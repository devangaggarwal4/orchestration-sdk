"""Router configuration: YAML file + defaults + environment overrides.

Resolution order (later wins):
  1. built-in defaults (``DEFAULTS``)
  2. the YAML file: explicit path, else $HYBRID_ROUTER_CONFIG, else ./hybrid_router.yaml
  3. environment variables ``HYBRID_ROUTER__<SECTION>__<KEY>...`` e.g.
     ``HYBRID_ROUTER__TIERS__LOCAL__MODEL=llama3.1:8b``. Values are parsed as YAML,
     so ``true``/``8192``/``[a, b]`` get their natural types.

Model names live here (and in the YAML), never in code paths.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, SerializeAsAny, field_validator

from hybrid_router.backends.base import BackendConfig
from hybrid_router.backends.registry import parse_backend_config
from hybrid_router.types import TIER_ORDER, Tier

CONFIG_ENV = "HYBRID_ROUTER_CONFIG"
ENV_PREFIX = "HYBRID_ROUTER__"
DEFAULT_CONFIG_FILE = "hybrid_router.yaml"

DEFAULTS: dict[str, Any] = {
    "tiers": {
        "local": {"kind": "ollama", "model": "qwen3:8b"},
        # Claude via OpenRouter. Claude 5.5 models 400 on temperature and on
        # forced tool_choice, so both are switched off for these tiers.
        "sonnet": {
            "kind": "openai_compat",
            "model": "anthropic/claude-sonnet-5.5",
            "sampling_params": False,
            "forced_tool_choice": False,
        },
        "opus": {
            "kind": "openai_compat",
            "model": "anthropic/claude-opus-5.5",
            "sampling_params": False,
            "forced_tool_choice": False,
        },
    },
}


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tiers: dict[Tier, SerializeAsAny[BackendConfig]]

    @field_validator("tiers", mode="before")
    @classmethod
    def _parse_tiers(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        parsed = {}
        for tier, cfg in value.items():
            try:
                parsed[tier] = parse_backend_config(cfg)
            except ValueError as e:
                raise ValueError(f"tiers.{tier}: {e}") from e
        return parsed

    @field_validator("tiers")
    @classmethod
    def _all_tiers_present(cls, value: dict[Tier, BackendConfig]) -> dict[Tier, BackendConfig]:
        missing = [t.value for t in TIER_ORDER if t not in value]
        if missing:
            raise ValueError(f"missing tier(s): {', '.join(missing)}")
        return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            # A tier switching backend kind replaces the old tier block wholesale,
            # otherwise fields from the default kind would leak in.
            if "kind" in val and val["kind"] != out[key].get("kind"):
                out[key] = copy.deepcopy(val)
            else:
                out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def env_overrides(environ: dict[str, str] | None = None) -> dict[str, Any]:
    environ = dict(os.environ) if environ is None else environ
    result: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in name[len(ENV_PREFIX) :].split("__") if p]
        if not path:
            continue
        node = result
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = yaml.safe_load(raw) if raw != "" else ""
    return result


def resolve_config_path(path: str | Path | None = None) -> Path | None:
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")
        return p
    if env := os.environ.get(CONFIG_ENV):
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"{CONFIG_ENV} points to a missing file: {p}")
        return p
    p = Path(DEFAULT_CONFIG_FILE)
    return p if p.is_file() else None


def load_config(
    path: str | Path | None = None, *, environ: dict[str, str] | None = None
) -> RouterConfig:
    data = copy.deepcopy(DEFAULTS)
    if (resolved := resolve_config_path(path)) is not None:
        loaded = yaml.safe_load(resolved.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{resolved}: top level must be a mapping")
        data = deep_merge(data, loaded)
    data = deep_merge(data, env_overrides(environ))
    return RouterConfig.model_validate(data)
