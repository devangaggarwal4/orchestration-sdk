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
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator, model_validator

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


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoutingConfig(_Section):
    strategy: Literal["hybrid", "pre_route", "cascade"] = "hybrid"
    # Zones: score < easy -> local (light checks, streamed live);
    # score > hard -> skip local (Opus above opus_threshold); otherwise cascade.
    # Conservative defaults for an 8B local model.
    easy_threshold: float = Field(default=0.25, ge=0, le=1)
    hard_threshold: float = Field(default=0.65, ge=0, le=1)
    opus_threshold: float = Field(default=0.85, ge=0, le=1)
    # Hard validator failures on the local tier skip Sonnet and go to Opus.
    escalate_hard_to_opus: bool = True
    # "original": resend the original request. "revise": also show the cloud
    # model the failed local draft and why it failed.
    escalation_mode: Literal["original", "revise"] = "original"
    # Upper bound on model calls per routed request (all tiers).
    max_attempts: int = Field(default=3, ge=1)
    # pre_route never escalates on validation, but a crashed/timed-out call is
    # not a routing result, so fall back to Sonnet when this is on.
    pre_route_fallback_on_error: bool = True
    # One retry for retryable cloud errors (rate limit, 5xx, timeout).
    cloud_retries: int = Field(default=1, ge=0)
    # Local generations allowed at once; an 8B model on 16 GB serves one well.
    local_concurrency: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> RoutingConfig:
        if not self.easy_threshold <= self.hard_threshold <= self.opus_threshold:
            raise ValueError("thresholds must satisfy easy <= hard <= opus")
        return self


class KeywordRule(_Section):
    pattern: str
    weight: float = Field(default=1.0, ge=0, le=1)


class RuleScorerConfig(_Section):
    # "learned" uses the model written by `hybrid-router train-scorer` (falls back
    # to rule if the file is missing).
    kind: Literal["rule", "learned"] = "rule"
    learned_model_path: str = "learned_scorer.json"
    # Each feature is an independent signal of difficulty, combined noisy-OR:
    #   score = 1 - prod(1 - weight * feature)
    # A weight in [0, 1] is how far that feature alone can push the score, so a
    # missing signal (no tools, short context) never drags a clear one down.
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "tokens": 0.6,
            "tools": 0.4,
            "task_type": 0.8,
            "keywords": 0.6,
            "schema": 0.4,
        }
    )
    # Context size: 0 at <= tokens_low, 1 at >= tokens_high (linear between).
    tokens_low: int = 2000
    tokens_high: int = 24000
    tools_saturation: int = 8
    task_types: dict[str, float] = Field(
        default_factory=lambda: {
            "classify": 0.05,
            "extract": 0.1,
            "summarize": 0.15,
            "rewrite": 0.15,
            "chat": 0.3,
            "tool_use": 0.4,
            "code": 0.6,
            "code_review": 0.8,
            "plan": 0.85,
            "multi_step_reasoning": 0.9,
        }
    )
    task_type_default: float = 0.4
    keywords: list[KeywordRule] = Field(
        default_factory=lambda: [
            KeywordRule(pattern=r"\b(prove|proof|derive|theorem)\b", weight=1.0),
            KeywordRule(pattern=r"\b(architect(ure)?|design a|trade-?offs?)\b", weight=0.8),
            KeywordRule(
                pattern=r"\b(refactor|race condition|concurren(t|cy)|deadlock)\b", weight=0.8
            ),
            KeywordRule(pattern=r"\b(step[- ]by[- ]step|multi[- ]step|plan out)\b", weight=0.6),
            KeywordRule(pattern=r"\b(security|vulnerab\w+|exploit)\b", weight=0.7),
            KeywordRule(pattern=r"\b(optimi[sz]e|complexity|algorithm)\b", weight=0.5),
        ]
    )
    # Schema complexity saturates at this many "units" (properties, nesting, unions).
    schema_saturation: int = 25
    # Additive boosts on top of the weighted average.
    sticky_steps: int = Field(default=3, ge=0)
    sticky_boost: float = Field(default=0.25, ge=0, le=1)
    tool_error_boost: float = Field(default=0.2, ge=0, le=1)
    # Only scan this many chars from each end of the latest user message for keywords.
    keyword_scan_chars: int = 4000
    # Reasons list contains features contributing at least this much.
    reason_min_contribution: float = 0.03


class StructuralConfig(_Section):
    enabled: bool = True


class HeuristicConfig(_Section):
    enabled: bool = True
    refusal_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\bI(?:'m| am) (?:not able|unable) to\b",
            r"\bI can(?:not|'t) (?:help|assist|do that|answer)\b",
            r"\bI(?:'m| am) not sure\b",
            r"\bI don't know\b",
            r"\bas an AI\b",
        ]
    )
    # Minimum answer length (chars) by task_type; "default" for the rest.
    min_chars: dict[str, int] = Field(
        default_factory=lambda: {"default": 1, "summarize": 40, "code": 20, "plan": 80}
    )
    # Fraction of repeated 6-word shingles above which output counts as looping.
    repetition_threshold: float = 0.5


class ConfidenceConfig(_Section):
    enabled: bool = True
    # Mean per-token logprob below this fails (skipped when no logprobs).
    min_mean_logprob: float = -1.0
    # Second local sample at higher temperature; disagreement is a soft failure.
    two_sample: bool = False
    two_sample_temperature: float = 0.8
    two_sample_min_similarity: float = 0.4


class ExecutionConfig(_Section):
    enabled: bool = True
    # Compile Python found in ```python fences (and in code task_types).
    python_compile: bool = True
    code_task_types: list[str] = Field(default_factory=lambda: ["code"])


class JudgeConfig(_Section):
    enabled: bool = False
    tier: Tier = Tier.LOCAL
    max_tokens: int = 200


class ValidatorsConfig(_Section):
    structural: StructuralConfig = StructuralConfig()
    heuristic: HeuristicConfig = HeuristicConfig()
    confidence: ConfidenceConfig = ConfidenceConfig()
    execution: ExecutionConfig = ExecutionConfig()
    judge: JudgeConfig = JudgeConfig()


class TelemetryConfig(_Section):
    enabled: bool = True
    db_path: str = "hybrid_router.sqlite"
    # pricing.yaml location; falls back to ./pricing.yaml, then the packaged copy.
    pricing_path: str | None = None
    # "What would it have cost if every call went to the cloud", priced as this tier.
    counterfactual_baseline: Tier = Tier.SONNET
    # For local routes, assume the session's observed cloud cache-hit ratio
    # ("observed") or no caching at all ("none").
    cache_assumption: Literal["observed", "none"] = "observed"
    # Local -> Claude token ratio, used until enough escalations calibrate it.
    token_ratio_default: float = Field(default=1.15, gt=0)
    token_ratio_min_samples: int = Field(default=20, ge=1)


class AgentConfig(_Section):
    # Directory the example agent's file tools may read.
    workspace: str = "."
    max_steps: int = Field(default=12, ge=1)
    # run_python executes model-written code; keep off unless you trust the setup.
    allow_code_exec: bool = False
    exec_timeout_s: float = 10.0


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tiers: dict[Tier, SerializeAsAny[BackendConfig]]
    routing: RoutingConfig = RoutingConfig()
    scorer: RuleScorerConfig = RuleScorerConfig()
    validators: ValidatorsConfig = ValidatorsConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    agent: AgentConfig = AgentConfig()

    def config_hash(self) -> str:
        """Short stable hash, recorded on every route so changes are traceable."""
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

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
