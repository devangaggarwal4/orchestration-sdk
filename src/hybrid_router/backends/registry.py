"""Maps a config ``kind`` to its Backend class.

Built-in backends register themselves through the ``hybrid_router.backends``
entry-point group (see pyproject.toml); third-party packages do the same, or
call ``register_backend`` directly.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, TypeVar

from hybrid_router.backends.base import Backend, BackendConfig
from hybrid_router.types import Tier

ENTRY_POINT_GROUP = "hybrid_router.backends"

_registry: dict[str, type[Backend]] = {}
_B = TypeVar("_B", bound=type[Backend])
_entry_points_loaded = False


def register_backend(cls: _B) -> _B:
    """Register a Backend class under ``cls.kind``. Usable as a decorator."""
    _registry[cls.kind] = cls
    return cls


def _load_entry_points() -> None:
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name not in _registry:
            _registry[ep.name] = ep.load()


def _load_builtins() -> None:
    # Imported for their register_backend side effect; works without an installed dist.
    from hybrid_router.backends import anthropic, ollama  # noqa: F401


def get_backend_class(kind: str) -> type[Backend]:
    if kind not in _registry:
        _load_builtins()
    if kind not in _registry:
        _load_entry_points()
    try:
        return _registry[kind]
    except KeyError:
        known = ", ".join(sorted(_registry)) or "none"
        raise ValueError(f"unknown backend kind {kind!r} (registered: {known})") from None


def parse_backend_config(data: dict[str, Any] | BackendConfig) -> BackendConfig:
    if isinstance(data, BackendConfig):
        return data
    kind = data.get("kind")
    if not kind:
        raise ValueError("backend config needs a 'kind' (e.g. 'ollama' or 'anthropic')")
    return get_backend_class(kind).config_model.model_validate(data)


def create_backend(config: BackendConfig, *, tier: Tier, name: str | None = None) -> Backend:
    return get_backend_class(config.kind)(config, tier=tier, name=name)
