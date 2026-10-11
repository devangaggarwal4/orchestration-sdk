"""Prices from pricing.yaml. Never hardcode a price in code."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from hybrid_router.types import Usage


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None
    verified: bool = False

    def cost(self, usage: Usage) -> float:
        read = (
            self.cache_read_per_mtok
            if self.cache_read_per_mtok is not None
            else (self.input_per_mtok)
        )
        write = (
            self.cache_write_per_mtok
            if self.cache_write_per_mtok is not None
            else (self.input_per_mtok)
        )
        return (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
            + usage.cache_read_tokens * read
            + usage.cache_write_tokens * write
        ) / 1_000_000


class LocalPrice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_cost_usd: float = 0.0
    per_hour_usd: float = 0.0

    def cost(self, latency_ms: float) -> float:
        return self.default_cost_usd + self.per_hour_usd * latency_ms / 3_600_000


class Pricing(BaseModel):
    model_config = ConfigDict(extra="forbid")

    currency: str = "USD"
    models: dict[str, ModelPrice] = {}
    local: LocalPrice = LocalPrice()

    def lookup(self, model: str) -> ModelPrice | None:
        if model in self.models:
            return self.models[model]
        matches = [k for k in self.models if model.startswith(k)]
        return self.models[max(matches, key=len)] if matches else None

    @property
    def all_verified(self) -> bool:
        return all(p.verified for p in self.models.values())


def load_pricing(path: str | Path | None = None) -> Pricing:
    """Load ``path``, else ./pricing.yaml, else the copy shipped with the package."""
    if path is not None and Path(path).is_file():
        text = Path(path).read_text()
    elif Path("pricing.yaml").is_file():
        text = Path("pricing.yaml").read_text()
    else:
        text = resources.files("hybrid_router.data").joinpath("pricing.yaml").read_text()
    return Pricing.model_validate(yaml.safe_load(text) or {})
