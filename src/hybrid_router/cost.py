"""Actual cost per call and the all-cloud counterfactual per route."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from hybrid_router.decision import AttemptRecord, RouteDecision
from hybrid_router.pricing import Pricing
from hybrid_router.types import Tier, Usage

CostSource = str  # "reported" | "pricing" | "local" | "unknown"


@dataclass
class CallCost:
    usd: float | None
    source: CostSource


def call_cost(attempt: AttemptRecord, pricing: Pricing) -> CallCost:
    """Provider-reported cost wins; then pricing.yaml; local calls use local pricing."""
    if attempt.reported_cost_usd is not None:
        return CallCost(attempt.reported_cost_usd, "reported")
    if attempt.tier is Tier.LOCAL:
        return CallCost(pricing.local.cost(attempt.latency_ms), "local")
    price = pricing.lookup(attempt.model)
    if price is None:
        return CallCost(None, "unknown")
    return CallCost(price.cost(attempt.usage), "pricing")


def validator_cost(attempt: AttemptRecord, pricing: Pricing) -> float:
    """Model usage spent inside validators (two-sample, judge)."""
    total = 0.0
    for v in attempt.validator_results:
        if v.extra_cost_usd is not None:
            total += v.extra_cost_usd
        elif v.extra_usage is not None and attempt.tier.is_cloud:
            price = pricing.lookup(attempt.model)
            total += price.cost(v.extra_usage) if price else 0.0
    return total


@dataclass
class Counterfactual:
    usd: float | None
    input_tokens: int
    output_tokens: int


def counterfactual(
    decision: RouteDecision,
    *,
    pricing: Pricing,
    baseline_model: str,
    token_ratio: float,
    session_cache_ratio: float,
) -> Counterfactual:
    """What this route would have cost had it gone straight to the baseline model.

    * served by the baseline model -> its actual cost
    * served by another cloud model (e.g. Opus) -> same tokens at baseline prices
      (so routes that needed Opus show negative savings, which is honest)
    * served locally -> local tokens x calibrated ratio, with the session's
      observed cache-hit ratio applied to input
    """
    served = _served_attempt(decision)
    price = pricing.lookup(baseline_model)
    if served is None:
        return Counterfactual(None, 0, 0)
    u = served.usage
    if served.tier.is_cloud:
        if served.model.startswith(baseline_model):
            cost = call_cost(served, pricing).usd
            return Counterfactual(cost, u.total_input_tokens, u.output_tokens)
        if price is None:
            return Counterfactual(None, u.total_input_tokens, u.output_tokens)
        return Counterfactual(price.cost(u), u.total_input_tokens, u.output_tokens)

    est_in = round(u.total_input_tokens * token_ratio)
    est_out = round(u.output_tokens * token_ratio)
    if price is None:
        return Counterfactual(None, est_in, est_out)
    cached = round(est_in * min(max(session_cache_ratio, 0.0), 1.0))
    est_usage = Usage(input_tokens=est_in - cached, output_tokens=est_out, cache_read_tokens=cached)
    return Counterfactual(price.cost(est_usage), est_in, est_out)


def _served_attempt(decision: RouteDecision) -> AttemptRecord | None:
    if decision.final_tier is None:
        return None
    for a in reversed(decision.attempts):
        if a.tier is decision.final_tier and not a.wasted:
            return a
    return None


def token_ratio_sample(decision: RouteDecision) -> float | None:
    """claude_input / local_input for one escalated route (same request, both counts)."""
    local = next((a for a in decision.attempts if a.tier is Tier.LOCAL), None)
    cloud = next((a for a in decision.attempts if a.tier.is_cloud and a.usage.output_tokens), None)
    if local is None or cloud is None or local.usage.total_input_tokens <= 0:
        return None
    if cloud.usage.total_input_tokens <= 0:
        return None
    return cloud.usage.total_input_tokens / local.usage.total_input_tokens


class TokenRatio:
    """Median of recent samples once there are enough, else the configured default."""

    def __init__(self, default: float, min_samples: int, window: int = 500):
        self.default = default
        self.min_samples = min_samples
        self.window = window
        self.samples: list[float] = []

    def add(self, value: float | None) -> None:
        if value is None or not 0.2 < value < 5.0:  # discard nonsense (e.g. revise payloads)
            return
        self.samples.append(value)
        del self.samples[: -self.window]

    @property
    def value(self) -> float:
        if len(self.samples) < self.min_samples:
            return self.default
        return statistics.median(self.samples)

    @property
    def calibrated(self) -> bool:
        return len(self.samples) >= self.min_samples
