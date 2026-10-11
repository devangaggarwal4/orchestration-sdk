"""Strategies decide where a request starts and what happens when validation fails.

hybrid     easy -> local, streamed, structural checks only
           uncertain -> local, buffered, full validator chain (cascade)
           hard -> straight to Sonnet (Opus above opus_threshold)
pre_route  tier from the score alone; never escalates on validation
cascade    always local first, buffered, full chain, escalate on failure
"""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_router.config import RoutingConfig
from hybrid_router.decision import AttemptRole, Strategy, Zone
from hybrid_router.types import Tier
from hybrid_router.validators.chain import ChainLevel
from hybrid_router.zones import cloud_tier_for


@dataclass(frozen=True)
class AttemptPlan:
    tier: Tier
    role: AttemptRole
    # Hold output back until validation passes.
    buffered: bool
    level: ChainLevel
    # Escalate to a bigger tier when validation fails.
    escalate: bool


def first_attempt(
    strategy: Strategy, zone: Zone, score: float, cfg: RoutingConfig, force_tier: Tier | None
) -> AttemptPlan:
    if force_tier is not None:
        return AttemptPlan(force_tier, "direct", buffered=False, level="light", escalate=False)
    if strategy == "pre_route":
        tier = cloud_tier_for(score, cfg) if zone is Zone.HARD else Tier.LOCAL
        # Full chain runs for the record only; pre_route never escalates on it.
        return AttemptPlan(tier, "direct", buffered=False, level="full", escalate=False)
    if strategy == "cascade":
        return AttemptPlan(
            Tier.LOCAL, "cascade_attempt", buffered=True, level="full", escalate=True
        )
    # hybrid
    if zone is Zone.EASY:
        return AttemptPlan(Tier.LOCAL, "direct", buffered=False, level="light", escalate=True)
    if zone is Zone.HARD:
        tier = cloud_tier_for(score, cfg)
        return AttemptPlan(tier, "direct", buffered=False, level="light", escalate=True)
    return AttemptPlan(Tier.LOCAL, "cascade_attempt", buffered=True, level="full", escalate=True)


def escalation_attempt(tier: Tier) -> AttemptPlan:
    return AttemptPlan(tier, "escalation", buffered=False, level="light", escalate=True)
