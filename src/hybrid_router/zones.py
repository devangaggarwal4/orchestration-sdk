"""Map a difficulty score to a zone and the tier that zone targets."""

from __future__ import annotations

from hybrid_router.config import RoutingConfig
from hybrid_router.decision import Zone
from hybrid_router.types import Tier


def zone_for(score: float, cfg: RoutingConfig) -> Zone:
    if score < cfg.easy_threshold:
        return Zone.EASY
    if score > cfg.hard_threshold:
        return Zone.HARD
    return Zone.UNCERTAIN


def cloud_tier_for(score: float, cfg: RoutingConfig) -> Tier:
    return Tier.OPUS if score > cfg.opus_threshold else Tier.SONNET


def planned_tier(score: float, cfg: RoutingConfig) -> Tier:
    """The tier the hybrid strategy starts at for this score."""
    return cloud_tier_for(score, cfg) if zone_for(score, cfg) is Zone.HARD else Tier.LOCAL
