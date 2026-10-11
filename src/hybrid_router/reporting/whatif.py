"""Replay recorded routes under a candidate config, with no model calls.

Every route stores its feature vector, so a candidate scorer/threshold set can
re-score it. Effects are *observed* when the route already ran the way the
candidate would send it (e.g. it escalated, so skipping local saves exactly the
waste we measured) and *estimated* otherwise (using local failure rates by score).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from hybrid_router.config import RouterConfig
from hybrid_router.reporting.queries import RouteRow, pct
from hybrid_router.scoring.rule import RuleScorer
from hybrid_router.zones import planned_tier, zone_for

# Share of local failures the validators don't catch. Unknown from routing data
# alone; the eval harness measures it. Used only for estimated quality risk.
UNDETECTED_FAILURE_SHARE = 0.25
LIGHT_CHAIN = {None, "structural", "error"}


def local_failure_rates(routes: list[RouteRow], width: float = 0.1) -> dict[int, float]:
    by_bin: dict[int, list[bool]] = {}
    for r in routes:
        if r.local_failed is not None and not r.forced:
            by_bin.setdefault(int(min(r.score, 0.999) / width), []).append(r.local_failed)
    return {b: sum(v) / len(v) for b, v in by_bin.items() if v}


def replay(
    routes: list[RouteRow], current: RouterConfig, candidate: RouterConfig
) -> dict[str, Any]:
    scorer = RuleScorer(candidate.scorer)
    rates = local_failure_rates(routes)
    overall_fail = (
        (
            sum(r.local_failed for r in routes if r.local_failed is not None)
            / max(sum(1 for r in routes if r.local_failed is not None), 1)
        )
        if any(r.local_failed is not None for r in routes)
        else 0.5
    )
    local_lat_p50 = pct([r.local_latency_ms for r in routes if r.local_latency_ms], 50) or 0.0
    cloud_lat_p50 = pct([r.cloud_latency_ms for r in routes if r.cloud_latency_ms], 50) or 0.0

    transitions: Counter[str] = Counter()
    zone_before: Counter[str] = Counter()
    zone_after: Counter[str] = Counter()
    cost_delta = latency_delta = risk = 0.0
    observed = estimated = 0
    changed_rows = []

    for r in routes:
        if r.forced or r.strategy != "hybrid" or not r.features:
            continue
        new = scorer.score_features(r.features)
        new_zone = zone_for(new.score, candidate.routing).value
        zone_before[r.zone] += 1
        zone_after[new_zone] += 1
        old_tier = planned_tier(r.score, current.routing).value
        new_tier = planned_tier(new.score, candidate.routing).value
        if new_zone == r.zone and new_tier == old_tier:
            continue
        transitions[f"{r.zone}→{new_zone}"] += 1
        cloud_cost = r.cloud_cost_usd or (r.counterfactual_cost_usd or 0.0)
        local_lat = r.local_latency_ms or local_lat_p50
        cloud_lat = r.cloud_latency_ms or cloud_lat_p50

        if old_tier == "local" and new_tier == "local":
            if r.zone == "uncertain" and new_zone == "easy":
                if r.local_failed_validator not in LIGHT_CHAIN:
                    # Full chain caught it; the light chain would have let it through.
                    risk += 1
                    cost_delta -= r.cloud_cost_usd
                    latency_delta -= r.cloud_latency_ms or 0.0
                observed += 1
            else:
                observed += 1  # easy->uncertain: more checking, no cost change
        elif old_tier == "local":  # now straight to cloud
            if r.escalated:
                latency_delta -= r.waste_latency_ms
                observed += 1
            else:
                cost_delta += cloud_cost
                latency_delta += cloud_lat - local_lat
                estimated += 1
        elif new_tier == "local":  # was cloud-direct, now tries local first
            p = rates.get(int(min(new.score, 0.999) / 0.1), overall_fail)
            cost_delta -= (1 - p) * (r.actual_cost_usd or 0.0)
            latency_delta += local_lat - (1 - p) * cloud_lat
            risk += p * UNDETECTED_FAILURE_SHARE
            estimated += 1
        else:
            estimated += 1  # sonnet <-> opus: counted, not priced
        if len(changed_rows) < 50:
            changed_rows.append(
                {
                    "route_id": r.route_id,
                    "task_type": r.task_type,
                    "old_score": r.score,
                    "new_score": new.score,
                    "old_zone": r.zone,
                    "new_zone": new_zone,
                    "old_tier": old_tier,
                    "new_tier": new_tier,
                }
            )

    considered = sum(zone_before.values())
    return {
        "routes_considered": considered,
        "routes_changed": sum(transitions.values()),
        "transitions": dict(transitions),
        "zone_mix_before": dict(zone_before),
        "zone_mix_after": dict(zone_after),
        "est_cost_delta_usd": cost_delta,
        "est_latency_delta_ms": latency_delta,
        "est_new_under_escalations": risk,
        "observed_changes": observed,
        "estimated_changes": estimated,
        "changed_routes": changed_rows,
    }
