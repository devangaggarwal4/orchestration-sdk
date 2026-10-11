import asyncio

import pytest

from hybrid_router.config import load_config
from hybrid_router.evals import run_evals
from hybrid_router.pricing import load_pricing
from hybrid_router.reporting import (
    apply_patch,
    build_report,
    load_routes,
    parse_since,
    replay,
    suggest,
)
from hybrid_router.reporting.queries import RouteRow, pct
from hybrid_router.reporting.tune import auc, fit_logistic
from hybrid_router.telemetry import connect

CFG = load_config(environ={})


@pytest.fixture(scope="module")
def sim_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("rep") / "sim.sqlite"
    asyncio.run(
        run_evals(CFG, strategies=["hybrid", "cascade"], simulate=True, repeat=2, db_path=path)
    )
    return path


def row(
    score,
    failed,
    *,
    zone="uncertain",
    escalated=None,
    strategy="hybrid",
    features=None,
    cloud_cost=0.0,
    cf=0.01,
    waste=0.0,
):
    escalated = failed if escalated is None else escalated
    return RouteRow(
        route_id=f"r{score}-{failed}",
        ts=0,
        session_id=None,
        task_type="t",
        strategy=strategy,
        score=score,
        zone=zone,
        features=features
        or {"tokens": 0, "tools": 0, "task_type": score / 0.8, "keywords": 0, "schema": 0},
        forced=False,
        planned_tier="local",
        final_tier="sonnet" if escalated else "local",
        escalated=escalated,
        escalation_reason="local failed heuristic: x" if escalated else None,
        success=True,
        total_latency_ms=1000,
        actual_cost_usd=cloud_cost,
        counterfactual_cost_usd=cf,
        config_hash="h",
        local_role="cascade_attempt",
        local_ok=not failed,
        local_succeeded=not failed,
        local_latency_ms=1500,
        local_failed_validator="heuristic" if failed else None,
        cloud_cost_usd=cloud_cost,
        cloud_latency_ms=3000 if escalated else None,
        waste_latency_ms=waste,
    )


def test_parse_since():
    assert parse_since("7d", now=1_000_000) == 1_000_000 - 7 * 86400
    assert parse_since("90m", now=10_000) == 10_000 - 5400
    assert parse_since(None) == 0
    assert parse_since("2026-10-01") > 0
    with pytest.raises(ValueError, match="7d"):
        parse_since("last tuesday")


def test_pct():
    assert pct([], 50) is None
    assert pct([5], 95) == 5
    assert pct(list(range(1, 101)), 50) == pytest.approx(50.5)


def test_report_sections(sim_db):
    rep = build_report(connect(sim_db), 0.0, load_pricing())
    o = rep["overview"]
    assert o["routes"] > 100 and 0 < o["local_route_share"] < 1
    assert o["savings_usd"] == pytest.approx(o["counterfactual_cost_usd"] - o["actual_cost_usd"])
    assert o["prices_verified"] is False
    assert {z["zone"] for z in rep["by_zone"]} == {"easy", "uncertain", "hard"}
    assert rep["top_escalation_reasons"][0]["count"] >= rep["top_escalation_reasons"][-1]["count"]
    assert rep["cascade_waste"]["wasted_calls"] > 0
    assert {t["tier"] for t in rep["latency_by_tier"]} == {"local", "sonnet", "opus"}
    assert len(rep["score_histogram"]) == 20
    assert sum(p["local"] + p["cloud"] for p in rep["timeseries"]["points"]) == o["routes"]


def test_report_reasons_are_grouped():
    from hybrid_router.reporting.report import _normalize_reason

    a = _normalize_reason("local failed heuristic: suspiciously short (12 < 40 chars)")
    b = _normalize_reason("local failed heuristic: suspiciously short (3 < 40 chars)")
    assert a == b == "local failed heuristic: suspiciously short"


def test_empty_window_report(sim_db):
    rep = build_report(connect(sim_db), 9e12)
    assert rep["overview"]["routes"] == 0


def test_tune_refuses_without_enough_data():
    result = suggest([row(0.5, True)] * 10, CFG)
    assert not result.enough_data and "at least 50" in result.message


def test_tune_lowers_hard_threshold_when_mid_scores_keep_failing():
    routes = (
        [row(0.1, False)] * 30
        + [row(0.3, False)] * 20
        + [row(0.47, True, waste=1500, cloud_cost=0.01)] * 18
        + [row(0.47, False)] * 2
        + [row(0.57, True, waste=1500, cloud_cost=0.01)] * 20
    )
    result = suggest(routes, CFG)
    by_param = {s.param: s for s in result.suggestions}
    hard = by_param["routing.hard_threshold"]
    assert hard.suggested == pytest.approx(0.45)
    assert "escalated 95% of the time" in hard.rationale
    assert hard.est_latency_delta_ms is not None and hard.est_latency_delta_ms < 0  # waste saved
    assert "hard_threshold: 0.45" in result.yaml_patch
    # The patch validates as a real config.
    patched = apply_patch(CFG, {"routing": {"hard_threshold": 0.45}})
    assert patched.routing.hard_threshold == 0.45


def test_tune_raises_easy_threshold_when_local_always_succeeds():
    routes = [row(s, False) for s in (0.1, 0.2, 0.3, 0.35) for _ in range(15)] + [
        row(0.5, True) for _ in range(10)
    ]
    result = suggest(routes, CFG)
    easy = next(s for s in result.suggestions if s.param == "routing.easy_threshold")
    assert easy.suggested == pytest.approx(0.4) and easy.current == 0.25


def test_whatif_counts_observed_and_estimated_effects():
    routes = [
        row(0.5, True, waste=2000, cloud_cost=0.02),  # escalated: skipping local is observed
        row(0.5, False, cf=0.03),  # succeeded locally: going to cloud costs ~cf (estimated)
        row(0.3, True, zone="uncertain"),  # caught by heuristic; easy zone would miss it
    ]
    candidate = apply_patch(CFG, {"routing": {"hard_threshold": 0.4, "easy_threshold": 0.35}})
    result = replay(routes, CFG, candidate)
    assert result["transitions"] == {"uncertain→hard": 2, "uncertain→easy": 1}
    assert result["est_latency_delta_ms"] < 0
    assert result["est_cost_delta_usd"] == pytest.approx(0.03)
    assert result["est_new_under_escalations"] == pytest.approx(1)
    assert result["observed_changes"] == 2 and result["estimated_changes"] == 1


def test_auc_and_logistic_fit():
    assert auc([0.9, 0.8, 0.1, 0.2], [True, True, False, False]) == 1.0
    assert auc([0.5], [True]) is None
    x = [[0.0], [0.1], [0.9], [1.0]] * 10
    y = [False, False, True, True] * 10
    assert fit_logistic(x, y)[1] > 1


def test_load_routes_joins_local_attempt(sim_db):
    routes = load_routes(connect(sim_db))
    cascades = [r for r in routes if r.local_role == "cascade_attempt"]
    assert cascades and all(r.local_ok is not None for r in cascades)
    escalated = [r for r in routes if r.escalated]
    assert escalated and all(r.waste_latency_ms > 0 for r in escalated if r.local_ok is False)
