import json

import pytest

from hybrid_router.config import load_config
from hybrid_router.cost import TokenRatio, call_cost, counterfactual, token_ratio_sample
from hybrid_router.decision import AttemptRecord, RouteDecision, Zone
from hybrid_router.pricing import LocalPrice, ModelPrice, Pricing, load_pricing
from hybrid_router.router import HybridRouter
from hybrid_router.scoring import RuleScorer
from hybrid_router.telemetry import TelemetryStore, connect
from hybrid_router.testing import Reply, ScriptedBackend
from hybrid_router.types import Message, Tier, ToolResultBlock, ToolSpec, Usage

SONNET = "anthropic/claude-sonnet-5.5"
OPUS = "anthropic/claude-opus-5.5"
PRICING = Pricing(
    models={
        SONNET: ModelPrice(
            input_per_mtok=2, output_per_mtok=10, cache_read_per_mtok=0.2, cache_write_per_mtok=2.5
        ),
        OPUS: ModelPrice(input_per_mtok=4, output_per_mtok=20),
    },
    local=LocalPrice(per_hour_usd=3.6),  # $0.001 per second
)
GOOD = "A complete, confident and correct answer to the question."


def attempt(
    tier, model, inp, out, *, idx=0, ok=True, wasted=False, cost=None, latency=1000.0, cache_read=0
):
    return AttemptRecord(
        attempt_index=idx,
        tier=tier,
        model=model,
        role="direct",
        buffered=False,
        ok=ok,
        wasted=wasted,
        usage=Usage(input_tokens=inp, output_tokens=out, cache_read_tokens=cache_read),
        latency_ms=latency,
        reported_cost_usd=cost,
    )


def decision(*attempts, final):
    return RouteDecision(
        strategy="hybrid",
        score=0.5,
        zone=Zone.UNCERTAIN,
        reasons=[],
        planned_tier=Tier.LOCAL,
        final_tier=final,
        attempts=list(attempts),
    )


# --- pricing & call cost ------------------------------------------------------------


def test_pricing_lookup_by_prefix_and_packaged_default():
    assert PRICING.lookup(SONNET + "-20260928") is PRICING.models[SONNET]
    assert PRICING.lookup("openai/gpt-x") is None
    shipped = load_pricing()
    assert SONNET in shipped.models and not shipped.all_verified


def test_call_cost_sources():
    reported = attempt(Tier.SONNET, SONNET, 1000, 100, cost=0.5)
    assert (call_cost(reported, PRICING).usd, call_cost(reported, PRICING).source) == (
        0.5,
        "reported",
    )
    priced = attempt(Tier.SONNET, SONNET, 1_000_000, 100_000)
    assert call_cost(priced, PRICING).usd == pytest.approx(2 + 1)
    assert call_cost(priced, PRICING).source == "pricing"
    local = attempt(Tier.LOCAL, "qwen3:8b", 500, 50, latency=2000)
    assert call_cost(local, PRICING).usd == pytest.approx(0.002)
    assert call_cost(attempt(Tier.OPUS, "mystery", 1, 1), PRICING).source == "unknown"


# --- counterfactual ---------------------------------------------------------------


def cf(d, ratio=1.0, cache=0.0):
    return counterfactual(
        d, pricing=PRICING, baseline_model=SONNET, token_ratio=ratio, session_cache_ratio=cache
    )


def test_counterfactual_for_baseline_route_is_its_actual_cost():
    d = decision(attempt(Tier.SONNET, SONNET, 1000, 100, cost=0.123), final=Tier.SONNET)
    assert cf(d).usd == 0.123


def test_counterfactual_reprices_opus_at_baseline():
    d = decision(attempt(Tier.OPUS, OPUS, 1_000_000, 0, cost=4.0), final=Tier.OPUS)
    assert cf(d).usd == pytest.approx(2.0)  # cheaper than actual: negative savings


def test_counterfactual_for_local_uses_ratio_and_cache():
    d = decision(attempt(Tier.LOCAL, "qwen3:8b", 1_000_000, 100_000), final=Tier.LOCAL)
    plain = cf(d, ratio=1.0)
    assert plain.usd == pytest.approx(2 + 1) and plain.input_tokens == 1_000_000
    scaled = cf(d, ratio=1.2)
    assert scaled.input_tokens == 1_200_000 and scaled.usd == pytest.approx(2.4 + 1.2)
    cached = cf(d, ratio=1.0, cache=0.5)  # half the input billed at cache-read price
    assert cached.usd == pytest.approx(1.0 + 0.1 + 1.0)


def test_counterfactual_counts_escalated_route_once():
    d = decision(
        attempt(Tier.LOCAL, "qwen3:8b", 900, 50, idx=0, ok=False, wasted=True),
        attempt(Tier.SONNET, SONNET, 1000, 100, idx=1, cost=0.01),
        final=Tier.SONNET,
    )
    assert cf(d).usd == 0.01
    assert token_ratio_sample(d) == pytest.approx(1000 / 900)


def test_token_ratio_calibrates_after_min_samples():
    r = TokenRatio(default=1.15, min_samples=3)
    r.add(1.3)
    r.add(1.1)
    assert r.value == 1.15 and not r.calibrated
    r.add(1.2)
    r.add(99)  # discarded as nonsense
    assert r.calibrated and r.value == pytest.approx(1.2)


# --- store ---------------------------------------------------------------------------


def make_router(tmp_path, local, sonnet=None):
    cfg = load_config(environ={})
    store = TelemetryStore(tmp_path / "t.sqlite", config=cfg, pricing=PRICING)
    backends = {
        Tier.LOCAL: ScriptedBackend(local, tier=Tier.LOCAL, model="qwen3:8b"),
        Tier.SONNET: ScriptedBackend(sonnet or [GOOD], tier=Tier.SONNET, model=SONNET),
        Tier.OPUS: ScriptedBackend([GOOD], tier=Tier.OPUS, model=OPUS),
    }
    return HybridRouter(cfg, backends=backends, telemetry=store), store


async def test_routes_and_calls_are_recorded_with_waste_and_costs(tmp_path):
    router, store = make_router(
        tmp_path,
        local=[Reply("I'm not sure.", usage=Usage(input_tokens=1000, output_tokens=10))],
        sonnet=[Reply(GOOD, usage=Usage(input_tokens=1150, output_tokens=40), cost=0.0027)],
    )
    result = await router.acomplete("what is the plan?", session_id="s1", metadata={"run_id": "r1"})
    store.flush()
    conn = connect(store.db_path)
    route = dict(conn.execute("SELECT * FROM routes").fetchone())
    calls = [dict(r) for r in conn.execute("SELECT * FROM calls ORDER BY attempt_index")]

    assert route["route_id"] == result.decision.route_id and route["run_id"] == "r1"
    assert route["escalated"] == 1 and route["final_tier"] == "sonnet"
    assert route["counterfactual_cost_usd"] == 0.0027  # served by the baseline model
    # Sonnet's reported cost plus the local call's 1 ms of wall-clock at $3.60/h.
    assert route["actual_cost_usd"] == pytest.approx(0.0027 + 0.000001)
    assert "task_type" in json.loads(route["features"])

    local, cloud = calls
    assert (local["tier"], local["wasted"], local["local_succeeded"]) == ("local", 1, 0)
    assert local["role"] == "cascade_attempt" and local["cost_source"] == "local"
    assert json.loads(local["validator_results"])[-1]["name"] == "heuristic"
    assert (cloud["role"], cloud["cost_source"], cloud["wasted"]) == ("escalation", "reported", 0)
    # The route's counterfactual sits on the serving call only, so SUM() is right.
    total_cf = conn.execute("SELECT SUM(counterfactual_cost_usd) FROM calls").fetchone()[0]
    assert total_cf == pytest.approx(route["counterfactual_cost_usd"])
    store.close()


async def test_local_route_counterfactual_and_tool_error_backfill(tmp_path):
    weather = ToolSpec(name="get_weather", input_schema={"type": "object"})
    router, store = make_router(
        tmp_path,
        local=[Reply(tool_calls=[("get_weather", {})], usage=Usage(input_tokens=1000_000))],
    )
    scorer = router.scorer
    assert isinstance(scorer, RuleScorer)
    scorer.cfg.task_types["tool_use"] = 0.0  # keep it in the easy zone
    first = await router.acomplete(
        "weather?", tools=[weather], task_type="tool_use", session_id="s"
    )
    assert first.decision.final_tier is Tier.LOCAL
    history = [
        Message(role="user", content="weather?"),
        first.response.as_message(),
        Message(
            role="user",
            content=[
                ToolResultBlock(tool_use_id=first.tool_calls[0].id, content="x", is_error=True)
            ],
        ),
    ]
    await router.acomplete(history, tools=[weather], task_type="tool_use", session_id="s")
    store.flush()
    conn = connect(store.db_path)
    row = conn.execute(
        "SELECT local_succeeded FROM calls WHERE call_id = ?", (first.decision.attempts[0].call_id,)
    ).fetchone()
    assert row[0] == 0  # back-filled after the tool errored
    cf_cost = conn.execute(
        "SELECT counterfactual_cost_usd FROM routes WHERE route_id = ?",
        (first.decision.route_id,),
    ).fetchone()[0]
    assert cf_cost == pytest.approx(1.15 * 2.0, rel=0.01)  # 1M local tokens x default ratio
    store.close()


async def test_ratio_samples_persist_across_restarts(tmp_path):
    cfg = load_config(environ={})
    store = TelemetryStore(tmp_path / "t.sqlite", config=cfg, pricing=PRICING)
    store.token_ratio.add(1.3)
    store.close()
    reopened = TelemetryStore(tmp_path / "t.sqlite", config=cfg, pricing=PRICING)
    assert reopened.token_ratio.samples == [1.3]
    reopened.close()


async def test_router_auto_creates_telemetry_from_config(tmp_path):
    cfg = load_config(environ={"HYBRID_ROUTER__TELEMETRY__DB_PATH": str(tmp_path / "auto.db")})
    router = HybridRouter(cfg, backends={t: ScriptedBackend([GOOD], tier=t) for t in Tier})
    assert isinstance(router.telemetry, TelemetryStore)
    await router.acomplete("hi")
    await router.aclose()
    assert connect(tmp_path / "auto.db").execute("SELECT COUNT(*) FROM routes").fetchone()[0] == 1

    off = HybridRouter(
        cfg, backends={t: ScriptedBackend([GOOD], tier=t) for t in Tier}, telemetry=None
    )
    assert off.telemetry is None
