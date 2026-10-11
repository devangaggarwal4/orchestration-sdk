import asyncio

import pytest

from hybrid_router.backends.base import BackendTimeout, BackendUnavailable
from hybrid_router.config import RouterConfig, load_config
from hybrid_router.decision import (
    AttemptEndEvent,
    RouteEndEvent,
    RoutingError,
    Zone,
)
from hybrid_router.router import HybridRouter, revise_request
from hybrid_router.scoring import ScoreResult, ScoringContext
from hybrid_router.testing import Reply, ScriptedBackend
from hybrid_router.types import (
    CompletionRequest,
    Message,
    TextDeltaEvent,
    Tier,
    ToolResultBlock,
    ToolSpec,
)

GOOD = "A complete, confident and correct answer to the question."
WEATHER = ToolSpec(
    name="get_weather",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
)


class FixedScorer:
    """Returns a preset score so tests pick the zone directly."""

    name = "fixed"

    def __init__(self, score: float):
        self.value = score
        self.contexts: list[ScoringContext] = []

    def score(self, req: CompletionRequest, ctx: ScoringContext) -> ScoreResult:
        self.contexts.append(ctx)
        boost = 0.25 if ctx.sticky_remaining else 0.0
        return ScoreResult(score=min(self.value + boost, 1.0), reasons=["fixed"], features={})


class Recorder:
    def __init__(self):
        self.routes = []
        self.tool_errors = []

    async def on_route(self, decision, request, response):
        self.routes.append(decision)

    async def on_local_tool_error(self, call_ids):
        self.tool_errors.append(call_ids)


def make(score=0.5, local=None, sonnet=None, opus=None, observers=None, **routing):
    cfg = load_config(environ={})
    if routing:
        cfg = cfg.model_copy(update={"routing": cfg.routing.model_copy(update=routing)})
    backends = {
        Tier.LOCAL: ScriptedBackend(local or [GOOD], tier=Tier.LOCAL),
        Tier.SONNET: ScriptedBackend(sonnet or [GOOD], tier=Tier.SONNET),
        Tier.OPUS: ScriptedBackend(opus or [GOOD], tier=Tier.OPUS),
    }
    router = HybridRouter(cfg, backends=backends, scorer=FixedScorer(score), observers=observers)
    return router, backends


def end(events) -> RouteEndEvent:
    assert isinstance(events[-1], RouteEndEvent)
    return events[-1]


def calls(backends) -> dict[str, int]:
    return {t.value: len(b.requests) for t, b in backends.items()}


# --- hybrid zones ---------------------------------------------------------------


async def test_easy_zone_stays_local_and_streams_live():
    router, backends = make(score=0.1)
    events = [e async for e in router.astream("hi")]
    end = events[-1]
    assert isinstance(end, RouteEndEvent) and end.response is not None
    d = end.decision
    assert d.zone is Zone.EASY and d.final_tier is Tier.LOCAL and d.success
    assert d.attempts[0].role == "direct" and not d.attempts[0].buffered
    assert [v.name for v in d.attempts[0].validator_results] == ["structural"]
    # Text arrives before the attempt is closed (live), not replayed after.
    kinds = [e.type for e in events]
    assert kinds.index("text_delta") < kinds.index("attempt_end")
    assert calls(backends) == {"local": 1, "sonnet": 0, "opus": 0}


async def test_hard_zone_skips_local():
    router, backends = make(score=0.7)
    result = await router.acomplete("design a distributed lock")
    assert result.decision.zone is Zone.HARD
    assert result.decision.final_tier is Tier.SONNET
    assert calls(backends) == {"local": 0, "sonnet": 1, "opus": 0}

    router, backends = make(score=0.9)
    assert (await router.acomplete("x")).decision.final_tier is Tier.OPUS


async def test_uncertain_zone_passes_after_buffered_validation():
    router, backends = make(score=0.5, local=[Reply(GOOD, logprobs=[-0.1, -0.2])])
    events = [e async for e in router.astream("hi")]
    kinds = [e.type for e in events]
    # Buffered: deltas only appear after validation, right before attempt_end.
    assert kinds.index("text_delta") > kinds.index("attempt_start")
    d = end(events).decision
    assert d.attempts[0].role == "cascade_attempt" and d.attempts[0].buffered
    assert [v.name for v in d.attempts[0].validator_results] == [
        "structural",
        "heuristic",
        "confidence",
        "execution",
    ]
    assert d.final_tier is Tier.LOCAL and not d.escalated


async def test_uncertain_soft_failure_escalates_to_sonnet_and_discards_draft():
    router, backends = make(score=0.5, local=["I'm not sure, sorry."])
    events = [e async for e in router.astream("hi")]
    texts = "".join(e.text for e in events if isinstance(e, TextDeltaEvent))
    assert "not sure" not in texts and GOOD in texts  # local draft never reached the caller
    d = end(events).decision
    assert d.tiers_attempted == [Tier.LOCAL, Tier.SONNET]
    assert d.attempts[0].wasted and not d.attempts[1].wasted
    assert d.attempts[1].role == "escalation"
    assert "heuristic" in (d.escalation_reason or "")
    assert d.final_tier is Tier.SONNET and d.success


async def test_hard_failure_skips_to_opus():
    router, backends = make(score=0.5, local=[Reply(tool_calls=[("rm_rf", {})])])
    d = (await router.acomplete("hi", tools=[WEATHER])).decision
    assert d.tiers_attempted == [Tier.LOCAL, Tier.OPUS]
    assert "nonexistent tool" in (d.escalation_reason or "")


async def test_hard_failure_goes_to_sonnet_when_configured():
    router, _ = make(
        score=0.5, local=[Reply(tool_calls=[("rm_rf", {})])], escalate_hard_to_opus=False
    )
    d = (await router.acomplete("hi", tools=[WEATHER])).decision
    assert d.tiers_attempted == [Tier.LOCAL, Tier.SONNET]


async def test_high_score_failure_skips_to_opus_even_if_soft():
    # Only cascade runs local for scores above opus_threshold.
    router, _ = make(score=0.9, local=["I'm not sure."])
    d = (await router.acomplete("hi", strategy="cascade")).decision
    assert d.tiers_attempted == [Tier.LOCAL, Tier.OPUS]


async def test_easy_zone_failure_discards_streamed_text():
    router, _ = make(score=0.1, local=[Reply("partial answ", stop_reason="max_tokens")])
    events = [e async for e in router.astream("hi")]
    first_end = next(e for e in events if isinstance(e, AttemptEndEvent))
    assert first_end.discarded and not first_end.ok and first_end.tier is Tier.LOCAL
    assert end(events).decision.final_tier is Tier.SONNET


async def test_sonnet_failure_goes_to_opus_then_gives_up():
    router, backends = make(score=0.7, sonnet=[""], opus=[""])
    events = [e async for e in router.astream("hi")]
    d = end(events).decision
    assert d.tiers_attempted == [Tier.SONNET, Tier.OPUS]
    assert not d.success and end(events).response is not None  # best effort returned


async def test_backend_errors_escalate_softly_and_all_errors_raise():
    router, _ = make(score=0.5, local=[BackendTimeout("slow")])
    d = (await router.acomplete("hi")).decision
    assert d.tiers_attempted == [Tier.LOCAL, Tier.SONNET]  # infra error -> soft -> Sonnet
    assert "BackendTimeout" in (d.attempts[0].error or "")

    down = BackendUnavailable("down")
    router, _ = make(score=0.5, local=[down], sonnet=[down], opus=[down], cloud_retries=0)
    with pytest.raises(RoutingError) as info:
        await router.acomplete("hi")
    assert len(info.value.decision.attempts) == 3


async def test_cloud_retry_on_retryable_error():
    router, backends = make(score=0.7, sonnet=[BackendUnavailable("blip"), GOOD])
    d = (await router.acomplete("hi")).decision
    assert d.success and len(d.attempts) == 1
    assert len(backends[Tier.SONNET].requests) == 2


async def test_max_attempts_is_respected():
    router, backends = make(score=0.5, local=["I'm not sure."], sonnet=[""], max_attempts=2)
    d = (await router.acomplete("hi")).decision
    assert len(d.attempts) == 2 and calls(backends)["opus"] == 0


# --- other strategies -------------------------------------------------------------


async def test_pre_route_never_escalates_on_validation():
    router, backends = make(score=0.5, local=["I'm not sure."])
    d = (await router.acomplete("hi", strategy="pre_route")).decision
    assert d.tiers_attempted == [Tier.LOCAL] and not d.success
    assert d.attempts[0].validator_results[-1].name == "heuristic"  # logged, not acted on


async def test_pre_route_falls_back_on_crash():
    router, _ = make(score=0.5, local=[BackendTimeout("slow")])
    d = (await router.acomplete("hi", strategy="pre_route")).decision
    assert d.tiers_attempted == [Tier.LOCAL, Tier.SONNET]
    router, _ = make(score=0.5, local=[BackendTimeout("slow")], pre_route_fallback_on_error=False)
    with pytest.raises(RoutingError):
        await router.acomplete("hi", strategy="pre_route")


async def test_pre_route_hard_goes_to_cloud():
    router, _ = make(score=0.9)
    assert (await router.acomplete("hi", strategy="pre_route")).decision.final_tier is Tier.OPUS


async def test_cascade_always_starts_local():
    router, backends = make(score=0.95)
    d = (await router.acomplete("hi", strategy="cascade")).decision
    assert d.tiers_attempted == [Tier.LOCAL] and d.attempts[0].buffered


async def test_force_tier_overrides_and_never_escalates():
    router, _ = make(score=0.1, opus=[""])
    d = (await router.acomplete("hi", force_tier="opus")).decision
    assert d.forced and d.tiers_attempted == [Tier.OPUS] and "forced to opus" in d.reasons


# --- escalation payload ------------------------------------------------------------


async def test_revise_mode_sends_the_draft():
    router, backends = make(score=0.5, local=["I'm not sure about 41."], escalation_mode="revise")
    await router.acomplete("what is 6*7?")
    sent = backends[Tier.SONNET].requests[0].messages[-1]
    assert "<draft>" in sent.text() and "41" in sent.text() and "heuristic" in sent.text()

    router, backends = make(score=0.5, local=["I'm not sure."])
    await router.acomplete("what is 6*7?")
    assert backends[Tier.SONNET].requests[0].messages[-1].text() == "what is 6*7?"


def test_revise_request_appends_new_turn_after_assistant():
    req = CompletionRequest(
        messages=[Message(role="user", content="q"), Message(role="assistant", content="a")]
    )
    from hybrid_router.testing import ScriptedBackend as _  # noqa: F401
    from hybrid_router.types import CompletionResponse, TextBlock, Usage

    draft = CompletionResponse(
        content=[TextBlock(text="d")],
        stop_reason="end_turn",
        usage=Usage(),
        model="m",
        backend="b",
        tier=Tier.LOCAL,
        latency_ms=1,
    )
    out = revise_request(req, draft, None)
    assert out.messages[-1].role == "user" and len(out.messages) == 3


# --- sessions ----------------------------------------------------------------------


async def test_stickiness_after_escalation():
    scorer = FixedScorer(0.5)
    router, _ = make(score=0.5, local=["I'm not sure.", GOOD, GOOD, GOOD, GOOD])
    router.scorer = scorer
    for _ in range(5):
        await router.acomplete("hi", session_id="s1")
    assert [c.sticky_remaining for c in scorer.contexts] == [0, 3, 2, 1, 0]
    # Other sessions are unaffected.
    await router.acomplete("hi", session_id="s2")
    assert scorer.contexts[-1].sticky_remaining == 0


async def test_step_index_increments_per_session():
    rec = Recorder()
    router, _ = make(score=0.1, observers=[rec])
    for sid in ("a", "a", "b", "a"):
        await router.acomplete("hi", session_id=sid)
    assert [(d.session_id, d.step_index) for d in rec.routes] == [
        ("a", 0),
        ("a", 1),
        ("b", 0),
        ("a", 2),
    ]


async def test_local_tool_error_raises_next_score_and_notifies():
    rec = Recorder()
    scorer = FixedScorer(0.1)
    router, _ = make(
        score=0.1, local=[Reply(tool_calls=[("get_weather", {"city": "x"})])], observers=[rec]
    )
    router.scorer = scorer
    first = await router.acomplete("weather?", tools=[WEATHER], session_id="s")
    tool_id = first.tool_calls[0].id
    history = [
        Message(role="user", content="weather?"),
        first.response.as_message(),
        Message(
            role="user",
            content=[ToolResultBlock(tool_use_id=tool_id, content="404", is_error=True)],
        ),
    ]
    await router.acomplete(history, tools=[WEATHER], session_id="s")
    assert scorer.contexts[-1].prior_local_tool_error
    assert rec.tool_errors == [[first.decision.attempts[0].call_id]]


# --- sync wrapper and config ---------------------------------------------------------


def test_sync_complete_reuses_one_loop():
    router, backends = make(score=0.1)
    a = router.complete("one")
    b = router.complete("two")
    assert a.text == b.text == GOOD
    router.close()


async def test_sync_complete_refuses_inside_a_loop():
    router, _ = make()
    with pytest.raises(RuntimeError, match="acomplete"):
        router.complete("hi")


async def test_observer_errors_do_not_break_routing():
    class Broken:
        async def on_route(self, *a):
            raise ValueError("boom")

        async def on_local_tool_error(self, *a):
            raise ValueError("boom")

    router, _ = make(observers=[Broken()])
    assert (await router.acomplete("hi")).decision.success


async def test_local_concurrency_is_limited():
    slow = [Reply(GOOD, latency_s=0.05)] * 4
    router, _ = make(score=0.1, local=slow, local_concurrency=1)
    results = await asyncio.gather(*(router.acomplete("hi") for _ in range(3)))
    queued = sorted(r.decision.attempts[0].queue_ms for r in results)
    assert queued[0] < 20 and queued[-1] >= 80  # third call waited for two others


async def test_hot_swap_config_and_hash_recorded():
    rec = Recorder()
    router, _ = make(score=0.5, observers=[rec])
    await router.acomplete("hi")
    new_cfg: RouterConfig = router.config.model_copy(
        update={"routing": router.config.routing.model_copy(update={"hard_threshold": 0.4})}
    )
    router.update_config(new_cfg)
    await router.acomplete("hi")
    assert rec.routes[0].zone is Zone.UNCERTAIN and rec.routes[1].zone is Zone.HARD
    assert rec.routes[0].config_hash != rec.routes[1].config_hash


async def test_real_rule_scorer_end_to_end():
    cfg = load_config(environ={})
    backends = {t: ScriptedBackend([GOOD], tier=t) for t in Tier}
    router = HybridRouter(cfg, backends=backends)
    easy = await router.acomplete("Is this positive? 'great movie'", task_type="classify")
    hard = await router.acomplete(
        "Prove step by step that the algorithm's complexity is optimal",
        task_type="multi_step_reasoning",
    )
    assert easy.decision.zone is Zone.EASY and easy.decision.final_tier is Tier.LOCAL
    assert hard.decision.zone is Zone.HARD
    assert hard.decision.final_tier is not None and hard.decision.final_tier.is_cloud
    assert hard.decision.reasons
