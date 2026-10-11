import time

import pytest

from hybrid_router.config import RoutingConfig, RuleScorerConfig
from hybrid_router.decision import Zone
from hybrid_router.scoring import RuleScorer, ScoringContext
from hybrid_router.scoring.features import ramp, schema_units
from hybrid_router.types import CompletionRequest, Message, Tier, ToolSpec
from hybrid_router.zones import cloud_tier_for, planned_tier, zone_for

NO_CTX = ScoringContext()


def req(text="hello", **kw) -> CompletionRequest:
    return CompletionRequest(messages=[Message(role="user", content=text)], **kw)


def score(r: CompletionRequest, ctx: ScoringContext = NO_CTX, **cfg):
    return RuleScorer(RuleScorerConfig(**cfg)).score(r, ctx)


def test_ramp_and_schema_units():
    assert ramp(0, 10, 20) == 0 and ramp(15, 10, 20) == 0.5 and ramp(99, 10, 20) == 1
    flat = {"type": "object", "properties": {"a": {"type": "string"}}}
    nested = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": flat},
            "kind": {"enum": ["a", "b"]},
            "x": {"anyOf": [{"type": "string"}, {"type": "number"}]},
        },
        "required": ["items", "kind"],
    }
    assert schema_units(flat) == 2  # one property + its nesting level
    assert schema_units(nested) > 8


def test_easy_request_scores_low_with_reasons():
    r = score(req("classify this tweet as positive or negative", task_type="classify"))
    assert r.score < 0.1
    assert r.features["task_type"] == 0.05


def test_each_feature_raises_the_score():
    base = score(req("do the thing")).score
    assert score(req("x " * 40000)).score > base  # tokens
    many_tools = [ToolSpec(name=f"t{i}") for i in range(8)]
    assert score(req("do the thing", tools=many_tools)).score > base
    assert score(req("do the thing", task_type="plan")).score > base
    assert score(req("please refactor this to avoid the race condition")).score > base
    schema = {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(30)}}
    assert score(req("do the thing", response_schema=schema)).score > base


def test_keyword_only_matches_latest_user_text():
    r = CompletionRequest(
        messages=[
            Message(role="user", content="prove the theorem"),
            Message(role="assistant", content="done"),
            Message(role="user", content="thanks, now summarize it"),
        ]
    )
    assert score(r).features["keywords"] == 0


def test_reasons_explain_contributions():
    r = score(req("prove it step by step", task_type="multi_step_reasoning"))
    assert any(reason.startswith("keywords: prove") for reason in r.reasons)
    assert any("task_type=multi_step_reasoning" in reason for reason in r.reasons)


def test_stickiness_boost_decays():
    r = req("hi", task_type="chat")
    plain = score(r).score
    boosts = [score(r, ScoringContext(sticky_remaining=n)).score - plain for n in (3, 2, 1)]
    assert boosts[0] == pytest.approx(0.25, abs=1e-3)
    assert boosts[0] > boosts[1] > boosts[2] > 0
    assert any("sticky" in x for x in score(r, ScoringContext(sticky_remaining=1)).reasons)


def test_prior_tool_error_boost():
    r = req("hi", task_type="chat")
    bumped = score(r, ScoringContext(prior_local_tool_error=True))
    assert bumped.score == pytest.approx(score(r).score + 0.2, abs=1e-3)
    assert any("tool call errored" in x for x in bumped.reasons)


def test_weights_come_from_config_and_replay_matches():
    r = req("refactor", task_type="code")
    live = score(r)
    no_keywords = RuleScorer(
        RuleScorerConfig(weights={"tokens": 0.6, "tools": 0.4, "task_type": 0.8, "schema": 0.4})
    )
    # Replaying stored features with new weights needs no request.
    assert no_keywords.score_features(live.features).score != live.score
    assert RuleScorer().score_features(live.features).score == live.score


@pytest.mark.parametrize(
    ("text", "task_type", "zone"),
    [
        ("is this review positive? 'loved it'", "classify", Zone.EASY),
        ("pull the dates out of this email", "extract", Zone.EASY),
        ("hey, how's it going?", "chat", Zone.EASY),
        ("write a function that parses ISO dates", "code", Zone.UNCERTAIN),
        ("what should I cook tonight?", None, Zone.UNCERTAIN),
        ("review this diff for bugs", "code_review", Zone.UNCERTAIN),
        ("prove step by step that the algorithm is optimal", "multi_step_reasoning", Zone.HARD),
        ("plan out the migration and its trade-offs", "plan", Zone.HARD),
    ],
)
def test_default_calibration(text, task_type, zone):
    s = score(req(text, task_type=task_type)).score
    assert zone_for(s, RoutingConfig()) is zone, s


def test_score_is_clamped():
    r = req("prove the theorem step by step", task_type="multi_step_reasoning")
    assert score(r, ScoringContext(sticky_remaining=3, prior_local_tool_error=True)).score <= 1


def test_scoring_is_fast_on_a_big_context():
    r = req("word " * 30000, tools=[ToolSpec(name=f"t{i}") for i in range(20)])
    scorer = RuleScorer()
    t0 = time.perf_counter()
    for _ in range(10):
        scorer.score(r, NO_CTX)
    assert (time.perf_counter() - t0) / 10 < 0.01  # well under 10 ms per call


@pytest.mark.parametrize(
    ("s", "zone", "tier"),
    [
        (0.0, Zone.EASY, Tier.LOCAL),
        (0.2499, Zone.EASY, Tier.LOCAL),
        (0.25, Zone.UNCERTAIN, Tier.LOCAL),
        (0.65, Zone.UNCERTAIN, Tier.LOCAL),
        (0.66, Zone.HARD, Tier.SONNET),
        (0.85, Zone.HARD, Tier.SONNET),
        (0.86, Zone.HARD, Tier.OPUS),
    ],
)
def test_zone_boundaries(s, zone, tier):
    cfg = RoutingConfig()
    assert zone_for(s, cfg) is zone
    assert planned_tier(s, cfg) is tier


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError, match="easy <= hard <= opus"):
        RoutingConfig(easy_threshold=0.7, hard_threshold=0.5)
    assert cloud_tier_for(0.9, RoutingConfig()) is Tier.OPUS
