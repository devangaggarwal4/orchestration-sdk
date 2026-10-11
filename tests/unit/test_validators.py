import pytest

from hybrid_router.config import ConfidenceConfig, JudgeConfig, ValidatorsConfig
from hybrid_router.testing import ScriptedBackend
from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    Message,
    StopReason,
    TextBlock,
    Tier,
    ToolSpec,
    ToolUseBlock,
    Usage,
)
from hybrid_router.validators import (
    ConfidenceValidator,
    ExecutionValidator,
    HeuristicValidator,
    JudgeValidator,
    StructuralValidator,
    ValidationContext,
    build_chain,
)
from hybrid_router.validators.heuristic import repetition_ratio

WEATHER = ToolSpec(
    name="get_weather",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


def resp(
    text: str = "",
    calls: list[ToolUseBlock] | None = None,
    stop: StopReason = "end_turn",
    logprobs: list[float] | None = None,
) -> CompletionResponse:
    content: list[ContentBlock] = [TextBlock(text=text)] if text else []
    content += calls or []
    return CompletionResponse(
        content=content,
        stop_reason=stop,
        usage=Usage(),
        model="m",
        backend="b",
        tier=Tier.LOCAL,
        latency_ms=1,
        token_logprobs=logprobs,
    )


def ctx(response, **req_kw) -> ValidationContext:
    req = CompletionRequest(messages=[Message(role="user", content="q")], **req_kw)
    return ValidationContext(request=req, response=response)


def call(name="get_weather", args=None, raw=None):
    return ToolUseBlock(id="t", name=name, input=args or {}, raw_arguments=raw)


# --- structural -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "req_kw", "severity", "reason"),
    [
        (resp(""), {}, "hard", "empty output"),
        (resp("cut", stop="max_tokens"), {}, "soft", "truncated"),
        (
            resp(calls=[call(args={"city": "x"})], stop="max_tokens"),
            {"tools": [WEATHER]},
            "hard",
            "mid tool call",
        ),
        (resp("no", stop="refusal"), {}, "hard", "refused"),
        (resp(calls=[call("launch_rockets")]), {"tools": [WEATHER]}, "hard", "nonexistent tool"),
        (resp(calls=[call(raw="{bad")]), {"tools": [WEATHER]}, "hard", "malformed"),
        (resp(calls=[call(args={"town": "x"})]), {"tools": [WEATHER]}, "hard", "invalid"),
        (resp("not json"), {"response_schema": SCHEMA}, "hard", "invalid JSON"),
        (resp('{"name": "a"}'), {"response_schema": SCHEMA}, "hard", "schema mismatch"),
    ],
)
async def test_structural_failures(response, req_kw, severity, reason):
    result = await StructuralValidator().run(ctx(response, **req_kw))
    assert not result.ok and result.severity == severity and reason in result.reason


async def test_structural_passes_valid_json_in_a_fence_and_valid_tool_call():
    ok_json = resp('```json\n{"name": "a", "age": 3}\n```')
    assert (await StructuralValidator().run(ctx(ok_json, response_schema=SCHEMA))).ok
    ok_call = resp(calls=[call(args={"city": "Oslo"})])
    assert (await StructuralValidator().run(ctx(ok_call, tools=[WEATHER]))).ok


# --- heuristic ------------------------------------------------------------------


async def test_heuristic_refusal_repetition_short_and_empty_fields():
    h = HeuristicValidator()
    refusal = await h.run(ctx(resp("I'm not able to help with that request.")))
    assert not refusal.ok and "refusal" in refusal.reason

    loop = " ".join(["the cat sat on the mat again"] * 30)
    assert repetition_ratio(loop) > 0.8
    assert "repetition" in (await h.run(ctx(resp(loop)))).reason

    short = await h.run(ctx(resp("Too short."), task_type="summarize"))
    assert not short.ok and "short" in short.reason

    empty = await h.run(ctx(resp('{"name": "", "age": 3}'), response_schema=SCHEMA))
    assert not empty.ok and "name" in empty.reason

    fine = await h.run(ctx(resp("A complete and reasonable answer to the question asked.")))
    assert fine.ok and fine.severity is None


# --- confidence -----------------------------------------------------------------


async def test_confidence_logprob_threshold_and_skip():
    c = ConfidenceValidator(ConfidenceConfig(min_mean_logprob=-1.0))
    assert not (await c.run(ctx(resp("x", logprobs=[-2.0, -3.0])))).ok
    assert (await c.run(ctx(resp("x", logprobs=[-0.1, -0.2])))).ok
    skipped = await c.run(ctx(resp("x")))
    assert skipped.ok and skipped.skipped


async def test_two_sample_disagreement():
    c = ConfidenceValidator(ConfidenceConfig(two_sample=True))
    backend = ScriptedBackend(["completely different words entirely"])
    context = ctx(resp("Paris is the capital of France."))
    context.backend = backend
    result = await c.run(context)
    assert not result.ok and "disagree" in result.reason
    assert backend.requests[0].temperature == 0.8
    assert result.extra_usage is not None


# --- execution ------------------------------------------------------------------


async def test_execution_compiles_python_and_runs_hook():
    e = ExecutionValidator()
    bad = await e.run(ctx(resp("```python\ndef f(:\n    pass\n```")))
    assert not bad.ok and bad.severity == "hard" and "compile" in bad.reason
    good_text = "```python\ndef f(x):\n    return x + 1\n```"
    assert (await e.run(ctx(resp(good_text)))).ok
    assert (await e.run(ctx(resp("no code here")))).skipped

    context = ctx(resp(good_text))
    context.test_runner = lambda code, req: "return x + 1" in code
    assert (await e.run(context)).ok

    async def failing(code, req):
        return False

    context.test_runner = failing
    assert "tests failed" in (await e.run(context)).reason


async def test_execution_assumes_python_for_code_task_type():
    unfenced = "def f(:\n  pass"
    assert not (await ExecutionValidator().run(ctx(resp(unfenced), task_type="code"))).ok


# --- judge and chain ------------------------------------------------------------


async def test_judge():
    j = JudgeValidator(JudgeConfig(enabled=True))
    context = ctx(resp("42"))
    context.judge_backend = ScriptedBackend(["FAIL\nwrong"])
    assert not (await j.run(context)).ok
    context.judge_backend = ScriptedBackend(["PASS\nlooks right"])
    assert (await j.run(context)).ok


async def test_chain_levels_and_short_circuit():
    cfg = ValidatorsConfig()
    assert [v.name for v in build_chain(cfg, "light").validators] == ["structural"]
    assert [v.name for v in build_chain(cfg, "full").validators] == [
        "structural",
        "heuristic",
        "confidence",
        "execution",
    ]
    results = await build_chain(cfg, "full").run(ctx(resp("")))
    assert [r.name for r in results] == ["structural"]  # stopped at first failure
