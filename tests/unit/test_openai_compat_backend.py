import json

import httpx2
import pytest

from hybrid_router.backends.base import (
    BackendAuthError,
    BackendBadResponse,
    BackendRateLimited,
    BackendRequestRejected,
    BackendUnavailable,
)
from hybrid_router.backends.openai_compat import (
    OpenAICompatBackend,
    OpenAICompatConfig,
    ReasoningDetailsMerger,
    ToolCallAccumulator,
    add_cache_breakpoints,
    parse_usage,
    to_openai_messages,
)
from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    ImageBlock,
    Message,
    MessageEndEvent,
    OpaqueBlock,
    TextBlock,
    TextDeltaEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    Tier,
    ToolChoice,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    ToolUseEvent,
)

WEATHER = ToolSpec(
    name="get_weather",
    description="weather",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
)

# How the default config ships the Claude tiers.
CLAUDE = {"sampling_params": False, "forced_tool_choice": False}


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def make_backend(handler, **cfg) -> tuple[OpenAICompatBackend, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    client = httpx2.AsyncClient(
        base_url="https://openrouter.test/api/v1", transport=httpx2.MockTransport(record)
    )
    backend = OpenAICompatBackend(OpenAICompatConfig(**cfg), tier=Tier.SONNET, http_client=client)
    return backend, seen


def sse(body: str):
    return lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=body.encode()
    )


def error(status: int, message: str = "nope"):
    return lambda request: httpx2.Response(
        status, json={"error": {"code": status, "message": message}}
    )


def final(events) -> CompletionResponse:
    assert isinstance(events[-1], MessageEndEvent)
    return events[-1].response


def ask(text: str = "hi", **kw) -> CompletionRequest:
    return CompletionRequest(messages=[Message(role="user", content=text)], **kw)


# --- request conversion -------------------------------------------------------


def test_message_conversion():
    reasoning = [{"type": "reasoning.text", "text": "t", "signature": "s", "index": 0}]
    req = CompletionRequest(
        system="be brief",
        messages=[
            Message(
                role="user",
                content=[TextBlock(text="look"), ImageBlock(media_type="image/png", data="aGk=")],
            ),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="t"),  # plain text form isn't replayed
                    OpaqueBlock(provider="openai_compat", data={"reasoning_details": reasoning}),
                    OpaqueBlock(provider="other", data={"x": 1}),
                    ToolUseBlock(id="t1", name="get_weather", input={"city": "Oslo"}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="t1", content="boom", is_error=True),
                    TextBlock(text="retry"),
                ],
            ),
        ],
    )
    assert to_openai_messages(req) == [
        {"role": "system", "content": "be brief"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
            ],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
                }
            ],
            "reasoning_details": reasoning,
        },
        {"role": "tool", "tool_call_id": "t1", "content": "ERROR: boom"},
        {"role": "user", "content": "retry"},
    ]


def test_cache_breakpoints_mark_system_and_latest_message():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "tool", "tool_call_id": "t", "content": "result"},
    ]
    add_cache_breakpoints(msgs)
    eph = {"type": "ephemeral"}
    assert msgs[0]["content"] == [{"type": "text", "text": "sys", "cache_control": eph}]
    assert msgs[1]["content"] == "first"  # only the newest one is marked
    assert msgs[3]["content"] == [{"type": "text", "text": "result", "cache_control": eph}]


def test_payload_for_claude_tier_drops_unsupported_params():
    backend, _ = make_backend(sse(""), **CLAUDE)
    payload, warnings = backend.build_payload(
        ask(
            system="sys",
            tools=[WEATHER],
            tool_choice=ToolChoice(type="tool", name="get_weather"),
            temperature=0.3,
            response_schema={"type": "object"},
        )
    )
    assert payload["model"] == "anthropic/claude-sonnet-5.5"
    assert payload["stream"] is True and payload["max_tokens"] == 16000
    assert payload["tool_choice"] == "auto" and "temperature" not in payload
    assert len(warnings) == 2
    assert payload["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert payload["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert "logprobs" not in payload and "reasoning" not in payload


def test_payload_for_permissive_model_and_passthroughs():
    backend, _ = make_backend(
        sse(""),
        model="openai/gpt-oss-120b",
        logprobs=True,
        reasoning={"effort": "low"},
        provider={"order": ["groq"]},
        extra_body={"seed": 1},
    )
    payload, warnings = backend.build_payload(
        ask(tools=[WEATHER], tool_choice=ToolChoice(type="any"), temperature=0.3),
        logprobs=True,
    )
    assert payload["tool_choice"] == "required" and payload["temperature"] == 0.3
    assert payload["logprobs"] is True
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["provider"] == {"order": ["groq"]} and payload["seed"] == 1
    assert isinstance(payload["messages"][0]["content"], str)  # no caching for non-Claude
    assert warnings == []


def test_schema_falls_back_to_system_prompt():
    backend, _ = make_backend(sse(""), structured_outputs=False, prompt_caching=False)
    payload, _ = backend.build_payload(ask(system="sys", response_schema={"type": "object"}))
    assert "response_format" not in payload
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"].startswith("sys\n\nRespond with only a JSON")


# --- stream helpers -----------------------------------------------------------


def test_tool_call_accumulator_reassembles_fragments():
    acc = ToolCallAccumulator()
    acc.add([{"index": 0, "id": "a", "function": {"name": "f", "arguments": '{"x"'}}])
    acc.add([{"index": 1, "id": "b", "function": {"name": "g", "arguments": "not json"}}])
    acc.add([{"index": 0, "function": {"arguments": ": 1}"}}])
    first, second = acc.result()
    assert (first.id, first.name, first.input, first.raw_arguments) == ("a", "f", {"x": 1}, None)
    assert second.input == {} and second.raw_arguments == "not json"


def test_reasoning_details_merger():
    m = ReasoningDetailsMerger()
    m.add([{"type": "reasoning.text", "text": "a", "index": 0}])
    m.add([{"type": "reasoning.text", "text": "b", "signature": None, "index": 0}])
    m.add([{"type": "reasoning.text", "signature": "sig", "index": 0}])
    m.add([{"type": "reasoning.encrypted", "data": "xx", "index": 1}])
    m.add(None)
    assert m.result() == [
        {"type": "reasoning.text", "text": "ab", "index": 0, "signature": "sig"},
        {"type": "reasoning.encrypted", "data": "xx", "index": 1},
    ]


def test_parse_usage_separates_cached_tokens_and_reads_cost():
    usage, cost = parse_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 5,
            "cost": 0.0012,
            "prompt_tokens_details": {"cached_tokens": 800, "cache_write_tokens": 150},
        }
    )
    assert (usage.input_tokens, usage.cache_read_tokens, usage.cache_write_tokens) == (50, 800, 150)
    assert usage.total_input_tokens == 1000 and cost == 0.0012
    assert parse_usage(None)[1] is None


# --- streaming responses --------------------------------------------------------


async def test_text_stream_with_usage_and_cost(fixture_text):
    backend, seen = make_backend(sse(fixture_text("openrouter_text.sse")))
    events = [e async for e in backend.stream(ask())]
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["po", "ng"]
    resp = final(events)
    assert resp.text == "pong" and resp.stop_reason == "end_turn"
    assert resp.usage.input_tokens == 12 and resp.usage.cache_read_tokens == 900
    assert resp.reported_cost_usd == 0.000248
    assert resp.raw is not None and resp.raw["id"] == "gen-1"
    assert seen[0].headers["authorization"] == "Bearer sk-or-test"
    assert seen[0].url.path == "/api/v1/chat/completions"
    assert json.loads(seen[0].content)["stream_options"] == {"include_usage": True}


async def test_tool_call_with_reasoning_round_trips(fixture_text):
    backend, _ = make_backend(sse(fixture_text("openrouter_tool_reasoning.sse")))
    events = [e async for e in backend.stream(ask("weather?", tools=[WEATHER]))]
    assert "".join(e.thinking for e in events if isinstance(e, ThinkingDeltaEvent)) == (
        "Need the weather tool."
    )
    assert [e.block.input for e in events if isinstance(e, ToolUseEvent)] == [{"city": "Tokyo"}]
    resp = final(events)
    assert resp.stop_reason == "tool_use" and resp.model == "anthropic/claude-opus-5.5"
    assert resp.reported_cost_usd == 0.002

    # Appending the response to history replays the signed reasoning to Claude.
    history = CompletionRequest(messages=[Message(role="user", content="q"), resp.as_message()])
    assistant = to_openai_messages(history)[1]
    assert assistant["reasoning_details"] == [
        {
            "type": "reasoning.text",
            "text": "Need the weather tool.",
            "format": "anthropic-claude-v1",
            "index": 0,
            "signature": "sig-xyz",
        }
    ]
    assert assistant["content"] == "Checking."


async def test_local_server_logprobs_and_length(fixture_text):
    backend, _ = make_backend(
        sse(fixture_text("llamacpp_logprobs.sse")), model="qwen3-8b", logprobs=True, api_key_env=""
    )
    resp = await backend.generate(ask(), logprobs=True)
    assert resp.token_logprobs == [-0.2, -0.4]
    assert resp.stop_reason == "max_tokens"
    assert resp.reported_cost_usd is None and resp.usage.input_tokens == 15


# --- errors ---------------------------------------------------------------------


async def test_midstream_error_raises(fixture_text):
    backend, _ = make_backend(sse(fixture_text("openrouter_midstream_error.sse")))
    with pytest.raises(BackendBadResponse, match="Provider disconnected"):
        await backend.generate(ask())


@pytest.mark.parametrize(
    ("status", "exc", "hint"),
    [
        (401, BackendAuthError, "OPENROUTER_API_KEY"),
        (402, BackendRequestRejected, "credits"),
        (429, BackendRateLimited, None),
        (400, BackendRequestRejected, None),
        (502, BackendUnavailable, None),
    ],
)
async def test_http_errors_map_to_backend_errors(status, exc, hint):
    backend, _ = make_backend(error(status))
    with pytest.raises(exc, match=hint) as info:
        await backend.generate(ask())
    assert info.value.status_code == status and info.value.backend == "sonnet:openai_compat"


async def test_missing_key_fails_fast_without_a_request(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    backend, seen = make_backend(sse(""))
    with pytest.raises(BackendAuthError, match="OPENROUTER_API_KEY"):
        await backend.generate(ask())
    assert seen == []
    assert not (await backend.health()).ok


async def test_connection_refused():
    def refuse(request):
        raise httpx2.ConnectError("refused", request=request)

    backend, _ = make_backend(refuse)
    with pytest.raises(BackendUnavailable, match="cannot reach"):
        await backend.generate(ask())


async def test_empty_stream_is_bad_response():
    backend, _ = make_backend(sse("data: [DONE]\n\n"))
    with pytest.raises(BackendBadResponse, match="without any output"):
        await backend.generate(ask())


# --- health ---------------------------------------------------------------------


async def test_health_checks_model_exists():
    def models(request):
        assert request.url.path == "/api/v1/models"
        return httpx2.Response(200, json={"data": [{"id": "anthropic/claude-sonnet-5.5"}]})

    ok, _ = make_backend(models)
    assert (await ok.health()).ok
    typo, _ = make_backend(models, model="anthropic/claude-sonet-5.5")
    status = await typo.health()
    assert not status.ok and "not offered" in status.detail
