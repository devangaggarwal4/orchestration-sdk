import json

import httpx2
import pytest

from hybrid_router.backends.base import (
    BackendBadResponse,
    BackendRequestRejected,
    BackendTimeout,
    BackendUnavailable,
)
from hybrid_router.backends.ollama import (
    OllamaBackend,
    OllamaConfig,
    parse_tool_call,
    to_ollama_messages,
)
from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    ImageBlock,
    Message,
    MessageEndEvent,
    TextBlock,
    TextDeltaEvent,
    ThinkingBlock,
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


def make_backend(handler, **cfg) -> tuple[OllamaBackend, list[dict]]:
    """Backend wired to a mock transport. Returns it plus the list of request bodies seen."""
    seen: list[dict] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        if request.content:
            seen.append(json.loads(request.content))
        return handler(request)

    client = httpx2.AsyncClient(
        base_url="http://ollama.test", transport=httpx2.MockTransport(record)
    )
    backend = OllamaBackend(OllamaConfig(**cfg), tier=Tier.LOCAL, http_client=client)
    return backend, seen


def ndjson(body: str, status: int = 200):
    return lambda request: httpx2.Response(status, content=body.encode())


def final(events) -> CompletionResponse:
    assert isinstance(events[-1], MessageEndEvent)
    return events[-1].response


def ask(text: str = "say pong", **kw) -> CompletionRequest:
    return CompletionRequest(messages=[Message(role="user", content=text)], **kw)


# --- request conversion -------------------------------------------------------


def test_message_conversion_covers_system_tools_results_and_images():
    req = CompletionRequest(
        system="be brief",
        messages=[
            Message(role="user", content="weather in Tokyo?"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="hmm", signature="sig"),
                    TextBlock(text="checking"),
                    ToolUseBlock(id="t1", name="get_weather", input={"city": "Tokyo"}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(tool_use_id="t1", content="boom", is_error=True),
                    TextBlock(text="and this photo?"),
                    ImageBlock(media_type="image/png", data="aGk="),
                ],
            ),
        ],
    )
    assert to_ollama_messages(req) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "weather in Tokyo?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {"id": "t1", "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
            ],
        },
        {
            "role": "tool",
            "content": "ERROR: boom",
            "tool_call_id": "t1",
            "tool_name": "get_weather",
        },
        {"role": "user", "content": "and this photo?", "images": ["aGk="]},
    ]


def test_payload_options_format_and_tool_choice_none():
    backend = OllamaBackend(OllamaConfig(num_ctx=8192, options={"seed": 7}), tier=Tier.LOCAL)
    req = ask(
        tools=[WEATHER],
        tool_choice=ToolChoice(type="none"),
        response_schema={"type": "object"},
        max_tokens=256,
        temperature=0.2,
        stop_sequences=["END"],
    )
    payload = backend.build_payload(req, logprobs=True)
    assert "tools" not in payload  # tool_choice none disables tools
    assert payload["format"] == {"type": "object"}
    assert payload["think"] is False and payload["stream"] is True
    assert payload["logprobs"] is True
    assert payload["options"] == {
        "num_ctx": 8192,
        "num_predict": 256,
        "seed": 7,
        "temperature": 0.2,
        "stop": ["END"],
    }
    no_lp = OllamaBackend(OllamaConfig(logprobs=False), tier=Tier.LOCAL)
    assert "logprobs" not in no_lp.build_payload(req, logprobs=True)


def test_parse_tool_call_handles_string_and_bad_arguments():
    ok = parse_tool_call({"function": {"name": "f", "arguments": '{"a": 1}'}})
    assert ok.input == {"a": 1} and ok.raw_arguments is None and ok.id.startswith("toolu_local_")
    bad = parse_tool_call({"id": "x", "function": {"name": "f", "arguments": "{not json"}})
    assert bad.input == {} and bad.raw_arguments == "{not json"
    arr = parse_tool_call({"function": {"name": "f", "arguments": [1, 2]}})
    assert arr.raw_arguments == "[1, 2]"


# --- responses ----------------------------------------------------------------


async def test_streams_text_and_collects_logprobs_and_usage(fixture_text):
    backend, seen = make_backend(ndjson(fixture_text("ollama_text_logprobs.ndjson")))
    events = [e async for e in backend.stream(ask(), logprobs=True)]
    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["po", "ng"]
    resp = final(events)
    assert resp.text == "pong" and resp.stop_reason == "end_turn"
    assert resp.token_logprobs == [-0.1, -0.3]
    assert (resp.usage.input_tokens, resp.usage.output_tokens) == (21, 2)
    assert resp.tier is Tier.LOCAL and resp.backend == "local:ollama"
    assert resp.ttft_ms is not None and resp.latency_ms >= resp.ttft_ms
    assert seen[0]["model"] == "qwen3:8b" and seen[0]["logprobs"] is True


async def test_tool_call_response(fixture_text):
    backend, seen = make_backend(ndjson(fixture_text("ollama_tool_call.ndjson")))
    events = [e async for e in backend.stream(ask("weather?", tools=[WEATHER]))]
    assert [e.block.name for e in events if isinstance(e, ToolUseEvent)] == ["get_weather"]
    resp = await backend.generate(ask("weather?", tools=[WEATHER]))
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].id == "call_abc"
    assert resp.tool_calls[0].input == {"city": "Tokyo"}
    assert seen[0]["tools"][0]["function"]["name"] == "get_weather"


async def test_length_done_reason_maps_to_max_tokens(fixture_text):
    backend, _ = make_backend(ndjson(fixture_text("ollama_truncated.ndjson")))
    resp = await backend.generate(ask())
    assert resp.stop_reason == "max_tokens"


async def test_thinking_is_kept_separate_from_text(fixture_text):
    backend, _ = make_backend(ndjson(fixture_text("ollama_thinking.ndjson")), think=True)
    resp = await backend.generate(ask())
    assert resp.text == "pong"
    assert isinstance(resp.content[0], ThinkingBlock)
    assert resp.content[0].signature is None


async def test_forced_tool_choice_is_downgraded_with_warning(fixture_text):
    backend, _ = make_backend(ndjson(fixture_text("ollama_tool_call.ndjson")))
    resp = await backend.generate(ask(tools=[WEATHER], tool_choice=ToolChoice(type="any")))
    assert any("cannot force" in w for w in resp.warnings)


# --- errors -------------------------------------------------------------------


async def test_midstream_error_line_raises(fixture_text):
    backend, _ = make_backend(ndjson(fixture_text("ollama_midstream_error.ndjson")))
    with pytest.raises(BackendBadResponse, match="runner has unexpectedly stopped"):
        await backend.generate(ask())


async def test_stream_without_done_raises():
    backend, _ = make_backend(ndjson('{"message":{"content":"hi"},"done":false}\n'))
    with pytest.raises(BackendBadResponse, match="before done"):
        await backend.generate(ask())


async def test_missing_model_404_says_how_to_fix():
    backend, _ = make_backend(ndjson('{"error":"model \\"qwen3:8b\\" not found"}', status=404))
    with pytest.raises(BackendRequestRejected, match="ollama pull qwen3:8b") as exc:
        await backend.generate(ask())
    assert exc.value.status_code == 404 and not exc.value.retryable


async def test_server_error_is_retryable_unavailable():
    backend, _ = make_backend(ndjson('{"error":"out of memory"}', status=500))
    with pytest.raises(BackendUnavailable, match="out of memory") as exc:
        await backend.generate(ask())
    assert exc.value.retryable


async def test_connection_refused_maps_to_unavailable():
    def refuse(request):
        raise httpx2.ConnectError("refused", request=request)

    backend, _ = make_backend(refuse)
    with pytest.raises(BackendUnavailable, match="ollama serve"):
        await backend.generate(ask())


async def test_read_timeout_maps_to_timeout():
    def slow(request):
        raise httpx2.ReadTimeout("slow", request=request)

    backend, _ = make_backend(slow)
    with pytest.raises(BackendTimeout):
        await backend.generate(ask())


async def test_overall_deadline_is_enforced(fixture_text):
    backend, _ = make_backend(ndjson(fixture_text("ollama_text_logprobs.ndjson")), timeout_s=0.0)
    with pytest.raises(BackendTimeout, match="exceeded"):
        await backend.generate(ask())


# --- health -------------------------------------------------------------------


async def test_health_reports_missing_model():
    def handler(request):
        if request.url.path == "/api/version":
            return httpx2.Response(200, json={"version": "0.13.0"})
        return httpx2.Response(200, json={"models": [{"name": "llama3.1:8b"}]})

    backend, _ = make_backend(handler)
    status = await backend.health()
    assert not status.ok and "ollama pull qwen3:8b" in status.detail

    backend_ok, _ = make_backend(handler, model="llama3.1:8b")
    assert (await backend_ok.health()).ok
