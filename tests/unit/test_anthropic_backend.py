import json

import anthropic
import httpx2
import pytest

from hybrid_router.backends.anthropic import (
    REFUSAL_FALLBACK_BETA,
    AnthropicBackend,
    AnthropicConfig,
    to_anthropic_messages,
)
from hybrid_router.backends.base import (
    BackendAuthError,
    BackendRateLimited,
    BackendRequestRejected,
    BackendUnavailable,
)
from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    ImageBlock,
    Message,
    MessageEndEvent,
    OpaqueBlock,
    RedactedThinkingBlock,
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


def make_backend(handler, **cfg) -> tuple[AnthropicBackend, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    sdk = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="http://anthropic.test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(record)),
    )
    backend = AnthropicBackend(AnthropicConfig(**cfg), tier=Tier.SONNET, client=sdk)
    return backend, seen


def sse(body: str):
    return lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, content=body.encode()
    )


def error(status: int, kind: str, message: str = "nope"):
    return lambda request: httpx2.Response(
        status, json={"type": "error", "error": {"type": kind, "message": message}}
    )


def final(events) -> CompletionResponse:
    assert isinstance(events[-1], MessageEndEvent)
    return events[-1].response


def ask(text: str = "hi", **kw) -> CompletionRequest:
    return CompletionRequest(messages=[Message(role="user", content=text)], **kw)


# --- request building -----------------------------------------------------------


def test_default_params_enable_caching_and_refusal_fallback():
    backend = AnthropicBackend(AnthropicConfig(), tier=Tier.SONNET)
    params, warnings = backend.build_params(ask(system="sys", tools=[WEATHER]))
    assert params["model"] == "claude-sonnet-5-5"
    assert params["max_tokens"] == 16000
    assert params["system"] == "sys"
    assert params["cache_control"] == {"type": "ephemeral"}
    assert params["betas"] == [REFUSAL_FALLBACK_BETA] and params["fallbacks"] == "default"
    assert params["tools"][0]["eager_input_streaming"] is True
    assert "thinking" not in params and "output_config" not in params
    assert warnings == []


def test_current_model_constraints_are_respected():
    backend = AnthropicBackend(AnthropicConfig(), tier=Tier.SONNET)
    params, warnings = backend.build_params(
        ask(
            tools=[WEATHER],
            tool_choice=ToolChoice(type="tool", name="get_weather"),
            temperature=0.3,
        )
    )
    assert params["tool_choice"] == {"type": "auto"}
    assert "temperature" not in params and "extra_body" not in params
    assert len(warnings) == 2

    legacy = AnthropicBackend(
        AnthropicConfig(forced_tool_choice=True, sampling_params=True), tier=Tier.SONNET
    )
    params, warnings = legacy.build_params(
        ask(
            tools=[WEATHER],
            tool_choice=ToolChoice(type="tool", name="get_weather"),
            temperature=0.3,
        )
    )
    assert params["tool_choice"] == {"type": "tool", "name": "get_weather"}
    assert params["extra_body"] == {"temperature": 0.3}
    assert warnings == []


def test_schema_effort_and_thinking_config():
    backend = AnthropicBackend(
        AnthropicConfig(effort="medium", thinking="adaptive", thinking_display="summarized"),
        tier=Tier.OPUS,
    )
    params, _ = backend.build_params(ask(response_schema={"type": "object"}))
    assert params["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}},
        "effort": "medium",
    }
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"}

    no_so = AnthropicBackend(AnthropicConfig(structured_outputs=False), tier=Tier.SONNET)
    params, _ = no_so.build_params(ask(system="sys", response_schema={"type": "object"}))
    assert "output_config" not in params
    assert params["system"].startswith("sys") and '{"type": "object"}' in params["system"]


def test_count_params_omit_streaming_only_fields():
    backend = AnthropicBackend(AnthropicConfig(), tier=Tier.SONNET)
    params, _ = backend.build_params(
        ask(tools=[WEATHER], tool_choice=ToolChoice(type="any")), for_count=True
    )
    for key in ("max_tokens", "betas", "fallbacks", "cache_control", "tool_choice"):
        assert key not in params
    assert "eager_input_streaming" not in params["tools"][0]


def test_message_conversion_round_trips_claude_blocks_and_drops_local_thinking():
    opaque = {"type": "server_tool_use", "id": "srv1", "name": "web_search", "input": {}}
    req = CompletionRequest(
        messages=[
            Message(role="user", content=[ImageBlock(media_type="image/png", data="aGk=")]),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="local musing"),  # unsigned -> dropped
                    ThinkingBlock(thinking="", signature="sig"),
                    RedactedThinkingBlock(data="enc"),
                    OpaqueBlock(provider="anthropic", data=opaque),
                    OpaqueBlock(provider="other", data={"type": "x"}),  # dropped
                    ToolUseBlock(id="t1", name="get_weather", input={"city": "Oslo"}),
                ],
            ),
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="t1", content=[TextBlock(text="3C")])],
            ),
        ]
    )
    out = to_anthropic_messages(req)
    assert out[0]["content"][0]["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": "aGk=",
    }
    assert [b["type"] for b in out[1]["content"]] == [
        "thinking",
        "redacted_thinking",
        "server_tool_use",
        "tool_use",
    ]
    assert out[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": [{"type": "text", "text": "3C"}],
        "is_error": False,
    }


# --- streaming responses -----------------------------------------------------------


async def test_stream_tool_use_with_thinking_and_cache_usage(fixture_text):
    backend, seen = make_backend(sse(fixture_text("anthropic_tool_use.sse")))
    events = [e async for e in backend.stream(ask("weather in Tokyo?", tools=[WEATHER]))]

    assert [e.text for e in events if isinstance(e, TextDeltaEvent)] == ["Let me check ", "that."]
    assert [e.thinking for e in events if isinstance(e, ThinkingDeltaEvent)] == [
        "Need the weather tool."
    ]
    tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
    assert tool_events[0].block.input == {"city": "Tokyo"}

    resp = final(events)
    assert resp.stop_reason == "tool_use"
    assert resp.text == "Let me check that."
    thinking = resp.content[0]
    assert isinstance(thinking, ThinkingBlock) and thinking.signature == "sig-abc"
    assert resp.usage.input_tokens == 12 and resp.usage.output_tokens == 42
    assert resp.usage.cache_read_tokens == 900 and resp.usage.cache_write_tokens == 50
    assert resp.model == "claude-sonnet-5-5" and resp.tier is Tier.SONNET
    assert resp.raw == {"id": "msg_01"}

    # The response can be appended to history and replayed losslessly.
    replay = to_anthropic_messages(CompletionRequest(messages=[resp.as_message()]))
    assert replay[0]["content"][0]["signature"] == "sig-abc"

    body = json.loads(seen[0].content)
    assert body["stream"] is True and body["fallbacks"] == "default"
    assert REFUSAL_FALLBACK_BETA in seen[0].headers["anthropic-beta"]
    assert seen[0].headers["x-api-key"] == "test-key"


@pytest.mark.parametrize(
    ("status", "kind", "exc"),
    [
        (429, "rate_limit_error", BackendRateLimited),
        (529, "overloaded_error", BackendRateLimited),
        (401, "authentication_error", BackendAuthError),
        (400, "invalid_request_error", BackendRequestRejected),
        (500, "api_error", BackendUnavailable),
    ],
)
async def test_api_errors_map_to_backend_errors(status, kind, exc):
    backend, _ = make_backend(error(status, kind))
    with pytest.raises(exc) as info:
        await backend.generate(ask())
    assert info.value.backend == "sonnet:anthropic"


async def test_missing_credentials_is_an_auth_error(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent-hybrid-router-test")
    monkeypatch.setenv("HOME", "/nonexistent-hybrid-router-test")
    backend = AnthropicBackend(
        AnthropicConfig(base_url="http://anthropic.test", max_retries=0), tier=Tier.SONNET
    )
    with pytest.raises(BackendAuthError, match="ANTHROPIC_API_KEY"):
        await backend.generate(ask())
    status = await backend.health()
    assert not status.ok and "credentials" in status.detail


async def test_count_tokens_uses_the_api():
    def handler(request):
        assert request.url.path == "/v1/messages/count_tokens"
        return httpx2.Response(200, json={"input_tokens": 1234})

    backend, _ = make_backend(handler)
    assert await backend.count_tokens(ask(tools=[WEATHER])) == 1234


async def test_health_uses_models_endpoint():
    def handler(request):
        assert request.url.path == "/v1/models/claude-sonnet-5-5"
        return httpx2.Response(
            200,
            json={
                "type": "model",
                "id": "claude-sonnet-5-5",
                "display_name": "Claude Sonnet 5.5",
                "created_at": "2026-01-01T00:00:00Z",
            },
        )

    backend, _ = make_backend(handler)
    status = await backend.health()
    assert status.ok and status.detail == "Claude Sonnet 5.5"
