"""Live smoke tests against real backends.

Skipped unless Ollama answers at the configured URL with the configured model
pulled AND ANTHROPIC_API_KEY is set. Each cloud tier makes two small calls.
"""

from __future__ import annotations

import os

import httpx2
import pytest

from hybrid_router.backends import create_backend
from hybrid_router.config import load_config
from hybrid_router.types import CompletionRequest, Message, Tier, ToolSpec

pytestmark = pytest.mark.integration

CONFIG = load_config()


def _ollama_ready() -> bool:
    local = CONFIG.tiers[Tier.LOCAL]
    base_url = getattr(local, "base_url", None)
    if base_url is None:
        return False
    try:
        tags = httpx2.get(f"{base_url}/api/tags", timeout=2.0).json().get("models", [])
    except (httpx2.HTTPError, ValueError):
        return False
    names = {m.get("name") for m in tags}
    return local.model in names or f"{local.model}:latest" in names


if not (_ollama_ready() and os.environ.get("ANTHROPIC_API_KEY")):
    pytest.skip(
        "needs a running Ollama with the local model pulled and ANTHROPIC_API_KEY",
        allow_module_level=True,
    )

WEATHER = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


@pytest.mark.parametrize("tier", list(Tier))
async def test_plain_reply(tier: Tier):
    async with create_backend(CONFIG.tiers[tier], tier=tier) as backend:
        resp = await backend.generate(
            CompletionRequest(
                messages=[Message(role="user", content="Reply with exactly one word: pong")],
                max_tokens=2000,
            )
        )
    assert "pong" in resp.text.lower()
    assert resp.usage.output_tokens > 0 and resp.latency_ms > 0
    assert resp.tier is tier


@pytest.mark.parametrize("tier", list(Tier))
async def test_tool_call(tier: Tier):
    async with create_backend(CONFIG.tiers[tier], tier=tier) as backend:
        resp = await backend.generate(
            CompletionRequest(
                messages=[Message(role="user", content="What's the weather in Tokyo?")],
                tools=[WEATHER],
                max_tokens=2000,
            )
        )
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "get_weather"
    assert "tokyo" in str(resp.tool_calls[0].input.get("city", "")).lower()


async def test_claude_count_tokens_is_exact():
    async with create_backend(CONFIG.tiers[Tier.SONNET], tier=Tier.SONNET) as backend:
        n = await backend.count_tokens(
            CompletionRequest(messages=[Message(role="user", content="hello " * 50)])
        )
    assert 40 < n < 200
