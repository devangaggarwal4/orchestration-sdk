"""Demo backends for ``hybrid-router web --fake-backends``.

They stream word by word with tier-typical delays so the UI behaves like the real
thing, without Ollama or an API key. The local demo model:
  * hedges ("I'm not sure...") on prompts with hard-task words or long context,
    so the validators catch it and the router escalates
  * calls the agent's search/read tools when asked where/which/find
Cloud tiers always answer and report a cost so the dashboard shows savings.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import AsyncIterator
from typing import Literal

from hybrid_router.backends.base import (
    Backend,
    BackendCapabilities,
    BackendConfig,
    HealthStatus,
)
from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    MessageEndEvent,
    MessageStartEvent,
    StreamEvent,
    TextBlock,
    TextDeltaEvent,
    Tier,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseEvent,
    Usage,
    estimate_tokens,
)

HARD_WORDS = re.compile(
    r"\b(prove|proof|architect|design|trade-?off|refactor|race|concurren|optimi[sz]e|"
    r"algorithm|plan|security|vulnerab|step by step)\w*",
    re.IGNORECASE,
)
FIND_WORDS = re.compile(r"\b(where|which file|find|search|look up|locate)\b", re.IGNORECASE)
# USD per token, only to make the demo dashboard meaningful.
DEMO_PRICE = {Tier.SONNET: (2e-6, 10e-6), Tier.OPUS: (4e-6, 20e-6)}
WORD_DELAY = {Tier.LOCAL: 0.012, Tier.SONNET: 0.02, Tier.OPUS: 0.03}
FIRST_TOKEN = {Tier.LOCAL: 0.15, Tier.SONNET: 0.5, Tier.OPUS: 0.9}


class DemoConfig(BackendConfig):
    kind: Literal["demo"] = "demo"
    model: str = "demo"


class DemoBackend(Backend):
    kind = "demo"
    config_model = DemoConfig

    def __init__(self, tier: Tier, model: str, *, speed: float = 1.0):
        super().__init__(DemoConfig(model=model), tier=tier, name=f"{tier.value}:demo")
        self.speed = speed

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(tools=True, logprobs=self.tier is Tier.LOCAL, json_schema=True)

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, detail="demo model", latency_ms=1.0)

    def _plan(self, req: CompletionRequest) -> tuple[str, list[ToolUseBlock], list[float]]:
        text = req.last_user_text()
        last = req.messages[-1] if req.messages else None
        results = [b for b in last.blocks() if isinstance(b, ToolResultBlock)] if last else []
        tool_names = {t.name for t in req.tools}
        hard = bool(HARD_WORDS.search(text)) or estimate_tokens(req) > 6000

        if results:
            found = "\n".join(r.text() for r in results)[:300]
            return (
                f"Here's what I found:\n\n{found}\n\nThat should answer your question.",
                [],
                [-0.2] * 30,
            )
        if "search_text" in tool_names and FIND_WORDS.search(text):
            word = max(re.findall(r"[A-Za-z_]{4,}", text) or ["main"], key=len)
            call = ToolUseBlock(
                id=f"toolu_demo_{time.time_ns()}", name="search_text", input={"pattern": word}
            )
            return "", [call], [-0.1] * 5
        if self.tier is Tier.LOCAL and hard:
            return (
                "I'm not sure I can do this well. It might involve several trade-offs.",
                [],
                [-1.8] * 20,
            )
        if req.response_schema is not None:
            return '{"result": "demo"}', [], [-0.1] * 6
        prefix = {
            Tier.LOCAL: "Quick answer",
            Tier.SONNET: "Detailed answer",
            Tier.OPUS: "In-depth answer",
        }[self.tier]
        body = (
            f"{prefix} from the {self.tier.value} demo model. In a real setup this tier is "
            f"{self.model}. The router picked it based on the request's difficulty score, "
            "and you can see why in the route card below this message."
        )
        return body, [], [-0.25] * 40

    async def stream(
        self, req: CompletionRequest, *, logprobs: bool = False
    ) -> AsyncIterator[StreamEvent]:
        start = time.perf_counter()
        text, calls, lps = self._plan(req)
        yield MessageStartEvent(model=self.model)
        await asyncio.sleep(FIRST_TOKEN[self.tier] / self.speed)
        ttft = (time.perf_counter() - start) * 1000
        words = text.split(" ") if text else []
        for i, w in enumerate(words):
            yield TextDeltaEvent(text=w if i == len(words) - 1 else w + " ")
            await asyncio.sleep(WORD_DELAY[self.tier] / self.speed)
        for c in calls:
            yield ToolUseEvent(block=c)
        usage = Usage(input_tokens=estimate_tokens(req), output_tokens=max(len(words) * 4 // 3, 1))
        cost = None
        if self.tier in DEMO_PRICE:
            pin, pout = DEMO_PRICE[self.tier]
            cost = usage.input_tokens * pin + usage.output_tokens * pout
        content: list[ContentBlock] = [TextBlock(text=text)] if text else []
        content += calls
        yield MessageEndEvent(
            response=CompletionResponse(
                content=content,
                stop_reason="tool_use" if calls else "end_turn",
                usage=usage,
                model=self.model,
                backend=self.name,
                tier=self.tier,
                latency_ms=(time.perf_counter() - start) * 1000,
                ttft_ms=ttft,
                token_logprobs=lps if (logprobs and self.tier is Tier.LOCAL) else None,
                reported_cost_usd=cost,
            )
        )


def demo_backends(models: dict[Tier, str], speed: float = 1.0) -> dict[Tier, Backend]:
    return {t: DemoBackend(t, models[t], speed=speed) for t in Tier}
