"""Test doubles. ``ScriptedBackend`` plays back pre-programmed responses or errors,
so routing can be unit-tested without a network."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

from hybrid_router.backends.base import (
    Backend,
    BackendCapabilities,
    BackendConfig,
    BackendError,
    HealthStatus,
)
from hybrid_router.backends.registry import register_backend
from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    MessageEndEvent,
    MessageStartEvent,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDeltaEvent,
    Tier,
    ToolUseBlock,
    ToolUseEvent,
    Usage,
)


class Reply:
    """One scripted model output."""

    def __init__(
        self,
        text: str = "",
        *,
        tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
        stop_reason: StopReason | None = None,
        logprobs: list[float] | None = None,
        usage: Usage | None = None,
        cost: float | None = None,
        latency_s: float = 0.0,
    ):
        self.text = text
        self.tool_calls = [
            ToolUseBlock(id=f"toolu_{uuid.uuid4().hex[:10]}", name=n, input=a)
            for n, a in (tool_calls or [])
        ]
        self.stop_reason: StopReason = stop_reason or (
            "tool_use" if self.tool_calls else "end_turn"
        )
        self.logprobs = logprobs
        self.usage = usage
        self.cost = cost
        self.latency_s = latency_s


Step = Reply | BackendError | str
Script = list[Step] | Callable[[CompletionRequest, int], Step]


class ScriptedConfig(BackendConfig):
    kind: Literal["scripted"] = "scripted"
    model: str = "scripted"


@register_backend
class ScriptedBackend(Backend):
    kind = "scripted"
    config_model = ScriptedConfig

    def __init__(
        self,
        script: Script | ScriptedConfig | None = None,
        *,
        tier: Tier = Tier.LOCAL,
        model: str | None = None,
        logprobs: bool = True,
        name: str | None = None,
        config: ScriptedConfig | None = None,
    ):
        if isinstance(script, ScriptedConfig):  # constructed by the registry
            config, script = script, None
        cfg = config or ScriptedConfig(model=model or f"scripted-{tier.value}")
        super().__init__(cfg, tier=tier, name=name)
        self.script: Script = script if script is not None else ["ok"]
        self._logprobs = logprobs
        self.requests: list[CompletionRequest] = []

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(tools=True, logprobs=self._logprobs, json_schema=True)

    def _next(self, req: CompletionRequest) -> Step:
        n = len(self.requests)
        self.requests.append(req)
        if callable(self.script):
            return self.script(req, n)
        if not self.script:
            raise AssertionError(f"{self.name}: script exhausted at call {n + 1}")
        return self.script[min(n, len(self.script) - 1)]

    async def stream(
        self, req: CompletionRequest, *, logprobs: bool = False
    ) -> AsyncIterator[StreamEvent]:
        step = self._next(req)
        if isinstance(step, BackendError):
            raise step
        reply = Reply(step) if isinstance(step, str) else step
        if reply.latency_s:
            await asyncio.sleep(reply.latency_s)
        yield MessageStartEvent(model=self.model)
        words = reply.text.split(" ")
        for i, w in enumerate(words):
            chunk = w if i == len(words) - 1 else w + " "
            if chunk:
                yield TextDeltaEvent(text=chunk)
        for call in reply.tool_calls:
            yield ToolUseEvent(block=call)
        content: list[ContentBlock] = []
        if reply.text:
            content.append(TextBlock(text=reply.text))
        content.extend(reply.tool_calls)
        usage = reply.usage or Usage(
            input_tokens=max(len(str(req.messages)) // 4, 1),
            output_tokens=max(len(reply.text) // 4, 1),
        )
        yield MessageEndEvent(
            response=CompletionResponse(
                content=content,
                stop_reason=reply.stop_reason,
                usage=usage,
                model=self.model,
                backend=self.name,
                tier=self.tier,
                latency_ms=reply.latency_s * 1000 + 1.0,
                ttft_ms=0.5,
                token_logprobs=(reply.logprobs if (logprobs and self._logprobs) else None),
                reported_cost_usd=reply.cost,
            )
        )

    async def health(self) -> HealthStatus:
        return HealthStatus(ok=True, detail="scripted")
