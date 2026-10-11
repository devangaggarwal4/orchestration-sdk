"""Ollama backend, talking to ``POST /api/chat`` directly over httpx2.

We use the raw HTTP API rather than the ``ollama`` package so we get every
field we route on: ``logprobs``, ``done_reason`` (truncation), token counts and
``think``. ``anthropic`` already depends on httpx2, so this adds no HTTP stack.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx2

from hybrid_router.backends.base import (
    Backend,
    BackendBadResponse,
    BackendCapabilities,
    BackendConfig,
    BackendError,
    BackendRequestRejected,
    BackendTimeout,
    BackendUnavailable,
    HealthStatus,
    elapsed_ms,
)
from hybrid_router.backends.registry import register_backend
from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    ContentBlock,
    ImageBlock,
    MessageEndEvent,
    MessageStartEvent,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDeltaEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseEvent,
    Usage,
)


class OllamaConfig(BackendConfig):
    kind: Literal["ollama"] = "ollama"
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"
    timeout_s: float = 60.0
    connect_timeout_s: float = 5.0
    # False keeps thinking models (qwen3) fast; True or "low"/"medium"/"high" enables it.
    think: bool | str = False
    keep_alive: str = "10m"
    # Context window to request. 16k suits an 8B model on a 16 GB Mac.
    num_ctx: int = 16384
    # Ask for token logprobs when the router wants them (needs a recent Ollama).
    logprobs: bool = True
    # Extra model options passed through verbatim (seed, top_p, ...).
    options: dict[str, Any] = {}


@register_backend
class OllamaBackend(Backend):
    kind = "ollama"
    config_model = OllamaConfig
    config: OllamaConfig

    def __init__(
        self, config: OllamaConfig, *, http_client: httpx2.AsyncClient | None = None, **kwargs: Any
    ):
        super().__init__(config, **kwargs)
        # Injectable for tests; otherwise created lazily with the configured timeouts.
        self._client: httpx2.AsyncClient | None = http_client

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            tools=True,
            streaming=True,
            logprobs=self.config.logprobs,
            json_schema=True,
            thinking=self.config.think is not False,
        )

    def _http(self) -> httpx2.AsyncClient:
        if self._client is None:
            timeout = httpx2.Timeout(self.config.timeout_s, connect=self.config.connect_timeout_s)
            self._client = httpx2.AsyncClient(base_url=self.config.base_url, timeout=timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- request -------------------------------------------------------------

    def build_payload(self, req: CompletionRequest, *, logprobs: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": to_ollama_messages(req),
            "stream": True,
            "think": self.config.think,
            "keep_alive": self.config.keep_alive,
        }
        tools_disabled = req.tool_choice is not None and req.tool_choice.type == "none"
        if req.tools and not tools_disabled:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in req.tools
            ]
        if req.response_schema is not None:
            payload["format"] = req.response_schema
        options: dict[str, Any] = {
            "num_ctx": self.config.num_ctx,
            "num_predict": req.max_tokens or self.config.default_max_tokens,
            **self.config.options,
        }
        if req.temperature is not None:
            options["temperature"] = req.temperature
        if req.stop_sequences:
            options["stop"] = req.stop_sequences
        payload["options"] = options
        if logprobs and self.config.logprobs:
            payload["logprobs"] = True
        return payload

    # --- call ----------------------------------------------------------------

    async def stream(
        self, req: CompletionRequest, *, logprobs: bool = False
    ) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(req, logprobs=logprobs)
        warnings: list[str] = []
        if req.tool_choice is not None and req.tool_choice.type in ("any", "tool"):
            warnings.append("ollama cannot force a tool call; tool_choice treated as 'auto'")

        start = time.perf_counter()
        ttft: float | None = None
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolUseBlock] = []
        token_logprobs: list[float] = []
        final: dict[str, Any] | None = None

        # A deadline checked per chunk, not asyncio.timeout: a timeout scope that
        # spans a ``yield`` would cancel the consumer's task instead of this call.
        deadline = start + self.config.timeout_s
        try:
            async with self._http().stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")
                    raise self._http_error(resp.status_code, body)
                yield MessageStartEvent(model=self.config.model)
                async for line in resp.aiter_lines():
                    if time.perf_counter() > deadline:
                        raise BackendTimeout(
                            f"ollama call exceeded {self.config.timeout_s}s", backend=self.name
                        )
                    if not line.strip():
                        continue
                    chunk = self._parse_line(line)
                    msg = chunk.get("message") or {}
                    if isinstance(think_delta := msg.get("thinking"), str) and think_delta:
                        ttft = ttft if ttft is not None else elapsed_ms(start)
                        thinking_parts.append(think_delta)
                        yield ThinkingDeltaEvent(thinking=think_delta)
                    if isinstance(text_delta := msg.get("content"), str) and text_delta:
                        ttft = ttft if ttft is not None else elapsed_ms(start)
                        text_parts.append(text_delta)
                        yield TextDeltaEvent(text=text_delta)
                    for call in msg.get("tool_calls") or []:
                        ttft = ttft if ttft is not None else elapsed_ms(start)
                        block = parse_tool_call(call)
                        tool_calls.append(block)
                        yield ToolUseEvent(block=block)
                    for lp in chunk.get("logprobs") or []:
                        if isinstance(lp, dict) and isinstance(lp.get("logprob"), int | float):
                            token_logprobs.append(float(lp["logprob"]))
                    if chunk.get("done"):
                        final = chunk
                        break
        except httpx2.TimeoutException as e:
            raise BackendTimeout(f"ollama timed out: {e}", backend=self.name) from e
        except httpx2.ConnectError as e:
            raise BackendUnavailable(
                f"cannot reach ollama at {self.config.base_url} (is `ollama serve` running?)",
                backend=self.name,
            ) from e
        except httpx2.HTTPError as e:
            raise BackendUnavailable(f"ollama transport error: {e}", backend=self.name) from e

        if final is None:
            raise BackendBadResponse("ollama stream ended before done=true", backend=self.name)

        content: list[ContentBlock] = []
        thinking = "".join(thinking_parts)
        if thinking:
            content.append(ThinkingBlock(thinking=thinking))
        text = "".join(text_parts)
        if text:
            content.append(TextBlock(text=text))
        content.extend(tool_calls)

        stop_reason: StopReason
        if final.get("done_reason") == "length":
            stop_reason = "max_tokens"
        elif tool_calls:
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"

        response = CompletionResponse(
            content=content,
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=int(final.get("prompt_eval_count") or 0),
                output_tokens=int(final.get("eval_count") or 0),
            ),
            model=str(final.get("model") or self.config.model),
            backend=self.name,
            tier=self.tier,
            latency_ms=elapsed_ms(start),
            ttft_ms=ttft,
            token_logprobs=token_logprobs or None,
            warnings=warnings,
            raw={k: v for k, v in final.items() if k != "message"},
        )
        yield MessageEndEvent(response=response)

    def _parse_line(self, line: str) -> dict[str, Any]:
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as e:
            raise BackendBadResponse(f"invalid JSON from ollama: {line[:200]!r}") from e
        if not isinstance(chunk, dict):
            raise BackendBadResponse(f"unexpected ollama chunk: {line[:200]!r}")
        if err := chunk.get("error"):
            raise BackendBadResponse(f"ollama error: {err}", backend=self.name)
        return chunk

    def _http_error(self, status: int, body: str) -> BackendError:
        try:
            detail = json.loads(body).get("error", body)
        except (json.JSONDecodeError, AttributeError):
            detail = body
        if status == 404:
            return BackendRequestRejected(
                f"ollama model {self.config.model!r} not found: {detail}. "
                f"Run `ollama pull {self.config.model}`.",
                backend=self.name,
                status_code=status,
            )
        if status >= 500:
            return BackendUnavailable(
                f"ollama {status}: {detail}", backend=self.name, status_code=status
            )
        return BackendRequestRejected(
            f"ollama {status}: {detail}", backend=self.name, status_code=status
        )

    async def health(self) -> HealthStatus:
        start = time.perf_counter()
        try:
            version = (await self._http().get("/api/version")).json().get("version", "?")
            tags = (await self._http().get("/api/tags")).json().get("models", [])
        except httpx2.HTTPError as e:
            return HealthStatus(
                ok=False, detail=f"ollama unreachable at {self.config.base_url}: {e}"
            )
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        wanted = self.config.model
        if wanted not in names and f"{wanted}:latest" not in names:
            return HealthStatus(
                ok=False,
                detail=f"ollama {version} is up but {wanted!r} is not pulled "
                f"(run `ollama pull {wanted}`)",
                latency_ms=elapsed_ms(start),
            )
        return HealthStatus(ok=True, detail=f"ollama {version}", latency_ms=elapsed_ms(start))


# --- conversion helpers (pure, unit-tested) ---------------------------------


def to_ollama_messages(req: CompletionRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if req.system:
        out.append({"role": "system", "content": req.system})
    tool_names: dict[str, str] = {}

    for m in req.messages:
        blocks = m.blocks()
        if m.role == "assistant":
            text = "".join(b.text for b in blocks if isinstance(b, TextBlock))
            entry: dict[str, Any] = {"role": "assistant", "content": text}
            calls = []
            for b in blocks:
                if isinstance(b, ToolUseBlock):
                    tool_names[b.id] = b.name
                    calls.append({"id": b.id, "function": {"name": b.name, "arguments": b.input}})
            if calls:
                entry["tool_calls"] = calls
            # Thinking and provider-specific blocks are not replayed to a local model.
            out.append(entry)
            continue

        # User turn: tool results become role=tool messages, ahead of any new text.
        for b in blocks:
            if isinstance(b, ToolResultBlock):
                text = b.text()
                if b.is_error:
                    text = f"ERROR: {text}"
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": text,
                    "tool_call_id": b.tool_use_id,
                }
                if name := tool_names.get(b.tool_use_id):
                    tool_msg["tool_name"] = name
                out.append(tool_msg)
        text = "".join(b.text for b in blocks if isinstance(b, TextBlock))
        images = [b.data for b in blocks if isinstance(b, ImageBlock)]
        if text or images:
            user_msg: dict[str, Any] = {"role": "user", "content": text}
            if images:
                user_msg["images"] = images
            out.append(user_msg)
    return out


def parse_tool_call(call: dict[str, Any]) -> ToolUseBlock:
    fn = call.get("function") or {}
    args = fn.get("arguments")
    raw: str | None = None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            raw, args = args, {}
    if not isinstance(args, dict):
        raw, args = json.dumps(args), {}
    return ToolUseBlock(
        id=call.get("id") or f"toolu_local_{uuid.uuid4().hex[:12]}",
        name=str(fn.get("name") or ""),
        input=args,
        raw_arguments=raw,
    )
