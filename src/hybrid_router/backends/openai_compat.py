"""Backend for OpenAI-compatible ``/chat/completions`` APIs.

The default target is OpenRouter, which is how the cloud tiers reach Claude.
The same wire format is spoken by llama.cpp's server, LM Studio, vLLM and MLX
servers, so this backend also covers those local runtimes: point ``base_url``
at them and turn on ``logprobs``.

OpenRouter extras used when present:
  * ``usage.cost`` (USD actually charged) becomes ``CompletionResponse.reported_cost_usd``
  * ``reasoning_details`` (Claude thinking) is kept and replayed on later turns,
    which Claude needs for tool use with reasoning
  * ``cache_control`` breakpoints for Anthropic models (prompt caching)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx2

from hybrid_router.backends.base import (
    Backend,
    BackendAuthError,
    BackendBadResponse,
    BackendCapabilities,
    BackendConfig,
    BackendError,
    BackendRateLimited,
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
    OpaqueBlock,
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

PROVIDER = "openai_compat"
REASONING_DETAILS = "reasoning_details"

_FINISH_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


class OpenAICompatConfig(BackendConfig):
    kind: Literal["openai_compat"] = "openai_compat"
    model: str = "anthropic/claude-sonnet-5.5"
    base_url: str = "https://openrouter.ai/api/v1"
    # Env var holding the bearer token. Empty for local servers that need none.
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_s: float = 600.0
    connect_timeout_s: float = 10.0
    default_max_tokens: int = 16000
    # Add cache_control breakpoints (system + latest message). Only meaningful for
    # Anthropic models; other providers cache automatically or not at all.
    prompt_caching: bool = True
    # Request per-token logprobs when the router asks (local servers; not Claude).
    logprobs: bool = False
    # Claude Sonnet/Opus 5.5 reject temperature and forced tool_choice with a 400,
    # so the Claude tiers ship with both off. Turn on for models that accept them.
    sampling_params: bool = True
    forced_tool_choice: bool = True
    # Sent as response_format json_schema when a request has response_schema.
    structured_outputs: bool = True
    # OpenRouter reasoning settings, e.g. {"effort": "medium"} or {"enabled": false}.
    reasoning: dict[str, Any] | None = None
    # OpenRouter provider routing, e.g. {"order": ["anthropic"], "allow_fallbacks": true}.
    provider: dict[str, Any] | None = None
    # Extra request headers (e.g. OpenRouter's HTTP-Referer / X-Title attribution).
    extra_headers: dict[str, str] = {}
    # Merged into every request body last; escape hatch for provider-specific fields.
    extra_body: dict[str, Any] = {}


@register_backend
class OpenAICompatBackend(Backend):
    kind = "openai_compat"
    config_model = OpenAICompatConfig
    config: OpenAICompatConfig

    def __init__(
        self,
        config: OpenAICompatConfig,
        *,
        http_client: httpx2.AsyncClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, **kwargs)
        self._client: httpx2.AsyncClient | None = http_client

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            tools=True,
            streaming=True,
            logprobs=self.config.logprobs,
            json_schema=self.config.structured_outputs,
            prompt_caching=self.config.prompt_caching and self._is_anthropic_model,
            thinking=True,
        )

    @property
    def _is_anthropic_model(self) -> bool:
        return self.config.model.startswith(("anthropic/", "claude"))

    def _api_key(self) -> str | None:
        return os.environ.get(self.config.api_key_env) if self.config.api_key_env else None

    def _http(self) -> httpx2.AsyncClient:
        if self._client is None:
            timeout = httpx2.Timeout(self.config.timeout_s, connect=self.config.connect_timeout_s)
            self._client = httpx2.AsyncClient(base_url=self.config.base_url, timeout=timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        if self.config.api_key_env and not self._api_key():
            raise BackendAuthError(f"no API key: set {self.config.api_key_env}", backend=self.name)
        headers = dict(self.config.extra_headers)
        if key := self._api_key():
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- request -------------------------------------------------------------

    def build_payload(
        self, req: CompletionRequest, *, logprobs: bool = False
    ) -> tuple[dict[str, Any], list[str]]:
        cfg = self.config
        warnings: list[str] = []
        if req.response_schema is not None and not cfg.structured_outputs:
            note = (
                "Respond with only a JSON document matching this schema:\n"
                f"{json.dumps(req.response_schema)}"
            )
            req = req.model_copy(update={"system": f"{req.system or ''}\n\n{note}".strip()})
        messages = to_openai_messages(req)
        if cfg.prompt_caching and self._is_anthropic_model:
            add_cache_breakpoints(messages)
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": messages,
            "stream": True,
            "max_tokens": req.max_tokens or cfg.default_max_tokens,
            # Ask for the usage block in the final chunk. OpenRouter always sends it
            # now; other servers (llama.cpp, vLLM) need this.
            "stream_options": {"include_usage": True},
        }
        if req.tools:
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
        if (tc := req.tool_choice) is not None and req.tools:
            if tc.type in ("any", "tool") and not cfg.forced_tool_choice:
                warnings.append(f"{cfg.model} does not accept forced tool_choice; sent 'auto'")
                payload["tool_choice"] = "auto"
            elif tc.type == "any":
                payload["tool_choice"] = "required"
            elif tc.type == "tool":
                payload["tool_choice"] = {"type": "function", "function": {"name": tc.name}}
            else:
                payload["tool_choice"] = tc.type
        if req.response_schema is not None and cfg.structured_outputs:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": req.response_schema},
            }
        if req.temperature is not None:
            if cfg.sampling_params:
                payload["temperature"] = req.temperature
            else:
                warnings.append(f"{cfg.model} does not accept temperature; dropped")
        if req.stop_sequences:
            payload["stop"] = req.stop_sequences
        if logprobs and cfg.logprobs:
            payload["logprobs"] = True
        if cfg.reasoning is not None:
            payload["reasoning"] = cfg.reasoning
        if cfg.provider is not None:
            payload["provider"] = cfg.provider
        payload.update(cfg.extra_body)
        return payload, warnings

    # --- call ----------------------------------------------------------------

    async def stream(
        self, req: CompletionRequest, *, logprobs: bool = False
    ) -> AsyncIterator[StreamEvent]:
        payload, warnings = self.build_payload(req, logprobs=logprobs)
        headers = self._headers()
        start = time.perf_counter()
        # Checked per chunk; asyncio.timeout around a yield would cancel the consumer.
        deadline = start + self.config.timeout_s
        ttft: float | None = None
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        reasoning_details = ReasoningDetailsMerger()
        tools = ToolCallAccumulator()
        token_logprobs: list[float] = []
        finish_reason: str | None = None
        usage_raw: dict[str, Any] | None = None
        model_served = self.config.model
        response_id: str | None = None

        try:
            async with self._http().stream(
                "POST", "/chat/completions", json=payload, headers=headers
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")
                    raise self._http_error(resp.status_code, body, resp.headers)
                yield MessageStartEvent(model=self.config.model)
                async for line in resp.aiter_lines():
                    if time.perf_counter() > deadline:
                        raise BackendTimeout(
                            f"call exceeded {self.config.timeout_s}s", backend=self.name
                        )
                    # SSE: skip blanks, comments (": OPENROUTER PROCESSING") and event: lines.
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = self._parse_chunk(data)
                    response_id = response_id or chunk.get("id")
                    model_served = chunk.get("model") or model_served
                    if isinstance(chunk.get("usage"), dict):
                        usage_raw = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                        if isinstance(reasoning, str) and reasoning:
                            ttft = ttft if ttft is not None else elapsed_ms(start)
                            reasoning_parts.append(reasoning)
                            yield ThinkingDeltaEvent(thinking=reasoning)
                        reasoning_details.add(delta.get(REASONING_DETAILS))
                        if isinstance(text := delta.get("content"), str) and text:
                            ttft = ttft if ttft is not None else elapsed_ms(start)
                            text_parts.append(text)
                            yield TextDeltaEvent(text=text)
                        if delta.get("tool_calls"):
                            ttft = ttft if ttft is not None else elapsed_ms(start)
                            tools.add(delta["tool_calls"])
                        lp = choice.get("logprobs") or {}
                        for tok in lp.get("content") or []:
                            if isinstance(tok, dict) and isinstance(
                                tok.get("logprob"), int | float
                            ):
                                token_logprobs.append(float(tok["logprob"]))
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
        except httpx2.TimeoutException as e:
            raise BackendTimeout(f"timed out: {e}", backend=self.name) from e
        except httpx2.ConnectError as e:
            raise BackendUnavailable(
                f"cannot reach {self.config.base_url}: {e}", backend=self.name
            ) from e
        except httpx2.HTTPError as e:
            raise BackendUnavailable(f"transport error: {e}", backend=self.name) from e

        if finish_reason is None and not text_parts and not tools:
            raise BackendBadResponse("stream ended without any output", backend=self.name)

        content: list[ContentBlock] = []
        if reasoning_parts:
            content.append(ThinkingBlock(thinking="".join(reasoning_parts)))
        if details := reasoning_details.result():
            content.append(OpaqueBlock(provider=PROVIDER, data={REASONING_DETAILS: details}))
        if text_parts:
            content.append(TextBlock(text="".join(text_parts)))
        tool_blocks = tools.result()
        content.extend(tool_blocks)

        stop_reason = _FINISH_REASONS.get(finish_reason or "stop", "end_turn")
        if tool_blocks and stop_reason == "end_turn":
            stop_reason = "tool_use"
        usage, cost = parse_usage(usage_raw)
        raw: dict[str, Any] = {"id": response_id, "finish_reason": finish_reason}
        if usage_raw is not None:
            raw["usage"] = usage_raw

        response = CompletionResponse(
            content=content,
            stop_reason=stop_reason,
            usage=usage,
            model=model_served,
            backend=self.name,
            tier=self.tier,
            latency_ms=elapsed_ms(start),
            ttft_ms=ttft,
            token_logprobs=token_logprobs or None,
            reported_cost_usd=cost,
            warnings=warnings,
            raw=raw,
        )
        for block in tool_blocks:
            yield ToolUseEvent(block=block)
        yield MessageEndEvent(response=response)

    def _parse_chunk(self, data: str) -> dict[str, Any]:
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as e:
            raise BackendBadResponse(f"invalid JSON chunk: {data[:200]!r}") from e
        if not isinstance(chunk, dict):
            raise BackendBadResponse(f"unexpected chunk: {data[:200]!r}", backend=self.name)
        if err := chunk.get("error"):
            # OpenRouter reports mid-stream provider failures this way.
            msg = err.get("message", err) if isinstance(err, dict) else err
            raise BackendBadResponse(f"error mid-stream: {msg}", backend=self.name)
        return chunk

    def _http_error(
        self, status: int, body: str, headers: httpx2.Headers | None = None
    ) -> BackendError:
        try:
            err = json.loads(body).get("error", body)
            detail = err.get("message", err) if isinstance(err, dict) else err
        except (json.JSONDecodeError, AttributeError):
            detail = body
        msg = f"{status}: {detail}"
        kw: dict[str, Any] = {"backend": self.name, "status_code": status}
        if status in (401, 403):
            return BackendAuthError(f"{msg} (check {self.config.api_key_env})", **kw)
        if status == 402:
            return BackendRequestRejected(f"{msg} (out of OpenRouter credits?)", **kw)
        if status in (408, 429):
            return BackendRateLimited(msg, **kw)
        if status >= 500:
            return BackendUnavailable(msg, **kw)
        return BackendRequestRejected(msg, **kw)

    async def health(self) -> HealthStatus:
        """Checks the API answers, the key is accepted, and the model id exists."""
        start = time.perf_counter()
        try:
            headers = self._headers()
            resp = await self._http().get("/models", headers=headers)
        except BackendError as e:
            return HealthStatus(ok=False, detail=str(e))
        except httpx2.HTTPError as e:
            return HealthStatus(ok=False, detail=f"cannot reach {self.config.base_url}: {e}")
        if resp.status_code >= 400:
            return HealthStatus(ok=False, detail=str(self._http_error(resp.status_code, resp.text)))
        try:
            ids = {m.get("id") for m in resp.json().get("data", [])}
        except (ValueError, AttributeError):
            ids = set()
        if ids and self.config.model not in ids:
            return HealthStatus(
                ok=False,
                detail=f"model {self.config.model!r} not offered at {self.config.base_url}",
                latency_ms=elapsed_ms(start),
            )
        return HealthStatus(ok=True, detail="model available", latency_ms=elapsed_ms(start))


# --- conversion helpers (pure, unit-tested) ---------------------------------


def to_openai_messages(req: CompletionRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if req.system:
        out.append({"role": "system", "content": req.system})
    for m in req.messages:
        blocks = m.blocks()
        if m.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": None}
            text = "".join(b.text for b in blocks if isinstance(b, TextBlock))
            if text:
                entry["content"] = text
            calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {"name": b.name, "arguments": json.dumps(b.input)},
                }
                for b in blocks
                if isinstance(b, ToolUseBlock)
            ]
            if calls:
                entry["tool_calls"] = calls
            details = [
                d
                for b in blocks
                if isinstance(b, OpaqueBlock) and b.provider == PROVIDER
                for d in b.data.get(REASONING_DETAILS, [])
            ]
            if details:
                entry[REASONING_DETAILS] = details
            if entry["content"] is None and not calls:
                entry["content"] = ""
            out.append(entry)
            continue

        # User turn: tool results first (as role=tool), then any new text/images.
        for b in blocks:
            if isinstance(b, ToolResultBlock):
                text = b.text()
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": b.tool_use_id,
                        "content": f"ERROR: {text}" if b.is_error else text,
                    }
                )
        parts: list[dict[str, Any]] = []
        for b in blocks:
            if isinstance(b, TextBlock) and b.text:
                parts.append({"type": "text", "text": b.text})
            elif isinstance(b, ImageBlock):
                url = f"data:{b.media_type};base64,{b.data}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
        if not parts:
            continue
        if all(p["type"] == "text" for p in parts):
            out.append({"role": "user", "content": "".join(p["text"] for p in parts)})
        else:
            out.append({"role": "user", "content": parts})
    return out


def add_cache_breakpoints(messages: list[dict[str, Any]]) -> None:
    """Mark the system prompt and the newest user/tool message as cache breakpoints.

    Agent loops resend a growing prefix, so caching up to the latest message makes
    the next step a cache hit. Uses 2 of Anthropic's 4 allowed breakpoints.
    """
    targets = [m for m in messages[:1] if m["role"] == "system"]
    for m in reversed(messages):
        if m["role"] in ("user", "tool"):
            targets.append(m)
            break
    for m in targets:
        content = m.get("content")
        if isinstance(content, str) and content:
            m["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(content, list) and content:
            content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}


def parse_usage(raw: dict[str, Any] | None) -> tuple[Usage, float | None]:
    """OpenAI-style usage -> our Usage (input excludes cached tokens) and USD cost."""
    if not raw:
        return Usage(), None
    details = raw.get("prompt_tokens_details") or {}
    cache_read = int(details.get("cached_tokens") or 0)
    cache_write = int(details.get("cache_write_tokens") or 0)
    prompt = int(raw.get("prompt_tokens") or 0)
    usage = Usage(
        input_tokens=max(prompt - cache_read - cache_write, 0),
        output_tokens=int(raw.get("completion_tokens") or 0),
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )
    cost = raw.get("cost")
    return usage, float(cost) if isinstance(cost, int | float) else None


class ToolCallAccumulator:
    """Reassembles streamed tool calls (arguments arrive as string fragments)."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def __bool__(self) -> bool:
        return bool(self._calls)

    def add(self, deltas: list[dict[str, Any]]) -> None:
        for pos, d in enumerate(deltas):
            idx = d.get("index", pos)
            call = self._calls.setdefault(idx, {"id": None, "name": "", "arguments": ""})
            call["id"] = call["id"] or d.get("id")
            fn = d.get("function") or {}
            if fn.get("name"):
                call["name"] += fn["name"]
            args = fn.get("arguments")
            if isinstance(args, str):
                call["arguments"] += args
            elif isinstance(args, dict):
                call["arguments"] += json.dumps(args)

    def result(self) -> list[ToolUseBlock]:
        blocks = []
        for idx in sorted(self._calls):
            c = self._calls[idx]
            raw_args = c["arguments"].strip() or "{}"
            raw: str | None = None
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed, raw = {}, raw_args
            if not isinstance(parsed, dict):
                parsed, raw = {}, raw_args
            blocks.append(
                ToolUseBlock(
                    id=c["id"] or f"call_{uuid.uuid4().hex[:12]}",
                    name=c["name"],
                    input=parsed,
                    raw_arguments=raw,
                )
            )
        return blocks


class ReasoningDetailsMerger:
    """Merges streamed ``reasoning_details`` fragments into whole entries.

    Fragments sharing an ``index`` belong to one entry: string pieces
    (``text``/``summary``/``data``) are concatenated, other fields take the
    latest non-null value (e.g. the ``signature`` that arrives last).
    """

    _CONCAT = ("text", "summary", "data")

    def __init__(self) -> None:
        self._entries: dict[Any, dict[str, Any]] = {}

    def add(self, items: Any) -> None:
        if not isinstance(items, list):
            return
        for pos, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            key = item.get("index", f"pos{len(self._entries) + pos}")
            entry = self._entries.setdefault(key, {})
            for field, value in item.items():
                if value is None:
                    continue
                if field in self._CONCAT and isinstance(value, str):
                    entry[field] = entry.get(field, "") + value
                else:
                    entry[field] = value

    def result(self) -> list[dict[str, Any]]:
        return list(self._entries.values())
