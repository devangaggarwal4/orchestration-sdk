"""Claude backend using the official ``anthropic`` SDK.

Calls always stream (``beta.messages.stream``) so long outputs never hit HTTP
timeouts and we can measure time-to-first-token. The beta namespace is used
because server-side refusal fallbacks are a beta parameter.

Current Claude models (Sonnet 5.5 / Opus 5.5) reject sampling parameters
(``temperature``) and forced ``tool_choice`` (``any``/``tool``) with a 400, so
this backend drops or downgrades those and records a warning on the response.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

import anthropic

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
    RedactedThinkingBlock,
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

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP_REASONS: set[str] = {
    "end_turn",
    "max_tokens",
    "tool_use",
    "stop_sequence",
    "refusal",
    "pause_turn",
}


class AnthropicConfig(BackendConfig):
    kind: Literal["anthropic"] = "anthropic"
    model: str = "claude-sonnet-5-5"
    timeout_s: float = 600.0
    default_max_tokens: int = 16000
    # SDK-level retries for 408/409/429/5xx and connection errors.
    max_retries: int = 2
    # Env var holding the key. When unset, the SDK's own credential chain applies
    # (ANTHROPIC_AUTH_TOKEN, `ant auth login` profiles...).
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str | None = None
    # "omit" lets the model use its default (adaptive on current models).
    # "between_tools" turns reasoning off on Sonnet 5.5; Opus 5.5 rejects it.
    thinking: Literal["omit", "adaptive", "between_tools"] = "omit"
    thinking_display: Literal["omitted", "summarized"] | None = None
    # low | medium | high | xhigh | max; None = model default.
    effort: str | None = None
    # Top-level automatic prompt caching (caches the growing conversation prefix).
    prompt_caching: bool = True
    # Ask the API to re-run a refused request on a fallback model in the same call.
    refusal_fallback: bool = True
    # Send response_schema as structured outputs (output_config.format). When off,
    # the schema is described in the system prompt instead.
    structured_outputs: bool = True
    # Stream large tool inputs as they're generated. The router validates tool
    # inputs itself, so the server-side validation this skips is not lost.
    eager_input_streaming: bool = True
    # Current models 400 on tool_choice any/tool; leave False unless you pin an
    # older model that accepts forced tool use.
    forced_tool_choice: bool = False
    # Older models accept temperature; current ones 400 on it.
    sampling_params: bool = False


@register_backend
class AnthropicBackend(Backend):
    kind = "anthropic"
    config_model = AnthropicConfig
    config: AnthropicConfig

    def __init__(self, config: AnthropicConfig, *, client: Any = None, **kwargs: Any):
        super().__init__(config, **kwargs)
        self._client: anthropic.AsyncAnthropic | None = client

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            tools=True,
            streaming=True,
            logprobs=False,
            json_schema=self.config.structured_outputs,
            prompt_caching=self.config.prompt_caching,
            exact_token_count=True,
            thinking=True,
        )

    def _sdk(self) -> anthropic.AsyncAnthropic:
        # Created lazily so configs load (and local-only use works) without credentials.
        if self._client is None:
            kwargs: dict[str, Any] = {
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout_s,
            }
            if key := os.environ.get(self.config.api_key_env):
                kwargs["api_key"] = key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            try:
                self._client = anthropic.AsyncAnthropic(**kwargs)
            except anthropic.AnthropicError as e:
                raise BackendAuthError(
                    f"no Anthropic credentials: set {self.config.api_key_env}",
                    backend=self.name,
                ) from e
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _auth_error(self, e: TypeError) -> BackendAuthError:
        # With no credentials at all the SDK raises TypeError when building the
        # request ("Could not resolve authentication method"), not an APIError.
        if "authentication" not in str(e).lower():
            raise e
        return BackendAuthError(
            f"no Anthropic credentials: set {self.config.api_key_env} "
            "(or ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile)",
            backend=self.name,
        )

    # --- request -------------------------------------------------------------

    def build_params(
        self, req: CompletionRequest, *, for_count: bool = False
    ) -> tuple[dict[str, Any], list[str]]:
        """SDK kwargs for ``beta.messages.stream`` plus any warnings."""
        cfg = self.config
        warnings: list[str] = []
        params: dict[str, Any] = {
            "model": cfg.model,
            "messages": to_anthropic_messages(req),
        }
        if not for_count:
            params["max_tokens"] = req.max_tokens or cfg.default_max_tokens

        system = req.system or ""
        output_config: dict[str, Any] = {}
        if req.response_schema is not None:
            if cfg.structured_outputs:
                output_config["format"] = {"type": "json_schema", "schema": req.response_schema}
            else:
                system = (
                    f"{system}\n\nRespond with only a JSON document matching this schema:\n"
                    f"{json.dumps(req.response_schema)}"
                ).strip()
        if cfg.effort:
            output_config["effort"] = cfg.effort
        if output_config:
            params["output_config"] = output_config
        if system:
            params["system"] = system

        if req.tools:
            tools = []
            for t in req.tools:
                tool: dict[str, Any] = {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                if cfg.eager_input_streaming and not for_count:
                    tool["eager_input_streaming"] = True
                tools.append(tool)
            params["tools"] = tools

        if (tc := req.tool_choice) is not None and not for_count:
            if tc.type in ("any", "tool") and not cfg.forced_tool_choice:
                warnings.append(
                    f"{cfg.model} does not accept forced tool_choice; sent 'auto' instead"
                )
                params["tool_choice"] = {"type": "auto"}
            elif tc.type == "tool":
                params["tool_choice"] = {"type": "tool", "name": tc.name}
            else:
                params["tool_choice"] = {"type": tc.type}

        if cfg.thinking != "omit":
            thinking: dict[str, Any] = {"type": cfg.thinking}
            if cfg.thinking == "adaptive" and cfg.thinking_display:
                thinking["display"] = cfg.thinking_display
            params["thinking"] = thinking

        if for_count:
            return params, warnings

        if req.temperature is not None:
            if cfg.sampling_params:
                params.setdefault("extra_body", {})["temperature"] = req.temperature
            else:
                warnings.append(f"{cfg.model} does not accept temperature; dropped")
        if req.stop_sequences:
            params["stop_sequences"] = req.stop_sequences
        if cfg.prompt_caching:
            params["cache_control"] = {"type": "ephemeral"}
        if cfg.refusal_fallback:
            params["betas"] = [REFUSAL_FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params, warnings

    # --- call ----------------------------------------------------------------

    async def stream(
        self, req: CompletionRequest, *, logprobs: bool = False
    ) -> AsyncIterator[StreamEvent]:
        params, warnings = self.build_params(req)
        start = time.perf_counter()
        ttft: float | None = None
        try:
            async with self._sdk().beta.messages.stream(**params) as stream:
                yield MessageStartEvent(model=self.config.model)
                async for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "text_delta":
                        ttft = ttft if ttft is not None else elapsed_ms(start)
                        yield TextDeltaEvent(text=delta.text)
                    elif delta.type == "thinking_delta":
                        ttft = ttft if ttft is not None else elapsed_ms(start)
                        yield ThinkingDeltaEvent(thinking=delta.thinking)
                    elif ttft is None:
                        ttft = elapsed_ms(start)
                message = await stream.get_final_message()
        except anthropic.APIError as e:
            raise map_anthropic_error(e, backend=self.name) from e
        except TypeError as e:
            raise self._auth_error(e) from e
        except ValueError as e:
            # The SDK raises ValueError for streamed tool input it cannot parse.
            raise BackendBadResponse(
                f"unparseable streamed tool input: {e}", backend=self.name
            ) from e

        response = from_anthropic_message(
            message,
            backend=self.name,
            tier=self.tier,
            latency_ms=elapsed_ms(start),
            ttft_ms=ttft,
            warnings=warnings,
        )
        for block in response.tool_calls:
            yield ToolUseEvent(block=block)
        yield MessageEndEvent(response=response)

    async def count_tokens(self, req: CompletionRequest) -> int:
        params, _ = self.build_params(req, for_count=True)
        try:
            result = await self._sdk().messages.count_tokens(**params)
        except anthropic.APIError as e:
            raise map_anthropic_error(e, backend=self.name) from e
        except TypeError as e:
            raise self._auth_error(e) from e
        return result.input_tokens

    async def health(self) -> HealthStatus:
        start = time.perf_counter()
        try:
            info = await self._sdk().models.retrieve(self.config.model)
        except BackendError as e:
            return HealthStatus(ok=False, detail=str(e))
        except anthropic.APIError as e:
            return HealthStatus(ok=False, detail=str(map_anthropic_error(e, backend=self.name)))
        except TypeError as e:
            return HealthStatus(ok=False, detail=str(self._auth_error(e)))
        return HealthStatus(
            ok=True, detail=getattr(info, "display_name", info.id), latency_ms=elapsed_ms(start)
        )


# --- conversion helpers (pure, unit-tested) ---------------------------------


def _block_to_param(b: ContentBlock) -> dict[str, Any] | None:
    if isinstance(b, TextBlock):
        return {"type": "text", "text": b.text}
    if isinstance(b, ImageBlock):
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": b.media_type, "data": b.data},
        }
    if isinstance(b, ToolUseBlock):
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    if isinstance(b, ToolResultBlock):
        content: Any = b.content
        if not isinstance(content, str):
            content = [_block_to_param(c) for c in content]
        return {
            "type": "tool_result",
            "tool_use_id": b.tool_use_id,
            "content": content,
            "is_error": b.is_error,
        }
    if isinstance(b, ThinkingBlock):
        # Unsigned (local) thinking can't be replayed to Claude.
        if b.signature is None:
            return None
        return {"type": "thinking", "thinking": b.thinking, "signature": b.signature}
    if isinstance(b, RedactedThinkingBlock):
        return {"type": "redacted_thinking", "data": b.data}
    if isinstance(b, OpaqueBlock):
        return b.data if b.provider == "anthropic" else None
    return None


def to_anthropic_messages(req: CompletionRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in req.messages:
        if isinstance(m.content, str):
            out.append({"role": m.role, "content": m.content})
            continue
        blocks = [p for b in m.content if (p := _block_to_param(b)) is not None]
        if blocks:
            out.append({"role": m.role, "content": blocks})
    return out


def from_anthropic_message(
    message: Any,
    *,
    backend: str,
    tier: Any,
    latency_ms: float,
    ttft_ms: float | None = None,
    warnings: list[str] | None = None,
) -> CompletionResponse:
    content: list[ContentBlock] = []
    for block in message.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            content.append(TextBlock(text=block.text))
        elif btype == "tool_use":
            raw_input = block.input
            if isinstance(raw_input, dict):
                content.append(ToolUseBlock(id=block.id, name=block.name, input=raw_input))
            else:
                content.append(
                    ToolUseBlock(id=block.id, name=block.name, raw_arguments=str(raw_input))
                )
        elif btype == "thinking":
            content.append(ThinkingBlock(thinking=block.thinking or "", signature=block.signature))
        elif btype == "redacted_thinking":
            content.append(RedactedThinkingBlock(data=block.data))
        else:
            content.append(
                OpaqueBlock(provider="anthropic", data=block.model_dump(mode="json", by_alias=True))
            )

    stop: StopReason = message.stop_reason if message.stop_reason in _STOP_REASONS else "end_turn"
    u = message.usage
    usage = Usage(
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", None) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", None) or 0,
    )
    raw: dict[str, Any] = {"id": message.id}
    if getattr(message, "stop_details", None) is not None:
        raw["stop_details"] = message.stop_details.model_dump(mode="json")
    return CompletionResponse(
        content=content,
        stop_reason=stop,
        usage=usage,
        model=message.model,
        backend=backend,
        tier=tier,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,
        warnings=warnings or [],
        raw=raw,
    )


def map_anthropic_error(e: anthropic.APIError, *, backend: str) -> BackendError:
    status = getattr(e, "status_code", None)
    msg = f"anthropic: {e}"
    # Most specific first: several of these subclass APIStatusError.
    if isinstance(e, anthropic.APITimeoutError):
        return BackendTimeout(msg, backend=backend)
    if isinstance(e, anthropic.APIConnectionError):
        return BackendUnavailable(msg, backend=backend)
    if isinstance(e, anthropic.RateLimitError | anthropic.OverloadedError):
        return BackendRateLimited(msg, backend=backend, status_code=status)
    if isinstance(e, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return BackendAuthError(msg, backend=backend, status_code=status)
    if isinstance(e, anthropic.APIStatusError):
        if e.status_code >= 500:
            return BackendUnavailable(msg, backend=backend, status_code=status)
        return BackendRequestRejected(msg, backend=backend, status_code=status)
    return BackendBadResponse(msg, backend=backend)
