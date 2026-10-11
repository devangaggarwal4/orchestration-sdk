"""Canonical request/response types shared by every backend and the router.

Messages are Anthropic-shaped (role + content blocks). Backends convert to and
from their native wire formats; nothing outside ``backends/`` sees a wire format.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Tier(StrEnum):
    """Escalation ladder, cheapest first."""

    LOCAL = "local"
    SONNET = "sonnet"
    OPUS = "opus"

    @property
    def is_cloud(self) -> bool:
        return self is not Tier.LOCAL


TIER_ORDER: tuple[Tier, ...] = (Tier.LOCAL, Tier.SONNET, Tier.OPUS)


# --- content blocks ---------------------------------------------------------


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    media_type: str
    data: str  # base64, no newlines


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    # Set when the backend returned arguments that were not a JSON object; the
    # structural validator treats this as a malformed tool call.
    raw_arguments: str | None = None


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[TextBlock | ImageBlock] = ""
    is_error: bool = False

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(b.text for b in self.content if isinstance(b, TextBlock))


class ThinkingBlock(BaseModel):
    """Reasoning output. Claude blocks carry a signature and must be replayed
    unchanged to the same model; local thinking has no signature."""

    type: Literal["thinking"] = "thinking"
    thinking: str = ""
    signature: str | None = None


class RedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


class OpaqueBlock(BaseModel):
    """A provider-specific block we don't model (e.g. OpenRouter ``reasoning_details``).

    Kept verbatim so an agent can append ``response.content`` to its history and
    replay it to the same provider without loss. Other backends drop it.
    """

    type: Literal["opaque"] = "opaque"
    provider: str
    data: dict[str, Any]


ContentBlock = Annotated[
    TextBlock
    | ImageBlock
    | ToolUseBlock
    | ToolResultBlock
    | ThinkingBlock
    | RedactedThinkingBlock
    | OpaqueBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]

    def blocks(self) -> list[ContentBlock]:
        if isinstance(self.content, str):
            return [TextBlock(text=self.content)] if self.content else []
        return list(self.content)

    def text(self) -> str:
        return "".join(b.text for b in self.blocks() if isinstance(b, TextBlock))


# --- tools ------------------------------------------------------------------


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class ToolChoice(BaseModel):
    type: Literal["auto", "any", "none", "tool"] = "auto"
    name: str | None = None


# --- request / response -----------------------------------------------------


class CompletionRequest(BaseModel):
    messages: list[Message]
    system: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: ToolChoice | None = None
    # When set, the answer must be a JSON document matching this schema.
    response_schema: dict[str, Any] | None = None
    # None means "use the backend's configured default".
    max_tokens: int | None = None
    # Ignored by backends whose models reject sampling parameters.
    temperature: float | None = None
    stop_sequences: list[str] = Field(default_factory=list)
    task_type: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def last_user_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "user":
                text = m.text()
                if text:
                    return text
        return ""


StopReason = Literal["end_turn", "max_tokens", "tool_use", "stop_sequence", "refusal", "pause_turn"]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


class CompletionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    content: list[ContentBlock]
    stop_reason: StopReason
    usage: Usage
    model: str
    backend: str
    tier: Tier
    latency_ms: float
    ttft_ms: float | None = None
    # Per-token log probabilities of the generated tokens, when the backend supports them.
    token_logprobs: list[float] | None = None
    # USD the provider says it charged for this call (OpenRouter's usage.cost).
    # None when the backend doesn't report cost; pricing.yaml is used then.
    reported_cost_usd: float | None = None
    # Non-fatal adjustments the backend made to the request (e.g. dropped temperature).
    warnings: list[str] = Field(default_factory=list)
    raw: dict[str, Any] | None = Field(default=None, repr=False)

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def mean_logprob(self) -> float | None:
        if not self.token_logprobs:
            return None
        return sum(self.token_logprobs) / len(self.token_logprobs)

    def as_message(self) -> Message:
        """The assistant turn to append to the conversation history."""
        return Message(role="assistant", content=list(self.content))


# --- streaming --------------------------------------------------------------


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    model: str


class TextDeltaEvent(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDeltaEvent(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class ToolUseEvent(BaseModel):
    """A complete tool call. Emitted once the call's arguments are fully known."""

    type: Literal["tool_use"] = "tool_use"
    block: ToolUseBlock


class MessageEndEvent(BaseModel):
    type: Literal["message_end"] = "message_end"
    response: CompletionResponse


StreamEvent = Annotated[
    MessageStartEvent | TextDeltaEvent | ThinkingDeltaEvent | ToolUseEvent | MessageEndEvent,
    Field(discriminator="type"),
]


# --- helpers ----------------------------------------------------------------

_CHARS_PER_TOKEN = 3.5
_TOKENS_PER_IMAGE = 1000


def estimate_tokens(req: CompletionRequest) -> int:
    """Fast, tokenizer-free input size estimate (used by the scorer and as a fallback)."""
    chars = len(req.system or "")
    images = 0
    for m in req.messages:
        for b in m.blocks():
            if isinstance(b, TextBlock):
                chars += len(b.text)
            elif isinstance(b, ToolUseBlock):
                chars += len(b.name) + len(json.dumps(b.input))
            elif isinstance(b, ToolResultBlock):
                chars += len(b.text())
            elif isinstance(b, ThinkingBlock):
                chars += len(b.thinking)
            elif isinstance(b, ImageBlock):
                images += 1
    for t in req.tools:
        chars += len(t.name) + len(t.description) + len(json.dumps(t.input_schema))
    if req.response_schema:
        chars += len(json.dumps(req.response_schema))
    return int(chars / _CHARS_PER_TOKEN) + images * _TOKENS_PER_IMAGE
