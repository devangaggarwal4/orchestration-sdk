"""Routing results: zones, validator results, attempt records and router events."""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, computed_field

from hybrid_router.types import (
    CompletionResponse,
    MessageEndEvent,
    MessageStartEvent,
    TextDeltaEvent,
    ThinkingDeltaEvent,
    Tier,
    ToolUseBlock,
    ToolUseEvent,
    Usage,
)


class Zone(StrEnum):
    EASY = "easy"
    UNCERTAIN = "uncertain"
    HARD = "hard"


Severity = Literal["soft", "hard"]
Strategy = Literal["hybrid", "pre_route", "cascade"]
AttemptRole = Literal["direct", "cascade_attempt", "escalation"]


class ValidatorResult(BaseModel):
    name: str
    ok: bool
    severity: Severity | None = None
    reason: str = ""
    # True when the validator had nothing to check (e.g. no logprobs available).
    skipped: bool = False
    ms: float = 0.0
    # Extra model usage spent by the validator itself (two-sample, judge).
    extra_usage: Usage | None = None
    extra_cost_usd: float | None = None


def new_id() -> str:
    return uuid.uuid4().hex


class AttemptRecord(BaseModel):
    """One model call made while routing a request."""

    call_id: str = Field(default_factory=new_id)
    attempt_index: int
    tier: Tier
    model: str
    role: AttemptRole
    # Output was held back until validation passed (not streamed live).
    buffered: bool
    validator_results: list[ValidatorResult] = Field(default_factory=list)
    ok: bool = False
    error: str | None = None
    # A local attempt whose output was thrown away because the request escalated.
    wasted: bool = False
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    # Time spent waiting for the local concurrency slot (not model latency).
    queue_ms: float = 0.0
    stop_reason: str | None = None
    reported_cost_usd: float | None = None
    tool_use_ids: list[str] = Field(default_factory=list)

    @property
    def failure(self) -> ValidatorResult | None:
        return next((v for v in self.validator_results if not v.ok and not v.skipped), None)


class RouteDecision(BaseModel):
    route_id: str = Field(default_factory=new_id)
    session_id: str | None = None
    step_index: int = 0
    task_type: str | None = None
    strategy: Strategy
    score: float
    zone: Zone
    reasons: list[str]
    features: dict[str, float] = Field(default_factory=dict)
    forced: bool = False
    planned_tier: Tier
    final_tier: Tier | None = None
    final_model: str | None = None
    attempts: list[AttemptRecord] = Field(default_factory=list)
    escalation_reason: str | None = None
    success: bool = False
    total_latency_ms: float = 0.0
    config_hash: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tiers_attempted(self) -> list[Tier]:
        return [a.tier for a in self.attempts]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def escalated(self) -> bool:
        return any(a.role == "escalation" for a in self.attempts)


class RoutedResponse(BaseModel):
    response: CompletionResponse
    decision: RouteDecision

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def tool_calls(self) -> list[ToolUseBlock]:
        return self.response.tool_calls


# --- router events ----------------------------------------------------------


class RouteStartEvent(BaseModel):
    type: Literal["route_start"] = "route_start"
    route_id: str
    score: float
    zone: Zone
    reasons: list[str]
    planned_tier: Tier
    strategy: Strategy


class AttemptStartEvent(BaseModel):
    type: Literal["attempt_start"] = "attempt_start"
    attempt_index: int
    tier: Tier
    model: str
    role: AttemptRole
    buffered: bool


class AttemptEndEvent(BaseModel):
    """Closes an attempt. When ``discarded`` is true, any text already streamed
    for this attempt (easy zone streams live) must be thrown away by the client."""

    type: Literal["attempt_end"] = "attempt_end"
    attempt_index: int
    tier: Tier
    ok: bool
    discarded: bool
    validator_results: list[ValidatorResult]
    error: str | None = None


class RouteEndEvent(BaseModel):
    type: Literal["route_end"] = "route_end"
    decision: RouteDecision
    response: CompletionResponse | None


RouterEvent = Annotated[
    RouteStartEvent
    | AttemptStartEvent
    | AttemptEndEvent
    | RouteEndEvent
    | MessageStartEvent
    | TextDeltaEvent
    | ThinkingDeltaEvent
    | ToolUseEvent
    | MessageEndEvent,
    Field(discriminator="type"),
]


class RoutingError(Exception):
    """Every attempt failed with an error, so there is no response to return."""

    def __init__(self, message: str, decision: RouteDecision):
        super().__init__(message)
        self.decision = decision


def dump_event(event: Any) -> dict[str, Any]:
    return event.model_dump(mode="json")
