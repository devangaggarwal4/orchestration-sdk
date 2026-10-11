"""Scorer interface: a fast difficulty estimate in [0, 1] plus human-readable reasons.

Scorers must be cheap (a few milliseconds) and must not call a model to judge
difficulty. ``score_features`` lets telemetry replay (what-if) re-score stored
feature vectors under different weights without the original request.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from hybrid_router.types import CompletionRequest


class ScoringContext(BaseModel):
    """Per-session facts the scorer can't see in the request itself."""

    # Steps left in the stickiness window after the last escalation.
    sticky_remaining: int = 0
    # The newest tool results include an error for a call made by a local step.
    prior_local_tool_error: bool = False


class ScoreResult(BaseModel):
    score: float
    reasons: list[str]
    # Normalized feature values in [0, 1] plus boost inputs, stored for replay.
    features: dict[str, float] = Field(default_factory=dict)


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, req: CompletionRequest, ctx: ScoringContext) -> ScoreResult: ...
