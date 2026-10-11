"""LearnedScorer: logistic regression on telemetry (label: did local fail?).

Trained with ``hybrid-router train-scorer``; enable with ``scorer.kind: learned``.
Uses the same features as RuleScorer, so reasons stay explainable, and keeps the
session boosts (stickiness, prior tool error) on top.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from pydantic import BaseModel

from hybrid_router.config import RuleScorerConfig
from hybrid_router.reporting.queries import RouteRow
from hybrid_router.reporting.tune import BASE_FEATURES, auc, fit_logistic
from hybrid_router.scoring.base import ScoreResult, ScoringContext
from hybrid_router.scoring.features import FeatureExtractor
from hybrid_router.types import CompletionRequest


class LearnedModel(BaseModel):
    features: list[str]
    bias: float
    weights: list[float]
    trained_on: int
    auc: float | None
    created_at: float

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> LearnedModel:
        return cls.model_validate_json(Path(path).read_text())


def train(routes: list[RouteRow], *, min_samples: int = 50) -> LearnedModel:
    data = [r for r in routes if r.local_failed is not None and not r.forced and r.features]
    if len(data) < min_samples:
        raise ValueError(f"need {min_samples} routes with a local attempt, have {len(data)}")
    x = [[r.features.get(f, 0.0) for f in BASE_FEATURES] for r in data]
    y = [bool(r.local_failed) for r in data]
    w = fit_logistic(x, y)
    model = LearnedModel(
        features=list(BASE_FEATURES),
        bias=w[0],
        weights=w[1:],
        trained_on=len(data),
        auc=None,
        created_at=time.time(),
    )
    scorer = LearnedScorer(model)
    model.auc = auc([scorer.score_features(r.features).score for r in data], y)
    return model


class LearnedScorer:
    name = "learned"

    def __init__(self, model: LearnedModel, cfg: RuleScorerConfig | None = None):
        self.model = model
        self.cfg = cfg or RuleScorerConfig()
        self._extractor = FeatureExtractor(self.cfg)

    @classmethod
    def load(cls, path: str | Path, cfg: RuleScorerConfig | None = None) -> LearnedScorer:
        return cls(LearnedModel.load(path), cfg)

    def score(self, req: CompletionRequest, ctx: ScoringContext) -> ScoreResult:
        features, details = self._extractor.extract(req)
        features["sticky_remaining"] = float(ctx.sticky_remaining)
        features["prior_local_tool_error"] = 1.0 if ctx.prior_local_tool_error else 0.0
        return self.score_features(features, details)

    def score_features(
        self, features: dict[str, float], details: dict[str, str] | None = None
    ) -> ScoreResult:
        details = details or {}
        contrib = {
            f: w * features.get(f, 0.0)
            for f, w in zip(self.model.features, self.model.weights, strict=True)
        }
        z = self.model.bias + sum(contrib.values())
        score = 1 / (1 + math.exp(-max(min(z, 30), -30)))
        reasons = [
            f"{details.get(f) or f} (logit {c:+.2f})"
            for f, c in sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
            if abs(c) >= 0.1
        ]
        reasons.insert(0, f"learned model, p(local fails)={score:.2f}")
        sticky = int(features.get("sticky_remaining", 0))
        if sticky > 0 and self.cfg.sticky_steps > 0:
            boost = (
                self.cfg.sticky_boost * min(sticky, self.cfg.sticky_steps) / self.cfg.sticky_steps
            )
            score += boost
            reasons.append(f"sticky after escalation (+{boost:.2f})")
        if features.get("prior_local_tool_error", 0) > 0:
            score += self.cfg.tool_error_boost
            reasons.append(f"previous local tool call errored (+{self.cfg.tool_error_boost:.2f})")
        return ScoreResult(
            score=round(min(score, 1.0), 4), reasons=reasons, features=dict(features)
        )
