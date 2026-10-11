"""RuleScorer: explainable features combined noisy-OR, plus session boosts.

Each reason shows a feature and its own signal strength (weight x value); the
score is 1 - prod(1 - signal), then sticky/tool-error boosts are added.
"""

from __future__ import annotations

from hybrid_router.config import RuleScorerConfig
from hybrid_router.scoring.base import ScoreResult, ScoringContext
from hybrid_router.scoring.features import FeatureExtractor
from hybrid_router.types import CompletionRequest

BOOST_KEYS = ("sticky_remaining", "prior_local_tool_error")


class RuleScorer:
    name = "rule"

    def __init__(self, cfg: RuleScorerConfig | None = None):
        self.cfg = cfg or RuleScorerConfig()
        self._extractor = FeatureExtractor(self.cfg)

    def score(self, req: CompletionRequest, ctx: ScoringContext) -> ScoreResult:
        features, details = self._extractor.extract(req)
        features["sticky_remaining"] = float(ctx.sticky_remaining)
        features["prior_local_tool_error"] = 1.0 if ctx.prior_local_tool_error else 0.0
        return self.score_features(features, details)

    def score_features(
        self, features: dict[str, float], details: dict[str, str] | None = None
    ) -> ScoreResult:
        """Score a stored feature vector. Used live and by what-if replay."""
        cfg = self.cfg
        details = details or {}
        weights = {k: min(max(w, 0.0), 1.0) for k, w in cfg.weights.items() if k in features}

        # Noisy-OR: each feature independently pushes toward "hard".
        signals = {k: w * min(max(features[k], 0.0), 1.0) for k, w in weights.items()}
        remaining = 1.0
        for v in signals.values():
            remaining *= 1.0 - v
        score = 1.0 - remaining
        reasons: list[str] = []
        for key, v in sorted(signals.items(), key=lambda kv: -kv[1]):
            if v >= cfg.reason_min_contribution:
                detail = details.get(key) or f"{key}={features[key]:.2f}"
                reasons.append(f"{detail} ({v:.2f})")

        sticky = int(features.get("sticky_remaining", 0))
        if sticky > 0 and cfg.sticky_steps > 0:
            boost = cfg.sticky_boost * min(sticky, cfg.sticky_steps) / cfg.sticky_steps
            score += boost
            reasons.append(f"sticky after escalation, {sticky} step(s) left (+{boost:.2f})")
        if features.get("prior_local_tool_error", 0) > 0:
            score += cfg.tool_error_boost
            reasons.append(f"previous local step's tool call errored (+{cfg.tool_error_boost:.2f})")

        score = min(max(score, 0.0), 1.0)
        return ScoreResult(score=round(score, 4), reasons=reasons, features=dict(features))
