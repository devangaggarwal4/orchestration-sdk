"""``hybrid-router tune``: suggest thresholds and weights from telemetry.

Suggestions are explained with the evidence behind them and priced with the
what-if replay. Nothing is applied automatically; the output includes a YAML
patch to review.
"""

from __future__ import annotations

import math
from typing import Any

import yaml
from pydantic import BaseModel

from hybrid_router.config import RouterConfig
from hybrid_router.reporting.queries import RouteRow
from hybrid_router.reporting.whatif import replay
from hybrid_router.scoring.rule import RuleScorer

BASE_FEATURES = ("tokens", "tools", "task_type", "keywords", "schema")


class Suggestion(BaseModel):
    param: str
    current: Any
    suggested: Any
    rationale: str
    est_cost_delta_usd: float | None = None
    est_latency_delta_ms: float | None = None
    est_new_under_escalations: float | None = None


class Bin(BaseModel):
    lo: float
    hi: float
    n: int
    local_fail_rate: float


class TuneReport(BaseModel):
    enough_data: bool
    message: str
    samples: int
    bins: list[Bin] = []
    suggestions: list[Suggestion] = []
    auc_current: float | None = None
    auc_fitted: float | None = None
    yaml_patch: str = ""


def _bins(routes: list[RouteRow], width: float) -> list[Bin]:
    out = []
    for i in range(int(round(1 / width))):
        lo, hi = round(i * width, 4), round((i + 1) * width, 4)
        rs = [r for r in routes if lo <= r.score < hi or (hi >= 1 and r.score == 1)]
        if rs:
            fails = sum(bool(r.local_failed) for r in rs)
            out.append(Bin(lo=lo, hi=hi, n=len(rs), local_fail_rate=fails / len(rs)))
    return out


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def fit_logistic(
    x: list[list[float]], y: list[bool], iters: int = 3000, lr: float = 0.5, l2: float = 0.01
) -> list[float]:
    """Plain gradient-descent logistic regression (no numpy needed). Returns [bias, *w]."""
    k = len(x[0])
    w = [0.0] * (k + 1)
    n = len(x)
    for _ in range(iters):
        grad = [0.0] * (k + 1)
        for xi, yi in zip(x, y, strict=True):
            z = w[0] + sum(wj * xj for wj, xj in zip(w[1:], xi, strict=True))
            p = 1 / (1 + math.exp(-max(min(z, 30), -30)))
            err = p - (1.0 if yi else 0.0)
            grad[0] += err
            for j in range(k):
                grad[j + 1] += err * xi[j]
        for j in range(k + 1):
            reg = l2 * w[j] if j else 0.0
            w[j] -= lr * (grad[j] / n + reg)
    return w


def suggest(
    routes: list[RouteRow],
    config: RouterConfig,
    *,
    min_samples: int = 50,
    min_bin: int = 5,
    width: float = 0.05,
    target_escalation: float = 0.7,
    easy_success: float = 0.9,
) -> TuneReport:
    data = [r for r in routes if r.local_failed is not None and not r.forced and r.features]
    if len(data) < min_samples:
        return TuneReport(
            enough_data=False,
            samples=len(data),
            message=f"need at least {min_samples} routes with a local attempt; have {len(data)}. "
            "Keep routing (or run `hybrid-router eval`) and try again.",
        )
    rc = config.routing
    bins = _bins(data, width)
    solid = [b for b in bins if b.n >= min_bin]
    suggestions: list[Suggestion] = []
    candidate = config.model_copy(deep=True)

    # hard_threshold: lowest score from which local keeps failing.
    new_hard = None
    for i, b in enumerate(solid):
        tail = solid[i:]
        if b.lo >= rc.easy_threshold and all(t.local_fail_rate >= target_escalation for t in tail):
            new_hard = b.lo
            break
    if new_hard is not None and new_hard < rc.hard_threshold - 1e-9:
        span = [b for b in solid if new_hard <= b.lo < rc.hard_threshold]
        n = sum(b.n for b in span)
        rate = sum(b.local_fail_rate * b.n for b in span) / max(n, 1)
        candidate.routing.hard_threshold = new_hard
        candidate.routing.opus_threshold = max(rc.opus_threshold, new_hard)
        suggestions.append(
            Suggestion(
                param="routing.hard_threshold",
                current=rc.hard_threshold,
                suggested=new_hard,
                rationale=f"scores {new_hard:.2f}-{rc.hard_threshold:.2f} escalated {rate:.0%} of "
                f"the time (n={n}); sending them straight to the cloud skips a local attempt "
                "that usually fails",
            )
        )
    else:
        above = [b for b in solid if b.lo >= rc.hard_threshold]
        if above and all(b.local_fail_rate < 1 - target_escalation for b in above):
            raised = min(above[-1].hi, rc.opus_threshold)
            if raised > rc.hard_threshold + 1e-9:
                candidate.routing.hard_threshold = raised
                suggestions.append(
                    Suggestion(
                        param="routing.hard_threshold",
                        current=rc.hard_threshold,
                        suggested=raised,
                        rationale=f"local succeeded {1 - max(b.local_fail_rate for b in above):.0%}"
                        f"+ of the time above {rc.hard_threshold:.2f} (cascade data); trying local "
                        "first there would save cloud cost",
                    )
                )

    # easy_threshold: highest score below which local almost always succeeds.
    new_easy = 0.0
    for b in solid:
        if 1 - b.local_fail_rate >= easy_success:
            new_easy = b.hi
        else:
            break
    new_easy = min(new_easy, candidate.routing.hard_threshold)
    if abs(new_easy - rc.easy_threshold) >= width - 1e-9:
        below = [b for b in solid if b.hi <= max(new_easy, rc.easy_threshold)]
        n = sum(b.n for b in below)
        ok = 1 - sum(b.local_fail_rate * b.n for b in below) / max(n, 1)
        direction = "raise" if new_easy > rc.easy_threshold else "lower"
        candidate.routing.easy_threshold = new_easy
        suggestions.append(
            Suggestion(
                param="routing.easy_threshold",
                current=rc.easy_threshold,
                suggested=new_easy,
                rationale=f"local succeeded {ok:.0%} of the time below {new_easy:.2f} (n={n}); "
                f"{direction} it so "
                + (
                    "more requests stream immediately with light checks"
                    if direction == "raise"
                    else "fewer risky requests skip full validation"
                ),
            )
        )

    # Weights: refit feature importance to predict local failure.
    x = [[r.features.get(f, 0.0) for f in BASE_FEATURES] for r in data]
    y = [bool(r.local_failed) for r in data]
    auc_current = auc([r.score for r in data], y)
    coef = fit_logistic(x, y)[1:]
    top = max((c for c in coef if c > 0), default=0.0)
    auc_fitted = None
    if top > 0:
        fitted = {
            f: round(max(c, 0.0) / top * 0.9, 2) for f, c in zip(BASE_FEATURES, coef, strict=True)
        }
        fitted = {f: max(v, 0.05) for f, v in fitted.items()}
        trial = config.scorer.model_copy(update={"weights": fitted})
        scorer = RuleScorer(trial)
        auc_fitted = auc([scorer.score_features(r.features).score for r in data], y)
        if auc_fitted is not None and auc_current is not None and auc_fitted > auc_current + 0.03:
            candidate.scorer.weights = fitted
            suggestions.append(
                Suggestion(
                    param="scorer.weights",
                    current=config.scorer.weights,
                    suggested=fitted,
                    rationale=f"refit on {len(data)} routes predicts local failure better "
                    f"(AUC {auc_current:.2f} → {auc_fitted:.2f})",
                )
            )

    if suggestions:
        impact = replay(routes, config, candidate)
        for s in suggestions:  # combined impact of the whole patch
            s.est_cost_delta_usd = impact["est_cost_delta_usd"]
            s.est_latency_delta_ms = impact["est_latency_delta_ms"]
            s.est_new_under_escalations = impact["est_new_under_escalations"]

    patch: dict[str, Any] = {}
    for s in suggestions:
        section, key = s.param.split(".")
        patch.setdefault(section, {})[key] = s.suggested
    if (
        "hard_threshold" in patch.get("routing", {})
        and candidate.routing.opus_threshold != rc.opus_threshold
    ):
        patch["routing"]["opus_threshold"] = candidate.routing.opus_threshold
    return TuneReport(
        enough_data=True,
        samples=len(data),
        message="no changes suggested; thresholds match the data"
        if not suggestions
        else f"{len(suggestions)} suggestion(s)",
        bins=bins,
        suggestions=suggestions,
        auc_current=auc_current,
        auc_fitted=auc_fitted,
        yaml_patch=yaml.safe_dump(patch, sort_keys=False) if patch else "",
    )


def apply_patch(config: RouterConfig, patch: dict[str, Any]) -> RouterConfig:
    """Return a validated copy of ``config`` with a tune/what-if patch applied."""
    from hybrid_router.config import deep_merge

    data = config.model_dump(mode="json")
    return RouterConfig.model_validate(deep_merge(data, patch))
