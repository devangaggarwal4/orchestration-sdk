"""Model confidence: mean token logprob, and optional two-sample disagreement."""

from __future__ import annotations

import difflib

from hybrid_router.backends.base import BackendError
from hybrid_router.config import ConfidenceConfig
from hybrid_router.decision import ValidatorResult
from hybrid_router.validators.base import ValidationContext, Validator


def _signature(text: str, calls: list) -> str:
    if calls:
        return "|".join(sorted(c.name for c in calls))
    return " ".join(text.lower().split())


class ConfidenceValidator(Validator):
    name = "confidence"

    def __init__(self, cfg: ConfidenceConfig | None = None):
        self.cfg = cfg or ConfidenceConfig()

    async def check(self, ctx: ValidationContext) -> ValidatorResult:
        resp = ctx.response
        mean = resp.mean_logprob
        if mean is not None and mean < self.cfg.min_mean_logprob:
            return self.fail(
                "soft", f"low confidence: mean logprob {mean:.2f} < {self.cfg.min_mean_logprob}"
            )
        if self.cfg.two_sample and ctx.backend is not None:
            return await self._two_sample(ctx)
        if mean is None:
            return self.skip("no logprobs from this backend")
        return self.ok(f"mean logprob {mean:.2f}")

    async def _two_sample(self, ctx: ValidationContext) -> ValidatorResult:
        assert ctx.backend is not None
        req = ctx.request.model_copy(update={"temperature": self.cfg.two_sample_temperature})
        try:
            second = await ctx.backend.generate(req)
        except BackendError as e:
            return self.skip(f"second sample failed: {e}")
        a = _signature(ctx.response.text, ctx.response.tool_calls)
        b = _signature(second.text, second.tool_calls)
        similarity = difflib.SequenceMatcher(None, a, b).ratio()
        result = (
            self.fail("soft", f"two samples disagree (similarity {similarity:.2f})")
            if similarity < self.cfg.two_sample_min_similarity
            else self.ok(f"two samples agree (similarity {similarity:.2f})")
        )
        result.extra_usage = second.usage
        result.extra_cost_usd = second.reported_cost_usd
        return result
