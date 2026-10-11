"""Optional LLM judge (off by default). Asks a model whether the answer is acceptable."""

from __future__ import annotations

from hybrid_router.backends.base import BackendError
from hybrid_router.config import JudgeConfig
from hybrid_router.decision import ValidatorResult
from hybrid_router.types import CompletionRequest, Message
from hybrid_router.validators.base import ValidationContext, Validator

JUDGE_SYSTEM = (
    "You review an assistant's answer for correctness and completeness. "
    "Reply with PASS or FAIL on the first line, then one short sentence of reasoning."
)


class JudgeValidator(Validator):
    name = "judge"

    def __init__(self, cfg: JudgeConfig | None = None):
        self.cfg = cfg or JudgeConfig()

    async def check(self, ctx: ValidationContext) -> ValidatorResult:
        if ctx.judge_backend is None:
            return self.skip("no judge backend")
        question = ctx.request.last_user_text()[:6000]
        answer = ctx.response.text[:6000] or str([c.model_dump() for c in ctx.response.tool_calls])
        prompt = f"Task:\n{question}\n\nAnswer:\n{answer}\n\nPASS or FAIL?"
        try:
            verdict = await ctx.judge_backend.generate(
                CompletionRequest(
                    system=JUDGE_SYSTEM,
                    messages=[Message(role="user", content=prompt)],
                    max_tokens=self.cfg.max_tokens,
                )
            )
        except BackendError as e:
            return self.skip(f"judge failed: {e}")
        first = verdict.text.strip().splitlines()[0].upper() if verdict.text.strip() else ""
        result = (
            self.ok(verdict.text.strip()[:200])
            if first.startswith("PASS")
            else self.fail("soft", f"judge: {verdict.text.strip()[:200]}")
        )
        result.extra_usage = verdict.usage
        result.extra_cost_usd = verdict.reported_cost_usd
        return result
