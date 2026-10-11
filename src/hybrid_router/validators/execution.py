"""Execution checks: Python in the answer must compile; optional caller test hook."""

from __future__ import annotations

import inspect
import re

from hybrid_router.config import ExecutionConfig
from hybrid_router.decision import ValidatorResult
from hybrid_router.validators.base import ValidationContext, Validator

_PY_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.DOTALL)


def extract_python(text: str, *, assume_python: bool) -> list[str]:
    blocks = _PY_FENCE.findall(text)
    if not blocks and assume_python:
        blocks = _ANY_FENCE.findall(text) or ([text] if "def " in text or "import " in text else [])
    return blocks


class ExecutionValidator(Validator):
    name = "execution"

    def __init__(self, cfg: ExecutionConfig | None = None):
        self.cfg = cfg or ExecutionConfig()

    async def check(self, ctx: ValidationContext) -> ValidatorResult:
        if ctx.response.tool_calls:
            # The tool runs after we return; its errors feed the next step's score.
            return self.skip("tool call; result is checked on the next step")
        assume = (ctx.request.task_type or "") in self.cfg.code_task_types
        blocks = extract_python(ctx.response.text, assume_python=assume)
        if not blocks:
            return self.skip("no code to check")
        if self.cfg.python_compile:
            for i, code in enumerate(blocks):
                try:
                    compile(code, f"<answer block {i + 1}>", "exec")
                except SyntaxError as e:
                    return self.fail(
                        "hard", f"code block {i + 1} does not compile: {e.msg} (line {e.lineno})"
                    )
        if ctx.test_runner is not None:
            code = "\n\n".join(blocks)
            try:
                passed = ctx.test_runner(code, ctx.request)
                if inspect.isawaitable(passed):
                    passed = await passed
            except Exception as e:  # the hook is user code; any crash is a failure
                return self.fail("hard", f"test runner raised {type(e).__name__}: {e}")
            if not passed:
                return self.fail("hard", "caller's tests failed")
            return self.ok(f"{len(blocks)} block(s) compile, tests pass")
        return self.ok(f"{len(blocks)} block(s) compile")
