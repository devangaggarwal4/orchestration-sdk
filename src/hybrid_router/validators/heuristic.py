"""Cheap text heuristics: refusals/uncertainty, repetition loops, too-short answers,
empty required fields."""

from __future__ import annotations

import json
import re

from hybrid_router.config import HeuristicConfig
from hybrid_router.decision import ValidatorResult
from hybrid_router.validators.base import ValidationContext, Validator
from hybrid_router.validators.structural import parse_json_answer


def repetition_ratio(text: str, n: int = 6) -> float:
    """Fraction of n-word shingles that are repeats. 0 = no repetition."""
    words = text.split()
    if len(words) < n * 4:
        return 0.0
    shingles = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(shingles)) / len(shingles)


class HeuristicValidator(Validator):
    name = "heuristic"

    def __init__(self, cfg: HeuristicConfig | None = None):
        self.cfg = cfg or HeuristicConfig()
        self._refusals = [re.compile(p, re.IGNORECASE) for p in self.cfg.refusal_patterns]

    async def check(self, ctx: ValidationContext) -> ValidatorResult:
        req, resp = ctx.request, ctx.response
        if resp.tool_calls:
            return self.ok("tool call; text heuristics not applicable")
        text = resp.text.strip()

        head = text[:400]
        for rx in self._refusals:
            if m := rx.search(head):
                return self.fail("soft", f"refusal/uncertainty phrase: {m.group(0)!r}")

        ratio = repetition_ratio(text)
        if ratio > self.cfg.repetition_threshold:
            return self.fail("soft", f"repetition loop ({ratio:.0%} repeated phrases)")

        min_chars = self.cfg.min_chars.get(req.task_type or "", self.cfg.min_chars["default"])
        if len(text) < min_chars:
            return self.fail("soft", f"suspiciously short ({len(text)} < {min_chars} chars)")

        if req.response_schema and isinstance(req.response_schema.get("required"), list):
            try:
                doc = parse_json_answer(text)
            except json.JSONDecodeError:
                return self.ok()  # structural already judged the JSON
            if isinstance(doc, dict):
                empty = [
                    k for k in req.response_schema["required"] if doc.get(k) in (None, "", [], {})
                ]
                if empty:
                    return self.fail("soft", f"required field(s) empty: {', '.join(empty)}")
        return self.ok()
