"""Cheapest checks: empty/truncated output, JSON + schema, malformed or unknown tool calls."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from hybrid_router.decision import ValidatorResult
from hybrid_router.validators.base import ValidationContext, Validator

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def parse_json_answer(text: str) -> Any:
    """Parse a JSON answer, tolerating a surrounding ```json fence."""
    if m := _FENCE.match(text):
        text = m.group(1)
    return json.loads(text)


class StructuralValidator(Validator):
    name = "structural"

    async def check(self, ctx: ValidationContext) -> ValidatorResult:
        req, resp = ctx.request, ctx.response
        calls = resp.tool_calls
        text = resp.text.strip()

        if resp.stop_reason == "refusal":
            return self.fail("hard", "model refused (stop_reason=refusal)")
        if not text and not calls:
            return self.fail("hard", "empty output")
        if resp.stop_reason == "max_tokens":
            if calls:
                return self.fail("hard", "truncated mid tool call (max_tokens)")
            return self.fail("soft", "output truncated (max_tokens)")

        tools = {t.name: t for t in req.tools}
        for call in calls:
            if call.name not in tools:
                return self.fail("hard", f"call to nonexistent tool {call.name!r}")
            if call.raw_arguments is not None:
                return self.fail("hard", f"malformed arguments for {call.name!r}")
            try:
                jsonschema.validate(call.input, tools[call.name].input_schema)
            except jsonschema.ValidationError as e:
                return self.fail("hard", f"{call.name} arguments invalid: {e.message}")
            except jsonschema.SchemaError:
                pass  # the caller's schema is broken; not the model's fault

        if req.response_schema is not None and not calls:
            try:
                doc = parse_json_answer(text)
            except json.JSONDecodeError as e:
                return self.fail("hard", f"invalid JSON: {e.msg}")
            try:
                jsonschema.validate(doc, req.response_schema)
            except jsonschema.ValidationError as e:
                return self.fail("hard", f"schema mismatch: {e.message}")
            except jsonschema.SchemaError:
                pass
        return self.ok()
