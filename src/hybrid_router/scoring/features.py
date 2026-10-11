"""Feature extraction shared by scorers. Every feature is normalized to [0, 1]."""

from __future__ import annotations

import re
from typing import Any

from hybrid_router.config import RuleScorerConfig
from hybrid_router.types import CompletionRequest, estimate_tokens


def ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value >= high else 0.0
    return min(max((value - low) / (high - low), 0.0), 1.0)


def schema_units(schema: Any, depth: int = 0) -> int:
    """Rough size of a JSON schema: properties, nesting, unions, enums, required."""
    if not isinstance(schema, dict):
        return 0
    units = depth  # deeper nesting costs more
    props = schema.get("properties")
    if isinstance(props, dict):
        units += len(props)
        units += sum(schema_units(v, depth + 1) for v in props.values())
    for key in ("anyOf", "oneOf", "allOf"):
        if isinstance(schema.get(key), list):
            units += 2 * len(schema[key])
            units += sum(schema_units(v, depth + 1) for v in schema[key])
    if isinstance(schema.get("items"), dict):
        units += 1 + schema_units(schema["items"], depth + 1)
    if isinstance(schema.get("enum"), list):
        units += 1
    if isinstance(schema.get("required"), list):
        units += len(schema["required"]) // 2
    for defs_key in ("$defs", "definitions"):
        if isinstance(schema.get(defs_key), dict):
            units += sum(schema_units(v, depth + 1) for v in schema[defs_key].values())
    return units


class FeatureExtractor:
    def __init__(self, cfg: RuleScorerConfig):
        self.cfg = cfg
        self._keywords = [(re.compile(k.pattern, re.IGNORECASE), k) for k in cfg.keywords]

    def extract(self, req: CompletionRequest) -> tuple[dict[str, float], dict[str, str]]:
        """Returns normalized features and a short human detail for each."""
        cfg = self.cfg
        tokens = estimate_tokens(req)
        n_tools = len(req.tools)
        task_value = cfg.task_types.get(req.task_type or "", cfg.task_type_default)

        text = req.last_user_text()
        if len(text) > 2 * cfg.keyword_scan_chars:
            # Instructions sit at the start or end; skip pasted context in the middle.
            text = text[: cfg.keyword_scan_chars] + "\n" + text[-cfg.keyword_scan_chars :]
        kw_value, kw_hits = 0.0, []
        for rx, rule in self._keywords:
            if m := rx.search(text):
                kw_value = max(kw_value, rule.weight)
                kw_hits.append(m.group(0).lower())

        units = schema_units(req.response_schema) if req.response_schema else 0
        tool_units = sum(schema_units(t.input_schema) for t in req.tools)

        features = {
            "tokens": ramp(tokens, cfg.tokens_low, cfg.tokens_high),
            "tools": min(n_tools / max(cfg.tools_saturation, 1), 1.0),
            "task_type": task_value,
            "keywords": kw_value,
            "schema": min((units + tool_units / 4) / max(cfg.schema_saturation, 1), 1.0),
        }
        details = {
            "tokens": f"~{tokens / 1000:.1f}k tokens",
            "tools": f"{n_tools} tools",
            "task_type": f"task_type={req.task_type or 'unknown'}",
            "keywords": "keywords: " + ", ".join(sorted(set(kw_hits))) if kw_hits else "",
            "schema": f"schema size {units}" + (f" (+tools {tool_units})" if tool_units else ""),
        }
        return features, details
