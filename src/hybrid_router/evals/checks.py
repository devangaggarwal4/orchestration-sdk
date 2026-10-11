"""Answer checks for eval tasks. Every check returns (passed, detail).

Checks run trusted code from tasks.yaml (json_predicate) and model-written code
(python_tests) — the latter in a separate interpreter with a timeout, in a temp dir.
"""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hybrid_router.validators.structural import parse_json_answer

_FENCE = r"```{lang}\s*\n(.*?)```"


@dataclass
class Answer:
    text: str
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def _code(text: str, lang: str) -> str:
    m = re.search(_FENCE.format(lang=lang), text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(_FENCE.format(lang=""), text, re.DOTALL)
    return m.group(1) if m else text


def _norm(v: Any) -> Any:
    return v.strip().lower() if isinstance(v, str) else v


def _same(got: Any, want: Any) -> bool:
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            return False
        return sorted(map(str, map(_norm, got))) == sorted(map(str, map(_norm, want)))
    if isinstance(want, int | float) and not isinstance(want, bool):
        return isinstance(got, int | float) and abs(got - want) < 1e-6
    return _norm(got) == _norm(want)


def run_check(check: dict[str, Any], answer: Answer) -> tuple[bool, str]:
    kind = check["type"]
    text = answer.text.strip()
    low = text.lower()

    if kind == "regex":
        ok = re.search(check["pattern"], text, re.IGNORECASE) is not None
        return ok, "" if ok else f"no match for /{check['pattern']}/"
    if kind == "contains_all":
        missing = [v for v in check["values"] if str(v).lower() not in low]
        return not missing, f"missing {missing}" if missing else ""
    if kind == "contains_any":
        ok = any(str(v).lower() in low for v in check["values"])
        return ok, "" if ok else f"none of {check['values']}"
    if kind == "contains_none":
        found = [v for v in check["values"] if str(v).lower() in low]
        return not found, f"contains {found}" if found else ""

    if kind in ("json_fields", "json_predicate"):
        try:
            doc = parse_json_answer(text)
        except json.JSONDecodeError as e:
            return False, f"not JSON: {e.msg}"
        if kind == "json_fields":
            if not isinstance(doc, dict):
                return False, "not a JSON object"
            bad = [k for k, want in check["expected"].items() if not _same(doc.get(k), want)]
            return not bad, f"wrong fields {bad}" if bad else ""
        scope: dict[str, Any] = {}
        exec(check["code"], scope)  # trusted: comes from tasks.yaml
        try:
            ok = bool(scope["check"](doc))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            return False, f"predicate error: {type(e).__name__}: {e}"
        return ok, "" if ok else "predicate false"

    if kind == "python_tests":
        return _python_tests(
            _code(text, "(?:python|py)"), check["tests"], float(check.get("timeout_s", 10))
        )

    if kind == "sql_result":
        query = _code(text, "sql").strip()
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(check["setup"])
            rows = [list(r) for r in conn.execute(query).fetchall()]
        except sqlite3.Error as e:
            return False, f"sql error: {e}"
        finally:
            conn.close()
        ok = rows == check["expected"]
        return ok, "" if ok else f"got {rows}"

    if kind == "tool_call":
        for name, args in answer.tool_calls:
            if name != check["name"]:
                continue
            want = check.get("args", {})
            if all(str(v).lower() in str(args.get(k, "")).lower() for k, v in want.items()):
                return True, ""
        return False, f"no {check['name']} call with {check.get('args')}"
    if kind == "tools_used":
        used = [n for n, _ in answer.tool_calls if n in check["names"]]
        ok = len(used) >= check.get("min_calls", 1)
        return ok, "" if ok else f"tool calls {used}"
    raise ValueError(f"unknown check type {kind!r}")


def _python_tests(code: str, tests: str, timeout_s: float = 10.0) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "answer_test.py"
        script.write_text(f"{code}\n\n# --- tests ---\n{tests}\n")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env={"PATH": "/usr/bin:/bin"},
            )
        except subprocess.TimeoutExpired:
            return False, f"tests timed out after {timeout_s}s"
    if proc.returncode == 0:
        return True, ""
    err = (proc.stderr.strip().splitlines() or ["failed"])[-1]
    return False, err[:200]


def run_checks(checks: list[dict[str, Any]], answer: Answer) -> tuple[bool, list[str]]:
    failures = []
    for c in checks:
        ok, detail = run_check(c, answer)
        if not ok:
            failures.append(f"{c['type']}: {detail}")
    return not failures, failures
