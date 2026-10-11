"""Read-side of telemetry, shared by ``report``, ``tune``, what-if and the web app."""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

_SINCE = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhdw])$")
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_since(value: str | None, now: float | None = None) -> float:
    """'7d', '12h', '30m', '2w', an ISO date/time, or None (everything) -> unix ts."""
    now = time.time() if now is None else now
    if not value:
        return 0.0
    if m := _SINCE.match(value.strip()):
        return now - float(m.group(1)) * _UNITS[m.group(2)]
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        raise ValueError(f"can't parse --since {value!r}; use e.g. 7d, 12h or 2026-10-01") from None


def pct(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    return statistics.quantiles(vals, n=100, method="inclusive")[max(1, min(99, round(q))) - 1]


@dataclass
class RouteRow:
    route_id: str
    ts: float
    session_id: str | None
    task_type: str | None
    strategy: str
    score: float
    zone: str
    features: dict[str, float]
    forced: bool
    planned_tier: str
    final_tier: str | None
    escalated: bool
    escalation_reason: str | None
    success: bool
    total_latency_ms: float
    actual_cost_usd: float | None
    counterfactual_cost_usd: float | None
    config_hash: str | None
    # First local attempt, if the route had one.
    local_role: str | None
    local_ok: bool | None
    local_succeeded: bool | None
    local_latency_ms: float | None
    local_failed_validator: str | None
    cloud_cost_usd: float
    cloud_latency_ms: float | None
    waste_latency_ms: float

    @property
    def local_failed(self) -> bool | None:
        """Local attempt failed validation, or its tool call errored afterwards."""
        if self.local_ok is None:
            return None
        return (not self.local_ok) or self.local_succeeded is False


ROUTES_SQL = """
SELECT r.*,
       l.role AS local_role, l.ok AS local_ok, l.local_succeeded, l.latency_ms AS local_latency_ms,
       l.validator_results AS local_vr,
       COALESCE(c.cloud_cost, 0) AS cloud_cost_usd, c.cloud_latency AS cloud_latency_ms,
       COALESCE(w.waste_ms, 0) AS waste_latency_ms
FROM routes r
LEFT JOIN calls l ON l.route_id = r.route_id AND l.tier = 'local'
     AND l.attempt_index = (SELECT MIN(attempt_index) FROM calls x
                            WHERE x.route_id = r.route_id AND x.tier = 'local')
LEFT JOIN (SELECT route_id, SUM(COALESCE(actual_cost_usd, 0)) AS cloud_cost,
                  SUM(latency_ms) AS cloud_latency
           FROM calls WHERE tier != 'local' GROUP BY route_id) c ON c.route_id = r.route_id
LEFT JOIN (SELECT route_id, SUM(latency_ms) AS waste_ms FROM calls WHERE wasted = 1
           GROUP BY route_id) w ON w.route_id = r.route_id
WHERE r.ts >= ? {extra}
ORDER BY r.ts
"""


def _failed_validator(raw: str | None) -> str | None:
    if not raw:
        return None
    for v in json.loads(raw):
        if not v.get("ok", True) and not v.get("skipped"):
            return v.get("name")
    return None


def _opt_bool(v: Any) -> bool | None:
    return None if v is None else bool(v)


def load_routes(
    conn: sqlite3.Connection, since_ts: float = 0.0, *, source: str | None = None
) -> list[RouteRow]:
    extra = ""
    params: list[Any] = [since_ts]
    if source:
        extra = "AND r.source = ?"
        params.append(source)
    rows = conn.execute(ROUTES_SQL.format(extra=extra), params).fetchall()
    out = []
    for r in rows:
        out.append(
            RouteRow(
                route_id=r["route_id"],
                ts=r["ts"],
                session_id=r["session_id"],
                task_type=r["task_type"],
                strategy=r["strategy"],
                score=r["score"],
                zone=r["zone"],
                features=json.loads(r["features"] or "{}"),
                forced=bool(r["forced"]),
                planned_tier=r["planned_tier"],
                final_tier=r["final_tier"],
                escalated=bool(r["escalated"]),
                escalation_reason=r["escalation_reason"],
                success=bool(r["success"]),
                total_latency_ms=r["total_latency_ms"] or 0.0,
                actual_cost_usd=r["actual_cost_usd"],
                counterfactual_cost_usd=r["counterfactual_cost_usd"],
                config_hash=r["config_hash"],
                local_role=r["local_role"],
                local_ok=_opt_bool(r["local_ok"]),
                local_succeeded=_opt_bool(r["local_succeeded"]),
                local_latency_ms=r["local_latency_ms"],
                local_failed_validator=_failed_validator(r["local_vr"]),
                cloud_cost_usd=r["cloud_cost_usd"],
                cloud_latency_ms=r["cloud_latency_ms"],
                waste_latency_ms=r["waste_latency_ms"],
            )
        )
    return out


def load_calls(conn: sqlite3.Connection, since_ts: float = 0.0) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM calls WHERE ts >= ? ORDER BY ts", (since_ts,)).fetchall()


def meta(conn: sqlite3.Connection, key: str) -> Any:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None
