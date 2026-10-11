"""SQLite telemetry: one row per model call (``calls``) and one per routed request
(``routes``). Writes go through a background thread so they never slow a route.
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hybrid_router.config import RouterConfig
from hybrid_router.cost import (
    TokenRatio,
    call_cost,
    counterfactual,
    token_ratio_sample,
    validator_cost,
)
from hybrid_router.decision import RouteDecision
from hybrid_router.pricing import Pricing, load_pricing
from hybrid_router.types import CompletionRequest, CompletionResponse, Tier

log = logging.getLogger("hybrid_router.telemetry")

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS routes (
    route_id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    session_id TEXT,
    step_index INTEGER,
    task_type TEXT,
    strategy TEXT,
    score REAL,
    zone TEXT,
    reasons TEXT,
    features TEXT,
    forced INTEGER,
    planned_tier TEXT,
    final_tier TEXT,
    final_model TEXT,
    n_attempts INTEGER,
    escalated INTEGER,
    escalation_reason TEXT,
    success INTEGER,
    total_latency_ms REAL,
    actual_cost_usd REAL,
    counterfactual_cost_usd REAL,
    cf_input_tokens_est INTEGER,
    cf_output_tokens_est INTEGER,
    token_ratio REAL,
    config_hash TEXT,
    source TEXT,
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS routes_ts ON routes(ts);
CREATE INDEX IF NOT EXISTS routes_session ON routes(session_id);
CREATE INDEX IF NOT EXISTS routes_run ON routes(run_id);

CREATE TABLE IF NOT EXISTS calls (
    call_id TEXT PRIMARY KEY,
    route_id TEXT NOT NULL REFERENCES routes(route_id),
    ts REAL NOT NULL,
    session_id TEXT,
    step_index INTEGER,
    task_type TEXT,
    strategy TEXT,
    score REAL,
    zone TEXT,
    reasons TEXT,
    tier TEXT,
    model TEXT,
    role TEXT,
    attempt_index INTEGER,
    buffered INTEGER,
    validator_results TEXT,
    ok INTEGER,
    error TEXT,
    wasted INTEGER,
    local_succeeded INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_read_tokens INTEGER,
    cache_write_tokens INTEGER,
    latency_ms REAL,
    ttft_ms REAL,
    queue_ms REAL,
    stop_reason TEXT,
    actual_cost_usd REAL,
    cost_source TEXT,
    validator_cost_usd REAL,
    counterfactual_cost_usd REAL
);
CREATE INDEX IF NOT EXISTS calls_ts ON calls(ts);
CREATE INDEX IF NOT EXISTS calls_route ON calls(route_id);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.commit()


class TelemetryStore:
    """Route observer that persists every route and call."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        config: RouterConfig,
        pricing: Pricing | None = None,
        source: str = "sdk",
    ):
        self.db_path = str(db_path)
        self.config = config
        self.pricing = pricing or load_pricing(config.telemetry.pricing_path)
        self.source = source
        tcfg = config.telemetry
        self.token_ratio = TokenRatio(tcfg.token_ratio_default, tcfg.token_ratio_min_samples)
        # session_id -> (cache_read_tokens, total_input_tokens) over cloud calls
        self._session_cache: dict[str, tuple[int, int]] = {}

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = connect(self.db_path)
        init_db(self._conn)
        self._load_ratio_samples()
        self._day, self._spent_today = _today_start(), 0.0
        row = self._conn.execute(
            "SELECT COALESCE(SUM(actual_cost_usd), 0) FROM calls WHERE tier != 'local' AND ts >= ?",
            (self._day,),
        ).fetchone()
        self._spent_today = float(row[0] or 0.0)

        self._queue: queue.Queue[Callable[[sqlite3.Connection], None] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._writer, name="telemetry", daemon=True)
        self._thread.start()

    @classmethod
    def from_config(cls, config: RouterConfig, **kwargs: Any) -> TelemetryStore:
        return cls(config.telemetry.db_path, config=config, **kwargs)

    # --- observer API --------------------------------------------------------

    async def on_route(
        self,
        decision: RouteDecision,
        request: CompletionRequest,
        response: CompletionResponse | None,
    ) -> None:
        route_row, call_rows = self.build_rows(decision, request)
        self._queue.put(lambda conn: self._insert(conn, route_row, call_rows))

    async def on_local_tool_error(self, call_ids: list[str]) -> None:
        def backfill(conn: sqlite3.Connection) -> None:
            conn.executemany(
                "UPDATE calls SET local_succeeded = 0 WHERE call_id = ?",
                [(cid,) for cid in call_ids],
            )

        self._queue.put(backfill)

    def update_config(self, config: RouterConfig) -> None:
        self.config = config

    def spent_today(self) -> float:
        """Cloud spend since local midnight (actual cost, as recorded)."""
        if _today_start() != self._day:
            self._day, self._spent_today = _today_start(), 0.0
        return self._spent_today

    # --- row building (pure apart from in-memory calibration state) -----------

    def build_rows(
        self, decision: RouteDecision, request: CompletionRequest
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        cfg = self.config
        now = time.time()
        if cfg.routing.escalation_mode == "original":
            self.token_ratio.add(token_ratio_sample(decision))

        cache_ratio = 0.0
        if cfg.telemetry.cache_assumption == "observed" and decision.session_id:
            read, total = self._session_cache.get(decision.session_id, (0, 0))
            cache_ratio = read / total if total else 0.0

        baseline_model = cfg.tiers[cfg.telemetry.counterfactual_baseline].model
        cf = counterfactual(
            decision,
            pricing=self.pricing,
            baseline_model=baseline_model,
            token_ratio=self.token_ratio.value,
            session_cache_ratio=cache_ratio,
        )

        reasons = json.dumps(decision.reasons)
        calls: list[dict[str, Any]] = []
        actual_total: float | None = 0.0
        served_idx = _served_index(decision)
        for a in decision.attempts:
            cost = call_cost(a, self.pricing)
            vcost = validator_cost(a, self.pricing)
            if cost.usd is None:
                actual_total = None
            elif actual_total is not None:
                actual_total += cost.usd + vcost
            if a.tier.is_cloud:
                self.spent_today()  # roll the day over if needed
                self._spent_today += (cost.usd or 0.0) + vcost
            if a.tier.is_cloud and decision.session_id:
                read, total = self._session_cache.get(decision.session_id, (0, 0))
                self._session_cache[decision.session_id] = (
                    read + a.usage.cache_read_tokens,
                    total + a.usage.total_input_tokens,
                )
            calls.append(
                {
                    "call_id": a.call_id,
                    "route_id": decision.route_id,
                    "ts": now,
                    "session_id": decision.session_id,
                    "step_index": decision.step_index,
                    "task_type": decision.task_type,
                    "strategy": decision.strategy,
                    "score": decision.score,
                    "zone": decision.zone.value,
                    "reasons": reasons,
                    "tier": a.tier.value,
                    "model": a.model,
                    "role": a.role,
                    "attempt_index": a.attempt_index,
                    "buffered": int(a.buffered),
                    "validator_results": json.dumps(
                        [v.model_dump(mode="json", exclude_none=True) for v in a.validator_results]
                    ),
                    "ok": int(a.ok),
                    "error": a.error,
                    "wasted": int(a.wasted),
                    "local_succeeded": int(a.ok) if a.tier is Tier.LOCAL else None,
                    "input_tokens": a.usage.input_tokens,
                    "output_tokens": a.usage.output_tokens,
                    "cache_read_tokens": a.usage.cache_read_tokens,
                    "cache_write_tokens": a.usage.cache_write_tokens,
                    "latency_ms": a.latency_ms,
                    "ttft_ms": a.ttft_ms,
                    "queue_ms": a.queue_ms,
                    "stop_reason": a.stop_reason,
                    "actual_cost_usd": cost.usd,
                    "cost_source": cost.source,
                    "validator_cost_usd": vcost,
                    # The route's counterfactual sits on the serving call so SUM() works.
                    "counterfactual_cost_usd": cf.usd if a.attempt_index == served_idx else 0.0,
                }
            )

        route = {
            "route_id": decision.route_id,
            "ts": now,
            "session_id": decision.session_id,
            "step_index": decision.step_index,
            "task_type": decision.task_type,
            "strategy": decision.strategy,
            "score": decision.score,
            "zone": decision.zone.value,
            "reasons": reasons,
            "features": json.dumps(decision.features),
            "forced": int(decision.forced),
            "planned_tier": decision.planned_tier.value,
            "final_tier": decision.final_tier.value if decision.final_tier else None,
            "final_model": decision.final_model,
            "n_attempts": len(decision.attempts),
            "escalated": int(decision.escalated),
            "escalation_reason": decision.escalation_reason,
            "success": int(decision.success),
            "total_latency_ms": decision.total_latency_ms,
            "actual_cost_usd": actual_total,
            "counterfactual_cost_usd": cf.usd,
            "cf_input_tokens_est": cf.input_tokens,
            "cf_output_tokens_est": cf.output_tokens,
            "token_ratio": self.token_ratio.value,
            "config_hash": decision.config_hash,
            "source": request.metadata.get("source", self.source),
            "run_id": request.metadata.get("run_id"),
        }
        return route, calls

    # --- writer thread -------------------------------------------------------

    @staticmethod
    def _insert(
        conn: sqlite3.Connection, route: dict[str, Any], calls: list[dict[str, Any]]
    ) -> None:
        _insert_row(conn, "routes", route)
        for row in calls:
            _insert_row(conn, "calls", row)

    def _writer(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                job(self._conn)
                self._conn.commit()
            except Exception:
                log.exception("telemetry write failed")
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Block until every queued write is committed."""
        self._queue.join()

    def _load_ratio_samples(self) -> None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='token_ratio_samples'"
        ).fetchone()
        if row:
            for v in json.loads(row[0]):
                self.token_ratio.add(float(v))

    def _save_ratio_samples(self) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('token_ratio_samples', ?)",
            (json.dumps(self.token_ratio.samples),),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=10)
        self._save_ratio_samples()
        self._conn.close()

    async def aclose(self) -> None:
        self.close()


def _today_start() -> float:
    now = time.localtime()
    return time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))


def _insert_row(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    conn.execute(f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({marks})", list(row.values()))


def _served_index(decision: RouteDecision) -> int | None:
    if decision.final_tier is None:
        return None
    for a in reversed(decision.attempts):
        if a.tier is decision.final_tier and not a.wasted:
            return a.attempt_index
    return None
