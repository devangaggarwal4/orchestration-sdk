"""Web app persistence (conversations, runs, run events) in the telemetry DB."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from hybrid_router.telemetry.store import connect, init_db
from hybrid_router.types import Message

WEB_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    settings TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS conversation_messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    decisions TEXT NOT NULL DEFAULT '[]',
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS conv_msgs ON conversation_messages(conversation_id, idx);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created REAL NOT NULL,
    finished REAL,
    params TEXT NOT NULL,
    summary TEXT,
    config_hash TEXT
);
CREATE TABLE IF NOT EXISTS run_events (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    event TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


class WebStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = connect(self.db_path)
        init_db(self._conn)
        self._conn.executescript(WEB_SCHEMA)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()

    def _exec(self, sql: str, params: tuple | list = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _all(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # --- conversations ---------------------------------------------------------

    def create_conversation(
        self, title: str = "New chat", settings: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        cid = uuid.uuid4().hex[:12]
        now = time.time()
        self._exec(
            "INSERT INTO conversations(id, title, created, updated, settings) VALUES (?,?,?,?,?)",
            (cid, title, now, now, json.dumps(settings or {})),
        )
        return {
            "id": cid,
            "title": title,
            "created": now,
            "updated": now,
            "settings": settings or {},
        }

    def list_conversations(self) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM conversations ORDER BY updated DESC")
        return [{**dict(r), "settings": json.loads(r["settings"])} for r in rows]

    def get_conversation(self, cid: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM conversations WHERE id = ?", (cid,))
        if not rows:
            return None
        msgs = self._all(
            "SELECT * FROM conversation_messages WHERE conversation_id = ? ORDER BY idx", (cid,)
        )
        return {
            **dict(rows[0]),
            "settings": json.loads(rows[0]["settings"]),
            "messages": [
                {
                    "id": m["id"],
                    "idx": m["idx"],
                    "role": m["role"],
                    "content": json.loads(m["content"]),
                    "decisions": json.loads(m["decisions"]),
                    "created": m["created"],
                }
                for m in msgs
            ],
        }

    def history(self, cid: str) -> list[Message]:
        conv = self.get_conversation(cid)
        if conv is None:
            return []
        return [Message(role=m["role"], content=m["content"]) for m in conv["messages"]]

    def append_messages(
        self, cid: str, messages: list[Message], decisions: list[list[dict[str, Any]]] | None = None
    ) -> list[str]:
        decisions = decisions or [[] for _ in messages]
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(idx), -1) FROM conversation_messages WHERE conversation_id=?",
                (cid,),
            ).fetchone()
            idx = int(row[0]) + 1
            ids = []
            now = time.time()
            for m, d in zip(messages, decisions, strict=True):
                mid = uuid.uuid4().hex[:12]
                ids.append(mid)
                content = [b.model_dump(mode="json") for b in m.blocks()]
                self._conn.execute(
                    "INSERT INTO conversation_messages VALUES (?,?,?,?,?,?,?)",
                    (mid, cid, idx, m.role, json.dumps(content), json.dumps(d), now),
                )
                idx += 1
            self._conn.execute("UPDATE conversations SET updated = ? WHERE id = ?", (now, cid))
            self._conn.commit()
        return ids

    def update_conversation(
        self, cid: str, *, title: str | None = None, settings: dict[str, Any] | None = None
    ) -> None:
        if title is not None:
            self._exec("UPDATE conversations SET title = ? WHERE id = ?", (title, cid))
        if settings is not None:
            self._exec(
                "UPDATE conversations SET settings = ? WHERE id = ?", (json.dumps(settings), cid)
            )

    def delete_conversation(self, cid: str) -> bool:
        self._exec("DELETE FROM conversation_messages WHERE conversation_id = ?", (cid,))
        return self._exec("DELETE FROM conversations WHERE id = ?", (cid,)).rowcount > 0

    # --- runs ---------------------------------------------------------------------

    def create_run(self, kind: str, params: dict[str, Any], config_hash: str) -> str:
        rid = uuid.uuid4().hex[:12]
        self._exec(
            "INSERT INTO runs(id, kind, status, created, params, config_hash) VALUES (?,?,?,?,?,?)",
            (rid, kind, "running", time.time(), json.dumps(params), config_hash),
        )
        return rid

    def add_run_event(self, run_id: str, seq: int, event: dict[str, Any]) -> None:
        self._exec("INSERT INTO run_events VALUES (?,?,?)", (run_id, seq, json.dumps(event)))

    def finish_run(self, run_id: str, status: str, summary: dict[str, Any] | None) -> None:
        self._exec(
            "UPDATE runs SET status = ?, finished = ?, summary = ? WHERE id = ?",
            (status, time.time(), json.dumps(summary) if summary is not None else None, run_id),
        )

    def run_events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._all("SELECT event FROM run_events WHERE run_id = ? ORDER BY seq", (run_id,))
        return [json.loads(r["event"]) for r in rows]

    def list_runs(self, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM runs", []
        if kind:
            sql += " WHERE kind = ?"
            params.append(kind)
        rows = self._all(sql + " ORDER BY created DESC LIMIT ?", [*params, limit])
        return [self._run_dict(r) for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        rows = self._all("SELECT * FROM runs WHERE id = ?", (run_id,))
        return self._run_dict(rows[0]) if rows else None

    def mark_interrupted(self) -> None:
        """Runs left 'running' by a previous process can't resume."""
        self._exec("UPDATE runs SET status = 'interrupted' WHERE status = 'running'")

    @staticmethod
    def _run_dict(r: sqlite3.Row) -> dict[str, Any]:
        return {
            **dict(r),
            "params": json.loads(r["params"]),
            "summary": json.loads(r["summary"]) if r["summary"] else None,
        }

    def close(self) -> None:
        self._conn.close()
