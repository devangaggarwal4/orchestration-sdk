"""Background runs (agent / compare / eval) with live event fan-out.

Every event is persisted, so a run can be watched live or replayed later; a
subscriber first gets the backlog, then live events, then an end marker.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from hybrid_router.web.store import WebStore

log = logging.getLogger("hybrid_router.web")

Emit = Callable[[dict[str, Any]], None]
RunBody = Callable[[Emit], Awaitable[dict[str, Any] | None]]
END = {"type": "run_end"}


def sse(data: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(data, default=str)}\n\n".encode()


@dataclass
class Run:
    id: str
    kind: str
    events: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)
    task: asyncio.Task[None] | None = None
    status: str = "running"


class RunManager:
    def __init__(self, store: WebStore):
        self.store = store
        self.runs: dict[str, Run] = {}

    def start(self, kind: str, params: dict[str, Any], config_hash: str, body: RunBody) -> str:
        run_id = self.store.create_run(kind, params, config_hash)
        run = Run(run_id, kind)
        self.runs[run_id] = run

        def emit(event: dict[str, Any]) -> None:
            seq = len(run.events)
            event = {"seq": seq, **event}
            run.events.append(event)
            self.store.add_run_event(run_id, seq, event)
            for q in run.subscribers:
                q.put_nowait(event)

        async def go() -> None:
            summary: dict[str, Any] | None = None
            try:
                summary = await body(emit)
                run.status = "done"
            except asyncio.CancelledError:
                run.status = "cancelled"
                emit({"type": "run_cancelled"})
            except Exception as e:  # a failing run must not take the server down
                log.exception("run %s failed", run_id)
                run.status = "error"
                emit({"type": "run_error", "error": f"{type(e).__name__}: {e}"})
            self.store.finish_run(run_id, run.status, summary)
            emit({**END, "status": run.status, "summary": summary})
            for q in run.subscribers:
                q.put_nowait(END)

        run.task = asyncio.create_task(go(), name=f"run-{run_id}")
        return run_id

    def cancel(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if run is None or run.task is None or run.task.done():
            return False
        run.task.cancel()
        return True

    async def subscribe(self, run_id: str, heartbeat_s: float = 15.0) -> AsyncIterator[bytes]:
        run = self.runs.get(run_id)
        if run is None:
            # Finished in an earlier server process: replay from the DB.
            for event in self.store.run_events(run_id):
                yield sse(event)
            return
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        backlog = list(run.events)
        run.subscribers.add(queue)
        try:
            for event in backlog:
                yield sse(event)
            if backlog and backlog[-1].get("type") == END["type"]:
                return
            seen = len(backlog)
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), heartbeat_s)
                except TimeoutError:
                    yield b": ping\n\n"
                    continue
                if event is END:
                    return
                if event.get("seq", seen) < seen:
                    continue  # already sent in the backlog
                yield sse(event)
                if event.get("type") == END["type"]:
                    return
        finally:
            run.subscribers.discard(queue)

    async def shutdown(self) -> None:
        for run in self.runs.values():
            if run.task and not run.task.done():
                run.task.cancel()
        tasks = [r.task for r in self.runs.values() if r.task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
