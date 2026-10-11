"""FastAPI app: chat, agent runs, compare/eval runs, config + tuning, stats.

One process owns one HybridRouter, so everything done in the UI is recorded in
the same telemetry DB as SDK and CLI usage.
"""

from __future__ import annotations

import asyncio
import difflib
import hmac
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from hybrid_router import __version__
from hybrid_router.agent import Agent, AgentEndEvent, AgentEvent, StepEndEvent, workspace_tools
from hybrid_router.config import RouterConfig, deep_merge, load_config, resolve_config_path
from hybrid_router.cost import call_cost
from hybrid_router.decision import RouteDecision, RouterEvent
from hybrid_router.evals import ALL_STRATEGIES, load_tasks, run_evals
from hybrid_router.evals.runner import BudgetExceeded, run_task
from hybrid_router.pricing import Pricing, load_pricing
from hybrid_router.reporting import apply_patch, build_report, load_routes, parse_since, replay
from hybrid_router.reporting.tune import TuneReport, suggest
from hybrid_router.router import HybridRouter, build_scorer
from hybrid_router.telemetry import TelemetryStore, connect
from hybrid_router.types import Message, Tier
from hybrid_router.web.demo import demo_backends
from hybrid_router.web.runs import RunManager, sse
from hybrid_router.web.store import WebStore

STATIC_DIR = Path(__file__).parent / "static"
Strategy = Literal["hybrid", "pre_route", "cascade"]


@dataclass
class AppState:
    config_path: Path
    router: HybridRouter
    store: WebStore
    runs: RunManager
    pricing: Pricing
    token: str | None
    fake: bool
    config_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    health_cache: tuple[float, dict[str, Any]] | None = None

    @property
    def config(self) -> RouterConfig:
        return self.router.config


# --- request bodies ---------------------------------------------------------------


class NewConversation(BaseModel):
    title: str = "New chat"
    settings: dict[str, Any] = Field(default_factory=dict)


class ConversationUpdate(BaseModel):
    title: str | None = None
    settings: dict[str, Any] | None = None


class ChatTurn(BaseModel):
    content: str = Field(min_length=1)
    strategy: Strategy | None = None
    task_type: str | None = None
    force_tier: Tier | None = None
    tools: bool = False


class RunRequest(BaseModel):
    kind: Literal["agent", "compare", "eval"]
    # agent
    task: str | None = None
    strategy: Strategy | None = None
    force_tier: Tier | None = None
    task_type: str | None = None
    # compare
    prompt: str | None = None
    task_id: str | None = None
    strategies: list[str] = Field(default_factory=lambda: list(ALL_STRATEGIES))
    # eval
    only: list[str] | None = None
    budget_usd: float | None = None


class ConfigChange(BaseModel):
    """Either a partial patch (merged into the current config) or full YAML text."""

    patch: dict[str, Any] | None = None
    yaml_text: str | None = None


class WhatIfRequest(ConfigChange):
    since: str = "30d"


class EventTypes(BaseModel):
    """Not an endpoint payload: exposes streamed event shapes in the OpenAPI schema
    so the frontend can generate types for them."""

    router: RouterEvent
    agent: AgentEvent
    decision: RouteDecision


# --- helpers ----------------------------------------------------------------------


def dump_config_yaml(config: RouterConfig) -> str:
    """Non-default values only, plus every tier's kind, so the file stays readable."""
    data = config.model_dump(mode="json", exclude_defaults=True)
    tiers = data.get("tiers", {})
    data["tiers"] = {
        t.value: {"kind": c.kind, **tiers.get(t.value, {})} for t, c in config.tiers.items()
    }
    return yaml.safe_dump(data, sort_keys=False)


def decision_cost(d: RouteDecision, pricing: Pricing) -> float:
    return sum((call_cost(a, pricing).usd or 0.0) for a in d.attempts)


def _ev(e: BaseModel) -> dict[str, Any]:
    return e.model_dump(mode="json")


def create_app(
    config_path: str | Path | None = None,
    *,
    fake: bool = False,
    token: str | None = None,
    router: HybridRouter | None = None,
    static_dir: Path | None = STATIC_DIR,
) -> FastAPI:
    # The file may not exist yet: the Tuning page creates it on first save.
    path = Path(config_path) if config_path else Path(resolve_config_path() or "hybrid_router.yaml")
    cfg = load_config(path if path.is_file() else None)
    if fake and router is None:
        # Keep demo data out of real telemetry.
        cfg = cfg.model_copy(
            update={
                "telemetry": cfg.telemetry.model_copy(
                    update={"db_path": str(Path(cfg.telemetry.db_path).with_suffix(".demo.sqlite"))}
                )
            }
        )
    if router is None:
        backends = demo_backends({t: c.model for t, c in cfg.tiers.items()}) if fake else None
        router = HybridRouter(cfg, backends=backends)
    db_path = router.config.telemetry.db_path
    store = WebStore(db_path)
    store.mark_interrupted()
    state = AppState(
        config_path=path,
        router=router,
        store=store,
        runs=RunManager(store),
        pricing=load_pricing(router.config.telemetry.pricing_path),
        token=token,
        fake=fake,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await state.runs.shutdown()
        await state.router.aclose()
        store.close()

    app = FastAPI(title="hybrid-router", version=__version__, lifespan=lifespan)
    app.state.hr = state

    def auth(request: Request) -> None:
        if not state.token:
            return
        header = request.headers.get("authorization", "")
        given = header.removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), state.token.encode()):
            raise HTTPException(401, "missing or wrong bearer token")

    protected = [Depends(auth)]

    # --- meta & health ---------------------------------------------------------

    @app.get("/api/meta")
    async def meta() -> dict[str, Any]:
        return {"version": __version__, "auth_required": bool(state.token), "fake": state.fake}

    @app.get("/api/health", dependencies=protected)
    async def health() -> dict[str, Any]:
        now = time.time()
        if state.health_cache and now - state.health_cache[0] < 10:
            return state.health_cache[1]

        async def one(tier: Tier) -> dict[str, Any]:
            b = state.router.backends[tier]
            try:
                h = await asyncio.wait_for(b.health(), 8)
                return {"tier": tier.value, "model": b.model, "ok": h.ok, "detail": h.detail}
            except TimeoutError:
                return {"tier": tier.value, "model": b.model, "ok": False, "detail": "timeout"}

        tiers = await asyncio.gather(*(one(t) for t in Tier))
        result = {
            "tiers": tiers,
            "config_hash": state.config.config_hash(),
            "fake": state.fake,
            "budget": budget(),
        }
        state.health_cache = (now, result)
        return result

    def budget() -> dict[str, Any]:
        return {
            "daily_cloud_budget_usd": state.config.routing.daily_cloud_budget_usd,
            "spent_today_usd": state.router.cloud_spent_today(),
            "cloud_allowed": state.router.cloud_allowed(),
        }

    @app.get("/api/budget", dependencies=protected)
    async def get_budget() -> dict[str, Any]:
        return budget()

    @app.get("/api/_schema/events", dependencies=protected, response_model=EventTypes)
    async def event_types() -> None:  # pragma: no cover - schema only
        raise HTTPException(404)

    # --- conversations -----------------------------------------------------------

    @app.get("/api/conversations", dependencies=protected)
    def list_conversations() -> list[dict[str, Any]]:
        return store.list_conversations()

    @app.post("/api/conversations", dependencies=protected)
    async def new_conversation(body: NewConversation) -> dict[str, Any]:
        return store.create_conversation(body.title, body.settings)

    @app.get("/api/conversations/{cid}", dependencies=protected)
    def get_conversation(cid: str) -> dict[str, Any]:
        conv = store.get_conversation(cid)
        if conv is None:
            raise HTTPException(404, "no such conversation")
        return conv

    @app.patch("/api/conversations/{cid}", dependencies=protected)
    async def update_conversation(cid: str, body: ConversationUpdate) -> dict[str, Any]:
        if store.get_conversation(cid) is None:
            raise HTTPException(404, "no such conversation")
        store.update_conversation(cid, title=body.title, settings=body.settings)
        return store.get_conversation(cid) or {}

    @app.delete("/api/conversations/{cid}", dependencies=protected)
    async def delete_conversation(cid: str) -> dict[str, bool]:
        return {"deleted": store.delete_conversation(cid)}

    @app.post("/api/conversations/{cid}/messages", dependencies=protected)
    async def chat(cid: str, body: ChatTurn) -> StreamingResponse:
        conv = store.get_conversation(cid)
        if conv is None:
            raise HTTPException(404, "no such conversation")
        cfg = state.config
        user = Message(role="user", content=body.content)
        history = store.history(cid) + [user]
        tools = (
            workspace_tools(
                cfg.agent.workspace,
                allow_code_exec=cfg.agent.allow_code_exec,
                exec_timeout_s=cfg.agent.exec_timeout_s,
            )
            if body.tools
            else []
        )
        agent = Agent(
            state.router,
            tools,
            max_steps=cfg.agent.max_steps if body.tools else 1,
            task_type=body.task_type or ("tool_use" if body.tools else "chat"),
            session_id=f"chat-{cid}",
            metadata={"source": "web"},
        )
        if conv["title"] == "New chat":
            store.update_conversation(cid, title=body.content.strip().splitlines()[0][:60])

        async def stream():
            decisions: list[dict[str, Any]] = []
            async for ev in agent.run(history, strategy=body.strategy, force_tier=body.force_tier):
                payload = _ev(ev)
                if isinstance(ev, StepEndEvent):
                    cost = decision_cost(ev.decision, state.pricing)
                    decisions.append({**payload["decision"], "cost_usd": cost})
                    payload["cost_usd"] = cost
                yield sse(payload)
                if isinstance(ev, AgentEndEvent):
                    # Attach each step's decision to the assistant message it produced.
                    per_msg: list[list[dict[str, Any]]] = [[]]
                    it = iter(decisions)
                    for m in ev.new_messages:
                        per_msg.append([next(it, {})] if m.role == "assistant" else [])
                    ids = store.append_messages(cid, [user, *ev.new_messages], per_msg)
                    yield sse({"type": "saved", "message_ids": ids})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- runs ---------------------------------------------------------------------

    @app.get("/api/runs", dependencies=protected)
    def list_runs(kind: str | None = None) -> list[dict[str, Any]]:
        return store.list_runs(kind)

    @app.get("/api/runs/{run_id}", dependencies=protected)
    async def get_run(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(404, "no such run")
        return run

    @app.get("/api/runs/{run_id}/events", dependencies=protected)
    async def run_events(run_id: str) -> StreamingResponse:
        if store.get_run(run_id) is None:
            raise HTTPException(404, "no such run")
        return StreamingResponse(
            state.runs.subscribe(run_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/runs/{run_id}/cancel", dependencies=protected)
    async def cancel_run(run_id: str) -> dict[str, bool]:
        return {"cancelled": state.runs.cancel(run_id)}

    @app.post("/api/runs", dependencies=protected)
    async def start_run(body: RunRequest) -> dict[str, str]:
        cfg = state.config
        if body.kind == "agent":
            if not body.task:
                raise HTTPException(422, "agent runs need a task")
            run_body = agent_run(body)
        elif body.kind == "compare":
            if not (body.prompt or body.task_id):
                raise HTTPException(422, "compare runs need a prompt or a task_id")
            bad = [s for s in body.strategies if s not in ALL_STRATEGIES]
            if bad or not body.strategies:
                raise HTTPException(422, f"unknown strategies: {bad}")
            run_body = compare_run(body)
        else:
            run_body = eval_run(body)
        params = body.model_dump(mode="json", exclude_none=True)
        return {"run_id": state.runs.start(body.kind, params, cfg.config_hash(), run_body)}

    def agent_run(body: RunRequest):
        async def go(emit) -> dict[str, Any]:
            cfg = state.config
            agent = Agent(
                state.router,
                workspace_tools(
                    cfg.agent.workspace,
                    allow_code_exec=cfg.agent.allow_code_exec,
                    exec_timeout_s=cfg.agent.exec_timeout_s,
                ),
                max_steps=cfg.agent.max_steps,
                task_type=body.task_type or "tool_use",
                metadata={"source": "web-agent"},
            )
            cost = 0.0
            tiers: list[str] = []
            end: AgentEndEvent | None = None
            async for ev in agent.run(
                body.task or "", strategy=body.strategy, force_tier=body.force_tier
            ):
                extra: dict[str, Any] = {}
                if isinstance(ev, StepEndEvent):
                    c = decision_cost(ev.decision, state.pricing)
                    cost += c
                    tiers.append(ev.decision.final_tier.value if ev.decision.final_tier else "-")
                    extra = {"cost_usd": c, "total_cost_usd": cost}
                if isinstance(ev, AgentEndEvent):
                    end = ev
                emit({**_ev(ev), **extra})
            return {
                "status": end.status if end else "error",
                "answer": end.answer if end else "",
                "steps": end.steps if end else 0,
                "tiers": tiers,
                "cost_usd": cost,
            }

        return go

    def compare_run(body: RunRequest):
        async def go(emit) -> dict[str, Any]:
            cfg = state.config
            spent = 0.0
            cap = body.budget_usd if body.budget_usd is not None else cfg.web.run_budget_usd

            def spend(usd: float) -> None:
                nonlocal spent
                spent += usd
                if spent > cap:
                    raise BudgetExceeded

            legs: dict[str, Any] = {}
            task = None
            if body.task_id:
                task = next((t for t in load_tasks() if t.id == body.task_id), None)
                if task is None:
                    raise ValueError(f"unknown task {body.task_id!r}")

            async def leg(strategy: str) -> None:
                emit({"type": "leg_start", "leg": strategy})
                try:
                    if task is not None:
                        r = await run_task(
                            state.router,
                            task,
                            strategy,  # type: ignore[arg-type]
                            "web",
                            state.pricing,
                            spend,
                        )
                        legs[strategy] = {
                            "passed": r.passed,
                            "failures": r.failures,
                            "tiers": r.tiers,
                            "cost_usd": r.cost_usd,
                            "latency_ms": r.latency_ms,
                            "steps": r.steps,
                            "escalated": r.escalated,
                            "error": r.error,
                        }
                    else:
                        force = (
                            Tier.LOCAL
                            if strategy == "all_local"
                            else cfg.telemetry.counterfactual_baseline
                            if strategy == "all_cloud"
                            else None
                        )
                        strat = (
                            strategy if strategy in ("hybrid", "pre_route", "cascade") else "hybrid"
                        )
                        text, decision = "", None
                        async for ev in state.router.astream(
                            body.prompt or "",
                            strategy=strat,
                            force_tier=force,  # type: ignore[arg-type]
                            task_type=body.task_type,
                            metadata={"source": "web-compare"},
                        ):
                            if ev.type == "text_delta":
                                emit({"type": "leg_delta", "leg": strategy, "text": ev.text})
                            elif ev.type == "attempt_end" and ev.discarded:
                                emit({"type": "leg_discard", "leg": strategy})
                            elif ev.type == "route_end":
                                decision = ev.decision
                                text = ev.response.text if ev.response else ""
                        assert decision is not None
                        c = decision_cost(decision, state.pricing)
                        spend(c)
                        legs[strategy] = {
                            "text": text,
                            "decision": _ev(decision),
                            "cost_usd": c,
                            "latency_ms": decision.total_latency_ms,
                            "tiers": [t.value for t in decision.tiers_attempted],
                        }
                except BudgetExceeded:
                    legs[strategy] = {"error": f"run budget ${cap} exceeded"}
                emit({"type": "leg_end", "leg": strategy, **legs[strategy]})

            # Local legs queue on the router's local semaphore; cloud legs overlap.
            await asyncio.gather(*(leg(s) for s in body.strategies))
            return {"legs": legs, "spent_usd": spent}

        return go

    def eval_run(body: RunRequest):
        async def go(emit) -> dict[str, Any]:
            cfg = state.config
            tasks = load_tasks()
            if body.only:
                tasks = [t for t in tasks if t.id in set(body.only)]
            report = await run_evals(
                cfg,
                strategies=body.strategies,  # type: ignore[arg-type]
                tasks=tasks,
                simulate=state.fake,
                budget_usd=body.budget_usd
                if body.budget_usd is not None
                else cfg.web.run_budget_usd,
                db_path=cfg.web.eval_db_path,
                on_result=lambda r: emit({"type": "eval_result", **r.__dict__}),
            )
            d = report.as_dict()
            emit(
                {
                    "type": "eval_summary",
                    "summaries": d["summaries"],
                    "budget_exhausted": report.budget_exhausted,
                }
            )
            return {
                "run_id": report.run_id,
                "summaries": d["summaries"],
                "budget_exhausted": report.budget_exhausted,
            }

        return go

    @app.get("/api/eval/tasks", dependencies=protected)
    async def eval_tasks() -> list[dict[str, Any]]:
        return [
            {
                "id": t.id,
                "difficulty": t.difficulty,
                "task_type": t.task_type,
                "min_tier": t.min_tier.value,
                "prompt": t.prompt,
                "tools": t.tools,
            }
            for t in load_tasks()
        ]

    # --- config, what-if, tune -------------------------------------------------

    def candidate_from(change: ConfigChange) -> RouterConfig:
        try:
            if change.yaml_text is not None:
                data = yaml.safe_load(change.yaml_text) or {}
                if not isinstance(data, dict):
                    raise ValueError("YAML top level must be a mapping")
                return load_config_from_dict(data)
            return apply_patch(state.config, change.patch or {})
        except (ValidationError, ValueError, yaml.YAMLError) as e:
            raise HTTPException(422, str(e)) from None

    def diff(new: RouterConfig) -> str:
        old_text = dump_config_yaml(state.config).splitlines(keepends=True)
        new_text = dump_config_yaml(new).splitlines(keepends=True)
        return "".join(difflib.unified_diff(old_text, new_text, "current", "proposed"))

    @app.get("/api/config", dependencies=protected)
    async def get_config() -> dict[str, Any]:
        return {
            "path": str(state.config_path.resolve()),
            "exists": state.config_path.is_file(),
            "hash": state.config.config_hash(),
            "config": state.config.model_dump(mode="json"),
            "yaml": dump_config_yaml(state.config),
        }

    @app.post("/api/config/validate", dependencies=protected)
    async def validate_config(change: ConfigChange) -> dict[str, Any]:
        new = candidate_from(change)
        return {
            "ok": True,
            "hash": new.config_hash(),
            "diff": diff(new),
            "config": new.model_dump(mode="json"),
        }

    @app.post("/api/config/whatif", dependencies=protected)
    def whatif(req: WhatIfRequest) -> dict[str, Any]:
        new = candidate_from(req)
        routes = load_routes(connect(state.config.telemetry.db_path), parse_since(req.since))
        return replay(routes, state.config, new)

    @app.put("/api/config", dependencies=protected)
    async def put_config(change: ConfigChange) -> dict[str, Any]:
        async with state.config_lock:
            new = candidate_from(change)
            if state.fake:
                # Demo mode: keep demo backends, never touch the real file.
                state.router.config = new
                state.router.scorer = build_scorer(new)
                return {"applied": True, "written": None, "hash": new.config_hash()}
            text = dump_config_yaml(new)
            backup = None
            if state.config_path.is_file():
                backup = state.config_path.with_name(
                    f"{state.config_path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
                )
                shutil.copy2(state.config_path, backup)
            state.config_path.write_text(text)
            state.router.update_config(new)
            state.health_cache = None
            return {
                "applied": True,
                "written": str(state.config_path),
                "backup": str(backup) if backup else None,
                "hash": new.config_hash(),
            }

    @app.get("/api/tune", dependencies=protected, response_model=TuneReport)
    def tune(since: str = "30d", min_samples: int = 50) -> TuneReport:
        routes = load_routes(connect(state.config.telemetry.db_path), parse_since(since))
        return suggest(routes, state.config, min_samples=min_samples)

    # --- stats ----------------------------------------------------------------

    @app.get("/api/stats", dependencies=protected)
    def stats(since: str = "7d", source: Literal["main", "eval"] = "main") -> dict[str, Any]:
        try:
            ts = parse_since(since)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        db = state.config.telemetry.db_path if source == "main" else state.config.web.eval_db_path
        if not Path(db).exists():
            TelemetryStore(db, config=state.config).close()  # creates an empty schema
        rep = build_report(connect(db), ts, state.pricing)
        rep["budget"] = budget()
        return rep

    # --- frontend -------------------------------------------------------------

    index = static_dir / "index.html" if static_dir else None
    if static_dir and index and index.is_file():
        if (static_dir / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404)
            candidate = (static_dir / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(static_dir.resolve()):
                return FileResponse(candidate)
            return FileResponse(index)
    else:

        @app.get("/", include_in_schema=False)
        async def not_built() -> HTMLResponse:
            return HTMLResponse(
                "<h1>hybrid-router</h1><p>The web UI isn't built. Run "
                "<code>cd frontend && pnpm install && pnpm build</code>, or use the API at "
                "<a href='/docs'>/docs</a>.</p>"
            )

    return app


def load_config_from_dict(data: dict[str, Any]) -> RouterConfig:
    from hybrid_router.config import DEFAULTS

    return RouterConfig.model_validate(deep_merge(DEFAULTS, data))
