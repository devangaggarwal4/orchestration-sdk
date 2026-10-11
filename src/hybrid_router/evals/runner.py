"""Run labeled tasks under several strategies and compare routing accuracy, cost,
latency and quality.

Strategies: hybrid, pre_route, cascade (router strategies), all_local and all_cloud
(every step forced to the local tier / the counterfactual baseline tier).
"""

from __future__ import annotations

import statistics
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from hybrid_router.config import RouterConfig
from hybrid_router.cost import call_cost
from hybrid_router.decision import RoutingError
from hybrid_router.evals.checks import Answer, run_checks
from hybrid_router.evals.simulate import simulated_backend
from hybrid_router.evals.tasks import EvalTask, load_tasks
from hybrid_router.evals.tools import TOOLS, run_tool
from hybrid_router.pricing import Pricing, load_pricing
from hybrid_router.router import HybridRouter
from hybrid_router.telemetry import TelemetryStore, connect
from hybrid_router.types import TIER_ORDER, Message, Tier, ToolResultBlock

EvalStrategy = Literal["hybrid", "pre_route", "cascade", "all_local", "all_cloud"]
ALL_STRATEGIES: list[EvalStrategy] = ["hybrid", "pre_route", "cascade", "all_local", "all_cloud"]
MAX_AGENT_STEPS = 6


@dataclass
class TaskResult:
    strategy: str
    task_id: str
    difficulty: str
    min_tier: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    tiers: list[str] = field(default_factory=list)  # tier that served each step
    steps: int = 0
    escalated: bool = False
    cost_usd: float = 0.0
    latency_ms: float = 0.0  # summed model latency across steps
    error: str | None = None
    skipped: bool = False

    @property
    def served_locally(self) -> bool:
        return bool(self.tiers) and all(t == Tier.LOCAL.value for t in self.tiers)


@dataclass
class StrategySummary:
    strategy: str
    tasks: int
    pass_rate: float
    pass_by_difficulty: dict[str, float]
    local_share: float
    routing_accuracy: float | None
    label_source: str
    under_escalations: int
    over_escalations: int
    cost_usd: float
    counterfactual_usd: float
    savings_usd: float
    latency_p50_ms: float
    latency_p95_ms: float
    waste_latency_ms: float
    waste_tokens: int


@dataclass
class EvalReport:
    run_id: str
    simulated: bool
    results: list[TaskResult]
    summaries: list[StrategySummary]
    budget_exhausted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "simulated": self.simulated,
            "budget_exhausted": self.budget_exhausted,
            "summaries": [asdict(s) for s in self.summaries],
            "results": [asdict(r) for r in self.results],
        }


class BudgetExceeded(Exception):
    pass


def _force(strategy: EvalStrategy, config: RouterConfig) -> Tier | None:
    if strategy == "all_local":
        return Tier.LOCAL
    if strategy == "all_cloud":
        return config.telemetry.counterfactual_baseline
    return None


async def run_task(
    router: HybridRouter,
    task: EvalTask,
    strategy: EvalStrategy,
    run_id: str,
    pricing: Pricing,
    spend: Callable[[float], None],
) -> TaskResult:
    result = TaskResult(
        strategy=strategy,
        task_id=task.id,
        difficulty=task.difficulty,
        min_tier=task.min_tier.value,
        passed=False,
    )
    force = _force(strategy, router.config)
    router_strategy = strategy if strategy in ("hybrid", "pre_route", "cascade") else "hybrid"
    tools = [TOOLS[name][0] for name in task.tools]
    messages: list[Message] = [Message(role="user", content=task.prompt)]
    tool_calls: list[tuple[str, dict[str, Any]]] = []
    session = f"eval-{run_id}-{strategy}-{task.id}-{uuid.uuid4().hex[:6]}"
    final_text = ""

    try:
        for _ in range(MAX_AGENT_STEPS):
            routed = await router.acomplete(
                messages,
                system=task.system,
                tools=tools,
                response_schema=task.response_schema,
                task_type=task.task_type,
                session_id=session,
                strategy=router_strategy,
                force_tier=force,
                metadata={"source": "eval", "run_id": f"{run_id}:{strategy}", "eval_task": task.id},
            )
            d = routed.decision
            result.steps += 1
            result.escalated |= d.escalated
            if d.final_tier is not None:
                result.tiers.append(d.final_tier.value)
            # Model-reported latency (incl. local queueing) so simulated runs are meaningful.
            result.latency_ms += sum(a.latency_ms + a.queue_ms for a in d.attempts)
            step_cost = sum((call_cost(a, pricing).usd or 0.0) for a in d.attempts)
            result.cost_usd += step_cost
            spend(step_cost)
            messages.append(routed.response.as_message())
            final_text = routed.text
            if not routed.tool_calls:
                break
            results = []
            for call in routed.tool_calls:
                tool_calls.append((call.name, call.input))
                out, is_error = run_tool(call.name, call.input)
                results.append(ToolResultBlock(tool_use_id=call.id, content=out, is_error=is_error))
            messages.append(Message(role="user", content=results))
    except RoutingError as e:
        result.error = str(e)

    if result.error is None:
        result.passed, result.failures = run_checks(task.checks, Answer(final_text, tool_calls))
    else:
        result.failures = [f"error: {result.error}"]
    return result


async def run_evals(
    config: RouterConfig,
    *,
    strategies: list[EvalStrategy] | None = None,
    tasks: list[EvalTask] | None = None,
    simulate: bool = False,
    budget_usd: float | None = 2.0,
    db_path: str | Path = "evals.sqlite",
    repeat: int = 1,
    on_result: Callable[[TaskResult], None] | None = None,
) -> EvalReport:
    strategies = strategies or list(ALL_STRATEGIES)
    tasks = tasks if tasks is not None else load_tasks()
    pricing = load_pricing(config.telemetry.pricing_path)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    store = TelemetryStore(db_path, config=config, pricing=pricing, source="eval")
    backends = None
    if simulate:
        backends = {t: simulated_backend(t, tasks, config.tiers[t].model) for t in Tier}
    router = HybridRouter(config, backends=backends, telemetry=store)

    spent = 0.0
    exhausted = False

    def spend(usd: float) -> None:
        nonlocal spent
        spent += usd
        if budget_usd is not None and spent > budget_usd:
            raise BudgetExceeded

    results: list[TaskResult] = []
    try:
        for strategy in strategies:
            for _ in range(repeat):
                for task in tasks:
                    if exhausted:
                        r = TaskResult(
                            strategy,
                            task.id,
                            task.difficulty,
                            task.min_tier.value,
                            passed=False,
                            skipped=True,
                            failures=["budget exhausted"],
                        )
                    else:
                        try:
                            r = await run_task(router, task, strategy, run_id, pricing, spend)
                        except BudgetExceeded:
                            exhausted = True
                            r = TaskResult(
                                strategy,
                                task.id,
                                task.difficulty,
                                task.min_tier.value,
                                passed=False,
                                skipped=True,
                                failures=["budget exhausted"],
                            )
                    results.append(r)
                    if on_result:
                        on_result(r)
    finally:
        await router.aclose()  # flushes and closes the store

    summaries = summarize(results, run_id, db_path, tasks)
    return EvalReport(run_id, simulate, results, summaries, budget_exhausted=exhausted)


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def summarize(
    results: list[TaskResult], run_id: str, db_path: str | Path, tasks: list[EvalTask]
) -> list[StrategySummary]:
    # Empirical label: did the local model alone pass the task (from all_local runs)?
    local_runs = [r for r in results if r.strategy == "all_local" and not r.skipped]
    empirical: dict[str, bool] = {}
    for r in local_runs:
        empirical[r.task_id] = empirical.get(r.task_id, True) and r.passed
    hand = {t.id: t.min_tier is Tier.LOCAL for t in tasks}
    label = empirical if len(empirical) == len(tasks) else hand
    label_source = "all_local run" if label is empirical else "hand labels"

    conn = connect(db_path)
    summaries = []
    for strategy in dict.fromkeys(r.strategy for r in results):
        rs = [r for r in results if r.strategy == strategy and not r.skipped]
        if not rs:
            continue
        by_diff = {}
        for diff in ("easy", "medium", "hard"):
            sub = [r for r in rs if r.difficulty == diff]
            if sub:
                by_diff[diff] = sum(r.passed for r in sub) / len(sub)
        correct = under = over = 0
        for r in rs:
            should_local = label.get(r.task_id, False)
            if r.served_locally == should_local:
                correct += 1
            if r.served_locally and not r.passed:
                under += 1
            if not r.served_locally and should_local:
                over += 1
        row = conn.execute(
            """SELECT COALESCE(SUM(actual_cost_usd), 0), COALESCE(SUM(counterfactual_cost_usd), 0)
               FROM routes WHERE run_id = ?""",
            (f"{run_id}:{strategy}",),
        ).fetchone()
        waste = conn.execute(
            """SELECT COALESCE(SUM(latency_ms), 0),
                      COALESCE(SUM(input_tokens + output_tokens + cache_read_tokens), 0)
               FROM calls c JOIN routes r USING (route_id)
               WHERE r.run_id = ? AND c.wasted = 1""",
            (f"{run_id}:{strategy}",),
        ).fetchone()
        latencies = [r.latency_ms for r in rs]
        summaries.append(
            StrategySummary(
                strategy=strategy,
                tasks=len(rs),
                pass_rate=sum(r.passed for r in rs) / len(rs),
                pass_by_difficulty=by_diff,
                local_share=sum(r.served_locally for r in rs) / len(rs),
                routing_accuracy=correct / len(rs),
                label_source=label_source,
                under_escalations=under,
                over_escalations=over,
                cost_usd=row[0],
                counterfactual_usd=row[1],
                savings_usd=row[1] - row[0],
                latency_p50_ms=_pct(latencies, 50),
                latency_p95_ms=_pct(latencies, 95),
                waste_latency_ms=waste[0],
                waste_tokens=int(waste[1]),
            )
        )
    conn.close()
    return summaries


def tier_rank(t: Tier) -> int:
    return TIER_ORDER.index(t)
