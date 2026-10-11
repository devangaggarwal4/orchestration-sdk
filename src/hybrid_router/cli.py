"""``hybrid-router`` command line. Phase 1 ships ``smoke`` and ``config``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.table import Table

from hybrid_router.backends import Backend, BackendError, create_backend
from hybrid_router.config import load_config
from hybrid_router.types import (
    CompletionRequest,
    Message,
    TextDeltaEvent,
    Tier,
    ToolSpec,
)

app = typer.Typer(no_args_is_help=True, help="Route LLM calls between local models and Claude.")
console = Console()

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="Path to hybrid_router.yaml")
]

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
)


@app.command("config")
def show_config(config: ConfigOpt = None) -> None:
    """Print the resolved configuration (defaults + file + env overrides)."""
    cfg = load_config(config)
    console.print(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))


@app.command()
def smoke(
    config: ConfigOpt = None,
    tier: Annotated[
        list[Tier] | None,
        typer.Option("--tier", "-t", help="Tier(s) to test; default all. Repeatable."),
    ] = None,
    skip_cloud: Annotated[bool, typer.Option(help="Only test the local tier.")] = False,
) -> None:
    """Check each tier: health, a plain streamed reply, and one tool call.

    Cloud tiers make two tiny real API calls each (a few hundred tokens).
    """
    cfg = load_config(config)
    tiers = tier or list(cfg.tiers)
    if skip_cloud:
        tiers = [t for t in tiers if not t.is_cloud]
    ok = asyncio.run(_smoke(cfg.tiers, tiers))
    raise typer.Exit(0 if ok else 1)


async def _smoke(tier_configs: dict, tiers: list[Tier]) -> bool:
    table = Table(title="hybrid-router smoke test")
    for col in (
        "tier",
        "model",
        "health",
        "reply",
        "ttft / total",
        "tokens in/out ($)",
        "tool call",
        "logprobs",
    ):
        table.add_column(col)
    all_ok = True
    notes: list[str] = []
    for t in tiers:
        backend = create_backend(tier_configs[t], tier=t)
        async with backend:
            row, problems = await _smoke_one(backend)
        all_ok &= not problems
        notes.extend(f"[bold]{t.value}[/]: {p}" for p in problems)
        table.add_row(t.value, backend.model, *row)
    console.print(table)
    for note in notes:
        console.print(f"[red]✗[/] {note}")
    return all_ok


async def _smoke_one(backend: Backend) -> tuple[list[str], list[str]]:
    """Returns the table cells and a list of problems (empty when everything passed)."""
    health = await backend.health()
    if not health.ok:
        return ["[red]✗[/]", "-", "-", "-", "-", "-"], [health.detail]
    cells = [f"[green]✓[/] {health.detail}"]
    problems: list[str] = []

    plain = CompletionRequest(
        messages=[Message(role="user", content="Reply with exactly one word: pong")],
        max_tokens=2000,
    )
    try:
        streamed = ""
        response = None
        async for ev in backend.stream(plain, logprobs=True):
            if isinstance(ev, TextDeltaEvent):
                streamed += ev.text
            elif ev.type == "message_end":
                response = ev.response
        assert response is not None
    except BackendError as e:
        return [*cells, "[red]✗[/]", "-", "-", "-", "-"], [f"plain reply: {e}"]

    reply = response.text.strip()
    if "pong" not in reply.lower():
        problems.append(f"plain reply did not contain 'pong': {reply[:80]!r}")
    if streamed != response.text:
        problems.append("streamed text differs from the final message text")
    cells.append(("[green]✓[/] " if not problems else "[yellow]?[/] ") + repr(reply[:30]))
    ttft = f"{response.ttft_ms:.0f}" if response.ttft_ms is not None else "-"
    cells.append(f"{ttft} / {response.latency_ms:.0f} ms")
    tokens = f"{response.usage.total_input_tokens}/{response.usage.output_tokens}"
    if response.reported_cost_usd is not None:
        tokens += f" (${response.reported_cost_usd:.5f})"
    cells.append(tokens)
    if response.mean_logprob is not None:
        logprobs = f"mean {response.mean_logprob:.2f}"
    elif backend.capabilities.logprobs:
        logprobs = "[yellow]none returned[/]"
        problems.append("logprobs requested but none returned (Ollama too old?)")
    else:
        logprobs = "n/a"

    tool_req = CompletionRequest(
        messages=[Message(role="user", content="What's the weather in Tokyo right now?")],
        tools=[WEATHER_TOOL],
        max_tokens=2000,
    )
    try:
        tool_resp = await backend.generate(tool_req)
        calls = tool_resp.tool_calls
        if calls and calls[0].name == "get_weather" and "city" in calls[0].input:
            cells.append(f"[green]✓[/] get_weather({calls[0].input['city']!r})")
        else:
            cells.append("[red]✗[/]")
            problems.append(f"no valid get_weather call (stop_reason={tool_resp.stop_reason})")
    except BackendError as e:
        cells.append("[red]✗[/]")
        problems.append(f"tool call: {e}")
    cells.append(logprobs)
    return cells, problems


@app.command("eval")
def eval_cmd(
    config: ConfigOpt = None,
    strategies: Annotated[
        str, typer.Option(help="Comma-separated: hybrid,pre_route,cascade,all_local,all_cloud")
    ] = "hybrid,pre_route,cascade,all_local,all_cloud",
    simulate: Annotated[
        bool, typer.Option(help="Use simulated tiers (no models, no cost).")
    ] = False,
    budget_usd: Annotated[float, typer.Option(help="Stop cloud spending past this.")] = 2.0,
    tasks_file: Annotated[Path | None, typer.Option("--tasks", help="Custom tasks.yaml")] = None,
    only: Annotated[str | None, typer.Option(help="Comma-separated task ids")] = None,
    repeat: Annotated[int, typer.Option(min=1)] = 1,
    db: Annotated[Path, typer.Option(help="Separate telemetry DB for eval runs")] = Path(
        "evals.sqlite"
    ),
    json_out: Annotated[Path | None, typer.Option("--json", help="Write full report")] = None,
) -> None:
    """Run the labeled tasks under several strategies and compare them."""
    import json as _json

    from hybrid_router.evals import ALL_STRATEGIES, load_tasks, run_evals

    cfg = load_config(config)
    chosen = [s.strip() for s in strategies.split(",") if s.strip()]
    bad = [s for s in chosen if s not in ALL_STRATEGIES]
    if bad:
        raise typer.BadParameter(f"unknown strategies: {bad}")
    tasks = load_tasks(tasks_file)
    if only:
        wanted = {t.strip() for t in only.split(",")}
        tasks = [t for t in tasks if t.id in wanted]
    total = len(tasks) * len(chosen) * repeat
    done = 0

    def progress(r) -> None:
        nonlocal done
        done += 1
        mark = "[green]✓[/]" if r.passed else ("[yellow]-[/]" if r.skipped else "[red]✗[/]")
        console.print(
            f"[dim]{done:>3}/{total}[/] {mark} {r.strategy:<9} {r.task_id:<24} "
            f"{'>'.join(r.tiers) or '-':<22} ${r.cost_usd:.4f}",
            highlight=False,
        )

    report = asyncio.run(
        run_evals(
            cfg,
            strategies=chosen,  # type: ignore[arg-type]
            tasks=tasks,
            simulate=simulate,
            budget_usd=budget_usd,
            db_path=db,
            repeat=repeat,
            on_result=progress,
        )
    )
    table = Table(title=f"eval {report.run_id}" + (" (simulated)" if simulate else ""))
    for col in (
        "strategy",
        "pass",
        "easy/med/hard",
        "local",
        "routing acc",
        "under/over",
        "cost",
        "all-cloud",
        "savings",
        "p50 / p95",
        "cascade waste",
    ):
        table.add_column(col)
    for s in report.summaries:
        diff = "/".join(f"{s.pass_by_difficulty.get(d, 0):.0%}" for d in ("easy", "medium", "hard"))
        table.add_row(
            s.strategy,
            f"{s.pass_rate:.0%}",
            diff,
            f"{s.local_share:.0%}",
            f"{s.routing_accuracy:.0%}" if s.routing_accuracy is not None else "-",
            f"{s.under_escalations}/{s.over_escalations}",
            f"${s.cost_usd:.4f}",
            f"${s.counterfactual_usd:.4f}",
            f"${s.savings_usd:.4f}",
            f"{s.latency_p50_ms / 1000:.1f}s / {s.latency_p95_ms / 1000:.1f}s",
            f"{s.waste_latency_ms / 1000:.1f}s, {s.waste_tokens} tok",
        )
    console.print(table)
    if report.summaries:
        console.print(
            f"[dim]routing accuracy vs {report.summaries[0].label_source}; under = served "
            "locally but failed, over = sent to cloud though local would have passed[/]"
        )
    if report.budget_exhausted:
        console.print(f"[yellow]budget of ${budget_usd} exhausted; remaining tasks skipped[/]")
    if json_out:
        json_out.write_text(_json.dumps(report.as_dict(), indent=2))
        console.print(f"wrote {json_out}")


if __name__ == "__main__":
    app()
