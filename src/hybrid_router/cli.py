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
        "tokens in/out",
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
    cells.append(f"{response.usage.total_input_tokens}/{response.usage.output_tokens}")
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


if __name__ == "__main__":
    app()
