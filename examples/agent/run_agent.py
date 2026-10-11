"""Example: a repo assistant agent using HybridRouter as its model provider.

    uv run python examples/agent/run_agent.py "Where is the escalation logic?"
    uv run python examples/agent/run_agent.py --simulate "..."

Each step prints the routing decision (tier, score, zone) so you can watch the
local/cloud split as the agent works.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from hybrid_router import HybridRouter, Tier, load_config
from hybrid_router.agent import (
    Agent,
    AgentEndEvent,
    RouterEventWrapper,
    StepEndEvent,
    ToolCallEvent,
    ToolResultEvent,
    workspace_tools,
)
from hybrid_router.testing import Reply, ScriptedBackend

REPO = Path(__file__).resolve().parents[2]


def simulated_backends() -> dict[Tier, ScriptedBackend]:
    """Local model searches, then answers; good enough to see the event flow."""
    local = ScriptedBackend(
        [
            Reply(tool_calls=[("search_text", {"pattern": "def _next_tier", "glob": "*.py"})]),
            Reply(tool_calls=[("read_file", {"path": "src/hybrid_router/router.py"})]),
            Reply(
                "Escalation lives in HybridRouter._next_tier (src/hybrid_router/router.py). "
                "A failed local attempt goes to Opus when the failure is hard (and "
                "escalate_hard_to_opus is on) or the score is above opus_threshold; "
                "otherwise it goes to Sonnet, and a failed Sonnet attempt goes to Opus."
            ),
        ],
        tier=Tier.LOCAL,
    )
    cloud = [Reply("(cloud answer)")]
    return {
        Tier.LOCAL: local,
        Tier.SONNET: ScriptedBackend(cloud, tier=Tier.SONNET),
        Tier.OPUS: ScriptedBackend(cloud, tier=Tier.OPUS),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--simulate", action="store_true", help="scripted models, no network")
    ap.add_argument("--workspace", default=str(REPO))
    args = ap.parse_args()

    cfg = load_config()
    router = HybridRouter(cfg, backends=simulated_backends() if args.simulate else None)
    agent = Agent(
        router,
        workspace_tools(args.workspace, allow_code_exec=cfg.agent.allow_code_exec),
        max_steps=cfg.agent.max_steps,
    )
    async for ev in agent.run(args.task):
        if isinstance(ev, StepEndEvent):
            d = ev.decision
            print(
                f"[step {ev.step}] {d.final_tier} · score {d.score:.2f} ({d.zone}) · "
                f"tiers {[t.value for t in d.tiers_attempted]}"
            )
        elif isinstance(ev, ToolCallEvent):
            print(f"    → {ev.name}({ev.input})")
        elif isinstance(ev, ToolResultEvent):
            first = ev.content.splitlines()[0] if ev.content else ""
            print(f"    ← {'ERROR ' if ev.is_error else ''}{first[:100]}")
        elif isinstance(ev, RouterEventWrapper):
            pass  # text deltas etc.; a UI would stream these
        elif isinstance(ev, AgentEndEvent):
            print(f"\n[{ev.status} after {ev.steps} steps]\n{ev.answer}")
    await router.aclose()


if __name__ == "__main__":
    asyncio.run(main())
