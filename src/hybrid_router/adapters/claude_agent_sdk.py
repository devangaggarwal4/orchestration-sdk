"""Claude Agent SDK integration: let Claude delegate simple subtasks to the router.

The Agent SDK talks to Claude directly (supported path); this adds an in-process
MCP tool ``delegate_local`` so Claude can hand off classify/extract/summarize-style
subtasks, which the router keeps local when it can.

    from claude_agent_sdk import ClaudeAgentOptions, query
    from hybrid_router.adapters.claude_agent_sdk import delegate_local_server

    options = ClaudeAgentOptions(
        mcp_servers={"hybrid": delegate_local_server(router)},
        allowed_tools=["mcp__hybrid__delegate_local"],
    )

Requires ``claude-agent-sdk`` (``uv sync --extra agent-sdk``).
"""

from __future__ import annotations

from typing import Any

from hybrid_router.decision import RoutingError
from hybrid_router.router import HybridRouter

DESCRIPTION = (
    "Delegate a self-contained, simple subtask (classify, extract, summarize, rewrite, short "
    "Q&A) to a cheaper model. Give complete instructions and all needed text in `task`; the "
    "delegate cannot see this conversation. Optional `task_type` is one of classify, extract, "
    "summarize, rewrite, chat."
)
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "Full instructions plus input text"},
        "task_type": {"type": "string"},
    },
    "required": ["task"],
}


async def delegate_local(router: HybridRouter, args: dict[str, Any]) -> dict[str, Any]:
    """Tool handler, returning MCP tool-result content. Usable without the SDK."""
    try:
        result = await router.acomplete(
            str(args["task"]), task_type=args.get("task_type") or "chat"
        )
    except RoutingError as e:
        return {"content": [{"type": "text", "text": f"delegate failed: {e}"}], "is_error": True}
    d = result.decision
    note = f"\n\n[served by {d.final_tier.value if d.final_tier else '?'}, score {d.score:.2f}]"
    return {"content": [{"type": "text", "text": result.text + note}]}


def delegate_local_server(router: HybridRouter, name: str = "hybrid") -> Any:
    from claude_agent_sdk import (  # pyright: ignore[reportMissingImports]
        create_sdk_mcp_server,
        tool,
    )

    @tool("delegate_local", DESCRIPTION, INPUT_SCHEMA)
    async def _delegate(args: dict[str, Any]) -> dict[str, Any]:
        return await delegate_local(router, args)

    return create_sdk_mcp_server(name=name, tools=[_delegate])
