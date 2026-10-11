"""A small tool-using agent that uses HybridRouter as its model provider.

Every step is routed separately (per step, not per task) under one session id, so
stickiness and tool-error feedback apply. Tools:

  read_file(path)              read a text file under the workspace root
  search_text(pattern, glob)   regex search across workspace files
  run_python(code)             run Python in a subprocess (off unless allowed)

``Agent.run`` yields events, so a UI can show the step timeline live.
"""

from __future__ import annotations

import asyncio
import fnmatch
import re
import sys
import tempfile
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from hybrid_router.decision import RouteDecision, RouteEndEvent, RouterEvent, RoutingError
from hybrid_router.router import HybridRouter
from hybrid_router.types import Message, ToolResultBlock, ToolSpec

SYSTEM = (
    "You are a careful assistant working in a code workspace. Use the tools to look things up "
    "instead of guessing. Keep tool calls focused. When you have the answer, reply with it "
    "directly and concisely."
)

MAX_READ_CHARS = 20_000
MAX_RESULT_CHARS = 8_000
MAX_SEARCH_BYTES = 1_000_000
SKIP_DIRS = {"__pycache__", "node_modules", "dist", "build", "venv", "site-packages"}


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return b"\0" in f.read(8192)
    except OSError:
        return True


# --- tools ----------------------------------------------------------------------

ToolFn = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class AgentTool:
    spec: ToolSpec
    fn: ToolFn


class ToolError(Exception):
    pass


def _confine(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise ToolError(f"path {rel!r} is outside the workspace")
    return target


def workspace_tools(
    root: str | Path, *, allow_code_exec: bool = False, exec_timeout_s: float = 10.0
) -> list[AgentTool]:
    root = Path(root).resolve()

    async def read_file(args: dict[str, Any]) -> str:
        path = _confine(root, str(args.get("path", "")))
        if not path.is_file():
            raise ToolError(f"no such file: {args.get('path')}")
        text = path.read_text(errors="replace")
        if len(text) > MAX_READ_CHARS:
            return text[:MAX_READ_CHARS] + f"\n... [truncated, {len(text)} chars total]"
        return text

    async def search_text(args: dict[str, Any]) -> str:
        try:
            rx = re.compile(str(args.get("pattern", "")))
        except re.error as e:
            raise ToolError(f"bad regex: {e}") from None
        glob = str(args.get("glob") or "*")
        hits: list[str] = []
        for path in sorted(root.rglob("*")):
            if len(hits) >= 50:
                break
            rel = path.relative_to(root).as_posix()
            parts = path.relative_to(root).parts
            if not path.is_file() or any(p.startswith(".") or p in SKIP_DIRS for p in parts):
                continue
            if path.stat().st_size > MAX_SEARCH_BYTES or _is_binary(path):
                continue
            if not fnmatch.fnmatch(path.name, glob) and not fnmatch.fnmatch(rel, glob):
                continue
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for i, line in enumerate(lines, 1):
                if rx.search(line):
                    hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(hits) >= 50:
                        break
        return "\n".join(hits) or "no matches"

    async def run_python(args: dict[str, Any]) -> str:
        if not allow_code_exec:
            raise ToolError("code execution is disabled (agent.allow_code_exec is false)")
        code = str(args.get("code", ""))
        with tempfile.TemporaryDirectory() as tmp:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-c",
                code,
                cwd=tmp,
                env={"PATH": "/usr/bin:/bin"},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), exec_timeout_s)
            except TimeoutError:
                proc.kill()
                await proc.wait()  # reap it, or it lingers as a zombie process
                raise ToolError(f"timed out after {exec_timeout_s}s") from None
        text = out.decode(errors="replace") + err.decode(errors="replace")
        if proc.returncode:
            raise ToolError(f"exit code {proc.returncode}\n{text[-2000:]}")
        return text or "(no output)"

    def spec(name: str, desc: str, props: dict[str, Any], required: list[str]) -> ToolSpec:
        return ToolSpec(
            name=name,
            description=desc,
            input_schema={"type": "object", "properties": props, "required": required},
        )

    tools = [
        AgentTool(
            spec(
                "read_file",
                "Read a text file in the workspace.",
                {"path": {"type": "string", "description": "Relative path"}},
                ["path"],
            ),
            read_file,
        ),
        AgentTool(
            spec(
                "search_text",
                "Regex search over workspace files; returns file:line hits.",
                {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string", "description": "e.g. *.py"},
                },
                ["pattern"],
            ),
            search_text,
        ),
    ]
    if allow_code_exec:
        tools.append(
            AgentTool(
                spec(
                    "run_python",
                    "Run a short Python snippet; returns stdout.",
                    {"code": {"type": "string"}},
                    ["code"],
                ),
                run_python,
            )
        )
    return tools


# --- events ---------------------------------------------------------------------


class StepStartEvent(BaseModel):
    type: Literal["step_start"] = "step_start"
    step: int
    task_type: str


class RouterEventWrapper(BaseModel):
    """A router event from inside a step (route start, text deltas, attempts...)."""

    type: Literal["router"] = "router"
    step: int
    event: RouterEvent


class ToolCallEvent(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    step: int
    tool_use_id: str
    name: str
    input: dict[str, Any]


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    step: int
    tool_use_id: str
    content: str
    is_error: bool


class StepEndEvent(BaseModel):
    type: Literal["step_end"] = "step_end"
    step: int
    decision: RouteDecision


class AgentEndEvent(BaseModel):
    type: Literal["agent_end"] = "agent_end"
    status: Literal["done", "max_steps", "error", "cancelled"]
    answer: str
    steps: int
    error: str | None = None
    # Messages added during the run (assistant turns and tool results).
    new_messages: list[Message] = []


AgentEvent = Annotated[
    StepStartEvent
    | RouterEventWrapper
    | ToolCallEvent
    | ToolResultEvent
    | StepEndEvent
    | AgentEndEvent,
    Field(discriminator="type"),
]


@dataclass
class Agent:
    router: HybridRouter
    tools: list[AgentTool]
    system: str = SYSTEM
    max_steps: int = 12
    task_type: str = "tool_use"
    session_id: str = field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:10]}")
    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(
        self,
        task: str | list[Message],
        *,
        strategy: Any = None,
        force_tier: Any = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run on a new task, or continue a conversation (list of messages)."""
        by_name = {t.spec.name: t for t in self.tools}
        messages: list[Message] = (
            [Message(role="user", content=task)] if isinstance(task, str) else list(task)
        )
        start = len(messages)
        answer = ""
        for step in range(self.max_steps):
            task_type = self.task_type
            yield StepStartEvent(step=step, task_type=task_type)
            end: RouteEndEvent | None = None
            try:
                async for ev in self.router.astream(
                    messages,
                    system=self.system,
                    tools=[t.spec for t in self.tools],
                    task_type=task_type,
                    session_id=self.session_id,
                    strategy=strategy,
                    force_tier=force_tier,
                    metadata=dict(self.metadata),
                ):
                    if isinstance(ev, RouteEndEvent):
                        end = ev
                    else:
                        yield RouterEventWrapper(step=step, event=ev)
            except RoutingError as e:  # pragma: no cover - astream doesn't raise it
                yield AgentEndEvent(
                    status="error",
                    answer=answer,
                    steps=step + 1,
                    error=str(e),
                    new_messages=messages[start:],
                )
                return
            assert end is not None
            yield StepEndEvent(step=step, decision=end.decision)
            if end.response is None:
                yield AgentEndEvent(
                    status="error",
                    answer=answer,
                    steps=step + 1,
                    error=end.decision.escalation_reason or "all attempts failed",
                )
                return
            response = end.response
            messages.append(response.as_message())
            answer = response.text or answer
            if not response.tool_calls:
                yield AgentEndEvent(
                    status="done", answer=answer, steps=step + 1, new_messages=messages[start:]
                )
                return
            results = []
            for call in response.tool_calls:
                yield ToolCallEvent(
                    step=step, tool_use_id=call.id, name=call.name, input=call.input
                )
                tool = by_name.get(call.name)
                try:
                    if tool is None:
                        raise ToolError(f"unknown tool {call.name!r}")
                    content, is_error = await tool.fn(call.input), False
                except ToolError as e:
                    content, is_error = f"error: {e}", True
                content = content[:MAX_RESULT_CHARS]
                yield ToolResultEvent(
                    step=step, tool_use_id=call.id, content=content, is_error=is_error
                )
                results.append(
                    ToolResultBlock(tool_use_id=call.id, content=content, is_error=is_error)
                )
            messages.append(Message(role="user", content=results))
        yield AgentEndEvent(
            status="max_steps", answer=answer, steps=self.max_steps, new_messages=messages[start:]
        )

    async def run_to_end(self, task: str, **kwargs: Any) -> AgentEndEvent:
        last: AgentEndEvent | None = None
        async for ev in self.run(task, **kwargs):
            if isinstance(ev, AgentEndEvent):
                last = ev
        assert last is not None
        return last
