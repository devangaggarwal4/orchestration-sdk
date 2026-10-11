"""Per-session routing state: step counter, stickiness window, and which tool
calls came from local steps (so a tool error can be blamed on the local model)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from hybrid_router.types import CompletionRequest, ToolResultBlock


@dataclass
class SessionState:
    session_id: str | None
    step_index: int = 0
    sticky_remaining: int = 0


@dataclass
class SessionStore:
    """In-memory and LRU-bounded. Swap for a persistent store for multi-process agents."""

    max_sessions: int = 1000
    max_tool_ids: int = 20000
    _sessions: OrderedDict[str, SessionState] = field(default_factory=OrderedDict)
    # tool_use_id -> call_id of the local attempt that produced it
    _local_tool_calls: OrderedDict[str, str] = field(default_factory=OrderedDict)

    def get(self, session_id: str | None) -> SessionState:
        if session_id is None:
            return SessionState(session_id=None)
        state = self._sessions.get(session_id)
        if state is None:
            state = SessionState(session_id=session_id)
            self._sessions[session_id] = state
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(session_id)
        return state

    def remember_local_tool_calls(self, call_id: str, tool_use_ids: list[str]) -> None:
        for tid in tool_use_ids:
            self._local_tool_calls[tid] = call_id
        while len(self._local_tool_calls) > self.max_tool_ids:
            self._local_tool_calls.popitem(last=False)

    def failed_local_calls(self, req: CompletionRequest) -> list[str]:
        """Call ids of local attempts whose tool calls errored in the newest user turn."""
        if not req.messages or req.messages[-1].role != "user":
            return []
        failed: list[str] = []
        for block in req.messages[-1].blocks():
            if isinstance(block, ToolResultBlock) and block.is_error:
                call_id = self._local_tool_calls.get(block.tool_use_id)
                if call_id and call_id not in failed:
                    failed.append(call_id)
        return failed

    def clear(self) -> None:
        self._sessions.clear()
        self._local_tool_calls.clear()
