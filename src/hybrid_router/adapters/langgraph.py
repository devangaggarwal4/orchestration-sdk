"""LangChain / LangGraph chat model backed by HybridRouter.

    from hybrid_router.adapters.langgraph import HybridRouterChatModel
    llm = HybridRouterChatModel(router=router, task_type="tool_use").bind_tools(tools)
    # use llm in a LangGraph node like any chat model

The RouteDecision of each call is in ``AIMessage.response_metadata["route"]``.
Requires ``langchain-core`` (``uv sync --extra langgraph``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from hybrid_router.router import HybridRouter
from hybrid_router.types import (
    ContentBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)


def to_router_messages(messages: Sequence[BaseMessage]) -> tuple[str | None, list[Message]]:
    system_parts: list[str] = []
    out: list[Message] = []
    for m in messages:
        text = m.content if isinstance(m.content, str) else json.dumps(m.content)
        if isinstance(m, SystemMessage):
            system_parts.append(text)
        elif isinstance(m, HumanMessage):
            out.append(Message(role="user", content=text))
        elif isinstance(m, AIMessage):
            blocks: list[ContentBlock] = [TextBlock(text=text)] if text else []
            blocks += [
                ToolUseBlock(id=c["id"] or "", name=c["name"], input=c["args"])
                for c in m.tool_calls
            ]
            out.append(Message(role="assistant", content=blocks))
        elif isinstance(m, ToolMessage):
            block = ToolResultBlock(
                tool_use_id=m.tool_call_id, content=text, is_error=m.status == "error"
            )
            # Consecutive tool results belong in one user turn.
            if (
                out
                and out[-1].role == "user"
                and all(isinstance(b, ToolResultBlock) for b in out[-1].blocks())
            ):
                out[-1] = Message(role="user", content=[*out[-1].blocks(), block])
            else:
                out.append(Message(role="user", content=[block]))
    return ("\n\n".join(system_parts) or None), out


class HybridRouterChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    router: HybridRouter
    task_type: str | None = None
    session_id: str | None = None
    strategy: str | None = None
    tools: list[ToolSpec] = []

    @property
    def _llm_type(self) -> str:
        return "hybrid-router"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> HybridRouterChatModel:  # type: ignore[override]
        specs = []
        for t in tools:
            fn = convert_to_openai_tool(t)["function"]
            specs.append(
                ToolSpec(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    input_schema=fn.get("parameters", {"type": "object", "properties": {}}),
                )
            )
        return self.model_copy(update={"tools": specs})

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system, msgs = to_router_messages(messages)
        result = await self.router.acomplete(
            msgs,
            system=system,
            tools=self.tools,
            task_type=kwargs.get("task_type", self.task_type),
            session_id=kwargs.get("session_id", self.session_id),
            strategy=kwargs.get("strategy", self.strategy),
        )
        resp = result.response
        ai = AIMessage(
            content=resp.text,
            tool_calls=[{"id": c.id, "name": c.name, "args": c.input} for c in resp.tool_calls],
            response_metadata={
                "model": resp.model,
                "tier": resp.tier.value,
                "route": result.decision.model_dump(mode="json"),
            },
            usage_metadata={
                "input_tokens": resp.usage.total_input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.total_input_tokens + resp.usage.output_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self.router._run_sync(self._agenerate(messages, stop, None, **kwargs))
