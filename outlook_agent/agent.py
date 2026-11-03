"""The agent loop: the model plans, calls tools, reads results, and repeats."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Callable

from .llm import OllamaClient
from .tools import Toolbox

SYSTEM_PROMPT = """You are an email assistant with access to the user's Outlook mailbox and calendar via tools.
User: {name} <{email}>. Current time: {now}.

How to work:
- Use tools to get real data. Never invent emails, senders, dates or ids.
- Emails are referenced by short refs like E1, E2 that tools return. Drafts use D1, D2.
- To summarize or triage the inbox, call summarize_inbox and show its digest.
- To act on "the email from X", first list_emails or search_emails to find its ref.
- Before replying to an email, read_email it so the reply is grounded.
- Replies: call draft_reply (saves a draft). Only send_draft / send_email when the user explicitly says to send.
- Write replies in a friendly, concise, professional tone and sign off with the user's first name.
- Destructive or outgoing actions ask the user for confirmation automatically; if they decline, accept it.
- Keep final answers short and skimmable (markdown bullets are fine). Mention refs so the user can follow up.
"""


class Agent:
    def __init__(self, llm: OllamaClient, toolbox: Toolbox, max_steps: int = 8,
                 on_tool: Callable[[str, dict], None] | None = None, history_limit: int = 30):
        self.llm = llm
        self.toolbox = toolbox
        self.max_steps = max_steps
        self.on_tool = on_tool
        self.history_limit = history_limit
        try:
            me = toolbox.mailbox.me()
        except Exception:
            me = {"name": "the user", "email": "unknown"}
        self.system = SYSTEM_PROMPT.format(name=me.get("name"), email=me.get("email"),
                                           now=datetime.now().astimezone().strftime("%A %Y-%m-%d %H:%M %Z"))
        self.history: list[dict] = []

    def reset(self) -> None:
        self.history.clear()

    def _trimmed(self) -> list[dict]:
        """Keep the context small for an 8B model: last N messages, starting at a user turn."""
        h = self.history[-self.history_limit:]
        while h and h[0]["role"] != "user":
            h = h[1:]
        return [{"role": "system", "content": self.system}, *h]

    def run(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        schemas = self.toolbox.schemas()
        seen: set[str] = set()
        for _ in range(self.max_steps):
            reply = self.llm.chat(self._trimmed(), tools=schemas)
            if not reply.tool_calls:
                answer = reply.content or "(no response)"
                self.history.append({"role": "assistant", "content": answer})
                return answer
            self.history.append({"role": "assistant", "content": reply.content, "tool_calls": [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in reply.tool_calls]})
            for call in reply.tool_calls:
                key = call.name + json.dumps(call.arguments, sort_keys=True)
                if key in seen and call.name not in ("summarize_inbox",):
                    result = json.dumps({"note": "You already made this exact call; use its earlier result."})
                else:
                    seen.add(key)
                    if self.on_tool:
                        self.on_tool(call.name, call.arguments)
                    result = self.toolbox.call(call.name, call.arguments)
                self.history.append({"role": "tool", "tool_name": call.name, "content": result})
        # Out of steps: ask for a final answer without tools.
        reply = self.llm.chat(self._trimmed() + [{"role": "user", "content":
                              "Stop calling tools and give your best final answer now."}])
        answer = reply.content or "Sorry, I couldn't finish that."
        self.history.append({"role": "assistant", "content": answer})
        return answer
