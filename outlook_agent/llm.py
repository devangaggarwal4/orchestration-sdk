"""Minimal Ollama chat client with tool-calling support."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def strip_think(text: str) -> str:
    """qwen3 may emit <think>…</think> even with thinking off; never show it to the user."""
    text = THINK_RE.sub("", text or "")
    if "</think>" in text:  # unterminated opening tag got eaten upstream
        text = text.split("</think>", 1)[1]
    return text.strip()


class OllamaClient:
    def __init__(self, host: str, model: str, num_ctx: int = 8192, think: bool = False,
                 temperature: float = 0.3, timeout: float = 300):
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.think = think
        self.temperature = temperature
        self.http = httpx.Client(timeout=timeout)

    def check(self) -> None:
        """Fail early with a helpful message if Ollama or the model is missing."""
        try:
            tags = self.http.get(f"{self.host}/api/tags").json()
        except httpx.HTTPError as e:
            raise LLMError(f"Can't reach Ollama at {self.host} ({e}). Is `ollama serve` running?") from e
        names = {m.get("name", "") for m in tags.get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in names:
            raise LLMError(f"Model '{self.model}' not pulled. Run: ollama pull {self.model}")

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             json_mode: bool = False) -> LLMReply:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["format"] = "json"
        try:
            resp = self.http.post(f"{self.host}/api/chat", json=payload)
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        if resp.status_code >= 400:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        msg = data.get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args))
        return LLMReply(content=strip_think(msg.get("content", "")), tool_calls=calls, raw=data)
