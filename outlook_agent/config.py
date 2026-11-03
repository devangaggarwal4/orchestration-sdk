"""Settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Settings:
    client_id: str = ""
    tenant: str = "common"
    scopes: list[str] = field(
        default_factory=lambda: ["Mail.ReadWrite", "Mail.Send", "Calendars.Read", "User.Read"]
    )
    token_cache: Path = Path.home() / ".outlook_agent" / "token_cache.json"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_num_ctx: int = 8192
    ollama_think: bool = False

    # Max characters of an email body handed to the model. Keeps an 8B model's context sane.
    max_body_chars: int = 2500

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(Path.cwd() / ".env")
        env = os.environ
        return cls(
            client_id=env.get("OUTLOOK_CLIENT_ID", ""),
            tenant=env.get("OUTLOOK_TENANT", "common"),
            ollama_host=env.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
            ollama_model=env.get("OLLAMA_MODEL", "qwen3:8b"),
            ollama_num_ctx=int(env.get("OLLAMA_NUM_CTX", "8192")),
            ollama_think=env.get("OLLAMA_THINK", "0").lower() in ("1", "true", "yes"),
            max_body_chars=int(env.get("OUTLOOK_MAX_BODY_CHARS", "2500")),
        )
