"""The Backend interface every model runtime implements.

The router only ever talks to ``Backend``. Adding a runtime (llama.cpp, MLX, an
NPU runtime...) means subclassing ``Backend`` and ``BackendConfig`` and
registering the class; nothing in the router changes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    MessageEndEvent,
    StreamEvent,
    Tier,
    estimate_tokens,
)


class BackendConfig(BaseModel):
    """Settings common to every backend. Subclasses add runtime-specific fields."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    kind: str
    model: str
    # Wall-clock limit for one whole call, streaming included.
    timeout_s: float = 120.0
    # Used when a request doesn't set max_tokens.
    default_max_tokens: int = 4096


class BackendCapabilities(BaseModel):
    tools: bool = False
    streaming: bool = True
    logprobs: bool = False
    json_schema: bool = False
    prompt_caching: bool = False
    exact_token_count: bool = False
    thinking: bool = False


class HealthStatus(BaseModel):
    ok: bool
    detail: str = ""
    latency_ms: float | None = None


# --- errors -----------------------------------------------------------------


class BackendError(Exception):
    """Base class. ``retryable`` says whether the same call might succeed later."""

    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, backend: str = "", status_code: int | None = None):
        super().__init__(message)
        self.backend = backend
        self.status_code = status_code


class BackendTimeout(BackendError):
    retryable = True


class BackendUnavailable(BackendError):
    """Connection refused, DNS failure, 5xx: the service isn't answering."""

    retryable = True


class BackendRateLimited(BackendError):
    retryable = True


class BackendAuthError(BackendError):
    """Missing or rejected credentials."""


class BackendRequestRejected(BackendError):
    """The backend refused this request as invalid (4xx, unknown model...)."""


class BackendBadResponse(BackendError):
    """The backend answered, but with something we couldn't parse."""

    retryable = True


# --- interface --------------------------------------------------------------


class Backend(ABC):
    kind: ClassVar[str]
    config_model: ClassVar[type[BackendConfig]]

    def __init__(self, config: BackendConfig, *, tier: Tier, name: str | None = None):
        self.config = config
        self.tier = tier
        self.name = name or f"{tier.value}:{config.kind}"

    @property
    def model(self) -> str:
        return self.config.model

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abstractmethod
    def stream(
        self, req: CompletionRequest, *, logprobs: bool = False
    ) -> AsyncIterator[StreamEvent]:
        """Yield events for one call. The last event is always ``MessageEndEvent``.

        Raises a ``BackendError`` subclass on failure.
        """

    async def generate(
        self, req: CompletionRequest, *, logprobs: bool = False
    ) -> CompletionResponse:
        async for event in self.stream(req, logprobs=logprobs):
            if isinstance(event, MessageEndEvent):
                return event.response
        raise BackendBadResponse("stream ended without a final message", backend=self.name)

    async def count_tokens(self, req: CompletionRequest) -> int:
        """Input token count. Exact where the backend supports it, else an estimate."""
        return estimate_tokens(req)

    @abstractmethod
    async def health(self) -> HealthStatus: ...

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release connections. Safe to call more than once."""

    async def __aenter__(self) -> Backend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} model={self.model!r}>"


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
