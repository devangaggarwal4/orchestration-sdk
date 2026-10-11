"""Validator interface. Validators run cheapest-first and the chain stops at the
first failure. Each returns ok/fail, a severity and a reason."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from hybrid_router.backends.base import Backend
from hybrid_router.decision import Severity, ValidatorResult
from hybrid_router.types import CompletionRequest, CompletionResponse

# Caller hook for the execution validator: given extracted code, return True if
# it passes your tests. May be sync or async.
TestRunner = Callable[[str, CompletionRequest], bool | Awaitable[bool]]


@dataclass
class ValidationContext:
    request: CompletionRequest
    response: CompletionResponse
    # The backend that produced the response (used for a second sample).
    backend: Backend | None = None
    # Backend used by the optional judge validator.
    judge_backend: Backend | None = None
    test_runner: TestRunner | None = None
    extras: dict = field(default_factory=dict)


class Validator(ABC):
    name: str

    async def run(self, ctx: ValidationContext) -> ValidatorResult:
        start = time.perf_counter()
        result = await self.check(ctx)
        result.ms = (time.perf_counter() - start) * 1000
        return result

    @abstractmethod
    async def check(self, ctx: ValidationContext) -> ValidatorResult: ...

    def ok(self, reason: str = "") -> ValidatorResult:
        return ValidatorResult(name=self.name, ok=True, reason=reason)

    def skip(self, reason: str) -> ValidatorResult:
        return ValidatorResult(name=self.name, ok=True, skipped=True, reason=reason)

    def fail(self, severity: Severity, reason: str) -> ValidatorResult:
        return ValidatorResult(name=self.name, ok=False, severity=severity, reason=reason)
