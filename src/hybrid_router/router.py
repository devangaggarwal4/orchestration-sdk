"""HybridRouter: the public entry point.

    router = HybridRouter.from_config()
    result = await router.acomplete(messages, tools=..., task_type="extract",
                                    session_id="run-42", strategy="hybrid")
    result.response   # CompletionResponse (append result.response.as_message() to history)
    result.decision   # RouteDecision: score, zone, reasons, attempts, validator results

``astream`` yields router events (see ``decision.RouterEvent``); ``acomplete`` is
``astream`` collected; ``complete`` is a blocking wrapper for sync code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Coroutine, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from hybrid_router.backends.base import Backend, BackendError
from hybrid_router.backends.registry import create_backend
from hybrid_router.config import RouterConfig, load_config
from hybrid_router.decision import (
    AttemptEndEvent,
    AttemptRecord,
    AttemptStartEvent,
    RouteDecision,
    RoutedResponse,
    RouteEndEvent,
    RouterEvent,
    RouteStartEvent,
    RoutingError,
    Strategy,
    ValidatorResult,
)
from hybrid_router.scoring import RuleScorer, Scorer, ScoringContext
from hybrid_router.session import SessionStore
from hybrid_router.strategies import AttemptPlan, escalation_attempt, first_attempt
from hybrid_router.types import (
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageEndEvent,
    StreamEvent,
    TextBlock,
    Tier,
    ToolChoice,
    ToolSpec,
)
from hybrid_router.validators import TestRunner, ValidationContext, ValidatorChain, build_chain
from hybrid_router.zones import zone_for

log = logging.getLogger("hybrid_router")

T = TypeVar("T")


class RouteObserver(Protocol):
    """Receives every finished route (telemetry implements this)."""

    async def on_route(
        self,
        decision: RouteDecision,
        request: CompletionRequest,
        response: CompletionResponse | None,
    ) -> None: ...

    async def on_local_tool_error(self, call_ids: list[str]) -> None: ...


@dataclass
class _AttemptOutcome:
    record: AttemptRecord
    response: CompletionResponse | None = None
    buffered_events: list[StreamEvent] = field(default_factory=list)


class HybridRouter:
    def __init__(
        self,
        config: RouterConfig | None = None,
        *,
        backends: Mapping[Tier, Backend] | None = None,
        scorer: Scorer | None = None,
        observers: list[RouteObserver] | None = None,
        telemetry: Any = "auto",
        test_runner: TestRunner | None = None,
    ):
        """``telemetry``: "auto" records to the configured SQLite DB when
        ``telemetry.enabled``; pass a ``TelemetryStore`` to use your own, or None."""
        self.config = config or load_config()
        self.backends: dict[Tier, Backend] = (
            dict(backends)
            if backends
            else {tier: create_backend(cfg, tier=tier) for tier, cfg in self.config.tiers.items()}
        )
        self._custom_scorer = scorer is not None
        self.scorer: Scorer = scorer or build_scorer(self.config)
        self.observers: list[RouteObserver] = list(observers or [])
        if telemetry == "auto":
            telemetry = None
            if self.config.telemetry.enabled:
                from hybrid_router.telemetry import TelemetryStore

                telemetry = TelemetryStore.from_config(self.config)
        self.telemetry = telemetry
        if telemetry is not None:
            self.observers.append(telemetry)
        self.test_runner = test_runner
        self.sessions = SessionStore()
        self._local_slots: asyncio.Semaphore | None = None
        self._sync_loop: asyncio.AbstractEventLoop | None = None
        self._sync_thread: threading.Thread | None = None
        self._chains: dict[tuple[str, str], ValidatorChain] = {}
        self._stale_backends: list[Backend] = []

    @classmethod
    def from_config(cls, path: str | Path | None = None, **kwargs: Any) -> HybridRouter:
        return cls(load_config(path), **kwargs)

    # --- configuration -------------------------------------------------------

    def update_config(self, config: RouterConfig) -> None:
        """Hot-swap configuration. Requests already in flight finish on the old one."""
        old = self.config
        self.config = config
        if not self._custom_scorer:
            self.scorer = build_scorer(config)
        for tier, cfg in config.tiers.items():
            if old.tiers.get(tier) != cfg:
                if (stale := self.backends.get(tier)) is not None:
                    # Closed on aclose(); in-flight calls may still be using it.
                    self._stale_backends.append(stale)
                self.backends[tier] = create_backend(cfg, tier=tier)
        if self.telemetry is not None and hasattr(self.telemetry, "update_config"):
            self.telemetry.update_config(config)
        if old.routing.local_concurrency != config.routing.local_concurrency:
            self._local_slots = None

    def _local_semaphore(self) -> asyncio.Semaphore:
        if self._local_slots is None:
            self._local_slots = asyncio.Semaphore(self.config.routing.local_concurrency)
        return self._local_slots

    # --- public API ----------------------------------------------------------

    def build_request(
        self,
        messages: list[Message] | list[dict[str, Any]] | str | None = None,
        *,
        request: CompletionRequest | None = None,
        tools: list[ToolSpec] | list[dict[str, Any]] | None = None,
        task_type: str | None = None,
        session_id: str | None = None,
        system: str | None = None,
        response_schema: dict[str, Any] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CompletionRequest:
        if request is not None:
            updates: dict[str, Any] = {}
            if task_type is not None:
                updates["task_type"] = task_type
            if session_id is not None:
                updates["session_id"] = session_id
            return request.model_copy(update=updates) if updates else request
        if messages is None:
            raise ValueError("pass messages or request")
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        return CompletionRequest(
            messages=[m if isinstance(m, Message) else Message.model_validate(m) for m in messages],
            system=system,
            tools=[
                t if isinstance(t, ToolSpec) else ToolSpec.model_validate(t) for t in tools or []
            ],
            tool_choice=tool_choice,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            task_type=task_type,
            session_id=session_id,
            metadata=metadata or {},
        )

    async def acomplete(
        self,
        messages: list[Message] | list[dict[str, Any]] | str | None = None,
        *,
        strategy: Strategy | None = None,
        force_tier: Tier | str | None = None,
        test_runner: TestRunner | None = None,
        **request_kwargs: Any,
    ) -> RoutedResponse:
        end: RouteEndEvent | None = None
        async for event in self.astream(
            messages,
            strategy=strategy,
            force_tier=force_tier,
            test_runner=test_runner,
            **request_kwargs,
        ):
            if isinstance(event, RouteEndEvent):
                end = event
        assert end is not None
        if end.response is None:
            raise RoutingError(
                f"all attempts failed: {end.decision.escalation_reason}", end.decision
            )
        return RoutedResponse(response=end.response, decision=end.decision)

    def complete(self, messages: Any = None, **kwargs: Any) -> RoutedResponse:
        """Blocking wrapper around ``acomplete`` for synchronous code.

        Runs on a private event loop thread so backend connections are reused
        across calls. Don't call it from inside a running event loop.
        """
        if _loop_running():
            raise RuntimeError("complete() called inside an event loop; use acomplete()")
        return self._run_sync(self.acomplete(messages, **kwargs))

    async def astream(
        self,
        messages: list[Message] | list[dict[str, Any]] | str | None = None,
        *,
        strategy: Strategy | None = None,
        force_tier: Tier | str | None = None,
        test_runner: TestRunner | None = None,
        **request_kwargs: Any,
    ) -> AsyncIterator[RouterEvent]:
        cfg = self.config  # snapshot: a hot-swap mid-request doesn't affect this one
        scorer = self.scorer
        backends = dict(self.backends)
        start = time.perf_counter()
        req = self.build_request(messages, **request_kwargs)
        strategy = strategy or cfg.routing.strategy
        forced = Tier(force_tier) if force_tier is not None else None

        state = self.sessions.get(req.session_id)
        step_index = state.step_index
        state.step_index += 1
        failed_calls = self.sessions.failed_local_calls(req)
        if failed_calls:
            await self._notify_tool_errors(failed_calls)

        scored = scorer.score(
            req,
            ScoringContext(
                sticky_remaining=state.sticky_remaining,
                prior_local_tool_error=bool(failed_calls),
            ),
        )
        zone = zone_for(scored.score, cfg.routing)
        plan = first_attempt(strategy, zone, scored.score, cfg.routing, forced)
        decision = RouteDecision(
            session_id=req.session_id,
            step_index=step_index,
            task_type=req.task_type,
            strategy=strategy,
            score=scored.score,
            zone=zone,
            reasons=scored.reasons + ([f"forced to {forced.value}"] if forced else []),
            features=scored.features,
            forced=forced is not None,
            planned_tier=plan.tier,
            config_hash=cfg.config_hash(),
        )
        yield RouteStartEvent(
            route_id=decision.route_id,
            score=decision.score,
            zone=zone,
            reasons=decision.reasons,
            planned_tier=plan.tier,
            strategy=strategy,
        )

        final: CompletionResponse | None = None
        final_rec: AttemptRecord | None = None
        last_response: CompletionResponse | None = None
        last_rec: AttemptRecord | None = None
        attempt_req = req
        pending: AttemptPlan | None = plan
        runner = test_runner or self.test_runner

        while pending is not None and len(decision.attempts) < cfg.routing.max_attempts:
            current = pending
            pending = None
            outcome = _AttemptOutcome(
                record=AttemptRecord(
                    attempt_index=len(decision.attempts),
                    tier=current.tier,
                    model=backends[current.tier].model,
                    role=current.role,
                    buffered=current.buffered,
                )
            )
            decision.attempts.append(outcome.record)
            yield AttemptStartEvent(
                attempt_index=outcome.record.attempt_index,
                tier=current.tier,
                model=outcome.record.model,
                role=current.role,
                buffered=current.buffered,
            )
            async for event in self._run_attempt(
                current, attempt_req, outcome, backends, cfg, runner
            ):
                yield event

            rec = outcome.record
            if outcome.response is not None:
                last_response, last_rec = outcome.response, rec
            if rec.ok:
                for event in outcome.buffered_events:
                    yield event
                final, final_rec = outcome.response, rec
                yield AttemptEndEvent(
                    attempt_index=rec.attempt_index,
                    tier=rec.tier,
                    ok=True,
                    discarded=False,
                    validator_results=rec.validator_results,
                )
                break

            failure = rec.failure
            next_tier = self._next_tier(current, rec, failure, scored.score, strategy, cfg)
            if next_tier is not None and len(decision.attempts) < cfg.routing.max_attempts:
                reason = f"{rec.tier.value} failed {failure.name if failure else 'call'}: " + (
                    failure.reason if failure else (rec.error or "unknown")
                )
                decision.escalation_reason = decision.escalation_reason or reason
                rec.wasted = True
                pending = escalation_attempt(next_tier)
                if cfg.routing.escalation_mode == "revise" and outcome.response is not None:
                    attempt_req = revise_request(req, outcome.response, failure)
                yield AttemptEndEvent(
                    attempt_index=rec.attempt_index,
                    tier=rec.tier,
                    ok=False,
                    discarded=True,
                    validator_results=rec.validator_results,
                    error=rec.error,
                )
            else:
                # No escalation left: return what we have (even if it failed checks).
                for event in outcome.buffered_events:
                    yield event
                final, final_rec = last_response, last_rec
                if decision.escalation_reason is None and failure is not None:
                    decision.escalation_reason = f"{rec.tier.value}: {failure.reason}"
                yield AttemptEndEvent(
                    attempt_index=rec.attempt_index,
                    tier=rec.tier,
                    ok=False,
                    discarded=outcome.response is None,
                    validator_results=rec.validator_results,
                    error=rec.error,
                )

        if final is None:
            final, final_rec = last_response, last_rec
        served = final_rec
        if final is not None and served is not None:
            decision.final_tier = served.tier
            decision.final_model = final.model
            served.wasted = False
        decision.success = any(a.ok for a in decision.attempts)
        decision.total_latency_ms = (time.perf_counter() - start) * 1000

        # Session bookkeeping: stickiness and local tool-call attribution.
        if decision.escalated:
            state.sticky_remaining = cfg.scorer.sticky_steps
        elif state.sticky_remaining > 0:
            state.sticky_remaining -= 1
        if final is not None and served is not None and served.tier is Tier.LOCAL:
            self.sessions.remember_local_tool_calls(served.call_id, served.tool_use_ids)

        await self._notify_route(decision, req, final)
        yield RouteEndEvent(decision=decision, response=final)

    # --- internals -----------------------------------------------------------

    async def _run_attempt(
        self,
        plan: AttemptPlan,
        req: CompletionRequest,
        outcome: _AttemptOutcome,
        backends: dict[Tier, Backend],
        cfg: RouterConfig,
        test_runner: TestRunner | None,
    ) -> AsyncIterator[StreamEvent]:
        rec = outcome.record
        backend = backends[plan.tier]
        want_logprobs = backend.capabilities.logprobs and plan.level == "full"
        retries = cfg.routing.cloud_retries if plan.tier.is_cloud else 0

        for attempt in range(retries + 1):
            try:
                async for event in self._call_backend(backend, req, want_logprobs, plan.tier, rec):
                    if isinstance(event, MessageEndEvent):
                        outcome.response = event.response
                    elif plan.buffered:
                        outcome.buffered_events.append(event)
                    else:
                        yield event
                rec.error = None
                break
            except BackendError as e:
                rec.error = f"{type(e).__name__}: {e}"
                outcome.buffered_events.clear()
                if not (e.retryable and attempt < retries):
                    break
                log.info("retrying %s after %s", plan.tier.value, rec.error)

        resp = outcome.response
        if resp is None:
            # Infrastructure failure, not evidence the task is hard: soft severity.
            rec.validator_results = [
                ValidatorResult(name="error", ok=False, severity="soft", reason=rec.error or "")
            ]
            rec.ok = False
            return

        rec.usage = resp.usage
        rec.latency_ms = resp.latency_ms
        rec.ttft_ms = resp.ttft_ms
        rec.stop_reason = resp.stop_reason
        rec.reported_cost_usd = resp.reported_cost_usd
        rec.model = resp.model
        rec.tool_use_ids = [c.id for c in resp.tool_calls]

        chain = self._chain(cfg, plan.level)
        judge_backend = backends.get(cfg.validators.judge.tier)
        rec.validator_results = await chain.run(
            ValidationContext(
                request=req,
                response=resp,
                backend=backend,
                judge_backend=judge_backend,
                test_runner=test_runner,
            )
        )
        rec.ok = ValidatorChain.passed(rec.validator_results)

    async def _call_backend(
        self,
        backend: Backend,
        req: CompletionRequest,
        logprobs: bool,
        tier: Tier,
        rec: AttemptRecord,
    ) -> AsyncIterator[StreamEvent]:
        if tier is Tier.LOCAL:
            queued = time.perf_counter()
            async with self._local_semaphore():
                rec.queue_ms = (time.perf_counter() - queued) * 1000
                async for event in backend.stream(req, logprobs=logprobs):
                    yield event
        else:
            async for event in backend.stream(req, logprobs=logprobs):
                yield event

    def _chain(self, cfg: RouterConfig, level: str) -> ValidatorChain:
        key = (cfg.config_hash(), level)
        if key not in self._chains:
            self._chains[key] = build_chain(cfg.validators, "full" if level == "full" else "light")
        return self._chains[key]

    @staticmethod
    def _next_tier(
        plan: AttemptPlan,
        rec: AttemptRecord,
        failure: ValidatorResult | None,
        score: float,
        strategy: Strategy,
        cfg: RouterConfig,
    ) -> Tier | None:
        if not plan.escalate:
            # pre_route and forced calls don't escalate on validation, but a crashed
            # local call under pre_route may fall back to Sonnet.
            if (
                strategy == "pre_route"
                and plan.role == "direct"
                and plan.tier is Tier.LOCAL
                and rec.error is not None
                and cfg.routing.pre_route_fallback_on_error
            ):
                return Tier.SONNET
            return None
        if plan.tier is Tier.LOCAL:
            if score > cfg.routing.opus_threshold:
                return Tier.OPUS
            if (
                failure is not None
                and failure.severity == "hard"
                and cfg.routing.escalate_hard_to_opus
            ):
                return Tier.OPUS
            return Tier.SONNET
        if plan.tier is Tier.SONNET:
            return Tier.OPUS
        return None

    async def _notify_route(
        self,
        decision: RouteDecision,
        req: CompletionRequest,
        response: CompletionResponse | None,
    ) -> None:
        for obs in self.observers:
            try:
                await obs.on_route(decision, req, response)
            except Exception:  # observers must never break routing
                log.exception("route observer %r failed", obs)

    async def _notify_tool_errors(self, call_ids: list[str]) -> None:
        for obs in self.observers:
            try:
                await obs.on_local_tool_error(call_ids)
            except Exception:
                log.exception("route observer %r failed", obs)

    def _run_sync(self, coro: Coroutine[Any, Any, T]) -> T:
        if self._sync_loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="hybrid-router", daemon=True)
            thread.start()
            self._sync_loop, self._sync_thread = loop, thread
        return asyncio.run_coroutine_threadsafe(coro, self._sync_loop).result()

    async def aclose(self) -> None:
        for backend in [*self.backends.values(), *self._stale_backends]:
            await backend.aclose()
        self._stale_backends.clear()
        for obs in self.observers:
            close = getattr(obs, "aclose", None)
            if close is not None:
                await close()

    def close(self) -> None:
        """Sync counterpart of ``aclose`` (also stops the private loop)."""
        if self._sync_loop is not None:
            self._run_sync(self.aclose())
            self._sync_loop.call_soon_threadsafe(self._sync_loop.stop)
            if self._sync_thread is not None:
                self._sync_thread.join(timeout=5)
            self._sync_loop = None
        elif not _loop_running():
            asyncio.run(self.aclose())

    async def __aenter__(self) -> HybridRouter:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def build_scorer(config: RouterConfig) -> Scorer:
    if config.scorer.kind == "learned":
        path = Path(config.scorer.learned_model_path)
        if path.is_file():
            from hybrid_router.scoring.learned import LearnedScorer

            return LearnedScorer.load(path, config.scorer)
        log.warning("learned scorer model %s not found; using the rule scorer", path)
    return RuleScorer(config.scorer)


def revise_request(
    req: CompletionRequest, draft: CompletionResponse, failure: ValidatorResult | None
) -> CompletionRequest:
    """Original request plus the failed local draft as 'revise this' context."""
    if draft.tool_calls:
        shown = json.dumps([{"name": c.name, "input": c.input} for c in draft.tool_calls])
    else:
        shown = draft.text
    why = f"{failure.name}: {failure.reason}" if failure else "it failed validation"
    note = (
        "A draft response from a smaller model was rejected "
        f"({why}).\n<draft>\n{shown[:8000]}\n</draft>\n"
        "Write a correct, complete response to the request above. You may reuse "
        "anything in the draft that is right."
    )
    messages = list(req.messages)
    if messages and messages[-1].role == "user":
        last = messages[-1]
        messages[-1] = Message(role="user", content=[*last.blocks(), TextBlock(text=note)])
    else:
        messages.append(Message(role="user", content=note))
    return req.model_copy(update={"messages": messages})


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
