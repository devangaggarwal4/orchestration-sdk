"""Simulated tiers for running the eval harness without any model.

Each tier solves a task when it is at least the task's ``min_tier`` label. When it
can't, it fails in one of three ways, picked deterministically per task and tier:
  * hedges ("I'm not sure...")          -> caught by the heuristic validator
  * answers with low token confidence    -> caught by the confidence validator
  * answers confidently and wrongly      -> not caught (under-escalation)
Latencies and token counts are reported, not slept, so a full run takes seconds.
"""

from __future__ import annotations

import hashlib

from hybrid_router.evals.tasks import EvalTask
from hybrid_router.testing import Reply, ScriptedBackend
from hybrid_router.types import TIER_ORDER, CompletionRequest, Tier, Usage

# Rough per-call latencies (ms) to make reports look like real ones.
LATENCY = {Tier.LOCAL: 1800.0, Tier.SONNET: 3500.0, Tier.OPUS: 7000.0}
WRONG = {
    "regex": "0",
    "json": '{"note": "unknown"}',
}


def _mode(task_id: str, tier: Tier) -> int:
    return int(hashlib.sha256(f"{task_id}:{tier.value}".encode()).hexdigest(), 16) % 3


def _rank(t: Tier) -> int:
    return TIER_ORDER.index(t)


def simulated_backend(tier: Tier, tasks: list[EvalTask], model: str) -> ScriptedBackend:
    by_id = {t.id: t for t in tasks}

    def respond(req: CompletionRequest, _n: int) -> Reply:
        task = by_id.get(str(req.metadata.get("eval_task")))
        prompt_tokens = max(sum(len(str(m.content)) for m in req.messages) // 4, 20)
        usage = Usage(input_tokens=prompt_tokens, output_tokens=120)
        lat = LATENCY[tier]
        if task is None:
            return Reply("Simulated answer.", usage=usage, reported_latency_ms=lat)
        capable = _rank(tier) >= _rank(task.min_tier)
        confident = [-0.15] * 20

        if task.is_agentic:
            steps = task.reference_steps or [{"final": "Done."}]
            n = sum(1 for m in req.messages if m.role == "assistant")
            if capable:
                step = steps[n] if n < len(steps) else {"final": steps[-1].get("final", "Done.")}
                if "tool" in step:
                    return Reply(
                        tool_calls=[(step["tool"], step["input"])],
                        usage=usage,
                        logprobs=confident,
                        reported_latency_ms=lat,
                    )
                return Reply(
                    step["final"], usage=usage, logprobs=confident, reported_latency_ms=lat
                )
            mode = _mode(task.id, tier)
            if mode == 0:
                return Reply(
                    tool_calls=[("lookup_everything", {})], usage=usage, reported_latency_ms=lat
                )  # nonexistent tool: structural catch
            if mode == 1:
                return Reply(
                    "I'm not sure which tool to use here.", usage=usage, reported_latency_ms=lat
                )
            return Reply(
                "It's all fine, nothing to report.",
                usage=usage,
                logprobs=confident,
                reported_latency_ms=lat,
            )

        if capable:
            return Reply(
                task.reference or "", usage=usage, logprobs=confident, reported_latency_ms=lat
            )
        wrong = WRONG["json"] if task.response_schema else WRONG["regex"]
        mode = _mode(task.id, tier)
        if mode == 0:
            return Reply(f"I'm not sure, but maybe {wrong}", usage=usage, reported_latency_ms=lat)
        if mode == 1:
            return Reply(wrong, usage=usage, logprobs=[-2.5] * 20, reported_latency_ms=lat)
        return Reply(wrong, usage=usage, logprobs=confident, reported_latency_ms=lat)

    return ScriptedBackend(respond, tier=tier, model=model, logprobs=True)
