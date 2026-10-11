"""Ordered validator chain, cheapest first, stopping at the first failure."""

from __future__ import annotations

from typing import Literal

from hybrid_router.config import ValidatorsConfig
from hybrid_router.decision import ValidatorResult
from hybrid_router.validators.base import ValidationContext, Validator
from hybrid_router.validators.confidence import ConfidenceValidator
from hybrid_router.validators.execution import ExecutionValidator
from hybrid_router.validators.heuristic import HeuristicValidator
from hybrid_router.validators.judge import JudgeValidator
from hybrid_router.validators.structural import StructuralValidator

ChainLevel = Literal["light", "full"]


class ValidatorChain:
    def __init__(self, validators: list[Validator]):
        self.validators = validators

    async def run(self, ctx: ValidationContext) -> list[ValidatorResult]:
        results: list[ValidatorResult] = []
        for v in self.validators:
            result = await v.run(ctx)
            results.append(result)
            if not result.ok:
                break
        return results

    @staticmethod
    def passed(results: list[ValidatorResult]) -> bool:
        return all(r.ok for r in results)


def build_chain(cfg: ValidatorsConfig, level: ChainLevel) -> ValidatorChain:
    """``light``: structural only (easy zone, cloud tiers). ``full``: the whole chain."""
    validators: list[Validator] = []
    if cfg.structural.enabled:
        validators.append(StructuralValidator())
    if level == "full":
        if cfg.heuristic.enabled:
            validators.append(HeuristicValidator(cfg.heuristic))
        if cfg.confidence.enabled:
            validators.append(ConfidenceValidator(cfg.confidence))
        if cfg.execution.enabled:
            validators.append(ExecutionValidator(cfg.execution))
        if cfg.judge.enabled:
            validators.append(JudgeValidator(cfg.judge))
    return ValidatorChain(validators)
