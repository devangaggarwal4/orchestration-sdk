"""Eval task model and loader."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hybrid_router.types import Tier


class EvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    difficulty: Literal["easy", "medium", "hard"]
    task_type: str
    min_tier: Tier
    prompt: str
    system: str | None = None
    response_schema: dict[str, Any] | None = None
    tools: list[str] = Field(default_factory=list)
    checks: list[dict[str, Any]]
    reference: str | None = None
    reference_steps: list[dict[str, Any]] | None = None

    @property
    def is_agentic(self) -> bool:
        return bool(self.tools)


def load_tasks(path: str | Path | None = None) -> list[EvalTask]:
    if path is not None:
        text = Path(path).read_text()
    else:
        text = resources.files("hybrid_router.evals").joinpath("tasks.yaml").read_text()
    tasks = [EvalTask.model_validate(t) for t in yaml.safe_load(text)]
    ids = [t.id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate task ids: {sorted(dupes)}")
    return tasks
