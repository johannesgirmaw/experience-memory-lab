from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    """Frozen knobs for a run. Prefer YAML files under configs/experiments/."""

    name: str = "m1_condition_a"
    exploratory: bool = True
    condition: Literal["A", "B", "C", "D"] = "A"
    seed: int = 0
    n_episodes: int = 3
    phase: Literal["P1", "P2", "P3"] = "P1"
    rule_id: str = "access_match_v1"
    max_steps: int = 6
    memory_tokens: int = 800
    k: int = 4
    temperature: float = 0.0
    llm: Literal["mock", "openai"] = "mock"
    mock_policy: str = "access_match"
    artifacts_dir: str = "artifacts/runs"
    deterministic_time: bool = True
    persist_db: bool = False

    def budget_dict(self) -> dict[str, Any]:
        return {
            "memory_tokens": self.memory_tokens,
            "k": self.k,
            "write_cadence": "episode",
        }


def load_config(path: Path | None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    data: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        data.update(raw)
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return ExperimentConfig.model_validate(data)
