from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from experience_memory_lab.types.contracts import Action, Observation, PhaseId


@runtime_checkable
class Environment(Protocol):
    def reset(self, seed: int, *, episode_index: int = 0) -> Observation: ...

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]: ...

    def hidden_rule_id(self) -> str: ...

    def phase(self) -> PhaseId: ...

    def applicability_truth(self, record: dict[str, Any]) -> bool | None: ...
