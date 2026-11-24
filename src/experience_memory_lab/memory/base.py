"""MemorySystem protocol. Conditions A–D implement this; the agent does not branch."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from experience_memory_lab.types.contracts import (
    EpisodeOutcome,
    MemoryBudget,
    MemoryStats,
    Observation,
    RecordId,
    RetrievedContext,
    Trajectory,
)


@runtime_checkable
class MemorySystem(Protocol):
    condition: str

    def configure(self, run_id: str, budget: MemoryBudget) -> None: ...

    def retrieve(self, observation: Observation, goal: str, k: int) -> RetrievedContext: ...

    def write(
        self,
        episode_id: str,
        trajectory: Trajectory,
        outcome: EpisodeOutcome,
    ) -> list[RecordId]: ...

    def notify_outcome(self, record_ids: list[RecordId], outcome: EpisodeOutcome) -> None: ...

    def stats(self) -> MemoryStats: ...
