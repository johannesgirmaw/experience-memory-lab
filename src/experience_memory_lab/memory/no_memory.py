"""Condition A: nothing persists across episodes."""

from __future__ import annotations

from experience_memory_lab.types.contracts import (
    EpisodeOutcome,
    MemoryBudget,
    MemoryStats,
    Observation,
    RecordId,
    RetrievedContext,
    Trajectory,
)


class NoMemory:
    condition = "A"

    def __init__(self) -> None:
        self._run_id: str | None = None
        self._budget = MemoryBudget()
        self._n_retrieves = 0
        self._n_writes = 0

    def configure(self, run_id: str, budget: MemoryBudget) -> None:
        self._run_id = run_id
        self._budget = budget

    def retrieve(self, observation: Observation, goal: str, k: int) -> RetrievedContext:
        del observation, goal, k
        self._n_retrieves += 1
        return RetrievedContext(text="", record_ids=[], token_estimate=0, truncated=False)

    def write(
        self,
        episode_id: str,
        trajectory: Trajectory,
        outcome: EpisodeOutcome,
    ) -> list[RecordId]:
        del episode_id, trajectory, outcome
        self._n_writes += 1
        return []

    def notify_outcome(self, record_ids: list[RecordId], outcome: EpisodeOutcome) -> None:
        del record_ids, outcome

    def stats(self) -> MemoryStats:
        return MemoryStats(
            n_records=0,
            n_retrieves=self._n_retrieves,
            n_writes=self._n_writes,
            n_llm_calls=0,
            tokens_retrieved=0,
        )
