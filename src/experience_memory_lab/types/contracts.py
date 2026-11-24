"""Shared contracts. Keep these stable — backends plug into them."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PhaseId = Literal["P1", "P2", "P3"]
RecordId = str


class Observation(BaseModel):
    """Agent-visible state. Must never contain hidden rule ids."""

    episode_index: int
    step: int
    goal: str
    visible: dict[str, Any]
    text: str
    phase: PhaseId = "P1"
    # Optional context cue for later phases (room tag, tool serial). Not a rule id.
    context_cue: str | None = None


class Action(BaseModel):
    """Normalized action. LLM output is parsed into this shape."""

    type: str
    args: dict[str, Any] = Field(default_factory=dict)
    raw: str | None = None


class StepRecord(BaseModel):
    step: int
    observation: Observation
    action: Action
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    episode_id: str
    steps: list[StepRecord] = Field(default_factory=list)


class EpisodeOutcome(BaseModel):
    success: bool
    total_reward: float
    n_steps: int
    terminated: bool
    truncated: bool


class MemoryBudget(BaseModel):
    memory_tokens: int = 800
    k: int = 4
    write_cadence: Literal["episode"] = "episode"


class RetrievedContext(BaseModel):
    text: str = ""
    record_ids: list[RecordId] = Field(default_factory=list)
    token_estimate: int = 0
    truncated: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


class MemoryStats(BaseModel):
    n_records: int = 0
    n_retrieves: int = 0
    n_writes: int = 0
    n_llm_calls: int = 0
    tokens_retrieved: int = 0
