"""Condition B: vector memory — episode notes + embedding similarity.

No applicability filter, no outcome-conditioned retirement. That is D.
"""

from __future__ import annotations

from experience_memory_lab.llm.base import LLM
from experience_memory_lab.memory.notes import build_episode_note
from experience_memory_lab.memory.store import InMemoryVectorStore, StoredNote
from experience_memory_lab.runlog.jsonl import estimate_tokens
from experience_memory_lab.types.contracts import (
    EpisodeOutcome,
    MemoryBudget,
    MemoryStats,
    Observation,
    RecordId,
    RetrievedContext,
    Trajectory,
)


class VectorMemory:
    condition = "B"

    def __init__(self, llm: LLM, *, note_max_tokens: int = 200) -> None:
        self._llm = llm
        self._note_max_tokens = note_max_tokens
        self._store = InMemoryVectorStore()
        self._run_id: str | None = None
        self._budget = MemoryBudget()
        self._n_retrieves = 0
        self._n_writes = 0
        self._n_llm_calls = 0
        self._tokens_retrieved = 0
        self._write_counter = 0

    def configure(self, run_id: str, budget: MemoryBudget) -> None:
        self._run_id = run_id
        self._budget = budget

    def retrieve(self, observation: Observation, goal: str, k: int) -> RetrievedContext:
        self._n_retrieves += 1
        query_text = f"{goal}\n{observation.text}"
        query_vec = self._embed_one(query_text)
        hits = self._store.search(query_vec, k=k)

        record_ids: list[str] = []
        scores: list[float] = []
        blocks: list[str] = []
        truncated = False
        used_tokens = 0

        for note, score in hits:
            block = f"[note {note.record_id} score={score:.4f}]\n{note.text.strip()}"
            block_tokens = estimate_tokens(block)
            if used_tokens + block_tokens > self._budget.memory_tokens and blocks:
                truncated = True
                break
            if used_tokens + block_tokens > self._budget.memory_tokens and not blocks:
                # Even a single note exceeds budget: include truncated text.
                char_budget = self._budget.memory_tokens * 4
                block = block[:char_budget] + "\n...[truncated]"
                truncated = True
                block_tokens = estimate_tokens(block)
            blocks.append(block)
            record_ids.append(note.record_id)
            scores.append(score)
            used_tokens += block_tokens
            if truncated:
                break

        text = "\n\n".join(blocks)
        self._tokens_retrieved += used_tokens
        return RetrievedContext(
            text=text,
            record_ids=record_ids,
            token_estimate=used_tokens,
            truncated=truncated,
            meta={"scores": scores, "backend": "vector"},
        )

    def write(
        self,
        episode_id: str,
        trajectory: Trajectory,
        outcome: EpisodeOutcome,
    ) -> list[RecordId]:
        self._n_writes += 1
        self._write_counter += 1
        note_text = build_episode_note(
            episode_id,
            trajectory,
            outcome,
            max_tokens=self._note_max_tokens,
        )
        embedding = self._embed_one(note_text)
        record_id = f"{self._run_id or 'run'}-note-{self._write_counter:04d}"
        self._store.add(
            StoredNote(
                record_id=record_id,
                text=note_text,
                embedding=embedding,
                meta={
                    "episode_id": episode_id,
                    "success": outcome.success,
                    "total_reward": outcome.total_reward,
                },
            )
        )
        return [record_id]

    def notify_outcome(self, record_ids: list[RecordId], outcome: EpisodeOutcome) -> None:
        # Condition B does not update notes from downstream outcomes.
        del record_ids, outcome

    def stats(self) -> MemoryStats:
        return MemoryStats(
            n_records=len(self._store),
            n_retrieves=self._n_retrieves,
            n_writes=self._n_writes,
            n_llm_calls=self._n_llm_calls,
            tokens_retrieved=self._tokens_retrieved,
        )

    def _embed_one(self, text: str) -> list[float]:
        self._n_llm_calls += 1
        vectors = self._llm.embed([text])
        if not vectors:
            raise RuntimeError("embed() returned no vectors")
        return vectors[0]
