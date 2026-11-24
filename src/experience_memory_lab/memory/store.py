"""In-memory vector store for Milestone 2.

Postgres/pgvector comes later as a drop-in backend behind the same methods.
Offline tests must not require Docker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoredNote:
    record_id: str
    text: str
    embedding: list[float]
    meta: dict[str, Any] = field(default_factory=dict)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._notes: list[StoredNote] = []

    def __len__(self) -> int:
        return len(self._notes)

    def add(self, note: StoredNote) -> None:
        self._notes.append(note)

    def search(self, query: list[float], k: int) -> list[tuple[StoredNote, float]]:
        if k <= 0 or not self._notes:
            return []
        scored = [(note, cosine_similarity(query, note.embedding)) for note in self._notes]
        # Stable tie-break on record_id so MockLLM runs stay bit-stable.
        scored.sort(key=lambda item: (-item[1], item[0].record_id))
        return scored[:k]

    def all(self) -> list[StoredNote]:
        return list(self._notes)
