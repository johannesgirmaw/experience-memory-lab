from __future__ import annotations

from experience_memory_lab.llm.base import LLM
from experience_memory_lab.memory.base import MemorySystem
from experience_memory_lab.memory.no_memory import NoMemory
from experience_memory_lab.memory.vector import VectorMemory


def build_memory(condition: str, *, llm: LLM | None = None) -> MemorySystem:
    key = condition.strip().upper()
    if key == "A":
        return NoMemory()
    if key == "B":
        if llm is None:
            raise ValueError("Condition B (VectorMemory) requires an LLM for embeddings")
        return VectorMemory(llm)
    raise ValueError(
        f"Memory condition {condition!r} is not implemented yet. "
        "Supported now: A (No Memory), B (Vector Memory). C/D come later."
    )
