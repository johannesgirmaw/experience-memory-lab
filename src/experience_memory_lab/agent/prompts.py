"""Acting prompts. Identical across conditions except the memory block."""

from __future__ import annotations

from experience_memory_lab.types.contracts import Observation, RetrievedContext

ACT_SYSTEM_PROMPT = (
    "You are a careful agent in a locked-door environment. "
    "Choose exactly one action per turn. "
    'Reply with JSON only: {"type":"pick_key|unlock|done","args":{...}}. '
    "No markdown. No commentary."
)


def build_user_prompt(observation: Observation, memory: RetrievedContext) -> str:
    memory_block = memory.text.strip() if memory.text.strip() else "(empty)"
    return f"{observation.text}\n\n--- memory ---\n{memory_block}\n--- end memory ---\n"
