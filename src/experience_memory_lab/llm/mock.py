"""Seed-stable mock LLM for offline harness tests.

Policy knowledge is *not* a scientific claim. It exists so the harness can
produce successful and unsuccessful trajectories without an API key.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from experience_memory_lab.llm.base import LLMResponse, Message


def _stable_embed(text: str, dims: int = 8) -> list[float]:
    """Deterministic pseudo-embedding from text bytes. Not a real embedder."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vals = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(dims)]
    norm = sum(v * v for v in vals) ** 0.5 or 1.0
    return [v / norm for v in vals]


class MockLLM:
    """Deterministic actor for RuleShift Access (matching rule).

    Parses the last user message for door color and inventory, then emits a
    JSON action. Same inputs always produce the same output.
    """

    model_id = "mock-llm-v0"

    def __init__(self, *, policy: str = "access_match") -> None:
        self.policy = policy
        self.n_complete = 0
        self.n_embed = 0

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> LLMResponse:
        del temperature, max_tokens, seed  # deterministic; knobs logged by caller
        self.n_complete += 1
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        action = self._decide(user)
        content = json.dumps(action, sort_keys=True, separators=(",", ":"))
        return LLMResponse(
            content=content,
            model=self.model_id,
            finish_reason="stop",
            usage={"prompt_tokens": len(user) // 4, "completion_tokens": len(content) // 4},
            request_id=f"mock-{self.n_complete:04d}",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.n_embed += 1
        return [_stable_embed(t) for t in texts]

    def _decide(self, user_text: str) -> dict[str, Any]:
        if self.policy == "always_done":
            return {"type": "done", "args": {}}
        if self.policy == "access_match":
            return self._access_match_policy(user_text)
        return {"type": "done", "args": {}}

    def _access_match_policy(self, user_text: str) -> dict[str, Any]:
        # Act on the live observation only. Retrieved notes often contain
        # prior door_open/inventory lines and must not hijack the policy.
        live = _live_observation(user_text)
        door = _extract(live, r"door_color['\"]?\s*[:=]\s*['\"]?(\w+)")
        if door is None:
            door = _extract(live, r"Open the (\w+) door")
        inventory = _extract(live, r"inventory['\"]?\s*[:=]\s*['\"]?(\w+)")
        if inventory is None and re.search(r"inventory['\"]?\s*[:=]\s*null", live, re.I):
            inventory = None
        door_open = bool(re.search(r"door_open['\"]?\s*[:=]\s*true", live, re.I))

        if door_open:
            return {"type": "done", "args": {}}
        if door is None:
            return {"type": "done", "args": {}}
        if inventory != door:
            return {"type": "pick_key", "args": {"color": door}}
        return {"type": "unlock", "args": {"door": door}}


def _live_observation(user_text: str) -> str:
    marker = "--- memory ---"
    if marker in user_text:
        return user_text.split(marker, 1)[0]
    return user_text


def _extract(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    val = m.group(1).lower()
    if val in {"none", "null"}:
        return None
    return val
