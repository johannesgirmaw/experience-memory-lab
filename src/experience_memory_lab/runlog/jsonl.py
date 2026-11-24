"""Append-only structured JSONL run logger.

Uses a logical clock for bit-stable mock runs. Wall clocks are a reproducibility
footgun; do not reintroduce them into equality-sensitive fields.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate. Not a tokenizer claim."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


class JsonlRunLogger:
    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        condition: str,
        seed: int,
        deterministic_time: bool = True,
    ) -> None:
        self.path = path
        self.run_id = run_id
        self.condition = condition
        self.seed = seed
        self.deterministic_time = deterministic_time
        self._seq = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate on create so re-runs with the same path are clean.
        self.path.write_text("", encoding="utf-8")

    def emit(
        self,
        event_type: str,
        *,
        episode_id: str | None = None,
        step: int | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        self._seq += 1
        event: dict[str, Any] = {
            "ts": self._ts(),
            "seq": self._seq,
            "run_id": self.run_id,
            "condition": self.condition,
            "seed": self.seed,
            "episode_id": episode_id,
            "step": step,
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True, separators=(",", ":"), default=str))
            fh.write("\n")
        return event

    def _ts(self) -> str:
        if self.deterministic_time:
            base = datetime(2026, 1, 1, tzinfo=UTC)
            return (base + timedelta(milliseconds=self._seq)).isoformat().replace("+00:00", "Z")
        return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
