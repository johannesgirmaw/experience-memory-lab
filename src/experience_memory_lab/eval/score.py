"""Programmatic scoring from JSONL. No LLM judge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def score_events_path(path: Path) -> dict[str, Any]:
    events = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return score_events(events)


def score_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") != "episode_end":
            continue
        payload = event.get("payload") or {}
        episodes.append(
            {
                "episode_id": event.get("episode_id"),
                "success": bool(payload.get("success")),
                "total_reward": float(payload.get("total_reward") or 0.0),
            }
        )

    n = len(episodes)
    n_success = sum(1 for e in episodes if e["success"])
    mean_reward = (sum(e["total_reward"] for e in episodes) / n) if n else 0.0
    success_rate = (n_success / n) if n else 0.0

    budget_faults = sum(1 for e in events if e.get("event_type") == "budget_fault")
    llm_calls = sum(1 for e in events if e.get("event_type") == "llm_call")

    return {
        "n_episodes": n,
        "n_success": n_success,
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "budget_faults": budget_faults,
        "llm_calls": llm_calls,
        "episodes": episodes,
    }
