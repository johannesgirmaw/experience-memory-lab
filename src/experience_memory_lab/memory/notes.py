"""Deterministic episode-note writer for condition B.

No LLM call on write: keeps A/B call budgets comparable on the write path and
keeps MockLLM runs bit-stable. Embedding is the only model call for B writes.
"""

from __future__ import annotations

from experience_memory_lab.runlog.jsonl import estimate_tokens
from experience_memory_lab.types.contracts import EpisodeOutcome, Trajectory


def build_episode_note(
    episode_id: str,
    trajectory: Trajectory,
    outcome: EpisodeOutcome,
    *,
    max_tokens: int = 200,
) -> str:
    if not trajectory.steps:
        text = (
            f"episode_id: {episode_id}\n"
            f"success: {str(outcome.success).lower()}\n"
            f"total_reward: {outcome.total_reward:.3f}\n"
            "actions: (none)\n"
        )
        return _truncate(text, max_tokens)

    first = trajectory.steps[0]
    last = trajectory.steps[-1]
    goal = first.observation.goal
    actions = "; ".join(_format_action(s.action.type, s.action.args) for s in trajectory.steps)
    final_vis = last.observation.visible
    text = (
        f"episode_id: {episode_id}\n"
        f"goal: {goal}\n"
        f"success: {str(outcome.success).lower()}\n"
        f"total_reward: {outcome.total_reward:.3f}\n"
        f"n_steps: {outcome.n_steps}\n"
        f"actions: {actions}\n"
        f"final_state: end_door_color={final_vis.get('door_color')} "
        f"end_inventory={final_vis.get('inventory')} "
        f"end_door_open={final_vis.get('door_open')}\n"
    )
    if first.observation.context_cue:
        text += f"context_cue: {first.observation.context_cue}\n"
    return _truncate(text, max_tokens)


def _format_action(atype: str, args: dict) -> str:
    if not args:
        return atype
    inner = ",".join(f"{k}={args[k]}" for k in sorted(args))
    return f"{atype}({inner})"


def _truncate(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    # Character budget ≈ 4 chars/token; keep a clean cut at a newline if possible.
    budget = max_tokens * 4
    cut = text[:budget]
    if "\n" in cut:
        cut = cut.rsplit("\n", 1)[0] + "\n"
    return cut + "...[truncated]\n"
