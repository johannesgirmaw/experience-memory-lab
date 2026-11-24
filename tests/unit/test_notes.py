from experience_memory_lab.memory.notes import build_episode_note
from experience_memory_lab.runlog.jsonl import estimate_tokens
from experience_memory_lab.types.contracts import (
    Action,
    EpisodeOutcome,
    Observation,
    StepRecord,
    Trajectory,
)


def _traj() -> Trajectory:
    obs = Observation(
        episode_index=0,
        step=0,
        goal="Open the red door, then done.",
        visible={"door_color": "red", "inventory": None, "door_open": False},
        text="door_color: red",
    )
    step = StepRecord(
        step=0,
        observation=obs,
        action=Action(type="pick_key", args={"color": "red"}),
        reward=0.0,
        terminated=False,
        truncated=False,
    )
    return Trajectory(episode_id="ep0", steps=[step])


def test_note_is_deterministic_and_capped() -> None:
    outcome = EpisodeOutcome(
        success=True, total_reward=1.0, n_steps=1, terminated=True, truncated=False
    )
    a = build_episode_note("ep0", _traj(), outcome, max_tokens=200)
    b = build_episode_note("ep0", _traj(), outcome, max_tokens=200)
    assert a == b
    assert "goal: Open the red door" in a
    assert "pick_key(color=red)" in a
    assert "applicability" not in a
    assert "confidence" not in a

    tiny = build_episode_note("ep0", _traj(), outcome, max_tokens=5)
    assert estimate_tokens(tiny) <= 8  # truncate marker may add a little
    assert "[truncated]" in tiny
