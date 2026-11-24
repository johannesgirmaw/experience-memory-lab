from experience_memory_lab.llm.mock import MockLLM
from experience_memory_lab.memory.vector import VectorMemory
from experience_memory_lab.types.contracts import (
    Action,
    EpisodeOutcome,
    MemoryBudget,
    Observation,
    StepRecord,
    Trajectory,
)


def _obs(goal: str = "Open the red door, then done.") -> Observation:
    return Observation(
        episode_index=0,
        step=0,
        goal=goal,
        visible={"door_color": "red", "inventory": None, "door_open": False},
        text=f"goal: {goal}\ndoor_color: red\ninventory: null\ndoor_open: false\n",
    )


def _traj(episode_id: str, door: str) -> Trajectory:
    obs = Observation(
        episode_index=0,
        step=0,
        goal=f"Open the {door} door, then done.",
        visible={"door_color": door, "inventory": door, "door_open": True},
        text=f"door_color: {door}",
    )
    return Trajectory(
        episode_id=episode_id,
        steps=[
            StepRecord(
                step=0,
                observation=obs,
                action=Action(type="unlock", args={"door": door}),
                reward=0.5,
                terminated=False,
                truncated=False,
            )
        ],
    )


def test_vector_memory_write_then_retrieve() -> None:
    llm = MockLLM()
    mem = VectorMemory(llm)
    mem.configure("run-b", MemoryBudget(memory_tokens=800, k=2))

    empty = mem.retrieve(_obs(), _obs().goal, k=2)
    assert empty.record_ids == []
    assert empty.text == ""

    outcome = EpisodeOutcome(
        success=True, total_reward=1.5, n_steps=1, terminated=True, truncated=False
    )
    ids = mem.write("ep0", _traj("ep0", "red"), outcome)
    assert len(ids) == 1

    ctx = mem.retrieve(_obs("Open the red door, then done."), "Open the red door, then done.", k=2)
    assert ids[0] in ctx.record_ids
    assert "score=" in ctx.text
    assert ctx.meta["backend"] == "vector"
    assert mem.stats().n_records == 1
    assert mem.stats().n_llm_calls >= 2  # write embed + retrieve embed


def test_token_budget_truncates_extra_notes() -> None:
    llm = MockLLM()
    mem = VectorMemory(llm, note_max_tokens=80)
    mem.configure("run-b", MemoryBudget(memory_tokens=40, k=4))
    outcome = EpisodeOutcome(
        success=True, total_reward=1.0, n_steps=1, terminated=True, truncated=False
    )
    for i, door in enumerate(["red", "blue", "green", "red"]):
        mem.write(f"ep{i}", _traj(f"ep{i}", door), outcome)

    ctx = mem.retrieve(_obs(), "Open the red door, then done.", k=4)
    assert ctx.truncated is True or ctx.token_estimate <= 40
    assert len(ctx.record_ids) >= 1
