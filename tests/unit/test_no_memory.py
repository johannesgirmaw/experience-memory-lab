from experience_memory_lab.memory.no_memory import NoMemory
from experience_memory_lab.types.contracts import (
    EpisodeOutcome,
    MemoryBudget,
    Observation,
    Trajectory,
)


def test_no_memory_returns_empty_context() -> None:
    mem = NoMemory()
    mem.configure("run", MemoryBudget())
    obs = Observation(
        episode_index=0,
        step=0,
        goal="g",
        visible={},
        text="t",
    )
    ctx = mem.retrieve(obs, "g", k=4)
    assert ctx.text == ""
    assert ctx.record_ids == []
    written = mem.write(
        "ep0",
        Trajectory(episode_id="ep0"),
        EpisodeOutcome(success=True, total_reward=1.0, n_steps=1, terminated=True, truncated=False),
    )
    assert written == []
    assert mem.stats().n_retrieves == 1
    assert mem.stats().n_writes == 1
