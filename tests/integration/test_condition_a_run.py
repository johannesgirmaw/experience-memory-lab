from pathlib import Path

from experience_memory_lab.experiments.config import ExperimentConfig
from experience_memory_lab.experiments.runner import run_experiment


def test_condition_a_mock_run_is_bit_stable(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        name="test_a",
        condition="A",
        seed=0,
        n_episodes=2,
        llm="mock",
        mock_policy="access_match",
        artifacts_dir=str(tmp_path / "runs"),
        deterministic_time=True,
        persist_db=False,
    )
    r1 = run_experiment(cfg, root=tmp_path)
    events1 = Path(r1["run_dir"]) / "events.jsonl"
    text1 = events1.read_text(encoding="utf-8")

    r2 = run_experiment(cfg, root=tmp_path)
    events2 = Path(r2["run_dir"]) / "events.jsonl"
    text2 = events2.read_text(encoding="utf-8")

    assert r1["run_id"] == r2["run_id"]
    assert text1 == text2
    assert r1["scores"]["n_episodes"] == 2
    assert r1["scores"]["success_rate"] == 1.0
    assert (Path(r1["run_dir"]) / "scores.json").exists()


def test_always_done_policy_fails(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        name="test_a_fail",
        condition="A",
        seed=0,
        n_episodes=1,
        llm="mock",
        mock_policy="always_done",
        artifacts_dir=str(tmp_path / "runs"),
        deterministic_time=True,
    )
    result = run_experiment(cfg, root=tmp_path)
    assert result["scores"]["success_rate"] == 0.0
