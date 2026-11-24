import json
from pathlib import Path

from experience_memory_lab.experiments.config import ExperimentConfig
from experience_memory_lab.experiments.runner import run_experiment


def test_condition_b_mock_run_is_bit_stable_and_writes_notes(tmp_path: Path) -> None:
    cfg = ExperimentConfig(
        name="test_b",
        condition="B",
        seed=0,
        n_episodes=3,
        llm="mock",
        mock_policy="access_match",
        artifacts_dir=str(tmp_path / "runs"),
        deterministic_time=True,
        persist_db=False,
        k=4,
        memory_tokens=800,
    )
    r1 = run_experiment(cfg, root=tmp_path)
    text1 = (Path(r1["run_dir"]) / "events.jsonl").read_text(encoding="utf-8")

    r2 = run_experiment(cfg, root=tmp_path)
    text2 = (Path(r2["run_dir"]) / "events.jsonl").read_text(encoding="utf-8")

    assert r1["run_id"] == r2["run_id"]
    assert text1 == text2
    assert r1["scores"]["success_rate"] == 1.0
    assert r1["memory_stats"]["n_records"] == 3
    assert r1["memory_stats"]["n_writes"] == 3

    events = [json.loads(line) for line in text1.splitlines() if line]
    retrieves = [e for e in events if e["event_type"] == "retrieve"]
    # First episode has empty memory; later episodes should retrieve prior notes.
    assert retrieves[0]["payload"]["record_ids"] == []
    later = [e for e in retrieves if e["payload"]["record_ids"]]
    assert later, "expected non-empty retrieves after the first write"
    assert all("scores" in e["payload"]["meta"] for e in later)
