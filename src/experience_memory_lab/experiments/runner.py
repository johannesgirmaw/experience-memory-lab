from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from experience_memory_lab.agent.loop import AgentLoop
from experience_memory_lab.env.ruleshift import RuleShiftEnv
from experience_memory_lab.eval.score import score_events_path
from experience_memory_lab.experiments.config import ExperimentConfig
from experience_memory_lab.llm.mock import MockLLM
from experience_memory_lab.llm.openai_compatible import OpenAICompatibleClient
from experience_memory_lab.memory.factory import build_memory
from experience_memory_lab.persist.db import get_database_url, get_session_factory
from experience_memory_lab.persist.models import EpisodeRow, RunRow
from experience_memory_lab.runlog.jsonl import JsonlRunLogger
from experience_memory_lab.types.contracts import MemoryBudget


def make_run_id(cfg: ExperimentConfig) -> str:
    """Deterministic run id from frozen knobs (mock-friendly)."""
    blob = json.dumps(cfg.model_dump(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]
    return f"{cfg.condition}-seed{cfg.seed}-{digest}"


def git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def build_llm(cfg: ExperimentConfig):
    if cfg.llm == "mock":
        return MockLLM(policy=cfg.mock_policy)
    if cfg.llm == "openai":
        return OpenAICompatibleClient()
    raise ValueError(f"Unknown llm backend: {cfg.llm}")


def run_experiment(cfg: ExperimentConfig, *, root: Path | None = None) -> dict[str, Any]:
    root = root or Path.cwd()
    run_id = make_run_id(cfg)
    run_dir = root / cfg.artifacts_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    import yaml

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(), sort_keys=True),
        encoding="utf-8",
    )

    meta = {
        "run_id": run_id,
        "git_hash": git_hash(),
        "env_version": RuleShiftEnv().env_version,
        "condition": cfg.condition,
        "seed": cfg.seed,
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger = JsonlRunLogger(
        run_dir / "events.jsonl",
        run_id=run_id,
        condition=cfg.condition,
        seed=cfg.seed,
        deterministic_time=cfg.deterministic_time,
    )
    logger.emit(
        "run_start",
        config=cfg.model_dump(),
        meta=meta,
    )

    env = RuleShiftEnv(rule_id=cfg.rule_id, phase=cfg.phase, max_steps=cfg.max_steps)
    llm = build_llm(cfg)
    memory = build_memory(cfg.condition, llm=llm)
    budget = MemoryBudget(
        memory_tokens=cfg.memory_tokens,
        k=cfg.k,
        write_cadence="episode",
    )
    memory.configure(run_id, budget)

    agent = AgentLoop(
        env=env,
        memory=memory,
        llm=llm,
        logger=logger,
        budget=budget,
        temperature=cfg.temperature,
        seed=cfg.seed,
    )

    outcomes = []
    for i in range(cfg.n_episodes):
        episode_id = f"{run_id}-ep{i:03d}"
        outcome = agent.run_episode(episode_index=i, episode_id=episode_id)
        outcomes.append({"episode_id": episode_id, "episode_index": i, **outcome.model_dump()})

    mem_stats = memory.stats().model_dump()
    logger.emit("run_end", n_episodes=cfg.n_episodes, memory_stats=mem_stats)

    scores = score_events_path(run_dir / "events.jsonl")
    (run_dir / "scores.json").write_text(
        json.dumps(scores, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if cfg.persist_db and get_database_url():
        _persist_run(cfg, run_id, meta, outcomes)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "scores": scores,
        "outcomes": outcomes,
        "memory_stats": mem_stats,
    }


def _persist_run(
    cfg: ExperimentConfig,
    run_id: str,
    meta: dict[str, Any],
    outcomes: list[dict[str, Any]],
) -> None:
    Session = get_session_factory()
    with Session() as session:
        session.merge(
            RunRow(
                id=run_id,
                condition=cfg.condition,
                seed=cfg.seed,
                env_version=meta["env_version"],
                config_json=cfg.model_dump(),
                git_hash=meta.get("git_hash"),
                status="completed",
            )
        )
        for row in outcomes:
            session.merge(
                EpisodeRow(
                    id=row["episode_id"],
                    run_id=run_id,
                    episode_index=row["episode_index"],
                    phase=cfg.phase,
                    hidden_rule_id=cfg.rule_id,
                    success=row["success"],
                    total_reward=row["total_reward"],
                    n_steps=row["n_steps"],
                )
            )
        session.commit()
