from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experience_memory_lab.eval.score import score_events_path
from experience_memory_lab.experiments.config import load_config
from experience_memory_lab.experiments.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experience_memory_lab.experiments",
        description="Run controlled memory experiments (research harness).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run an experiment")
    run_p.add_argument("--config", type=Path, default=None, help="YAML config path")
    run_p.add_argument("--condition", type=str, default=None, help="A|B|C|D")
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--n-episodes", type=int, default=None)
    run_p.add_argument("--llm", type=str, choices=["mock", "openai"], default=None)
    run_p.add_argument("--artifacts-dir", type=str, default=None)
    run_p.add_argument("--persist-db", action="store_true", default=False)

    score_p = sub.add_parser("score", help="Score an existing events.jsonl")
    score_p.add_argument("events", type=Path, help="Path to events.jsonl")
    score_p.add_argument("--out", type=Path, default=None, help="Optional scores.json path")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        overrides = {
            "condition": args.condition.upper() if args.condition else None,
            "seed": args.seed,
            "n_episodes": args.n_episodes,
            "llm": args.llm,
            "artifacts_dir": args.artifacts_dir,
            "persist_db": True if args.persist_db else None,
        }
        cfg = load_config(args.config, overrides)
        result = run_experiment(cfg)
        print(json.dumps({"run_id": result["run_id"], "scores": result["scores"]}, indent=2))
        print(f"artifacts: {result['run_dir']}", file=sys.stderr)
        return 0

    if args.command == "score":
        scores = score_events_path(args.events)
        text = json.dumps(scores, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        print(text, end="")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
