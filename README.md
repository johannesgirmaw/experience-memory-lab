# experience-memory-lab

A research laboratory for comparing AI-agent memory systems under controlled, reproducible experiments.

**Question.** Can an agent that converts experience into context-bounded, evidence-backed knowledge achieve better long-horizon performance and lower negative transfer than conventional vector or procedural memory?

This is a research project, not a product. The first deliverable is a reproducible four-condition experiment, not a service.

| Document | Purpose |
|---|---|
| [`RESEARCH.md`](RESEARCH.md) | Question, hypothesis, constructs, threats to validity |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | System design, interfaces, directory layout |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Protocol, metrics, analysis plan |
| [`TODO.md`](TODO.md) | Implementation milestones |

## Status

**Milestones 1–2 are implemented:** conditions **A (No Memory)** and **B (Vector Memory)** run end-to-end with `MockLLM`, RuleShift-v0 Access (matching rule), JSONL artifacts, and programmatic scoring. Conditions C/D are not implemented yet. No scientific results are claimed — MockLLM success only proves the harness works.

## Quick start

```bash
# Python 3.12+ (uv recommended)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run condition A or B (offline, no API key)
python -m experience_memory_lab.experiments run \
  --config configs/experiments/m1_condition_a.yaml \
  --condition A --seed 0

python -m experience_memory_lab.experiments run \
  --config configs/experiments/m2_condition_b.yaml \
  --condition B --seed 0

# Score an existing run
python -m experience_memory_lab.experiments score artifacts/runs/<run_id>/events.jsonl

# Tests
pytest -q
```

Optional Postgres (for later persistence / pgvector):

```bash
docker compose up -d
export DATABASE_URL=postgresql+psycopg://emlab:emlab@127.0.0.1:5432/emlab
alembic upgrade head
```

JSONL under `artifacts/runs/` is the source of truth for Milestone 1 scoring. Database persistence is opt-in (`--persist-db`).
