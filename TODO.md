# TODO

Research first. Do not start auth, UI, Kubernetes, or a second service.

## Milestone 0 — Lab definition

- [x] Inspect repository (empty)
- [x] `RESEARCH.md`
- [x] `ARCHITECTURE.md`
- [x] `EXPERIMENTS.md`
- [x] `TODO.md`
- [x] Proposed directory skeleton

## Milestone 1 — Runnable harness, condition A only

**Goal.** One command runs RuleShift-v0 with No Memory, writes JSONL, and scores success without an API key.

- [x] `pyproject.toml` (Python ≥3.12, pytest, pydantic, sqlalchemy, alembic, ruff)
- [x] `.gitignore` and `artifacts/` exclusion
- [x] `docker-compose.yml`: Postgres + pgvector only
- [x] SQLAlchemy models for `Run`, `Episode`, `Step` (no user/org tables)
- [x] Alembic initial migration
- [x] Structured JSONL logger (`runlog/`, logical clock for bit-stability)
- [x] `LLM` protocol + `MockLLM` (seed-stable) + OpenAI-compatible client
- [x] `MemorySystem` protocol + `NoMemory`
- [x] RuleShift-v0: Access family, matching rule, programmatic reward
- [x] Agent loop using only `MemorySystem` + env + LLM
- [x] CLI: `python -m experience_memory_lab.experiments run --condition A --seed 0`
- [x] `eval` that computes per-episode success from logs
- [x] Unit tests: env determinism, observation does not leak `rule_id`, mock LLM stability
- [x] Integration test: A run with MockLLM produces scorable artifacts

**Exit criterion.** Met: two MockLLM runs with the same config produce identical `events.jsonl`.

## Milestone 2 — Condition B (vector memory)

- [x] Note writer (single summary per episode, token-capped, deterministic — no LLM on write)
- [x] Embeddings via LLM port; in-memory vector store (pgvector deferred; offline-first)
- [x] Top-\(k\) retrieve; enforce memory token budget
- [x] Log retrieve events (ids, scores, truncated via meta)
- [x] Tests with deterministic MockLLM embeddings
- [x] Wire `build_memory("B")`
- [x] Bit-stable MockLLM integration test for B

## Milestone 3 — Condition C (procedural memory)

- [ ] Skill schema (trigger + steps), write on success only (default)
- [ ] Retrieve by trigger match, with similarity fallback if pre-registered
- [ ] Same \(k\) and token budget as B
- [ ] Tests: a conflicting skill is still retrievable by similar trigger

## Milestone 4 — Condition D (experience memory)

- [ ] `ExperienceRecord` + `ExperienceOutcome` tables
- [ ] Compiler: trajectory → claim + structured applicability + evidence + provenance
- [ ] Retrieve: embed candidates → applicability filter → confidence/outcome rank
- [ ] `notify_outcome` updates confidence and status
- [ ] Retirement rule frozen in config
- [ ] Tests: failing applicability not inserted into act prompt
- [ ] Tests: ground-truth ITR scorer never writes into agent context

## Milestone 5 — Full RuleShift-v0 protocol

- [ ] Three phases, aligned and conflicting rule cards, context cues in P3
- [ ] Empty-memory P2 control runs
- [ ] Frozen YAMLs for confirmatory vs exploratory
- [ ] Eval: long-horizon success, NTI, ITR, phase curves, cost counts
- [ ] Script to run A/B/C/D × seeds
- [ ] Ablation configs (D-no-filter, D-no-outcomes)

## Milestone 6 — Analysis, not product

- [ ] Offline plots from `scores.json` (script, not a dashboard app)
- [ ] Write-up of whatever the numbers are, including null results
- [ ] List of protocol deviations, if any

## Explicitly deferred

- Django / REST / admin
- Auth, billing, multi-tenancy
- Kubernetes, multiple microservices
- Frontend
- Multi-env generalization (WebArena, OSWorld, …)
- Hyperparameter search aimed at making D win
- Public leaderboard

## Next implementation milestone

**Milestone 3 — Procedural Memory (C).** Skill schema (trigger + steps), write on success, retrieve by trigger/similarity, same \(k\) and token budget as B. Agent loop stays unchanged.
