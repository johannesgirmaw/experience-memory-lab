# Architecture

The system is an **experiment harness**, not a platform. Every module exists to run the four memory conditions under a shared agent loop, persist artifacts, and score them.

Django is **not** used in the first implementation. An HTTP API is not required to run or score experiments. Persistence is SQLAlchemy + Alembic. If a read-only inspection API becomes useful later, it can be added as a thin layer over the same models.

## Design principles

1. **One agent loop, four memory backends.** Conditions differ only behind `MemorySystem`.
2. **The environment is the source of truth.** Rewards, shifts, and ground-truth applicability come from code, not from the model.
3. **Budgets are first-class.** Token, retrieval, and LLM-call budgets are configured and enforced, not hoped for.
4. **Runs are artifacts.** A run directory plus database rows must be enough to reproduce scoring without the original process.
5. **Determinism is layered.** Environment and task sequence are deterministic given a seed. The LLM may not be; seeds, model ids, and raw responses are always logged.
6. **No product surface.** No auth, billing, UI, or cluster topology.

## Logical view

```
                  ┌─────────────────┐
                  │  Experiment CLI │
                  │  (configs YAML) │
                  └────────┬────────┘
                           │
           ┌───────────────▼────────────────┐
           │           Runner                │
           │  seed, condition, phase, budget │
           └─┬───────────┬───────────┬───────┘
             │           │           │
     ┌───────▼──┐  ┌─────▼─────┐  ┌─▼──────────┐
     │ RuleShift │  │   Agent   │  │  LLM port  │
     │    env    │  │   loop    │  │ OpenAI-c.  │
     └───────┬──┘  └─────┬─────┘  │ + MockLLM  │
             │           │        └────────────┘
             │     ┌─────▼──────┐
             │     │ MemorySystem│
             │     │ A B C D     │
             │     └─────┬──────┘
             │           │
           ┌─▼───────────▼──┐     ┌─────────────┐
           │  PostgreSQL    │     │ JSONL logs  │
           │  + pgvector    │     │ run artifacts│
           └────────────────┘     └─────────────┘
```

Scoring reads logs and env events. It does not ask the LLM whether the agent “did well.”

## Directory structure

Proposed layout. Packages are empty until the first implementation milestone.

```
experience-memory-lab/
  README.md
  RESEARCH.md
  ARCHITECTURE.md
  EXPERIMENTS.md
  TODO.md
  pyproject.toml                 # later: package, pytest, ruff
  docker-compose.yml             # later: postgres + pgvector only
  alembic.ini                    # later
  configs/
    experiments/                 # frozen YAML: condition, seed, budgets
    models/                      # model id, temperature, timeout
  src/experience_memory_lab/
    agent/                       # observe–retrieve–act–write loop
    memory/
      base.py                    # MemorySystem protocol
      no_memory.py               # A
      vector.py                  # B
      procedural.py              # C
      experience.py              # D
    env/
      ruleshift/                 # RuleShift-v0
    eval/                        # metrics from logs, no LLM judge
    llm/                         # OpenAI-compatible client + MockLLM
    runlog/                      # structured JSONL (not named logging/: avoids stdlib shadow)
    persist/                     # SQLAlchemy models, session, Alembic env
    experiments/                 # runner, config loading, run ids
    types/                       # shared Pydantic contracts
  tests/
    unit/
    integration/                 # compose + mock LLM
  scripts/                       # run_experiment.py, score_run.py
  artifacts/                     # generated; gitignored
```

Keep `src/experience_memory_lab` importable. Do not add `services/`, `apps/`, or a `frontend/`.

## Shared interfaces

These are the contracts. Implementation comes later; names should stay stable.

### Environment

```text
reset(seed) -> Observation
step(action) -> (Observation, Reward, Terminated, Truncated, Info)
hidden_rule_id() -> str          # logged, never shown to the agent
phase() -> PhaseId
applicability_truth(record) -> Optional[bool]
  # used only by eval, never by A/B/C, and never injected into D's context
```

`Info` may include ground-truth predicates for scoring. The agent observation must not include hidden rules.

### MemorySystem

```text
configure(run_id, budget) -> None
write(episode_id, trajectory, outcome) -> list[RecordId]
retrieve(observation, goal, k) -> RetrievedContext
notify_outcome(record_ids, outcome) -> None   # no-op for A/B; required for D
stats() -> MemoryStats                        # sizes, calls, tokens
```

`RetrievedContext` is a token-bounded payload plus the record ids used. The runner truncates if a backend exceeds the budget; exceeding the budget is a logged fault, not a silent advantage.

### Agent loop

Per episode, for every condition:

1. `obs = env.reset`
2. `ctx = memory.retrieve(obs, goal, k)`
3. `action = llm.act(obs, goal, ctx)`
4. `obs, reward, done, ... = env.step(action)`
5. Repeat 2–4 until termination.
6. `memory.write(...)` and `memory.notify_outcome(...)` once per episode unless a config says per-step writes (frozen in YAML).

The loop does not branch on condition id except to construct the `MemorySystem`.

### LLM port

```text
complete(messages, *, temperature, max_tokens, seed) -> LLMResponse
embed(texts) -> list[list[float]]
```

Two implementations: `OpenAICompatibleClient` and `MockLLM`. Tests use `MockLLM`. `MockLLM` must be able to play a scripted, seed-stable policy so the harness can be scored without a network.

No provider-specific code outside `llm/`.

## Persistence

**PostgreSQL + pgvector** store:

- runs, episodes, steps
- memory records (notes, skills, experience records)
- embeddings
- retrieval events (what was fetched, rank, scores)
- application events (what was used, outcome)

**JSONL artifacts** store the same events in append-only files under `artifacts/runs/<run_id>/`. The database is for query and vector search. JSONL is the reproducibility backup and the default input to `eval/`.

SQLAlchemy models map 1:1 to these tables. Alembic owns migrations. Do not introduce a second ORM.

### Experience record (condition D)

The schema is the intervention.

```text
ExperienceRecord
  id
  run_id, source_episode_id, source_step_ids
  claim                 # short knowledge statement
  applicability         # JSON predicates the env can evaluate
  evidence_refs         # step / observation ids
  confidence            # float, update rule named in config
  provenance            # model, prompt_hash, env_hash, seed, writer
  status                # candidate | supported | contradicted | retired
  embedding             # optional, candidate retrieval only
```

```text
ExperienceOutcome
  record_id, episode_id
  applied               # bool
  applicability_passed  # bool (agent-side check, not ground truth)
  reward_delta
  success
```

Vector notes and procedural skills get simpler tables. They must **not** grow D-only columns “for convenience.”

## Logging

One JSON object per line. Required fields on every event:

```text
ts, run_id, condition, seed, episode_id, step, event_type
```

Event types: `run_start`, `phase_change`, `retrieve`, `llm_call`, `action`, `env_step`, `write`, `outcome`, `budget_fault`, `run_end`.

Payloads include hashes of prompts and of the env spec. No free-text logging that cannot be parsed.

## Major design decisions

| Decision | Choice | Reason |
|---|---|---|
| Web framework | None for v0 | Experiments are batch jobs. Django would add an unused request cycle and a second persistence story. |
| ORM | SQLAlchemy + Alembic | Matches Postgres/pgvector use; migrations are explicit. |
| API | CLI + config files | Reproducible, scriptable, reviewable in git. |
| Memory abstraction | Single protocol | Prevents condition-specific agent forks, which are an unmeasured intervention. |
| Primary scoring | Programmatic | Avoids LLM-as-judge circularity on the hypothesis. |
| D retrieval | Embed candidates, then filter | Isolates the hypothesis (bounded use) from “no vectors allowed.” |
| Applicability language | Structured predicates first | Natural-language-only licenses cannot be scored for inappropriate transfer. |
| LLM | OpenAI-compatible + mock | Swap providers without rewriting the loop; tests stay offline. |
| Compose | Postgres + pgvector only | Enough to persist embeddings. No app container required at first. |
| Frontend | None | Plots come later from eval scripts, not from a product UI. |

## Assumptions that can bias the research

These are load-bearing. If an assumption is false, a D win (or loss) is easy to misread.

1. **Matched budgets are a fair comparison.** Forcing B and D to the same token cap may clip useful traces for B or clip evidence for D. Report a budget-sensitivity sweep before interpreting a single operating point.
2. **Structured predicates are expressible.** If RuleShift rules cannot be written as the same predicate language D uses, we either leak env structure into D or make D inarticulate. Both bias the result.
3. **Writes are equally frequent.** If D writes richer records at the same cadence, it may win on write quality rather than on applicability tracking. A write-ablation (D schema without outcome updates; D schema without applicability filter) is required before attributing gains.
4. **The mock and the real LLM test the same harness.** A harness that only “works” under `MockLLM` can hide prompt-size bugs that appear only with real models.
5. **Similarity is the right candidate generator for D.** If the true analogue is not near in embedding space, D never sees the record and cannot demonstrate bounded use.
6. **One environment is informative.** RuleShift-v0 can at best support a *local* claim. It cannot support a general memory-architecture claim.
7. **Temperature and decoding are a wash.** Residual LLM nondeterminism can look like a memory effect if seed counts are small.

## Explicit non-goals in code

- Shared `User`, `Org`, or `ApiKey` models.
- Redis, task queues, or multiple deployable services.
- A React/Vue app.
- Automatic hyperparameter search over prompts until D wins.
