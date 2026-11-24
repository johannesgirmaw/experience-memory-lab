# Experiments

This document is the protocol for the **first comparison**. Configs that implement it will live in `configs/experiments/`. If a run diverges from this file, the run is exploratory, not confirmatory.

There are **no results** in this repository.

## Question mapped to a measurement

Hypothesis: experience memory (D) improves long-horizon performance and reduces negative transfer versus A/B/C.

We measure that on **RuleShift-v0**, a synthetic, seed-deterministic environment with three phases and known conflicting rules.

## Environment: RuleShift-v0

A small, fully observed object–tool world. The agent receives a goal in text. Hidden **rule cards** change the correct action sequence. Rule cards are never shown in the observation.

Design constraints:

- State transition is deterministic given `(state, action, rule_id, seed)`.
- Every episode has a programmatic success predicate and a dense or sparse reward that does not depend on the LLM.
- Some pairs of rule cards are **aligned** (same policy, different wording of the goal).
- Some pairs are **conflicting** (similar goal wording, opposite correct action).
- Some episodes include an explicit **context cue** in the observation (for example a room tag or tool serial) that is the correct applicability key.
- The env can evaluate whether a stored applicability predicate is true in the current state. That API is for **scoring only**.

Suggested initial rule families (to be specified in code, not here as results):

| Family | Example of a correct regularity | Conflict |
|---|---|---|
| Access | Color–lock mapping | Mapping reverses after shift |
| Tool use | Tool A is safe for material M | After shift, A damages M; B is required |
| Ordering | File-then-seal | After shift, seal-then-file |

The first shipped world should stay smaller than this table if needed. Completeness of the *shift design* matters more than the number of object types.

### Phases (one run)

| Phase | Episodes | Prior available | Purpose |
|---|---|---|---|
| P1 Learn | \(N_1\) episodes, rule set \(R\) | Writes allowed | Acquisition / in-distribution |
| P2 Conflict | \(N_2\) episodes, conflicting \(R'\) | Memory from P1 remains | Negative transfer |
| P3 Cued mix | \(N_3\) episodes, \(R\) or \(R'\) with a reliable context cue | Full memory | Appropriate vs inappropriate transfer |

Default proposal (to freeze in YAML before the first real comparison): \(N_1=N_2=N_3=12\), 3 rule families, 8 seeds. These numbers are a starting budget, not a power analysis.

A **no-prior control** for P2/P3 is condition A, and, when needed, a B/C/D run that starts at P2 with empty memory (`warm_start: false`). The negative-transfer index needs both.

## Conditions

Implemented as `MemorySystem` backends. Shared agent, env, \(k\), memory-token budget, write cadence.

| ID | Write | Retrieve | Outcome update |
|---|---|---|---|
| A | none | empty context | n/a |
| B | short episode note + embedding | top-\(k\) cosine | none |
| C | skill/recipe if episode succeeded | trigger or similarity | none (or count-only, if pre-registered) |
| D | experience record | top-\(k\) candidates, drop those that fail applicability, rank by confidence and outcomes | yes |

**Forbidden silent differences**

- D must not receive extra observations.
- B must not grow an applicability field.
- C must not receive D’s outcome schema unless that variant is named and registered as C′.
- Prompts for *acting* are identical except for the memory block format.

## Budgets (freeze before confirmatory runs)

Propose and then lock in config (values below are placeholders):

| Knob | Proposed lock | Notes |
|---|---|---|
| Memory tokens per retrieve | 800 | Truncation is a fault if the backend exceeds this |
| \(k\) | 4 | Same for B, C, D |
| Writes | 1 per episode | Per-step writes are a different experiment |
| Act temperature | 0 | Still log residual variance |
| Extra LLM calls for write/filter | count and cap | D’s compiler/filter calls are a cost axis |
| Embedding model | one frozen id | Shared by B and D candidates |

## Primary metrics

Computed by `eval/` from JSONL + env events. No LLM judge.

1. **Long-horizon success.** Mean episode success over P1+P2+P3. Also report by phase.
2. **Negative-transfer index (NTI).**

   \[
   \mathrm{NTI} = \mathbb{E}[S(P2 \mid \text{memory after } P1)] - \mathbb{E}[S(P2 \mid \text{empty memory})]
   \]

   \(S\) is episode success (or mean reward; freeze one). **Lower (more negative) NTI is worse.** The hypothesis predicts D’s NTI is closer to zero or positive, relative to B and C, without collapsing P1 success.

3. **Inappropriate-transfer rate (ITR).** Among retrievals used for an action in P2/P3, the fraction where ground-truth applicability is false. Defined for C and D when a record has a scorable license; for B, use a declared proxy (retrieved note’s source rule_id ≠ current rule_id) and label it as a proxy.

## Secondary metrics

- P3 success split by whether the cued rule matches P1 or P2.
- Recovery: episodes after the P1→P2 boundary until success rate returns to a pre-registered threshold.
- Calibration of D confidence (ECE) against subsequent success when applied.
- Memory size (records) and retrieve/write LLM-call counts.
- Token use and budget faults.
- Appropriate-transfer rate: used record, applicability true, action consistent with that record.

## Analysis plan

1. Freeze configs and code hash. Record `git rev-parse HEAD`, env spec hash, prompt hash.
2. Run all four conditions × seeds. Same seeds across conditions.
3. Primary comparison: paired by seed. Report means and seed-level intervals (bootstrap over seeds). Do not report a single pooled run as the result.
4. Pre-registered pairwise tests: D vs B and D vs C on long-horizon success and on NTI. Treat p-values as descriptive if \(n_{\text{seeds}}\) is small; the interval and the phase curves matter more.
5. Ablations (after the main four, not instead of them):
   - D-no-filter: same writes, retrieve like B.
   - D-no-outcomes: filter on applicability, no outcome updates.
   - D-no-structure: natural-language applicability only.
6. Cost axis: plot success against LLM calls / tokens. A D win at 3× calls is a different claim.

**Multiple comparisons.** The confirmatory claim uses the two primary metrics and the two pairwise contrasts above. Everything else is exploratory.

## Reproducibility

A complete run artifact:

```
artifacts/runs/<run_id>/
  config.yaml          # frozen copy
  meta.json            # git hash, model ids, image digests
  events.jsonl
  scores.json          # written by eval, not by the agent
```

Runner requirements:

- `--seed` and `--condition` are required.
- Task sequence is a pure function of seed + env version.
- `MockLLM` path must produce bit-stable events.jsonl given the same seed.
- Real LLM path logs every request/response id and cannot guarantee bit-stability; scoring must still be deterministic given the logs.

Docker Compose provides Postgres + pgvector. Tests that need the database are marked and skipped if Compose is down.

## Pre-registration discipline

Before the first confirmatory comparison:

- [ ] RuleShift-v0 rules and \(N_1,N_2,N_3\) written in config, not edited after seeing D’s scores.
- [ ] Prompts frozen (hash logged).
- [ ] Budgets frozen.
- [ ] Primary metric (success vs reward) chosen.
- [ ] NTI uses empty-memory P2, not P1, as the control.

Exploratory runs are allowed and should be tagged `exploratory: true` in config.

## Failure modes of the protocol

| Failure | How it shows up | What to do |
|---|---|---|
| D is conservative | NTI improves, P1 and P3-aligned success drop | Report both; hypothesis is not supported |
| B starved by token cap | D wins only at 800 tokens | Sweep 400/800/1600 |
| Rules are pretrained clichés | A is already strong in P1 | Change surface forms; keep rules synthetic |
| Applicability API leaks into prompts | D sees hidden rule ids | Code review + unit test on observation payload |
| Seed count too low | Intervals cover every contrast | Do not claim a difference |

## What this experiment cannot say

It cannot say that experience memory is generally better, that the schema is new, or that the same numbers would appear on web, desktop, or embodied agents. It can say whether, on RuleShift-v0, under a frozen budget, D moved the two primary metrics relative to A/B/C.
