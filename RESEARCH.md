# Research plan

## Question

Can an AI agent that converts experience into **context-bounded, evidence-backed knowledge** achieve better **long-horizon performance** and **lower negative transfer** than conventional **vector** or **procedural** memory?

## Hypothesis

An agent equipped with experience memory that tracks **applicability**, **evidence**, **confidence**, **provenance**, and **downstream outcomes** should achieve better long-horizon performance while reducing inappropriate transfer of previous experience.

This is a testable comparison, not a claim of novelty. The lab exists to measure the hypothesis under a fixed protocol.

## Why this comparison

Long-horizon agents fail in two common ways that memory can worsen:

1. **Forgetting.** Useful regularities from earlier episodes are unavailable later.
2. **Negative transfer.** A regularity that was true in one context is applied in another where it is false.

Vector memory typically retrieves by semantic similarity. That helps (1) and can aggravate (2) when two situations *look* alike but *require* different policies.

Procedural memory stores reusable how-to knowledge. That helps when the skill still applies, and can aggravate (2) when the trigger for a skill is too coarse.

The intervention under test is not “more memory.” It is **binding stored knowledge to the conditions under which it is licensed**, and updating that binding from evidence and later outcomes.

## Constructs

Operational definitions used throughout the lab. Informal synonyms in quotes are not additional metrics.


| Construct                  | Definition in this lab                                                                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Episode**                | One finite interaction with the environment under a single hidden rule set and a single goal.                                                                             |
| **Horizon**                | Number of episodes in a run, including at least one **distribution shift**. Long-horizon performance is measured on the full run, not a single episode.                   |
| **Memory write**           | Any persistent record created from an episode that can be retrieved in a later episode.                                                                                   |
| **Memory read**            | Records inserted into the agent’s context (or otherwise used) before or during an action.                                                                                 |
| **Vector memory**          | Unstructured or lightly annotated traces retrieved by embedding similarity, without structured applicability or outcome-conditioned retirement.                           |
| **Procedural memory**      | Stored recipes or skills (text or structured steps) retrieved by task/skill similarity or trigger match.                                                                  |
| **Experience memory**      | Structured records containing a claim, applicability conditions, supporting evidence, confidence, provenance, and a log of downstream outcomes when the record was used.  |
| **Applicability**          | A stated condition under which a record is licensed. May be structured predicates, constrained language, or both.                                                         |
| **Evidence**               | Pointers to observations, actions, or rewards that justified writing the record.                                                                                          |
| **Confidence**             | A numeric score in `[0, 1]` attached to a record, updated by a declared rule (not a free-form adjective).                                                                 |
| **Provenance**             | Immutable metadata: writer identity, model id, prompt hash, env hash, seed, source episode/step ids.                                                                      |
| **Downstream outcome**     | Observed result after a record was applied: success/fail, reward delta, and whether the applicability check passed.                                                       |
| **Negative transfer**      | Performance on a shifted task after prior experience, relative to an otherwise identical agent that did not see the conflicting prior. Formal metric in `EXPERIMENTS.md`. |
| **Inappropriate transfer** | A retrieved record is used when its applicability conditions are false in the current state. Requires ground-truth conditions in the environment.                         |




## Conditions

The initial experiment compares four **memory conditions**. The acting policy, environment, token budget for memory context, retrieval cardinality k, and write cadence are matched unless a deviation is pre-registered.


| ID  | Name                         | What persists                       | How it is retrieved                                    | What must not happen                          |
| --- | ---------------------------- | ----------------------------------- | ------------------------------------------------------ | --------------------------------------------- |
| A   | No Memory                    | Nothing across episodes             | n/a                                                    | Hidden scratch files, leaked traces           |
| B   | Vector Memory                | Episode notes / traces + embeddings | Top-k cosine similarity                                | Applicability filter, outcome retirement      |
| C   | Procedural Memory            | Skills / recipes                    | Skill-trigger or task similarity                       | Outcome-conditioned applicability (that is D) |
| D   | Contextual Experience Memory | Structured experience records       | Candidates, then applicability + confidence + outcomes | Silent fallback to unconstrained RAG          |


Condition D may *use* embeddings to form a candidate set. The experimental contrast is the **post-retrieval filter and the write schema**, not the absence of vectors.

## What would support or weaken the hypothesis

**Support (not proof).** On the pre-registered RuleShift-v0 protocol, D has higher long-horizon success (or reward) than A/B/C **and** a lower negative-transfer index than B and C, with uncertainty intervals that do not include a reversal on the primary metrics, under matched budgets.

**Weaken.** D matches or underperforms B or C on long-horizon success, or reduces negative transfer only by becoming overly conservative (large drop in appropriate transfer / Phase-1 success). Either result is a valid scientific outcome.

**Do not treat as support.** Qualitative “the memories look nicer,” a win on a single seed, a win after unregistered prompt edits, or a win when D is given extra tokens, extra retrievals, or extra model calls.

## Scope

**In scope for the first experiment**

- Four memory conditions on one synthetic environment with known shift points.
- Programmatic rewards (no LLM-as-judge for primary metrics).
- Structured run logs and a deterministic task sequence given a seed.
- A mock LLM so the harness can be tested without network or API keys.

**Out of scope until the first comparison exists**

- Authentication, billing, multi-tenancy, dashboards, Kubernetes, microservices.
- Production serving, complicated frontend, SaaS packaging.
- Claims of generality across real websites, robots, or production assistants.
- Claims of novelty relative to MemGPT, Voyager, Reflexion, ExpeL, case-based reasoning, or related systems.



## Related work (positioning only)

This lab is a **controlled comparison**, not a survey and not a priority claim. Systems that store traces, skills, or reflections (for example retrieval-augmented agents, skill libraries, and verbal reinforcement) are the baselines we re-implement in simplified form. We do not claim that applicability, evidence, or outcome tracking are original ideas. We ask whether those attributes, implemented as a single write/read schema, change measured transfer on a shift benchmark we fully control.

## Threats to validity

These are the main ways the project can fool itself. See also “Assumptions that can bias the research” in `ARCHITECTURE.md`.

1. **Environment–hypothesis alignment.** A synthetic world built to reward applicability filters will favor D by construction. RuleShift-v0 must include cases where similarity is a *correct* cue, not only cases where it is a trap.
2. **Unequal compute.** Extra filter/rerank calls for D are a different intervention than “better memory.” All extra LLM calls must be logged and either matched or reported as a cost axis.
3. **Self-written licenses.** If the same model writes applicability text and later interprets it, D can launder its own biases. Prefer structured predicates the environment can evaluate.
4. **Pretraining leakage.** If the hidden rules are familiar (lock-and-key, common office scripts), No Memory is not a true zero-memory baseline.
5. **Credit assignment.** Outcome updates on long episodes are noisy. Bad credit assignment can make D retire true knowledge or keep false knowledge.
6. **Researcher degrees of freedom.** Prompt wording, k, write cadence, and retirement thresholds can be tuned until D wins. Freeze them in `configs/` before the comparison run.



## What we will not claim

- That D is a new memory algorithm, a product, or a general cognitive architecture.
- That any unpublished number in this repository is a benchmark result. There are no results yet.
- That success on RuleShift-v0 implies success on WebArena, OSWorld, or human assistants.

