# SLM Factory Pipeline

**Effective date: 2026-07-29.** Rewritten against the code as ground truth. Every claim below
was checked against a symbol in the repository, cited inline as `path::symbol`. Where the
previous documentation and the code disagreed, the code won; the corrections are listed in
[§12.3](#123-audit-claims-that-were-stale-and-are-now-corrected).

This document absorbs and replaces `docs/intervention_capability_audit.md`. Items from that
audit that are **not** implemented are collected in [§12](#12-not-implemented) rather than
described as behavior.

> **Update 2026-08-19 — the `task_type` channel is gone, and with it a lot of what was written
> here.** Eight concrete benchmarks used to share five abstract channels (`classification`, `NER`,
> `math_reasoning`, `code_generation`, `generation`, plus `function_call`/`diff` bolted on later),
> and behaviour was decided by `if task_type == ...` chains. Behaviour now comes from a per-task
> `TaskSpec` in `tasks/`, one module per benchmark, **every field required and no defaults**, so a
> decision nobody made for a task is an import-time error instead of a silent runtime fallthrough.
> `state["task"]` and `EvalSet.task` hold a registry NAME. `TaskSpec.family` survives only as a
> descriptive tag for reports and model-selection hints and **must never be read as a dispatch
> key** (`tests/test_task_registry.py` enforces that).
>
> Every section below that described a per-task-type table, a three-strategy `data_rebuild`, a
> curriculum rebuilt to a target size, or a paid-acquisition budget has been corrected in place and
> the correction dated. **[`interventions.md`](interventions.md) is the authority for the
> iterate → `data_rebuild` → curate loop**; the sections here point at it rather than restating it.
> The reasoning behind the rebuild is in `Evan's Notes/08-19b-task-registry-rebuild.md`, and the
> defects it introduced and fixed are B299–B305.
>
> Also deleted on 2026-08-18, and therefore no longer described as behaviour anywhere below:
> `code_generation` / APPS / MBPP / HumanEval **and their execution sandbox**, the `diff` scorer,
> `sms_spam`, `fpb`, `arc`, `multilingual`, `structured_extraction`, `NAMED_BENCHMARK_TASK_TYPES`,
> `TASK_REQUIRED_FIELDS`, `TASK_METRIC_NAMES` (now `eval.harness.task_metric_name(task)`), and the
> paid-acquisition ledger.
>
> **Amended 2026-08-23 — `sms_spam` is back, and the registry is now NINE tasks.** It is a rewrite
> rather than a revert: the deleted loader split the corpus unshuffled (B44), never deduplicated
> (415 repeated messages, so copies straddled the eval firewall), and fetched an unchecksummed zip
> over plain HTTP. The new one deduplicates before splitting, stratifies the holdout on a pinned
> seed, and reads a checksummed `data/local/sms_spam` bundle. It scores `minority_f1` because the
> corpus is ~87% ham and both accuracy and macro-F1 reward answering `ham` for everything. Nothing
> else in that deletion list has returned. See `Evan's Notes/08-23-calendar-variance-sms-spam-small-models.md`.

---

## Table of contents

1. [Graph topology](#1-graph-topology)
2. [Global guards](#2-global-guards)
3. [Pre-graph: hardware research and filtering](#3-pre-graph-hardware-research-and-filtering)
4. [Cold-start entry nodes](#4-cold-start-entry-nodes)
5. [Model selection strategies](#5-model-selection-strategies)
6. [The shared loop](#6-the-shared-loop)
7. [Interventions](#7-interventions)
8. [Hyperparameter contract](#8-hyperparameter-contract)
9. [Model pool, tiers, and variant identity](#9-model-pool-tiers-and-variant-identity)
11. [Durability and checkpoint/requeue](#11-durability-and-checkpointrequeue)
12. [Not implemented](#12-not-implemented)
13. [State field reference](#13-state-field-reference)

---

## 1. Graph topology

`agent/graph.py::build_graph(mode, checkpointer)` builds the LangGraph state machine over
`agent/state.py::AgentState`. `graph_topology_descriptor(mode)` is the canonical structure and is
used to reject unsafe checkpoint resumes.

**There is one mode, `cold_start`.** The `mode` parameter is retained because it is part of the
checkpoint compatibility fingerprint and the run manifest, and `graph_topology_descriptor` raises
on anything else. A second `production` mode existed until 2026-07-29; its entry chain
(`trace_ingest` → `live_confirm` → `parent_awareness`) was never wired into the topology and the
idea was scrapped, so those nodes and `agent/nodes/production/` no longer exist. Sections below that
still discuss production behaviour are retained as history and are marked where they are.

**Every edge is conditional.** There are no unconditional `add_edge` calls — each transition
passes through a guard that can divert to `END` (see [§2](#2-global-guards)).

### Cold-start mode

```
 (entry) ──▶ task_analysis ──▶ eval_setup ──▶ model_selection ──┐
                                                                │
      ┌─────────────────────────────────────────────────────────┘
      ▼
  ┌────────┐      ┌───────┐      ┌──────────┐
  │ curate │─────▶│ train │─────▶│ evaluate │
  └────────┘      └───────┘      └──────────┘
      ▲               ▲                │
      │               │        ┌───────┴───────┐
      │               │        ▼               ▼
      │               │   ┌──────────┐    ┌─────────┐
      │               │   │ rollback │───▶│ iterate │
      │               │   └──────────┘    └─────────┘
      │               │                        │
      │               │   ┌────────────────────┼───────────────┬────────────────┐
      │               │   ▼                    ▼               ▼                ▼
      │               └─ train               curate      ┌──────────┐   ┌────────────────┐
      │                                        │         │ escalate │   │ downward_probe │
      └────────────────────────────────────────┘         └──────────┘   └────────────────┘
                                                              │             │   ▲
                                            curate / END ◀────┘             └───┘ self-loop
                                                                            │
                                                                           END
```

### Exact routing table

| From | Router | Destinations |
|---|---|---|
| entry | `_route_before(entry)` | `task_analysis`, `END` |
| `task_analysis` | `_route_before` | `eval_setup`, `END` |
| `eval_setup` | `_route_before` | `model_selection`, `END` |
| `model_selection` | `_route_before` | `curate`, `END` |
| `curate` | `_route_before` | `train`, `END` |
| `train` | `_route_before` | `evaluate`, `END` |
| `evaluate` | `_route_after_evaluate` | `rollback` if `should_rollback`, else `iterate`; `END` |
| `rollback` | `_route_before` | `iterate`, `END` |
| `iterate` | `_route_after_iterate` | `train`, `curate`, `escalate`, `downward_probe`, `END` |
| `escalate` | `_route_after_escalate` | `train`, `curate`, `END` |
| `downward_probe` | `_route_after_downward_probe` | `downward_probe` (self-loop), `END` |

**Three structural facts worth stating explicitly:**

- **`rollback` never routes to `train`.** It routes to `iterate`. Training is
  (near-)deterministic, so re-training the identical config after a regression reproduces the
  same regressing score and rolls back again — forever. `iterate` is forced to choose a
  *different* action.
- **`downward_probe` self-loops.** Each execution is one durable step (plan, then train+eval),
  so a crashed training process resumes without repeating a paid model-choice call or
  re-adopting a probe.
- **`escalate`'s `train` branch is registered but unreachable** under current code —
  `escalate_node` only ever writes `next_action = "curate"` or `"terminate"`. Kept for
  forward compatibility.

---

## 2. Global guards

Every node is wrapped twice: `guard_graph_node(name, instrument_node(name, node))`.

### `guard_graph_node` — `agent/graph.py`

Runs **before** the node body:

1. `_graph_steps >= MAX_TURNS_MAIN` (1500) → raise `RecursionBudgetExhausted`. This is a
   cumulative, checkpoint-durable count, distinct from LangGraph's per-invocation
   `recursion_limit`.
2. `_wallclock_exceeded()` → **skip the node entirely**, set
   `_wallclock_terminated_before = <node name>` and `next_action = "terminate"`, return.
   This is why a long node (train, evaluate) cannot start near the deadline and get
   hard-killed mid-write.
3. Run the node; raise `TypeError` if it returns a non-`Mapping`.
4. Increment `_graph_steps`.

`_must_terminate(state)` = `_wallclock_exceeded() or _graph_steps >= MAX_TURNS_MAIN`, and is
consulted by every router — so termination is checked on both sides of every edge.

### Guard constants

| Constant | Value | Where | Meaning |
|---|---|---|---|
| `MAX_TURNS_MAIN` | 1500 | `config/config.py` | Cumulative node executions; also the LangGraph `recursion_limit` |
| `MAX_WALLCLOCK_S` | `14*3600` | `config/config.py` | 0 disables. Requeue scripts set 0 and let Slurm `USR1` drive rollover |
| `turn_budget` | 1500 cold / 500 prod | `AgentState` | Charged at 2 turns per iteration (`curate` + `train`) |
| `STAGNATION_WINDOW` | 15 | `agent/nodes/iterate.py` | Evals examined by the stagnation test (over the append-only `eval_history`, so rollback cannot reset it) |
| `STAGNATION_MIN_DELTA` | 0.02 | `agent/nodes/iterate.py` | Minimum window gain that counts as progress |
| `MAX_EVALS_BEFORE_ESCALATION` | 30 | `agent/nodes/iterate.py` | Unconditional ceiling: escalate after this many evals on one model without meeting the goal |
| `SLM_SURGICAL_MAX_CATEGORIES` | 5 | `agent/nodes/curate.py` | Most failure categories targeted per surgical rebuild |
| `SURGICAL_MIN/MAX_ROWS_PER_CATEGORY` | 25 / 600 | `agent/nodes/curate.py` | Bounds on a category's share of the rebuild, which is otherwise proportional to its failure count |
| `MIN_REBUILD_ROWS` / `MAX_REBUILD_ROWS` | 50 / 2000 | `agent/data_rebuild.py` | Clamp on a plan's `rows`. The floor stops a whole train+eval cycle being spent on a handful of rows; the ceiling stops one turn dominating the run |
| `MAX_FAILED_DISCOVERY_ROUNDS` | 2 | `agent/data_rebuild.py` | Consecutive web-research rounds contributing zero novel rows before `mine_new_real` is retired for the run |
| `TaskSpec.initial_train_cap` | 5000 (all 8 tasks) | `tasks/` | Gold rows requested at cold start. The loader returns as many as it has, up to this. **Not a fraction of anything** |
| `TaskSpec.eval_cap` | 1000 (all 8 tasks) | `tasks/` | Held-out eval rows requested. `SLM_EVAL_SIZE_CAP` overrides |
| `MIN_CURRICULUM_ROWS` | 500 | `agent/nodes/curate.py` | Viability FLOOR — curate raises below it. There is no target and nothing pads |
| `SLM_EVAL_JUDGE_OVERLAP_CHUNK` | 100 | `eval/harness.py` | Rows per generate-then-judge chunk so judging overlaps the next generation batch; `0` disables |
| `DEFAULT_STOP_THRESHOLD` | 0.96 | `config/config.py` | Used only if the planner supplies none |

**Update 2026-08-19 — four constants left this table and one changed meaning.**

- `SLM_SURGICAL_SYNTH_SHARE` (0.20) is gone because there is no longer a balanced fill to take the
  other 80%. Surgical synthesis is now the whole of synthesis, so a share of it is meaningless.
  `SLM_SURGICAL_MAX_PAIRS` became `SLM_SURGICAL_MAX_CATEGORIES` when confusion *pairs* were replaced
  by the task's own failure **categories** (B296).
- **`MAX_PAID_ACQUIRE_ROUNDS_PER_RUN` (9), `MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN` (3), the durable
  reservation ledger, `plan_budget_identity` and `reserve_paid_acquisition` were all removed.** They
  bounded paid Exa calls during dataset *discovery*, but the same counter gated the re-read path,
  where re-reading a corpus already in the local cache costs nothing. What is metered now is
  failure, not spend: `MAX_FAILED_DISCOVERY_ROUNDS`.
- **`CURRICULUM_SIZE_FLOOR`, `EVAL_SET_SIZE`, `DATA_SIZE_CEILING` and `agent/data_sizing.py` are all
  DELETED** (2026-08-19). The per-tier novelty × capacity target they bounded had exactly one
  consumer — the `× 0.65` split that produced the mystery 3,250-row curriculum — and once that split
  was removed nothing read the figure at all (B305). The curriculum is cumulative and has no target:
  the initial load is `TaskSpec.initial_train_cap` (5,000) gold rows and `eval_cap` (1,000) eval
  rows, as many as the source has up to those, and rebuilds grow it from there.
  Scores recorded before this change were measured on 800 eval rows and are not strictly comparable
  to ones measured on 1,000.

`STAGNATION_*`, `MAX_EVALS_BEFORE_ESCALATION`, and both size targets are env-overridable
(`SLM_STAGNATION_WINDOW`, `SLM_MAX_EVALS_BEFORE_ESCALATION`, `SLM_CURRICULUM_SIZE`,
`SLM_EVAL_SET_SIZE`, …). **`MAX_STALL_EVALS` was removed on 2026-08-05** — it answered the same
question as the stagnation window and reset on every improvement, so an improvement every 14
evals could defer escalation indefinitely.

**Wall-clock accounting across resumes.** `_wallclock_elapsed_s()` =
`SLM_RUN_ELAPSED_S` (seconds completed in prior requeue segments) + `now − SLM_RUN_START_TS`.
A requeued run inherits its predecessor's elapsed time rather than restarting the budget.

### API failures are fatal

`agent/llm_errors.py::raise_if_fatal` / `is_api_transport_error` walk `type(exc).__mro__`
for a class defined in an `anthropic.*` or `httpx.*` module. **Any** such exception aborts the
run instead of degrading to a fallback, so a run cannot silently limp along on heuristics
after an auth/quota/billing failure. Call sites: `iterate`, `escalate`, `downward_probe`,
`hardware_research`.

JSON/schema validation failures are deliberately **not** fatal — those are recoverable via the
reask and the deterministic fallbacks.

---

## 3. Pre-graph: hardware research and filtering

Runs in the driver (`tests/pipeline/run.py`) before `build_graph`, and is skipped on resume.

### `agent/nodes/cold_start/hardware_research.py::research_device`

1. **Local DB** — fuzzy match against `data/devices.csv` (Kaggle phone specs, refreshed by
   `scripts/refresh_device_db.py`). Free, no API call.
2. **Exa fallback** — web search for spec snippets when the device is not in the DB.
3. **LLM resolve** — `ORCHESTRATOR_MODEL` emits **only device-specific values**:
   `usable_ram_mb`, `storage_budget_mb`, and `reference_chip` (allow-listed against
   `config/android_pool.py::KNOWN_CHIPS`).

Latency, power, and throughput floors are **fixed system constants** in `config/config.py`
(`HW_*`), not device-derived. Output is saved as `device_research.json`.

> `KNOWN_CHIPS` is a bare list of names. It replaced `CHIP_SCALE_FACTORS`, a table of invented
> per-chip decode multipliers — no performance claim is attached to a chip name anywhere.

### `agent/nodes/cold_start/hardware_filter.py::run_hardware_filter`

Two stages, returning variants largest→smallest.

**Stage 1** — `config/android_pool.py::filter_pool(constraints)`. Filters **only on
quantities that are known rather than modelled**:

- `size_mb <= storage_mb` — the on-disk weight file fits
- `size_mb <= memory_mb` — the weight bytes fit in RAM. A lower bound on true peak (which
  adds KV cache and runtime), but a model whose weights alone exceed RAM cannot run.
- if `config/measured_metrics.json` has a real measurement for this
  `(model_id, quant, chip)`, it *additionally* gates on measured peak RAM and tok/s.

**`min_tok_s` is never applied to an unmeasured candidate.** Throughput is not estimated.

**Stage 2** — `hardware_eval/on_device_eval.py::run_on_device_eval` per candidate. At
pre-training screening no GGUF exists yet, so this falls back to the *unmeasured* profile (all
fields `None`) regardless of `SLM_HW_BACKEND`, and nothing is eliminated on a guess. Real
measured gating happens post-convergence in the driver, opt-in via
`SLM_HW_VERIFY_ON_DEVICE=1`.

---

## 4. Cold-start entry nodes

### Node 1 — `task_analysis`

`agent/nodes/cold_start/task_analysis.py::task_analysis_node`

- If `autonomous`, calls `agent/task_planner.py::plan_task` for labels, Exa queries, benchmark,
  stop threshold, and data sizes. A run naming a registered task skips this entirely.
- `_apply_data_targets` clamps the planner's `curriculum_size` / `eval_size` into
  the task's own `initial_train_cap` / `eval_cap`. There is no floor or ceiling to clamp to.
- `SLM_STOP_THRESHOLD` pins both `stop_threshold` **and** `initial_stop_threshold` (the
  immutable floor), taking precedence over the planner.
- Runs the hardware filter, sorts feasible **largest→smallest by `size_mb`**, stores in
  `feasible_models`. Raises if empty.
- **Does not set `selected_model`** — that is Node 1b's job.

**Update 2026-08-19 — the five-task-type table that stood here has been deleted, not corrected.**
It listed `classification`, `NER`, `math_reasoning`, `code_generation` and `generation` with a
supervision format, an eval metric and a synthesis-eligibility flag per *type*. Every one of those
columns is now a field on the task's own spec, because two tasks sharing a type do not share those
answers: `routerbench` and `clinc150` were both `classification` and reported the same metric name
while computing different quantities (B301), and `xlam_bfcl` and `calendar_json` were both
`function_call` and received the same one-line description in every teacher prompt.

The suite is eight named tasks, listed with their metrics in
[§6.5](#the-curated-benchmark-suite-organised-by-what-fine-tuning-is-expected-to-do).
`code_generation` and its APPS/MBPP execution sandbox were **deleted** on 2026-08-18 — those
benchmarks are not in the suite, and a sandbox that nothing runs is a liability rather than a
capability. `task_analysis` no longer validates a `task_type`; `tasks.get_task` raises when the
run's state names a task the registry does not hold, which happens before the graph is built.

The autonomous path is the one place the old vocabulary survives: `agent/task_planner.py` still
asks the orchestrator to classify a free-text description into one of those strings, because a task
that is not in the registry has no spec to read. That path is now nearly unreachable — every task
this project runs is curated — and is discussed in [§12](#12-not-implemented).

### Node 2 — `eval_setup`

`agent/nodes/cold_start/eval_setup.py::eval_setup_node`

Builds the held-out eval set `E` **before any training**, fixed for the whole run. `E` is a
single flat sample of examples (`EvalSet.all`) — there are no pos/neg/boundary slices (removed
2026-08-02; they had no functional effect and, for non-classification families, were a
meaningless random partition).

- **Shared-dataset path** (`SLM_SHARED_DATASET_DIR`) loads a frozen bundle so competing
  strategies see identical data. Requires `manifest.json` + `checksums.sha256`, verifies every
  file hash, checks `bundle_type`/`schema_version`, validates the row schema against
  `required_fields_for_task`, checks manifest counts against the JSONL, and **rejects any
  normalized train/test overlap**.
- **Curated path (every task this project runs)** — `_load_named_benchmark` calls the task's own
  loader, named on its spec, for `max_train = TaskSpec.initial_train_cap` and
  `max_test = TaskSpec.eval_cap`. There is no per-channel fallback branch: a task the registry does
  not know cannot reach this point. It also seeds `state["source_progress"]` with what was consumed
  per mining source, which is what later lets `mine_new_real` tell an exhausted corpus from one we
  only read the first few thousand rows of (B297).

  > **Update 2026-08-19 — where 3,250 came from, and why it is gone.** This used to load
  > `0.65 × curriculum_size_target` gold rows, which on a 5,000-row "target" is 3,250 — a number
  > that appeared in every xlam log with no stated derivation. It was 65% of a target the
  > curriculum was then never allowed to reach, because the only mechanism that could have closed
  > the gap was re-drawing rows it already had. The fraction, the split and the target are all
  > removed: the loader returns as many rows as the source has, up to a flat 5,000, and the
  > curriculum **grows** from there. See [§6.1](#61-curate--build-one-dataset-artifact).

- **Autonomous path** (`task_plan` present, no registered task) — `web_acquire.py::acquire_dataset`
  with `gold_target = 0.65 × curriculum_size_target` and request headroom `×1.15 + 40` to survive
  eval-overlap removal and quality-control drops. This is the only surviving reader of
  `curriculum_size_target` (B305).
- `_eval_target(target)` clamps to a min-30 floor and passes it to `build_eval_set` as the total
  sample size. (Historically `build_eval_set` defaulted to 100, which silently capped every eval set
  at 100 rows regardless of how many test rows were acquired.) On the curated path the target is the
  size of the split the loader already capped, because re-applying a separate `eval_size_target`
  here capped it a second time and undid the cap (B288). Sampling is the task's own
  `TaskSpec.eval_sampling`: `label_balanced` round-robins across classes so `E` spans the full label
  range, `shuffled` is a plain top-N draw. A short eval set is accepted and **logged as short**,
  because fewer rows means more variance and scores that are not comparable across tasks.
- `_author_task_brief` — after the real data is loaded, the orchestrator is shown real rows and
  writes the **task brief** (`agent/task_brief.py`): what the benchmark is, its exact output
  contract, and its likely failure modes. Every synthesis and verification prompt is built from it.
  New 2026-08-19; see [`PROMPTS.md` §1.11](PROMPTS.md#111-task-brief-authoring).
- **Leak firewall (layer 1)** — after *all* acquisition paths, any train row whose normalized
  text matches a test row raises `ValueError`. Logged as
  `official train/test separation: normalized overlap=0`.
- **Difficulty stratification** via `test_agent.label_difficulty` — see [§6.4](#64-test-data-agent).
- Persists `artifacts/eval_set.json` (`counts.total`, the `examples` rows, difficulty buckets).

**Three-way split.** The held-out eval set is *not* the trainer's validation set. The trainer
carves its own 12% validation split from the curriculum (seed 1234) for best-checkpoint early
stopping. `eval_set` is never seen during training.

---

## 5. Model selection strategies

Node 1b. Chosen by `config/config.py::MODEL_SELECTION_STRATEGY`
(`SLM_MODEL_SELECTION_STRATEGY`), resolved by
`model_selection/__init__.py::get_model_selection_node`. All four honour `SLM_FORCE_MODEL`
(must resolve inside the feasible pool, else `RuntimeError`).

| Strategy | Probe cost | Picks | Downward probe reachable? |
|---|---|---|---|
| `smallest_first` **(default)** | none | smallest `size_mb` | ❌ |
| `largest_first` | 1 extra training run | largest `size_mb`; sets `_largest_first_phase="probe"` | ❌ |
| `interpolation` | 3 training runs | closest to the RAM target on a fitted curve | ✅ |
| `orchestrator_choice` | 1 API call | LLM picks | ✅ |

### `smallest_first`

Most resource-conservative. The first `curate→train→evaluate` cycle already uses the
production config — there is no throwaway probe.

### `largest_first`

Establishes feasibility first: if the largest feasible model cannot reach the goal, no smaller
one will. `check_probe_result(state)` is called from `iterate` when the probe clears the
threshold:

- **Probe succeeded** → switch to smallest, reset per-model state,
  `_largest_first_phase = "escalate"`, `next_action = "curate"`. Normal escalation takes over
  from the bottom.
- **Probe succeeded but smallest == largest** → `_largest_first_phase = "done"`.
- **Probe stagnated below goal** → `iterate` sets `next_action = "terminate"` and
  `_largest_first_phase = "done"`. The task is declared infeasible within the budget.

### `interpolation`

Probes up to 3 variants (smallest / middle / largest via `_pick_candidates`), fits F1 against
**log on-disk weight size**, then picks the variant closest to the RAM target that still meets
the goal. Probe config is fixed: 3 epochs, r16/α32/dropout 0, wd 0.01, lr 2e-4, micro-batch 8,
capped at `SLM_PROBE_MAX_EXAMPLES` (300) rows drawn from the **acquired curriculum**
(`train_examples`), falling back to the eval seed only when no train data exists.
`_fit_scaling_curve` averages duplicate footprints and refuses to fit fewer than two unique
x-values.

> The probe uses 3 epochs on real curriculum data specifically because a cheaper probe
> systematically underestimates fully-trained F1, which drags the fitted curve down, pushes the
> goal-crossing out to an enormous parameter count, disqualifies everything, and silently
> defaults to the largest model.

### `orchestrator_choice`

One `ORCHESTRATOR_MODEL` call. The system prompt explicitly optimizes for **resource
efficiency**: the lowest-resource candidate that can *plausibly* reach the goal, not the
highest-benchmark one. Injects `config/model_capabilities.py::capability_sections` plus
`METRIC_COMPARABILITY_CAVEAT`. Deterministic fallback: lowest `size_mb`.

---

## 6. The shared loop

### 6.1 `curate` — build one dataset artifact

`agent/nodes/curate.py::curate_node`

**Early exit:** if `last_intervention != "data_rebuild"`, logs
`SKIP: intervention=… — dataset held fixed` and returns unchanged. A hyperparameter
intervention never rebuilds data.

Requires `eval_set`; raises `RuntimeError` if absent.

**The curriculum is CUMULATIVE (2026-08-19).** Cold start loads the gold rows the loader returned;
every rebuild **adds** to what is already on disk; rows leave only via quality control or the eval
firewall. Nothing re-draws from a pool it has already drawn from, and there is no target size.
[`interventions.md` §4](interventions.md#4-data_rebuild) is the authority for what a rebuild does;
what follows is the node's own order of operations.

1. **First build** (no previous dataset on disk) — the curriculum is `state["train_examples"]`
   tagged `train_anchor`, decontaminated against the frozen eval set. No plan, no strategy: there
   is nothing to add to yet.
2. **Otherwise, resolve the plan** — `state["data_rebuild_plan"]` if present, else
   `fallback_data_rebuild_plan`; then `normalize_data_rebuild_plan`, which **rejects unknown
   fields** rather than dropping them. No dedup or rotation: the plan is used as-is.
3. **Seed** — `_entropy_seed()` (fresh OS entropy per sampler call). **Non-deterministic** by
   design; there is no reproducible per-plan seed.
4. **Execute the one named sub-strategy.** `mine_new_real` walks the ladder (re-read unexhausted
   known sources → web discovery only once all are exhausted → retire after
   `MAX_FAILED_DISCOVERY_ROUNDS`); `surgical_synthesis` generates rows aimed at the failure
   categories costing the most points. Mined rows also join `state["train_examples"]`, so a later
   re-read counts them as consumed.
5. **Eval firewall, per row**, on whatever was added, with each blocked row logged by provenance,
   label, length and a SHA-8 of its normalized text — never the text itself.
6. `_dedupe_into` appends only additions whose normalized text is not already present. **This is
   the only place the curriculum grows.** If it added zero rows, that is logged as an **ERROR**: an
   intervention was chosen, a plan was built, and the mechanism it named could not do the thing it
   exists to do, so this iteration would retrain the previous curriculum exactly.
7. **CoT annotation**, only if the task declares `cot_annotation=True` (gsm8k alone in the current
   suite); skipped under `SLM_CHEAP=1`.
8. `apply_quality_controls` → **eval firewall again** → `MIN_CURRICULUM_ROWS = 500` viability floor,
   which raises rather than warns, because below it the run would burn GPU hours producing a number
   nobody should trust and the cause is always upstream where it can be fixed.
9. Atomic write to `artifacts/dataset_v{N}.jsonl`; every row stamped `_dataset_version`.
10. Record `last_curation`: rows added, novel rows, strategy, target categories, the mining report,
    `source_progress`, `failed_discovery_rounds`, provenance/source composition, label distribution,
    and the per-layer eval-firewall tally.

> **Update 2026-08-19 — three mechanisms described here were removed, and the `target_rows` floor
> with them.** `resample` went on 2026-08-16 because it re-drew rows from the pool the curriculum
> was already built from: it could change *which* gold rows were present but never add information
> (one traced rebuild resampled 3,308 rows of which 122 were novel). The **universal gold FILL** is
> the same defect one level down — because the curriculum was rebuilt to a target size every
> iteration, something had to refill it from the train pool, and with nothing else changed it
> re-selected the identical ~3,235 rows and honestly reported `0 novel` on eight consecutive
> rebuilds of one run. Untargeted **balanced `synthesize`** went because it spent most of the
> teacher budget on rows chosen for class balance rather than for anything the model was getting
> wrong. With no target there is no shortfall to cover, so `allocation_fallbacks` and its
> `rewrite_noop_strategy` entry are gone too — a rebuild that adds nothing is now an ERROR line
> rather than a silently-padded curriculum.

**Quality control is per task, not per channel (B299).** `apply_quality_controls` runs the ordered
steps *this task declared* in `TaskSpec.quality_controls`, from the named units in
`data/quality_controls.py`: `require_fields`, `label_space`, `balance_labels`, `length_outliers`
(>3× the median over TRUSTED rows only, B260), `dedup_surface` (Jaccard ≥ 0.9), `entity_diversity`
(≤3 occurrences of a surface form), and `valid_json_answer`. An empty tuple is a legal, visible
choice; falling through is impossible because there is no branch. A step told to filter on a field
no row carries **says so loudly and skips** instead of passing — that silent pass is what made QC a
no-op for four of the eight tasks.

### 6.2 `train` — one LoRA configuration

`agent/nodes/train.py::train_node`

**Always trains from the base model, never from a prior adapter.** Always produces a LoRA
adapter for on-device adapter-manager deployment.

`_build_config` resolves hyperparameters in this order:

| Situation | Config used |
|---|---|
| `intervention == "hyperparameter"` with LLM `hyperparams` | the LLM's, normalized |
| …and that exact `(dataset, H)` identity was already tried | `_next_untried_config` deterministic replacement |
| `intervention == "hyperparameter"`, no config supplied | `_next_untried_config(best_prior)` — never a silent repeat |
| any other intervention (`data_rebuild`, `rollback`) | **carry-forward best prior config** (unlabelled — holding the best config is the definition of a non-hyperparameter intervention) |
| no prior config at all | `_DEFAULT_CONFIG` (r16, α32, dropout 0, wd 0.01, lr 2e-4, 3 epochs, mb 8, ga 1) |

The carry-forward rule matters: previously every `data_rebuild` collapsed to the hardcoded
r=16 default, so a rebuilt dataset was judged with *weaker* hyperparameters than the current
best, always regressed, and was always rolled back — the data change could never be evaluated
fairly. A config-less `hyperparameter` fallback had the mirror bug: r=16 on the same dataset
produced a bit-identical repeat of the prior iteration.

`_config_diff(best_prior, new)` logs a field-by-field diff
(`Diff vs best prior config: weight_decay 0.01→0.05`) or `unchanged from best prior config`.

Writes `_pending_weights_refs` / `_pending_training_outputs` / `_pending_configs`, keyed by the
config label. `run_training_atomically(final_dir, produce)` makes the checkpoint
write-or-nothing.

### 6.3 `evaluate` — score, reap, log the DAG

`agent/nodes/evaluate.py::evaluate_node`

**Baseline (iteration 1 only).** Runs the base model with no adapter; on a quantized variant it
builds the base GGUF first.

- `QuantizationInfrastructureError` (and, for `generation`, `JudgeInfrastructureError`)
  **re-raise** — a missing toolchain must never be scored as a real result.
- Any other failure records **`baseline_f1 = None`, reported as `n/a`, never `0.0`.**
  Conflating a failed measurement with a genuine zero-shot score of zero once credited
  fine-tuning with a fabricated `+0.8476` improvement in the NER run.

**`judge_mean_0_1` requires a local Qwen3.6 judge, and there is no cloud fallback.**
`eval/judge_client.py::LocalJudgeClient` preflights `JUDGE_ENDPOINT` before it scores anything: the
host must be loopback unless `SLM_JUDGE_ALLOW_REMOTE=1`, and `JUDGE_MODEL` (default
`Qwen/Qwen3.6-35B-A3B`) must name a Qwen3.6 model. Either check failing raises
`JudgeInfrastructureError`, which aborts the eval rather than falling back — a silently substituted
judge changes what every generation score in the run means, and the scores would still be reported
as comparable. Scoring runs `SLM_JUDGE_CONCURRENCY` (16) requests concurrent, and
`SLM_EVAL_JUDGE_OVERLAP_CHUNK` hands each finished generation chunk to the judge while the next
chunk is still generating.

**Quantized accuracy eval.** A GGUF is built and scored when `quant is not None` **and**
(`config.QUANT_ACCURACY_EVAL` — default on — **or** `config.HW_ONDEVICE_BACKEND != "theoretical"`).
With `SLM_QUANT_EVAL=0`, HF/LoRA weights are scored via Unsloth instead (`gguf_path=None`).

GGUF cache key: `sha1(f"{weights_ref}|{quant}")[:12]` →
`artifacts/gguf/<model_id>/<key>/model-<method>.gguf`. **Both parts of the key are
essential** — the same base model can be selected at two tiers as different quant variants, and
the checkpoint path collides across tiers because the iteration counter resets on escalation.
Keying on `weights_ref` alone made a Q8_0 tier silently reuse the earlier Q4_K_M file, so both
tiers scored identically. A cached file is reusable only when its size and SHA-256 match an
atomic sidecar written after a real llama.cpp load.

**The zero-shot baseline competes as a candidate** at iteration 1. If fine-tuning does not beat
the base model, the base model is kept — preventing a shipped fine-tune that is *worse* than
zero-shot, and letting a strong base model converge on its own.

**GGUF retention.** `_reap_gguf` keeps a GGUF only when its iteration set a new best for the
tier; every other one is deleted along with its validation sidecar. Measured reuse was 0/138
(NER) and 2/66 (math) at ~2.6 GB apiece. Earlier new-bests are protected by
`retained_gguf_paths`. A rollback or probe that needs a reaped GGUF simply rebuilds it —
correctness is unaffected.

**Score bookkeeping.** `state["scores"] = list(old) + [current]` — a **new list**, not an
in-place append. `scores` has no LangGraph reducer, so an in-place mutation keeps the same
object identity and is not reliably persisted to the channel; the list froze after ~3 entries,
stagnation never fired, and the run looped forever (B122). Same for `state["dag"]`.

On a new best: `IMPROVEMENT — iteration N — <intervention> — X → Y (Δ=+Z)`. Failures are logged
as `failures=K/N` against the full eval size.

**DAG node** — one per evaluation: `parent_iteration`, `selector`, `model_id`, `quant`,
`weights_ref`, `score`, `best_config`, `intervention`, `failures`, `pruned`, `trained_configs`
(every actually-trained identity, so a losing config cannot be reproposed even when the
baseline won), `evaluation_state` (`last_eval` + `test_report`), and the full triple:

- `pi.D` — dataset `version`, `path`, `plan`, `plan_identity`, `config`, `composition`
- `pi.H` — the complete hyperparameter identity (retired fields included, for replay)
- `pi.S` — `task` (the registry NAME, not a channel), `supervision`,
  `loss_masking="assistant_only"`, `loss_contract_version`

**Two numbers per evaluation, not one (2026-08-19).** Every log line, DAG node and report now
carries a **content** score (`EvalResult.f1`, under the task's own `metric` name) *and* a **format**
score (`EvalResult.format_valid` — the fraction of predictions the scorer could read at all). Seven
of the eight tasks have a real parse step, and gsm8k is the one the guesses got wrong: a model that
reasons correctly and never states a parseable number is a *format* failure, fixable by the prompt
or the answer marker rather than by more data. Reporting only content made those indistinguishable,
which is exactly how B290 hid for two runs — every prediction carried two stray `<think>` tags, so
content collapsed while format told the real story.

| task | what `format_valid` measures | needs parsing? |
|---|---|---|
| `xlam_bfcl`, `calendar_json` | output parses as a JSON list of `{name, arguments}` | yes |
| `ner_bc5cdr` | output parses as a JSON array of `{text, type}` | yes |
| `clinc150`, `routerbench`, `proactive_listening` | an in-vocabulary label was extractable | yes |
| `gsm8k` | a final numeric answer was extractable | yes |
| `dialogsum` | non-empty output; there is no contract to satisfy | no |

Finally writes one `data-curation.md` row via `data/curation_log.py::CurationLog`.

### 6.4 Test-data agent

`agent/nodes/test_agent.py`. **Owns the held-out set and reports only aggregates** — never raw
rows. This is the contamination firewall between evaluation and the decision prompt.

**Difficulty labeling** (`label_difficulty`, at `eval_setup`): run the smallest and largest
unique **base** model IDs zero-shot once, then

- `easy` — both correct
- `medium` — only the large model correct
- `hard` — neither correct

This captures the small→large capability gap that drives escalation.
`_unique_base_endpoints` deduplicates quant siblings by `model_id` (a quant variant is a
deployment choice, not a capacity endpoint) and prefers the BF16 object. Fallback: length-tercile
heuristic — used under `SLM_DIFFICULTY=heuristic`, on any failure, or if the zero-shot split is
degenerate (all three buckets empty).

**`build_test_report`** returns `overall`, `by_difficulty`, `confusion_pairs` (top 8),
`diagnosis`, `suggested_intervention`, `band`.

**Failure categories come from the task's own scorer (2026-08-19).** `TaskSpec.failure_category`
maps one failure record to an actionable category, so a reported category names something the
scorer measured. For a closed-label task the pair is the real `(gold, predicted)` class confusion;
for everything else the category names the error KIND and the "predicted" slot reads `incorrect`. A
task that declares `failure_category=None` reports one honest aggregate rather than a fake taxonomy.

| task | categories it can report |
|---|---|
| `xlam_bfcl`, `calendar_json` | `unparseable_output`, `undeclared_function`, `wrong_function`, `wrong_call_count`, `wrong_arguments` |
| `ner_bc5cdr` | `unparseable_output`, `no_entities_predicted`, `entities_hallucinated`, `wrong_entity_type`, `wrong_span_boundaries` |
| `clinc150`, `routerbench`, `proactive_listening` | the real `(gold, predicted)` class confusions, plus `extraction_failed` vs `wrong_label` |
| `gsm8k` | `empty_output`, `no_numeric_answer`, `wrong_value` |
| `dialogsum` | `empty_output`, `unrelated_output`, `partially_correct` |

> **What this replaced (B296).** Under the `task_type` design every non-classification, non-NER
> failure collapsed to the single literal `gold_verifier → incorrect`, whose count is the failure
> count the orchestrator already had. Across a dozen iterations of the xlam run the orchestrator
> wrote hypotheses like *"the dominant confusion gold_verifier->incorrect (147) essentially
> unchanged since iter2"* — paragraphs of reasoning about a constant, used as evidence. Surgical
> synthesis now aims at these categories, so it is aiming at something measured.

**`diagnose`** turns the per-bucket *pattern* into an action:

- `overall >= threshold` → `none` (converged)
- `easy < 0.6` → `data_rebuild` — missing even simple cases means data quality / label format /
  prompt, not capacity
- `medium` or `hard < 0.6` → `hyperparameter` — an optimization/capacity gap
- below goal with no single failing bucket → `data_rebuild`

### 6.5 `iterate` — the decision node

`agent/nodes/iterate.py::iterate_node`. **Evaluated strictly in this order:**

| # | Condition | Action | API call? |
|---|---|---|---|
| 0 | `not scores` | → `train` | no |
| 1 | `turn_budget` and `(iteration+1)*2 >= turn_budget` | → `terminate` | no |
| 2 | `_wallclock_exceeded()` | → `terminate` | no |
| 3 | `score >= stop_threshold` | `_route_score_at_threshold` (below) | no |
| 4 | `score < threshold` **and** (stagnant or eval-cap reached) | → `escalate`, or `terminate` for a `largest_first` probe | **no** |
| 5 | otherwise | `_llm_iterate` | yes (1, + ≤1 reask) |
| 6 | after a threshold adjustment | re-run `_route_score_at_threshold` | no |
| 7 | `hyperparameter` | → `train` | — |
| 8 | `data_rebuild` | → `curate` | — |

**Step 3 — `_route_score_at_threshold`**, in order:

0. **Bank the convergence, then ask about a stretch goal** — `_maybe_raise_threshold`.
   `_bank_convergence` records the cleared goal into `convergence_banked` unconditionally, then
   (unless `SLM_THRESHOLD_RAISE=0` or `SLM_CHEAP=1`) a dedicated `stage="threshold_raise"` call
   asks the orchestrator whether the goal should be **raised**. A raise returns `False`, so the
   caller treats the score as below-threshold again and the run keeps training against the new
   goal. This runs *before* steps 1–4 because those all treat the goal as settled — probing for a
   smaller model that clears a goal we are about to abandon wastes a tier. See
   "Stretch goals" below.
1. `_largest_first_phase == "probe"` → `check_probe_result` → `curate` if it switched to the
   smallest model.
2. `hw_gating_enabled` and the selected variant fails `check_hardware_constraints` → **not
   accepted as terminal**. Routes deterministically (`escalate` if stagnant, else the
   score-band intervention) *without* an API call, so an auth/quota failure cannot break a run
   that already met its accuracy goal.
3. strategy ∈ {`interpolation`, `orchestrator_choice`}, `not downward_probe_done`, and an
   untried lower tier exists → `downward_probe`.
4. else → `terminate`.

The lower-tier test unions `downward_tiers_tried` with
`downward_probe.tiers_already_explored(state, feasible)`, which derives tiers from
`model_baselines` — every tier the *main escalation ladder* actually trained. Without that
union, the probe could re-select a tier whose true best score is already known and retrain it
with a single fixed config that has no reason to beat the real search.

**Stagnation — ONE guard (2026-08-05), plus an unconditional ceiling:**

Stagnation is measured over `state["eval_history"]`, which is append-only and includes
rolled-back evals. It is NOT measured over `state["scores"]`, which `rollback` pops on every
regression — that older design meant a mostly-regressing run never filled the window and the
check silently never fired. Improvements inside the window do not reset it; only cumulative
gain above `STAGNATION_MIN_DELTA` avoids escalation.

*(historical note below)*

- `_is_stagnant(scores)` — requires ≥50 scores, then `max(window) − window[0] <
  STAGNATION_MIN_DELTA`. Declines and below-origin oscillation count as no progress. A
  `math.isclose` tolerance keeps the exact 0.02 boundary non-stagnant despite binary float
  representation.
- `consecutive_no_improvement >= MAX_STALL_EVALS` — the backstop. `should_rollback` **pops**
  the regressing score, so the stagnation window may never fill during rollback churn;
  `consecutive_no_improvement` is set in `evaluate` and is *not* popped.

**Step 4 spends no API call by design.** Escalation on stagnation is a rule the LLM cannot
override, and plateauing is exactly when a run makes the most `iterate` calls.

**Step 5 — the LLM decision.** One tool-free `ORCHESTRATOR_MODEL` call carrying the **run memory**
(`agent/run_memory.py::build_run_memory`), the test-agent report, tried `(dataset, H)` identities
*including pruned ones*, tried rebuild-plan identities with yield status, source novelty, and
remaining budgets.

Run memory is built from `state["dag"]`, which is append-only and marks discarded attempts
`pruned` rather than deleting them, so rolled-back iterations stay visible even though
`state["scores"]` pops them. It is per-model: `escalate` resets the DAG on a tier change, matching
the escalation policy's scope. Four sections:

| Section | Contents |
|---|---|
| `MOST RECENT ITERATION` | full detail: intervention, delta, `KEPT`/`ROLLED BACK`, per-bucket accuracy, top confusions, full reasoning |
| `WHAT WORKED` | every kept improvement with its delta and full reasoning |
| `FAILED SINCE THE LAST IMPROVEMENT` | aggregated by intervention/sub-strategy, the `MAX_DETAILED_FAILURES` (5) most recent narrated in full, plus an explicit "prefer a DIFFERENT intervention type" conclusion when one dominates |
| `SURGICAL SPEND` | per confusion pair: targeted when, movement, and `EXHAUSTED`/`improving`/`RESOLVED` |

Sub-strategy is shown only for `data_rebuild`: `pi.D.plan` persists across iterations, so labelling
a `hyperparameter` node with the last rebuild's strategy would credit it with a data change it
never made (B239).

Nothing in the block is character-truncated — it is bounded by how many attempts are narrated in
full. The orchestrator's hypothesis is capped only by `HYPOTHESIS_MAX_CHARS` (2000,
`SLM_HYPOTHESIS_MAX_CHARS`), and exceeding it logs a warning rather than cutting silently (B238).

`agent/context_manager.py::compact_trajectory` remains as the fallback for iteration 1, before the
DAG has any nodes.

- A `tool_calls` response is never executed or reflected back — it triggers `_reask_json_only`
  from the original bounded context, so a tool-use block cannot open a side channel.
- `_parse_decision_json` tolerates content-block lists, code fences, and prose wrapping, and
  **always** raises `ValueError` rather than a bare `JSONDecodeError`.
- `_validate_decision_json` is re-applied to the result with `allow_internal=True` as defense in
  depth, so mock/alternate provider paths cannot bypass the contract.
- **Failure ladder:** `raise_if_fatal` → test-agent `suggested_intervention` → score bands.

**Score bands are fallback only**, not enforced boundaries on a valid LLM decision
(`apply_iteration_policy`): `<0.80` → `data_rebuild` / `mine_new_real`; `0.80–0.95` →
`hyperparameter`; `≥0.95` → `data_rebuild` / `surgical_synthesis`. The plan itself is filled in by
`fallback_data_rebuild_plan`, which prefers real rows while any source has them — real data is free
of teacher error, and a gold-only curriculum produced the best result anyone has measured on this
project (BC5CDR, 0.8098).

**Threshold adjustment (lowering).** `new_threshold` is clamped to
`max(value, initial_stop_threshold)` and applied only if it *lowers* the current threshold. The LLM
is told the floor is system-enforced. Legitimate reasons are genuine capacity limits (world knowledge
beyond parametric memory, reasoning chains too long, adversarial OOD) — explicitly not "the task is
hard but learnable." When `stop_threshold` already equals the floor, this path is inert. This is the
below-threshold path and is separate from stretch goals, which only fire on a *met* goal.

### Data rebuild: what an intervention can actually do

`DATA_REBUILD_STRATEGIES = ("mine_new_real", "surgical_synthesis")` — exactly two, and a plan names
one. **[`interventions.md` §4](interventions.md#4-data_rebuild) is the authority**; this is the
summary.

| Sub-strategy | Effect |
|---|---|
| `mine_new_real` | add REAL rows, by a ladder: re-read the task's own unexhausted `mining_sources` for a larger head slice (free — no provider call, no LLM mapping, no schema risk) → web research for a dataset never used, only once every known source is exhausted → **retired** for the run after `MAX_FAILED_DISCOVERY_ROUNDS` (2) rounds that contribute nothing |
| `surgical_synthesis` | add TEACHER-GENERATED rows aimed at the failure categories costing the most points, budgeted proportionally to each category's failure count and bounded to [25, 600] rows. A category already targeted whose count did not fall is EXHAUSTED and skipped |

Both exhaustion events are logged loudly, because "mining added nothing" and "mining had nothing
left to add" are different facts that the run used to report identically. A source is marked
exhausted only when its loader returns *fewer* rows than asked for — the only reliable evidence a
head slice has reached the end of the split — and progress is tracked per source in
`state["source_progress"]`.

**A candidate dataset is filtered PER ROW.** It is no longer rejected wholesale for carrying columns
we do not need, for rows duplicating the curriculum (`_dedupe_into` drops those per row), for rows
overlapping the eval set (the firewall drops those per row), for a single-class slice, or for its
own internal train/test split structure. That last check was a tautology: `_materialize_from_mapping`
sliced train and test from the front of the same split, so it always "found" overlap exactly equal
to `max_test` and rejected every candidate — including both canonical xLAM repositories. A source is
useless only when *nothing* survives per-row filtering.

> **What was removed, and why (2026-08-16 through 2026-08-19).**
> - **`resample`** re-drew rows from the pool the curriculum was already built from, so it could
>   change which gold rows were present but never add information: one traced rebuild resampled
>   3,308 rows of which 122 were novel.
> - **train-pool-gold / the universal gold FILL** was the same defect one level down. Because the
>   curriculum was rebuilt to a target size every iteration, something had to refill it from the
>   train pool; with nothing else changed it re-selected the identical ~3,235 rows and honestly
>   reported `0 novel` on eight consecutive rebuilds of run 38566712. The curriculum is now
>   cumulative, so nothing needs to refill it.
> - **synth-fill** padded to `target_rows` with generated rows. The target was itself a heuristic,
>   BC5CDR's best-in-project 0.8098 came from a gold-only curriculum ~7,100 rows below target, and
>   one traced rebuild spent 749 generations to keep 64 post-QC rows.
> - **Untargeted balanced `synthesize`** spent most of the teacher budget on rows chosen for class
>   balance rather than for anything the model was getting wrong. Targeted generation is strictly
>   better use of the same calls, so surgical synthesis is now the whole of synthesis.
>
> A plan naming any of them is now **rejected**, not silently redirected. A plan written against the
> wrong contract means the orchestrator believes it asked for something it did not.

**Synthesis coverage is derived from the task, not chosen by a channel.** `synthesize_examples`
produces one of two row shapes: for a task with a **closed label space** (`clinc150`, `routerbench`,
`proactive_listening`) a new *input* for an existing class, so the row inherits a real anchor's
label and only the phrasing can be wrong; for an **open-ended target** (everything else) a whole new
*(input, answer)* pair, which is why those are gated twice. Under the old dispatch table
`function_call` matched neither branch and fell through to a bare `return []`, so six rebuilds on
xlam announced 250–500 rows against a healthy teacher endpoint and produced zero, silently — and the
exact verifiers written for that path had never executed in production (B291). The table it lived in
no longer exists.

**`MIN_CURRICULUM_ROWS = 500`** (`SLM_MIN_CURRICULUM_ROWS`) is the viability floor: curate RAISES
below it, because the cause is always upstream (loader or QC) where it can be fixed.

### Model selection strategies

`SLM_MODEL_SELECTION_STRATEGY` ∈ `smallest_first` (default), `largest_first`, `interpolation`,
`orchestrator_choice`, **`single_model`**.

`single_model` is the naive-baseline ablation control: the orchestrator picks one model and the run
stays on it — no escalation on failure, no downward regression on success. Everything else
(hyperparameter search, data rebuilds, rollback, accuracy goal, stretch goals) runs unchanged, so
`single_model` vs `smallest_first` isolates what the model ladder buys. Enforced at all three ladder
gates through `config.model_ladder_enabled()`.

### Eval-set size is capped per task at 1,000

The cap is `TaskSpec.eval_cap`, declared by each task and currently **1,000 for all eight**
(`SLM_EVAL_SIZE_CAP` overrides every task). It replaced a module-level `_EVAL_SIZE_CAP` on
2026-08-19, which in turn replaced the old `eval_size_target` default of 800 that was applied
per-benchmark regardless of how much held-out data existed (B282). The cap is now a per-task field
because it is a per-task trade-off, and because leaving it in `eval_setup` meant a task could not
state its own answer.

Why 1,000 and not the whole split: the eval runs on EVERY iteration, so an unbounded split makes each
loop turn proportionally slower — RouterBench's full held-out set is 7,267 rows, 9x the old per-iteration
cost, for a variance improvement that flattens out long before that. At n=1,000 the standard error on a
proportion is about 1.5 percentage points, comfortably below the score differences this project resolves.

Every task **asks** for 1,000; the loader returns as many as its split has, and a short eval set is
accepted and logged as short. `dialogsum` (renamed from `dialogsum_samsum` on 2026-08-19) and
`calendar_json` have historically returned fewer — 667 and 478 rows respectively — because their
whole held-out splits are smaller. **Scores measured before the cap moved to 1,000 used 800 rows and
are not strictly comparable.**

### Synthetic rows pass TWO gates, exact first

```
generate (5-shot) → PROGRAMMATIC verify (exact) → MODEL verify (judgement) → quality control
```

`data/synth_verifiers.py` supplies the exact gate for the tasks where correctness is decidable by
computation. The verifier is named on the task's own spec as `TaskSpec.synth_verifier`;
`programmatic_verifier_for` and its side registry are gone.

| task | exact verifier | what it proves for free |
|---|---|---|
| `xlam_bfcl` | `verify_function_call_row` | JSON parses; the call targets a **declared** tool; arguments exist in its schema; required parameters present |
| `calendar_json` | `verify_calendar_row` | the above, plus datetimes parse, `end` follows `start`, an unstated duration is 60 minutes, and the event resolves near the request's own reference instant |
| `ner_bc5cdr` | `verify_ner_row` *(new 2026-08-19)* | every span appears **verbatim** in the row's own text, with no duplicates |
| the other five | `None` | only the teacher's judgement is available, and that is logged as such rather than implied |

The exact gate runs FIRST because it is free, cannot be fooled, and a row it rejects should never
cost a teacher call. `None` is the honest answer for `dialogsum` (summary quality is not decidable
by computation) and for the closed-label tasks (rows inherit a real anchor's label, so there is no
answer to verify) — those get the teacher label-verification pass instead.

**NER span synthesis looked unverifiable and is not**, which is why BC5CDR moved from "gold-only by
design" to having a verifier. The substring check catches the dominant teacher error — a plausible
entity that was never written down. What it *cannot* catch is a **missed** entity, so the teacher
pass still runs and that limit is stated in the code rather than left implicit.

Generated rows have `tools` and `_instruction` **pinned from the anchor**: the tool signature is the
constraint rather than something being invented, and without it a row cannot be schema-checked at all.

**All synthesis and verification prompts are 5-shot** (`SLM_SYNTH_SHOTS`) and are built from the
**task brief** — the orchestrator's own description of the benchmark, authored once at cold start
from real rows (`agent/task_brief.py`). The brief replaced a one-line description keyed by task
*type*, under which `xlam_bfcl` and `calendar_json` were described identically, omitting every
convention that makes a calendar row correct — so a verifier judging against it was judging its own
guess (B269, calendar synthesis at 0.2176). Worked examples shown to the teacher are always **real
rows** sampled from the task's own training split, never orchestrator-invented: a wrong example is
worse than none. The teacher scores 0.1131 span-F1 zero-shot on BC5CDR NER and 0.7190 with five
demonstrations (B276/B281).

### Per-task preflight

`scripts/preflight_tasks.py` checks every registered task without a GPU: data loads with the required
fields, gold scores 1.0 through the real scorer, a DEGENERATE answer scores ~0 (a metric that rewards
collapse cannot detect it), `_training_turn` produces a non-empty prompt/target byte-identical to the
eval prompt, and there is no train/eval overlap. Run it before submitting anything — it is the check
that would have caught the `function_call` trainer crash and it takes ~2 minutes.

### The curated benchmark suite, organised by what fine-tuning is expected to do

The registry in `tasks/` is the single source of truth, keyed by `SLM_BENCHMARK_TASK`, which equals
both the module name and `TaskSpec.name`. `coedit` and `medqa` were removed 2026-08-15 (neither ever
produced a run); `proactive_listening` was added.

| Category — what FT should buy | Task | `family` (descriptive) | Metric |
|---|---|---|---|
| **in-distribution** — base model can already do it; FT buys format discipline. Good baseline, small delta. | `gsm8k` | generation | `exact_match` |
| | `dialogsum` | generation | `judge_mean_0_1` |
| **format-bound** — knows the content, cannot produce the contract. Near-zero baseline, large delta. | `xlam_bfcl` | structured_output | `ast_arg_match` |
| | `calendar_json` | structured_output | `ast_arg_match` |
| | `ner_bc5cdr` | extraction | `span_f1` |
| **out-of-distribution** — label is not a property of the input's surface form. Low/noisy baseline, delta bounded by label noise. | `clinc150` | classification | `macro_f1` (151-way) |
| | `routerbench` | classification | `minority_f1` |
| | `proactive_listening` | classification | `minority_f1` |

> **Update 2026-08-19 — three things changed in this table.** `NAMED_BENCHMARK_TASK_TYPES` (and the
> parallel loader dict a test kept in lockstep with it) were replaced by the registry. The
> `task_type` column became `family`, which is **descriptive only** — a report label and a
> model-selection hint, never a dispatch key. **`gsm8k` was promoted** from the autonomous path to a
> first-class task with its own loader (`data/loaders/gsm8k.py`), making the suite eight, and
> `dialogsum_samsum` was renamed `dialogsum`. The metric column now states the string the scorer
> actually returns: `routerbench` and `proactive_listening` reported a minority-class F1 under the
> name `macro_f1` for their entire history (B301).

Evidence for the categories is in `Evan's Notes/08-16-extraction-collapse-verdict.md` §9. Three
caveats that matter when reading results:

- **`clinc150` behaves like an in-distribution control, not an OOD task** — its label *is* the
  utterance's meaning, so the teacher scores 0.8919 and fine-tuning adds +0.003. The predictive
  variable is not in/out of distribution but **whether the label is inferable from surface form**.
- **`dialogsum` showed +0.0000 over 15 iterations at tier 3** (baseline 0.7157 = best FT). For
  a genuinely in-distribution task the honest output of the loop may be "use the base model".
- **`gsm8k` is genuinely format-sensitive despite sitting in the in-distribution row.** Its scorer
  regexes a final number out of the working, so a model that reasons correctly and never states a
  parseable number scores zero on content and is only visible in `format_valid`.

### The label vocabulary is pinned at eval_setup and closed thereafter

`_pin_label_space` writes `state["task_label_space"]` from the frozen eval set immediately after it is
built — `{"labels": [...], "definitions": {...}, "benchmark": …, "source": "frozen_eval_set"}`. The eval
set is the authority because it is what the score is computed against, so a class absent from it cannot
be scored. Every later stage may only DROP rows outside the vocabulary, never extend it, and **no LLM
may introduce a label**. Enforcement points and the strict-vs-inferred rule are documented in
`DATA_CURATION_AND_CAPS.md`; the history is in `data/label_space.py` (B259).

`definitions` additionally tells the synthesis and verification prompts what each class MEANS, so the
teacher judges the task rather than the label's wording (B267).

### Stretch goals — raising the accuracy target

The goal is the Qwen-3.6 teacher's zero-shot score floored at 0.80, so when the teacher scores below
the floor the goal reflects what we refused to go below rather than what the task permits. BC5CDR
cleared a floored 0.8000 in five iterations and 81 minutes and stopped with hours of budget left.

`_maybe_raise_threshold` (`agent/nodes/iterate.py`) runs first in `_route_score_at_threshold`:

1. **Bank** — `convergence_banked = {threshold, score, iteration, selector}`, keeping the highest
   goal ever cleared. Unconditional, including when raising is disabled or declined.
2. **Ask** — a small dedicated call, not a field on the intervention decision (that prompt is only
   built for below-threshold scores). The orchestrator is shown the current goal, its **provenance**
   (whether the floor or the teacher set it), the score and margin, iterations used, the last 12
   scores, turns used/remaining, the ceiling, the minimum step, and previous raises. It returns
   `{"raise_goal": true, "new_threshold": …, "reason": …}` or `{"raise_goal": false, "reason": …}`.
3. **Ratchet** — the proposal must exceed `max_stop_threshold` (the run's high-water mark) by at
   least `_THRESHOLD_RAISE_MIN_STEP = 0.005`, and is clamped to `THRESHOLD_CEILING = 0.99`. Clamping
   against the high-water mark rather than the current goal is what prevents a lower-then-raise cycle
   from reusing the same band indefinitely.
4. **Continue** — returns `False`; the run trains on against the new goal.

**Termination.** Raises are strictly increasing with a fixed minimum step under a hard ceiling, so a
run performs at most `(0.99 − initial) / 0.005` of them; each also requires clearing the previous
goal, and at most one is asked per iteration (`_threshold_raise_asked_iteration`). Every other budget
(turn, wall clock, graph steps, stagnation, eval cap) is unchanged.

**A missed stretch goal is still a success.** `run.py` treats a banked convergence as convergence, so
raising can never turn an already-successful run into a reported failure. When the stretch goal is
missed it prints `⚠ CONVERGED AT THE ORIGINAL GOAL, stretch goal missed: …`.

**Controls.** `SLM_THRESHOLD_RAISE=0` disables; `SLM_CHEAP=1` skips. Default **off under pytest**
(`conftest.py`), because every convergence assertion would otherwise make a live call.
`scores.json` carries `convergence_banked` and `threshold_raises`.

### Goal provenance in the run summary

A floored goal and a teacher-set goal print the same number, so BC5CDR's `threshold 0.8000` hid a
teacher score of 0.0999. `threshold_calibration` now records `floored: bool`, and
`agent.threshold.describe_threshold_provenance()` renders it. The summary always prints:

```
  goal source: floor 0.80 OVERRODE the Qwen-3.6 teacher, which scored only 0.0999 span_f1 …
  teacher    : Qwen-3.6 zero-shot 0.0999 span_f1  (no fine-tuning; …)
```

`scores.json` carries `initial_stop_threshold` and the full `threshold_calibration` block.

**`llm_iterate_decision` is always overwritten**, including with `None` on failure. A stale
non-`None` decision would let `train._build_config` reuse the previous iteration's
hyperparameters while bypassing the carry-forward / untried-identity logic.

### 6.6 `rollback`

`agent/nodes/rollback.py`. Triggered by `should_rollback`:
`len(scores) >= 2 and scores[-1] < scores[-2]`. Cold start uses this simple rule; the dual-gate
is production-only.

1. Pop the regressing score; `last_intervention = "rollback"`.
2. Mark the newest DAG node `pruned = True`.
3. Restore from the best **non-pruned** node: `best_weights_ref`, `best_score`,
   `current_dataset_path`, `dataset_version`, `last_curation`, `data_rebuild_plan`,
   `data_rebuild_plan_identity`, `last_eval`, `test_report`, and `_pending_configs` /
   `_pending_weights_refs`.
4. Raises `RuntimeError` if the winning node has no dataset path, or if **all** nodes are pruned
   (an inconsistent state, not a recoverable one).
5. Routes to `iterate`.

### 6.7 `escalate`

`agent/nodes/escalate.py::escalate_node`

1. Finds the **nearest higher non-empty tier** among `filter_pool(constraints)`. Tiers are size
   buckets and can be empty for a given budget, so gaps are stepped over rather than terminating
   on the first empty one. No higher tier → `terminate`.
2. `_llm_choose_model(direction="up")` picks within that tier, injecting `capability_sections`,
   `METRIC_COMPARABILITY_CAVEAT`, and a per-task-type `_BENCHMARK_HINT` (which steers away from
   defaulting to GSM8K on a non-math task). Deterministic fallback:
   `min(candidates, key=size_mb)`.
3. Appends to `escalation_history`: selector, model_id, quant, tier, `baseline_f1`
   (**`None`, never `0.0`**), `best_score`, `iterations`, `scores`, and that model's **full
   DAG** — the DAG resets on escalation, so this is the only record of earlier tiers.
4. `clear_inference_cache()` frees the previous model's VRAM so it does not sit resident
   competing with the next (usually larger) model.
5. Resets per-model state: `scores=[]`, `dag=[]`, `iteration=0`, `best_score=0.0`,
   `best_weights_ref=None`, `last_eval=None`, `consecutive_no_improvement=0`, rebuild plan, and
   all downward-probe state. `lifetime_best_score` retains the max across tiers.
6. `next_action = "curate"`. **The dataset is carried forward** until curate writes the new
   version.

Because tiers are quant-specific, escalation can move to a *different quant variant of the same
base model*.

### 6.8 `downward_probe`

`agent/nodes/downward_probe.py::downward_probe_step_node`. Reachable only from
`_route_score_at_threshold` under `interpolation` / `orchestrator_choice`. Goal: find the
**smallest** feasible model that still clears the goal, to minimize on-device resource use.

Two durable phases, one graph commit each:

- **Plan** — record `origin` (once), compute untried lower tiers, pick a candidate with
  `_llm_choose_model(direction="down")`, write `downward_probe_pending`, return. The choice is
  checkpointed *before* training.

  > The separate `_should_reexplore_downward` gate — one extra LLM call answering "is another
  > lower tier worth probing?", with a `margin >= 0.03` heuristic fallback — **no longer exists**.
  > It parsed its answer with `bool(...)`, so a JSON string `"false"` evaluated true and the gate
  > could not decline (B202), and it was never given the probe's cost or the candidate's RAM saving,
  > so it was not actually the cost-aware decision it was written to be. The probe now simply runs
  > while an untried lower tier exists and stops at the first tier that misses the goal.
- **Execute** — revalidate that the pending selector is still feasible and that its `H` matches
  the fixed contract, then train + eval and append to `history["attempts"]`.
  - `f1 >= threshold` → **adopt**: swap `selected_model`, `best_weights_ref`, `best_score`,
    `converged_model_ref`, and continue to the next lower tier.
  - `f1 < threshold` → stop the downward search and keep the current model.

The probe config is **fixed and serialized in three places** (`downward_probe_pending["H"]`,
`history["fixed_H"]`, and each attempt) so a resume cannot silently change it:

```
DOWNWARD_PROBE_H = r16, α32, dropout 0, weight_decay 0.01, lr 2e-4,
                   3 epochs, micro_batch 8, grad_accum 1, effective_batch 8
```

`skip_optional_error(stage, exc)` treats the whole probe as **optional**: any failure in the
reexploration gate, model chooser, pending validation, or training records a
`history["termination"]` reason and preserves the converged model. **The probe always terminates
the graph.**

---

## 7. Interventions

Exactly **two** intervention types, as a discriminated union — merging their payloads is
rejected.

### 7.1 `data_rebuild`

Routes to `curate`; hyperparameters are held at the current best so the data change is isolated
and comparable.

`agent/data_rebuild.py` defines **two** orchestrator-selectable sub-strategies
(`DATA_REBUILD_STRATEGIES = ("mine_new_real", "surgical_synthesis")`), chosen singly with no
task-type or score gating. What each one does is in
[§6.5](#data-rebuild-what-an-intervention-can-actually-do) and, in full, in
[`interventions.md`](interventions.md).

**Plan schema version 3** (`normalize_data_rebuild_plan`), rewritten 2026-08-19:

| Field | Domain |
|---|---|
| `schema_version` | must be `3` |
| `strategy` | `mine_new_real` \| `surgical_synthesis` |
| `rows` | [50, 2000], clamped. How many rows this rebuild may ADD |
| `target_categories` | ≤8 entries of `{"category": <name>, "count": <observed failures>}`, drawn from the task's own failure taxonomy |
| `pattern_hint` | free text, ≤1200 chars |

**Unknown fields are rejected, not ignored.** A plan that sets something we removed is a plan
written against the wrong contract, and silently dropping it would let the orchestrator believe it
had asked for something. So `target_rows`, `resample_fraction`, `new_real_rows`, `synth_rows`,
`max_acquire_rounds`, `difficulty_buckets` and `confusion_pairs` are all now hard validation errors
rather than fields that get trimmed. `difficulty_buckets` went because nothing ever read it back to
sample rows with; per-difficulty targeting is expressed through failure categories instead, which is
more direct. Validation also rejects any raw held-out text anywhere in the plan.

**One rewrite survives, and it is not silent.** A `mine_new_real` plan is rewritten to
`surgical_synthesis` when `mining_available_for_state` is False — every known source exhausted AND
web research already out of its allowance — because the alternative is spending a full train+eval
cycle on an intervention that provably cannot add a row. The orchestrator is told mining is retired
in its own prompt before it writes the plan.

**Non-determinism.** There is no plan-identity dedup, no untried-plan rotation, and no
`DataRebuildPlanSpaceExhausted`. Sampling and synthesis draw fresh OS entropy each call, so repeated
rebuilds genuinely vary. The orchestrator freely re-picks a strategy each turn; escalation on
no-improvement (`STAGNATION_WINDOW` = 15, plus the `MAX_EVALS_BEFORE_ESCALATION` = 30 ceiling) is the
sole stuck-run backstop, plus the wall-clock guard. Exact checkpoint-resume reproducibility is
intentionally dropped.

> **The paid-acquisition ledger is gone (2026-08-19.)** Mining used to reserve a round against a
> locked append-only ledger under the stable run directory, bounded by
> `MAX_PAID_ACQUIRE_ROUNDS_PER_RUN` (9) and `MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN` (3). The ledger did
> what it was built to do — a crash loop could not re-spend budget — but it metered the wrong thing:
> the same counter gated **re-reading a corpus already in the local cache**, which costs nothing at
> all, and on xlam that meant the loop could not reach ~57,000 unused rows it already had while
> paying Exa to rediscover mirrors of them (B297). `data/acquisition_budget.py`,
> `plan_budget_identity` and `reserve_paid_acquisition` are no longer on any code path. What is
> metered now is **failure, not spend**: `MAX_FAILED_DISCOVERY_ROUNDS` (2) fruitless discovery
> rounds retire mining for the run.

**Fallback plan** (`fallback_data_rebuild_plan`) is now deterministic in its strategy choice:
`mine_new_real` while any source still has rows, `surgical_synthesis` otherwise, aimed at whatever
the last test report says is failing most. It prefers real rows because real data is free of teacher
error and a gold-only curriculum produced the best result anyone has measured on this project
(BC5CDR, 0.8098). This replaced a signal-weighted random draw over three strategies, which in turn
replaced a keyword/rotation chooser that sent every NER fallback to `resample_existing`.

**A stray `hyperparams` block on a `data_rebuild` is stripped, not rejected.** The rule being
enforced is "a data rebuild must not also change hyperparameters"; dropping the field enforces
it exactly. Raising did not — it discarded the orchestrator's entire data plan and fell through
to a heuristic that *also* held hyperparameters fixed, so the invariant was never what was at
stake. Claude attaches `hyperparams` to essentially every `data_rebuild` it proposes, so the NER
run rejected 65 of them and executed **zero** orchestrator-authored data plans across 142
iterations while its logs still credited each rebuild to the orchestrator.

### 7.2 `hyperparameter`

Routes straight to `train`; the dataset is held fixed. See [§8](#8-hyperparameter-contract).

### 7.3 Contamination firewalls

Four independent layers, all on **normalized** text
(`data/loaders/dataset_integrity.py::normalize_text`):

| Layer | Where | Effect |
|---|---|---|
| Source split separation | `eval_setup` | Raises on any train/test overlap after acquisition |
| Candidate filtering | `curate` — `train_examples`, mined, synthesized, replay, and final | Drops overlapping rows, logs the count |
| Decision-prompt rejection | `iterate::_reject_eval_text_strings` | Recursively rejects any held-out text in any decision string (≥12 chars normalized, substring or exact) |
| Reask sanitization | `iterate::_sanitize_reask_error` | Redacts eval text from validator error messages before replaying them |

Gold and synthesis anchors come only from decontaminated `train_examples`. Held-out rows
influence the loop **only** through aggregate difficulty and confusion counts.

---

## 8. Hyperparameter contract

**Exactly five fields are tunable** by the orchestrator
(`iterate.py::_validate_decision_json`, `training/hparams.py`):

| Field | Domain | Default |
|---|---|---|
| `lora_rank` | {4, 8, 16, 32, 64} | 16 |
| `alpha_ratio` | {1, 2, 4} — alpha derived as `rank × ratio` | 2 |
| `weight_decay` | {0.0, 0.01, 0.05, 0.1} | 0.01 |
| `learning_rate` | [1e-5, 5e-4] | 2e-4 |
| `nr_epochs` | [1, 8] | 3 |

**Retired fields** are accepted from checkpoint/DAG replay but rejected with an actionable
message if the LLM emits them:

| Retired | Message |
|---|---|
| `lora_alpha` | use `alpha_ratio` (alpha = rank × ratio) |
| `lora_dropout` | removed; regularize with `weight_decay` |
| `micro_batch_size` | derived by the trainer to fit VRAM |
| `gradient_accumulation_steps` | derived by the trainer to fit VRAM |
| `effective_batch_size` | fixed; not part of the search |
| `batch_size` | legacy alias; not part of the search |

**Why batch shape was removed:** only the *effective* batch changes what the model learns, while
the micro/accum split is a VRAM-fitting decision the trainer makes better from the actual
device. In the math run the orchestrator burned iterations 14/19/24/39/52 re-shuffling that
split with every learning parameter held fixed, measuring a ±0.01 spread that is pure
run-to-run noise.

**Why dropout was removed:** it duplicates `weight_decay` as a regularizer and never produced a
new best in either run, while `weight_decay` produced the single largest hyperparameter gain
(+0.033). Dropout 0 is also Unsloth's optimized path; nonzero dropout can materially increase
training time.

**Type discipline.** `lora_rank` and `nr_epochs` must have **integer** JSON type; `alpha_ratio`,
`weight_decay`, `learning_rate` must be numeric. Booleans are rejected everywhere.
Out-of-domain numerics are snapped to the nearest valid rung (lower wins ties); non-numerics are
rejected. `TrainingConfig` re-enforces every range strictly at runtime.

**Repeat prevention.** `_tried_hparam_configs` / `_tried_trial_identities` collect complete
`(dataset_version, dataset_path, H…)` identities from the DAG — including `trained_configs` on
nodes where the baseline won, and including **pruned** nodes. Training is
(near-)deterministic, so an exact repeat cannot produce new information and is forbidden. The
same `H` **is** allowed after a data rebuild, because the dataset identity changed.
`deterministic_neighbor_configs` supplies the replacement ladder over the same five axes.

### Fixed training choices (not interventions)

`training/lora_trainer.py`:

- **Max sequence length is per TASK** (2026-08-19), from `training.slm_helpers.task_max_seq_length`
  → `TaskSpec.max_seq_length`: 1024 for `clinc150` and `routerbench`, 2048 for `gsm8k`,
  `dialogsum`, `ner_bc5cdr`, `xlam_bfcl`, `calendar_json` and `proactive_listening`.
  `SLM_MAX_SEQ_LENGTH` overrides every task; the training side clamps to [128, 32768]. Training and
  eval read the same field, so a row cannot fit one side and be truncated on the other. These are
  ceilings with headroom, not tight fits — measured rows are p50≈228 / p99≈601 tokens. Output
  reserves are `TaskSpec.max_new_tokens`: 50 for the classification tasks, 256 for the two
  function-calling tasks, 512 for `gsm8k` / `dialogsum` / `ner_bc5cdr`. `TaskSpec.__post_init__`
  rejects a spec whose reserve leaves no prompt budget, so the pair is validated at import.
  There is no longer a `.get(..., 4096)` default for an unrecognised type: an unregistered task
  raises in `get_task` long before this.
- **Eval batch size** 32 short-output / 16 long-output (`SLM_EVAL_BATCH_SIZE`). A CUDA OOM
  halves the active batch and retries in place.
- **Nothing truncates.** Training (`_validate_training_sequence_lengths`), HF inference, and
  GGUF inference all tokenize with `truncation=False` and **raise** on an over-length row,
  rather than silently cutting a prompt or gold completion. `_log_sequence_length_report`
  emits the min/p50/p95/p99/max token distribution plus a warning for any row at or above
  90% of the window (`SLM_LENGTH_WARN_FRACTION`), so approaching the limit is visible before
  it becomes a crash.
- Base loaded in 4-bit whenever rank is not `None`.
- Text LoRA targets q/k/v/o + gate/up/down; multimodal tunes language/attention/MLP and freezes
  vision. Bias `none`.
- **Completion-only loss.** Prompt tokens are `-100`; assistant labels, NER JSON,
  generation/CoT, and code targets are trained. Text-only multimodal training applies the same
  explicit mask through the inner tokenizer. The contract version is stamped into `pi.S`.
- Early-stop split 12% when ≥60 formatted examples exist; best validation-loss checkpoint
  loaded with patience 3.
- On an early-stop/checkpoint failure the failed model+trainer are **discarded**, a fresh
  base+LoRA stack is reloaded, rows are rebuilt with the fresh tokenizer, and the **complete
  original dataset** is retrained without validation for the originally requested epoch count.
  It never continues partial weights or the reduced split.
- `_purge_trainer_checkpoints(output_dir)` runs after the final checkpoint save. It previously
  existed only on the fallback-retry path, which leaked 93 GB.

---

## 9. Model pool, tiers, and variant identity

`config/android_pool.py`.

### Variant identity

Every pool entry is an **independent quant variant** with the stable selector
`model_id@bf16|Q8_0|Q4_K_M`. Selection prompts, `SLM_FORCE_MODEL`, DAG nodes,
`model_baselines`, `escalation_history`, and `converged_model_ref` all preserve it. A legacy
bare `model_id` resolves to the **lowest-size feasible sibling**, not list order
(`resolve_model_selector`).

### Tiers are on-disk size buckets

`_size_tier(size_mb)`:

| Tier | On-disk size |
|---|---|
| 0 | < 750 MB |
| 1 | 750–1499 MB |
| 2 | 1500–2499 MB |
| 3 | ≥ 2500 MB |

> **Changed from the old audit.** This previously bucketed a *modelled* peak-inference-RAM
> figure (`_ram_tier`). Tier is used only to order and group candidates by rough scale, which
> real weight size serves equally well without inventing a runtime number. Q4_K_M, Q8_0, and
> BF16 variants of the same base model can still occupy different tiers, and tiers are **not**
> parameter-count or base-capability classes.

### No modelled metrics anywhere

Removed pool-wide: `CHIP_SCALE_FACTORS`, `tok_s_snapdragon_*`, `peak_memory_mb`,
`tok_s_for_chip()`, `_SPEED_FACTOR`. What remains:

- `size_mb` — a measured value when `config/measured_metrics.json` records one for this
  `(model_id, quant)`, else bytes-per-parameter arithmetic (BF16 2.0, Q8_0 1.0,
  Q4_K_M 0.55 GB per 1B params). **As of the 2026-07-29 sweep (job 37905779) all 18 pool
  variants are measured**, and the arithmetic was confirmed accurate to +4.9%/-1.1% with
  **zero tier changes** — so the fallback is trustworthy for a future pool addition.
- `ModelSpec.measured(chip)` / `measured_metrics_for(model_id, quant, chip)` — returns a
  recorded measurement or `None`. **Exact-match only**: no interpolation across chips or
  quants. An unmeasured combination reads as *unknown*, which is the honest answer, rather than
  a number borrowed from a different configuration.
- `hardware_eval/measure_model.py` — the CLI that writes real measurements with provenance.
- `hardware_eval/on_device_eval.py::unmeasured_profile` returns all-`None` (renamed from
  `theoretical_profile`); `measure_llama_cpp` measures real peak RSS via
  `getrusage(RUSAGE_CHILDREN)` and reports `avg_watts=None` rather than a size proxy.
- `training/quantize.py::theoretical_hardware_profile` keeps its name for call-site
  compatibility but returns only `size_mb`, `size_source` (`measured` vs
  `bytes-per-parameter arithmetic`), `tier`, and any recorded `measured` block.

### Capability evidence

- `config/model_capabilities.md` — the offline, human-readable, sourced capability document.
- `config/model_capabilities.py::capability_sections(model_ids)` — caches it and returns the
  shared caveat plus only the requested sections.
- `ModelSpec` carries `CapabilityMeasurement(metric, value, artifact, mode, protocol, source)`
  records. Missing measurements are `None` / `not reported`, **never numeric zero**. MMLU,
  MMLU-Pro, and MMLU-Redux stay separately named. Measurements rank only when metric, mode, and
  protocol are all identical.
- `filter_pool` sorts by `(tier, size_mb)` only. `filter_pool_by_task` uses a benchmark within a
  tier **only** when every candidate has a present measurement with an identical
  metric/mode/protocol key; mixed metrics, differing protocols, or any missing value falls back
  to deterministic resource ordering.
- `METRIC_COMPARABILITY_CAVEAT` is injected into every selection prompt.

### Capability injection points

| Stage | Symbol |
|---|---|
| Initial LLM selection | `model_selection/orchestrator_choice.py::orchestrator_choice_node` |
| Upward escalation | `escalate.py::_llm_choose_model` (`direction="up"`) |
| Downward selection | `downward_probe.py` → the same `_llm_choose_model` (`direction="down"`) |
| Autonomous planning | `task_planner.py::_pool_summary` — renders explicit metric names, modes, and source URLs under the same contract |

### Evaluation artifact and mode

- Q4/Q8 zero-shot baselines build/reuse the selected base GGUF before scoring; interpolation
  probes and downward probes use the same cache-aware helper.
- HF training and inference pass `enable_thinking=False` through the tokenizer template.
- llama-cpp-python 0.3.34's `create_chat_completion` has no `chat_template_kwargs`, so hybrid
  Qwen3/Qwen3.5 GGUF uses the verified ChatML empty-think prefix. Non-thinking-only
  Qwen3-4B-Instruct-2507 uses its plain assistant prefix. **Unknown templates fail loudly**
  rather than silently changing mode.
- GGUF execution requires the compute-node runtime: the login node's `libstdc++` lacks
  `GLIBCXX_3.4.29`. Activate `.venv_gpu` (see `scripts/setup_gpu_env.sh`).

---

## 11. Durability and checkpoint/requeue

Driver: `tests/pipeline/run.py`. Machinery: `agent/checkpoint.py`, `agent/state_codec.py`.

### Long-run checkpoint/requeue contract

The checkpoint/requeue-enabled weeklong jobs use Slurm `--time=7-00:00:00`,
`--signal=B:USR1@7200`, and `--requeue`. The
batch shell forwards `USR1` to the pipeline, waits for its atomic JSON **and** SQLite
checkpoint, verifies durable resume state, and only then calls `scontrol requeue` with the same
stable run directory. This produces segment rollover with continuous progress rather than a
terminal seven-day run.

Those scripts set `SLM_MAX_WALLCLOCK_S=0`. The application guard is **aggregate across resume
segments**, so a nonzero value such as the former 6d20h setting would silently terminate before
Slurm's 6d22h signal and prevent the safe requeue path. Non-requeue runs may still set a nonzero
guard when graceful termination before their scheduler limit is wanted.

`run-manifest.json` stores both the resume compatibility fingerprint and a human-readable
`effective_config` snapshot. Resume recomputes every resume-sensitive setting after the runner
loads `.env`, then reports the specific stored/current values that drifted. The shell's
`durable_resume_available` probe checks only structural durability and agreement between stored
manifest/checkpoint identities; it does not import credentialed runtime config. The run-local
curation path is explicit in the environment and in graph state, so concurrent runs and resumed
segments never read project-root history.

On `TERM`, the batch shell forwards the signal once, then waits up to `SLM_TERM_GRACE_S`
(default 120 s) for Python to publish SQLite/JSON checkpoint state, summaries, and observability
artifacts. It preserves the pipeline's final exit status; if the child remains hung after the
grace period, the shell sends `KILL`, reaps it, and exits with that forced-termination status.
`TERM` takes precedence over requeue, while the `USR1` durable-checkpoint verification and
requeue path is unchanged.

### Other durability properties

- **Dual authority.** A SQLite LangGraph checkpointer (`sqlite_checkpointer`) is the authority
  for graph position; `checkpoint.json` is a human-readable mirror.
  `reconcile_checkpoint_from_sqlite` repairs the mirror from SQLite. If the mirror records graph
  progress but SQLite has none, the run **aborts** rather than silently restarting.
- `graph_topology_descriptor(mode)` is compared on resume, so a checkpoint written by a
  different graph shape is rejected. A `thread_id` mismatch against the manifest is a hard error.
- **Atomic writes.** `atomic_write_json` / `atomic_write_jsonl` for datasets, eval sets, and
  checkpoints. `run_training_atomically` makes each training checkpoint write-or-nothing.
- ~~**Paid-acquisition ledger.**~~ **Removed 2026-08-19.** It was locked, append-only and under the
  stable run directory, and it did prevent a crash loop from re-spending budget — but it metered
  free re-reads of already-cached corpora alongside genuinely paid Exa discovery, which is what
  kept mining off its own data (B297). Discovery is now bounded by `MAX_FAILED_DISCOVERY_ROUNDS`,
  which needs no durable state: `state["failed_discovery_rounds"]` and `state["source_progress"]`
  ride the normal checkpoint.
- **Cost ledger.** `agent/cost.py` appends one `CostEvent` per instrumented call to
  `SLM_COST_EVENT_PATH` under an OS file lock, so the runner, forked acquisition processes, and
  spawned CUDA workers share one ledger. Provider SDK calls are wrapped explicitly at their call
  sites; there is intentionally no global monkey patch.
- **Logs.** GPU/environment setup logs are separated under `logs/gpu_setup/`, created by
  `mkdir -p` in `scripts/setup_gpu_env.sh` and `_l40s_task_body.sh` (Slurm does not create
  `--output` directories, and `logs/` is gitignored so a `.gitkeep` would not work).

### Driver phase order

| # | Phase | Notes |
|---|---|---|
| 1 | Run directory + tee logger | Before any import that might print |
| 2 | Observability paths | Inherited by forked/spawned children |
| 3 | Artifact path redirection | Must precede any node module caching `ARTIFACTS_DIR` |
| 4 | `hardware_research` | Skipped on resume |
| 5 | Initial state / checkpoint restore | |
| 6 | `is_torch_fx_available` compat shim | B111 |
| 7 | **Synthesis preflight** | Blocks until the local synth server answers, up to `SLM_SYNTH_WAIT_S`. Aborts rather than progressing on a dead endpoint, so a run cannot silently go gold-only. Opt out with `SLM_REQUIRE_SYNTH=0` |
| 8 | Graph stream with checkpoints | `stream_with_checkpoints` |
| 9 | On-device hardware verification | Opt-in `SLM_HW_VERIFY_ON_DEVICE=1`; writes `hardware_eval.json`, re-checks the constraints against **measured** values |
| 10 | Artifacts, DAG summary, baseline-vs-fine-tuned report, run summary, convergence metrics, full progression across all tiers | `build_run_progression` separates post-convergence downward attempts from the original trajectory, preventing adopted-model relabeling |

---

## 12. Not implemented

Carried over from `docs/intervention_capability_audit.md`. **These are gaps and
recommendations, not behavior.** No code was changed to produce this list.

### 12.1 Real gaps in shipped paths

| # | Gap | Evidence |
|---|---|---|
| 2 | **Gold-only degradation on a dead synth endpoint.** `_surgical_synthesize` logs "synthesis endpoint unavailable — no rows generated" and returns nothing. The driver's preflight (phase 7) blocks *at startup*, but an endpoint that dies mid-run degrades to a zero-row rebuild. That is at least now an **ERROR** line rather than a silent pad. | `curate.py::_surgical_synthesize` |
| 3 | **Synthesis has never run at scale in production.** The two completed runs that predate this note show only `synth_preflight` events in the cost ledger — 8/9 failed (NER), 18/43 failed (math) — and no generation events at all; the xlam run that followed had a healthy endpoint and produced zero rows for a different reason (B291). Every claim about synthesis quality remains untested in production. | `logs/runs/*/cost.json` |
| 4 | **Generation scorer mislabels its metric field.** The average LLM-judge score is carried in `EvalResult.f1`. The `metric` field now names what it really is (`judge_mean_0_1`), so reports cannot misattribute it, but `f1` itself is deliberately never renamed — checkpoints and DAG replay depend on the field name. | `eval/scorers/generation.py` |
| 7 | **`_annotate_ner_entities` does not re-validate spans.** Contrary to its prompt, returned spans are not rechecked as exact substrings and types are not allow-listed at that call site. A parse failure becomes an empty-entity gold row, indistinguishable from a genuine negative. Reachable only from the autonomous acquisition path. | `web_acquire.py::_annotate_ner_entities` |
| 10 | **`calendar_json`'s mining source is unverified.** `TOPv2/reminder` is a GitHub tarball, not a hub dataset, and nothing has confirmed its loader answers a request for a larger head slice — so rung 1 of the ladder may be a no-op for that task. **B303**, 🔴 open. | `tasks/calendar_json.py` |
| 11 | **`allow_paid_discovery=True` on `routerbench` and `proactive_listening`,** whose labels are derived rather than observed, so discovery there can only find data whose labels an LLM must invent. **B304**, ⚪ design gap. | `tasks/routerbench.py`, `tasks/proactive_listening.py` |
| 12 | ~~`agent/data_sizing.py` has no production call site.~~ **RESOLVED 2026-08-19** — the module is deleted, along with `CURRICULUM_SIZE_FLOOR`/`EVAL_SET_SIZE`/`DATA_SIZE_CEILING`. Per-task caps on `TaskSpec` replaced it. **B305** closed. | (deleted) |

> **Update 2026-08-19 — items 5, 8 and 9 were resolved by deletion, not by a fix.** `agent/tools/`
> (item 5, `delegate_task` and the four `@tool` wrappers) no longer exists. `agent/nodes/production/`
> (item 9, live confirmation) went with production mode on 2026-07-29. `_should_reexplore_downward`
> (item 8, the `bool("false")` coercion) is no longer in `downward_probe.py`. Their B-numbers —
> B21, B23, B199, B202, B207 — are retained in BUGS.md as history.
>
> The numbers are the original audit's and are left as they are so an entry can still be matched to
> the audit it came from; the gaps predate this sweep.

### 12.2 Recommended interventions (none implemented)

1. **Prompt-contract repair** — a first-class intervention that versions and edits the shared
   train/eval prompt builder, then re-evaluates without relabeling data.
2. **Verified hard-case generation** — *partly implemented as of 2026-08-19.* Three tasks now have
   exact programmatic verifiers on generated rows (`xlam_bfcl`, `calendar_json`, `ner_bc5cdr`; see
   `TaskSpec.synth_verifier`), and surgical synthesis targets measured failure categories rather
   than reusing unchanged failures. What is still missing is an exact verifier for `gsm8k` —
   `synth_verifier=None` there, so a generated math row is gated only by the teacher's own answer
   check, which is the weakest gate in the suite and the one an arithmetic check could replace
   outright.
3. **Preference optimization** — store generation negatives as explicit chosen/rejected pairs
   and train with a preference loss. Never as positive SFT.
4. **Class weighting** — still not implemented; nothing weights the loss or oversamples a class at
   the row level. *Confusion-pair oversampling, however, now exists in a better form:*
   `surgical_synthesis` budgets generation per failure **category** in proportion to that
   category's measured failure count, so the evidence does steer row production. `difficulty_buckets`
   and `confusion_pairs` were removed from the plan schema on 2026-08-19 precisely because nothing
   read them back to sample with; the buckets are still computed once at cold start and remain
   **advisory only**, feeding reporting and the orchestrator prompt (and a small-right/large-wrong
   row is still bucketed as `hard`, B298). `pattern_hint` survives in the plan and reaches the
   teacher prompt, not the sampler.
5. **LoRA target-module search** — target sets are fixed; a model-aware choice is possible.
6. **Optimizer schedule intervention** — warmup and scheduler are fixed. (Weight decay *is* now
   a bounded intervention; effective batch deliberately is not — see
   [§8](#8-hyperparameter-contract).)
7. **Context-length intervention** — sequence length is a fixed per-task ceiling (see §11);
   nothing truncates and the distribution is reported, so an over-length task fails loudly rather
   than silently. The orchestrator still cannot *choose* a longer context, so a genuinely
   long-context task has to be resized by hand with `SLM_MAX_SEQ_LENGTH`.
8. **Cross-run source cache** — reuse novelty fingerprints across independent runs without
   weakening source/split restrictions.
9. **Additional verified-positive strategies** — an exact-answer verifier for `gsm8k`, which is the
   one remaining open-ended task where correctness is decidable by computation and currently is not
   checked. (Execution verification is no longer relevant: `code_generation` and its sandbox were
   deleted 2026-08-18.)

### 12.3 Audit claims that were stale and are now corrected

Recorded so the delta is auditable.

| Old audit claim | Current code |
|---|---|
| Maximum sequence length: 512 | **4096** (`SLM_MAX_SEQ_LENGTH`); B188 raised it from 512 and added explicit zero-budget validation. This doc previously repeated the stale 512 and inferred a silent-truncation failure mode that does not exist — nothing truncates anywhere. |
| `_ram_tier` buckets modelled peak RAM | `_size_tier` buckets real on-disk size |
| Interpolation fits F1 vs log peak RAM | fits vs log on-disk weight size |
| The LLM may set rank, LR, epochs, **and per-device batch size** | five fields; batch shape retired |
| Micro batch and grad accum independently selected from {1,2,4,8} by the LLM | trainer-derived; rejected from LLM decisions |
| Alpha from {rank, 2·rank, 4·rank} and dropout from {0, .05, .1} are bounded interventions | `alpha_ratio` ∈ {1,2,4}; **dropout is not tunable at all** |
| Config-less fallback "steps rank to the next higher untried rung" | `deterministic_neighbor_configs` over all five axes |
| Graph node is `downward_probe_node` | `downward_probe_step_node` (durable, self-looping); the old name is a compatibility wrapper that loops ≤100 steps |
| Resource-safe fallbacks pick "lowest peak RAM" | lowest `size_mb` |
| `downward_probe → END` is a hard edge | conditional self-loop; `END` only when the probe finishes |
| `rollback → iterate` is a hard edge | conditional, like every other edge |
| `MAX_TURNS_MAIN` is dead config | wired as both the cumulative `_graph_steps` cap and the LangGraph `recursion_limit` |

---

## 13. State field reference

`agent/state.py::AgentState`, grouped as in the source.

**Task specification** — `description`, `target_metric`, `hardware_constraints`

**Task analysis** — `task` (the registry NAME — `xlam_bfcl`, `clinc150`, …, **not** an abstract
`task_type`), `selected_model`, `feasible_models` (largest→smallest), `stop_threshold`,
`initial_stop_threshold` (immutable floor), `threshold_calibration`, `task_label_space`,
`task_plan`, `autonomous`

**Data** — `train_examples`, `eval_set`, `data_source`, `current_dataset_path`,
`dataset_version`, `data_rebuild_plan`, `data_rebuild_plan_identity`, `curation_log_path`. Also
still declared and initialised: `source_acquire_rounds_used`, which nothing increments or reads now
that the paid-acquisition ledger is gone — as is `data/acquisition_budget.py` itself, which has no
importers.

**Curriculum growth bookkeeping (added 2026-08-19)** — `source_progress`
(`{source_id: {consumed, asked_for, url, exhausted}}`, so `mine_new_real` can tell an exhausted
corpus from one we read the first few thousand rows of), `failed_discovery_rounds` (consecutive
web-research rounds contributing zero novel rows; at `MAX_FAILED_DISCOVERY_ROUNDS` mining is
retired), `task_brief` (the orchestrator's own description of the benchmark, authored once at cold
start and used to build every teacher prompt), and `surgical_category_history` (written by curate:
which failure categories were targeted, at what count, so an unresponsive one is skipped as
exhausted).

**Stretch goals** — `convergence_banked`, `threshold_raises`, `threshold_lowers`,
`max_stop_threshold`

**Search state** — `best_weights_ref`, `best_score`, `lifetime_best_score` (max across all
tiers), `iteration`, `scores`, `dag`, `consecutive_no_improvement`, `downward_probe_done`,
`retained_gguf_paths`

**Last iteration** — `last_eval`, `last_curation`, `last_intervention`, `last_hypothesis`,
`llm_iterate_decision`, `next_action`

**Baselines** — `model_baselines`: one entry per selector that reached `evaluate`
(`{selector, model_id, quant, baseline_f1, best_finetuned_f1}`). `baseline_f1` is `None` when
unmeasured. Doubles as the source of truth for "which tiers the main ladder tried."

**Phase 2 flags** — `quantize_enabled`, `hw_gating_enabled`

**Turn budget** — `turn_budget`, charged at ~2 productive turns per iteration (curate + train). The
production fields that used to sit alongside it (`mode`, `deployed_model_ref`, `traces`,
`regression_set`, `replay_buffer`) went with production mode.

**Graph internals** — `_graph_steps` (durable cumulative node count),
`_wallclock_terminated_before`

**Strategy state** — `_largest_first_phase` (`probe`/`escalate`/`done`), `escalation_history`
(per-variant record including that variant's full DAG)

**Train→evaluate carry** — `_pending_weights_refs`, `_pending_training_outputs`,
`_pending_configs` (all keyed by config label)

**Data targets** — `curriculum_size_target`, `eval_size_target` (planner-chosen, clamped). Read
only by the **autonomous** acquisition path since 2026-08-19; a curated run sizes its initial load
from `TaskSpec.initial_train_cap` / `eval_cap` and then grows without a target. See B305.

**Provenance** — `eval_source_ban`, `data_sources`

**Difficulty / test agent** — `eval_difficulty` (`{easy, medium, hard}` text lists),
`test_report`

**Downward re-exploration** — `downward_tiers_tried`, `converged_model_ref`,
`downward_probe_history` (`{origin, fixed_H, attempts, termination?}`),
`downward_probe_pending`

> **Note.** `agent/state.py`'s inline comment on the `task` field still lists the old channel
> vocabulary (`"classification" | "NER" | "math_reasoning" | …`). The field itself holds a registry
> name — every reader resolves it through `tasks.get_task` — so the comment is stale, not the
> behaviour.

Written by a node but **not** declared in the `TypedDict`: `termination_reason`.
