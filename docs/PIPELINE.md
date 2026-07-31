# SLM Factory Pipeline

**Effective date: 2026-07-29.** Rewritten against the code as ground truth. Every claim below
was checked against a symbol in the repository, cited inline as `path::symbol`. Where the
previous documentation and the code disagreed, the code won; the corrections are listed in
[§12.3](#123-audit-claims-that-were-stale-and-are-now-corrected).

This document absorbs and replaces `docs/intervention_capability_audit.md`. Items from that
audit that are **not** implemented are collected in [§12](#12-not-implemented) rather than
described as behavior.

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
10. [Production mode](#10-production-mode)
11. [Durability and checkpoint/requeue](#11-durability-and-checkpointrequeue)
12. [Not implemented](#12-not-implemented)
13. [State field reference](#13-state-field-reference)

---

## 1. Graph topology

`agent/graph.py::build_graph(mode, checkpointer)` builds one of two LangGraph state machines
over `agent/state.py::AgentState`. `graph_topology_descriptor(mode)` is the canonical
structure and is used to reject unsafe checkpoint resumes.

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

### Production mode

Replaces the three cold-start entry nodes with four; the loop from `curate` onward is the
same code.

```
(entry) ──▶ trace_ingest ──▶ taxonomy_construct ──▶ live_confirm ──▶ parent_awareness ──▶ curate ──▶ (same loop)
```

### Exact routing table

| From | Router | Destinations |
|---|---|---|
| entry | `_route_before(entry)` | `task_analysis` / `trace_ingest`, `END` |
| `task_analysis` | `_route_before` | `eval_setup`, `END` |
| `eval_setup` | `_route_before` | `model_selection`, `END` |
| `model_selection` | `_route_before` | `curate`, `END` |
| `trace_ingest` | `_route_before` | `taxonomy_construct`, `END` |
| `taxonomy_construct` | `_route_before` | `live_confirm`, `END` |
| `live_confirm` | `_route_before` | `parent_awareness`, `END` |
| `parent_awareness` | `_route_before` | `curate`, `END` |
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
| `STAGNATION_WINDOW` | 20 | `agent/nodes/iterate.py` | Evals examined by the stagnation test |
| `STAGNATION_MIN_DELTA` | 0.02 | `agent/nodes/iterate.py` | Minimum window gain that counts as progress |
| `MAX_STALL_EVALS` | 20 | `agent/nodes/iterate.py` | Consecutive non-improving evals before escalation (sole stuck-run backstop) |
| `CURRICULUM_SIZE_FLOOR` | 3000 | `config/config.py` | Per-task floor; curricula are synth-filled up to this |
| `EVAL_SET_SIZE` | 800 | `config/config.py` | Eval floor; below n≈100 F1 CIs under-cover |
| `DATA_SIZE_CEILING` | 10000 | `config/config.py` | Hard cap on both targets (also the `target_rows` upper clamp) |
| `DEFAULT_STOP_THRESHOLD` | 0.96 | `config/config.py` | Used only if the planner supplies none |
| `MAX_PAID_ACQUIRE_ROUNDS_PER_RUN` | 9 | `agent/data_rebuild.py` | Exa spend ceiling for the whole run |
| `MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN` | 3 | `agent/data_rebuild.py` | Per data-rebuild plan |

`STAGNATION_*`, `MAX_STALL_EVALS`, and both size targets are env-overridable
(`SLM_STAGNATION_WINDOW`, `SLM_MAX_STALL_EVALS`, `SLM_CURRICULUM_SIZE`, `SLM_EVAL_SET_SIZE`, …).

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

- If `autonomous` or `task_type` is not one of the five valid types, calls
  `agent/task_planner.py::plan_task` for labels, Exa queries, benchmark, stop threshold, and
  data sizes.
- `_apply_data_targets` clamps the planner's `curriculum_size` / `eval_size` into
  `[floor, DATA_SIZE_CEILING]`. `SLM_CURRICULUM_SIZE` / `SLM_EVAL_SET_SIZE` override.
- `SLM_STOP_THRESHOLD` pins both `stop_threshold` **and** `initial_stop_threshold` (the
  immutable floor), taking precedence over the planner.
- Runs the hardware filter, sorts feasible **largest→smallest by `size_mb`**, stores in
  `feasible_models`. Raises if empty.
- **Does not set `selected_model`** — that is Node 1b's job.

The five valid task types, each differing in at least two of {model selection, supervision
format, eval metric, curation strategy}:

| `task_type` | Supervision | Eval metric | Positive synthesis eligible |
|---|---|---|---|
| `classification` | single label | macro-F1 / accuracy | ✅ |
| `NER` | typed spans as JSON | span-F1 (exact match, pooled TP/FP/FN) | ✅ |
| `math_reasoning` | CoT mandatory | final-answer exact match | ❌ |
| `code_generation` | code | APPS/MBPP execution pass@1 | ❌ |

**Code-execution safety.** APPS/MBPP scoring executes candidate code in an isolated
subprocess with a per-case timeout and a bounded per-problem deadline. The trusted controller
retains the expected outputs; candidate workers inherit no success FD and no expected-output
payload, so a candidate cannot signal a false pass. This is **trusted-input only** — the
process boundary and limits bound accidental damage, but it is **not a hostile-code sandbox**
(no seccomp, no namespace isolation). Only run benchmark code you obtained from a source you
trust.
| `generation` | free text | local Qwen3.6 LLM-as-judge `[0,1]` | ❌ |

### Node 2 — `eval_setup`

`agent/nodes/cold_start/eval_setup.py::eval_setup_node`

Builds `E = E_pos ∪ E_neg ∪ E_boundary` **before any training**, fixed for the whole run.

- **Shared-dataset path** (`SLM_SHARED_DATASET_DIR`) loads a frozen bundle so competing
  strategies see identical data. Requires `manifest.json` + `checksums.sha256`, verifies every
  file hash, checks `bundle_type`/`schema_version`, validates the row schema against
  `required_fields_for_task`, checks manifest counts against the JSONL, and **rejects any
  normalized train/test overlap**.
- **Acquisition path** — `data/loaders/web_acquire.py::acquire_dataset` with
  `gold_target = 0.65 × curriculum_size_target` and request headroom `×1.15 + 40` to survive
  eval-overlap removal and quality-control drops.
- `_eval_split_sizes(target)` scales the pos/neg/boundary slices at a fixed **0.4 / 0.4 / 0.2**
  ratio to `eval_size_target`. This was previously hardcoded 40/40/20, which silently capped
  every eval set at 100 rows regardless of how many test rows were acquired.
- **Leak firewall (layer 1)** — after *all* acquisition paths, any train row whose normalized
  text matches a test row raises `ValueError`. Logged as
  `official train/test separation: normalized overlap=0`.
- **Difficulty stratification** via `test_agent.label_difficulty` — see [§6.4](#64-test-data-agent).
- Persists `artifacts/eval_set.json` (counts, all three slices, difficulty buckets).

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

Requires `eval_set`; raises `RuntimeError` if absent (see [§10](#10-production-mode)).

1. **Eval firewall (layer 2)** — `_exclude_eval_rows` over `train_examples`.
2. **Resolve the plan** — `state["data_rebuild_plan"]` if present, else
   `fallback_data_rebuild_plan`; then `normalize_data_rebuild_plan`. No dedup/rotation:
   the plan is used as-is (redesign 2026-07-31).
3. **Seed** — `_entropy_seed()` (fresh OS entropy per sampler call). **Non-deterministic** by
   design; there is no reproducible per-plan seed.
4. **Execute the one strategy** — `acquire` mines new real rows, `synthesize` generates
   task-adaptive rows, `resample` reshuffles; then resample-fill covers the remainder to
   `target_rows`.
5. **Synth-fill to `target_rows`** — `_synth_fill_to_target` tops up any shortfall with
   task-adaptive synthesis (covers the initial curriculum and every rebuild); degrades
   gracefully if the endpoint is down.
6. **CoT annotation** for math/code/generation (`_annotate_generation_cot`); skipped under
   `SLM_CHEAP=1`.
7. `apply_quality_controls` → truncate to `target_rows` → **eval firewall (layer 3)**.
8. Atomic write to `artifacts/dataset_v{N}.jsonl`; every row stamped `_dataset_version`.
9. Record `last_curation`: provenance/source/difficulty composition, per-origin
   `rows`/`novel_rows`, `plan_yield`, `source_novelty`, and `allocation_fallbacks`.

**Allocation fallbacks are recorded, not silent.** When `acquire` mining yields no novel rows,
or `synthesize` produces nothing (endpoint down / cheap mode), the shortfall is covered by
resample-fill and an honest entry (`base_fill` / `synth_unavailable_degrade`) is appended to
`allocation_fallbacks` — never a crash and never a misattributed strategy.

`apply_quality_controls` (`data/curriculum.py`) implements four controls, task-routed:
label balancing, context-length outlier removal (>3× median), entity-value capping (≤3
occurrences, NER), and Jaccard>0.9 near-duplicate removal.

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
| any other intervention (`data_rebuild`, `rollback`) | **carry-forward best prior config**, labelled `[carry-fwd best]` |
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
- `pi.S` — `task_type`, `supervision`, `loss_masking="assistant_only"`,
  `loss_contract_version`

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
`diagnosis`, `suggested_intervention`, `band`. Confusion is task-aware:

| task_type | gold | predicted |
|---|---|---|
| `classification` | label | predicted label |
| `NER` | sorted gold entity types | sorted predicted types, or `incorrect_entity_set` |
| everything else | `error_type` / `judge_category` / `gold_verifier` | `"incorrect"` |

Open-ended targets are reduced to a verifier category because the raw target *is* the held-out
answer.

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
| 4 | `score < threshold` **and** (stagnant or stalled) | → `escalate`, or `terminate` for a `largest_first` probe | **no** |
| 5 | otherwise | `_llm_iterate` | yes (1, + ≤1 reask) |
| 6 | after a threshold adjustment | re-run `_route_score_at_threshold` | no |
| 7 | `hyperparameter` | → `train` | — |
| 8 | `data_rebuild` | → `curate` | — |

**Step 3 — `_route_score_at_threshold`**, in order:

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

**Stagnation vs stall — two different guards, both needed:**

- `_is_stagnant(scores)` — requires ≥50 scores, then `max(window) − window[0] <
  STAGNATION_MIN_DELTA`. Declines and below-origin oscillation count as no progress. A
  `math.isclose` tolerance keeps the exact 0.02 boundary non-stagnant despite binary float
  representation.
- `consecutive_no_improvement >= MAX_STALL_EVALS` — the backstop. `should_rollback` **pops**
  the regressing score, so the stagnation window may never fill during rollback churn;
  `consecutive_no_improvement` is set in `evaluate` and is *not* popped.

**Step 4 spends no API call by design.** Escalation on stagnation is a rule the LLM cannot
override, and plateauing is exactly when a run makes the most `iterate` calls.

**Step 5 — the LLM decision.** One tool-free `ORCHESTRATOR_MODEL` call carrying the compacted
trajectory (`agent/context_manager.py::compact_trajectory` when `should_compact`), the
test-agent report, tried `(dataset, H)` identities *including pruned ones*, tried rebuild-plan
identities with yield status, source novelty, and remaining budgets.

- A `tool_calls` response is never executed or reflected back — it triggers `_reask_json_only`
  from the original bounded context, so a tool-use block cannot open a side channel.
- `_parse_decision_json` tolerates content-block lists, code fences, and prose wrapping, and
  **always** raises `ValueError` rather than a bare `JSONDecodeError`.
- `_validate_decision_json` is re-applied to the result with `allow_internal=True` as defense in
  depth, so mock/alternate provider paths cannot bypass the contract.
- **Failure ladder:** `raise_if_fatal` → test-agent `suggested_intervention` → score bands.

**Score bands are fallback only**, not enforced boundaries on a valid LLM decision:
`<0.80` → `data_rebuild` / `acquire`; `0.80–0.95` → `hyperparameter`;
`≥0.95` → `data_rebuild` / `synthesize`.

**Threshold adjustment.** `new_threshold` is clamped to `max(value, initial_stop_threshold)` and
applied only if it *lowers* the current threshold. The LLM is told the floor is system-enforced.
Legitimate reasons are genuine capacity limits (world knowledge beyond parametric memory,
reasoning chains too long, adversarial OOD) — explicitly not "the task is hard but learnable."
When `stop_threshold` already equals the floor, this path is inert.

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

- **Plan** — record `origin` (once), compute untried lower tiers, ask
  `_should_reexplore_downward` (LLM; fallback `margin >= 0.03`), pick a candidate with
  `_llm_choose_model(direction="down")`, write `downward_probe_pending`, return. The choice is
  checkpointed *before* training.
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

`agent/data_rebuild.py` defines **three** strategies (redesign 2026-07-31), chosen singly with
no task-type or score gating:

| Strategy | What it does |
|---|---|
| `resample` | Reshuffle / re-draw rows from the existing pool (entropy-seeded) |
| `acquire` | Add new rows from the same or a new provenance (bounded real-source mining: local → deterministic benchmark → paid Exa) |
| `synthesize` | Task-adaptive synthetic generation — hard negatives for classification/NER, new *correct* in-distribution examples for math/code/generation |

Regardless of strategy, the curriculum is **synth-filled up to `target_rows`** when real data
falls short (covers the initial curriculum and every rebuild); if synthesis is unavailable it
degrades gracefully (resample-fill + a logged `allocation_fallbacks` entry, never a crash).

**Plan schema** (`normalize_data_rebuild_plan`) — every field is snapped to a bounded, stepped
range:

`strategy` (one of the three) · `target_rows` [16, `DATA_SIZE_CEILING`] step 8 ·
`resample_fraction` [0.10,1.00] step 0.05 · `new_real_rows` [0,500] step 5 ·
`synth_rows` [0,2000] step 5 · `max_acquire_rounds` [0,3] · `difficulty_buckets` ·
`confusion_pairs` (≤8) · `pattern_hint`

Validation rejects: an unknown `strategy`, and any raw held-out text anywhere in the plan. A
material strategy (`acquire`/`synthesize`) with a zero budget has a sensible positive budget
auto-filled rather than being rejected.

**Non-determinism.** There is no plan-identity dedup, no untried-plan rotation, and no
`DataRebuildPlanSpaceExhausted`. Sampling and synthesis draw fresh OS entropy each call, so
repeated rebuilds genuinely vary. The orchestrator freely re-picks any strategy each turn;
**escalation after `MAX_STALL_EVALS` (20) non-improving evals** is the sole stuck-run backstop
(plus the wall-clock guard). Exact checkpoint-resume reproducibility is intentionally dropped.

**Real-source mining** (for `acquire`) tries local and deterministic benchmark sources before
process-isolated paid discovery. Local candidates require an explicit benchmark match or strong
task+label+schema agreement. Paid rounds use a locked append-only reservation ledger under the
stable run directory: reservation precedes the call and remains spent after completion, failure,
or crash.

**Fallback plan** (`fallback_data_rebuild_plan` → `_fallback_strategy_from_signal`) is
**non-deterministic and signal-weighted**: it draws a weighted-random strategy biased by the
measured failure signal — failing easy bucket biases toward `acquire`; weak medium/hard (or
confusion pairs) biases toward `synthesize`; otherwise all three are roughly equal. Difficulty
weights are computed **inversely to measured accuracy** (`0.1 + 0.7 × deficit/total_deficit`,
floored at 0.1). This replaced the old deterministic keyword/rotation chooser that sent every
NER fallback to `resample_existing` and exhausted the (then-bounded) plan space.

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

- **Max sequence length 4096** (`SLM_MAX_SEQ_LENGTH`, clamped to [128, 32768]). Task-specific
  output reserves: 50 classification, 512 NER/math/generation, 1024 APPS.
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

## 10. Production mode

`mode="production"` (paper §2.6). Four entry nodes, then the identical shared loop.

| Node | Symbol | Behavior |
|---|---|---|
| `trace_ingest` | `production/trace_ingest.py` | Loads judged traces from `state["traces"]` or `traces.jsonl`; partitions `T_fail`/`T_pass`; seeds `train_examples` from `T_fail` corrected outputs |
| `taxonomy_construct` | `production/taxonomy.py` | One `ORCHESTRATOR_MODEL` call clusters up to **40** sampled failures into 3–8 categories, each labeled `fixable` or external. The sampled window and the tagged window are the same 40 — they previously mismatched (20 shown, 50 tagged) |
| `live_confirm` | `production/live_confirm.py` | Pre-screens by the `cluster` **key** (traces with `cluster=None` are kept as unclassified, not dropped), then **re-runs M0** on each candidate and keeps only failures M0 still reproduces. Inference errors count conservatively as confirmed |
| `parent_awareness` | `production/parent_awareness.py` | Regression set `R` = stratified sample of passing traces (≥50, or half); replay buffer `D_replay` = 15% of the parent dataset (paper's 10–20%) |

`curate` consumes `replay_buffer` at ≤20% of `target_rows`, tagged `_provenance="replay"`, and
applies the eval firewall to it.

> **Production mode is not runnable end-to-end.** `curate_node` raises when `eval_set` is
> `None`, and the production graph never builds one. See [§12](#12-not-implemented).

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
- **Paid-acquisition ledger.** Locked, append-only, under the stable run directory. Reservation
  precedes the Exa call and stays spent after completion, failure, or crash, so a crash loop
  cannot re-spend budget.
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
| 1 | **Production mode cannot start.** `curate_node` raises when `eval_set is None`, and the production graph has no node that builds one. A caller must pre-populate it; nothing validates that at graph entry. | `curate.py` guard vs `graph.py` production branch |
| 2 | **Open generation and math/code have no augmentation strategy.** `synthesize_hard_negatives` returns gold anchors unchanged for `generation`, and warns and returns originals for math/code. Never writing rejected answers as positive SFT is correct — but it leaves those families with real-data sampling/mining and optional CoT only. | `data/curriculum.py` generation/math/code branches |
| 3 | **Gold-only degradation on a dead synth endpoint.** `curate` logs a warning and proceeds without synthesis. The driver's preflight (phase 7) blocks *at startup*, but an endpoint that dies mid-run degrades silently. | `curate.py::_synthesize_positive_rows` |
| 4 | **Synthesis has never actually run at scale.** Both completed runs show only `synth_preflight` events in the cost ledger — 8/9 failed (NER), 18/43 failed (math) — and **zero** `hard_negative_synthesis` events. Every claim about synthesis quality is therefore untested in production. | `logs/runs/*/cost.json` |
| 5 | **Generation scorer mislabels its metric.** The average LLM-judge score is reported in the `"f1"` field. | `eval/scorers/generation.py:592` |
| 6 | **`delegate_task` and the four `@tool`-decorated tools have zero call sites.** No sub-agent or tool-using path is reachable from either graph. | `agent/tools/` |
| 7 | **No baseline/SOTA survey.** Design §2.4 stage 3 calls for a web-search survey of published baselines at plan time; `task_analysis` relies on the planner LLM's own recall. | `task_analysis.py` |
| 8 | **`_annotate_ner_entities` does not re-validate spans.** Contrary to its prompt, returned spans are not rechecked as exact substrings and types are not allow-listed at that call site. A parse failure becomes an empty-entity gold row, indistinguishable from a genuine negative. | `web_acquire.py::_annotate_ner_entities` |
| 9 | **`_should_reexplore_downward` coerces with `bool(...)`.** A JSON string `"false"` would evaluate true. | `downward_probe.py` |
| 10 | **Live confirmation does not replay the serving prompt.** It sends `trace["input"]` raw and compares by exact string, which is wrong for most classification/NER/generation outputs. | `production/live_confirm.py` |

### 12.2 Recommended interventions (none implemented)

1. **Prompt-contract repair** — a first-class intervention that versions and edits the shared
   train/eval prompt builder, then re-evaluates without relabeling data.
2. **Verified hard-case generation** — task-specific generators for math/code that emit new
   problems *plus* executable or exact-answer verification, instead of reusing unchanged
   failures.
3. **Preference optimization** — store generation negatives as explicit chosen/rejected pairs
   and train with a preference loss. Never as positive SFT.
4. **Class weighting and confusion-pair oversampling** — `difficulty_weighted_sampling` is
   implemented and `confusion_pairs` reach the `pattern_hint`, but explicit class weights and
   confusion-pair oversampling are not.
5. **LoRA target-module search** — target sets are fixed; a model-aware choice is possible.
6. **Optimizer schedule intervention** — warmup and scheduler are fixed. (Weight decay *is* now
   a bounded intervention; effective batch deliberately is not — see
   [§8](#8-hyperparameter-contract).)
7. **Context-length intervention** — sequence length is a fixed 4096 for every task. Nothing
   truncates and the distribution is now reported, so an over-length task fails loudly rather
   than silently; but the orchestrator still cannot *choose* a longer context, so a genuinely
   long-context task has to be resized by hand.
8. **Cross-run source cache** — reuse novelty fingerprints across independent runs without
   weakening source/split restrictions.
9. **Additional verified-positive strategies** — math/code generators, gated on exact-answer or
   execution verification for every synthesized row.

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

**Task analysis** — `task_type`, `selected_model`, `feasible_models` (largest→smallest),
`stop_threshold`, `initial_stop_threshold` (immutable floor), `task_plan`, `autonomous`

**Data** — `train_examples`, `eval_set`, `data_source`, `current_dataset_path`,
`dataset_version`, `data_rebuild_plan`, `data_rebuild_plan_identity`,
`source_acquire_rounds_used`, `curation_log_path`

**Search state** — `best_weights_ref`, `best_score`, `lifetime_best_score` (max across all
tiers), `iteration`, `scores`, `dag`, `consecutive_no_improvement`, `downward_probe_done`,
`retained_gguf_paths`

**Last iteration** — `last_eval`, `last_curation`, `last_intervention`, `last_hypothesis`,
`llm_iterate_decision`, `next_action`

**Baselines** — `model_baselines`: one entry per selector that reached `evaluate`
(`{selector, model_id, quant, baseline_f1, best_finetuned_f1}`). `baseline_f1` is `None` when
unmeasured. Doubles as the source of truth for "which tiers the main ladder tried."

**Phase 2 flags** — `quantize_enabled`, `hw_gating_enabled`

**Production** — `mode`, `deployed_model_ref`, `traces`, `failure_taxonomy`, `regression_set`,
`replay_buffer`, `turn_budget`

**Graph internals** — `_graph_steps` (durable cumulative node count),
`_wallclock_terminated_before`

**Strategy state** — `_largest_first_phase` (`probe`/`escalate`/`done`), `escalation_history`
(per-variant record including that variant's full DAG)

**Train→evaluate carry** — `_pending_weights_refs`, `_pending_training_outputs`,
`_pending_configs` (all keyed by config label)

**Data targets** — `curriculum_size_target`, `eval_size_target` (planner-chosen, clamped)

**Provenance** — `eval_source_ban`, `data_sources`

**Difficulty / test agent** — `eval_difficulty` (`{easy, medium, hard}` text lists),
`test_report`

**Downward re-exploration** — `downward_tiers_tried`, `converged_model_ref`,
`downward_probe_history` (`{origin, fixed_H, attempts, termination?}`),
`downward_probe_pending`

Written by a node but **not** declared in the `TypedDict`: `termination_reason` (set by `curate`
on plan-space exhaustion).
