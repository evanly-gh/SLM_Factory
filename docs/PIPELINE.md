# SLM Factory Pipeline

How the LangGraph state machine is wired, what each node does, and what decisions are made at each step.

---

Three conditional edges and one hard edge control the loop:

- **`evaluate → rollback | iterate`** — decided by `should_rollback(state)`, a pure function, no LLM
- **`rollback → iterate`** — hard edge; restores the best checkpoint then re-enters the decision loop so the next action is *different* from the config that just regressed (a bare re-train of the identical config would deterministically regress again — an endless loop)
- **`iterate → train | curate | escalate | downward_probe | terminate`** — decided by `state["next_action"]`, set by the LLM inside `iterate_node`
- **`escalate → train | curate | terminate`** — decided by `state["next_action"]`; `escalate_node` always writes `"curate"` or `"terminate"`, so the `"train"` branch is registered but unreachable under current code (kept for forward-compatibility)
- **`downward_probe → END`** — hard edge; downward_probe always terminates

---

## Pre-Graph Step: `hardware_research`

Runs in `run.py` **before the LangGraph graph is invoked**. Implemented in [agent/nodes/cold_start/hardware_research.py](../agent/nodes/cold_start/hardware_research.py).

- Three-stage pipeline: (1) local DB lookup against `data/devices.csv`; (2) Exa web search fallback if no match; (3) Claude Sonnet call that returns a structured JSON object with device specs.
- Constructs a `HardwareConstraints` object with fields: `storage_mb`, `memory_mb`, `latency_ttft_ms`, `power_watts`, `target_chip`, `min_tok_s`.
- `target_chip` is resolved to the closest anchor in `CHIP_SCALE_FACTORS` (14 chips covering Snapdragon, Dimensity, Exynos, and Tensor families); falls back to `snapdragon_778g` if unrecognized.
- Falls back to conservative defaults (`3000 MB RAM`, `1500 MB storage`, `snapdragon_778g`) if the API call or JSON parsing fails.
- Result is written to `{RUN_DIR}/device_research.json` and passed into `state["hardware_constraints"]` for the entire graph run.

**Writes to state (initial_state):** `hardware_constraints`
**Decision made:** what the hardware budget is for model selection and hardware-gate checks

**Note on `hardware_filter` (Stages 1+2):** `run_hardware_filter` (implemented in `agent/nodes/cold_start/hardware_filter.py`) runs **inside `task_analysis_node`**, not as a separate pre-graph step. It applies Stage 1 (memory/storage/latency inequality checks) and Stage 2 (on-device benchmark stub, largest→smallest ordering) against `ANDROID_POOL` to produce the `feasible_models` list. This list is written to state by `task_analysis_node` and consumed by the model selection node.

**Model pool — `ANDROID_POOL` (Qwen family):** The pool contains **18 entries** — 6 Qwen-family base models each expanded to three quantization siblings (`bf16`, `Q4_K_M`, `Q8_0`):
- `unsloth/Qwen3-0.6B` — text-only; Tier 0 seed
- `Qwen/Qwen2.5-1.5B-Instruct` — text-only general 1.5B; Tier 1 seed
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` — Qwen arch, R1 distilled, math/reasoning; Tier 1 seed
- `Qwen/Qwen2.5-3B-Instruct` — text-only, top-capability; Tier 2/3 seed
- `Qwen/Qwen3.5-0.8B` — **multimodal**; text-only LoRA via `text_tokenizer()`
- `Qwen/Qwen3.5-2B` — **multimodal**; base transformers repo (not `-GGUF`); text-only LoRA via `text_tokenizer()`

Tier coverage after quant expansion: tier 0 ×2, tier 1 ×4, tier 2 ×6, tier 3 ×6 — so escalation can walk tiers 0→3. `ANDROID_POOL` is sorted by `(tier, size_mb)`. `ModelSpec` includes `quant` (`None`/`"Q4_K_M"`/`"Q8_0"`) and `multimodal` (bool).

> **Multimodal handling (B123/B107):** the two Qwen3.5 models are natively multimodal (image-text-to-text) and load as a *processor*. Text-only LoRA works because `training.lora_trainer.text_tokenizer()` extracts the inner text tokenizer so the vision path is never exercised (used in both `lora_trainer` and `slm_helpers`). Qwen3.5-2B uses the **base** transformers repo `Qwen/Qwen3.5-2B`, never the `-GGUF` repo (which has no transformers config and cannot be fine-tuned). Multimodal LoRA on this stack is **best-effort / unverified** without a GPU run; the four text-only Qwen models are the reliable path and `smallest_first` starts on one of them (Qwen3-0.6B).

---

## Node-by-Node

### Node 1: `task_analysis`

Entry point for cold-start. Reads `state["description"]` and `state["autonomous"]`.

- If autonomous mode (or no valid `task_type` provided), calls `plan_task()` — a Claude Sonnet call that returns `task_type`, `task_name`, `labels`, `multi_label`, `schema`, `multilingual`, `exa_queries`, `benchmark`, `stop_threshold`, and `rationale` as JSON.
- Valid task types: `classification`, `NER`, `math_reasoning`, `code_generation`, `generation`. Variants (multi-label classification, schema-constrained extraction, multilingual tasks) are expressed as flags on the plan dict, not separate types.
- Calls `run_hardware_filter(hardware_constraints, ANDROID_POOL)` internally (Stages 1+2) to produce `feasible_models` — the hardware-feasible subset of the pool, sorted largest→smallest by `size_mb` descending. This list is written to `state["feasible_models"]`. **`task_analysis_node` does NOT set `selected_model`**; model selection is deferred to the configurable `model_selection` node.
- Sets `state["stop_threshold"]` from the planner's output (calibrated relative to model-size SOTA, not a fixed 0.96) and records `state["initial_stop_threshold"]` as the immutable floor.

**Writes to state:** `task_type`, `task_plan`, `feasible_models`, `stop_threshold`, `initial_stop_threshold`
**Decision made:** what the accuracy target is, which task flags apply, which models are hardware-feasible

---

### Node 1b: `model_selection` (configurable strategy)

**Purpose:** Select the initial model from `feasible_models` for LoRA fine-tuning. The strategy is configurable via `config.config.MODEL_SELECTION_STRATEGY` (env: `SLM_MODEL_SELECTION_STRATEGY`).

**Input state fields:** `feasible_models`, `stop_threshold`, `eval_set`, `current_dataset_path`, `task_type`, `hardware_constraints`, `description`, `task_plan`

**Output state fields:** `selected_model`

**Position in graph:** `eval_setup → model_selection → curate`

**Available strategies** (defined in `agent/nodes/cold_start/model_selection/`):

#### `smallest_first` (default)

Start with the smallest feasible model (lowest peak RAM). If the training loop stagnates, normal escalation moves to the next tier up. No probing overhead — the first training cycle uses the production config.

**Algorithm:** `state["selected_model"] = min(feasible, key=peak_memory_mb)`

#### `largest_first`

Start with the largest feasible model to probe whether the task is solvable within the hardware budget. If the largest model reaches the accuracy goal, switch to the smallest model and escalate from there (resource-optimized). If the largest model stagnates without reaching the goal, terminate early — the task is infeasible.

**Algorithm:**
1. `state["selected_model"] = max(feasible, key=peak_memory_mb)`
2. Set `state["_largest_first_phase"] = "probe"`
3. On iterate: if probe model reaches threshold → switch to smallest, reset scores, `next_action = "curate"`
4. On iterate: if probe model stagnates → `next_action = "terminate"` (task infeasible)

#### `interpolation`

Probe 3 models (smallest, middle, largest from feasible set) with 1-epoch LoRA training, fit a log-linear accuracy-vs-size curve, then select the model whose peak RAM is closest to the hardware memory budget while still meeting the accuracy goal. (Refined version of the original `scaling_curve` approach from the paper.)

**Algorithm:**
1. Pick 3 candidates (largest, middle, smallest) from `feasible_models`.
2. For each: 1-epoch LoRA probe → record `(log(params), f1)`.
3. Fit `f1 = a * log(params) + b` via least-squares.
4. Filter models whose predicted f1 ≥ `stop_threshold`.
5. From qualifiers, pick the one closest to `hardware_constraints.memory_mb`.
6. If no qualifier, fall back to the highest-capability model.

#### `orchestrator_choice`

Let the orchestrator LLM choose the starting model directly, using the task type, task plan, benchmark affinities, and hardware constraints as context. No probing overhead — single API call. Falls back to the largest feasible model if the LLM call fails.

**Writes to state:** `selected_model`, optionally `_largest_first_phase`
**Decision made:** which model to fine-tune first

---

### Node 2: `eval_setup`

Runs once. The eval set is fixed from this point forward and never modified again.

- If `state["task_plan"]` exists (i.e. autonomous mode produced a plan), calls `acquire_dataset()` which uses Exa web search to pull real examples for each label.
- If no plan exists and `task_type == "classification"`, falls back to the hardcoded SMS Spam loader. For any other task type without a plan, raises `NotImplementedError`.
- Calls `build_eval_set()` which partitions test examples into `E = E_pos ∪ E_neg ∪ E_boundary`.
- Persists the eval set to `artifacts/eval_set.json` on disk.

**Writes to state:** `train_examples`, `eval_set`
**Decision made:** none — deterministic data loading

---

### Node 3: `curate`

Builds or augments the training dataset. Behavior branches on `state["last_intervention"]` (defaults to `"data_rebuild"` if unset):

| Condition | What happens |
|---|---|
| `"data_rebuild"` or `current_dataset_path is None` | Full rebuild from scratch (see dataset sizing below) |
| `"surgical"` | Load existing dataset from disk, append targeted examples per failure pattern |
| Anything else (including `"hyperparameter"`) | **Returns immediately** — dataset held fixed, no curation |

**Dataset sizing by task type (data_rebuild path):**

| `task_type` | N total | Rationale |
|---|---|---|
| `classification` | 150 | Simple decision surface |
| `multi_label_classification` | 300 | Label co-occurrence patterns need more coverage |
| `NER` | 300 | Entity diversity requirement |
| `structured_extraction` | 400 | Schema field coverage + negative schema examples |
| `math_reasoning` | 1 000 | Step-by-step CoT chains; quality over quantity |
| `code_generation` | 1 000 | Function diversity + execution harness coverage |
| `multilingual` | 400 | Language pair coverage adds dimensionality |
| `generation` | 1 000 | Open-ended; needs diverse inputs |

Split is always 65% gold (`build_initial_curriculum`) + 35% hard negatives (`synthesize_hard_negatives`, Claude Sonnet API call), then `apply_quality_controls` (label balancing, dedup). For `math_reasoning`, `code_generation`, and `generation`, additionally calls `annotate_cot()` with a teacher model (DeepSeek-R1 or GPT-4.1).

**Surgical path:** loads existing dataset, synthesizes `min(max(len(failures) * 2, 10), 20)` targeted examples from the LLM's `targeted_patterns` field, then re-applies quality controls.

In production mode, mixes the `replay_buffer` into the dataset at this step.

Saves the dataset to `artifacts/dataset_v{N}.jsonl` and increments `dataset_version`.

**Writes to state:** `current_dataset_path`, `dataset_version`, `last_curation`
**Decision made:** none — executes the intervention type the LLM decided in the prior `iterate` call

---

### Node 4: `train`

Trains a single LoRA configuration. Always produces a LoRA adapter (no full fine-tuning) for on-device adapter-manager deployment.

Config selection logic:
- If `iterate_node` set `last_intervention = "hyperparameter"` and provided a `hyperparams` dict, uses those values (clamped to valid LoRA ranks). If the LLM requested FFT (`lora_rank: null`), it is overridden to LoRA r=8.
- Otherwise, defaults to LoRA r=8, 3 epochs, lr=2e-4.

Always trains from the base model — never loads a prior adapter. Stores the `weights_ref` in `state["_pending_weights_refs"]`.

**Writes to state:** `_pending_weights_refs`, `_pending_configs`, increments `iteration`
**Decision made:** none — executes what curate set up

---

### Node 5: `evaluate`

Scores the trained config against the fixed eval set.

- Calls `run_eval()` for the `weights_ref` — dispatches to three scorer modules by task type:
  - `eval.scorers.classification` — accuracy/F1 for `classification`
  - `eval.scorers.ner` — entity span-F1 for `NER`
  - `eval.scorers.generation` — handles `math_reasoning` (final-answer exact match), `code_generation` (execution pass@1), and `generation` (LLM-as-judge) by inspecting `task_type` internally
- Updates `best_score`, `best_weights_ref`, `consecutive_no_improvement`.
- Appends a node to the linear DAG (`state["dag"]`) with the full `π = (D, H, S)` triple.
- Writes the iteration record to `data-curation.md` via `CurationLog`, including score band, hardware PASS/FAIL, config labels, and the hypothesis from the prior iterate call.

The routing decision is made via `should_rollback(state)` (defined in `rollback.py`):
- `scores[-1] < scores[-2]` → routes to `rollback`
- Otherwise → routes to `iterate`

**Writes to state:** `best_score`, `best_weights_ref`, `scores`, `last_eval`, `consecutive_no_improvement`, `dag`

---

### Node 6: `rollback` (conditional)

Only reached if the latest score was worse than the previous.

- Pops the last score from `state["scores"]`.
- Sets `last_intervention = "rollback"`.
- Marks the last DAG node as `pruned = True`.
- Restores `best_weights_ref` and `best_score` to the highest-scoring non-pruned DAG node.
- Increments `consecutive_no_improvement`.

After rollback the graph proceeds to `iterate` (**changed** — it previously went straight to `train`). Restoring the best checkpoint and re-training the *same* dataset + hyperparameters is (near-)deterministic, so it reproduces the same regressing score and rolls back again — an endless loop that was observed to burn the entire turn budget (e.g. ARC-Challenge and GSM8K oscillating for 20+ iterations at a fixed best score). Routing to `iterate` forces the next step to be a genuinely different action: a `data_rebuild` with a rotated sampling seed, a hyperparameter change, escalation to a bigger model, or a clean termination via the stall backstop.

**Writes to state:** `scores`, `dag`, `best_weights_ref`, `best_score`, `last_intervention`, `consecutive_no_improvement`
**Decision made:** none — all logic is rule-based

---

### Node 7: `iterate`

The only node with an LLM call that controls graph routing. This is the EXPAND operator (paper §2.2 Eq. 2).

Calls `_llm_iterate()` — a multi-round tool-use loop with Claude Sonnet that:

1. Compacts the `data-curation.md` trajectory via `compact_trajectory()` if it exceeds ~8,000 tokens.
2. Feeds the compacted trajectory, score history, current score, and up to 10 sample failures to the LLM.
3. Allows up to 5 rounds of tool use (bash, read_file, edit_file, web_search) before requiring a final JSON decision.
4. Parses the JSON decision: `{intervention, hypothesis, hyperparams?, targeted_patterns?, threshold_adjustment?}`.

**Threshold adjustment:** The LLM may include a `threshold_adjustment.new_threshold` value in its decision. If provided, `iterate_node` lowers `stop_threshold` to that value, clamped to `initial_stop_threshold` as a floor. This is used when the LLM identifies that the dominant failure cluster reflects a genuine model capacity limit (world knowledge gaps, reasoning chains longer than the model can produce, adversarial OOD inputs) rather than a data or hyperparameter problem. The floor is enforced in code — the LLM cannot set it below `initial_stop_threshold`.

Falls back to `apply_iteration_policy()` (pure score-band rules) if the LLM call fails.

**Stagnation detection:** escalation is triggered by a sliding-window delta check, not a consecutive-no-improvement count:

```python
STAGNATION_WINDOW = 3        # number of recent eval runs to inspect
STAGNATION_MIN_DELTA = 0.02  # minimum cumulative improvement over that window to avoid escalation
```

If `max(scores[-3:]) - min(scores[-3:]) < 0.02`, the model is stagnant and `next_action = "escalate"` regardless of what intervention the LLM chose.

**Stall backstop (`MAX_STALL_EVALS = 4`):** `should_rollback` pops the regressing score, so on a rollback→re-decide churn the stagnation window can stay short and never fire. As a hard backstop, `iterate_node` also escalates when `consecutive_no_improvement >= MAX_STALL_EVALS` (that counter is set in `evaluate_node` on every non-improving eval and is **not** popped by rollback). Escalation promotes to a bigger model if one fits the hardware budget, otherwise terminates — guaranteeing the loop always makes progress or ends.

Routing logic (evaluated in this order):

```
no scores yet                              → "train"       (first iteration, nothing to reason about)
iteration*2 >= turn_budget                → "terminate"   (turn budget exhausted; ~2 turns per iteration)
current_score >= stop_threshold:
  hw_gating_enabled AND hw fails          → continue (escalate if stagnant, else train or curate)
  not downward_probe_done AND tier > 0    → "downward_probe"  (try smaller model once before accepting)
  else                                    → "terminate"
_is_stagnant(scores)                       → "escalate"   (delta-based, window=3, min_delta=0.02)
intervention == "hyperparameter"           → "train"       (skip curate, same dataset)
else (data_rebuild or surgical)            → "curate"
```

**Writes to state:** `last_intervention`, `last_hypothesis`, `llm_iterate_decision`, `next_action`, optionally `stop_threshold`
**Decision made:** primary decision node — determines everything that happens next

---

### Node 8: `escalate`

Reached when `_is_stagnant(scores)` fires inside `iterate`.

- Calls `filter_pool(hardware_constraints)` to get the hardware-feasible subset, then collects **all** models in `tier == current_tier + 1`.
- If `current_tier + 1 > 3` or no candidates exist in the next tier, sets `next_action = "terminate"`.
- Calls `_llm_choose_model` (Claude Sonnet) on the full set of next-tier candidates to pick the best fit for the task; falls back to the largest candidate (highest `int4_size_mb`) on failure.
- Checks hardware constraints for the chosen model; if it does not fit, terminates.
- Otherwise promotes `selected_model` to the chosen next-tier model and resets score history and DAG.
- Also resets `downward_probe_done = False` so the downward probe is available for the new model.
- The curated dataset path is **preserved** — `current_dataset_path` carries over to the new model. The curated data is still valid; only the weights and score trajectory are stale.
- Sets `next_action = "curate"` — the new model gets a fresh curation round before training.

The state reset is intentional: the old model's scores would corrupt the rollback gate for the new model.

Routing after escalate:

```
next_action = "curate"     → curate → train (new model gets a fresh curate round)
next_action = "terminate"  → END
next_action = "train"      → train  (registered edge; not currently set by escalate_node)
```

**Writes to state:** `selected_model`, `scores`, `dag`, `iteration`, `dataset_version`, `best_score`, `best_weights_ref`, `last_eval`, `last_curation`, `last_intervention`, `last_hypothesis`, `llm_iterate_decision`, `consecutive_no_improvement`, `downward_probe_done`, `next_action`

---

### Node 9: `downward_probe`

Reached from `iterate` when `current_score >= stop_threshold`, `downward_probe_done` is False, and `current_model.tier > 0`. The intent is to confirm that a model one tier smaller cannot also clear the threshold before accepting the current model as the final result. Always terminates (hard edge to END, no loop).

- Sets `downward_probe_done = True` and `next_action = "terminate"` immediately (unconditional termination regardless of probe outcome).
- Collects all hardware-feasible models in `tier == current_tier - 1`. If none exist, returns without probing.
- Reuses `escalate._llm_choose_model` (Claude Sonnet) to pick the best candidate from the lower tier; falls back to the largest (by `int4_size_mb`).
- Trains the chosen model on `current_dataset_path` using a fixed LoRA config (rank=8, lr=2e-4, 3 epochs, batch=8) via `run_lora_training`, then evaluates with `run_eval`. Uses the honest quantized-eval path (merge → GGUF → `run_eval(..., gguf_path=...)`) when `model.quant` is set.
- If `result.f1 >= stop_threshold`: adopts the smaller model — updates `selected_model`, `best_weights_ref`, `best_score`, and `last_eval`.
- If the smaller model does not clear the threshold, keeps the current model unchanged.
- On any training/eval exception, logs a warning and keeps the current model.

**Writes to state:** `downward_probe_done`, `next_action`, and conditionally `selected_model`, `best_weights_ref`, `best_score`, `last_eval`
**Decision made:** whether to adopt a smaller model as the terminal model

---

## Full Loop

```
[hardware_research]  (pre-graph, run.py)
        │
        ▼
task_analysis → eval_setup → model_selection → curate → train → evaluate
                                                  │
                      ┌── score regressed? ───────┤
                      ↓ yes                        ↓ no
                   rollback ──────→ iterate ←───── iterate
                   (restore best              (LLM reasons
                    checkpoint,               about trajectory)
                    then re-decide)
                                                  │
                   ┌──────────────────────────────┤
                   ↓         ↓         ↓           ↓              ↓
              terminate  downward   escalate     curate          train
             (score ≥    probe      (stagnated:  (data_rebuild   (hyperparameter
              threshold  (score ≥   window delta  or surgical)    intervention)
              or budget;  thresh;   < 0.02)           │                │
              already    tier > 0;                → train          → evaluate
              probed)    not done)                    │                │
                              │               evaluate          ← (loop)
                              ▼                    │
                    train+eval tier-1         ← (loop)
                    model; adopt if
                    clears threshold
                         → END
                              │             promote tier
                              │             LLM picks model
                              │             carry dataset
                              │             reset scores
                              │             → curate → train
```

---

## Key Invariants

1. **Eval set never changes.** Fixed after `eval_setup`, never touched again. The same `E` measures every iteration.
2. **Always trains from the base model.** `train_node` never loads a prior adapter. Each run is fully determined by the current dataset. Always LoRA — no full fine-tuning — for adapter-manager deployment.
3. **Rollback re-enters the decision loop.** On regression the graph goes `evaluate → rollback → iterate`: the best checkpoint is restored and then `iterate` chooses a *different* next action. (Previously rollback went straight to `train`, which caused an endless rollback→re-train→rollback loop because re-training an identical config deterministically regresses again.) A `consecutive_no_improvement >= MAX_STALL_EVALS` backstop in `iterate` guarantees the loop escalates or terminates rather than churning.
4. **Escalation preserves the dataset.** When promoting to the next model tier, `current_dataset_path` is carried forward. Only weights and score history are reset. The new model goes through a fresh curation round on the carried-over dataset (`escalate → curate → train`).
5. **Escalation resets all score history.** The new model starts from zero — stale DAG nodes from the prior model do not pollute the rollback gate.
6. **Hyperparameter interventions skip curate entirely.** `iterate → train` directly; the dataset is held fixed to isolate the optimization effect.
7. **Curate early-returns for any intervention that isn't `data_rebuild` or `surgical`.** The `else` branch catches `"hyperparameter"`, `"rollback"`, and any other value — returning state unchanged with no disk writes.
8. **Escalation trigger is delta-based, not count-based.** `escalate` fires when the total improvement across the last 3 evaluations is < 0.02 — slow but real progress does not trigger it, only genuine plateaus.
9. **`stop_threshold` can be lowered at runtime, never raised.** `iterate_node` may lower it when the LLM identifies OOD failures, but it is clamped to `initial_stop_threshold` as a hard floor.
10. **Model selection strategy is configurable.** Set `MODEL_SELECTION_STRATEGY` in config (or `SLM_MODEL_SELECTION_STRATEGY` env var). All strategies share the same interface: `(AgentState) → AgentState`, setting `state["selected_model"]`. The `largest_first` strategy additionally uses `state["_largest_first_phase"]` to coordinate with `iterate_node`.
11. **Model pool is Qwen-only.** The `ANDROID_POOL` contains 6 Qwen-family base models (18 entries with quant variants), spanning tiers 0–3: 4 text-only + 2 multimodal (Qwen3.5). Multimodal models are fine-tuned text-only via `text_tokenizer()` (B123); Qwen3.5-2B uses the base repo, not `-GGUF` (B107).

---

## State Fields Reference

Key fields that flow through every node:

| Field | Type | Set by | Used by |
|---|---|---|---|
| `task_type` | str | `task_analysis` | all nodes |
| `feasible_models` | `list[ModelSpec]` | `task_analysis` | `model_selection` |
| `selected_model` | `ModelSpec` | `model_selection`, `escalate` | `train`, `evaluate`, `escalate` |
| `hardware_constraints` | `HardwareConstraints` | `hardware_research` (pre-graph) | `task_analysis`, `escalate` |
| `stop_threshold` | float | `task_analysis`, `iterate` | `iterate` |
| `initial_stop_threshold` | float | `task_analysis` | `iterate` (floor for threshold adjustments) |
| `eval_set` | `EvalSet` | `eval_setup` | `evaluate` |
| `train_examples` | list | `eval_setup` | `curate` |
| `current_dataset_path` | str | `curate` | `train`, `escalate` (carried forward) |
| `dataset_version` | int | `curate` | `evaluate` (logging) |
| `_largest_first_phase` | str or None | `largest_first` model selection, `iterate` | `iterate` (gates probe→smallest switch) |
| `_pending_weights_refs` | dict | `train` | `evaluate` |
| `_pending_configs` | dict | `train` | `evaluate` |
| `_pending_training_outputs` | dict | `train` | *(reserved; not currently consumed by any node)* |
| `model_baselines` | list[dict] | `evaluate` (appended on iter==1), `escalate` (best_finetuned_f1 update) | run summary, `escalate` |
| `quantize_enabled` | bool | `run.py` (init False) | *(not currently read by any node; quantization is gated on `selected_model.quant` instead)* |
| `scores` | list[float] | `evaluate`, `rollback` | `iterate`, `rollback` |
| `best_score` | float | `evaluate`, `rollback` | `iterate`, `escalate` |
| `best_weights_ref` | str | `evaluate`, `rollback` | `escalate` |
| `consecutive_no_improvement` | int | `evaluate`, `rollback` | logging only (escalation uses delta check) |
| `last_eval` | `EvalResult` | `evaluate` | `curate`, `iterate` |
| `last_intervention` | str | `iterate`, `rollback` | `curate`, `train` |
| `last_hypothesis` | str | `iterate` | `evaluate` (logging) |
| `llm_iterate_decision` | dict | `iterate` | `train`, `curate` |
| `next_action` | str | `iterate`, `escalate` | graph routing |
| `downward_probe_done` | bool | `downward_probe` (set True), `escalate` (reset False) | `iterate` (gates downward probe) |
| `dag` | list[dict] | `evaluate`, `rollback` | `evaluate`, `escalate` |
| `iteration` | int | `train`, `escalate` | `evaluate`, `escalate` |
