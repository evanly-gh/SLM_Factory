# SLM Factory Pipeline

How the LangGraph state machine is wired, what each node does, and what decisions are made at each step.

---

Three conditional edges and one hard edge control the loop:

- **`evaluate → rollback | iterate`** — decided by `should_rollback(state)`, a pure function, no LLM
- **`rollback → train`** — hard edge; rolls back to the best checkpoint then re-trains without an LLM intervention round
- **`iterate → train | curate | escalate | terminate`** — decided by `state["next_action"]`, set by the LLM inside `iterate_node`
- **`escalate → train | curate | terminate`** — decided by `state["next_action"]`; current code only ever produces `"train"` or `"terminate"`, but the graph edge map also accepts `"curate"` (latent path, not triggered by any current escalate logic)

---

## Pre-Graph Step: `hardware_research`

Runs in `run.py` **before the LangGraph graph is invoked**. Implemented in [agent/nodes/cold_start/hardware_research.py](../agent/nodes/cold_start/hardware_research.py).

- Passes the full natural-language task description to an Exa web search (`_exa_snippets`) and then to a Claude Sonnet call that returns a structured JSON object with device specs.
- Constructs a `HardwareConstraints` object with fields: `storage_mb`, `memory_mb`, `latency_ttft_ms`, `power_watts`, `target_chip`, `min_tok_s`.
- Falls back to conservative defaults (`3000 MB RAM`, `1500 MB storage`, `snapdragon_778g`) if the API call or JSON parsing fails.
- Result is written to `{RUN_DIR}/device_research.json` and passed into `state["hardware_constraints"]` for the entire graph run.

**Writes to state (initial_state):** `hardware_constraints`
**Decision made:** what the hardware budget is for model selection and hardware-gate checks

---

## Node-by-Node

### Node 1: `task_analysis`

Entry point for cold-start. Reads `state["description"]` and `state["autonomous"]`.

- If autonomous mode (or no valid `task_type` provided), calls `plan_task()` — a Claude Sonnet call that returns `task_type`, `task_name`, `labels`, `multi_label`, `schema`, `multilingual`, `exa_queries`, `benchmark`, `stop_threshold`, and `rationale` as JSON.
- Valid task types: `classification`, `NER`, `math_reasoning`, `code_generation`, `generation`. Variants (multi-label classification, schema-constrained extraction, multilingual tasks) are expressed as flags on the plan dict, not separate types.
- Filters `ANDROID_POOL` by hardware constraints, preference-sorted by task type via `filter_pool_by_task`. The sort key varies by type: `math` sorts by `(tier, -gsm8k)`, `classification` sorts by `(tier, smol_bonus - mmlu)`, `code`/`ner` prefer Qwen, and `generation` (no pool key) sorts by `(tier, int4_size_mb)`.
- Picks the first model in the preference-sorted feasible list as `selected_model`. For `math_reasoning`, further front-loads DeepSeek-R1, Qwen3, and Phi-4 models. `SLM_FORCE_MODEL` env var overrides the selection.
- Sets `state["stop_threshold"]` from the planner's output (calibrated relative to model-size SOTA, not a fixed 0.96) and records `state["initial_stop_threshold"]` as the immutable floor.

**Writes to state:** `task_type`, `task_plan`, `selected_model`, `stop_threshold`, `initial_stop_threshold`
**Decision made:** which model to start from, what the accuracy target is, which task flags apply

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

After rollback the graph always proceeds directly to `train` — **not** `iterate`. The best checkpoint is restored and the model re-trains on the existing dataset without an LLM intervention round. This avoids the LLM being asked to reason about a regression before it can observe whether the rollback alone recovers performance.

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

Routing logic after the LLM decision:

```
current_score >= stop_threshold          → "terminate"
_is_stagnant(scores)                     → "escalate"   (delta-based, window=3, min_delta=0.02)
intervention == "hyperparameter"         → "train"       (skip curate, same dataset)
intervention == "data_rebuild"           → "curate"
intervention == "surgical"               → "curate"
```

**Writes to state:** `last_intervention`, `last_hypothesis`, `llm_iterate_decision`, `next_action`, optionally `stop_threshold`
**Decision made:** primary decision node — determines everything that happens next

---

### Node 8: `escalate`

Reached when `_is_stagnant(scores)` fires inside `iterate`.

- Calls `filter_pool(hardware_constraints)` to get the hardware-feasible subset, then finds the current model's index within that filtered list and looks up `current_idx + 1`.
- If there is no next model (already at the ceiling of the feasible pool), sets `next_action = "terminate"`.
- Checks hardware constraints for the next model; if it does not fit, terminates.
- Otherwise promotes `selected_model` to the next tier and resets score history and DAG.
- The curated dataset path is **preserved** — `current_dataset_path` carries over to the new model. The curated data is still valid; only the weights and score trajectory are stale.
- Sets `next_action = "train"` — the new model goes straight to training on the existing dataset, with no LLM intervention or re-curation round.

The state reset is intentional: the old model's scores would corrupt the rollback gate for the new model.

Routing after escalate:

```
next_action = "train"      → re-train immediately on the carried-over dataset
next_action = "terminate"  → END
```

**Writes to state:** `selected_model`, `scores`, `dag`, `iteration`, `best_score`, `best_weights_ref`, `last_eval`, `last_hypothesis`, `llm_iterate_decision`, `consecutive_no_improvement`, `next_action`

---

## Full Loop

```
[hardware_research]  (pre-graph, run.py)
        │
        ▼
task_analysis → eval_setup → curate → train → evaluate
                                                  │
                      ┌── score regressed? ───────┤
                      ↓ yes                        ↓ no
                   rollback ──────→ train       iterate
                   (restore best              (LLM reasons
                    checkpoint)               about trajectory)
                                                  │
                   ┌──────────────────────────────┤
                   ↓             ↓                ↓                 ↓
              terminate      escalate           curate            train
             (score ≥        (stagnated:       (data_rebuild     (hyperparameter
              threshold       window delta      or surgical)      intervention)
              or budget)      < 0.02)               │                 │
                                  │             → train          → evaluate
                                  ▼                │                 │
                         promote model         evaluate          ← (loop)
                         carry dataset
                         reset scores
                              │
                         → train (no re-curate)
```

---

## Key Invariants

1. **Eval set never changes.** Fixed after `eval_setup`, never touched again. The same `E` measures every iteration.
2. **Always trains from the base model.** `train_node` never loads a prior adapter. Each run is fully determined by the current dataset. Always LoRA — no full fine-tuning — for adapter-manager deployment.
3. **Rollback bypasses iterate.** `should_rollback` fires before `iterate` is called. On regression, the graph goes `evaluate → rollback → train` — the LLM never sees a regressed score and is not asked to reason about it before the rollback re-train completes.
4. **Escalation preserves the dataset.** When promoting to the next model tier, `current_dataset_path` is carried forward. Only weights and score history are reset. The new model trains immediately on the existing curated data (`escalate → train`, no `curate` round).
5. **Escalation resets all score history.** The new model starts from zero — stale DAG nodes from the prior model do not pollute the rollback gate.
6. **Hyperparameter interventions skip curate entirely.** `iterate → train` directly; the dataset is held fixed to isolate the optimization effect.
7. **Curate early-returns for any intervention that isn't `data_rebuild` or `surgical`.** The `else` branch catches `"hyperparameter"`, `"rollback"`, and any other value — returning state unchanged with no disk writes.
8. **Escalation trigger is delta-based, not count-based.** `escalate` fires when the total improvement across the last 3 evaluations is < 0.02 — slow but real progress does not trigger it, only genuine plateaus.
9. **`stop_threshold` can be lowered at runtime, never raised.** `iterate_node` may lower it when the LLM identifies OOD failures, but it is clamped to `initial_stop_threshold` as a hard floor.

---

## State Fields Reference

Key fields that flow through every node:

| Field | Type | Set by | Used by |
|---|---|---|---|
| `task_type` | str | `task_analysis` | all nodes |
| `selected_model` | `ModelSpec` | `task_analysis`, `escalate` | `train`, `evaluate`, `escalate` |
| `hardware_constraints` | `HardwareConstraints` | `hardware_research` (pre-graph) | `task_analysis`, `escalate` |
| `stop_threshold` | float | `task_analysis`, `iterate` | `iterate` |
| `initial_stop_threshold` | float | `task_analysis` | `iterate` (floor for threshold adjustments) |
| `eval_set` | `EvalSet` | `eval_setup` | `evaluate` |
| `train_examples` | list | `eval_setup` | `curate` |
| `current_dataset_path` | str | `curate` | `train`, `escalate` (carried forward) |
| `dataset_version` | int | `curate` | `evaluate` (logging) |
| `_pending_weights_refs` | dict | `train` | `evaluate` |
| `_pending_configs` | dict | `train` | `evaluate` |
| `scores` | list[float] | `evaluate`, `rollback` | `iterate`, `rollback` |
| `best_score` | float | `evaluate`, `rollback` | `iterate`, `escalate` |
| `best_weights_ref` | str | `evaluate`, `rollback` | `escalate` |
| `consecutive_no_improvement` | int | `evaluate`, `rollback` | logging only (escalation uses delta check) |
| `last_eval` | `EvalResult` | `evaluate` | `curate`, `iterate` |
| `last_intervention` | str | `iterate`, `rollback` | `curate`, `train` |
| `last_hypothesis` | str | `iterate` | `evaluate` (logging) |
| `llm_iterate_decision` | dict | `iterate` | `train`, `curate` |
| `next_action` | str | `iterate`, `escalate` | graph routing |
| `dag` | list[dict] | `evaluate`, `rollback` | `evaluate`, `escalate` |
| `iteration` | int | `train`, `escalate` | `evaluate`, `escalate` |
