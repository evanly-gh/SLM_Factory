# SLM Factory Pipeline

How the LangGraph state machine is wired, what each node does, and what decisions are made at each step.

## Long-run checkpoint/requeue contract

The checkpoint/requeue-enabled weeklong jobs use Slurm `--time=7-00:00:00`,
`--signal=B:USR1@7200`, and `--requeue`. The batch shell forwards `USR1` to the
pipeline, waits for its atomic JSON and SQLite checkpoint, verifies durable resume
state, and only then calls `scontrol requeue` with the same stable run directory.
This creates segment rollover with continuous progress rather than a terminal
seven-day run.

Those scripts set `SLM_MAX_WALLCLOCK_S=0`. The application guard is aggregate
across resume segments, so a nonzero value such as the former 6d20h setting would
silently terminate before Slurm's 6d22h signal and prevent the safe requeue path.
Non-requeue runs may still set a nonzero application guard when graceful
termination before their scheduler limit is desired.

`run-manifest.json` stores both the resume compatibility fingerprint and a
human-readable `effective_config` snapshot. Resume recomputes every
resume-sensitive setting after the runner loads `.env`, then reports the
specific stored/current values that drifted. The shell's
`durable_resume_available` probe checks only structural durability and agreement
between stored manifest/checkpoint identities; it does not import credentialed
runtime config. The run-local curation path is explicit in the environment and
graph state, so concurrent runs and resumed segments never read project-root
history.

On `TERM`, the batch shell forwards the signal once, then waits up to
`SLM_TERM_GRACE_S` (default 120 seconds) for Python to publish SQLite/JSON
checkpoint state, summaries, and observability artifacts. It preserves the
pipeline's final exit status; if the child remains hung after the grace period,
the shell sends `KILL`, reaps it, and exits with that forced-termination status.
`TERM` takes precedence over requeue, while the existing `USR1` durable
checkpoint verification and requeue path remains unchanged.

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

**Model pool — `ANDROID_POOL` (official Qwen only):** The pool contains **18 entries** — 6 official-Qwen base models each expanded to three quantization siblings (`bf16`, `Q4_K_M`, `Q8_0`). See [`docs/model_pool.md`](model_pool.md) for the full capability write-up.
- `Qwen/Qwen3-0.6B` — text; Tier 0 Q4 seed; Table-8 Base proxies are not attached
- `Qwen/Qwen3-1.7B` — text; Tier 1 Q4 seed; non-thinking MMLU-Pro 40.2 / Redux 64.4
- `Qwen/Qwen3-4B-Instruct-2507` — text, non-thinking; Tier 3 Q4 seed; MMLU-Pro 69.6 / Redux 84.2
- `Qwen/Qwen3.5-0.8B` — multimodal; Tier 0 Q4 seed; non-thinking MMLU-Pro 29.7 / Redux 48.5
- `Qwen/Qwen3.5-2B` — multimodal; Tier 2 Q4 seed; non-thinking MMLU-Pro 55.3 / Redux 69.2
- `Qwen/Qwen3.5-4B` — multimodal; Tier 3 Q4 seed; MMLU-Pro 79.1 / Redux 88.8, with mode unspecified by that card table

No Qwen2.5, no distilled models, no thinking-only models. Unsupported GSM8K rows are not attached to post-trained pool IDs. Tier coverage after quant expansion: tier 0 ×2, tier 1 ×3, tier 2 ×5, tier 3 ×8. `ANDROID_POOL` is sorted by `(tier, size_mb)`. Every variant has an exact selector (`model_id@bf16|Q8_0|Q4_K_M`); a legacy ambiguous bare ID deterministically chooses the lowest-peak-RAM feasible sibling. Capability measurements retain metric, artifact, mode, protocol, and source.

> **Multimodal handling (B136):** Qwen3.5 is a "Causal LM with Vision". Text-only LoRA uses Unsloth `FastVisionModel` with `finetune_vision_layers=False`; merge-for-quantization also selects the vision loader. Qwen3.5 uses base Transformers repositories, never `-GGUF` repositories for training.

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

All strategies store an exact `ModelSpec` variant. `SLM_FORCE_MODEL` and orchestrator
responses should use `model_id@bf16|Q8_0|Q4_K_M`; legacy bare IDs use the documented
lowest-peak-RAM default when several siblings are feasible.

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

Probe 3 variants (smallest, middle, largest from feasible set) with the configured representative probe training (3 epochs by default), fit F1 against log deployment peak RAM, then select the variant whose peak RAM is closest to the hardware memory budget while still meeting the accuracy goal. Duplicate footprints are averaged; fewer than two unique footprints skip fitting. Quantized probes build/reuse and score their exact GGUF.

**Algorithm:**
1. Pick 3 candidates (largest, middle, smallest) from `feasible_models`.
2. For each: LoRA probe → exact-variant eval → record `(log(params), f1)`.
3. Fit `f1 = a * log(params) + b` via least-squares.
4. Filter models whose predicted f1 ≥ `stop_threshold`.
5. From qualifiers, pick the one closest to `hardware_constraints.memory_mb`.
6. If no qualifier, fall back to the highest-capability model.

#### `orchestrator_choice`

Let the orchestrator LLM choose an exact selector directly, using the task type, task plan,
sourced capability records, and hardware constraints as context. No probing overhead —
single API call. Bare-ID replies use deterministic compatibility resolution. Call failure or
an unknown selector chooses the lowest-peak-RAM feasible variant, never arbitrary max-BF16.

**Writes to state:** `selected_model`, optionally `_largest_first_phase`
**Decision made:** which model to fine-tune first

---

### Node 2: `eval_setup`

Runs once. The eval set is fixed from this point forward and never modified again.

- If `state["task_plan"]` exists, calls `acquire_dataset()`. The default order is
  checksum-verified local bundle, deterministic known-benchmark loader, then agentic
  discovery; readiness mode may exercise discovery first but still falls back locally.
  A local bundle is eligible only by exact recognized benchmark alias or a strong
  task+label+schema match; an unrelated same-task bundle is skipped rather than aborting
  the acquisition ladder.
  Agent-discovered APPS/MBPP/BC5CDR/GSM8K/SAMSum IDs use their task-aware converters, and
  every discovered result must pass row-schema, task-integrity, and normalized-overlap
  validation. APPS uses the official introductory train/test splits and retains solutions,
  starter code, difficulty, and call-based or stdin/stdout tests.
- If no plan exists and `task_type == "classification"`, falls back to the hardcoded SMS Spam loader. For any other task type without a plan, raises `NotImplementedError`.
- Calls `build_eval_set()` which partitions test examples into `E = E_pos ∪ E_neg ∪ E_boundary`.
- Difficulty labeling deduplicates quant siblings by `model_id`, chooses the smallest and
  largest unique parameter-capacity endpoints, and intentionally evaluates their BF16/base
  model IDs. Quant siblings are deployment formats, not separate capacity endpoints.
- Persists the eval set to `artifacts/eval_set.json` on disk.

**Writes to state:** `train_examples`, `eval_set`
**Decision made:** none — deterministic data loading

---

### Node 3: `curate`

Builds or augments the training dataset. Behavior branches on `state["last_intervention"]` (defaults to `"data_rebuild"` if unset):

| Condition | What happens |
|---|---|
| `"data_rebuild"` | Executes the validated declarative rebuild plan |
| Anything else (including `"hyperparameter"`) | **Returns immediately** — dataset held fixed, no curation |

**Dataset sizing by task type (data_rebuild fallback targets):** the autonomous planner can
request a different size; `CURRICULUM_SIZE_FLOOR` (default 1000) and
`DATA_SIZE_CEILING` (default 10000) are then enforced.

| `task_type` | N total | Rationale |
|---|---|---|
| `classification` | 150 | Simple decision surface |
| `multi_label_classification` | 300 | Label co-occurrence patterns need more coverage |
| `NER` | 200 | Entity diversity requirement |
| `structured_extraction` | 400 | Schema field coverage + negative schema examples |
| `math_reasoning` | 700 | Step-by-step CoT chains; quality over quantity |
| `code_generation` | 300 | Function diversity + execution harness coverage |
| `multilingual` | 400 | Language pair coverage adds dimensionality |
| `generation` | 600 | Open-ended; needs diverse inputs |

Each plan has one primary strategy and at most two support strategies selected from:
`resample_existing`, `preserve_elite_resample`, `mine_new_real_source`,
`source_diversification`, `difficulty_weighted_sampling`, and
`targeted_synth_positive`. The last strategy is eligible only for classification and NER.
Fractions, target/new/synthesis row counts, paid acquisition rounds, query variants,
difficulty weights, aggregate confusion pairs, pattern hints, and elite
provenance/version are bounded and normalized before execution. Additive strategies require
positive material budgets; only one sampling strategy may appear, and a no-op source
diversification is explicitly rewritten and logged.

Source mining uses clean local and deterministic benchmark data before bounded,
process-isolated paid discovery. New rows are deduplicated against training and held-out
texts and checked against source/split restrictions. Every paid round is durably reserved
in `acquisition-reservations.jsonl` under the stable run directory before the provider
call, then reconciled as completed or
failed; pending/failed reservations remain spent across crashes and resumes. Mining
zero-novelty does not override novelty produced by another composed strategy. Accepted
real rows, including source metadata and mined provenance, are also merged into
`state["train_examples"]`; this persisted pool survives checkpoints and remains available
to future rebuilds even when a particular capped dataset version does not select the row.

Elite rows are selected deterministically by quality/provenance and must resolve to an
existing declared dataset version; they pass the same balance, dedup, length, entity, and
eval gates as every other row. Difficulty allocation never fills from zero-weight buckets;
an unavailable nonzero quota leaves rows unfilled under a logged fallback policy. Positive
synthesis uses only non-eval train anchors and local Qwen; math, code, and open generation
remain gold/CoT-only. After all quality controls, replay, and composed strategies,
`target_rows` is the hard final cap. A final normalized eval-text gate protects every saved
dataset.

In production mode, mixes the `replay_buffer` into the dataset at this step.

Saves the dataset to `artifacts/dataset_v{N}.jsonl` and increments `dataset_version`.

**Writes to state:** `train_examples`, `current_dataset_path`, `dataset_version`,
`last_curation`, `data_rebuild_plan`, `data_rebuild_plan_identity`,
`source_acquire_rounds_used`
**Decision made:** none — executes the intervention type the LLM decided in the prior `iterate` call

---

### Node 4: `train`

Trains a single LoRA configuration. Always produces a LoRA adapter (no full fine-tuning) for on-device adapter-manager deployment.

Config selection logic:
- The bounded search is rank `{4,8,16,32,64}`, alpha `{r,2r,4r}`, dropout
  `{0,.05,.1}`, weight decay `{0,.01,.05,.1}`, learning rate
  `[1e-5,5e-4]`, and epochs `[1,8]`.
- `micro_batch_size` and `gradient_accumulation_steps` are each selected from
  `{1,2,4,8}`. `effective_batch_size` is derived as their product (maximum 64);
  micro batch controls per-step activation memory while accumulation changes effective
  batch without increasing that peak. Legacy `batch_size` is accepted as a micro-batch
  alias, but conflicting aliases and false effective-batch claims are rejected.
- Numeric orchestrator suggestions are deterministically snapped/clamped into this space.
  Runtime `TrainingConfig` is strict and rejects values outside it.
- The first iteration defaults to `r=16`, alpha 32, dropout 0, weight decay .01,
  3 epochs, lr `2e-4`, micro batch 8, and accumulation 1. Data-only changes and
  rollback carry every winning optimizer field forward.
- Trial identity is the current dataset identity plus every normalized field above.
  Pruned trials and trained candidates that lost to the zero-shot baseline count as
  tried. An exact repeat on the same dataset is rejected and replaced with a
  deterministic untried neighbor; carrying the same optimizer config onto a rebuilt
  dataset is allowed.

Always trains from the base model — never loads a prior adapter. Qwen chat formatting passes
`enable_thinking=False`; a Qwen tokenizer without a chat template fails clearly instead of
silently training under another prompt mode. Code training and eval share one prompt builder
that includes starter code, required signatures/entry points, and execution-mode
instructions. The APPS code profile uses a 4096-token context, reserves up to 1024 tokens
for output, and rejects overlength prompts or training rows rather than silently truncating
target-critical content. Stores the `weights_ref` in `state["_pending_weights_refs"]`.
Both text and multimodal PEFT paths receive the selected alpha/dropout. `SFTConfig`
always receives micro batch, gradient accumulation, and weight decay, including the
no-validation fallback path.
Nonzero LoRA dropout can leave Unsloth's optimized zero-dropout path and make training
materially slower; select `.05` or `.1` only when the overfitting hypothesis justifies
the cost.
Training rows are pre-tokenized from the same `enable_thinking=False` chat template used
for serving. Their explicit completion mask drives a Trainer-compatible collator:
all prompt labels are `-100`, while assistant/CoT/code target tokens retain their token
IDs. This applies equally to text models and text-only training through `FastVisionModel`.
If early-stop/checkpoint training fails after any optimizer work, the failed trainer and
model are discarded. The fallback reloads a fresh base model plus the exact same LoRA H,
rebuilds tokenized rows with the fresh tokenizer, and trains all original rows without
the validation split. The `except` suite captures only an immutable error summary and
exits before cleanup; no exception or traceback remains active while the failed object
graph is cleared, `gc.collect()`/`empty_cache()` run, or the fresh stack reloads. This
prevents traceback-held tensors from surviving into fallback. It never continues
partially trained weights or the reduced split.
Pending configs and DAG identities serialize every canonical field plus the legacy
`batch_size` alias. Legacy rank/LR/epochs/batch checkpoints are normalized on read at
their consumer boundary. The resume compatibility fingerprint includes the LoRA search
contract version, maximum effective batch, and assistant-only SFT loss-contract version.

**Writes to state:** `_pending_weights_refs`, `_pending_configs`, increments `iteration`
**Decision made:** none — executes what curate set up

---

### Node 5: `evaluate`

Scores the trained config against the fixed eval set.

- Calls `run_eval()` for the `weights_ref` — dispatches to three scorer modules by task type:
  - `eval.scorers.classification` — accuracy/F1 for `classification`
  - `eval.scorers.ner` — entity span-F1 for `NER`
  - `eval.scorers.generation` — handles `math_reasoning` (final-answer exact match),
    `code_generation` (execution pass@1), and `generation` (required local Qwen3.6
    LLM-as-judge) by inspecting `task_type` internally
- Open-generation judging uses `eval/judge_client.py` against the configured local
  OpenAI-compatible vLLM endpoint. Before scoring, it requires the exact `JUDGE_MODEL`
  Qwen3.6-family identity from `/models`, verifies any completion `response.model`, and
  disables thinking. Loopback, `localhost`, and Unix-socket endpoints are accepted by
  default; every remote hostname—including private cluster hosts—hard-fails unless
  `SLM_JUDGE_ALLOW_REMOTE=1` is explicitly set. Long-run scripts do not enable that opt-in,
  and proxy environment variables are ignored.
- Question, gold, and prediction are serialized as fields in a marked untrusted JSON block.
  The system prompt explicitly forbids following embedded instructions.
- Normalized triples are cached in a process-locked JSONL ledger under the stable run's
  `artifacts/` directory. Keys include the exact model and prompt version/fingerprint, so
  disposable eval workers and restarted processes reuse valid scores; corrupt lines are
  ignored. An in-memory cache layers over the disk ledger.
- Unique cache misses use a bounded sliding window of concurrent requests. Results remain
  in eval-set order; the first failure stops new submissions and cancels pending work.
- Judge output must be exactly one finite number in `[0,1]`. A missing or unreachable
  endpoint, model mismatch, request error, malformed response, or out-of-range value aborts
  the evaluation. There is no cloud fallback and no conversion of judge failures into model
  scores. Local preflight/request cost and timing events are process-safe and report
  `provider=local` at `$0`.
- APPS predictions execute every preserved case as either call-based functions or
  stdin/stdout programs. Each case gets its own timeout, while a configurable per-problem
  total wall deadline (`SLM_APPS_PROBLEM_TIMEOUT_S`, default 6 seconds) bounds pathological
  case collections without subsampling. Budget exhaustion is reported separately from a
  per-case timeout, and diagnostics record cases executed/total. MBPP assertions remain the
  lightweight CI smoke. Expected outputs and
  tests stay in the trusted controller; candidate workers receive one input at a time and
  inherit no success channel. Workers use process-group cleanup, a sanitized environment,
  and rlimits. This is trusted-benchmark isolation, not a hostile-code sandbox. It also
  does not provide seccomp and is trusted-input only; arbitrary untrusted submissions
  must not be executed.
- APPS compatibility follows the pinned `codeparrot/apps_metric` `testing_util.py` prelude
  and argument conventions: common stdlib helpers, optional `numpy as np`, single-threaded
  BLAS/OpenMP runtimes, tuple/list normalization, integer-key dictionaries, ListNode
  construction, and the dataset's wrapped Two Sum scalar argument. The official singleton
  expected-output wrapper is accepted only for structured actual values; scalar/list false
  passes remain rejected. Stdin comparison normalizes line/whitespace tokenization and
  numeric spelling while preserving token order and duplicate multiplicity. The reference
  harness's global unordered-set fallbacks are intentionally not copied because they can
  turn ordered-output errors into false passes; no problem-specific unordered checker is
  declared by this pinned introductory source. Codeforces 1294F uses a scoped semantic
  checker for its explicitly non-unique vertex triple. Rows whose supplied gold solutions
  still cannot satisfy the preserved exact/semantic checks are retained for provenance,
  marked `runner_compatible=false`, counted by reason in the manifest, and skipped by the
  eval loader.
- On the first iteration of each exact selector, measures a zero-shot baseline. Q4/Q8
  baselines build/reuse and score that exact base GGUF; a BF16 score is never credited to a
  quantized identity. Generation baseline judge-infrastructure failures are re-raised, so
  an unavailable judge cannot be recorded as a false `0.0` baseline.
- HF inference renders the same non-thinking template as training. GGUF uses
  `chat_template_kwargs={"enable_thinking": false}` when supported; the installed
  llama-cpp-python 0.3.34 signature lacks that parameter. Hybrid Qwen3/Qwen3.5 use the
  verified empty-think ChatML prefix; non-thinking-only Qwen3-4B-Instruct-2507 uses its
  plain assistant prefix without think tags. Unsupported unknown templates fail clearly.
- Updates `best_score`, `best_weights_ref`, `consecutive_no_improvement`.
- Appends a node to the linear DAG (`state["dag"]`) with the exact selector and full
  `π = (D, H, S)` triple. `π.D` snapshots dataset path/version, normalized rebuild plan,
  plan identity, rebuild config, composition/yield, and the matching aggregate evaluation
  state needed by rollback.
- Writes the iteration record to `data-curation.md` via `CurationLog`, including actual
  total, initial gold, mined real rows, generated positive rows, replay rows, strategy
  composition, source novelty/yield, score band, hardware PASS/FAIL, config labels, and
  hypothesis. `_llm_iterate` reads this trajectory verbatim (or compacted), so its
  intervention decision sees the same aggregate counts.

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
- Restores `best_weights_ref`, `best_score`, dataset path/version, curation composition,
  rebuild plan/identity, and matching test/eval reports from the highest-scoring
  non-pruned DAG node.

After rollback the graph proceeds to `iterate` (**changed** — it previously went straight to `train`). Restoring the best checkpoint and re-training the *same* dataset + hyperparameters is (near-)deterministic, so it reproduces the same regressing score and rolls back again — an endless loop that was observed to burn the entire turn budget (e.g. ARC-Challenge and GSM8K oscillating for 20+ iterations at a fixed best score). Routing to `iterate` forces the next step to be a genuinely different action: a `data_rebuild` with a rotated sampling seed, a hyperparameter change, escalation to a bigger model, or a clean termination via the stall backstop.

**Writes to state:** `scores`, `dag`, `best_weights_ref`, `best_score`, `last_intervention`, `consecutive_no_improvement`
**Decision made:** none — all logic is rule-based

---

### Node 7: `iterate`

The only node with an LLM call that controls graph routing. This is the EXPAND operator (paper §2.2 Eq. 2).

Calls `_llm_iterate()` — one bounded, tool-free Claude decision:

1. Compacts the `data-curation.md` trajectory via `compact_trajectory()` if it exceeds ~8,000 tokens.
2. Feeds the compacted trajectory, score history, aggregate difficulty/confusion report,
   source novelty/yield, prior plan identities, prior hypothesis, and remaining budgets
   to the LLM. Raw eval text is never included.
3. Makes exactly one tracked `iterate` call. The model has no bound tools, shell,
   filesystem, web-search, or eval-artifact access. A malformed/prose/tool-use response
   receives at most one fresh tracked `iterate_json_reask`; no tool request is executed.
4. Parses strict JSON: `{intervention, hypothesis, data_rebuild?,
   hyperparams?, threshold_adjustment?}`.
   Hyperparameter payloads are normalized to the complete bounded config before being
   stored. The prompt includes full tried/pruned identities, derived effective batch,
   deployment peak RAM, and the device memory budget.
   The intervention enum and required payload are validated (`hyperparams` for
   hyperparameter and a bounded declarative plan for data rebuild). Exact rebuild-plan
   repeats, including pruned and zero-yield trials, are deterministically rotated.
   Unknown top-level/nested keys, wrong scalar types, cross-intervention payloads, and
   Python-literal responses are rejected. `threshold_adjustment` must be an object whose
   `new_threshold` is null or finite numeric; a numeric adjustment requires a non-empty
   reason. Any string containing normalized held-out text is rejected recursively.
   Validation is
   repeated at the node boundary, so malformed provider/mocked decisions enter the
   logged safe fallback and cannot crash later threshold handling.

**Threshold adjustment:** The LLM may include a `threshold_adjustment.new_threshold` value in its decision. If provided, `iterate_node` lowers `stop_threshold` to that value, clamped to `initial_stop_threshold` as a floor. This is used when the LLM identifies that the dominant failure cluster reflects a genuine model capacity limit (world knowledge gaps, reasoning chains longer than the model can produce, adversarial OOD inputs) rather than a data or hyperparameter problem. The floor is enforced in code — the LLM cannot set it below `initial_stop_threshold`.

Falls back to `apply_iteration_policy()` (pure score-band rules) if the LLM call fails.

**Stagnation detection:** escalation is triggered by a sliding-window delta check, not a consecutive-no-improvement count:

```python
STAGNATION_WINDOW = 50       # env: SLM_STAGNATION_WINDOW
STAGNATION_MIN_DELTA = 0.02  # minimum cumulative improvement over that window to avoid escalation
```

Once the full window exists, chronological gain is
`max(window) - window[0]`. A gain below `.02` is stagnant; the exact `.02`
boundary is non-stagnant with floating-point tolerance.

**Stall backstop (`MAX_STALL_EVALS = 50`, env: `SLM_MAX_STALL_EVALS`):**
`should_rollback` pops the regressing score, so on a rollback→re-decide churn the
stagnation window can stay short and never fire. `iterate_node` escalates when
`consecutive_no_improvement >= MAX_STALL_EVALS`; that counter survives rollback.

Routing logic (evaluated in this order):

```
no scores yet                              → "train"       (first iteration, nothing to reason about)
iteration*2 >= turn_budget                → "terminate"   (turn budget exhausted; ~2 turns per iteration)
current_score >= stop_threshold:
  hw_gating_enabled AND hw fails          → continue (escalate if stagnant, else train or curate)
  not downward_probe_done AND tier > 0    → "downward_probe"  (try smaller model once before accepting)
  else                                    → "terminate"
_is_stagnant(scores)                       → "escalate"   (chronological gain, default window=50, min_delta=0.02)
intervention == "hyperparameter"           → "train"       (skip curate, same dataset)
intervention == "data_rebuild"             → "curate"
```

**Writes to state:** `last_intervention`, `last_hypothesis`, `llm_iterate_decision`, `next_action`, optionally `stop_threshold`
**Decision made:** primary decision node — determines everything that happens next

---

### Node 8: `escalate`

Reached when `_is_stagnant(scores)` fires inside `iterate`.

- Calls `filter_pool(hardware_constraints)` and selects the nearest higher non-empty
  peak-RAM tier.
- If no higher feasible tier exists, sets `next_action = "terminate"`.
- Calls `_llm_choose_model` with `direction="up"` on that tier. Failure/unknown output
  deterministically chooses the lowest-peak-RAM exact variant in the target tier, never
  arbitrary max-BF16.
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
- Selects the nearest untried lower non-empty peak-RAM tier and can continue progressively
  lower after a successful adoption.
- Reuses `escalate._llm_choose_model` with `direction="down"` to choose an exact selector;
  call failure chooses that lower tier's lowest-peak-RAM variant.
- Trains on `current_dataset_path` with an explicit fixed config: rank 16, alpha 32,
  dropout 0, weight decay .01, lr `2e-4`, 3 epochs, micro batch 8, accumulation 1,
  and derived effective batch 8.
- The canonical fixed H is copied into `downward_probe_pending`,
  `downward_probe_history.fixed_H`, and every attempt record before training. The
  exact serialized H is passed to the trainer, so checkpoint/resume history identifies
  what was actually realized.
  Quantized probes call the shared cache-aware GGUF builder and pass the exact artifact to
  `run_eval`.
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
             (score ≥    probe      (stagnated:  (structured     (hyperparameter
              threshold  (score ≥   window delta  data_rebuild)   intervention)
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
7. **Curate executes only `data_rebuild`.** Hyperparameter, rollback, and unknown values
   return unchanged with no disk writes.
8. **Escalation uses chronological gain.** With the default 50-score window,
   `max(window) - window[0] < 0.02` escalates; slow progress that reaches the
   boundary does not.
9. **`stop_threshold` can be lowered at runtime, never raised.** `iterate_node` may lower it when the LLM identifies OOD failures, but it is clamped to `initial_stop_threshold` as a hard floor.
10. **Model selection strategy is configurable.** Set `MODEL_SELECTION_STRATEGY` in config (or `SLM_MODEL_SELECTION_STRATEGY` env var). All strategies share the same interface: `(AgentState) → AgentState`, setting `state["selected_model"]`. The `largest_first` strategy additionally uses `state["_largest_first_phase"]` to coordinate with `iterate_node`.
11. **Model pool is official-Qwen-only.** The `ANDROID_POOL` contains 6 official Qwen base models (18 variants), spanning tiers 0–3: 3 text-only Qwen3 (0.6B, 1.7B, 4B-Instruct-2507) + 3 multimodal Qwen3.5 (0.8B, 2B, 4B). No Qwen2.5, distilled, or thinking-only models. Qwen3.5 is fine-tuned text-only via `FastVisionModel` (B136), base repos only (not `-GGUF`, B107). See `docs/model_pool.md`.

Structured data rebuilds and complete dataset rollback are implemented together: every
winning `π.D` carries the dataset artifact identity, declarative plan/config/composition,
and matching aggregate evaluation state.

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
| `train_examples` | list | `eval_setup`, `curate` (mined real rows) | `curate` |
| `curation_log_path` | str | runner | `evaluate`, `iterate` |
| `current_dataset_path` | str | `curate` | `train`, `escalate` (carried forward) |
| `dataset_version` | int | `curate` | `evaluate` (logging) |
| `data_rebuild_plan` | dict | `iterate`, `curate`, `rollback` | `curate`, `evaluate` |
| `data_rebuild_plan_identity` | str | `iterate`, `curate`, `rollback` | repeat blocking, DAG lineage |
| `source_acquire_rounds_used` | int | `curate` + durable reservation ledger | `iterate`, plan validation |
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
