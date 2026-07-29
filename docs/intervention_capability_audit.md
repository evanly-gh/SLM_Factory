# Intervention and capability audit

Effective date: **2026-07-21**

This document records current behavior and recommendations. The recommendations are not
implemented here.

## 1. Capability data: storage, meaning, and injection

### Authoritative locations

- `config/android_pool.py::ModelSpec` stores deployment resources plus a tuple of
  `CapabilityMeasurement(metric, value, artifact, mode, protocol, source)` records.
- `config/model_capabilities.md` is the offline, human-readable, sourced capability document.
- `config/model_capabilities.py::capability_sections` caches that document and returns the
  shared caveat plus only the requested model sections.
- `docs/model_pool.md` is the operator-facing pool reference.

Missing measurements are `None`/`not reported`; they are not numeric zero. MMLU, MMLU-Pro,
and MMLU-Redux remain separately named. Measurements rank only when metric, mode, and
protocol are all identical. Unsupported Qwen3 report GSM8K rows are not attached to
post-trained pool artifacts.

### Active injection points

- Initial LLM selection:
  `agent/nodes/cold_start/model_selection/orchestrator_choice.py::orchestrator_choice_node`.
- Upward escalation:
  `agent/nodes/escalate.py::_llm_choose_model`.
- Downward candidate selection:
  `agent/nodes/downward_probe.py::downward_probe_node` calls the same
  `escalate._llm_choose_model`, so it receives the same capability sections and metric
  contract.
- Autonomous planning:
  `agent/task_planner.py::_pool_summary` does not inject prose sections, but renders explicit
  optional metric names, modes, and source URLs with the same comparability contract.

### Sorting behavior

- `config/android_pool.py::filter_pool` sorts only by deployed variant tier and size.
- `filter_pool_by_task` uses a benchmark within a tier only when every candidate has a
  present measurement with the same metric/mode/protocol comparison key. Mixed metrics,
  differing protocols, or any missing GSM8K causes deterministic resource ordering.
- The main cold-start hardware path subsequently sorts feasible variants largest to smallest
  by file size for strategy contracts
  (`agent/nodes/cold_start/task_analysis.py::task_analysis_node`); it does not use capability
  scores.
- Interpolation fits F1 against log deployment peak RAM, averages duplicate footprints, and
  declines to fit fewer than two unique x-values.
- Test-agent difficulty endpoints deduplicate quant siblings by base `model_id`, choose
  smallest/largest unique parameter-capacity models, and intentionally run BF16/base IDs.

### Tier semantics

`config/android_pool.py::_ram_tier` assigns tiers from each deployed variant’s configured
peak RAM:

- Tier 0: below 750 MB
- Tier 1: 750–1499 MB
- Tier 2: 1500–2499 MB
- Tier 3: at least 2500 MB

Tiers are not parameter-count or base-model capability classes. Q4_K_M, Q8_0, and BF16
variants of the same model can occupy different tiers.

Each variant’s stable identity is `model_id@bf16|Q8_0|Q4_K_M`. Selection prompts, force
overrides, DAG/baseline/escalation histories, and convergence references preserve it. Legacy
bare IDs choose the lowest-peak-RAM feasible sibling rather than list order.

### Evaluation artifact and mode

- Q4/Q8 zero-shot baselines build/reuse the selected base GGUF before scoring.
- Quantized interpolation and downward probes use the same cache-aware GGUF helper.
- HF training and inference pass `enable_thinking=False` through the tokenizer template.
- The installed llama-cpp-python 0.3.34 `create_chat_completion` signature has no
  `chat_template_kwargs`; hybrid Qwen3/Qwen3.5 GGUF therefore uses the verified ChatML
  empty-think prefix. Non-thinking-only Qwen3-4B-Instruct-2507 uses the plain assistant
  prefix from its HF template, without empty think tags. Unknown templates fail clearly.
- The installed llama-cpp shared library cannot load on the login-node runtime because its
  `libstdc++` lacks `GLIBCXX_3.4.29`; actual GGUF execution still requires a compatible
  compute/runtime environment.

## 2. Intervention selection and routing

### Decision sources

`agent/nodes/iterate.py::iterate_node` chooses the next action:

1. Hard guards terminate on turn or wall-clock budget.
2. Stagnation/stall below the target bypasses the LLM and routes to upward escalation
   (or terminates a `largest_first` feasibility probe).
3. Otherwise `_llm_iterate` makes one tool-free bounded call for `data_rebuild` or
   `hyperparameter`, with at most one JSON-only reask.
4. On a non-fatal LLM failure, a test-agent recommendation is preferred; the static score
   bands are the final fallback.
5. A score at/above threshold terminates or enters downward probing, subject to strategy and
   hardware gates.

Static fallback bands are below 0.80 → broad data rebuild, 0.80–0.95 →
hyperparameter, and at least 0.95 → aggregate-confusion refinement through a
task-eligible data-rebuild strategy. These are fallback heuristics, not enforced
boundaries on a valid LLM decision.

### Data rebuild

Current behavior is in `agent/nodes/curate.py::curate_node`:

- Executes one normalized primary strategy and up to two support strategies.
- Supports existing-data resampling, elite-preserving resampling, bounded real-source
  mining, source diversification, train-only difficulty weighting, and classification/NER
  positive synthesis.
- Uses deterministic plan identities and seeds; exact pruned or zero-yield repeats count as
  tried.
- Classification and NER can use local-Qwen positive augmentation. Math, code, and open
  generation are gold/CoT-only: wrong code/answers and rejected generation responses are
  never stored as positive SFT targets.
- Real-source mining tries local and deterministic benchmark sources before process-isolated
  paid discovery. Local candidates require an explicit benchmark match or strong
  task+label+schema agreement. Paid rounds use a locked append-only reservation ledger under
  the stable run directory; reservation precedes the call and remains spent after
  completion, failure, or crash.
- Gold and synthesis anchors come only from normalized-decontaminated `train_examples`.
  Held-out rows influence only aggregate difficulty/confusion counts.
- Requires positive strategy-specific budgets and rejects redundant sampling compositions.
- Resolves elite provenance/version before execution, ranks elite candidates
  deterministically by quality, and applies all normal quality/firewall gates.
- Applies task-specific balancing, length filtering, entity caps, deduplication, and a final
  normalized eval-text firewall. `target_rows` is the post-QC cap across elite, mined,
  sampled, synthesized, and replay rows; zero-weight difficulty buckets never backfill.
- Computes overall novelty across all composed strategies and attributes strategy
  composition from explicit causal row origins.
- Persists row provenance/version and complete plan/config/composition/yield metadata.
- `data-curation.md` records strategy composition and source novelty/yield for the next
  iterate decision.

Important limitations:

- If the local synthesis endpoint is unavailable, rebuild becomes gold-only.
- Open-generation rejected-answer augmentation remains unavailable until a preference
  objective exists or a task-specific verifier can certify positive responses.
- Production `data_rebuild` raises when `eval_set` is absent. The production graph does not
  create that set, so callers must populate it.

### Hyperparameter intervention

Routing and config construction are in `agent/nodes/iterate.py` and
`agent/nodes/train.py::_build_config`:

- Routes directly to `train`; the current dataset is held fixed.
- A valid LLM proposal can set rank, learning rate, epochs, and per-device batch size.
- If the LLM/fallback supplies no config, the best prior config is retained; for a
  config-less hyperparameter fallback, rank steps to the next higher untried rung when one
  exists.
- Every training run starts from the base model, never the previous adapter.
- Previously tried configurations remain visible in the DAG even when pruned by rollback.

### Positive-synthesis rebuild strategy

`targeted_synth_positive` is a data-rebuild strategy, not a separate intervention.
It is eligible only for classification and NER, requires the local synthesis endpoint,
uses normalized non-eval training anchors, and consumes only aggregate confusion pairs and
a bounded pattern hint. NER outputs must contain valid JSON, non-empty spans present in the
rewritten text, and source-approved entity types. Other task families cannot select this
strategy; they use real-data sampling/mining and optional CoT instead.

### Rollback

`agent/nodes/rollback.py` rolls back whenever the newest score is lower than the immediately
preceding score:

- Pops the regressing score.
- Marks the newest DAG node pruned.
- Restores weights/score plus dataset path/version, curation state, rebuild plan/identity,
  and matching aggregate test/eval reports from the best non-pruned node.
- Routes back to `iterate` so the system chooses a different next action.

### Upward escalation

`agent/nodes/escalate.py::escalate_node`:

- Finds the nearest non-empty higher peak-RAM tier among hardware-feasible variants.
- Uses the sourced capability prompt to choose within that tier.
- On choice failure, selects the lowest-peak-RAM exact variant within the already selected
  higher tier.
- Saves the completed model’s trajectory in `escalation_history`.
- Clears inference cache, selects the new variant, resets per-model score/DAG/iteration state,
  and routes to a new data rebuild.
- Carries forward the acquired source data/current dataset path until curate writes the new
  version.

Because tiers are quant/peak-RAM specific, escalation can move to a different quant variant
of the same base model.

### Downward re-exploration

`agent/nodes/downward_probe.py::downward_probe_node` is reachable after convergence only for
`interpolation` and `orchestrator_choice`:

- Records the converged model as the fallback answer.
- Considers the nearest untried lower tier first.
- Asks whether another probe is worthwhile; call failure uses `margin >= 0.03`.
- Uses the same sourced capability chooser as escalation to select a lower-tier candidate.
- Choice failure selects the lowest-peak-RAM exact variant within that lower tier.
- Trains once on the converged dataset and evaluates the actual candidate quant path.
- Adopts a candidate that clears the threshold and continues lower; stops at the first
  lower-tier failure or operational error.
- Always terminates the graph when downward search finishes.

The probe uses an explicit fixed training config: rank 16, alpha 32, dropout 0,
weight decay .01, learning rate `2e-4`, 3 epochs, micro batch 8, gradient
accumulation 1, and effective batch 8.
That canonical H is serialized in the pending probe, the history-level `fixed_H`,
and each attempt, then passed unchanged to training.

## 3. LoRA and optimizer setting handling

### Rank

- Prompt-allowed and trainer-valid ranks: 4, 8, 16, 32, 64.
- Default: rank 16.
- Explicit invalid numeric ranks are snapped to the nearest valid rank (lower wins ties).
- Missing rank uses 16; non-numeric values are rejected.
- A config-less fallback selects a deterministic untried complete config, preferring a
  higher untried rank before changing another bounded axis.
- `TrainingConfig` rejects any remaining invalid rank.

### Alpha and dropout

- Alpha is selected from `{rank, 2*rank, 4*rank}` (default `2*rank`).
- Dropout is selected from `{0, 0.05, 0.1}` (default 0).
- Numeric suggestions are snapped deterministically; runtime config is strict.
- The same selected alpha/dropout is wired to text and multimodal language-layer LoRA.
- Dropout 0 is Unsloth's optimized path. Nonzero dropout may use a slower path and
  materially increase training time; `.05` or `.1` should therefore be tied to an
  explicit overfitting hypothesis.
- Bias remains `none`.

### Learning rate

- Default: `2e-4`.
- Orchestrator values are clamped to `[1e-5, 5e-4]`.
- `TrainingConfig` strictly enforces the inclusive range.
- There is no task/model-specific range or scheduler choice in the intervention contract.

### Epochs

- Default/request fallback: 3; orchestrator values are clamped to `[1,8]`.
- The selected value is the actual epoch ceiling, so different epoch identities do not
  silently collapse to the same eight-epoch run.
- The best validation-loss checkpoint is loaded with patience 3 by default.
- If the early-stopping/checkpoint path errors, training retries without validation for the
  originally requested epoch count.

### Batch and optimizer regularization

- Micro batch and gradient accumulation are independently selected from `{1,2,4,8}`.
- Effective batch is derived as `micro_batch_size * gradient_accumulation_steps`, so it
  is one of `{1,2,4,8,16,32,64}` and never exceeds 64.
- Legacy `batch_size` maps to micro batch. Conflicting aliases and a supplied effective
  batch that disagrees with the derived value are rejected.
- Weight decay is selected from `{0,0.01,0.05,0.1}` and is always passed to
  `SFTConfig`, with or without validation/early stopping.
- The iterate prompt shows effective batch and explains that micro batch controls peak
  activation memory while accumulation raises effective batch without raising that peak.

### Other fixed training choices

- Maximum sequence length: 512.
- LoRA training loads the base in 4-bit whenever rank is not `None`.
- Text LoRA targets q/k/v/o and gate/up/down projections; multimodal mode tunes language,
  attention, and MLP modules while freezing vision.
- Early-stop split defaults to 12% when at least 60 formatted examples exist.
- Every supported task uses completion-only labels: prompt tokens are `-100`, while
  assistant labels, NER JSON, generation/CoT, and code target tokens are trained.
  Text-only multimodal training uses the same explicit mask through the inner tokenizer.
- An early-stop/checkpoint failure discards the failed model/trainer, reloads a fresh
  base+LoRA stack, rebuilds rows with the fresh tokenizer, and retries the complete
  original dataset without validation. It never continues partial weights or the
  reduced train split.
- Full `(dataset, H)` identities are stored in the DAG/checkpoint. Pruned identities count
  as tried; exact repeats on the same dataset are forbidden. Data rebuilds and rollback
  retain every winning optimizer field.

Structured data rebuild and complete dataset path/version rollback are implemented.

## 4. Recommendations for additional interventions

These are recommendations only:

1. **Prompt-contract repair:** first-class intervention that changes a versioned shared
   train/eval prompt builder, then re-evaluates without relabeling data.
2. **Verified hard-case generation:** task-specific generators for math/code that produce
   new problems plus executable/exact-answer verification, instead of unchanged failures.
3. **Preference optimization for rejected answers:** store generation negatives as
   chosen/rejected pairs and train with an appropriate preference loss; never as positive SFT.
4. **Sampling/reweighting:** class weights, confusion-pair oversampling, and difficulty-bucket
   sampling without fabricating labels.
5. **LoRA target-module search:** alpha/dropout are now bounded interventions; target-module
   sets remain fixed and could become a future model-aware choice.
6. **Optimizer schedule intervention:** weight decay and effective batch are now bounded;
   warmup/scheduler choices remain fixed and could use a validation protocol.
7. **Context-length intervention:** task-aware sequence length and truncation diagnostics
   before increasing model capacity.
8. **Cross-run source cache:** reuse novelty fingerprints across independent runs without
   weakening source/split restrictions.
9. **Additional verified-positive strategies:** add math/code generators only when exact
   answer or execution verification can gate every synthesized row.
