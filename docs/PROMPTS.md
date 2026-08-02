# Prompt inventory

**Effective date: 2026-07-29.** Code-referenced inventory of every prompt reachable from the
cold-start and production graphs. Static audit — no model API calls were made.

"Active" means reachable in a configured runtime path, including optional fallbacks. Symbols are
cited with paths so this survives line-number drift. Companion docs:
[`PIPELINE.md`](PIPELINE.md) for control flow, [`BUGS.md`](BUGS.md) for tracked defects.

**Completeness check.** Every cost-tracked LLM/search `stage=` in the repo maps to a section
below:

| `stage` | Section | | `stage` | Section |
|---|---|---|---|---|
| `task_analysis` | [1.1](#11-task-analysis-and-run-planning) | | `cot_fallback` | [2.1](#21-cotimplementation-reasoning-annotation) |
| `hardware_research` | [1.2](#12-hardware-resolution) | | `hard_negative_synthesis` | [2.2](#22-classification-hard-negative-generation), [2.3](#23-ner-hard-example-generation) |
| `acquire_schema_mapping` | [1.3](#13-dataset-schema-mapping) | | `local_synthesis` | [2](#2-teacher-and-local-qwen-prompts) (transport) |
| `acquire_seed_synthesis` | [1.4](#14-last-resort-seed-synthesis) | | `synth_preflight` | [2](#2-teacher-and-local-qwen-prompts) (transport) |
| `acquire_ner_annotation` | [1.5](#15-web-acquired-ner-annotation) | | `generation_judge` | [3.1](#31-open-generation-semantic-judge) |
| `acquire_dataset_discovery` | search input, not a prompt — [§6](#6-present-but-not-active) | | `generation_judge_preflight` | [3.1](#31-open-generation-semantic-judge) |
| `acquire_exa` | search input, not a prompt — [§6](#6-present-but-not-active) | | `production_taxonomy` | [5.1](#51-failure-taxonomy) |
| `model_selection` | [1.6](#16-initial-orchestrator-model-choice) | | `delegate_task` | [§6](#6-present-but-not-active) — no call sites |
| `escalate` | [1.7](#17-escalation-and-downward-candidate-choice) | | `iterate_web_search` | [§6](#6-present-but-not-active) — no call sites |
| `downward_probe` | [1.8](#18-downward-re-exploration-decision) | | | |
| `iterate` | [1.9](#19-iterationexpand-decision) | | | |
| `iterate_json_reask` | [1.10](#110-tool-free-json-re-ask) | | | |

Prompts with no cost stage (they run on the SLM being trained, not a provider): [§4](#4-slm-training-and-evaluation-prompts), [§5.2](#52-live-confirmation).

---

## 1. Orchestrator prompts

All use `config.config.ORCHESTRATOR_MODEL`. Every call site routes exceptions through
`agent/llm_errors.py::raise_if_fatal`, so an `anthropic.*` / `httpx.*` transport error **aborts
the run** rather than falling back — see [`PIPELINE.md` §2](PIPELINE.md#2-global-guards).

### 1.1 Task analysis and run planning

- **Code:** `agent/task_planner.py::_PLANNER_PROMPT`, formatted by `plan_task`.
- **Purpose:** classify the task, choose labels/flags/Exa queries, propose curriculum and eval
  sizes, calibrate the stop threshold.
- **In:** free-text description, parameter range, one deduplicated summary per pool model.
  Capability values retain explicit metric names, missing values read `not reported`, source URLs
  included (`_pool_summary`).
- **Out:** one JSON object — task type/name, labels, flags, queries, benchmark, stop threshold,
  data sizes, rationale.
- **Validation:** `_extract_json` accepts a whole reply or the first greedy `{...}`. Only
  `task_type` is validated; missing keys get defaults; size clamping is deferred to
  `task_analysis.py::_apply_data_targets`.
- **Critique:** one call controls task schema, data acquisition, *and* the termination target.
  Numeric fields, labels, and query-map shape are not schema-validated. Still asks the model to
  use leaderboard recall, which may be stale even though pool metrics are sourced.
- **Improve:** typed JSON schema; split schema inference from threshold calibration; derive
  known-benchmark thresholds from a versioned local registry and ask only for a cited adjustment.

### 1.2 Hardware resolution

- **Code:** `agent/nodes/cold_start/hardware_research.py::_PROMPT`, via `research_device`.
- **Purpose:** convert a device description plus local-DB or Exa snippets into usable model RAM,
  storage budget, and a reference chip.
- **In:** user description, ≤3 device records/snippets, allowed `REFERENCE_CHIPS`
  (= `android_pool.KNOWN_CHIPS` — names only, no performance claims attached).
- **Out:** strict JSON — device, chipset, total capacities, `usable_ram_mb`,
  `storage_budget_mb`, `reference_chip`, rationale. Latency/power/throughput floors are
  explicitly **excluded**; they are fixed `HW_*` constants.
- **Validation:** `_extract_json`; `reference_chip` is allow-listed; numerics cast with
  defaults. **No cross-check that usable RAM < total RAM** or that storage is plausible.
- **Critique:** deterministic DB rows are routed through an LLM for arithmetic code could do.
  Prompt-injected web snippets can influence output.
- **Improve:** compute budgets deterministically for complete DB rows; quote/sanitize external
  snippets; add range and consistency validation.

### 1.3 Dataset schema mapping

- **Code:** `data/loaders/web_acquire.py::_llm_map_dataset`
- **Purpose:** decide whether a Hugging Face dataset fits, and map its columns/splits.
- **In:** task plan, repo/config, advertised splits/columns, one sample row.
- **Out:** task-specific mapping JSON, or `{"suitable": false}`.
- **Validation:** first greedy object; requires one of `text_col` / `question_col` /
  `tokens_col`. Materialization later accesses the proposed names with no full preflight check.
- **Critique:** one row may not expose nullable or alternate schemas; repo data is untrusted
  prompt content; suitability and mapping are conflated in one answer.
- **Improve:** validate every proposed split/column against dataset metadata, inspect several
  bounded rows, separate suitability classification from a deterministic mapping validator.

### 1.4 Last-resort seed synthesis

- **Code:** `data/loaders/web_acquire.py::synthesize_seed_examples`
- **Purpose:** generate minimally viable training rows only when real/local/web data cannot meet
  the viability floor.
- **In:** task name/type, allowed labels/entity types, requested count.
- **Out:** strict `{"examples": [...]}` in one of **three hardcoded task-family schemas** —
  `{text,label}` (classification), `{text,entities[{text,type}]}` (NER), `{text,answer}`
  (everything else).
- **Validation:** one object; dedup by text; classification labels must be in the allowed set;
  NER spans must be exact substrings; generation answers must be non-empty. **Factual
  correctness is not verified.**
- **Mode:** temperature 1.0, `max_tokens=4096`.
- **Critique:** the same model plans the task and fabricates "gold," so errors are correlated. A
  large requested count goes in one response and may truncate. The three fixed schemas are also
  the reason a task needing a fourth output shape (function call, diff, recurrence rule) cannot
  be expressed — see [`PIPELINE.md` §12.2](PIPELINE.md#122-recommended-interventions-none-implemented).
- **Improve:** bounded batches; independent verification; per-example provenance/confidence;
  never admit unverified synthetic gold to held-out evaluation.

### 1.5 Web-acquired NER annotation

- **Code:** `data/loaders/web_acquire.py::_annotate_ner_entities`
- **Purpose:** label entity spans in unannotated web passages.
- **In:** the **first 500 characters** of each passage.
- **Out:** JSON list of exact-span/type objects, or `[]`.
- **Validation:** greedy list extraction; presence checks for `text`/`type` only. **Contrary to
  the prompt, spans are not rechecked as exact substrings and types are not allow-listed here.**
- **Cost:** one call per passage — not batched.
- **Critique:** a silent parse/call failure becomes an empty-entity gold example, which is
  indistinguishable from a genuine negative and teaches "no entities." Truncation can remove
  entity context. **Tracked as [B203](BUGS.md#b203--_annotate_ner_entities-does-not-re-validate-spans-or-types).**
- **Improve:** validate spans/types; mark annotation failures instead of converting them to
  negatives; batch requests and retry only malformed items.

### 1.6 Initial orchestrator model choice

- **Code:** `agent/nodes/cold_start/model_selection/orchestrator_choice.py::_CHOOSE_SYSTEM`,
  `orchestrator_choice_node`
- **Purpose:** choose the **least-resource** candidate that can plausibly meet the goal after
  LoRA. The system prompt states this explicitly ("NOT simply the largest or
  highest-benchmark model").
- **In:** task description/type/schema, goal, hardware budget, sourced `capability_sections`,
  exact variant selectors, quant/`size_mb`/notes, and measurements carrying
  metric/artifact/mode/protocol/source.
- **Out:** strict JSON — exact `selector` + one-sentence reason.
- **Validation:** `_parse_choice` accepts JSON, a greedy object, or bare text; the selector must
  match a candidate. Unknown/failed output → **lowest `size_mb`** feasible variant.
- **Active only when** `MODEL_SELECTION_STRATEGY == "orchestrator_choice"`.
- **Critique:** selector ambiguity is resolved; legacy bare IDs resolve to the lowest-size
  sibling. Whether the conservative fallback can actually meet the goal is uncalibrated.

### 1.7 Escalation and downward candidate choice

- **Code:** `agent/nodes/escalate.py::_CHOOSE_SYSTEM`, `_llm_choose_model`. Called by
  `escalate_node` (`direction="up"`) **and** `downward_probe.py::downward_probe_step_node`
  (`direction="down"`).
- **Purpose:** pick a candidate within the next higher or lower size tier. All candidates already
  fit the device, so the prompt frames the choice purely as expected task capability after LoRA,
  breaking ties toward lower resource use.
- **In:** task type/name/labels, current best F1, quant/`size_mb`/notes, `_BENCHMARK_HINT` for
  the task type (steers away from defaulting to GSM8K on a non-math task),
  `capability_sections`, `METRIC_COMPARABILITY_CAVEAT`, and the explicit direction + target tier.
- **Out:** exact selector + reason, strict JSON.
- **Validation:** same tolerant parsing and membership check. Failure → `min(candidates,
  key=size_mb)` **within the already-selected direction-appropriate tier**.
- **This is the single capability-injection point for both directions.**
- **Critique:** capability prediction stays qualitative when comparable evidence is absent.
- **Improve:** add measured task-local evidence when available.

### 1.8 Downward re-exploration decision

- **Code:** `agent/nodes/downward_probe.py::_should_reexplore_downward`
- **Purpose:** decide whether resource savings justify probing another lower tier after
  convergence.
- **In:** winning score, goal, margin, iterations on the winner, models/tiers tried, untried
  lower tiers.
- **Out:** `{"reexplore": boolean, "reason": string}`.
- **Validation:** greedy object, then `bool(...)`. Any failure → heuristic `margin >= 0.03`.
- **Critique:** **`bool("false")` is `True`** — a JSON string flips the decision. The prompt also
  omits probe cost and candidate RAM savings, so the call is not actually cost-aware. **Tracked
  as [B202](BUGS.md#b202--_should_reexplore_downward-coerces-a-json-string-to-a-boolean).**
- **Improve:** require a real JSON boolean; include resource deltas and remaining wall-clock; or
  replace the call with an explicit policy.

### 1.9 Iteration/EXPAND decision

- **Code:** `agent/nodes/iterate.py::_ITERATE_SYSTEM`, `_llm_iterate`
- **Purpose:** diagnose the trajectory and choose `data_rebuild` **or** `hyperparameter`, with a
  bounded declarative rebuild plan or LoRA settings.
- **In:** compacted curation trajectory (`context_manager.compact_trajectory` when
  `should_compact`), model/iteration/scores/threshold, test-agent aggregate
  difficulty + confusion report, tried `(dataset, H)` identities **including pruned ones**, tried
  rebuild-plan identities with yield status, prior hypothesis, source novelty, remaining turn and
  paid-acquisition budgets, on-disk weight size and device memory budget. **Raw eval rows are
  excluded by construction.**
- **Out:** one decision JSON object — `intervention`, `hypothesis`, then exactly one of
  `data_rebuild` / `hyperparams`, plus optional `threshold_adjustment`.
- **Contract (as of 2026-07-29):**
  - The schema is a **discriminated union**; merging branch payloads is rejected.
  - **Exactly five hyperparameters are tunable**: `lora_rank` ∈ {4,8,16,32,64}, `alpha_ratio` ∈
    {1,2,4} (alpha = rank × ratio), `weight_decay` ∈ {0,0.01,0.05,0.1}, `learning_rate` ∈
    [1e-5,5e-4], `nr_epochs` ∈ [1,8]. Any other key is rejected with an actionable message
    naming the replacement (`lora_alpha` → use `alpha_ratio`; `lora_dropout` → regularize with
    `weight_decay`; batch-shape fields → derived by the trainer to fit VRAM). The prompt states
    the batch-shape exclusion and its reason explicitly.
  - A stray `hyperparams` block on a `data_rebuild` is **stripped, not rejected** — see
    [`PIPELINE.md` §7.1](PIPELINE.md#71-data_rebuild) for why.
  - **Three** rebuild strategies (`resample`, `acquire`, `synthesize`) with per-field
    bounded/stepped ranges. No task-type or score gating — any strategy is valid for any task
    (redesign 2026-07-31); `synthesize` is task-adaptive (hard negatives for classification/NER,
    new correct examples for generation-family). **One availability gate:** when the whole train
    pool is already in the curriculum, `resample` is removed from the menu — the validator/fallback
    redirect it to `synthesize` and the prompt carries a "resample unavailable this turn" note,
    since a reshuffle there adds no novelty.
- **Validation:** `_parse_decision_json` handles content-block lists, code fences, and prose
  wrapping, and **always** raises `ValueError` (never a bare `JSONDecodeError`).
  `_validate_decision_json` enforces field allow-lists, branch exclusivity, integer-vs-numeric
  JSON types, finite threshold values, and **recursive rejection
  of any held-out eval text** in any string. It is re-applied to the result with
  `allow_internal=True` as defense in depth, so mock/alternate provider paths cannot bypass the
  contract.
- **Cost bound:** one tracked `iterate` call plus at most one `iterate_json_reask`. Tool-use
  blocks are never executed or reflected back.
- **Threshold semantics:** can only *decrease*, clamped at `initial_stop_threshold`. When the
  current threshold already equals that floor, this path is inert.
- **Remaining critique:** the score-band guidance (`<0.80` data, `0.80–0.95` hyperparameter,
  `≥0.95` refinement) can still anchor the diagnosis even though it is only a fallback rule.

### 1.10 Tool-free JSON re-ask

- **Code:** `agent/nodes/iterate.py::_reask_json_only`
- **Purpose:** recover a decision after prose, malformed JSON, or an attempted tool-use block.
- **In:** the original bounded context plus a short "no tools, JSON only" instruction and the
  **sanitized** validator error. `_sanitize_reask_error` collapses known error prefixes, redacts
  any eval text, strips control characters, and truncates to 500 chars. The malformed first
  response is never replayed.
- **Out/validation:** same decision JSON, same parser. A fresh non-tool-bound client is used.
- **Critique:** a second full-context paid call. In the NER run it fired on **86 of 141**
  iterate calls (61%, $4.90) versus 2 of 63 in the math run — **tracked as
  [B206](BUGS.md#b206--iterate_json_reask-fired-on-86-of-141-iterate-calls-in-the-ner-run)**;
  re-measure now that the `hyperparams` strip is in place.
- **Improve:** provider-side structured output when available.

---

## 2. Teacher and local-Qwen prompts

`data/synth_client.py::get_generate_fn` sends all local-Qwen prompts in **non-thinking** mode
with `top_p=0.8`, `top_k=20`, caller-supplied temperature and token limit. Transport-only
stages: `synth_preflight` (reachability) and `local_synthesis` (generation); both are recorded at
$0 as local-provider cost events.

`agent/nodes/curate.py` uses local `SYNTH_MODEL` first and **never falls back to Claude for hard
negatives.** CoT is authored **only** by the local Qwen3.6 synth endpoint — there is no cloud CoT
teacher; if the endpoint is unavailable the example is left CoT-less.

> **Neither completed run produced a single synthetic row.** The ledgers contain only
> `synth_preflight` events (8/9 failed in NER, 18/43 in math) and zero
> `hard_negative_synthesis` events. Everything in this section is therefore
> **implemented but unexercised in production** — see
> [B200](BUGS.md#b200--synthesis-has-never-executed-at-scale-both-completed-runs-were-gold-only).

### 2.1 CoT/implementation-reasoning annotation

- **Code:** `data/curriculum.py::annotate_cot::_build_prompt`
- **Purpose:** add reasoning to generation-family gold examples that lack CoT.
- **In:** problem/prompt plus the correct answer/solution.
- **Out:** reasoning steps only. Code tasks request an implementation plan *without code*; other
  tasks request step-by-step reasoning *without the final answer*.
- **Validation:** any non-empty stripped text is accepted verbatim as `cot_reasoning`. Existing
  CoT is preserved. Sole backend: local Qwen3.6 (no cloud teacher); if it is unreachable the
  example is left unchanged.
- **Critique:** no check that the reasoning is consistent with the supplied gold, does not leak
  the final answer, or fits the training context after formatting.
- **Improve:** verify answer consistency, reject leakage when the prompt forbade it, and record
  backend/model/prompt-version per annotation.

### 2.2 Classification hard-negative generation

- **Code:** `data/curriculum.py::synthesize_hard_negatives`, classification branch.
- **Purpose:** produce text that superficially resembles the source class but genuinely belongs
  to another — a contrastive pair.
- **In:** source text/label, **the first alternate target label**, and an aggregate
  `pattern_hint` from the rebuild plan.
- **Out:** raw text only for the target class (explicitly: no preamble, no quotes, no label
  prefix).
- **Validation:** any non-empty text is *assigned* the target label by code. Quality controls
  later dedup and cap label imbalance. The 2-for-1 result carries a real source anchor **and** a
  generated row; curate reports `n_hard_source` and `n_hard_generated` separately.
- **Cost/parallelism:** generations run concurrently (vLLM continuous-batches them); order is
  preserved so each anchor stays adjacent to its synthetic counterpart. Three consecutive
  failures abort synthesis and degrade to gold-only.
- **Critique:** semantic membership in the target class is **not verified**. Picking the first
  alternate label can overproduce one class on multiclass tasks. The prompt is a **hardcoded
  f-string** — the only orchestrator-controlled inputs are `pattern_hint` (one sentence) and
  `temperature`.
- **Improve:** choose the target from the observed confusion pair; require an independent label
  verifier; reject near-copies before admitting them as gold.

### 2.3 NER hard-example generation

- **Code:** `data/curriculum.py::synthesize_hard_negatives`, NER branch.
- **Purpose:** rewrite a passage into a more ambiguous context while preserving correct entity
  types.
- **In:** original passage plus up to five entity text/type pairs.
- **Out:** JSON object with rewritten text and typed spans.
- **Validation:** greedy object parse, then **strict**: malformed JSON, empty entity lists, spans
  absent from the rewritten text, and types absent from the source row are all discarded. Source
  anchors and rewrites are accounted separately.
- **Critique:** span/type integrity is enforced, but nothing confirms the rewrite preserved the
  intended difficulty.

### 2.4 Open generation, math, and code: task-adaptive *new-correct* synthesis (redesign 2026-07-31)

- **Code:** `data/curriculum.py::synthesize_examples` → `_synthesize_new_correct` for the
  generation family; `synthesize_hard_negatives` still serves classification/NER.
- **Behavior:** synthesis is **ungated** for all task types. For math/code/generation it generates
  **new, correct, in-distribution** examples in the same schema as the anchors (via
  `_new_example_prompt`), verified by a `verify_fn` where one exists (math answer / code tests),
  and kept only if they pass. It never writes a wrong answer as a positive SFT target.
- **Safety property (preserved):** contrastive *wrong-answer* pairs are still confined to
  classification/NER; generation-family synthesis is correct-only.
- **Consequence:** every family now has a synthesis path and curricula are synth-filled to the
  target size; unverifiable generation rows fall back to standard quality controls.
- **Improve:** stronger verifiers (full execution harness for code, symbolic checks for math)
  and a preference/ranking objective for open generation.

---

## 3. Judge prompt

### 3.1 Open-generation semantic judge

- **Code:** `eval/judge_client.py::JUDGE_SYSTEM`, `JUDGE_PROMPT_VERSION`
  (`"qwen36-json-rubric-v2"`), `_build_judge_prompt`, `LocalJudgeClient`. Dispatched from
  `eval/scorers/generation.py`.
- **Purpose:** score a generated answer against gold on a 0–1 anchored rubric.
- **In:** question, gold, prediction.
- **Out:** one decimal in `[0,1]`.
- **Validation:** strict full-response numeric parse plus a finite closed-range check. Extra
  text, non-numerics, NaN/inf, or out-of-range values **abort evaluation** rather than becoming a
  false score.
- **Mode/backend:** `task_type == "generation"` only. Required local Qwen3.6 via the configured
  OpenAI-compatible vLLM endpoint. `/models` must expose the exact configured Qwen3.6-family
  model and every completion's `response.model` must match. `enable_thinking=False` on every
  request. There is **no cloud fallback**. Remote hosts hard-fail by default, including private cluster
  addresses; only loopback/localhost/Unix sockets are accepted without
  `SLM_JUDGE_ALLOW_REMOTE=1`.
- **Prompt safety:** normalized question/gold/prediction are encoded in a marked **untrusted**
  JSON object, and the system prompt says never to follow instructions inside those fields.
- **Cache:** keys include normalized inputs, model, prompt version, and prompt fingerprint. A
  locked append-only JSONL cache under the stable run artifacts is shared across
  workers/restarts and ignores corrupt records. Misses use a bounded sliding window; the first
  failure stops new submissions and cancels pending futures while output ordering stays
  deterministic.
- **Failure semantics:** endpoint, model identity, response identity, event, cache, executor,
  request, and parse failures all become `JudgeInfrastructureError` and **propagate** — including
  out of zero-shot baseline evaluation. Math exact match and APPS/MBPP executable scoring are
  unaffected.
- **Observability:** `generation_judge_preflight` (model identity) and `generation_judge`
  (completions) are recorded as local-provider cost events at $0, plus timing events.
- **Critique:** judge agreement is not periodically measured against a fixed calibration set. The
  scorer also reports this average in a field named `"f1"` —
  [B45](BUGS.md#b45--generation-scorer-misnames-metric).

---

## 4. SLM training and evaluation prompts

These run on the model being trained, so they have no provider cost stage. **Train/eval parity is
the property that matters here**; where the two strings differ it is called out.

| Task | Eval prompt | Training prompt | Parity | Extraction |
|---|---|---|---|---|
| Classification | `eval/scorers/classification.py::CLASSIFY_PROMPT`, `build_classify_prompt` | reuses the same builder | ✅ shared builder | case-insensitive word-boundary → substring → `__EXTRACTION_FAILED__` |
| NER | `eval/scorers/ner.py::NER_PROMPT` | **duplicated** in `lora_trainer.py` NER `format_example` | ❌ training omits the explicit `[]` instruction and punctuation differs; entity vocabulary is not enumerated | whole-list JSON or greedy list; keeps objects with both keys; **no exact-substring or type allow-list check** |
| Generation / math | `eval/scorers/generation.py::GENERATE_PROMPT` | generation branch in `lora_trainer.py`; may prepend `<reasoning>` to the assistant target | ❌ eval prepends "Answer the following question", training sends the raw prompt | generation → LLM judge; math → explicit final-answer marker, else the last number |
| Code | `eval/scorers/generation.py::CODE_GENERATE_PROMPT`, `build_code_prompt` | reuses the same builder | ✅ explicit parity | strip Python/empty fences, then execute every preserved APPS case in an isolated worker with per-case timeout and a bounded per-problem deadline |

Notes:

- **Classification** critique: the substring fallback can accept a label embedded in unwanted
  prose or inside another token. Improve with constrained decoding or exact normalized output
  before a carefully delimited fallback, plus an input delimiter resistant to instruction text
  inside the message.
- **NER** improvement: share one prompt builder for train/eval, enumerate allowed types/schema,
  and validate exact spans at prediction extraction.
- **Math** improvement: require a stable final-answer marker, since the parser depends on one
  but the prompt never asks for it.
- **Code:** the trusted controller retains expected outputs; candidate workers inherit no
  success FD or expected-output payload. Optional code reasoning is rendered as executable Python
  comments. MBPP assertions remain the CI smoke. The process boundary and limits reduce
  accidental damage but are **not** seccomp or a hostile-code sandbox.

### 4.5 Inference chat wrapping

- **Code:** `training/slm_helpers.py::infer`, `infer_batch_gguf`
- **Purpose:** apply the model/GGUF chat template around each already-built task prompt.
- **Behavior:** one user turn, assistant continuation; parsing delegated to the task scorer. HF
  inference is greedy and passes `enable_thinking=False`. GGUF uses `chat_template_kwargs` where
  supported — the installed llama-cpp-python 0.3.34 signature lacks it, so hybrid Qwen3/Qwen3.5
  use the verified empty-think ChatML prefix and non-thinking-only Qwen3-4B-Instruct-2507 uses
  its plain assistant prefix. **Unknown templates fail rather than silently changing mode.**
- **Quant identity:** Q4/Q8 baselines and interpolation/downward probes build or reuse their
  exact GGUF and pass `gguf_path`; histories retain `model_id@quant` selectors.
- **Improve:** record the template path/version with eval artifacts; drop the manual path once
  the deployed llama-cpp-python accepts template kwargs.

---

## 5. Production prompts

### 5.1 Failure taxonomy

- **Code:** `agent/nodes/production/taxonomy.py::taxonomy_construct_node`, inline system + user
  prompts.
- **Purpose:** cluster up to **40** sampled failed traces into 3–8 categories, each labeled
  `fixable` (by training data) or external.
- **In:** total/sample counts plus full sampled trace JSON with stable indices. The sampled
  window and the tagged window are now the same 40 — they previously mismatched (20 shown, 50
  tagged), leaving silently untagged traces.
- **Out:** strict JSON — clusters with counts, root causes, boolean fixability, trace indices,
  and a summary.
- **Validation:** greedy object parse, or a raw-summary fallback. Code does **not** enforce
  unique/complete indices, recompute counts, type-check `fixable`, or validate cluster names.
- **Critique:** raw deployed inputs/outputs are untrusted prompt content. The prompt demands a
  partition; the implementation accepts overlaps and omissions.
- **Improve:** schema-validate, deterministically repair or reject invalid partitions, recompute
  counts from indices, and delimit traces as untrusted data.

### 5.2 Live confirmation

- **Code:** `agent/nodes/production/live_confirm.py::live_confirm_node`
- **Purpose:** re-run each prescreened failure through deployed model M0 to confirm it is
  systematic rather than sampling noise.
- **Behavior:** sends `trace["input"]` **directly, with no task prompt**, and compares the
  stripped output to `corrected_output` by exact string equality. Inference errors count
  conservatively as confirmed failures. Pre-screening is by the `cluster` key, and traces with
  `cluster=None` are kept as unclassified rather than dropped.
- **Critique:** this does not reproduce the original serving prompt/template, and exact string
  equality is wrong for most classification, NER, and generation outputs — so genuine passes
  enter the training set as confirmed failures. **Tracked as
  [B207](BUGS.md#b207--live-confirmation-does-not-replay-the-original-serving-prompt).**
- **Improve:** store the rendered prompt plus task/parser metadata per trace, replay through the
  same scorer, and separate infrastructure errors from real failures.

### 5.3 Production training prompts

The production graph routes confirmed examples through the **same** shared `curate` → `train` →
`evaluate` → `iterate` nodes, so every prompt above applies unchanged.

**However:** `curate_node` raises when `eval_set is None`, and the production graph contains no
node that builds one — so production startup depends on caller-populated state that is not
expressed or validated as a contract. **Tracked as
[B199](BUGS.md#b199--production-graph-cannot-start-curate-requires-an-eval_set-nothing-builds).**

---

## 6. Present but not active

- **`agent/tools/delegate_task.py::delegate_task`** — a generic no-tools sub-agent system prompt
  that forwards an arbitrary task description. **Zero call sites** outside `agent/tools/`
  (verified 2026-07-29). [B21](BUGS.md)
- **`agent/tools/web_search.py::web_search`** (cost stage `iterate_web_search`) — an
  `@tool`-decorated Exa wrapper, bound to no LLM and never invoked. Same for the other three
  decorated tools. [B23](BUGS.md)
- **Legacy Claude hard-negative backend** — the fallback inside `synthesize_hard_negatives` is
  reachable only by external/legacy callers that omit `generate_fn`. `curate_node` always
  supplies local Qwen or skips synthesis.
- **Exa search strings** (`acquire_dataset_discovery`, `acquire_exa`, and
  `use_autoprompt=True`) — search *inputs*, not generation prompts, so they are outside this
  inventory even though they are cost-tracked.

---

## 7. Highest-priority improvements

1. **Unify train/eval builders** for NER and general/math generation (classification and code
   already share one). This is the only remaining train/serve parity gap.
2. **Typed JSON-schema validation** for planner, hardware, downward-probe, acquisition, and
   taxonomy outputs. `iterate` already has the strictest validator in the repo; the others do
   not.
3. **Validate NER annotation output** at acquisition so a failed call cannot become an
   empty-entity gold row. [B203](BUGS.md#b203--_annotate_ner_entities-does-not-re-validate-spans-or-types)
4. **Verified-positive or preference-objective augmentation** for open generation, math, and
   code — those three families currently have no augmentation path at all.
5. **Reproduce the serving prompt and parser** during live confirmation.
   [B207](BUGS.md#b207--live-confirmation-does-not-replay-the-original-serving-prompt)
6. **Treat external content as untrusted data** — dataset rows, web snippets, and production
   traces reach decision prompts. The judge already does this correctly (marked untrusted JSON +
   an explicit system-prompt instruction); use it as the template.
