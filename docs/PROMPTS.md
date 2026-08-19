# Prompt inventory

**Effective date: 2026-07-29.** Code-referenced inventory of every prompt reachable from the
cold-start and production graphs. Static audit — no model API calls were made.

"Active" means reachable in a configured runtime path, including optional fallbacks. Symbols are
cited with paths so this survives line-number drift. Companion docs:
[`PIPELINE.md`](PIPELINE.md) for control flow, [`BUGS.md`](BUGS.md) for tracked defects.

> **Update 2026-08-19 — the `task_type` channel is gone, and several prompts changed with it.**
> Behaviour now comes from a per-task `TaskSpec` in `tasks/` rather than from an abstract channel
> five benchmarks shared, so anywhere below that a prompt was described as "per task type", it is
> now per task. Three specific consequences, each corrected in place: the iterate contract offers
> **two** rebuild sub-strategies rather than three ([§1.9](#19-iterationexpand-decision)); every
> synthesis and verification prompt is now built from an orchestrator-authored **task brief**, which
> is itself a new prompt ([§1.11](#111-task-brief-authoring), [§2.2](#22-task-adaptive-synthesis));
> and the code-generation training/eval prompts and their execution sandbox were **deleted** on
> 2026-08-18 along with APPS/MBPP ([§4](#4-slm-training-and-evaluation-prompts)).
>
> Two whole sections describe prompts that no longer exist and are retained only as history:
> [§5](#5-production-prompts--removed) (production mode was removed 2026-07-29, and
> `agent/nodes/production/` with it) and [§6](#6-present-but-not-active) (`agent/tools/` no longer
> exists).

**Completeness check.** Every cost-tracked LLM/search `stage=` in the repo maps to a section
below:

| `stage` | Section | | `stage` | Section |
|---|---|---|---|---|
| `task_analysis` | [1.1](#11-task-analysis-and-run-planning) | | `local_synthesis` | [2](#2-teacher-and-local-qwen-prompts) (transport) |
| `hardware_research` | [1.2](#12-hardware-resolution) | | `synth_preflight` | [2](#2-teacher-and-local-qwen-prompts) (transport) |
| `acquire_schema_mapping` | [1.3](#13-dataset-schema-mapping) | | `generation_judge` | [3.1](#31-open-generation-semantic-judge) |
| `acquire_seed_synthesis` | [1.4](#14-last-resort-seed-synthesis) | | `generation_judge_preflight` | [3.1](#31-open-generation-semantic-judge) |
| `acquire_ner_annotation` | [1.5](#15-web-acquired-ner-annotation) | | `acquire_dataset_discovery` | search input, not a prompt |
| `model_selection` | [1.6](#16-initial-orchestrator-model-choice) | | `acquire_exa` | search input, not a prompt |
| `escalate` | [1.7](#17-escalation-and-downward-candidate-choice) | | | |
| `iterate` | [1.9](#19-iterationexpand-decision) | | | |
| `iterate_json_reask` | [1.10](#110-tool-free-json-re-ask) | | | |
| `task_brief` | [1.11](#111-task-brief-authoring) — **new 2026-08-19** | | | |
| `threshold_raise` | [1.12](#112-stretch-goal-decision) — **new** | | | |

Stages that left the table: `downward_probe` (the re-exploration gate was deleted, [§1.8](#18-downward-re-exploration-decision--removed)), `cot_fallback` (there is no cloud CoT teacher), and `delegate_task` / `iterate_web_search` (`agent/tools/` was deleted).

Prompts with no cost stage (they run on the SLM being trained, not a provider): [§4](#4-slm-training-and-evaluation-prompts).

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
- **Reachability (2026-08-19):** the **autonomous** path only. A run naming one of the eight
  registered tasks skips this call entirely and reads its labels, metric, caps and flags from the
  task's `TaskSpec`. This prompt is also the last surviving user of the old channel vocabulary —
  it still asks the orchestrator to classify a free-text description as `classification` / `NER` /
  `math_reasoning` / `code_generation` / `generation` / `function_call` / `diff` — because a task
  with no spec has nothing else to be described by. Those strings no longer dispatch anything
  downstream of a registered task.
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
- **Reachability (2026-08-19):** this is on the **autonomous** acquisition path only — the branch
  taken when the run names a task the registry does not hold. Every task this project runs is
  curated and loads from its own loader, so nothing here executes on a normal run.
- **Critique:** the same model plans the task and fabricates "gold," so errors are correlated. A
  large requested count goes in one response and may truncate. The three fixed schemas are also
  the reason a task needing a fourth output shape (a function call, a recurrence rule) cannot be
  expressed on this path — which is one of the reasons the eight benchmarks are curated rather than
  discovered. See [`PIPELINE.md` §12.2](PIPELINE.md#122-recommended-interventions-none-implemented).
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
- **In:** the task name and its labels, current best F1, quant/`size_mb`/notes,
  `capability_sections`, `METRIC_COMPARABILITY_CAVEAT`, and the explicit direction + target tier.
  The per-task-type `_BENCHMARK_HINT` (which steered away from defaulting to GSM8K on a non-math
  task) was removed with the channels on 2026-08-19; the ranking benchmark is now a per-task field,
  `TaskSpec.model_ranking_metric` — `GSM8K` for `gsm8k`, `MMLU` for the three classification tasks,
  and `None` for the rest, which says plainly that no published benchmark ranks candidates for
  function calling or span extraction.
- **Out:** exact selector + reason, strict JSON.
- **Validation:** same tolerant parsing and membership check. Failure → `min(candidates,
  key=size_mb)` **within the already-selected direction-appropriate tier**.
- **This is the single capability-injection point for both directions.**
- **Critique:** capability prediction stays qualitative when comparable evidence is absent.
- **Improve:** add measured task-local evidence when available.

### 1.8 Downward re-exploration decision — REMOVED

`agent/nodes/downward_probe.py::_should_reexplore_downward` no longer exists, and neither does its
`downward_probe` cost stage. It asked the orchestrator whether resource savings justified probing
another lower tier after convergence, and returned `{"reexplore": boolean, "reason": string}`.

Two things were wrong with it and only one was fixable. It parsed the answer with `bool(...)`, so a
JSON string `"false"` evaluated **true** and the gate could not actually decline
([B202](BUGS.md#b202--_should_reexplore_downward-coerces-a-json-string-to-a-boolean)) — that part
was a bug. But the prompt was also never given the probe's cost or the candidate's RAM saving, so
the "is it worth it?" question it purported to ask could not be answered from what it was shown.
Replacing a miscalibrated paid call with an explicit policy was the better trade: the probe now runs
while an untried lower tier exists and stops at the first tier that misses the goal. Model *choice*
within a tier is still an LLM call — [§1.7](#17-escalation-and-downward-candidate-choice), shared
with escalation.

### 1.9 Iteration/EXPAND decision

- **Code:** `agent/nodes/iterate.py::_ITERATE_SYSTEM`, `_llm_iterate`
- **Purpose:** diagnose the trajectory and choose `data_rebuild` **or** `hyperparameter`, with a
  bounded declarative rebuild plan or LoRA settings.
- **In:** the run memory (`agent/run_memory.py::build_run_memory`; `context_manager.compact_trajectory`
  is the iteration-1 fallback, before the DAG has nodes), model/iteration/scores/threshold,
  test-agent aggregate difficulty + failure-category report, tried `(dataset, H)` identities
  **including pruned ones**, prior hypothesis, on-disk weight size and device memory budget,
  remaining turn budget, and **which datasets still have rows left to mine**. **Raw eval rows are
  excluded by construction.**

  > **Update 2026-08-19 — the mining line replaced a budget count.** The prompt used to state a
  > remaining *paid-acquisition budget* ("3 paid rounds left"), which told the orchestrator how much
  > money was left rather than whether more real data existed. It now states the mining position
  > concretely, per source, from `state["source_progress"]` — and when mining has been retired it
  > says so, so the orchestrator stops proposing an intervention that provably cannot add a row.
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
  - **Two** rebuild sub-strategies (`mine_new_real`, `surgical_synthesis`) under plan
    **schema version 3** (rewritten 2026-08-19): `strategy`, `rows` [50, 2000],
    `target_categories` (≤8 `{category, count}` entries drawn from the task's own failure
    taxonomy), `pattern_hint`. No task-type or score gating — either sub-strategy is valid for any
    task. **Unknown fields are rejected, not trimmed**, which is a deliberate change of contract:
    silently dropping `target_rows` or `synth_rows` would let the orchestrator believe it had asked
    for something. **One availability gate:** when every known source is exhausted *and* web
    research has spent its allowance, `mine_new_real` is rewritten to `surgical_synthesis` and the
    prompt says mining is retired, since the alternative is a full train+eval cycle on an
    intervention that cannot add a row.

    > *Superseded 2026-08-19:* this previously described three strategies (`resample`, `acquire`,
    > `synthesize`) with per-field bounded/stepped ranges, and before that six. `resample` and the
    > universal gold fill were removed because they re-drew rows from a pool the curriculum was
    > already built from; untargeted balanced `synthesize` was removed because it spent the teacher
    > budget on class balance rather than on measured failures. The `resample`→`synthesize`
    > redirect described here is gone with them.
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
- **Remaining critique:** the score-band guidance (`<0.80` data, `0.80–1.0` hyperparameter in the
  prompt; `apply_iteration_policy` additionally sends `≥0.95` to surgical refinement) can still
  anchor the diagnosis even though it is only a fallback rule.

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

### 1.11 Task-brief authoring

*New 2026-08-19. This is the prompt every teacher prompt is now built from.*

- **Code:** `agent/task_brief.py::_BRIEF_PROMPT`, `build_task_brief`. Called once, from
  `eval_setup._author_task_brief`, after the real data is loaded.
- **Purpose:** have the orchestrator write the description of the benchmark that the teacher model
  will read when generating rows and when judging whether a generated row is correct.
- **In:** the task's title, registry name, category and metric; a one-line statement of **how a
  prediction is graded**, derived from the spec (`_grading_description` — the comparison scalar,
  whether the prediction must be one of a fixed set of labels, and whether generated rows also face
  an exact programmatic check); and **five REAL rows** from the benchmark's own training split,
  serialized exactly as the pipeline stores them, minus `_`-prefixed private fields.
- **Out:** strict JSON — `summary` (2–4 sentences), `output_contract` (2–5 sentences naming fields,
  types, units and conventions), `failure_modes` (3–6 short phrases). Clipped to 700/700/6×200
  characters, because these prompts are issued once per generated row, thousands of times per
  rebuild.
- **The instruction that matters:** *"Write the output_contract from the ROWS, not from the
  benchmark's reputation. If the rows show a convention the name would not tell you — a default
  duration, a date resolved against a reference instant, an answer marker, a fixed tool name — state
  it, because a teacher that does not know it will produce plausible rows that are graded wrong."*
- **Validation:** first greedy `{...}`, then a required non-empty `summary` **and**
  `output_contract`. Any failure — unreachable orchestrator, no JSON, missing field — falls back to
  `fallback_brief`, which is deliberately *honest rather than helpful*: it states
  `"UNAVAILABLE — the orchestrator could not be reached"` as the contract and tags `source:
  "fallback"`, so a reader of the log can see synthesis ran without a real brief instead of
  assuming the prose came from the model. Never fatal: a run that can train and score is not stopped
  by a missing description.
- **Observability:** logged in full by `log_task_brief` — summary, contract and every failure mode —
  because a run whose synthesis behaved oddly needs this text in the log, not only in the artifacts.
- **What it replaced:** `data/curriculum.py::_TASK_DESCRIPTIONS`, a hardcoded one-line table keyed
  by task *type*. It could not distinguish two tasks sharing a type, so `xlam_bfcl` and
  `calendar_json` were both described as "converting a request into a JSON function call using only
  the declared tools" — omitting the 60-minute default, ISO 8601, and resolving relative dates
  against the request's own reference instant. A verifier asked to judge against that description
  was judging its own guess, and calendar synthesis scored 0.2176 (B269).
- **What it is NOT:** a source of examples. Worked examples shown to the teacher are always real
  rows sampled from the task's own training split. An orchestrator-invented example could be wrong,
  and a wrong example is worse than none.
- **Critique:** the orchestrator writes the contract from five rows, so a convention that appears in
  none of them is not in the brief. The brief is authored once and never revised, even after
  failure categories reveal a contract detail it missed.

### 1.12 Stretch-goal decision

- **Code:** `agent/nodes/iterate.py::_THRESHOLD_RAISE_SYSTEM`, `_maybe_raise_threshold`. A small
  dedicated `stage="threshold_raise"` call, deliberately not a field on the intervention decision —
  that prompt is only built for below-threshold scores.
- **Purpose:** when a score clears the goal, ask whether the goal should be **raised**, so a model
  that converged on iteration 2 is pushed further rather than stopping at a bar it cleared easily.
- **In:** the task, the model, the current goal and **its provenance** (whether the 0.80 floor or
  the Qwen teacher's own zero-shot score set it), the score and its margin, iterations used, the
  last 12 scores, turns used/remaining, the `THRESHOLD_CEILING` (0.99), the minimum raise step
  (0.005), and every previous raise.
- **Out:** `{"raise_goal": bool, "new_threshold": float|null, "reason": str}`.
- **Validation:** the proposal must exceed the run's high-water mark `max_stop_threshold` by at
  least the minimum step and is clamped to the ceiling. Clamping against the high-water mark rather
  than the current goal is what prevents a lower-then-raise cycle reusing the same band. A malformed
  reply is **not** re-asked: declining to raise is always safe, and the cleared goal is already
  banked in `convergence_banked`.
- **Controls:** `SLM_THRESHOLD_RAISE=0` disables; `SLM_CHEAP=1` skips; off by default under pytest,
  because every convergence assertion would otherwise make a live call.

---

## 2. Teacher and local-Qwen prompts

`data/synth_client.py::get_generate_fn` sends all local-Qwen prompts in **non-thinking** mode
with `top_p=0.8`, `top_k=20`, caller-supplied temperature and token limit. Transport-only
stages: `synth_preflight` (reachability) and `local_synthesis` (generation); both are recorded at
$0 as local-provider cost events.

`agent/nodes/curate.py` uses local `SYNTH_MODEL` and **never falls back to Claude for
synthesis.** CoT is authored **only** by the local Qwen3.6 synth endpoint — there is no cloud CoT
teacher; if the endpoint is unavailable the example is left CoT-less.

> **Neither completed run produced a single synthetic row.** The ledgers contain only
> `synth_preflight` events (8/9 failed in NER, 18/43 in math) and no generation events at all.
> Everything in this section is therefore
> **implemented but unexercised in production** — see
> [B200](BUGS.md#b200--synthesis-has-never-executed-at-scale-both-completed-runs-were-gold-only).
> The xlam run that followed had a healthy endpoint and still produced zero rows, for a different
> reason: `function_call` matched neither branch of the synthesis dispatch table and fell through
> to a bare `return []` (B291). The dispatch table no longer exists, but the conclusion stands —
> **nothing in this section has been exercised at scale in production.**

### 2.1 CoT/implementation-reasoning annotation

- **Code:** `data/curriculum.py::annotate_cot::_build_prompt`
- **Purpose:** add reasoning to gold examples that lack CoT, for the tasks that ask for it.
- **Which tasks:** `TaskSpec.cot_annotation`, per task since 2026-08-19 — **only `gsm8k`** in the
  current suite. CoT pays for itself when the gold answer is the end of a multi-step derivation.
  `dialogsum` forced the distinction: its summary is a compression of text already sitting in the
  prompt, so a reasoning chain adds nothing the model cannot read off its own input, while costing
  one teacher call for EVERY row.
- **In:** problem/prompt plus the correct answer/solution.
- **Out:** step-by-step reasoning *without the final answer*. (The code-task variant, which
  requested an implementation plan without code, went with `code_generation` on 2026-08-18.)
- **Validation:** any non-empty stripped text is accepted verbatim as `cot_reasoning`. Existing
  CoT is preserved. Sole backend: local Qwen3.6 (no cloud teacher); if it is unreachable the
  example is left unchanged.
- **Critique:** no check that the reasoning is consistent with the supplied gold, does not leak
  the final answer, or fits the training context after formatting.
- **Improve:** verify answer consistency, reject leakage when the prompt forbade it, and record
  backend/model/prompt-version per annotation.

### 2.2 Task-adaptive synthesis

*Redesigned 2026-07-31; rewritten 2026-08-19 when the task-type dispatch was removed.*

- **Code:** `data/curriculum.py::synthesize_examples`. Entered only from
  `curate._surgical_synthesize`, which is now the sole synthesis path.
- **Which shape a task gets is DERIVED, not dispatched.** `TaskSpec.closed_label_space` picks
  between the two branches below. That replaced an `if task_type == ...` table in which
  `function_call` appeared in neither branch and fell through to `return []` — six rebuilds on xlam
  announced 250–500 rows against a healthy endpoint and produced zero, silently, and the exact
  verifiers written for that path had never run in production (B291).
- **Every prompt here carries the task brief** ([§1.11](#111-task-brief-authoring)) — the
  orchestrator's own summary, output contract and expected failure modes — plus the **failure
  category this batch is aimed at**, so the teacher is asked for rows exercising what the model is
  actually getting wrong rather than for more of the same. All synthesis and verification prompts
  are **5-shot** (`SLM_SYNTH_SHOTS`), and the demonstrations are always real rows: the teacher scores
  0.1131 span-F1 zero-shot on BC5CDR NER and 0.7190 with five (B276/B281).
- **Purpose:** add rows that are **correct by construction**, never a wrong answer as a positive SFT
  target. It no longer "tops the curriculum up to its target size" — there is no target, and
  synthesis adds to a cumulative curriculum.
- **Closed-label-space branch** (`clinc150`, `routerbench`, `proactive_listening`): one new in-class
  example per anchor, with anchors drawn round-robin across labels so rare classes get the same
  attention as common ones.
  - *In:* the anchor's label and its text as a reference example, plus the label definitions.
  - *Out:* raw text only (explicitly: no preamble, no explanation, no quotes, no label prefix).
  - *Validation:* the generated row **inherits the anchor's label** — the model is never asked to
    choose one — so an out-of-vocabulary label is impossible by construction and the class
    histogram is left undisturbed. Rows are then label-verified (below).
- **Open-ended-target branch** (`gsm8k`, `dialogsum`, `xlam_bfcl`, `calendar_json`, `ner_bc5cdr`):
  - *In:* the anchor row reduced to its public keys, serialized as the target JSON schema. For the
    function-calling tasks `tools` and `_instruction` are **pinned from the anchor**, so the tool
    signature is a constraint rather than something being invented — without it a row cannot be
    schema-checked at all.
  - *Out:* one JSON object with the same keys and value types.
  - *Validation:* the task's `TaskSpec.synth_verifier` runs FIRST where one exists — it is free and
    unfoolable, and a row it rejects should never cost a teacher call. `verify_function_call_row`
    (xlam), `verify_calendar_row` (calendar), `verify_ner_row` (NER, new 2026-08-19). `gsm8k` and
    `dialogsum` have none, so the teacher's own answer check is the only gate, and that is logged as
    such rather than implied. Then `verify_generated_answers` asks "does this answer directly
    satisfy the request", which had previously been running with no check at all, keeping 100% of
    whatever the teacher produced (`450/450 kept` every batch, B269).
- **Label verification:** `verify_generated_labels` asks the same local model, at temperature 0,
  whether each generated row genuinely belongs to its assigned label, answering strict
  `{"valid": bool, "reason": "<max 15 words>"}`. Rejections are dropped with their stated reason
  logged, so a bad *generator* prompt is visible rather than silently absorbed. Any verification
  failure — unparseable reply or endpoint error — **keeps** the row, so the verifier can never
  empty a dataset. Disable with `SLM_VERIFY_SYNTH=0`.
- **Cost/parallelism:** generations and verifications both run concurrently (vLLM
  continuous-batches them) at `_synth_concurrency` workers, tunable via `SLM_SYNTH_CONCURRENCY`
  (default 16).
- **Critique:** teacher verification reuses the generator's own model, so correlated blind spots
  survive; deciding "does this belong to class X" is nonetheless a much easier task than writing the
  row. `verify_ner_row` cannot catch a **missed** entity, only an invented one.
- **Improve:** an exact arithmetic verifier for `gsm8k`, which is the one remaining open-ended task
  whose correctness is decidable by computation and currently is not checked; an independent
  verification model; and a preference/ranking objective for open generation.

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
- **Mode/backend:** the tasks whose spec sets `needs_judge=True` — **`dialogsum` alone** in the
  current suite, which also sets `judge_overlap=True` so judging of finished chunks overlaps the
  next generation batch. (Before 2026-08-19 this was `task_type == "generation"`, which also
  covered `gsm8k`; gsm8k is scored by exact match on the extracted final answer and never calls the
  judge.) Required local Qwen3.6 via the configured
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
  out of zero-shot baseline evaluation. This is what `needs_judge` exists to express: a judge outage
  on a task that depends on it must fail loudly rather than score zero. Every other task's scoring
  is unaffected.
- **Observability:** `generation_judge_preflight` (model identity) and `generation_judge`
  (completions) are recorded as local-provider cost events at $0, plus timing events.
- **Critique:** judge agreement is not periodically measured against a fixed calibration set. The
  scorer still carries this average in the field named `"f1"` —
  [B45](BUGS.md#b45--generation-scorer-misnames-metric) — but `EvalResult.metric` now names what the
  number really is (`judge_mean_0_1`), so reports cannot misattribute it. `f1` itself is
  deliberately never renamed: checkpoints and DAG replay depend on the field name.

---

## 4. SLM training and evaluation prompts

These run on the model being trained, so they have no provider cost stage. **Train/eval parity is
the property that matters here.**

> **Update 2026-08-19 — parity is now structural, not a discipline.** Each task names a
> `TaskSpec.build_training_turn`, and every one of the four builders in `tasks/_builders.py`
> **imports its prompt from the eval scorer** rather than reproducing it. That is not tidiness:
> when the two were written separately the model was fine-tuned on one input shape and scored on
> another (B250 for generation; and again for NER, where the training copy had quietly dropped the
> "Reply with `[]` if there are no entities" sentence). Importing makes them incapable of drifting,
> and `scripts/preflight_tasks.py` asserts the training prompt is byte-identical to the eval prompt
> for every registered task. The two ❌ rows that stood in this table are therefore closed.

| Task(s) | Shared prompt builder | Training turn | Extraction |
|---|---|---|---|
| `clinc150`, `routerbench`, `proactive_listening` | `eval/scorers/classification.py::build_classify_prompt` (from `CLASSIFY_PROMPT`) | `classification_turn` — same builder, target is the bare label | exact label → label word in the answer's TAIL → label anywhere in a SHORT answer → `__EXTRACTION_FAILED__` |
| `ner_bc5cdr` | `eval/scorers/ner.py::NER_PROMPT` | `ner_turn` — same string, target is the JSON span list | whole-list JSON or greedy list, keeping objects with both keys; **`None` on a parse failure** (B300); no exact-substring or type allow-list check at extraction |
| `gsm8k`, `dialogsum` | `eval/scorers/generation.py::build_generation_prompt` | `generation_turn` — same builder; prepends a `<reasoning>` block to the target when the row carries CoT | `dialogsum` → LLM judge; `gsm8k` → explicit final-answer marker, else the last number |
| `xlam_bfcl`, `calendar_json` | `eval/scorers/function_call.py::build_function_call_prompt` | `function_call_turn` — same builder; **raises** on an empty gold `answer` rather than training on nothing | JSON list of `{name, arguments}`, or `None` when it does not parse |

Notes:

- **Classification** critique: the short-answer fallback can still accept a label embedded in
  unwanted prose. The prompt now fences the message as DATA and restates the output contract
  *after* it, which is the mitigation for rows that are themselves instructions — RouterBench
  prompts like "Print only a single choice from A/B/C/D" made base models answer the row's embedded
  question instead of classifying (B271). Improve further with constrained decoding.
- **NER** improvement: enumerate the allowed types in the prompt and validate exact spans at
  prediction extraction. (Span validation *does* now run on synthesized rows, via
  `verify_ner_row`.)
- **`gsm8k`** improvement: require a stable final-answer marker, since the parser depends on one
  but the prompt never asks for it. Until then a correct derivation with no parseable number is a
  format failure, and `format_valid` is what makes that visible.
- **Code generation was deleted on 2026-08-18.** `CODE_GENERATE_PROMPT`, `build_code_prompt`, the
  APPS/MBPP execution scoring and the isolated-subprocess sandbox are all gone with the
  `code_generation` task type. The sandbox's properties are worth recording because they were the
  reason it could exist at all: the trusted controller retained expected outputs, candidate workers
  inherited no success FD or expected-output payload, and per-case timeouts bounded a run. It was
  never seccomp or a hostile-code sandbox, only a trusted-input harness — which is a liability to
  keep for benchmarks that are no longer in the suite.

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

## 5. Production prompts — REMOVED

> **Production mode was removed on 2026-07-29 and `agent/nodes/production/` no longer exists.**
> `build_graph` accepts only `cold_start` and `graph_topology_descriptor` raises on anything else;
> the `mode` parameter survives solely because it is part of the checkpoint compatibility
> fingerprint. The production entry chain (`trace_ingest` → `live_confirm` → `parent_awareness`) was
> never wired into the topology in the first place, which is what B199 recorded. This section is
> retained as history — the critiques below are still the right ones if production mode is ever
> rebuilt.

### 5.1 Live confirmation *(historical)*

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

### 5.2 Production training prompts *(historical)*

The production graph would have routed confirmed examples through the **same** shared `curate` →
`train` → `evaluate` → `iterate` nodes, so every prompt above would have applied unchanged. It never
ran: `curate_node` raises when `eval_set is None` and the production entry chain contained no node
that built one, so startup depended on caller-populated state that was never expressed or validated
as a contract. **Tracked as
[B199](BUGS.md#b199--production-graph-cannot-start-curate-requires-an-eval_set-nothing-builds)**,
and resolved by removing the mode rather than by wiring it.

---

## 6. Present but not active

- ~~**`agent/tools/delegate_task.py::delegate_task`**~~ and ~~**`agent/tools/web_search.py`**~~ —
  **the whole `agent/tools/` package was deleted.** A generic no-tools sub-agent prompt and four
  `@tool`-decorated wrappers (including the `iterate_web_search` Exa wrapper) sat there bound to no
  LLM, with zero call sites, from 2026-07-29 until they were removed. [B21](BUGS.md),
  [B23](BUGS.md) are retained as history.
- **Exa search strings** (`acquire_dataset_discovery`, `acquire_exa`, and
  `use_autoprompt=True`) — search *inputs*, not generation prompts, so they are outside this
  inventory even though they are cost-tracked. Still live, on rung 2 of the mining ladder.

---

## 7. Highest-priority improvements

*Reordered 2026-08-19: items 1, 4 and 5 were resolved, and what replaced them is listed below.*

1. **An exact verifier for `gsm8k`.** It is the only open-ended task in the suite whose correctness
   is decidable by computation and is currently checked by nothing but the teacher's own judgement
   (`synth_verifier=None`). The other four format-bound/extraction tasks are exact-verified.
2. **Typed JSON-schema validation** for the planner, hardware and acquisition outputs. `iterate`
   already has the strictest validator in the repo — a discriminated union, field allow-lists,
   integer-vs-numeric JSON typing, and recursive rejection of held-out eval text — and the others do
   not.
3. **Validate NER annotation output** at acquisition so a failed call cannot become an
   empty-entity gold row. [B203](BUGS.md#b203--_annotate_ner_entities-does-not-re-validate-spans-or-types).
   Reachable only from the autonomous path now, which lowers its priority but does not close it.
4. **Revisit the task brief mid-run.** It is authored once at cold start from five rows and never
   updated, even after the failure categories reveal a contract detail it missed. A brief that is
   wrong is worse than one that is thin, because every synthesis and verification prompt inherits it.
5. **Treat external content as untrusted data** — dataset rows and web snippets reach decision
   prompts. The judge already does this correctly (marked untrusted JSON + an explicit
   system-prompt instruction), and the classification prompt now fences its message the same way
   after B271; use those two as the template for the rest.

**Closed since the last revision.** *Unify train/eval builders* — done structurally: every
`build_training_turn` imports its prompt from the eval scorer, and `scripts/preflight_tasks.py`
asserts the two are byte-identical per task. *Verified-positive augmentation for open generation* —
partly done: three tasks gained exact verifiers and every open-ended path now runs teacher answer
verification, which previously kept 100% of whatever was generated (B269). *Reproduce the serving
prompt during live confirmation* — moot; production mode was removed.
