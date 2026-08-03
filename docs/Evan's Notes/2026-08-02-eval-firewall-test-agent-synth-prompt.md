# Three Questions: Eval Firewall at Setup, the Test Agent, and the Qwen3.6 Synthesis Prompt

**Date:** 2026-08-02
**Branch:** `data-curation-redesign`
**Purpose:** Direct answers to three questions, verified against current code (code is ground
truth). Companion to [2026-08-01-firewall-filtering-rebuild-and-escalation.md](2026-08-01-firewall-filtering-rebuild-and-escalation.md),
which this doc updates and extends.

Also records the two behavior changes shipped 2026-08-02 (curate no longer truncates to
`target_rows`; `resample` is gated when the pool is exhausted) — see the companion doc §2.5 and §3.2.

---

## Q1 — Does the orchestrator EVER look at the eval set during initial creation?

**No. The orchestrator LLM never receives a single raw eval row — not at setup, not during any
iteration.** Your concern ("even if it looks once, it will remember and cheat later") is structurally
prevented: there is no code path that puts eval rows in front of the orchestrator, so there is
nothing for it to memorize.

Where the eval set is built and used, and who touches it:

| Phase | Who reads raw eval rows | Is it the orchestrator? |
|---|---|---|
| Eval-set construction (`build_eval_set`, `eval_setup.py`) | deterministic Python (sampling/label-coverage stratification, normalized-overlap firewall) | **No** — no LLM call |
| Difficulty labeling (`label_difficulty`, `test_agent.py`) | the **base models** run zero-shot via the eval harness | **No** — base model inference, not the orchestrator |
| Qwen goal calibration (threshold) | the **local Qwen-3.6 reference model** + the first fine-tune | **No** |
| Per-iteration decision (`_llm_iterate`, `iterate.py`) | the orchestrator sees the **test-agent report only** (aggregates) | orchestrator — but aggregates only |
| Curate (`curate_node`) | uses `eval_set.all` **only** as the overlap firewall (drops matching train rows) | **No** — no LLM sees rows here |

### Why memorization-cheating cannot happen

1. **The orchestrator's only window into eval performance is the test-data agent's aggregate
   report** — per-difficulty accuracy (easy/medium/hard + counts), top-8 aggregate confusion
   *counts*, a diagnosis string, a suggested intervention, and a band. No row text. Assembled at
   `_llm_iterate` and fed to the model; see Q2 for exactly what those fields are.
2. **The system prompt states the rule** (`_ITERATE_SYSTEM`, `iterate.py`): *"never inspect,
   request, quote, or copy raw eval text. Use only aggregate difficulty scores and aggregate
   confusion counts."*
3. **Four independent firewall layers on normalized text** stop eval rows from reaching either the
   model *or* the training data (full detail in the companion doc §1.1):
   - **Layer 1 — build-time raise:** `eval_setup` raises `ValueError` on ANY normalized train/test
     overlap after acquisition.
   - **Layer 2 — curate drops overlap** at every touch point (train anchors, mined rows, synth
     rows, and the final dataset) via `_exclude_eval_rows`.
   - **Layer 3 — decision-prompt rejection:** `_reject_eval_text_strings` walks every string in the
     orchestrator's decision and raises if a normalized eval string (≥12 chars) appears; also
     applied specifically to `hypothesis` and to the plan's `pattern_hint`.
   - **Layer 4 — reask sanitization:** `_sanitize_reask_error` redacts eval text from validator
     error strings before they are replayed to the model.
4. **Synthesis and gold anchors draw only from the decontaminated `train_examples` pool**, so even
   generated data cannot smuggle an eval row back in.

### The honest boundary

The firewall matches on `normalize_text` (exact/near-exact string identity after normalization). It
catches verbatim leakage and duplicates; it does **not** catch a *paraphrase* of an eval item. And
difficulty labeling does let the **base models** (not the orchestrator) see eval rows once at setup —
that is base-model inference used to bucket difficulty, and those models never author training data
or decisions. If you want zero model exposure of any kind, the only remaining lever is difficulty
labeling, which you can switch to the length-heuristic mode (`SLM_DIFFICULTY=heuristic`) so **no
model of any kind reads an eval row** — buckets then come purely from text length terciles.

**Bottom line: the orchestrator is denied the eval rows by construction. It cannot cheat from
memory because it is never shown anything to memorize.**

---

## Q2 — What is the test agent, exactly? Difficulties, and the report fields?

### What it is

The **test-data agent** (`agent/nodes/test_agent.py`) is not an LLM. It is deterministic Python that
**owns the held-out eval set** and is the *only* component allowed to look at raw eval rows in bulk.
Its entire job is to convert the eval outcome into **aggregates** the orchestrator can act on without
ever seeing a row — it is the contamination firewall between "how did we do on the test set" and
"what should we change next."

It does three things:
1. **Labels each eval row's difficulty** once at setup (`label_difficulty`).
2. **Scores the latest eval bucketed by difficulty** (`score_by_difficulty`).
3. **Diagnoses the score pattern** into an actionable suggestion (`diagnose`) and packages everything
   into the report (`build_test_report`).

### The difficulties (easy / medium / hard)

Labeled by the **base-model zero-shot capability gradient** (`label_difficulty`,
[test_agent.py:47-88](../../agent/nodes/test_agent.py#L47-L88)) — run the **smallest** and
**largest** feasible base models zero-shot once at setup:

| Bucket | Definition |
|---|---|
| **easy** | both the smallest and largest base model get it right |
| **medium** | only the large model gets it right |
| **hard** | neither gets it right |

This deliberately captures the small→large capability gap that drives escalation: if even the big
model can't do it zero-shot, it's genuinely hard. **Fallback:** a text-length tercile heuristic
(`_length_heuristic_buckets`, longer = harder) is used when `SLM_DIFFICULTY=heuristic`, on any error,
or on a degenerate all-empty split — so buckets always exist.

### The report fields (`build_test_report`, [test_agent.py:173-224](../../agent/nodes/test_agent.py#L173-L224))

The report is a dict with exactly these keys:

- **`overall`** — the best model's overall score (`best_result.f1`; the metric is task-appropriate —
  macro-F1, accuracy, judge score, etc.).

- **`by_difficulty`** — `{easy/medium/hard: {"n": count, "accuracy": fraction | None}}`. Accuracy is
  the fraction correct *within that bucket*; `None` when the bucket is empty. Correctness per example
  is reconstructed by text-matching against `best_result.failures`.

- **`confusion_pairs` (top 8)** — the dominant **aggregate** error modes, as
  `{"gold": …, "predicted": …, "count": N}`, sorted by descending count then gold then predicted,
  capped at 8. What "gold"/"predicted" mean is task-adaptive:
  - **classification:** the true label vs the model's predicted label (truncated to 64 chars).
  - **NER:** the sorted set of gold entity *types* vs the predicted set of types (e.g.
    `"ORG,PER" → "ORG"`); empty sets render as `"no_entity"` / `"incorrect_entity_set"`.
  - **open-ended (generation/math/code/function_call/diff):** the raw target *is* the answer, so
    there is no safe "confusion pair" — it is reduced to an aggregate **verifier category**
    (`error_type` / `judge_category` / `"gold_verifier"`) vs `"incorrect"`. This is a firewall
    measure: it prevents the held-out answer from leaking through the confusion field.

- **`diagnosis`** — a human-readable string explaining *why* the score is where it is, derived from
  the bucket pattern (see `diagnose`, [test_agent.py:130-170](../../agent/nodes/test_agent.py#L130-L170)):
  - `overall ≥ threshold` → "converged."
  - **easy bucket < 0.6** → "misses even simple cases → data-quality / label-format / prompt
    problem, not capacity."
  - **easy solid but medium/hard < 0.6** → "optimization/capacity gap → tune hyperparameters, then
    escalate."
  - below goal with no single failing bucket → "add more balanced data and continue."

- **`suggested_intervention`** — the machine-actionable version of the diagnosis: one of `none`,
  `data_rebuild`, or `hyperparameter`. This is the failure-fallback intervention (used if the
  orchestrator LLM call fails) and is also fed into the orchestrator prompt as a recommendation.

- **`band`** — a short tag for the diagnosis regime: `converged`, `data`, `optimization`, or
  `general`. (Distinct from the score-band in `apply_iteration_policy`, which maps the raw score to
  data-vs-hyperparameter guidance in the prompt.)

Everything in this report is an aggregate. No raw eval row ever leaves the test agent.

---

## Q3 — What prompt does Qwen3.6 get when generating synthetic data on each curate?

### The backend (same for every synthesis call)

There is **one** synthesis backend: a local **Qwen3.6-35B-A3B** model served on an OpenAI-compatible
vLLM endpoint (`config.SYNTH_ENDPOINT`), accessed through
`data/synth_client.py::get_generate_fn`, which returns
`generate(prompt, temperature=0.7, max_tokens=200)`. Every call runs in **non-thinking mode**
(`chat_template_kwargs={"enable_thinking": False}`, `top_p=0.80`, `top_k=20`) for fast, direct output
with no `<think>` preamble. There is **no cloud fallback** — if the endpoint is unreachable, synthesis
degrades gracefully (fewer/zero rows, logged) rather than calling Claude. The prompt is passed as a
single user message; there is no separate system prompt.

### There is no single fixed prompt — it is task-adaptive, built per row

The prompt is **constructed in code per anchor row and per task type** in `data/curriculum.py`. There
is no prompt template file and no per-run configuration; the wording is hard-coded in the synthesis
functions and specialized by `task_type`. `synthesize_examples`
([curriculum.py:550-586](../../data/curriculum.py#L550-L586)) is the single dispatch:

| task_type | Path | What is generated |
|---|---|---|
| `classification` | `synthesize_hard_negatives` | contrastive **hard negatives** (2-for-1) |
| `NER` | `synthesize_hard_negatives` | harder-to-tag passages with **correct** types |
| `math_reasoning`, `code_generation`, `generation`, `multilingual`, `structured_extraction` | `_synthesize_new_correct` | **new correct** in-distribution examples |

### The actual prompts

**1. Classification hard negative** (`synthesize_hard_negatives`,
[curriculum.py:344-353](../../data/curriculum.py#L344-L353)) — for an anchor with label `X`, pick a
different target label `Y` and ask for a boundary-crossing counterexample:

> You are generating a HARD NEGATIVE for a text classifier: a realistic example that superficially
> resembles the '`{src_label}`' class but genuinely belongs to the '`{target_label}`' class. The
> surface features should mislead toward '`{src_label}`' while the true meaning is unambiguously
> '`{target_label}`'.`{pattern_hint clause, if present}`
>
> Reference '`{src_label}`' example: `{ex['text']}`
>
> Output ONLY the new example text for the '`{target_label}`' class — no preamble, no explanation, no
> quotation marks, no label prefix.

Run 2-for-1 (each gold row kept, its synthetic counterpart appended), concurrently via a bounded
thread pool (vLLM continuous-batches), `max_tokens=200`.

**2. NER hard example** (`synthesize_hard_negatives`,
[curriculum.py:390-403](../../data/curriculum.py#L390-L403)) — rewrite so the same entities sit in a
more confusable context but keep their **correct** types (an earlier version asked for wrong types
and caused negative transfer — that is fixed):

> You are generating a HARD training example for a named-entity recognizer. Rewrite the passage so
> the SAME entities appear in a more ambiguous or confusable context … so their correct type is
> harder to infer from surface form alone — but keep each entity's CORRECT type unchanged.
>
> Original entities (text → correct type): `{entity_desc}`
> Original passage: `{ex['text']}`
> Aggregate pattern to emphasize: `{pattern_hint or 'general ambiguity'}`
> Reply with JSON only: `{"text": "<rewritten passage>", "entities": [{"text": "<span>", "type": "<CORRECT_TYPE>"}]}`
> … JSON only, no prose.

Parsed as JSON, `max_tokens=400`; a synthetic row is kept only if every returned entity span is
present in the rewritten text and carries a type from the original valid set.

**3. New-correct example** (generation family) (`_new_example_prompt`,
[curriculum.py:493-504](../../data/curriculum.py#L493-L504)) — schema-preserving, from a randomly
shuffled anchor:

> Generate ONE new, correct `{task_type}` example in EXACTLY this JSON schema (same keys, same value
> types): `{json schema of the anchor, minus underscore keys}`. It must be a genuinely new, diverse,
> and CORRECT instance — not a copy or a paraphrase of the reference, and never a wrong answer. Return
> only the JSON object, no preamble or code fences.

Called with `temperature=0.7, max_tokens=512`, up to `n*4` attempts to reach `n` kept rows; each
candidate is parsed as JSON and, when a `verify_fn` is supplied (e.g. a math answer-checker or code
test-runner), kept only if it verifies. **Math/code/generation never receive wrong-answer negatives**
— wrong-answer SFT actively harms those families.

### CoT annotation (a separate Qwen3.6 call, generation family only)

On math/code/generation curates, `_annotate_generation_cot` → `annotate_cot`
([curriculum.py:18-118](../../data/curriculum.py#L18-L118)) makes a **separate** Qwen3.6 call per row
that lacks a gold chain-of-thought, at **low temperature (0.3), 512 tokens**:

- **code_generation:** *"Explain the reasoning behind this code solution as a concise implementation
  plan … Do NOT restate the full code. … Reply with only the step-by-step implementation reasoning."*
- **everything else:** *"Solve this problem step by step … The correct answer is: `{gold}` … Provide
  a clear step-by-step explanation … Reply with only the reasoning steps, not the final answer."*

Rows that already carry gold CoT (e.g. GSM8K) are left untouched — the teacher is not re-run on them.

### Where the prompt comes from, and does it change per task?

- **Comes from:** hard-coded strings in `data/curriculum.py` (no template file, no config, no
  per-run override). Anchors are the eval-decontaminated `train_examples` pool.
- **Changes per task:** **yes.** `synthesize_examples` dispatches on `task_type`; classification/NER
  get contrastive prompts, generation-family get schema-preserving new-correct prompts, and CoT
  annotation has its own code-vs-prose variant. The prompt is further parameterized per **row**
  (anchor text, labels, entity list, JSON schema).
- **`pattern_hint`:** the classification/NER contrastive prompts *do* interpolate `pattern_hint` when
  one is passed in. But note (companion doc §5.3): `curate_node` calls `synthesize_examples` **without**
  a `pattern_hint`, so in the live pipeline the hint the orchestrator writes into the plan does **not**
  currently reach generation. That remains an open lever.

---

## Q4 — How is the eval set actually split into pos / neg / boundary?

**Short answer: it isn't anymore. The pos/neg/boundary split was REMOVED (2026-08-02) because it
had no functional effect.** The eval set is now a single flat sample (`EvalSet.all`).

The investigation that prompted this question is *why* it was removed. The original design carried
three slices, but tracing every read showed:

- `pos`/`neg`/`boundary` were only ever recombined into `.all` — which is what all eval, prompting,
  difficulty labeling, and firewall matching actually use.
- The three per-slice scores flowed to exactly one place: a cosmetic `Epos | Eneg | Eboundary` line
  printed into the curation log. **Nothing branched on them.**
- For the NER and generation families the split was a plain shuffled positional partition — three
  statistically interchangeable random thirds, not semantic buckets. Only classification built
  label-aware buckets, and even those fed nothing downstream.

So the slices shaped neither training nor any decision — they were noise fed to the orchestrator via
one log line. They were deleted (code, serialization, tests, docs).

**What `build_eval_set` does now** ([eval_set.py](../../data/eval_set.py), `_eval_target` in
[eval_setup.py](../../agent/nodes/cold_start/eval_setup.py)):

- **Multi-class classification (>2 labels)** — keeps **label-coverage stratification**: a round-robin
  draw across every label up to the target, so `E` spans the full label range even when the target is
  smaller than the pool.
- **Every other task type** (binary classification, NER, math/code/generation/function_call/diff) — a
  plain shuffled top-N sample of `eval_size_target` rows (min-30 floor).

The **real** difficulty signal is easy/medium/hard (Q2), which is independent and was never part of
the pos/neg/boundary machinery.

**Bottom line:** your suspicion was correct — for the generation/NER families it was a three-way
random split, not true positive/negative/boundary examples, and since even the classification buckets
drove nothing, the whole split is gone.

---

## Q5 — How does the pipeline initially find the dataset it pulls the eval set from?

`eval_setup_node` ([eval_setup.py:214-389](../../agent/nodes/cold_start/eval_setup.py#L214-L389)) has
**four acquisition paths**, tried in order. Each returns `(train_examples, test_examples)`, and the
eval set is carved from `test_examples`:

1. **`SLM_SHARED_DATASET_DIR`** — a frozen, checksum- and manifest-verified train/test bundle (built
   once by `scripts/prepare_shared_dataset.py`). Used so competing strategy runs load *identical* data
   and the comparison isn't confounded by acquisition nondeterminism.
2. **Autonomous plan present** — `acquire_dataset(plan)` in
   [web_acquire.py](../../data/loaders/web_acquire.py). For a benchmark the plan *names* and the loader
   recognizes (`_BENCHMARK_ALIASES` — gsm8k, fpb, sms_spam, …), it loads the **actual HF dataset** with
   its official train/test split. For anything unknown it falls back to **Exa web search** — one query
   per class label (classification) or per topic (NER/generation) — scraping documents as raw text;
   supervision is synthesized later at curate.
3. **`SLM_BENCHMARK_TASK`** — one of the six curated loaders (`clinc150`, `dialogsum_samsum`,
   `xlam_bfcl`, `coedit`, `routerbench`, `medqa`).
4. **Fallback** — bundled SMS-Spam for `classification`; otherwise `NotImplementedError`.

After *any* path: a **normalized train/test overlap check raises `ValueError` on any leak**
([eval_setup.py:319-332](../../agent/nodes/cold_start/eval_setup.py#L319-L332)) — this is firewall
Layer 1 (Q1). Then `build_eval_set` samples the test rows (Q4), difficulty labeling runs (Q2), and
the Qwen-3.6 goal is calibrated. The eval set is frozen here, before any training.

**Sizing:** the orchestrator's plan chooses `curriculum_size_target` / `eval_size_target`; gold ≈
**65%** of the curriculum target, requested with ×1.15 + 40 headroom (to survive overlap removal +
quality-control drops), capped at `DATA_SIZE_CEILING`.

---

## Q6 — What does "data-constrained" mean here, and does it match "far from pretraining"?

**Yes — "data-constrained" in this project means the task is *not (significantly) in the base model's
pretraining distribution*, and that intent is wired into the planner.** It does **not** mean "few rows
exist upstream" — the benchmarks (CLINC150, MedQA, xLAM, …) have thousands of rows available. The
constraint is the *regime*, and the planner sizes data against it explicitly
([task_planner.py:104-117](../../agent/task_planner.py#L104-L117)):

> *"Task complexity / distance from pretraining: obscure, niche, or specialized benchmarks (little
> public data, unlikely to be well-covered in pretraining) need MORE data to instill the behavior.
> Widely-known, popular benchmarks (heavily represented in pretraining) need less — the base model
> already has the capability, so fine-tuning mostly selects the format."*

So the design's own definition of the scarce resource is **pretraining coverage of the task**, exactly
your intended meaning. Two reinforcing factors:

1. **Distance from pretraining** — the planner biases `curriculum_size` **up** for tasks unlikely to be
   in pretraining, **down** for popular ones the base already knows (where fine-tuning only instills the
   output *format*, not the capability).
2. **Small-model instillation regime** — the pool is sub-4B, which the config notes need *more* data
   than an 8B for the same task (the "selection→instillation regime shift", `CURRICULUM_SIZE_FLOOR=5000`
   with synth-fill to the floor).

### The honesty caveat

"Distance from pretraining" is judged by the **orchestrator's subjective popularity/complexity prior at
plan time** — it is *not measured*. The one empirical signal that actually reflects pretraining exposure
is the **base-model zero-shot difficulty gradient** (Q2) and the **Qwen-3.6 baseline threshold**: if the
base models already ace the eval zero-shot, the task is in-distribution and little instillation is
needed. But that empirical signal is **not fed back into `curriculum_size`** — sizing stays the
planner's up-front guess. And the six curated benchmarks are a *mix* (MedQA and GSM8K-style tasks are
well-represented in pretraining), so the framework *supports* the OOD-first intent but does not *enforce*
it. If you want the sizing to track real OOD-ness rather than a prior, the lever is to feed the
zero-shot base-model score back into the curriculum target — currently an open design point.

---

## Changes shipped alongside this note (2026-08-02)

1. **curate no longer truncates to `target_rows`.** The old `dataset = dataset[:target_rows]` is
   removed; `target_rows` is a floor for synth-fill, not a cap. Extra rows from an
   `acquire`/`synthesize` overshoot are kept — we only guard against too *few* rows.
2. **`resample` is gated when the pool is exhausted.** When the whole train pool is already in the
   curriculum, `resample` is removed from the strategy menu — the plan normalizer/fallback redirect
   it to `synthesize`, and the orchestrator prompt carries a "resample unavailable this turn" note,
   because reshuffling there adds no novelty.
3. **Dead code removed:** `_difficulty_sample` / `_with_train_difficulty` / curate-local
   `_DIFFICULTY_BUCKETS` (zero call sites), and `build_initial_curriculum` (no live callers).
4. **Full orchestrator context is now logged each iterate turn.** `_llm_iterate`
   ([iterate.py](../../agent/nodes/iterate.py)) dumps the complete decision input — system prompt
   (`_ITERATE_SYSTEM`) + the entire `user_content` (trajectory, current-iteration summary, test-agent
   aggregate report, already-tried configs, data-rebuild notes, source novelty/yield) — bracketed with
   `===== ORCHESTRATOR CONTEXT =====` markers before the LLM call. Every decision is now
   auditable/reproducible from the run log; no raw eval rows appear by construction (the report is
   aggregates only).

Full detail: companion doc §2.5 / §3.2, and [BUGS.md B211](../BUGS.md).
