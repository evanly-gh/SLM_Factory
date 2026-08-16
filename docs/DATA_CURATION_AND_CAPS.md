# Data Curation, Failure Analysis & Token Caps

> **Redesign 2026-07-31.** The data-curation subsystem was reworked (see
> [spec](superpowers/specs/2026-07-31-data-curation-redesign-design.md)). The six rebuild
> strategies collapsed to **three** — `resample`, `acquire`, `synthesize` — chosen freely by
> the orchestrator with no task/score gating. Curation is now **non-deterministic**
> (entropy-seeded sampling; the plan-identity dedup, untried-plan rotation, and
> `DataRebuildPlanSpaceExhausted` were removed). Synthesis is **ungated** and **task-adaptive**
> and **synth-fills** every curriculum up to the target size.
>
> **UPDATE 2026-08-05 — escalation policy.** ONE mechanism: escalate after
> `STAGNATION_WINDOW` (15) evals without a gain greater than `STAGNATION_MIN_DELTA` (2%),
> measured over an append-only eval history so rollback cannot reset it, plus an
> unconditional ceiling at `MAX_EVALS_BEFORE_ESCALATION` (30) evals. `MAX_STALL_EVALS` was
> deleted. Curriculum floor is 5000, ceiling 25000.
>
> **UPDATE 2026-08-05 — dataset sizing is now per-tier and deterministic.** `DATASET_SIZE_BY_TYPE`
> was deleted (every value sat below the floor and was clamped away). `agent/data_sizing.py`
> computes the target from two measured signals — task novelty (`1 − zero_shot_baseline`) and model
> capacity (inverse parameter count) — clamped to [5000, 25000], recomputed on entry to every tier:
>
> ```
> target = clamp(5000 × (0.5 + novelty) × clamp(1e9/n_params, 0.5, 2.0), 5000, 25000)
> ```
>
> A bigger model therefore gets a SMALLER target. Override with `SLM_CURRICULUM_SIZE`.
>
> **UPDATE 2026-08-05 — `synthesize` splits into fill and surgical.** `surgical_synthesize` spends
> `SLM_SURGICAL_SYNTH_SHARE` (default 20%) of the plan's `synth_rows` on the top
> `SLM_SURGICAL_MAX_PAIRS` (5) confusion pairs, budgeted **proportionally to each pair's confusion
> count** and clamped to 10–100 rows per pair. Anchors come from the **gold** class (the one the
> model should have predicted). A pair that was targeted before and whose count did not fall is
> logged `EXHAUSTED` and skipped. The remainder is `fill` — balanced across the whole label space.
>
> Sections below marked *(pre-redesign)* describe the old model and are retained for history.

A walkthrough of three tightly-related parts of the pipeline:

1. How the `data_rebuild` intervention works, including synthetic data generation.
2. How the system analyzes failures, and what that analysis is used for.
3. Every token cap in the repo and the rationale behind each.

---

## 1. The `data_rebuild` intervention & synthetic data generation

### The intervention model

The pipeline is a LangGraph loop (`agent/graph.py`):

```
task_analysis → eval_setup → model_selection → curate → train → evaluate → iterate ↺
```

Each iteration the orchestrator LLM picks **exactly one** of two mutually-exclusive
interventions (`agent/nodes/iterate.py:510-513`):

- **`data_rebuild`** → changes the training data → routes to `curate`
- **`hyperparameter`** → changes LoRA settings → routes to `train`

Only one thing changes per iteration so score movements stay causally attributable.
`curate_node` is a **no-op unless `last_intervention == "data_rebuild"`**
(`agent/nodes/curate.py:503-506`) — otherwise the dataset is frozen.

### The declarative "data_rebuild plan"

Curation is driven by a bounded, JSON-only plan (validated in `agent/data_rebuild.py`)
that declares a `primary_strategy` + up to two support strategies. The six strategies
(`agent/data_rebuild.py:24-31`):

| Strategy | What it does |
|---|---|
| `resample_existing` | reshuffle current pool |
| `preserve_elite_resample` | resample top-quality rows from a prior elite version |
| `mine_new_real_source` | acquire **new real** rows from HF/Exa |
| `source_diversification` | round-robin across sources |
| `difficulty_weighted_sampling` | sample by easy/medium/hard quotas |
| `targeted_synth_positive` | **LLM synthetic data generation** |

The plan validator (`normalize_data_rebuild_plan`, `agent/data_rebuild.py:313-517`)
clamps every numeric field (`target_rows` [16,2000], `synth_rows` [0,200], etc.), and
`ensure_untried_data_rebuild_plan` rotates the plan to an untried variant — raising
`DataRebuildPlanSpaceExhausted` to **terminate cleanly** (added after a 44.8-hour NER
run crashed on exhaustion, `agent/nodes/curate.py:547-560`).

### The curriculum never shrinks

The per-tier target is a **floor to build up to, never a reason to discard rows**. The sizing
formula is recomputed from each tier's own baseline and parameter count, so a *larger* model can
legitimately compute a *smaller* number — Qwen3-4B asked for 1961 (floored to 5000) immediately
after Qwen3.5-0.8B had asked for 5754. Applied literally that rebuilt the curriculum at 5000 and
threw away 754 rows that had already passed quality control.

Two guards enforce the contract:

- `agent/data_sizing.py::resize_curriculum_for_tier` ratchets the stored
  `curriculum_size_target` so it never drops below the previous tier's.
- `agent/nodes/curate.py` floors the effective `target_rows` at the size of the dataset already
  on disk, because the orchestrator can also write `target_rows` straight into a rebuild plan
  and bypass the sizing function.

Rows leave the curriculum only through quality control or the eval firewall. `SLM_CURRICULUM_SIZE`
still pins the size explicitly when that is what you want.

> **Stale below:** the strategy table above lists a six-strategy `primary_strategy` schema. The
> current validator accepts three strategies (`resample`, `acquire`, `synthesize`) and *rejects*
> `primary_strategy` outright. This section needs rewriting against `agent/data_rebuild.py`.

### Which synthesis path a task uses, and who produces the label

There are exactly TWO generators, selected by task family, and the distinction between them is the
single most important fact about synthetic-data safety here:

| Generator | Task types | Who produces the LABEL | Teacher skill matters? |
|---|---|---|---|
| `_synthesize_new_gold` (new in-class gold) | classification, NER | **we do** — copied from a real anchor | **No** — it never solves the task |
| `_synthesize_new_correct` (new correct example) | generation, function_call, diff | **the teacher does** | **Yes, critically** |

`_synthesize_new_gold` is safe *only when the label is a property of the text the generator
controls*. CLINC150 qualifies ("write another utterance meaning `accept_reservations`"). RouterBench
does NOT: `local` means "a small model answers this correctly", which is a property of a different
model's behaviour on the new text, so copying the anchor's label fabricates it.

**For NER this path is a documented no-op** (B268): NER rows carry no `label` to anchor on, so it
returns immediately without any teacher call. It must stay that way — the generator emits
`{text, label}` and never entity spans, so a row it produced would carry the anchor sentence's spans
against new text. NER curricula are gold-only, and BC5CDR converged at 0.8098 that way.

**Both generators are now verified.** `verify_generated_labels` checks classification labels (with
the label definitions above, so the teacher judges the class and not the label's wording, B267).
`verify_generated_answers` checks generation-family answers — "does this answer directly satisfy the
request, in the context of {task}" — which had been running with NO check at all, keeping 100% of
whatever the teacher produced (`450/450 kept` every batch, B269). Both fail OPEN: an unparseable
reply or endpoint error keeps the row, because a verifier must never be able to empty a dataset.

**Observed keep-rates track teacher competence directly**, which makes the keep-rate a usable signal:

| Task | Teacher zero-shot | Rows the teacher kept |
|---|---|---|
| `clinc150` | 0.8919 | 91% |
| `routerbench` | 0.5311 | 18-34% |
| `ner_bc5cdr` | 0.0999 | n/a (path is a no-op) |
| generation family | varies | was 100% (unverified) |

### Quality control: the length bound is anchored on TRUSTED rows

`apply_quality_controls` drops rows longer than **3× the median**. That ratio is fine; what it is
measured *against* was the bug (B260). The median used to be taken over the whole dataset, so injecting
short foreign rows collapsed it and pulled the cutoff down onto the real data:

| RouterBench dataset | injected rows | median | cutoff | share of the REAL benchmark deleted |
|---|---|---|---|---|
| clean | 0 | 715 | 2145 | **0.6%** |
| `v5` | 958 | 295 | 885 | 47% |
| `v10` | 1,155 | 269 | 807 | **48%** |

`_filter_length_outliers` now computes the median over rows whose `_provenance` is in
`_TRUSTED_LENGTH_PROVENANCE` (`train_anchor`, `resample`) — the task's own real data — falling back to
all rows when nothing is tagged. Genuine outliers are still removed; injected rows can no longer move
the goalposts. **None of the QC thresholds were loosened**; the contamination was the problem.

### The label vocabulary is pinned once and CLOSED

`eval_setup` pins `state["task_label_space"]` from the frozen eval set — definitionally the set of
classes the model is scored against, so a class absent from it cannot be scored and training rows
carrying it are unusable. After that the vocabulary is closed: every later stage may only DROP rows,
never extend it, and **no LLM may introduce a label**. See `data/label_space.py` for the full history
(B259: four hallucinated classes entered a two-class RouterBench task, one per acquire round).

| Stage | Enforcement |
|---|---|
| `eval_setup._pin_label_space` | pins from the frozen eval set, logs `[label-space] PINNED …` |
| `web_acquire._labels_are_usable` | rejects a whole source unless every label it carries is already in the vocabulary |
| `web_acquire.accept` | per-ROW drop of anything that still slips through a converter |
| `_llm_map_dataset` | states the exact permitted labels and forbids inventing one |
| `_materialize_from_mapping` | strips `label_map` entries targeting a non-existent label, and drops unmapped rows instead of passing them through verbatim |
| `apply_quality_controls` | unchanged last line of defence (`[qc] label-space`) |

**Strict only when the vocabulary is authoritative.** A space pinned from the eval set or declared in
a plan is complete, so strict subset rejection is safe. A space merely *inferred* from the rows a run
happens to hold may be missing real classes, and being strict there would discard good sources for a
class the pool had not seen yet — so the inferred case keeps the older any-overlap check (which still
catches B222's raw integer class ids).

**Label definitions.** `label_definitions_for(benchmark)` supplies what each class MEANS, injected into
both the synthesis and verification prompts. Without it the teacher judges the label's wording rather
than the task — it rejected 70% of RouterBench rows because a math problem "is not a local query"
(B267). Tasks whose label already describes the text (CLINC150 intents) need none.

### Rebuild KINDS — what a `data_rebuild` actually did

The plan's `strategy` names only the primary lever. Every rebuild additionally runs resample-fill
and, when still short, synth-fill — so a single `data_rebuild` normally combines several kinds, and
`strategy=synthesize` alone never told you whether rows came from the confusion-pair targeting or
from padding to target. Each producer stamps `_strategy_origin`; `REBUILD_KIND_NAMES` /
`rebuild_kind_name()` in `agent/nodes/curate.py` map those to one canonical token:

| `_strategy_origin` | Kind | What it does | Set by |
|---|---|---|---|
| `resample` | **reshuffle** | re-draw from the existing pool; no new material | `_balanced_sample` (universal filler) |
| `mine_new_real_source` | **mine-new-real** | local bundles → HF → paid Exa discovery | `_tag_mined_rows`, `strategy="acquire"` |
| `surgical_synthesize` | **surgical-synth** | targeted gold for the top confusion pairs | `_surgical_synthesize`, 20% of `synth_rows` |
| `synthesize` | **fill-synth** | balanced across the label space | `_synthesize_positive_rows`, remaining budget |
| `synth_fill` | **synth-fill** | pad to `target_rows` when everything else fell short | `_synth_fill_to_target` |

An unmapped origin passes through unchanged rather than being hidden as "unknown".

Every rebuild logs the kinds that actually produced rows, with novelty:

```
DATA REBUILD kinds (plan strategy=synthesize): reshuffle=3308 row(s) (122 novel);
synth-fill=64 row(s) (62 novel); fill-synth=20 row(s) (19 novel); surgical-synth=9 row(s) (9 novel)
```

`strategy_composition` entries in `last_curation` carry `kind` alongside `strategy`.

**Counting synthetic rows.** `n_hard` / `n_hard_generated` count only `_provenance == "synthetic"`
(the plan's `synthesize` strategy). Use **`n_synth_total`** for every teacher-generated row —
`synthetic` + `synthetic_fill` + `synthetic_positive` — with the split in `n_synth_by_provenance`.
The old field made a rebuild that generated 990 rows and kept 93 report 29, and `run_graphics` drew
a synthetic band of ~1% while the rest sat in the grey "unattributed" band. `run_graphics` now plots
`n_synth_total`, falling back to `n_hard_generated` for runs that predate the field.

### Where synthetic data is actually generated

Entered via `_synthesize_positive_rows` (plan `synthesize` strategy) or
`_synth_fill_to_target` (top-up), both in `agent/nodes/curate.py`, then
`data/curriculum.py::synthesize_examples`.

**The backend is a LOCAL vLLM endpoint serving Qwen3.6-35B-A3B, *not* Claude**
(`data/synth_client.py:1-13`) — chosen for zero Claude cost, reproducibility, and
contamination safety. It runs in non-thinking mode (`enable_thinking=False`,
`top_p=0.80`, `top_k=20`).

> **Correction 2026-08-15.** The claim below that label inheritance makes a mismatched label
> "impossible by construction" is **only true when the label is a property of the text that the
> generator controls**. CLINC150 qualifies — "write another utterance meaning `accept_reservations`"
> controls the intent the label names. RouterBench does **not**: `local` means "a small on-device
> model answers this correctly", which is a property of a *different model's behaviour on the new
> text*, so copying the anchor's label fabricates it. The teacher's own keep-rate tracks this
> directly — `clinc150` 91%, `routerbench` 18–34%, `calendar_json` unverified at 100%. See
> `Evan's Notes/2026-08-15b-label-space-lockdown-synthesis-quality-and-open-decisions.md` §0.
>
> Also corrected: **NER synthesis has never produced a row** (B268). NER rows carry no `label`, so
> the bucketing below is always empty and the generator returns immediately. It is now a documented,
> intentional no-op — NER curricula are gold-only.

**Gold-only generation (2026-08-05).** For every task family, synthesis produces NEW CORRECT
in-distribution examples. A generated row **inherits its label from a real anchor row** rather
than being assigned one by the model, so an out-of-vocabulary or mismatched label is impossible
by construction *for tasks whose label the generator controls* (see the correction above):

- **classification / NER** — `_synthesize_new_gold`: anchors are drawn **round-robin across the
  label space** (so rare classes get equal attention), and the model is asked for one new
  utterance of the *same* class as its anchor.
- **math/code/generation** — `_synthesize_new_correct`: new correct instances in the anchor's
  JSON schema, kept only if a supplied `verify_fn` accepts them. Wrong-answer SFT harms these
  families. They also get **chain-of-thought annotation** (`annotate_cot`).

**Teacher label verification** (`verify_generated_labels`, on by default, `SLM_VERIFY_SYNTH=0`
to disable). Every generated classification row is shown back to the teacher model, which
answers `{"valid": bool, "reason": str}`. Rejected rows are dropped and the teacher's reason is
logged. Verification failures (unparseable reply, endpoint error) KEEP the row — the verifier
can never empty a dataset.

### The "honest attribution" invariant

A key design rule: a plan declaring synthesis that produces **zero** synthetic rows is a
"strategy attribution lie." So if `SLM_REQUIRE_SYNTH != "0"` (default) and the local
endpoint isn't reachable, the run **stops** with `SynthesisUnavailableError` rather than
silently falling back to gold-only (`agent/nodes/curate.py:385-403`). When a strategy
legitimately no-ops, it's rewritten to `base_fill` and logged in `allocation_fallbacks`.

### Real-data acquisition ladder

For `mine_new_real_source` and initial setup, `data/loaders/web_acquire.py` walks:
local bundle → deterministic benchmark loaders (GSM8K, CoNLL, BC5CDR…) → agentic HF
discovery via Exa + orchestrator column-mapping → bounded Exa web scrapes →
**last-resort gold synthesis via the Claude orchestrator** (distinct from the local curriculum
synthesis above).

> **Update 2026-08-12.** Rung 2 is more fragile than it reads. Under `datasets>=4` any HF repo
> with a loading script at its root is permanently unloadable (`RuntimeError: Dataset scripts are
> no longer supported`), which kills `tner/bc5cdr`, `AmazonScience/massive` and
> `iohadrubin/smcalflow` outright, and repos whose files don't match a split-name pattern raise
> `DataFilesNotFoundError`. Every Stage-0 entry needs a script-free candidate — the raw-JSON
> fallback at `web_acquire.py:153-161` is the pattern to copy. Prefer rung 1: a frozen bundle
> under `data/local/` never breaks and is checksummed. See B253–B255 and
> `docs/Evan's Notes/2026-08-12-two-new-tasks-and-loader-blockers.md`.

Everything runs through an **eval firewall** (`_exclude_eval_rows`, applied at mining,
synthesis, and final write) to prevent contamination, and output is atomically written to
`artifacts/dataset_v{N}.jsonl`.

---

## 2. Failure analysis: the "test-data agent"

### What it produces

Failure analysis lives in `agent/nodes/test_agent.py`. It's a **contamination
firewall**: it reports only aggregate numbers, never raw failing examples. It produces
three things:

1. **Difficulty bucketing** (easy/medium/hard) by base-model capability gradient — easy =
   both smallest & largest base models pass; hard = neither
   (`agent/nodes/test_agent.py:47-88`).
2. **A confusion table** — top-8 `(gold, predicted)` pairs per task type in a `Counter`
   (`agent/nodes/test_agent.py:180-216`).
3. **A rule-based diagnosis → intervention** (`agent/nodes/test_agent.py:130-170`): easy
   bucket < 0.6 → `data_rebuild`; easy solid but medium/hard weak → `hyperparameter`;
   below goal with no single failing bucket → `data_rebuild`.

### What the failure analysis is used for

- **Iteration decisions** (`agent/nodes/iterate.py:778-801`): per-difficulty accuracy +
  diagnosis + confusion pairs feed the orchestrator prompt, driving the next intervention
  and even letting the LLM lower `stop_threshold` if a failure reflects genuine capacity
  limits.
- **Reporting** (`data/curation_log.py:82-102`): top confusion pairs are written to the
  curation log under an "Aggregate confusion counts" heading.
- **Targeted curriculum synthesis** (`agent/nodes/curate.py::_surgical_synthesize`): the top
  confusion pairs set the budget for extra in-class gold rows, so synthesis effort follows the
  evidence about which decision boundaries the model is actually getting wrong.

---

## 3. Every token cap in the repo, and why

No token caps live in YAML/JSON — they're Python constants, function defaults,
env-var-with-default, or the `_EVAL_OUTPUT_TOKEN_SETTINGS` table.

### Response/generation caps that affect the shipped SLM

**Eval output reserve — the central task-aware cap** (`eval/harness.py:8-14`):

```python
_EVAL_OUTPUT_TOKEN_SETTINGS = {
    "classification": ("SLM_EVAL_MAX_NEW_TOKENS_CLASSIFICATION", 50),
    "NER":            ("SLM_EVAL_MAX_NEW_TOKENS_NER", 512),
    "math_reasoning": ("SLM_EVAL_MAX_NEW_TOKENS_MATH", 512),
    "generation":     ("SLM_EVAL_MAX_NEW_TOKENS_GENERATION", 512),
    "code_generation":("SLM_EVAL_MAX_NEW_TOKENS_APPS", 1024),
}
```

**Why these values:** they match the shape of a correct answer. Classification emits a
single label → 50 tokens is plenty and prevents rambling into wrong territory. NER/math/
generation need room for entity lists / reasoning chains / summaries → 512 (raised from
256 in B147). Code needs the most → 1024. `eval_output_token_reserve()`
(`eval/harness.py:17-49`) validates the reserve is positive and `< max_seq_length`,
guaranteeing prompt budget remains. Defaults are duplicated in
`agent/checkpoint.py:150-154` so resumed runs reproduce identical caps.

**Max sequence length — the per-task context cap** (`training.slm_helpers.task_max_seq_length`,
overridable for every task with `SLM_MAX_SEQ_LENGTH`): 1024 classification, 2048
generation/math/NER/function_call/diff, 4096 code_generation, 4096 unrecognised. Used for model
load, GGUF `n_ctx`, and training `SFTConfig.max_length`. **Why:** input budget is always
`max_seq_length - max_new_tokens`, and the code **never truncates** — over-length rows raise
rather than silently dropping content. Notably `generation_config.max_length = None` disables
Qwen's built-in 40960 cap so `max_new_tokens` is the *sole* length knob (fix B128).

A single global 4096 was raised from 512 (B204/B188) to fit APPS code prompts, then split per
task in 2026-08 — context is *allocated* whatever the rows contain, so 4096 against a measured
max of 1206 tokens left ~70% of the KV cache and position buffers untouched, memory a larger
eval batch can use instead. Code generation keeps the full window. This governs the SMALL MODEL
only; the orchestrator's Claude context is an unrelated budget.

**Inference primitive defaults** — `max_new_tokens=50` on
`infer`/`infer_batch`/`infer_batch_gguf` (`training/slm_helpers.py:475,622,663`).
**Why:** a safe minimal default for the common classification case; eval always overrides
it via the table above.

### Synthetic-data generation caps

- **Local synth default 200 tokens** (`data/synth_client.py:177,188`) — generated classification
  rows are short texts; 200 keeps them tight and fast under continuous batching.
- **CoT authoring 512 tokens** (`data/curriculum.py:147`) — reasoning chains need room. This is
  the only CoT path: the local Qwen3.6 synth endpoint. There is no cloud CoT fallback; if it is
  unreachable the example is left CoT-less.

### Judge and orchestrator caps

- **LLM-judge: `max_tokens=8`** (`eval/judge_client.py:489-490`) — the judge outputs only
  a score token, so 8 forbids explanation drift and cuts cost.
- **Orchestrator (Claude) per-call caps** — hardcoded per decision, sized to the JSON each
  stage returns: task analysis 1024 (`agent/task_planner.py:263`), iterate 1536
  (`agent/nodes/iterate.py:49`), escalate 256, downward_probe 120, model_selection 256,
  schema-mapping 400, **seed synthesis 4096** (`data/loaders/web_acquire.py:1779`).
  **Why varied:** each is tuned to the expected output size — a yes/no probe gets 120,
  seed-data generation gets 4096. The orchestrator's *input* side is separately governed
  by the Sonnet 1M-token beta (`config/config.py:90-104`) so long trajectories aren't
  truncated.
- **Trajectory compaction at 8000 tokens** (`agent/context_manager.py:118-120`) —
  proactively compacts agent history before it bloats orchestrator calls.

### Overarching justification

Three consistent principles explain every cap:

1. **Right-size output to the task** — a label gets 50, code gets 1024, a judge score gets
   8. Tight caps prevent runaway generation, cut cost/latency, and stop models wandering
   into wrong answers.
2. **Never silently truncate** — input budget = context − output reserve, and over-length
   rows raise rather than lose data. Caps are enforced as hard invariants, not lossy
   conveniences.
3. **Reproducibility** — defaults are duplicated into checkpoint snapshots and SLURM
   exports so a resumed or re-run job reproduces identical caps.
