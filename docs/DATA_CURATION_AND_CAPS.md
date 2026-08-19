# Data Curation, Failure Analysis & Token Caps

> **Redesign 2026-07-31.** The data-curation subsystem was reworked (see
> [spec](superpowers/specs/2026-07-31-data-curation-redesign-design.md)). The six rebuild
> strategies collapsed to **three** — `resample`, `acquire`, `synthesize` — chosen freely by
> the orchestrator with no task/score gating. Curation is now **non-deterministic**
> (entropy-seeded sampling; the plan-identity dedup, untried-plan rotation, and
> `DataRebuildPlanSpaceExhausted` were removed). Synthesis is **ungated** and **task-adaptive**.
> (The synth-fill-to-target part of that redesign was removed again on 2026-08-16 — nothing pads a
> curriculum to its target size. See the 2026-08-16/17 update below.)
>
> **UPDATE 2026-08-05 — escalation policy.** ONE mechanism: escalate after
> `STAGNATION_WINDOW` (15) evals without a gain greater than `STAGNATION_MIN_DELTA` (2%),
> measured over an append-only eval history so rollback cannot reset it, plus an
> unconditional ceiling at `MAX_EVALS_BEFORE_ESCALATION` (30) evals. `MAX_STALL_EVALS` was
> deleted. Curriculum floor is 5000, ceiling 25000.
>
> **UPDATE 2026-08-05 — dataset sizing is now per-tier and deterministic.** `DATASET_SIZE_BY_TYPE`
> was deleted (every value sat below the floor and was clamped away). `agent/data_sizing.py`
> (itself deleted 2026-08-19, B305)
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
> **UPDATE 2026-08-19 — the task-registry rebuild supersedes most of §1.** Three things changed at
> once and they touch nearly every claim below. (1) **`task_type` no longer exists**: behaviour
> comes from a per-task `TaskSpec` in `tasks/`, one module per benchmark, every field required.
> Anywhere this document says "for classification/NER" or "for the generation family", read
> "for the tasks whose spec says so". (2) **The curriculum is CUMULATIVE and has no target** — cold
> start loads `TaskSpec.initial_train_cap` (a flat 5,000 for all eight tasks), every rebuild ADDS,
> rows leave only via quality control or the eval firewall. The per-tier sizing formula still exists
> but nothing calls it (B305). (3) **`data_rebuild` has exactly two sub-strategies**,
> `mine_new_real` and `surgical_synthesis`. `resample`, the universal gold fill, synth-fill and
> untargeted balanced synthesis are all gone, and a plan naming any of them is now *rejected*.
>
> **[`interventions.md`](interventions.md) is the authority for the rebuild loop.** Corrections are
> made in place below and dated; the reasoning is in
> `Evan's Notes/08-19b-task-registry-rebuild.md`, and B299–B305 record what was found on the way.
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
interventions (`agent/nodes/iterate.py`):

- **`data_rebuild`** → changes the training data → routes to `curate`
- **`hyperparameter`** → changes LoRA settings → routes to `train`

Only one thing changes per iteration so score movements stay causally attributable.
`curate_node` is a **no-op unless `last_intervention == "data_rebuild"`** — otherwise the dataset
is frozen.

### The declarative "data_rebuild plan"

Curation is driven by a bounded, JSON-only plan validated in `agent/data_rebuild.py`. **Schema
version 3, rewritten 2026-08-19**, has five fields and exactly two strategies:

| Field | Domain |
|---|---|
| `schema_version` | must be `3` |
| `strategy` | `mine_new_real` \| `surgical_synthesis` |
| `rows` | [50, 2000] — how many rows this rebuild may ADD |
| `target_categories` | ≤8 `{"category", "count"}` entries from the task's own failure taxonomy |
| `pattern_hint` | free text, ≤1200 chars |

| Strategy | What it does |
|---|---|
| `mine_new_real` | add REAL rows: re-read the task's own unexhausted `mining_sources` for a larger head slice (free) → web research only once all are exhausted → retired after `MAX_FAILED_DISCOVERY_ROUNDS` (2) fruitless rounds |
| `surgical_synthesis` | add TEACHER-GENERATED rows aimed at the failure categories costing the most points, budgeted proportionally to each category's failure count and bounded to [25, 600] rows per category |

**Unknown fields are rejected, not trimmed.** A plan that sets something we removed is a plan
written against the wrong contract, and silently dropping it would let the orchestrator believe it
had asked for something. The rejected set is `target_rows`, `resample_fraction`, `new_real_rows`,
`synth_rows`, `max_acquire_rounds`, `difficulty_buckets` and `confusion_pairs`.

> **What this table used to say, and why the whole schema was replaced.** Two generations ago it
> listed a `primary_strategy` plus up to two support strategies drawn from six
> (`resample_existing`, `preserve_elite_resample`, `mine_new_real_source`,
> `source_diversification`, `difficulty_weighted_sampling`, `targeted_synth_positive`); the
> 2026-07-31 redesign collapsed that to three (`resample`, `acquire`, `synthesize`). Both schemas
> shared a defect: they let the orchestrator specify *quantities* for mechanisms that could not
> deliver them. `difficulty_weighted_sampling` never sampled by difficulty, `synth_rows` was
> announced on a task whose synthesis path did not exist (B291), and `target_rows` named a
> curriculum size the loop had no way to reach. Version 3 asks for one mechanism and one row count
> and rejects everything else. `ensure_untried_data_rebuild_plan` and
> `DataRebuildPlanSpaceExhausted` were removed with the 2026-07-31 redesign — escalation on
> no-improvement is the stuck-run backstop now.

### The curriculum only grows

**Update 2026-08-19.** The curriculum is CUMULATIVE and has **no target size**. Cold start loads
`TaskSpec.initial_train_cap` gold rows — a flat 5,000 for every task in the suite, capped by
whatever the source actually has — and every rebuild ADDS to what is already on disk via
`_dedupe_into`, which is the single place it grows. Rows leave only through quality control or the
eval firewall. `MIN_CURRICULUM_ROWS = 500` is a **viability floor** that raises when real data
cannot supply a usable curriculum; it is not a target and nothing pads toward it.

*(pre-redesign, retained for history)* The per-tier target used to be a floor to build up to. The
sizing formula was recomputed from each tier's own baseline and parameter count, so a *larger* model
could legitimately compute a *smaller* number — Qwen3-4B asked for 1961 (floored to 5000)
immediately after Qwen3.5-0.8B had asked for 5754, and applied literally that rebuilt the curriculum
at 5000 and threw away 754 rows that had already passed quality control. Two guards existed for
that: `resize_curriculum_for_tier` ratcheted `curriculum_size_target` so it never dropped below the
previous tier's, and curate floored the effective `target_rows` at the size of the dataset already
on disk.

**Both guards are now moot, and the formula they protected has no production call site (B305).**
`resize_curriculum_for_tier` is referenced only by tests and this documentation set;
`curriculum_size_target` survives in state and is read only by the *autonomous* acquisition path,
where the gold request is still `0.65 × curriculum_size_target`. The 3,250 that appeared in every
xlam log was that same fraction of a 5,000-row target — 65% of a number the curriculum was never
allowed to reach. `SLM_CURRICULUM_SIZE` no longer pins a curated run's size.

### Which synthesis path a task uses, and who produces the label

There are exactly TWO row shapes, and which one a task gets is **derived from the task**
(`TaskSpec.closed_label_space`) rather than chosen by a channel. The distinction between them is
still the single most important fact about synthetic-data safety here:

| Row shape | Tasks | Who produces the LABEL | Teacher skill matters? |
|---|---|---|---|
| new **input** for an existing class (`_synthesize_new_gold`) | `clinc150`, `routerbench`, `proactive_listening` | **we do** — inherited from a real anchor | **No** — it never solves the task |
| new **(input, answer)** pair (`_synthesize_new_correct`) | `gsm8k`, `dialogsum`, `xlam_bfcl`, `calendar_json`, `ner_bc5cdr` | **the teacher does** | **Yes, critically** |

Inheriting the anchor's label is safe *only when the label is a property of the text the generator
controls*. CLINC150 qualifies ("write another utterance meaning `accept_reservations`"). RouterBench
does NOT: `local` means "a small model answers this correctly", which is a property of a different
model's behaviour on the new text, so copying the anchor's label fabricates it.

> **Update 2026-08-19 — two corrections to what stood here.** The old table keyed these paths on
> `task_type`, and `function_call` appeared in **neither** branch, so `synthesize_examples` fell
> through to a bare `return []`: six rebuilds on xlam announced 250–500 rows against a healthy
> teacher endpoint and produced zero, silently, and the exact verifiers written for that path had
> never executed in production (B291). The dispatch table it lived in no longer exists.
>
> **NER synthesis is no longer a no-op** (this section previously recorded it as a documented one,
> B268). BC5CDR rows carry no `label`, so the closed-label path was never right for them — but they
> have an open-ended target, which is the *other* path, and that path emits whole rows including
> spans. NER now generates through `_synthesize_new_correct` and is gated by `verify_ner_row`, which
> checks that every span appears **verbatim** in the row's own text with no duplicates. That catches
> the dominant teacher error, a plausible entity that was never written down; it cannot catch a
> **missed** entity, so the teacher pass still runs and that limit is stated in the code. BC5CDR's
> 0.8098 was reached with a gold-only curriculum, which remains the number to beat.

**Both paths are verified, and three tasks are verified twice.** `verify_generated_labels` checks
inherited class labels (with the label definitions above, so the teacher judges the class and not
the label's wording, B267). `verify_generated_answers` checks open-ended answers — "does this answer
directly satisfy the request, in the context of {task}" — which had been running with NO check at
all, keeping 100% of whatever the teacher produced (`450/450 kept` every batch, B269). Both fail
OPEN: an unparseable reply or endpoint error keeps the row, because a verifier must never be able to
empty a dataset.

Ahead of the teacher pass, a task may declare an **exact programmatic verifier**
(`TaskSpec.synth_verifier`), which runs first because it is free and cannot be fooled — a row it
rejects should never cost a teacher call:

| task | exact verifier | what it proves |
|---|---|---|
| `xlam_bfcl` | `verify_function_call_row` | parses; targets a **declared** tool; arguments exist in its schema; required ones present |
| `calendar_json` | `verify_calendar_row` | the above, plus datetimes parse, `end` follows `start`, an unstated duration is 60 minutes, and the event resolves near the request's own reference instant |
| `ner_bc5cdr` | `verify_ner_row` *(new 2026-08-19)* | every span appears verbatim in the row's own text, no duplicates |
| the other five | `None` | only the teacher's judgement is available, and that is logged as such rather than implied |

`programmatic_verifier_for` and the `synth_verifiers._BY_BENCHMARK` side registry are gone; the
verifier is a required field on the spec, so a task with none had to say so explicitly.

**Observed keep-rates track teacher competence directly**, which makes the keep-rate a usable signal:

| Task | Teacher zero-shot | Rows the teacher kept |
|---|---|---|
| `clinc150` | 0.8919 | 91% |
| `routerbench` | 0.5311 | 18-34% |
| `ner_bc5cdr` | 0.0999 | n/a at the time — the path was a no-op then; it now generates and is exact-verified |
| open-ended tasks | varies | was 100% (unverified) before B269 |

### Quality control is now per task, not per channel (B299)

**Update 2026-08-19.** `apply_quality_controls` used to be one `if task_type == ...` chain ending in
`else: return dataset`. Measured across the suite, that meant **four of the eight tasks received no
quality control at all**, silently: `xlam_bfcl` and `calendar_json` (`function_call`) fell into the
`else` and were returned untouched, and `gsm8k` and `dialogsum` entered their branch but filtered
length and near-duplicates on a `"prompt"` key their rows do not carry — a 100,000-character row
survived and nothing was logged. Only `clinc150`, `routerbench`, `proactive_listening` and
`ner_bc5cdr` were actually filtered.

Each task now lists the steps it wants, in order, in `TaskSpec.quality_controls`, composed from the
named units in `data/quality_controls.py`:

| Step | What it drops |
|---|---|
| `require_fields(...)` | rows missing any field the task named |
| `label_space(strict=...)` | rows whose label is outside the pinned closed vocabulary (B222/B229) |
| `balance_labels(max_ratio=3)` | rows past 3× the smallest class |
| `length_outliers(key, max_ratio=3.0)` | rows whose *stated* field exceeds 3× the trusted median |
| `dedup_surface(key, threshold=0.9)` | near-duplicates by word-set Jaccard over a sliding window |
| `entity_diversity(cap=3)` | rows re-using an entity surface form already seen `cap` times |
| `valid_json_answer()` | rows whose gold `answer` is not parseable JSON |

An empty tuple is a legal, visible choice; falling through is impossible because there is no branch.
Two properties matter more than the list itself. **The field a step filters on is stated by the
task, not guessed from a channel** — guessing is exactly what made this a no-op for gsm8k and
dialogsum. And **a step that cannot find its field logs loudly and skips** rather than passing
silently. `xlam_bfcl` and `calendar_json` also gained `valid_json_answer`, which matters most there:
a gold answer that does not parse trains the model to emit something the scorer marks wrong no
matter what it predicts.

### The length bound is anchored on TRUSTED rows

`length_outliers` drops rows longer than **3× the median**. That ratio is fine; what it is
measured *against* was the bug (B260). The median used to be taken over the whole dataset, so injecting
short foreign rows collapsed it and pulled the cutoff down onto the real data:

| RouterBench dataset | injected rows | median | cutoff | share of the REAL benchmark deleted |
|---|---|---|---|---|
| clean | 0 | 715 | 2145 | **0.6%** |
| `v5` | 958 | 295 | 885 | 47% |
| `v10` | 1,155 | 269 | 807 | **48%** |

`length_outliers` computes the median over rows whose `_provenance` is in
`TRUSTED_LENGTH_PROVENANCE` (`train_anchor`, `resample`) — the task's own real data — falling back to
all rows when nothing is tagged. Genuine outliers are still removed; injected rows can no longer move
the goalposts. **None of the QC thresholds were loosened**; the contamination was the problem.
(The step was called `_filter_length_outliers` in `data/curriculum.py` before it became a composable
unit on 2026-08-19; `resample` is retained in the trusted set so historical artifacts still measure
correctly, even though no code path produces that provenance now.)

### The label vocabulary is pinned once and CLOSED

`eval_setup` pins `state["task_label_space"]` from the frozen eval set — definitionally the set of
classes the model is scored against, so a class absent from it cannot be scored and training rows
carrying it are unusable. After that the vocabulary is closed: every later stage may only DROP rows,
never extend it, and **no LLM may introduce a label**. See `data/label_space.py` for the full history
(B259: four hallucinated classes entered a two-class RouterBench task, one per acquire round).

| Stage | Enforcement |
|---|---|
| `eval_setup._pin_label_space` | pins from the frozen eval set, logs `[label-space] PINNED …` |
| `web_acquire.accept` | per-ROW drop of anything that slips through a converter |
| `_llm_map_dataset` | states the exact permitted labels and forbids inventing one |
| `_materialize_from_mapping` | strips `label_map` entries targeting a non-existent label, and drops unmapped rows instead of passing them through verbatim |
| `quality_controls.label_space` | unchanged last line of defence (`[qc] label-space`) |

> **Update 2026-08-19 — `_labels_are_usable` is gone, and so are two other whole-source
> rejections.** It refused an entire dataset unless *every* label it carried was already in the
> vocabulary, which threw away a source whose mapping got most rows right for the sake of the few it
> did not. Labels are filtered **per row** now, alongside the two other checks that were relaxed at
> the same time: a source is no longer refused for carrying extra columns we do not need, nor for
> yielding a single-class slice (narrow is not useless). The eval-overlap check that compared a
> discovered dataset's own train slice to its own test slice was removed outright — both were sliced
> from the front of the same split, so it always "found" overlap exactly equal to `max_test` and
> rejected every candidate, including both canonical xLAM repositories. A source is useless only
> when *nothing* survives per-row filtering.

**Strict only when the vocabulary is authoritative.** A space pinned from the eval set or declared in
a plan is complete, so strict subset rejection is safe. A space merely *inferred* from the rows a run
happens to hold may be missing real classes, and being strict there would discard good sources for a
class the pool had not seen yet — so the inferred case keeps the older any-overlap check (which still
catches B222's raw integer class ids).

**Label definitions.** `label_definitions_for(benchmark)` supplies what each class MEANS, injected into
both the synthesis and verification prompts. Without it the teacher judges the label's wording rather
than the task — it rejected 70% of RouterBench rows because a math problem "is not a local query"
(B267). Tasks whose label already describes the text (CLINC150 intents) need none.

> **Update 2026-08-16/17.** Two of the kinds below are GONE. `resample` is no longer an
> orchestrator-selectable strategy (it added no information), and **synth-fill is removed entirely**
> (it padded to a heuristic target; BC5CDR's best result came from a gold-only curriculum 7,100 rows
> under target). What remains: `acquire`, `synthesize` (surgical + spread), and the universal
> resample-FILL that supplies the gold rows. A 500-row viability floor replaces synth-fill's guarantee.
>
> Also: the teacher's ZERO-SHOT score is not a valid proxy for its synthesis fitness — it scores 0.1131
> zero-shot on BC5CDR NER and **0.7190 with five demonstrations** (B276). And the claim that systematic
> teacher errors are worse than random noise is **withdrawn** (B277); the literature contradicts it.

> **Update 2026-08-17 — the kind NAMES changed, and synthesis was not reaching format-bound tasks.**
> `resample-fill` read as a survival of the removed `resample` strategy when it is the universal gold
> filler, and `fill-synth` was one character from the deleted `synth-fill`, so xlam run 38566712 looked
> like it was running mechanisms that no longer exist. The kinds are renamed below to say what they do.
> Separately, `function_call` and `diff` were missing from `synthesize_examples`' dispatch table, so the
> `synthesize` strategy was a **silent no-op** on every format-bound task and the exact verifiers had
> never executed (B291). Both fixed.

### Rebuild KINDS — what a `data_rebuild` actually did

*(largely historical as of 2026-08-19)* A rebuild used to combine several kinds at once, because the
plan's `strategy` named only the primary lever and every rebuild additionally refilled the remainder
of the curriculum with gold rows from the train pool. `strategy=synthesize` alone therefore never
told you whether rows came from confusion-pair targeting or from balanced spread. Each producer
stamps `_strategy_origin`, and `REBUILD_KIND_NAMES` / `rebuild_kind_name()` in
`agent/nodes/curate.py` map those to one canonical token:

| `_strategy_origin` | Kind | What it did | Still produced? |
|---|---|---|---|
| `resample` | **train-pool-gold** | draw gold rows from the existing real train pool; no new material | **No** — the universal gold fill was removed 2026-08-19 |
| `mine_new_real_source` | **mine-new-real** | real rows from a known source or a discovered one | **Yes** — set by `_tag_mined_rows` under `mine_new_real` |
| `surgical_synthesize` | **surgical-synth** | targeted rows for the top failure categories | **Yes** — set by `_surgical_synthesize`, now the whole synthesis budget rather than 20% of it, and no longer classification-only |
| `synthesize` | **balanced-synth** | teacher-generated rows balanced across the label space | **No** — untargeted balanced synthesis was removed 2026-08-19 |

The mapping is retained so historical artifacts and resumed checkpoints still render, and an
unmapped origin passes through unchanged rather than being hidden as "unknown". `synth_fill` has not
been a named kind since 2026-08-16 — keeping a display name for a deleted mechanism made every log
line that mentioned it look live.

A rebuild now produces rows from exactly one kind, so the report is simpler and the interesting
number is what it **added**:

```
CURRICULUM: 3235 → 4187 row(s) (+952 added this rebuild, 952 novel overall)
```

and a rebuild that added nothing is an **ERROR**, not a line among twenty:

```
✗ ERROR: data_rebuild/mine_new_real added 0 new rows. The curriculum is unchanged at 3235 row(s),
  so training this iteration would repeat the previous one exactly.
```

`last_curation` carries `rows_added`, `novel_rows`, `strategy`, `target_categories`, the
`mining_report`, `source_progress`, `failed_discovery_rounds`, and the per-layer eval-firewall tally.

**Counting synthetic rows.** `n_hard` / `n_hard_generated` count only `_provenance == "synthetic"`
(the plan's `synthesize` strategy). Use **`n_synth_total`** for every teacher-generated row —
`synthetic` + `synthetic_fill` + `synthetic_positive` — with the split in `n_synth_by_provenance`.
The old field made a rebuild that generated 990 rows and kept 93 report 29, and `run_graphics` drew
a synthetic band of ~1% while the rest sat in the grey "unattributed" band. `run_graphics` now plots
`n_synth_total`, falling back to `n_hard_generated` for runs that predate the field.

### Where synthetic data is actually generated

Entered via `_surgical_synthesize` in `agent/nodes/curate.py` — the only remaining entry point since
2026-08-19 — then `data/curriculum.py::synthesize_examples`. (`_synthesize_positive_rows` and
`_synth_fill_to_target` are both gone.) The teacher prompt is built from the task brief and carries
the failure category the batch is aimed at, so it asks for rows exercising what the model is
actually getting wrong rather than for more of the same.

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
> `Evan's Notes/08-15b-label-space-lockdown.md` §0.
>
> Also corrected: **NER synthesis has never produced a row** (B268). NER rows carry no `label`, so
> the bucketing below is always empty and the generator returns immediately. It is now a documented,
> intentional no-op — NER curricula are gold-only.

**Gold-only generation (2026-08-05).** Synthesis produces NEW CORRECT in-distribution examples, never
a wrong answer as a positive SFT target. Which of two shapes it produces is derived from
`TaskSpec.closed_label_space`:

- **closed label space** (`clinc150`, `routerbench`, `proactive_listening`) —
  `_synthesize_new_gold`: anchors are drawn **round-robin across the label space** (so rare classes
  get equal attention), and the model is asked for one new utterance of the *same* class as its
  anchor. The row **inherits** the anchor's label, so an out-of-vocabulary or mismatched label is
  impossible by construction *for tasks whose label the generator controls* (see the correction
  above).
- **open-ended target** (everything else) — `_synthesize_new_correct`: new correct instances in the
  anchor's JSON schema, kept only if the task's `synth_verifier` accepts them where one exists, then
  checked by the teacher. Wrong-answer SFT harms these tasks most.

**Chain-of-thought annotation is per task too** (`TaskSpec.cot_annotation`), not per family. Only
`gsm8k` declares it in the current suite: the gold answer there is the end of a multi-step
derivation the model has to reason its way to. `dialogsum` is the case that forced the distinction —
its summary is a compression of text already sitting in the prompt, so a reasoning chain adds
nothing the model cannot read off its own input, while costing one teacher call for EVERY row.

**Teacher label verification** (`verify_generated_labels`, on by default, `SLM_VERIFY_SYNTH=0`
to disable). Every generated classification row is shown back to the teacher model, which
answers `{"valid": bool, "reason": str}`. Rejected rows are dropped and the teacher's reason is
logged. Verification failures (unparseable reply, endpoint error) KEEP the row — the verifier
can never empty a dataset.

### The "honest attribution" invariant

A key design rule: a plan declaring an intervention that produces **zero** rows is a "strategy
attribution lie." The driver's synthesis preflight still blocks at startup when
`SLM_REQUIRE_SYNTH != "0"` (default), so a run cannot silently begin gold-only against a dead
endpoint.

**Update 2026-08-19 — the invariant is now enforced by reporting rather than by padding.** There is
no target to fill, so there is nothing to rewrite a no-op strategy *to*: `base_fill` and the
`allocation_fallbacks` ledger are gone. Instead, a rebuild that adds zero rows is logged as an
**ERROR** naming the strategy and the unchanged curriculum size, because an intervention was chosen,
a plan was built, and the mechanism it named could not do the thing it exists to do — so training
this iteration would repeat the previous one exactly. The old entry also used to *guess* the cause
("endpoint unavailable or cheap mode"), which was false every time on xlam run 38566712 where the
real cause was the B291 dispatch gap and the endpoint was healthy.

### Real-data acquisition ladder

**Update 2026-08-19 — mid-run mining is a two-rung ladder, and rung 1 is new.**
`agent/nodes/curate.py::_mine_new_real` walks:

1. **Re-read the task's own `mining_sources`** for a larger head slice than we have consumed. Every
   loader takes a head slice, so a bigger ask returns a superset and the tail is novel by
   construction — free, with no provider call, no LLM column mapping and no schema risk.
   `state["source_progress"]` records what we consumed per source, and a source is marked exhausted
   only when the loader returns *fewer* rows than asked for.
2. **Web research**, only once every known source is exhausted: Exa finds candidate HuggingFace
   datasets, the orchestrator judges fit and maps columns, and rows are taken per row.

This did not exist before 2026-08-19. `acquire` went straight to paid discovery, which then found
mirrors of the very corpus sitting in the local cache and rejected them, while ~57,000 unused xLAM
rows stayed unreachable (B297). Two other things went with it: the `candidates[:6]` shortlist (both
canonical repositories were returned at positions 7 and 8, B294) and the `str(e)[:80]` error
truncation — every candidate is probed now and errors are logged whole.

*(pre-redesign, retained for history)* Cold-start acquisition on the **autonomous** path still walks
the longer ladder in `data/loaders/web_acquire.py`: local bundle → deterministic benchmark loaders →
agentic HF discovery via Exa + orchestrator column-mapping → bounded Exa web scrapes →
**last-resort gold synthesis via the Claude orchestrator** (distinct from the local curriculum
synthesis above). A curated run loads its data from the task's own loader and never enters it.

> **Update 2026-08-12.** Rung 2 is more fragile than it reads. Under `datasets>=4` any HF repo
> with a loading script at its root is permanently unloadable (`RuntimeError: Dataset scripts are
> no longer supported`), which kills `tner/bc5cdr`, `AmazonScience/massive` and
> `iohadrubin/smcalflow` outright, and repos whose files don't match a split-name pattern raise
> `DataFilesNotFoundError`. Every Stage-0 entry needs a script-free candidate — the raw-JSON
> fallback at `web_acquire.py:153-161` is the pattern to copy. Prefer rung 1: a frozen bundle
> under `data/local/` never breaks and is checksummed. See B253–B255 and
> `docs/Evan's Notes/08-12-new-tasks-loaders.md`.

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
   both smallest & largest base models pass; hard = neither (`test_agent.label_difficulty`).
   Computed once at cold start and **advisory only**: nothing samples training rows by difficulty,
   and a row the small model gets right and the large one wrong still falls into `hard` (B298).
2. **A confusion table** — the top 8 entries of a `Counter`, built from the task's **own** failure
   taxonomy (`TaskSpec.failure_category`) rather than per task type. For a closed-label task the
   entries are real `(gold, predicted)` class confusions; for everything else they name the error
   KIND — `wrong_arguments`, `entities_hallucinated`, `no_numeric_answer` — and the second slot
   reads `incorrect`. A task declaring `failure_category=None` reports one honest aggregate instead
   of a fabricated taxonomy.
3. **A rule-based diagnosis → intervention** (`test_agent.diagnose`): easy bucket < 0.6 →
   `data_rebuild`; easy solid but medium/hard weak → `hyperparameter`; below goal with no single
   failing bucket → `data_rebuild`.

> **Update 2026-08-19 (B296).** Item 2 used to be keyed by `task_type`, and every task that was
> neither `classification` nor `NER` collapsed to the single literal `gold_verifier → incorrect` —
> a constant whose count is the failure count the orchestrator already had. Across a dozen
> iterations of the xlam run the orchestrator wrote hypotheses like *"the dominant confusion
> gold_verifier->incorrect (147) essentially unchanged since iter2"*: paragraphs of reasoning about
> a constant, used as evidence. Because surgical synthesis budgets off these categories, the fix
> also gave synthesis something real to aim at.

### What the failure analysis is used for

- **Iteration decisions** (`agent/nodes/iterate.py`): per-difficulty accuracy + diagnosis +
  failure categories feed the orchestrator prompt, driving the next intervention and even letting
  the LLM lower `stop_threshold` if a failure reflects genuine capacity limits.
- **Reporting** (`data/curation_log.py`): top confusion pairs are written to the curation log under
  an "Aggregate confusion counts" heading.
- **Targeted curriculum synthesis** (`agent/nodes/curate.py::_surgical_synthesize`): each category's
  share of the rebuild is proportional to its failure count, bounded to [25, 600] rows, over at most
  `SLM_SURGICAL_MAX_CATEGORIES` (5) categories. A category already targeted whose count did not fall
  is logged `EXHAUSTED` and skipped, so the run stops paying for something that is not responding
  (B224).

---

## 3. Every token cap in the repo, and why

No token caps live in YAML/JSON — they're Python constants, function defaults,
env-var-with-default, or the `_EVAL_OUTPUT_TOKEN_SETTINGS` table.

### Response/generation caps that affect the shipped SLM

**Eval output reserve — the central task-aware cap.** Since 2026-08-19 this is
`TaskSpec.max_new_tokens`, read by `eval/harness.py::eval_output_token_reserve(task)`:

| Reserve | Tasks |
|---|---|
| 50 | `clinc150`, `routerbench`, `proactive_listening` |
| 256 | `xlam_bfcl`, `calendar_json` |
| 512 | `gsm8k`, `dialogsum`, `ner_bc5cdr` |

**Why these values:** they match the shape of a correct answer. A classification task emits a
single label → 50 tokens is plenty and prevents rambling into wrong territory. A function call is a
short JSON object → 256. Entity lists, reasoning chains and summaries need room → 512 (raised from
256 in B147). `SLM_EVAL_MAX_NEW_TOKENS` overrides every task and is re-validated against that task's
own `max_seq_length`, so an override that leaves no prompt budget raises.

> **Update 2026-08-19.** This was `_EVAL_OUTPUT_TOKEN_SETTINGS`, a dict keyed by `task_type` with
> five entries and a `.get(..., 4096)` on the context side — so an unrecognised type silently
> received a generous default instead of an error. The `code_generation` entry (1024, for APPS) is
> gone with the task type itself. The per-task pair is now validated **at import**:
> `TaskSpec.__post_init__` refuses a spec whose reserve leaves no prompt budget inside its context
> window, which is a stronger guarantee than checking at eval time.

**Max sequence length — the per-task context cap** (`training.slm_helpers.task_max_seq_length` →
`TaskSpec.max_seq_length`, overridable for every task with `SLM_MAX_SEQ_LENGTH`): **1024** for
`clinc150` and `routerbench`, **2048** for the other six. Used for model load, GGUF `n_ctx`, and
training `SFTConfig.max_length`. **Why:** input budget is always `max_seq_length - max_new_tokens`,
and the code **never truncates** — over-length rows raise rather than silently dropping content.
Notably `generation_config.max_length = None` disables Qwen's built-in 40960 cap so `max_new_tokens`
is the *sole* length knob (fix B128).

A single global 4096 was raised from 512 (B204/B188) to fit APPS code prompts, then split per
task in 2026-08 — context is *allocated* whatever the rows contain, so 4096 against a measured
max of 1206 tokens left ~70% of the KV cache and position buffers untouched, memory a larger
eval batch can use instead. With `code_generation` deleted on 2026-08-18 nothing in the suite needs
the 4096 window, and the unrecognised-task fallback is gone: `get_task` raises on a task the
registry does not hold. This governs the SMALL MODEL only; the orchestrator's Claude context is an
unrelated budget.

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

1. **Right-size output to the task** — a label gets 50, a function call 256, a reasoning chain or
   entity list 512, a judge score 8. Tight caps prevent runaway generation, cut cost/latency, and
   stop models wandering into wrong answers. Since 2026-08-19 "the task" means the task, not a
   channel it shares with another benchmark.
2. **Never silently truncate** — input budget = context − output reserve, and over-length
   rows raise rather than lose data. Caps are enforced as hard invariants, not lossy
   conveniences.
3. **Reproducibility** — defaults are duplicated into checkpoint snapshots and SLURM
   exports so a resumed or re-run job reproduces identical caps.
