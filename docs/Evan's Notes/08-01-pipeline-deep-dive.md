# Pipeline Deep-Dive: Firewall, Filtering, data_rebuild, Benchmarks, Difficulty, iterate + Escalation Policy

**Date:** 2026-08-01
**Branch:** `data-curation-redesign`
**Purpose:** One reference answering six questions about how the loop actually behaves *in current
code* (not the 2026-07-29 [PIPELINE.md](../PIPELINE.md), which is stale in several places noted
inline), plus the recorded escalation policy: **max 30 evals per tier, escalate on <2% gain over
15 evals.**
**Ground truth:** every claim is cited `path:line`. Where a shipped doc disagreed with the code,
the code won and the drift is flagged as ⚠ STALE DOC.

---

## Table of contents

1. [The data firewall — keeping the orchestrator blind to eval rows](#1-the-data-firewall)
2. [The filtering / quality-control process](#2-the-filtering--quality-control-process)
3. [How `data_rebuild` works, and whether reshuffle is valid](#3-how-data_rebuild-works-and-whether-reshuffle-is-valid)
4. [What each of the six benchmarks measures](#4-what-each-of-the-six-benchmarks-measures)
5. [easy/medium/hard difficulty in the training pipeline](#5-easymediumhard-difficulty-in-the-training-pipeline)
6. [What the orchestrator sees at `iterate`, and what it emits](#6-what-the-orchestrator-sees-at-iterate-and-what-it-emits)
7. [Escalation policy: 30 evals / 15-eval window / 2% — decision + how to wire it](#7-escalation-policy)
8. [Appendix: stale-doc corrections found while writing this](#8-appendix-stale-doc-corrections)

---

## 1. The data firewall

**Short answer to all three of your questions:** the orchestrator never receives a single raw
eval row — only aggregate per-difficulty accuracy and aggregate confusion *counts*. The exact
eval rows cannot be injected into training because normalized train/eval overlap is (a) a hard
raise at build time and (b) dropped at four points during curation. And the orchestrator's
reasoning cannot be "driven by the direct failure cases" because it is structurally denied the
failure cases — it sees only bucketed statistics.

### 1.1 Four independent firewall layers (all on normalized text)

Normalization is `data/loaders/dataset_integrity.py::normalize_text`; every layer compares on it.

| # | Layer | Where | Effect |
|---|---|---|---|
| 1 | **Source split separation** | `eval_setup` | Raises on ANY normalized train/test overlap after acquisition |
| 2 | **Candidate filtering** | `curate` (train, mined, synth, final) | Silently *drops* overlapping rows and logs the count |
| 3 | **Decision-prompt rejection** | `iterate` | Rejects any held-out text (≥12 normalized chars) appearing anywhere in a decision |
| 4 | **Reask sanitization** | `iterate` | Redacts eval text from validator error strings before replaying them to the LLM |

**Layer 1 — build-time raise.** Live path: [eval_setup.py:319-332](../../agent/nodes/cold_start/eval_setup.py#L319-L332) builds the set of normalized eval texts and raises `ValueError` if any train row matches; success logs `official train/test separation: normalized overlap=0`. Shared-bundle path does the same via `normalized_text_overlap` at [eval_setup.py:200-204](../../agent/nodes/cold_start/eval_setup.py#L200-L204).

**Layer 2 — curate drops overlap at four touch points.** Helper `_exclude_eval_rows` at [curate.py:73-87](../../agent/nodes/curate.py#L73-L87). Applied to:
- train anchors — [curate.py:509-512](../../agent/nodes/curate.py#L509-L512)
- mined (acquire) rows — [curate.py:593](../../agent/nodes/curate.py#L593), plus inside `_merge_persistent_train_rows` at :295
- synthesized rows — [curate.py:365](../../agent/nodes/curate.py#L365) (and synth-fill at :427)
- the FINAL dataset, after quality controls — [curate.py](../../agent/nodes/curate.py) (`_exclude_eval_rows` immediately before the version stamp/write)

**Layer 3 — the orchestrator prompt cannot contain eval text.** `_reject_eval_text_strings` ([iterate.py:57-97](../../agent/nodes/iterate.py#L57-L97)) walks every string in the decision recursively and raises if a normalized eval string (≥12 chars) is an exact or substring match. Applied in `_validate_decision_json` at [iterate.py:226](../../agent/nodes/iterate.py#L226) and again specifically to `hypothesis` at [iterate.py:236-246](../../agent/nodes/iterate.py#L236-L246). The `data_rebuild` plan validator also refuses raw eval text in `pattern_hint`.

**Layer 4 — reask can't leak either.** `_sanitize_reask_error` ([iterate.py:127-145](../../agent/nodes/iterate.py#L127-L145)) regex-redacts every eval text out of the validator error before it is shown back to the model on the JSON-only reask.

### 1.2 The orchestrator is fed aggregates only

The **test-data agent** (`agent/nodes/test_agent.py`) owns the held-out set and emits only
aggregates — never rows. `build_test_report` ([test_agent.py:173-224](../../agent/nodes/test_agent.py#L173-L224)) returns `overall`, `by_difficulty` (accuracy + n per bucket), `confusion_pairs` (top 8), `diagnosis`, `suggested_intervention`, `band`. For open-ended tasks the "confusion" is reduced to a *verifier category* ([test_agent.py:200-208](../../agent/nodes/test_agent.py#L200-L208)) precisely because the raw target *is* the answer.

What actually goes into the decision prompt is assembled at [iterate.py:786-833](../../agent/nodes/iterate.py#L786-L833): per-difficulty accuracy strings, the diagnosis, the suggested intervention, and `gold=… predicted=… count=…` confusion lines. No row text. The system prompt states the rule explicitly at [iterate.py:478-481](../../agent/nodes/iterate.py#L478-L481): *"never inspect, request, quote, or copy raw eval text. Use only aggregate difficulty scores and aggregate confusion counts."*

Synthesis and gold anchors are drawn only from the decontaminated `train_examples` pool, so even
generated data cannot smuggle eval rows back in.

### 1.3 Known limitation (worth writing down)

The firewall matches on `normalize_text` — i.e. exact / near-exact string identity after
normalization. It catches verbatim leakage and duplicates; it does **not** catch a paraphrase of
an eval item. That is the honest boundary of the current guarantee.

---

## 2. The filtering / quality-control process

### 2.1 Where filtering sits in `curate_node`

`curate_node` is a single linear path ([curate.py:492-819](../../agent/nodes/curate.py#L492-L819)). Ordered steps:

1. Model/task setup; **early-exit gate** — if `last_intervention != "data_rebuild"` it SKIPs and returns unchanged ([curate.py:499-502](../../agent/nodes/curate.py#L499-L502)).
2. Eval-set precondition (raises if `eval_set is None`) — :503-507.
3. **Firewall layer-1 on train anchors** — :509-517.
4. Resolve plan (`fallback_data_rebuild_plan` or `normalize_data_rebuild_plan`) — :518-546.
5. Strategy + entropy seed — :548-553.
6. Execute the material strategy (`acquire` / `synthesize`) — :573-643.
7. Allocation: reserve strategy rows, then **resample-fill** the remainder — :645-681.
8. **Synth-fill to `target_rows`** (`_synth_fill_to_target`) — :686-701.
9. CoT annotation (math/code/generation; skipped under `SLM_CHEAP=1`).
10. **`apply_quality_controls`**.
11. **Firewall layer-2 (final)** — `_exclude_eval_rows` on the finished dataset.
12. Version stamp + atomic write; record `last_curation`.

Note the order: **quality controls run BEFORE the final firewall.** There is **no upper truncation
step** — `target_rows` is a floor for synth-fill, not a cap (change 2026-08-02: see §2.5). Extra
rows above `target_rows` (e.g. an `acquire`/`synthesize` overshoot) are kept; the pipeline only
guards against too *few* rows, never too many.

### 2.2 The four quality controls — `apply_quality_controls`

`data/curriculum.py::apply_quality_controls` ([:188-310](../../data/curriculum.py#L188-L310)), routed by task family. Empty input is a no-op (:206-207); an unrecognized `task_type` is returned unchanged (:272-273).

| Filter | Task family | What it does | Constant |
|---|---|---|---|
| **A. Label balancing** | classification | No label may exceed 3× the rarest label's count | `max_allowed = 3 * min_count` ([:216-217](../../data/curriculum.py#L216-L217)) |
| **B. Length-outlier removal** | all families | Drop rows whose text length > 3× the median length | `max_ratio=3.0`, `cutoff = median*3` ([:277,284-285](../../data/curriculum.py#L277-L285)) |
| **C. Surface-form dedup** | classification, math, code (**not** generation) | Drop near-duplicate rows by Jaccard word-set similarity ≥ 0.9 vs the last 50 kept rows | `threshold=0.9`, window `[-50:]` ([:290,301-304](../../data/curriculum.py#L289-L310)) |
| **D. Entity diversification** | NER | Cap any single entity surface value at ≤3 occurrences | `> 3` over-rep, drop at `running>=3` ([:239,247](../../data/curriculum.py#L239-L247)) |

Uncommon terms:
- **Macro / label balancing** — balancing so each *class* contributes comparably to the loss regardless of raw frequency, so a dominant class can't swamp a rare one. Here it's a cap at 3× the rarest class.
- **Length outlier** — a row far longer than typical (>3× median chars) — usually a malformed or pathological example; removed so it doesn't distort training or blow the sequence budget.
- **Jaccard similarity** — |A∩B| / |A∪B| over the two rows' word sets; 1.0 = identical word sets. ≥0.9 flags near-duplicates. The `[-50:]` window makes dedup *local/approximate* (only the 50 most recent kept rows are checked), not global.
- **generation is deliberately NOT deduped** (comment [curriculum.py:266-267](../../data/curriculum.py#L266-L267)) — legitimate summaries/answers share a lot of vocabulary and would be over-pruned.

### 2.3 Is filtering run on EVERY curate, including the first curriculum? — **Yes.**

`apply_quality_controls` and all three firewall touch-points are on the one linear path after the
early-exit gate; there is no "initial build" bypass. The first curate still runs because the gate
defaults to `"data_rebuild"` (`state.get("last_intervention", "data_rebuild")`,
[curate.py:499](../../agent/nodes/curate.py#L499)) **and** cold-start model-selection explicitly
sets `last_intervention="data_rebuild"` ([largest_first.py:98](../../agent/nodes/cold_start/model_selection/largest_first.py#L98)). So the initial curriculum gets the exact same filters + firewall as every rebuild.

### 2.4 What happens when data is UNDER the goal

The goal is `target_rows` (normalized from `curriculum_size_target`; floor is now **5000**, ceiling
**25000** — [config.py:141-143](../../config/config.py#L141-L143)). Under-goal handling:

- Resample-fill first tops up from the existing pool ([curate.py:668-679](../../agent/nodes/curate.py#L668-L679)).
- Then `_synth_fill_to_target` ([curate.py:371-437](../../agent/nodes/curate.py#L371-L437)) synthesizes the remaining `deficit = target_rows - len(dataset)` via the local Qwen synth endpoint.
- If synthesis is unavailable or `SLM_CHEAP=1`, it does **not** crash — it leaves the dataset short and records an honest fallback in `last_curation["allocation_fallbacks"]`:
  - `synth_unavailable_degrade` (endpoint down / cheap mode) with `unfilled_rows: deficit` — :393-407
  - `synth_fill_empty` (endpoint returned nothing usable) — :428-435
  - `rewrite_noop_strategy` → `base_fill` when `acquire`/`synthesize` yielded nothing and resample covered it — :621-643

`base_fill` is not a function; it's the label meaning "resample/existing-pool fill covered what the
primary strategy couldn't." **Under `SLM_CHEAP=1`** the pipeline skips positive synthesis, the
fill generator, and CoT annotation — an under-goal run stays under goal on real+resampled data and
logs the degrade honestly rather than erroring.

### 2.5 `target_rows` is a floor, not a cap (2026-08-02 change)

Curate used to `dataset = dataset[:target_rows]` right before the final firewall, hard-capping the
curriculum at `target_rows`. That truncation was **removed**: `target_rows` is now only the *floor*
that drives `_synth_fill_to_target` (top up when short). The allocation loop still stops adding
resample/strategy rows once it reaches `target_rows`, so in the common case the dataset lands at
exactly `target_rows`; but if a strategy legitimately overshoots (e.g. `acquire` returns more novel
real rows than requested, or the 2-for-1 hard-negative synthesis path expands rows), those extra
rows are **kept** rather than sliced off. The design intent: never train on *too few* rows; extra
rows are strictly better than discarding real signal to hit an exact count.

---

## 3. How `data_rebuild` works, and whether reshuffle is valid

### 3.1 The three strategies

Defined in `agent/data_rebuild.py`, executed inside `curate_node`. Chosen singly, with **no**
task-type or score gating (2026-07-31 redesign).

| Strategy | Executor | Behavior |
|---|---|---|
| `resample` | `_balanced_sample` ([curate.py:159-177](../../agent/nodes/curate.py#L159-L177)) | Re-draw / reshuffle rows from the existing train pool. Also the **universal filler** for all three strategies. |
| `acquire` | `mine_additional_real_rows` ([web_acquire.py:571](../../data/loaders/web_acquire.py#L571)) | Add NEW real rows: local → deterministic benchmark → paid Exa/HF. |
| `synthesize` | `_synthesize_positive_rows` ([curate.py:312-368](../../agent/nodes/curate.py#L312-L368)) | Task-adaptive synthetic rows (hard negatives for cls/NER; new correct examples for math/code/gen). |

**Key facts about `resample`:**
- Draws from `tagged_train` = the decontaminated `train_examples` pool ([curate.py:614-618](../../agent/nodes/curate.py#L614-L618)) — **not** a stored curriculum; the previous dataset artifact is read only for novelty accounting.
- **WITHOUT replacement** (`_round_robin_sample` pops per-label; the non-cls path slices a shuffle), and `allocate` additionally dedups by normalized text. So it can never emit duplicates and never more than the number of unique pool rows.
- **Entropy-seeded / non-deterministic**: every sampler call takes a fresh `_entropy_seed() = int.from_bytes(os.urandom(8), "big")` ([curate.py:34-41](../../agent/nodes/curate.py#L34-L41)). No reproducible per-plan seed.

### 3.2 Your question: reshuffle when the whole pool is already in the curriculum — now GATED (2026-08-02)

This was the important case, and it is now handled directly: **`resample` is taken off the menu
whenever the entire training pool is already in the curriculum**, because a reshuffle there produces
the identical set and adds no novelty.

How the gate works (three layers, choice-time not just execution-time):

1. **The signal.** `resample_pool_exhausted(pool_texts, curriculum_texts)`
   ([data_rebuild.py](../../agent/data_rebuild.py)) returns True when the normalized train pool is a
   subset of the normalized current curriculum. An empty pool returns False (nothing to draw yet, so
   resample stays nominally allowed).
2. **curate — the precise guarantee.** `curate_node` computes `resample_available` from the
   eval-decontaminated pool vs the previous artifact **before** resolving the plan, and passes it
   into both `fallback_data_rebuild_plan` and `normalize_data_rebuild_plan`. When
   `resample_available` is False, `normalize_data_rebuild_plan` **redirects `strategy="resample"` →
   `"synthesize"`**, and the fallback planner drops `resample` from its weighted menu entirely.
3. **iterate — the advisory copy.** `iterate_node` sets `state["resample_available"] =
   resample_available_for_state(state)` and threads it into the validator, the fallback plan call,
   and the orchestrator prompt. The system prompt tells the orchestrator resample only reshuffles
   the existing pool, and the per-turn user prompt adds an explicit *"RESAMPLE IS UNAVAILABLE THIS
   TURN"* note when the pool is exhausted, so the LLM proactively picks `acquire`/`synthesize`.

`resample` remains fully valid and useful when the pool is **larger** than the curriculum (fresh
draw / class rebalance) — the gate only fires when pool ⊆ curriculum. When the pool is tapped out,
the levers that actually add information are **`acquire`** (new real rows, bounded by the paid
ledger) and **`synthesize`** (new synthetic rows), and the redirect steers there automatically.
Escalation after the stall window (see §7) is still the backstop for a run that stops making
progress even with novel data.

### 3.3 Plan schema, ledger caps, fallback

- Plan ranges (`normalize_data_rebuild_plan`): `target_rows` [16, ceiling] step 8; `resample_fraction` [0.10,1.0] step 0.05; `new_real_rows` [0,500] step 5; `synth_rows` [100,500] step 5; `max_acquire_rounds` [0,3]; `confusion_pairs` ≤8; `pattern_hint` (≤240 chars, raw eval text rejected).
- Paid acquisition ledger: `MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN = 3`, `MAX_PAID_ACQUIRE_ROUNDS_PER_RUN = 9` ([data_rebuild.py:28-29](../../agent/data_rebuild.py#L28-L29)). Reservations are locked, append-only, never refunded — a crash loop can't re-spend.
- `acquire` source order: local dataset → deterministic benchmark (both set `paid_limit=0` on success) → paid Exa/HF discovery loop.
- Fallback plan (`_fallback_strategy_from_signal`, [data_rebuild.py](../../agent/data_rebuild.py)) is **signal-weighted random**: base weights 1/1/1; easy-bucket <0.6 → +2 acquire; medium/hard <0.6 → +2 synthesize; confusion pairs → +1 synthesize; then `random.choices`. When `resample_available=False` the `resample` weight is dropped before the draw, so the fallback can only pick `acquire`/`synthesize`.

---

## 4. What each of the six benchmarks measures

Two per category, wired via `SLM_BENCHMARK_TASK` → loader registry ([eval_setup.py:27-44](../../agent/nodes/cold_start/eval_setup.py#L27-L44)). "Right answer" = the gold field each loader emits.

| # | Category | Benchmark | task_type / metric | Row schema | What a right answer looks like |
|---|---|---|---|---|---|
| 1 | In-distribution | **CLINC150** | classification / macro-F1 | `{text, label}` | The correct intent out of 150 intents **+ out-of-scope**. Label is the intent string; OOS is the built-in "none of these." ([clinc150.py:39](../../data/loaders/clinc150.py#L39)) |
| 2 | In-distribution | **DialogSum + SAMSum** | generation / LLM-judge [0,1] | `{text=dialogue, answer=summary}` | A faithful summary of the dialogue; scored 0–1 by the local Qwen-3.6 judge against the reference summary. ([dialogsum_samsum.py:24](../../data/loaders/dialogsum_samsum.py#L24)) |
| 3 | Format-bound | **xLAM / BFCL** | function_call / AST arg-match | `{text, answer=JSON [{name,arguments}]}` | The correct function name(s) with required args (values type-coerced). Gold is a JSON call list; scored by BFCL-style AST comparison. ([xlam_bfcl.py:68-72](../../data/loaders/xlam_bfcl.py#L68-L72)) |
| 4 | Format-bound | **CoEdIT** | diff / `git apply --check` + applied-match | `{text=instruction+src, answer=unified diff}` | A unified diff (computed via `difflib` from src→tgt) that cleanly `git apply`s and produces the target text. ([coedit.py:69-74](../../data/loaders/coedit.py#L69-L74)) |
| 5 | Data-scarce (small, ~300M) | **RouterBench** | classification / binary-F1 | `{text, label∈{local,route}}` | The correct routing call: `local` = the on-device model answers correctly (keep on device); `route` = it fails, escalate to cloud. ([routerbench.py:17-18,65](../../data/loaders/routerbench.py#L17-L18)) |
| 6 | Data-scarce (large, 3B+) | **MedQA-USMLE (4-opt)** | classification / accuracy | `{text=question+options, label∈{A,B,C,D}}` | The correct option letter for a USMLE-style medical MCQ. MCQ-only by design (no free-text medical advice). ([medqa.py:66](../../data/loaders/medqa.py#L66)) |

Category intent: **1** = find the smallest model matching base quality (near-saturated controls);
**2** = format compliance with executable, judge-free verifiers (the format-vs-bitwidth result);
**3** = decouple data cost from model size (RouterBench trains on synthetic, evaluates on 405k real
outcomes offline; MedQA is knowledge-bound and anchors the top of the size range). Full rationale:
[07-31-benchmark-selection.md](07-31-benchmark-selection.md).

---

## 5. easy/medium/hard difficulty in the training pipeline

**Headline (verified against current code): difficulty is an EVAL-side construct. It does not
directly shape the training curriculum today.** It influences training only *indirectly*, as a
decision signal that steers strategy selection and the orchestrator's prompt.

### 5.1 pos / neg / boundary — REMOVED (2026-08-02)

The eval set used to carry a three-way `pos`/`neg`/`boundary` split, but it had **no functional
effect**: every consumer used the `.all` union, `curate.py`/`curriculum.py` never read the slices,
and for the NER/generation families the split was a meaningless random partition. The three slice
scores were only ever printed as one cosmetic line in the curation log — nothing branched on them.
The slices were deleted; `EvalSet` is now a single flat `.all` sample. `build_eval_set` keeps
**label-coverage stratification** for multi-class classification (round-robin across labels up to
the target) and is a plain shuffled top-N sample for every other task type. The real difficulty
signal is easy/medium/hard below, which is independent and untouched.

### 5.2 easy / medium / hard — eval-only, drives diagnosis not sampling

`label_difficulty` ([test_agent.py:47-88](../../agent/nodes/test_agent.py#L47-L88)) labels each
held-out row by the base-model capability gradient: **easy** = smallest & largest base both correct;
**medium** = only the large one; **hard** = neither ([test_agent.py:68-77](../../agent/nodes/test_agent.py#L68-L77)). Fallback is a length-tercile heuristic under `SLM_DIFFICULTY=heuristic`, on any error, or on a degenerate all-empty split ([test_agent.py:103-113](../../agent/nodes/test_agent.py#L103-L113)).

Those labels feed `build_test_report` → `diagnose` ([test_agent.py:130-170](../../agent/nodes/test_agent.py#L130-L170)):
- `overall ≥ threshold` → `none` (converged)
- **easy < 0.6 → `data_rebuild`** (missing simple cases = data quality / label format / prompt)
- **medium/hard < 0.6 → `hyperparameter`** (optimization/capacity gap)
- below goal, no single failing bucket → `data_rebuild`

That suggestion goes into the orchestrator prompt and is the failure-fallback intervention.

### 5.3 How difficulty_buckets / confusion_pairs / pattern_hint touch training — barely

- They are validated + stored on the plan (`data_rebuild.py`), and in the **fallback** planner the
  difficulty weights are computed inversely to per-bucket accuracy and confusion biases the strategy
  choice ([data_rebuild.py:356-462](../../agent/data_rebuild.py#L356-L462)).
- **But `curate_node` reads none of `difficulty_buckets`, `confusion_pairs`, or `pattern_hint`.** It
  reads only `strategy`, `target_rows`, `synth_rows`, `new_real_rows`, `max_acquire_rounds`,
  `resample_fraction`. And `pattern_hint` never reaches synthesis — `synthesize_examples`
  ([curriculum.py:619-655](../../data/curriculum.py#L619-L655)) has no `pattern_hint` parameter, so
  the hint is dropped before generation.
- `difficulty_weighted_sampling` **does not exist** (removed in the redesign). The leftover
  `_difficulty_sample` / `_with_train_difficulty` / `_DIFFICULTY_BUCKETS` helpers in curate were
  dead code with zero call sites and have been **deleted** (2026-08-02). The only difficulty
  bucketing that remains is the eval-side `_DIFFICULTY_BUCKETS` in `data_rebuild.py`, used to
  validate the plan's `difficulty_buckets` weights.

⚠ STALE DOC: [PIPELINE.md §12.2 #4](../PIPELINE.md) says "difficulty_weighted_sampling is
implemented and confusion_pairs reach the pattern_hint." Against current code: the sampler is gone,
and confusion→pattern_hint only happens if the LLM writes it into prose (and even then it isn't
consumed). The "explicit class weights and confusion-pair oversampling are NOT implemented" half is
correct. **This is a real gap:** difficulty/confusion currently steer *which strategy* runs, not
*which rows* get oversampled. Turning them into actual sampling weights / confusion oversampling is
the obvious next lever.

---

## 6. What the orchestrator sees at `iterate`, and what it emits

### 6.1 Context assembled for the decision (`_llm_iterate`, [iterate.py:760-920](../../agent/nodes/iterate.py#L760-L920))

**System prompt** (`_ITERATE_SYSTEM`, [iterate.py:473-625](../../agent/nodes/iterate.py#L473-L625)): the firewall rule; "choose exactly ONE intervention"; worked JSON examples; per-field guidance for the data plan; score-band guidance; threshold-adjustment guidance.

**User content** ([iterate.py:876-920](../../agent/nodes/iterate.py#L876-L920)):
- **Trajectory** — the `data-curation.md` rows (one per past iteration: dataset version, intervention, score), compacted if long.
- **Current iteration summary** — task type; model variant selector; iteration #; current f(π); best f(π); full score history; current dataset identity `vN:path`; selected variant on-disk weight size (MB) + note that peak RAM is unmeasured; device memory budget; **recent chronological gain over the last window** + the escalation delta; stop threshold + immutable floor; prior hypothesis; **remaining turn budget**; **remaining paid acquisition rounds**.
- **Test-data agent report** — per-difficulty accuracy (easy/medium/hard + n), diagnosis, suggested intervention, aggregate confusion counts.
- **Tried (dataset, H) identities** — including pruned/rolled-back ones (training is deterministic, so exact repeats are forbidden).
- **Data-rebuild plan notes** — a reminder that plans are NOT deduped (repeat/vary freely).
- **Source novelty + prior plan yield** — `requested / novel_rows / novel_fraction`, and prior plan `status / novel_rows / final_rows`.

It never sees raw eval rows (§1).

### 6.2 Deterministic routing *before* any LLM call (`iterate_node`, [iterate.py:1108-1321](../../agent/nodes/iterate.py#L1108-L1321))

Evaluated in order, and most exits spend **no** API call:
0. no scores → `train`
1. turn budget hit (`(iteration+1)*2 >= turn_budget`) → `terminate`
2. wall-clock exceeded → `terminate`
3. score ≥ threshold → `_route_score_at_threshold` (largest_first probe / hw-gate / downward_probe / terminate)
4. score < threshold AND (stagnant OR stalled) → `escalate` (or terminate for a largest_first probe) — **rule-based, no API call**
5. otherwise → one `_llm_iterate` call (+ ≤1 JSON-only reask)

### 6.3 Output it generates

One validated JSON decision (`_validate_decision_json`, [iterate.py:195-421](../../agent/nodes/iterate.py#L195-L421)):
```json
{
  "intervention": "data_rebuild" | "hyperparameter",
  "hypothesis": "<required, causal, names the failure evidence>",
  "data_rebuild": { strategy, target_rows, resample_fraction, new_real_rows,
                    synth_rows, max_acquire_rounds, difficulty_buckets,
                    confusion_pairs, pattern_hint },   // iff data_rebuild
  "hyperparams":  { lora_rank, alpha_ratio, weight_decay, learning_rate, nr_epochs }, // iff hyperparameter
  "threshold_adjustment": { "new_threshold": <num|null>, "reason": "<str>" }
}
```
The two intervention payloads are a discriminated union — a stray `hyperparams` on a `data_rebuild`
is **stripped** (not rejected) and logged ([iterate.py:248-280](../../agent/nodes/iterate.py#L248-L280)). `threshold_adjustment` can only ever *lower* the threshold, clamped to `initial_stop_threshold`. On LLM failure it falls back to the test-agent's suggestion, then to score bands.

---

## 7. Escalation policy

> **Decision (2026-08-01): the pipeline should run at most 30 evals per model tier, and should
> escalate to the next tier if accuracy does not improve by more than 2% over 15 evaluations.**

### 7.1 What the code does today

Per-tier state resets on every `escalate` (`scores=[]`, `iteration=0`, dag reset), so **`len(state["scores"])` already equals "evals in the current tier."** The current backstops (all in
`iterate.py`, all env-overridable):

| Knob | Value | Meaning |
|---|---|---|
| `STAGNATION_WINDOW` | **20** ([iterate.py:645](../../agent/nodes/iterate.py#L645)) | evals examined by the stagnation test |
| `STAGNATION_MIN_DELTA` | **0.02** ([iterate.py:646](../../agent/nodes/iterate.py#L646)) | min gain over the window that counts as progress |
| `MAX_STALL_EVALS` | **20** ([iterate.py:687](../../agent/nodes/iterate.py#L687)) | consecutive non-improving evals → escalate (survives rollback) |
| turn_budget | 1500 cold | charged 2/iteration ([iterate.py:1127](../../agent/nodes/iterate.py#L1127)) |
| `MAX_WALLCLOCK_S` | 14h ([config.py:183](../../config/config.py#L183)) | graceful terminate |

`_is_stagnant` fires when `len(scores) >= STAGNATION_WINDOW` and `max(window) - window[0] < 0.02`
([iterate.py:736-757](../../agent/nodes/iterate.py#L736-L757)). ⚠ STALE DOC: PIPELINE.md §6.5 says
"requires ≥50 scores" — the code uses `STAGNATION_WINDOW` (20), not 50.

**Gaps vs the decision:**
- The **2%-over-15** target is a pure env change: `SLM_STAGNATION_WINDOW=15`, `SLM_STAGNATION_MIN_DELTA=0.02`, and set `SLM_MAX_STALL_EVALS=15` so the stall backstop matches the window.
- The **hard 30-evals-per-tier cap does not exist.** A model climbing slowly but steadily (>2% per 15 evals) would never trip stagnation and could exceed 30 evals in a tier. A true cap needs one guard.

### 7.2 How to realize it

**(a) Env-only, gets you the 2%/15 behavior now:**
```
SLM_STAGNATION_WINDOW=15
SLM_STAGNATION_MIN_DELTA=0.02
SLM_MAX_STALL_EVALS=15
```

**(b) The hard 30-eval-per-tier cap — one small guard in `iterate_node`.** Add alongside the
stall/stagnation block (~[iterate.py:1166-1187](../../agent/nodes/iterate.py#L1166-L1187)), before
the LLM call, gated on below-threshold like the stall guard:
```python
MAX_EVALS_PER_TIER = int(_os.environ.get("SLM_MAX_EVALS_PER_TIER", "30"))
...
if current_score < state["stop_threshold"] and len(state["scores"]) >= MAX_EVALS_PER_TIER:
    state["next_action"] = "escalate"
    state["last_intervention"] = "escalate"
    state["last_hypothesis"] = f"per-tier eval cap reached ({MAX_EVALS_PER_TIER})"
    return state
```
`escalate` promotes to the next non-empty higher tier, or terminates cleanly if none fits — so the
cap degrades safely at the top of the pool.

**Terminology note to resolve before wiring:** "15 turns" vs "15 evals." The stagnation window
counts **evaluations** (one per iteration). `turn_budget` counts *turns* at 2/iteration, so "15
turns" would be ~7–8 evals. I've written the policy as **15 evals** (matches the stagnation
machinery and is almost certainly the intent). If you literally mean 15 turn-budget turns, halve
the window to ~8. Flagging rather than guessing silently.

Nothing here is implemented yet — this section is the spec. Say the word and I'll wire (b) + a test.

---

## 8. Appendix: stale-doc corrections found while writing this

Recorded so the drift is auditable (code is ground truth on branch `data-curation-redesign`):

| Claim in a shipped doc | Current code |
|---|---|
| PIPELINE.md: `CURRICULUM_SIZE_FLOOR = 3000`, `DATA_SIZE_CEILING = 10000` | **5000 / 25000** ([config.py:141-143](../../config/config.py#L141-L143)) |
| PIPELINE.md §6.5: `_is_stagnant` "requires ≥50 scores" | requires `STAGNATION_WINDOW` (**20**) ([iterate.py:747](../../agent/nodes/iterate.py#L747)) |
| 2026-07-31 note §4b: Qwen goal "degrades to the 0.8 floor if the endpoint is unreachable" | **FATAL now** — `QwenBaselineUnavailableError`, run stops ([eval_setup.py:78-121](../../agent/nodes/cold_start/eval_setup.py#L78-L121), [threshold.py:24-25](../../agent/threshold.py#L24-L25)) |
| PIPELINE.md §12.2 #4: "difficulty_weighted_sampling is implemented" | removed; the dead `_difficulty_sample` helper has been deleted (2026-08-02), `difficulty_weighted_sampling` does not exist |
| PIPELINE.md §6.1 / §2: floor 3000, ceiling 10000 in the guard table | see row 1 |
