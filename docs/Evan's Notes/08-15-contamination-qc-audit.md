# The RouterBench contamination loop, why QC only looks aggressive, and stretch goals

*2026-08-15 — investigation + implementation*

Six questions answered about the NER and RouterBench runs, a full scan of the ten Slurm logs, and
three changes implemented: naming the rebuild kind in the logs, always reporting the teacher's own
score next to the accuracy goal, and letting the orchestrator raise the goal when a model converges
quickly.

Companions: `08-14-routerbench-postmortem.md` (the
immediately preceding post-mortem, which this extends), `08-13-overnight-run-log.md`
(the campaign log), `08-06-benchmark-tasks-reference.md` (task reference).

> **Follow-up 2026-08-15b** — `08-15b-label-space-lockdown.md`
> implements the three upstream fixes recommended in §6.5 and **corrects §2 of this note**. The claim
> that classification/NER synthesis is safe because the label is copied from an anchor holds only when
> the label is a property of the text the generator controls; RouterBench's `local` is a property of a
> *different model's behaviour*, so copying it fabricates the label. That note also records four new
> bugs found while implementing (B267–B270), including that NER synthesis has never produced a single
> row and that generation-family synthesis is entirely unverified.

**The headline is a contamination loop nobody was looking for.** RouterBench's acquire path mines a
foreign dataset, has its labels invented by an LLM, banks them permanently in the train pool, and
those short foreign rows then drag the length-outlier threshold down until it deletes roughly half
the *real* benchmark data on every rebuild. QC is not too aggressive. QC is the only thing standing
between the run and a poisoned label space, and it has been doing that job silently for 36
iterations.

---

## 1. The BC5CDR baseline of 0.0000 was real

Not a bug. The untrained Q4_K_M `Qwen3-0.6B` emitted an empty entity list inside a markdown fence on
every one of the 800 rows:

```
      [eval] sample predictions (3 of 800):
        input : Dexamethasone induced the transcriptional factor CHOP , a marker for chronic ER stress …
        gold  :
        raw   : ```json [] ```
        parsed:
```
(`logs/slurm/slm-ner-bc5cdr-cse-38455148.out:256-268`)

Zero true positives means exact-match span F1 is exactly 0. Three things confirm the measurement
rather than the instrumentation:

- A **failed** baseline measurement records `n/a`, never `0.0` — `agent/nodes/evaluate.py:234-246`
  does this deliberately, because a caught exception and a genuine zero used to be
  indistinguishable and once credited fine-tuning with a phantom +0.8476.
- The same model, one iteration later, produced well-formed spans and scored **0.7701**. The base
  model *could* emit the format; it just did not without training.
- `QuantizationInfrastructureError` re-raises rather than degrading, so a GGUF problem could not
  have produced a scored zero here.

This is the clean version of the argument the project exists to make: the task is dominated by
format compliance, not knowledge. The 35B teacher scored 0.0999 on the same rows.

### 1.1 Bug found: the sample-prediction display never prints NER gold

`gold :` is blank on **every** NER row, in the baseline block and the fine-tuned block alike (lines
258, 262, 266, 298, 302, 306). The display reads a field NER rows do not carry — they hold
`entities`, not `label`/`answer` — so the one line a human uses to eyeball gold against prediction is
empty exactly where it matters most. Cosmetic in severity, but it defeats the purpose of the display
and it is why the 0.0000 looked suspicious in the first place. **Open.**

---

## 2. A low-scoring teacher is dangerous — but only on one of the two synthesis paths

The teacher is `Qwen/Qwen3.6-35B-A3B` (`config/config.py:142-143`). Whether its score disqualifies
it depends entirely on what it is asked to do, and the codebase happens to split along exactly that
line.

| Path | Task types | What the teacher is asked | Teacher skill matters? |
|---|---|---|---|
| `_synthesize_new_gold` (`data/curriculum.py:500-578`) | classification, NER | "write another utterance of class X", label copied from the anchor | **No** — it never solves the task |
| `_synthesize_new_correct` (`data/curriculum.py:391-403`) | generation family incl. `function_call`, `diff` | invent the input **and** the correct output | **Yes, critically** |

The first path is safe by construction, and its docstring is right that an out-of-vocabulary label
is impossible — *relative to the anchors it is given*. §6 is about what happens when the anchors
themselves are poisoned.

The second path has no protection at all. `curate._verifier_for` returns `None` for those task
types, so nothing checks the output. On `calendar_json` the teacher scores **0.2176** and the
curriculum target was 8,929 rows against 3,250 real ones — asking a model that gets the task right
22% of the time to manufacture ~5,700 training targets, unchecked. That is a wrong-label factory.
**Open, and it should block any format-bound run at scale.**

---

## 3. RouterBench labels a 7B model, and no choice of column can fix that

The routing boundary is `mistralai/mistral-7b-chat` (`data/loaders/routerbench.py:33`). I enumerated
every correctness column in the pickle:

| Candidate | Params | `local` rate |
|---|---|---|
| `meta/code-llama-instruct-34b-chat` | 34B | 0.208 |
| **`mistralai/mistral-7b-chat`** | **7B** | **0.299** ← current |
| `meta/llama-2-70b-chat` | 70B | 0.351 |
| `WizardLM/WizardLM-13B-V1.2` | 13B | 0.452 |
| `mistralai/mixtral-8x7b-chat` | 8×7B | 0.568 |
| `gpt-3.5-turbo-1106` / `claude-instant-v1` / `claude-v1` / `claude-v2` / `Yi-34B-Chat` | — | 0.636–0.687 |
| `gpt-4-1106-preview` | — | 0.843 |

The pool is 0.6B–4B (`docs/model_pool.md`). **The smallest model in RouterBench is 7B — over 10×
the largest pool member.** The loader comment is honest that mistral-7b is the closest available
analogue, but "would a 7B get this right" and "would Qwen3-0.6B get this right" are different
questions and the second is strictly harder. No `small_model_key` produces pool-aligned labels.

Getting pool-aligned labels means relabelling against our own pool, which needs gold answers per
prompt. RouterBench does not ship them — the columns are `sample_id`, `prompt`, `eval_name`,
per-model graded correctness, per-model `|model_response`, `|total_cost`, and
`oracle_model_to_route_to`. Two viable routes:

1. **Recover gold from the source benchmarks.** `eval_name` names 86 upstream evals — hellaswag
   (10,042 rows), grade-school-math (7,450), mmlu-professional-law (1,534), arc-challenge (1,470),
   winogrande (1,267), … — all on HF. Join by prompt text, then run Qwen3-0.6B/1.7B/4B and label
   `local` where the pool model is actually correct. Exact, and it makes the label a property of
   *our* deployment rather than someone else's.
2. **Use `gpt-4-1106-preview|model_response` as pseudo-gold** on the 84% of rows where gpt-4 was
   graded correct. Much cheaper, adds label noise.

**Open — needs a decision.** Not changed, because it is a task redefinition, not a bug fix.

---

## 4. The eight tasks, their sources, and where the data lives

Registry: `agent/nodes/cold_start/eval_setup.py:31-43`, selected by `SLM_BENCHMARK_TASK`. One loader
per task under `data/loaders/`, one Slurm script per task under `tests/pipeline/`. There is no
per-task YAML. HF cache root for all cluster runs is
`HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache` (~185 GB), set in
`tests/pipeline/_l40s_task_body.sh:27`.

| Slug | Input → output | Metric | Upstream source | Local materialisation |
|---|---|---|---|---|
| `clinc150` | utterance → 1 of 151 intents | macro-F1 | HF `clinc/clinc_oos` cfg `plus` | HF cache only |
| `dialogsum_samsum` | dialogue → 1–3 sentence summary | LLM-judge mean 0–1 | HF `knkarthick/dialogsum` + `knkarthick/samsum` | HF cache; unused frozen copy at `data/local/samsum/` |
| `xlam_bfcl` | request + tool schemas → JSON call list | `ast_arg_match` | train HF `Salesforce/xlam-function-calling-60k` (gated, needs `HF_TOKEN`); eval BFCL v3 JSON via `hf_hub_download` | HF cache only |
| `coedit` | edit instruction + text → unified diff | `apply_match` | HF `grammarly/coedit` | HF cache only |
| `routerbench` | prompt → `local` / `route` | minority-class F1 | HF `withmartian/routerbench` (`routerbench_0shot.pkl`, pandas) | HF cache pickle |
| `medqa` | USMLE MCQ → A/B/C/D | macro-F1 | HF `GBaker/MedQA-USMLE-4-options` | HF cache only |
| `ner_bc5cdr` | sentence → Chemical/Disease spans | exact span-F1 | frozen bundle; T-NER raw JSON fallback | **`data/local/bc5cdr/`** — 5,096 train / 5,865 test, checksummed |
| `calendar_json` | scheduling request → `calendar.events.insert` JSON | `ast_arg_match` | train HF `WillHeld/top_v2` (`reminder`/`CREATE_REMINDER`); eval SGD `Calendar_1` `AddEvent` | TOPv2 in HF cache; **SGD fetched live from GitHub** |

Metric wiring is `eval/harness.py:58-68` → `eval/scorers/<task_type>.py`. Note that for a
**binary** task the reported number is minority-class F1, not macro over both classes
(`eval/scorers/classification.py:62-65`) — which is why a `route`-collapsed RouterBench model scores
exactly 0.000 rather than a flattering 0.63.

`coedit` and `medqa` have Slurm scripts but have never produced a run log.

### 4.1 One example row per task, verbatim

```json
clinc150          {"text": "does village inn let you make reservations", "label": "accept_reservations"}
routerbench       {"text": "猜字谜，根据我给的描述猜出一个字…", "label": "route"}
ner_bc5cdr        {"text": "Famotidine - associated delirium .",
                   "entities": [{"text": "Famotidine", "type": "Chemical"},
                                {"text": "delirium", "type": "Disease"}]}
medqa             {"text": "A junior orthopaedic surgery resident is completing a carpal tunnel repair…",
                   "label": "B"}
dialogsum_samsum  {"text": "Sarah: how much longer?\nDaina: I need to put my make up\n…",
                   "answer": "Daina needs about an hour more to get ready.", "label": "generation"}
coedit            {"text": "Paraphrase this sentence", "src": "Why are you arresting me?\n",
                   "tgt": "Why am I being arrested?\n",
                   "answer": "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-Why are you arresting me?\n+Why am I being arrested?\n"}
xlam_bfcl         {"text": "Where can I find live giveaways for beta access and games?",
                   "answer": "[{\"arguments\": {\"type\": \"beta\"}, \"name\": \"live_giveaways_by_type\"}, …]"}
calendar_json     {"text": "…Current date and time: 2026-08-09T11:00:00 (Sunday).\n\nRemind me to pack my lunch for tomorrow.",
                   "answer": "[{\"arguments\": {\"end\": {\"dateTime\": \"2026-08-10T10:00:00\"}, \"start\": {\"dateTime\": \"2026-08-10T09:00:00\"}, \"summary\": \"pack my lunch\"}, \"name\": \"calendar.events.insert\"}]"}
```

### 4.2 Reproducibility hole

`calendar_json`'s eval half is downloaded from
`raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue` at load time and
is **not cached**. One upstream commit silently moves the eval set out from under every past result.
**Open.**

---

## 5. Synthetic rows do not accumulate — every rebuild discards the previous ones

This is the answer to "why is the synthetic band so small", and it is a mental-model correction
rather than a bug.

`curate_node` begins each rebuild with `selected_rows = []` (`agent/nodes/curate.py:881`) and refills
from the train pool. The previous dataset is read only to compute the no-shrink target and to decide
whether resampling can still yield novelty. **Synthetic rows from iteration N−1 are never carried
into iteration N.** 500 rows generated 36 times is not 18,000 rows; it is ~500 rows, 36 times over,
each from scratch. Confirmed on disk: `dataset_v1`, `v5`, `v10` all sit at 3,420–3,510 rows.

On top of that the per-rebuild yield is brutal. One rebuild traced end to end
(`slm-routerbench-l40s-38493142.out:14130-14183`):

| Stage | Rows |
|---|---|
| Plan `synthesize` requested | 241 → **kept 62** (teacher rejected 74%) |
| Synth-fill requested | 749 → **kept 225** (teacher rejected 524) |
| Admitted pre-QC | 287 |
| Surviving QC | **93** (`synthetic` 29 + `synthetic_positive` 64) |

**990 generated → 93 survive: 9.4% yield.**

And the chart under-reported even that. `_plot_composition` drew `n_hard_generated`, which counted
only `_provenance == "synthetic"` — 29 of 3,401 rows, 0.9%. The other 64 fell into the grey
"unattributed" band because synth-fill never stamped `_strategy_origin`. Both effects push the same
way, which is why the orange band looked like a rounding error. **Fixed — see §7.1.**

The dataset also lands far below target, and to its credit says so:

```
Quality control removed 2256 row(s) total (5657 → 3401)
⚠ PROCEEDING BELOW DATA TARGET: 3401 row(s) vs target 6181 (short by 2780). Synth-fill runs
BEFORE quality control and there is no refill afterwards…
```

---

## 6. QC is not too aggressive — it is containing an upstream contamination loop

I went in expecting to lower thresholds. The measurements say the opposite. Here is the chain.

### 6.1 Acquire substitutes a foreign dataset

`web_acquire._BENCHMARK_ALIASES` has no `routerbench` entry, so Stage-0 fails on the very benchmark
the task is about and agentic discovery substitutes something else — 21 times in the l40s run:

```
[acquire] peek withmartian/routerbench (config=None) failed: No (supported) data files found
[acquire] loaded AGENTIC HF dataset 'anasnassar/llm-query-complexity-benchmark' (train=4800 test=80)
```
(`slm-routerbench-l40s-38493142.out:658-660`)

### 6.2 That dataset has no `local`/`route` labels — an LLM invented them

I loaded it. Its columns are `text`, `source`, `subject`, `domain`, `ground_truth`, `id`, and
`ground_truth` takes values **`LOW` (1,423) / `MEDIUM` (1,440) / `HIGH` (1,137)** — query-complexity
tiers over StackExchange/MMLU/PubMedQA text. Nothing resembling a routing decision.

Yet `dataset_v10.jsonl` contains 1,155 of its rows carrying `local`/`route`, split **579/576**.
RouterBench's true base rate is 30/70. That near-perfect 50/50 is the tell.

The labels come from `_llm_map_dataset` (`data/loaders/web_acquire.py:953-995`), which asks Claude
for a `label_map` from the foreign schema into the run's label space, and `_materialize_from_mapping`
then applies `lmap.get(str(lab), lab)` (line 1017–1022) — **unmapped values pass through verbatim**.
Each acquire round gets a *fresh, non-deterministic* mapping, which is exactly why the
out-of-vocabulary label set grows monotonically through the run:

| Iterations | Out-of-vocabulary labels QC removed |
|---|---|
| 1–9 | `cloud` |
| 10–19 | `cloud`, `on_device` |
| 20 | `+ router` |
| 21 onward | `+ remote` |

Every round hallucinated a new class, and `_merge_persistent_train_rows`
(`agent/nodes/curate.py:250-272`) banks them in the train pool **permanently**.

### 6.3 The guard that should have stopped it is per-source, not per-row

```636:638:data/loaders/web_acquire.py
        overlap = mined_labels & _known_labels
        if overlap:
            return True
```

B222 added this to catch a source whose labels are *entirely* disjoint (raw integer class ids). It
passes the whole source on **any** overlap. Because Claude mapped some rows to `local`/`route`, the
source passed and `cloud`/`on_device`/`router`/`remote` rode along. QC's label-space filter then
deleted 1,166 of them per rebuild for the rest of the run.

### 6.4 Contamination turns the length filter into a wrecking ball

`length-outlier` cuts anything longer than 3× the dataset's **own** median. The mined rows are short
— median 75 characters against real RouterBench's 715 — so they drag the median down and pull the
cutoff with them. Measured against the real benchmark:

| Dataset version | mined rows | median len | 3× cutoff | share of real RouterBench that cutoff deletes |
|---|---|---|---|---|
| uncontaminated | 0 | 715 | 2145 | **0.6%** (216 / 36,497) |
| `v1` | 0 | 416 | 1248 | 9% (3,315) |
| `v5` | 958 | 295 | 885 | 47% (17,041) |
| `v10` | 1,155 | 269 | 807 | **48%** (17,667) |

The same filter that removes 0.6% of clean data removes about half of it once contaminated. You can
watch it eat the real rows: `train_anchor` median length falls **572 → 432 → 396** across v1/v5/v10
while the pool it is sampled from never changes. In-run that was 813 real rows deleted per rebuild,
and the per-iteration removals climb 58 → 39 → … → 566 → … → **813** as mined rows accumulate.

One reassurance: it is not strongly label-biased. Deleted rows are 32.6% `local` against 29.9%
overall, so it is near-uniform destruction rather than skew.

### 6.5 Verdict, and the worse problem QC cannot see

**Keep every QC threshold where it is.** On uncontaminated RouterBench the filters are almost
inert. Three upstream fixes make QC quiet on its own:

1. Add a `routerbench` alias to `web_acquire._BENCHMARK_ALIASES` so acquire uses the real loader.
2. Make the label guard drop out-of-vocabulary **rows** rather than waving through a whole source
   on any overlap.
3. Compute the length-outlier median over **trusted** rows only, so mined data cannot move the
   goalposts. (Or make the bound absolute and task-aware.)

None implemented — all three change acquisition behaviour, and 38493142 was still running.

And the thing QC structurally cannot catch: **1,155 rows — 34% of the final training set — are
foreign-dataset rows with LLM-fabricated labels at the wrong base rate.** They survive because
`local` and `route` are spelled correctly. That is worse than everything QC removes, and it has been
in every RouterBench rebuild since the first acquire round. It is a plausible contributor to the
`route` mode collapse in §8.

---

## 7. What was implemented

### 7.1 Item 1 — every rebuild names its KIND

`data_rebuild` in a trajectory was ambiguous, and a single rebuild runs several kinds: the plan
strategy, then resample-fill, then synth-fill. Changes in `agent/nodes/curate.py`:

- `REBUILD_KIND_NAMES` + `rebuild_kind_name()` give one canonical token per producer:
  `resample`→**reshuffle**, `mine_new_real_source`→**mine-new-real**,
  `surgical_synthesize`→**surgical-synth**, `synthesize`→**fill-synth**, `synth_fill`→**synth-fill**.
  An unmapped origin passes through unchanged rather than being hidden as "unknown".
- The opening banner names the plan's primary kind and states that fill kinds may also run.
- A new closing line reports every kind that actually produced rows, with novelty:

```
DATA REBUILD kinds (plan strategy=synthesize): reshuffle=3308 row(s) (122 novel);
synth-fill=64 row(s) (62 novel); fill-synth=20 row(s) (19 novel); surgical-synth=9 row(s) (9 novel)
```

- **Bug fixed:** `_synth_fill_to_target` set `_provenance` but never `_strategy_origin`, so its rows
  reported as `unattributed`. Now stamped `synth_fill`.
- `last_curation` gains `n_synth_total` (all of `synthetic` + `synthetic_fill` +
  `synthetic_positive`) and `n_synth_by_provenance`; `strategy_composition` entries gain `kind`.
- `agent/run_graphics.py` now plots `n_synth_total`, falling back to `n_hard_generated` for older
  runs. The orange band finally reflects the real synthetic share.

### 7.2 Item 2 — the teacher's score is always reported next to the goal

A floored goal and a teacher-set goal printed the same number, so BC5CDR's `threshold 0.8000` hid a
teacher score of 0.0999.

- `eval_setup._calibrate_qwen_goal_if_pending` now records `floored: bool` and says which input won
  at calibration time:
  `[threshold] Qwen baseline 0.0999 → goal 0.8000 (floor 0.80) — FLOOR WON: the teacher scored
  below 0.80, so the goal is the floor, not the teacher's 0.0999`
- `agent.threshold.describe_threshold_provenance()` renders it for the summary, handling the
  floored, teacher-set, pending, manual-override and unrecorded cases.
- `tests/pipeline/run.py` prints two new summary lines under `best F1`:

```
  goal source: floor 0.80 OVERRODE the Qwen-3.6 teacher, which scored only 0.0999 span_f1 …
  teacher    : Qwen-3.6 zero-shot 0.0999 span_f1  (no fine-tuning; this is the score the goal
               is calibrated against)
```

- `scores.json` gains `initial_stop_threshold` and the full `threshold_calibration` block.

### 7.3 Item 3 — the orchestrator can raise the goal on a fast win

Previously the orchestrator could only ever **lower** the threshold. BC5CDR cleared a floored 0.8000
in five iterations and 81 minutes and stopped with hours of budget left.

**How it works.** On any score that meets `stop_threshold`, `_route_score_at_threshold` calls
`_maybe_raise_threshold` *before* the downward-probe/terminate decision — both of those treat the
goal as settled, and if the bar is about to move the whole convergence question reopens.

1. **Bank first.** `_bank_convergence` records `{threshold, score, iteration, selector}` into
   `state["convergence_banked"]`, keeping the highest goal ever cleared. This happens
   unconditionally, even when raising is disabled or the orchestrator declines.
2. **Ask the orchestrator.** A small dedicated call (`stage="threshold_raise"`, its own system
   prompt) — not a field on the main intervention decision, because that prompt is only built for
   below-threshold scores. It is shown: current goal, **goal provenance from §7.2**, score and
   margin, iterations used, last 12 scores, turns used/remaining, the ceiling, the minimum step, and
   previous raises. It returns `{"raise_goal": true, "new_threshold": …, "reason": …}` or
   `{"raise_goal": false, "reason": …}`.
3. **Validate and ratchet.** The proposal must exceed the run's high-water mark
   (`max_stop_threshold`) by at least `_THRESHOLD_RAISE_MIN_STEP = 0.005`, and is clamped to
   `THRESHOLD_CEILING = 0.99`. Clamping against the high-water mark rather than the current goal is
   what stops a lower-then-raise cycle from reusing the same band forever.
4. **Continue.** A successful raise returns `False` from `_route_score_at_threshold`, so the caller
   treats the score as below-threshold again and the run keeps training against the new goal.

**Why it terminates.** Raises are strictly increasing with a fixed minimum step and a hard ceiling,
so a run can perform at most `(0.99 − initial) / 0.005` of them; each additionally requires the model
to actually clear the previous goal, and at most one is asked per iteration
(`_threshold_raise_asked_iteration`). Every pre-existing budget — turn budget, wall clock, graph
steps, stagnation, eval cap — applies unchanged. `test_repeated_raising_terminates` asserts the bound
directly against a maximally greedy orchestrator.

**Why a missed stretch goal is not a failure.** Judging only against the final threshold would
report "budget exhausted" for a run that cleared the bar it was calibrated against. `run.py` now
treats a banked convergence as convergence and prints:

```
  ⚠ CONVERGED AT THE ORIGINAL GOAL, stretch goal missed: cleared 0.8000 with 0.8098 at
    iteration 5; the raised goal 0.8700 was not reached (best 0.8300). The run is a SUCCESS
    against its calibrated goal.
```

**Controls.** `SLM_THRESHOLD_RAISE=0` disables it; `SLM_CHEAP=1` skips it (an optimisation, not a
correctness requirement). Default under pytest is **off**, set in `conftest.py`, so the many existing
tests that assert terminal routing on a threshold-clearing score do not each make a live call.
`scores.json` gains `convergence_banked` and `threshold_raises`.

### 7.4 Two bugs the test suite caught during implementation

- **`_threshold_raise_asked_iteration` was undeclared in `AgentState`.**
  `test_every_key_iterate_persists_is_declared` caught it. Undeclared keys are dropped on the
  LangGraph state merge, so the once-per-iteration guard would have silently done nothing and the
  orchestrator would have been asked twice per turn. Declared.
- **`tests/nodes/test_iterate_prompt.py` had been failing since the 2026-07-31 curation redesign.**
  Its fixture returned `{"primary_strategy": "resample_existing"}`, a field the redesign retired, so
  validation failed, the reask failed identically, and the test's real assertion — that the curation
  counts reach the orchestrator prompt — had stopped running. Fixture updated to `strategy`.

### 7.5 Test status

`955 passed, 1 failed` across `tests/` (excluding `tests/pipeline/`). The single failure is
`test_capability_prompts.py::test_code_planner_and_model_choice_prompts_target_apps_introductory`
(`APPS introductory`), a **pre-existing** failure recorded in
`08-11-throughput-sizing-fixes.md` and untouched by this work.

New tests: `tests/nodes/test_threshold_stretch_goal.py` (22 — banking, ratchet, ceiling, high-water
clamp, termination bound, once-per-iteration, disabled paths, validation, routing, provenance) and
`tests/nodes/test_rebuild_kind_logging.py` (6). `test_cost_observability.py`'s paid-call-site guard
updated from 2 to 3 sites and now also asserts both billing stage names.

---

## 8. Other issues found in the log scan

Ordered by severity. All ten Slurm logs scanned.

| # | Issue | Evidence | Severity |
|---|---|---|---|
| 1 | **`route` mode collapse scored as 0.0000 and iterated through.** `easy=0.000 medium=0.000 hard=0.882` — answers `route` for everything; the hard bucket is route-heavy so it looks healthy. Metric is correct; the orchestrator burns full train/quantize/eval cycles on a degenerate solution. | l40s 5539-5542, 6467, 6719, 11822, 14252; cse 8818-8821 | **High** |
| 2 | **`calendar_json` has two disagreeing baselines for one model.** Reference path `ast_arg_match=0.2176`; GGUF iteration path `0.0000` with `failures=478/478`. The orchestrator optimises against whichever it sees. | `slm-calendar-json-cse-38505239.out:95, 257, 297-301` | **High** |
| 3 | **`calendar_json` eval set is 478 rows against a target of 800**, so its scores are not comparable to the other tasks. | same log :58; `38455147.out:134` | **High** |
| 4 | **`xlam_bfcl` spun on a dead data lever** — 60 `no_novelty` hits and 14 rollbacks after acquire stopped producing, including a 0.690 → 0.464 collapse. Never escalated past tier 0 in 3h51m. | `slm-xlam-bfcl-cse-38505237.out:126, 692, 1124, 1428` | **High** |
| 5 | **Orchestrator JSON truncated at 4,096 output tokens, 20×**, each costing a ~$0.02 reask. The error message is clear (`too long, NOT malformed`); the fix is a bigger budget, not a retry. | `slm-dialogsum-samsum-l40s-38303490.out:316-317` | **High** |
| 6 | **144 rollbacks and 157 GGUF quantize steps across all runs**, mostly discarded. RouterBench alone threw away 103 full cycles (59 cse + 44 l40s). | all logs | Medium |
| 7 | **NER test agent reports `medium=None` 8×** — that eval set has only easy and hard buckets, yet the orchestrator was handed `medium=0.25` as a difficulty weight. Converged anyway, but it reasoned about a bucket that does not exist. | `slm-ner-bc5cdr-cse-38455148.out:315, 495, 524, 754` | Medium |
| 8 | **DialogSum tier 3 gained nothing over a strong baseline** — 15 iterations, `baseline=0.7157 Best FT=0.7157 Δ=+0.0000`, budget exhausted at 0.716 vs 0.800. | `…38303490.out:13657-13681` | Medium |
| 9 | **Synth endpoint unreachable for ~35 min at RouterBench startup** (67 poll lines); first rebuilds ran gold-only. Recovered cleanly at 07:34. | l40s 23-25, 158 | Medium (benign after recovery) |
| 10 | **Agentic discovery repeatedly proposes contaminated sources** — 15 `REJECTED by schema/integrity/overlap` across RouterBench and calendar (80-row train/test overlap). The firewall works; the discovery quality is the problem. | l40s 657; calendar 116-118 | Medium |
| 11 | **`__EXTRACTION_FAILED__` grows with training** — 283 occurrences; route-class failures rise 4 (iter 1) → 22 (iter 3). Becomes a top confusion pair. | l40s 1135 | Medium |
| 12 | Paid Exa acquisition rounds consumed on RouterBench mining (16 `paid-round` lines) for modest novel yield against 4,800 candidates. | l40s 661 | Medium |
| 13 | Local inference failure rate 0.01–2% (`local calls=280182 failures=35`; calendar 10/492). No systemic outage. | routerbench-cse 20798 | Low |
| 14 | Unsloth import-order `UserWarning` ~620×. Zero OOM events across all logs. | all | Cosmetic |

---

## 9. Open items

| # | Item | Status |
|---|---|---|
| 1 | `web_acquire` has no `routerbench` alias → foreign dataset substituted (§6.1) | **fixed 08-15b** |
| 2 | Label guard is per-source on any overlap, not per-row (§6.3) | **fixed 08-15b** — vocabulary pinned and closed |
| 3 | `length-outlier` median computed over untrusted rows → deletes ~48% of real data (§6.4) | **fixed 08-15b** — median anchored on trusted rows |
| 4 | 34% of the RouterBench curriculum is LLM-fabricated labels at the wrong base rate (§6.5) | **open** |
| 5 | RouterBench labels a 7B model; pool is 0.6B–4B (§3) | **open — needs a decision** |
| 6 | `_synthesize_new_correct` unverified for generation-family; teacher at 0.2176 on `calendar_json` (§2) | **open** — confirmed as B269 (`450/450 kept`, verify hook always None) |
| 7 | NER sample-prediction display never prints gold (§1.1) | **fixed 08-15b** (B263) |
| 8 | `calendar_json` SGD eval fetched live from GitHub, uncached (§4.2) | **open** |
| 9 | Synth-fill runs before QC with no refill → dataset chronically below target (§5) | **open** (logged loudly) |
| 10 | Items 1–14 in §8 | **open** |
| 11 | Name the rebuild kind in the logs | **implemented** (§7.1) |
| 12 | Report the teacher's score next to the goal | **implemented** (§7.2) |
| 13 | Orchestrator-decided stretch goals | **implemented** (§7.3) |
