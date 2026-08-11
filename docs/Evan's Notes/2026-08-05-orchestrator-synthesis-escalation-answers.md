# Orchestrator, synthesis and escalation — answers, fixes, and a full log audit (2026-08-05)

Follow-up to `2026-08-04-clinc150-run-review-questions-and-log-changes.md`.
Reference run: **`slm-clinc150-cse-38155022`** (converged, macro-F1 0.8971 vs 0.8891).

Each question is quoted, then answered against the code. **The code was re-read for every claim**;
where a previous note or a subagent was wrong, that is called out.

---

## 1. "use claude sonnet 5 at the cheaper setting, switch all the configs and ensure every run uses that as its orchestrator from now on"

**Done.** Verified `claude-sonnet-5` is the correct canonical API model id (dateless ids are pinned
snapshots from the 4.6 generation onward). Changed:

| File | Change |
|---|---|
| `config/config.py:48` | `ORCHESTRATOR_MODEL` default → `claude-sonnet-5` |
| `tests/pipeline/_l40s_task_body.sh` | `SLM_ORCHESTRATOR_MODEL` default → `claude-sonnet-5` |
| `config/config.py` docstring | model table refreshed; 1M-context note corrected |

That covers **every** pipeline run: all eight benchmark scripts source `_l40s_task_body.sh`, and
anything else picks up the `config.py` default.

`agent/cost.py` already carried correct Sonnet 5 pricing ($2/$10 with
`valid_through: 2026-08-31`), so cost tracking is accurate with no change.

**Cost impact:** the reference run's 324,233 in / 19,153 out would cost **$0.84** instead of
**$1.26** — about 33% less. ⚠️ **Only until 2026-08-31**, after which Sonnet 5 is $3/$15, identical
to Sonnet 4.6. There is no permanent saving here, just an introductory window.

⚠️ **Untested compatibility risk.** One source states Sonnet 5 has adaptive thinking on by default
and **returns HTTP 400 for explicit temperature/top_p/top_k**. I grepped the orchestrator call
paths and found **no** temperature/top_p/top_k passed on any Anthropic call, so this should be
safe — but it has not been exercised against the live API. Watch the first run's `stage=iterate`
cost events for `status=error`.

---

## 2. "Why do the hardware specs even need to be reviewed by claude?"

Because the CSV is a raw spec dump, not a constraint set. `_lookup_local_db`
(`hardware_research.py:89`) returns matched rows of `data/devices.csv`; the pipeline needs
`memory_mb`, `storage_mb`, `latency_ttft_ms`, `power_w` and a reference chip. Claude does three
things the CSV cannot: pick which of the fuzzy matches is the right device, convert raw specs into
a *usable* budget (12,288 MB RAM → 10,240 MB usable after OS overhead), and normalise
inconsistent CSV formats.

**Whether it should is a fair question.** The step is one call (~$0.006) but it is
**non-deterministic**: two runs on identical input produced storage budgets of 204,800 MB and
153,600 MB — a 33% swing in a constraint that gates model selection. For a *known* device a
deterministic table lookup would be strictly better. My recommendation: keep the LLM path as the
fallback for unknown devices, but add a small curated `device → constraints` table for the phones
you actually target, so repeat runs are reproducible.

---

## 3. "Autonomous dataset sizing is not routed yet"

Correct, and worth stating precisely. `_apply_data_targets` (`task_analysis.py:45-78`) reads
`plan.get("curriculum_size")`, but on the `SLM_BENCHMARK_TASK` path `task_plan` is `None`, so it
falls back to `DATASET_SIZE_BY_TYPE["classification"] = 150` and clamps **up** to
`CURRICULUM_SIZE_FLOOR = 5000`. The LLM planner (`task_planner.py:104-135`) *can* choose these
values, and logs `[planner] data targets (pre-clamp)` + `[planner] rationale` when it does — those
lines are absent from this run, which is the tell.

So on curated-benchmark runs the sizes are pure config. Not changed — routing the planner into the
benchmark path is a behaviour decision for you, not a bug fix.

---

## 4. "Fix this: ... 164 mined_real rows are labelled '0', '1', '2'" / 14. "Explain this bug"

**Explanation.** `web_acquire._convert` already tries to resolve integer class ids:

```python
label_names = ds.features[lcol].names if hasattr(ds.features.get(lcol), "names") else None
if isinstance(lab, int) and label_names:
    lab = label_names[lab]
```

That works when the column is a HF `ClassLabel`. I checked the actual dataset:

```
DeepPavlov/clinc150 features: {'utterance': Value('string'), 'label': Value('int64')}
```

It is a plain `int64`. There are **no `.names`** to resolve, and the integers are dataset-local ids
with no mapping shipped alongside them — they are unrecoverable from the source. So the ids were
stringified into `"0"/"1"/"2"` and merged as if they were intent names, adding 164 rows (~5% of the
curriculum) labelled with classes that exist in no label space and can never match an eval label.

**Fixed.** `mine_additional_real_rows` now derives the run's label space from `existing_rows` and
**rejects any mined classification source whose labels are entirely disjoint from it**, with an
explicit reason in the log:

```
[mine] REJECTED local source: none of its 3 label(s) ['0', '1', '2'] exist in the run's
151-label space — most likely raw class ids from a non-ClassLabel column. Merging them would
inject unmatchable labels into training (B222).
```

Sources that share the label space are unaffected. Two regression tests cover reject and accept.
I rejected rather than remapped deliberately: remapping needs an id→name table the source does not
provide, and guessing one would be worse than dropping the rows.

---

## 5. "I would like you to print out in the logs maybe 3 examples of the model's answers for the eval set"

**Done** — `eval/harness.py::_log_prediction_samples`, called on every eval. It prefers examples
that **failed extraction**, since those are the diagnostic ones:

```
      [eval] sample predictions (3 of 800; 82 extraction failure(s) this eval):
        input : what expression would i use to say i love you in italian
        gold  : translate
        raw   : The user is asking about translation, so the intent is translate.
        parsed: __EXTRACTION_FAILED__  <-- EXTRACTION FAILED
```

This is exactly what distinguishes "picked the wrong class" from "answered in a format the
extractor cannot read". Count is configurable via `SLM_EVAL_SAMPLE_LOG_N` (default 3, 0 disables).

---

## 6. "requested: 200 ... candidate_rows: 4049 ... why were only 200 requested? why were 4000 fetched? aren't all the candidate rows novel since the dataset is being built for the first time?"

Three separate things:

**Why 200 requested.** `requested_rows = plan["new_real_rows"]`, chosen by the orchestrator in its
rebuild plan and clamped by the plan validator. It asked for 200 new real rows — a deliberately
small increment on top of a ~3,400-row pool, not a full rebuild.

**Why 4,049 fetched.** The loader requests
`max(300, len(existing_rows) + requested * 4)` (`web_acquire.py:673`) — deliberate over-fetch,
because most fetched rows will be discarded as duplicates of what you already have. It then stops
as soon as it has 200 novel rows (`if len(novel) >= requested: break`). So 4,049 is the *candidate
pool size*, not work done or rows kept.

**Why only ~5% novel — and no, they are not all novel.** This is the part worth understanding:
`novel` means *not already in the run's row set*, not "never seen before". The mined source was
`DeepPavlov/clinc150` — a **different packaging of the same CLINC150 corpus** already loaded as the
curated benchmark. Nearly every fetched row was a text-duplicate of an existing row and was
dropped by the `seen` set. `novel_fraction: 0.0494` is measuring that overlap correctly.

So the mining spent a paid discovery round to re-fetch a dataset the run already had, and the ~200
rows it did keep were the ones that differed — which, per Q4, were exactly the integer-labelled
junk. Both problems have the same root: the acquisition path had no notion of "this is the same
corpus I already loaded".

---

## 7. "Explain more about this issue [the reask not firing], I want it to be able to correct itself if the iterate LLM call fails"

**First, a correction.** My previous note said 6 iterate calls; the correct figure from the ledger
is **19 iterate calls, 6 validation failures, 0 reasks**. A subagent also proposed that the run
predated the reask code — **that is wrong**, and I verified it: `git show HEAD` contains the
`except ValueError → _reask_json_only` block, and the cost events record callsite
`_llm_iterate:950`, which matches HEAD's line numbering. The run *did* execute code containing the
reask.

**What I could verify.** The mechanism works in isolation — driving `_parse_decision_json` with the
exact failing payload raises, and calling `_reask_json_only` records a `iterate_json_reask` stage.
Both `_parse_decision_json` return paths validate, so there is no unvalidated escape. All six
failures were the identical `ValueError`. **I still cannot explain why the reask did not run**, and
I am not going to invent a cause.

**What I changed so it self-corrects regardless** (`agent/nodes/iterate.py`):

1. **Catch `Exception`, not just `ValueError`.** Self-correction should be attempted for any
   validation failure; a narrow clause silently skips the reask for anything else raised.
2. **Explicit logging on both sides**, so the next run states plainly what happened:
   ```
   Decision failed validation (ValueError: ...) — asking the orchestrator to correct itself (1 reask)
   Reask succeeded — using the corrected decision
   ```
   (or `Reask also failed (...) — falling back`.) If a reask is skipped again, the log will now say so.
3. **A salvage path.** If the reask *also* returns retired knobs, `_strip_retired_hyperparams`
   drops just those keys and keeps the tunable ones, tagging `_dropped_fields:
   ["retired_hyperparams"]`. Those knobs are never applied by the trainer, so dropping them yields
   exactly the decision the orchestrator could legally have written. This mirrors the existing
   precedent for a stray `hyperparams` block on a `data_rebuild`, whose code comment records that
   rejecting whole decisions cost the NER run 65 orchestrator-authored plans.

Verified end-to-end: with a model that repeats the mistake, the pipeline now records
`['iterate_json_reask']` and returns a usable decision instead of discarding it.

---

## 8. "you need to distribute the synthetic data generation amongst the different labels ... tell me what you currently do and what you'd recommend. reiterate what the quality controls are again?"

**What it did (the bug).** `target_label = target_labels[0]` — the *same* target for every example.
On 151 intents, all 48 surviving synthetic rows landed on 2 labels (`recipe` ×39,
`book_flight` ×9). `all_labels` also came from an unordered `set`, so the choice was not even
reproducible across a requeue.

**What it does now (fixed).** `all_labels` is `sorted()`, and each anchor draws its target from a
seeded `random.Random(20260804)` across the full label space. Verified on a 5-label fixture:
synthetics now spread across all 5 instead of 1.

**The quality controls, in full** (`data/curriculum.py::apply_quality_controls`):

| Control | Rule | Why it deleted the synthetics |
|---|---|---|
| **Label balancing** | no label may exceed **3× the smallest** label's count | ~1,500 rows piled on one label — this rule exists precisely to delete that |
| **Length outliers** | drop rows longer than **3× the median** length | minor |
| **Surface-form dedup** | Jaccard similarity dedup | minor |
| **NER entity diversification** | (NER only) cap repeated entities | n/a here |

There is **no refill after QC**, which is why 1,549 generated rows became 48 and the dataset was
written at 3,461 instead of 5,000.

**Recommendation beyond the fix:** add a post-QC top-up pass so the target is enforced *after*
quality control rather than before it. Right now `target_rows` is a pre-QC aspiration.

---

## 9. "My idea behind generating synthetic data is that it would generate gold examples, while keeping the hard negative ratio ... I want the intervention capable of generating gold examples as well"

**What it did.** For classification/NER, synthesis was **hard negatives only**. Every generated row
was filed under a *different* label than its anchor, so synthesis could never add in-class
coverage and always skewed the label histogram.

**What it does now.** `synthesize_examples` generates a **mix**, controlled by
`SLM_SYNTH_HARD_NEGATIVE_RATIO` (default **0.35**):

- **65% new gold** — `_synthesize_new_gold` produces new in-class utterances that **keep the
  anchor's label**, with anchors drawn round-robin across labels so rare classes get equal
  attention. Tagged `_provenance: synthetic_positive`.
- **35% hard negatives** — the existing contrastive path, now spread across the label space.

Logged explicitly:
```
[synth] requested 40 row(s): 26 new-gold + 14 hard-negative (ratio=0.35) -> produced 54
```

This should also largely fix the QC deletion problem, since gold examples preserve the class
distribution instead of skewing it.

⚠️ **Still open:** nothing verifies a generated row actually belongs to its assigned label. One
surviving row from the run reads `label=recipe, text="what is the best way to make a reservation
for a table at red robin"` — a reservation query stored as a recipe example. A verifier (e.g.
round-tripping the generated text through the reference model and keeping only rows it agrees with)
is the real fix and is not implemented.

---

## 10. "How much synthetic data does the intervention create every time?"

Two different paths, and they are now separately labelled in the log:

| Path | Amount | Bounds |
|---|---|---|
| **Plan `synthesize` strategy** | `plan["synth_rows"]`, chosen by the orchestrator | clamped to **100–500** by the plan validator |
| **Synth-fill (top-up)** | `target_rows - len(dataset)` — whatever the deficit is | unbounded by the plan; in iteration 1 that was **1,549 rows** |

In this run: one synth-fill of 1,549 rows, then later fills of 18 and 22 rows, plus targeted
`synthesize` batches. Note the deficit-driven fill can be far larger than any plan-authored batch.

---

## 11. "How does the pipeline even use the hf:DeepPavlov/clinc150/train dataset? the labels are only 0 1 and 2, isn't that just putting noise in the training?"

**Yes, it was noise, and you are right to flag it.** Those 164 rows went into the curriculum with
labels that match no eval label, so at best they were dead weight and at worst they taught the
model to emit `"0"`. The final `dataset_v2.jsonl` still contained **160** such rows.

It got there because the orchestrator chose `acquire`, and the mining ladder found
`DeepPavlov/clinc150` via paid discovery as a "novel source" — the novelty test only compares row
*text*, never labels, so a source with unusable labels passed. Now fixed per Q4: label-space
validation rejects it outright.

---

## 12. "the orchestrator isn't getting a diverse enough context and is overweighting these confusion pairs ... give me what it currently gets as context as a structure and an example, and suggest some solutions"

**Your read is correct, and the numbers support it.** `Hard-bucket accuracy is 0.683` appears
**163 times** in the run log. The test agent *did* compute fresh numbers every eval (`hard=`
values across the run include 0.683, 0.493, 0.549, 0.641, 0.761) — but the orchestrator's own past
hypotheses are replayed verbatim into every later prompt, so one early figure is restated dozens of
times against a single current one. Logged as **B224**.

**Current context structure** (per turn, `_llm_iterate`):

```
[system] _ITERATE_SYSTEM — fixed rules: choose exactly one intervention, the five tunable
         hyperparameters, the eval firewall, score-band guidance
[user]
  ## Task / model / tier
  ## Training trajectory so far        <- every past iteration, WITH its full hypothesis text
  ## Current iteration summary          <- current score, per-class, by-difficulty
  ## Test-data agent report             <- diagnosis + suggested_intervention + confusion pairs
  ## Hardware constraints               <- Memory: NoneMB ... PASS  (mostly empty, see below)
  ## Already-tried hyperparameter configs
  ## Data-rebuild plan notes
  ## Source novelty and prior plan yield
  ## Remaining budget / stop threshold / recent chronological gain
```

Real excerpt from the run:

```
- Recent chronological gain (last 2 evals, improvement to trigger escalation = 0.02): 0.0472
- Stop threshold: 0.8891 (initial floor: 0.889)
- Iter 4: f(π)=0.8294, band=0.80-0.95, intervention=hyperparameter, model=Qwen/Qwen3-0.6B —
  Hard-bucket accuracy is 0.683 (n=142), far below the 0.8891 stop threshold. Dominant confusion pairs...
- Iter 6: f(π)=0.8055, ... Hard-bucket accuracy is 0.683 (n=142), well below ...
- Iter 7: f(π)=0.8530, ... Hard-bucket accuracy is 0.683 (n=142), far below ...
```

Three of those lines restate the same stale number.

**Suggested solutions**, in the order I would try them:

1. **Stop replaying hypothesis prose.** In the trajectory, keep `iter / score / intervention /
   sub-strategy` and drop the free-text. The rationale for iteration 4 has no predictive value at
   iteration 19, and it is what carries the stale numbers.
2. **One current-metrics block, clearly marked.** Present per-difficulty and confusion data
   **once**, labelled "CURRENT EVAL", so there is exactly one live copy and no ambiguity about which
   number is fresh.
3. **Show deltas, not absolutes.** "hard bucket 0.683 → 0.761 (+0.078 over 5 iterations)" tells the
   orchestrator whether its interventions are working; a repeated absolute cannot.
4. **Surface intervention-effectiveness explicitly.** Give it "synthesize: tried 15×, mean Δ =
   −0.04" — it chose `synthesize` 15 times while its own test agent kept suggesting
   `hyperparameter`, and nothing in the context told it that synthesize was not paying off.
5. **Fix the empty hardware block.** It currently reads `Memory: NoneMB vs M_max=10240MB — PASS`,
   which is noise that dilutes real signal.

I have **not** implemented these — they change model input and therefore run behaviour, so they
deserve a deliberate decision rather than being folded into a logging cleanup.

---

## 13. "I want the pipeline to escalate if after 15 evals the model has not improved more than 2 percent. I want the model to escalate after 30 evals no matter what if it hasn't yet hit the goal."

**Done, and it also fixed five long-failing tests.**

| Knob | Was | Now | Env override |
|---|---|---|---|
| `STAGNATION_WINDOW` | 20 (tests asserted 50) | **15** | `SLM_STAGNATION_WINDOW` |
| `STAGNATION_MIN_DELTA` | 0.02 | **0.02** | `SLM_STAGNATION_MIN_DELTA` |
| `MAX_STALL_EVALS` | 20 | **15** | `SLM_MAX_STALL_EVALS` |
| `MAX_EVALS_BEFORE_ESCALATION` | *(did not exist)* | **30** | `SLM_MAX_EVALS_BEFORE_ESCALATION` |

**The 30-eval ceiling is deliberately built differently.** The stagnation window counts entries in
`state["scores"]`, and `rollback.py:40` **pops the regressing score** — so in a run where most
iterations regress, that list never grows and the window can never fire. In this run it sat at
**2 entries** for all 20 iterations while reporting a "healthy" 0.0472 gain. The new ceiling counts
`state["iteration"]` — evals actually performed — so rollback cannot hide from it.

No slurm script sets any of these, so the defaults now apply everywhere. Tests updated to the real
policy, plus two new cases (fires at 30 with a deliberately tiny score history; does not fire at
29). Documented as **B226**.

---

## Test status

**826 passed, 2 failed.** Down from 7 pre-existing failures — the escalation alignment fixed the
five drifted `test_iterate_stall.py` assertions. The 2 remaining are pre-existing and unrelated
(`test_iterate_prompt_receives_complete_curation_counts`,
`test_code_planner_and_model_choice_prompts_target_apps_introductory`).

---

## Part 2 — Full log audit (as requested)

Audit of `logs/slurm/slm-clinc150-cse-38155022.out` (23,900 lines), the run artifacts, and the vLLM
server log. Grouped by severity. **Zero Python tracebacks and zero fatal errors** — everything below
is a *quiet* problem.

### Definite problems

1. **Every mined row carried an unusable label.** 164 rows in v1, **160 in the final v2**, all
   labelled `"0"/"1"/"2"`. → B222, **fixed**.
2. **The winning dataset contained no synthetic data at all.** Final `dataset_v2.jsonl`:
   3,249 `train_anchor` + 160 `mined_real` + 17 untagged, **0 synthetic** — despite
   `data_sources.json` recording 790 synthesized rows. Iteration 20 was a resample-only rebuild.
   The headline 0.8971 therefore cannot be attributed to synthesis. → **B225, open**.
3. **Intermediate datasets are unrecoverable.** The same `dataset_v2.jsonl` path was rewritten 17+
   times; only the first and last states survive. → **B225, open**.
4. **Curriculum never reached its floor.** 3,461 then 3,426 rows against a 5,000 floor (−31%),
   across six synth-fill attempts. → root cause B221, **fixed**; the missing post-QC refill remains.
5. **The orchestrator ran on a frozen diagnosis.** `Hard-bucket accuracy is 0.683` ×163 while the
   true value moved 0.683 → 0.493 → 0.549 → 0.761. → **B224, open**.

### Suspicious / worth review

6. **`SLM_REQUIRE_SYNTH=1` did not require synth.** Nine preflight failures logged
   `synthesis will be skipped (gold-only)` before the server came up on attempt 10. It recovered,
   but the "required" flag did not enforce anything — if the server had never started, the run
   would have proceeded gold-only under a flag saying it must not.
7. **Six orchestrator decisions discarded**, all the same retired-hyperparameter `ValueError`,
   none reasked. → addressed in Q7.
8. **Hyperparameter search effectively stopped after iteration 2.** All 20 iterations used
   `r=32 a=64 drop=0 wd=0.01 lr=2e-04 ep=3`; only iteration 1 differed. 17 of 20 DAG nodes pruned.
9. **`Unsloth should be imported before transformers`** ×20 — may cost training performance/memory.
10. **Paid-round counter mislabels.** The second acquire also logs `paid-round-1`, and a later
    prompt still reported `run_paid_rounds_spent: 1` after two rounds had been spent.
11. **`timings.json` reports 60 train calls** against 20 `op=train` starts in the log — likely
    sub-phase double counting, but the numbers disagree.
12. **Confusion counts disagree with failure counts** in the same prompt
    (`alarm→__EXTRACTION_FAILED__: 6 failures` vs `count=3` for the same pair).
13. **Empty hardware metrics always PASS** (`Memory: NoneMB ... PASS`).
14. **KV cache headroom was tight** — 2.43 GiB, max concurrency 10.91× against `max_num_seqs=16`.
    No failures occurred, but there was little margin.

### Benign

15. 14,079 vLLM requests, **all HTTP 200**; no OOM, no preemption, no truncation.
16. `enforce_eager` disabling CUDA graphs, GPFS auto-prefetch disabled, Triton JIT latency spikes —
    all expected.
17. `peak_allocated_mib: 0.0` on eval workers — correct, llama.cpp scores on CPU.
18. `trust_remote_code` warnings — already removed yesterday.

### Overall read

The run converged, but **the mechanism it converged by is not the one the design intends**.
Synthesis produced almost nothing that survived to training, the winning dataset was essentially
the raw benchmark re-sampled, ~5% of training rows were mislabelled junk, and the orchestrator
spent 15 of 19 decisions on an intervention its own evidence never showed working. The +0.2996
jump at iteration 20 on a frozen configuration is most consistent with a favourable resample draw.
**I would not treat 0.8971 as reproducible** until a re-run with the B221/B222 fixes confirms it.

---

## Recommended next steps

1. **Re-run CLINC150** with the B221/B222 fixes and the gold-example synthesis, and compare both
   the score *and* the final dataset composition. That is the real test of whether synthesis helps.
2. **Fix B225** — version datasets per rebuild instead of overwriting `dataset_v2.jsonl`.
3. **Decide on the B224 context changes** (§12) — the highest-leverage remaining item, since it
   governs every intervention choice.
4. **Add a post-QC refill** so `target_rows` is enforced after quality control, not before.
5. **Add a synthetic-label verifier** so generated rows are checked against their assigned label.
