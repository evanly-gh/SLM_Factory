# CLINC150 run review — questions, answers, and log/graphics changes (2026-08-04)

Reference run: **`slm-clinc150-cse-38155022`** — converged at macro-F1 **0.8971** vs threshold
**0.8891** on `Qwen/Qwen3-0.6B @ Q4_K_M`, 20 iterations, 2h26m, $1.27.
Log: `logs/slurm/slm-clinc150-cse-38155022.out`.

Every question below is quoted verbatim, then answered with code references.

---

## Part 1 — Questions

### Q1. "Does the orchestrator actually have 1m token context window and how do you know? how much more does it cost than the 300k context window? is sonnet 5 available? how much does it cost in comparison?"

**Yes, but the mechanism you configured is now a no-op — you'd have 1M either way.**

`tests/pipeline/_l40s_task_body.sh` exports `SLM_ORCHESTRATOR_1M=1`. That flows into
`config/config.py:82-95`:

```python
ORCHESTRATOR_1M = os.environ.get("SLM_ORCHESTRATOR_1M", "0") == "1"
ANTHROPIC_BETAS = [... "context-1m-2025-08-07" ...]

def orchestrator_client_kwargs() -> dict:
    if ORCHESTRATOR_1M and ANTHROPIC_BETAS and not CHEAP_MODE:
        return {"default_headers": {"anthropic-beta": ",".join(ANTHROPIC_BETAS)}}
```

So the run does send `anthropic-beta: context-1m-2025-08-07`. **However**, on 2026-03-13 Anthropic
made the 1M window generally available for Claude 4.6-and-later at standard pricing, and states
the beta header is now **ignored**. So the header is vestigial — harmless, but it is no longer
what grants the window.

**Cost:** there is no longer a long-context premium. The old surcharge (2x input / 1.5x output
above 200K tokens) was removed in the same change. A 900K-token request bills at the same
per-token rate as a 9K one. Note your question said "300k" — the tier boundary was **200K**, and
it no longer exists for these models.

**Sonnet 5 is available, and is currently cheaper than what you're running:**

| Model | Input /MTok | Output /MTok | Context |
|---|---|---|---|
| Claude Sonnet 4.6 (what this run used) | $3.00 | $15.00 | 1M |
| **Claude Sonnet 5** (through 2026-08-31, introductory) | **$2.00** | **$10.00** | 1M |
| Claude Sonnet 5 (from 2026-09-01) | $3.00 | $15.00 | 1M |
| Claude Opus 4.6 | $5.00 | $25.00 | 1M |

This run used 324,233 input / 19,153 output tokens across 22 calls = **$1.26**, which matches
Sonnet 4.6 rates exactly. The same run on Sonnet 5 introductory pricing would be
`324233/1e6*2 + 19153/1e6*10` = **$0.84**, a ~33% saving — but only until **2026-08-31**, after
which Sonnet 5 costs the same as 4.6. Switch with
`SLM_ORCHESTRATOR_MODEL=<sonnet-5-model-id>`; `config/config.py:48` is the single point of change.

*(Caveat: I verified pricing and GA status from Anthropic's public pricing page. I have not
verified the exact Sonnet 5 API model string, so confirm that before switching.)*

---

### Q2. "In the hardware research portion, is an API ever called or does it just pull everything from a local database? or how does it get the metrics for the phone?"

**Both — a local CSV first, then always one Claude call.** `agent/nodes/cold_start/hardware_research.py:191-207`
documents the ladder: `local DB → Exa fallback → Claude LLM resolve`.

1. **Local DB** (`_lookup_local_db`, line 89) fuzzy-matches `data/devices.csv` (a Kaggle phone-spec
   dump). Your log shows `[hw] Local DB matched 7 device(s), using top 3` and `(source: local_db)`.
2. **Exa fallback** (`_exa_snippets`, line 172) only runs if the device is *not* in the CSV. It did
   **not** run here — the 2 Exa searches in the cost summary were for dataset discovery.
3. **Claude resolve** always runs — it turns the raw spec snippets into the structured constraint
   set. That's the `stage=hardware_research` cost event ($0.0059).

So the S24 Ultra numbers are CSV specs interpreted by Claude, not invented. Worth knowing: this
step is **not deterministic**. Run 38084400 chose a 204,800 MB storage budget; this run chose
153,600 MB ("~60% of 256 GB") from the identical input.

---

### Q3. "How did this run decide the accuracy goal? did it actually use the unfinetuned baseline from qwen 3.6? How did it determine the data targets? was it LLM based? if so where is the reasoning in the logs?"

**Accuracy goal — yes, genuinely measured from unfine-tuned Qwen3.6-35B.**

Two phases. `task_analysis` parks the goal as unreachable so nothing converges early
(`task_analysis.py:109-126`), because the eval set doesn't exist yet. Then `eval_setup`
(`_calibrate_qwen_goal_if_pending`, `eval_setup.py:117-168`) calls
`measure_endpoint_baseline(...)`, which scores the **hosted reference model zero-shot** on the
frozen 800-row eval set using the same scorer as the fine-tuned model, at temperature 0.

The transform is `threshold_from_endpoint_baseline` (`agent/threshold.py:43-65`):
`min(0.99, max(measured, floor=0.80))`, rounded to 4 dp. **There is no threshold-raising logic.**

On the `0.8843` vs `0.8891` discrepancy you spotted: those are **two different runs**. `0.8843` is
from the earlier failed job 38154619; this run measured `0.8891`. Same procedure, ~4 rows out of
800 different — run-to-run nondeterminism in the vLLM server, not a code path.
`Stop threshold: 0.8891 (initial floor: 0.889)` is just `:.3f` display rounding in the prompt.

**Data targets — NOT LLM-based on this run.** Because `SLM_BENCHMARK_TASK=clinc150` takes the
non-autonomous path, `task_planner.plan_task()` never runs. `_apply_data_targets`
(`task_analysis.py:45-78`) falls back to `DATASET_SIZE_BY_TYPE["classification"] = 150`, which is
then clamped **up** to `CURRICULUM_SIZE_FLOOR = 5000`. Eval likewise clamps to `EVAL_SET_SIZE = 800`.

That's why the log shows only the clamped line. On an autonomous run you would also see
`[planner] data targets (pre-clamp): ...` and `[planner] rationale: ...`
(`task_planner.py:279-281`). Their absence is the tell that the planner was skipped. Override with
`SLM_CURRICULUM_FLOOR` / `SLM_EVAL_SET_SIZE`.

---

### Q4. "(lines 868-875) What is it doing here? why is it looking at all the local databases?"

That's `load_local_dataset` (`web_acquire.py:1463-1478`), the **offline rung of the acquisition
ladder**. When the `acquire` strategy runs, it tries local bundles → curated HF benchmark → paid
Exa discovery, in that order, and only pays when the free options fail.

It walks *every* directory under `data/local/` and scores each with `_local_manifest_match`
(`web_acquire.py:1396-1444`). The first gate is `manifest["task_type"] != task_type → reject`,
which is why `gsm8k` (math), `mbpp`/`apps` (code) and `samsum` (generation) are rejected instantly
for a classification run. Scanning them is expected — it's a directory listing, not a load.

The message is arguably misleading though: it says "rejected: no explicit benchmark or strong
task+label+schema match" for what is usually a plain task-type mismatch.

---

### Q5. "(lines 899-903) At this part here after Exa finds HF datasets, why is the Qwen3.6 model going again? what is it synthesizing? If its the rest of the data there should be a message like Synthesizing rest of the data..."

Your instinct was right — it *is* topping the dataset back up, and there was no message saying so.

Two different things call the same generator and were indistinguishable in the log:
- `_synth_fill_to_target` (`curate.py:308-376`) — padding the dataset up to `target_rows`.
- `_synthesize_positive_rows` (`curate.py:247-305`) — the orchestrator's `synthesize` strategy.

**Changed.** Both now announce themselves:

```
  ▶ SYNTH-FILL (top-up to target): have 3413 row(s), need 1587 more to reach target 5000
  ▶ TARGETED SYNTHESIS (plan strategy=synthesize): generating from 200 anchor(s) (plan synth_rows=200)
```

---

### Q6. "(lines 2451-2453) instead of showing like 1000 of these rows can you instead put like a progress bar"

**Changed.** Two parts:

1. Local (self-hosted vLLM) cost events are **no longer echoed to the console**. They were always
   $0 and this run emitted **4,203** of them. They are still written to `cost-events.jsonl` and
   still counted in the end-of-run total, so nothing is lost for auditing. Paid providers always
   print, and local *failures* still print. Restore with `SLM_LOG_LOCAL_COST_EVENTS=1`.
2. A `_progress_map` helper (`curriculum.py`) now reports ~10 progress lines per synthesis batch,
   covering both hard-negative generation and CoT annotation:

```
      [synth] hard negatives (classification, 151 labels): 620/1549 (40%)
```

---

### Q7. "synth_fill=2, does this mean that the firewall removed two synthetically generated rows? why? were they too similar to the eval set rows? make sure to output the scripts justification for why a row was blocked"

**Yes — exactly two synthetic rows were blocked at the synth-fill checkpoint.** `synth_fill` is a
*layer name*, not a count of fill operations; the number is that layer's firewall drop count
(`curate.py:_exclude_eval_rows`, tallied per layer). The other layers in that line
(`train_anchor`, `mined`, `persistent_merge`, `final`) dropped 0.

"Too similar" is stricter than that: the test is **exact equality of normalized text**
(NFKC + casefold + whitespace collapse), not fuzzy similarity. A synthetic row was blocked because
the generator happened to reproduce a held-out eval utterance verbatim.

**Changed.** The firewall now logs a per-row justification (bounded to 20 rows per layer via
`SLM_FIREWALL_LOG_LIMIT`, and hashed so eval content never lands in the log):

```
    [firewall:synth_fill] BLOCKED row (provenance=synthetic_fill, label=recipe, len=42 chars,
    text_sha8=9f3c1a02): normalized text exactly matches a held-out eval row, so training on it
    would leak the eval set
```

---

### Q8. "(lines 2457-2459) What is the unknown category?? what does mined_real mean?"

- **`train_anchor` (3,249)** — rows re-drawn from the initial acquired pool (`_tag_train_rows`,
  `curate.py:195-206`). For this run that's the curated CLINC150 official train split.
- **`mined_real` (164)** — *new real* rows found during an `acquire` rebuild
  (`_tag_mined_rows`, `curate.py:209-217`). Here: `hf:DeepPavlov/clinc150/train`.
- **`unknown` (48)** — **a tagging gap, not a category.** Synth-fill rows got `_source="synth"` but
  never a `_provenance`, and the composition counter defaults missing tags to `"unknown"`.

**Changed.** Synth-fill rows are now tagged `synthetic_fill`, so the composition report
distinguishes them from the plan's `synthetic` rows and `unknown` should stay at 0.

⚠️ **Auditing this surfaced a real bug — see Part 3, B222.** Those 164 `mined_real` rows are
labelled `"0"`, `"1"`, `"2"` instead of intent names.

---

### Q9. "What is this stuff about batching ... Batch: micro=8 × grad_accum=1 → effective=8"

Three related numbers:
- **micro batch** — examples per forward/backward pass. Sets **peak activation VRAM**.
- **grad accumulation** — how many micro-batches accumulate before an optimizer step.
- **effective batch** = micro × accum — the number of examples per weight update. **This is the
  only one that changes what the model learns.**

`8 × 1 = 8` means it fits the whole batch in memory with no accumulation. The split is derived by
the trainer to fit VRAM and is deliberately **not** orchestrator-tunable — see Q14.

---

### Q10. "'__EXTRACTION_FAILED__' What is this in the test-data agent report? is this a bug?"

**Not a bug — an intentional sentinel.** `eval/scorers/classification.py:31-51` tries an exact
word-boundary match, then a substring match, and if neither finds a known label returns
`__EXTRACTION_FAILED__`. It has an explicit test.

It exists so an unparseable answer is *counted wrong loudly* rather than silently defaulting to
the majority class (the older behaviour, logged as B46). Seeing it in a confusion breakdown —
e.g. `account_blocked→EXTRACTION_FAILED (3)` — means the model emitted something outside the label
vocabulary. That's a **formatting** failure, distinct from picking the wrong intent, and it's
useful signal.

---

### Q11. "(lines 2754-2757) explain to me what these rows mean"

- **`Strategy composition`** — where the final rows came from, by sub-strategy:
  `mine_new_real_source` 164 (newly mined), `resample` 3,249 (re-drawn from the pool),
  `unattributed` 48 (the untagged synth-fill rows from Q8).
- **`Source novelty`** — the mining scorecard. `requested: 200` asked for 200 new rows;
  `candidate_rows: 4049` were fetched; `novel_rows: 200` were not already present;
  `novel_fraction: 0.0494` means only ~5% of what was fetched was new; `paid_rounds_used: 1`
  spent one paid discovery round of the 9-per-run budget; `status: novel` = it did find new material.
- **`Plan yield`** — what the rebuild produced end to end: `final_rows: 3461`,
  `novel_rows: 3460`, `novel_fraction: 1.0` (this is the first dataset, so essentially all of it
  is new relative to the previous version, of which there was none).

---

### Q12. "(lines 3621-3622) I'm pretty sure that here is where you realize you need to fill back to the ceiling"

Correct — that's synth-fill, and it's now explicitly labelled. See Q5. Note it fills to the plan's
`target_rows`, which is **not** a fixed ceiling: it was 5,000 on iteration 1 but the orchestrator
lowered it later (3,512 on one rebuild, 3,200 on another).

---

### Q13. "(lines 11185-11194) What is this and is it a problem?"

```
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset 'DeepPavlov/clinc150' isn't based on a loading script
```

**Not a problem — it's dead-argument noise.** `datasets` 4.x removed script-based loading, so the
`trust_remote_code=True` kwarg is ignored and warned about. The dataset loaded fine (164 rows came
back). It printed repeatedly because it fires on every load attempt.

**Changed.** Removed all 6 now-meaningless `trust_remote_code=True` arguments (4 in
`web_acquire.py`, 2 in `xlam_bfcl.py`). Behaviour is unchanged; the warnings are gone.

---

### Q14. "(lines 2855-2856) Why is there an error here? tell me what the problem is and if it's old stuff i got rid of just erase it from the logs and fix the error. explain to me what failed exactly"

**What failed:** the orchestrator returned a `hyperparameter` decision containing six knobs that
were deliberately retired. `_validate_decision_json` (`iterate.py:316-339`) rejected the whole
decision, and the pipeline fell back to the test-agent's suggestion — which also said
`hyperparameter`, so the *intervention type* survived and only the LLM's specific values were lost.
It happened **once in 22 orchestrator calls** and did not affect the outcome.

The five tunable knobs are `lora_rank`, `alpha_ratio`, `weight_decay`, `learning_rate`,
`nr_epochs`. The six rejected ones and why:

| Rejected field | Why |
|---|---|
| `lora_alpha` | superseded by `alpha_ratio` (alpha = rank x ratio) |
| `lora_dropout` | duplicates `weight_decay`, which produced the larger gain |
| `micro_batch_size`, `gradient_accumulation_steps`, `effective_batch_size` | batch shape is a VRAM-fitting decision the trainer makes better |
| `batch_size` | legacy alias |

The system prompt already states this correctly ("EXACTLY FIVE hyperparameters are tunable"), so
this was the model ignoring a clear instruction, not a prompt defect.

**Changed (your "old stuff, erase it"):** the iterate logger was still *printing* the retired
fields, so every successful hyperparameter decision logged `alpha=None dropout=None
micro_batch=None grad_accum=None effective_batch=None`. That dead logging is removed; it now
prints only the five real knobs.

**One open item I could not explain.** There is a JSON-only reask designed to catch exactly this
(`iterate.py:967-979`) — it should replay the sanitized error and ask the model to fix itself. It
**did not fire**: the cost ledger records 6 `iterate` events and **zero** `iterate_json_reask`.
I verified the validator does raise, the handler wraps it, and the error sanitizer handles this
message cleanly, so I cannot yet account for the skip. Low severity (one discarded decision), but
it is a real gap and I did not want to guess at a cause.

---

### Q15. "I thought the dataset size was floored at 5000 with synth fill to ensure it always stays at least that amount. Why did the pipeline end up training with only about 3500 rows?"

**Synth-fill did its job. Quality control then deleted almost all of it, and nothing refills after
QC.** From the log:

```
Synth-fill added 1549 row(s) toward 5000     ← reached ~4,962 pre-QC
Total examples : 3461                        ← after quality controls
Provenance : {'mined_real': 164, 'train_anchor': 3249, 'unknown': 48}
```

3,413 real rows + 1,549 synthetic ≈ 4,962, essentially at target. But only **48 of the 1,549
synthetic rows survived** — about 1,501 were dropped by `apply_quality_controls`, and
`_synth_fill_to_target` runs *before* QC with no second pass afterwards. So 5,000 is a
**pre-quality-control target, not a guaranteed floor**.

**Why QC ate them is B221** (Part 3): every hard negative was assigned the *same* target label, and
the label-balancing rule ("no label more than 3x the smallest") exists precisely to delete that
kind of pile-up. Fixing the targeting should let far more synthetic rows survive.

---

### Q16. "What were the prompts that the orchestrator gave to qwen3.6 to generate synthetic data? how did qwen 3.6 generate the data? show me a few examples of real rows and the synthetic rows and where to look over the data myself."

**Important framing:** the orchestrator (Claude) does **not** write these prompts. It only picks a
strategy and a row count; the prompt is a fixed template in `data/curriculum.py`. For
classification the template is (`curriculum.py`, hard-negative branch):

> You are generating a HARD NEGATIVE for a text classifier: a realistic example that superficially
> resembles the '{src_label}' class but genuinely belongs to the '{target_label}' class. The surface
> features should mislead toward '{src_label}' while the true meaning is unambiguously
> '{target_label}'.
>
> Reference '{src_label}' example:
> {ex['text']}
>
> Output ONLY the new example text for the '{target_label}' class — no preamble, no explanation, no
> quotation marks, no label prefix.

Sent to the local vLLM Qwen3.6 server as a single user message, `max_tokens=200`, temperature 1.0,
in "2-for-1" form (the anchor row and its synthetic counterpart are both kept).

Note `pattern_hint` — the slot meant to inject the observed failure mode — is **never passed by
`curate_node`**, so it is always empty in live runs. The generator therefore does not know which
confusion pairs it is supposed to attack, even though the orchestrator's hypothesis names them.

**Where to look yourself:**

```
logs/runs/slm-clinc150-cse-38155022/artifacts/dataset_v1.jsonl   # one JSON object per row
logs/runs/slm-clinc150-cse-38155022/artifacts/eval_set.json      # frozen held-out set
logs/runs/slm-clinc150-cse-38155022/data-curation.md             # per-iteration composition
```

Filter by `_provenance` (`train_anchor` / `mined_real` / `synthetic` / `synthetic_fill`) or
`_source` (`"synth"` = generated). Real vs synthetic from this run:

```jsonc
// REAL (train_anchor)
{"_provenance":"train_anchor","_source":"CLINC150 (clinc_oos/plus)","_strategy_origin":"resample",
 "label":"accept_reservations","text":"what locations of applebee's take reservations"}

// MINED (mined_real) — note the broken label, B222
{"_provenance":"mined_real","_source":"hf:DeepPavlov/clinc150/train","label":"0", ...}

// SYNTHETIC (synth-fill) — plausible
{"_source":"synth","label":"recipe","text":"can i substitute sour cream for buttermilk in this cake recipe"}

// SYNTHETIC — mislabelled; this is a reservation query stored as `recipe`
{"_source":"synth","label":"recipe","text":"what is the best way to make a reservation for a table at red robin"}
```

---

### Q17. "Why did the orchestrator pick data rebuild like everytime? is it bias? why did it never pick hyperparameter search except the time it errored?"

**It is not a coded bias.** There is no hardcoded preference in the normal path; the LLM chooses
freely. The fallbacks actually lean the *other* way — `apply_iteration_policy` returns
`hyperparameter` for scores in 0.80–0.95, which is where this run lived, and the test agent
repeatedly suggested `hyperparameter` too.

So the orchestrator overrode its own advice. Actual distribution of the 19 decisions:

| Intervention | Sub-strategy | Count |
|---|---|---|
| data_rebuild | **synthesize** | **15** |
| data_rebuild | acquire | 2 |
| data_rebuild | resample | 1 |
| hyperparameter | — | 1 (the one that errored) |

The most plausible reading from the prompt content: the test-agent report handed it very concrete,
actionable *data* evidence every single turn — the same confusion pairs
(`change_ai_name↔change_user_name`, `change_language→translate`, ...) and the same hard-bucket
number, 0.683, iteration after iteration. Named confusion pairs read as a data problem, so it kept
choosing to synthesize against them.

The bitter irony: because of **B221** those synthesize interventions were mostly deleted by quality
control before training, so the evidence it was reacting to never changed — the hard-bucket number
was **identical (0.683, n=142) across iterations 4–17**. It was stuck in a loop reacting to a
signal its own intervention could not move.

I'd treat this as evidence-driven behaviour on a broken feedback loop rather than model bias, and
re-evaluate after B221 is fixed.

---

### Q18. "explain to me generally why the model made gains on iterations 1 and 2 and 20, and why there wasn't an escalation. I thought i set the escalation limit to 15 turns without improvement??"

**The gains:**

- **Iteration 1 (0.0 → 0.8261).** Not a real "gain" — it's the first trained model, going from
  nothing to a fine-tuned 0.6B on ~3,461 rows covering all 151 intents. Against a 0.3152 zero-shot
  baseline, fine-tuning alone bought +0.51.
- **Iteration 2 (0.8261 → 0.8734, +0.0472).** The **only** hyperparameter change of the run:
  `lora_rank 16 → 32`. Doubling adapter capacity on a 151-class problem is exactly the right lever
  when easy/medium are strong and hard is weak, which is what the test agent had diagnosed. This
  is the clearest genuine improvement in the run.
- **Iterations 3–19 (all regressions).** Every one was `data_rebuild`, config frozen at
  `r=32 a=64`, all carry-forward-best. Scores swung 0.54–0.865 with no trend — that is dataset
  churn (re-drawing and re-synthesizing rows) around a fixed configuration, and the swings are
  large because the injected synthetic rows were noisy and label-imbalanced.
- **Iteration 20 (0.5975 → 0.8971).** A +0.30 jump in one step with the same hyperparameters.
  A swing that large from a data-only change on a frozen config is most consistent with a
  favourable dataset draw rather than a durable improvement — and it crossed 0.8891, so the run
  terminated immediately and never had to reproduce it. **I'd treat 0.8971 as a single lucky draw
  in a high-variance sequence, not a validated result.** Re-running with a different seed is the
  honest way to confirm it.

**Why no escalation — and the "15" you remember does not exist anywhere.**

Actual defaults (`iterate.py:653-695`): `STAGNATION_WINDOW=20`, `STAGNATION_MIN_DELTA=0.02`,
`MAX_STALL_EVALS=20`. Env overrides `SLM_STAGNATION_WINDOW`, `SLM_MAX_STALL_EVALS`. **The slurm
script sets none of them**, so defaults applied. There is no 15 in code, config, or the script.

There are two triggers, and both missed:

1. **Stagnation window — structurally cannot fire in a run like this.** It needs ≥20 retained
   scores, but `rollback.py:40` does `state["scores"].pop()` on every regression. Since iterations
   3–19 all regressed, `scores` stayed at **2 entries** `[0.826, 0.873]` for the whole run. Its
   gain, 0.0472, is above the 0.02 threshold, so it read as *healthy progress* for 17 straight
   iterations. That's why the prompt kept saying `Recent chronological gain (last 2 evals...)`.
   **This looks like a real design flaw:** rollback and stagnation detection disagree about what
   the score history means, and rollback wins.
2. **Stall backstop — came within 3.** `consecutive_no_improvement` is *not* popped by rollback, so
   it worked as intended and reached **17** (iterations 3–19) against a limit of **20**. Iteration
   20 improved and reset it.

If you want escalation after 15 stalls, set `SLM_MAX_STALL_EVALS=15`. Do **not** rely on
`SLM_STAGNATION_WINDOW` until the rollback interaction is fixed.

Related: 5 of the 7 long-standing test failures are `tests/nodes/test_iterate_stall.py` asserting
`STAGNATION_WINDOW == 50` ("raised to 50 in B161") against a code default of 20. So the intended
value has drifted from the actual one, and the tests have been failing ever since.

---

## Part 2 — Changes made

### Logging

| Change | Where |
|---|---|
| Local (`$0`) cost events no longer echoed to console — 4,203 lines removed from this run. Still in `cost-events.jsonl` and the final total. Local *failures* and all paid calls still print. Restore: `SLM_LOG_LOCAL_COST_EVENTS=1` | `agent/cost.py` |
| Synthesis + CoT annotation now emit ~10 progress lines instead of one line per row | `data/curriculum.py::_progress_map` |
| `▶ SYNTH-FILL (top-up to target)` vs `▶ TARGETED SYNTHESIS (plan strategy=synthesize)` | `agent/nodes/curate.py` |
| `DATA REBUILD` line now names what the sub-strategy actually does | `agent/nodes/curate.py` |
| Per-row eval-firewall justification, hashed + bounded (`SLM_FIREWALL_LOG_LIMIT`, default 20) | `agent/nodes/curate.py::_exclude_eval_rows` |
| Banner separating `BASELINE EVAL` from `ITERATION N EVAL` | `agent/nodes/evaluate.py` |
| `── STEP 1/2: QUANTIZE ──` / `── STEP 2/2: RUN EVAL ──` | `agent/nodes/evaluate.py` |
| Full orchestrator prompt logged **once**; later turns log only the changing user content. Restore: `SLM_LOG_FULL_ITERATE_PROMPT=1` | `agent/nodes/iterate.py` |
| Retired hyperparameter fields removed from the decision log (they always printed `None`) | `agent/nodes/iterate.py` |
| Dead `trust_remote_code=True` args removed (6 sites) | `web_acquire.py`, `xlam_bfcl.py` |
| Synth-fill rows tagged `synthetic_fill` instead of falling through to `unknown` | `agent/nodes/curate.py` |

### Graphics

- **`dataset_composition.png`** — removed the second right-hand y-axis. The stacked bars already
  sum to the total, so the twin axis both duplicated the information and drew it at a different
  scale. The total is now a dashed line on the **same** axis, where it visibly tracks the top of
  each stack.
- **`hypotheses.md`** — now opens with the untrained baseline and the best score achieved; adds a
  **sub-strategy** column; and the hypothesis column is **shifted to read forward**, so row N shows
  the score iteration N achieved *and the hypothesis it then formed to beat it*. Iteration 1 is no
  longer blank.

Regenerated for this run at `logs/graphics/slm-clinc150-cse-38155022-v2/` (the original is left
untouched for comparison).

### Not done

- **Per-iteration curation history in the prompt.** You asked that the `## Iteration N` block show
  only the current iteration. I did not change this: that block is assembled from
  `data-curation.md` and is also the orchestrator's only trajectory memory, so trimming it changes
  *model input*, not just logging. It needs a deliberate decision about how much history the
  orchestrator should see. Flagging rather than silently altering behaviour.

---

## Part 3 — New bugs found while answering

Both were discovered auditing `dataset_v1.jsonl`; both are written up in `docs/BUGS.md`.

- **B221 (fixed) — every classification hard negative targeted the same label.** `target_labels[0]`
  meant all synthesis landed on 1–2 of 151 classes (39 `recipe`, 9 `book_flight`). The
  label-balancing quality control then deleted ~1,501 of 1,549 generated rows. This is the direct
  cause of Q15's missing 1,500 rows and a large part of Q17's stuck feedback loop. Also fixed a
  determinism defect: `all_labels` came from an unordered `set`. **Still open:** nothing verifies a
  generated row actually belongs to its assigned label — one surviving row labels a restaurant
  reservation query as `recipe`.
- **B222 (open) — mined rows carry unmapped integer labels.** All 164 rows from
  `DeepPavlov/clinc150` were labelled `"0"`, `"1"`, `"2"` instead of intent names. The curated
  loader resolves ids through the HF `ClassLabel` names; the generic acquisition path has no
  equivalent step. ~5% of the training set is labelled with classes that exist in no label space.

---

## Suggested next steps

1. **Re-run CLINC150 with B221 fixed.** Most synthetic rows should now survive quality control, so
   the dataset should actually reach ~5,000 and the `synthesize` interventions should finally be
   able to move the hard bucket.
2. **Fix B222** before trusting any `acquire` result — validate mined labels against the
   established label space and reject or remap disjoint ones.
3. **Decide the escalation policy explicitly.** Set `SLM_MAX_STALL_EVALS=15` if that is what you
   want, and treat the rollback/stagnation-window interaction as a bug to fix rather than a knob.
4. **Consider Sonnet 5** before 2026-08-31 for ~33% lower orchestrator cost, after confirming the
   API model id.
5. **Pass `pattern_hint` into synthesis.** The orchestrator names the exact confusion pairs in its
   hypothesis every turn, and that information is currently thrown away before the generator sees it.
