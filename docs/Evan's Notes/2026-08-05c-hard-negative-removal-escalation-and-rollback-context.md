# Removing hard negatives, single-mechanism escalation, and rollback context (2026-08-05)

Fourth follow-up. Each question restated, then answered against the code.
**You corrected me on a serious mistake — see Q5 first.**

---

## ⚠️ Q5 (first, because you were right and I was wrong)

> "note that this is cheating: *[mine hard negatives from the model's own confusion matrix — take
> real eval-clean rows of class B that the model predicted as A]*. This is called training on the
> test set, and you should not do it."

**You are completely right and I was wrong to suggest it.** Those rows come from the held-out eval
set. Selecting the ones the model gets wrong and feeding them back as training data is textbook
test-set leakage — it would inflate the eval score without improving the model, and it would have
quietly invalidated every number the pipeline reports. The fact that they are "real and correctly
labelled" is irrelevant; the problem is *which* rows they are and *how they were chosen*. The whole
point of the four-layer eval firewall is to prevent exactly this, and I proposed driving a hole
through it. Thank you for catching it.

The recommendation is withdrawn entirely and is **not** implemented anywhere.

---

## Q1. "get rid of all the wiring you currently have for generating hard negative rows, only allow qwen 3.6 to generate gold rows. delete all the docs and code and remove it as an option."

**Done.**

| Removed | Where |
|---|---|
| `synthesize_hard_negatives` (235 lines: classification blend prompt, NER rewrite prompt, math/code guards, 2-for-1 pairing, Claude fallback) | `data/curriculum.py` |
| `_hard_negative_ratio` + `SLM_SYNTH_HARD_NEGATIVE_RATIO` | `data/curriculum.py` |
| `hard_negative_synthesis` cost stage and its Claude backend | `data/curriculum.py` |
| `tests/test_curriculum_hardneg.py` (12.6 KB) | deleted |
| Live doc sections | `PIPELINE.md`, `PROMPTS.md`, `DATA_CURATION_AND_CAPS.md` |

`synthesize_examples` now routes classification/NER to `_synthesize_new_gold` only. Confirmed:
`hasattr(curriculum, "synthesize_hard_negatives") == False`.

**Not touched:** dated records under `docs/Evan's Notes/`, `docs/superpowers/`, `PAPER.md`, and
`BUGS.md`. Those describe what the pipeline *was* and why it changed; rewriting them would destroy
the reasoning trail. `PROMPTS.md` §2.2/§2.3 are kept but now carry a banner saying the prompts no
longer exist and are retained only as a record of why the approach failed.

---

## Q2. "Delete all the stuff in the docs and code about the old small numbers ... a deterministic formula based on the parameter count and zero-shot baseline sounds good ... bigger model should decrease the target ... never cap other than the hard cap of 25000 ... recomputed every time it does evaluations on a model from a new tier ... I want the pipeline to always regress to the next lowest tier unless it is already at the lowest tier, it has already tried the lower tier, or it didn't meet the goal."

**IMPLEMENTED 2026-08-05** (this section replaces the earlier "needs your answers" version).

You confirmed: always regress a tier on meeting the goal unless already at the lowest tier or the
tier below was already tried; if the downward tier fails, keep the last passing tier.

**Sizing — `agent/data_sizing.py`.** Deterministic, no LLM:

```
novelty     = 1 - zero_shot_baseline          # measured, not guessed
size_factor = clamp(1e9 / n_params, 0.5, 2.0) # smaller model -> more data
target      = clamp(5000 x (0.5 + novelty) x size_factor, 5000, 25000)
```

| Case | Target |
|---|---|
| Qwen3-0.6B on CLINC150 (baseline 0.3152) | **9,873** |
| Same task on a 4B model | **5,000** |
| Easy task (baseline 0.85), 0.6B | 5,417 |
| No baseline yet (first sizing), 0.6B | 8,333 |

Recomputed whenever the selected model changes. I hooked it in `curate_node` rather than at the
ten places that assign `selected_model`, so initial selection, escalation and downward regression
are all covered by one call site. `SLM_CURRICULUM_SIZE` overrides.

**`DATASET_SIZE_BY_TYPE` deleted.** It was worse than dead: every value (classification 150, NER
200, generation 600 ...) sat below the floor and was clamped away on every path, so it *looked*
like per-task sizing was happening when the effective target was always the floor.

**Tier regression — two gates removed.**

1. `probes_down = strategy in ("interpolation", "orchestrator_choice")` meant a `smallest_first`
   run could never regress *even after escalating*, so it could not come back down when the
   smaller model might now succeed on an improved dataset.
2. `_should_reexplore_downward` (61 lines) spent an orchestrator API call deciding whether trying
   a smaller model was "worth it" — i.e. it could decline the one thing the run exists to
   determine. Deleted.

Regression is now unconditional; the only stopping conditions are structural (no lower tier, or
all lower tiers tried). Under `smallest_first` the run starts at tier 0 so it still never
regresses — a consequence of where it starts, not a rule about the strategy, which is what you
predicted.

**One correction to your premise, for the record:** you said "it will regress if it meets the
accuracy goal". Before this change, meeting the goal **terminated** the run under
`smallest_first` — that is exactly how the CLINC150 run ended at 0.8971. The behaviour you
described is now true; it was not before.

Tests: 8 in `tests/test_data_sizing.py`, 28 in `tests/nodes/test_downward_probe.py`. BUGS B234/B236.

## Q3. "I want to know why in the last run an error ever occurred at all. I thought the orchestrator was very explicitly instructed that it cannot choose both data_rebuild and hyperparameter search so theoretically this error should never happen?"

**Two different rules are being conflated — the error was not the mutual-exclusion one.**

The mutual-exclusion rule *is* stated very firmly, and the orchestrator **never violated it**. From
the system prompt:

```
=============================== CHOOSE EXACTLY ONE ===============================
  A) "intervention": "data_rebuild"  -> REQUIRED: "data_rebuild"  FORBIDDEN: "hyperparams"
  B) "intervention": "hyperparameter" -> REQUIRED: "hyperparams"  FORBIDDEN: "data_rebuild"
WHY THIS IS STRICT: exactly one thing may change per iteration...
```

The actual error was different: it chose `hyperparameter` correctly, but **inside** the
`hyperparams` object it included six **retired** knobs — `lora_alpha`, `lora_dropout`,
`micro_batch_size`, `gradient_accumulation_steps`, `effective_batch_size`, `batch_size`. That is a
field-level rule, not the intervention-level one.

**Why it still happened 6 times in 19.** The prompt states the rule positively ("EXACTLY FIVE
hyperparameters are tunable") *and* the JSON example shows only the five. But it also **names the
forbidden fields** in prose:

> - Batch shape (micro_batch_size / gradient_accumulation_steps / effective_batch_size) ...
> - lora_dropout is NOT tunable. Regularize with weight_decay instead.

Naming them primes them. Worse, `lora_alpha` is the *universal* LoRA parameter every model has seen
thousands of times in training data, while `alpha_ratio` is this repo's own invention. Asked to
write a LoRA config, the model reaches for the familiar name. The failing decision's own rationale
even reasoned about batch shape ("micro batch=8 controls peak activation memory") — it was thinking
about the exact fields the prompt told it not to send.

**So: the instruction is clear, and the model still ignored it ~30% of the time.** That is why I did
not stop at "the prompt says not to" — the salvage path now strips those keys and keeps the
decision. A prompt-side improvement worth trying separately: state the allowed set only, and say
"any field not in this list is rejected", without ever naming the forbidden ones.

---

## Q4 (your Q5 body). "implement those interventions with your suggestions ... just tell me details, like what proportions you're setting it at, how you're doing the per-pair effectiveness stuff"

**IMPLEMENTED 2026-08-05** (replaces the earlier "blocked" version). You said the guard could go
and the design was good, so both are done.

**The `surgical` naming guard is removed.** For the record on why it existed:
`test_data_rebuild_routing.py` forbade the token because it named a ROUTE deleted in the
2026-07-31 redesign. The guard now documents that this is a *synthesize sub-strategy under the
data_rebuild plan*, not a resurrection of that route — so the distinction survives in the
codebase rather than only in this conversation.

**Proportions.**

| Knob | Default | Meaning |
|---|---|---|
| `SLM_SURGICAL_SYNTH_SHARE` | **0.20** | share of the plan's `synth_rows` spent surgically; the rest is fill |
| `SLM_SURGICAL_MAX_PAIRS` | **5** | most confusion pairs targeted per rebuild |
| per-pair min / max | **10 / 100** | clamp on any single pair's budget |

Per-pair budget is **proportional to confusion count**, not split evenly:

```
rows_for_pair = clamp(round(TOTAL x count_pair / sum(counts)), 10, 100)
```

so a pair confused 8x gets materially more than one confused 2x. Anchors are drawn from the
**gold** class — the one the model should have predicted — and produce in-class GOLD rows. With
hard negatives removed, "surgical" necessarily means *reinforce the confused class*, not
*generate a contrastive negative*.

**Per-pair effectiveness.** `state["surgical_pair_history"]` records the confusion count at the
moment each pair was targeted. On a later rebuild, if a pair is up for targeting again and its
count has **not fallen**, it is skipped:

```
[surgical] SKIP change_ai_name->change_user_name: targeted before at count=7, still 7 —
EXHAUSTED, spending elsewhere
```

That is the specific guard against the loop the CLINC150 run was stuck in: 15 `synthesize`
interventions against the same unchanging confusion pairs.

Tests: 5 in `tests/nodes/test_surgical_synthesis.py` covering proportionality, exhaustion,
re-targeting after a pair improves, the no-pairs case, and that anchors come from the gold side.
BUGS B235.

## Q6. "Implement the verifier pass by the teacher model over the rows it generated ... add to the logs how many rows it validated, and if it eliminated any rows have it log qwen's reasoning as well."

**Done** — `verify_generated_labels` in `data/curriculum.py`, on by default
(`SLM_VERIFY_SYNTH=0` disables).

Each generated row goes back to the teacher as a *classification* question, not a generation one:

> Utterance: {text}
> Proposed label: {label}
> Does this utterance genuinely belong to the '{label}' class? Answer strictly as JSON:
> `{"valid": true|false, "reason": "<max 15 words>"}`. Answer false if the utterance actually
> belongs to a different class, is incoherent, or mixes two intents.

Logging, exactly as you asked:

```
      [synth] label verification: 120/120 (100%)
      [verify] teacher validated 103/120 generated row(s); rejected 17
        REJECTED [recipe] 'how do i make a reservation at the chef's table' — teacher: this is a restaurant reservation request
        REJECTED [recipe] 'set the oven alarm for 30 minutes' — teacher: this is an alarm request
        ... and 5 more rejected
```

Capped at 10 quoted rejections (`SLM_VERIFY_LOG_LIMIT`) so a bad batch cannot bury the log.

**One deliberate safety property:** if verification fails for any *mechanical* reason — endpoint
error, unparseable reply — the row is **kept**, not dropped. A broken verifier must never be able to
silently empty a dataset.

I share your scepticism about self-checking, which is why I want you to see the numbers. But note
the two tasks are not equally hard: *generating* "text that looks like A but is B" is open-ended,
whereas *classifying* an utterance is the task the reference model already scores 0.889 on. On the
17 bad rows from the last run, I would expect it to catch most of the obvious ones.

---

## Q7. "what I would actually like is for the orchestrator to see the whole context of every previous run ... whenever the model fails and regresses it should note the reason why and remember it, and whenever it succeeds and improves it should remember the reason as well ... in terms of logging I'd like the whole details of the most recent iteration, along with the general summary of the training trajectory including all the failed runs since the last improvement."

### What Q7 is actually about

Everything else we have fixed is about *what the orchestrator is told about right now*. Q7 is
about its **memory** — what it knows about everything it has already tried, and whether it can
tell what worked from what did not. It is the difference between an agent that learns across
iterations and one that re-litigates the same decision twenty times.

### Current state (verified against the real run, not assumed)

The trajectory comes from `data-curation.md`, read fresh each turn and passed through
`agent/context_manager.compact_trajectory` when it exceeds ~8,000 tokens. In the CLINC150 run
compaction triggered on **16 of 19** turns. It keeps the **last 3 iterations in full detail** and
compresses everything older to one line each.

Three things already work:

- **Failed iterations are not lost.** The curation log is append-only on disk, so rolled-back
  iterations still appear in the history even though `state["scores"]` popped them.
- **The most recent iterations are shown in full**, which is roughly the first half of what you
  asked for.
- **`last_failed_attempt`** (added in Q9 above) gives an explicit, prominent memo of the attempt
  that was just discarded.

But the compacted one-liners are close to useless, and here is a real one from the run:

```
## Compacted history (older iterations)
- Iter 1: f(pi)=0.8261, band=0.80-0.95, intervention=hyperparameter, model=Qwen/Qwen3-0.6B — ### Hardware profile (Phase 1: theoretical)
- Iter 4: f(pi)=0.8294, band=0.80-0.95, intervention=hyperparameter, model=Qwen/Qwen3-0.6B — Hard-bucket accuracy is 0.683 (n=142), far below the 0.8891 stop threshold. Dominant confusion pairs
- Iter 6: f(pi)=0.8055, band=0.80-0.95, intervention=hyperparameter, model=Qwen/Qwen3-0.6B — Hard-bucket accuracy is 0.683 (n=142), well below the 0.8891 stop threshold. The dominant failure mo
```

Four separate defects visible in three lines:

1. **The `intervention=` field was WRONG** — and this is the one I would call serious. The DAG
   (authoritative) records iterations 4-8 as `data_rebuild/synthesize`. The trajectory told the
   orchestrator they were `hyperparameter`. **Its memory of what it had already tried was
   factually incorrect**, which makes "propose something you have not tried" impossible to
   satisfy. Root cause: `evaluate.py` wrote `next_intervention=policy["intervention"]` — the
   *score-band policy's guess* — into the curation log, while the DAG correctly used
   `state["last_intervention"]`. **Fixed today (B237)**; it now records the executed intervention.
2. **Each line is dominated by stale hypothesis prose**, truncated mid-sentence at ~100 chars.
   This is the vector for the `hard=0.683` repetition — the number appears 163 times in the run
   log almost entirely through this replay path.
3. **Iteration 1's summary is garbage** — `— ### Hardware profile (Phase 1: theoretical)`. The
   extractor grabbed a markdown header instead of the hypothesis. Still open.
4. **Nothing distinguishes success from failure.** Iterations 2 (a new best) and 6 (a rollback)
   are rendered identically. There is no marker, no delta, and no grouping of "failures since the
   last improvement" — the exact framing you asked for.

### What I am suggesting

Replace the raw curation-log dump with a purpose-built memory block. Same information, organised
around the question the orchestrator is actually being asked. Concretely, instead of the above:

```
## MOST RECENT ITERATION (full detail)
  iteration 19 | data_rebuild/synthesize | 0.5975 (best 0.8734, delta -0.2759) | ROLLED BACK
  dataset v2, 3426 rows (2 dropped by eval firewall, 1688 by QC)
  by bucket: easy=0.918(n=214)  medium=0.702(n=444)  hard=0.197(n=142)
  top confusions: change_ai_name->change_user_name (7), change_language->translate (4)

## WHAT WORKED (kept improvements)
  iter 1  data_rebuild/acquire     0.0000 -> 0.8261  (+0.8261)   first trained model
  iter 2  hyperparameter           0.8261 -> 0.8734  (+0.0473)   lora_rank 16 -> 32

## FAILED SINCE THE LAST IMPROVEMENT (17 attempts, none kept)
  data_rebuild/synthesize  x15   deltas -0.28 .. -0.01   mean -0.041
  data_rebuild/acquire     x2    deltas -0.12, -0.03     mean -0.075
  -> nothing tried since iteration 2 has beaten it. Consider a different intervention TYPE.

## SURGICAL SPEND (per confusion pair)
  change_ai_name<->change_user_name  targeted 2x, count 7 -> 7   EXHAUSTED
  change_language->translate         targeted 1x, count 4 -> 2   improving
```

Four deliberate changes from today's format:

- **Drop the hypothesis prose from replayed history.** The rationale for iteration 4 has no
  predictive value at iteration 19, and it is what carries stale numbers forward. Keep prose only
  for the most recent iteration and the `last_failed_attempt` memo.
- **Mark outcomes explicitly.** `KEPT` vs `ROLLED BACK`, with the delta. This is your
  "remember why it succeeded / why it failed" requirement, and it costs almost no tokens.
- **Group failures since the last improvement**, aggregated by intervention and sub-strategy
  rather than listed one by one. Seventeen near-identical lines compress to two, and the pattern
  ("synthesize has been tried 15 times and never worked") becomes visible instead of implicit.
- **Show intervention effectiveness.** Nothing in today's prompt tells the orchestrator that its
  chosen strategy is not paying off. This is the single most likely fix for the intervention
  imbalance, alongside B227/B233.

**IMPLEMENTED 2026-08-05** in `agent/run_memory.py`, wired into the iterate prompt in place of
the raw dump. It falls back to the old dump on iteration 1, before the DAG has any nodes.

Source of truth is `state["dag"]` rather than the markdown: it is append-only, and rollback marks
nodes `pruned` instead of deleting them, so discarded attempts stay visible even though
`state["scores"]` pops them. `escalate` resets it on a tier change, which is correct — this memory
is per-model, the same scope as the escalation policy.

Rendered from the **real** CLINC150 DAG as the orchestrator would have seen it at iteration 19:

```
## MOST RECENT ITERATION (full detail)
  iteration 19 | data_rebuild/resample | f(pi)=0.5975 (delta -0.2758) | ROLLED BACK
  by bucket: easy=0.818(n=214)  medium=0.660(n=444)  hard=0.197(n=142)
  top confusions: accept_reservations->restaurant_reservation (6), calendar->calendar_update (6)

## WHAT WORKED (kept improvements, oldest first)
  iter 1  data_rebuild/acquire  -> 0.8261 (+0.8261)
  iter 2  hyperparameter        -> 0.8734 (+0.0472)
      because: easy cases are handled but hard cases are weak (hard=0.493) — an optimization gap.

## FAILED SINCE THE LAST IMPROVEMENT (17 attempts, none kept)
  data_rebuild/synthesize  x15   deltas -0.3336 .. -0.0087   mean -0.1428
  data_rebuild/acquire     x1    deltas -0.1306 .. -0.1306   mean -0.1306
  data_rebuild/resample    x1    deltas -0.2758 .. -0.2758   mean -0.2758
  --- the 5 most recent, in full ---
  ...
  (12 older failure(s) counted in the totals above)
  => data_rebuild/synthesize has been tried 15x since the last improvement and has never once
     been kept. Prefer a DIFFERENT intervention type.
```

That last line is the sentence the orchestrator never got to read during the real run.

**One defect found while validating against the real DAG**, worth recording because it is the same
class as B237: `pi.D.plan` persists across iterations, so a `hyperparameter` node still carries the
plan from whichever rebuild last ran, and iteration 2 rendered as `hyperparameter/acquire` — a data
strategy credited to an iteration that never touched the data. Sub-strategy is now shown only for
`data_rebuild`. 22 tests in `tests/test_run_memory.py`. BUGS B239.

---

## The hypothesis truncation (you asked what was going on here)

**It was being cut in five separate places, and the worst one was at the source.** Every long
hypothesis in the run landed at exactly **240 characters**, ending mid-word:

```
...change_language→translate (4), cancel→freeze_a
...(7 combined), change_langua
...(3), account_blocked→EX
```

Fourteen of twenty iterations were cut this way. The cut consistently landed *inside the
confusion-pair list* — so what survived was the generic preamble ("Hard-bucket accuracy is 0.683
(n=142)...") and what was destroyed was the specific evidence. That is a direct contributor to the
stale-0.683 problem: the reusable part was deleted and only the boilerplate was carried forward.

| # | Where | Was | Now |
|---|---|---|---|
| 1 | `iterate._validate_decision_json` | `[:240]` — **the source**, silent | `HYPOTHESIS_MAX_CHARS` = 2000, warns |
| 2 | `data_rebuild` hypothesis | `maximum=240` | `HYPOTHESIS_MAX_CHARS` |
| 3 | `data_rebuild` pattern_hint | `[:240]` | 1200 (it steers synthesis prompts) |
| 4 | `context_manager` summary | a further `[:100]` | removed |
| 5 | rollback memo | `[:200]` | removed |

Cut 1 is why it appeared everywhere: it ran at validation, so the console log, `data-curation.md`,
`dag.json` and the next prompt all inherited an already-severed string. Nothing logged that it had
happened, which is why it went unnoticed for the whole run. Truncation now prints a warning — if
you ever see it, raise `SLM_HYPOTHESIS_MAX_CHARS` rather than accept the loss.

**The iteration-1 garbage has the same origin.** Fields were read with
`re.search(rf'{label}:\s*(.+)')`, and `\s*` matches newlines — so an **empty** field captured the
next non-empty line. Iteration 1 has no orchestrator hypothesis, so the model was shown:

```
- Iter 1: f(pi)=0.8261, ... — ### Hardware profile (Phase 1: theoretical)
```

a markdown heading presented as its own past reasoning. Reads are now anchored to the line, so an
empty field stays empty. This affected every field, not just the hypothesis.

10 tests in `tests/test_hypothesis_integrity.py`. BUGS B238.

## Q8. "you can remove MAX_STALL_EVALS ... I only need one of these ... escalate if after 15 evals without an improvement of over 2%. It should not reset after a rollback, meaning an improvement followed by 9 rollbacks, an improvement, and then 4 more rollbacks should escalate if in those two improvements the accuracy did not go up more than 2 percent."

**Done, exactly as specified.** `MAX_STALL_EVALS` is deleted from code, tests, and docs.

The key change is *what the window counts*. It used to read `state["scores"]`, which `rollback`
pops — so a mostly-regressing run kept a permanently short list and the check could never fire.
There is now an append-only `state["eval_history"]`, written by `evaluate_node` on every eval
including ones that get rolled back, and reset only on a tier change.

Your worked example is now a test:

```python
history = [0.300] + [0.28] * 9 + [0.310] + [0.29] * 4   # 15 evals, 2 "improvements"
# best 0.310 vs window start 0.300 = +0.010 gain, under the 2% bar
assert iterate_node(...)["next_action"] == "escalate"
```

and its counterpart, which must *not* escalate:

```python
history = [0.300] + [0.28] * 9 + [0.350] + [0.29] * 4   # +0.05 over the window
assert iterate_node(...)["next_action"] != "escalate"
```

Improvements do not reset anything — only cumulative gain across the window avoids escalation.
`MAX_EVALS_BEFORE_ESCALATION = 30` remains as an unconditional ceiling.

---

## Q9. "I actually do want it to run off the eval numbers of the latest most successful run rather than the run it just failed on ... the only thing that should change after a rollback is the model should have some extra context for the failure case of the intervention it just tried, and know to try something different."

**You are right, and I have reverted yesterday's B227 fix in favour of your design.**

My reasoning yesterday was wrong in an important way: after a rollback the *live weights* are the
restored best checkpoint, so the per-difficulty scores and confusion pairs that describe the current
model genuinely are the best node's. Showing the failed attempt's numbers would have the
orchestrator reasoning about a model that no longer exists. Your framing — "same numbers, plus a
memo about what just failed" — is the correct one.

So: `test_report` is restored from the best node again (original behaviour), and rollback now also
writes `state["last_failed_attempt"]`, which is surfaced at the **top** of the report block:

```
## LAST ATTEMPT WAS ROLLED BACK — do not repeat it
- Tried: intervention='data_rebuild' (sub-strategy='synthesize') on iteration 14
- Result: scored 0.5986 vs best 0.8734 (Δ=-0.2748) — REGRESSION, so it was discarded and the
  previous best checkpoint restored.
- That attempt's difficulty profile: easy=0.802  medium=0.611  hard=0.176
- Its stated hypothesis was: Hard-bucket accuracy is 0.683 (n=142), far below...
- The scores in the report BELOW describe the restored best checkpoint (the current live
  weights), NOT the failed attempt. Choose something materially different from the failed
  attempt above.
```

It is cleared by `evaluate_node` as soon as a new eval runs, so it always refers to the immediately
preceding failure and never goes stale.

**This does mean the frozen-diagnosis behaviour from B227 is back by design** — and that is
defensible now, because the previously-missing piece (any signal at all that the last attempt
failed, and what it was) is present. What the old run lacked was not fresh numbers so much as *any
indication that 17 attempts in a row had been discarded*.

---

## Test status

**818 passed, 2 failed** (32 of those are new: 10 for hypothesis integrity, 22 for run memory) — the same two pre-existing, unrelated failures
(`test_iterate_prompt_receives_complete_curation_counts`,
`test_code_planner_and_model_choice_prompts_target_apps_introductory`).

## Docs updated (and now tracked going forward)

- **`PIPELINE.md`** — escalation table (`STAGNATION_WINDOW` 15, `MAX_EVALS_BEFORE_ESCALATION` 30,
  `MAX_STALL_EVALS` removed), the surgical knobs, the per-tier sizing formula, routing row, and a
  note explaining the `eval_history` change.
- **`PROMPTS.md`** — banner marking the hard-negative prompt sections as removed-but-retained;
  synthesis description updated; legacy Claude hard-negative backend marked removed.
- **`DATA_CURATION_AND_CAPS.md`** — gold-only synthesis, the teacher verifier, why hard negatives
  were removed, per-tier sizing, the fill/surgical split, floor 5000 / ceiling 25000, and the new
  escalation policy.

## New bugs found while writing this up

- **B237 — the orchestrator's memory of its own history was factually wrong.** The curation log
  recorded the score-band policy's *guess* as the iteration's intervention instead of the one
  actually executed, so the trajectory told the orchestrator that iterations 4-8 were
  `hyperparameter` when the DAG shows all five were `data_rebuild/synthesize`. Fixed — the log now
  records the executed intervention. See Q7 above.
- **B238 — the hypothesis was truncated in five places**, the worst at the source, silently.
  Fixed; see the truncation section under Q7.
- **B239 — the run-memory block** replaces the raw curation-log dump. Built; see Q7.

## Still open

1. **Synthetic label verification is unproven in production.** The teacher verifier is implemented
   and logged but has never run on a real batch; the next run is the test.
2. **Run memory has never been exercised end to end.** It is unit-tested and was validated by
   rendering the real CLINC150 DAG, but it changes what the model sees, so its effect on run
   behaviour is unmeasured until the next full run.
3. **Historical hypotheses stay truncated.** The 240-char cut is fixed going forward, but the text
   already written to `slm-clinc150-cse-38155022` is lost — comparisons against that run should
   account for it.
