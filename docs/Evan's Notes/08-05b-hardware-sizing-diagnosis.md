# Deterministic hardware, dataset sizing, synthesis quality, and the frozen diagnosis (2026-08-05)

Third follow-up on run **`slm-clinc150-cse-38155022`**. Each question is restated, then answered
against the code. **Two of my earlier claims were wrong and are corrected below.**

---

## ⚠️ Correction first: I was wrong about "zero synthetic rows"

I told you the winning dataset contained no synthetic data. **You were right, I was wrong.**
`dataset_v2.jsonl` contains **17 synthetic rows**. My count keyed on `_provenance`, and synth-fill
rows never get that field — they only have `_source: "synth"`, so a provenance-based count misses
them entirely. That is the same tagging gap I fixed yesterday, and it fooled my own audit.

Here is every synthetic row that made it into the winning dataset:

| # | label | text | verdict |
|---|---|---|---|
| 1 | recipe | how do i make a reservation at the chef's table | ❌ `accept_reservations` |
| 2 | recipe | can you explain why there is a hold on my savings account | ❌ `account_blocked` |
| 3 | recipe | mix eggs with flour for 5 minutes | ✅ |
| 4 | recipe | i would like an update on the progress of my chocolate cake recipe | ❌ `order_status` |
| 5 | recipe | can you give me information about the nutritional value of my recipe | ❌ `nutrition_info` |
| 6 | recipe | I want to know if you are a real person or an ai | ❌ `are_you_a_bot` |
| 7 | recipe | check if the recipe allows for reservations | ❌ incoherent |
| 8 | recipe | my dough seems to have been blocked for no reason | ❌ incoherent |
| 9 | recipe | set the oven alarm for 30 minutes | ❌ `alarm` |
| 10 | book_flight | what's the best way to make a non stop flight from new york to london | ⚠️ awkward |
| 11 | book_flight | how to book a flight from new york to london | ✅ |
| 12 | book_flight | can you help me figure out the best way to cook a turkey for thanksgiving | ❌ `recipe` |
| 13 | book_flight | i need to book a flight for my husband who loves to cook scottish haggis | ⚠️ contrived |
| 14 | book_flight | what ingredients do i need to book a flight from new york to london | ❌ incoherent |
| 15 | book_flight | can you find the best flight deals to new york for this weekend please | ✅ |
| 16 | book_flight | what are the steps i need to follow to book a flight to paris | ✅ |
| 17 | book_flight | how do i book economy class to london | ✅ |

**~10 of 17 are mislabelled or incoherent.** This is the single best piece of evidence in the whole
investigation, and it directly answers your Q9.

---

## Q1. "if there's a phone that's found in the dataset, the rest of the process is done through deterministic python ... Exa and Claude should only be used if the phone is not found"

**Implemented.** `research_device` now has a **Stage 0**: `_exact_local_device` resolves everything
in Python and returns before any client is even constructed.

It requires an *unambiguous* hit — if the top two CSV rows tie on score, or RAM/storage are
missing, or the chipset does not map to a known reference chip, it logs why and defers to the
research path. That prevents silently guessing between "Galaxy S24" and "Galaxy S24 Ultra".

The budget model is now explicit arithmetic, including the runtime overhead you asked for:

```
usable_ram_mb     = total_RAM − OS/foreground overhead − inference runtime overhead
storage_budget_mb = STORAGE_BUDGET_FRACTION × total_storage
```

| Constant | Default | Env override | Rationale |
|---|---|---|---|
| `ANDROID_OS_OVERHEAD_MB` | 2048 | `SLM_HW_OS_OVERHEAD_MB` | Android OS + foreground app |
| `INFERENCE_RUNTIME_OVERHEAD_MB` | 1024 | `SLM_HW_RUNTIME_OVERHEAD_MB` | **KV cache, activations, tokenizer, allocator fragmentation** — the "space required for hosting the model other than the model size itself" |
| `STORAGE_BUDGET_FRACTION` | 0.60 | `SLM_HW_STORAGE_FRACTION` | phones are not empty |

Live result, with **zero API calls**:

```
[hw] EXACT local-DB hit: Samsung Galaxy S24 Ultra — resolved deterministically, no Exa/Claude call
[hw] Samsung Galaxy S24 Ultra | Qualcomm Snapdragon 8 Gen 3 | RAM 12GB → usable 9216MB |
     model budget 157286MB | ref_chip=snapdragon_8gen3 (source: local_db_exact)
[hw] rationale: deterministic: usable RAM = 12288MB − 2048MB OS/foreground − 1024MB inference
     runtime (KV cache, activations, allocator headroom) = 9216MB; storage budget = 60% of
     262144MB = 157286MB
```

Verified identical across repeated calls. Note usable RAM is now **9216 MB** rather than the LLM's
10240 MB, because the runtime overhead is now subtracted explicitly rather than being hand-waved.

I also had to fix the chipset mapper: `"Qualcomm Snapdragon 8 Gen 3"` and `snapdragon_8gen3` share
only the token "snapdragon", so a token-overlap test failed. It now compares compacted forms
(`qualcommsnapdragon8gen3` contains `snapdragon8gen3`).

---

## Q2. "get rid of all code and docs related to those old numbers ... have the orchestrator decide the dataset size ... based on how novel the task is and the model size ... recompute every model tier ... floor 5000, cap 25000. Ask me any questions about this design and validate whether it is realistic."

**Not implemented — I want your answers first, because two parts of the design have problems.**
The floor/cap you specified (5000/25000) are already the live values
(`CURRICULUM_SIZE_FLOOR=5000`, `DATA_SIZE_CEILING=25000`), so nothing needs removing there; the
"old small numbers" that remain are `DATASET_SIZE_BY_TYPE` (classification=150), which is dead in
practice since it is always clamped up to 5000.

**Where I think the design is right:** recomputing per tier is clearly correct. A 0.6B and a 4B
model genuinely need different data volumes, and today the target is computed once and reused
across escalations, which is just wrong.

**Where I think it has problems:**

1. **The orchestrator cannot estimate "task novelty" from what it is given.** It sees a task
   description and a label list — it has no measurement of whether CLINC150 is in Qwen3-0.6B's
   pretraining distribution. It would be guessing, and its guess would look authoritative. **You
   already have a real measurement of exactly this: the zero-shot baseline.** Qwen3-0.6B scored
   0.3152 on CLINC150 before any training. That *is* the novelty signal, and it is empirical.
   I would strongly prefer `f(zero_shot_baseline, model_size)` over asking the LLM.

2. **Within 5000–25000, does the size actually matter?** This run trained on 3,461 rows and hit
   0.8971 against a 0.8891 goal. There is no evidence in any run so far that 5,000 vs 25,000 rows
   changes the outcome — and CLINC150 only *has* ~15,250 real train rows, so anything above that
   is necessarily synthetic. Given synthesis is currently ~60% mislabelled (see Q9), a bigger
   target would actively make things worse right now.

**Questions I need answered before implementing:**

- **Q2a.** Do you want the LLM to choose, or a deterministic formula from the zero-shot baseline
  and parameter count? I recommend the formula, with the LLM able to override within bounds.
- **Q2b.** On escalation to a bigger model, should the target go **up** or **down**? Your message
  says "the smaller the model, the more data it will require", which implies escalation should
  *lower* the target. That is the opposite of what most people expect, and it means escalating
  discards data. Confirm this is intended.
- **Q2c.** Where does the extra data come from above ~15k? For CLINC150 it can only be synthetic.
  Should the target be capped at available real data until synthesis quality is fixed?
- **Q2d.** "recomputed every time it escalates **or regresses to a new tier**" — is there a
  downward-tier path? I only see escalation and the `largest_first` downward probe.

---

## Q3. "confirm what you did exactly here ... you should have a set list of classes ... you should not allow any data that doesn't match those classes to enter the pool ... that's what you did right?"

**Partly — and you were right that it belonged in QC, so I have now added that too.**

What I did yesterday was a guard at the **acquisition** boundary only: reject a mined source whose
labels are *entirely disjoint* from the run's label space. That stops the DeepPavlov case, but it
is weaker than what you described in two ways: it works per-source rather than per-row, and it
only triggers on *total* disjointness, so a source with 90% good labels and 10% junk would pass
with the junk included.

**Now also implemented, exactly as you described** — a per-row label-space filter inside
`apply_quality_controls`:

```python
if allowed_labels:
    clean = [e for e in clean if str(e["label"]) in allowed_labels]
```

The allowed set is derived from the **frozen eval set** — those are precisely the classes the model
will be scored against, so anything outside them is unusable by construction. It runs on *every*
row from *every* source (real, mined, synthetic) and logs what it dropped:

```
[qc] label-space: removed 164 row(s) — label not in the task's 151 established classes
     ['0'x60, '1'x60, '2'x44]
```

So there are now two layers: reject bad sources at acquisition, and filter stray rows at QC.

---

## Q4. "you made it so that if there is a value error or exception raise, you run the reask which is basically just telling the orchestrator to redo its prompt right? won't it just regenerate the same error? what happens if the reask fails?"

**Your understanding is right, and your scepticism is justified.** The reask replays the same
conversation plus the sanitized validator error and asks for JSON only. It is a single retry.

**Will it repeat the mistake?** Sometimes, yes — that is a real risk, and it is why I did not stop
at "add a reask". The chain now is:

1. **First parse fails** → log the failure and attempt one reask.
2. **Reask returns** → if it validates, use it. (Logged: `Reask succeeded`.)
3. **Reask fails the same way** → **salvage**: `_strip_retired_hyperparams` drops just the
   offending retired keys and keeps the tunable ones, tagging
   `_dropped_fields: ["retired_hyperparams"]`. Those keys are never applied by the trainer, so
   dropping them yields exactly the decision the orchestrator could legally have written.
4. **Still unusable** → fall back to the test-agent suggestion, then to score-band rules — as
   before, but now with an explicit log line saying so.

So for the specific failure that hit your run 6 times, the answer to "won't it just repeat?" is:
it may, and it no longer matters, because step 3 recovers the decision anyway. Verified: with a
mock that repeats the mistake, the pipeline records `iterate_json_reask` and still returns a usable
decision.

This mirrors an existing precedent — the code already drops a stray `hyperparams` block from a
`data_rebuild` rather than rejecting it, with a comment noting that rejecting whole decisions cost
the NER run 65 orchestrator-authored plans.

---

## Q5. "split synthesize into fill_synthesize and surgical_synthesize ... tell me if my analysis is correct, what you'd recommend, and criticize and evaluate my ideas"

**Your analysis is correct in principle and I want to build it — but not yet, and here is the
honest reason.**

**What is right about it:** the orchestrator names specific confusion pairs every turn
(`change_ai_name↔change_user_name`, `change_language→translate`) and that information is currently
**thrown away**. `curate_node` never passes `pattern_hint` into `synthesize_examples`, so the
generator has no idea which failure it is supposed to attack. The prompt template even has a
`pattern_hint` slot that is always empty. Splitting fill (broad, balanced) from surgical (small,
targeted) is the right shape.

**Why I did not implement it yet:** surgical synthesis multiplies the *value* of each generated row
— and right now ~60% of generated rows are mislabelled (the table at the top). Pointing a broken
generator precisely at your hardest, most confusable classes would inject noise exactly where the
model is weakest, which is worse than injecting it randomly. **Fix generation quality first, then
add targeting.** I would rather tell you that than ship something that looks like progress.

**My criticisms of the design as stated:**

- "Surgical adds a smaller amount" — I would make it proportional to the confusion count, not a
  fixed small number. Two classes confused 7 times need more than a pair confused twice.
- Surgical rows are the *most* likely to be mislabelled, because near-synonym classes are exactly
  where the generator's "resembles X but is really Y" framing collapses. They need verification
  more than fill rows do, not less.
- There should be a feedback check: if surgical synthesis on a pair does not improve that pair's
  confusion count next iteration, stop spending on it.

**What I would build, in order:** (1) fix generation quality per Q9, (2) add a label verifier,
(3) then add `surgical_synthesize` with per-pair budgets and a per-pair effectiveness check.

---

## Q6 + Q9. "how is the qwen model going to generate the positive examples ... is it referencing a real data row? ... what's the prompt? is it made by the orchestrator?" + "most of it is bogus ... your way of generating hard negatives is like 'rephrasing rows' ... this seems unfeasible ... it is costly and ineffective to have qwen self check its work. give me a solution"

**How the prompts are made:** not by the orchestrator. They are **fixed templates** in
`data/curriculum.py`. The orchestrator only picks the strategy and a row count. Qwen never sees the
task description or the orchestrator's hypothesis.

**New-gold generation (added yesterday)** *does* reference a real row — it takes an anchor of the
target class and asks for a new utterance of the *same* class:

> Write ONE new, realistic user utterance that belongs to the '{label}' class... It must be
> genuinely NEW and phrased differently from the reference — not a paraphrase, not a copy — while
> unambiguously belonging to '{label}'.
> Reference '{label}' example: {anchor text}

**Hard negatives** also reference a real row, but the instruction is the problem:

> generating a HARD NEGATIVE ... a realistic example that superficially **resembles the
> '{src_label}' class** but genuinely **belongs to the '{target_label}' class**. The surface
> features should mislead toward '{src_label}' while the true meaning is unambiguously
> '{target_label}'.

**You are essentially right, and the data proves it.** It is not quite "rephrasing" — it is asking
for a *blend*: keep the surface of class A, but be truly class B. That is a genuinely hard
instruction, and a 35B model asked to do it 1,500 times produces Frankenstein sentences:

- `"what ingredients do i need to book a flight from new york to london"` — recipe surface +
  flight target = belongs to neither
- `"my dough seems to have been blocked for no reason"` — recipe surface + account_blocked target
- `"check if the recipe allows for reservations"` — incoherent

The failure is structural: **there is no such thing as a valid "looks like A, is really B" example
for most CLINC150 pairs**, because the classes are defined by intent, and intent *is* the surface
meaning. The instruction asks for something that often cannot exist.

**My recommended solution, in priority order:**

1. **Stop generating hard negatives from a blend instruction.** Instead, **mine them from the
   model's own confusion matrix.** You already know which pairs it confuses. Take *real* eval-clean
   rows of class B that the model predicted as A — those are real, correctly-labelled, genuinely
   hard examples, at zero generation cost and with zero label risk. This is strictly better than
   synthesizing and it uses data you already have.
2. **Default the ratio toward gold.** New-gold generation is a much easier task ("write another
   `alarm` request") and its label is inherited from the anchor rather than asserted by the model,
   so it cannot be mislabelled the same way. It is already the 65% majority; I would go further
   and make hard negatives opt-in.
3. **On verification** — you are right that Qwen self-checking is weak, but note it is *not* the
   same task. Generating "text that looks like A but is B" is hard; *classifying* an utterance
   into 151 intents is exactly the task the reference model scores 0.889 on. A round-trip check
   ("does the generator's own model classify this row as the label we assigned?") is cheap
   (one short call, no generation) and would have caught 10 of the 17 bad rows. I would use it as
   a filter, not a rewriter.
4. **Cheapest immediate win:** drop any generated row whose text contains a strong surface marker
   of a *different* class than its label (e.g. "recipe" appearing in a `book_flight` row). Crude,
   but it catches the incoherent hybrids for free.

---

## Q7. "This should be part of QC right? throw out all the rows with labels that are not in the set of verified labels established in the beginning."

**Yes — implemented, see Q3.** It now runs as QC step 0 for classification, per row, from every
source, with a log line naming the rejected labels and counts.

---

## Q8. "LOG THE WHOLE THING ... explain more about what you mean by your suggested solutions, show me exactly what it would look like. what do you mean by free-text?"

**Full logging restored.** `SLM_LOG_FULL_ITERATE_PROMPT` now defaults to `1`, so the complete
system prompt and user content are logged verbatim every turn. (Set it to `0` for the compact
delta mode.)

> **SUPERSEDED 2026-08-05 (B246).** The default is now `0`. Iteration 1 still logs the complete
> prompt so the run log stays self-contained and replayable, but later turns log only the
> changing user content — re-printing the constant system prompt every turn is most of the
> 23,900 lines in `slm-clinc150-cse-38155022` and buried the parts that actually differ. Set
> `SLM_LOG_FULL_ITERATE_PROMPT=1` to restore the behaviour described above.

**"Free-text" means the orchestrator's `hypothesis` field** — the prose it writes to explain its
decision, e.g.:

> "Hard-bucket accuracy is 0.683 (n=142), far below the 0.8891 stop threshold. Dominant confusion
> pairs — change_ai_name↔change_user_name (7 combined), change_language→translate (4)..."

That prose is stored per DAG node and **replayed verbatim** into every later prompt as part of the
trajectory. It is where the stale `0.683` was coming from — the model quoting its own old text back
to itself, 163 times.

**Concretely, here is the change I proposed.** Today's trajectory block:

```
- Iter 4: f(π)=0.8294, band=0.80-0.95, intervention=hyperparameter, model=Qwen/Qwen3-0.6B —
  Hard-bucket accuracy is 0.683 (n=142), far below the 0.8891 stop threshold. Dominant confusion
  pairs — change_ai_name↔change_user_name (7 combined), change_language→translate (4),
  cancel→freeze_account (3), account_blocked→extraction_failed
- Iter 6: f(π)=0.8055, ... Hard-bucket accuracy is 0.683 (n=142), well below ...
- Iter 7: f(π)=0.8530, ... Hard-bucket accuracy is 0.683 (n=142), far below ...
```

What I would replace it with — trajectory as a compact table (no prose), plus **one** current-state
block:

```
## Trajectory (score + what was changed; rationale omitted by design)
  iter | score  | intervention  | sub-strategy | Δ vs prev
     1 | 0.8261 | data_rebuild  | acquire      |   —
     2 | 0.8734 | hyperparameter| —            | +0.0473
     4 | 0.8294 | data_rebuild  | synthesize   | -0.0440
     6 | 0.8055 | data_rebuild  | synthesize   | -0.0239
     7 | 0.8530 | data_rebuild  | synthesize   | +0.0475

## Intervention effectiveness so far
  synthesize   : tried 15x, mean Δ -0.041, best Δ +0.048  <-- not working
  acquire      : tried  2x, mean Δ +0.012
  hyperparameter: tried 1x, mean Δ +0.047

## CURRENT EVAL (iteration 19 — this is the only live measurement)
  overall   : 0.5975   (goal 0.8891, gap -0.2916)
  by bucket : easy=0.918(n=214)  medium=0.702(n=444)  hard=0.197(n=142)
  vs iter 2 : easy -0.045   medium -0.192   hard -0.486
  top confusions: change_ai_name→change_user_name (7), change_language→translate (4)
```

Three differences that matter: the stale prose is gone; there is exactly **one** copy of the
current numbers, explicitly labelled; and the model is shown whether its chosen intervention is
actually working, which nothing in the current prompt tells it.

I have **not** implemented this — it changes model input and therefore run behaviour, so it is your
call. It is the single highest-leverage change remaining.

---

## Q10. "What's the difference between STAGNATION_WINDOW and MAX_STALL_EVALS?"

They count different things, and only one of them survives rollback.

| | `STAGNATION_WINDOW` (15) | `MAX_STALL_EVALS` (15) |
|---|---|---|
| Counts | entries in `state["scores"]` | `consecutive_no_improvement` |
| Question | "over the last 15 *retained* scores, did the best beat the first by ≥2%?" | "how many evals in a row failed to beat the best?" |
| Catches | a model creeping along with trivial gains | a model repeatedly regressing |
| Survives rollback? | **No** — `rollback.py:40` pops the score | **Yes** — never popped |
| Fired in your run? | **Never could** (2 scores retained) | reached 17/20 |

That is why they look redundant but are not: stagnation catches *flat improvement*, stall catches
*repeated failure*. In a run where most iterations regress, only the stall counter works — which is
exactly why I added the third guard, `MAX_EVALS_BEFORE_ESCALATION = 30`, which counts evals
actually performed and cannot be defeated by either mechanism.

---

## Q11. "It shouldn't save intermediates because that's a lot of storage. It's fine if it doesn't refill back up to the floor, just make it clear that it's under because of QC. add a message saying how much QC clears each time and for what reasons. Also send a warning that it's proceeding below the data goal floor."

**All three done. No intermediate versioning added** (agreed — storage cost is real, and B225 is
withdrawn as a recommendation).

Per-control QC reporting:

```
[qc] schema: removed 3 row(s) — row missing a 'text' or 'label' field
[qc] label-space: removed 164 row(s) — label not in the task's 151 established classes ['0'x60, '1'x60, '2'x44]
[qc] label-balance: removed 1501 row(s) — label over the cap of 3x the smallest class (66 rows/label; smallest class has 22)
[qc] length-outlier: removed 12 row(s) — text longer than 3x the median length
[qc] surface-dedup: removed 8 row(s) — near-duplicate text (Jaccard > 0.9)
Quality control removed 1688 row(s) total (5149 → 3461)
```

And the explicit warning:

```
⚠ PROCEEDING BELOW DATA TARGET: 3461 row(s) vs target 5000 (short by 1539). Synth-fill runs
BEFORE quality control and there is no refill afterwards, so QC removals land the final dataset
under target. See the [qc] lines above for exactly what was removed and why.
```

---

## Q12. "Figure out why the hard bucket accuracy was frozen, and check the other difficulty sets too. Make sure the numbers are legit and it's not a bug that it just returns the same number every time."

**Found it, and it is a real bug — the worst one in this whole investigation.**

**The numbers themselves are legitimate.** The test agent recomputes every eval; its diagnoses
across the run show `hard=` at 0.493, 0.408, 0.585, 0.577, 0.514, 0.310, 0.134, 0.289, 0.401,
0.535, 0.275, 0.246, 0.176, 0.549, 0.197. Not frozen, not a scoring bug.

**But the orchestrator was shown a frozen copy.** The prompt's per-difficulty line:

| Value shown in prompt | Turns |
|---|---|
| `easy=0.963(n=214) medium=0.894(n=444) hard=0.683(n=142)` | **18** |
| `easy=0.935 medium=0.885 hard=0.493` | 1 |

Line 3704 computed `hard=0.408`; the very next prompt (line 4064) showed `hard=0.683`.

**Root cause — `rollback.py`:**

```python
state["test_report"] = deepcopy(evaluation_state.get("test_report"))
```

Rollback restores the *best node's* `test_report`. Iterations 3–19 all regressed, so every one of
them restored iteration 2's report. **All three buckets were frozen, not just hard** — easy and
medium too.

**Fixed.** Rollback now stores the restored copy under `restored_test_report` and leaves
`test_report` describing the eval that actually just ran. The orchestrator is asked "what should we
try next?" — it must see the failure it is reacting to, not a checkpoint that succeeded.

---

## Q13. "I still want to know why hyperparameter wasn't chosen at all last run and how to balance it out."

**It was not bias, and it was not the orchestrator's judgement — it was the same frozen report.**

| What the prompt told the orchestrator | Turns |
|---|---|
| `Test-agent suggested intervention: data_rebuild` | **18** |
| `Test-agent suggested intervention: hyperparameter` | 1 |

Meanwhile the **live** test agent said "Tune hyperparameters" **21 times** across the run.

The frozen iteration-2 report said *"below goal (overall 0.873) with no single failing bucket — add
more balanced data and continue"* → `suggested_intervention: data_rebuild`. That recommendation was
replayed 18 times. The orchestrator was doing what it was told; it was told the wrong thing.

**How it balances out now:** the Q12 fix alone should largely correct it, because the live report
will actually reach the prompt — and the live report favoured `hyperparameter`. Beyond that, the
"intervention effectiveness" block from Q8 would let it see that `synthesize` had a mean Δ of
−0.041 over 15 attempts, which nothing currently tells it.

I would re-run before adding any explicit balancing rule. There is a good chance the bias
disappears once the orchestrator can see real data.

---

## Summary of changes

| Area | Change |
|---|---|
| Hardware | Deterministic Stage 0 on exact CSV hit — no Exa, no Claude; explicit RAM/storage budget model incl. inference runtime overhead; compact chipset matcher |
| Rollback | No longer overwrites `test_report` with the best node's copy (**B227**) |
| QC | Per-row label-space filter from the frozen eval set; per-control removal reporting with reasons; below-target warning (**B228/B229**) |
| Iterate | Full prompt logged verbatim every turn again (`SLM_LOG_FULL_ITERATE_PROMPT=1` default) — ~~superseded by B246, default is now `0`: full prompt on iteration 1, user content only thereafter~~ |
| Acquisition | (yesterday) source-level label-space rejection |

**Tests: 826 passed, 2 failed** — the same two pre-existing, unrelated failures.

## Open, awaiting your decision

1. **Dataset sizing** — needs answers to Q2a–Q2d before I build it.
2. **Prompt restructure** (Q8) — highest-leverage remaining change; changes model input.
3. **Synthesis quality** (Q6/Q9) — I recommend mining hard negatives from the real confusion
   matrix instead of generating them.
4. **`surgical_synthesize`** (Q5) — worth building, but after synthesis quality is fixed.
