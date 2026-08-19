# The corrupt-label ablation, what mining actually does, and the calendar gold explained plainly

*2026-08-19 — measurement + explanation*

Companions: `08-18-exact-verifiers-fewshot.md`,
`08-17-task-status.md`.

**Two of my earlier claims were wrong and are corrected here.** The DialogSum "contradictory gold"
issue is real but ~0.2% of rows and I misidentified its cause (§4). And constrained label decoding —
which I called "the biggest remaining eval win" twice — turns out to be **not worth doing**, because
extraction failures are essentially a zero-shot-only phenomenon (§5). You were right to push on both.

---

## 1. Eval set capped at 1,000

`_EVAL_SIZE_CAP = 1000` (`SLM_EVAL_SIZE_CAP` to override). Taking the whole split was my overcorrection
and you were right that it would be slow: the eval runs on **every iteration**, so RouterBench's 7,267
held-out rows would have been ~9× the old per-iteration cost.

1,000 is the sensible middle. It is 25% more rows than the old 800, and the standard error on a
proportion at n=1,000 is about **1.5 percentage points** — well below the score differences this project
actually resolves. Current sizes: 1,000 for `routerbench` / `ner_bc5cdr` / `clinc150` /
`proactive_listening` / `xlam_bfcl`; `dialogsum_samsum` 667 and `calendar_json` 478 because their whole
splits are smaller than the cap.

Preflight: **7/7 PASS** at the new size.

---

## 2. What happens when the intervention is "mine real data"

The orchestrator picks `acquire`, and `curate_node` calls `mine_additional_real_rows`. It walks a
**three-rung ladder, cheapest and most trustworthy first**, and stops as soon as a rung yields rows.

### Rung 1 — local frozen bundles (free, offline)

`load_local_dataset` checks `data/local/` for a bundle matching the task: `bc5cdr`, `gsm8k`, `samsum`,
`emotion`, `apps`, `mbpp`, and now `calendar_sgd` and `proactive_listening`. If a candidate matches, the
paid budget is set to zero — we never pay when the data is already on disk.

You saw this rung reject everything in the RouterBench logs:

```
[acquire] LOCAL candidate 'apps' rejected: no explicit benchmark or strong task+label+schema match
[acquire] LOCAL candidate 'bc5cdr' rejected: no explicit benchmark or strong task+label+schema match
```

That is correct behaviour — `apps` is a coding benchmark, it has nothing to do with routing.

### Rung 2 — the task's own benchmark (free, HuggingFace cache)

`load_benchmark_dataset` resolves the task's benchmark name through `_BENCHMARK_ALIASES` and loads it
directly. **This rung is where the RouterBench disaster happened**: there was no `routerbench` alias, so
the rung was skipped entirely, and — worse — curated runs carried no `task_plan` at all, so there was no
benchmark name to resolve in the first place. Both fixed; `routerbench` now routes to its real pickle
loader here, and curated runs name their own benchmark.

### Rung 3 — paid Exa discovery (costs money, bounded)

Only reached if rungs 1 and 2 produced nothing. Bounded hard: `MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN = 3`,
`MAX_PAID_ACQUIRE_ROUNDS_PER_RUN = 9`, metered by a durable on-disk ledger so a resumed checkpoint
cannot re-spend. Each round asks Exa for candidate HF datasets, then for each candidate:

1. `_peek_hf_dataset` reads two rows to see the columns.
2. `_llm_map_dataset` asks Claude to map those columns onto our schema.
3. `_materialize_from_mapping` loads and converts.
4. `_accept_discovered_result` validates schema, integrity, and train/test overlap.

### After a rung yields rows

- **Eval firewall** — rows matching held-out eval text are removed.
- **Novelty check** — rows already in the pool are dropped; `plan_yield.status` becomes `no_novelty` if
  nothing new survives.
- **`_merge_persistent_train_rows`** — the survivors are added to `state["train_examples"]`
  **permanently**, so every later rebuild can draw on them. This is the step that made the RouterBench
  contamination irreversible: foreign rows entered once and stayed for 36 iterations.

### 2.1 What happens if Exa finds a same-topic dataset with a different label format

This is exactly what went wrong, and it is worth walking through what the pipeline does *now* versus
what it did then.

**What happened then.** Exa found `anasnassar/llm-query-complexity-benchmark` — genuinely a
query-routing-adjacent dataset, so a reasonable hit on topic. But its labels are `LOW` / `MEDIUM` /
`HIGH` query-complexity tiers, and our task's labels are `local` / `route`. Claude was asked to map the
columns and produced a `label_map`; the code applied it as `lmap.get(str(lab), lab)` — **so any value
Claude did not map passed through verbatim**. Claude also invented plausible-sounding routing classes.
Because the mapping was re-requested per round and is non-deterministic, each round could mint a new
one: `cloud` at iterations 1–9, `+on_device` at 10, `+router` at 20, `+remote` at 21. The guard that
should have caught this accepted a source on **any** label overlap, so a source that got *some* rows
right admitted all of them.

**What happens now — four gates, in order:**

1. **The LLM is told the exact permitted labels** and that inventing one is forbidden. The prompt says
   that if the dataset's classes do not map cleanly onto that exact set, reply `{"suitable": false}`,
   because a partial mapping is worse than no dataset.
2. **`sanitize_label_map` strips any mapping entry whose TARGET is not in the vocabulary**, so a
   hallucinated class cannot survive even if the model ignores the instruction. Logged:
   `IGNORED 2 label_map target(s) that are not in the task's label space: ['cloud', 'on_device']`.
3. **Unmapped rows are DROPPED, not passed through.** This was the actual leak.
4. **The source-level guard is now a strict subset test** — a source is rejected outright unless
   *every* label it carries already exists:
   ```
   [mine] REJECTED agentic source: 3 of its 5 label(s) are NOT in the task's pinned 2-class
   vocabulary ['cloud'x688, 'on_device'x269, 'router'x110]. The label space is closed…
   ```

So the specific dataset you are asking about would now be **rejected at gate 1 or 4** rather than
silently relabelled. One honest caveat: strictness is only applied when the vocabulary is
**authoritative** (pinned from the frozen eval set, or plan-declared). If it were merely inferred from
whatever rows the run happened to hold, that list could be missing real classes, and being strict would
throw away good data — so the inferred case keeps the looser any-overlap check.

**The one thing still not caught.** If a foreign dataset's labels *happen to be spelled the same* as
ours, nothing here stops it. That is what let 1,155 rows with LLM-fabricated `local`/`route` labels
survive at a 50/50 rate against a true 30/70 base rate. Rung 2 being fixed is what prevents that in
practice — the real benchmark now answers first, so discovery is never reached for these tasks.

---

## 3. The calendar gold problem, slowly

### What the task is

Give the model a scheduling request plus **the current date and time**, and it must output a Google
Calendar API call with the date fully resolved:

```
Current date and time: 2026-02-21T15:00:00 (Saturday).
Add "Chris Webby concert" to my calendar on March 13th at 12:30 pm
→ start: 2026-03-13T12:30:00,  end: 2026-03-13T13:30:00
```

The skill being tested is **date arithmetic**: turn "March 13th" plus a known "today" into a full
timestamp. That is a real, learnable, useful skill.

### What was wrong

The request says "March 13th". It does **not** say which year. So the model has to infer the year from
the reference date — and that is only unambiguous if the reference is *before* the event.

Here is the trap. The eval data comes from SGD's calendar dialogues, and those are **almost all set in
March**. But the code that picked the "current date and time" for each row scattered it **randomly
across all of 2026**. So most rows looked like this:

```
Current date and time: 2026-08-19     ← August. March has already passed.
Request: "on 2nd of March"
```

Now, what does "on 2nd of March" mean when today is August 19th? Two readings, both defensible:

- **March 2nd, 2026** — the March in the current year. This is what most people and most models say.
- **March 2nd, 2027** — the *next* March, because this year's already gone.

The gold-answer generator applied a rule: *if the date has already passed this year, roll forward to
next year.* So the gold said **2027**. The model said **2026**. Marked wrong.

**This happened on 82% of the eval set.**

### Why that hurts training, not just scoring

Three distinct harms, and the third is the one that matters most:

1. **The score stopped measuring the task.** The model was producing the right month, right day, right
   time, right JSON structure, right function name — and failing on one digit of the year. A score of
   0.0000 read as "the model cannot do calendar formatting", when the truth was "the model does calendar
   formatting well and disagrees about an unstated convention."

2. **Fine-tuning would have taught it the wrong lesson.** You made this point yourself and you were
   right: a student *can* learn an arbitrary convention. But look at what it would have been learning
   from — 82% of rows saying "when in doubt, add a year." That is not date arithmetic. The model would
   get better at the benchmark while getting *worse* at the actual task, because the convention it
   absorbed is not a real-world rule.

3. **The ceiling was set by the same defect.** The accuracy goal is derived from the teacher's score, and
   the teacher fails for exactly the same reason (0.2176). So the target was calibrated to
   convention-guessing too. The whole measurement was circular.

### What I changed

The reference date is now placed **1 to 21 days before the event**, instead of randomly:

```
Current date and time: 2026-02-21    ← three weeks before the event
Request: "on March 13th"            → unambiguously 2026-03-13
```

No rollforward is ever needed, so the year is derivable from the prompt. The offset still varies per row
(hashed from the row id), so the model cannot memorise one fixed "today" — it still has to do the
arithmetic.

I also fixed a second, separate defect: the request used to be built as `"Schedule {title} on {date}"`,
so a row whose event was called `Food` read *"Schedule Food on March 1st"* and the model reasonably
extracted `summary="Schedule Food"`. The word added to make it a sentence became part of the thing being
extracted. It now reads `Add "Food" to my calendar on March 1st`, with the title quoted.

### Is it fixed? Yes — measured on the real eval set

| | before | after |
|---|---|---|
| gold year matches the reference year | 18% | **99%** (475/478) |
| gold rolled forward a year | **82%** | **1%** (3/478) |
| `summary` containing the request's imperative | systematic | **0** |

The remaining 1% are genuine cross-year cases (an event in early January referenced from December),
which is correct behaviour, not a defect.

**What is not yet known:** whether the task now trains well. It has never had a clean run. The gold is
fair, the preflight passes, and the new exact verifier independently checks the year against the row's
own reference instant — but the actual result is unmeasured.

---

## 4. DialogSum — you were right that it is negligible, and I had the cause wrong

Two corrections.

**It is not a DialogSum/SAMSum cross-source problem.** I said the two datasets share dialogues. Measured
on a 500+500 draw, every collision is **`samsum_train` × `samsum_test`** — SAMSum's own official train
and test splits are not disjoint, and it wrote a different summary for the same dialogue in each. Nothing
to do with DialogSum.

**On whether they are "genuinely separate rows":** the dialogue text is **byte-identical**, not merely
similar — I verified with an exact string comparison, not normalization. So it is the same input appearing
in both splits with two different gold summaries:

```
train: "Jeff has a skin allergy. He doesn't take meds all the time..."
eval : "Serena's skin condition is fine now and she doesn't have to take medication..."
```

They are separate *rows* in SAMSum's files, but they are the same *example*. Reading the dialogue, the
training summary is the accurate one, so the eval row is unwinnable.

**And you are right that it does not matter.** It is **2 rows out of 1,000 (~0.2%)**. It did not move the
run's 0.7157, and curate's eval firewall already removed the training side before training — so no run
was ever contaminated. My earlier note presented this as a finding that undermined DialogSum's tier-3
+0.0000 result. **It does not**, and I have corrected that in the 08-18 note and downgraded B285 to
cosmetic.

I kept the deduplication because it is three lines and it makes the reported curriculum size honest
rather than having the firewall silently shrink it. But it is hygiene, not a fix for anything that
mattered.

---

## 5. Extraction failures — I measured it, and constrained decoding is NOT worth doing

You said you were not convinced this mattered unless failures are frequent, and that you did not want a
big eval-harness change without necessity. **You were right on both counts.** Here is the measurement I
should have done before recommending it twice.

Extraction-failure rate per eval, split by whether the model was fine-tuned:

| Run | BASELINE (zero-shot) | FINE-TUNED |
|---|---|---|
| `routerbench` l40s | median **50.3%**, max 63.9% | median **0.0%**, mean 2.6%, **47/77 evals had zero** |
| `routerbench` cse | median **50.3%** | median **0.0%**, mean 1.7%, **50/75 zero** |
| `clinc150` | 34.0% | median **2.1%** |
| `ner_bc5cdr` | 0.0% | **0.0% on all 7 evals** |
| `xlam_bfcl` | 0.0% | **0.0% on all 23 evals** |
| `dialogsum_samsum` | 0.0% | **0.0% on all 33 evals** |

**Extraction failure is a zero-shot phenomenon.** After fine-tuning the model emits the bare label
reliably — the median is exactly zero on every task, and three of six tasks never had a single failure at
any point. The occasional 50% spike in the RouterBench fine-tuned column lines up with the mode-collapse
iterations (the model emitting `</tool_call>` on 522/800 rows), which is a *training* pathology; a
constrained decoder would have turned those into confident wrong labels rather than fixing anything.

**So I am withdrawing the recommendation.** Constrained decoding would:

- fix a problem that affects **baselines only**, not the fine-tuned numbers that the project reports;
- require a log-probability path in **both** inference backends (Unsloth and llama-cpp-python), which
  expose scoring differently;
- invalidate every classification baseline ever measured, requiring a deliberate before/after.

That is a large change to the eval harness for a ~0–2% effect on the numbers that matter. **Not worth
it.** The prompt hardening already shipped (fenced payload, contract restated after it, tail-anchored
extraction) is the proportionate fix, and the measurement above says it is sufficient.

If baseline honesty becomes important later, the cheap version is to **report a few-shot baseline
alongside the zero-shot one** — which reuses the probe harness that already exists and touches no
inference code.

---

## 6. The corrupt-label ablation — what it does, and the result

### The question it answers

We know the teacher scores **0.1131 zero-shot** and **0.7215 with five demonstrations** on BC5CDR NER —
a 6.3× jump. The question is *why*, because the answer decides whether the zero-shot score should gate
whether we trust the teacher to generate data.

Two possible explanations:

- **FORMAT.** The demonstrations taught it the *output contract* — that spans go in a JSON list, that the
  types are exactly `Chemical` and `Disease`, that there should be no markdown fence. If so, the teacher
  always knew what a biomedical entity was, and 0.1131 was measuring our formatting requirements rather
  than its knowledge.
- **KNOWLEDGE.** The demonstrations taught it *what to look for*. If so, 0.1131 is a real measure of
  ability and it should gate synthesis.

### How the ablation separates them

Show the teacher five demonstrations that are **perfectly formatted and factually wrong.**

Concretely: keep the JSON shape, keep the `Chemical`/`Disease` vocabulary, keep the same number of spans
— and replace the actual span text with entity names borrowed from *other* rows, so they are real
biomedical terms that simply do not appear in this sentence:

```
TEXT:            Naloxone reverses the antihypertensive effect of clonidine .

REAL gold      : [{"text": "Naloxone", "type": "Chemical"}, {"text": "clonidine", "type": "Chemical"}]
CORRUPTED gold : [{"text": "lithium", "type": "Chemical"},  {"text": "myocardial infarction", "type": "Chemical"}]
```

The corrupted demonstration is a flawless example of *how to answer* and a wrong example of *what the
answer is*. So:

- if the score **holds up**, the demonstrations were teaching format;
- if the score **collapses** back toward 0.1131, they were teaching knowledge.

This is Min et al.'s method (arXiv:2202.12837), who found for classification that *"randomly replacing
labels in the demonstrations barely hurts performance"* because demonstrations work by specifying the
label space, the input distribution and the format. Nobody had run it on exact-span extraction, where
the output space is combinatorial rather than a small label set, so the transfer was not obvious.

### The result (job 38558845, 200 eval rows)

| condition | span-F1 | empty/unparseable |
|---|---|---|
| 0-shot | 0.1147 | 130/200 |
| 1-shot | 0.4836 | 102/200 |
| 3-shot | 0.6678 | 91/200 |
| **5-shot, correct demos** | **0.7215** | 85/200 |
| **5-shot, CORRUPTED demos** | **0.5833** | 131/200 |

**Corrupted demonstrations retain 77% of the few-shot gain.**

Arithmetic: the clean gain is 0.7215 − 0.1147 = 0.6068. The corrupted gain is 0.5833 − 0.1147 = 0.4686.
0.4686 / 0.6068 = **77%**.

### What it means

**Mostly FORMAT.** Even when every demonstration's answer is factually wrong, the teacher still scores
0.5833 against 0.1147 zero-shot — five times better — purely from seeing the shape of a correct answer.
Its zero-shot score was mostly measuring whether it guessed our JSON contract and our type vocabulary,
which you can see directly in the raw output:

```
0-shot:  ```json [ { "text": "CYP", "type": "CHEMICAL" }, { "text": "P2X3", "type": "GENE_OR_PROTEIN" } ...
5-shot:  [{"text": "CYP", "type": "Chemical"}, {"text": "P2X3", "type": "Chemical"}, ...
```

Three format errors zero-shot — a markdown fence, `CHEMICAL` in the wrong case, and `GENE_OR_PROTEIN`,
a class BC5CDR does not have. The demonstrations fixed all three, and the corrupted ones fixed them just
as well, because format is exactly what survives corruption.

**And 23% is real knowledge, which is worth not overstating.** The 0.1382 gap between correct and
corrupted demonstrations (0.7215 vs 0.5833) is the demonstrations genuinely teaching the task. So this is
not "format explains everything" — it is "format is roughly three-quarters of it."

### The consequence for the synthesis decision

**A teacher's zero-shot score is not a valid gate on whether it can generate training data.** On this
task it understated usable ability by roughly 5×, and it did so for a reason that has nothing to do with
competence — our output contract.

Which means the synthesis policy I proposed earlier (block generation when the teacher's zero-shot score
is below a floor) was resting on a bad signal. The right gate, if we want one, is either:

- the **observed keep-rate** — how much of its own output the teacher's verifier rejects, which we
  already compute and which measures the thing we actually care about; or
- a **few-shot** measurement rather than a zero-shot one, since that is the regime synthesis actually
  runs in.

It also retroactively justifies making all synthesis and verification 5-shot (B281): we were running the
generator in exactly the regime this probe shows is 5× better, and we had been measuring it in the regime
that is 5× worse.

One honest limit: this is **one task, one model**. BC5CDR is unusually format-dominated — exact
`(surface, type)` multiset matching is about as unforgiving a contract as the suite has. On
`calendar_json`, where the difficulty is a date convention rather than an output shape, the split between
format and knowledge could look quite different. The probe harness takes `--task`, so that is one command
away if it matters.

---

## 7. Code changes

| # | Change | Where |
|---|---|---|
| 1 | Eval-set cap set to 1,000 (was: whole split) | `agent/nodes/cold_start/eval_setup.py` |
| 2 | Preflight default matches the pipeline cap | `scripts/preflight_tasks.py` |
| 3 | Corrupt-label ablation mode (`--corrupt-shots`) with a format-vs-knowledge verdict | `scripts/probe_teacher_fewshot.py` |
| 4 | DialogSum dedup root cause corrected in the docstring; severity downgraded | `data/loaders/dialogsum_samsum.py` |

**Tests: 236 pass** across the touched suites. **Preflight: 7/7 PASS.**

---

## 8. Unresolved

Two, and neither is urgent.

1. **Does the synthesis gate need to exist at all?** §6 removes the signal I was going to gate on. The
   keep-rate is the better candidate and we already compute it, but nothing consumes it yet. This only
   matters once we actually want synthesis on for `calendar_json`/`xlam_bfcl`; for the classification
   tasks the label-inheriting path is safe regardless.

2. **`calendar_json` has never had a clean run.** The gold is fixed and verified, the exact verifier is
   in place, preflight passes — but the task's actual trainability is unmeasured. It is the obvious next
   run.

**Closed, not problems:**

- `xlam_bfcl`'s 0.870 goal being the teacher's own score — agreed, not a bug, we will see what it does.
- Constrained label decoding — **withdrawn** (§5), measured as not worth the change.
- DialogSum contradictory gold — **downgraded to cosmetic** (§4), ~0.2% of rows, already firewalled.
- Everything from the previous notes: label-space lockdown, length anchoring, extraction hardening,
  reshuffle and synth-fill removal, exact verifiers, calendar gold rebuild, SGD vendoring, whole-split
  eval (now capped), difficulty buckets, the `single_model` strategy, and the preflight harness.
