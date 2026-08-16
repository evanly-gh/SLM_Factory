# Locking the label vocabulary, fixing the teacher's questions, and what's still open

*2026-08-15 (second note) — implementation + plain-English status*

Companion to `2026-08-15-routerbench-contamination-qc-audit-and-stretch-goals.md` (the audit that
found these) and `2026-08-14-routerbench-postmortem-crashes-and-the-ner-teacher-baseline.md`.

This note is written to be readable without holding the codebase in your head. Part 1 is what I
changed. Part 2 is everything I did **not** change, explained plainly, with the decision each one
needs from you.

---

## Part 0 — The correction that reframes everything

In the morning note I said synthetic data was safe for classification and NER because the generated
row copies its label from a real anchor row, so the label "cannot" be wrong. **That was wrong, and
you were right to push on it.**

The argument only works when **the label is a property of the text that the generator controls**.

- **CLINC150** — the label is `accept_reservations`. Ask the teacher "write another utterance that
  means *accept reservations*" and it directly controls the thing the label describes. The label
  genuinely carries over to the new sentence. ✅
- **RouterBench** — the label `local` means *"mistral-7b answers this correctly."* The teacher writes
  a brand-new sentence. Whether `local` is still true depends on whether mistral-7b would get **that
  new sentence** right — which nobody knows, and estimating it is exactly the task the teacher only
  scores 0.53 on. Copying the anchor's label is **fabricating** it. ❌

So the rule is not "classification and NER are safe." It is:

> **Synthesis is safe only when the label describes something the generator controls.**

And the run logs already measured this without anyone noticing. The teacher's own rejection rate is
a competence thermometer:

| Task | Teacher zero-shot score | Rows kept by the teacher's own check | What it means |
|---|---|---|---|
| `clinc150` | **0.8919** | 457/500, 2076/2256 → **91%** | good teacher, usable data |
| `routerbench` | **0.5311** | 90/500, 541/2624, 261/758 → **18–34%** | mediocre teacher, mostly junk |
| `ner_bc5cdr` | 0.0999 | `requested 5693 → kept 0` → **0%** | path was dead (see 1.5) |
| `dialogsum` (generation) | — | 450/450, 250/250 → **100%** | **nothing was checking it at all** |

---

## Part 1 — What I implemented today

### 1.1 The label vocabulary is now pinned and closed

**The problem, plainly.** RouterBench has two answers: `local` and `route`. The runs trained on rows
labelled `cloud`, `on_device`, `router` and `remote` too. Here is how, step by step:

1. Mining wanted more real routing data. It tried to load RouterBench itself and failed (the dataset
   ships as a pickle file, which the loading library can't read).
2. So it fell back to searching the web and found `anasnassar/llm-query-complexity-benchmark`.
3. **That dataset has no routing labels at all.** Its labels are `LOW` / `MEDIUM` / `HIGH` — query
   *complexity* tiers over StackExchange and MMLU questions.
4. The pipeline asked Claude to translate that dataset's columns onto ours. Claude produced a
   translation table, and the code applied it with a default that **passed anything it couldn't
   translate through unchanged**.
5. That translation was re-requested on **every** mining round and isn't deterministic, so each
   round could invent a new class. You can see it accumulate: `cloud` alone for iterations 1–9,
   `+on_device` at 10, `+router` at 20, `+remote` at 21.
6. The guard meant to stop this accepted a dataset if **any** of its labels matched ours. Since some
   rows did get translated to `local`/`route`, the whole thing was let in — hallucinated classes
   included.

**What I built.** A new module, `data/label_space.py`, holding one idea: the label vocabulary is
decided **once**, at the start, from the frozen eval set — and then it is **closed**.

The eval set is the right authority because it is literally what the model is scored against. A class
that isn't in it *cannot be scored*, so training rows carrying it are useless by definition. "Widen
the vocabulary to fit the data we found" is never correct.

Four enforcement points, so no single failure lets a bad label through:

| Where | Rule |
|---|---|
| `eval_setup` | Pins the vocabulary from the frozen eval set and logs it |
| `web_acquire` (source level) | Rejects an entire dataset unless **every** label it carries is already in the vocabulary |
| `web_acquire` (row level) | Drops any individual row whose label is outside it |
| `_llm_map_dataset` | Tells Claude the exact permitted labels and that inventing one is forbidden; **and** strips any translation entry pointing at a label that doesn't exist, so it can't get through even if Claude ignores the instruction |
| quality control | Unchanged — still the last line of defence |

At run start you now see:

```
[label-space] PINNED 2 class(es) from the frozen eval set: 'local', 'route'
[label-space] this vocabulary is CLOSED — mined sources whose labels are not a subset are
              rejected, and no LLM may introduce a new class
```

**One subtlety worth knowing about,** because I got it wrong first and the tests caught it. Strict
rejection is only safe when the vocabulary is **complete**. If the pipeline is guessing the
vocabulary from whatever rows it happens to hold, that list can be missing real classes — and being
strict would then throw away perfectly good data for a class the pool simply hadn't seen yet. So:

- vocabulary from the eval set or an explicit plan → **strict** (every label must already exist);
- vocabulary merely inferred from existing rows → falls back to the older, looser check.

In real runs the pinned vocabulary is always present, so RouterBench gets the strict rule.

### 1.2 Mining now uses RouterBench itself

`web_acquire` had no entry for `routerbench`, so the mining stage had no way to load the benchmark
the task is *about*. Added the alias plus a loader branch that goes through the real pickle reader,
and made curated runs name their own benchmark (they previously carried no plan at all, so this stage
was skipped entirely and went straight to paid web discovery — even though the dataset was sitting in
the local cache).

### 1.3 The length filter can no longer be tricked into deleting real data

**The problem, plainly.** One quality check removes rows more than **3× longer than the median** row.
The trap is that "median" was computed over *the whole dataset* — so the moment mining injected a
pile of short foreign rows, the median collapsed and the cutoff came down with it, onto the real data.

Real RouterBench prompts have a median of **715** characters. The injected foreign rows: **75**. The
result:

| Dataset version | injected rows | median | cutoff | share of the *real* benchmark this deletes |
|---|---|---|---|---|
| clean | 0 | 715 | 2145 | **0.6%** |
| `v5` | 958 | 295 | 885 | 47% |
| `v10` | 1,155 | 269 | 807 | **48%** |

The same filter that removes 0.6% of clean data was removing nearly half of it. You can watch it
happen in the artifacts: the median length of *real* rows fell 572 → 432 → 396 across three rebuilds
even though the pool they're drawn from never changed — the long ones were being eaten.

**The fix:** compute the median over **trusted rows only** (the task's own real training data). The 3×
ratio is fine and unchanged — the bug was what it was measured against. Genuine outliers are still
removed; injected rows can no longer move the goalposts.

### 1.4 The teacher is now told what the labels *mean*

**This is the direct cause of the over-rejection you spotted.** The old verification prompt showed
the teacher the label and nothing else:

> *"Does this utterance genuinely belong to the 'local' class?"*

Nothing told it what `local` means here. So it read `local` as the ordinary English word and asked
itself *"is this a local-information query?"* — and a grade-school math problem honestly isn't one.
Every rejection you quoted is this one failure:

```
REJECTED [local] 'What is 15 percent of 200?'      — a math question, not related to local services
REJECTED [local] 'Which term refers to a material's ability to resist heat flow?' — a factual question
REJECTED [cloud] 'The thick fog rolled in...'      — the utterance describes fog, not clouds
```

**The teacher wasn't being too strict. It was answering a different question than the task asks.**
Given the prompt it was handed, those verdicts are correct.

Both the generator and the verifier now receive real definitions, and the verifier is explicitly told
not to reject on topic mismatch:

```
What the labels mean for THIS task (judge by these definitions, NOT by the everyday
meaning of the label word):
  - 'local': the request is simple enough that a SMALL on-device model can answer it
             correctly — judge difficulty for a small model, NOT whether the topic is
             'local' in the everyday sense of nearby/location-based
  - 'route': the request is hard enough that it should be escalated to a LARGER cloud
             model — judge difficulty, NOT whether the topic involves routing
```

Definitions live in `data/label_space.py` and are per-benchmark. Tasks whose label already *is* a
description of the text (CLINC150) correctly get none, and behave exactly as before.

**Honest caveat.** This makes the teacher ask the *right* question. It does not make the teacher
*good* at it — that question is "would a small model get this right", and the teacher scores 0.53 on
it. Expect the rejection rate to drop and the remaining labels still to be noisy. The real fix for
RouterBench is not to synthesize at all, which is the open decision in 2.1.

### 1.5 NER synthesis: never worked, and now says so

`requested 5693 new-gold row(s) -> kept 0` — every call, in every NER run.

The generator groups anchor rows by their `label` field. **NER rows have no `label`** — their answer
lives in `entities`. So the grouping was always empty and the function returned nothing before making
a single API call.

No teacher compute was wasted, but the BC5CDR curriculum ran **~7,100 rows below target** with no
explanation, and the docs claimed this path worked.

**I deliberately did not "fix" it to start generating.** The generator returns `{text, label}` and
never produces entity spans, so a row it made would either have no entities at all (dropped later) or
carry the *original* sentence's spans attached to *new* text — invented gold. Given your directive,
the correct behaviour is to stay off and be honest about it:

```
[synth] SKIPPED: NER rows have no 'label' field to anchor in-class generation, and this
generator cannot produce entity spans. NER curricula are gold-only by design — no rows
generated, no teacher calls made.
```

Worth noting BC5CDR **converged anyway** (0.8098), gold-only, in 81 minutes on a 0.6B. That is
evidence real data alone was sufficient there.

### 1.6 NER gold is visible in the logs again

The sample-prediction display read only `answer` and `label`, so NER printed a **blank** gold on every
row — in the baseline block and the fine-tuned block. That deleted the one human check of gold against
prediction from the task where it matters most, and it's why a legitimate `Baseline F1 = 0.0000` looked
like a broken harness.

Before / after:

```
gold  :
gold  : ['Famotidine':Chemical, 'delirium':Disease]
```

Rendered as exact `(surface, type)` pairs — the way the scorer actually compares them — so the display
matches what's being matched. An empty gold now shows as `[]` rather than nothing.

### 1.7 The Model Improvement Report shows the first fine-tuned score

`Baseline → Best FT` couldn't separate "fine-tuning worked" from "the search loop worked":

```
  Tier  Model                            Quant     Baseline  First FT   Best FT    Δ base  Δ search
  ----------------------------------------------------------------------------------------------------
     0  Qwen/Qwen3-0.6B                  Q4_K_M      0.0000    0.7701    0.8098   +0.8098   +0.0397
     3  Qwen/Qwen3-4B-Instruct-2507      Q8_0        0.5443    0.2675    0.7584   +0.2141   +0.4909
```

Tier 0 is now legible at a glance: **one round of fine-tuning did essentially all the work** (+0.7701),
and four further iterations added +0.0397. Tier 3 is the opposite — its first fine-tune was *worse*
than the base model (0.2675 vs 0.5443) and the search recovered +0.4909.

One implementation note: this could **not** be read from the score history, because when the zero-shot
baseline beats the first fine-tune it *becomes* iteration 1's recorded score — the two are
indistinguishable afterwards. So it's captured at the moment of measurement, before the baseline joins
the candidate pool. That's exactly the tier-3 row above.

### 1.8 Tests

**1,013 tests: 1,012 pass.** The single failure is `APPS introductory`, pre-existing and documented in
the 08-11 notes, untouched by this work.

New: `tests/test_label_space_closed.py` (19), `tests/test_length_outlier_trusted_median.py` (6),
`tests/test_synth_label_context.py` (7), `tests/eval/test_gold_display_and_first_ft.py` (10).

The length-median suite deliberately includes a test asserting the **old** behaviour would have
deleted the real rows, so the fix can't silently regress into "we just raised the threshold."

---

## Part 2 — What I have NOT changed, in plain terms

### 2.1 🔴 Should synthesis be blocked when the teacher is bad at the task? — **your call, deferred**

You said you don't want low-quality synthetic data even when the labels are legal. I agree, and the
data supports it (Part 0). I did not implement the block because the threshold and the fallback
behaviour are policy choices with real trade-offs.

**Two signals are available, and we already compute both:**

1. **The teacher's measured score.** Known before any synthesis happens, at calibration. Clean and
   predictable — `clinc150` 0.89 vs `routerbench` 0.53 vs `calendar_json` 0.22.
2. **The observed keep-rate.** How much of its own output the teacher rejects. Adapts per task
   without needing a threshold, but only measurable *after* spending the generation.

**The options as I see them:**

| Option | Upside | Downside |
|---|---|---|
| Both signals, hard block | Catches it before *and* during; strongest guarantee | Two knobs to tune |
| Teacher score only | One clean gate, fully predictable, cheapest | A task where the score misleads gets no second chance |
| Keep-rate only | Self-tuning per task, no magic number | Pays for the generation before deciding |
| Warn only | Zero behaviour risk | Doesn't actually stop bad data |

**My recommendation:** both signals, hard block. Teacher below **0.70** → synthesis off for the run;
keep-rate below **0.50** mid-run → synthesis off for the rest of it. Both env-tunable. On today's data
that turns synthesis off for `routerbench` and `calendar_json`, leaves `clinc150` untouched, and NER is
already off.

**What I need from you:** the two thresholds, or a "use your recommendation."

### 2.2 🔴 What happens when the curriculum can't reach its target size? — **your call, deferred**

Follows directly from 2.1. Blocking synthesis means the dataset stays smaller than planned.

This is **already happening and already fine**: BC5CDR ran ~7,100 rows short of an 8,929 target and
converged at 0.8098 in 81 minutes. So training on real-data-only is demonstrably not fatal — and the
8,929 target was itself a guess.

Options: accept it and lower the effective target so the warning stops being permanent noise
(my recommendation); accept it and keep the single warning line; or fail the run outright.

There's also a **separate structural oddity** worth deciding on: synth-fill runs *before* quality
control, and nothing refills afterwards. So the pipeline pads to target, QC then deletes 40% of it,
and the dataset lands under target every time by construction. Either fill after QC, or over-fill to
compensate, or accept it — but right now it's accidental rather than chosen.

### 2.3 🔴 RouterBench's labels describe a 7B model — **needs your decision, and you partly answered**

You said: *the 0.6B should be a router, sending work either to an on-device model or to the cloud;
ideally the same on-device model both routes and answers, but router-only is fine as a POC.*

Here's the tension, plainly. RouterBench's label answers **"would `mistralai/mistral-7b-chat` get this
right?"** The smallest model in the entire benchmark is 7B. Our pool is 0.6B–4B.

- If the thing on the phone is a **~7B-class model**, mistral-7b is a fine stand-in and the current
  labels are roughly right.
- If the thing on the phone is **our pool** (which is what "the same model routes and answers" implies),
  the labels answer the wrong question — 7B is over 10× the largest pool member, so many things
  labelled `local` are things a 0.6B would fail.

**Your answer points at the second case as the ideal and the first as an acceptable POC.** So I left
the boundary alone and documented it. Nothing is broken today; the numbers just mean "can a 7B do
this", not "can our model do this."

**To fix it properly** you'd relabel by running our own pool models on the prompts and marking `local`
where they actually succeed. That needs correct answers per prompt, which RouterBench doesn't ship.
Two routes:

1. **Recover answers from the source benchmarks.** Every prompt is tagged with where it came from —
   hellaswag (10,042), grade-school-math (7,450), MMLU law (1,534), ARC (1,470), winogrande (1,267),
   and 81 others — all publicly available. Match them back, then score our models. Exact, and it makes
   the label a property of *our* deployment. Maybe a day of work.
2. **Use GPT-4's stored answers as a stand-in** where GPT-4 was graded correct (84% of rows). Much
   faster, adds some noise.

**What I need from you:** whether to do this, and if so which route.

### 2.4 ⚠️ A ceiling on RouterBench that no amount of work removes

Worth stating because it bounds what "success" can mean. RouterBench records correctness as a
*fraction* (0.0, 0.1, 0.25 …) and we threshold at 0.5. So a question mistral got right **by luck** is
labelled `local`, and a nearly identical one it fumbled is `route`. That's irreducible noise, and it
caps achievable F1 below 1.0 no matter how good the model or the data.

This is not a bug to fix — it's a property of the benchmark to be aware of when reading the 0.80 goal.

### 2.5 🔴 Generation-family synthesis is completely unchecked

For summarization and function-calling, the teacher invents **both** the input and the correct output,
and **nothing verifies it** — the verification hook exists but is always `None`. Result: `450/450 kept`
every time. On `calendar_json` the teacher scores **0.2176**, and the curriculum target was 8,929 rows
against 3,250 real ones — roughly 5,700 machine-invented training targets from a model that gets the
task right 22% of the time.

This is the single riskiest thing still open, and it's the same issue as 2.1 in a place where the label
copying trick doesn't even apply. Whatever you decide in 2.1 should cover this path too.

### 2.6 🔴 Other open items from the morning audit

Unchanged, in rough priority order:

| Item | Plain description |
|---|---|
| **`route` mode collapse** | Five iterations where the model answered `route` for *everything*. The score correctly reads 0.0000, but nothing detects the collapse, so full train→quantize→eval cycles get spent on it. Should be flagged as its own diagnosis. |
| **`calendar_json` has two different baselines** | The same model on the same eval set scores 0.2176 one way and 0.0000 another. The orchestrator optimises against whichever it's shown. Its eval set is also 478 rows against a target of 800. |
| **`calendar_json` gold requires guessing the year** | 82% of its answers need rolling the date forward to 2027, a convention the prompt never states. Don't re-run this task until rebuilt. |
| **Orchestrator JSON truncated 20×** | Its decision got cut off at the output limit, each time costing a retry. Needs a bigger budget, not a retry. |
| **144 rollbacks / 157 re-quantizations** | Mostly discarded work. RouterBench alone threw away 103 full cycles. |
| **NER reports `medium=None`** | That eval set has only easy and hard buckets, but the orchestrator was still handed a `medium` weight — it reasoned about a bucket that doesn't exist. |
| **`calendar_json` eval data fetched live from GitHub** | Never cached, so an upstream change silently moves your eval set and past results stop being reproducible. |
| **The two deferred stretch-goal knobs** | From this morning: thresholds for raising the accuracy goal are in place, but see that note's §7.3. |

### 2.7 The 34% already in the current run

The fixes in 1.1–1.3 prevent this going forward. They do **not** clean the run that's on disk:
`slm-routerbench-l40s-38493142` has ~1,155 foreign rows with invented labels — 34% of its curriculum —
in every rebuild since its first mining round. **I would not read much into that run's final number.**
A fresh run would be the first clean measurement of the router task.

---

## Questions for you

1. **Synthesis quality gate (2.1)** — my recommendation is teacher-score floor 0.70 plus keep-rate
   floor 0.50, both hard blocks and both env-tunable. Take it, or set your own numbers?
2. **Below-target curriculum (2.2)** — accept it and lower the effective target (recommended), or keep
   the loud warning as-is? And separately: should synth-fill move to *after* quality control so the
   dataset actually reaches target?
3. **RouterBench relabelling (2.3)** — worth doing now against our own pool? If yes: recover answers
   from the 86 source benchmarks (exact, ~a day), or use GPT-4's stored answers (fast, noisier)?
4. **Generation-family synthesis (2.5)** — block it under the same gate as 2.1, or leave it running
   unverified for now?
5. **Restart RouterBench?** Its current curriculum is a third contaminated. Kill and restart clean, or
   let it finish for comparison?
6. **Label definitions** — I wrote them for `routerbench` and `medqa`. Do you want them for the other
   tasks too, and does my wording of `local`/`route` match your intent?
