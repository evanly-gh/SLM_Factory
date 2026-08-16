# RouterBench run 38455150, why the crashed runs crashed, and what the teacher actually scored

*2026-08-14 — post-mortem*

Four questions answered: what `slm-routerbench-cse-38455150` was and why a second one is running,
whether the RouterBench data was any good and where the `0.000` scores come from, why the two
function-calling runs crashed and what it takes to fix them, and what the Qwen3.6 teacher scored on
BC5CDR NER.

Companion: `2026-08-13-overnight-four-task-run-log.md` (the campaign log) and
`2026-08-12-two-new-tasks-and-loader-blockers.md` (the pre-flight audit).

> **Follow-up 2026-08-15** — `2026-08-15-routerbench-contamination-qc-audit-and-stretch-goals.md`
> extends this note and settles several of its open items. Most importantly, §5's open item — "the
> `routerbench` acquire path substitutes a foreign dataset" — turned out to be far worse than a
> wrong-source annoyance: that dataset has **no routing labels at all** (its labels are
> `LOW`/`MEDIUM`/`HIGH` query complexity), an LLM invents `local`/`route` for it at a 50/50 rate
> against a true 30/70 base rate, and those rows are banked permanently in the train pool — 34% of
> the final curriculum (B259). They also collapse the dataset's median row length, which makes the
> length-outlier QC filter delete ~48% of the *real* benchmark data on every rebuild (B260). §2.2's
> conclusion that the task may be ill-posed still stands independently, and §4's teacher-baseline
> analysis is unchanged.

**The headline is a correction to my own report.** I previously told you 38455150 was a redundant
duplicate that got cancelled. It was not. It ran for 22 hours, climbed the entire model ladder,
checkpointed cleanly, and **my supervisor script killed it four minutes later** while it sat in the
queue waiting to resume. The new run exists because of that bug, not by design.

---

## 1. What `slm-routerbench-cse-38455150` was, and why a second one is running

### 1.1 It was the real run, and it was doing fine

It started 2026-08-13 08:49 and ran **79,151 s = 22.0 hours**, all the way to the top of the model
ladder:

| Tier | Model | Quant | Zero-shot baseline | Best fine-tuned | Δ |
|---|---|---|---|---|---|
| 0 | Qwen/Qwen3-0.6B | Q4_K_M | 0.4615 | 0.6486 | +0.1870 |
| 1 | Qwen/Qwen3-1.7B | Q4_K_M | 0.1685 | 0.6960 | +0.5274 |
| 2 | Qwen/Qwen3.5-2B | Q8_0 | 0.1701 | 0.6561 | +0.4860 |
| 3 | Qwen/Qwen3-4B-Instruct-2507 | Q8_0 | 0.5443 | **0.7584** | +0.2141 |

Lifetime best **0.7584** against a 0.800 threshold. 99 Claude calls, $4.99. Not converged, but a
long way from nothing, and its best tier-3 iteration had per-difficulty accuracy of easy 0.736 /
medium 0.799 / hard 0.849 — i.e. it *was* learning the task.

### 1.2 It hit the 24h wall clock and correctly requeued itself

This is the part that worked exactly as designed:

```
[06:49:06]   outcome : FAILED: _SignalInterruption: received SIGUSR1
checkpoint complete; requeueing Slurm job 38455150
*** JOB 38455150 ON g3106 CANCELLED AT 2026-08-14T06:49:45 DUE TO JOB REQUEUE ***
```

`--signal=B:USR1@7200` fired two hours before the CSE 24-hour cap, the pipeline published a
19 MB atomic `checkpoint.json`, and `_l40s_task_body.sh` called `scontrol requeue`. The "FAILED"
and "CANCELLED" wording is misleading: `_SignalInterruption: received SIGUSR1` is the checkpoint
signal, not a crash, and `CANCELLED ... DUE TO JOB REQUEUE` is how Slurm records a requeue. **The
run was healthy and had banked its state.** Requeuing put it back to PENDING with the same job ID,
so `SLM_RUN_DIR` would still resolve to the same directory and it would have resumed from that
checkpoint.

### 1.3 Then my supervisor destroyed it

From `logs/watchdog/supervisor.log`:

```
06:51:44     SUBMIT int-sys copy of routerbench (2 GPUs free)
Submitted batch job 38493142
06:54:45     routerbench: RUNNING as slm-routerbench-l40s (38493142)
06:54:45     CANCEL twin 38455150 of routerbench (superseded by running 38493142)
```

Timeline:

| Time | Event |
|---|---|
| 06:49:06 | 38455150 checkpoints after 22h and requeues → goes PENDING, releases its GPUs |
| 06:51:44 | supervisor sees int-sys has room, submits a **from-scratch** copy, 38493142 |
| 06:52:59 | 38493142 starts |
| 06:54:45 | supervisor sees routerbench RUNNING, cancels 38455150 as a "redundant twin" |

The auto-cancel rule — cancel the twins once a copy is running — is the rule you asked for and it
is right in the normal case. What it could not see is the difference between **a duplicate that
never ran** and **the primary run, three minutes into a requeue, holding 22 hours of progress**.
Both look identical to `squeue`: same task, state PENDING. It killed a 22-hour run in favour of one
that was three minutes old.

So the honest answer to "why are you running a new one": **I shouldn't be.** 38493142 is not a
deliberate re-run, it is the replacement my own tooling forced. It started from zero, and after
~6 hours its best is **0.6551** — worse than the 0.7584 that was already on disk.

### 1.4 The fix, and what is recoverable

`scripts/supervise_overnight_runs.sh` now refuses to cancel any pending twin whose run directory
contains a `checkpoint.json`, and raises `ALERT DUPLICATE-PROGRESS` instead so the choice comes to
a human. A pending job holding a durable checkpoint is worth strictly more than a fresh one, so it
is never the thing to kill.

**The checkpoint survived.** `logs/runs/slm-routerbench-cse-38455150/checkpoint.json` is intact
(19 MB, written 06:48) and `agent.checkpoint.durable_resume_available()` returns **True** on that
directory. The 22 hours are recoverable with a resubmit that pins `SLM_RUN_DIR` to it.

Whether that is *worth* doing is a real judgement call, not an obvious yes:

- **For resuming:** 0.7584 already banked versus 0.6551 now, and 22h of curriculum work plus $4.99
  of orchestrator reasoning are on disk.
- **Against:** the killed run had already reached **tier 3, the top of the ladder**, and still only
  managed 0.7584 with four iterations there. Resuming mostly buys more tier-3 iterations against a
  threshold it had not cleared. The fresh run will climb to the same ceiling and probably land in
  the same place.

Either way the more useful conclusion is already in hand and it is in §2: this task may not be
reaching 0.80 for reasons that have nothing to do with how long it runs.

---

## 2. Was the data good, and where do the `0.000` scores come from?

### 2.1 The data loaded correctly

No complaints on the mechanical side:

```
[eval_setup] loading named benchmark 'routerbench' (RouterBench): train≤3250 test≤800
[routerbench] 36497 rows; routing boundary = 'mistralai/mistral-7b-chat'
[routerbench] train=3250 test=800
[eval_setup] official train/test separation: normalized overlap=0
[eval_setup] eval set built: 800 examples (target 800)
```

All 36,497 rows read from the pickle, zero train/test overlap, 800 eval rows, and self-consistency
was verified at exactly 1.0 before launch. The loader is fine.

### 2.2 But the task is much harder than it looks, and possibly ill-posed

Here is what an actual eval row is:

```
label=route  | TASK: Solve the following grade school math problem and provide a numerical
               answer. ... Question: There are 15 trees in the grove...
label=local  | As a habitat with a small pond goes through a long drought, which of these is
               most likely to happen to many of the fish in the pond?  A) They would be unable...
```

**The prompt is an ordinary question. The label is whether a completely different model —
`mistralai/mistral-7b-chat` — happened to answer it correctly.**

In plain terms: we are not asking the small model "can you answer this?". We are asking it "would
*that other model over there* get this right?" Nothing in the text says. The model has to infer
another system's competence from the question alone, and RouterBench's correctness scores are
graded fractions (0.0, 0.1, 0.25, …) thresholded at 0.5, so a question mistral-7b got right by
luck is labelled `local` and a near-identical question it fumbled is labelled `route`. That is
irreducible label noise, and no amount of training removes it.

It is not *impossible* — harder questions really are more likely to be missed, and the tier-3 model
did reach easy 0.736 / medium 0.799 / hard 0.849. But it is a fundamentally noisier target than
the other three tasks, where the label is a property of the input itself (which entities are in
this sentence; what JSON does this request mean). This is worth saying plainly: **routerbench is
the only one of the four tasks whose label is not a function of its own input.**

Two secondary problems, both real:

**The curriculum difficulty split is badly skewed** — easy n=72, medium n=159, hard n=569. The
minority `local` signal lives mostly in the easy/medium buckets, which together are a quarter of
the data.

**Curate is still using the broken loader.** `eval_setup` uses the fixed pickle reader;
`mine_new_real_source` goes through `web_acquire`, which does not, and the log shows the
consequence:

```
[acquire] peek withmartian/routerbench (config=None) failed: No (supported) data files found
[acquire] loaded AGENTIC HF dataset 'anasnassar/llm-query-complexity-benchmark' (train=4800 test=80)
```

When the orchestrator asked for more real routing data, Stage-0 failed on the very benchmark the
task is about, agentic discovery substituted a *different* corpus, and 200 rows of it went into the
curriculum. `web_acquire._BENCHMARK_ALIASES` has no `routerbench` entry. Not yet fixed — changing
acquisition mid-run would invalidate the running trajectory.

### 2.3 The `0.000` scores, in simple terms

There are two labels, and they are unbalanced: in the eval set for 38455150, **502 `route` and 298
`local`**. For a two-label task the harness reports **minority-class F1** — the F1 of the *rarer*
label, here `local`.

So if the model gives up and answers `route` for every single row:

- it gets every `route` row right, all 502 of them,
- and every `local` row wrong, all 298,
- F1 on `local` = **0.000**, because it never once said `local`.

That is what a `0.000` means: **the model collapsed to always predicting the majority answer.** It
is not a broken metric, an empty eval set, or a crashed harness — it is the score correctly
refusing to give credit for a degenerate strategy. Plain accuracy would have flattered the same
model with 63%, which is exactly why minority-class F1 is the right metric here.

The `route`-heavy confusion confirms it: gold `local` predicted as `route` **213** times against
only **7** the other way. The model's failure is one-directional. It is not confused; it is
defaulting.

The scores in between (0.05, 0.17) are partial collapses — the model says `local` a handful of
times and mostly gets those wrong too.

For completeness, this behaviour was verified before launch: feeding the gold labels back in scores
exactly 1.0, and an all-`route` prediction scores exactly 0.0. Both controls behaved correctly, so
the zeros in the run are real model behaviour rather than instrumentation.

---

## 3. Why the two function-calling runs crashed, and what fixing them takes

### 3.1 The crash

`slm-xlam-bfcl-cse-38454799` (68 min) and `slm-calendar-json-cse-38455147` (42 min) both died with
exactly the same exception, at the first training step:

```
File "training/lora_trainer.py", line 357, in _training_turn
    raise ValueError(
ValueError: completion-only SFT does not support task_type='function_call'
```

`_training_turn` builds the (prompt, target) pair for supervised fine-tuning. It had branches for
`classification`, `NER`, `code_generation`, `generation` and `math_reasoning`, and a bare `raise`
for everything else — which meant `function_call` and `diff`.

**The pipeline could score the format-bound task types but had never been able to train them.**
Their scorers were built on 2026-08-01, complete with the `format_valid` / `content_correct` split;
this one function was never extended to match. Both runs loaded their data cleanly, measured a
baseline, and then hit the wall.

This is also the real reason `xlam_bfcl` and `coedit` had never produced a run log. The loader
breakage documented on 08-12 was true, but it was not the whole story: fixing the data just moved
the failure 68 minutes later.

### 3.2 Why the pre-flight missed it

The validation I ran before launching checked, for all four tasks, that the dataset loads, that the
eval set is non-empty, and that feeding the gold back as the prediction scores exactly 1.0. That is
the right check for data quality and it caught two genuine defects (a BFCL row whose gold calls an
undeclared function; RouterBench's list-literal prompts).

What it did not do is **execute a single training step**. A `raise` sitting on the one code path
both tasks had to traverse was invisible to every check I ran. Validating the data is not validating
the run, and the eval harness being correct says nothing about the trainer.

### 3.3 What it took to fix

Small, and it follows a pattern the codebase already uses everywhere else:

1. `build_function_call_prompt(example)` added to `eval/scorers/function_call.py` and
   `build_diff_prompt(example)` to `eval/scorers/diff.py`, with each scorer's `build_prompts`
   refactored to call it. Previously only the batch builder existed, so there was nothing for the
   trainer to import.
2. `_training_turn` grew a `function_call` / `diff` branch that **imports those builders** rather
   than reproducing the prompt text. That is the whole point: classification, NER and generation
   all import their eval-side builder, which is what makes train/serve drift impossible. Copying
   the string is how B250 happened, and how the NER prompt skew survived a 44.8-hour run.
3. The training target is the gold string the scorer already parses out of `answer` — a JSON call
   list for `function_call`, a unified diff for `diff`. An empty `answer` raises instead of quietly
   teaching the model to emit nothing.

Verified byte-identical prompts on both sides for both task types, plus 4 new tests
(`test_function_call_training_prompt_is_byte_identical_to_eval`, the `diff` equivalent, the
empty-answer guard, and an assertion that a genuinely unknown task type still raises). 122 tests
pass across the touched areas.

Both tasks resubmitted 12:22 on both accounts: `xlam_bfcl` 38505237 / 38505240, `calendar_json`
38505239 / 38505241. All four are `AssocGrpGRES` — both accounts at cap — so they wait for a
co-tenant to finish.

**One caveat on those runs, stated up front:** this is the first time the trainer has ever executed
a `function_call` step, so the first iteration is the real test. The prompts and targets are
verified, but nothing downstream of `_training_turn` — token-boundary handling for a long JSON
target, the GGUF eval path on JSON output — has been exercised for this task type before.

---

## 4. What the Qwen3.6 teacher scored on NER

**0.0999 span-F1.**

```
[baseline] scoring reference model on 800 eval rows (task_type=NER, max_new_tokens=512)
[baseline] reference span_f1=0.0999
[threshold] Qwen baseline 0.0999 → goal 0.8000 (floor 0.80)
```

That is `Qwen/Qwen3.6-35B-A3B` — the 35B teacher that serves synthesis and judging — scored
**0.0999** on the same 800 BC5CDR rows where the fine-tuned **0.6B** student scored **0.8098**.

The student beat its own teacher by roughly **8×** on this task.

Worth being precise about what that does and does not mean. It is not evidence that a 0.6B model is
better at biomedical entity recognition than a 35B one in any general sense. It is evidence that
**zero-shot prompting cannot produce this output format**, and that the task is dominated by format
compliance and span-boundary conventions rather than knowledge. The metric is exact
`(surface_text, entity_type)` multiset match: a model that identifies the right chemical but writes
`the clonidine` instead of `clonidine`, or emits prose around its JSON, or labels a span
`CHEMICAL` instead of `Chemical`, scores zero on that row. The 35B model knows perfectly well what
naloxone is; it does not know, without being shown, that this evaluation wants exactly
`[{"text": "Naloxone", "type": "Chemical"}]` and nothing else.

This is the whole argument for the project, quantified in one line: **fine-tuning a 0.6B model on
the task's own conventions beats prompting a 35B model by 8×, and the 0.6B is the one that fits on
the phone.**

### 4.1 It also explains the threshold change

The previous BC5CDR run (`slm-ner-l40s-37531245`) failed at 0.8628 against a **0.880** threshold.
This one converged at 0.8098 against **0.800**. The difference is not a policy change I made — it
is a consequence of pinning the benchmark:

| | previous run | this run |
|---|---|---|
| Path | autonomous (`TASK=` only) | curated (`SLM_BENCHMARK_TASK=ner_bc5cdr`) |
| Threshold source | **planner recall** | measured baseline + floor |
| Value | 0.880 | 0.800 (the floor; the 0.0999 baseline is far below it) |

On the autonomous path the planner picked 0.88 from its own memory of published BC5CDR numbers —
exactly the failure mode `config/benchmark_baselines.md` exists to prevent. On the curated path the
threshold comes from `measure_endpoint_baseline` plus a 0.80 floor. Because the reference model
scored 0.0999, the derived goal fell below the floor and the floor won.

So the campaign's convergence is partly a threshold artefact and I would not claim otherwise:
0.8098 is genuinely *below* the 0.8628 the older run achieved. What the threshold does **not**
explain is that this run needed **5 iterations, 81 minutes and a 0.6B model** where the old one
spent 142 iterations, 44.8 hours and a 4B. The plausible causes are the two pre-launch changes —
the train/serve prompt fix, so the model is now tuned on the exact string it is scored on, and
5,096 training rows instead of 3,403. Which of those mattered is not established here and would
need an ablation.

---

## 5. Open items

| # | Item | Status |
|---|---|---|
| 1 | Supervisor could cancel a checkpointed requeue | **fixed** — never cancels a twin holding `checkpoint.json`; raises `ALERT DUPLICATE-PROGRESS` |
| 2 | Supervisor silently ignored a task whose only run had crashed | **fixed** — `ALERT ORPHANED` with the exception attached |
| 3 | `function_call` / `diff` untrainable | **fixed** — prompt builders shared, 4 tests |
| 4 | Resume the 22h RouterBench checkpoint, or let the fresh run continue? | **open — needs a decision**, see §1.4 |
| 5 | `web_acquire` has no `routerbench` alias, so curate substitutes a foreign dataset | **open** — deliberately not changed mid-run |
| 6 | RouterBench label is not a function of its own input (§2.2) | **open, design-level** — may be why 0.80 is unreachable |
| 7 | `NOPROGRESS` false positives during GGUF quantize phases | cosmetic; poll interval is shorter than a quantize step |
| 8 | Nothing downstream of `_training_turn` has run for `function_call` | will be known on the first iteration of 38505237/38505240 |

---

## 6. Post-cancellation review (16:49)

`xlam_bfcl` 38505237 (3h51m) and `calendar_json` 38505239 (34m) cancelled on request. Both ran
long enough to be informative, and both say something the pre-flight could not have.

### 6.1 The trainer fix worked

Neither run hit `completion-only SFT does not support task_type='function_call'`. `xlam_bfcl`
completed **10 train→eval iterations** and `calendar_json` completed one. The format-bound training
path works end to end: token boundaries, GGUF quantization, and JSON-output eval all held up. §3 is
closed.

### 6.2 `xlam_bfcl`: working, but the bar is the teacher's own score

| | value |
|---|---|
| Qwen3.6-35B reference | **0.8700** ast_arg_match |
| Threshold (derived from that baseline, above the 0.80 floor) | **0.8700** |
| Qwen3-0.6B zero-shot | 0.2675 |
| Best fine-tuned (iteration 3) | **0.6900** |
| Trajectory | 0.660 → 0.669 → **0.690** → 0.666 → 0.659 → 0.464 → 0.611 → 0.616 → 0.644 → 0.657 |

This is the mirror image of NER. On BC5CDR the teacher scored 0.0999 and the floor set the goal at
0.80; here the teacher is genuinely **good** at function calling (0.87), so the threshold became
0.87 — and a 0.6B has to match a 35B to converge.

The run is not broken, it is plateaued: best at iteration 3, then nine iterations of oscillation
between 0.46 and 0.69 with no new high. It never escalated past tier 0 in 3h51m, which is the thing
worth questioning — the escalation trigger wants a 0.02 chronological gain, and noisy oscillation
around a plateau keeps producing apparent "gains" that reset the counter.

### 6.3 `calendar_json`: 0.0000 is my eval's fault, not the model's

Every one of 478 rows failed. That looks catastrophic until you read what the model actually
produced:

```
raw : [{"name": "calendar.events.insert", "arguments": {"summary": "GP appointment",
        "start": {"dateTime": "2026-03-...
gold: [{"arguments": {"end": {"dateTime": "2027-03-01T17:45:00"}, ...
```

Valid JSON, correct function name, correct nested structure, sensible summary, right month and day.
It is failing on **the year** — and that is a convention I built into the gold that the prompt never
states.

**82% of the eval set (393 of 478 rows) requires rolling the year forward to 2027.**

| | rows |
|---|---|
| gold year == reference year | 85 |
| gold year rolled to the next year | **393 (82%)** |

The cause is a systematic bias I introduced. SGD's calendar dialogues are almost all set in
**March**; my `reference_for()` scatters reference dates uniformly across 2026. So for most rows the
reference falls *after* March, `_parse_date` applies its "if the date already passed, roll to next
year" rule, and the gold lands in 2027. The model sees "on 2nd of March", a reference of
2026-08-19, and answers 2026-03-02 — which is the *more natural* reading, and is marked wrong.

A second, separate flaw in `convert_sgd_rows`: I synthesise the request as
`f"Schedule {summary} on {date} at {time}"`, so a row whose title is `Food` reads
*"Schedule Food on March 1st at 16:45"*. The model reasonably extracts `summary="Schedule Food"`.
The word I added to make it a sentence became part of the thing being extracted.

**Why the pre-flight missed both.** Self-consistency proves gold scores 1.0 against gold. It cannot
detect that the gold encodes a convention no model could infer, because the gold agrees with itself
by construction. The eval set was internally consistent and **not fair** — a distinct failure from
"the data does not load", and one I had no check for.

Three fixes are needed before this task is worth running again:

1. **Pin the reference date near the corpus's own timeframe** (SGD is March; use a reference a few
   days to weeks before the event) so the year is unambiguous, *or* state the year in the request,
   *or* score date-without-year. Any of the three removes 82% of the failures.
2. **Stop leaking "Schedule" into the summary.** Build the request from the source utterance, or
   word it so the title is unambiguous.
3. **Re-verify with a model in the loop, not just gold-vs-gold.** The right pre-flight for a
   format-bound task is: run the *reference model* zero-shot and read its failures. The teacher
   scored 0.2176 here, and looking at *why* would have surfaced the year problem in minutes.

### 6.4 A design concern this surfaced

For `function_call` the curriculum is topped up by `_synthesize_new_correct`, which asks the local
teacher to invent new correct examples. On `calendar_json` that teacher scores **0.2176**. Asking a
model that gets the task right 22% of the time to generate training targets is a way to manufacture
wrong labels at scale — and the curriculum target was 8,929 rows against 3,250 real ones.

NER avoided this by construction: its synthesis is "new in-class gold" anchored to a real label, so
the teacher never has to *solve* anything. The generation-family path has no such protection, and
`curate._verifier_for` returns `None` for these task types, so nothing checks the output. Worth
addressing before any format-bound run is trusted at scale.

### 6.5 Where things stand

| Task | Verdict |
|---|---|
| `ner_bc5cdr` | **Working.** Converged 0.8098 vs 0.800, 0.6B, 81 min. Threshold came from the floor, so the bar was soft — but the mechanism is sound. |
| `xlam_bfcl` | **Working, plateaued.** Data and training correct; 0.690 vs a 0.870 bar set by the teacher's own score. Needs faster tier escalation, not more iterations at 0.6B. |
| `calendar_json` | **Eval set needs rebuilding.** Training path fine, data loads, but 82% of gold rows encode an unguessable year. Do not re-run before §6.3. |
| `routerbench` | **Questionable at the task level.** Still running (38493142, ~10h, best 0.6551). Label is not a function of its own input (§2.2); original 22h run reached 0.7584 at tier 3 and its checkpoint is preserved. |
