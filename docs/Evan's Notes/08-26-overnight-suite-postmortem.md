# Overnight suite postmortem: 3 runs, 8 issues

**Date:** 2026-08-26
**Companions:** `08-25-toolbench-bringup-and-overnight-suite.md`, `08-23-calendar-variance-sms-spam-small-models.md`

## Results

| run | outcome | best score | rerun? |
|---|---|---|---|
| `ner_bc5cdr` 38832588 | FAILED @ 1h42 | **0.7302** (goal 0.80) | **yes** — killed by false rejections |
| `calendar_json` 38832587 | COMPLETED @ 17h51 | **0.8262** (goal 0.80) ✓ | no — result stands, curve does not |
| `toolbench` 38832586 | FAILED @ 21h38 | **0.0908** | **yes** — synthesis contributed nothing |

## The issues, numbered

1. **The teacher verifier rejects rows the exact verifier just accepted.** NER: 25/25 accepted by
   computation, 25/25 rejected by the teacher, twice — this stopped the run. Calendar: 22.6% rejected
   (1,965 of 8,704) with provably wrong reasons. **Fixed two ways** (§1).
2. **Calendar's score is really a coin flip on whether the model stops generating.** Correct answers
   need 76 tokens of a 256 budget; on bad iterations the model never emits EOS, blows the budget, and
   the truncated JSON does not parse. Score ≈ `format_valid × 0.8`. **Not fixed — root cause needs one
   more check** (§2).
3. **Toolbench synthesis produced literally nothing**: 0 rows kept from 1,019 generation attempts,
   because I had set it to zero-shot. **Fixed** (§3).
4. **Synthetic calendar data is mode-collapsed**: 28.4% unique summaries vs gold's 77.5%; "call the
   dentist" appears 175 times. Individually correct, collectively useless. **Not fixed** (§4).
5. **The synthesis gate was reading the wrong number** — best of 0-shot and 5-shot, when synthesis
   prompts 5-shot. **Fixed** (§5).
6. **A parser regression I introduced** understated toolbench's teacher numbers ~2.3x. **Fixed**
   (already covered in the 08-25 note).
7. **NER hit its data ceiling**: `entity_diversity(cap=3)` removes 2,321 of 5,000 rows, and BC5CDR is
   a fixed corpus, so mining had nothing left. Working as designed, but it means data is not a
   available lever on that task (§6).
8. **Toolbench costs 16x NER per run** and the eval is the fixable half (§7).
9. **The run-health guard stopped runs for the wrong reason.** An empty mining round counted toward a
   kill budget, so two exploratory rounds that correctly found nothing could end a run with hours of
   budget left. **Fixed** (§10).

Terms I used loosely and you asked me to pin down — oscillation, repeated summaries, wrong argument
rows, sorted keys, and why xlam is mentioned at all — are in §8.

Suite: **1608 passing.**

---

## 1. Issue 1 — the teacher verifier

**NER, the clearest case.** The programmatic verifier checks the property that matters and cannot be
faked: every labelled span must appear *verbatim* in the abstract. It passed 25 of 25. The teacher
then rejected 25 of 25:

```
[synth] answer verification: 25/25 (100%)          <- verify_ner_row, exact
[verify] teacher validated 0/25; rejected 25
  REJECTED 'The administration of aspirin significantly reduced the incidence of m'
     — teacher: Output format is invalid; must be a JSON object with 'text' and 'entities' fields
```

The abstracts are textbook BC5CDR. The "format" the teacher objected to is an artifact of the prompt
showing it the entity *list* as the proposed answer rather than the whole row. A 100% rejection rate
is never a quality signal — it means the two sides are describing different objects. `run_health`
said exactly that and stopped the run, 0.07 short of goal.

**Calendar, the same mechanism.** 22.6% rejected, including:

```
"Incorrect end time; 7pm start plus 60 mins is 8pm, not 20:00."    8pm IS 20:00
"Start time should be 20:00, not 21:00, as 8pm is 8:00 PM."        x5, same confusion
"...08:00 + 60m = 09:00. Correct."                                 rejected anyway
6x  rejections of the year-roll the gold itself uses
```

The teacher does not know 8 pm and 20:00 are the same time, and does not know the year-roll
convention, so it overrules a computation that had already checked both.

### What I did about it

**First, `verifier_notes` on `TaskSpec`** — task-level conventions the teacher is told before it
judges. Calendar gets the 12h/24h equivalence, the 60-minute default, tonight=20:00, bare-date=09:00,
the forward year-roll, and the summary rule. NER gets told it is being shown an entity list, which
properties are already machine-checked, and that its only job is judging whether the right chemicals
and diseases were labelled. No default on the field, so all ten tasks state it.

**Second, and this is the answer to "are you confident":** *no, I am not.* I verified the notes reach
the prompt, and that the text is factually right. I have **not** verified that the teacher's rejection
rate actually drops, and that cannot be established by a unit test — it is "give an LLM better
instructions and hope it complies". A false rejection is expensive twice over: it discards a good row
*and* it can stop the run.

So I took your preference: **`SLM_VERIFY_SYNTH=0` on the four tasks that have an exact verifier**
(`calendar_json`, `ner_bc5cdr`, `toolbench`, `xlam_bfcl`). Those tasks already check every decidable
property of a generated row by computation. The teacher pass was only ever meant to add semantic
judgement on top, and on measurement it subtracts. It stays available (`SLM_VERIFY_SYNTH=1`) so the
notes can be measured against a live run later, but it is off by default for these four now.

Tasks without an exact verifier (gsm8k, dialogsum, the classification three) keep the teacher pass —
it is the only check they have.

---

## 2. Issue 2 — calendar, and where my earlier explanation was wrong

**I said the format collapse was caused by mode-collapsed synthetic data. You pushed back that mode
collapse is a content problem, not a format problem. You were right, and the data agrees with you.**

Here is the tier-3 table with the intervention that preceded each iteration:

```
  iter  content  format   intervention
   1    0.7701   1.0000   surgical_synthesis     <- best format
   2    0.0019   0.0019   surgical_synthesis     <- worst format
   3    0.6692   0.8505   hyperparameter
   4    0.1664   0.4187   surgical_synthesis
   5    0.3327   0.4299   hyperparameter
   6    0.0953   0.9981   hyperparameter
   7    0.8262   1.0000   surgical_synthesis     <- best format
```

Synthesis precedes the **best** format score and the **worst**. There is no correlation. My §2 claim
was an overreach — I had two real findings (the score tracks format; the synthetic data is
mode-collapsed) and asserted a causal link between them that the numbers do not support.

### What is actually happening

I pulled the raw outputs from the collapsed iteration (format 0.0019):

```
gold  : [{"arguments": {"end": {"dateTime": "2026-11-28T09:00:00"}, …
raw   : [{"arguments": {"end": {"dateTime": "2026-11-28T09:00:00"}, "start": {"dateTime": "2026-11-28T08:00:00"}, "sum…
parsed:                                                                              <- EMPTY
```

The output is **correct** as far as you can see it — right schema, right nested `dateTime`, right
date. And it does not parse. So the failure is in the tail, not the shape.

Then I measured how long a correct answer actually is:

```
calendar max_new_tokens = 256
gold answer tokens: median 76   p90 85   p99 95   max 99
gold answers over budget: 0/300      headroom at p99: 161 tokens
```

A correct answer needs 76 tokens and has 161 to spare. So an output that gets truncated at 256 is one
where **the model never stopped generating** — it wrote a correct call and then kept going, and the
captured text is an incomplete JSON array that cannot parse.

**So the real issue is termination, not format and not content.** Whether a given LoRA fit learns to
emit the stop token is close to binary, and it is uncorrelated with the intervention — which is
exactly the pattern in the table. When the model stops: ~0.8. When it does not: ~0.002.

That also explains the pieces that confused me:

- **Why tiers 1–2 were flat at ~0.** Different cause entirely. SmolLM2-360M reached format 0.89 with
  content 0.0000 — it terminates fine and gets the *dates* wrong. That is a capability wall: resolve a
  relative date against a stated reference instant, add exactly 60 minutes, roll the year forward.
  40 iterations cannot buy a capability the model lacks.
- **Why tier 3 looks unstable.** Qwen3.5-2B can do the task. It is unstable at stopping.

**Not fixed, and here is the honest next step.** The obvious suspect is whether the stop token is
inside the trained completion span — if the target is masked before EOS, the model never learns to
stop and whether it does becomes luck. That is one focused read of `_build_completion_only_rows` and
the completion mask, and I did not want to assert it without doing that read properly. Raising
`max_new_tokens` would *not* help: a model that does not stop just rambles longer.

---

## 3. Issue 3 — toolbench synthesis kept nothing

```
[synth] new-correct (zero-shot): 0/318 kept (636 attempts)
[synth] new-correct (zero-shot): 0/163 kept (326 attempts)
[synth] new-correct (zero-shot): 0/25  kept  (57 attempts)
```

**What that line means:** the rebuild asked for 25 rows; the generator plans up to `n×4` calls and
made 57; of the rows returned, zero survived `verify_toolbench_row` (parses as Action/Action Input
turns, calls only declared APIs, valid JSON arguments, ends in `Finish->give_answer`, within the call
budget). Zero from 1,019 attempts across three rebuilds.

**This one was my fault.** I set `SLM_SYNTH_SHOTS=0` for toolbench because five *random* full-prompt
demonstrations are ~11,500 tokens against a teacher served at 8,192 and every request returned HTTP
400. Zero-shot was the wrong cure — a format-bound task cannot be synthesized without showing the
format (B276), and the verifier correctly discarded everything.

**Fixed** by choosing demonstrations that *fit* rather than sampling at random:
`fit_demonstrations` takes the shortest suitable rows. ToolBench's shortest complete paths are ~809
tokens, so five are ~4,000 and fit inside 8,192 with room for the question. Five-shot is restored on
every task, and the same rule now applies to the generation prompt and the verification prompt.

---

## 4. Issue 4 — the synthetic data itself

Your doubt was right, just not where you expected. Each row I checked is **correct**:

```
"Remind me to pick up the dry cleaning tomorrow at 5 pm."   ref 2026-03-15T14:30
  -> start 2026-03-16T17:00  end 18:00  summary "pick up the dry cleaning"     CORRECT
```

Tomorrow resolved, 5 pm → 17:00, +60 minutes, imperative stripped. The problem is variety:

```
              n      unique requests    unique summaries
  GOLD     3000     2990 (99.7%)       2326 (77.5%)
  SYNTH    1048      750 (71.6%)        298 (28.4%)

  175x "call the dentist"   104x "buy groceries"   99x "pick up the dry cleaning"
```

Top 8 summaries are 59% of synthetic rows; synthetic rows are 26% of the final curriculum. And
calendar deliberately declares **no `dedup_surface`** — correct for gold (calendar utterances are
templated and a trigram filter deletes legitimately distinct events) and wrong for mode-collapsed
synthetic rows, with nothing distinguishing the two.

**Not fixed.** The fix is a diversity cap on *synthetic* rows only, like `entity_diversity` for NER —
cap repeats of the same `summary` at ~3, leave gold alone. It wants its own measurement run.

To be clear about what this does and does not explain: it is a real defect in the data, and it is
*not* the cause of issue 2.

---

## 5. Issue 5 — the fitness gate, and what each number is for

You read the log right. Per task:

```
                gate measurement          rows       accuracy-goal baseline (0-shot)
 toolbench      0-shot 0.0050             200/760    0.0171
 calendar_json  5-shot 0.5400             200/535    0.2822
 ner_bc5cdr     5-shot 0.6140             200/1000   0.1011
```

- **200 rows is upstream's default** (`SLM_TEACHER_FITNESS_ROWS`), not something I reduced — 200 rows
  put the standard error near 3 points, far tighter than a go/no-go decision needs.
- **Calendar and NER were 5-shot as designed.** Only toolbench was 0-shot, for the context reason in
  §3.
- **The 28% you saw is calendar's accuracy-goal baseline**, which is *supposed* to be zero-shot. Its
  synthesis decision used the 5-shot 0.5400, which is the right number for that decision.
- **0.0050 vs 0.0171 are two different measurements**, not an inconsistency: fitness on 200 rows vs
  the goal baseline on all 760.

**Changed to your spec:** the goal stays zero-shot (already was); the gate now reads the **5-shot**
number rather than best-of, because synthesis prompts 5-shot and a teacher authorised on a zero-shot
score it will never reproduce is authorised on the wrong evidence. Both numbers are still measured
and logged, and when zero-shot is higher the log says so and points at B320, since that remains the
signature of a prompt-assembly defect.

---

## 6. Issue 7 — NER's data ceiling

`entity_diversity(cap=3)` removes **2,321 of 5,000** initial rows ("entity surface form already
present 3 times"). That is the cap doing its job — BC5CDR abstracts repeat the same drug names — but
it leaves 2,603 rows, and BC5CDR is a fixed corpus, so mining is exhausted almost immediately. Two
rebuilds added 0 rows.

The orchestrator handled this well rather than thrashing:

> "Two consecutive hyperparameter escalations (rank16→32→64, ep3→4→5) show the axis is exhausted:
> rank32/ep4 gave +0.0168 but rank64/ep5 regressed −0.0361."

It reached 0.7302 with both levers spent. Synthesis was the only remaining one, and issue 1 broke it.

---

## 7. Issue 8 — timing

```
              train                    eval                total worker
 toolbench   7 x 129.3min = 15.1h    8 x 41.1min = 5.5h      20.8h
 calendar   47 x  11.5min =  9.0h   52 x  3.1min = 2.7h      12.9h
 ner         9 x   4.5min =  0.7h   12 x  2.8min = 0.6h       1.3h
```

One toolbench iteration (~2.9h) costs more than the entire NER run. Two structural multipliers:

- **Rows are 5–24x longer.** Every prompt carries the full callable API list, which *is* most of the
  prompt: mean 2,408 tokens vs xlam's 493 and BC5CDR's 102. Training reads 12.0M tokens vs 306K.
- **Eval writes 6x more, one token at a time.** 1,536 output tokens vs 512/256, and each token needs
  its own forward pass: 760 rows / batch 16 × 1,536 = 73,728 sequential passes at ~57 ms.

Improvements, best return first:

1. **`max_new_tokens` 1536 → 1024.** Gold-target p90 is 995 tokens, so almost no truncation cost, and
   eval drops ~33%.
2. **Route eval through the idle vLLM server.** The GGUF path is single-context by necessity (8
   contexts killed runs 38734202/3 with a `GGML_ASSERT` abort for ~20% gain), so the fix is a
   batching engine, not more contexts. The teacher's server sits idle during every student eval.
3. **Cap the curriculum.** 5,000 rows × 2,408 tokens × 3 epochs is the 129-minute train.

---

## 8. The terms I used loosely, defined

I used four pieces of shorthand in chat without defining them. Each hid a real question.

### "Oscillation"

Bad word, and it implied something false. Oscillation suggests a system swinging between two states
with some period — which would point at a feedback loop, and would mean the fix is damping. That is
not what calendar does.

What the numbers show is **a per-fit coin flip.** Each iteration trains a fresh LoRA fit, and each
fit independently either learns to emit the stop token or does not. When it does, the score is ~0.8;
when it does not, ~0.002. Nothing carries over between iterations, and the outcome does not depend on
what the previous iteration scored. Look at the sequence again with that reading:

```
  iter   score    stopped?
   1    0.7701     yes
   2    0.0019     no
   3    0.6692     mostly
   4    0.1664     mostly not
   5    0.3327     mostly not
   6    0.0953     yes, but wrong content
   7    0.8262     yes
```

So "the score oscillates" should have been **"roughly one fit in three fails to learn EOS, and a fit
that fails scores near zero regardless of its data."** That matters because the two framings have
different fixes: damping a feedback loop is pointless here, while making termination reliable — the
completion-mask read in §2 — fixes every iteration at once.

The reason this masquerades as instability is that the reported score multiplies two nearly
independent things: `score ≈ format_valid × content_accuracy`. `content_accuracy` is stable at ~0.8
on tier 3. `format_valid` is the coin flip. A stable factor times a binary factor looks like noise.

### "Repeated summaries", and why a cap

`summary` is a field in the gold answer — the event title in the calendar call:

```json
{"name": "calendar.events.insert", "arguments": {"summary": "call the dentist", ...}}
```

"Repeated summaries" means the teacher generated the same event over and over: 175 rows whose summary
is `call the dentist`, 104 `buy groceries`, 99 `pick up the dry cleaning`. The top 8 summaries are 59%
of all synthetic rows. Each row is individually correct — that is what makes it slip through, since
every gate the pipeline has asks "is this row right?" and none asks "is this row *new*?".

Why cap it: 1,048 synthetic rows carrying 298 distinct summaries teach roughly what 298 rows would,
while costing 1,048 rows' worth of training time and 26% of the curriculum's weight. Worse, the model
sees `call the dentist` 175 times against a gold distribution where nearly every summary is unique, so
it is being pushed toward memorizing a handful of phrases on exactly the field that has to generalize.

Why a *cap* rather than dedup: a few repeats are fine and realistic, so the cap is ~3 occurrences of a
given summary, which is what `entity_diversity` already does for NER surface forms. And it has to
apply to **synthetic rows only** — gold calendar utterances are templated, so a general dedup filter
deletes legitimately distinct events, which is why calendar declares no `dedup_surface` at all.

### "Wrong argument rows"

I should have said which of two things I meant, because they need opposite fixes.

1. **Structurally wrong** — the JSON does not carry the keys the scorer reads: a missing `name`, an
   `arguments` value that is a string instead of an object, a truncated array. The scorer cannot
   compare these to anything, so they count as format failures. This is what calendar's collapsed
   iterations produce, and it is a *termination* bug (§2), not an argument bug.
2. **Semantically wrong** — perfectly formed JSON whose argument VALUES do not match the request:
   `"start": "2026-03-16T05:00:00"` for "5 pm", or a summary that keeps the imperative ("Remind me to
   call the dentist" instead of "call the dentist"). The scorer can read these and marks them wrong on
   content.

When I wrote "wrong argument rows" I was talking about **type 2 in the teacher's rejection reasons** —
rows the teacher *claimed* had wrong arguments. Those turned out to be mostly the teacher's error, not
the data's: it called 8pm-vs-20:00 a mismatch (§1). The measured reality is that 100% of kept synthetic
calendar rows pass `verify_calendar_row`, which checks the datetime arithmetic exactly. So there is no
significant type-2 population — the defect in that data is repetition (above), not wrongness.

### "Sorting the keys" — what was actually wrong

Nothing is wrong with sorted keys in general. JSON objects are unordered and every parser in this
repo, including `parse_calls`, reads them by name. Key order cannot break parsing.

What it breaks is **what the model learns to emit, and in what order.** The loader built each gold
answer with `json.dumps(..., sort_keys=True)`, and `"arguments"` sorts before `"name"`:

```
prompt said:      {"name": "calendar.events.insert", "arguments": {...}}
gold answer was:  {"arguments": {...}, "name": "calendar.events.insert"}
```

Two concrete costs. First, the instruction and the target disagreed, so the model was being taught to
ignore the one worked example in its own prompt. Second — and this is the real one — it moved `name`
from the first token of the answer to the **last**, behind the entire nested `start`/`end`/`summary`
block. `name` is the field the scorer requires. On a task where roughly a third of fits fail to
terminate and get truncated mid-object, putting the required field last means a truncated output loses
exactly the field that decides whether it can be scored at all. Sorting the keys took the most
important token in the answer and placed it where truncation eats it first.

Fixed in the loader, and `scripts/audit_task_prompts.py` now checks prompt-vs-gold key order for every
task so it cannot drift back silently.

### Why xlam appears at all

Fair challenge — the overnight suite was toolbench, calendar_json, ner_bc5cdr. xlam_bfcl did not run.
It shows up twice and for different reasons, one legitimate and one not:

- **Legitimate:** it is one of the four tasks in the `SLM_VERIFY_SYNTH=0` change (§1), because that
  change is a per-task default in the registry and xlam has an exact verifier like the other three.
  Editing its default now is what stops it re-hitting issue 1 whenever it next runs. Same for the
  key-order audit: `xlam_bfcl` shares the function-call prompt and scorer with `calendar_json`, so
  auditing one without the other leaves the bug live on the untested half.
- **Not legitimate:** the token-length comparison in §7 quotes xlam's 493 tokens as a reference point.
  That is a number from an earlier run, not this suite, and I should have labelled it as such.

---

## 9. Output quality: clean

Every `input / gold / raw / parsed` block in both long runs:

```
              sample blocks   empty input   empty raw   empty gold   unparseable
 toolbench          8              0            0           0         3/24  (12%)
 calendar          52              0            0           0        51/156 (33%)
```

No empty or truncated inputs, no missing golds. Unparseable counts match the reported `format_valid`
exactly, so the harness measures what it claims.

**Toolbench's 3-round majority vote is confirmed live:** `JUDGE_ROUNDS=3`, rubric temperature 0.7,
7,209 successful judge calls, 2-of-3 → solved and 1-of-3 → unsolved. Its *teacher* numbers are
understated ~2.3x by issue 6; the *student* trajectory (0.0000 → 0.0908) is unaffected, because the
fine-tuned model imitates gold's same-line `Action:` format rather than the prompt's.

---

## 10. Issue 9 — the intervention ladder now retires routes instead of killing runs

The old guard counted *any* empty rebuild toward one budget and stopped the run at four. That is the
wrong shape, because the two data routes fail for opposite reasons:

- **Mining finding nothing is an answer.** BC5CDR is a fixed corpus; once `entity_diversity` has taken
  its share there is genuinely nothing left. Asking again cannot change that, and the run still has
  synthesis and hyperparameter tuning available.
- **Synthesis producing nothing is a malfunction.** The teacher was reachable, a category was chosen,
  and nothing survived — so the generator and the verifier disagree about the contract, and they will
  disagree identically next turn.

New behaviour, per your spec:

| route | empty rounds | action |
|---|---|---|
| `mine_new_real` | 2 | **retired for the rest of the run**, run continues |
| `mine_new_real` | 3 shutouts (candidates seen, none accepted) | retired, run continues |
| `surgical_synthesis` | 3 consecutive | **run stops** |
| both closed | — | every remaining turn takes `hyperparameter` |

The retirement is written to run state (`mining_retired_reason`) rather than only logged, which is the
part that makes it real: `data_rebuild.mining_available_for_state` reads that key, so a retired route
stops being *offered* to the orchestrator instead of being offered and then refused by the validator.
That check deliberately overrides the `source_progress` bookkeeping — evidence from actually running
mining beats the optimistic assumption that a corpus of unknown length still has rows. When both
routes are closed, `data_rebuild_available` goes false, and the existing `iterate` path already tells
the orchestrator that a data plan will be rejected and to choose `hyperparameter`.

`agent/run_health.py`, `agent/data_rebuild.py`, `agent/nodes/iterate.py`. Thresholds are
`SLM_MAX_EMPTY_MINING` (2) and `SLM_MAX_EMPTY_SYNTHESIS` (3).

---

## 11. Routing eval through a vLLM engine

**The idea in §7 cannot be done as written**, and it is worth recording why so it is not re-proposed:

- The teacher's server hosts Qwen3.6-35B-A3B. A LoRA adapter only applies to the base model it was
  trained against, so a SmolLM2 or Gemma adapter cannot be loaded into it at any price.
- That card is at 0.90 memory utilization by design — the 2-GPU profile gives the teacher its own GPU
  and spends the headroom on KV cache — so there is no room to stand a second engine beside it.

The useful half is still true: a GPU is idle. So `eval/student_server.py` starts its **own** vLLM
server for the student, uses it, and shuts it down. It defaults to the *pipeline* GPU, not the
teacher's, because under `SLM_CUDA_ISOLATION=1` training ran in a disposable worker that has already
exited — that card is genuinely empty at eval time, while the teacher's is committed.

Why this should be faster at all: the current path pads 4–16 rows into one batch and waits for the
slowest row. On toolbench, output lengths vary by more than 10x, so most of those 74,000 forward passes
are computing padding. Continuous batching retires a finished sequence and admits the next one.

**Parity is the whole correctness argument**, since a faster eval that scores differently is a
different experiment. The served path renders prompts with `_render_inference_prompt` — the same
function the in-process path calls — and posts them to `/v1/completions` as raw text, so the prompt
bytes are identical by construction rather than by inspection. Sampling is `temperature=0` to match
`do_sample=False`, and decoding skips special tokens to match `tokenizer.decode(...)`.

That argument covers the BF16/adapter path only. It does **not** cover `SLM_QUANT_EVAL=1`, where the
in-process path scores a Q4_K_M GGUF through llama.cpp's own chat template — a different artifact under
a different template. Since the overnight runs all set `SLM_QUANT_EVAL=1`, that is the case that would
actually need to change for the toolbench saving to land, and swapping it is a measurement change
rather than a speedup.

So this ships **off** (`SLM_EVAL_BACKEND=auto`) and is gated on measurement, not on being faster:
`scripts/compare_eval_backends.py` runs both backends over the same rows and prints agreement, the
metric delta, and the speedup, refusing the swap if the metric moves by more than 0.01. Verified per
(task, model) pair via `tests/pipeline/verify_eval_backend.slurm`. If the probe fails for any reason
the harness logs it and falls back in-process — a slower eval is a far better outcome than a dead run.

---

## 12. Run 38985393 — the toolbench synthesis fix was necessary but not sufficient

Stopped at 6h20 for review. It answered the question it was launched to answer, and the answer was no.

**What worked.** `fit_demonstrations` is engaged and synthesis is five-shot again — the log reads
`[synth] new-correct (5-shot)` where the previous run read `(zero-shot)`. `SLM_EVAL_SIZE_CAP=500` cut
the eval to 496 rows. The student is healthy: baseline 0.0000 → iteration 1 **0.0685** with
`format_valid=0.9476`, and the failure taxonomy is informative rather than degenerate
(`no_finish_call` 357, `judged_unsolved` 74, `unparseable_path` 26).

**What did not.** Synthesis kept **0 rows again** — 0/80, 0/28, 0/25 — and the log said why:

```
[synth] new-correct generation: only 1 of 5 demonstration(s) available
[synth] new-correct generation: only 2 of 5 demonstration(s) available
```

Measured against that run's own 4,995-row curriculum artifact:

```
  cheapest demonstration   3,383 chars       median   8,582 chars
  five cheapest            17,814 chars  =  7,126 tokens
  demo budget at 8192      10,240 chars  =  4,096 tokens   -> 2 of 5 fit
  five demos + question + 1,536 output reserve            ~= 12,100 tokens
```

So `fit_demonstrations` did exactly what it was written to do and still could not reach five.
**Shortest-first cannot shrink an API list.** A toolbench demonstration is a complete prompt, and the
callable API list is most of that prompt, so demonstrations are ~2,300 tokens each irreducibly. My §3
claim that "toolbench's shortest complete paths are ~809 tokens, so five are ~4,000 and fit" measured
the *answer* — the solution path — and ignored the prompt the answer hangs off. That was the error.

**The real constraint is the served context, and it always was.** Fixed by raising it for this task:
`SLM_SYNTH_MAX_MODEL_LEN=16384` with `SLM_SYNTH_MAX_NUM_SEQS` lowered 64 → 16 to pay for the KV cache.
Not a model limit — Qwen3.6-35B-A3B declares `max_position_embeddings=262144`.

**Two lessons worth more than the fix.**

1. **Lowering the shot count is not a remedy for a prompt that does not fit, and I have now proved it
   twice.** 0 shots kept 0 of 1,019 attempts; the 1–2 shots that fit 8,192 kept 0 of 133. Both are the
   same B276 failure. The log message in `_prefix_fits` was *recommending* `SLM_SYNTH_SHOTS=0` as the
   first remedy — which is how I got here — so it now recommends raising the context and explicitly
   names both failed attempts.
2. **The `_l40s` and `_cse` launchers had silently diverged.** `run_toolbench_l40s.slurm` carried
   `SLM_VERIFY_SYNTH=0` and `SLM_EVAL_SIZE_CAP=500`; its CSE twin carried neither, and the same was
   true for calendar, ner and xlam. Since which account a task lands on is decided by whichever quota
   frees first, that made the settings a lottery. `SLM_TEACHER_SYNTH_BYPASS` was worse — run 38832586
   had it on from my shell, so the flag lived in a terminal rather than the repo and neither launcher
   mentioned it. Both are now pinned in both twins and held there by
   `test_a_task_is_configured_identically_on_both_accounts`.

### Status

| job | task | account | state |
|---|---|---|---|
| 39041380 | toolbench (context fix) | cse | queued |
| 39040698 | sms_spam | intelligentsystems | queued |
| 39040699 | clinc150 | intelligentsystems | queued |

Both accounts are saturated by other users — `gpu-l40s-intelligentsystems` caps at 10 GPUs with ~9 in
use — so the three are spread one/two across the pools rather than stacked in one queue.

`sms_spam` and `clinc150` both pass `scripts/preflight_tasks.py`. The property worth stating for
sms_spam, whose harness has not run live before: its metric is `minority_f1` and a degenerate
always-`ham` predictor scores **0.0000** on it, against an eval split of 872 ham / 128 spam. A metric
that rewarded collapse could not detect collapse, and this one does not.

The eval-backend parity probe (38985077) did not produce a verdict: it failed in the *in-process*
baseline, before vLLM was reached, on `Unsloth: Could not find a valid pad token for
HuggingFaceTB/SmolLM2-135M`. That is an artifact of comparing an untrained BASE model — a real run
always has an adapter whose saved tokenizer carries a pad token — so the probe needs a trained adapter
rather than a fix. Re-run it against one of tonight's adapters.

---

## 13. The actual root cause: `max_tokens=512` (supersedes §3, §12)

**Every explanation in §3 and §12 was wrong, and they were wrong in the same way: I believed the log
line that said the verifier rejected the rows.** Nothing was ever verified.

`data/curriculum.py` generated each synthetic row with `max_tokens=512`, hardcoded, for every task in
the registry. Measured against toolbench's own curriculum (4,995 rows, run 39041380):

```
  toolbench row serialised as JSON, which is what the generator must emit in full
      min     3,913 chars  ~=   978 tokens
      median  9,997 chars  ~= 2,499 tokens
      p90    14,678 chars  ~= 3,670 tokens
  budget                        512 tokens
  rows that could possibly fit    0 of 4,995
```

So every reply was cut off mid-object, `json.loads` raised, and a bare `except Exception: return None`
dropped the row without a word. The counter that fed `"verification rejected EVERY generated row"`
counts attempted-minus-kept and attributes the gap to verification, so a truncation upstream reads
exactly like a verifier disagreement.

This explains all three runs at once, and explains why nothing I tried helped:

| run | shots | attempts | kept | what I blamed |
|---|---|---|---|---|
| 38832586 | 0 | 1,019 | 0 | zero-shot on a format-bound task (B276) |
| 38985393 | 1–2 | 133 | 0 | demonstrations not fitting the 8,192 context |
| 39041380 | 5 | 425 | 0 | — |

**Shot count was never the variable.** Zero-, one- and five-shot all returned exactly zero, because
the limit that mattered was on the OUTPUT and none of those changes touched it. `fit_demonstrations`
and the 16,384-token context were real fixes to real problems — the prompt genuinely did not fit — but
they were never going to move this number.

Worth being precise about why my instrumentation from 2026-08-27 did not catch it either: it was aimed
one layer too low. Publishing the verifier's reason via `.checker` is correct and worth keeping, but a
truncated row never reaches the verifier, so the `[verify:exact]` breakdown would have printed nothing
even had it been loaded. (It was not loaded: run 39041380 imported its modules at 13:15 on 08-27 and
the patch landed after, and Python does not re-read source in a live process.)

### Fixed

- **`_row_output_budget(anchor, prompt)`** sizes the budget from the anchor the row must resemble —
  the prompt asks the teacher to reproduce that schema, so its serialised length is a measurement
  rather than a guess. 1.5x plus a margin, floored at the old 512 so short-answer tasks are unchanged,
  and clamped to what the served context can still return after the prompt is spent. Coverage on
  toolbench goes from **0/4,995 to 4,995/4,995**.
- **`[generate:failed]` logging**, reported separately from and above the verifier breakdown, counted
  by kind, quoting the tail of the truncated reply. An object that stops mid-string is unmistakable
  once it is visible, and the two failures have opposite fixes: raise the budget versus change the
  prompt.
- The clamp reads `SLM_SYNTH_MAX_MODEL_LEN` from the environment rather than importing
  `config.config`, which raises on any unset API key and was previously wrapped in `except: pass` —
  the same species of silent fallback as the bug itself.

### The lesson

A log line that names a stage is a hypothesis, not evidence. `"verification rejected EVERY generated
row"` was emitted by code that never called the verifier, and I quoted it back three times. The check
that would have caught this in minutes is the one now in the tests: does the output budget exceed the
size of the smallest row the task can produce?

---

## 14. All bespoke token caps replaced by one global ceiling

Nine call sites each carried a hand-picked output limit — 512, 512, 200, 160, 120, 200, 192, plus two
character clips at 1,200 and 400. None was derived from anything, and nothing checked any of them
against the data it had to carry. One of them cost three runs (§13).

`config/token_budget.py` now owns all of it:

```
SLM_MAX_OUTPUT_TOKENS = 16384        # one global ceiling
output_budget(prompt, needed_chars)  # min(ceiling, max_model_len - prompt - margin)
prompt_char_budget(fraction)         # a share of the real context, for input-side clips
```

The rule is: ask for as much as the endpoint can physically return, never less because someone
guessed. Over-asking costs nothing, because generation stops at the EOS token — a budget larger than
the reply changes neither the output nor the latency, it only removes a ceiling that could be hit.
The one real bound is arithmetic: a request for more than `max_model_len - prompt` is an HTTP 400,
which is a failure rather than a truncation, so it is computed rather than approximated.

| call site | was | now |
|---|---|---|
| synthesized row | 512 | ceiling, floored by the anchor's measured JSON length |
| chain-of-thought | 512 | ceiling |
| answer-verification verdict | 160 | ceiling |
| label-verification verdict | 120 | ceiling |
| new synthesized utterance | 200 | ceiling |
| `synth_client` default | 200 | ceiling |
| ToolEval judge verdict | 192 | ceiling |
| row shown to the brief author | 1,200 chars | `prompt_char_budget(0.20)` |
| reference examples for the verifier | 400 chars | `prompt_char_budget(0.10)` |

Verified against toolbench's real 4,995-row curriculum at a 16,384 context: **0 requests would exceed
the context**, worst-case prompt+budget 16,320 of 16,384, and **4,995/4,995 rows have room for their
full JSON** (previously 0). LOSSY caps in the audit go **6 → 0**, and
`test_no_cap_can_silently_lose_data` now requires zero rather than ratcheting down from six.

### The one cap deliberately left alone

`TaskSpec.max_new_tokens`, the student's eval reserve. It cannot be globally raised, because the
reserve and the prompt share ONE window: the input budget is `max_seq_length - max_new_tokens`, and
`eval_output_token_reserve` raises when a reserve leaves no prompt room. Raising it does not uncap the
model — it starts REFUSING rows at load time, and it moves every task's measured score and eval cost.
That is a per-task measurement decision, so it stays in the registry where it is visible rather than
being buried in plumbing.

Worth noting where that interacts with §13's open question: toolbench's dominant failure is
`no_finish_call` (281 of 455), and some unknown fraction of it may be `max_new_tokens=1536` stopping
generation before the model reaches `Finish` rather than the model declining to terminate. Same family
of bug as the 512. Checkable by counting how many of those outputs land within a token or two of
1,536, and worth doing before spending another run on capacity tuning.
