# ToolBench works — and five bugs it took to get there, four of them not about ToolBench

**Date:** 2026-08-25 (live; appended as runs land)
**Companions:** `08-24-toolbench-tooleval-harness.md` (the implementation), `08-23-calendar-variance-sms-spam-small-models.md`

**The headline.** The new `toolbench` task learns, and the trajectory is the most interpretable one
this project has produced on a hard task:

```
SmolLM2-360M-Instruct @ Q4_K_M          score    format_valid
  baseline (no fine-tuning)             0.0000        0.0632
  iteration 1                           0.0461        0.9566
  iteration 2                           0.0816        0.9763
```

Read the second column first. The base model could not produce a readable solution path at all — 6%
of its outputs parsed. One iteration of fine-tuning took that to **96%**, and the score followed. The
failure categories then migrated in exactly the direction the task's design predicts:

```
                       iter 1   iter 2
  no_finish_call          533  →   425     learning to terminate
  judged_unsolved         148  →   241     now producing paths the JUDGE gets to reject
  unparseable_path         33  →    18
  undeclared_api           11  →    13     fabrication stays ~1.7%
  judge_unsure              0  →     1     the judge decides; it does not abstain
```

That migration is the thing worth having. The model first learns the FORM, then learns to terminate,
and its failures move from "malformed" to "well-formed but unconvincing" — which is the only regime
where the ToolEval judge is measuring anything. `judge_unsure` at 1/760 says the judge is deciding
rather than shrugging, and `undeclared_api` at 1.7% says almost nothing is being invented. Both were
the numbers I was most worried about when the harness went in.

**But getting there took five bugs, and only one of them was in the ToolBench code.** The other four
were pre-existing assumptions that had never been contradicted, because every prior task has short
rows, a narrow vocabulary, and a small output budget. ToolBench has none of those, so it walked into
all of them. That is the durable value of this bring-up and it is what §2 is about.

---

## 1. What is running tonight

Three concurrent runs on `gpu-l40s-intelligentsystems` (6 GPUs, 2 each), 7-day walltime with the
USR1 checkpoint/requeue contract, all `smallest_first`, all with the synthesis gate bypassed:

| job | task | why |
|---|---|---|
| 38832586 | `toolbench` | relocated off the 20h CSE box; needs the long box (§3) |
| 38832587 | `calendar_json` | the task the gate used to refuse synthesis on (0.7950 vs 0.80) |
| 38832588 | `ner_bc5cdr` | queued behind capacity |

then `clinc150` + `sms_spam` as the first two land.

The bypass is applied per-submission rather than written into the shared launchers, because it is a
bring-up tool and not a default:

```bash
sbatch --export=ALL,SLM_TEACHER_SYNTH_BYPASS=1 tests/pipeline/run_<task>_l40s.slurm
```

Verified inside the live process rather than assumed, since a silently-dropped export would have
looked exactly like the gate working:

```
SLM_BENCHMARK_TASK=toolbench   SLM_MODEL_SELECTION_STRATEGY=smallest_first
SLM_MAX_SEQ_LENGTH=8192        SLM_SYNTH_SHOTS=0        SLM_TEACHER_SYNTH_BYPASS=1
```

`scripts/monitor_runs.py` is the watch tool. It parses whole logs rather than tailing them, because
every question worth asking — did the score move plausibly, is the failure FORM or CONTENT, did the
intervention target what cost points, is the judge deciding — is a number that appears once per
iteration, thousands of lines apart.

**Preflight before submitting** (`scripts/preflight_tasks.py`, no GPU, 50s): 4/4 pass.

```
PASS  calendar_json  train=3250  eval= 535  ast_arg_match  gold=1.0  degen=0.0
PASS  ner_bc5cdr     train=3250  eval=1000       span_f1   gold=1.0  degen=0.0
PASS  clinc150       train=3250  eval=1000      macro_f1   gold=1.0  degen=0.0001
PASS  sms_spam       train=3250  eval=1000   minority_f1   gold=1.0  degen=0.0
```

Two things it surfaced that are worth knowing but not worth blocking on:

- **`calendar_json`'s eval set is 535 of a 1,000 target (54%).** Accepted by design — the loader's
  held-out split is the limit — but it means calendar scores carry materially more variance than the
  other three and are not directly comparable to them. This is the same task that swung 0.0019 →
  0.4598 on byte-identical data (see the 08-23 note), so the small eval set is not an academic
  concern.
- **`clinc150` ships 1 train row that appears verbatim in its eval set.** `curate`'s eval firewall
  removes it before training, so the run is not contaminated, but the loader should not be emitting
  it. Cosmetic tonight, a real defect if the firewall ever changes.

---

## 2. The five bugs

Ordered by how much they would have cost if they had reached a results run.

### 2.1 A shell default silently overrode a task spec

`tests/pipeline/_l40s_task_body.sh` defaults `SLM_MAX_SEQ_LENGTH` to **4096**, and
`training.slm_helpers.task_max_seq_length` gives that environment variable precedence **over**
`TaskSpec.max_seq_length`. So `toolbench`'s measured 8192 — justified in its spec with token
distributions — was clamped to 4096 at runtime, and job 38812203 died twice from it:

```
eval : 4020 tokens, exceeding input budget 2560 ... inside configured max sequence length 4096
train: row 43 contains 4110 tokens, exceeding configured context 4096
```

This is precisely the failure mode the task registry was built to eliminate — a global default
overriding a per-task decision — except that this instance lives in **shell**, which is why no
existing test caught it. Fixed by exporting 8192 in the six toolbench launchers, and guarded by
`test_no_launcher_silently_clamps_its_task_context_below_the_spec`, which fails with the exact
diagnosis if any launcher under-provisions its task. A second test pins the ordering, because
exporting after `source` parses fine and does nothing.

### 2.2 Few-shot prompting is structurally impossible on this task

`build_training_turn` returns a **complete** prompt, so k demonstrations cost k full prompts. A
ToolBench row is ~2,300 tokens, so the default 5 shots is ~11,500 tokens of prefix against a teacher
served at 8,192. All **199** five-shot fitness requests returned HTTP 400.

The wasted calls were not the damage. The gate takes the *best* of its 0-shot and k-shot
measurements, so an all-failed k-shot pass returned `format_valid=0.0000` and the code logged
**"demonstrations made this teacher WORSE on this task"** — which is the B320 wording for a
prompt-assembly defect. It pointed at the model when the cause was that the prompt did not fit.

Worse, and the part that mattered: **synthesis prompts the same way.** At 5 shots every
`surgical_synthesis` call would have failed identically and produced zero rows, silently — the exact
intervention the bypass was enabled to exercise.

Fixed two ways. `SLM_SYNTH_SHOTS=0` for toolbench, and a guard in `measure_teacher_fitness` that
checks the demonstration block against the teacher's context *before* spending calls.

**My first version of that guard was wrong in an instructive way.** I bounded the prefix by
`spec.max_seq_length` — the *student's* window — and it broke five existing tests. routerbench is the
proof: its 5-shot prefix is ~2,240 tokens against a 974-token student budget and fits the teacher's
8,192 perfectly well. Bounding a teacher prompt by the student's window would have disabled
demonstrations on exactly the tasks B276 says need them. The guard now reads
`config.SYNTH_MAX_MODEL_LEN`, which `_l40s_task_body.sh` exports as the same value it passes to
`vllm serve`, so the two cannot drift.

### 2.3 A reporting aid was spending the whole GPU budget

`eval_setup` calls `label_difficulty`, which in `zeroshot` mode runs **two full generative evals over
the entire eval set** — smallest and largest feasible base models, both BF16 — *before* the run
selects a model or trains anything. On every prior task that was cheap. On toolbench:

```
760 rows x 1,536 output tokens = 1.17M generated tokens per probe model
largest feasible model = Qwen/Qwen3.5-4B @ bf16
smallest model's pass alone = 70 minutes
```

Both runs spent their entire allocation on it and never reached a training step. The buckets it
produces feed the per-difficulty **report** — not the curriculum, not the score, not any routing
decision.

Bounded at 1,000,000 generated tokens per probe model, which leaves every other task untouched (next
highest is 512,000) and catches toolbench's 1,536,000, with `SLM_DIFFICULTY=zeroshot-force` as the
escape hatch. The bound ignores prefill, so a task that trips it is comfortably over rather than
marginally so.

**A correction to my own first assessment.** I said the 4B pass "would not finish inside the 20-hour
allocation." That was wrong, and the reason matters. Cost here is dominated by *step count*, not
FLOPs — 1,536 output tokens is 1,536 sequential forward passes no matter how big the model is — so a
4B bf16 model is perhaps 2–4x the 135M per-step time, i.e. **2.5–5 hours**, not unbounded. Skipping
was still right (4–6 hours of a 20-hour box, before model selection, on a reporting aid), but the
justification is "disproportionate", not "impossible".

### 2.4 The smallest model in the pool ran out of memory on a 44 GB card

`google/gemma-3-270m-it` OOM'd while training a 270M-parameter model. The allocation it died on was
28.69 GiB, and it is exactly the fp32 cross-entropy logits tensor:

```
micro_batch 8 x 6,006 padded tokens x 262,144 vocab x 4 bytes = 46.9 GiB
```

That model spends **168M of its 270M parameters on a 262,144-token embedding table**, so its output
projection is **5.33x wider** than SmolLM2's 49,152. The same batch on SmolLM2 needs 8.8 GiB and
trains fine — which is why nothing had caught this: the OOM needs *both* a wide vocabulary and long
sequences, and no prior task supplied the second.

So the smallest artifact in the pool is the one that cannot train, and not because it is big — because
it is *wide*. Fixed by predicting it from three numbers known before the first step and shrinking the
micro-batch to fit, with gradient accumulation scaled up so the **effective batch is preserved**:

```
[train] ⚠ micro_batch 8 → 2 (grad_accum 1 → 4): the fp32 logits tensor for google/gemma-3-270m-it
        would be 46.9 GiB ... at 2 it is 11.7 GiB. This is a WIDE-VOCABULARY model on a
        LONG-SEQUENCE task, not a large model. effective batch size unchanged.
```

Preserving the effective batch was not automatic. `TrainingConfig` computes
`effective = micro x accum` once in `__post_init__`, so shrinking the micro-batch alone would have
quietly divided the effective batch by 4 and changed the learning dynamics rather than just the
memory profile. The existing OOM recovery path also reloaded and retried with the *same* micro-batch,
so it OOM'd twice and lost the run.

### 2.5 The scorer punished models for obeying the prompt

ToolBench's own system prompt asks for a format that omits a colon, while its training data includes
one:

```
Your output should follow this format:      every gold turn reads:
Thought:                                    Thought: ...
Action            <- no colon               Action: <name>
Action Input:                               Action Input: {...}
```

My parser required the colon, so a model that obeyed the **instruction** instead of imitating the
**data** was scored `unparseable_path` — a measurement error attributed to the model. The colon is now
optional, and I verified the relaxation does not launder real failures: all three degenerate patterns
SmolLM2 actually produced (`Thought: Action Action Input: Finish:` and friends) still correctly fail
to parse, because they contain the format words with no function name.

---

## 3. Cost characterisation: why this task is an order of magnitude heavier

Measured, same three models, against the probe log `38765131`:

| task | mean tokens/row | rows | training tokens |
|---|---|---|---|
| `ner_bc5cdr` | 102 | 3,000 | 306K |
| `xlam_bfcl` | 493 | 3,000 | 1.48M |
| **`toolbench`** | **2,408** | **4,995** | **12.0M** |

Every ToolBench prompt carries the full callable API list, and that list *is* most of the prompt. So
training reads **8x** what xlam does. Eval is worse, because two factors multiply:

```
xlam:      1000 rows / batch 32 x  256 max_new =  8,000 sequential decode steps
toolbench:  760 rows / batch 16 x 1536 max_new = 73,728 sequential decode steps
```

and each toolbench step attends over a 1,250–6,656-token context instead of ~450. At the measured
57 ms/step that is the 70 minutes. A 135M model spending 57 ms per decode step is **overhead-bound,
not compute-bound** — the FLOPs are trivial and the time goes to per-step Python/kernel overhead plus
attention over a padded KV cache. Model size barely enters into it, which is the counter-intuitive
part and the reason "it's only 4B" is the wrong intuition here.

The multiplier that turns inefficiency into 70 minutes is that **a base model never emits EOS on this
task**, so it runs to the full 1,536-token ceiling on essentially every row. A fine-tuned model
terminates, which is visible in the numbers: gemma's baseline eval took **5.0 min** (short degenerate
output) while SmolLM2's fine-tuned evals took **33–45 min** (real paths).

### Measured per-iteration cost

```
SmolLM2-360M (micro_batch 8)     train  91–120 min   eval 33–45 min   ->  ~2.2 h/iteration
gemma-3-270m (micro_batch 2)     train     301 min   eval  5 min      ->  ~5.1 h/iteration
```

gemma trains **3.3x slower** because the vocab-driven micro-batch reduction means 4x the optimizer
steps. That is the price of the fix in §2.4 and it is worth stating plainly: gemma-3-270m is now
trainable on this task but not economically so.

**Consequence.** The 15-eval stagnation window needs ~32 h at SmolLM2's rate. That is why toolbench
was moved to the 7-day intsys box tonight — on the 20 h CSE box it would have terminated on the wall
clock at ~8 evals, which is not one of the three stopping conditions and would not have been a clean
result.

If a cheaper toolbench run is ever wanted, the honest lever is `max_new_tokens`: gold-target p90 is
995 tokens against the current 1,536 ceiling, so 1024 would cut eval by a third at negligible
truncation cost. I did not take it tonight because it changes the metric mid-comparison.

---

## 4. Notable, smaller

- **The GGUF eval concurrency is 1 and should stay there.** I went looking for a 16x speedup and
  found the answer already recorded at `slm_helpers.py:823`: raising it to 8 killed runs 38734202/3
  with a llama.cpp `GGML_ASSERT` → `abort()` (SIGABRT, invisible to the OOM-halving retry) for a
  measured ~20% gain, because 8 contexts on one device serialise anyway. Real throughput needs
  llama.cpp's multi-sequence decode or routing eval through the idle vLLM server. Worth doing
  eventually; it is the single largest lever on this task's cost.
- **Gemma emits a duplicate `<bos>`.** `llama_cpp` warns `Detected duplicate leading "<bos>" in
  prompt` on every gemma eval row — its chat template adds one and llama.cpp adds another. Harmless
  to throughput, unquantified effect on quality, and it only shows up for gemma.
- **`gemma-3-270m` needs a full-size SWA cache under llama.cpp** (`llama_kv_cache_iswa: using
  full-size SWA cache`), so its interleaved sliding-window attention gives up the memory saving it
  exists for at an 8192 context.
- **The ToolEval judge behaves.** 24 judge calls on a 760-row eval, all successful — because rows
  that fail an exact rule (no `Finish`, gave up, over budget) never reach the judge by design. When
  the model started producing real paths, judged rows rose accordingly. That is the cost control
  working as intended rather than the judge being unused.
- **The teacher scores 0.0184–0.0250 on ToolBench zero-shot** with `format_valid=1.0000`. So the goal
  correctly falls back to the 0.80 floor, and the reference model — a 35B MoE — solves ~2% of this
  benchmark in one pass. Useful context for reading any small-model number on it.

---

## 5. Overnight monitoring: what the three runs are actually doing

All three `smallest_first` runs independently chose **SmolLM2-360M-Instruct @ Q4_K_M** from Tier 1,
each with a reason grounded in the right measured evidence:

```
ner_bc5cdr    "Highest measured ner_bc5cdr span-F1 (73.4) among Tier 1 candidates, with lossless
               Q4_K_M quantization giving the best structured-extraction..."
calendar_json "calendar_json requires structured/compositional output like function-calling, and
               SmolLM2-360M-Instruct scores far higher..."
toolbench     "For toolbench-style function-calling/JSON adherence, SmolLM2-360M-Instruct shows by
               far the highest xlam_bfcl ast_arg_match..."
```

That is the model-selection path working: it read `config/model_capabilities.md`, and it did not fall
for the fact that gemma-3-270m is 17 MB smaller — which the capability doc explicitly warns costs
~0.35 on structured output.

**The bypass is doing what it was turned on for.** All three teachers fail the 0.80 gate, and all
three runs are using synthesis anyway:

```
ner_bc5cdr     0-shot 0.1302   5-shot 0.6140     -> surgical_synthesis(600) fired
calendar_json  0-shot 0.3000   5-shot 0.5400     -> surgical_synthesis(700) fired
toolbench      0-shot 0.0050   (k-shot skipped, §2.2)
```

Note the 0-shot vs 5-shot gap on the two non-toolbench tasks — 0.13 → 0.61 and 0.30 → 0.54. That is
B276 in the wild, and it is why bounding the demonstration guard by the TEACHER's context rather
than the student's (§2.2) mattered: had I shipped my first version, both of these would have lost
their demonstrations and the gate would have read the depressed 0-shot number.

### 5.1 `ner_bc5cdr` — healthy, at its data ceiling

Seven iterations in 85 minutes (train ~6 min, eval ~10 min — two orders of magnitude cheaper than
toolbench):

```
baseline  0.0000  fv 0.7260
iter 1    0.6885  fv 0.9900   Δ +0.6885
iter 2    0.6517  fv 0.9790   Δ -0.0368
iter 3    0.7054  fv 0.9790   Δ +0.0537
iter 4    0.6693  fv 0.9930   Δ -0.0361
iter 5    0.7144  fv 0.9950   Δ +0.0451
iter 6    0.7144  fv 0.9950   Δ  0.0000
iter 7    0.7169  fv 0.9980   Δ +0.0025
```

The +0.6885 first jump triggered my monitor's alert and is the expected base→fine-tuned step: the
base model does not know the output contract (`fv` 0.726 → 0.990). Failures are a sensible NER
taxonomy — `wrong_span_boundaries (148), partial_span_set (127), wrong_entity_type (105),
entities_hallucinated (23)`.

**Two things worth recording.** First, quality control removes **2321 of 5000** initial rows —
`entity-diversity: entity surface form already present 3 times`. That is the declared cap doing
exactly its job (BC5CDR abstracts repeat the same drug names constantly), but it means the effective
curriculum is 2603 and it caps how much mining can ever add. Second, and following from it, the
corpus is now genuinely **exhausted**: two rebuilds added 0 rows, `run_health` logged
`WASTED ITERATION (2/4)`, and the orchestrator said so plainly rather than thrashing:

> "Two consecutive hyperparameter escalations (rank16→32→64, ep3→4→5) show the axis is exhausted:
> rank32/ep4 gave +0.0168 but rank64/ep5 regressed −0.0361, meaning the axis is exhausted."

That is the intervention loop reading its own per-difficulty report, tracking which levers it has
already pulled, and noticing a regression. It is the healthiest orchestrator reasoning I have seen
in these logs. Plateaued at 0.7169 against a 0.80 goal, and it should terminate on stagnation.

### 5.2 `calendar_json` — the score is real, and it is date arithmetic

```
baseline  0.0000  fv 0.7252
iter 1    0.0056  fv 0.5458
iter 2    0.0019  fv 0.9196
iter 3    0.0019  fv 0.5757
```

I checked whether this is a harness fault and it is not. Two separate things are happening.

**The baseline 0.0000 at fv 0.7252 is correct.** The untrained model echoes the TOOL SCHEMA back
instead of calling it — `{"name": "calendar.events.insert", "description": "Create an event on the
user's primary Google Calendar..."}`. That parses as JSON (so it counts as format-valid) but is a
tool definition, not a call. Exactly the format/content split working.

**The fine-tuned failures are date resolution, not schema.** This is NOT the flat-vs-nested `end`
bug from the 08-23 note — the schema is right now:

```
gold: end 2026-11-28T09:00:00   ->  model: 2026-11-26T10:00:00   (2 days off)
gold: end 2027-08-09T10:00:00   ->  model: 2026-08-09T11:00:00   (one YEAR off)
gold: end 2026-07-13T09:00:00   ->  model: 2026-07-12T10:00:00   (1 day off)
```

`wrong_arguments (305)` is that. The year case is the rollover the 08-23 note measured at 82% of
gold rows: the request says a bare date, the reference instant is later in the year, and the gold
rolls forward. It is learnable from the prompt (which states the reference instant) but it is exact
arithmetic, and a 360M model is not doing it. One iteration-2 sample also shows the old summary
contamination — `"summary": "go see Christopher Robin next saturday at 8 am."` — the whole request
including the date phrase in the title.

**And `format_valid` is unstable across iterations: 0.726 → 0.546 → 0.920 → 0.576.** Same task, same
schema, similar data. That is the 08-23 variance finding reproducing on a different model: which
convention a given fit lands in is uncontrolled, and on an all-or-nothing metric that is worth the
whole score. `unparseable_output (227)` of 535 at iteration 3 is the 0.576 from the other side.

So: not a bug, a genuinely hard task plus a weak model plus known instability. Historical context
for reading it — a larger model reached 0.4598 on this task in the 08-23 run, so 0.0019 is a
capability gap and not a ceiling.

### 5.3 `toolbench` — training, on the long box

At 1h29 it is at epoch 2.1 of 3, `loss 0.61`, `eval_loss 0.687`, both falling smoothly. No eval yet,
which is expected: this task's first eval lands ~2h in (§3). It is on the 7-day box precisely so the
15-eval stagnation window is reachable, which it was not on the 20h CSE box.

### 5.4 Monitoring tooling

`scripts/monitor_runs.py` parses whole logs and prints per-run scores with deltas, format_valid,
failure taxonomy, interventions, curriculum growth, teacher fitness, the orchestrator's stated
reason, and worker-op timings — with explicit ALERTs for score-0-with-format-0 (a FORM problem),
score-0-with-format-1 (a CONTENT problem), single-iteration jumps over 0.25, identical consecutive
scores, zero-row rebuilds, and a judge_unsure rate over 0.15.

It caught its own bug on first use: the model-selection regex only matched `single_model`'s "PINNED
to" line, so it reported "(not selected yet)" on three runs that had chosen a model 40 minutes
earlier. A monitor that misreports is worse than no monitor, so it now reads both strategies'
announcements.

---

## 6. Run log (appended as runs land)

| job | task | outcome | notes |
|---|---|---|---|
| 38812041/38812203 | toolbench | FAILED | §2.1 context clamp |
| 38817759 | toolbench | cancelled | §2.2 199 five-shot 400s |
| 38818333/38818334 | toolbench | cancelled | §2.3 difficulty probe ate the box |
| 38820306 | toolbench (gemma) | FAILED | §2.4 logits OOM |
| 38820307 | toolbench (SmolLM2) | cancelled at 2 iters | the validation evidence in the headline |
| 38820472 | toolbench (gemma) | cancelled at 1 iter | trained, but 5.1 h/iteration |
| 38832586 | toolbench | running | 7-day intsys, smallest_first, bypass |
| 38832587 | calendar_json | running | " |
| 38832588 | ner_bc5cdr | queued | " |
