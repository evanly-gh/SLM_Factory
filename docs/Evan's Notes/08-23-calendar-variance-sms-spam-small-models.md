# The calendar zeros were noise, SMS spam is back, and three sub-billion models on the bench

**Date:** 2026-08-23
**Companions:** `08-22-sub-billion-pool-and-task-evidence.md`, `08-21-run-38708719-review.md`

**The headline is a correction to the last run's own report.** You asked whether calendar_json's
opening zeros were a bug, a real improvement, or a train/eval mismatch. It is the first one, and
the specific finding is worse than a mismatch would have been: **iterations 2, 3 and 5 trained on
byte-identical data with byte-identical hyperparameters and scored 0.0037, 0.0019 and 0.4598.**
The +0.4561 the report credits to `mine_new_real` came from a rebuild that its own log says added
zero rows.

The reason a 0.46 swing is possible without any data change is **not** that the score jitters by
0.46. It is that `ast_arg_match` is all-or-nothing per row, and every calendar row hinges on the
same key: emit `"end": {"dateTime": ...}` and the row can score, emit `"end": "..."` and it cannot.
One convention, applied to all 535 rows, is worth the entire score. Which convention a given fit
lands in is what is uncontrolled — and on several iterations the model was not merely wrong but
**unparseable on half the eval set**. §1.3 has the numbers, and §1.4 names a second candidate cause
I cannot rule out from the logs: the GGUF re-quantization, which this model has a documented
history of getting wrong on this cluster.

---

## 1. Why calendar_json scored ~0 until iteration 5

### 1.1 It is not a train/eval schema mismatch

That was the right first hypothesis, because the symptom looks exactly like one. Here is what the
model emitted at iterations 1–3 against what the gold wanted:

```
gold  : [{"arguments": {"end": {"dateTime": "2026-11-28T09:00:00"}, ...
iter 1: [{"arguments": {"end": "2026-11-30T09:00:00",  "start": {"dateTime": "2026-11-30T08:00:00"}, ...
iter 2: [{"arguments": {"end": "2026-11-30T09:00:00",  "start": {"dateTime": "2026-11-30T10:00:00"}, ...
iter 3: [{"arguments": {"end": "2026-11-30T09:00:00",  "start": {"dateTime": "2026-11-30T08:00:00"}, ...
iter 5: [{"arguments": {"end": {"dateTime": "2026-11-28T09:00:00"}, "start": {"dateTime": "2026-11-28T08:00:00"}, ...
```

Two things are wrong before iteration 5 and both are fixed at it. `end` is a **flat ISO string**
where gold has a **nested `{"dateTime": ...}` object** — and note `start` on the same row is
nested correctly, so the model learned the schema for one key and not the other. The date is also
two days off. At iteration 5 the model emits the gold bytes exactly.

So: did the training data teach the flat shape? **No.** I checked every target in all four
curriculum versions the run wrote:

```
dataset_v1.jsonl: n=3000  nested_end=3000  flat_end=0
dataset_v2.jsonl: n=3929  nested_end=3929  flat_end=0
dataset_v3.jsonl: n=3929  nested_end=3929  flat_end=0
dataset_v4.jsonl: n=3982  nested_end=3982  flat_end=0
```

**Zero flat targets out of 14,840.** The training data and the eval gold use the same schema. The
model was inventing the flat form, not copying it. `format_valid` was `1.0000` throughout, which
says the same thing from the other side: the JSON always parsed, it was just wrong.

Worth stating because it is the thing that would have made this a data bug: `calendar_json` pools
TOPv2 and SGD and re-splits them 88/12 on a fixed seed, so train and eval are draws from one
distribution by construction. That was the B321 fix and it is holding.

### 1.2 It is not a real improvement either

Iteration 5's own curate block:

```
✗ ERROR: data_rebuild/mine_new_real added 0 new rows. The curriculum is unchanged at 3929 row(s),
  so training this iteration would repeat the previous one exactly.
CURRICULUM: 3929 → 3929 row(s) (+0 added this rebuild, 0 novel overall)
```

All fourteen web-discovered candidates were rejected or unusable, and TOPv2 was already exhausted.
So iteration 5 trained on the same curriculum as iteration 3. I confirmed that directly rather
than trusting the log — `dataset_v2.jsonl` and `dataset_v3.jsonl` are identical in content **and
row order**, differing only in a `_dataset_version` bookkeeping integer:

```
same ORDER + content (ignoring _dataset_version): True
```

And the configs match. From the run's own tier-0 ledger:

| Iter | Score | Config | Dataset | Stopped at | train_loss |
|---|---|---|---|---|---|
| 2 | 0.0037 | `r=16 a=32 wd=0.01 lr=2e-04 ep=3` | v2 (3929) | epoch 0.6467 | 0.07134 |
| 3 | 0.0019 | `r=16 a=32 wd=0.01 lr=2e-04 ep=3` | v3 (3929) | epoch **1.247** | **0.04064** |
| 5 | **0.4598** | `r=16 a=32 wd=0.01 lr=2e-04 ep=3` | v3 (3929) | epoch 0.6467 | 0.07076 |

**Three fits of the same model on the same rows with the same hyperparameters: 0.0037, 0.0019,
0.4598.** Iterations 2 and 5 even stopped at the same epoch with train losses within 0.0006 of
each other, and their scores differ by 124×.

### 1.3 So what is it — corrected

> **I first wrote this section as "run-to-run variance" and left the impression it was smooth
> random noise of magnitude 0.46. That framing is wrong and the objection to it is correct: a 0.46
> swing does not come out of jitter.** What follows replaces it. The *fact* in §1.2 is unchanged
> and is not in question — identical data, identical hyperparameters, 0.0037 / 0.0019 / 0.4598 —
> but the mechanism below is a different and much more specific claim.

**The metric is all-or-nothing per row, and a handful of discrete output conventions gate every
row at once.** `ast_arg_match` scores each row 1 or 0 on an exact match of the whole argument dict:

```307:313:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/eval/scorers/function_call.py
        content = 1.0 if _call_correct(gold_calls, pred, allowed) else 0.0
```

Every calendar row carries `start` and `end`. So if the model adopts one wrong convention for
`end`, **every row fails**, and the score is ~0.00 regardless of how well it did everything else.
If it adopts the right one, the score jumps to whatever the remaining per-row errors allow. There
is no smooth middle. The observed tier-0 scores are exactly that shape — 0.0000, 0.0019, 0.0037,
0.0262, 0.2542, 0.3589, 0.4598 — clusters, not a distribution around a mean.

The three conventions the model is flipping between are all about the `end` key, and all three are
visible in the sampled predictions:

```
correct   : {"arguments": {"end": {"dateTime": "..."}, "start": {"dateTime": "..."}, ...}}
flat      : {"arguments": {"end": "2026-11-30T09:00:00", "start": {"dateTime": "..."}, ...}}
misplaced : {"arguments": {"start": {...}, "summary": "..."}, "end": {"dateTim…    ← outside arguments, truncated
```

The second is a type error (string where a dict is required). The third closes `arguments` early
and puts `end` at the call level, which then also runs out of tokens — that row parses to nothing.
Note `start` is nested correctly in all three. It is specifically `end`, and the reason is
positional: gold is serialized with sorted keys, so `end` is the **first** key inside `arguments`,
emitted before the model has produced any datetime and while it has the least context to copy from.

**The parse rate itself swings enormously, which is the part I missed.** `format_valid` across the
nineteen tier-0 iterations:

| iter | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| format_valid | 1.000 | 1.000 | .998 | 1.000 | **.757** | .953 | 1.000 | 1.000 | .998 | **.494** | **.766** | .996 | **.535** | .888 |
| score | .004 | .002 | .000 | **.460** | .000 | .002 | .254 | .000 | .359 | .002 | .000 | .026 | .000 | .008 |

On iterations 11 and 14 **half the eval set produced output the scorer could not parse at all.**
So yes — on several iterations the model genuinely was emitting garbage, and I should have led
with that rather than with "variance".

### 1.4 What I have NOT ruled out, and it matters

**Quantization instability.** Every iteration re-merges the adapter and rebuilds a Q4_K_M GGUF, and
the eval scores *that artifact*, not the bf16 checkpoint (`SLM_QUANT_EVAL=1`). Qwen3-0.6B at
Q4_K_M has a documented history on this exact cluster of quantizing to a **degenerate** artifact —
runs 38569606 and 38569608 both died with:

```
QuantizationInfrastructureError: Q4_K_M GGUF for Qwen/Qwen3-0.6B decoded to degenerate output on
two consecutive independent builds ... decodes to degenerate output '////////////////////////////////'
```

A build that is *partially* degenerate would not trip that check — it only has to generate one
sane token sequence for a "Hello" prompt — and would show up downstream as exactly what the table
above shows: wildly varying parse rates on the same task.

So there are two candidate mechanisms and **the logs cannot separate them**, because the bf16
checkpoint was never scored:

| Hypothesis | Predicts |
|---|---|
| Training lands in different output modes | bf16 and Q4 wobble together |
| Quantization produces variable artifacts | bf16 stable, Q4 wobbles |

Both are consistent with everything in §1.2 and §1.3. What is *not* in doubt either way is the
finding that matters: **iteration 5's +0.4561 was not caused by the data, because no data
changed.**

### 1.5 The mechanism, if it is the training side

Two defects that would compound. Stated as the leading hypothesis, not as established.

**(a) The training run is not seeded or pinned to determinism.** `_build_sft_config`
(`training/lora_trainer.py:905`) passes no `seed` and no `data_seed`, and nothing sets
`torch.use_deterministic_algorithms` or the cuDNN flags. The only seeded randomness in the whole
path is the validation-split shuffle:

```876:878:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/training/lora_trainer.py
        _rng = _rnd.Random(1234)
        _idx = list(range(len(_formatted)))
        _rng.shuffle(_idx)
```

**(b) Checkpoint selection optimizes a quantity the score does not track.** Early stopping keeps
the best checkpoint by `eval_loss`:

```935:937:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/training/lora_trainer.py
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
```

That is token-level cross-entropy on 471 rows carved out of the training set. On `calendar_json`
the scored metric is `ast_arg_match` — exact argument equality. The two are close to uncorrelated
here: writing `2026-11-30` instead of `2026-11-28` is a couple of tokens of cross-entropy and a
1.0 → 0.0 swing on the metric. Emitting `"end": "..."` instead of `"end": {"dateTime": "..."}` is
three tokens and, again, the whole row.

Iteration 3 is the clean demonstration. It trained nearly twice as long and reached a **lower**
loss (0.04064 vs 0.07076) and scored **worse** (0.0019 vs 0.4598). The selection criterion
preferred it. The metric did not.

Put together: the optimizer path varies run to run, and the rule that picks which checkpoint to
keep cannot tell the good outcome from the bad one. Checkpoint selection is a lottery with respect
to the number that gets reported.

### 1.6 Why this matters more than the one score

The attribution table at the end of that run reads:

```
mine_new_real   +0.4561   (iteration 5)
```

for a rebuild that added zero rows. Everything downstream inherits that error. Concretely:

- **The tier-0 ledger is not interpretable.** Nineteen iterations, scores from 0.0000 to 0.4598,
  and the spread on a *fixed* configuration is at least ±0.46. Iteration 8 (0.2542), iteration 10
  (0.3589) and iteration 13 (0.0262) are all inside the noise of iteration 5.
- **Rollback is deciding on differences far smaller than the noise.** At tier 1 the run kept two
  changes worth +0.0093 and +0.0150 and rolled back seventeen. On a task whose repeat-run spread
  is measured in tenths, those two "improvements" are not distinguishable from luck.
- **The orchestrator was reasoning about a phantom.** It wrote several paragraphs attributing the
  jump to curriculum thinness and to mining TOPv2, which is a coherent story about an event that
  did not happen.
- **`dataset_v3.jsonl` was written twice** — the version counter does not advance when a rebuild
  adds no rows, so iteration 3's artifact was overwritten by iteration 5's. That is B225, and it
  is why the two identical-looking files needed comparing by content rather than by name.

### 1.7 The experiment that separates them

The design has to change from what I proposed first. Scoring five repeat fits tells you the spread
but **not which half of the pipeline produced it**, so it would leave §1.4 unresolved. The version
that answers it:

Train `calendar_json` on the frozen `dataset_v3` curriculum with the fixed
`r=16 a=32 wd=0.01 lr=2e-04 ep=3` config **five times**, and for each fit record:

1. the **bf16** score (`SLM_QUANT_EVAL=0`),
2. the **Q4_K_M** score on the GGUF built from that same checkpoint,
3. the **parse rate** for both,
4. the **flat / nested / misplaced `end`** counts for both.

Then:

| Result | Reading |
|---|---|
| bf16 tight, Q4 scattered | quantization — fix the GGUF path, and every quantized score in the project is suspect |
| both scattered together | training — seed it and change checkpoint selection |
| both tight | something specific to that run; look at the GGUF cache keys next |

(4) is the cheap addition that makes the whole thing legible: it turns "score was 0.002" into
"483 of 535 rows emitted a flat `end`", which is a fact you can act on rather than a number to
stare at.

**Follow-on fixes, only once the above says which one is needed:**

- *If training:* seed `SFTConfig` (`seed`, `data_seed`) plus the torch determinism flags, and stop
  selecting checkpoints on `eval_loss`. Cross-entropy on a ~120-token target barely distinguishes
  `"end": "2026` from `"end": {"dat` — three or four tokens — while the metric treats it as the
  whole row. Scoring the top-k checkpoints on the task metric is the honest fix.
- *If quantization:* the load-validation gate is too weak. It generates once for "Hello" and
  passes; a partially degenerate artifact sails through. It should score a small fixed sample of
  the task's own eval rows and compare against the bf16 checkpoint before the artifact is accepted.

I have not made any of these changes. (1)–(4) is a measurement; everything after it moves recorded
scores, so it is yours to call.

---

## 2. SMS spam is back in the registry

Rewritten rather than reverted. The old `data/loaders/sms_spam.py` went out with the `task_type`
channel on 2026-08-18 (commit `387f4ad`), and it carried three defects the current registry makes
it possible to fix properly.

### 2.1 What was wrong with the old one

| Defect | Old behaviour | Now |
|---|---|---|
| **B44 — unshuffled split** | `examples[:80%]` as train. The UCI file is ordered, so the halves had different class balance. | Stratified per class on a pinned seed; both halves carry the corpus base rate |
| **No deduplication** | The corpus has repeated messages; a random split puts copies on both sides | **415 duplicates removed BEFORE the split.** Measured train/eval leak: 0 |
| **Unverified plain-HTTP zip** | Fetched from `archive.ics.uci.edu` at load time, no checksum | Checksummed bundle at `data/local/sms_spam/`, HuggingFace `ucirvine/sms_spam` as fallback |

The 415 figure is worth dwelling on: that is 7.4% of the corpus, and under the old loader an
unknown fraction of it sat on both sides of the eval firewall.

### 2.2 The task as built

```
data/local/sms_spam/{train,test}.jsonl + manifest.json + checksums.sha256
  5574 raw → 415 duplicates removed → 5159 rows
  train 4128 (spam 514, 12.4%)   test 1031 (spam 128, 12.4%)   leak 0
```

The two decisions worth challenging:

**The metric is `minority_f1`, not accuracy or macro-F1.** The corpus is ~87% ham. Answering
`ham` for every message scores 0.87 accuracy and ~0.47 macro-F1 — both of which read like a
working model. Minority-class F1 scores it **0.0**. The preflight confirms this end to end:

```
PASS  sms_spam  train=3250  eval=1000  minority_f1  gold=1.0  degen=0.0 (always 'ham')
```

That makes this the cheapest collapse detector in the suite, and it is the same reasoning
`routerbench` uses.

**`ham` is defined for the teacher.** Left undefined it is not English, and the teacher reads it
as the food — the identical failure RouterBench hit when it read `local` as "nearby" and discarded
70% of generated rows for the wrong reason (B267). `label_definitions` states both classes.

Two smaller choices, recorded so they can be argued with: `initial_train_cap=3000` rather than
5000, so `mine_new_real` has somewhere to go on its first attempt — that is the lesson
`calendar_json` taught, where a cap that consumed the whole pool at cold start killed the data
intervention outright. And `balance_labels(max_ratio=8)` rather than the 3 the other classification
tasks use, because the real base rate is ~6.5:1 and clamping to 3:1 would teach a prior the eval
set does not have.

### 2.3 Everything that ships with it

| File | What |
|---|---|
| `data/loaders/sms_spam.py` | Loader: local bundle first, HF fallback, dedup → stratified split |
| `tasks/sms_spam.py` | `TaskSpec` — every field declared, no defaults exist to fall through |
| `tasks/__init__.py` | Registered; the registry is now nine tasks |
| `scripts/vendor_sms_spam.py` | Materializes the checksummed bundle and refuses to write one with any leak |
| `data/local/sms_spam/` | The frozen bundle, manifest and checksums |
| `tests/pipeline/run_sms_spam_l40s.slurm` | Launcher, `intelligentsystems` quota |
| `tests/pipeline/run_sms_spam_cse.slurm` | Launcher, CSE quota |
| `tests/data/test_task_context_block.py` | Fixture row added — the registry guard failed until it was |

That last one is the registry working as designed. Adding the task made
`test_the_fixture_table_covers_the_whole_registry` fail immediately with
`At index 7 diff: 'xlam_bfcl' != 'sms_spam'`, which is precisely the "a new task cannot silently
inherit no coverage" contract the 2026-08-18 rebuild was built to enforce.

### 2.4 Every task you can run right now

All nine pass `scripts/preflight_tasks.py`, measured today, not quoted from a previous note:

```
PASS  calendar_json        train= 3250  eval= 535  ast_arg_match   gold=1.0  degen=0.0
PASS  clinc150             train= 3250  eval=1000  macro_f1        gold=1.0  degen=0.0001
PASS  dialogsum            train= 3228  eval= 667  judge_mean_0_1  gold=judge degen=judge
PASS  gsm8k                train= 3250  eval=1000  exact_match     gold=1.0  degen=0.0
PASS  ner_bc5cdr           train= 3250  eval=1000  span_f1         gold=1.0  degen=0.0
PASS  proactive_listening  train= 3250  eval=1000  minority_f1     gold=1.0  degen=0.0
PASS  routerbench          train= 3250  eval=1000  minority_f1     gold=1.0  degen=0.0
PASS  sms_spam             train= 3250  eval=1000  minority_f1     gold=1.0  degen=0.0
PASS  xlam_bfcl            train= 3250  eval=1000  ast_arg_match   gold=1.0  degen=0.0

9/9 passed
```

With launcher coverage:

| Task | Category | Metric | `_l40s` | `_cse` | Ever run? |
|---|---|---|---|---|---|
| `gsm8k` | in_distribution | exact_match | ✓ | — | Yes — CONVERGED 0.8263 vs 0.820 |
| `dialogsum` | in_distribution | judge_mean_0_1 | ✓ | — | Yes — 4 tiers, real gains at tiers 0–2 |
| **`sms_spam`** | **in_distribution** | **minority_f1** | **✓ new** | **✓ new** | **No — never run** |
| `xlam_bfcl` | format_bound | ast_arg_match | ✓ | ✓ | Yes — best 0.8600 vs 0.8660 |
| `calendar_json` | format_bound | ast_arg_match | ✓ | ✓ | Yes — 0.8673, but see §1 |
| `ner_bc5cdr` | format_bound | span_f1 | ✓ | ✓ | Yes — CONVERGED 0.8098 vs 0.800 |
| `routerbench` | out_of_distribution | minority_f1 | ✓ | ✓ | Yes — numbers invalidated by contamination |
| `proactive_listening` | out_of_distribution | minority_f1 | ✓ | ✓ | No — ready, never submitted |
| `clinc150` | out_of_distribution | macro_f1 | ✓ | **✓ new** | Yes — CONVERGED 0.8952 vs 0.8919 |

`clinc150` picked up the CSE twin it was missing. `gsm8k` and `dialogsum` still have only the
`intelligentsystems` launcher; say the word and they get twins too.

**Caveat on all nine, from §1:** every "CONVERGED" above was decided by comparing scores whose
run-to-run spread is unmeasured. The convergence verdicts may well hold — BC5CDR going
0.0000 → 0.8098 is far outside any plausible noise band — but the small deltas are not safe.

---

## 3. Banking77

On hold as instructed. Nothing added, nothing removed, and the literature argument from
`08-22-sub-billion-pool-and-task-evidence.md` §4.1 is parked rather than withdrawn.

---

## 4. The three sub-billion models

You asked for a baseline and a fine-tuned number on `ner_bc5cdr` and `xlam_bfcl`, and for whether
our quantization path can handle them. The measurement is running as job **38764845**; the numbers
go in §4.4 when it lands. Everything else about the setup is settled and worth reading first,
because two of the answers do not depend on the scores.

### 4.1 The blocker is gone, and it was one function

`08-22`'s §1.3 identified a single line that refused to score any non-Qwen GGUF. That was real —
the installed llama-cpp-python 0.3.34 has no `chat_template_kwargs` parameter, so the code always
took a fallback branch whose prompt was a hardcoded Qwen ChatML string, and it correctly refused
rather than serve a Gemma the wrong turn markers.

It now renders from the served model's own tokenizer:

```python
def _serving_prompt_prefix(prompt: str, base_model: str) -> str:
    if _is_qwen_model_id(base_model):
        return _qwen_no_think_prompt(prompt, base_model)   # bytes unchanged
    tokenizer = _serving_tokenizer(base_model)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    return prompt
```

Three things about that are deliberate:

**Qwen keeps its hand-written bytes.** Every score this project has recorded was produced with
them, so routing Qwen through its own template — even if it produced the same string — would put
every historical number in question for no gain. There is a regression test asserting the exact
Qwen prefix is unchanged.

**Stop tokens are derived too.** The old path passed `stop=["<|im_end|>"]` unconditionally. That is
ChatML, which is right for Qwen and for SmolLM2 and wrong for Gemma, which ends a turn with
`<end_of_turn>`. Sending the wrong one does not error — it just never stops, so every row runs to
`max_tokens` and trails generated text into the scorer. That would have looked like a capability
failure.

**A missing chat template is an answer, not an error.** You linked the BASE checkpoints
(`gemma-3-270m`, not `-it`; `SmolLM2-360M`, not `-Instruct`). A base model may ship no chat
template at all, and training already takes the plain-text branch in that case, so serving the bare
prompt is what keeps train and serve aligned. Wrapping it in some other family's turn markers is
what would break parity.

The old refusal was pinned by a test. That test is replaced rather than deleted, by three that
assert the new contract: a templated non-Qwen renders through its own template and gets its own
stop token, an untemplated one gets the bare prompt, and Qwen is byte-identical to before.

### 4.2 Can we quantize them — the parts already known

The conversion half is settled and does not need the run to answer it. The local llama.cpp
checkout (`a320cbfc`) registers all three architectures:

```
Gemma3ForCausalLM  → gemma    (gemma-3-270m)
LlamaForCausalLM   → llama    (both SmolLM2 sizes declare this)
```

So `convert_hf_to_gguf` will accept them. What the run is actually testing on this axis is the two
steps after conversion, which is where a small model is most likely to break:

1. **`llama-quantize` to Q4_K_M.** Gemma 3 270M is the interesting case — 170M of its 270M
   parameters are the embedding table over a 256k vocabulary, and its published Q8_0 is only
   1.15× its Q4_K_M where the pool's arithmetic assumes 1.82×. A quantizer that handles a
   normally-shaped model fine can behave differently when the embedding dominates the file.
2. **`validate_and_record_gguf` — an actual load through llama-cpp-python.** This is the step that
   caught the degenerate-decode corruption on runs 38569606 and 38569608, where a Q4_K_M of
   Qwen3-0.6B loaded and then emitted `////////////////////////////////`. A converter can emit a
   structurally valid file that decodes to garbage, and only a load-and-generate catches it.

The probe reports both, plus the resulting file size, so a failure lands as "quantization failed at
step N" rather than as a mystery zero.

### 4.3 What the probe does, and what it deliberately does not

`scripts/probe_small_models.py`, launched by `tests/pipeline/run_small_model_probe{,_cse}.slurm`.
Three numbers per (model, task) cell, all through the **same** `eval.harness.run_eval` and the same
`TaskSpec` a real run uses, so they are comparable to run scores rather than to a private harness:

| Number | What it is |
|---|---|
| `zero_shot` | base checkpoint, no adapter — `weights_ref == model_id`, exactly what `evaluate`'s baseline does |
| `finetuned` | one LoRA fit, scored bf16 |
| `quantized` | that adapter merged, Q4_K_M'd, load-validated, scored through llama-cpp-python |

The fit is **fixed** at `r=16 a=32 lr=2e-4 ep=3` for every cell. Comparing models needs the fit
held constant; tuning per model would answer a different question.

Three deliberate omissions:

- **Nothing touches `config/android_pool.py`.** You said not to think about the pool yet. These
  models are not pool members and model selection cannot reach them; this measures whether they
  *could* be.
- **No teacher, no synthesis, no mining, no orchestrator.** Hence one GPU rather than two — a
  pipeline run needs the second for the vLLM teacher, and reserving an idle L40S for hours on a
  saturated queue is how everyone else's jobs get starved.
- **300 eval rows, not 1,000.** Enough to separate "learned the task" from "did not" at a third of
  the cost. It is a screen, not a result, and §1 is a live reminder that a single number on this
  pipeline should not be over-read anyway.

Results are written to the JSON after every cell, so a job killed at hour six still reports hours
one through five.

**One caveat I want on the record before the numbers arrive.** Nemotron-Research-Tool-N1's size
sweep found that post-training gains are *muted* specifically at 0.5B and 1.5B on function calling:

> "performance improvements from post-training are limited for smaller models (0.5B and 1.5B),
> whereas larger models exhibit substantial gains"

So a flat line on `xlam_bfcl` is the predicted outcome, not a surprise, and it would not by itself
condemn the models. `ner_bc5cdr` is the fairer first read — it is the task where the published
evidence reaches furthest down the size curve (OpenMed's 209M GLiNER goes 0.5721 → 0.8848 on
BC5CDR-Disease) and where our own 0.6B result is cleanest (0.0000 → 0.8098).

### 4.4 The naming trap — measured, and it cost a job

The first probe (38764845) was launched against the checkpoints as linked, and the very first thing
it printed was the reason to kill it:

```
google/gemma-3-270m         arch=['Gemma3ForCausalLM']  vocab=262145  chat_template=False
HuggingFaceTB/SmolLM2-360M  arch=['LlamaForCausalLM']   vocab= 49152  chat_template=False
HuggingFaceTB/SmolLM2-135M  arch=['LlamaForCausalLM']   vocab= 49152  chat_template=False
```

The architectures are what §4.2 predicted, so conversion is fine. **`chat_template=False` on all
three is the problem: those are base checkpoints, and they were about to be compared against an
instruct one.**

**The two families use opposite naming conventions, and this is worth pinning down because it will
recur:**

| Publisher | Base checkpoint | Post-trained checkpoint |
|---|---|---|
| **Qwen** | `Qwen/Qwen3-0.6B-Base` | **`Qwen/Qwen3-0.6B`** ← plain name is instruct |
| **Google** | **`google/gemma-3-270m`** ← plain name is base | `google/gemma-3-270m-it` |
| **HuggingFaceTB** | **`HuggingFaceTB/SmolLM2-360M`** ← plain name is base | `HuggingFaceTB/SmolLM2-360M-Instruct` |

So the pool's `Qwen/Qwen3-0.6B` is instruction-tuned even though nothing in the name says so. Its
card states `Training Stage: Pretraining & Post-training`, its model tree lists
`Qwen/Qwen3-0.6B-Base` as its base model, it is tagged `conversational`, and it carries the
`enable_thinking` switch. Our own run logs confirm it from the other direction — every calendar
iteration printed `Pinned chat template to the SERVED model Qwen/Qwen3-0.6B`, and
`lora_trainer.py:318` hard-fails on a Qwen with no chat template, so it demonstrably has one.

Why it matters beyond the score: **the loop consumes zero-shot numbers structurally.** The
zero-shot baseline is counted as a candidate checkpoint (`evaluate.py:327`), the accuracy goal is
calibrated against a zero-shot teacher, and difficulty stratification labels rows easy/medium/hard
by whether the smallest and largest models pass them zero-shot. Feed it a base model and all three
degenerate at once: every difficulty bucket collapses, the baseline candidate is meaningless, and
the comparison against Qwen3-0.6B is base-versus-instruct rather than small-versus-large.

38764845 was cancelled 38 minutes in, still inside its first cell. That slowness was itself a
symptom — a base model has no reliable turn-end behaviour, so it generated the full 512-token
reserve on every one of 300 rows instead of stopping. The instruct variants emit EOS and should be
several times faster.

Relaunched as **38765131 / 38765132** against `gemma-3-270m-it`, `SmolLM2-360M-Instruct` and
`SmolLM2-135M-Instruct`. The launcher now defaults to the instruct ids and documents the trap
inline; `PROBE_MODELS` overrides it if the base numbers are ever wanted as a contrast.

One thing worth keeping from the cancelled run: Gemma's vocabulary is **262,145 tokens against
SmolLM2's 49,152**, 5.3×. That is the embedding-table design from `08-22` §1.5 showing up in
practice, and it is why Gemma's published Q8_0 is barely larger than its Q4_K_M.

And a second: Unsloth loaded Gemma 3 on its fast path (`Unsloth 2026.7.2: Fast Gemma3 patching`),
not a generic fallback — so architecture support is confirmed against the installed version rather
than inferred from the docs.

### 4.5 Results

Job **38765131**, 300 eval rows, one fixed LoRA fit per cell (`r=16 a=32 lr=2e-4 ep=3`) on 3,000
gold rows. Cells fill in as they land.

| Model | Params | Task | Zero-shot | Fine-tuned (bf16) | Q4_K_M | GGUF |
|---|---|---|---|---|---|---|
| `gemma-3-270m-it` | 270M | `ner_bc5cdr` | 0.0000 | **0.6926** | 0.6529 | 241 MB |
| `gemma-3-270m-it` | 270M | `xlam_bfcl` | 0.0000 | **0.0567** | 0.1000 | 241 MB |
| `SmolLM2-360M-Instruct` | 362M | `ner_bc5cdr` | 0.0000 | **0.7339** | **0.7339** | 258 MB |
| `SmolLM2-360M-Instruct` | 362M | `xlam_bfcl` | 0.0000 | **0.4500** | **0.4500** | 258 MB |
| `SmolLM2-135M-Instruct` | 135M | `ner_bc5cdr` | 0.0000 | **0.6107** | 0.5476 | 101 MB |
| `SmolLM2-135M-Instruct` | 135M | `xlam_bfcl` | 0.0000 | **0.2300** | 0.1867 | 101 MB |

Reference points from the pool, same eval sets, same metrics:

| | `ner_bc5cdr` (span_f1) | `xlam_bfcl` (ast_arg_match) |
|---|---|---|
| Qwen3-0.6B@Q4_K_M, fine-tuned | **0.8098** (converged) | — |
| Qwen3-4B-Instruct@Q4_K_M, fine-tuned | — | **0.8600** |
| Qwen3.6-35B teacher, 5-shot | 0.7190 | 0.8350 |

### 4.5a The zero-shot zeros are real, and here is the proof

Six cells, six 0.0000 baselines, is exactly the pattern a data-loading or scoring bug produces, so
it was worth confirming rather than assuming. It is not a bug. Three independent checks:

**The eval set loaded.** Every cell reports `n=300` — the rows are there, with real gold.

**The models produced output the scorer could read.** `format_valid` on the zero-shot pass is not
zero on any cell, and on one it is nearly total:

| Model | Task | zero-shot score | **format_valid** |
|---|---|---|---|
| `SmolLM2-360M-Instruct` | `ner_bc5cdr` | 0.0000 | **0.8967** |
| `gemma-3-270m-it` | `ner_bc5cdr` | 0.0000 | 0.5500 |
| `SmolLM2-360M-Instruct` | `xlam_bfcl` | 0.0000 | 0.2033 |
| `SmolLM2-135M-Instruct` | `xlam_bfcl` | 0.0000 | 0.2067 |
| `gemma-3-270m-it` | `xlam_bfcl` | 0.0000 | 0.0833 |
| `SmolLM2-135M-Instruct` | `ner_bc5cdr` | 0.0000 | 0.0367 |

**SmolLM2-360M parsed 269 of 300 entity lists successfully and matched gold on none of them.** A
loader returning nothing, or a scorer that could not read the output, cannot produce that row — it
would show `format_valid=0.0`.

**The raw outputs are wrong in a legible way.** From that same cell:

```
gold  : []
raw   : [ { "text": "On PND71", "type": "behavioral assay" } ]
parsed: [{'text': 'On PND71', 'type': 'behavioral assay'}]
```

Well-formed JSON, parsed cleanly, and wrong twice over: the sentence has no entities, and
`behavioral assay` is not one of BC5CDR's two classes. The model is inventing a plausible-looking
answer to a task whose contract it has never seen. That is what a real 0.0 looks like.

Two supporting facts. `scripts/preflight_tasks.py` independently verifies for both tasks that gold
predictions score 1.0 and a degenerate answer scores ~0 through the same scorer, so the metric is
not stuck at zero. And the *fine-tuned* pass on the same eval sets reports `format_valid` of
0.9867–0.9967 — the models go from unparseable-or-wrong to near-perfectly formatted after training,
which is the whole point and could not happen if the harness were broken.

**What actually predicts the score is non-embedding parameters, not parameter count.**

> I wrote a different conclusion here after the two Gemma cells — that the extraction-versus-
> composition split was a property of the *tasks*, so no sub-billion model would do tool calling.
> The very next cell refuted it. `SmolLM2-360M-Instruct` scores **0.4500** on xlam where
> `gemma-3-270m-it` scored **0.0567**: eight times better, same recipe, same rows, and only 1.34×
> the total parameters. Recorded rather than quietly edited, because calling it one cell early is
> the same mistake §1 is about.

Line the models up by where their parameters actually sit:

| Model | Total | Embedding | **Non-embedding** | BC5CDR | xlam |
|---|---|---|---|---|---|
| `SmolLM2-135M-Instruct` | 135M | 28M (49k vocab) | **~107M** | 0.6107 | — |
| `gemma-3-270m-it` | 270M | 168M (262k vocab) | **~102M** | 0.6926 | 0.0567 |
| `SmolLM2-360M-Instruct` | 362M | 47M (49k vocab) | **~315M** | 0.7339 | 0.4500 |
| `Qwen3-0.6B` | 600M | 160M | **440M** | 0.8098 | — |

**With all six cells in, the pure capacity story does not survive either — and the residual is the
most useful thing here.** The final cell landed at 0.2300, not near Gemma's 0.0567 as the
hypothesis predicted:

```
xlam_bfcl, by non-embedding capacity
    ~102M   gemma-3-270m-it          0.0567
    ~107M   SmolLM2-135M-Instruct    0.2300    ← 4x Gemma at the same capacity
    ~315M   SmolLM2-360M-Instruct    0.4500

ner_bc5cdr, same ordering
    ~107M   SmolLM2-135M-Instruct    0.6107
    ~102M   gemma-3-270m-it          0.6926    ← Gemma AHEAD at the same capacity
    ~315M   SmolLM2-360M-Instruct    0.7339
```

Two things are true at once. **Within** SmolLM2, capacity orders the results cleanly on both tasks
(107M → 0.23/0.61, 315M → 0.45/0.73). **Across** families it does not: at essentially identical
non-embedding budgets Gemma is *ahead* on BC5CDR and 4× *behind* on xlam. That is a task × family
interaction, and capacity alone cannot produce it.

The reading that fits — offered as a hypothesis, not a finding — is that Gemma's design tradeoff is
exactly right for one of these tasks and exactly wrong for the other. BC5CDR is **rare-token
heavy**: `Naloxone`, `clonidine`, `thrombocytopenia`. A 262k-token vocabulary represents those in
fewer, better-conditioned pieces than a 49k one, and the answer is largely *copied from the input*,
so vocabulary quality converts directly into score. `xlam_bfcl` is **structural**: the tokens are
ordinary, and the work is composing a call from a schema, which lives in the transformer stack that
Gemma spent its budget away from. Same 270M, opposite outcomes, for the same reason.

**What I am not claiming.** The BC5CDR spread across the two ~100M models is 0.08, and §1.7 applies
to it directly — the noise floor is unmeasured and 0.08 is inside what one fit could move. The xlam
spread is 0.0567 vs 0.2300, four-fold, which is much harder to write off but still n=1 per cell.
This is the third reading of these data I have offered today; the first two were each refuted by the
next cell. Treat the vocabulary story as the best current explanation and not as established.

SmolLM2-360M is 1.34× Gemma's total size but carries **3.1× the transformer**. Gemma spends 63% of
its budget on a 262k-token vocabulary; SmolLM2 spends 13%.

That reframes both columns. On **BC5CDR** the three models sit at 0.69 / 0.73 / 0.81 — a shallow
curve, because the task is largely copying spans out of the input and learning a contract
(`Chemical` vs `Disease`, JSON shape, no markdown fence). Even 102M of transformer does most of it.
On **xlam** they sit at 0.06 / 0.45 / (unmeasured) — a cliff, because selecting a tool from a
declared schema and *synthesizing* argument values that appear nowhere in the prompt is
compositional work that scales with depth and width, not vocabulary.

**The practical consequence is a pool-design problem.** `ModelSpec.size_mb` is on-disk weight size,
and it ranks these two as near-identical: 253 MB against 271 MB at Q4_K_M, a 7% difference.
`select_smallest` would therefore pick Gemma first on every task. On BC5CDR that costs 0.04; on
xlam it costs **0.39**. If sub-billion models are ever added to the pool, size alone is the wrong
ordering key and non-embedding parameter count needs to be a recorded field.

A large vocabulary is not a defect — it is what makes Gemma quantize so flatly (Q8_0 only 1.15× its
Q4_K_M) and it may pay off on multilingual or rare-token work. It is simply not what these two
tasks are bottlenecked on.

**And BC5CDR gives a clean size curve, with one result that matters for the paper.** Three models,
one recipe, same eval set:

```
gemma-3-270m-it         270M   0.6926
SmolLM2-360M-Instruct   362M   0.7339
Qwen3-0.6B              600M   0.8098   ← current pool floor
                     ---------
Qwen3.6-35B teacher, 5-shot    0.7190
```

Monotonic in size, no inversion, and the middle row is the interesting one: **a fine-tuned 362M
model scores 0.7339 against the 35B teacher's 0.7190 five-shot.** A model ~97× smaller than the
teacher beats it on this task, which is the project's central claim holding at 1.7× below the
current pool floor. It reaches 91% of Qwen3-0.6B's score on 60% of the parameters.

This is also the Nemotron-Research-Tool-N1 finding reproducing on our own harness at a smaller
scale than they tested:

> "performance improvements from post-training are limited for smaller models (0.5B and 1.5B),
> whereas larger models exhibit substantial gains"

They saw it at 0.5B on BFCL. We see it at 0.27B on BFCL-derived data, harder. **So the flat xlam
number is the predicted outcome and should not be read as condemning the model** — the BC5CDR
number from the same checkpoint family is the counter-evidence.

### 4.6 The quantization column is missing, and that one is my bug

Every cell reports:

```
quantize   FAILED ImportError: cannot import name 'merge_for_quantization' from 'training.quantize'
```

`merge_for_quantization` lives in **`training.lora_trainer`**, not `training.quantize` — it needs
Unsloth to load and merge the adapter, so it belongs to the trainer rather than the converter.
`agent/nodes/evaluate.py` imports it from the right place; I assumed it sat with the other quantize
helpers and did not check before submitting.

Contained, because the step is wrapped: the run continues and still produces zero-shot and
fine-tuned for all six cells. Only the `quantized` column is lost, and the fix is one import.

The running job holds the old bytecode, so it will keep skipping. Rather than restart and pay for
training again, the script is now **resumable**: a cell whose workdir already holds
`training/final_checkpoint` reuses it unless `--retrain` is passed, and `--skip-zero-shot` skips the
expensive baseline. The quantization answer therefore costs one short follow-up job over the saved
checkpoints — merge, convert, load-validate, score — instead of a full re-run.

Worth noting what this near-miss says about the probe's design: wrapping each step in its own
`try/except` and writing the JSON after every cell is why an import error in step 4 cost the
quantization column rather than the whole job.

Ran as **38765655** (1h07m, COMPLETED). Results below.

### 4.6a Quantization works — all six, no degenerate artifacts

Every cell merged, converted, **load-validated** and scored. `load_validated=True` on all six, no
`QuantizationInfrastructureError`, no `////////` decode. Both architectures convert and run:
`Gemma3ForCausalLM` → gemma and `LlamaForCausalLM` → llama, through the local llama.cpp checkout
and the generalized non-Qwen serving path from §4.1.

| Model | Task | bf16 | Q4_K_M | Δ | GGUF | merge+quant |
|---|---|---|---|---|---|---|
| `gemma-3-270m-it` | `ner_bc5cdr` | 0.6926 | 0.6529 | **−0.0397** | 241.4 MB | 46s + 224s |
| `gemma-3-270m-it` | `xlam_bfcl` | 0.0567 | 0.1000 | +0.0433 | 241.4 MB | 32s + 126s |
| `SmolLM2-360M-Instruct` | `ner_bc5cdr` | 0.7339 | 0.7339 | **0.0000** | 258.1 MB | 36s + 124s |
| `SmolLM2-360M-Instruct` | `xlam_bfcl` | 0.4500 | 0.4500 | **0.0000** | 258.1 MB | 27s + 191s |
| `SmolLM2-135M-Instruct` | `ner_bc5cdr` | 0.6107 | 0.5476 | **−0.0631** | 100.6 MB | 19s + 152s |
| `SmolLM2-135M-Instruct` | `xlam_bfcl` | 0.2300 | 0.1867 | −0.0433 | 100.6 MB | 14s + 97s |

**On SmolLM2-360M the Q4 scores are identical to bf16 to four decimals on both tasks, and I checked
that rather than reporting it.** Identical scores are what a silently-cached or silently-fallen-back
eval looks like. It is a real, independent run: `format_valid` **differs** between the two passes
(0.9967 → 0.9933 on NER, 0.9867 → 0.9500 on xlam), so the quantized model produced different output
that happened to land on the same score. On xlam that is 135/300 both times, unremarkable under
binary scoring. On NER one extra row became unparseable and it was a row that scored zero anyway.

Three things worth carrying forward:

**Quantization damage scales inversely with model size.** SmolLM2-360M loses nothing, SmolLM2-135M
loses 0.06 on NER. At 100 MB there is not much redundancy left to throw away. Gemma's `+0.0433` on
xlam is not a gain — it is a near-floor score moving inside its own noise.

**The Q4 eval is 5–8× faster than bf16** (19.9s vs 157.9s on SmolLM2-360M NER). llama.cpp on a
sub-billion model is quick, which matters for §1.7: scoring both precisions per fit is cheap.

**Gemma is slower to quantize and smaller on disk** — 224s against SmolLM2-360M's 124s on the same
task, and 241 MB against 258 MB. Both are the 262k vocabulary again, and the size figure sharpens
the pool-ordering problem in §4.5: the model that is *cheaper* by `size_mb` is the one that scores
0.10 where its rival scores 0.45.

**Does this settle §1.4?** No, but it narrows it. The GGUF toolchain is demonstrably not broken in
general — six clean builds across two architectures. That makes "quantization is inherently
unstable here" less likely and leaves the Qwen3-0.6B-specific history (runs 38569606/38569608) as
the open question. §1.7's experiment still has to be run against Qwen3-0.6B itself.

### 4.7 Verdict on the three candidates

The run completed all six cells in 2h31m. Every zero-shot cell scored exactly 0.0000 — three
models, two architectures, two tasks — so each fine-tuned number is a clean delta from nothing.

**`SmolLM2-360M-Instruct` is the pick, and quantization makes the case stronger.** It is the only
candidate usable on both tasks, and on BC5CDR it does the thing this project exists to demonstrate:
**0.7339 against the Qwen3.6-35B teacher's 0.7190 at five-shot**, from a model ~97× smaller. It
reaches 91% of Qwen3-0.6B's BC5CDR score on 60% of the parameters, its 0.4500 on xlam is the only
sub-billion function-calling result here that is not near floor, and it is the **only one of the
three that loses nothing to Q4_K_M** — 258 MB deployed, same score as bf16 on both tasks.

**`SmolLM2-135M-Instruct` is the interesting floor.** 0.6107 on BC5CDR at **105 MB** — 4.4× smaller
than the current pool floor — and 0.2300 on xlam, which is weak but not zero. If the question is
"how small can this go", this is the answer to beat.

**`gemma-3-270m-it` is task-dependent and should not be a general pool entry.** It is the best of
the three on BC5CDR per unit of transformer, and the worst by a wide margin on xlam. A pool that
selects on `size_mb` would pick it over SmolLM2-360M on every task — 253 MB vs 271 MB, a 7%
difference — and on xlam that choice costs 0.39. It belongs in the pool only if selection can see
something other than file size.

**None of this should move the pool yet**, for the reason in §1: these are single fits and the
noise floor is unmeasured. The ordering that is safe to act on is the large gaps (0.06 vs 0.45 vs
0.23 on xlam); the small ones (0.61 vs 0.69) are not.

---

## 5. What changed in code

| # | Change | Where |
|---|---|---|
| 1 | `sms_spam` loader — dedup before split, stratified seeded holdout, checksummed bundle | `data/loaders/sms_spam.py` |
| 2 | `sms_spam` task spec — `minority_f1`, `ham`/`spam` definitions, cap 3000, `balance_labels(8)` | `tasks/sms_spam.py` |
| 3 | Registered; registry is nine tasks | `tasks/__init__.py` |
| 4 | Bundle vendoring script, refuses to write a bundle with any train/eval leak | `scripts/vendor_sms_spam.py` |
| 5 | Frozen bundle + manifest + sha256 | `data/local/sms_spam/` |
| 6 | GGUF serving prompt + stop tokens rendered from the served model, not hardcoded ChatML | `training/slm_helpers.py` |
| 7 | Sub-billion probe: zero-shot / fine-tuned / quantized per (model, task) | `scripts/probe_small_models.py` |
| 8 | Launchers: `sms_spam` ×2, `clinc150` CSE twin, probe ×2 | `tests/pipeline/` |

**Tests: 1,455 pass, 0 fail** (full suite, after the updates below).

Six suites needed updating and every one of them failed for the right reason — the
registry's coverage guards are load-bearing, not decorative. Adding one task broke
`test_all_eight_tasks_are_registered`, `test_every_task_module_is_registered`,
`test_categories_partition_the_suite`, `test_the_data_caps_are_uniform_across_the_suite`,
`test_the_fixture_table_covers_the_whole_registry`, `test_every_registered_task_has_a_row_builder`,
`test_every_registered_task_has_a_fixture`, and both slurm-script coverage invariants. Each one
was pointing at a real gap: a task with no QC fixture, no scoring fixture, no launcher, or an
undeclared cap exception. `test_the_deleted_scorers_are_gone` also failed, because it asserted
`data.loaders.sms_spam` cannot be imported — correct until today, and now removed from that list
with the reason recorded inline.

One test was rewritten rather than fixed: `test_infer_batch_gguf_fails_clearly_when_mode_cannot_be_controlled`
pinned the old non-Qwen refusal. It is replaced by three tests asserting the new contract.

## 6. Open items

| # | Item | Status |
|---|---|---|
| 1 | **Five repeat fits, scored bf16 AND Q4, with flat/nested `end` counts** (§1.7) — separates training variance from quantization instability | **recommended next; one short job** |
| 2 | Seed `SFTConfig` and pin torch determinism (§1.7) | open — only if (1) says training |
| 3 | Select checkpoints on the task metric rather than `eval_loss` (§1.7) | open — only if (1) says training |
| 3b | Strengthen `validate_and_record_gguf` to score real eval rows, not one "Hello" (§1.7) | open — only if (1) says quantization |
| 4 | `sms_spam` has never been run | ready — preflight green, both launchers exist |
| 5 | B225 — dataset version counter does not advance on an empty rebuild, so artifacts get overwritten | open, and it obstructed this diagnosis |
| 6 | `gsm8k` and `dialogsum` still have no CSE twin | open, trivial |
| 7 | Banking77 | **on hold** by your instruction |
| 8 | Sub-billion pool membership | **not started** by your instruction — §4 measures, it does not adopt |

## 7. Questions

1. **Run the noise-floor measurement?** It is the cheapest thing on the list and it decides how much
   of the last four runs' attribution survives. I would do this before another results run.
2. **`sms_spam`: submit now, or wait?** It is green and unrun. Worth noting it is the only task in
   the suite whose metric scores a collapsed model at exactly 0.0, so it doubles as a check on the
   loop itself.
3. **Do you want `gsm8k` and `dialogsum` CSE twins** while I am in the launcher directory?
