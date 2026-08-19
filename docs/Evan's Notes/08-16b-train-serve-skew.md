# Train/serve skew, the single_model run, and the pipeline launch — 2026-08-16

**Headline:** the `single_model` xlam run took three submissions, because the first two surfaced a bug
that had been silently destroying fine-tuning on `Qwen3-4B-Instruct-2507`. Training was teaching the
model to emit a `<think></think>` block that inference never pre-filled, so every fine-tuned prediction
began with two stray tags. Fixed and verified: the first fine-tune went from **0.6120 (worse than the
0.8010 untrained baseline) to 0.8230, and the run's best is now 0.8460**. The six other curated tasks
are launched under `smallest_first`.

---

## 1. What was asked, and what happened

Run xlam under `single_model` (orchestrator picks one model, then no escalation and no regression),
diagnose anything that looks wrong, fix and resubmit as needed, then launch the full pipeline under
`smallest_first`.

Three submissions:

| job | outcome |
|---|---|
| `38561204` | aborted — found B288 (eval cap) and the scores that led to B290 |
| `38565344` | aborted — isolated the real cause, B290 |
| `38566712` | **healthy, running** — this is the real ablation |

The two aborted logs are kept as `logs/slurm/ABORTED-b28x-*.out`.

## 2. The bug that mattered: B290

`Qwen3-4B-Instruct-2507` is thinking-free — its official chat template renders a bare assistant prefix.
But `FastLanguageModel.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")` **does not load that repo**. It
redirects to `unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit` and hands back the mirror's tokenizer,
whose template applies the hybrid-Qwen3 think-block convention anyway:

```
official Qwen/Qwen3-4B-Instruct-2507  -> '<|im_start|>assistant\nANSWER<|im_end|>\n'
unsloth mirror (what training loads)  -> '<|im_start|>assistant\n<think>\n\n</think>\n\nANSWER<|im_end|>\n'
```

`_build_completion_only_rows` builds the prompt with `add_generation_prompt=True` — which gives the bare
prefix even under the mirror's template — and the full text with the assistant message. So the think
block falls *inside* `completion_mask`: the model was explicitly **supervised to emit it**. Inference
follows the official template and doesn't pre-fill it, so those supervised tokens came out as the first
tokens of the answer.

The control that settles it, same three eval rows:

```
BASELINE   (no adapter):  [{"name": "create_histogram", "arguments": {...}}]   <- clean
ITERATION 1 (fine-tuned): </tool_call> </tool_call> [{"arguments": {...}}]      <- two stray tags
```

The parser salvages rows where the JSON still follows and fails rows where the model stops after the
tags. That is the entire spread of scores I was chasing — **0.0000, 0.0887, 0.3010, 0.6120, 0.7887,
0.8137** — against a stable untrained baseline of 0.8010. The loop kept concluding, correctly given its
numbers, that *"fine-tuning did not improve on zero-shot."*

The tags print as `</tool_call>` rather than `</think>` because llama.cpp renders those GGUF
special-token IDs under different names. That cost me hours on the wrong hypothesis.

### Why it would have shipped

`merge_for_quantization` pins the **official** base, so the deployed GGUF is served under the official
template. This was not an artifact of our harness — the skew would have followed the model onto the
phone. That is why the fix pins the **served** template for training rather than teaching inference to
send Unsloth's prefix; the latter would have hidden the skew in our eval while shipping a broken model.

### Fix

1. `_pin_serving_chat_template` — replace the loaded tokenizer's template with the official base
   model's, so the training target and the deployment contract are the same object. Confirmed live:
   `[train] Pinned chat template to the SERVED model Qwen/Qwen3-4B-Instruct-2507 — the loaded
   (Unsloth mirror) template differed`.
2. `_assert_train_serve_prefix_alignment` — before training, require the inference prompt to be a
   strict prefix of the rendered training text with **nothing** between it and the answer. Raises, not
   warns: an hour of GPU time yielding a silently crippled adapter is worse than a fast failure.

19 tests in `tests/training/test_train_eval_prefix_alignment.py`, two of which run against the **real
cached vendor tokenizers** — one asserting the official templates satisfy the invariant, one asserting
the Unsloth mirror violates it, so if upstream fixes their template we find out rather than carrying the
workaround forever.

### Scope

Only `Qwen3-4B-Instruct-2507` was affected — it is the only pool model with a thinking-free official
template, hence the only one where our inference prefix omits a block the mirror inserts. Hybrid Qwen3
tiers (0.6B / 1.7B / 8B) pre-fill it on both sides and were always aligned.

**So: previously reported tier-0/1/2 numbers stand. Any run that selected the 4B Instruct model was
measuring the base model, not fine-tuning.**

## 3. Result after the fix

| | before | after |
|---|---|---|
| baseline (untrained) | 0.8010 | 0.8010 |
| first fine-tune | 0.6120 | **0.8230** |
| best so far | base model kept | **0.8460** |
| trajectory | 0.000-0.814, chaotic | 0.823, 0.846, 0.837, 0.826 |
| hard bucket | 0.377 | 0.439 -> **0.507** |

Predictions are pure JSON with no tags. All three difficulty buckets are populated and improving.
Rollback discards regressions correctly. No escalation or downward probe — `single_model` is holding the
pin, which is the thing the ablation was meant to test.

Goal is 0.8680, calibrated from the Qwen3.6-35B teacher's own zero-shot score on the same 1,000 rows.
Best is 0.8460, so it is 0.022 short and still iterating. **Not converging is a legitimate outcome
here** — the goal asks a 4B Q4 model to match a 35B teacher.

## 4. Two smaller bugs

**B288 — the eval cap was applied twice, so B282's fix was a no-op.** `eval_size_target` (800) was used
both as the loader's `max_test` and again as `build_eval_set(target=...)`; B282 only changed the first,
so curated benchmarks still got 800 rows. Compounding it, my own first attempt at this fix targeted a
source string *that did not exist in the file* — `str.replace` with no match is a silent no-op, and
nothing asserted the match, so the "fix" shipped as nothing. Now 1,000 rows, verified live:
`eval set built: 1000 examples`. The lesson is to assert new **behaviour**, which the 7 new tests do.

**B289 — a GGUF that loads but decodes to garbage was scored as a real 0.0000.**
`validate_and_record_gguf` only ever asked llama.cpp to *open* the file, never to generate. Added a
generation smoke test plus a rebuild-once retry (transient corruption recovers; a second failure raises
and stops the run rather than being scored).

**I got B289's root cause wrong at first and the BUGS.md entry now says so.** I argued "final eval_loss
was 0.1221, so the model cannot be emitting `</tool_call>` on 800/800 rows." Teacher-forced eval loss
supplies the gold prefix at every position, so it is blind to what the model emits *first* when
generating from the prompt alone — precisely this failure mode. The fix is still worth having as a
backstop, but it was not the cause of those scores.

## 5. Pipeline launch

Six curated tasks submitted under `smallest_first` (the default):

| task | job |
|---|---|
| clinc150 | 38569251 |
| routerbench | 38569298 |
| ner_bc5cdr | 38569299 |
| calendar_json | 38569300 |
| dialogsum_samsum | 38569301 |
| proactive_listening | 38569302 |

xlam under `smallest_first` is deliberately **not** submitted yet — the `single_model` ablation is still
using it, and running both concurrently would contend for GPUs and share the `artifacts/merged` and
`artifacts/gguf` trees. Submit it when `38566712` finishes.

**All six benefit from the B290 fix**, which matters most for any task whose ladder reaches the 4B
Instruct tier.

## 6. Open / for review

- **B290's underlying cause is upstream.** Unsloth ships a template that is wrong for this checkpoint.
  We work around it; a bug report upstream would be reasonable.
- **B289's corruption question is moot but unproven.** Now that B290 explains the scores, there may
  never have been intermittent GGUF corruption at all. The smoke test is cheap insurance either way.
- **Per-row eval predictions are not persisted.** This whole diagnosis ran off the 3-row sample display
  that happens to be printed. Saving predictions per iteration would have made it minutes, not hours.
  Worth doing.
- **`tests/test_capability_prompts.py::test_code_planner_and_model_choice_prompts_target_apps_introductory`
  still fails.** Pre-existing, unrelated, untouched. Everything else is green: 1,168 pass, 2 skipped.
- **Routerbench mode collapse is still unresolved** and was not revisited here.
