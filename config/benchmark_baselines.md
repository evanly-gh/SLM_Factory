# Benchmark baselines registry

**Purpose.** Published reference scores for benchmarks this pipeline may target, used to
calibrate `stop_threshold`. This file exists so the accuracy goal comes from a **reviewable,
diffable, sourced record** instead of the orchestrator's frozen training recall.

**Why recall was not good enough.** The planner prompt used to say "anchor `stop_threshold` to
the PUBLISHED STATE-OF-THE-ART for this benchmark at ~{param_range} scale". That is exactly the
kind of number an LLM recalls confidently and wrongly, and the error is asymmetric:

- set **too low** → a base model's zero-shot already clears it, the run "converges" at
  iteration 1 having learned nothing;
- set **too high** → every tier fails, the run burns its whole budget concluding "infeasible".

Both failure modes were observed. Benchmark research on 2026-07-28 found BANKING77 SOTA is
~94.8% from a **110M** encoder (so a 1–4B target is nonsense), and that MedQA is saturating.

---

## Contract

Every row must carry: `metric`, `value`, `model`, `params`, `source`, `checked`. A row with a
missing field is ignored by the loader rather than guessed at.

**Metric names are load-bearing.** They must match `eval/harness.py::TASK_METRIC_NAMES` for the
task type, because a value is only comparable to what this pipeline measures if the metric is
the same. `macro_f1` ≠ `accuracy` ≠ `span_f1` ≠ `exact_match`. A mismatch means the row cannot
calibrate the threshold and the run falls back to the measured anchor.

**`params` is the model scale the value was achieved at**, not the scale you plan to deploy. A
SOTA from a 110M encoder tells you the task is easy; it does not tell you what a 1.7B decoder
will do. The loader returns it so the caller can reason about the gap.

**`checked` is the date a human verified the row.** A stale row is still better than recall
because you can see that it is stale.

---

## Registry

### GSM8K (math_reasoning)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| exact_match | 0.8779 | Qwen3-4B Base | 4B | https://arxiv.org/pdf/2505.09388 | 2026-07-28 |

Note: the pipeline's own measurement of Qwen3.5-4B **base, zero-shot** on its held-out GSM8K
slice was bf16 0.8213 / Q8_0 0.8100 / Q4_K_M 0.7913 (slurm 37818082) — i.e. below the published
figure, which is expected: different eval slice, different prompt, non-thinking mode.

### BC5CDR (NER)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| span_f1 | n/a | — | — | — | — |

No comparable published span-F1 with a stated protocol was located. Leave absent so the run
uses the measured anchor.

⚠ **2026-08-12: leaving this `n/a` had a cost.** With no row here the planner fell back to
recall and set `stop_threshold=0.88` for `slm-ner-l40s-37531245`. The run reached **0.8628**
(Qwen3.5-4B @ Q4_K_M, 900-row eval, exact `(text, type)` multiset match), never cleared the bar,
and burned 44.8 h and $13.43 before crashing on an exhausted data-rebuild plan space. Our own
measured figures, for calibration — these are pipeline measurements, not published SOTA, so they
are deliberately **not** formatted as a registry row:

| what | value |
|---|---|
| Qwen3.5-4B Q4_K_M, zero-shot | 0.0254 |
| Qwen3.5-2B Q4_K_M, best fine-tuned (74 iters) | 0.8476 |
| Qwen3.5-4B Q4_K_M, best fine-tuned (68 iters) | 0.8628 |

Published encoder-class span-F1 on BC5CDR sits around 0.87–0.90 at ~110M params, but under
exact-match protocols that do not demonstrably agree with `eval/metrics.py::entity_f1` — which is
exactly why no row is asserted. Set the next NER run's threshold from the 0.8628 measured anchor,
not from recall.

### Calendar NL→JSON (function_call)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| ast_arg_match | n/a | — | — | — | — |

No benchmark with a comparable metric exists. SMCalFlow is the canonical calendar semantic-parsing
task and reports **0.7375 exact-match program accuracy** on its hidden test set
(https://microsoft.github.io/task_oriented_dialogue_as_dataflow_synthesis/, checked 2026-08-12),
but that is Lispress program equality, not JSON argument matching, so it will not calibrate this
pipeline and is recorded as context only. TOPv2 exact-match accuracy is likewise a bracket-parse
metric. Use the measured anchor.

### BANKING77 (classification)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| accuracy | 0.9477 | SPACE 2.0 | <1B (encoder) | https://arxiv.org/pdf/2305.02468 | 2026-07-28 |

⚠ Achieved by a sub-1B **encoder**; SetFit reaches few-shot SOTA on a 110M sentence encoder
(https://arxiv.org/pdf/2209.11055). Also has documented label errors
(https://aclanthology.org/2022.insights-1.19.pdf), so treat >0.95 as unreachable noise.
The registry metric is `accuracy`, which does **not** match this pipeline's `macro_f1` for
classification — so this row is informational and will **not** auto-calibrate.

### i2b2-2014 de-identification (NER)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| span_f1 | 0.9785 | BiLSTM-CRF | <10M | https://arxiv.org/abs/1606.03475 | 2026-07-28 |

### MedQA / USMLE (classification, multiple choice)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| accuracy | 0.8650 | Med-PaLM 2 | 340B | https://arxiv.org/pdf/2212.13138 | 2026-07-28 |

⚠ Saturating for frontier models; no 1–4B figure located. Metric is `accuracy`, not
`macro_f1` — informational only.

### SMS Spam Collection (classification)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| macro_f1 | n/a | — | — | — | — |

Widely used but without a canonical split or a single citable SOTA. Absent on purpose.

---

## Adding a row

1. Find a source that states the metric **by name** and the model scale.
2. Confirm the metric name matches `TASK_METRIC_NAMES[task_type]`, or expect the row to be
   informational only.
3. Add the row with today's date in `checked`.
4. Never estimate a value from a different benchmark, a different metric, or a parameter count.
   Leaving it `n/a` makes the pipeline use its own measured anchor, which is the safer default.
