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

## On-device SFT suite (added 2026-09-06)

### DialogSum (generation, dialogue summarization)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| rouge_l | 0.3812 | BART-large fine-tuned | 400M | https://aclanthology.org/2021.findings-acl.449/ | 2026-09-06 |
| rouge_l | 0.3945 | BART-large + speaker/turn embeddings | 400M | https://aclanthology.org/2021.findings-acl.449/ | 2026-09-06 |

**THE CEILING IS HUMAN, AND IT IS NOT 1.0.** One annotator's summary scored against the other two
reaches **ROUGE-1 53.35 / ROUGE-2 26.72 / ROUGE-L 50.84**. Out-of-the-box models sit near 36
ROUGE-1 and the best fine-tuned near 47. So fine-tuning buys ~11 points and ~6 remain — and a 47
is about 88% of human, not 47% of perfect. The scorer carries all three ceiling numbers in
`per_class` so a report cannot lose them.

Full published ladder, for reading a run against: pointer-generator 33.77 / 9.24 / 32.18,
Transformer 35.91 / 8.74 / 33.50, distilBART out of the box 35.93 / 11.71 / 28.86,
UniLM 42.38 / 16.88 / 34.36 (R-1 / R-2 / R-L).

### W&I+LOCNESS BEA-2019 (generation, grammatical error correction)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| errant_f05 | 0.4300 | GPT-4 zero-shot | — | https://www.cl.cam.ac.uk/research/nl/bea2019st/ | 2026-09-06 |
| errant_f05 | 0.7124 | recent fine-tuned system (+/- 0.28 over seeds) | — | https://www.cl.cam.ac.uk/research/nl/bea2019st/ | 2026-09-06 |

⚠ **TWO CEILINGS APPLY AND BOTH ARE BELOW 1.0.**

1. **The oracle ceiling here is ~0.89**, measured: feeding the gold correction back in as the
   hypothesis scores F0.5 0.8934 on 120 dev sentences. The reference edits are the annotator's own
   segmentation while the hypothesis edits are derived by ERRANT's alignment rules, so the two
   lists differ even for identical sentences. A system at 0.75 is at ~84% of achievable.
2. **F0.5 is reference-count dependent and the effect is ~12 points.** These numbers are
   SINGLE-reference BEA-19 dev. CoNLL-14 at two references puts top systems near 68; the same
   systems re-scored against a 10-annotator extension reach 80-81 against a human 72.58. Never
   compare our F0.5 to a number computed under a different reference set.

Fine-tuning GPT-4o gained +22.07 F0.5 over its own zero-shot — the largest SFT delta in the suite.

### MultiCoNER II English (extraction, 33-class fine-grained NER)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| macro_f1 | 0.5300 | XLM-R baseline | 270M | https://aclanthology.org/2023.semeval-1.310/ | 2026-09-06 |
| micro_f1 | 0.6100 | XLM-R-Large + feature/loss engineering | 550M | https://arxiv.org/abs/2401.00698 | 2026-09-06 |

The published ladder is RoBERTa-base 0.31 -> XLM-R-Large 0.53 -> 0.61, with roughly 30 of those
points from feature, model and loss choices rather than scale. Note the metric split: `macro_f1`
is the headline and `micro_f1` is what this pipeline SELECTS on, so the 0.61 row is not a target
for the in-loop number.

### GoEmotions (classification, 28-label multi-label emotion)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| macro_auprc | n/a | — | — | — | — |
| macro_f1_28 | 0.4600 | BERT-base (std 0.19) | 110M | https://arxiv.org/abs/2005.00547 | 2026-09-06 |
| macro_f1_28 | 0.5400 | BERT-base + clipped asymmetric loss | 110M | https://arxiv.org/abs/2403.06108 | 2026-09-06 |

⚠ `macro_auprc` is this task's HEADLINE and is deliberately `n/a`: the literature reports
threshold-dependent macro-F1, and there is no citable macro-AUPRC to anchor against. The
`macro_f1_28` rows are informational — they are a thresholding artifact, which is exactly why the
headline is threshold-free — and a straight reproduction lands near 0.49.

⚠ Never let the tail carry a headline. Test support: grief 6, relief 11, pride 16,
nervousness 23, against neutral 1,787. A published `grief` 0.00 -> 0.57 F1 is +2 macro points
earned on six examples.

### TOPv2 (structured output, compositional semantic parsing)

| metric | value | model | params | source | checked |
|---|---|---|---|---|---|
| exact_match | n/a | — | — | — | — |

⚠ Deliberately `n/a` despite published numbers existing (RINE +13.0 EM over the seq2seq-pointer
baseline on reminder at 25 SPIS; shift-reduce in-order +3.5 / +2.4). Two independent reasons make
them non-comparable to ours, and a row here would invite exactly that comparison:

1. **Our SPIS splits are reconstructed, not the released files.** The official low-resource splits
   ship with the gated release and have no public mirror, so `data/loaders/topv2.py` reimplements
   the sampling rule. It lands within 1.6% of the released 25-SPIS sizes, which validates the rule
   — it does not make it the same file.
2. **EM depends on the parse serialization**, and ours is the mirror's `semantic_parse` string
   verbatim.

So this task is a controlled comparison against our own baseline. The pipeline uses its own
measured anchor, which is the safer default.

---

## Adding a row

1. Find a source that states the metric **by name** and the model scale.
2. Confirm the metric name matches `TASK_METRIC_NAMES[task_type]`, or expect the row to be
   informational only.
3. Add the row with today's date in `checked`.
4. Never estimate a value from a different benchmark, a different metric, or a parameter count.
   Leaving it `n/a` makes the pipeline use its own measured anchor, which is the safer default.
