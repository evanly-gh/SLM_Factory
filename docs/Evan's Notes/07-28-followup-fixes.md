# Follow-up Answers & Fixes

**Date:** 2026-07-28
Companion to [07-26-ner-math-review.md](07-26-ner-math-review.md). Answers the follow-up questions and records
the code changes made in response.

---

# Section 1 — Concept answers

## 1.1 What are pos / neg / boundary for?

The eval set is deliberately **stratified into three disjoint slices** so a single F1
number can't hide a specific failure mode. From
[`data/eval_set.py`](../../data/eval_set.py):

| Slice | NER (your run) | What it catches |
|---|---|---|
| **pos** (360) | Entity-rich passages with gold annotations | Can it find entities at all? |
| **neg** (360) | Entity-free passages | **Hallucination test** — does it invent entities that aren't there? |
| **boundary** (180) | Overlapping entity types / partial matches | Does it get *edges* right (Chemical vs Disease when both plausible)? |

Per family:
- **classification** — pos = clear positive class, neg = clear negative, boundary = confusable pairs
- **math/generation** — pos = well-formed problems, neg = adversarial/ill-posed, boundary = multi-step edge cases

**Why it matters:** a model that outputs "no entities" for everything scores 100% on
`neg` and 0% on `pos`. A single aggregate F1 would show ~50% and look mediocre; the
slices show it's *degenerate*. `EvalSet.all = pos + neg + boundary`, and the F1 you see
reported is computed over all three.

This is a **different axis** from easy/medium/hard. pos/neg/boundary is about *what kind
of input*; difficulty is about *how hard*. Every example has both labels.

## 1.2 What is `iterate_json_reask`? Is the prompt growth a bug?

**`iterate_json_reask` is a retry.** The orchestrator is asked to return a JSON decision.
If the first response is prose, a tool call, or fails schema validation, the pipeline
sends one follow-up: *"Do NOT call any tools and do NOT include any prose… Respond NOW
with ONLY the decision JSON"* — plus the exact validation error. That second call is
logged under the `iterate_json_reask` stage. One retry only; if it fails again, the
heuristic fallback takes over.

**On prompt growth — you're right, and I was wrong to imply otherwise.**

Growth *because you're attaching more trajectory context* is correct and desirable. The
orchestrator needs the score history, tried configs, and diagnosis to make a good
decision; a decision made without them would be worse. The 3.3K → 23K token growth is
mostly that, and it's the system working as intended.

What is genuinely worth flagging is narrower:

1. **The reask doubles it.** Each failed decision paid for a full-size prompt *twice* and
   then discarded both. In the NER run that was **$4.90 of $13.41 (36%)** on retries
   whose results were thrown away. That's the waste — not the context itself. **Now fixed**
   (§2.3): the failure that caused every one of those reasks no longer occurs.
2. **Nothing caps it.** Prompt cost grows with run length and there's no compaction or
   windowing. At 142 iterations it's affordable ($13); the concern is a run 5× longer.

So: growth = good, unbounded growth with a 2× retry multiplier = the actual problem.

## 1.3 Span-F1, exactly

Span-F1 scores **which text spans you extracted and what type you gave each**. A
prediction counts as a true positive only if it matches a gold span **exactly** — same
start, same end, same entity type.

Take one abstract:

```
Text:  "Lidocaine-induced cardiac asystole occurred after the second dose."

Gold:  [Lidocaine        → Chemical]
       [cardiac asystole → Disease ]

Model: [Lidocaine        → Chemical]   ✅ exact match          → TP
       [asystole         → Disease ]   ❌ wrong boundary       → FP (and the gold
                                          span it should have matched → FN)
       [dose             → Chemical]   ❌ not a gold entity    → FP
```

Counting:
```
TP = 1      (Lidocaine)
FP = 2      (asystole — wrong boundary; dose — spurious)
FN = 1      (cardiac asystole — gold span never matched)

Precision = TP / (TP + FP) = 1 / 3 = 0.333     "of what I predicted, how much was right"
Recall    = TP / (TP + FN) = 1 / 2 = 0.500     "of what existed, how much did I find"

F1 = 2 · P · R / (P + R) = 2(0.333)(0.500) / (0.833) = 0.400
```

Three properties worth internalizing:

- **No partial credit for a near-miss.** `asystole` vs `cardiac asystole` scores exactly
  the same as a completely wrong guess. This is why boundary precision dominates BC5CDR.
- **A near-miss is punished twice** — once as a false positive, once as the false
  negative for the gold span it failed to match.
- **Counts pool across the whole eval set**, not per-example. All TPs/FPs/FNs from all
  900 examples are summed, *then* P/R/F1 are computed once. This is exactly why the
  span-F1 (0.8272) and the per-example bucket accuracies (mean 0.691) don't reconcile —
  see §2.7 of the previous doc.

## 1.4 Does quantization matter enough to keep evaluating it every iteration?

**Yes — and the new data says so more strongly than the NER result did.**

The NER sweep (fine-tuned, span extraction) showed a 0.0008 gap, which argued for
skipping. The base-model GSM8K sweep argues the opposite:

| | bf16 | Q8_0 | Q4_K_M | bf16→Q4 gap |
|---|---|---|---|---|
| **NER, fine-tuned** (job 37810980) | 0.8636 | 0.8627 | 0.8628 | **0.0008** |
| **GSM8K, base model** (job 37818082) | **0.8213** | **0.8100** | **0.7913** | **0.0300** |

On-disk size for the GSM8K row: bf16 8888.1 MB → Q8_0 4397.0 MB (49%) → Q4_K_M 2654.5 MB (30%).

```
TRADEOFF vs bf16
  Q8_0         loses 1.370% accuracy for 2.02x smaller
  Q4_K_M       loses 3.653% accuracy for 3.35x smaller
```

**The bf16→Q4 gap on math is 0.0300 — 37.5× the 0.0008 gap on NER**, and the degradation is
monotonic in bit-width (0.8213 → 0.8100 → 0.7913), which is what you'd expect from a real
effect rather than noise. The fresh Q4_K_M measurement (0.7913) reproduces the math run's
own recorded base zero-shot exactly, so this is not a harness artifact.

That matches the mechanism: multi-step arithmetic compounds a perturbed token across a
long chain and only the final answer is scored, whereas NER copies short spans out of
the input with high confidence.

**Recommendation: keep evaluating the quantized artifact every iteration.** The 7.6 h of
GGUF builds is the price of knowing your reported number is the number the phone will
produce. On a reasoning task, scoring BF16 and shipping Q4 would have overstated
accuracy by **0.0300** — comparable to the entire improvement fine-tuning achieved
(+0.035 across the whole 4B tier). You would have reported most of your gain as real
when a third of it evaporates on the device.

If you still want the time back, the safe version is (b) from the earlier doc — skip the
*rebuild* when an iteration's config is hyperparameter-identical to one already built —
which changes no numbers at all. Do not take option (a).

> **Resolved 2026-07-28.** Job 37818082 completed; the falsifiable condition stated here
> was "if all three land within 0.002, the conclusion doesn't hold." They landed 0.0300
> apart, monotonically. The conclusion holds. Full report:
> `logs/quant_eval/qwen35-4b-base-gsm8k/report.md`.
>
> Note this generalizes the earlier NER finding in the *opposite* direction from how I
> first framed it: quantization is nearly free on span extraction and materially costly
> on reasoning. Neither result transfers to the other task type — which is the argument
> for measuring it per task rather than deciding once.

## 1.5 The GGUF load failure — what was it, and is it fixed?

**It was a transient, and no, I did not fix it — because there is nothing to fix in our
code.** Being precise since I glossed this earlier:

```
[evaluate] Baseline measurement failed (CUDA worker 'eval' failed (exit=1,
  ValueError: Failed to load model from file:
  artifacts/gguf/Qwen_Qwen3.5-2B/663bb41fea0a/model-q4_k_m.gguf)
```

llama.cpp failed to load a GGUF that had just been written. The *same* build path
succeeded moments later for the training evals, and the run went on to complete 74
iterations on that model — so the file was fine and the load was a one-off (most likely
the file not being fully flushed/visible on Lustre at open time).

There is already a guard for exactly this: `validate_and_record_gguf` loads every
freshly-built GGUF through llama.cpp and writes a SHA-256 sidecar only on success, so a
genuinely corrupt artifact is caught at build time and never cached.

**What I did fix is the consequence, not the cause** (§2.2): the failure was silently
recorded as `baseline_f1 = 0.0`, which is what turned a transient into a permanently
wrong number in the final report.

## 1.6 Train / validation / test — is the held-out set being used for training?

**No. There are genuinely three separate splits, and I verified the firewall.**

| Split | Size (NER) | Where it comes from | What uses it |
|---|---|---|---|
| **Train** | ~88% of curriculum | `dataset_vN.jsonl` minus the val slice | Gradient updates |
| **Validation** | ~12% of curriculum | Carved out **inside the trainer** at [`lora_trainer.py:704-715`](../../training/lora_trainer.py#L704-L715) (`SLM_VAL_FRACTION=0.12`, seed 1234) | `eval_loss`, early stopping, best-checkpoint pick **within one training run** |
| **Test** | 900 held out | `artifacts/eval_set.json`, built once at run start | The F1 the pipeline optimizes **across iterations** |

So the `eval_loss` in row 2 of the training output is computed on a slice of the
**training data**, never on the test set. The two never meet.

**The firewall is real and it fires.** Three independent layers:
1. **Source-level** — train/test overlap removal at acquisition: *"overlap removal for
   'BC5CDR': removed 24 train row(s); official test rows unchanged"*. In math it
   rejected an entire candidate dataset for 128 overlapping rows.
2. **Curation-level** — `_exclude_eval_rows()` runs on every candidate row at three
   call sites in `curate.py`, matching on normalized text.
3. **Decision-level** — `_reject_eval_text_strings()` refuses any orchestrator decision
   that quotes held-out text, so eval content can't leak through a hypothesis string.

The one thing worth knowing: the validation slice is **12% of your training data taken
away from training**. On the NER curriculum (3403 rows) that's ~408 rows not learned
from. That's a deliberate, standard trade for early stopping — but if you ever want it
back, `SLM_EARLY_STOPPING=0` disables it.

---

# Section 2 — Changes made

## 2.1 A Claude API failure now stops the run

[`agent/llm_errors.py`](../../agent/llm_errors.py) — previously only billing/auth/quota
errors were fatal; a timeout, 500, overload, or rate limit fell through to a hard-coded
heuristic and the run continued while its logs still attributed each step to the
orchestrator.

`is_api_transport_error()` now walks the exception's MRO for an `anthropic.*` or
`httpx.*` module (plus bare `ConnectionError`/`TimeoutError`) — unambiguous, unlike
message matching. Any hit is fatal.

**Deliberately NOT fatal:** JSON-parse and schema-validation failures. Those mean the
API answered fine and *we* rejected the content, which the reask path handles.

Also closed a silent hole: `hardware_research` caught an API failure and substituted
`{usable_ram_mb: 3000, storage_budget_mb: 1500, rationale: "fallback defaults"}`. That
quietly changes which models the run may even consider. It now raises.

## 2.2 Failed baselines record `n/a`, never `0.0`

[`agent/nodes/evaluate.py`](../../agent/nodes/evaluate.py) records `baseline_f1 = None`
on failure and logs `Baseline F1 = n/a (measurement failed)`. The report renderer
already handled `None → "n/a"` — the lie was purely upstream.

Also fixed [`escalate.py`](../../agent/nodes/escalate.py), which defaulted a *missing*
baseline to `0.0` and would have reintroduced the same problem.

Effect on the NER report: tier 2 becomes `n/a` instead of claiming a `+0.8476`
improvement over a measurement that never happened.

## 2.3 `data_rebuild` is now actually orchestrator-driven

Three separate defects, all fixed:

**(a) The rejection that disabled the whole path.** The validator raised on
`hyperparams` accompanying a `data_rebuild`. Claude attaches that block to essentially
every data plan it writes, so **65 of 65 were rejected** and zero orchestrator-authored
plans ran in 142 iterations.

The rule's *intent* is "a data_rebuild must not also change hyperparameters." **Stripping
the field enforces that exactly**; raising did not — it discarded the data plan and fell
through to a heuristic that also held hyperparameters fixed. Now stripped.

**(b) The fallback couldn't reach 5 of its 6 strategies.** It keyword-matched the
hypothesis prose (`"hard" in hypothesis` → difficulty_weighted_sampling). But the only
two test-agent diagnoses that *suggest* `data_rebuild` contain none of the matched
keywords — so every fallback fell through to the catch-all `resample_existing`, 31/31.

Replaced with `_fallback_strategy_from_signal()`, which reads the same measured signals
the orchestrator sees — per-difficulty accuracy, confusion pairs, which strategies have
already been tried — and prefers an untried strategy. Also fixed: material budgets
(`new_real_rows`, `synth_rows`) are now set to match the chosen strategy, and difficulty
weights are computed **inversely to measured bucket accuracy** rather than from prose.

**(c) Synthesis was unreachable.** Gated at `score >= 0.95`, above both stop thresholds
(0.88, 0.82) — it could only fire in a run that had already won. Synthesis is now also
reachable from a difficulty gap at any score. The near-ceiling path is preserved.

**Plus: exhaustion no longer crashes.** The plan search now rotates the **primary
strategy** before giving up (previously it bounded the space at ~1 × 8 query_variants ×
5 fractions ≈ 40 plans), and raises a typed `DataRebuildPlanSpaceExhausted`.
`curate_node` catches it and terminates cleanly with the best model preserved — instead
of propagating an uncaught `ValueError` that killed a 44.8-hour run.

**10 new tests** in `tests/nodes/test_data_rebuild_flexibility.py`.

## 2.4 Hyperparameters reduced 9 → 5

The orchestrator now tunes exactly: **`lora_rank`, `alpha_ratio`, `weight_decay`,
`learning_rate`, `nr_epochs`**.

| Removed | Why |
|---|---|
| `micro_batch_size`, `gradient_accumulation_steps`, `effective_batch_size` | Only the *effective* batch changes what the model learns; the micro/accum split is a VRAM-fitting decision. Math iterations 14/19/24/39/52 held every learning parameter fixed and only reshuffled this split — measuring ±0.01 of pure noise across ~35 min of GPU time. |
| `lora_dropout` | Duplicates `weight_decay` as a regularizer and never produced a new best in either run, while `weight_decay` produced the single largest hyperparameter gain (+0.033). |
| `lora_alpha` | Replaced by `alpha_ratio` — alpha only matters as `alpha/rank`, so exposing both invited incoherent pairs. `alpha = rank × ratio`, ratio ∈ {1,2,4}. |

Retired fields are **rejected with an actionable message** rather than silently ignored,
and are still accepted when replaying old checkpoints/DAG history. The deterministic
fallback ladder was narrowed to the same five axes.

---

# Verification

**Full suite: 791 passed, 0 failed** (228 s). The two flakes seen on earlier runs —
`test_apps_prelude_exposes_numpy…` (cold-Lustre numpy import) and
`test_checkpoint_kill_resume_smoke` (24 s against a 30 s subprocess limit) — both passed
here, confirming they were machine-saturation artifacts rather than real breakage. Both
also pass in isolation.

New tests this session: 10 (data_rebuild flexibility) + 9 (no-modelled-metrics) + 5
(config diff) + 4 (downward probe) + 4 (checkpoint purge) + 6 (GGUF retention) + 5
(log split).

**Not committed** — consistent with the earlier decision, since these files carry
substantial unrelated uncommitted work of yours.
