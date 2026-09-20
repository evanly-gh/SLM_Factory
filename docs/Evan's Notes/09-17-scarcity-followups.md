# 09-17 — Five follow-ups on the data-scarcity ablations

Answers to five questions about the sms_spam (100 real rows) and clinc150 (151 real rows) A/B pairs.
Every number here is measured, not inferred; probe jobs are named so each claim can be re-run.
Results tables live in `09-16-baselines-ablations-and-probes.md` Part III.

---

## 1. sms_spam's tier-1 +0.2981 is real, not a bug, harness artifact or warmup

Three independent checks, any one of which would have caught an artifact.

**The gap reproduces from disk, cold.** Probe **40256115** re-scored the two saved tier-1 best
adapters against each run's own frozen 1,000-row eval set, months of wall-clock after the fact:

| checkpoint | run reported | re-score pass 1 | pass 2 |
|---|---|---|---|
| arm A tier-1 best (`iter2-d1`) | 0.6369 | 0.6387 | 0.6369 |
| arm B tier-1 best (`iter18-d3`) | 0.9350 | 0.9350 | 0.9350 |

**The two arms' tier-1 models fail in opposite directions**, which no measurement error produces.
Reconstructed from every tier-1 iteration's confusion counts (872 ham / 128 spam):

| | false positives (ham→spam) | precision | models |
|---|---|---|---|
| arm A, 100 real rows | **114 – 779** | 0.138 – 0.496 | all 15 fine-tunes |
| arm B, + synthetic rows | **0 – 18** | 0.867 – 1.000 | 20 of 21 fine-tunes |

Arm A never stops calling legitimate messages spam; its *best* model still mislabels 114 of 872 ham
rows. Arm B essentially never false-positives and its residual error is missed spam instead. That
is a regime change, not a scoring wobble.

**A config-matched control isolates the data as the cause.** Both arms trained the default config
(`r=16 a=32 wd=0.01 lr=2e-04 ep=3`) on the md5-identical 100-row file and both scored **0.2353**
with 819/1000 failures. Arm B then ran that same config on 100 real + 208 synthetic rows and scored
**0.7700** with 49 failures. Two further pairs match on config across arms: 0.6369 → 0.8111 and
0.3303 → 0.8927.

Warmup is ruled out by arm A's own trajectory: 15 fine-tunes spanning rank 16–64, lr 5e-5 to 2e-4,
weight decay 0–0.1 and 3–6 epochs, every one landing between 0.2353 and 0.6369.

---

## 2. Why carried synthetic data sometimes made things worse

**The curriculum's class prior drifts away from the deployment prior, and every synthetic batch
makes it worse.** The eval stream is 12.8% spam. The anchor was built 50/50, and each synthesis
round added rows that were 52–59% spam:

| curriculum | v1 | v2 | v5 | v9 | v14 |
|---|---|---|---|---|---|
| rows | 100 | 308 | 676 | 1,261 | 1,836 |
| spam share | 50.0% | 51.6% | 53.7% | 55.3% | **55.8%** |

At 55.8% spam the curriculum is **4.4× spam-enriched** relative to what the model will see. Training
longer on that distribution moves the decision boundary toward "spam", and the failures confirm it
— the regressions are false-positive blowups, not missed spam:

| arm B iteration | rows | score | ham→spam | spam→ham |
|---|---|---|---|---|
| T5 iter 1 (carried curriculum, default config) | 1,261 | 0.6740 | **114** | 5 |
| T5 iter 7 (kept, best of the run) | 1,679 | 0.9528 | 5 | 7 |
| T5 iter 13 (rolled back, +174 rows) | 1,853 | 0.7011 | **98** | 6 |
| T2 iter 10 (rolled back) | 625 | 0.6649 | **127** | 1 |

The clearest case is tier 5 iteration 1, the one place carried data looks strictly harmful: 1,261
inherited rows scored 0.6740 where arm A's 100 rows scored 0.7595 on the same model. Arm A's model
was barely trained (38 optimizer steps) and so stayed near the base model's ham-leaning prior; arm
B's saw 473 steps of a 55%-spam curriculum and inherited its prior instead.

It was recoverable rather than fatal. Continuing to add rows *with a config that compensated* walked
tier 5 back: false positives fell 114 → 82 → 58 → 32 → 5 as the curriculum grew 1,261 → 1,679, and
the arm finished at 0.9528 against arm A's 0.9339. So the failure is a **calibration** effect of a
prior-mismatched curriculum, not evidence that synthetic rows are bad — and the fix is to generate
to the deployment prior instead of 50/50.

---

## 3. The clinc150 iteration-1 gap is a real model difference; the eval is not the noisy part

Probe **40256115** re-scored both arms' saved iteration-1 adapters, twice each:

| checkpoint | run reported | pass 1 | pass 2 | stable across passes |
|---|---|---|---|---|
| arm A `iter1-d1` | 0.2781 | 0.2781 | 0.2781 | yes |
| arm B `iter1-d1` | 0.2934 | 0.2943 | 0.2943 | yes |

Both reproduce their run's number to within 0.0009, and repeat identically. **So the 0.0153 gap is
two genuinely different models being measured accurately**, not eval noise. Of four checkpoints
re-scored twice, three were bit-stable and one (sms_spam arm A) moved 0.0018, so eval variance is
≤0.002.

The difference originates in training, which is not bit-reproducible (Unsloth's fused kernels — see
09-16 Part III.3), and the scoring path amplifies it:

| the same two adapters, scored… | gap |
|---|---|
| bf16, 5,500-row report split | 0.0013 |
| Q4_K_M, 5,500-row report split | 0.0061 |
| Q4_K_M, 1,000-row in-loop eval | 0.0153 |

**Are all results subject to ±2.5%? No.** Two figures get conflated. The ±2.5 points in
`scripts/report_eval.py`'s docstring is the *sampling* confidence interval for an absolute score at
n=1,000 — it applies to "this model scores X", and it cancels in a paired comparison because both
arms are scored on the same rows. The run-to-run variability *between* two identically-trained
models is what these probes measure: **0.0013 on the report split, ~0.015 in the loop.** So
per-iteration numbers carry roughly ±0.015 and published numbers should come from `report_eval.py`,
where the same nondeterminism costs ten times less.

---

## 4. Generated data quality: correct, in-distribution, and less diverse than gold

Audited every synthetic row in the final curriculum of both arm Bs (1,736 sms_spam rows, 1,119
clinc150 rows).

| check | sms_spam | clinc150 |
|---|---|---|
| synthetic rows whose text appears in the **eval set** | **0** | **0** |
| exact duplicates of another synthetic row | 0 | 0 |
| verbatim copies of a gold anchor row | 0 | 0 |
| labels outside the gold label set | 0 | 0 |
| label-space coverage | 2 of 2 | **151 of 151** intents, min 2 / median 8 rows each |
| text length, p10/p50/p90 chars (gold) | 53/124/189 (29/126/160) | 21/38/59 (22/35/59) |
| **near-duplicate rate** (4-gram Jaccard ≥ 0.6) | **11.6%** (gold 0.0%) | **3.8%** (gold 0.0%) |
| size-matched type/token ratio (gold) | **0.208** (0.431) | 0.354 (0.377) |

No contamination, no invalid labels, no degenerate copying, and lengths that match gold closely.
Samples read like the real thing: `URGENT! Your account has been compromised. Click
http://secure-bank-verify.com/login…` / `hey can u pick up some milk on ur way home? thx`, against
gold's `FREE RINGTONE text FIRST to 87131…` / `U sick still can go shopping?`.

Two real defects. **sms_spam's rows are template-collapsed** — 11.6% are near-duplicates of another
row and, size for size, they carry less than half gold's lexical variety, with several independent
"pick up milk on your way home" ham messages. **And the class balance is wrong for the task**, at
55.8% spam against the eval's 12.8%, which is the cause analysed in question 2. clinc150's rows are
much healthier on both counts (3.8% near-duplicates, 94% of gold's diversity), which is consistent
with its synthesis helping at every tier where sms_spam's helped only at tier 1.

---

## 5. Does the hypothesis hold?

> *Surgical data can make up for model capability ceilings for certain smaller model sizes in data
> scarce environments.*

**It holds in a specific and useful form: at the small end, synthetic data buys one to two tiers of
model scale.** Comparing the two arms' per-tier bests directly:

| | arm A (no synth) | arm B (synth) |
|---|---|---|
| sms_spam 360M | 0.6369 | **0.9350** |
| sms_spam 0.8B | 0.9349 | 0.9569 |
| sms_spam 4B Q8_0 | 0.9339 | 0.9528 |
| clinc150 360M | 0.7537 | **0.8047** |
| clinc150 0.8B | 0.7990 | 0.8577 |

On sms_spam, 100 real rows plus synthesis made a **360M model match arm A's 0.8B (0.9350 vs
0.9349) and its 4B Q8_0 (0.9339)** — an 11× parameter saving. On clinc150 it bought about one tier:
360M with synthesis (0.8047) edges arm A's 0.8B (0.7990) but not its 1.7B (0.8229).

**Three qualifications the data forces.**

It does not lift the task's ceiling, only the small model's. clinc150's full-data control still
beats the best scarce arm by 0.0268 (0.9372 vs 0.9104), and within arm B bigger models still win
(0.8047 → 0.9104 up the ladder). The effect is a substitute for scale, not for data.

"Capability ceiling" is the wrong description of what was fixed on sms_spam. Arm A's 360M was not
at a capability ceiling — it was **miscalibrated**, mislabelling 114–779 ham rows as spam in every
one of 15 configurations. What synthesis supplied was enough examples to place a boundary, which is
why the fix shows up as false positives collapsing from 116 to 3 rather than as a uniform lift.

Where the model was already competent, synthesis added nothing measurable. sms_spam tiers 2–5 moved
+0.012 to +0.022, at or below the ±0.015 in-loop noise floor. clinc150 is the counter-case — it
gained at all five tiers, +0.0218 even at the top — and the difference between the two tasks is
label-space size (151 classes versus 2), not model size.

**What still needs testing.** A dose-response arm holding the model at 360M and varying only
synthetic volume (0 / 200 / 500 / 1,000 rows), which neither pair provides and which is the actual
shape of the claim. A repeat with synthesis generating to the **deployment class prior** instead of
50/50, since question 2 shows the current mix actively fights calibration. At least one
non-classification task, since both of these are classifiers. And scoring through
`scripts/report_eval.py` rather than the in-loop eval, which alone would shrink the noise floor
from ±0.015 to ±0.001 and make the tier 2–5 effects measurable instead of ambiguous.
