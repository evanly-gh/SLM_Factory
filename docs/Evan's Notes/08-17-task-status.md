# The teacher few-shot probe, removing reshuffle and synth-fill, and a task-by-task status

*2026-08-17 — measurement + implementation*

Companions: `08-16-extraction-collapse-verdict.md`,
`08-15b-label-space-lockdown.md`.

**The headline is a measurement that changes the synthetic-data argument.** The Qwen3.6 teacher goes
from **0.1131 to 0.7190 span-F1 on BC5CDR NER with five demonstrations — 6.4×.** Its zero-shot score
was never measuring whether it can find biomedical entities; it was measuring whether it guessed our
JSON contract and span conventions. Every conclusion that rested on "the teacher scores 0.0999, so it
cannot generate NER data" was resting on the wrong number.

**A correction up front.** In the previous note I claimed that systematic teacher errors are more
damaging than random ones, citing model collapse. I went looking for a paper that says this and
**found the opposite** — the best controlled study reports structured/real-world noise is *less*
harmful than synthetic uniform noise. §5.3 sets that straight. I was wrong.

---

## 1. Why the model would ever predict only one class

You asked why this happens at all, and it is worth answering before the safeguards.

The metric is **minority-class F1** on a task whose eval set is 502 `route` / 298 `local`. Predicting
`route` for everything gets **63% of rows right**. Under plain accuracy that looks like learning. So
if the training signal is weak or the label prior is wrong, gradient descent finds the majority-class
shortcut because it genuinely is the lowest-loss simple hypothesis available.

Four things were feeding that shortcut, and three are now fixed:

| Cause | Status |
|---|---|
| The training label prior was **wrong**: 1,155 foreign rows at 50/50 against a true 30/70 base rate, so the model was taught the wrong prior outright (B259) | **fixed** — label space closed |
| The length filter deleted ~48% of real rows, thinning the genuine signal it had to learn from (B260) | **fixed** — median anchored on trusted rows |
| Extraction assigned labels by substring accident, so gradient noise was partly measurement noise (B271) | **fixed** — contract enforced |
| `resample` rebuilds re-drew the same pool and re-trained, giving many chances to land in the degenerate basin | **fixed** — removed as a strategy (§2) |

So I did not add a collapse detector, as you asked. The honest position is that collapse was a
*symptom* of a corrupted label prior, and the prior is what was wrong. If it recurs on a clean run
that is real information — it would mean the task itself is degenerate, not that the pipeline is —
and the existing rollback already discards the iteration.

One thing worth stating plainly: **the metric was never the problem.** Minority-class F1 scoring
exactly 0.0 for an all-majority prediction is the metric working. Plain accuracy would have reported
63% and hidden it.

---

## 2. Reshuffle removed as an intervention

You were right that it does nothing. `resample` re-drew rows from the pool the curriculum was already
built from, so it could change *which* gold rows were present but never add information. The plan-yield
numbers say it outright — one traced rebuild: **3,308 resampled rows, 122 of them novel**.

`DATA_REBUILD_STRATEGIES` is now `("acquire", "synthesize")`. What changed:

- The orchestrator can no longer spend a turn, a training run and an eval on a reshuffle.
- The prompt now says so, and tells it to prefer a hyperparameter intervention if neither data
  strategy fits the evidence.
- A plan naming `resample` (an old checkpoint being resumed) is redirected to `synthesize` rather
  than failing the run.
- The `resample_available` gating machinery is inert — it existed only to redirect resample.

**The universal resample-FILL is untouched and must stay.** It is what assembles every curriculum
from the train pool; without it there would be no gold rows at all, especially now that synth-fill is
gone. Its rows are relabelled `resample-fill` in the composition report so it can no longer be
mistaken for a chosen strategy.

---

## 3. Synth-fill removed, and a 500-row floor

Dropped, as you asked. The reasoning, recorded at the call site:

- The size target is a **heuristic** (a function of the zero-shot baseline and parameter count), so
  "reach the target" was never a real requirement.
- **BC5CDR converged at 0.8098 — the project's best result — on a gold-only curriculum that ran
  ~7,100 rows below its 8,929-row target.**
- The SFT literature is consistent that a small clean set beats a large noisy one (LIMA; AlpaGasus,
  where 9k filtered rows beat 52k unfiltered).
- It consumed most of the teacher budget: one traced rebuild spent **749 generations to keep 225
  rows, of which 64 survived quality control**.

In its place, `MIN_CURRICULUM_ROWS = 500` (`SLM_MIN_CURRICULUM_ROWS`). Curate **raises** rather than
warns if the curriculum falls below it — below that point the run would burn GPU hours producing a
number nobody should trust, and the cause is always upstream where it can be fixed. The error names
the two places to look (the loader's train split, the `[qc]` removals).

All seven tasks currently supply 3,250 training rows, so the floor is not close to binding. It is a
tripwire for a loader silently returning nothing, not a target.

The below-target log line no longer blames synth-fill and now reports the shortfall as a percentage:

```
⚠ PROCEEDING BELOW DATA TARGET: 3401 row(s) vs target 6181 (short by 2780, 55% of target).
This is real-data-only by design — synth-fill was removed 2026-08-16. Above the 500-row
viability floor. See the [qc] lines above for what quality control removed and why.
```

---

## 4. Generator A and Generator B — the naming was mine, not the code's

Apologies for the jargon; I invented those labels and they are not in the codebase. Here is the real
picture, and **your understanding of data_rebuild is now correct** — after this session there are
exactly two things a rebuild can do.

### What a `data_rebuild` can do (current state)

| Plan strategy | What it does |
|---|---|
| `acquire` | **Mine new real rows** — the task's own benchmark first, then local bundles, then bounded paid Exa discovery |
| `synthesize` | **Generate new gold rows.** 20% of the budget goes to *surgical* synthesis targeting the top confusion pairs; the rest is spread across the label space |

Plus one thing that always runs: **resample-fill** from the train pool, which is what supplies the
gold rows. That is it. Reshuffle-as-a-choice and synth-fill are both gone.

### The two generators the code actually has

The distinction I was gesturing at is real and it is the single most important fact about synthesis
safety here — it is just about **who produces the label**:

| Function | Task types | Who makes the label | Teacher skill matters? |
|---|---|---|---|
| `_synthesize_new_gold` ("Generator A") | classification, NER | **we do** — copied from a real anchor row | **No** — it never decides anything |
| `_synthesize_new_correct` ("Generator B") | generation, function_call, diff | **the teacher does** — it invents input *and* output | **Yes, critically** |

For your tasks that means:

- `routerbench`, `clinc150`, `proactive_listening` → Generator A. The teacher is handed a label and
  an example and asked for another utterance of that class.
- `xlam_bfcl`, `calendar_json`, `dialogsum_samsum` → Generator B. The teacher invents the answer too.
- `ner_bc5cdr` → **neither.** Generator A needs a `label` field to anchor on and NER rows have none,
  so it returns nothing. That is now explicit and deliberate (B268): the generator emits
  `{text, label}` and never entity spans, so a row it produced would carry the anchor sentence's
  spans against new text — fabricated gold.

---

## 5. The three variables, with what the papers actually say

You asked me to research this properly and quote the papers. I did, and **one of my three claims does
not survive contact with the literature.**

### 5.1 Who produces the label — SUPPORTED

**Self-Instruct** (Wang et al., 2023, [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)) §2.2,
verbatim:

> "However, we found that this approach can generate inputs biased toward one label, especially for
> classification tasks (e.g., for grammar error detection, it usually generates grammatical input).
> Therefore, we additionally propose an Output-first Approach for classification tasks, where we
> first generate the possible class labels, and then condition the input generation on each class
> label."

That is exactly the design `_synthesize_new_gold` uses — label first, input conditioned on it. Note
precisely what they diagnose: **label imbalance**, not label incorrectness.

**The honest caveat.** I went looking for a paper stating that label-conditioning removes label noise
and **could not find one.** The two canonical class-conditional generation papers argue the opposite.
ZeroGen (Ye et al., 2022, [arXiv:2202.07922](https://arxiv.org/abs/2202.07922)):

> "we observe noisy examples in synthetic dataset on difficult tasks such as NLI and QA, this
> situation progressively deteriorates when incorporating more diverse decoding strategy"

And SuperGen (Meng et al., 2022) adds *three* further defences on top of label-conditioning —
generation-probability filtering, label smoothing, temporal ensembling — precisely because the
conditioned output is still unreliable. So: conditioning on the label is the right architecture and it
removes a *bias* failure mode, but it does not make the output trustworthy on its own. Our teacher
label-verification pass is doing necessary work, not belt-and-braces.

### 5.2 An independent verifier — STRONGLY SUPPORTED, and this is the actionable one

**STaR** (Zelikman et al., 2022, [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)) states the
assumption outright:

> "We assume that rationales that lead to correct answers are of better quality than those that lead
> to incorrect answers. Therefore, we filter the generated rationales to include only the ones which
> result in the correct answer"

And, directly relevant to us:

> "We propose a bootstrapping mechanism to iteratively generate a rationale dataset from a few
> initial examples with rationales—without needing to check new rationales' correctness."

**Bansal et al.** (ICLR 2025, [arXiv:2408.16737](https://arxiv.org/abs/2408.16737)) is the result that
most directly attacks the "teacher must be accurate" intuition. The exact numbers:

> "Our human evaluations suggest that the FPR for the WC-generated solutions is 7% and 2% higher than
> SE-generated solutions on the MATH and GSM-8K, respectively."

> "The Gemma-7B finetuned with the synthetic data from WC consistently outperforms the one finetuned
> on data from SC with a relative gain of 6% and 5.8% at the low and high sampling budgets"

A generator with a **higher false-positive rate produced better students**, because coverage and
diversity dominated. **Two caveats I will not paper over:** their filter is gold-answer matching, so
that 7% is noise *surviving* a verifier — this is evidence for "weak generator **plus** verifier", not
weak generator alone. And their "weak" model is weak by parameter count (9B vs 27B), not weak in the
sense of scoring 0.11 on the task. The analogy to our situation is suggestive, not established.

### 5.3 Random vs systematic errors — I WAS WRONG, and the evidence runs the other way

I claimed systematic teacher errors get learned and amplified while random noise averages out. **No
paper I could find supports the asymmetry, and the best controlled study reports the reverse.**

Jiang et al. (ICML 2020, [arXiv:1911.09781](https://arxiv.org/abs/1911.09781)), *Beyond Synthetic
Noise: Deep Learning on Controlled Noisy Labels* — a head-to-head of synthetic uniform noise against
real-world structured noise:

> "Our studies reveal several new findings: (1) DNNs generalize much better on web label noise"

> "The real-world label noise from the web appears to be less harmful, yet it is more difficult for
> our current robust learning methods to tackle."

And Rolnick et al. (2017, [arXiv:1705.10694](https://arxiv.org/abs/1705.10694)) tested confusion-biased
noise specifically:

> "Such behavior holds across multiple patterns of label noise, even when erroneous labels are biased
> towards confusing classes."

The two papers I cited for the amplification claim do not say what I used them for either:

- **Shumailov et al.** (Nature 2024) is about **recursive** training — generation *n* trains on
  generation *n−1*'s output, repeatedly. We do a single distillation step onto a separate student,
  which is not the collapse setting. Worse for my claim, they name *statistical* (sampling) error as
  the **primary** driver, which is close to the reverse of "random noise averages out."
- **Gudibande et al.** ([arXiv:2305.15717](https://arxiv.org/abs/2305.15717)) is about **breadth**,
  and it arguably supports our approach rather than warning against it:

  > "training on 100k ChatGPT outputs from broad-coverage user inputs provides no benefits to Natural
  > Questions accuracy … but training exclusively on ChatGPT responses for Natural-Questions-like
  > queries drastically improves task accuracy."

  We do narrow, single-task distillation. That is the regime they say works.

**So the corrected rule has two variables, not three:** who makes the label, and whether a verifier
exists. Drop the third. If we want to claim teacher errors are correlated with the student's inductive
biases in a way i.i.d. noise is not, that is a hypothesis about our setting, not a citation.

### 5.4 Your intuition about low initial accuracy — now measured, and it was right

You said you expected format-bound and OOD tasks to give a low initial score, and asked whether that
should govern trust in the teacher's data. **The measurement says it should not.**

| shots | span-F1 | empty/unparseable |
|---|---|---|
| 0 | **0.1131** | 131/200 |
| 1 | 0.4910 | 101/200 |
| 3 | 0.6736 | 90/200 |
| **5** | **0.7190** | 84/200 |

**6.4×, from five demonstrations.** The raw outputs show exactly what changed. Zero-shot:

```
```json
[ { "text": "CYP", "type": "CHEMICAL" }, { "text": "P2X3", "type": "GENE_OR_PROTEIN" }, ...
```

Wrong casing (`CHEMICAL`), a class that does not exist in BC5CDR (`GENE_OR_PROTEIN`), wrapped in a
markdown fence. One demonstration fixed all three:

```
[{"text": "CYP", "type": "Chemical"}, {"text": "P2X3", "type": "Chemical"}, ...
```

The literature has a precise account of why. **Min et al.** (EMNLP 2022,
[arXiv:2202.12837](https://arxiv.org/abs/2202.12837)):

> "we find that other aspects of the demonstrations are the key drivers of end task performance,
> including the fact that they provide a few examples of (1) the label space, (2) the distribution of
> the input text, and (3) the overall format of the sequence."

That is our three failure modes exactly: label space (`GENE_OR_PROTEIN`), input distribution, and
format (the fence). And **Holtzman et al.** (EMNLP 2021,
[arXiv:2104.08315](https://arxiv.org/abs/2104.08315)) states the framing directly:

> "we argue that current work underestimates the zero-shot capabilities of these models on
> classification tasks."

One honest counterweight: **Sclar et al.** (ICLR 2024,
[arXiv:2310.11324](https://arxiv.org/abs/2310.11324)) report format sensitivity of "up to 76 accuracy
points" and say it "is not eliminated by adding few-shot examples." So demonstrations are not a
general cure for format sensitivity — but on this task they recovered most of the gap.

**Where that leaves the 0.7190.** It is below the fine-tuned 0.6B student's **0.8098**. So on BC5CDR
the fine-tuned small model still beats the few-shot 35B teacher — the project's central claim
survives, and is in fact *better supported* now, because the comparison is no longer against a
teacher crippled by a format artifact.

### 5.5 The one experiment that would settle it — worth running

Min et al.'s method suggests a decisive ablation I have **not** run: **corrupt the labels in the five
demonstrations while holding the format fixed.** Randomize the entity spans but keep the JSON shape,
casing and class vocabulary.

- If the teacher stays near 0.7190 → the gain was format/label-space, its task knowledge was never
  the bottleneck, and its zero-shot score is irrelevant to synthesis fitness.
- If it falls back toward 0.1131 → the demonstrations were carrying task knowledge, and the low
  zero-shot score is real and should gate synthesis.

That is one Slurm job on the probe harness already written. **This is the measurement I would do
before deciding the synthesis policy**, and it is cheaper than any ablation involving training.

---

## 6. Proactive listening — the TLDR

**The task.** An earbud assistant listens to your conversation. At every pause it makes one binary
decision: whisper a 1–3 word hint now (`interrupt`), or stay quiet (`wait`). That is it. LlamaPIE
splits this across two models — a small one that decides *when*, a big one that decides *what to
say* — and we implement the small one, because that is the part that runs continuously on the phone.

**The data is real and it is now in the repo.** It is not on HuggingFace; it is a 477 MB Google Drive
tarball linked from the paper's GitHub README. I downloaded it, verified it, and vendored it at
`data/local/proactive_listening/` with checksums — **7,065 training dialogues, 878 held-out**, each
annotated by Claude with where the assistant should speak.

**What one row looks like.** The transcript up to a pause, plus what the assistant knows about the
user, and the model answers `interrupt` or `wait`:

```
… Can you share more about the themes in your "Morning Light" series? What inspired you to
focus on daily routines? |SILENCE > User: Thank you! Well, the series is r
```
→ `interrupt` (she is about to need a detail from her own profile)

**Why it is worth running.** Three reasons:

1. **It is a fair out-of-distribution task, unlike RouterBench.** RouterBench's label describes a
   *different model's* behaviour, which you cannot read off the prompt. Here the label is a property
   of *this* conversation — hesitation, a trailing clause, an about-to-be-needed name. The signal is
   actually in the text.
2. **It has a published number to beat.** LlamaPIE's fine-tuned Llama-3.2-1B gets **F1 ≈ 0.74**, and
   our metric is the same quantity. No other task in the suite has an external reference point.
3. **It is a real product.** "Should I interrupt right now" is the kind of always-on, low-latency,
   privacy-sensitive decision that belongs on a phone rather than in the cloud.

**Two bugs I caught before it ever ran.** Half the training corpus (`synthetic0`) attaches whispers to
*emotion* markers (`|ANGRY >`) rather than silences, so matching silences alone found **zero
positives in 7,065 dialogues**. And because the file is grouped by sub-corpus, truncating to 3,250 rows
drew entirely from that one corpus — giving 4.2% positives for training against 33.4% for eval. Both
fixed, both pinned by tests.

**Status: ready, not submitted.** Slurm scripts exist for both accounts. Pre-flight all green.

---

> **Corrected 2026-08-18.** Two errors in the table below. **DialogSum was a VALID, complete run** —
> 4 tiers, 73 iterations, real gains at tiers 0-2 (+0.122/+0.135/+0.055); the +0.0000 I quoted was its
> tier-3 row only, where the baseline was already 0.7157. And **GSM8K is missing entirely**: it
> CONVERGED at 0.8263 vs a 0.820 goal on Qwen3.5-4B@Q4_K_M via the autonomous path. `xlam_bfcl` was
> stopped by hand, not by a defect. Corrected table in
> `08-18-exact-verifiers-fewshot.md`.

## 7. Status of every task

Seven tasks (`coedit` and `medqa` removed at your request). "Preflight" is
`scripts/preflight_tasks.py`, which checks load → eval → train → hygiene without a GPU.

| Task | Category | Ran cleanly? | Best result | Preflight | Ready for a real run? |
|---|---|---|---|---|---|
| `ner_bc5cdr` | format-bound | **Yes — converged** | **0.8098** vs 0.800 goal, 0.6B, 81 min. Baseline 0.0000 → first FT 0.7701 | **PASS** | **Yes** — the reference result |
| `clinc150` | in-distribution | **Yes** | **0.8952**, teacher 0.8919, FT gain **+0.003** | **PASS** (151 classes, vocab matches) | **Yes** — but it is a control; there is no headroom |
| `xlam_bfcl` | format-bound | Yes, plateaued | 0.6900 at iter 3, then 9 iterations oscillating 0.46–0.69. Never left tier 0 in 3h51m | **PASS** | **Yes** — but the 0.870 goal is the teacher's own score, so a 0.6B must match a 35B |
| `routerbench` | out-of-distribution | Ran, **numbers invalid** | 0.7584 (22h run), 0.7467 (fresh) — **34% of the curriculum was contaminated** | **PASS** | **Yes, and worth rerunning clean** — all three upstream causes fixed |
| `dialogsum_samsum` | in-distribution | Yes | **0.7157 baseline = 0.7157 best FT, Δ+0.0000** over 15 iterations at tier 3 | **FAIL** — see below | **No** — fix the overlap first |
| `calendar_json` | format-bound | **No — eval set unfair** | 0.0000; 82% of gold needs an unguessable year rollforward | **PASS** (mechanically) | **No** — rebuild the gold first |
| `proactive_listening` | out-of-distribution | Never run | — (reference: LlamaPIE 1B ≈ 0.74) | **PASS** | **Yes** |

### 7.2 Preflight results in full

```
PASS  calendar_json        train= 3250  eval= 300  gold=   1.0  degen=   0.0
PASS  clinc150             train= 3250  eval= 300  gold=   1.0  degen=0.0001
FAIL  dialogsum_samsum     train= 3250  eval= 300  gold= judge  degen= judge
PASS  ner_bc5cdr           train= 3250  eval= 300  gold=   1.0  degen=   0.0
PASS  proactive_listening  train= 3250  eval= 300  gold=   1.0  degen=   0.0
PASS  routerbench          train= 3250  eval= 300  gold=   1.0  degen=   0.0
PASS  xlam_bfcl            train= 3250  eval= 300  gold=   1.0  degen=   0.0
```

Every task: gold scores 1.0, a degenerate answer scores ~0, the training prompt is byte-identical to
the eval prompt, and there is no train/eval overlap (except the one DialogSum row). `dialogsum_samsum`
is scored by an LLM judge so its scoring half cannot be checked offline — that is reported as skipped
rather than passed.

The preflight also **caught one of my own bugs**: it initially reported classification train/serve skew
for `routerbench` and `proactive_listening`. That was an artifact of comparing against a one-row eval
set (which has one label, so the enumerated label list differs). The fix turned it into a *better*
check — it now compares the label **vocabularies** of train and eval, which is the real skew risk for
a 151-class task like CLINC150 where a rare class could be missing from one side.

---

## 8. The open items you asked me to explain

### 8.1 Generation-family synthesis (was open item 2)

**What it is.** For `dialogsum_samsum`, `xlam_bfcl` and `calendar_json`, synthesis asks the teacher to
invent **both** the input and the correct output. Nothing constrains correctness — unlike
classification, there is no anchor label to copy.

**Why it was alarming.** It was running with **no check whatsoever**: `curate._verifier_for` returns
`None` for every task type, so the `if verify_fn is not None` branch had never executed. Every batch
logged `450/450 kept`. On `calendar_json` the teacher scores 0.2176, against a curriculum target of
8,929 rows from 3,250 real ones — roughly 5,700 machine-invented targets from a model that gets the
task right 22% of the time.

**Where it stands now.** Two things changed. `verify_generated_answers` gives it the model-based check
you asked for ("given this answer and this request, is it correct in the context of this task"), and
**synth-fill is gone**, which removes the path that was generating thousands of these rows. It now
only runs when the orchestrator explicitly picks `synthesize`, bounded to 100–500 rows.

**My recommendation, unchanged:** default it off, and turn it back on per-task once an *exact*
verifier exists (§8.2). A model-based check on a task the model is bad at is weak.

### 8.2 Exact verifiers (was open item 4) — the best available work

This is the highest-value thing left, and §5.2 is why: STaR and Bansal both show that a mediocre
generator behind a **real** verifier produces good data. We have two tasks where the verifier is free
and exact, and we are not using it.

**`xlam_bfcl`** — a generated row can be checked mechanically: does `answer` parse as JSON, is it a
list of `{name, arguments}`, does every `name` appear in the row's declared `tools`, and does every
argument key exist in that tool's schema? All four are pure computation, no model involved. The
codebase *already has* this logic — it is how the scorer grades — it is simply not wired into
`_verifier_for`.

**`calendar_json`** — stronger still. Parse the ISO-8601 datetimes; check the end is after the start,
that the duration is 60 minutes unless stated, and that the date is consistent with the reference
datetime in the prompt. That last check would have **caught the unguessable-year bug** (§8.4) as a
data defect rather than a mystery 0.0000.

Estimated effort: small — a few dozen lines each, reusing scorer internals. Effect: Generator B stops
being a wrong-label factory on exactly the two tasks where it was worst.

### 8.3 `calendar_json` eval fetched live from GitHub (was open item 5)

Its eval half is downloaded from
`raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue` **at load time, and
never cached**. Two consequences: one upstream commit silently changes your eval set, so past scores
stop being comparable and are not reproducible; and the run fails outright without network access.

The fix is the treatment `bc5cdr` gets and `proactive_listening` now gets: download once, vendor under
`data/local/` with a manifest and sha256 checksums, read locally thereafter. This is mechanical — the
proactive-listening bundle is a working template.

### 8.4 The unguessable year (was open item 6)

82% of `calendar_json`'s gold answers require rolling the date forward to 2027. **You were right that
a student should learn a convention through fine-tuning rather than being told it in the harness** —
and I stand by that. The problem is narrower:

- SGD's dialogues are almost all set in **March**. `reference_for()` scatters the reference date
  uniformly across 2026. So for most rows the reference falls *after* March, a "roll forward if the
  date has passed" rule fires, and the gold lands in 2027.
- The model sees "on 2nd of March" with a reference of 2026-08-19 and answers **2026**-03-02 — which
  is the more natural reading and is marked wrong.
- A student *can* learn this. But the **ceiling** is set by a teacher that scores 0.2176 for the same
  reason, and 82% of the score then rides on one arbitrary rule rather than on the task.

So: don't tell the model. **Make the gold unambiguous** — pin the reference date near the corpus's own
March timeframe so no rollforward is needed. Separately, `convert_sgd_rows` builds the request as
`f"Schedule {summary} on {date}"`, so a row titled `Food` reads *"Schedule Food on March 1st"* and the
model extracts `summary="Schedule Food"`. Plain data bug.

### 8.5 Empty difficulty buckets (was open item 8) — implemented

You asked me to implement this and log it. Done.

The buckets come from a two-model capability gradient: `easy` = both the smallest and largest model
pass zero-shot, `medium` = only the largest passes, `hard` = neither. On a **format-bound** task
neither model can produce the output contract zero-shot, so nothing separates them and `medium` is
empty for the whole run. BC5CDR reported `medium=None` eight times while the orchestrator was handed
`difficulty_buckets.medium = 0.25` — a quarter of the curriculum budget aimed at rows that do not
exist.

Three changes:

1. `_normalized_difficulty` takes the set of **populated** buckets, zeroes the empty ones, and
   redistributes their weight. If all weight landed on empty buckets it spreads evenly over the
   populated ones instead of raising.
2. `curate` derives that set from `state["eval_difficulty"]` and **logs it**:
   `Difficulty buckets present: easy=182, hard=618 — EMPTY: medium. Their curriculum weight is
   redistributed to the populated buckets, because weighting a bucket with no eval rows spends budget
   on nothing.`
3. The per-difficulty line now reads `medium=n/a(n=0, no eval row in this bucket)` instead of a bare
   `medium=n/a`, which read like a measurement failure.

**One thing I did not change, and it is the deeper issue.** BC5CDR's `easy` bucket is essentially the
**182 rows whose gold entity list is empty** — where emitting `[]` is correct and both models manage
it. It measures *abstention*, not extraction, which is consistent with 39.5% of training rows having
no entities and with the fine-tuned model scoring easy=0.984 / hard=0.605. The gradient is measured
zero-shot, which is exactly the regime where format-bound tasks carry no signal. A length-tercile
fallback would be more informative there. Left open deliberately — it changes what "difficulty" means
and is worth deciding rather than sliding into.

### 8.6 Constrained label decoding (was open item 9)

**What it is.** Instead of letting the model generate free text and then regex-parsing a label out of
it, compute the model's likelihood for each candidate label and take the highest. The model never
emits anything but a label.

**Why it matters.** It makes extraction failure **structurally impossible**. Every failure mode in
B271 — the model answering the row's embedded question, markdown fences, `<think>` wrappers, prose
that happens to contain "route" — simply cannot occur, because the model is not asked to produce a
string at all. It is what `lm-eval-harness` does for multiple choice, and it is the standard approach.

**Why I have not done it.** It needs a log-probability path in **both** inference backends — Unsloth
for BF16/LoRA and llama-cpp-python for the GGUF quantized path — and the two expose scoring
differently. It is a real piece of work, not a patch, and it changes every classification baseline
ever measured (upward, and honestly). It is the single biggest remaining eval improvement.

### 8.7 Eval-set shortfall now logged (your request)

You said a short eval set is fine as long as it is logged. Done:

```
⚠ EVAL SET IS SHORT: 478/800 rows (short by 322, 60% of target). Accepted — the loader's
held-out split is the limit — but scores carry more variance than a full-size eval set and
are not directly comparable to tasks that reached target.
```

Both facts a reader needs: it is accepted, and it changes how the number should be read.

---

## 9. What changed in code

| # | Change | Where |
|---|---|---|
| 1 | `resample` removed as a selectable strategy; old plans redirect; prompt rewritten | `agent/data_rebuild.py`, `agent/nodes/iterate.py` |
| 2 | Synth-fill removed entirely (97-line function deleted) | `agent/nodes/curate.py` |
| 3 | `MIN_CURRICULUM_ROWS = 500` viability floor, raises with an actionable message | `agent/nodes/curate.py` |
| 4 | Empty difficulty buckets: weight redistributed, `n=0` reported, logged | `agent/data_rebuild.py`, `agent/nodes/curate.py`, `agent/nodes/evaluate.py` |
| 5 | Eval-set shortfall logged with variance caveat | `agent/nodes/cold_start/eval_setup.py` |
| 6 | `single_model` strategy — naive baseline, no escalation, no regression | `agent/nodes/cold_start/model_selection/single_model.py`, `config/config.py`, `agent/nodes/iterate.py` |
| 7 | Teacher few-shot probe (script + Slurm) | `scripts/probe_teacher_fewshot.py`, `tests/pipeline/run_teacher_fewshot_probe.slurm` |
| 8 | Per-task preflight harness | `scripts/preflight_tasks.py` |

**Tests: 203 node tests pass; new suites** `test_single_model_strategy.py` (9),
`test_curate_synth_fill.py` rewritten (6, now asserting synth-fill is *gone*).

Bugs my own tooling caught this session, all silent:

- `set -u` in the probe Slurm script broke lmod's module init, so `module load cuda` failed and the
  job died 11 minutes into weight loading with a misleading `Permission denied: 'nvcc'`. Fixed, plus
  an explicit early `nvcc` check so this fails in seconds instead of minutes.
- `STAGNATION_WINDOW` is 15, not 5 — my first `single_model` test wasn't triggering stagnation at all.
- A module-reload hack in that test leaked `MAX_EVALS_BEFORE_ESCALATION=3` into every later test in
  the file. Replaced with a direct constant patch.
- `MIN_CURRICULUM_ROWS` is read at import time, so `monkeypatch.setenv` silently did nothing in tests;
  needed `setattr`.

---

## 10. Open items after this session

| # | Item | Status |
|---|---|---|
| 1 | **Corrupt-label few-shot ablation** — separates format from knowledge definitively (§5.5) | **recommended next; one Slurm job** |
| 2 | Exact verifiers for `xlam_bfcl` and `calendar_json` (§8.2) | **open — best available work** |
| 3 | Generation-family synthesis default (§8.1) | **open — needs your decision** |
| 4 | `calendar_json` gold: unguessable year + "Schedule" leak (§8.4) | **open — blocks rerun** |
| 5 | `calendar_json` eval vendored instead of live-fetched (§8.3) | **open** |
| 6 | `dialogsum_samsum` train/eval overlap + contradictory summaries (§7.1) | **open — blocks rerun** |
| 7 | Constrained label decoding (§8.6) | **open — biggest eval win** |
| 8 | NER `easy` bucket measures abstention, not extraction (§8.5) | **open, design-level** |
| 9 | RouterBench relabelling against our own pool | **closed by decision** — not doing it |
| 10 | Mode-collapse detector | **closed by decision** — safeguards address the cause (§1) |

---

## 11. Questions

1. **Run the corrupt-label ablation?** It is the cheapest way to settle whether the teacher's low
   zero-shot score should gate synthesis at all, and the harness already exists.
2. **Implement the exact verifiers?** Small, free accuracy, and it is what the literature says
   actually makes a weak generator usable.
3. **Generation-family synthesis: default off?** It is now verified but the check is weak on the tasks
   where it matters most.
4. **`proactive_listening`: submit now?** Preflight is green. I would run the zero-shot reference
   measurement first — one command — since that is exactly the check `calendar_json` never got.
5. **`clinc150` recategorised as in-distribution** — done as you instructed; flagging that it means the
   suite now has two in-distribution tasks and both show near-zero fine-tuning gain, which is itself
   a result worth stating in the paper.
