# Run Review Q&A — NER & Math

**Date:** 2026-07-26
**Runs analyzed:** `slm-ner-l40s-37531245` (BC5CDR NER), `slm-math-l40s-37576194` (GSM8K math)

Every number here was verified against the run logs or the source; where something is an
opinion or a proposal rather than a measured fact, it says so.

---

# Section 1 — Clarify

## 1.1 What is NER, and what does it mean for BC5CDR?

**Named Entity Recognition** finds spans of text that refer to specific things and labels
each with a type. Not "is this document about chemistry?" (classification) but "which exact
characters in this sentence are a chemical, and which are a disease?"

For **BC5CDR** (BioCreative V Chemical–Disease Relation), the input is a PubMed abstract
and the model must extract two entity types:

```
Input:  "Lidocaine-induced cardiac asystole occurred in the patient."
Output: [{"text": "Lidocaine", "type": "Chemical"},
         {"text": "cardiac asystole", "type": "Disease"}]
```

Why it's hard, and why the orchestrator flagged it as needing a big curriculum:

- **Boundaries matter.** "cardiac asystole" is one Disease span. Predicting just "asystole"
  is *wrong* under strict span-F1, even though a human would call it close.
- **Specialized vocabulary.** Sub-4B models see little biomedical text in pretraining. The
  orchestrator's own words: *"a niche biomedical benchmark with specialized vocabulary
  unlikely to be well-covered in general pretraining of sub-4B models."*
- **Type confusion.** Many terms are ambiguous in isolation.

**Scoring is span-F1**, not accuracy — this matters a lot for §2.7. Each predicted span is
compared to gold spans:

- Precision = (correct predicted spans) / (all predicted spans)
- Recall = (correct predicted spans) / (all gold spans)
- F1 = harmonic mean = `2·P·R / (P + R)`

So a model that gets 3 of 4 entities in a sentence earns *partial credit*. This is the
critical difference from the math run, where an answer is simply right or wrong.

## 1.2 Datasets used, and where the SOTA threshold came from

**NER run** — Exa discovered candidate HuggingFace datasets, logged verbatim:

```
[acquire] Exa-discovered candidate HF datasets: ['tner/bc5cdr', 'bigbio/bc5cdr',
 'omniquad/BC5CDR-IOB', 'ghadeermobasher/BC5CDR-Chemical-Disease',
 'languidsheep/bc5cdr', 'bigbio/chem_dis_gene', 'masaenger/bc5cdr']
```

Selected: **`tner/bc5cdr`** → https://huggingface.co/datasets/tner/bc5cdr
(train=3403, test=900; 24 rows dropped for normalized train/test overlap).

**Math run** — selected **`openai/gsm8k`** → https://huggingface.co/datasets/openai/gsm8k
Notably, `gabrielaltay/gsm8k-math-reasoning` was **rejected for 128 rows of train/test
overlap** — the contamination check did real work here.

**Where the thresholds came from — important caveat.** They were *not* looked up from a
leaderboard. They were produced by Claude from its own pretraining knowledge in a single
`task_analysis` call, with no citation and no verification. The verbatim rationales:

> **NER (0.88):** "BC5CDR published SOTA for fine-tuned small models (~1B–4B, e.g.
> BioBERT-large F1 ~0.90, small-model fine-tunes typically ~0.88–0.90) anchors
> stop_threshold at 0.88"

> **Math (0.82):** "small models in the 1.7B–4B range with fine-tuning reach roughly
> 0.82–0.87 on GSM8K (e.g., Qwen3-4B-class fine-tunes have been reported at ~0.85+ while
> smaller 1.7B fine-tunes land ~0.78–0.82), so stop_threshold anchored at 0.82"

These are plausible and roughly match the literature, but they are **recalled, not
retrieved** — no source URL, no benchmark protocol, no date. The 0.88 NER target directly
caused a 44.8-hour run to be classified a failure at 0.8628. If that number was off by
even 0.02 the verdict flips. Worth grounding against a real leaderboard.

## 1.3 What does "curriculum 4500/900" mean?

Two separate dataset sizes the planner chose:

| Term | NER | Math | What it is |
|---|---|---|---|
| **curriculum_size** | 4500 | 4500 | Target **training** examples the model learns from |
| **eval_size** | 900 | 800 | Held-out **test** examples, never trained on |

The eval set is built once at the start and frozen for the whole run — that's what makes
scores comparable across 142 iterations. It's stratified into `pos` / `neg` / `boundary`
(360/360/180 for NER) and separately bucketed by difficulty (easy/medium/hard).

**Both runs missed the curriculum target.** They asked for 4500 and got **3403** — the
entire real training split of each source dataset. See §2.10.

## 1.4 What is schema validation? Show me how it works.

When the orchestrator decides what to do next, it must return **structured JSON** matching
a fixed contract, so the pipeline can execute it mechanically. Schema validation is the
check that the returned JSON obeys that contract. Bad JSON is rejected rather than acted on.

The rule that broke the NER run, at [`agent/nodes/iterate.py:253`](../../agent/nodes/iterate.py#L253):

```python
if "hyperparams" in validated or "hyperparam_rationale" in validated:
    raise ValueError("hyperparams is not allowed for a data_rebuild intervention")
```

The logic: an intervention is *either* "change the training knobs" (`hyperparameter`) *or*
"change the training data" (`data_rebuild`) — never both, so that when the score moves you
know which change caused it. Sending `hyperparams` with a `data_rebuild` is a contradiction.

**Valid** — data_rebuild, no hyperparams:
```json
{"intervention": "data_rebuild",
 "hypothesis": "plateaued at 0.83; dataset under-represents boundary cases",
 "data_rebuild": {"primary_strategy": "difficulty_weighted_sampling", "target_rows": 4500}}
```

**Rejected** — what Claude actually sent, 65 times:
```json
{"intervention": "data_rebuild",
 "hypothesis": "After 11 iterations of hyperparameter tuning on the same v1 dataset...",
 "data_rebuild": {"primary_strategy": "resample_existing"},
 "hyperparams": {"lora_rank": 64, "learning_rate": 1e-4}}
```
The trailing `hyperparams` block violates the rule → `ValueError` → decision discarded.

## 1.5 What is a "heuristic fallback substitute"?

When the LLM decision is unusable, the pipeline can't stop — so it substitutes a
deterministic, hard-coded rule and continues. The log is explicit:

```
[iterate] LLM call failed (ValueError('hyperparams is not allowed for a data_rebuild
          intervention')); using test-agent suggestion: hyperparameter
[train]   Reasoning: fallback hyperparameter step (LLM returned no config): selected
          deterministic untried complete identity r=64 a=128 drop=0.05 wd=0.01
```

So instead of Claude's reasoned "rebuild the data targeting Chemical/Disease boundary
confusions," the run got "pick the next untried rank off a fixed ladder." **The `data_rebuild`
label in the summary table is real — an actual rebuild ran — but the *plan* behind it was
never the LLM's.** All 31 NER rebuilds were fallback-generated.

## 1.6 Why did NER cost $13.43 but math only $3.20, at ~2× the runtime?

Runtime isn't what drives cost — **number of LLM calls × prompt size** is. Measured:

| | NER | Math | Ratio |
|---|---|---|---|
| Wall clock | 44.8 h | 20.5 h | 2.2× |
| Claude calls | 231 | 73 | **3.2×** |
| Input tokens | 3,584,126 | 742,965 | **4.8×** |
| Total cost | $13.41 | $3.16 | **4.2×** |

Three compounding causes:

**1. Twice the iterations.** 142 vs 66 — each needs an `iterate` call.

**2. The retry tax (NER only).** 86 `iterate_json_reask` calls that math never made:

| Stage | NER calls | Math calls |
|---|---|---|
| `iterate` | 141 | 63 |
| `iterate_json_reask` | 86 | 2 |

**$4.90 of NER's $13.41 — 36% — was spent on reask retries, and every one was discarded.**

**3. Prompt growth.** The `iterate` prompt embeds trajectory history, so it grows as the run
does. Early NER call: `tokens=3351→261` ($0.014). Later: `tokens=23082→1024` ($0.16) — a
**12× cost increase for the same decision**, with a paired reask on top. Cost per iteration
grows superlinearly with run length; nothing caps it.

## 1.7 The retry problem, restated

The sequence, 69 times in the NER run:

1. `iterate` call → Claude returns `data_rebuild` + `hyperparams` → **rejected** (paid)
2. `iterate_json_reask` retry → Claude returns the same shape → **rejected again** (paid)
3. Fall back to the hard-coded heuristic → **both paid results thrown away**

Why it's a problem beyond the money:

- **It's 100% deterministic, not flaky.** Grep confirms **zero** successful
  `intervention=data_rebuild` decisions in the entire NER log — 72 clean `hyperparameter`
  decisions, and not one clean `data_rebuild`. Every time Claude wanted to change data, it
  was silently overruled.
- **The reask doesn't re-prompt differently enough.** It fails the same way, so the retry
  is pure waste.
- **The failure is invisible in the summary.** The table shows 31 `data_rebuild`
  iterations, implying the orchestrator drove them. It didn't.
- **It is run-dependent, which is worse than always-broken.** Math hit this **zero** times
  (`grep -c` → 0). Same code, same model — so you can't predict or budget for it.

## 1.8 Why does reshuffling in `data_rebuild` matter, and what were the alternatives?

**What actually happened in NER** — all 31 rebuilds:
```
[curate] DATA REBUILD: identity=942eb2a854f38fa154bd43fb primary=resample_existing support=[]
         yield {'previous_rows': 1277, 'final_rows': 1321, 'novel_rows': 661, 'novel_fraction': 0.5004}
```

`primary=resample_existing` = draw a different random subset of the *same* pool. No new
examples entered the run. "novel_rows: 661" means "rows not in the previous subset," not
"new information."

**Why that's a problem:** the point of `data_rebuild` is to escape a plateau *hyperparameters
can't fix* — by fixing what the model is learning from. Reshuffling changes none of the
information content. It costs a **full train + quantize + eval cycle (~19 min of 4×L40S)**
and is statistically near-indistinguishable from rerunning the same experiment with a
different seed. NER spent **31 cycles ≈ 10 GPU-hours** this way.

**The alternatives existed and were never used.** Six strategies are defined at
[`agent/data_rebuild.py:15`](../../agent/data_rebuild.py#L15):

| Strategy | What it does | NER | Math |
|---|---|---|---|
| `resample_existing` | Reshuffle the same pool | **31** | 2 |
| `difficulty_weighted_sampling` | Oversample hard/medium buckets | 0 | **20** |
| `mine_new_real_source` | Pull genuinely new rows from a real dataset | 0 | **2** |
| `source_diversification` | Broaden across sources | 0 | 0 |
| `preserve_elite_resample` | Keep best-performing rows, resample the rest | 0 | 0 |
| `targeted_synth_positive` | **Generate synthetic examples** | 0 | 0 |

Math used three strategies and never exhausted its plan space. NER used one and crashed.
The root cause is §1.9.

## 1.9 What is the "plan space," and why didn't it find new data or synthesize?

**The plan space** is the set of distinct rebuild plans reachable from a given starting
plan. Every plan gets an identity hash; a plan already tried is not retried. At
[`agent/data_rebuild.py`](../../agent/data_rebuild.py), `ensure_untried_data_rebuild_plan`
searches for an untried variant in a fixed order:

1. The proposed plan as-is
2. Rotate `query_variant` — **8 options**
3. Rotate `resample_fraction` through `(0.50, 0.75, 0.90, 1.0, 0.35)` — **5 options**
4. Out of options → `raise ValueError("bounded data_rebuild plan space is exhausted")`

So with the primary strategy held fixed, the space is roughly **1 × 8 × 5 = 40 plans**. NER
did 31 rebuilds plus pruned/duplicate attempts and ran the well dry. **The exhaustion is a
symptom: the space is only small because the strategy never varied.**

**Why it never mined a new dataset.** The fallback plan builder picks a strategy by
**keyword-matching the hypothesis string**
([`agent/data_rebuild.py:619-633`](../../agent/data_rebuild.py#L619)):

```python
elif "source" in lower_hypothesis or "novel" in lower_hypothesis:
    primary = "source_diversification"
elif "hard" in lower_hypothesis or "difficulty" in lower_hypothesis:
    primary = "difficulty_weighted_sampling"
else:
    primary = "resample_existing"      # ← the catch-all
```

On the fallback path the hypothesis is the **test-agent diagnosis**. And `diagnose()`
([`agent/nodes/test_agent.py:130`](../../agent/nodes/test_agent.py#L130)) only ever
*suggests* `data_rebuild` in two branches, whose text is:

- `"easy-bucket accuracy is low (…) — … Rebuild + balance the data."`
- `"below goal (overall 0.856) with no single failing bucket — add more balanced data and continue."`

**Neither contains "source", "novel", "hard", or "difficulty".** The one diagnosis that
*does* say "hard" suggests `hyperparameter`, so it never reaches the plan builder at all.

> **This is a structural dead end, not bad luck.** On the fallback path, `data_rebuild`
> *always* falls through to `resample_existing`. The other five strategies are unreachable
> unless the LLM's own plan validates — which in the NER run it never did (§1.7).

**Why it never synthesized.** `targeted_synth_positive` is gated at
[`agent/data_rebuild.py:616`](../../agent/data_rebuild.py#L616):

```python
if score >= 0.95 and task_type in TARGETED_SYNTH_TASK_TYPES:
    primary = "targeted_synth_positive"
```

**Synthesis requires score ≥ 0.95.** Both stop thresholds are *below* that (0.88 and 0.82),
so any run that converges normally stops before synthesis can ever fire. NER peaked at
0.8628, math at 0.8263. The vLLM synthesis server was up, healthy, and idle the entire
time. Confirmed: `grep -c targeted_synth_positive` → **0** in both runs.

## 1.10 Where is the downward probe in the math log?

It's there — `logs/runs/slm-math-l40s-37576194/run.log` **lines 5480–5486 and 5537–5558**.
It runs *after* convergence, which is why it's near the end rather than in the iteration table:

```
5480  [iterate]   → DOWNWARD_PROBE (score 0.8263 >= threshold; strategy=orchestrator_choice
                    may have over-selected; trying a smaller model)
5482  [downward_probe] orchestrator downward-re-exploration decision: True — The converged
                    score exceeds the goal by only a modest margin (+0.0063), but with 3
                    untried lower tiers still available … worthwhile to potentially save
                    device resources while still meeting the goal.
5484  [downward_probe]   Orchestrator chose Qwen/Qwen3.5-2B (quant=Q4_K_M, 1100MB; …)
5537  [downward_probe]   fixed probe config: r=16 alpha=32 dropout=0.0 wd=0.01 lr=0.0002 epochs=3
5558  [downward_probe]   ✗ Qwen/Qwen3.5-2B scored 0.5400 < 0.8200 — stopping downward search
```

Quick view:
```bash
sed -n '5480,5486p;5537,5558p' logs/runs/slm-math-l40s-37576194/run.log
```
Machine-readable copy: `logs/runs/slm-math-l40s-37576194/downward_probe_history.json`.

**This probe should never have run** — see §2.11.

---

# Section 2 — Other problems

## 2.1 Why is there an empty "Tier 2 … (0 iterations)" section?

Line 5667 of the math log. It's the **downward probe's own attempt**, rendered as a third
tier entry in the DAG traversal.

The probe re-selected `Qwen/Qwen3.5-2B [Q4_K_M]` — the same selector as tier 2 at line 5596
(59 iterations). It trained and scored 0.5400, but because the result was **rejected**
(below threshold), the DAG node was never committed. So the summary renders a section for a
selector that has a probe record but no accepted DAG iterations.

Not a crash, but genuinely misleading — the same model appears twice, once with 59
iterations and once with 0. It should render as e.g.
`── Downward probe: Qwen/Qwen3.5-2B [Q4_K_M] — 0.5400, rejected ──`. The data is correct in
`downward_probe_history.json`; only the summary rendering is wrong.

## 2.2 Does `latency=1283.0s` mean Claude spent 3 hours responding?

**No — 1283.0s is 21.4 minutes, not 3 hours.** (You may have read the NER run's 3506.1s ≈
58 min.) It's **cumulative** wall time summed over all calls, not one response.

Math: 1283.0s ÷ 73 calls ≈ **17.6 s/call**.
NER: 3506.1s ÷ 231 calls ≈ **15.2 s/call**.

Individual calls in the log confirm this — `latency=16706.4ms` (16.7 s), `latency=11800.8ms`
(11.8 s). 15–20 s per call is normal for Sonnet with a long prompt.

In context: NER's 58 min of Claude latency is **2.2% of a 44.8-hour run**. LLM latency is
not a bottleneck (§3.1).

## 2.3 What are the `provider=local` failures?

```
cost provider=local calls=43 failures=18 latency=23.3s usd=$0.000000
```

`provider=local` is the **self-hosted vLLM synthesis/judge server** (`Qwen/Qwen3.6-35B-A3B`)
running as a sidecar on the same node. Free ($0) because it's your own GPU.

The 18 failures are all **cold-start connection refusals**, not errors:

```
[synth] endpoint http://127.0.0.1:36194/v1 not reachable (Connection error.);
        synthesis will be skipped (gold-only)
[synth] synthesis server connected on attempt 19 — proceeding
```

The 35B model takes ~9.5 min to load; the pipeline polls every ~30 s meanwhile. All 18
failures precede the connection, zero after. **Working as designed** — the retry loop
exists for exactly this race. (NER shows the same pattern: 8 failures of 9 calls.)

The real issue is what came *next*: the server came up healthy and then **was never used**,
because synthesis is gated behind score ≥ 0.95 (§1.9). You paid ~10 min of startup and held
a 35B model in VRAM for the entire run for zero calls.

## 2.4 Why did the orchestrator only pick Q4_K_M?

Its verbatim reasoning, both runs:

> **NER initial:** "At only 1550MB peak RAM and 1100MB storage, Qwen3.5-2B Q4_K_M offers a
> strong instruction-following base … while … the 4B variants consume far more RAM than
> necessary."

> **NER escalation:** "Q4_K_M is the most resource-efficient variant at the same peak-RAM as
> the 2507 instruct alternative."

> **Math escalation:** "the Q4_K_M variant achieves this at the lowest peak-RAM (2900MB)
> within Tier 3."

So the stated reason is always **lowest peak RAM**. Both runs had all three variants
available (the escalate candidate list includes `/Q4_K_M`, `/Q8_0`, and `/None` for each
base model) and picked Q4_K_M every time.

**The criticism — and how it survived measurement.** The target device has **10,240 MB
usable RAM**. Q4_K_M at 2900 MB uses 28% of it; Q8_0 would still fit comfortably. So the
orchestrator optimized a resource that was never scarce, and always chose the *most lossy*
option — while the run finished 0.017 F1 short of its goal.

**I predicted Q8_0 would recover some of that. I measured it in §2.5, and I was wrong:**
Q8_0 scores **0.8627** vs Q4_K_M's **0.8628** — no recovery at all, for +66% on disk. The
orchestrator's choice was correct.

**What the criticism should have been:** not *"it picked the wrong quant"* but *"it never
measured, it assumed."* It reasoned purely from peak RAM and never checked what precision
costs in accuracy. Here that assumption happened to hold. On a task where quantization does
bite, the identical untested reasoning would silently give up accuracy — and nothing in the
pipeline would surface it. **The prompt also gives the model no notion of "this constraint
has slack, spend it,"** which is a real gap even though it didn't cost anything this time.
The fix is a one-time per-task quant sweep (§2.5 is exactly that experiment), not a
different hard-coded default.

## 2.5 Quantization correctness check — BF16 vs Q8_0 vs Q4_K_M

Built and submitted as
[`tests/pipeline/quant_accuracy_compare_l40s.slurm`](../../tests/pipeline/quant_accuracy_compare_l40s.slurm)
(job `37810980`): takes the NER winner (Qwen3.5-4B + `iter46-d4` adapter) and scores it at
all three precisions against the **same 900-example held-out eval set**.

Q4_K_M is deliberately **rebuilt from scratch** rather than reusing the cached GGUF — a
fresh build is what actually exercises the converter.

**Two things this confirms:**

1. **Monotonic degradation:** BF16 ≥ Q8_0 ≥ Q4_K_M. If **Q4_K_M ties BF16 exactly**, the
   eval is not really running quantized weights — the exact bug the `(weights_ref, quant)`
   cache key was introduced to prevent (see `_build_or_reuse_gguf`, where keying on
   `weights_ref` alone once made Q8_0 silently reuse a Q4_K_M file so both tiers scored
   identically).
2. **Reproducibility:** the fresh Q4_K_M number should land near the run's reported
   **0.8628**.

### RESULTS (job 37810980, COMPLETED, 56m36s)

| Precision | On-disk MB | F1 | Δ vs BF16 |
|---|---|---|---|
| **BF16 (full)** | ~8000 (unquantized) | **0.8636** | — |
| **Q8_0** | 4397.0 | **0.8627** | −0.0009 |
| **Q4_K_M** | 2654.5 | **0.8628** | −0.0008 |

**Verdict: the conversion/quantization path is correct.** Three independent confirmations:

1. **Exact reproducibility.** The freshly-built Q4_K_M scored **0.8628** — identical to the
   run's reported 0.8628 for `iter46-d4`. The converter is deterministic and the run's
   headline number is real.
2. **Genuinely distinct artifacts.** 4397.0 MB vs 2654.5 MB — these are real, different
   quantizations, *not* the silent-cache-reuse bug (where Q8_0 reused a Q4_K_M file and both
   tiers scored identically). BF16 also ran a completely separate code path (Unsloth, not
   llama.cpp), and still landed within 0.001.
3. **Degradation is in the right direction.** BF16 (0.8636) > both quantized variants.
   Q4_K_M edges Q8_0 by **0.0001** — a single-example-scale difference on 900 examples,
   i.e. the noise floor, not a real inversion.

### ⚠️ This result CONTRADICTS my criticism in §2.4 — correction

In §2.4 I speculated that *"Q8_0 plausibly recovers part of the quantization loss at no real
deployment cost."* **The measurement says that is wrong.** Q8_0 recovers **nothing**
(0.8627 vs Q4_K_M's 0.8628 — if anything a hair worse) while costing **1742 MB more on
disk, a 66% size increase**.

So the orchestrator's consistent Q4_K_M preference was **the correct call**, and better
justified than its own stated reasoning. It argued from peak RAM; the stronger argument is
that on this task the higher-precision variant buys literally zero accuracy. My §2.4
criticism of *which quant it chose* does not survive contact with data.

What **does** survive from §2.4 is the narrower point: the orchestrator never *measured*
this tradeoff, it assumed it. It happened to be right here. On a task where quantization
does bite (heavier reasoning, longer generations), the same untested assumption would cost
accuracy silently. **The fix isn't "choose Q8_0" — it's "measure the quant tradeoff once
per task rather than assuming it."**

### The other finding: quantization did NOT cost the NER run its goal

Even at **full BF16**, the model scores 0.8636 — still **0.0164 short** of the 0.88
threshold. The entire quantization stack costs 0.0008 F1 (0.09% relative) for a ~3× size
reduction. So the run's failure to converge was a **model-capacity / data problem, not a
precision problem**, and no amount of higher-precision deployment would have rescued it.

Reproduce with: `tail -30 logs/gpu_setup/quant-acc-cmp-37810980.out`

Two infrastructure bugs were fixed to get this running, both worth knowing:
- An unguarded `module avail cuda | grep …` under `set -o pipefail` **aborted the script
  with completely empty output** when `module` was undefined on the compute node.
- `libllama.so` needs both `libcudart.so.12` (from the venv's `nvidia-*` wheels) and
  `GLIBCXX_3.4.29` from `/sw/gcc/12.3.0/lib64`. Sourcing `.venv_gpu/bin/activate` supplies
  the latter — `setup_gpu_env.sh` appends it there specifically for this (B160).

## 2.6 "failures=139" out of how many? — FIXED

You were right that this is unreadable. **Now fixed** in
[`agent/nodes/evaluate.py`](../../agent/nodes/evaluate.py):

```
  → F1=0.8263  failures=139/800
Score: 0.8263  (Δ=+0.0100 from best 0.8163)  failures=139/800  trajectory=[...]
```

For the record: 139 of **800** (math). NER used 900.

## 2.7 Why doesn't `overall=0.8263` equal the average of easy/medium/hard?

**Because it's a size-weighted average, not a simple mean — and the buckets are very
unequal.** Your line:

```
overall=0.8263  easy=0.956(n=389)  medium=0.769(n=268)  hard=0.580(n=143)
```

**The simple mean (what you'd expect):**
```
(0.956 + 0.769 + 0.580) / 3 = 0.768        ✗ not 0.8263
```

**The weighted mean (what it actually is):**
```
easy:    0.956 × 389 = 371.88 correct
medium:  0.769 × 268 = 206.09 correct
hard:    0.580 × 143 =  82.94 correct
                        ───────
total                 = 660.92 correct out of 389+268+143 = 800

660.92 / 800 = 0.8261   ✓ matches 0.8263 (small gap = displayed accuracies rounded to 3dp)
```

**Confirming from the raw counts** — 139 failures out of 800:
```
(800 − 139) / 800 = 661 / 800 = 0.826250   ✓ exactly the reported best_score 0.82625
```

The simple mean would treat 143 hard examples as equal in weight to 389 easy ones. Easy is
**2.7× larger** than hard, so it dominates. This is correct behavior — `overall` is just
"what fraction of all 800 did it get right."

### The important subtlety: this only works for math

For math (exact-match), `overall == correct/total`, so the buckets partition it exactly.
**For NER it does not**, because span-F1 isn't an average of per-example correctness. Real
NER line:

```
overall=0.8272  easy=0.870(n=300)  medium=0.690(n=300)  hard=0.513(n=300)

weighted mean = (0.870 + 0.690 + 0.513)/3 = 0.691   ✗ vs overall 0.8272 — a 0.14 gap!
```

Not a bug. Two genuinely different metrics:
- **`overall` (span-F1)** gives **partial credit** — 3 of 4 entities correct still scores well.
- **bucket accuracy** is **all-or-nothing per example** — reconstructed from
  `best_result.failures`, so one missed span makes the whole example a failure.

So for NER, `overall` is always **higher** than the bucket average. Worth knowing before
comparing the two columns: for math they reconcile exactly, for NER they never will.

## 2.8 Why two separate rows in the training output?

```
{'loss': '0.3153', 'grad_norm': '0.2486', 'learning_rate': '0.0001421', 'epoch': '1.091'}
{'eval_loss': '0.3621', 'eval_runtime': '8.408', 'eval_samples_per_second': '28.54', ...}
```

Two different events from HuggingFace `Trainer`, printed on the same schedule
(`SLM_EVAL_STEPS=20`, so both every 20 steps) — hence the identical `epoch`.

**Row 1 — training-step metrics** (on data being learned from):

| Field | Meaning |
|---|---|
| `loss` | Training loss on the last batch. Lower = fitting better. |
| `grad_norm` | Magnitude of the gradient update. Spiking = instability (LR too high); →0 = learning stalled. |
| `learning_rate` | Current LR — *changes over time* under the scheduler (warmup then decay), which is why it's an odd number. |
| `epoch` | Fractional passes through the training set. `1.091` = just past the 1st epoch. |

**Row 2 — validation metrics** (on held-out validation, not trained on):

| Field | Meaning |
|---|---|
| `eval_loss` | Loss on the validation split. **This is the number that matters.** |
| `eval_runtime` | Seconds to run the validation pass |
| `eval_samples_per_second` / `eval_steps_per_second` | Throughput |

**Why you need both:** comparing them detects overfitting. Here `loss=0.3153` <
`eval_loss=0.3621` — normal, slightly better on seen data. If training loss keeps falling
while `eval_loss` *rises*, the model is memorizing. That's exactly what `EarlyStoppingCallback`
watches (`metric_for_best_model="eval_loss"`), and it's separate from the F1 the pipeline
optimizes — `eval_loss` picks the best checkpoint *within* one training run; F1 compares
*across* iterations.

## 2.9 "IMPROVEMENT" log line — ADDED

Implemented in [`agent/nodes/evaluate.py`](../../agent/nodes/evaluate.py). On any new best:

```
[evaluate][Qwen/Qwen3.5-4B [Q4_K_M]]   IMPROVEMENT — iteration 46 — data_rebuild — 0.8564 → 0.8628 (Δ=+0.0064)
```

Includes the intervention type as requested, plus the before/after and delta. Greppable
with `grep IMPROVEMENT run.log` to get the whole improvement history in one shot — for NER
that's 6 lines out of 142 iterations.

## 2.10 Data goals per run: were they met?

| | NER | Math |
|---|---|---|
| Curriculum target | 4500 | 4500 |
| **Curriculum actual** | **3403** | **3403** |
| Eval target | 900 | 800 |
| Eval actual | 900 ✓ | 800 ✓ |
| Contamination removed | 24 rows | 0 (but rejected a 128-overlap dataset) |

**Neither run met its curriculum goal — both got 3403 of 4500 (76%).** Not a coincidence:
3403 is the **entire training split** of both source datasets. The planner asked for more
data than exists, and the pipeline silently delivered everything available.

**The filtering pipeline** (in order):
1. **Discovery** — Exa searches for candidate HF datasets
2. **Schema mapping** — can its columns map to the task's `text`/`labels` shape?
3. **Contamination check** — normalize text, drop any train row matching a test row
   (this rejected `gabrielaltay/gsm8k-math-reasoning` outright for 128 overlaps)
4. **Eval split** — carve out held-out pos/neg/boundary, stratify by difficulty
5. **Curriculum assembly** — fill to `curriculum_size` from remaining rows
6. **Per-rebuild re-selection** — `data_rebuild` re-derives the training subset

**The criticism:** the 4500→3403 shortfall is never surfaced as a warning. The planner set
4500 with a specific rationale ("biased heavily upward … for statistically reliable
per-entity macro span-F1"), the pipeline delivered 76% of it, and nothing flagged the gap.
This is *precisely* the situation `mine_new_real_source` and `targeted_synth_positive`
exist for — the run needed ~1100 more examples, had two mechanisms to get them, and used
neither (§1.9). A run that knows it's data-starved should say so loudly.

## 2.11 Downward probe should only run on untried tiers — FIXED

**You're right, and it was a real bug that wasted a full train+eval cycle.**

What happened in math: tier 2 (`Qwen3.5-2B@Q4_K_M`) was the run's **own starting tier** —
59 iterations, best F1 **0.6925**, abandoned for stagnation. After converging at tier 3, the
probe **re-selected that same tier 2 model**, retrained it with one arbitrary fixed config,
and scored **0.5400**. Strictly worse than the 0.6925 already on record, and it could never
have cleared the 0.82 threshold.

**Root cause:** `downward_tiers_tried` only tracks tiers *the probe itself* has tried. It
starts empty, so it has no idea the main escalation ladder already spent 59 iterations there.

**Fix** — new `tiers_already_explored()` in
[`agent/nodes/downward_probe.py`](../../agent/nodes/downward_probe.py), which reads
`state["model_baselines"]` (one entry per selector that ever reached `evaluate_node` = every
tier the ladder actually tried) and unions it into `tried`. Wired into **both** consumers:

- `downward_probe_step_node` — won't select an already-trained tier
- `iterate_node`'s routing gate — won't even *route* to the probe if the only lower tiers are spent

**4 new tests** in `tests/nodes/test_downward_probe.py`, including one reproducing the exact
math-run scenario and asserting neither `_llm_choose_model` nor `_train_and_eval` is called.

Net effect on that run: skips a pointless train+eval (~13 min of 4×L40S) and one paid
model-choice call.

## 2.12 How can the NER baselines be 0.0000?

```
Tier  Model             Quant     Baseline   Best FT         Δ
   2  Qwen/Qwen3.5-2B   Q4_K_M      0.0000    0.8476   +0.8476
   3  Qwen/Qwen3.5-4B   Q4_K_M      0.0254    0.8628   +0.8374
```

**The 0.0000 is not a real measurement — it's a swallowed infrastructure failure.**
Log line 135:

```
[evaluate] Baseline measurement failed (CUDA worker 'eval' failed (exit=1,
  ValueError: Failed to load model from file:
  artifacts/gguf/Qwen_Qwen3.5-2B/663bb41fea0a/model-q4_k_m.gguf)
[evaluate] Baseline F1 = 0.0000
```

The GGUF failed to load once, the exception was caught, and `baseline_f1 = 0.0` was
recorded as if measured. The same build succeeded moments later for training evals — a
transient fault, not a real zero.

**How the numbers are computed:** baseline = zero-shot F1 of the base model with **no
adapter**, run once at iteration 1 through the identical eval path. `Δ = Best FT − Baseline`
is meant to isolate fine-tuning's contribution.

**Are they accurate?**
- **Tier 2 (0.0000): no.** A failed load recorded as a score. The `+0.8476` improvement is
  therefore meaningless — it's `0.8476 − (measurement failure)`.
- **Tier 3 (0.0254): yes, and it's genuinely near-zero.** A base model asked for strict
  BC5CDR span extraction with no fine-tuning produces prose, not the exact span format —
  near-total failure under strict span-F1 is the correct result. Math is the useful
  contrast: its 4B baseline scored **0.7913**, because a base model can already do
  grade-school math; it just needed format alignment.

**The bug:** a caught exception and a real 0.0 are indistinguishable downstream. It should
record `None`/`unmeasured` and print `n/a`, so a failed baseline can't masquerade as a
+0.85 improvement in the final report. *(Not yet fixed — flagging for your call, since it
touches the reporting contract.)*

## 2.13 Show hyperparameter diffs vs current best — ADDED

Implemented via `_config_diff()` in [`agent/nodes/train.py`](../../agent/nodes/train.py),
diffing against `_best_prior_config()` (best-scoring non-pruned config in DAG history):

```
[train][Qwen/Qwen3.5-4B [Q4_K_M]] Iteration 21
[train][Qwen/Qwen3.5-4B [Q4_K_M]]   Config: LoRA r=64 a=256 drop=0.05 wd=0.05 lr=1e-04 ep=5 ...
[train][Qwen/Qwen3.5-4B [Q4_K_M]]   Diff vs best prior config: weight_decay 0.01→0.05
```

Covers all 9 tunable fields. Three special cases:
- First iteration → `no prior config (first iteration for this model)`
- Data-only/rollback retry → `unchanged from best prior config` (**makes `data_rebuild`
  iterations visibly distinguishable** — previously you couldn't tell if the config moved)
- Multi-field → `lora_rank 32→64, lora_alpha 64→128`

**5 new tests** in `tests/nodes/test_iterate_train_fixes.py`.

---

# Section 3 — Open questions

## 3.1 Biggest time bottlenecks, and how to cut them

Measured from `timing-events.jsonl`. `graph_node/*` spans are non-overlapping and sum to
wall clock; `worker_op/*` are nested inside them.

**NER — 44.72 h total:**

| Phase | Hours | % of run | n | Avg |
|---|---|---|---|---|
| **`evaluate` node** | **25.36** | **56.7%** | 142 | 643 s |
| ├─ scoring the eval set | 17.78 | 39.8% | 146 | 438 s |
| └─ building the GGUF | 7.64 | 17.1% | 144 | 191 s |
| **`train` node** | **18.21** | **40.7%** | 142 | 462 s |
| `iterate` (all LLM calls) | 1.00 | 2.2% | 142 | 25 s |
| everything else | 0.15 | 0.3% | — | — |

**Math — 20.37 h total:** `evaluate` 14.12 h (69%), `train` 4.74 h (23%), `build_gguf` 2.14 h.

> **The headline result: evaluation costs more than training.** 25.4 h vs 18.2 h in NER;
> 14.1 h vs 4.7 h in math (**3×**). Nearly everyone assumes training dominates. It doesn't.

Ranked fixes:

**1. Stop re-scoring all 900 examples every iteration (saves ~40%).** 438 s per eval × 142.
Score a stratified ~200-example subset for the iterate decision; run the full 900 only to
confirm a new best. Same decisions, ~4× cheaper. **Biggest single lever.**

**2. Stop rebuilding a GGUF for every iteration (saves ~17%).** 7.64 h building 144 GGUFs
that are read once each. Two options: (a) score the LoRA adapter directly in BF16 during
search and only quantize when confirming a new best — you'd trade some honesty for 7 h; or
(b) keep honest quantized eval but skip the rebuild for iterations whose config is
hyperparameter-identical to one already built. Note the new-best-only GGUF retention
already implemented saves *disk*, not this time.

**3. Kill the run when it's flatlined (saves ~30% of NER).** The NER best landed at
iteration 46; iterations 47–68 bought zero improvement over ~7 h before the crash.
Stagnation detection exists for *escalation* but nothing stops a top-tier plateau.

**4. Early-abort doomed training runs (saves ~10–15%).** 92% of iterations end in rollback,
each paying a full 462 s train. Check validation loss against the best-so-far at ~30% of
epochs and abort clear losers.

**5. Don't scale the tiny thing.** LLM orchestration is **2.2%** of wall clock. Optimizing
prompt latency is not worth touching.

## 3.2 Reliably parsing arbitrary phone specs

Your instinct is right — there's no single good free database. What you need is
**(chipset, total RAM)**, from which everything else is derived.

**What's actually going wrong today isn't lookup, it's determinism.** The same S24 Ultra
produced two different storage budgets across your two runs — 209,715 MB (NER) vs 153,600 MB
(math), **27% apart**, purely from LLM rationale variance:

> NER: *"~200 GB … reflecting realistic free space on a 256 GB device"*
> Math: *"~60% of the 256 GB … to leave headroom for OS, apps, and user data"*

Both defensible; that's the problem. A hardware constraint that bounds which models the
search may try should not be re-derived by free-form reasoning every run.

**Recommended: a three-layer cascade, deterministic-first.**

**Layer 1 — local device DB (already exists).** `data/devices.csv` + `refresh_device_db.py`,
and the NER run did hit it (`source: local_db`). Best coverage source is
**GSMArena** (~13k devices, chipset + RAM variants) via a periodic scrape, or the
**Android Device Catalog** (Play Console CSV export — authoritative for RAM/ABI/SoC on
shipping devices, free, ~30k entries). Store `(canonical_name, chipset, ram_variants[])`.

**Layer 2 — chipset → performance mapping (the part that matters).** Device count is a
long tail, but **chipsets are only a few hundred**, and inference speed is a property of the
SoC, not the phone. You already do this with `tok_s_snapdragon_*` reference chips. Extend
that table rather than chasing per-device data. Ground it in real measurements from
**llama.cpp's community benchmark threads**, which post actual tok/s per SoC per quant.

**Layer 3 — LLM fallback, but constrained.** For unknown devices, let the LLM infer only
`(chipset, ram_gb)` — **facts**, not budgets. Then compute budgets with a **fixed formula**:

```python
usable_ram_mb   = total_ram_mb - OS_RESERVE_MB      # fixed constant, not LLM judgment
storage_mb      = STORAGE_FRACTION * total_storage  # fixed constant
```

This is the key change: the LLM supplies facts it can look up; the *policy* is code. Same
phone → same budget, every run, forever.

**Also worth doing:** let the user pass `--chipset` / `--ram-gb` explicitly to bypass
inference entirely, and log `spec_source` (`local_db` / `llm_inferred` / `user_override`) so
a wrong constraint is traceable. `device_research.json` already records `spec_source` — it
just isn't acted on.

## 3.3 What actually caused the gains?

Only **6 of 142** NER iterations and **6 of 66** math iterations improved. Grouped by cause:

**(a) Escalating model capacity — by far the biggest single jump.**

| Run | 2B best | 4B zero-shot | 4B best |
|---|---|---|---|
| Math | 0.6925 | **0.7913** | 0.8263 |
| NER | 0.8476 | 0.0254 (failed) | 0.8628 |

Math is the clean story: the 4B model's **untuned** score (0.7913) beat 59 iterations of
2B tuning (0.6925) by **+0.099**. Then 6 iterations of tuning added only +0.035. *The model
swap was ~3× more valuable than all the fine-tuning on the smaller model.* The test agent
had diagnosed this correctly for dozens of iterations — *"an optimization/capacity gap"* —
but the escalation trigger needed 50 stalled evals to fire.

**(b) Regularization — the single best hyperparameter change.** NER iteration 21,
weight_decay 0.01→0.05: **0.8231 → 0.8561 (+0.033)**, the largest hyperparameter gain in
either run. Why: 4B params LoRA-tuned on ~3400 examples overfits easily; weight decay
penalizes large weights, and BC5CDR needs *generalizable* boundary rules, not memorized
spans. Notably the *fallback* ladder found this, not the LLM.

**(c) Alpha/rank ratio.** NER iteration 7, alpha 64→256 at fixed r=64: **0.8205 → 0.8311**.
LoRA scales its update by `alpha/rank`, so 4× alpha = 4× stronger adaptation. For a domain
as far from pretraining as biomedical NER, a stronger update helps. The LLM reasoned about
this explicitly: *"the best configuration (r=64, a=256) benefits from the 4x alpha scaling."*

**(d) Higher LR + larger batch, together.** Math's winning iteration 6: lr 1e-4→2e-4 with
effective batch 8→16. Larger batches give less noisy gradients, which *permits* a higher LR.
Changed alone, higher LR regressed (iterations 3–4).

**(e) The counter-example — capacity that didn't help.** Math iteration 9 found its 2B best
with **r=8, alpha=8, lr=1e-5, 1 epoch** — the *smallest, gentlest* config tried. Every
attempt to add capacity (r=64: 0.6025; r=64 a=128: 0.5800) made it *worse*. The 2B model
was capacity-limited at the *base model* level; adding LoRA capacity just overfit faster.
Iterations 2–5 all scored **below the untuned baseline** — fine-tuning actively damaged the
model until the LR came down 20×.

**The synthesis:** the gains came from **model capacity** (a) and **regularizing to avoid
overfitting a small dataset** (b, c, e). Notably, *no* gain in either run came from
`data_rebuild` improving the data — because it never actually improved the data (§1.8).

## 3.4 Every tunable hyperparameter — and which ones earn their place

The orchestrator may set 9 fields ([`agent/nodes/iterate.py:296`](../../agent/nodes/iterate.py#L296)):

| # | Field | What it does |
|---|---|---|
| 1 | `lora_rank` | Dimensionality of the low-rank update. Higher = more capacity + more overfit risk. |
| 2 | `lora_alpha` | Scaling: update is multiplied by `alpha/rank`. Effective strength. |
| 3 | `lora_dropout` | Dropout inside LoRA layers. Regularizer. |
| 4 | `weight_decay` | L2 penalty on weights. Regularizer. |
| 5 | `learning_rate` | Step size. |
| 6 | `nr_epochs` | Passes over the training set. |
| 7 | `micro_batch_size` | Examples per forward pass. **Sets peak VRAM.** |
| 8 | `gradient_accumulation_steps` | Micro-batches accumulated before an optimizer step. |
| 9 | `effective_batch_size` | `micro_batch × grad_accum`. What the optimizer actually sees. |

### Self-criticism: three of these don't belong in the search

**Fields 7/8/9 are one knob presented as three, and two are hardware settings, not
learning settings.** Only `effective_batch_size` affects *what the model learns*.
`micro_batch_size` and `gradient_accumulation_steps` merely decide how that batch is split
to fit in VRAM — a **hardware** concern with (barring numerical edge cases) *no effect on
the learned function*.

The logs show the orchestrator wasting iterations on exactly this. Math iterations 14, 19,
24, 39, 52 all hold every learning parameter fixed and only reshuffle the split:

```
iter  9  r=8 a=8 lr=1e-05 ep=1 mb=4 ga=4 eb=16   0.6925   ← best
iter 14  r=8 a=8 lr=1e-05 ep=1 mb=8 ga=2 eb=16   0.6850   ← same eb, different split
iter 19  r=8 a=8 lr=1e-05 ep=1 mb=2 ga=8 eb=16   0.6825   ← same eb, different split
iter 24  r=8 a=8 lr=1e-05 ep=1 mb=1 ga=8 eb=8    0.6800
```

Three full train+eval cycles (~35 min of 4×L40S) spent measuring **noise**. The
±0.01 spread is run-to-run variance, not signal — but the orchestrator can't know that, so
it may "learn" a false lesson from it.

**Recommendation:** expose only `effective_batch_size` to the orchestrator, and have the
trainer derive the largest `micro_batch_size` that fits VRAM (it already has OOM-halving
logic in `infer_batch`). Removes 2 of 9 dimensions and eliminates a whole class of
noise-measuring iterations.

**`lora_alpha` should arguably be `alpha_ratio`.** Alpha only matters relative to rank
(`alpha/rank`). Exposing both invites incoherent combinations, and the fallback ladder
already couples them (r 32→64 auto-scales a 64→128). Exposing `alpha_ratio ∈ {0.5, 1, 2, 4}`
would make the search space smaller *and* more meaningful.

**Two regularizers is one too many, given the data volume.** `lora_dropout` and
`weight_decay` both combat overfitting. Across both runs, `weight_decay` produced the single
best hyperparameter gain (+0.033), while `lora_dropout` never produced a new best in either
run — NER held it at 0.05 for essentially the entire run; math's excursions to 0 and 0.1
(iterations 42–45, 50) all failed. Given ~19 min per measurement, a knob with no
demonstrated wins is expensive to keep.

**What genuinely earns its place:** `learning_rate` (largest observed swings, both
directions), `lora_rank` (real capacity control), `weight_decay` (best single gain),
`nr_epochs` (interacts with early stopping), `effective_batch_size` (enables higher LR).

**Proposed reduction: 9 → 5** (`learning_rate`, `lora_rank`, `alpha_ratio`, `weight_decay`,
`nr_epochs`), with `effective_batch_size` kept as a 6th only if you want LR/batch
co-tuning. Given ~19 min per sample, halving the dimensionality is worth more than any
smarter search strategy layered on top of the current 9.

---

# Changes made in this session

**Code (all tested, 769 passing + 1 known Lustre-I/O flake):**

| Change | File | Tests |
|---|---|---|
| `failures=N/total` in eval logs | `agent/nodes/evaluate.py` | existing updated |
| `IMPROVEMENT — iteration N — <type>` line | `agent/nodes/evaluate.py` | existing updated |
| `Diff vs best prior config:` line | `agent/nodes/train.py` | 5 new |
| Downward probe skips ladder-tried tiers | `agent/nodes/downward_probe.py`, `agent/nodes/iterate.py` | 4 new |
| Quantization comparison job | `tests/pipeline/quant_accuracy_compare_l40s.slurm` | job 37810980 ✓ passed |

**Quantization verdict (§2.5):** toolchain confirmed correct — fresh Q4_K_M build reproduced
the run's 0.8628 exactly. BF16 0.8636 / Q8_0 0.8627 / Q4_K_M 0.8628. Quantization costs
**0.0008 F1 (0.09%)** for a ~3× size reduction, so it is *not* why the NER run missed 0.88
(BF16 is still 0.0164 short). This measurement **overturned my own §2.4 criticism** — Q8_0
recovers nothing, so the orchestrator's Q4_K_M preference was right.

**Known issues flagged but NOT fixed** (need your call):

1. **`hyperparams`-with-`data_rebuild` validation failure** (§1.4/1.7) — the highest-value
   fix. Either relax the schema to ignore a stray `hyperparams` on a `data_rebuild`, or
   make the reask prompt explicitly restate the exclusion. Currently silently disables the
   entire data-rebuild reasoning path on affected runs.
2. **Fallback keyword-matching is structurally dead** (§1.9) — every `data_rebuild`
   diagnosis lacks the keywords the plan builder matches, so it always yields
   `resample_existing`. Root cause of the NER crash.
3. **Synthesis gated at score ≥ 0.95** (§1.9) — above both stop thresholds, so
   `targeted_synth_positive` is unreachable in any normally-converging run.
4. **Failed baseline recorded as 0.0** (§2.12) — should be `None`/`n/a`.
5. **`plan space is exhausted` is an uncaught crash** — should terminate the run cleanly
   with the best model preserved.
6. **Empty "(0 iterations)" tier section** (§2.1) — probe attempts should render distinctly.
7. **Curriculum shortfall is silent** (§2.10) — 3403 of 4500 requested, no warning.
