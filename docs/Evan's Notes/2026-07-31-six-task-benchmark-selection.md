# Six-Task Benchmark Selection — 2 per Category

**Date:** 2026-07-31
**Purpose:** Pick 2 benchmarks from each of the three task categories (In-Distribution,
Format-Bound, Data-Scarce), verify a live public dataset exists for each, and specify how the
eval harness gets built for each against the current pipeline.
**Companion to:** [2026-07-28-task-suite-benchmark-research.md](2026-07-28-task-suite-benchmark-research.md)
(the 18→6 survivor analysis this re-frames onto the 3-category structure).

---

## 0. The binding constraint

The pipeline speaks exactly **5 task types** and **3 output schemas**, with one scorer per type:

| `task_type` | schema | scorer | metric (`TASK_METRIC_NAMES`) |
|---|---|---|---|
| `classification` | `{text,label}` | `eval/scorers/classification.py` | macro-F1 |
| `NER` | `{text,entities}` | `eval/scorers/ner.py` | span-F1 |
| `math_reasoning` | `{text,answer}` | `eval/scorers/generation.py` | exact-match |
| `code_generation` | `{text,answer}` | `eval/scorers/generation.py` | execution pass@1 |
| `generation` | `{text,answer}` | `eval/scorers/generation.py` | local-judge [0,1] |

Dispatch is in [`eval/harness.py:137-147`](../../eval/harness.py#L137-L147). Every pick below is
chosen for the subsystem it stresses, and is tagged **native** (loader only) or **needs a new
verifier** (the `TaskContract` / verifier-registry gap already flagged in the 2026-07-28 notes §5.3).

---

## 1. The six tasks

| # | Category | Task | `task_type` | Primary dataset (train / test) | Metric | Build effort |
|---|---|---|---|---|---|---|
| 1 | **1 In-Distribution** | Intent classification | `classification` | **CLINC150** 15,200 / 5,500 (+3,100 val; 150 classes +OOS) | macro-F1 | ✅ Native |
| 2 | **1 In-Distribution** | Dialogue/email summarization | `generation` | **DialogSum** 12,460 / 1,500 · **SAMSum** 14,732 / 819 | local-judge [0,1] (+ROUGE 2nd) | ✅ Native |
| 3 | **2 Format-Bound** | Intent → app action (function calling) | `code_generation`* | **xlam-function-calling-60k** 60,000 / — · eval on **BFCL** | AST arg-match + format-valid | ⚠ New verifier |
| 4 | **2 Format-Bound** | Text edit as unified diff (prose) | `generation`* | **CoEdIT** ~69k public (train+val); carve test | `git apply --check` + result match | ⚠ New verifier |
| 5 | **3a Data-Scarce (small)** | Escalate-to-cloud router | `classification` | **RouterBench** ~30k prompts / 405,467 outcomes | binary-F1 + cost-quality curve | ⚠ Label derivation |
| 6 | **3b Data-Scarce (large)** | Health / medication QA | `classification` | **MedQA-USMLE-4-opt** 10,178 / 1,273 · **MedMCQA** 182,822 / 6,150 | accuracy (MCQ) | ✅ Native |

\* closest existing type; the true home is a new `function_call` / `diff` contract (§4).

---

## 2. Why these six best exercise *this* pipeline

Each pick stresses a different subsystem, not just a different domain.

### Category 1 — the control (find the smallest model matching base quality)

Both are deliberately near-saturated; that is the point. They test the parts that must work *cheaply*.

- **CLINC150 over BANKING77.** 150 classes (not 77) resist tiny-model saturation slightly longer,
  and the built-in **out-of-scope** class gives a free `neg`/hallucination slice for
  `eval_setup`'s pos/neg/boundary split. Pure native `classification` — this is the
  `smallest_first` + `downward_probe` efficiency demonstration.
- **Summarization** is the only **decode-heavy / long-context** control and the one that exercises
  the `generation` local-judge path. It will surface the known bug that the judge mean is reported
  in the `f1` field ([PIPELINE.md §12.1 #5](../PIPELINE.md)) — fix before relying on it.

### Category 2 — where the pipeline's value is provable, and where its biggest gap lives

Both are format-bound with **executable, judge-free verifiers**, giving the two-column
*format-valid vs. content-correct* split for free. That split is the substrate for the one novel
result available: **format compliance as a function of quantization bit-width** (the GSM8K sweep
already shows Q4 costs 0.0300 on reasoning vs. 0.0008 on NER — [followups §1.4](2026-07-28-followups.md)).
These are also the two tasks that fit **no current schema**, so building them is what forces the
`TaskContract`/verifier-registry work. Function calling additionally has a real **0.6B→3B accuracy
gradient** (BFCL 45.8%→65.7%) — the size search has something genuine to find.

### Category 3 — decouple data cost from model size (the joint-optimization thesis)

- **RouterBench (3a, ~300M).** All 11 models' correctness+cost are pre-recorded, so you can **train
  a router on purely synthetic data and evaluate against 405k real outcomes offline, at zero GPU
  cost.** If a 300M synthetic-only router matches routers fit to real traces, that is external
  validation of the data-generation claim that no no-benchmark task can give. Its label is
  counterfactual (does-local-succeed) — exactly the "unobservable in the wild" property of 3a.
  *(Alternative, lower-risk: ambiguity detection — CLAMBER 3k / AmbigQA 14k — cleaner native
  `{text,label}` fit, no label derivation, but no external-validation property.)*
- **MedQA (3b, 3B+).** Knowledge-bound, private-by-construction; anchors the top of the size range
  so Category 3 spans sizes instead of collapsing to "hard = big." Keep it **MCQ-only** (objective
  accuracy, no free-text medical advice) to sidestep both the safety risk and the
  "factory grades its own homework" objection.

**Hardware-profile spread:** router = pure prefill/TTFT; function-calling = balanced;
summarization = decode/peak-memory; MedQA = KV-cache. **Size spread:** ~300M → 3B+.

---

## 3. Datasets — verified live (2026-07-31), with splits and links

| Task | Dataset | Train | Val | Test | License | Link |
|---|---|---|---|---|---|---|
| Intent (1) | CLINC150 (DeepPavlov) | 15,200 | 3,100 | 5,500 | CC-BY-3.0 | [HF](https://huggingface.co/datasets/DeepPavlov/clinc150) |
| Intent (1, alt) | BANKING77 | 10,003 | — | 3,080 | CC-BY-4.0 | [HF](https://huggingface.co/datasets/PolyAI/banking77) |
| Summ (2) | DialogSum | 12,460 | 500 | 1,500 | CC-BY-NC-SA-4.0 | [HF](https://huggingface.co/datasets/knkarthick/dialogsum) |
| Summ (2) | SAMSum | 14,732 | 818 | 819 | CC-BY-NC-ND-4.0 | [HF](https://huggingface.co/datasets/knkarthick/samsum) |
| Summ (2, email) | EmailSum | 2,549 threads | — | — | research | [paper](https://aclanthology.org/2021.acl-long.537/) |
| Func-call (3) | xlam-function-calling-60k | 60,000 | — | — | CC-BY-4.0 (gated) | [HF](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) |
| Func-call (3, eval) | BFCL | — | — | ~4k cases | Apache-2.0 | [repo](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) |
| Func-call (3, alt) | TOPv2 / MTOP | 125k / 100k | — | — | research | [MTOP HF](https://huggingface.co/datasets/iohadrubin/mtop) |
| Diff (4) | CoEdIT | ~69k public (train+val)† | — | carve yourself | Apache-2.0 | [HF](https://huggingface.co/datasets/grammarly/coedit) |
| Router (5) | RouterBench | ~30k prompts / 405,467 outcomes | — | split yourself | Apache-2.0 | [HF](https://huggingface.co/datasets/withmartian/routerbench) |
| Med (6) | MedQA-USMLE-4-opt | 10,178 | 1,272 | 1,273 | MIT | [HF](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) |
| Med (6) | MedMCQA | 182,822 | 4,183 | 6,150‡ | MIT | [HF](https://huggingface.co/datasets/openlifescienceai/medmcqa) |
| Med (6, held-out) | MedicationQA | 674 (real consumer) | — | — | research | [repo](https://github.com/abachaa/Medication_QA_MedInfo2019) |

† CoEdIT HF release is ~69k (paper used 82k; ~13k Simplification/Formality rows withheld for
licensing). It ships `src`/`tgt` **full-text pairs**, not diffs — compute the unified diff as gold
with Python `difflib` (free, exact).
‡ MedMCQA test labels are **not public** (submission-only) — use the 4,183-row `validation` split as
the held-out `E`.

---

## 4. How the eval harness gets built, per task

The pipeline already builds, per run, a frozen `E = E_pos ∪ E_neg ∪ E_boundary` at fixed 0.4/0.4/0.2,
difficulty-stratified (easy/medium/hard) by the small-vs-large base-model gradient, behind a 4-layer
contamination firewall ([PIPELINE.md §4](../PIPELINE.md)). Each task plugs in via a shared-dataset
bundle (`SLM_SHARED_DATASET_DIR`) so competing candidates see identical data.

### Native tasks (1, 2, 6) — loader + slice mapping only

- Deterministic loader alongside `data/loaders/*` that pulls the HF split into the correct schema.
- pos/neg/boundary per [followups §1.1](2026-07-28-followups.md) conventions:
  - CLINC150 → pos = in-scope, neg = OOS, boundary = confusable domain pairs.
  - MedQA → pos = single-fact recall, neg = distractor-heavy, boundary = multi-step vignettes.
  - Summarization → pos = well-formed threads, neg = degenerate/empty, boundary = multi-topic.
- Set `stop_threshold` from a **real leaderboard**, not planner recall (the NER run's failure came
  from an ungrounded 0.88 — [2026-07-26 §1.2](2026-07-26.md)).
- Summarization needs the `generation` `f1`-field mislabel fixed so the judge mean isn't reported as F1.

### Format-bound tasks (3, 4) — the real build: verifier registry + two-column scorer

Implement the `TaskContract` from the 2026-07-28 notes §5.3:

- New `eval/scorers/` modules returning **two** numbers — `format_valid` and `content_correct` —
  routed through `EvalResult`. The pipeline's comparison scalar stays `content_correct`;
  `format_valid` is logged as a first-class metric (mirrors Aider's separate columns).
- **Function calling** → a `function_call` verifier doing BFCL-style AST comparison:
  (1) name ∈ allowed set (catches hallucinated functions), (2) right function, (3) required args
  present, (4) values match with type coercion. Train on xlam-60k; hold out BFCL non-live-AST +
  relevance as the frozen `E`.
- **Diff** → a `git_apply` verifier: gold = `difflib` unified diff of `src`→`tgt`;
  `format_valid` = `git apply --check` exit code, `content_correct` = string match after applying.
  No judge, no sandbox.
- Both go in a **registry the orchestrator selects from**
  (`{exact_match, macro_f1, git_apply_check, ast_call_match, judge_rubric,…}`) — ~200 lines, no RCE.

### Router (5) — native scorer, custom label derivation

Runs as plain `classification` once labels exist. Loader reads RouterBench's recorded per-model
correctness, picks the on-device model's column as "local," sets label = `local_correct`. Train the
router on **synthetic-only** curriculum; evaluate on held-out RouterBench prompts. Report binary-F1
(fits the harness directly) **and** the offline cost-quality curve area (a post-hoc script over the
frozen outcome table — zero inference).

### Cross-cutting

- Run the format-bound pair (3, 4) through the BF16/Q8_0/Q4_K_M sweep with the two-column scorer and
  **constrained decoding off** — this produces the format-vs-bitwidth result no one has published.
- Take the eval-cost lesson from the NER run (`evaluate` was 57% of wall-clock): score a stratified
  ~200-row subset for `iterate` decisions, full `E` only to confirm a new best.

---

## 5. Sources

CLINC150 [HF](https://huggingface.co/datasets/DeepPavlov/clinc150) ·
BANKING77 [HF](https://huggingface.co/datasets/PolyAI/banking77) ·
DialogSum [HF](https://huggingface.co/datasets/knkarthick/dialogsum) ·
SAMSum [HF](https://huggingface.co/datasets/knkarthick/samsum) ·
EmailSum [ACL](https://aclanthology.org/2021.acl-long.537/) ·
xlam-60k [HF](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) ·
BFCL [repo](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) ·
CoEdIT [HF](https://huggingface.co/datasets/grammarly/coedit) ·
RouterBench [HF](https://huggingface.co/datasets/withmartian/routerbench) · [paper](https://arxiv.org/abs/2403.12031) ·
MedQA-USMLE-4-opt [HF](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) ·
MedMCQA [HF](https://huggingface.co/datasets/openlifescienceai/medmcqa) ·
MedicationQA [repo](https://github.com/abachaa/Medication_QA_MedInfo2019)
