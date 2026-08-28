# Going below 0.6B, and what the evidence says about the five tasks

**Date:** 2026-08-22 — research + status audit, no code changed yet
**Companions:** `08-17-task-status.md`, `08-19b-task-registry-rebuild.md`, `08-21-run-38708719-review.md`

> **Superseded in two places, 2026-08-23.**
> **(1) `sms_spam` now exists.** §2.1 below reports there is no spam task and §3.4 recommends not
> building one, on the grounds that the fine-tuning headroom is ~2 points. You decided to build it
> anyway and the recommendation is overruled, not withdrawn — the headroom argument still stands and
> the run will test it. The task is registered, vendored and preflight-green; the registry is nine
> tasks. §4.1's Banking77 proposal is **on hold** by your instruction.
> **(2) The one-line GGUF blocker in §1.3 is fixed.** `training/slm_helpers.py` now renders the
> fallback prompt from the served model's own chat template instead of hardcoded Qwen ChatML, and
> derives its stop tokens the same way. Qwen bytes are unchanged and pinned by a regression test.
> Both are written up in `08-23-calendar-variance-sms-spam-small-models.md`.

Three questions, answered against the code as it stands today rather than against the last note.

**The short version.**

1. **DistilBERT and TinyBERT cannot work here**, and it is not a packaging problem — every task in the
   registry is scored by generating text and parsing it, and an encoder has nothing to generate. The
   models that *do* fit are small **decoder** LMs, and there are four good ones. The best single pick
   is **Gemma 3 270M IT** (253 MB at Q4_K_M, 1.8× smaller than the current floor).
2. **Exactly one line of code blocks all of them**, and I found it:
   `training/slm_helpers.py:960` refuses to score any non-Qwen GGUF. It is a ~20-line fix, not an
   ecosystem change.
3. **Spam classification is not in the pipeline.** It is not a stale harness — there is no task
   module, no loader, no slurm script and no run. It was deleted on 2026-08-18 and `PAPER.md` still
   quotes a headline number for it.
4. On the literature: **xlam_bfcl and ner_bc5cdr have exactly the evidence you asked for.**
   calendar_json has good evidence only by analogy. **clinc150 and spam do not, and the reason is
   the same for both — the ceiling is too close to the floor.** §4 names replacements.

---

## Part I — Models below 0.6B

### 1.1 Why DistilBERT and TinyBERT are the wrong shape

Not a dependency problem. A contract problem, and it is worth being precise because the instinct
("small BERT, small model, same thing") is reasonable and wrong here.

Every one of the eight registered tasks is scored the same way, in `eval/harness.py`:

```299:321:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/eval/harness.py
    def _infer(chunk: list[str]) -> list[str]:
        if gguf_path is not None:
            return infer_batch_gguf(chunk, gguf_path, ...)
        return infer_batch(chunk, weights_ref, base_model, ...)
```

Build a prompt, **generate a string**, parse the string, score it. CLINC150 does not read a softmax
over 151 logits — it generates the literal text `accept_reservations` and the scorer matches it.
BC5CDR generates a JSON array of spans. An encoder with a classification head produces logits and
never produces a string, so it does not fail at the eval step, it has no eval step to fail at.

Supporting that would mean a second training path (`AutoModelForSequenceClassification`, different
collator, different loss), a second eval path, a second scorer contract per task, and pool entries
with no meaningful `quant` variants. It is a parallel pipeline, not a pool entry. And it would only
ever cover the three classification tasks — BC5CDR, xlam and calendar have no fixed label set for a
head to predict over.

One thing worth noting because it looks like a counterexample: the local llama.cpp checkout *does*
convert BERT.

```
"DistilBertForSequenceClassification": "bert",
```

But llama.cpp serves BERT as an **embedding** model. There is no `create_chat_completion` for it.
The conversion existing does not mean the eval path exists.

### 1.2 What the pipeline actually requires of a model

I traced this end to end so the shopping list is exact. A candidate must satisfy five things:

| # | Requirement | Enforced at |
|---|---|---|
| 1 | Decoder-only causal LM | `FastLanguageModel.from_pretrained` |
| 2 | Unsloth supports the architecture | `training/lora_trainer.py:395` |
| 3 | LoRA targets are `q/k/v/o_proj`, `gate/up/down_proj` | `training/lora_trainer.py:424` |
| 4 | Tokenizer ships a chat template | `training/lora_trainer.py:606` |
| 5 | `convert_hf_to_gguf` knows the architecture | `training/quantize.py:336` |

Requirement 3 is the quiet one. Those are Llama/Qwen-style projection names, hardcoded as a flat
list. Any architecture that names its modules differently attaches zero adapters and trains
nothing — silently, since PEFT does not raise on an empty target match.

I checked the local converter at `/mmfs1/gscratch/intelligentsystems/evanly/llama.cpp`
(commit `a320cbfc`, 2026-07-16) for requirement 5 directly:

```
"Gemma3ForCausalLM":  "gemma"   ✓
"LlamaForCausalLM":   "llama"   ✓   (SmolLM2 declares this)
"Lfm2ForCausalLM":    "lfm2"    ✓
"Qwen3ForCausalLM":   "qwen"    ✓   (what we run today)
```

All three candidate families convert. No llama.cpp upgrade needed.

### 1.3 The one real blocker — and it is small

This is the finding that makes the whole thing cheap. GGUF scoring has two paths:

```960:986:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/training/slm_helpers.py
    if not supports_mode_kwargs and not _is_qwen_model_id(base_model):
        raise RuntimeError(
            "Installed llama-cpp-python cannot enforce non-thinking chat-template "
            f"kwargs for {base_model or 'an unspecified model'}; refusing to mix "
            "prompt modes."
        )

    def _score_one(worker_llama, prompt: str, index: int) -> str:
        if supports_mode_kwargs:
            ...
            resp = worker_llama.create_chat_completion(...)
            return resp["choices"][0]["message"]["content"]
        rendered_prompt = _qwen_no_think_prompt(prompt, base_model)
```

`supports_mode_kwargs` is computed by introspecting `create_chat_completion` for a
`chat_template_kwargs` parameter. **I checked the installed wheel: llama-cpp-python 0.3.34 does not
have it**, in both `.venv` and `.venv_gpu`. So `supports_mode_kwargs` is always `False` today, the
code always takes the fallback branch, and the fallback is this:

```59:67:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/training/slm_helpers.py
    prefix = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if base_model == "Qwen/Qwen3-4B-Instruct-2507":
        return prefix
    return prefix + "<think>\n\n</think>\n\n"
```

A hardcoded Qwen ChatML string. Every quantized eval in the project has gone through it. The guard
at line 960 exists precisely so a Gemma model does not get silently served ChatML — which is the
right instinct, and it is why nothing has broken.

**The fix is to render the fallback from the base model's own tokenizer** rather than from a
hardcoded string: load the tokenizer for `base_model`, call `apply_chat_template(...,
add_generation_prompt=True)`, and use that as the completion prefix. Qwen keeps producing exactly the
bytes it produces now; Gemma gets `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`;
SmolLM2 gets its own. The Qwen special-cases become one branch of a general rule instead of the rule.

Three smaller items ride along, all in `training/lora_trainer.py`:

- `_pin_serving_chat_template` (line 451) replaces Unsloth's mirror template with the official one,
  Qwen-only. Non-Qwen models loaded from `unsloth/*` mirrors need the same treatment or they inherit
  the same class of B290 skew.
- `_assert_train_serve_prefix_alignment` (line 505) is gated on Qwen. This check should become
  **universal**, not extended — it is the thing that catches train/serve skew, and a new family is
  exactly when you want it running.
- The `"qwen" in name_or_path` hard-fail on a missing chat template (line 318) should become
  family-agnostic: *any* model without a chat template should fail loudly.

Net: one real change, three generalizations of existing Qwen-shaped guards. No new dependencies, no
new venv, no llama.cpp rebuild.

### 1.4 The candidates

Everything below is decoder-only, Unsloth-supported, GGUF-convertible by the local checkout, and
uses the standard projection names.

Sizes below are the **published upstream GGUF file sizes** from the `unsloth/*-GGUF` repos, not
arithmetic from parameter counts — so they are directly comparable to the `size_mb` the pool already
stores. (A pool sweep would still re-measure them locally before any run relies on them.)

| Model | Params | Q4_K_M | Q8_0 | Arch → converter | Unsloth | Why it is on the list |
|---|---|---|---|---|---|---|
| **`google/gemma-3-270m-it`** | **270M** | **253 MB** | 292 MB | `Gemma3ForCausalLM` → gemma | dedicated notebook + GGUF repo | Built for task-specific fine-tuning; 256k vocab; QAT INT4 checkpoints |
| **`HuggingFaceTB/SmolLM2-360M-Instruct`** | **362M** | 271 MB | 386 MB | `LlamaForCausalLM` → llama | catalog: GGUF + 4-bit | Plain Llama architecture — the most boring, best-tested path in the stack |
| **`HuggingFaceTB/SmolLM2-135M-Instruct`** | **135M** | **105 MB** | 145 MB | `LlamaForCausalLM` → llama | catalog: GGUF + 4-bit | The floor — 4.4× below anything the project has run |
| `Qwen/Qwen2.5-0.5B-Instruct` | 494M | ~400 MB | — | `Qwen2ForCausalLM` → qwen | native | The control (see below) |
| `google/functiongemma-270m-it` | 270M | ~250 MB | — | `Gemma3ForCausalLM` → gemma | GGUF repo | Pre-trained *for function calling*. Targeted, not general — see 1.6 |
| ~~`LiquidAI/LFM2.5-350M`~~ | 350M | — | — | `Lfm2ForCausalLM` → lfm2 | GGUF only | **Skip.** Hybrid conv/attention blocks; highest risk of silently matching zero LoRA targets |

For scale, the current floor is `Qwen/Qwen3-0.6B@Q4_K_M` at a **measured 462 MB**:

```
SmolLM2-135M     105 MB   4.4x smaller than today's floor
Gemma-3-270M     253 MB   1.8x
SmolLM2-360M     271 MB   1.7x
Qwen2.5-0.5B    ~400 MB   1.2x
Qwen3-0.6B       462 MB   ← today's floor
```

One quirk worth recording, because it will look like a bug in the pool table otherwise: Gemma 3
270M's Q8_0 is only **292 MB against a 253 MB Q4_K_M**, a 1.15× ratio where the pool's arithmetic
assumes 1.82×. That is the 256k-vocabulary embedding table — it dominates the file and does not
compress the way transformer weights do. `_variant()`'s synthetic sizing would be badly wrong for
this model, so its entries must carry **measured** sizes in `config/measured_metrics.json` rather
than seeded ones.

No tier change is needed. Tier 0 is `< 750 MB` and `select_smallest` sorts on `size_mb`, so a 253 MB
entry is picked ahead of the 462 MB one automatically. The tier *labels* just get coarser — five
models would share tier 0 — which is a reporting nit, not a mechanism problem.

### 1.5 The recommendation

**Add three, in this order: Gemma 3 270M IT, SmolLM2-360M-Instruct, SmolLM2-135M-Instruct.**

The reasoning is about what each one buys that the others do not, and I want the second pick to be
justified on its own terms rather than as padding:

**Gemma 3 270M** is the strongest candidate on the merits. Google shipped it explicitly as a
fine-tuning base rather than a chat model, and the architecture reflects that — 170M of its 270M
parameters are the embedding table over a 256k vocabulary, leaving only 100M in the transformer
blocks. That allocation is unusually well suited to what this suite actually tests: four of our five
tasks are format-bound or structured, where the work is emitting exact tokens (`Chemical`,
`accept_reservations`, an ISO-8601 instant) rather than reasoning. A model that is mostly vocabulary
is a model that is mostly good at exactly that. It also has QAT INT4 checkpoints, which is relevant
because `SLM_QUANT_EVAL=1` means our scored artifact is the quantized one.

**SmolLM2** earns its place for the opposite reason — it is architecturally boring. It declares
`LlamaForCausalLM`, which is the single most exercised path in Unsloth, llama.cpp and PEFT. If Gemma
hits an integration snag, SmolLM2 is the control that tells you whether the snag is Gemma or the
pipeline. And the 135M gives a genuine floor: at ~100 MB it is 4.6× below anything the project has
ever run, which is where "how small can this go" stops being rhetorical.

**`Qwen2.5-0.5B-Instruct` is worth adding if and only if you want the family confound removed.**
It is barely smaller than Qwen3-0.6B, so it buys nothing on size. What it buys is an answer to the
question the results will otherwise raise: when Gemma-270M underperforms Qwen3-0.6B, is that size or
is that family? A same-family point at a similar size settles it. My inclination is to skip it in the
first pass and add it only if the size/family question actually becomes contested.

### 1.6 FunctionGemma, and why I would run it separately

`google/functiongemma-270m-it` is Gemma 3 270M continued-trained for function calling, and its
numbers are directly relevant to two of our tasks:

| Benchmark | 0-shot |
|---|---|
| BFCL Simple | 61.6 |
| BFCL Multiple | 63.5 |
| BFCL Parallel | 39.0 |

Google's own fine-tuning result on their Mobile Actions set is **58% → 85%**.

For context, `xlam_bfcl` measures `ast_arg_match` on the BFCL AST categories, and our best-ever score
on that task is **0.8600** from a fine-tuned Qwen3-**4B**. A 270M model starting at ~0.62 zero-shot is
a genuinely interesting starting point, 8.8× smaller.

I would not put it in the general pool, though. Its model card says it "has the same architecture as
Gemma 3, but uses a **different chat format**" — control tokens for declarations and calls, which is
why the BFCL leaderboard needs a custom handler for it. Dropping that into a pool where every task
shares one prompt builder is how you get a B290-class skew bug. It belongs as a **pinned
`single_model` run on `xlam_bfcl` and `calendar_json`**, where the prompt contract can be checked
once against that one model.

### 1.7 The honest counterweight

I went looking for evidence against this plan and found some. Nemotron-Research-Tool-N1 swept model
sizes on BFCL and reports:

> "performance improvements from post-training are limited for smaller models (0.5B and 1.5B),
> whereas larger models exhibit substantial gains"

That is a direct warning about the sub-billion regime **on the function-calling task specifically**,
which is where our best evidence for large gains also lives (§3.1). Both can be true — TinyAgent's
+66 points came from a narrow, heavily curated single-application dataset, while Tool-N1's sweep is
over general BFCL — and the difference between those two setups is roughly the difference between
what our pipeline does and what a general tool-use benchmark measures. But it should temper the
expectation: **a 270M model on xlam is the experiment most likely to produce a flat line**, and
`ner_bc5cdr` or `clinc150` are the safer first tests of the small-model path.

---

## Part II — Where the five tasks actually stand

### 2.1 Spam classification does not exist

Checked the registry, the loaders, the slurm directory and the whole of `logs/slurm/`. There is:

- no entry in `TASKS` (`tasks/__init__.py` registers eight modules; spam is not one)
- no loader — `data/loaders/sms_spam.py` was **deleted in commit `387f4ad`**, the 2026-08-18
  registry rebuild, and only a stale `__pycache__/sms_spam.cpython-311.pyc` remains
- no slurm script
- no run, ever, under any job name

`PIPELINE.md` records the deletion explicitly:

> Also deleted on 2026-08-18 … `sms_spam`, `fpb`, `arc`, `multilingual`, `structured_extraction`

So the answer to "is the spam harness up to date" is that there is no harness to update. What does
still exist is a **claim**, in `PAPER.md`:

```
| SMS Spam | GLiNER2-base | F1: 0.159 | 0.997 | +83.8 | 10 |
```

And §15.3 of that same document already dismantles it:

> GLiNER2 is a span-extraction NER model. Applied in NER mode to a binary spam classification
> task, it attempts to extract entity spans rather than classify documents — producing near-zero
> recall by design. … Even a simple logistic regression over TF-IDF achieves F1>0.95 on SMS Spam
> (UCI) out of the box.

That is the right verdict and §3.4 below independently confirms it from the outside literature. The
`+83.8` is a broken baseline, not a result. **The paper should stop quoting it**, which is a doc
change I am flagging rather than making unilaterally since it removes a headline number.

### 2.2 Do the datasets load? Yes — measured today

I ran `scripts/preflight_tasks.py` just now rather than quoting the last note:

```
PASS  xlam_bfcl      train= 3250  eval=1000  ast_arg_match  gold=   1.0  degen=   0.0
PASS  calendar_json  train= 3250  eval= 535  ast_arg_match  gold=   1.0  degen=   0.0
PASS  ner_bc5cdr     train= 3250  eval=1000       span_f1    gold=   1.0  degen=   0.0
PASS  clinc150       train= 3250  eval=1000      macro_f1    gold=   1.0  degen=0.0001

  4/4 passed
```

All four load, gold scores 1.0 through the real scorer, a degenerate answer scores ~0, and the
training prompt is byte-identical to the eval prompt. Two caveats it surfaced:

- **calendar_json's eval set is 535/1000 rows** (54% of target). Accepted and logged, but its scores
  carry more variance than the others' and are not directly comparable to them.
- **clinc150 ships 1 train row that appears verbatim in eval** at the loader level. `curate`'s
  firewall removes it before training so no run is contaminated, but the loader should not emit it.

### 2.3 What has been run, and how it ended

| Task | Best run | Outcome | Score vs goal | Model | Iters | Wall |
|---|---|---|---|---|---|---|
| `ner_bc5cdr` | 38455148 | **CONVERGED** | **0.8098** vs 0.8000 ✓ | Qwen3-0.6B@Q4_K_M (tier 0) | 5 | 1.35 h |
| `clinc150` | 38180646 | **CONVERGED** | **0.8952** vs 0.8919 ✓ | Qwen3-0.6B@Q4_K_M (tier 0) | 4 | 1.11 h |
| `calendar_json` | 38735780 | converged at floor, then SIGTERM | 0.8673 vs 0.8800 stretch ✗; **0.8430 vs 0.8000 floor ✓** | 0.6B → Qwen3-1.7B@Q4_K_M (tier 1) | 38 | 7.44 h |
| `xlam_bfcl` | 38708719 | budget exhausted | 0.8600 vs 0.8660 ✗ | Qwen3-4B-Instruct@Q4_K_M (tier 2) | 10 | 5.50 h |
| spam | — | **never existed** | — | — | — | — |

Nothing is queued right now (`squeue` is empty). Both of the recent runs died in the same
2026-08-21 16:54 window to cluster SIGTERM, not to a defect.

**What went wrong per task, and what is still open:**

**`xlam_bfcl`** — covered in full in `08-21-run-38708719-review.md`. It stopped 0.006 short of goal
with the eval cap set to 10, still improving. That cap is back to 30. The unresolved substance is the
failure mix: ~84% of failures are `wrong_arguments→incorrect` and the hard difficulty bucket sat at
0.45–0.49 through every intervention. Open bugs: **B253** (loader/BFCL file resolution), **B297**
(no mining alias for the local xLAM corpus, so `acquire` pays Exa to rediscover a cached dataset).

**`calendar_json`** — the more interesting failure, and it is a *policy* failure rather than a data
one. Reading the final ledger:

```
  What each strategy actually bought (kept steps only):
    hyperparameter          +0.0243   2/18 attempt(s) kept
    surgical_synthesis       0.8430   starting point (1/1 kept)
    FINAL                    0.8673   = 0.8430 start +0.0243 from interventions

  teacher fitness: ast_arg_match=0.5100 5-shot on 200 eval row(s) vs a 0.80 gate
                   → synthetic data REFUSED
```

**Seventeen of nineteen tier-1 iterations were hyperparameter tuning**, and two of them were kept.
That is not the orchestrator being stubborn — it is the orchestrator having one lever. Synthesis was
refused by the fitness gate (teacher 0.51 against a 0.80 threshold) and mining was exhausted (`2 of 4
rebuild(s) added no rows`, and the tier-0 phase logged `✗ ERROR: data_rebuild/mine_new_real added 0
new rows`). With `data_rebuild` dead by both routes, the loop degenerates into a hyperparameter grid
search that burns a full train+eval cycle per point.

This is the single most actionable finding in the whole audit and it is not a calendar-specific bug:
**any task where the teacher scores below 0.80 and the mining pool is finite will collapse to the
same behaviour.** Open bugs: **B262**, **B297**, **B303** (`TOPv2/reminder` mining source never
verified — likely why mining returns nothing), **B275**.

**`ner_bc5cdr`** — the reference result and still the cleanest thing in the suite: 0.0000 baseline →
0.7701 first fine-tune → 0.8098 converged, on the smallest model in the pool, in 81 minutes. Open:
**B263** (NER eval display prints a blank `gold:`, cosmetic but it hides a sanity check). One
historical crash mode worth remembering — run 38569606 on the CSE partition died on a GGUF that
decoded to `////////////////////////////////`, the same corruption that killed 38569608.

**`clinc150`** — converged, and it is a control rather than a result. The teacher scores 0.8919
zero-shot and the fine-tuned 0.6B scores 0.8952. **A +0.003 gain.** §3.3 explains why that is exactly
what the literature predicts. Open: **B224** (orchestrator replays stale hypotheses), **B225**
(dataset versioning overwrites intermediates).

### 2.4 The harness audit — better than I expected, with one real gap

I diffed all the slurm scripts. The four `_l40s` launchers are **structurally identical**: same
`#SBATCH` block (2× L40S, 16 CPU, 160G, 7-day, `--requeue`, `--signal=B:USR1@7200`), same
`SLM_CUDA_ISOLATION=1`, same `SLM_BENCHMARK_TASK`, same `source _l40s_task_body.sh`. Only the job
name, the log path and the `TASK` prose differ. The `_cse` variants differ from their `_l40s` twins
only in `--account` and `--time`. That is genuinely uniform, and every behavioural knob — model
selection, escalation, eval caps, orchestrator, quantized eval — lives once in the shared body:

```
export SLM_MODEL_SELECTION_STRATEGY="${SLM_MODEL_SELECTION_STRATEGY:-smallest_first}"
export SLM_QUANT_EVAL=1
export SLM_MAX_WALLCLOCK_S="${SLM_MAX_WALLCLOCK_S:-0}"
```

Same for the interventions and escalation policy: there is **no per-task override anywhere**. No
task spec carries a threshold, an eval cap or a tier pin; `STAGNATION_WINDOW=15`,
`STAGNATION_MIN_DELTA=0.02` and `MAX_EVALS_BEFORE_ESCALATION=30` are global. I grepped for
`if task ==` across `agent/`, `eval/` and `training/` and found none for any registered task — the
2026-08-18 registry rebuild really did remove them.

So the answer to "are they running identically" is **yes at the code level**. The divergences are all
at the launcher level, and there are three:

| # | Divergence | Consequence |
|---|---|---|
| 1 | **`xlam_bfcl` has two extra `single_verify` launchers** that set `SLM_MODEL_SELECTION_STRATEGY=single_model` | **The important one.** xlam's last four runs pinned one model with *no escalation and no downward probe*. calendar/ner/clinc ran `smallest_first` *with* both. These results are not comparable. |
| 2 | **`clinc150` has no `_cse` launcher** | Can only run on the `intelligentsystems` quota; the other three can use either. Pure availability loss. (`dialogsum` and `gsm8k` are missing theirs too.) |
| 3 | `single_verify` also sets `SLM_MAX_WALLCLOCK_S=21600`, `--time=08:00:00`, and drops `--requeue`/`--signal` | Deliberate (a verification run should stay dead), but it means xlam's runs are time-boxed where the others are not. |

Divergence 1 is what you were asking about, and it is real. `single_verify`'s own header says its
purpose is finished:

> That verification is done, and 10 turned out to be the binding constraint rather than a safety net

Its `SLM_MAX_EVALS_BEFORE_ESCALATION` is already back to the project default of 30, so the only thing
still making xlam different is the pinned strategy.

---

## Part III — Does the literature support fine-tuning gains on these tasks?

You asked for papers reporting 20–80% improvements from fine-tuning, preferably at ≤4B and at worst
7B. Here is what I found, per task, with the caveats where they exist.

### 3.1 `xlam_bfcl` — YES, and it is the strongest case in the suite

**TinyAgent** (Berkeley SqueezeAILab, [arXiv:2409.00608](https://arxiv.org/abs/2409.00608)) is the
paper you want. It fine-tunes **TinyLlama-1.1B** — under our 4B bar — with LoRA on curated
function-calling data:

| Model | Before | After |
|---|---|---|
| TinyLlama-1.1B | **12.71%** | **78.89%** (80.06% with ToolRAG) |
| WizardLM-2-7B | 41.25% | 83.09% |
| GPT-4-Turbo | — | 79.08% |

**+66.2 points absolute, 6.2×, on a 1.1B model — which then beats GPT-4-Turbo.** That is at the top
of your requested range and then some.

Second, and closer to home: **APIGen**
([arXiv:2406.18518](https://arxiv.org/abs/2406.18518)) is *the paper that produced xLAM-60k*, the
exact dataset `xlam_bfcl` trains on:

> "models trained with our curated datasets, even with only 7B parameters, can achieve
> state-of-the-art performance on the Berkeley Function-Calling Benchmark, outperforming multiple
> GPT-4 models. Moreover, our **1B model** achieves exceptional performance, surpassing
> GPT-3.5-Turbo and Claude-3 Haiku."

So both our training corpus and our eval benchmark come from a line of work whose central claim is
that sub-billion-to-7B models fine-tuned on this data beat frontier models. **Keep this task.**

The counterweight from §1.7 applies: Tool-N1 reports muted post-training gains specifically at 0.5B
and 1.5B on general BFCL. TinyAgent's setting is narrow and curated; ours is closer to that than to
general BFCL, but the risk is real.

### 3.2 `ner_bc5cdr` — YES, on our exact dataset, at our exact scale

*Harnessing Large Language Models for Biomedical NER*
([arXiv:2512.22738](https://arxiv.org/abs/2512.22738)) benchmarks BC5CDR directly, strict-match F1:

| Model | BC5CDR-Chemical | BC5CDR-Disease |
|---|---|---|
| GPT-4 (not fine-tuned) | 83.70 | **67.80** |
| UniNER-7B | 88.82 | 80.09 |
| BioMedBERT | **93.33** | **85.62** |
| Qwen3-8B-SFT | 91.58 | 86.11 |
| **Qwen3-4B-SFT** | **91.71** | — |

A fine-tuned **Qwen3-4B** — same family, same size class as our pool ceiling — reaches 91.71 on
BC5CDR-Chemical against GPT-4's 83.70. And the gap on the *disease* half is 67.80 → 85.62, **+17.8
points over GPT-4**.

The cleaner before/after comes from OpenMed's GLiNER fine-tunes on BC5CDR-Disease, which report base
and fine-tuned for the same model:

| Model | Base F1 | Fine-tuned F1 | ΔF1 % |
|---|---|---|---|
| 209M | 0.5721 | 0.8848 | **+54.7%** |
| 459M | 0.5890 | 0.9029 | **+53.3%** |

**+53% relative at 209M parameters** — squarely inside your 20–80% band, and at a parameter count
below every model we are considering adding. **Keep this task; it is also the best first target for
the sub-billion models.**

Our own number sits comfortably in this company: 0.0000 zero-shot → 0.8098 fine-tuned on a 0.6B.

### 3.3 `clinc150` — NO. The task has no headroom, and this is a measurement, not an opinion

The literature is consistent, and it says CLINC150 is *easy*:

| Setting | Accuracy |
|---|---|
| BERT-base, full 15k train | **96.31%** |
| BERT-base, 50-shot | 95.67% |
| BERT-base, **10-shot** | **90.38%** |

A 110M encoder reaches 90% from **ten examples per intent**. CLINC150 was constructed as a clean,
balanced, well-separated intent set, and that is exactly what makes it a bad fine-tuning benchmark —
there is nowhere to improve *to*.

Our own run agrees precisely: teacher zero-shot **0.8919** → fine-tuned 0.6B **0.8952**, a gain of
**+0.003**. That is not a pipeline failure. It is the correct answer for this dataset, and the
`08-17` note already recategorised it as a control for that reason.

The general finding that fine-tuned small models beat zero-shot LLMs on classification *is* well
supported — Bucher & Martini's four-application study concludes "fine-tuning smaller LLMs with
dedicated training data is consistently superior to zero-shot prompting larger models," and an
ASRJETS comparison puts the gap at "~10-25 points in accuracy, with a larger gap on the fine-grained
task." But that gap has to exist in the dataset for us to measure it, and on CLINC150 it does not.

**Recommendation: keep `clinc150` only as the declared control, and add a fine-grained intent task
that actually has headroom.** §4.1.

### 3.4 Spam classification — NO, and adding it would be a mistake

The evidence runs against it from both directions.

Fine-tuning works, but there is nothing to win. SpaLLM-Guard
([arXiv:2501.04985](https://arxiv.org/abs/2501.04985)) finds fine-tuned Mixtral 8×7B at **98.61%**
accuracy; a separate comparative study fine-tunes **DistilBERT to 99%**, matching Phi-3.5 and
H2O-Danube. A 66M encoder saturates the task.

And the zero-shot floor is high. GPT-5 zero-shot with adaptive thresholding reaches **97.00%
accuracy / 96.98% F1**. Flan-T5 zero-shot gets 90% F1 on SpamAssassin.

So the realistic delta is **97% → 99%**, roughly two points. Compare that to the +66 points TinyAgent
reports on function calling. Adding spam would cost a loader, a task module, two slurm scripts and a
GPU run, to measure a two-point gain on a task a logistic regression over TF-IDF already scores >0.95
on — which is the same objection `PAPER.md` §15.3 already raises against the existing spam claim.

**Recommendation: do not add it. Retire the `PAPER.md` claim instead.**

### 3.5 `calendar_json` — PARTIAL. Good evidence by analogy, weak evidence directly

This one I want to be careful about, because it is easy to overstate.

The direct literature on TOPv2 and SGD is about **low-resource transfer**, not zero-shot vs
fine-tuned. Two credible results:

- *Low-Resource Task-Oriented Semantic Parsing via Intrinsic Modeling*
  ([arXiv:2104.07224](https://arxiv.org/abs/2104.07224)): "+15 EM absolute (44% relative) when
  fine-tuning on 10 samples from an unseen domain" on a TOPv2-derived benchmark.
- *The Power of Prompt Tuning for Low-Resource Semantic Parsing* (ACL 2022): "On TOPv2, prompt tuning
  achieves an absolute improvement of 15% mean accuracy over fine-tuning on the lowest SPIS split."

Both are +15-ish and both compare *training method A against training method B*, not
*untrained against trained*. They support "small models can learn this task sample-efficiently."
They do not directly support "fine-tuning yields a 20–80% gain over zero-shot."

**Our own run is the better evidence, and it is strong:** teacher Qwen3.6-35B scores **0.2579**
zero-shot; the fine-tuned Qwen3-1.7B reaches **0.8673**. That is **+0.61 absolute, 3.4×** — a larger
gain than anything in §3.1 or §3.2, produced by our own pipeline on a 1.7B model. It is unpublished
and it is n=1, but it is a real measurement on our real eval set.

**Recommendation: keep it, and cite our own number rather than reaching for TOPv2 papers that
measure something adjacent.** The blocker is §2.3's finding, not the task's merit: fix the data
intervention or the run degenerates into hyperparameter search again.

### 3.6 Verdict

| Task | Published evidence at ≤7B? | Gain reported | Keep? |
|---|---|---|---|
| `xlam_bfcl` | **Yes** — TinyAgent 1.1B, APIGen 1B/7B | **+66 pts / 6.2×** | **Keep** |
| `ner_bc5cdr` | **Yes** — Qwen3-4B-SFT, GLiNER 209M | **+53% rel** | **Keep — and test small models here first** |
| `calendar_json` | Partial — TOPv2 low-resource only | +15 EM (different comparison) | Keep, cite our own 0.2579→0.8673 |
| `clinc150` | Yes, but the ceiling is 96% from a 110M encoder | our own: **+0.003** | Keep as declared control only |
| spam | Yes, but 97% zero-shot → 99% fine-tuned | ~2 pts | **Do not add** |

---

## Part IV — Replacements where the evidence is strong

Two of the five tasks are weak. These are the best-evidenced swaps, both of which reuse existing
machinery rather than needing new scorers.

### 4.1 Banking77 — replaces `clinc150` as the classification task

Same shape as CLINC150: N-way intent classification over short utterances, so it drops straight onto
`classification_turn`, `eval.scorers.classification`, `macro_f1` and `label_balanced` sampling. The
task module would be a near-copy of `tasks/clinc150.py` with a different loader and `mteb/banking77`
as the mining source.

The difference is headroom. Banking77's 77 intents are *deliberately* fine-grained and confusable
(`card_arrival` vs `card_delivery_estimate` vs `card_not_working`), and the numbers reflect it:

| Model | Zero-shot | Fine-tuned |
|---|---|---|
| **Llama-3.2-1B-Instruct** (QLoRA, single T4, ~50 min) | **0.00%** | **90.21%** |
| Qwen3-8B (QLoRA, 1 epoch) | 68.00% | 91.85% |
| Mistral-7B (QLoRA) | — | 94.6 µF1 |

The 1B result is the striking one — 0/3076 zero-shot because the base model cannot produce a valid
label string at all, then 2775/3076 after fine-tuning. That is the format-plus-knowledge gain this
project exists to measure, on a 1B model, in under an hour on one T4.

It also gives a direct read on §1.7's open question, since the same benchmark has a published 1B
number to compare our sub-billion models against.

### 4.2 Text-to-SQL — the strongest sub-billion evidence anywhere

If you want one more task and are willing to add a scorer, this is where the evidence is best for
*exactly* the model sizes we are about to add.

**SLM-SQL** (Findings of IJCNLP 2025) fine-tunes five models from **0.5B to 1.5B**:

> "On the BIRD development set, the five models achieved an average improvement of **31.4 points**,
> with the **0.5B model reaching 56.87%** execution accuracy and the 1.5B model achieving 67.08% …
> The 0.5B model achieved **73.50%** EX [on Spider], while the 1.5B model reached 79.06%."

**FINER-SQL** ([arXiv:2605.03465](https://arxiv.org/abs/2605.03465)) gets a 3B model to 67.73% on
BIRD and 85.0% on Spider, "surpassing CodeS-15B (58.47%) … and Reasoning-SQL 14B (65.31%)."

A +31.4-point average gain at 0.5B–1.5B is the single best-matched result to the sub-billion pool
expansion. The cost is honest: it needs an execution-accuracy scorer (run the predicted SQL against
the database and compare result sets), which is a new scorer family and a sandboxing question — and
the project deleted its last execution sandbox on 2026-08-18. So this is a *bigger* piece of work
than Banking77, and I would not do both at once.

### 4.3 Ranking

1. **Banking77** — highest value per unit of work. Reuses every existing component, published 1B
   before/after numbers, direct replacement for the one task with no headroom.
2. **FunctionGemma-270M pinned to `xlam_bfcl`** — not a new task, a new run. Zero new code beyond
   §1.3, and it tests the most interesting hypothesis in this note.
3. **Text-to-SQL** — best evidence, most work. Worth it if the suite needs a fifth strong task.

---

## Part V — What I propose to change

Nothing in this section is done yet. Splitting it out because two of these are decisions rather than
fixes.

**Code — I would just do these:**

| # | Change | Where |
|---|---|---|
| 1 | Render the GGUF fallback prompt from the base model's own chat template instead of hardcoded Qwen ChatML; delete the non-Qwen refusal | `training/slm_helpers.py:960`, `:43` |
| 2 | Make `_assert_train_serve_prefix_alignment` universal rather than Qwen-gated | `training/lora_trainer.py:505` |
| 3 | Make the missing-chat-template hard-fail family-agnostic | `training/lora_trainer.py:318` |
| 4 | Generalize `_pin_serving_chat_template` beyond Qwen mirrors | `training/lora_trainer.py:451` |
| 5 | Add `tests/pipeline/run_clinc150_cse.slurm` — the only one of these four without a CSE twin (`dialogsum` and `gsm8k` also lack one) | `tests/pipeline/` |
| 6 | Fix the 1 clinc150 loader-level train/eval overlap row | `data/loaders/clinc150.py` |
| 7 | Verify or replace the `TOPv2/reminder` mining source (B303) — likely why calendar mining returns nothing | `tasks/calendar_json.py` |

**Decisions I want from you before touching:**

- **The three pool entries.** Adding non-Qwen models breaks the "official Qwen only, for a controlled
  ablation" policy that `docs/model_pool.md` states. That policy exists for a reason and I do not
  want to quietly drop it. Adding them also means a `hardware_eval/measure_pool.py` sweep for real
  `size_mb` values before any run uses them.
- **`xlam_bfcl`'s escalation policy.** Retire the `single_verify` launchers so xlam runs
  `smallest_first` like the other three, or keep `single_model` and add matching pinned launchers for
  the others? Right now the five-task matrix cannot be compared, and either direction fixes it.
- **`clinc150` and spam.** Swap clinc150 for Banking77, or keep it as the explicitly-labelled control
  and add Banking77 alongside? And confirm we are dropping spam rather than building it.
- **The `PAPER.md` spam row.** §15.3 already says the +83.8 is a broken baseline. Remove the row, or
  keep it with the caveat inline?

**Docs I will update once the above lands:** `model_pool.md` (new entries + the family policy),
`PIPELINE.md` (the non-Qwen inference path), `interventions.md` (the `SURGICAL_MAX_CATEGORIES` 5-vs-8
mismatch I found between `data_rebuild.py:63` and `curate.py:435`), `BUGS.md` (B303 and the clinc150
overlap row), and `PAPER.md` (the spam claim).

---

## Questions

1. **Add Gemma 3 270M, SmolLM2-360M and SmolLM2-135M to the pool?** The Qwen-only policy is the thing
   I am actually asking about — the engineering is one small function.
2. **Which way on xlam's escalation policy?** It is the only thing making the five-task comparison
   invalid.
3. **Banking77 as a replacement for clinc150, or alongside it?**
4. **Confirm spam is dropped**, and say whether the `PAPER.md` row comes out.
5. **Where should the first sub-billion run go?** My vote is `ner_bc5cdr` — it converges in 81
   minutes, it has the cleanest signal in the suite (0.0000 → 0.8098), and §3.2 gives a published
   209M reference point to compare against. `xlam_bfcl` is the more interesting question and the more
   likely flat line (§1.7).
