# BUGS.md — running bug log

A living record of bugs found while building/validating the SLM Factory pipeline.
Newest entries appended at the bottom. Each entry notes **where** it lives, **when** it
was discovered, **how** it was found, and current **status**.

Status legend: 🔴 open · 🟢 fixed · 🟡 suspected/unconfirmed · ⚪ design gap (not a crash)

| ID | Status | Area | One-line |
|----|--------|------|----------|
| B1 | 🟢 | eval_setup | classification path hardcoded SMS-spam for every task |
| B2 | ⚪ | orchestration | iteration loop is deterministic Python, not an LLM per-iteration (design says LLM) |
| B3 | 🟢 | curation log | dataset composition logged as 0 (placeholders) |
| B4 | 🔴 | train | checkpoint output dirs named `iter{N-1}` (off-by-one) |
| B5 | 🔴 | rollback | docstring claims it restores weights; it only pops the score |
| B6 | 🟢 | eval_set/scorer | binary-only; multi-class collapsed to 2 labels |
| B7 | 🟢 | web_acquire | Exa `search()` rejects `text=` kwarg |
| B8 | 🟢 | eval_set | multi-class eval degenerated to `pos=1/neg=0` |
| B9 | 🟢 | eval_setup | eval set never persisted to disk |
| B10 | 🟢 | run_autonomous | `_Tee` flush-after-close `ValueError` at exit |
| B11 | 🟢 | lora_trainer | train prompt ≠ eval prompt (no train/serve parity) |
| B12 | 🟢 | runner | hardware constraints hardcoded; no autonomous device research |
| B13 | 🔴 | curriculum | Dgold:Dhard ≈ 98:2, far from the paper's 65:35 target |
| B14 | 🟢 | lora_trainer | `target_modules="all-linear"` string → PEFT char-iteration crash |
| B15 | 🟢 | slm_helpers | inference used vanilla `AutoModelForCausalLM`; Unsloth's global patches break it |
| B16 | 🟢 | run_autonomous | `_Tee` stdout missing `isatty()` → transformers loading report crashes |
| B17 | 🟢 | lora_trainer | mid-training checkpoint save can't pickle Unsloth-patched `SFTConfig` |
| B18 | 🟢 | train | two trainers in parallel threads → Accelerate `device_map='auto'` distributed-mode error |

---

## B1 — eval_setup hardcoded SMS-spam for any classification task
- **Where:** `agent/nodes/eval_setup.py`
- **When:** 2026-06-23, during initial code read-through
- **How found:** reading the node; `task_type=="classification"` always called `download_sms_spam()`.
- **Impact:** any classification task trained/evaluated on SMS spam regardless of the request.
- **Status:** 🟢 fixed 2026-06-24 — routes through `data/loaders/web_acquire.py` when a `task_plan` exists.

## B2 — orchestration loop is deterministic, not LLM-driven per iteration
- **Where:** `agent/nodes/iterate.py`, `rollback.py`, `escalate.py` (the graph routing)
- **When:** 2026-06-23, code review + first instrumented run
- **How found:** the design (paper §2.2 `EXPAND` via the orchestrator LLM) calls for an LLM
  reasoning step each iteration; the implementation uses hardcoded score-band rules. The only
  LLM calls are the planner (once) and hard-negative synthesis (in `curate`).
- **Impact:** behavior diverges from the paper; "if it ran longer" adds only data-gen calls,
  not orchestrator-reasoning calls.
- **Status:** ⚪ design gap — partially addressed (added an LLM task planner + Exa acquisition).

## B3 — curation log records dataset composition as 0
- **Where:** `agent/nodes/evaluate.py` (call to `CurationLog.write_iteration`)
- **When:** 2026-06-23 (noticed), confirmed 2026-06-24 from user question about `data-curation.md`
- **How found:** `data-curation.md` showed `Total examples: 0 / Dgold 0 / Dhard 0 / Distribution {}`
  even though the curriculum on disk was non-empty; the node passed literal `n_gold=0, n_hard=0`.
- **Status:** 🟢 fixed 2026-06-24 — `curate_node` stores `state["last_curation"]`; `evaluate_node` logs it.

## B4 — checkpoint output dirs are off-by-one (`iter{N-1}`)
- **Where:** `agent/nodes/train.py` (builds `output_dir` from `state['iteration']` before incrementing)
- **When:** 2026-06-23, first instrumented run
- **How found:** DAG reported `iteration=4` but the weights lived in `artifacts/iter3_A/`.
- **Impact:** cosmetic/confusing lineage; weights selection still correct.
- **Status:** 🔴 open.

## B5 — rollback_node doesn't restore best weights
- **Where:** `agent/nodes/rollback.py`
- **When:** 2026-06-23, code review
- **How found:** docstring says "Restores previous weights_ref" but the code only pops the last
  score and bumps a counter; `best_weights_ref` is untouched.
- **Impact:** works today only because `best_weights_ref` is managed in `evaluate_node`; the
  docstring is misleading and the intent is unimplemented.
- **Status:** 🔴 open.

## B6 — eval set + scorer were binary-only
- **Where:** `data/eval_set.py`, `eval/scorers/classification.py`
- **When:** 2026-06-23 (identified), 2026-06-24 (bit hard on multi-class)
- **How found:** classification path used `_infer_pos_label`/`_infer_neg_label` (binary) and
  `binary_f1`; with >2 classes the eval set covered only 2 labels.
- **Status:** 🟢 fixed 2026-06-24 — multi-class stratified slices + macro-F1; binary path unchanged.

## B7 — Exa `search()` rejects the `text=` kwarg
- **Where:** `data/loaders/research_papers.py`, `data/loaders/web_acquire.py`, `agent/hardware_research.py`
- **When:** 2026-06-24 ~17:00, first autonomous research-paper run
- **How found:** run crashed: `Exa.search() got an unexpected keyword argument 'text'`; the new
  `search()` returns text by default and doesn't accept `text=`.
- **Status:** 🟢 fixed 2026-06-24 — switched to `search_and_contents(...)`.

## B8 — multi-class eval degenerated to `pos=1/neg=0/boundary=6`
- **Where:** `data/eval_set.py`
- **When:** 2026-06-24 ~16:48, first full autonomous run
- **How found:** log showed `eval_set pos=1/neg=0` for a 6-class task; F1 quantized to 0.4/0.5.
- **Status:** 🟢 fixed 2026-06-24 (same change as B6).

## B9 — held-out eval set never persisted to disk
- **Where:** `agent/nodes/eval_setup.py`
- **When:** 2026-06-24, user question 2
- **How found:** eval set existed only in `state["eval_set"]`; no artifact to inspect.
- **Status:** 🟢 fixed 2026-06-24 — writes `artifacts/eval_set.json`.

## B10 — `_Tee` flush after close throws at interpreter exit
- **Where:** `run_autonomous.py` (`_Tee`)
- **When:** 2026-06-24, first autonomous run (exit code 120)
- **How found:** `Exception ignored in: _Tee ... ValueError: I/O operation on closed file` after
  the run printed its summary.
- **Status:** 🟢 fixed 2026-06-24 — guard `closed` streams; restore `sys.stdout` before close.

## B11 — training prompt ≠ evaluation prompt (no train/serve parity)
- **Where:** `training/lora_trainer.py` vs `eval/scorers/classification.py`
- **When:** 2026-06-24, prepping the real GPU run
- **How found:** trainer formatted `"Task: classification...\nInput:...\nLabel:..."` while eval
  prompted `"Classify this message... Message:..."` — the model would be trained on a different
  format than it is scored on.
- **Status:** 🟢 fixed 2026-06-24 — trainer now reuses `CLASSIFY_PROMPT` + gold label as completion.

## B12 — hardware constraints hardcoded; no autonomous device research
- **Where:** `run_autonomous.py` (previously a hardcoded `HardwareConstraints(...)`)
- **When:** 2026-06-24, user question 3
- **How found:** the Moto G's real chip (Snapdragon 6 Gen 1) was never resolved; constraints were
  my manual guess (and `target_chip` a proxy not in the pool's benchmark columns).
- **Status:** 🟢 fixed 2026-06-24 — `agent/hardware_research.py` (Exa + Sonnet) emits real constraints.

## B13 — Dgold:Dhard far from the paper's 65:35 target
- **Where:** `data/curriculum.py` (`synthesize_hard_negatives`) + `agent/nodes/curate.py`
- **When:** 2026-06-24, after fixing B3 the honest log showed `Dgold 62 / Dhard 1` (≈98:2)
- **How found:** `synthesize_hard_negatives` (classification) only generates hard negatives for the
  *minority/positive* class, so multi-class curricula get almost none.
- **Impact:** the curriculum doesn't follow the paper's composition; hard negatives are the lever
  the paper calls "essential."
- **Status:** 🔴 open.

## B14 — LoRA `target_modules="all-linear"` crashes PEFT
- **Where:** `training/lora_trainer.py` (`FastLanguageModel.get_peft_model`)
- **When:** 2026-06-24 ~21:53, first real GPU run (Slurm job 36429526, Qwen3-0.6B on an L40)
- **How found:** `ValueError: Target modules {'a','-','l','e','i','n','r'} not found in the base
  model.` — PEFT iterated the *string* character-by-character. (Full fine-tune config trained fine;
  only the LoRA config crashed, which then failed the whole `train_node`.)
- **Status:** 🟢 fixed 2026-06-24 — pass a list `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`. Resubmitted as job 36430179.

## B15 — inference path used vanilla loader, incompatible with Unsloth
- **Where:** `training/slm_helpers.py` (`infer` / `infer_batch`)
- **When:** 2026-06-24, predicted after B14; **confirmed ~22:41**, fifth real GPU run (Slurm job
  36434997) — first run to finish training and reach `evaluate`.
- **How found:** real inference crashed with `AttributeError: 'Qwen3Attention' object has no
  attribute 'apply_qkv'`. Once `unsloth` is imported it globally patches the model classes
  (Qwen3 attention's fast path expects `apply_qkv`), so a model loaded with vanilla
  `AutoModelForCausalLM.from_pretrained` has no such attribute and breaks at `generate`. (The
  predicted LoRA-adapter loading problem is the same root: the inference path must use Unsloth.)
- **Status:** 🟢 fixed 2026-06-24 — `infer` loads via `FastLanguageModel.from_pretrained` +
  `for_inference` (also handles LoRA adapter dirs); `infer_batch` made sequential (drops the
  20-thread `ThreadPoolExecutor`, same global-state hazard as B18).
- **Confirmed:** 2026-06-24 ~22:55, run 36436130 — **first fully real end-to-end run** completed:
  Qwen3-0.6B, real Unsloth train (363s) + real inference eval → **F1=0.553** on the 6-class
  research-paper task. (Stopped after 1 increase only due to the smoke caps SLM_MAX_STEPS=6.)

## B16 — `_Tee` stdout shim is not a complete stream (`isatty` missing)
- **Where:** `run_autonomous.py` (`_Tee`, installed as `sys.stdout`)
- **When:** 2026-06-24 ~22:02, second real GPU run (Slurm job 36430179)
- **How found:** during model load, `transformers` loading report did `sys.stdout.isatty()`:
  `AttributeError: '_Tee' object has no attribute 'isatty'`. Crashed `train_node` before training.
  (Confirms B14 is cleared — we got past the LoRA target-modules error this time.)
- **Status:** 🟢 fixed 2026-06-24 — added `isatty()`/`fileno()` + `__getattr__` delegation to `sys.__stdout__`.

## B17 — mid-training checkpoint save fails to pickle Unsloth's `SFTConfig`
- **Where:** `training/lora_trainer.py` (`TrainingArguments(save_strategy="epoch")`)
- **When:** 2026-06-24 ~22:19, third real GPU run (Slurm job 36431939)
- **How found:** both configs trained for ~8 min, then at the first epoch-end checkpoint save:
  `_pickle.PicklingError: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>: it's not the
  same object as trl.trainer.sft_config.SFTConfig`. Unsloth's `unsloth_compiled_cache` patches
  `SFTConfig` to a different class identity, so `torch.save(self.args)` fails. (Confirms B16 cleared
  — model load + training now work.)
- **Status:** 🟢 fixed 2026-06-24 — `save_strategy="no"`; final model persisted via `save_pretrained`. Resubmitting.

## B18 — parallel training threads break under Unsloth/Accelerate
- **Where:** `agent/nodes/train.py` (`ThreadPoolExecutor(max_workers=2)`)
- **When:** 2026-06-24 ~22:26, fourth real GPU run (Slurm job 36432812, with B17 fixed)
- **How found:** `ValueError: You can't train a model that has been loaded with device_map='auto'
  in any distributed mode...` raised from inside `concurrent/futures/thread.py`. Two trainers
  initializing Accelerate's process-global `PartialState` concurrently race; combined with
  Unsloth's `device_map='auto'` load, training is rejected. (B17 was the same concurrency hazard
  surfacing as a pickling error; run 36431939 only reached training by thread-scheduling luck.)
- **Root cause:** Unsloth/trl/Accelerate keep process-global singletons and patch classes at
  runtime; they are not safe to run as two trainers in one process.
- **Status:** 🟢 fixed 2026-06-24 — train the 2 configs sequentially (no speedup lost on 1 GPU).
