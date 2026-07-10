# BUGS.md — running bug log

A living record of bugs found while building/validating the SLM Factory pipeline.
Newest entries appended at the bottom. Each entry notes **where** it lives, **when** it
was discovered, **how** it was found, and current **status**.

Status legend: 🔴 open · 🟢 fixed · 🟡 suspected/unconfirmed · ⚪ design gap (not a crash)

| ID | Status | Area | One-line |
|----|--------|------|----------|
| B1 | 🟢 | eval_setup | classification path hardcoded SMS-spam for every task |
| B2 | 🟢 | orchestration | iteration loop is deterministic Python, not an LLM per-iteration (design says LLM) |
| B3 | 🟢 | curation log | dataset composition logged as 0 (placeholders) |
| B4 | 🟢 | train | checkpoint output dirs named `iter{N-1}` (off-by-one) |
| B5 | 🟢 | rollback | docstring claims it restores weights; it only pops the score |
| B6 | 🟢 | eval_set/scorer | binary-only; multi-class collapsed to 2 labels |
| B7 | 🟢 | web_acquire | Exa `search()` rejects `text=` kwarg |
| B8 | 🟢 | eval_set | multi-class eval degenerated to `pos=1/neg=0` |
| B9 | 🟢 | eval_setup | eval set never persisted to disk |
| B10 | 🟢 | run_autonomous | `_Tee` flush-after-close `ValueError` at exit |
| B11 | 🟢 | lora_trainer | train prompt ≠ eval prompt (no train/serve parity) |
| B12 | 🟢 | runner | hardware constraints hardcoded; no autonomous device research |
| B13 | 🟢 | curriculum | Dgold:Dhard ≈ 98:2, far from the paper's 65:35 target |
| B14 | 🟢 | lora_trainer | `target_modules="all-linear"` string → PEFT char-iteration crash |
| B15 | 🟢 | slm_helpers | inference used vanilla `AutoModelForCausalLM`; Unsloth's global patches break it |
| B16 | 🟢 | run_autonomous | `_Tee` stdout missing `isatty()` → transformers loading report crashes |
| B17 | 🟢 | lora_trainer | mid-training checkpoint save can't pickle Unsloth-patched `SFTConfig` |
| B18 | 🟢 | train | two trainers in parallel threads → Accelerate `device_map='auto'` distributed-mode error |
| B19 | 🟢 | iterate/train | `llm_iterate_decision["hyperparams"]` produced but never consumed by `train_node` |
| B20 | 🟢 | curate | `targeted_patterns` from surgical decision ignored; `synthesize_hard_negatives` got no pattern hint |
| B21 | ⚪ | delegate_task | sub-agent has no file-writing tool; zero call sites — no parallel sub-agent work |
| B22 | 🟢 | orchestration | Context Manager not implemented; no turn compaction for long runs |
| B23 | ⚪ | tools | bash/read_file/edit_file/web_search are `@tool`-decorated but bound to no LLM and never invoked |
| B24 | ⚪ | config | `MAX_TURNS_MAIN=1500` is dead config; only `recursion_limit` (50/200) actually caps runs |
| B25 | 🟢 | dag | DAG is a flat node list with no edges; `π=(D,H,S)` not stored per node; lineage attribution impossible |
| B26 | ⚪ | curriculum | Teacher models (DeepSeek-R1/GPT-4.1) never called despite `DEEPSEEK_API_KEY`/`OPENAI_API_KEY` in `.env` |
| B27 | 🟢 | curriculum | 3 of 5 quality controls missing: context-length matching, NER entity diversification, generation CoT annotation |
| B28 | ⚪ | quantize | `quantize.py` returns a profile dict only; no actual INT4/GGUF export |
| B29 | 🟢 | curriculum | NER hard negatives have `entities: []` (mislabeled); generation hard negatives have `response: None` (untrainable) |
| B30 | 🟢 | lora_trainer | No `apply_chat_template` / assistant-only loss masking; raw concatenated-text SFT (train/serve parity holds only for classification) |
| B31 | 🟢 | curate | Dataset size hardcoded `N_TOTAL=150`; generation tasks need 500–3,000 examples |
| B32 | ⚪ | task_analysis | Baseline/SOTA survey (design §2.4 stage 3) never actually runs via web search |
| B33 | 🟢 | train | `train_node` never passes `task_type` to `slm_train()` → always trains with classification prompt format |
| B34 | 🟢 | curate | Double-applied `gold_fraction`: `curate_node` passes `n_total=97` to `build_initial_curriculum` which internally applies `gold_fraction=0.65` again → selects ~63 gold, not 97 |
| B35 | 🟢 | eval_set | `BOUNDARY_KEYWORDS` is SMS-spam-specific (`"free"`, `"win"`, `"prize"`…) — any other classification task gets no real boundary detection |
| B36 | 🟢 | curriculum | 2-for-1 rule is documented in the docstring but NOT implemented: only the synthetic counterexample is added, the original gold example that inspired it is NOT paired alongside |
| B37 | 🟢 | curriculum | No surface-pattern diversity enforcement: paper requires 3–5 distinct surface-text patterns per label; no code checks or enforces this |
| B38 | 🟢 | android_pool | `filter_pool()` ignores `latency_ttft_ms` and `power_watts` from `HardwareConstraints`; design doc §6.2 says these should be logged Phase 1, gating Phase 2 |
| B39 | ⚪ | android_pool | Missing models from design doc pool: HRM-Text-1B (research candidate), Gemma3n-E2B (Tier 2) |
| B40 | 🟢 | android_pool | Tier 3 removed entirely — research shows ≤2B params is the practical Android limit |
| B41 | 🟢 | curation_log | `data-curation.md` schema missing fields from design doc §4.3: per-slice failure taxonomy text, hardware PASS/FAIL lines |
| B42 | 🟢 | lora_trainer | Generation `format_example` concatenates prompt+response without separator; model cannot learn where prompt ends and answer begins |
| B43 | 🟢 | escalate | `escalate_node` does not reset `iteration`, `scores`, `dag`, or `dataset_version` — new model inherits old model's training history |
| B44 | 🟢 | sms_spam | Train/test split is NOT shuffled — first 80% of file is train, last 20% is test — introduces ordering bias |
| B45 | ⚪ | scorer/generation | Generation scorer reports average LLM-judge score as `"f1"` field — semantically misleading; also requires 1 API call per eval example with no batching |
| B46 | 🟢 | scorer/classification | `extract_predictions` falls back to majority (non-positive) label for garbled output — inflates majority-class accuracy |
| B47 | 🟢 | slm_helpers | `_inference_cache` dict never clears — on long runs with many checkpoints, all models stay in VRAM/RAM and will OOM |
| B48 | 🟢 | web_acquire | NER data acquired via Exa lacks entity annotations — documents now annotated via Claude after acquisition |
| B49 | ⚪ | state | `messages: list[Any]` initialized as `[]` but never populated; no LangGraph message-passing occurs between nodes |
| B50 | ⚪ | hardware | No on-device eval harness: design doc §6.3/§6.5 calls for measuring latency, power, and memory on a reference chip (Phase 2) |
| B51 | 🟢 | orchestration | `MAX_TURNS_MAIN`/`turn_budget` never enforced; runs bounded only by recursion_limit |
| B52 | 🟢 | iterate | termination on score>=threshold never re-checked hardware constraints |
| B53 | 🟢 | android_pool | tiers based on peak_memory_mb (RAM), not parameter count; siblings got different tiers than base |
| B54 | 🟢 | escalate | escalation stepped by pool index, not by tier; LLM never involved in model selection |
| B55 | 🟢 | scaling_curve | _probe_model returned 0.0 always; current_dataset_path is None before first curate |
| B56 | 🟢 | task_analysis | task-preference sort immediately overwritten by size-descending sort — dead code |
| B57 | 🟢 | task_planner | _param_range_label formula underestimated param count by 8×; misled stop_threshold calibration |
| B58 | 🟢 | lora_trainer | math_reasoning and code_generation fell to else branch with no answer — model trained on empty targets |
| B59 | 🟢 | harness | max_new_tokens=50 hardcoded; truncates math derivations and code completions |
| B60 | 🟢 | scorer/classification | substring label match produced wrong label when one label is a substring of another |
| B61 | 🟢 | metrics | entity_f1 used set (dedup), undercounting TP/FN for repeated entity mentions |
| B62 | 🟢 | curriculum | math/code hard negatives stored wrong answers as SFT targets, training model to produce errors |
| B63 | 🟢 | downward_probe | original downward probe was dead code (scanned reset DAG for cross-tier entries that never exist); replaced with active train+eval probe node |
| B64 | 🟢 | evaluate | `_pending_weights_refs` None → `AttributeError: 'NoneType'.items()` when train_node hasn't populated it |
| B65 | 🟢 | evaluate | `max(scored, …)` crashes with `ValueError: max() arg is empty sequence` when scored dict is empty |
| B66 | 🟢 | evaluate | `config_labels = list(config_descriptions.values())` extracts config dicts not keys → `KeyError['label']` |
| B67 | 🟢 | scorer/generation | invalid model ID `claude-haiku-4-5-20251001` crashes every generation scorer call |
| B68 | 🟢 | scorer/generation | math_reasoning and code_generation always use LLM-as-judge instead of exact-match / pass@1 |
| B69 | 🟢 | scorer/generation | neg-slice gold labels all `"correct"` regardless of whether example has a reference answer |
| B70 | 🟢 | scorer/ner | regex `r'\[.*?\]'` truncates JSON array on first `]` inside entity text |
| B71 | 🟢 | scorer/ner | failures computed with set comparison (dedup), inconsistent with Counter-based entity_f1 |
| B72 | 🟢 | metrics | `acc_from_lists` divides by `len(preds)` not `min(len(preds), len(golds))` → under-reports accuracy |
| B73 | 🟢 | metrics | `per_slice_scores` boundary slice is open-ended; overlong predictions inflate boundary score |
| B74 | 🟢 | iterate | turn-budget formula `iteration * 2` fires one iteration early; should be `(iteration+1) * 2` |
| B75 | 🟢 | iterate | `initial_stop_threshold` floor uses `or` instead of `is None` — falsy 0.0 bypasses floor |
| B76 | 🟢 | iterate | `intervention` variable uninitialized at function scope; `LLM fail` branch could reference unbound var |
| B77 | 🟢 | escalate | `dataset_version`, `last_intervention`, `last_curation` not reset on escalation |
| B78 | 🟢 | evaluate | DAG `intervention` field uses `apply_iteration_policy` fallback even when LLM already decided |
| B79 | 🟢 | runner | `lifetime_best_score` missing from initial state dict in `tests/pipeline/run.py` |
| B80 | 🟢 | runner | `last_intervention` initialized to `""` instead of `"data_rebuild"` → first curate no-ops |
| B81 | 🟢 | lora_trainer | dataset JSONL opened without `encoding="utf-8"` → `cp1252` mojibake on Windows |
| B82 | 🟢 | lora_trainer | `ex['label']` hard-subscript in classification format — KeyError on missing label field |
| B83 | 🟢 | lora_trainer | `fp16/bf16` computed with two separate `is_bf16_supported()` calls; CPU-only env crashes |
| B84 | 🟢 | slm_helpers | cache eviction calls `del old` but never `torch.cuda.empty_cache()` — GPU OOM persists |
| B85 | 🟢 | task_analysis | `initial_stop_threshold` guard uses `not state.get(…)` — threshold 0.0 would be overwritten |
| B86 | 🟢 | downward_probe | `filter_pool` returns triplicated entries; LLM prompt misleading, fallback always picks Q8_0 |
| B87 | 🟢 | downward_probe | quant selection silently resolves to unquantized base variant when LLM picks by model_id only |
| B88 | 🟢 | downward_probe | artifact paths use relative `'artifacts'` root — wrong dir if CWD differs |
| B89 | 🟢 | bash_tool | `_HELPERS_INJECT` built at import time with frozen CWD; PYTHONPATH wrong on any other CWD |
| B90 | 🟢 | bash_tool | `export` keyword invalid on Windows cmd.exe; PYTHONPATH injection fails every bash() call |
| B91 | 🟢 | delegate_task | `response.content[0].text` crashes when first block is ThinkingBlock or ToolUseBlock |
| B92 | 🟢 | query_traces | `int(query.split(':')[1])` unguarded — ValueError on non-integer suffix |
| B93 | 🟢 | web_search | `r.text[:500]` TypeError when Exa returns `r.text=None` for unscrapable results |
| B94 | 🟢 | android_pool | `_q4_sibling.peak_memory_mb` = `int4_size_mb + 400` can understate base model's true peak → OOM |
| B95 | 🟢 | android_pool | power gate `if measured.get('avg_watts')` falsy-checks 0.0, always passing when sensor returns 0 |
| B96 | 🟢 | curate | surgical path calls `synthesize_hard_negatives([], …)` when `failures` is empty — silent no-op wasting an iteration |
| B97 | 🟢 | curate | `eval_set is None` in production `data_rebuild` raises uninformative `AttributeError` instead of descriptive RuntimeError |
| B98 | ⚪ | live_confirm | M0 re-inference not implemented; taxonomy label filter (itself broken) used instead |
| B99 | 🟢 | live_confirm | cluster membership checked via `cluster in str(t)` substring match — produces false positives on trace content |
| B100 | 🟢 | eval_set | `boundary_texts = {e["text"] …}` hard-subscript → KeyError when example lacks "text" key |

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
- **Status:** 🟢 fixed 2026-06-24 — `iterate_node` now calls the orchestrator LLM (Claude Sonnet 4.6)
  with the full `data-curation.md` trajectory + current failures. LLM returns structured JSON with
  `{intervention, hypothesis, hyperparameter_changes?, targeted_patterns?}`. Score-band rules remain
  as fallback if the LLM call fails. `hypothesis` field written to `data-curation.md` via
  `state["last_hypothesis"]` passed through to `evaluate_node`.

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
- **Status:** 🟢 fixed 2026-06-24 — increment `state["iteration"]` BEFORE building `output_dir`, so
  dir name matches the DAG and `data-curation.md` iteration number.

## B5 — rollback_node doesn't restore best weights
- **Where:** `agent/nodes/rollback.py`
- **When:** 2026-06-23, code review
- **How found:** docstring says "Restores previous weights_ref" but the code only pops the last
  score and bumps a counter; `best_weights_ref` is untouched.
- **Impact:** works today only because `best_weights_ref` is managed in `evaluate_node`; the
  docstring is misleading and the intent is unimplemented.
- **Status:** 🟢 fixed 2026-06-24 — marks the regressing DAG node as pruned, then scans
  all non-pruned nodes for the best score and restores `best_weights_ref` + `best_score`
  from that node.

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
- **Status:** 🟢 fixed 2026-06-24 — `curate_node` now explicitly targets `N_TOTAL=150` with
  `n_gold=int(150*0.65)=97` and `n_hard=53`. `synthesize_hard_negatives` (classification) now
  uses all labels as source material (not just the minority class), generating a boundary-crossing
  counterexample for each candidate.

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

## B19 — `llm_iterate_decision["hyperparams"]` produced but never consumed
- **Where:** `agent/nodes/train.py` (lines 22–25, hardcoded configs)
- **When:** 2026-06-26, post-B2 code review
- **How found:** `iterate_node` stores structured hyperparameter fields in `state["llm_iterate_decision"]`
  but `train_node` always ran the same two hardcoded configs (LoRA r=8 / full_ft). A hyperparameter
  intervention therefore retrained identically, could not move the score, and burned rounds until
  `consecutive_no_improvement >= 2` triggered model escalation.
- **Impact:** `hyperparameter` intervention was a no-op; H dimension of π=(D,H,S) search was fake.
- **Status:** 🟢 fixed 2026-06-26 — `train_node` now calls `_build_configs(state)` which reads
  `llm_iterate_decision["hyperparams"]` (`lora_rank`, `lr`, `nr_epochs`, `batch_size`) for Config A and
  derives a contrasting Config B. Default configs used as fallback when no LLM decision is present.

## B20 — `targeted_patterns` from surgical decision ignored in `synthesize_hard_negatives`
- **Where:** `agent/nodes/curate.py` (surgical branch); `data/curriculum.py` (`synthesize_hard_negatives`)
- **When:** 2026-06-26, post-B2 code review
- **How found:** `iterate_node` populates `llm_iterate_decision["targeted_patterns"]` with the LLM's
  description of the failure pattern to address, but `curate_node` never extracted it and
  `synthesize_hard_negatives` had no `targeted_pattern` parameter — the LLM surgical guidance was
  silently discarded.
- **Impact:** surgical synthesis was blind to the LLM diagnosis; generated generic hard negatives
  instead of targeted ones.
- **Status:** 🟢 fixed 2026-06-26 — `curate_node` extracts `targeted_pattern` from
  `state["llm_iterate_decision"]`; `synthesize_hard_negatives` accepts a `targeted_pattern: str`
  kwarg and injects it into the generation prompt as a "focus on this failure pattern" hint.

## B21 — `delegate_task` is broken and has zero call sites
- **Where:** `agent/tools/delegate_task.py`
- **When:** 2026-06-26, design gap audit
- **How found:** the sub-agent is told to write its result to `output_file` but is given no
  file-writing tool (single `messages.create` call, no tools bound). The file is never
  written and the function falls back to returning raw text — the opposite of the paper's
  "summary-to-disk, no raw context" isolation pattern. `delegate_task` is never called
  anywhere in the codebase.
- **Impact:** no parallel sub-agent work; the paper's context isolation pattern is unimplemented.
- **Status:** ⚪ design gap — deferred to Phase 2. Fix requires binding `edit_file` tool to the
  sub-agent and adding call sites in e.g. `curate_node` (synthesize dataset while training runs).

## B22 — Context Manager not implemented
- **Where:** no file
- **When:** 2026-06-26, design gap audit
- **How found:** `agent/state.py` and all nodes hold no conversation history; there is nothing to
  compact. The paper's Context Manager compacts old turns while preserving key decisions and eval
  results for 500–1,500-turn runs.
- **Impact:** lower priority than other gaps because each node is a fresh stateless LLM call
  (not a growing conversation). Becomes critical only if nodes are wired into a multi-turn chat loop.
- **Status:** ⚪ design gap — not yet started. `data-curation.md` serves as the durable complement;
  the stateless node architecture avoids most compaction pressure for now.

## B23 — 4 named tools are decorated but never invoked
- **Where:** `agent/tools/bash_tool.py`, `file_tools.py`, `web_search.py`
- **When:** 2026-06-26, design gap audit
- **How found:** `grep -r "bind_tools\|tool_node\|tools=" agent/` returns nothing. The tools exist
  as `@tool` functions but no LangGraph `ToolNode` or `bind_tools` call connects them to any LLM.
  All data acquisition, file reading, and shell execution happen via direct Python calls.
- **Impact:** the paper's "4 named tools" interface (§2.3) is not replicated; the agent cannot
  use them for open-ended reasoning steps.
- **Status:** ⚪ design gap — wiring tools into the LangGraph graph requires replacing deterministic
  node functions with a ReAct loop.

## B24 — `MAX_TURNS_MAIN` is dead config
- **Where:** `config.py` (`MAX_TURNS_MAIN = 1500`)
- **When:** 2026-06-26, design gap audit
- **How found:** the value is defined but never read. Run length is controlled only by
  `recursion_limit` in `graph.compile()` / `run_autonomous.py`.
- **Impact:** cosmetic — effective turn limit is whatever `recursion_limit` is set to, not 1,500.
- **Status:** ⚪ design gap — wire `MAX_TURNS_MAIN` into `graph.compile(recursion_limit=config.MAX_TURNS_MAIN)`.

## B25 — DAG has no edges; π=(D,H,S) not stored per node
- **Where:** `agent/nodes/evaluate.py` (DAG node construction)
- **When:** 2026-06-26, design gap audit
- **How found:** `dag_node` dict stores score/model/weights_ref but no parent pointer, no `D`
  (dataset version), no `H` (hyperparameter config), no `S` (learning strategy). The DAG is a
  flat append-only list with no edges. Lineage attribution requires edges and full π triples.
- **Impact:** the DAG purpose (tracking why accuracy changed) is unachievable; rollback can only
  restore weights, not reproduce the exact (D,H,S) that achieved a given score.
- **Status:** ⚪ design gap — extend `dag_node` with `parent_iteration`, `dataset_version`,
  `hyperparams`, `strategy` fields and a parent pointer on each `dag.append()` call.

## B26 — Teacher models (DeepSeek-R1/GPT-4.1) never called
- **Where:** `data/curriculum.py` (`synthesize_hard_negatives`, generation branch); `config.py`
- **When:** 2026-06-26, design gap audit
- **How found:** `DEEPSEEK_API_KEY` and `OPENAI_API_KEY` are in `.env` / `config.py` but never
  imported or used anywhere. The generation branch calls `claude-sonnet-4-6` for all task types,
  including generation tasks where the design spec (§2.4) calls for DeepSeek-R1 (math/science CoT)
  or GPT-4.1 (code/QA).
- **Impact:** generation hard negatives lack CoT annotation; training data for generation tasks
  is weaker than paper spec.
- **Status:** ⚪ design gap — add DeepSeek and OpenAI client paths in `curriculum.py` generation
  branch, guarded by task sub-type (math → DeepSeek-R1, code/QA → GPT-4.1).

## B27 — 3 of 5 quality controls missing
- **Where:** `data/curriculum.py` (`apply_quality_controls`)
- **When:** 2026-06-26, design gap audit
- **How found:** only label balancing (classification) is implemented. Missing:
  1. **Context-length matching** — training examples should match the length distribution of eval examples.
  2. **NER entity diversification** — no entity value should appear >2–3× in the training set.
  3. **CoT annotation for generation** — generation examples need chain-of-thought traces, not just answers.
- **Impact:** training data quality below paper spec for NER and generation tasks.
- **Status:** ⚪ design gap — implement each control in the relevant task-type branch of
  `apply_quality_controls`.

## B28 — `quantize.py` returns a profile dict; does no quantization
- **Where:** `training/quantize.py`
- **When:** 2026-06-26, design gap audit
- **How found:** function returns benchmark estimates from the Android pool definition. No INT4/GGUF
  export, no ONNX conversion, no QNN packaging.
- **Impact:** "Android-deployable" claim is theoretical only.
- **Status:** ⚪ design gap — correctly Phase 2. Phase 1 scope is the training loop, not on-device export.

## B29 — NER and generation hard negatives are untrainable
- **Where:** `data/curriculum.py` (`synthesize_hard_negatives`)
- **When:** 2026-06-26, design gap audit
- **How found:**
  - NER branch appends `{"text": ..., "entities": []}` — empty entities is ambiguous: teaches the
    model that passages with this surface form have no entities, including ones that actually do.
  - Generation branch appends `{"prompt": ..., "response": None}` — `None` is not a valid training
    target; the trainer will crash or skip these examples.
- **Impact:** NER hard negatives teach incorrect labeling; generation hard negatives are unusable.
- **Status:** 🟢 fixed 2026-06-26 — NER hard negatives now ask Claude to return spans with INCORRECT
  entity types (wrong type, not absent). Generation hard negatives now ask Claude for a plausible-but-wrong
  answer instead of `None`. Both branches also include the original gold example (2-for-1 rule, B36).

## B30 — No `apply_chat_template` / assistant-only loss masking
- **Where:** `training/lora_trainer.py` (prompt formatting)
- **When:** 2026-06-26, design gap audit
- **How found:** trainer concatenates `PROMPT + LABEL` as a single text field and feeds it to
  `SFTTrainer`. Loss is computed over prompt tokens as well as label tokens. Proper SFT applies
  loss only to the assistant/label portion (assistant-only masking) and uses the model chat template.
- **Impact:** for classification, the prompt is short so the label dominates the loss — works in
  practice. For NER/generation with long prompts, prompt tokens dominate and degrade learning.
  Also affects train/serve parity for non-classification tasks (B11 was fixed for classification only).
- **Status:** ⚪ design gap — use `tokenizer.apply_chat_template` + `DataCollatorForCompletionOnlyLM`
  to mask prompt tokens from the loss.

## B31 — Dataset size hardcoded at `N_TOTAL=150`
- **Where:** `agent/nodes/curate.py` (`N_TOTAL = 150`)
- **When:** 2026-06-26, design gap audit
- **How found:** paper specifies 100–200 examples for classification, 500–3,000 for generation.
  `N_TOTAL=150` is reasonable for classification but too small for generation tasks.
- **Impact:** generation tasks will underfit with 150 examples.
- **Status:** ⚪ design gap — make `N_TOTAL` task-type-aware:
  `classification → 150`, `NER → 300`, `generation → 1000`.

## B32 — Baseline/SOTA survey (design §2.4 stage 3) never runs
- **Where:** `agent/nodes/task_analysis.py`
- **When:** 2026-06-26, design gap audit
- **How found:** `task_analysis_node` sets `stop_threshold=0.96` from a hardcoded default, not from
  a web survey of published baselines. The design spec calls for an Exa search to find SOTA accuracy
  on the target benchmark and calibrate `stop_threshold` accordingly.
- **Impact:** agent may stop too early or aim too high for the model tier.
- **Status:** ⚪ design gap — add an Exa `web_search` call in `task_analysis_node` querying
  "{task_name} state of the art accuracy benchmark" to calibrate `stop_threshold`.

## B33 — `train_node` never passes `task_type` to `slm_train()`
- **Where:** `agent/nodes/train.py` → `training/slm_helpers.py` → `training/lora_trainer.py`
- **When:** 2026-06-26, deep codebase audit
- **How found:** `train_node` calls `slm_train(dataset_path, base_model, nr_epochs, ...)` but never
  sends `task_type`. `slm_helpers.train()` accepts `task_type` but defaults to `"classification"`.
  `lora_trainer._run_unsloth_training()` uses `task_type` to select the prompt template for formatting
  training examples. So NER and generation tasks are **always trained with the classification prompt
  template** (`"Classify this message as spam or ham..."`) regardless of their actual task type.
- **Impact:** NER/generation training is fundamentally broken — the model learns a classification format
  for a non-classification task. The eval scorer uses the correct task-type-specific prompt, so there is
  a train/serve format mismatch for all non-classification tasks.
- **Status:** 🟢 fixed 2026-06-26 — `train_node` now passes `task_type=state["task_type"]` to `slm_train()`.

## B34 — `curate_node` double-applies `gold_fraction`, halving gold count
- **Where:** `agent/nodes/curate.py` (line ~38-39)
- **When:** 2026-06-26, deep codebase audit
- **How found:** For `data_rebuild`, `curate_node` computes `n_gold_target = int(150 * 0.65) = 97` and
  passes it as `n_total=97` to `build_initial_curriculum()`. But `build_initial_curriculum()` has its own
  `gold_fraction=0.65` default, and internally computes `n_gold = int(97 * 0.65) = 63`. So the actual
  gold count is ~63, not the intended ~97. The 65:35 fraction is applied twice.
- **Impact:** gold dataset is 30% smaller than intended. The overall dataset composition is ~63 gold +
  53 hard = ~116 total at roughly 54:46 — more aggressive than the paper's 65:35 target.
- **Status:** 🟢 fixed 2026-06-26 — `curate_node` now passes `gold_fraction=1.0` to
  `build_initial_curriculum()` since it already computed the gold target. The function no longer
  double-applies the 65% fraction.

## B35 — Boundary keywords in `eval_set.py` are SMS-spam-specific
- **Where:** `data/eval_set.py` (`BOUNDARY_KEYWORDS` on line ~39)
- **When:** 2026-06-26, deep codebase audit
- **How found:** The keywords used to detect boundary examples (`"www"`, `"http"`, `"free"`, `"win"`,
  `"prize"`, `"offer"`, `"click"`, `"call"`, `"urgent"`, `"limited"`, `"txt"`) are spam-specific terms.
  For any other classification task (e.g., intent classification, sentiment), these keywords match nothing
  and the boundary set degrades to "short positive examples" only — which is a poor approximation of
  decision-boundary examples.
- **Impact:** eval set quality degrades for any non-SMS classification task. The paper's concept of
  `E_boundary` (confusable pairs at decision boundaries) is only properly implemented for SMS spam.
- **Status:** 🟢 fixed 2026-06-26 — replaced SMS-spam-specific keyword set with task-agnostic
  length-based boundary detection: negative-class examples whose text length is closest to the
  positive-class mean are selected as boundary candidates. Works for any binary classification task.

## B36 — 2-for-1 rule documented but not implemented
- **Where:** `data/curriculum.py` (`synthesize_hard_negatives` docstring)
- **When:** 2026-06-26, deep codebase audit
- **How found:** the docstring says "Generate n hard negatives using the 2-for-1 rule" but the actual
  code only generates the synthetic counterexample. The paper's 2-for-1 rule (§2.3) means: for each
  challenging case, include BOTH the original gold example AND one hard negative. The implementation
  returns only the hard negatives; the originals are not paired alongside them.
- **Impact:** the training curriculum lacks the "positive anchor" half of each contrastive pair. The model
  sees what NOT to predict but not what TO predict for the same surface form. This reduces the
  effectiveness of hard negatives.
- **Status:** ⚪ design gap — `synthesize_hard_negatives` should return both the original example and
  the synthetic counterexample as a pair.

## B37 — No surface-pattern diversity enforcement (3-5 patterns per label)
- **Where:** `data/curriculum.py` (`apply_quality_controls`)
- **When:** 2026-06-26, deep codebase audit
- **How found:** the paper (§2.3) requires "each label requires 3–5 distinct surface-text patterns to
  enforce diversity beyond individual example counts." No code checks or enforces this. The quality
  controls only implement label balancing (3x cap).
- **Impact:** training data may contain many similar examples for the same label (e.g., 20 "You've won a
  prize!" variants) without syntactic diversity, limiting generalization.
- **Status:** ⚪ design gap — add a diversity check that clusters training examples by surface pattern
  (e.g., TF-IDF or embedding similarity) and ensures >= 3 distinct patterns per label.

## B38 — `filter_pool()` ignores `latency_ttft_ms` and `power_watts`
- **Where:** `android_pool.py` (`filter_pool`)
- **When:** 2026-06-26, deep codebase audit
- **How found:** `HardwareConstraints` has `latency_ttft_ms` and `power_watts` fields, and the design
  doc (§6.2) says latency and power should be "logged, not gating" in Phase 1. But `filter_pool()` only
  filters on `storage_mb` and `memory_mb` — it doesn't even log whether models pass the latency/power
  thresholds. The theoretical tok/s data is in `ModelSpec` but never compared against the constraints.
- **Impact:** hardware constraint logging is incomplete. The `data-curation.md` hardware profile section
  cannot report PASS/FAIL for latency or power because no comparison code exists.
- **Status:** ⚪ design gap — add latency/power comparison to `filter_pool()` return value or a separate
  `check_all_constraints()` function that returns per-constraint PASS/FAIL status.

## B39 — Missing models from design doc pool
- **Where:** `android_pool.py` (`ANDROID_POOL`)
- **When:** 2026-06-26, deep codebase audit
- **How found:** the design doc §6.4 lists HRM-Text-1B (sapientinc/HRM-Text-1B, ~600MB, Tier 1,
  research candidate) and Gemma3n-E2B (~1.3GB, Tier 2, MatFormer arch). Neither is in the
  implementation's `ANDROID_POOL` list.
- **Impact:** search space is smaller than designed. HRM-Text is flagged as needing custom handling
  (no llama.cpp/Ollama support), so deferral is reasonable. Gemma3n-E2B has no such caveat.
- **Status:** ⚪ design gap — add Gemma3n-E2B to Tier 2. Add HRM-Text-1B with a `notes` field
  flagging its custom runtime requirement.

## B40 — Ministral-3B exceeds Tier 3 upper bound
- **Where:** `android_pool.py` (`ANDROID_POOL`)
- **When:** 2026-06-26, deep codebase audit
- **How found:** Ministral-3B is listed at `int4_size_mb=1900` but the design doc §6.4 defines Tier 3
  as "generous upper bound" at ~1.5GB. 1.9GB exceeds the stated ceiling by 400MB and the CLAUDE.md
  constraint of "≤~1.5GB INT4". The design doc does list it but with "~<2GB, 256K context."
- **Impact:** if selected, the model would exceed the Android deployment target. `filter_pool()` would
  correctly exclude it for storage_mb <= 1500, but if constraints are relaxed it could slip through.
- **Status:** ⚪ design gap — either update the Tier 3 bound to ~2GB in docs, or remove Ministral-3B
  from the pool, or flag it as "exceeds nominal bound" in `notes`.

## B41 — `data-curation.md` schema missing design doc fields
- **Where:** `data/curation_log.py` (`write_iteration`)
- **When:** 2026-06-26, deep codebase audit
- **How found:** the design doc §4.3 specifies these fields that are not in the implementation:
  1. Per-slice failure taxonomy text (description of what failed and why)
  2. Hardware PASS/FAIL lines: `Storage: {size}MB vs S_max={S_max}MB — PASS/FAIL`, same for memory,
     latency, and power
  3. Escalation availability: `Escalation available: yes/no`, `Next model: {id}`, `Would pass
     constraints: yes/no`
- **Impact:** the lineage artifact is less informative than designed. Phase 2 hardware gating will need
  the PASS/FAIL lines to make escalation decisions from the log.
- **Status:** ⚪ design gap — add `failure_taxonomy`, `hw_pass_fail` dict, and `escalation_info` dict
  to `write_iteration()` parameters.

## B42 — Generation training concatenates prompt+response without separator
- **Where:** `training/lora_trainer.py` (`format_example`, generation branch)
- **When:** 2026-06-26, deep codebase audit
- **How found:** the generation branch of `format_example` returns
  `GENERATE_PROMPT.format(text=ex["text"]) + "\n\n" + ex.get("response", ex.get("label", ""))`.
  This concatenation has no explicit delimiter between the prompt portion and the response portion.
  The model sees continuous text and cannot learn where to start generating.
- **Impact:** loss is computed over prompt+response jointly (see B30 — no assistant-only masking). Even
  if masking were added, there's no structural marker (like `<|assistant|>`) to anchor it. Combined
  with B33, generation tasks are triply broken: wrong prompt template + no separator + no loss masking.
- **Status:** ⚪ design gap — use `tokenizer.apply_chat_template` with role-tagged messages, or at
  minimum insert a clear `\n\nAnswer:` separator that the inference prompt also uses.

## B43 — `escalate_node` doesn't reset state for new model
- **Where:** `agent/nodes/escalate.py`
- **When:** 2026-06-26, deep codebase audit
- **How found:** when escalating to a larger model, the node sets `state["selected_model"]` to the new
  model and `state["consecutive_no_improvement"] = 0`, but does NOT reset `iteration`, `scores`, `dag`,
  or `dataset_version`. The new model inherits the old model's entire training history.
- **Impact:** the `iterate_node`'s LLM call reads `data-curation.md` which contains the old model's
  trajectory. The LLM may make decisions based on the smaller model's scores and failure patterns,
  which are irrelevant to the new model's capabilities. The `scores` list mixes scores from different
  models, making rollback logic incorrect (comparing model A's score to model B's score).
- **Status:** ⚪ design gap — on escalation, either reset `scores`/`dag`/`iteration` or clearly mark the
  escalation boundary in the trajectory so the LLM knows to ignore pre-escalation data.

## B44 — SMS spam train/test split is not shuffled
- **Where:** `data/loaders/sms_spam.py`
- **When:** 2026-06-26, deep codebase audit
- **How found:** the loader uses `train = examples[:split]`, `test = examples[split:]` where split is 80%.
  The UCI SMS Spam Collection is not randomly ordered — it has blocks of spam and ham messages. A
  sequential split may produce a test set with a different class distribution than the training set.
- **Impact:** potential train/test distribution mismatch. In practice, the dataset's natural ordering is
  close enough to random that this hasn't caused visible problems, but it's technically incorrect.
- **Status:** ⚪ design gap — shuffle with a fixed seed before splitting.

## B45 — Generation scorer misnames metric and doesn't batch API calls
- **Where:** `eval/scorers/generation.py`
- **When:** 2026-06-26, deep codebase audit
- **How found:** two issues:
  1. The scorer returns the average LLM-judge score (0.0-1.0) in the `f1` field. This is semantically
     wrong — it's a judge score, not an F1 metric. Downstream code treats it as F1.
  2. Each eval example requires a separate Anthropic API call to the judge model. For 100 examples,
     that's 100 sequential API calls with no batching or parallelism.
- **Impact:** (1) misleading metric name; (2) generation eval is very slow and expensive.
- **Status:** ⚪ design gap — rename field to `judge_score` or `metric` throughout, and batch judge
  calls using the Anthropic batch API or concurrent requests.

## B46 — Classification `extract_predictions` silently defaults to majority class
- **Where:** `eval/scorers/classification.py` (`extract_predictions`)
- **When:** 2026-06-26, deep codebase audit
- **How found:** when the model outputs garbled text that doesn't contain any known label, the extractor
  falls back to the majority (non-positive) label instead of flagging it as an extraction failure. This
  means a model that outputs random garbage scores as well as one that consistently predicts the
  majority class.
- **Impact:** inflates majority-class accuracy. Masks output-format problems that would be caught if
  extraction failures were tracked separately.
- **Status:** ⚪ design gap — track extraction failures as a separate metric; use `"UNKNOWN"` as the
  fallback label and count it in the failure list.

## B47 — Inference model cache never clears; OOM risk on long runs
- **Where:** `training/slm_helpers.py` (`_inference_cache`)
- **When:** 2026-06-26, deep codebase audit
- **How found:** `_inference_cache` is a module-level dict mapping `weights_ref → (model, tokenizer)`.
  Each new checkpoint loads a new model into VRAM/RAM but old models are never evicted. Over 10+
  iterations with 2 configs each, 20+ models accumulate in memory.
- **Impact:** will OOM on GPU servers with limited VRAM, especially with 1B+ models. On the L40 (48GB),
  this allows ~30 Qwen3-0.6B checkpoints before exhaustion, but fewer for larger models.
- **Status:** ⚪ design gap — implement LRU eviction (keep only the last 2-3 checkpoints cached) or
  clear the cache between iterations.

## B48 — NER web-acquired data lacks entity annotations
- **Where:** `data/loaders/web_acquire.py` (NER acquisition path)
- **When:** 2026-06-26, deep codebase audit
- **How found:** for NER tasks, `acquire_dataset()` searches Exa for topic-relevant documents and returns
  `{"text": ..., "label": <topic_name>}`. But NER training requires `{"text": ..., "entities":
  [{"text": "...", "type": "..."}]}`. The acquired data has no span-level entity annotations.
- **Impact:** NER training data is unusable without a separate entity annotation step. The `curate_node`
  and `build_initial_curriculum` don't add entity annotations. Combined with B29 (NER hard negatives
  have `entities: []`), the entire NER pipeline produces empty or mislabeled training data.
- **Status:** ⚪ design gap — add an LLM-driven entity annotation step after web acquisition: call
  Claude to extract entities from each acquired passage and produce gold `entities` lists.

## B49 — `messages` field in `AgentState` is populated but never used
- **Where:** `agent/state.py`
- **When:** 2026-06-26, deep codebase audit
- **How found:** `messages: list[Any]` is initialized as `[]` in `run.py`'s `initial_state` but no node
  ever reads or writes to it. LangGraph supports message-passing between nodes via this field (it's
  the standard channel for chat-style agent interactions), but the current architecture bypasses it
  entirely — each node reads/writes state fields directly.
- **Impact:** cosmetic for now. Becomes relevant if the architecture is refactored toward a ReAct/tool-use
  loop (B23) where message history matters.
- **Status:** ⚪ design gap — remove the field to reduce confusion, or wire it into the iterate_node's
  LLM call to accumulate a conversation transcript.

## B53 — tiering by RAM, not params; siblings mis-tiered
- **Where:** `config/android_pool.py` (`_q4_sibling`, `_q8_sibling`, 12 base ModelSpec entries)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** tier should reflect model capability (parameter count) not hardware footprint;
  siblings were re-tiered by peak_memory_mb, so a Llama-3.2-1B Q4 was tier 0 while its base was
  tier 1 — escalation stepping by tier would skip the sibling.
- **Impact:** tier-based escalation (Task 4) would mis-order the escalation path; capability-tier
  concept was confused with hardware-budget concept.
- **Status:** 🟢 fixed 2026-07-08 — tier computed from params_b = int4_size_mb * 2 / 1000 for
  base models; siblings inherit base.tier without recomputing.

## B52 — accuracy-goal termination ignored hardware constraints
- **Where:** `agent/nodes/iterate.py`
- **When:** 2026-07-08, pipeline hardening review
- **How found:** `iterate_node` terminated the instant score>=threshold, no hardware re-check.
- **Impact:** accuracy always won over hardware at termination.
- **Status:** 🟢 fixed 2026-07-08 — when `hw_gating_enabled`, a hardware-failing model is not
  accepted as terminal; the loop continues (escalate on stagnation / else iterate). Gating off
  (Phase 1 default) is unchanged.

## B51 — turn budget never enforced
- **Where:** `config.py`, `agent/graph.py`, `agent/nodes/iterate.py`
- **When:** 2026-07-08, pipeline hardening review
- **How found:** `MAX_TURNS_MAIN=1500` defined but never read (was B24); no node counts turns.
- **Impact:** a stuck loop ran to recursion_limit; budget was advisory only.
- **Status:** 🟢 fixed 2026-07-08 — `graph.compile().with_config(recursion_limit=MAX_TURNS_MAIN)`;
  `iterate_node` terminates when `iteration*2 >= turn_budget`.

## B50 — No on-device eval harness for latency/power/memory measurement
- **Where:** no file
- **When:** 2026-06-26, deep codebase audit
- **How found:** the design doc §6.3 and §6.5 describe a Phase 2 eval harness that quantizes the
  checkpoint to INT4, deploys it to a reference device (or Qualcomm AI Hub simulator), and measures:
  - **Latency** (TTFT in ms, tok/s throughput)
  - **Power** (average watts during sustained inference, via Android Battery Historian)
  - **Memory** (peak RSS during inference)
  No code skeleton, interface definition, or stub exists for any of these measurements. The
  `theoretical_hardware_profile()` function in `quantize.py` returns only static lookup values.
- **Impact:** the "hardware-in-the-loop" aspect of SLM Factory — the core differentiator from the
  Pioneer Agent paper — has no implementation path started. Phase 2 will need to build this from
  scratch.
- **Status:** ⚪ design gap — Phase 2 scope. Recommended approach: define a `HardwareEvalResult`
  dataclass and a `measure_on_device(weights_ref, model_id, chip) -> HardwareEvalResult` interface
  now; implement via Qualcomm AI Hub API or ADB shell profiling in Phase 2.

## B55 — scaling_curve probes always returned 0.0
- **Where:** `agent/nodes/cold_start/scaling_curve.py` (`_probe_model`)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** scaling_curve runs before curate_node, so current_dataset_path is always None;
  every probe returned 0.0 immediately; the linear fit produced a flat zero line; the node always
  fell back to the largest model. The scaling curve was entirely dead code.
- **Impact:** model selection always defaulted to largest feasible model regardless of task.
- **Status:** 🟢 fixed 2026-07-09 — when current_dataset_path is None, _probe_model constructs
  a temporary JSONL seed from eval_set.pos + neg + boundary examples, trains a 1-epoch probe on
  that seed, and cleans up in a finally block. This gives a real ranking signal without restructuring
  graph edges.

## B56 — dead task-preference sort in task_analysis_node
- **Where:** `agent/nodes/cold_start/task_analysis.py`
- **When:** 2026-07-08, pipeline hardening review
- **How found:** sort by (tier, benchmark_score) was applied then immediately overwritten by a
  size-descending re-sort before storing to state. scaling_curve_node (which reads feasible_models)
  only needs size order, not task preference.
- **Impact:** dead code; no functional effect but confuses maintainers.
- **Status:** 🟢 fixed 2026-07-08 — removed task-preference sort and _TASK_TYPE_TO_POOL_KEY dict.

## B54 — escalation stepped by index not tier; no LLM model selection
- **Where:** `agent/nodes/escalate.py`
- **When:** 2026-07-08, pipeline hardening review
- **How found:** escalate_node took `feasible[current_idx + 1]` (next entry in ascending size
  sort) — with 36 pool entries, the "next" model was often a quant sibling of the same model,
  not a genuinely larger architecture. The orchestrator LLM was never consulted on which model
  in the next tier to try.
- **Impact:** escalation didn't reliably advance capability; LLM task knowledge was unused.
- **Status:** 🟢 fixed 2026-07-08 — escalation finds all feasible models in `current_tier + 1`,
  calls `_llm_choose_model` to select among them for the task, falls back to largest on LLM failure.
  A downward probe was also added here, but that DAG-scanning version was dead code — see B63 for
  the corrected active implementation (a real downward_probe graph node that trains + evals the best
  one-tier-down candidate at termination).

## B57 — _param_range_label underestimated param count by 8×
- **Where:** `agent/task_planner.py` (`_param_range_label`)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** formula `int4_size_mb / 1024 / 4` gave 0.244B for a 1000MB model; correct is
  int4_size_mb * 2 / 1000 ≈ 1.32B. The LLM planner was told the pool was "0.1B–0.6B" when
  it was actually "0.6B–5.0B", degrading stop_threshold calibration for larger models.
- **Status:** 🟢 fixed 2026-07-08 — formula changed to params_b = int4_size_mb * 2 / 1000;
  also filters to base models (quant=None) to avoid double-counting siblings.

## B58 — math/code training format produced empty completions
- **Where:** `training/lora_trainer.py` (`_run_unsloth_training`, format_example dispatch)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** `math_reasoning` and `code_generation` hit the `else` branch returning only
  `{"text": ex.get("text", "")}` with no answer. SFT loss was computed over a blank completion.
- **Impact:** math and code models trained on this data learned nothing task-relevant.
- **Status:** 🟢 fixed 2026-07-08 — both task types now route through the `generation` branch,
  which handles prompt/answer/response/cot_reasoning keys correctly.

## B59 — max_new_tokens=50 truncates math/code eval outputs
- **Where:** `eval/harness.py`
- **When:** 2026-07-08, pipeline hardening review
- **How found:** math derivations and code functions routinely exceed 50 tokens; the model's
  answer was truncated, degrading eval scores in a non-representative way.
- **Status:** 🟢 fixed 2026-07-08 — classification/NER use 50 tokens; math_reasoning,
  code_generation, generation use 256 tokens.

## B60 — classification label extraction used substring match, breaking on sub-label names
- **Where:** `eval/scorers/classification.py` (`extract_predictions`)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** "positive" is a substring of "very_positive"; the extractor non-deterministically
  returned whichever label iterated first from a set, not the more-specific one.
- **Impact:** multi-class F1 artificially degraded for tasks with overlapping label names.
- **Status:** 🟢 fixed 2026-07-08 — word-boundary re.search (longest label first) takes priority
  over substring; substring is a fallback only.

## B61 — entity_f1 used set intersection, missing duplicate entity mentions
- **Where:** `eval/metrics.py` (`entity_f1`)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** set intersection deduplicates: gold has "Apple" ORG twice, pred has it once →
  set gives TP=1, FN=0, F1=1.0 (wrong). Counter gives TP=1, FN=1, F1=0.667 (correct).
- **Impact:** NER eval overestimated recall for passages with repeated entity mentions.
- **Status:** 🟢 fixed 2026-07-08 — Counter multiset arithmetic replaces set intersection.

## B62 — math/code hard negatives trained model on wrong answers
- **Where:** `data/curriculum.py` (`synthesize_hard_negatives`)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** synthesize_hard_negatives for math_reasoning called teacher LLM to produce a
  plausible-but-wrong answer and stored it as {"prompt": ..., "response": wrong_answer}.
  SFTTrainer maximizes likelihood of whatever is in "response", so the model learned to output
  wrong answers for those prompts.
- **Impact:** math/code fine-tuned models were actively trained to produce incorrect outputs on
  the hard-negative examples — negative transfer rather than positive learning signal.
- **Status:** 🟢 fixed 2026-07-08 — math/code hard-negative synthesis removed. Gold examples
  are returned unchanged (CoT annotation in curate_node provides the relevant augmentation).

## B63 — downward probe was dead code; replaced with active probe node
- **Where:** `agent/nodes/iterate.py`, new `agent/nodes/downward_probe.py`, `agent/graph.py`
- **When:** 2026-07-09, final whole-branch review of pipeline hardening
- **How found:** the DAG-scanning probe in iterate_node could never fire — the DAG only holds
  current-model nodes (reset on escalation) and siblings share the base tier, so the
  `tier < current_tier` condition was unsatisfiable. Escalation only fires on stagnation, so a
  smaller model that cleared the bar would have terminated earlier, never escalated past.
- **Impact:** the minimum-resource "downward probe" feature was silently non-functional.
- **Status:** 🟢 fixed 2026-07-09 — new downward_probe graph node actively trains + evals the best
  one-tier-down candidate (via escalate._llm_choose_model) on the current dataset, using the honest
  quantized-eval path when quant is set, and adopts it if it clears the threshold. iterate routes
  iterate→downward_probe→END once per terminal model (guarded by state["downward_probe_done"],
  reset on escalation).


## B64 -- evaluate_node: None pending_weights_refs causes AttributeError
- **Where:** `agent/nodes/evaluate.py` line 28
- **When:** 2026-07-09, full-repo audit
- **How found:** `state.get('_pending_weights_refs', {})` returns stored None; .items() crashes.
- **Impact:** CRITICAL -- evaluate_node crashes when _pending_weights_refs is None.
- **Status:** Already fixed -- `state.get('_pending_weights_refs') or {}`.

## B65 -- evaluate_node: max() on empty sequence crashes
- **Where:** `agent/nodes/evaluate.py` line 72
- **When:** 2026-07-09, full-repo audit
- **How found:** no guard before max(scored, ...) when scored dict is empty.
- **Impact:** CRITICAL -- crashes graph with ValueError if no configs were evaluated.
- **Status:** Already fixed -- RuntimeError guard added before max() call.

## B66 -- evaluate_node: config_labels extracts config dicts not label keys
- **Where:** `agent/nodes/evaluate.py` line 131
- **When:** 2026-07-09, full-repo audit
- **How found:** `list(config_descriptions.values())` gives dict values; [0]['label'] then KeyErrors.
- **Impact:** HIGH -- curation log write crashes, suppressing all config logging.
- **Status:** Already fixed -- changed to `list(config_descriptions.keys())`.

## B67 -- generation scorer: invalid Anthropic model ID
- **Where:** `eval/scorers/generation.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** `claude-haiku-4-5-20251001` does not exist.
- **Impact:** CRITICAL -- every generation eval call fails; scorer non-functional.
- **Status:** Already fixed -- changed to `claude-haiku-4-5`.

## B68 -- generation scorer: math/code always uses LLM-as-judge
- **Where:** `eval/scorers/generation.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** score() had no task_type branching; LLM judge applied unconditionally.
- **Impact:** HIGH -- math and code tasks scored incorrectly.
- **Status:** Already fixed -- exact-match for math_reasoning, pass@1 for code_generation.

## B69 -- generation scorer: neg-slice gold labels always correct
- **Where:** `eval/scorers/generation.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** gold_labels_for_slice was ["correct"] * len(scores) for all examples.
- **Impact:** HIGH -- adversarial robustness measurement broken.
- **Status:** Already fixed -- gold conditional on whether example has a reference answer.

## B70 -- NER scorer: non-greedy regex truncates JSON on embedded ]
- **Where:** `eval/scorers/ner.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** `r'\[.*?\]'` stops at first ] inside entity text.
- **Impact:** HIGH -- NER returns [] for entities whose text contains ].
- **Status:** Already fixed -- greedy `r'\[.*\]'` with re.DOTALL; tries json.loads first.

## B71 -- NER scorer: failures uses set dedup inconsistent with entity_f1
- **Where:** `eval/scorers/ner.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** set comparison deduplicates repeated mentions; entity_f1 uses Counter.
- **Impact:** HIGH -- failure detection inconsistent with metric.
- **Status:** Already fixed -- Counter comparison.

## B72 -- metrics: acc_from_lists wrong denominator
- **Where:** `eval/metrics.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** divides by len(preds) not min(len(preds), len(golds)).
- **Impact:** HIGH -- silently wrong accuracy on mismatched-length inputs.
- **Status:** Already fixed -- `/ min(len(preds), len(golds))`.

## B73 -- metrics: boundary slice open-ended
- **Where:** `eval/metrics.py` per_slice_scores
- **When:** 2026-07-09, full-repo audit
- **How found:** bnd_preds has no upper bound; excess predictions inflate boundary.
- **Impact:** MEDIUM -- boundary accuracy distorted.
- **Status:** 2026-07-09 fixed -- added explicit upper bound n_pos + n_neg + n_bnd.

## B74 -- iterate: turn-budget check fires one iteration early
- **Where:** `agent/nodes/iterate.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** turns_used = state['iteration'] * 2 uses past cost not upcoming.
- **Impact:** MEDIUM -- terminates one iteration early with small budgets.
- **Status:** 2026-07-09 fixed -- changed to (state['iteration'] + 1) * 2.

## B75 -- iterate: initial_stop_threshold floor uses falsy or check
- **Where:** `agent/nodes/iterate.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** `or` operator treats 0.0 as absent, bypassing floor enforcement.
- **Impact:** MEDIUM -- threshold can go below floor if floor is 0.0.
- **Status:** 2026-07-09 fixed -- changed to `state['initial_stop_threshold']`.

## B76 -- iterate: intervention variable uninitialized at function scope
- **Where:** `agent/nodes/iterate.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** only assigned inside try or except; structurally latent UnboundLocalError.
- **Impact:** MEDIUM -- latent risk if except clause is modified.
- **Status:** 2026-07-09 fixed -- initialized to policy["intervention"] before try block.

## B77 -- escalate: dataset_version, last_intervention, last_curation not reset
- **Where:** `agent/nodes/escalate.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** escalate reset iteration/scores/dag but not dataset_version/last_intervention/last_curation.
- **Impact:** MEDIUM -- new model gets wrong dataset versioning and stale first-curate strategy.
- **Status:** 2026-07-09 fixed -- added dataset_version=0, last_intervention="data_rebuild", last_curation=None.

## B78 -- evaluate: DAG intervention field ignores LLM decision
- **Where:** `agent/nodes/evaluate.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** apply_iteration_policy always used for DAG even when LLM had already decided.
- **Impact:** MEDIUM -- DAG and curation log record wrong intervention type.
- **Status:** 2026-07-09 fixed -- DAG intervention reads state.get('last_intervention') first.

## B79 -- runner: lifetime_best_score missing from initial state
- **Where:** `tests/pipeline/run.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** lifetime_best_score in AgentState TypedDict but absent from initial_state dict.
- **Impact:** HIGH -- KeyError on first escalation call.
- **Status:** 2026-07-09 fixed -- added "lifetime_best_score": 0.0.

## B80 -- runner: last_intervention initialized to "" causes first curate to no-op
- **Where:** `tests/pipeline/run.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** curate_node falls through to else branch on empty string, skipping dataset build.
- **Impact:** HIGH -- first train crashes with None dataset_path.
- **Status:** 2026-07-09 fixed -- changed initial value to "data_rebuild".

## B81 -- lora_trainer: JSONL opened without encoding=utf-8
- **Where:** `training/lora_trainer.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** Windows default cp1252 mojibake on non-ASCII training data.
- **Impact:** MEDIUM -- any dataset with non-ASCII characters fails on Windows.
- **Status:** 2026-07-09 fixed -- added encoding="utf-8".

## B82 -- lora_trainer: classification format hard-subscripts ex['label']
- **Where:** `training/lora_trainer.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** KeyError if any example lacks label field; inconsistent with other branches.
- **Impact:** MEDIUM -- malformed training example crashes entire training run.
- **Status:** 2026-07-09 fixed -- changed to ex.get("label", "") and ex.get("text", "").

## B83 -- lora_trainer: CPU-only env crashes on fp16/bf16 setup
- **Where:** `training/lora_trainer.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** fp16=True on CPU-only; PyTorch does not support CPU fp16 training.
- **Impact:** MEDIUM -- training crashes in CI/test environments without GPU.
- **Status:** 2026-07-09 fixed -- gated on torch.cuda.is_available() first.

## B84 -- slm_helpers: cache eviction missing torch.cuda.empty_cache()
- **Where:** `training/slm_helpers.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** del old releases Python ref but CUDA holds GPU memory until explicit flush.
- **Impact:** MEDIUM -- GPU OOM persists despite LRU cache (B47 fix incomplete).
- **Status:** 2026-07-09 fixed -- unpacks model/tokenizer explicitly and calls torch.cuda.empty_cache().

## B85 -- task_analysis: initial_stop_threshold guard uses falsy not check
- **Where:** `agent/nodes/cold_start/task_analysis.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** `if not state.get("initial_stop_threshold")` overwrites legitimate 0.0 threshold.
- **Impact:** MEDIUM -- floor enforcement broken if planner returns 0.0.
- **Status:** 2026-07-09 fixed -- changed to `is None` check.

## B86-B87 -- downward_probe: triplicated LLM prompt and quant selection lost
- **Where:** `agent/nodes/downward_probe.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** filter_pool returns 3 entries per base model; LLM match resolves to first (unquantized base).
- **Impact:** CRITICAL -- probe always trains unquantized model regardless of quant level.
- **Status:** Already fixed in B63 implementation -- deduplicated with quant-preference logic.

## B88 -- downward_probe: artifact paths use relative root
- **Where:** `agent/nodes/downward_probe.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** os.path.join('artifacts', ...) relative to CWD at call time.
- **Impact:** HIGH -- artifacts written to wrong directory when CWD differs.
- **Status:** Already fixed in B63 implementation -- uses PROJECT_ROOT = Path(__file__).parents[2].

## B89-B90 -- bash_tool: PYTHONPATH injection broken on Windows
- **Where:** `agent/tools/bash_tool.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** export keyword invalid in cmd.exe; CWD frozen at import time.
- **Impact:** CRITICAL -- every bash() call fails on Windows.
- **Status:** Already fixed -- env dict approach; computed at call time.

## B91 -- delegate_task: response.content[0].text crashes on non-TextBlock
- **Where:** `agent/tools/delegate_task.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** ThinkingBlock/ToolUseBlock have no .text attribute.
- **Impact:** HIGH -- sub-agent responses with thinking blocks crash.
- **Status:** Already fixed -- uses next((b.text for b in response.content if b.type == 'text'), '').

## B92 -- query_traces: sample:N crashes on non-integer suffix
- **Where:** `agent/tools/query_traces.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** int(query.split(':')[1]) unguarded.
- **Impact:** HIGH -- malformed query crashes agent turn.
- **Status:** Already fixed -- try/except returning error string.

## B93 -- web_search: r.text[:500] TypeError when Exa returns None text
- **Where:** `agent/tools/web_search.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** Exa returns text=None for unscrapable results.
- **Impact:** HIGH -- web_search crashes mid-loop dropping subsequent results.
- **Status:** Already fixed -- (r.text or '')[:500].

## B94 -- android_pool: _q4_sibling understates peak_memory_mb
- **Where:** `config/android_pool.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** peak = int4_size_mb + 400 can be less than base model actual peak.
- **Impact:** HIGH -- device OOM when Q4 sibling passes filter that base correctly fails.
- **Status:** Already fixed -- max(base.peak_memory_mb, base.int4_size_mb + 400).

## B95 -- android_pool: power gate passes when avg_watts == 0.0
- **Where:** `config/android_pool.py` check_hardware_constraints
- **When:** 2026-07-09, full-repo audit
- **How found:** falsy check treats 0.0 as absent; broken sensor always passes power gate.
- **Impact:** HIGH -- broken sensor makes all models pass power gate.
- **Status:** Already fixed -- is not None check.

## B96 -- curate: surgical path wastes iteration when failures is empty
- **Where:** `agent/nodes/curate.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** synthesize_hard_negatives called with empty source; returns empty; dataset unchanged.
- **Impact:** HIGH -- silent no-op wastes entire iteration.
- **Status:** Already fixed -- early return with log message when failures is empty.

## B97 -- curate: production data_rebuild raises uninformative AttributeError
- **Where:** `agent/nodes/curate.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** eval_set is None in production mode; AttributeError on eval_set.task_type.
- **Impact:** HIGH -- confusing crash points to curriculum internals not actual cause.
- **Status:** Already fixed -- explicit RuntimeError guard with descriptive message.

## B98 -- live_confirm: M0 re-inference not implemented (design gap)
- **Where:** `agent/nodes/production/live_confirm.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** paper sec 2.6 requires re-running M0 on failures; original code only filtered by taxonomy label.
- **Impact:** Design gap -- failures not verified as systematic vs sampling artifact.
- **Status:** Already fixed -- implemented actual M0 re-inference via slm_helpers.infer.

## B99 -- live_confirm: cluster membership checked via substring match
- **Where:** `agent/nodes/production/live_confirm.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** cluster in str(t) matches any trace whose content contains cluster name.
- **Impact:** HIGH -- filter semantically vacuous.
- **Status:** Already fixed -- taxonomy_construct_node tags traces with cluster key; filter uses t.get('cluster').

## B100 -- eval_set: hard-subscript e["text"] KeyError on missing field
- **Where:** `data/eval_set.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** boundary_texts = {e["text"] for e in boundary} raises KeyError.
- **Impact:** HIGH -- eval set construction crashes for tasks without text field.
- **Status:** Already fixed -- changed to e.get("text", "") throughout.


## B101 -- curriculum: placeholder API keys in .env are truthy, defeating the Claude teacher fallback
- **Where:** `.env` (lines 3-4) + `data/curriculum.py:44,51,57,62` (`if ... and DEEPSEEK_API_KEY:` truthy gate)
- **When:** 2026-07-10, overnight validation campaign pre-submit check
- **How found:** `.env` shipped `OPENAI_API_KEY=your_key_here` / `DEEPSEEK_API_KEY=your_key_here` (placeholders, non-empty). `get_teacher_client()` gates specialist routing on `if task_type == "math_reasoning" and DEEPSEEK_API_KEY:` — a truthy check, not a validity check. Non-empty placeholder → truthy → the code builds `OpenAI(api_key="your_key_here", base_url="https://api.deepseek.com")` and calls it during CoT annotation, hitting an auth error, instead of falling back to the Claude/Haiku teacher.
- **Impact:** HIGH for tests 3 (GSM8K, task_type=math_reasoning) and 4 (ARC-Challenge, generation + math/science benchmark) — both route to a dead DeepSeek endpoint. Tests 1 (classification) and 2 (NER) are unaffected (they never request a specialist teacher).
- **Fix:** Blanked the two placeholder values in `.env` (`OPENAI_API_KEY=` / `DEEPSEEK_API_KEY=`) so they parse falsy and the intended Claude/Haiku CoT fallback triggers, matching the campaign premise ("no DeepSeek/OpenAI key available"). `.env` is gitignored so this is a working-tree-only change (not committed). Code hardening (treat `your_*` placeholders as unset) noted as a follow-up but not applied to avoid touching routing logic mid-campaign.
- **Status:** 🟢 fixed (config) — related to B26.

## B102 -- slurm: GSM8K/ARC scripts request gpu:a100 on ckpt-g2, which has no A100 nodes
- **Where:** `tests/pipeline/run_gsm8k_math.slurm:5`, `tests/pipeline/run_arc_challenge.slurm:5`
- **When:** 2026-07-10, overnight validation campaign submit
- **How found:** `sbatch` rejected both with "Batch job submission failed: Requested node configuration is not available". `sinfo -p ckpt-g2 -N -o "%N %G"` shows ckpt-g2 exposes only gpu:l40 / gpu:l40s / gpu:h200 — no a100 (A100 nodes live on the `ckpt`/`gpu-a100` partitions). The two L40 jobs (sms, ner) submitted fine.
- **Impact:** MEDIUM (environment/config, not pipeline logic) — tests 3 and 4 could not launch as written. A100 was chosen only "for faster generation-task training", not for correctness.
- **Fix:** Changed `--gres=gpu:a100:1` → `--gres=gpu:l40:1` on both scripts, keeping partition ckpt-g2 (matches the two working jobs). L40 (48GB) is ample for tier-0/1/2 models; only slower. Committed.
- **Status:** 🟢 fixed (config).

## B103 -- setup/slurm: venv guards check directory existence, not interpreter validity → zombie venv
- **Where:** `scripts/setup_gpu_env.sh:11` (`if [ ! -d .venv_gpu ]`) and all four `tests/pipeline/*.slurm` (`[ -d .venv_gpu ] || bash scripts/setup_gpu_env.sh`)
- **When:** 2026-07-10, overnight validation campaign — first launch of all four jobs
- **How found:** All four jobs FAILED at 00:00:00 with ExitCode 127. slurm .out shows the GPU allocated, then `line NN: python: command not found`. Root cause: `.venv_gpu/bin/python` symlinks to `~/.local/share/uv/python/cpython-3.11-.../bin/python3.11`, but that uv-managed interpreter directory was deleted (garbage-collected). The venv directory still existed, so both guards passed and setup was skipped, leaving a zombie venv whose every `python` invocation fails with 127. The 18G uv package cache was intact — only the interpreter was gone.
- **Impact:** BLOCKER — no pipeline job could run until fixed. Not a pipeline-logic bug; an environment/tooling robustness gap.
- **Fix:** (1) `setup_gpu_env.sh` now validates the interpreter (`.venv_gpu/bin/python -c ''`) and `rm -rf` + recreates on failure, self-healing zombie venvs. (2) All four slurm scripts changed their guard from `[ -d .venv_gpu ]` to `.venv_gpu/bin/python -c '' >/dev/null 2>&1` so a broken venv triggers a rebuild. (3) Rebuilt `.venv_gpu` from the intact cache (fast; only the ~30MB cpython interpreter re-downloaded). Committed.
- **Status:** 🟢 fixed (config/tooling).

## B104 -- hardware_research: data/devices.csv missing → ALL devices use Exa fallback (not just the intended one)
- **Where:** `data/devices.csv` (absent); consumed by `agent/nodes/cold_start/hardware_research.py:28,90-93`
- **When:** 2026-07-10, overnight validation campaign — pre-run artifact review
- **How found:** `ls data/devices.csv` → not found. `_lookup_local_db()` logs "Local device DB not found (run scripts/refresh_device_db.py to populate)" and returns None for EVERY device, so `research_device()` always advances to the Exa fallback (`source="exa"`). The four slurm headers assume a populated DB (e.g. run_conll_ner: "Samsung Galaxy A14 5G ... confirmed in devices.csv: 4096MB"; run_arc_challenge: "Redmi 9A ... confirmed in devices.csv: 2048MB").
- **Impact:** MEDIUM / needs-review. Not a crash (graceful None). Test 1 (Moto G Stylus 5G 2023) still gets spec_source="exa" as intended — its PASS criterion is unaffected. BUT tests 2 and 4 no longer exercise the local-DB path; their usable-RAM values are resolved by Exa snippets + Haiku instead of the deterministic CSV. Risk: if Haiku over-estimates usable RAM (e.g. Redmi 9A resolved >~2.5GB), test 4 may admit a larger tier and stop being "impossible"; if it over-estimates A14 RAM, test 2's tier-0/1 start (and thus escalation) may not trigger. Both are checkable from device_research.json after the runs.
- **Fix:** NOT applied mid-campaign — regenerating devices.csv while jobs are queued/running would be read non-deterministically across jobs and could flip test 1's spec_source away from "exa". Deferred: if test 2 or 4 fails due to mis-resolved RAM, populate the DB via scripts/refresh_device_db.py and rerun only those two.
- **Status:** 🟡 needs-review (environment/data gap).

## B105 -- run.py: orchestrator model never logged → cannot confirm which model drove a run
- **Where:** `tests/pipeline/run.py` (startup banner)
- **When:** 2026-07-10, overnight validation campaign — verifying the Haiku pin
- **How found:** Campaign requires confirming "run.log shows the orchestrator resolving to claude-haiku-4-5". No line in run.log, slurm .out, or any node logs the model name. The cost ledger (`agent/cost.py`) is model-agnostic — it applies a hardcoded Sonnet 4.6 rate to all token counts — so cost.json cannot distinguish Haiku from Sonnet either. There was no artifact-level evidence of which model actually ran.
- **Impact:** LOW (observability), but it blocks the campaign's model-confirmation check. (Separately: cost.json over-reports cost whenever a cheaper model like Haiku is used, since it always bills at Sonnet rates — noted, not fixed.)
- **Fix:** Added `log(f"orchestrator model: {config.ORCHESTRATOR_MODEL}  |  judge model: {config.JUDGE_MODEL}")` to the run.py startup banner. Picked up by the three still-pending jobs (NER/GSM8K/ARC). The already-running SMS job (36983575) predates the edit; its Haiku usage is confirmed via the slurm script's `export SLM_ORCHESTRATOR_MODEL=claude-haiku-4-5` (grepped pre-submit) which config.py reads with that exact env override.
- **Status:** 🟢 fixed (observability).

## B106 -- lora_trainer/slm_helpers: model load omits trust_remote_code=True → custom-code models crash
- **Where:** `training/lora_trainer.py:45,175`; `training/slm_helpers.py:108,114` (all `FastLanguageModel.from_pretrained` sites)
- **When:** 2026-07-10, overnight campaign — first successful model-load attempts (ARC job 36983578)
- **How found:** ARC selected `openbmb/MiniCPM4-0.5B` (tier-0, only model that fits 512MB) and crashed at load: "The repository openbmb/MiniCPM4-0.5B contains custom code which must be executed... Please pass the argument `trust_remote_code=True`." iterations=0, converged=false. Many pool models (MiniCPM4/5, Qwen3.5, Gemma3n) ship custom modeling code.
- **Impact:** BLOCKER for any task whose selected model needs remote code (tier-0 devices like ARC's Redmi 9A always land on MiniCPM4-0.5B). Llama models (SMS/GSM8K) are unaffected, which is why they loaded.
- **Fix:** Added `trust_remote_code=True` to all four `FastLanguageModel.from_pretrained` call sites (train, merge-for-quant, inference-adapter, inference-merged). Committed.
- **Status:** 🟢 fixed (verification pending on rerun).

## B107 -- android_pool: Qwen3.5-2B entry uses a -GGUF repo as trainable model_id
- **Where:** `config/android_pool.py:398` (now :400)
- **When:** 2026-07-10, overnight campaign — NER job 36983576
- **How found:** NER (A14, usable 2300MB) selected `unsloth/Qwen3.5-2B-GGUF` (largest tier that fits) and crashed at load: "Unrecognized model in unsloth/Qwen3.5-2B-GGUF. Should have a `model_type` key in its config.json." A GGUF (llama.cpp deployment) repo has no transformers config and cannot be LoRA-fine-tuned. It is the ONLY pool entry whose model_id is a GGUF repo; all siblings use base repos (`Qwen/Qwen3.5-0.8B`, etc.). The `model_id` field is documented as "HuggingFace model ID" with `quant` separately holding the GGUF variant.
- **Impact:** BLOCKER for any task selecting this model (mid-tier devices ~2-2.5GB usable, e.g. NER's A14).
- **Fix:** Changed `model_id` to the base repo `Qwen/Qwen3.5-2B` (verified to exist on HF as a transformers/safetensors repo; the GGUF's own metadata lists `base_model:Qwen/Qwen3.5-2B`). Committed.
- **Status:** 🟢 fixed (verification pending on rerun).

## B108 -- evaluate: GGUF quantization not gated on on-device mode → accuracy-only runs crash on missing llama.cpp
- **Where:** `agent/nodes/evaluate.py:57` (`if quant is not None:`) → `training/quantize.py:183`
- **When:** 2026-07-10, overnight campaign — SMS job 36983575 (the happy-path classification test)
- **How found:** SMS trained iteration 1 on Llama-3.2-3B, then crashed in evaluate_node: "Quantization failed... convert_hf_to_gguf not found. Clone llama.cpp and add it to PATH." The model selector set `quant="Q4_K_M"` on Llama-3.2-3B to fit 4GB RAM; evaluate_node then unconditionally built a real GGUF (needs the llama.cpp toolchain, which is not installed) even though the run is accuracy-only (on-device eval OFF, `HW_ONDEVICE_BACKEND=theoretical`). eval/harness.py only needs gguf_path for on-device (llama.cpp) inference; with gguf_path=None it scores the HF/LoRA weights via Unsloth.
- **Impact:** BLOCKER for ALL four tests — every task trains then evaluates, and any feasible model on a RAM-constrained phone carries a quant, so eval always tried (and failed) to quantize. This is the reason SMS/NER/GSM8K/ARC all produced empty scores.
- **Fix:** Gated the GGUF build on `config.HW_ONDEVICE_BACKEND != "theoretical"` in evaluate_node. In the default theoretical/accuracy-only mode the GGUF is skipped (gguf_path=None) and eval scores the HF weights via Unsloth infer_batch — matching the campaign's "accuracy only (theoretical hardware filter)" requirement. Committed. (llama.cpp remains uninstalled; that's only needed for actual on-device measurement, which is intentionally OFF.)
- **Status:** 🟢 fixed (verification pending on rerun).

## B109 -- lora_trainer: math_reasoning training crashes in Unsloth fused-CE (logits/labels batch mismatch); stale trl API
- **Where:** `training/lora_trainer.py:133-161` (trainer construction) → Unsloth `unsloth_zoo/fused_losses/cross_entropy_loss.py`
- **When:** 2026-07-10, overnight campaign — GSM8K job 36983577 (and expected for ARC 36983578 once it reaches training)
- **How found:** GSM8K trained on Llama-3.2-3B (loads fine) and crashed at the first optimizer step: `TorchRuntimeError: Dynamo failed... cross_entropy... Expected input batch_size (hint=3072) to match target batch_size (hint=3390)`. iterations=0. Classification (SMS, same model+trainer) trains fine — the difference is sequence length: math CoT completions are long/variable; classification labels are single short tokens.
- **Two-layer root cause:**
  1. **Stale trl API.** The code builds `args = TrainingArguments(...)` and calls `SFTTrainer(..., tokenizer=, dataset_text_field="text", max_seq_length=512, args=args)`. In the installed trl, `SFTTrainer` no longer accepts `tokenizer`/`dataset_text_field`/`max_seq_length`; text field + truncation live in `SFTConfig` (`dataset_text_field` default "text", `max_length` default **1024**, not 512; `max_seq_length` removed), and the tokenizer is `processing_class`. So the intended 512-token truncation is silently dropped and a plain `TrainingArguments` is passed where an `SFTConfig` is expected → SFT-specific fields fall to defaults.
  2. **Unsloth fused-CE mismatch.** Downstream, Unsloth's fused cross-entropy (vmap/Dynamo-compiled) receives logits and labels of different flattened lengths (3072 vs 3390) on long sequences — consistent with a `num_logits_to_keep`/logit-slicing path that diverges from full-length labels. This is inside Unsloth 2026.7.2 + transformers 5.5.0 internals.
- **Impact:** BLOCKER for math_reasoning tasks (tests 3 GSM8K & 4 ARC). Likely also at risk for NER (long PubMed abstracts) — to be confirmed on rerun. Classification (test 1, short sequences) is unaffected.
- **Fix:** NOT applied. Migrating the SFTTrainer construction to `SFTConfig(dataset_text_field=..., max_length=512, ...)` + `processing_class=tokenizer` is the correct API alignment, but (a) it may not resolve the Unsloth-internal fused-CE mismatch (which looks like an Unsloth/transformers version bug), and (b) it risks regressing the currently-working classification path — cannot be verified without a GPU rerun. Per campaign guardrails (risky/ambiguous → log, don't guess destructively) this is left for review. Candidate follow-ups for a human: (i) migrate to SFTConfig and pin/patch Unsloth's fused CE, (ii) disable Unsloth fused-CE loss for long-sequence tasks, (iii) pin trl/transformers/unsloth to a mutually-tested set.
- **Status:** 🟡 needs-review (BLOCKS tests 3 & 4).

## B110 -- run.py: concurrent jobs share one run dir (second-resolution timestamp) → clobber each other
- **Where:** `tests/pipeline/run.py:38-40`
- **When:** 2026-07-10, overnight campaign — batch rerun 36988105-108
- **How found:** All four jobs launched together started Python in the same second and computed identical `TS=20260710_105759`; `os.makedirs(..., exist_ok=True)` + `open(run.log, "w")` meant they shared one `logs/runs/<TS>/` and truncated/overwrote each other's run.log, scores.json, dag.json, device_research.json, and artifacts. Only ONE cold-start banner survived in the shared run.log.
- **Impact:** HIGH — concurrent validation runs produce corrupted, unattributable artifacts. (Per-job `logs/slurm/*-<jobid>.out` stay clean, so those remain the source of truth.)
- **Fix:** Append `SLURM_JOB_ID` (or PID off-cluster) to the timestamp and use `exist_ok=False`. Committed.
- **Status:** 🟢 fixed (code).

## B111 -- stack: `is_torch_fx_available` removed in transformers 5.5.0 → scaling-curve probe fails AND training crashes
- **Where:** third-party (`transformers.utils.import_utils`); surfaced via `agent/nodes/cold_start/scaling_curve.py` probe and the training forward path
- **When:** 2026-07-10, overnight campaign — all reruns
- **How found:** `ImportError: cannot import name 'is_torch_fx_available' from 'transformers.utils.import_utils'`. A dependency (peft/accelerate/unsloth chain) imports this symbol, which transformers 5.5.0 deleted. (a) scaling_curve's per-model probe catches it → "Probe failed for openbmb/MiniCPM4-0.5B" → "Too few probe points; defaulting to highest-capability". (b) During ARC(108) training it is raised uncaught at `[train] Iteration 1` → whole run crashes (iterations=0).
- **Impact:** HIGH — (1) breaks the scaling-curve probe, so model selection defaults to the LARGEST feasible model instead of starting small — this is why SMS/GSM8K/NER all jumped straight to Llama-3.2-3B / Qwen3.5-2B and undercuts the escalation design (test 2). (2) Crashes tier-0 training (ARC/MiniCPM4).
- **Fix:** NOT a code fix in this repo (the symbol is imported by third-party libs). Resolved by rebuilding the venv on a known-good version matrix (transformers/torch/unsloth/peft that mutually agree). Part of the planned rebuild.
- **Status:** 🟡 needs-review (env/version) — targeted by venv rebuild.

## B112 -- iterate: langchain_anthropic not installed → LLM per-iteration decision never runs (silent rule fallback)
- **Where:** `agent/nodes/iterate.py:135` (`from langchain_anthropic import ChatAnthropic`); missing from `scripts/setup_gpu_env.sh`
- **When:** 2026-07-10, overnight campaign
- **How found:** `[iterate] LLM call failed (ModuleNotFoundError("No module named 'langchain_anthropic'")), falling back to score-band rules`. The agentic LLM-driven iteration decision (design §) is never exercised; every iteration silently uses deterministic score-band rules.
- **Impact:** MEDIUM-HIGH — a core "agentic" behavior is silently disabled on every run. Not a crash (it's caught), but it means the pipeline is not doing what it claims.
- **Fix:** Add `langchain-anthropic` to setup_gpu_env.sh deps (installed on the venv rebuild). Committed to setup script.
- **Status:** 🟡 fix staged (dep added; effective after venv rebuild).

## B113 -- hardware/model download: Qwen/Qwen3.5-2B xet snapshot stalls (incomplete snapshot)
- **Where:** `unsloth_zoo/hf_xet_fallback.py` during model download for NER (Qwen/Qwen3.5-2B)
- **When:** 2026-07-10, overnight campaign — NER 36988106
- **How found:** `DownloadStallError: Download for 'Qwen/Qwen3.5-2B' returned an incomplete snapshot even with HF_HUB_DISABLE_XET=1 -- missing files, check your network connection` (preceded by an automatic retry with xet disabled).
- **Impact:** MEDIUM — partly environmental (network/xet on compute nodes). Can nondeterministically fail runs that must download a not-yet-cached model. Qwen/Qwen3.5-2B is the B107 model, not previously cached (only its GGUF was).
- **Fix:** Not a code bug. Mitigations for the rebuild: pre-download models to HF_HOME on a login node, and/or set `HF_HUB_DISABLE_XET=1` in the slurm scripts. Noted for follow-up.
- **Status:** 🟡 needs-review (env/network).

## B114 -- model pool: gemma-3n-e2b-it requires `timm` (not installed) → probe fails
- **Where:** third-party (`TimmWrapperModel`); pool entry `google/gemma-3n-e2b-it`; missing from `scripts/setup_gpu_env.sh`
- **When:** 2026-07-10, overnight campaign
- **How found:** `TimmWrapperModel requires the timm library but it was not found... Probe failed for google/gemma-3n-e2b-it`. gemma-3n is multimodal and pulls a timm vision wrapper.
- **Impact:** MEDIUM — gemma-3n-e2b-it can never be probed/selected/trained; contributes to the scaling-curve probe having too few points (compounds B111).
- **Fix:** Add `timm` to setup_gpu_env.sh deps (installed on rebuild). Committed to setup script.
- **Status:** 🟡 fix staged (dep added; effective after venv rebuild).

## B116 -- MiniCPM4-0.5B remote code incompatible with transformers 5.5.0 (tied-weights list vs dict)
- **Where:** `openbmb/MiniCPM4-0.5B` remote `modeling_minicpm.py:1163` (__init__ → post_init) → `transformers/modeling_utils.py:2472 get_expanded_tied_weights_keys`
- **When:** 2026-07-10, overnight campaign — ARC canary 36988759 (after the B111 shim)
- **How found:** With the B111 shim in place MiniCPM4 got past the `is_torch_fx_available` import, then crashed at model init: `AttributeError: 'list' object has no attribute 'keys'` — transformers 5.5.0's `get_expanded_tied_weights_keys` does `tied_mapping.keys() | tied_mapping.values()`, but MiniCPM4's remote code supplies `_tied_weights_keys` as a LIST (older format). Also breaks the scaling-curve probe for MiniCPM4.
- **Impact:** BLOCKER for ARC (test 4) — its 2GB Redmi 9A restricts the pool to tier-0, and MiniCPM4-0.5B is the selected tier-0 model. This is the SECOND MiniCPM4-vs-transformers-5.5.0 incompatibility (after B111); the model's remote code predates transformers 5.5.0's API and is multiply incompatible.
- **Fix:** NOT applied. Per-symbol/-attribute shims are whack-a-mole and fragile. Correct fixes: (a) rebuild the venv with a transformers version compatible with MiniCPM4's remote code (risks the newer Qwen3.5/gemma-3n model types), or (b) replace/remove tier-0 MiniCPM models in the pool with ones that load on transformers 5.5.0, or (c) pin a MiniCPM4 revision whose remote code targets transformers 5.5.0 if one exists. Needs a deliberate version/pool decision.
- **Status:** 🟡 needs-review (BLOCKS test 4; version/model-compat).

## B115 -- inference: train/serve prompt mismatch (chat template not applied at eval) → F1 ~0
- **Where:** `training/slm_helpers.py` `infer()` (used by `infer_batch`, the accuracy eval path)
- **When:** 2026-07-10, overnight campaign — SMS rerun 36988105 completed the full loop but scored F1=0.0000 (15/15 failures) every iteration
- **How found:** Training formats every example with the chat template (`lora_trainer.format_example` → `tokenizer.apply_chat_template(user+assistant)`), but `infer()` tokenized the RAW prompt string (`tokenizer(prompt, ...)`) with no chat template and no assistant generation prompt. The fine-tuned model therefore saw out-of-distribution input at eval and emitted text that never matched a label → classification F1 collapses to 0 (and NER/generation scoring is similarly degraded). This is a regression introduced when chat-template formatting was added to training (B30) without updating inference.
- **Impact:** HIGH — every run "completes" but produces meaningless ~0 scores, so no task can converge regardless of training quality. Masqueraded as a modeling/data problem.
- **Fix:** In `infer()`, when the tokenizer has a chat_template, wrap the prompt as a user turn and apply the template with `add_generation_prompt=True` before tokenizing — matching training's formatting. Committed. (The GGUF path `infer_batch_gguf` has the same raw-prompt issue but is only used for on-device eval, which is gated off in accuracy-only mode via B108; noted for later.)
- **Status:** 🟢 fixed (verification pending on rerun).
