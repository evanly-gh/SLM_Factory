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
- **Status:** 🔴 open — NER: label should use ambiguous entity spans or a different strategy (wrong
  entity type, not absent). Generation: `response` should be a Claude-generated plausible-but-wrong
  answer, not `None`.

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
- **Status:** 🔴 open — `train_node` must pass `task_type=state["task_type"]` to `slm_train()`.

## B34 — `curate_node` double-applies `gold_fraction`, halving gold count
- **Where:** `agent/nodes/curate.py` (line ~38-39)
- **When:** 2026-06-26, deep codebase audit
- **How found:** For `data_rebuild`, `curate_node` computes `n_gold_target = int(150 * 0.65) = 97` and
  passes it as `n_total=97` to `build_initial_curriculum()`. But `build_initial_curriculum()` has its own
  `gold_fraction=0.65` default, and internally computes `n_gold = int(97 * 0.65) = 63`. So the actual
  gold count is ~63, not the intended ~97. The 65:35 fraction is applied twice.
- **Impact:** gold dataset is 30% smaller than intended. The overall dataset composition is ~63 gold +
  53 hard = ~116 total at roughly 54:46 — more aggressive than the paper's 65:35 target.
- **Status:** 🔴 open — either pass `gold_fraction=1.0` to `build_initial_curriculum()` (since
  `curate_node` already computed the gold target), or pass `n_total=150` and let the function handle
  the split internally. Don't split in both places.

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
- **Status:** 🔴 open — for non-SMS tasks, boundary detection should be LLM-driven (ask Claude to
  identify which examples are near the decision boundary) or use embedding-based nearest-neighbor
  selection between classes.

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
