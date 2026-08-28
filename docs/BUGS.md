# BUGS.md — running bug log

A living record of bugs found while building/validating the SLM Factory pipeline.
Newest entries appended at the bottom. Each entry notes **where** it lives, **when** it
was discovered, **how** it was found, and current **status**.

Status legend: 🔴 open · 🟢 fixed · 🟡 suspected/unconfirmed · ⚪ design gap (not a crash)

> **Read [Status reconciliation — 2026-07-29](#status-reconciliation--2026-07-29) first.**
> It is a code-verified sweep of every 🔴/🟡/⚪ entry and is **authoritative where it
> disagrees with an older entry's inline `Status:` line**. 15 entries moved to fixed, 3 were
> superseded by deliberate design changes, and B199–B207 were opened.

---

## Overnight validation campaign — 2026-07-10 (summary)

Ran the four `tests/pipeline/*.slurm` pipeline tests on the SLURM GPU cluster
(orchestrator pinned to Haiku; on-device HW eval OFF; only Anthropic + Exa keys).
Every early run crashed; root-causing surfaced a **19-bug chain (B101–B119)** across
environment, model-loading, training, download, scoring, and data acquisition. Status per test:

| # | Test | Result | Blocking issue |
|---|------|--------|----------------|
| 1 | **Financial sentiment (classification)** | 🟢 **pipeline functional, ✗ not converged** — trains/evaluates/iterates with real scores; F1 0.111→**0.697** best, then plateaus ~0.66 and exhausts budget (threshold 0.82). `spec_source="exa"` ✓, `task_type=classification` ✓, Haiku ✓. Mechanics validated; convergence capped by data/curation quality + degraded iterate decision (B117) | data/curation + B117 |
| 2 | Biomedical NER | 🟢 **superseded — completed** | **B113** was the blocker; NER later ran end-to-end for 44.8 h (`slm-ner-l40s-37531245`, best span-F1 0.8263). B113 remains a live infra risk, not a blocker — see its entry. |
| 3 | GSM8K math | 🟢 **superseded — completed** | **B119 fix applied** (real benchmark loaders); math later ran end-to-end (`slm-math-l40s-37576194`). |
| 4 | ARC-Challenge (impossible) | 🟢 **superseded** | **B116 fix applied** — tier-0 pool entry swapped from MiniCPM4-0.5B to Qwen3-0.6B. |

> **⚠ This table is a snapshot of 2026-07-10 and is retained as campaign history only.**
> Rows 2–4 were marked 🔴 blocked at the time; every one of those blockers has since had a fix
> applied (B116, B119) or been demoted to an infra risk (B113), and both the NER and math
> pipelines have since completed full multi-day runs. For current status always read
> [Status reconciliation — 2026-07-29](#status-reconciliation--2026-07-29).

**Bottom line (as of 2026-07-10):** the crash chain was fixed and the core loop
(research → select → curate → train → eval → iterate) was proven functional on classification
with a real learning curve. No test met its strict PASS criteria at that date.

**Fixed & committed:** B101 (placeholder .env keys), B102 (A100→L40 partition), B103
(zombie-venv guards), B105 (log orchestrator model), B106 (`trust_remote_code`), B107
(GGUF→base model_id), B108 (gate GGUF quant on on-device mode), B109 (SFTConfig
truncation), B110 (per-job run dir), B111 (`is_torch_fx_available` shim), B112/B114
(missing `langchain-anthropic`/`timm`), B115 (chat-template train/serve parity), B118
(math final-answer extraction).

**Open / needs human decision (in priority order):**
- **B119** (blocks GSM8K/ARC math + likely NER data quality): benchmark "data" is Exa-scraped
  repo/webpage metadata — no real questions, no gold answers. Structured benchmarks must be
  loaded from source (`datasets.load_dataset`, etc.). Deepest issue; masked earlier because
  classification (FinancialPhraseBank) happened to get usable data.
- **B116** (blocks ARC): MiniCPM4/transformers version conflict — needs a version-matrix
  rebuild or a tier-0 pool-model swap. Per-symbol shims are whack-a-mole.
- **B113** (blocks NER): compute-node download reliability for the large multimodal
  Qwen3.5-2B — needs a warmed shared cache, the `unsloth/` mirror, or a retry wrapper.
- **B117**: resolved — iterate is now a single bounded tool-free decision with one JSON-only
  reask, so tool-round exhaustion is impossible.
- **B104** (`data/devices.csv` missing → all hardware research via Exa; harmless but off-spec).
- **B111** relies on a runtime shim; **B109** truncates to 512 rather than fixing Unsloth internals.

**Not done / caveats:** on-device HW eval left OFF as instructed; DeepSeek/GPT-4.1 CoT
teachers unavailable (keys absent) so CoT is Haiku-annotated (expected). No run was
marked "passed" that wasn't confirmed from its per-job `logs/slurm/*.out` (the shared
`logs/runs/<ts>/` artifacts were unreliable until B110 was fixed).

---

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
| B20 | 🟢 | curate | legacy fine-grained decision hint ignored by positive synthesis |
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
| B77 | 🟢 | escalate | escalation mixed stale action state with carried dataset metadata |
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
| B96 | 🟢 | curate | legacy positive-synthesis path silently no-ops when anchors are empty |
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
  `{intervention, hypothesis, hyperparameter_changes?, data_rebuild?}`. Score-band rules remain
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

## B20 — fine-grained decision hint ignored in positive synthesis
- **Where:** `agent/nodes/curate.py`; `data/curriculum.py` (`synthesize_hard_negatives`)
- **When:** 2026-06-26, post-B2 code review
- **How found:** the decision carried a failure-pattern description, but `curate_node` never
  passed it to `synthesize_hard_negatives`, so the guidance was silently discarded.
- **Impact:** positive synthesis was blind to the diagnosis; generated generic hard negatives
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

## B45 — Generation scorer misnames metric
- **Where:** `eval/scorers/generation.py`, `eval/judge_client.py`
- **When:** 2026-06-26, deep codebase audit
- **How found:** two issues:
  1. The scorer returns the average LLM-judge score (0.0-1.0) in the `f1` field. This is semantically
     wrong — it's a judge score, not an F1 metric. Downstream code treats it as F1.
  2. Originally each eval example required a separate sequential Anthropic API call.
- **Impact:** the remaining issue is the misleading metric name.
- **Status:** 🟡 partially fixed — judging now requires local Qwen3.6, deduplicates repeated
  triples, scores unique requests concurrently in order, and costs `$0`; strict failures abort
  instead of becoming model scores. Renaming `f1` to `judge_score` or `metric` throughout remains
  a separate compatibility migration.
- **Status update 2026-07-22:** The paid/sequential judge defect is superseded by B173's
  strict local-only judge, durable cache, locality checks, and fail-fast semantics. The
  metric-field rename remains open; TP4 co-location job 37486488 is runtime validation
  pending, not a code fix.

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
- **Status update 2026-07-22:** 🟢 superseded by B189. LRU eviction and
  `empty_cache()` were insufficient for hidden Unsloth/Trainer references; disposable CUDA
  workers now provide the hard cleanup boundary. Job 37430554 returned to 0 MiB after all
  100 × 4 GiB allocation cycles.

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

## B77 -- escalate: action state and carried dataset metadata were conflated
- **Where:** `agent/nodes/escalate.py`
- **When:** 2026-07-09, full-repo audit
- **How found:** escalation needs a fresh action decision while retaining the exact carried
  dataset identity and composition.
- **Impact:** MEDIUM -- resetting dataset metadata breaks lineage; retaining the prior action
  plan can repeat a rebuild.
- **Status:** updated by structured data rebuild -- preserve dataset version/composition,
  set `last_intervention="data_rebuild"`, and clear only the pending rebuild plan/identity.

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

## B96 -- curate: positive synthesis wastes iteration when anchors are empty
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
- **Fix:** Applied in the current trainer. It uses `SFTConfig(max_length=...)` and `processing_class`, pre-tokenizes exact non-thinking chat turns, and supplies explicit completion masks through a Trainer-compatible collator: prompt labels are `-100`, assistant/NER/CoT/code labels remain trainable. Overlength rows fail before SFT instead of silently truncating. If the early-stop/checkpoint path throws after optimizer work, only an immutable error summary leaves the `except` suite; the exception/traceback is out of scope before the failed trainer/model references are cleared, GC/CUDA cache cleanup runs, and a fresh base+LoRA stack reloads all original rows. This prevents both partially trained fallback and traceback-held tensors.
- **Status:** 🟢 code + CPU mask/cleanup ordering tests fixed; a bounded GPU readiness run is still required to verify the third-party fused-CE path on the installed stack.

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

## B113 (update) -- HF_HUB_DISABLE_XET did NOT resolve the Qwen3.5-2B download stall
- **When:** 2026-07-10, clean round NER 36988856
- **Finding:** With `export HF_HUB_DISABLE_XET=1` in the slurm script, NER STILL failed: `DownloadStallError: ... returned an incomplete snapshot even with HF_HUB_DISABLE_XET=1 -- missing files`. unsloth_zoo already retries with xet disabled, so the env var is redundant. Llama-3.2-3B downloads fine on the same compute nodes, so this is specific to `Qwen/Qwen3.5-2B` (a large multimodal repo with image/video preprocessors) — a compute-node network/download-reliability issue, not pipeline logic.
- **Status:** 🔴 open (infra). Options for a human: (a) warm the shared HF cache for this model from a node with reliable network (standard HPC practice — the model weights are infra, distinct from the loop's decisions); (b) switch the B107 pool entry to the `unsloth/Qwen3.5-2B` mirror which may fetch/load more reliably under Unsloth; (c) add a bounded download-with-retry wrapper before training. NER (test 2) is blocked on this. NOT re-pre-staged per user direction to keep the loop autonomous.

## B117 -- iterate: legacy tool loop exhausted before producing a decision
- **Where:** `agent/nodes/iterate.py` (former decision loop)
- **When:** 2026-07-10, clean round GSM8K 36988857 (after B112 installed langchain-anthropic)
- **How found:** `[iterate] LLM call failed (RuntimeError('LLM exhausted tool rounds without producing a final JSON decision')), falling back to score-band rules` on every iteration. With langchain-anthropic now installed the LLM call runs, but Haiku doesn't emit a final decision within the allotted tool rounds.
- **Impact:** MEDIUM — the agentic per-iteration decision still degrades to deterministic score-band rules (as it did under B112, now for a different reason). Not a crash. Likely Haiku being too weak for the tool-calling protocol, or the tool-round cap being too low.
- **Status:** 🟢 superseded by the strict structured-rebuild decision boundary: one
  tool-free call plus at most one JSON-only reask, with no shell/file/web execution.

## B118 -- scorer/generation: math_reasoning used full-string exact match → always 0.0 (even baseline)
- **Where:** `eval/scorers/generation.py` `_exact_match`
- **When:** 2026-07-10, clean round GSM8K 36988857
- **How found:** GSM8K trained fine (B109 fixed) but scored F1=0.0000 every iteration INCLUDING the zero-shot baseline — impossible for a base Llama-3.2-3B on GSM8K if scoring were correct. Root cause: `_exact_match` compared the model's ENTIRE chain-of-thought generation to the gold answer by normalized string equality, which never matches (gold is often '#### 42'; the model emits a paragraph ending in the number).
- **Impact:** HIGH — math_reasoning tasks can never score above 0, so GSM8K/ARC could never converge regardless of model quality. Masqueraded as the "Haiku CoT is too weak" config-limitation.
- **Fix:** Extract the final answer from both gold and prediction (explicit '#### N' / 'answer: N' markers, else the last number; normalize $, commas, trailing .0) and compare those. Unit-tested on 7 GSM8K-style cases. Committed.
- **Status:** 🟢 fixed (verification pending on a math rerun).

## B119 -- data acquisition: benchmark datasets scraped as web/repo METADATA via Exa, not the actual data
- **Where:** the web-acquire / curation path that builds train + eval sets from a benchmark name
- **When:** 2026-07-10, clean round GSM8K 36988857 / 36989010
- **How found:** GSM8K trained (B109 ✓) and scored 0.0 even after the B118 scorer fix. Inspecting the run's `artifacts/eval_set.json` and `dataset_v1.jsonl`:
  - eval example: `text="openai/grade-school-math. # Repository: ... Stars: 1437 ..."`, `label="math_reasoning"`, and **no `answer` field**.
  - train example: `text="README.md at master · openai/grade-school-math ..."`, `cot_reasoning="... This is a README.md file ..."`.
  The pipeline Exa-searched the GSM8K *repository/webpage* and used the returned snippets as "examples" — so the model is trained to reason about README files and the eval has no gold answers. GSM8K/ARC (and likely BC5CDR NER) can never score meaningfully.
- **Impact:** CRITICAL for structured benchmarks — no gold answers, no real questions. Classification (FinancialPhraseBank) happened to get usable data (SMS reached F1 0.70), so this had been masked. This is the deepest blocker for the generation/NER tasks.
- **Fix:** NOT applied (design-level). Structured benchmarks should be loaded from their source (e.g. `datasets.load_dataset("openai/gsm8k")`, BC5CDR, ARC) with real question/answer fields, reserving Exa for genuinely open-web tasks. Needs a deliberate acquisition redesign + a check that eval examples carry a gold `answer`.
- **Status:** 🔴 open (design) — blocks meaningful GSM8K/ARC (and NER) evaluation regardless of the model/scorer fixes.

## B119 (fix applied) -- load real benchmark datasets instead of Exa-scraping
- **Fix:** Added `load_benchmark_dataset()` in `data/loaders/web_acquire.py` and call it first in `acquire_dataset()`. Known benchmarks load their REAL data (verified on the login node):
  - GSM8K → `openai/gsm8k` (question + gold final number + real CoT solution)
  - FinancialPhraseBank → `ChanceFocus/flare-fpb` (sentence + sentiment label)
  - ARC-Challenge → `allenai/ai2_arc` (question + correct-choice text)
  Unknown benchmarks (e.g. BC5CDR) still fall back to the Exa path. Datasets are small parquet downloads (fine on compute nodes with HF_HUB_DISABLE_XET=1).
- **Status:** 🟢 fixed for GSM8K/FPB/ARC (rerun to confirm end-to-end scores). BC5CDR NER still uses Exa+Claude annotation.

## B116 (fix applied) -- swap ARC's tier-0 model from MiniCPM4-0.5B to Qwen3-0.6B
- **Fix:** MiniCPM4-0.5B (the only model fitting ARC's 512MB device) is multiply incompatible with transformers 5.5.0 (B111 is_torch_fx_available shimmed, B116 tied-weights list-vs-dict, likely more) — per-symbol shims are whack-a-mole. Replaced the tier-0 pool entry with `unsloth/Qwen3-0.6B` (peak ~500MB, tier 0), a standard safetensors/qwen3 model natively supported by Unsloth/transformers 5.5.0. Preserves the ARC test intent (a tiny tier-0 model that can't reach ARC-Challenge SOTA → impossible → graceful escalate/terminate).
- **Status:** 🟢 fix applied (rerun to confirm it loads/trains). MiniCPM5-1B (tier 1) shares MiniCPM's remote-code family and is likely similarly broken if ever selected — noted.

## B120 -- scaling_curve (startup model selection): fell back to LARGEST model, breaking start-small-and-escalate
- **Where:** `agent/nodes/cold_start/scaling_curve.py` (final fallback)
- **When:** 2026-07-10, reviewing GSM8K 36988857 (user observation)
- **How found:** The startup model-selection probes 3 candidates (train+eval each) and fits an accuracy-vs-log(params) curve. When probes fail (they do often — e.g. gemma-3n "mat1/mat2 shapes", MiniCPM4 tied-weights) or no variant is predicted to clear the threshold, it selected `max(feasible, key=params)` — the LARGEST model. So the run jumped straight to the biggest model (e.g. Llama-3.2-3B), and the main loop's escalate_node (grow the model when it stalls — "find the next best model") never fired. Choosing the strongest model is not startup's job; it contaminated the start-model decision with growth logic that belongs to the main loop.
- **Impact:** MEDIUM (design) — defeated the start-small-and-escalate design the escalation tests (NER, ARC) exercise, wasted compute on the largest model, and prevented escalation from ever being demonstrated.
- **Fix:** Startup now starts SMALL on the no-qualifier fallback (lowest peak-RAM feasible model) and defers growth to escalate_node in the main loop. The "smallest variant predicted to clear threshold" happy path is unchanged. Keeps model *selection* (startup) cleanly separated from model *growth* (main loop).
- **Status:** 🟢 fixed.

## B122 -- state: in-place list appends not persisted → scores froze, stagnation never fired, run looped forever
- **Where:** `agent/nodes/evaluate.py` (`state["scores"].append(...)`, `state["dag"].append(...)`); `agent/state.py` (channels have no LangGraph reducer)
- **When:** 2026-07-10, SMS 36989310 (ran 13+ iterations without terminating)
- **How found:** SMS trained/evaluated on real data (F1~0.768) but the trajectory froze at exactly `['0.768','0.768','0.765']` from iteration 3 onward; "Stagnation detected"/"ESCALATE" fired 0 times, so it looped on data_rebuild indefinitely. `scores` is a plain `list[float]` (no reducer); evaluate_node mutated it IN PLACE, so the channel value kept the same object identity and LangGraph did not persist the change past the first few super-steps. `_is_stagnant` reads this window, so stagnation could never trigger.
- **Impact:** HIGH — no run could ever terminate via stagnation/escalation; every non-converging run burned its full SLURM time budget looping.
- **Fix:** Assign a NEW list (`state["scores"] = list(...) + [x]`, same for `dag`) so the channel change is detected and persisted. (Proper long-term fix: annotate these channels with an `operator.add` reducer and have nodes return partial updates.)
- **Status:** 🟢 fixed (verification on rerun).

## B123 -- android_pool: Qwen3.5 family is multimodal → text-only LoRA crashes ("Incorrect image source")
- **Where:** `config/android_pool.py` (`Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`)
- **When:** 2026-07-10, NER 36989327 crashed after B113 fixed its download
- **How found:** NER selected Qwen/Qwen3.5-2B; after loading, training crashed with `ValueError: Incorrect image source. Must be a valid URL ... Got <|im_start|>user`. The Qwen3.5 family is natively multimodal (image-text-to-text); Unsloth routes it through a vision processor that rejects the text chat template.
- **Impact:** BLOCKER for any task selecting a Qwen3.5 model (NER's mid-tier device did). Same class as gemma-3n (B121).
- **Fix:** Removed both Qwen3.5 entries from the selectable pool; text-only Qwen3-0.6B (tier 0) remains. Pool is now 24 text-only models that load/train on the stack.
- **Status:** 🟢 fixed.

---

## Log-readability + control-flow pass — 2026-07-15 (B124–B128)

Driven by a review of ARC-Challenge run 36989407 (`logs/slurm/slm-arc-challenge-36989407.out`).

## B124 -- endless rollback→re-train loop; regression never triggered a different action
- **Where:** `agent/graph.py` (`rollback → train` edge); `agent/nodes/iterate.py`; `agent/nodes/rollback.py`
- **When:** 2026-07-15, ARC 36989407 + GSM8K 36989406 (user observation)
- **How found:** On every regression the graph went `evaluate → rollback → train`, re-training the SAME dataset + hyperparameters. Training is (near-)deterministic, so it reproduced the same regressing score and rolled back again. ARC churned ~7 iterations and GSM8K ~23, each pinned at a fixed best score, until the turn/recursion budget ran out. `should_rollback` also pops the regressing score, so the stagnation window never filled and escalation never fired.
- **Impact:** HIGH — any model that beats its best once then can't again burns the entire budget oscillating; makes no progress and never terminates cleanly.
- **Fix:** (1) Re-routed `rollback → iterate` so a regression forces a *different* next action (data_rebuild with a rotated seed / hyperparameter / escalate / terminate). (2) Added a stall backstop in `iterate_node` — escalate when `consecutive_no_improvement` (set in evaluate, not popped by rollback) reaches it. The original fix used 4; the current env-overridable default is `MAX_STALL_EVALS = 50`, matching the 50-score chronological-gain window. Escalation promotes to a bigger model if one fits, else terminates. Updated `PIPELINE.md` invariant #3 and the loop diagram.
- **Status:** 🟢 fixed.

## B125 -- data_rebuild regenerated a byte-identical gold slice every iteration
- **Where:** `agent/nodes/curate.py`; `data/curriculum.py` (`build_initial_curriculum(seed=42)`)
- **When:** 2026-07-15, ARC 36989407 (user observation: "synthesizes the exact same data")
- **How found:** `build_initial_curriculum` used a fixed `seed=42`, and curate seeded hard-negative sourcing off the unshuffled train list, so repeated `data_rebuild` rounds produced the same curriculum — feeding B124's loop with identical retrains.
- **Impact:** MEDIUM — repeated rebuilds could not diversify data, so re-training could not escape a plateau.
- **Fix:** curate now rotates the seed per rebuild (`seed = 42 + dataset_version`) for both gold selection and hard-negative source shuffling. NOTE: when the acquired corpus is smaller than the gold target (e.g. ARC's 150 real examples < 650 target), the gold slice is necessarily identical regardless of seed — that is a data-availability cap (B119 family), now surfaced by an explicit warning (B126), not a shuffling bug.
- **Status:** 🟢 fixed (diversifies whenever the corpus has surplus).

## B126 -- curate: no provenance logging; gold-cap was silent
- **Where:** `agent/nodes/curate.py`; `agent/nodes/cold_start/eval_setup.py`; `data/loaders/web_acquire.py`; `agent/state.py`
- **When:** 2026-07-15 (user question: "why only 150 gold vs target 650? where does data come from?")
- **How found:** curate logged `gold_target=650` then `Gold examples built: 150` with no explanation. The 150 cap is simply the number of REAL examples acquired (`load_benchmark_dataset` caps ARC/GSM8K at `max_train=150`); hard negatives are LLM-synthesized, not scraped. None of this was visible in the log.
- **Fix:** Threaded a `data_source` provenance string from `acquire_dataset`/`load_benchmark_dataset` (via a `meta` out-param) into `state["data_source"]`, and curate now logs: the gold data source, the number of acquired examples, that hard negatives are LLM-synthetic (contrastive 2-for-1), the CoT teacher, and a ⚠ warning when gold is capped below target with the reason + how to raise the cap.
- **Status:** 🟢 fixed (observability). Underlying small-corpus cap is the B119-family design limit.

## B127 -- ML-stack log flood buried the signal
- **Where:** new `agent/logging_setup.py`; `tests/pipeline/run.py`; `training/lora_trainer.py`; `training/slm_helpers.py`
- **When:** 2026-07-15 (user: "these lines are flooding the logs")
- **How found:** Each run emitted hundreds of transformers deprecation aliases (`Accessing is_flash_linear_attention_available from .models.<X>.image_processing_<X>` — one per image processor, triggered when Unsloth Zoo patches every module), per-load `Loading weights:` bars, `Unsloth: Tokenizing [...]` bars, per-generate `Both max_new_tokens (=256) and max_length (=40960)` warnings, and FutureWarnings.
- **Fix:** `set_ml_env()` at the top of `run.py` sets `TRANSFORMERS_VERBOSITY=error` (+ `DATASETS_VERBOSITY`, `HF_HUB_DISABLE_PROGRESS_BARS`, `TOKENIZERS_PARALLELISM`) BEFORE any heavy import, and filters Future/Deprecation warnings. `quiet_ml_logging()` (called after unsloth import in train/merge/infer) additionally calls `transformers.logging.disable_progress_bar()` + `datasets.disable_progress_bars()`. `SFTConfig(disable_tqdm=True)` drops the training progress bar while KEEPING the periodic `{'loss':..., 'grad_norm':..., 'learning_rate':..., 'epoch':...}` lines (printed by PrinterCallback, not gated by verbosity).
- **Status:** 🟢 fixed.

## B128 -- infer: max_new_tokens vs generation_config.max_length conflict (per-call warning)
- **Where:** `training/slm_helpers.py` (`infer`)
- **When:** 2026-07-15 (user question about the "two max tokens")
- **How found:** We always pass an explicit `max_new_tokens` (50 classification / 256 generation), but chat models like Qwen3 also ship `generation_config.max_length=40960`; with both set transformers logs "Both max_new_tokens and max_length seem to have been set" on EVERY generate() (i.e. once per eval example).
- **Fix:** After loading, set `model.generation_config.max_length = None` so `max_new_tokens` (a cap on NEWLY generated tokens — the correct control for our short answers) is the single length knob. This is a root-cause fix, not just a warning suppression.
- **Status:** 🟢 fixed.

## B129 (improvement) -- iterate: malformed first response fell straight to score-band rules
- **Where:** `agent/nodes/iterate.py` (`_llm_iterate`)
- **When:** 2026-07-15 (user question: "what is 'LLM exhausted tool rounds'?")
- **How found:** the former multi-round decision path could consume its local exploration
  budget without returning JSON.
- **Fix:** current code makes one tracked tool-free decision and at most one fresh
  JSON-only reask. Attempted tool blocks are never executed.
- **Status:** 🟢 fixed; score-band fallback remains only for genuine decision/reask failure.

---

## NER task review — 2026-07-15 (B130–B133)

Driven by a review of CoNLL/biomedical-NER run 36989405 (`logs/slurm/slm-conll-ner-36989405.out`).

## B130 (B104 resolved) -- local device DB missing → every run went to Exa; short model codes never matched
- **Where:** `data/devices.csv` (absent); `agent/nodes/cold_start/hardware_research.py` (`_lookup_local_db`)
- **When:** 2026-07-15 (user: "I need to refresh the database")
- **How found:** Every run logged `Local device DB not found (run scripts/refresh_device_db.py to populate)` and fell back to Exa for hardware research. `refresh_device_db.py` needs the Kaggle CLI + credentials + network, unavailable on the compute nodes. Separately, the matcher required ≥2 keyword hits but dropped tokens ≤2 chars, so single-distinctive-token phones (Pixel 6a → only "pixel"; Redmi 9A → only "redmi") could never reach the threshold even with a DB.
- **Fix:** (1) Committed a curated `data/devices.csv` (30 common phones incl. all four test devices) in the schema `_lookup_local_db` expects (device, ram_mb, storage_mb, chipset) — the local DB now resolves without Kaggle. (`refresh_device_db.py` still performs a full Kaggle refresh when creds exist.) (2) Improved keyword extraction to keep alphanumeric MODEL CODES containing a digit ("9a", "6a", "5g", "a14", "2023") and strip punctuation. Verified all four test devices now match the correct row locally.
- **Status:** 🟢 fixed. Full 8k-device Kaggle DB still requires `pip install kaggle` + `~/.kaggle/kaggle.json`, then `python scripts/refresh_device_db.py`.

## B131 -- NER curriculum: acquired data tiny; gold reported as 0; dataset ~100% synthetic
- **Where:** `data/loaders/web_acquire.py`; `agent/nodes/curate.py`
- **When:** 2026-07-15, NER 36989405 (user: "total=32 train=22 test=10 ... 0 gold?!")
- **How found:** NER named a real benchmark (BC5CDR) but there was no benchmark loader for it, so it Exa-scraped 32 generic web docs (22 train / 10 test) and Claude-annotated entities — and yes, iteration 1 trained on that. Curate then reported `Gold: 0 (0.0%) / Hard: 100%`: the composition was computed as `n_hard = min(hard_synth_count, total); n_gold = total − n_hard`, so whenever the synthetic count exceeded the QC-filtered total (64 > 25) it underflowed gold to 0 — a reporting bug, and the dataset really was mostly synthetic because the full hard target (105) was generated against only ~14 gold.
- **Fix:** (1) Added real NER benchmark loaders (`_load_ner_benchmark`) for BC5CDR (`tner/bc5cdr`) and CoNLL-2003 (`eriktks/conll2003`) that convert token+BIO tags into `{text, entities}` spans; defensive (falls back to Exa if a dataset is unreachable). (2) Raised acquisition caps (`max_train 150→300`, `max_test 25→80`) and the Exa fallback `n_per_label` (16→24) so the curriculum + eval set are larger. (3) Fixed composition accounting with provenance tags (`_slice`) counted AFTER quality controls (stripped before the JSONL is written). (4) Scaled the hard-negative target to the ACTUAL gold count to preserve the ~65:35 ratio when gold is scarce, instead of producing a 100%-synthetic set.
- **Status:** 🟢 fixed (real-benchmark load is best-effort/defensive; Exa fallback improved regardless).

## B132 -- NER eval truncated at 50 tokens → empty predictions → F1=0 on every run
- **Where:** `eval/harness.py`
- **When:** 2026-07-15, NER 36989405 (user: "how is it 0 accuracy on every run?")
- **How found:** `max_new_tokens = 256 if generation else 50` gave NER only 50 tokens. A JSON entity list for a passage can exceed that, and the selected model (DeepSeek-R1-Distill-Qwen-1.5B — a *reasoning* model whose notes literally say "Do not use for classification/NER") emits a long `<think>` preamble before any JSON, so 50 tokens produced no parseable entity list → all predictions `[]` → entity-F1 0. Compounded by poor synthetic gold (Claude-annotated generic web text, exact-(text,type)-match metric) and a tiny 10-example eval set.
- **Fix:** Give NER the same 256-token budget as generation. (The model-family mismatch — a math/reasoning model chosen for NER — is a separate model-selection concern; noted, not yet auto-corrected.)
- **Status:** 🟢 fixed (token budget). Model-selection preference for NER remains a follow-up.

## B133 (not a bug) -- NER escalation "no feasible models above tier 2" is correct
- **Where:** `agent/nodes/escalate.py` + `config/android_pool.py`
- **When:** 2026-07-15, NER 36989405 (user question)
- **Explanation:** Galaxy A14 5G has 4 GB RAM → hardware_research resolved ~2300 MB usable. `filter_pool` keeps only variants with `peak_memory_mb ≤ 2300`; the pool's tier-3 models (Llama-3.2-3B peak 3400, Ministral 3200, Phi-4-mini 4100) all exceed it, so tier 2 is genuinely the top feasible tier. Escalation *did* fire (3 zero-score evals → stagnation → escalate) and correctly terminated because there is nothing bigger that fits. This is the intended behavior, unlike the GSM8K rollback loop (B124).
- **Status:** ⚪ working as designed (documented for clarity).

---

## Merge of `qwen-only` branch — 2026-07-15 (B134)

## B134 -- merged configurable model-selection strategies; reconciled qwen-only pool with B123
- **Where:** `agent/nodes/cold_start/model_selection/` (new package), `agent/graph.py`, `agent/nodes/iterate.py`, `config/config.py`, `config/android_pool.py`, `agent/state.py`, `tests/cold_start/test_model_selection.py`
- **When:** 2026-07-15 (user: "merge the qwen-only branch")
- **What merged:** The branch adds a configurable initial-model-selection stage (replacing the hard-wired `scaling_curve` node) with 4 strategies selected via `config.MODEL_SELECTION_STRATEGY` (env `SLM_MODEL_SELECTION_STRATEGY`): `smallest_first` (default), `largest_first` (feasibility probe → drop to smallest), `interpolation` (the original 3-probe scaling curve, now picking the model closest to the RAM budget), and `orchestrator_choice` (LLM picks). `graph.py` wires `eval_setup → model_selection → curate`; `iterate.py` gained largest-first probe hooks; `state.py` gained `_largest_first_phase`.
- **Merge conflict resolution:**
  - `agent/nodes/iterate.py`: combined the branch's `largest_first` probe-stagnation handling with this session's stall backstop — `elif stagnant or stalled:` now terminates if in the largest_first probe phase, else escalates.
  - `agent/graph.py`: kept BOTH the branch's `model_selection` wiring AND this session's `rollback → iterate` edge (B124).
  - `config/android_pool.py`: **the branch's pool reintroduced the multimodal `Qwen/Qwen3.5-0.8B` and `unsloth/Qwen3.5-2B-GGUF`, which crash text-only LoRA (B123).** Reconciled by keeping the qwen-only intent but seeding 4 TEXT-ONLY Qwen-family models that load cleanly and span tiers 0-3: `unsloth/Qwen3-0.6B`, `Qwen/Qwen2.5-1.5B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `Qwen/Qwen2.5-3B-Instruct` (12 variants total: tier0×1, tier1×3, tier2×4, tier3×4).
- **Added testing knob:** `SLM_STOP_THRESHOLD` env override in `task_analysis_node` — pins the stop threshold (and its floor) so a validation run can be steered deterministically (e.g. above the pool's best benchmark to force escalation through every tier).
- **Validation test:** `tests/pipeline/run_qwen_escalation.slurm` — GSM8K on a 16 GB device with `smallest_first` + `SLM_STOP_THRESHOLD=0.90`, designed to walk tier 0→3 and exercise data_rebuild / hyperparameter / rollback / escalate / terminate. Downward probing and classification/NER positive synthesis are covered by complementary tests.
- **Status:** 🟢 merged; all 29 model-selection + iterate + pool tests pass with full deps.

---

## Cleanup + Qwen3.5 support pass — 2026-07-15 (B135–B138)

## B135 -- dead code removed
- **Where:** `agent/nodes/cold_start/scaling_curve.py`, `tests/cold_start/test_scaling_curve.py`, `training/on_device_eval.py`
- **When:** 2026-07-15
- **What:** `scaling_curve.py` was replaced by the `model_selection/` package at merge time (graph no longer imports it; `interpolation.py` is its successor) — deleted with its test. `training/on_device_eval.py` was unified into `hardware_eval/on_device_eval.py` and imported by nothing — deleted. Stale `scaling_curve_node` references in `task_analysis.py`/`hardware_filter.py` comments updated to `model_selection`.
- **Status:** 🟢 done.

## B136 -- Qwen3.5 multimodal LoRA support (text-only path)
- **Where:** `training/lora_trainer.py` (`text_tokenizer()`), `training/slm_helpers.py`, `config/android_pool.py`
- **When:** 2026-07-15 (user: "make the pipeline work for Qwen 3.5")
- **How addressed:** Qwen3.5 is natively multimodal and loads as a *processor*; text-only LoRA previously crashed ("Incorrect image source ... Got <|im_start|>user", B123) because the vision processor received the text chat template. Added `text_tokenizer()`, which unwraps the processor's inner text tokenizer (gated on the class name ending in "Processor" so plain tokenizers / test mocks are untouched) and is applied in both training and inference. Re-added `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B` to the pool with a `multimodal=True` `ModelSpec` flag, using the BASE transformers repo (not `-GGUF`, B107).
- **Status:** 🟡 implemented; **UNVERIFIED** without a GPU run + model download. The 4 text-only Qwen models remain the reliable path and `smallest_first` starts on one of them. If the processor still routes text through the vision path, the fallback is a FastVisionModel loader.
- **Status update 2026-07-22:** 🟢 superseded for Qwen3.5-0.8B by B194.
  Job 37449798 passed text-only LoRA, separate HF inference, FastVisionModel merge,
  Q4_K_M build, and deployment eval; job 37483464 additionally passed real four-prompt
  batched inference and VRAM-return criteria. The 2B/4B siblings still lack their own
  standalone GPU smoke.

## B137 -- quantization separated from on-device eval
- **Where:** new `hardware_eval/quantize_model.py`; `hardware_eval/run_autobench.py`
- **When:** 2026-07-15 (user request)
- **What:** `run_autobench.py` used to convert HF→GGUF inline via an external `~/Model-Conversion/convert_to_gguf.py` (which does not exist here). Split into two clean steps sharing one engine (`training/quantize.py`): `quantize_model.py` (HF checkpoint → GGUF) and `run_autobench.py` (GGUF → on-device metrics, no conversion). `on_device_eval.py` already only consumed a `gguf_path`.
- **Does quantization work?** The code is correct but **NON-FUNCTIONAL in this environment**: `convert_hf_to_gguf` and `llama-quantize` (llama.cpp) are not on PATH, so `quantize_from_model_spec` raises a clear "install llama.cpp tools" error. Accuracy-only pipeline runs are unaffected (they gate GGUF behind a non-`theoretical` HW backend, B108). To enable: build/clone llama.cpp and add its tools to PATH.
- **Status:** 🟢 separated; quantization itself blocked on the missing llama.cpp toolchain (infra).

## B138 -- data_rebuild variety (rotating generation temperature)
- **Where:** `data/curriculum.py` (`synthesize_hard_negatives` `temperature` param), `agent/nodes/curate.py`
- **When:** 2026-07-15 (user: "data rebuild should have a variety of data")
- **What:** Beyond the per-rebuild seed rotation (B125, varies gold sampling + hard-neg source order), the synthetic hard negatives are now generated with a temperature that rotates per rebuild (0.7 → 0.9 → 1.1), so successive `data_rebuild` rounds produce genuinely different negatives instead of a near-identical regeneration. Combined with the gold-cap warning + real-benchmark loaders (B131), this gives the rebuild real variety when the gold pool has surplus; when the corpus is fully consumed the negatives still differ via temperature.
- **Status:** 🟢 done.

---

## Official-Qwen pool + FastVisionModel + acquisition ladder — 2026-07-15 (B139–B141)

## B139 -- pool restricted to official Qwen (Qwen3 + Qwen3.5); FastVisionModel for multimodal
- **Where:** `config/android_pool.py`, `training/lora_trainer.py`, `training/slm_helpers.py`, `docs/model_pool.md` (new), `config/android_pool.md` (archived)
- **When:** 2026-07-15 (user: official Qwen only; make Qwen3.5 work)
- **What:** Pool is now 6 official Qwen base models (18 variants): text `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B-Instruct-2507`; multimodal `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`. Removed Qwen2.5, DeepSeek-R1-Distill, Qwen3-4B-Thinking-2507 (thinking-only reintroduces the reasoning-model eval failure). Real Qwen3 benchmarks sourced from the Qwen3 report + 4B-2507 card; Qwen3.5 numbers are estimates. New authoritative `docs/model_pool.md`; old `config/android_pool.md` kept but marked ARCHIVED.
- **Multimodal integration (supersedes B136's text_tokenizer hack):** Qwen3.5 is a "Causal LM with Vision" (confirmed by Unsloth's Qwen3.5 fine-tuning guide). Text-only LoRA now loads via `FastVisionModel.from_pretrained` + `get_peft_model(finetune_vision_layers=False, finetune_language_layers=True, ...)`, selected by `is_multimodal_model()` (pool `multimodal` flag lookup) in both `lora_trainer` and `slm_helpers.infer`. Base transformers repos only (never `-GGUF`, B107).
- **Status:** 🟡 pool + wiring done and unit-tested; multimodal LoRA is **UNVERIFIED** without a GPU run + model download (follows Unsloth's documented recipe). Text-only Qwen3 models are the reliable path.
- **Status update 2026-07-22:** Exact-selector and sourced-capability correctness is now
  covered by B168, and Qwen3.5-0.8B runtime support is verified by B194. The official
  six-model pool contract is unit-tested, but this does not claim direct runtime coverage
  for every Qwen3/Qwen3.5 size.

## B140 -- data-acquisition ladder (bounded diversified Exa + verified synthesis)
- **Where:** `data/loaders/web_acquire.py`, `agent/nodes/cold_start/eval_setup.py`
- **When:** 2026-07-15 (user-approved "recommended" design)
- **What:** Replaced single-pass Exa scraping with a 3-stage ladder: (1) real benchmark loader (B131/B119); (2) **bounded, diversified** Exa rounds — up to `MAX_ACQUIRE_ROUNDS=3`, each round rephrases queries (`_diversify_query`) so re-runs fetch NEW docs (not duplicates), deduped, stopping at `target_examples`; (3) if still below the viability floor (`target*0.5`), **verified synthesis** (`synthesize_seed_examples`) tops up with orchestrator-generated, deduped, label-validated examples. `target_examples` is an UPPER bound (quality-over-quantity), passed per task type from `eval_setup`. Provenance (web vs synth counts) logged and stored in `state["data_source"]`.
- **Why not the literal "rerun Exa forever + generate to the cap" plan:** more noisy web docs amplify label noise (B119); identical re-runs return duplicates; unbounded retries risk cost blowup; unfiltered synthesis trains on the teacher's hallucinations. The ladder bounds retries, diversifies queries, and validates/dedups synthetic gold.
- **Status:** 🟢 implemented (synthesis path needs live API keys to exercise end-to-end).

---

## Live escalation run (job 37110415) — bugs caught — 2026-07-15 (B141–B143)

Caught while running `run_qwen_escalation.slurm` end-to-end (the run validated the whole loop
through tier-0→3 escalation before hitting B142; see the run report).

## B141 -- annotate_cot regenerated gold CoT that the benchmark already provides
- **Where:** `data/curriculum.py` (`annotate_cot`)
- **How found:** the run sat ~15 min in "CoT annotation" per curate. The GSM8K loader ships REAL gold `cot_reasoning` for all 300 gold examples, but `annotate_cot` called the teacher for every one anyway — 300 sequential Haiku calls per curate, per tier — AND replaced the gold CoT with a weaker teacher CoT.
- **Fix:** `annotate_cot` now (1) skips examples that already carry a non-empty `cot_reasoning` (preserves gold), and (2) runs the remaining calls concurrently (ThreadPoolExecutor, ≤16 workers). For GSM8K this makes CoT annotation instant.
- **Status:** 🟢 fixed.

## B142 -- tier-3 (4B) inference crashed: "Invalid target device: None" (VRAM/meta offload)
- **Where:** `training/slm_helpers.py` (inference cache), `training/lora_trainer.py` (post-train cleanup)
- **How found:** on the 4B tier-3 model, iteration 4's eval crashed with `ValueError: Invalid target device: None` from Unsloth's `move_to_device`. The preceding log line was accelerate's "Some parameters are on the meta device because they were offloaded to the cpu" — i.e. VRAM was full, so the 4B model was partially offloaded to CPU, and Unsloth fast-generate then got a `None` device. The inference cache held up to 3 FULL-PRECISION models and the just-trained model's VRAM was never freed. (The partial offload also likely corrupted iters 2–3, explaining the tier-3 score collapse 0.333→0.042→0.014.)
- **Fix:** inference cache `_MAX_CACHED` 3→1 (eval is sequential; evicting frees VRAM + `empty_cache`), and `_run_unsloth_training` now `del`s the trained model + `torch.cuda.empty_cache()` before returning so the checkpoint loads for eval on a clear GPU.
- **Status:** 🟢 fixed; **re-verification blocked** — a clean re-run could not complete because the Anthropic API ran out of credits (billing, not code). The fix is unit-safe and addresses the exact offload cause.

## B143 -- device DB matcher dropped 2-digit model numbers ("OnePlus 12")
- **Where:** `agent/nodes/cold_start/hardware_research.py` (`_lookup_local_db`)
- **How found:** the run logged "No match in local DB ... falling back to Exa" for "OnePlus 12" even though it's in `devices.csv`. The matcher kept only 4-digit numbers, so "12" was dropped → only "oneplus" matched (1 < the ≥2 threshold).
- **Fix:** keep short bare numbers (≤4 digits) as keywords — they are model numbers ("12", "8") or years. Verified all four test devices + OnePlus 12 now resolve locally (no Exa fallback).
- **Status:** 🟢 fixed.

---

## Run-review batch — 2026-07-15 (B145–B153)

Driven by review of the escalation run (job 37110415). Anthropic credits were exhausted mid-review, so these are unit-tested but NOT re-verified end-to-end.

- **B145 — math/code hard-negs were gold DUPLICATES.** `synthesize_hard_negatives` returns gold passthrough for math/code (no safe wrong-answer negative, B62); curate appended them → 162 duplicate examples, and the report showed "162 synthesized" then "0 hard / 462 gold". Fixed: math/code are now **gold-only** (no hard negs), so the dataset and the report are accurate (300 gold, 0 hard). `agent/nodes/curate.py`.
- **B146 — benchmark data hard-capped at 300.** `load_benchmark_dataset(max_train=300)` capped gold at 300 vs a 650 target for math. Fixed: `eval_setup` now sizes `benchmark_max_train` to the task's gold target (×1.15 headroom, ≤1200) and threads it through `acquire_dataset` → `load_benchmark_dataset`; test kept at 80 (eval cost). `web_acquire.py`, `eval_setup.py`.
- **B147 — CoT eval truncation.** Generation/math/NER eval used `max_new_tokens=256`; verbose CoT on the 4B model overran it, cutting off the final answer before the exact-match extractor → collapsing scores. Raised to 512. `eval/harness.py`.
- **B148 — default LoRA rank 8→16.** Research (IJCNLP-2025 rank sweep; "LoRA Learns Less and Forgets Less") shows r=8 underfits multi-step reasoning; r=16 (α=32) is the neutral default, r=32-64 the capacity sweet spot (watch overfit on small data). Default bumped to 16; iterate can still tune. `agent/nodes/train.py`.
- **B149 — stall/stagnation escalates WITHOUT an LLM call.** iterate now takes the (rule-based) escalation decision before calling the orchestrator, saving one API call every plateau. `agent/nodes/iterate.py`.
- **B150 — robust decision-JSON parsing.** `JSONDecodeError('Expecting value: line 1 column 1')` came from empty/prose LLM replies. `_parse_decision_json` strips fences, regex-extracts the JSON object, and raises a readable ValueError on empty. `agent/nodes/iterate.py`.
- **B151 — fail-fast on billing/auth API errors.** Credit/auth/quota errors recur on every call; silently falling back for a whole run produces garbage. New `agent/llm_errors.py` (`raise_if_fatal`) is called in iterate + escalate to stop the run with a clear message instead of spamming fallbacks. (Other LLM-calling nodes should adopt it too — follow-up.)
- **B152 — downward_probe gated to interpolation/orchestrator_choice only (Q3).** smallest_first/largest_first already end at the smallest feasible model, so a downward probe is redundant; only prediction-based strategies can over-select. `agent/nodes/iterate.py`.
- **B153 — verbose end-of-run summary + escalate logging.** Added `escalation_history` (recorded by escalate before it resets scores/DAG) so the summary prints the FULL per-tier progression, not just the final model. escalate now logs the orchestrator's model-choice reason + clean quant. `agent/state.py`, `escalate.py`, `run.py`. Also: Unsloth banner/kernel-warning/offload log noise filtered at the `_Tee` and via accelerate log level (Q7); B142 VRAM crash + B141 CoT-skip carried in.
- **B154 — B143 matcher extended** to keep short model numbers (already above).

## Data sizing + agentic acquisition — 2026-07-15 (B155–B157)
- **B155 — dataset-size targets recalibrated + centralized.** Moved the per-task `N_TOTAL` map to `config.DATASET_SIZE_BY_TYPE` (single source for curate + eval_setup) and adjusted toward the paper's §4.3 quality-over-quantity guidance: NER 300→200, math_reasoning 1000→700, code_generation 1000→300 (paper: 173>348 on HumanEval), generation 1000→600 (500 selected > 2000 random). classification 150, multi_label 300, structured 400, multilingual 400 unchanged.
- **B156 — quant variants are trained/eval'd IDENTICALLY (documented limitation).** A model's Q4_K_M / Q8_0 / bf16 pool entries share benchmark scores and all train the SAME base HF model via LoRA; in the default `theoretical` HW backend, eval scores the HF/LoRA weights (no GGUF), so Q4_K_M and Q8_0 of one model produce identical results — the quant only changes the size/tier/speed the selector sees. Honest per-quant accuracy needs llama.cpp + `SLM_HW_BACKEND!=theoretical` (absent here). Not a code bug; a Phase-2 gap.
- **Status update 2026-07-22:** 🟢 superseded by B169. Quant identities now receive
  exact Q4/Q8 zero-shot baselines and exact-GGUF interpolation/downward/fine-tuned
  evaluation; cache keys include both weights and quant. Jobs 37449798 and 37483464
  exercised the Qwen3.5 Q4 deployment path.
- **B158 — quantized ACCURACY eval decoupled from on-device eval.** New `SLM_QUANT_EVAL=1`
  (`config.QUANT_ACCURACY_EVAL`) makes `evaluate_node` merge→quantize→score the real GGUF on
  CPU (via llama-cpp-python) for honest per-quant accuracy, WITHOUT any phone/latency/power
  measurement — so Q4_K_M vs Q8_0 finally differ (was B156). `infer_batch_gguf` now uses the
  GGUF's chat template (`create_chat_completion`) for train/serve parity. Added standalone
  `hardware_eval/quant_accuracy_eval.py` to compare Q4_K_M/Q8_0/bf16 accuracy side-by-side.
  Requires llama.cpp tools on PATH + `pip install llama-cpp-python` (neither installed yet).

- **B157 — agentic HF-dataset acquisition (paper §6.1).** `acquire_dataset` now: (0) hardcoded known-benchmark fast path → (1) **agentic discovery**: Exa locates candidate HuggingFace dataset repos, the orchestrator picks the best + maps its columns to our schema, and `datasets.load_dataset()` downloads the REAL data (`discover_and_load_hf_dataset`) → (2) web-scrape + verified-synthesis only as last resort. This replaces "scrape web pages and call the text a dataset" with "locate + download the actual dataset," matching the paper. Best-effort/defensive; unverified without API credits + HF network.

---

## Multi-day pipeline hardening and runtime readiness — 2026-07-21/22 (B159–B194)

## B159 -- eval-size target was silently capped at 100 instead of the full 800
- **Where:** `agent/nodes/cold_start/eval_setup.py`, `data/eval_set.py`,
  `scripts/prepare_shared_dataset.py`.
- **How found/evidence:** The planner and config requested an 800-row floor, but
  `build_eval_set()` was called with its fixed defaults `40/40/20`; only 100 acquired rows
  were scored. Focused tests now assert `320/320/160 = 800`, including code generation and
  frozen shared bundles; runs 37371493 and 37372065 logged the corrected full 800.
- **Impact:** Small, noisy evals produced unstable macro-F1, weak rare-class coverage, and
  misleading rollback/stagnation decisions while claiming an 800-example evaluation.
- **Root cause:** Acquisition sizing and eval-slice sizing were separate; the caller never
  translated `eval_size_target` into slice counts.
- **Fix:** Added `_eval_split_sizes()` and threaded the dynamic 40/40/20 allocation through
  live and shared-dataset paths. APPS filtering occurs before the 800-row cap.
- **Status:** 🟢 fixed and structure-tested.

## B160 -- fixed vLLM port collided on shared nodes and a foreign endpoint passed readiness
- **Where:** `tests/pipeline/run_emotion_orch_full_l40s.slurm`,
  `tests/pipeline/_l40s_task_body.sh`, `data/synth_client.py`.
- **How found/evidence:** Job 37371493 launched its server on `localhost:8000`; its vLLM log
  failed with `OSError: [Errno 98] Address already in use`, while the pipeline reported
  `/models` reachable and proceeded. It had connected to another tenant/job's listener.
- **Impact:** A run could generate training data with an unowned model/configuration, charge
  the wrong service, or kill/reuse another job while its own server had already failed.
- **Root cause:** All jobs shared the node network namespace and fixed port 8000; readiness
  checked reachability but not ownership or exact served-model identity.
- **Fix:** Derive job-unique high ports from `SLURM_JOB_ID`, retain the launched PID for
  cleanup/liveness checks, and require `/models` to contain the exact configured model.
- **Status:** 🟢 fixed; job 37372065 used unique port 32065 successfully.

## B161 -- L40S localhost readiness was proxy-sensitive and did not cover first-request JIT
- **Where:** `data/synth_client.py`, `scripts/setup_vllm_env.sh`,
  `tests/pipeline/manual_qwen36_cot_smoke_l40s.slurm`, L40S task scripts.
- **How found/evidence:** Jobs 37449799/37449899 saw repeated localhost HTTP 503s even though
  vLLM later logged a healthy server (cluster proxy interception). Job 37450415 reached
  `/models` but the Python smoke did not receive the shell's endpoint/model config. Job
  37450646 reached readiness, then the first completion triggered Triton MoE/GDN JIT and
  outlived the old request/server bound. Earlier jobs 37366122/37366249 also exposed missing
  `ninja` and fragile FlashInfer JIT/host-compiler failures.
- **Impact:** A healthy local server looked dead, or a shallow `/models` probe passed while
  the first real synthesis call failed; long jobs then aborted or silently became gold-only.
- **Root cause:** HTTP clients/curl inherited proxy variables, smoke config was not fully
  exported, readiness tested only the control plane, and runtime kernels compile lazily.
- **Fix:** Use `httpx.Client(trust_env=False)` and `curl --noproxy '*'`; export exact
  `SLM_SYNTH_*` keys; load CUDA/nvcc and install `ninja`; use eager/text-only settings; and
  independently bound readiness, first completion, and total server lifetime.
- **Status:** 🟢 fixed and single-L40S verified by final smoke job 37450865, whose first
  non-thinking local Qwen3.6 completion passed after the expected JIT warnings.

## B162 -- hard-negative ratio and composition accounting counted anchors as negatives
- **Where:** legacy curriculum assembly in `agent/nodes/curate.py`,
  `data/curriculum.py`, `data/curation_log.py`.
- **How found/evidence:** Job 37372065 scaled a hard target to 485, then logged 970
  “synthesized” because the 2-for-1 API returned 485 source anchors plus 485 generated rows;
  post-QC reporting guessed composition from the pre-QC count and reported 84% gold/16%
  hard. Job 37387566 showed the ratio varying again after balance/dedup.
- **Impact:** The claimed 65:35 curriculum was not auditable; gold could be reported as hard,
  and iterate decisions reasoned over incorrect composition.
- **Root cause:** Provenance was implicit and counts were calculated before quality controls;
  the return cardinality of a contrastive pair was mistaken for generated-row cardinality.
- **Fix:** Tag source anchors and generated rows separately, count provenance after all QC,
  record total/source/generated/replay components explicitly, and scale legacy targets from
  actual available gold. Structured rebuilds now use explicit post-QC budgets instead of
  promising a ratio that balancing can change.
- **Status:** 🟢 fixed; composition is now causal/provenance-based.

## B163 -- data rebuilds discarded winning hyperparameters and fallback retries repeated exactly
- **Where:** `agent/nodes/train.py`, `training/hparams.py`, `agent/nodes/iterate.py`,
  DAG `pi.H`.
- **How found/evidence:** In job 37372065, r=32/lr=5e-4/5 epochs reached 0.723, but the next
  data rebuild silently reverted to r=16. After an iterate parse failure, iterations 4 and
  5 used the same dataset and r=16 and produced identical loss/eval traces and F1=0.5609.
- **Impact:** Data interventions were confounded by weaker optimizer settings, and
  deterministic duplicate trials burned hours without exploring H.
- **Root cause:** Missing explicit hyperparameters always selected defaults; stale
  `llm_iterate_decision` could survive a failed call; tried memory omitted complete/pruned
  `(dataset,H)` identities.
- **Fix:** Carry every winning optimizer field across data-only changes, clear failed
  decisions, canonicalize the full bounded H, count losing/pruned candidates as tried, and
  replace exact repeats with a deterministic untried neighbor.
- **Status:** 🟢 fixed with complete DAG/checkpoint identity coverage.

## B164 -- ChatAnthropic content blocks and prose responses broke decision JSON parsing
- **Where:** `agent/nodes/iterate.py` (`_coerce_to_text`,
  `_parse_decision_json`, `_reask_json_only`).
- **How found/evidence:** Job 37372065 logged
  `JSONDecodeError(... line 1 column 2 (char 1))`. LangChain returned
  `AIMessage.content` as a list of text blocks; `str(list)` produced a single-quoted Python
  repr. A later live response was prose with no JSON.
- **Impact:** Valid orchestrator decisions were discarded, causing static/test-agent
  fallbacks and the identical-training defect in B163.
- **Root cause:** The parser handled only strings, extracted the transport envelope rather
  than block text, and left a second `json.loads` unguarded.
- **Fix:** Flatten text blocks, accept fenced/prose-wrapped JSON but reject Python literals,
  convert all parse failures to readable `ValueError`, validate the schema, and make one
  fresh tool-free JSON-only reask.
- **Status:** 🟢 fixed; content-block, fence, prose, empty, and garbage cases are covered.

## B165 -- cost ledger missed whole processes/providers and priced/cache-bucketed calls incorrectly
- **Where:** `agent/cost.py`, provider call sites, `training/cuda_worker.py`,
  `tests/pipeline/run.py`.
- **How found/evidence:** A long run reported only 3 Claude calls/~$0.03 while its log
  contained 63 iterate decisions. `ChatAnthropic.invoke`, forked acquisition, CUDA-worker
  judge calls, and OpenAI/DeepSeek fallbacks bypassed the process-local ledger. All Claude
  usage was priced as Sonnet, and Anthropic 5-minute/1-hour creation totals could be charged
  in the wrong or duplicate cache bucket.
- **Impact:** Spend, provider attribution, latency, failure counts, and resume accounting
  were materially false; budget decisions could not be trusted.
- **Root cause:** In-memory counters and partial SDK wrapping did not cross process
  boundaries or preserve model/provider-specific usage metadata.
- **Fix:** Use a flock-protected append-only JSONL ledger inherited by fork/CUDA workers;
  wrap Anthropic SDK, ChatAnthropic, OpenAI-compatible, Exa, and local calls explicitly;
  classify OpenAI/DeepSeek/local vLLM; and apply a dated model registry with separate cache
  read, 5-minute write, and 1-hour write rates plus overrides/unknown-pricing warnings.
- **Status:** 🟢 fixed; local Qwen calls are recorded at `$0`.

## B166 -- offline/shared bundles lacked deterministic order, provenance, checksums, and overlap gates
- **Where:** `scripts/download_datasets.py`, `scripts/prepare_shared_dataset.py`,
  `data/loaders/dataset_integrity.py`, `data/loaders/web_acquire.py`, local manifests.
- **How found/evidence:** Local files could be loaded without hashes or source revision;
  paid discovery could run before a clean local copy; provenance records were conflated with
  eval bans; overlap checks after capping could hide contamination outside the selected
  prefix; shared strategy runs had no integrity-sealed frozen bundle.
- **Impact:** Runs were non-reproducible, could pay unnecessarily, and could train on held-out
  rows or silently use a tampered/misattributed bundle.
- **Root cause:** The local fallback was a convenience directory rather than a versioned
  artifact contract.
- **Fix:** Default order is checksum-verified local bundle → deterministic known benchmark
  → paid agentic discovery (agent-first exists only as an explicit readiness mode).
  Schema-v2 manifests pin source/config/revision/splits/roles/eval bans/counts; SHA-256
  sidecars and manifest hashes are verified; complete splits are schema/normalized-overlap
  checked before caps; APPS also checks URL/solution fingerprints. Shared bundles seal plan,
  sources, bans, difficulty, and all content.
- **Status:** 🟢 fixed; explicit schema-v1 bundles remain compatibility-only and are logged
  as unhashed legacy data.

## B167 -- unrelated local datasets matched solely because the task type agreed
- **Where:** `data/loaders/web_acquire.py` (`_local_manifest_match`,
  `load_local_dataset`).
- **How found/evidence:** An emotion-classification bundle could satisfy an SMS-spam or
  FinancialPhraseBank request because all were `classification`; focused tests reproduce
  both wrong-match cases.
- **Impact:** The pipeline could train/evaluate the wrong task while reporting a successful
  offline fallback.
- **Root cause:** Bundle selection treated task-family equality as semantic task identity.
- **Fix:** Recognized benchmarks require an exact alias/source match. Unknown benchmarks
  require strong task + label Jaccard/coverage + row-schema agreement; rejected candidates
  do not abort the remaining acquisition ladder.
- **Status:** 🟢 fixed.

## B168 -- Qwen3.5 selection was biased to BF16 and capability evidence was ambiguous
- **Where:** `config/android_pool.py`, `config/model_capabilities.{md,py}`,
  orchestrator/escalation/downward selection prompts.
- **How found/evidence:** Job 37372065 logged “LLM chose Qwen3-1.7B [bf16]” while the LLM's
  reason explicitly selected Q4_K_M/1350 MB. Three siblings shared the same bare model ID,
  and first-match/list order selected BF16. Prompts also mixed MMLU/MMLU-Pro/Redux, attached
  base/proxy numbers to post-trained artifacts, and represented missing scores as zero.
- **Impact:** Selection ignored the model's stated resource choice and could rank models
  using fabricated or incomparable capability data.
- **Root cause:** Deployment quant was not part of stable identity, and capability fields
  lacked artifact/mode/protocol/source provenance.
- **Fix:** Every variant now uses `model_id@bf16|Q8_0|Q4_K_M`; prompts and histories require
  exact selectors, while a legacy bare ID deterministically chooses the lowest-RAM feasible
  sibling. Capability measurements retain named metric, exact artifact, mode, protocol, and
  official source; unknown remains “not reported,” and unsupported Qwen3 base proxies are
  not attached.
- **Status:** 🟢 fixed; supersedes the selector/provenance limitations in B139.

## B169 -- quantized variants received BF16 baselines/probes and could reuse the wrong GGUF
- **Where:** `agent/nodes/evaluate.py`, interpolation/downward probes,
  `training/slm_helpers.py`.
- **How found/evidence:** B156's Q4/Q8 identities were scored through the same HF/BF16 path.
  Later, tier history showed identical Q4/Q8 results while artifacts contained only
  `model-q4_k_m.gguf`: the cache key used `weights_ref` alone, so a Q8 request reused an
  earlier Q4 file when iteration paths collided after escalation.
- **Impact:** Baseline deltas, interpolation, escalation, downward adoption, and final
  variant reports attributed accuracy to artifacts that were never evaluated.
- **Root cause:** Quant was treated as resource metadata, not part of evaluation identity
  or artifact-cache identity.
- **Fix:** Q4/Q8 zero-shot baselines and interpolation/downward/fine-tuned probes build and
  score the exact GGUF. Cache keys use `(weights_ref, quant)` and require the specific
  `model-<method>.gguf`; required quant paths fail closed rather than labeling BF16 as quant.
- **Status:** 🟢 fixed; supersedes B156 and is exercised for Q4 by jobs 37449798/37483464.

## B170 -- thinking mode differed across training, HF inference, and GGUF inference
- **Where:** `training/lora_trainer.py`, `training/slm_helpers.py`, code/task prompt builders.
- **How found/evidence:** Hybrid Qwen models think by default; prior train/eval templates
  could therefore mix direct targets with `<think>` continuations, truncating labels/JSON
  or making HF and GGUF scores incomparable. Installed llama-cpp-python 0.3.34 lacks
  `chat_template_kwargs`.
- **Impact:** Classification/NER extraction collapsed, generation budgets were consumed by
  hidden reasoning, and deployment-format parity was not trustworthy.
- **Root cause:** “Non-thinking” was an assumption rather than an explicit cross-backend
  contract.
- **Fix:** Training and HF serving use the same chat template with
  `enable_thinking=False`; Qwen tokenizers without a template fail clearly. GGUF uses the
  kwarg when supported, otherwise the verified Qwen ChatML empty-think prefix; the
  non-thinking-only 4B-Instruct artifact uses its plain prefix. Unknown unsupported
  templates fail closed.
- **Status:** 🟢 fixed and covered by parity tests plus Qwen3.5 runtime smokes.

## B171 -- generation CoT used the wrong backend priority and sometimes the task label as the answer
- **Where:** `data/curriculum.py` (`get_cot_fallbacks`, `annotate_cot`),
  `agent/nodes/curate.py`.
- **How found/evidence:** The old route could construct the Claude orchestrator as teacher,
  did not make local Qwen3.6 primary, and bundle rows with both `answer` and a generic
  `label="generation"` could place the task label—not the gold answer—in the CoT prompt.
- **Impact:** Paid calls replaced available local reasoning, specialist routing was lost,
  and CoT could explain an incorrect/non-answer target.
- **Root cause:** Backend choice and answer extraction were legacy single-client fallbacks
  with inconsistent row-schema precedence.
- **Fix:** Per example: local non-thinking Qwen3.6 first; math/science falls back
  DeepSeek→OpenAI, code/QA/general OpenAI→DeepSeek; Claude/orchestrator is excluded.
  Gold uses first present `answer`→`response`→`label`, existing CoT is preserved, and failed
  backends advance without dropping the row.
- **Status:** 🟢 fixed; local Qwen3.6 CoT was runtime-verified in job 37450865.

## B172 -- hard-negative generation ignored the local generator and wrote unsafe SFT targets
- **Where:** `data/curriculum.py::synthesize_hard_negatives`,
  `agent/nodes/curate.py`.
- **How found/evidence:** The generation branch could bypass an injected local
  `generate_fn` for a paid client, then store a “plausible wrong answer” as the positive
  SFT response. NER synthesis accepted malformed/empty entity lists, absent spans, or
  changed entity types.
- **Impact:** Math/code/open-generation models were trained to answer incorrectly; NER was
  trained on false spans/types; locality/cost claims were false.
- **Root cause:** “Hard negative” semantics from discriminative training were copied into
  positive-likelihood SFT without a preference objective or verifier.
- **Fix:** Classification and NER synthesis use the supplied local generator. Math, code,
  and open generation are gold/CoT-only until verified-positive or preference training
  exists. NER accepts only parseable non-empty JSON whose spans occur in rewritten text and
  whose types come from the source row; malformed output is discarded non-fatally.
- **Status:** 🟢 fixed.

## B173 -- paid generation judging lacked locality, cache durability, and fail-fast semantics
- **Where:** `eval/judge_client.py`, `eval/scorers/generation.py`, cost/timing ledgers.
- **How found/evidence:** B45's path made one sequential Anthropic call per eval row and
  could turn endpoint/model/parse failures into ordinary low model scores. A configured
  “local” endpoint could be a remote/private host or a proxy target.
- **Impact:** Open-generation evaluation was expensive, slow, non-reproducible, and could
  train/rollback in response to judge infrastructure rather than model quality.
- **Root cause:** Judge execution was embedded in the scorer without an explicit local
  identity, transport, failure, or persistent-cache contract.
- **Fix:** Require exact Qwen3.6 identity from `/models` and completion responses; allow
  loopback/localhost/Unix sockets by default and require explicit opt-in for any remote
  host; ignore proxies; disable thinking; serialize inputs as untrusted JSON; parse exactly
  one finite `[0,1]` number; dedupe/cache triples in a locked run-local JSONL keyed by model
  and prompt fingerprint; score misses concurrently in order and abort on first failure.
- **Status:** 🟢 code fixed; B45's metric-field rename remains open. Full shared-GPU
  judge-during-eval validation belongs to pending TP4 smoke job 37486488 (B194), not to the
  correctness fix itself.

## B174 -- raw eval failures leaked into intervention prompts and candidate training data
- **Where:** `agent/nodes/test_agent.py`, `agent/nodes/iterate.py`,
  `agent/nodes/curate.py`, `agent/data_rebuild.py`.
- **How found/evidence:** Review found rebuild hints/anchors could be derived from held-out
  failure rows and that free-form LLM payloads could quote eval text; this violates the
  fixed-eval firewall even if exact train/test splits were initially disjoint.
- **Impact:** The loop could memorize the evaluation set and report artificial gains.
- **Root cause:** Failure diagnosis, plan generation, and data synthesis shared raw example
  objects rather than a one-way aggregate boundary.
- **Fix:** Test-agent output is limited to difficulty scores, diagnoses, and aggregate
  confusion counts. Decision/rebuild schemas recursively reject held-out text; synthesis
  anchors come only from normalized train sources; mined/replay/elite/final rows all pass
  the normalized eval-text gate.
- **Status:** 🟢 fixed with explicit leakage tests.

## B175 -- iterate could execute unrestricted tools for a declarative routing decision
- **Where:** former iterate tool loop; current `agent/nodes/iterate.py::_llm_iterate`.
- **How found/evidence:** B117/B129 showed tool-round exhaustion. Hardening review also found
  the model could request shell/file/web operations even though the output is only a bounded
  intervention JSON, opening unnecessary leakage and side-effect paths.
- **Impact:** A routing call could inspect artifacts/raw eval text, mutate files, spend
  unbounded rounds, or fail without ever emitting a decision.
- **Root cause:** A general ReAct loop was used for a fixed declarative schema.
- **Fix:** One tracked, tool-free ChatAnthropic call receives only bounded trajectory and
  aggregate reports. A tool-use/prose/malformed reply gets one fresh JSON-only reask from
  the original context; no tool request is executed or reflected back.
- **Status:** 🟢 fixed; supersedes the operational mechanism described in B117/B129.

## B176 -- structured data rebuilds lacked enforceable caps, lineage, persistence, and crash-safe spend
- **Where:** `agent/data_rebuild.py`, `agent/nodes/curate.py`,
  `data/acquisition_budget.py`, DAG/rollback state.
- **How found/evidence:** Cross-feature review found unresolved elite references, additive
  budgets exceeding final size, zero-weight difficulty buckets being backfilled, a
  no-novelty mining result overwriting novelty from another strategy, mined rows disappearing
  from future rebuilds, and paid retries being replayable after crash/resume.
- **Impact:** Plans could exceed cost/data bounds, lose useful real data, misattribute gains,
  repeat zero-yield work, or preserve rows from the wrong dataset version.
- **Root cause:** Strategies were loosely composed and only the final JSONL—not the full
  causal plan/budget/source state—was durable.
- **Fix:** Strictly normalize one primary + ≤2 support strategies; cap/snap rows,
  fractions, difficulty weights, query variants, and per-plan/run paid rounds; require
  positive additive budgets and resolvable elite provenance/version; apply common QC and
  final `target_rows`; attribute origin/novelty after composition; merge accepted mined rows
  into durable `train_examples`; reserve paid rounds in a locked ledger before calls and
  never refund pending/failed reservations; persist full D and restore it on rollback.
- **Status:** 🟢 fixed.

## B177 -- `surgical` duplicated targeted rebuild behavior and bypassed the new plan contract
- **Where:** former iterate enum/graph route/state fields and curate surgical branch.
- **How found/evidence:** Design review found `surgical` and `targeted_patterns` overlapped
  high-score targeted synthesis but had separate routing, validation, accounting, and
  rollback semantics; on hard tasks it was also effectively unreachable.
- **Impact:** Two mechanisms could express the same intervention while only one had bounded
  budgets, plan identity, repeat prevention, provenance, and dataset rollback.
- **Root cause:** The legacy score-band action survived after hypothesis-driven rebuilds
  became first-class.
- **Fix:** Remove `surgical`/`targeted_patterns` from enum, graph, state, fallback, and docs.
  High-score refinement is now task-gated `targeted_synth_positive` inside the same validated
  structured data-rebuild contract.
- **Status:** 🟢 fixed; redundant path removed.

## B178 -- MBPP “pass@1” was syntax/execution-only and exposed spoofable success signaling
- **Where:** `eval/scorers/generation.py`, `eval/scorers/code_execution.py`,
  MBPP local bundle/tests.
- **How found/evidence:** Audit showed code could compile (or simply `pass`) and receive
  credit without running each row's `test_list`; earlier worker designs exposed payload/result
  files or a success descriptor that candidate code could forge.
- **Impact:** Code-generation scores measured syntax and harness manipulation rather than
  functional correctness.
- **Root cause:** The scorer had no trusted controller retaining hidden tests and no
  isolated request/result protocol.
- **Fix:** Execute all MBPP assertions and imports in a disposable candidate process;
  enforce/derive the entry point; keep tests/expected outcomes in the controller; sanitize
  argv/env/cwd; expose no success FD/file; apply process-group cleanup and rlimits; report
  timeout/runtime/assertion diagnostics fail-closed.
- **Status:** 🟢 fixed; MBPP remains the lightweight CI smoke, not the long-run benchmark.

## B179 -- initial APPS migration confused split semantics and silently collapsed the dataset
- **Where:** `data/loaders/apps.py`, `scripts/download_datasets.py`,
  `data/loaders/web_acquire.py`, `data/local/apps/manifest.json`.
- **How found/evidence:** APPS introductory has 2,639 train/1,000 test source rows. Early
  conversion treated test rows without solutions as unusable and did not preserve all
  `solutions`, `starter_code`, execution mode, tests, or source revision. Gold validation
  then shrank train data without explaining why.
- **Impact:** Valid executable test rows disappeared, training/eval contracts were mixed,
  and an apparently full benchmark could collapse to a small, unattributable subset.
- **Root cause:** APPS was forced into an MBPP-like “every split must have one gold body”
  schema instead of its official train-supervision/test-execution semantics.
- **Fix:** Pin source revision `21e74d…`; train requires a passing compilable solution while
  test requires executable `input_output` and may omit gold. Preserve all solutions,
  starter/interface/difficulty/mode/problem/url metadata; validate supplied golds against
  the production runner; retain incompatible test rows with explicit status/provenance and
  skip only those at eval load. The manifest records every removal reason.
- **Status:** 🟢 fixed as an explicit fail-closed bundle. Current intentional artifact is
  721 validated train rows and 1,000 retained test rows (976 runner-compatible), not a
  silent claim that all 2,639 source train rows are usable.

## B180 -- APPS executor mismatched official argument/output semantics and allowed false passes
- **Where:** `eval/scorers/code_execution.py`, `eval/scorers/generation.py`.
- **How found/evidence:** Supplied APPS golds failed under the first runner because common
  imports, NumPy, `fractions.gcd`, ListNode/cycle, wrapped Two Sum arguments, integer dict
  keys, tuple/list equivalence, and stdin numeric/whitespace conventions were absent.
  Conversely, a global unordered-set fallback could accept wrong order or multiplicity.
- **Impact:** The bundle either discarded correct solutions or scored incorrect predictions
  as passing; candidate code could also inspect expected-output payloads.
- **Root cause:** A generic subprocess executor did not implement the pinned APPS harness
  conventions or separate trusted expected outputs from candidate state.
- **Fix:** Add the pinned compatibility prelude and wire types/adapters; compare structured
  and stdin output deterministically while preserving order/multiplicity; scope non-unique
  semantics only to declared Codeforces 1294F; keep one input/expected output at a time in
  the trusted controller; sanitize environment/argv and remove success channels.
- **Status:** 🟢 fixed with gold-parity and adversarial/spoof tests.

## B181 -- APPS context, eval sizing, case coverage, and timeout budgets were mutually inconsistent
- **Where:** `eval/harness.py`, `training/slm_helpers.py`,
  `eval/scorers/{generation,code_execution}.py`, L40S code script.
- **How found/evidence:** The code path combined a 512-token context with a 1,024-token
  completion reserve, the old 100-row eval cap, a legacy `SLM_APPS_MAX_CASES` subsample, and
  a per-case timeout whose cost multiplied without a per-problem bound.
- **Impact:** APPS could fail before generation, truncate target-critical code, under-score
  hidden cases, or make an 800-row evaluation run for hours.
- **Root cause:** Prompt/output/context sizing and executor runtime limits were added
  independently.
- **Fix:** Use 4,096 context with 1,024 output reserve and reject overlength rows/prompts;
  filter compatibility before selecting the full 800; execute every preserved case; retain
  an independent per-case bound while enforcing a configurable 6-second default
  per-problem total wall deadline; report budget exhaustion separately from case timeout.
- **Status:** 🟢 fixed; `6s × 800 < 90m` is asserted and the TP4 smoke contains a
  four-prompt 4096/1024 batch-pressure phase.

## B182 -- APPS parity test exercised a one-case compatibility branch, not the full-case path
- **Where:** old artifact parity assertion; current
  `scripts/smoke_apps_gold_parity.py`,
  `tests/eval/test_apps_gold_parity_smoke.py`.
- **How found/evidence:** The old artifact test passed `max_cases=1`, so a PASS did not prove
  production executes every preserved case or reaches timeout/accounting branches.
- **Impact:** Case-cap removal and full-suite timeout regressions could ship behind a green
  “gold parity” check.
- **Root cause:** A fast sample-level test was mistaken for production-path verification.
- **Fix:** Production ignores the legacy `max_cases` argument. The deterministic offline
  smoke pins one train and one test problem identity, executes every preserved case, checks
  `cases_executed == cases_total`, reports elapsed time, and fails closed on drift or any
  incompatible gold; separate tests force per-case and per-problem timeout branches.
- **Status:** 🟢 fixed (test defect).

## B183 -- batched HF eval lacked overlength atomicity, OOM-safe cleanup, and adapter detection
- **Where:** `training/slm_helpers.py`, `eval/harness.py`.
- **How found/evidence:** Sequential eval cost ~5–7 minutes per 1,000 examples. Initial
  batching risked truncating an overlength row while still generating shorter rows; CUDA OOM
  tracebacks retained failed-batch tensors during cleanup; adapter-only directories could be
  loaded as full models.
- **Impact:** Evaluation was slow and could silently corrupt a mixed batch, leak VRAM, lose
  output ordering, or fail to apply the trained adapter.
- **Root cause:** `infer`/`infer_batch` had separate loaders/renderers and no explicit batch,
  context, retry, or checkpoint-format contract.
- **Fix:** Share loader/non-thinking renderer; unwrap multimodal text tokenizers; left-pad
  and generate ordered batches (16 short / 4 long, overrideable); reject every overlength
  prompt before generation with no truncation; convert OOM to immutable details, exit the
  exception scope, GC/empty cache, halve and retry to 1 with a diagnostic; detect
  adapter-only checkpoints, load the base, then apply the adapter.
- **Status:** 🟢 fixed and real batched Qwen3.5 inference passed job 37483464.

## B184 -- LoRA trained prompt tokens and early-stop fallback reused partially trained state
- **Where:** `training/lora_trainer.py`, `training/hparams.py`, CUDA training payloads.
- **How found/evidence:** Review confirmed SFT loss covered the full prompt, especially
  harmful for long NER/generation/APPS rows. If checkpoint/early stopping failed after
  optimizer work, fallback reused the mutated model and reduced train split; the active
  exception traceback could retain its Trainer/CUDA graph.
- **Impact:** The realized training run differed from recorded H, prompts dominated the
  objective, examples were dropped, and fallback could OOM or continue corrupt partial
  weights.
- **Root cause:** Raw concatenated-text SFT and an in-place retry path were used instead of
  an explicit assistant-loss/fresh-stack contract.
- **Fix:** Pretokenize exact user+assistant turns, require the prompt tokenization to be a
  prefix, and collate labels with prompt/padding=`-100` for every task and text-only
  FastVisionModel path. On early-stop failure, retain only an immutable error string, leave
  the exception scope, clear failed trainer/model/tokenizer and CUDA state, reload a fresh
  base+identical LoRA H, rebuild all original rows, and retry without validation.
- **Status:** 🟢 fixed; supersedes the functional part of B30.

## B185 -- threshold payloads and stagnation arithmetic could crash or misclassify progress
- **Where:** `agent/nodes/iterate.py`, `training/hparams.py`.
- **How found/evidence:** Schema tests found strings, booleans, NaN/inf, extra keys, or a
  numeric threshold without a reason could reach float conversion. The old stagnation
  statistic could treat below-origin oscillation as progress, while exact decimal gain
  `0.12-0.10` is slightly below 0.02 in binary floating point.
- **Impact:** Malformed LLM output could crash the node; declining/oscillating models could
  avoid escalation, or a mathematically exact boundary could escalate early.
- **Root cause:** Prompt instructions were trusted as validation, and stagnation ignored
  chronology/tolerance.
- **Fix:** Strict JSON types/allow-lists/finite checks and node-boundary revalidation;
  threshold changes require a reason and can only lower to the immutable floor. Stagnation
  uses `max(window)-window[0]`, requires the full window, treats below-origin motion as no
  gain, and uses a tight tolerance so the exact boundary is non-stagnant. Convergence routes
  before escalation/API calls.
- **Status:** 🟢 fixed.

## B186 -- downward adoption lost the winning trajectory and post-convergence API failures failed runs
- **Where:** `agent/nodes/downward_probe.py`, `agent/pipeline_status.py`,
  checkpoint state.
- **How found/evidence:** Review found that replacing `selected_model` after a successful
  downward probe relabeled the original model's scores/DAG, and a credit/auth/model-choice
  failure after convergence could turn an already successful run into a failure. Retry also
  risked repeating paid gate/choice calls.
- **Impact:** Final reports lied about which model produced which score; optional
  resource-minimization work could destroy a valid result or spend twice after preemption.
- **Root cause:** Downward search had no durable origin/attempt/pending state distinct from
  the normal trajectory.
- **Fix:** Snapshot the converged exact selector, score, weights, scores, and DAG as
  `origin`; serialize fixed H and each attempt; checkpoint the selected pending candidate
  before train/eval; adopt/reject without relabeling origin; record gate/chooser failures as
  optional termination and preserve the converged model. Reports render each probe as a
  separate progression entry.
- **Status:** 🟢 fixed.

## B187 -- iterate read stale project-root `data-curation.md` instead of run-local history
- **Where:** `data/curation_log.py`, `tests/pipeline/run.py`,
  `agent/nodes/{evaluate,iterate}.py`.
- **How found/evidence:** Cross-feature review found evaluate had been redirected to the
  run artifact path while iterate constructed a default `CurationLog()` and could read old
  emotion history from the project root.
- **Impact:** Concurrent runs contaminated one another's reasoning; a resumed task could
  choose an intervention from an unrelated trajectory.
- **Root cause:** Artifact directories were monkeypatched, but curation history had an
  independent implicit global path.
- **Fix:** Set an absolute run-local `SLM_CURATION_LOG_PATH`, persist
  `curation_log_path` in state/checkpoints, pass it explicitly to both writer and reader,
  reject/repair path drift on resume, append logs across segments, and make iteration writes
  idempotent by stable entry marker.
- **Status:** 🟢 fixed.

## B188 -- 512-token context plus a 512-token output reserve left zero prompt budget
- **Where:** `eval/harness.py`, `training/slm_helpers.py`,
  `training/lora_trainer.py`, L40S task defaults.
- **How found/evidence:** Final integration review computed
  `input_budget = max_seq_length - max_new_tokens = 512 - 512 = 0` for NER, math, and open
  generation; those tasks deterministically raised before meaningful inference.
- **Impact:** Three long-run task types could not evaluate at all, and changing context only
  after failure would violate resume compatibility.
- **Root cause:** Output-token increases were not reconciled with the independent legacy
  context default.
- **Fix:** Unify training/HF/GGUF context at 4,096 for long jobs, retain task-specific
  reserves (50 classification, 512 NER/math/generation, 1,024 APPS), validate
  `reserve < context` before loading a model, and include the contract in the effective
  resume configuration.
- **Status:** 🟢 fixed with explicit zero-budget and 4096-reserve tests.

## B189 -- long-lived Unsloth/Trainer state accumulated 43.71 GiB and OOMed at iteration 64
- **Where:** GPU ownership across `training/lora_trainer.py`,
  `training/slm_helpers.py`, `eval/harness.py`; new
  `training/cuda_{isolation,worker}.py`.
- **How found/evidence:** Job 37387566 failed evaluating iteration 64: GPU 0 had only
  35.31 MiB free and PyTorch held 43.71 GiB allocated (only 113.11 MiB unused reserved).
  This proves live tensors/references, not allocator fragmentation; `empty_cache()` could
  not reclaim them.
- **Impact:** Any long 1,500-step run eventually OOMed despite cache eviction and explicit
  cleanup, losing the active segment.
- **Root cause:** Unsloth, Accelerate, PEFT, Trainer, compiled graphs, tracebacks, and caches
  retained hidden live references in one Python/CUDA context.
- **Fix:** Keep the LangGraph parent model-free and execute train, infer/eval/probes, and
  GGUF build/merge in one-shot subprocesses. Only paths/serializable results cross IPC;
  worker exit destroys the CUDA context. Explicit cache cleanup remains defense-in-depth.
- **Status:** 🟢 fixed. L40S soak job 37430554 completed 100 isolated 4 GiB cycles with
  `post_exit_used_mib=0` and `peak_delta_mib=0`.

## B190 -- graph exceptions were swallowed and printed `RUN COMPLETE`
- **Where:** `tests/pipeline/run.py`, `agent/pipeline_status.py`.
- **How found/evidence:** Immediately after the OOM traceback in job 37387566, the runner
  printed `RUN COMPLETE — 42203.6s` and exited through the normal summary path.
- **Impact:** Slurm/automation could mark crashed runs successful, and the summary described
  exceptions as non-convergence/budget exhaustion.
- **Root cause:** A broad exception handler logged the error but discarded it before final
  heading/outcome/exit-code selection.
- **Fix:** Retain `pipeline_error`, reload the freshest atomic checkpoint for reporting,
  write summary/DAG/cost/timing artifacts in finalization, render `RUN FAILED` with the
  exception, then exit nonzero. Successful non-convergence remains distinct.
- **Status:** 🟢 fixed with status-helper and worker-error tests.

## B191 -- scheduler preemption/requeue restarted in a new run and lost completed graph work
- **Where:** `agent/checkpoint.py`, `agent/state_codec.py`, `agent/graph.py`,
  `tests/pipeline/run.py`, weeklong SLURM scripts.
- **How found/evidence:** Multi-day design review found no stable thread/run identity or
  node-granular persistence: a new process built a timestamped run and reinvoked the graph
  from initial state; partial datasets/checkpoints could be mistaken for complete artifacts.
- **Impact:** Preemption near seven days could discard days of completed acquisition,
  curation, training, scoring, histories, spend, and remaining-turn accounting.
- **Root cause:** Files were end-of-run reports, not a transactional graph resume protocol.
- **Fix:** Stable run directory/thread manifest; SQLite LangGraph checkpointer plus atomic
  JSON mirror after each committed node; typed state codec for exact selectors and new
  fields; atomic datasets and manifested training checkpoints; compatibility/effective-config
  fingerprints; append-only ledgers; resume only the pending node with cumulative graph and
  wall budgets. USR1 requests checkpoint, verifies durability, then requeues the same job.
- **Status:** 🟢 fixed. Offline real-SQLite SIGKILL smoke resumes only pending `eval` after
  committed `prepare→train`, without duplicate worker or ledger events.

## B192 -- JSON/SQLite authority and requeue signal/credential races could resume stale state
- **Where:** `agent/checkpoint.py`, `tests/pipeline/run.py`,
  `tests/pipeline/_l40s_task_body.sh`, emotion SLURM script.
- **How found/evidence:** Review found progressed JSON could be injected as fresh input when
  SQLite was missing/empty/wrong-thread; the shell durability probe imported credentialed
  config before `.env`; and TERM could let the shell exit while Python was publishing
  SQLite/JSON/observability. These were reproduced by authority and credential-free probe
  tests.
- **Impact:** Resume could replay completed nodes, overwrite a run, refuse a valid requeue,
  or kill final checkpoint publication.
- **Root cause:** The JSON mirror and SQLite journal lacked an explicit authority boundary,
  and structural scheduler checks were coupled to runtime secrets/finalization timing.
- **Fix:** Pre-graph JSON is authoritative only before the first SQLite generation;
  afterwards missing/empty/wrong-thread SQLite fails closed and startup reconciles JSON from
  the latest SQLite generation. Structural `durable_resume_available` needs no API keys;
  the runner loads `.env` then performs strict setting-by-setting drift checks. TERM is
  forwarded once and the shell waits a bounded grace period for finalization (then KILL);
  TERM wins over requeue, and USR1 requeues only after manifest+JSON+SQLite agree.
- **Status:** 🟢 fixed; final durability review found no remaining Critical/Important code
  issue.

## B193 -- fixed 24h/application wall guards conflicted with checkpointed weeklong requeue
- **Where:** `config/config.py`, `agent/nodes/iterate.py`, weeklong SLURM scripts,
  `docs/PIPELINE.md`.
- **How found/evidence:** The original 24h-style process-local guard (and a later 6d20h
  value) could terminate a healthy run before Slurm's 6d22h USR1 notice, so the checkpoint
  requeue path never ran. A process-local timer also reset or double-counted across resume
  segments.
- **Impact:** “Seven-day resumable” jobs could end cleanly but prematurely, or receive an
  incorrect remaining wall budget after requeue.
- **Root cause:** Application termination and scheduler segment rollover were independent
  clocks with incompatible ownership.
- **Fix:** Track cumulative wall time in SQLite/JSON across segments. Non-requeue runs retain
  an optional 14h graceful guard for 16h allocations; checkpoint/requeue weeklong scripts
  set `SLM_MAX_WALLCLOCK_S=0` and let Slurm USR1 own rollover. Resume compatibility records
  the effective setting.
- **Status:** 🟢 fixed and documented.

## B194 -- Qwen3.5/Qwen3.6 runtime support lacked bounded end-to-end evidence
- **Where:** `tests/pipeline/manual_qwen35_readiness_l40s.slurm`,
  `manual_qwen36_cot_smoke_l40s.slurm`,
  `manual_qwen36_tp4_colocation_l40s.slurm`; corresponding SLURM logs.
- **How found/evidence:** B136/B139 wiring was initially unit-only. Job 37449798 passed
  Qwen3.5-0.8B text-only LoRA → single HF infer → FastVisionModel merge → Q4_K_M GGUF →
  deployment eval. Job 37483464 added one real four-prompt variable-length batch and passed
  VRAM criteria (`3839/46068 MiB` peak, `0 MiB` post-worker delta; total 955.569s). Final
  Qwen3.6 job 37450865 served FP8 on one L40S and returned one non-thinking local CoT after
  first-request JIT.
- **Impact:** Without real smokes, processor unwrapping, adapter load, batching, merge,
  llama.cpp deployment, vLLM startup, and first-call kernels could all remain falsely green.
- **Root cause:** CPU mocks do not exercise Unsloth/FastVisionModel, CUDA memory ownership,
  llama.cpp binaries, or Qwen3.6 vLLM runtime compilation.
- **Fix:** Added bounded, manually submitted, no-cloud smokes with explicit phase/time/VRAM
  criteria and non-empty artifact checks.
- **Status:** 🟢 Qwen3.5-0.8B and standalone one-L40S Qwen3.6 paths verified by the jobs
  above. 🟡 TP4 shared-GPU judge/train/APPS/GGUF co-location job 37486488 is pending runtime
  validation; pending is not “fixed” and no result is claimed here.

## B195 -- raw Qwen3.5 baseline merge produced and cached an unloadable GGUF
- **Where:** `agent/nodes/evaluate.py`, `training/quantize.py`.
- **How found/evidence:** The zero-shot `Qwen/Qwen3.5-2B` Q4_K_M artifact reported
  `qwen35.block_count=25`/`n_layer_all=25` but contained only 320 tensors; llama.cpp
  rejected it with `missing tensor blk.24.attn_norm.weight`. A LoRA-merged artifact from
  the same architecture contained 335 tensors and loaded.
- **Impact:** A raw base model passed through `FastVisionModel.save_pretrained_merged`,
  lost the MTP/final layer, and was then trusted forever based only on its cache filename.
  Baseline quantization failures could also be converted into an apparent model score of
  `0.0`, mixing infrastructure failure with evaluator output.
- **Root cause:** Base HF IDs and adapter checkpoints shared the same Unsloth merge path;
  cached GGUF files had no load validation, content hash, or validation provenance.
- **Fix:** Convert raw base IDs directly from a pinned immutable HF snapshot, retain
  FastVisionModel merge only for adapter/local checkpoints, validate new GGUFs with a real
  CPU llama.cpp model load, and atomically write a size/SHA-256/tool-version sidecar only
  after success. Cache reuse now requires a matching sidecar and hash; stale/corrupt
  artifacts rebuild automatically. Quantization infrastructure failures propagate as hard
  errors, including across disposable-worker transport, and shared HF snapshots are never
  deleted.
- **Status:** 🟢 fixed with regression coverage for raw snapshots, missing-layer cache
  invalidation, sidecar/hash tampering, adapter merge, and baseline error propagation.

## B196 -- Qwen3.5-4B zero-shot inference bypassed incomplete-snapshot mitigation
- **Where:** `training/lora_trainer.py`, `training/slm_helpers.py`, and the shared Hugging
  Face cache snapshot at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **How found/evidence:** The beginning difficulty probe failed in its eval worker with
  `DownloadStallError: incomplete snapshot even with HF_HUB_DISABLE_XET=1`. At failure
  time the cached snapshot contained config/index/tokenizer metadata, but neither of the
  two shards named by `model.safetensors.index.json` (about 9.32 GB total).
- **Impact:** The zero-shot probe ran before training, so it entered Unsloth directly and
  bypassed training's prefetch. Training's prefetch also treated any `snapshot_download`
  return as success and swallowed all three failures, allowing Unsloth to retry an already
  known-incomplete cache.
- **Root cause:** Prefetch was training-only and checked neither immutable revision identity
  nor indexed shard completeness. The downloader used its default concurrency and failures
  were downgraded to a log message instead of an infrastructure error.
- **Fix:** Added a reusable HF snapshot prefetch/verifier used by training and inference.
  It first resolves and verifies a local-only snapshot without calling `model_info`, so
  fully warmed compute nodes need no network. An absent or incomplete local snapshot falls
  through to online resolution, which pins one commit; retries reuse the existing partial
  shared cache without deletion, back off boundedly, and download with
  `SLM_HF_DOWNLOAD_WORKERS` (default 1). Every shard named by a weight index must exist and
  be nonempty. Exhausted retries now raise a clear infrastructure error before Unsloth
  loads; local checkpoints remain no-ops.
- **Status:** 🟢 code fixed and unit-tested with mocked downloads/filesystems. No download
  or cluster job was run as part of this fix. A read-only post-fix check found that the
  shared cache had separately been warmed before this session: both indexed shards are
  present and nonempty (5,329,398,688 + 3,990,429,408 bytes). Their combined file size is
  9,319,828,096 bytes; the index's 9,319,737,856-byte tensor payload excludes 90,240 bytes
  of safetensors file overhead.

## B197 -- iterate prompt and generic reask produced contradictory branch payloads
- **Where:** `agent/nodes/iterate.py` (`_ITERATE_SYSTEM`, `_reask_json_only`,
  `_llm_iterate`).
- **How found/evidence:** Run `slm-ner-l40s-37531245` repeatedly logged an initial iterate
  response followed by `iterate_json_reask`, then
  `ValueError('hyperparams is not allowed for a data_rebuild intervention')`. At log lines
  662 and 757 the initial response consumed the full 1024-token output allowance; the
  parseable reask selected `data_rebuild` while still emitting `hyperparams`.
- **Impact:** Correct strict validation rejected the contradictory decision, after which
  fallback changed the action to the test-agent's `hyperparameter` recommendation. The
  intended LLM data intervention was therefore lost.
- **Root cause:** The system prompt showed both mutually exclusive payloads inside one
  combined JSON example. The sole retry asked generically for JSON again without reporting
  the validator error, so it did not tell the model which contradiction to repair. The
  1024-token ceiling also demonstrably truncated two initial decisions.
- **Fix:** Describe the decision as a discriminated union and provide separate valid
  `data_rebuild` and `hyperparameter` examples while retaining all bounded field
  constraints. The one allowed reask now receives the exact bounded/sanitized validation
  error without replaying malformed output or raw eval data. Both initial and reask output
  allowances are 1536 tokens; call count and tracked cost stages remain unchanged.
- **Status:** 🟢 fixed with regression coverage for branch-separated examples, retained
  schema bounds, exact-error repair, strict contradictory-payload rejection, the eval
  firewall, and the one-initial-plus-one-reask limit.

## B198 -- Unsloth assumed every adapter base had tokenizer.model during merge
- **Where:** `training/hf_cache.py`, `training/lora_trainer.py`; triggered by third-party
  `unsloth_zoo/saving_utils.py` during `save_pretrained_merged`.
- **How found/evidence:** Every Qwen3.5-2B adapter merge in run
  `slm-ner-l40s-37531245` printed `Cache check failed: tokenizer.model not found in local
  cache` and attempted a filtered Hub download. Official snapshot
  `15852e8c16360a2fea060d615a32b45270f8a8fc` intentionally has no
  `tokenizer.model`; it has a valid BPE `tokenizer.json` (12,807,982 bytes),
  `vocab.json` (6,722,759), and `merges.txt` (3,353,259). Unsloth had already called
  `tokenizer.save_pretrained`, and every adapter merge, GGUF conversion, and subsequent
  llama.cpp load validation succeeded.
- **Impact:** Performance/noise only for this Qwen snapshot: the warning did not indicate
  tokenizer or model corruption. It caused a redundant local cache miss and could make an
  unnecessary network request for a file that does not exist. A genuinely absent
  tokenizer representation still must be fatal.
- **Root cause:** Unsloth's remote-base merge branch unconditionally probes and downloads
  `tokenizer.model`, although its own tokenizer loader and saver support BPE tokenizers.
- **Fix:** Resolve and verify the immutable base snapshot with `local_files_only=True`.
  Stage the existing adapter with only `base_model_name_or_path` rewritten to that local
  snapshot, then let Unsloth load the local base plus the real adapter and merge from local
  shards. No site-package edits or fake tokenizer files are used. Shared snapshot
  validation now requires complete weights plus one valid tokenizer representation:
  `tokenizer.json`, `tokenizer.model`, or valid `vocab.json` + `merges.txt`; merged output
  is revalidated before GGUF conversion. Because Unsloth still owns the merge over the
  complete indexed local shard, Qwen3.5 MTP tensors remain on the same validated path.
- **Status:** 🟢 fixed with RED/GREEN coverage for Qwen BPE, SentencePiece, missing
  tokenizer rejection, local-only adapter resolution/loading, and full local-checkpoint
  regression. No GPU job, model download, or API call was run. A bounded GPU merge→GGUF
  load smoke remains advisable to validate this exact installed Unsloth/PEFT runtime path.

---

# Status reconciliation — 2026-07-29

A code-verified sweep of every entry still carrying 🔴 / 🟡 / ⚪. Each row below was checked
against a live symbol; the evidence column names it. This section is authoritative where it
disagrees with an older entry's inline `Status:` line.

## Now fixed (verified against code)

| ID | Was | Evidence it is now implemented |
|---|---|---|
| B22 | ⚪ Context Manager not implemented | `agent/context_manager.py::compact_trajectory` / `should_compact` / `estimate_token_count`; called from `iterate.py::_llm_iterate` |
| B24 | ⚪ `MAX_TURNS_MAIN` is dead config | `graph.py::guard_graph_node(max_steps=MAX_TURNS_MAIN)` enforces a durable cumulative `_graph_steps` cap **and** `compiled.with_config(recursion_limit=MAX_TURNS_MAIN)` |
| B25 | ⚪ DAG has no edges; π=(D,H,S) not stored | `evaluate.py::dag_node` writes `parent_iteration` plus `pi.D` (version/path/plan/plan_identity/config/composition), `pi.H` (complete identity), `pi.S` (task_type/supervision/loss_masking/loss_contract_version) |
| B26 | ⚪ Teacher models never called | CoT is now authored by the local Qwen3.6 synth endpoint only (`curate.py::_annotate_generation_cot` → `curriculum.py::annotate_cot` via `data.synth_client.get_generate_fn`); the cloud DeepSeek/OpenAI CoT teachers were removed (no cloud fallback) |
| B27 | ⚪ 3 of 5 quality controls missing | `curriculum.py::apply_quality_controls` implements label balancing, >3×-median length filtering, entity-value capping (≤3, NER), and Jaccard>0.9 dedup, task-routed |
| B28 | ⚪ `quantize.py` does no quantization | `training/quantize.py::quantize_from_model_spec` + `validate_and_record_gguf` + `validated_gguf_cache_hit` produce and load-verify real GGUFs |
| B30 | ⚪ No `apply_chat_template` / assistant-only masking | `lora_trainer.py::SFT_LOSS_CONTRACT_VERSION = 2` ("explicit assistant/completion-only labels"); `apply_chat_template` used incl. the multimodal inner-tokenizer path; prompt tokens are `-100` |
| B31 | ⚪ Dataset size hardcoded `N_TOTAL=150` | `task_analysis.py::_apply_data_targets` clamps the planner's `curriculum_size`/`eval_size` into `[floor, DATA_SIZE_CEILING]`; `curate` reads `curriculum_size_target` |
| B36 | ⚪ 2-for-1 rule not implemented | `curriculum.py::synthesize_hard_negatives` — "Generate hard negatives using the 2-for-1 rule (paper §2.3)"; returns gold anchor + synthetic pair, counted separately as `n_hard_source` / `n_hard_generated` |
| B43 | ⚪ `escalate_node` doesn't reset state | `escalate.py` resets `scores`/`dag`/`iteration`/`best_score`/`best_weights_ref`/`last_eval`/`consecutive_no_improvement`/rebuild plan/downward state, and preserves `lifetime_best_score` |
| B46 | ⚪ Classification extraction defaults to majority class | `scorers/classification.py::_UNKNOWN_LABEL = "__EXTRACTION_FAILED__"`; priority is word-boundary → substring → explicit failure |
| B47 | ⚪ Inference model cache never clears | `slm_helpers.clear_inference_cache()`, called by `escalate_node` on promotion |
| B48 | ⚪ NER web-acquired data lacks entity annotations | `web_acquire.py::_annotate_ner_entities` exists and runs during acquisition. **Caveat: see B203** — it does not re-validate spans/types |
| B49 | ⚪ `messages` field populated but never used | Field removed from `AgentState` entirely (`grep -c messages agent/state.py` → 0) |
| B98 | ⚪ `live_confirm` M0 re-inference not implemented | `production/live_confirm.py` Step 2 loads `deployed_model_ref` and re-runs each candidate. **Caveat: see B207** — it does not replay the serving prompt |

## Superseded by a deliberate design change (not "fixed", but no longer a gap)

| ID | Was | Why it no longer applies |
|---|---|---|
| B38 | ⚪ `filter_pool()` ignores `latency_ttft_ms` / `power_watts` | `filter_pool` now **deliberately** gates only on quantities that are known rather than modelled (`size_mb` vs storage/RAM, plus any real recorded measurement). `min_tok_s` is applied only where `config/measured_metrics.json` has a measurement; nothing is eliminated on an estimate. |
| B53 | ⚪ Tiering by RAM, not params; siblings mis-tiered | `_size_tier` buckets **real on-disk weight size**, replacing `_ram_tier`'s modelled peak-RAM figure. Tiers are explicitly documented as scale buckets, not capability classes; quant siblings occupying different tiers is now intended. See `PIPELINE.md` §9. |
| B39 / B40 | ⚪ Missing/oversized pool models from the design doc | Obsolete under **B139** (pool restricted to official Qwen). |

## Still open, unchanged

| ID | Status | Note |
|---|---|---|
| B21 | 🟢 | Closed by deletion — see **B208**. |
| B23 | 🟢 | Closed by deletion — see **B208**. |
| B32 | 🟢 | **Implemented 2026-07-30** — see **B32 (IMPLEMENTED)** below: sourced registry, else measured anchor + bounded headroom. 28 tests. |
| B45 | 🟢 | Fixed 2026-07-29. `EvalResult.metric` + `TASK_METRIC_NAMES` name the real measurement (`macro_f1` / `span_f1` / `exact_match` / `execution_pass@1` / `judge_mean_0_1`); every scorer returns it and the run summary prints it. The `f1` **field** is retained deliberately — it is the universal comparison scalar and checkpoint/DAG replay depend on the name. |
| B50 | 🟡 partial | The harness now exists (`hardware_eval/measure_model.py`, `on_device_eval.py::measure_llama_cpp` with real `getrusage` peak RSS). It is **opt-in** (`SLM_HW_VERIFY_ON_DEVICE=1`) and post-convergence only; pre-training screening still uses the all-`None` unmeasured profile. |
| B113 | ⚫ | **Closed-unreproducible 2026-07-30** — cold-cache test (job 37911562) downloaded and loaded it cleanly alongside two larger controls. See **B113 (cold-cache reproduction test)**. Not 'fixed': no code changed. |
| B116 / B119 | 🟢 | Both have "(fix applied)" entries; the 2026-07-10 summary table that still called them blockers is now annotated as historical. |

---

## B199 — production graph cannot start: `curate` requires an `eval_set` nothing builds

- **Where:** `agent/nodes/curate.py::curate_node` vs `agent/graph.py::build_graph(mode="production")`
- **Found:** 2026-07-29, code read while rewriting `PIPELINE.md`.
- **Symptom:** `curate_node` raises `RuntimeError("curate_node requires a fixed eval_set before
  rebuilding data")`. The production entry chain (`trace_ingest → taxonomy_construct →
  live_confirm → parent_awareness`) never constructs one — only the cold-start `eval_setup` node
  does, and it is not in the production graph.
- **Impact:** `mode="production"` is unrunnable end-to-end from the graph alone. A caller must
  pre-populate `state["eval_set"]`, and nothing validates that at graph entry, so the failure
  surfaces as a mid-graph crash rather than a startup error.
- **Status:** 🟢 **resolved 2026-07-29 by removing production mode.** The idea was scrapped, so
  the unrunnable path was deleted rather than completed. Removed: `agent/nodes/production/`
  (trace_ingest, taxonomy, live_confirm, parent_awareness), the production branch of
  `build_graph`, the production topology in `graph_topology_descriptor` (which now raises on
  any mode but `cold_start`), the `mode`/`deployed_model_ref`/`traces`/`failure_taxonomy`/
  `regression_set`/`replay_buffer` state fields, curate's replay-buffer allocation,
  `curation_log`'s `replay_count`/`failure_taxonomy` parameters, and `trace_ingest` from
  checkpoint's pregraph set.
  **Resume compatibility is preserved:** the `mode` parameter is retained through
  `checkpoint_compatibility` / `runtime_config_snapshot` / `run-manifest.json`, and the
  cold-start topology fingerprint is byte-identical because production nodes were never part
  of the cold-start descriptor. B207 (live-confirm prompt replay) is closed by the same
  removal.

## B200 — synthesis has never executed at scale; both completed runs were gold-only

- **Where:** `data/synth_client.py`, `agent/nodes/curate.py::_synthesize_positive_rows`
- **Found:** 2026-07-29, reading `logs/runs/*/cost.json`.
- **Evidence:** **zero** `hard_negative_synthesis` events in either ledger.
- **CORRECTED CAUSE (2026-07-30).** I originally attributed this to an unreliable endpoint
  ("8/9 preflight failures"). That was wrong — those 8 errors are the **startup preflight
  polling while vLLM boots**, 31 s apart from 15:58:49 to 16:02:28, followed by success. The
  endpoint came up in **both** runs (NER 1 success; math 25 successes / 18 errors).
  The real reason synthesis never ran is that the strategy was **never selected**:

  | Run | rebuilds | strategies actually executed | why no synthesis |
  |---|---|---|---|
  | NER | 31 | `resample_existing` ×31 | eligible, but all 65 LLM plans were rejected (the `hyperparams` bug) and the fallback keyword-matched to `resample_existing` (the prose-matching bug) |
  | math | 66 | `difficulty_weighted_sampling` ×40, `mine_new_real_source` ×4, `resample_existing` ×2 | `math_reasoning` is **ineligible** for `targeted_synth_positive` by design — correct behavior, not a bug |

- **Impact:** every claim about synthesis quality, yield, or cost is still untested in
  production — but the blocker was **selection logic, not infrastructure**. Both NER-side bugs
  are now fixed, so the next NER-shaped run is the first real test.
- **Status:** 🟡 unblocked, unverified. The fix for B201 (blocking + raising on a dead endpoint)
  addresses a real hypothetical risk but was **not** the observed cause here.

## B201 — a mid-run synth endpoint death degrades silently to gold-only

- **Where:** `agent/nodes/curate.py::_synthesize_positive_rows`
- **Found:** 2026-07-29, code read.
- **Symptom:** the driver's startup preflight (`tests/pipeline/run.py` phase 7) blocks until the
  synth server answers and aborts if it never does. But if the endpoint dies *after* that,
  `_synthesize_positive_rows` logs `Positive synthesis endpoint unavailable; retaining real rows
  only`, appends an `allocation_fallbacks` entry, and continues.
- **Impact:** a `targeted_synth_positive` plan silently becomes a `base_fill`. The run keeps its
  logged strategy attribution, so the DAG says synthesis was chosen while no synthesis occurred.
- **Status:** 🟢 **fixed 2026-07-29.** `data/synth_client.py::wait_until_available` blocks and
  polls (default 10 min via `SLM_SYNTH_MIDRUN_WAIT_S`, 15 s interval), then
  `curate._synthesize_positive_rows` raises `SynthesisUnavailableError` instead of returning
  gold-only rows. The wait is **clamped to the remaining aggregate wall-clock budget minus a
  5-minute reserve**, so blocking can never consume the time the run needs to checkpoint and
  write its summary. `SLM_REQUIRE_SYNTH=0` restores the old gold-only degradation.
  The mid-run wait is deliberately shorter than the 40-minute startup preflight: at startup
  nothing has been spent, mid-run every minute is charged against the wall clock.

## B202 — `_should_reexplore_downward` coerces a JSON string to a boolean

- **Where:** `agent/nodes/downward_probe.py::_should_reexplore_downward`
- **Found:** 2026-07-29, code read (also flagged in `PROMPTS.md` §1.8).
- **Symptom:** `decision = bool(obj.get("reexplore", False))`. A model replying
  `{"reexplore": "false"}` yields `bool("false") is True`, so the probe runs when the
  orchestrator declined.
- **Impact:** one unnecessary train+eval cycle per occurrence. Not a correctness risk to the
  final model (the probe is optional and adoption is gated on the real threshold), but it is
  wasted GPU time and a decision that does not match the log.
- **Status:** 🟢 **fixed 2026-07-29.** `_should_reexplore_downward` now requires a real JSON
  boolean and raises `ValueError` otherwise, which routes into the existing `skip_optional_error`
  path — a malformed reply preserves the converged model instead of triggering a probe the
  orchestrator declined.

## B203 — `_annotate_ner_entities` does not re-validate spans or types

- **Where:** `data/loaders/web_acquire.py::_annotate_ner_entities`
- **Found:** 2026-07-29, code read (also flagged in `PROMPTS.md` §1.5).
- **Symptom:** the prompt demands exact substrings and allowed types, but the parser only checks
  that each returned object has `text` and `type` keys. Spans are not rechecked as exact
  substrings of the passage and types are not allow-listed at this call site. Passages are also
  truncated to the first 500 characters, which can cut entity context.
- **Impact:** a silent parse or call failure produces an **empty-entity gold example**, which is
  indistinguishable downstream from a genuine negative — so acquisition noise becomes training
  signal that teaches the model to predict "no entities."
- **Never executed in either completed run (verified 2026-07-30).** Both ledgers contain
  **zero** `acquire_ner_annotation` events. The NER run loaded real gold from
  `tner/bc5cdr` (train=3403 / test=900) via the B119 benchmark loader, so the Exa-scrape +
  annotate fallback was never reached. **Every defect below is latent, not observed** — there
  were no annotation call failures to explain, because there were no annotation calls.
- **Status:** 🟢 **fixed 2026-07-29**, and the diagnosis grew: the worst defect was a **window
  mismatch** I had not spotted. The annotator saw `text[:500]` while the emitted row stored the
  **full** passage, so every entity past character 500 was unlabeled — a systematic
  false-negative generator on exactly the long passages NER finds hardest.

  Five changes in `_annotate_ner_entities`:
  1. **The annotated window IS the stored text** (`_NER_ANNOTATION_WINDOW_CHARS`, default 500,
     `SLM_NER_ANNOTATION_WINDOW`). No more prefix-annotate / whole-store.
  2. **Failures are dropped, not emitted as negatives.** `except Exception: entities = []`
     produced an empty-entity gold row indistinguishable from a genuine negative. Call errors
     and unparseable replies now drop the passage and are counted.
  3. **Spans are validated** as exact substrings of the annotated window via
     `_validate_ner_annotation`, with per-reason rejection counts
     (`not_substring` / `bad_type` / `malformed` / `duplicate`) logged.
  4. **Types are allow-listed** — from `task_plan["labels"]` when present (so BC5CDR's
     CHEMICAL/DISEASE are accepted rather than rejected as non-CoNLL), else
     `_DEFAULT_NER_TYPES`.
  5. **`raise_if_fatal` is called**, matching every other orchestrator call site. Previously a
     dead API key silently produced an entirely unlabeled NER corpus; now it aborts. One
     bounded retry covers a transient malformed reply.

  A **validated-empty** row is still kept — the call succeeded, the reply parsed, and no span
  survived. That is a legitimate negative and is now distinguishable from a failure in the log.

## B204 — RETRACTED (was: "context length pinned at 512 with no truncation diagnostics")

- **Status:** ⚫ **retracted 2026-07-29 — not a bug. The premise was false.**
- **What I claimed:** context is 512 tokens, rows are silently truncated, and a truncated row
  is misdiagnosed as a capacity limit → wrong escalation.
- **What the code actually does:**
  - `training/lora_trainer.py::_configured_max_seq_length` → **4096** (`SLM_MAX_SEQ_LENGTH`,
    clamped to [128, 32768]). The 512 figure was fixed by **B188** and no longer exists.
  - `_validate_training_sequence_lengths` tokenizes with `truncation=False` and **raises**:
    *"Refusing to silently truncate target-critical prompt or completion content."*
  - `slm_helpers.infer` and `infer_batch_gguf` both validate
    `prompt_length <= max_seq_length - max_new_tokens` and **raise** with
    *"No truncation was applied, so shorter rows were not generated with a corrupted batch."*
- **Where the error came from:** `docs/intervention_capability_audit.md` stated "Maximum
  sequence length: 512". I carried that into the `PIPELINE.md` rewrite without checking it
  against `lora_trainer.py`, then reasoned a failure mode out of it. The audit line was stale;
  the inferred failure mode never existed. Both docs are corrected.
- **What was actually missing, and is now added:** *visibility*. Over-length was a hard crash
  at 100% of the window with no warning at 95%. `_log_sequence_length_report` now emits the
  formatted-row token distribution (min/p50/p95/p99/max/mean, headroom) and warns on any row
  at or above `SLM_LENGTH_WARN_FRACTION` (default 0.90), to stdout and as a
  `sequence_length_report` timing event. `truncated` is reported as a structural `0`.

## B206 — `iterate_json_reask` fired on 86 of 141 iterate calls in the NER run

- **Where:** `agent/nodes/iterate.py::_reask_json_only`
- **Found:** 2026-07-29, reading `logs/runs/*/cost.json`.
- **Evidence:** NER: 141 `iterate` + **86** `iterate_json_reask` (61% reask rate, $4.90).
  Math: 63 `iterate` + **2** reasks (3%).
- **Root cause — CONFIRMED from the run log, not inferred.** `logs/runs/slm-ner-l40s-37531245/run.log`:

  ```
  [cost] stage=iterate            tokens=5454->877
  [cost] stage=iterate_json_reask tokens=5494->678
  [iterate] LLM call failed (ValueError('hyperparams is not allowed for a data_rebuild
            intervention')); using test-agent suggestion: hyperparameter
  ```

  The full chain: Claude proposed `data_rebuild` **plus** a `hyperparams` block →
  `_validate_decision_json` raised → reask (paid call #2) → the reask re-attached
  `hyperparams` and raised again → **both calls wasted** → fell through to the test-agent
  suggestion. That log has **86 reasks and 72 `intervention=hyperparameter` decisions, and
  zero `intervention=data_rebuild` decisions** — the same 65/65 rejection pattern behind the
  data_rebuild flexibility bug.

- **Why math showed only 2/63:** not because it was fixed, but because the task differed. The
  math run's difficulty profile kept steering the orchestrator to `hyperparameter`, which
  legitimately carries a `hyperparams` block and validated fine. NER's test agent kept
  suggesting `data_rebuild`, so NER kept hitting the rejection. The rate was
  **task-dependent, not run-dependent.**
- **Status:** 🟢 **fixed** by the `validated.pop("hyperparams")` strip in
  `_validate_decision_json` (the same change that fixed the data_rebuild flexibility bug). The
  exact exception in the log above can no longer be raised. Confirm on the next NER-shaped run
  that `iterate_json_reask` has collapsed toward the math run's ~3%.

## B207 — live confirmation does not replay the original serving prompt

- **Where:** `agent/nodes/production/live_confirm.py`
- **Found:** 2026-07-29, code read (also flagged in `PROMPTS.md` §5.2).
- **Symptom:** sends `trace["input"]` directly with no task prompt, then compares
  `output.strip()` to `corrected_output` by exact string equality.
- **Impact:** exact string equality is wrong for most classification (label casing/prose), NER
  (JSON key/entity ordering), and generation outputs, so genuine passes are recorded as confirmed
  failures and enter the training set as "failures M0 reproduces." Infrastructure errors are also
  conservatively counted as confirmed failures, compounding it.
- **Status:** ⚪ design gap. Store the rendered prompt plus task/parser metadata in each trace and
  replay through the same scorer; distinguish infrastructure errors from real failures.

---

## B113 (analysis) — why only Qwen3.5-2B stalls, and what actually fixes it

Reopened for analysis 2026-07-29. Status: 🟡 **live infra risk, not a blocker.** The model is
still in the pool (`Qwen/Qwen3.5-2B`, tier 2), so escalation can still land on it with a cold
cache.

### Why this model and not the other five — MY EARLIER THEORY WAS WRONG

On 2026-07-29 I wrote that the cause was "file count × xet × a partially-warm cache." The
cache inventory taken 2026-07-30 **falsifies that**:

| Model | weight shards | files | cached size | ever stalled? |
|---|---|---|---|---|
| Qwen3-0.6B | 1 | 10 | 1.5 G | no |
| Qwen3.5-0.8B | 1 | 13 | 1.7 G | no |
| Qwen3-1.7B | 2 | 12 | 3.9 G | no |
| **Qwen3.5-2B** | **1** | **13** | **4.3 G** | **yes** |
| Qwen3-4B-Instruct-2507 | 3 | 13 | 7.6 G | no |
| Qwen3.5-4B | 2 | 14 | 8.8 G | no |

Qwen3.5-2B has **one** weight shard, not many. Qwen3-4B-Instruct-2507 has the same file count
and is 1.8× larger; Qwen3.5-4B has more files and is 2× larger. Both download fine. File
count, total size, and shard count therefore all fail to explain it.

**Honest position: the cause is not determined from available evidence.** What is factual:

- It stalled twice on 2026-07-10, with and without `HF_HUB_DISABLE_XET=1` — and `unsloth_zoo`
  already retries with xet disabled internally, so the env var was redundant either way.
- It **completed successfully later that same day**: every blob in
  `.hf-cache/hub/models--Qwen--Qwen3.5-2B` is timestamped 2026-07-10 11:32–11:33, with **zero**
  broken symlinks and **zero** `.incomplete` blobs.
- Nothing has exercised the download path since, because a complete local snapshot
  short-circuits it.

The most likely remaining explanation is a **transient compute-node network fault during one
large blob transfer**, which self-resolved on retry the same day. That is consistent with every
observation and is not a property of the model. It is a guess, and it is labelled as one.

### The three fixes, and why each works

| Fix | Mechanism | Trade-off |
|---|---|---|
| **(a) Warm the shared HF cache from a login node** | Removes the download from the critical path entirely — `resolve_hf_snapshot(..., local_files_only=True)` then finds a complete snapshot and never calls the Hub. Login nodes have reliable network; compute nodes do not. | Manual step, and it must be repeated whenever the pool gains a model. This is standard HPC practice: model weights are infra, not part of the loop's decisions. **Recommended.** |
| **(b) Switch the pool entry to the `unsloth/Qwen3.5-2B` mirror** | A different repo layout with fewer/consolidated files and one that Unsloth's loader is tested against, so the per-file stall surface shrinks. | Changes model identity in the DAG/baselines, and the mirror can lag upstream. Also does not fix the class of problem for the next multimodal repo. |
| **(c) Bounded download-with-retry wrapper before training** | Re-drives `snapshot_download` with backoff until the snapshot passes the same completeness check the pipeline already runs (`verify_manifest_hashes` style), so a transient partial fetch self-heals instead of failing the iteration. | Adds code on the hot path and can mask a genuinely bad repo; needs a hard attempt cap so a permanently-missing file still fails loudly. |

**Recommendation: (a) now, (c) as the durable fix.** (a) costs one command and eliminates the
risk for the current pool; (c) is what makes the pipeline autonomous across future pool
changes. (b) is a workaround that trades one unverified repo for another.

**Note:** the pool-measurement sweep (job 37905779) sequences 3.5-2B **last** and continues
past a failure, so it doubles as a live test of whether this still reproduces — and whether
the cache is warm enough that (a) is already effectively satisfied.

---

## B32 (design) — orchestrator-driven leaderboard research

Status: ⚪ **open by design; no code written.** Recorded here because the fix is a real design
choice, not a patch.

**The problem.** `task_planner.plan_task` asks the orchestrator to calibrate `stop_threshold`
against "published SOTA at the target model size" using nothing but its own recall. That recall
is frozen at training time and is systematically wrong in the direction that matters: the
benchmark research (`docs/Evan's Notes/07-28b-benchmark-research.md`) found
BANKING77 SOTA is ~94.8% from a **110M** encoder, and that MedQA is saturating. A threshold set
from stale recall either (i) sits under a base model's zero-shot, so the run "converges"
immediately having learned nothing, or (ii) sits above achievable SOTA, so every tier fails and
the run burns its full budget concluding "infeasible."

**Four implementations, cheapest first.**

**1. Versioned local registry + LLM adjustment only (no tools).**
A checked-in `config/benchmark_baselines.md` — same pattern as `config/model_capabilities.md`,
which already solves this problem for model capabilities: sourced, human-readable, cached by
`capability_sections()`, with a metric-comparability contract. The orchestrator gets the
registry rows for the matched benchmark and may only propose a *delta* with a cited row. No
network, no new failure mode, fully reproducible.
*Cost: ~0. Staleness: whatever you last curated. Best default.*

**2. One bounded Exa search at plan time, results as untrusted data.**
Re-enable a `web_search` call (the deleted `agent/tools/web_search.py` is the template) inside
`task_analysis`, restricted to a fixed query template
(`"{benchmark} state of the art {param_range} 2026 leaderboard"`), capped at N results, with
snippets injected in a marked-untrusted block — exactly the containment the judge prompt
already uses. Extract `(metric_name, value, model, params, source_url)` and require the metric
name to match the eval harness's `TASK_METRIC_NAMES` entry before it can move the threshold.
*Cost: ~$0.007/run (Exa) + one orchestrator call. Risk: prompt injection from search results,
which the untrusted-block pattern bounds.*

**3. Structured leaderboard APIs instead of free-text search.**
Query sources that return typed rows rather than prose: the HF Open LLM Leaderboard dataset,
`paperswithcode` SOTA endpoints, or BFCL's published JSON. A typed row carries metric name,
value, and model size natively, so the comparability check is mechanical rather than an LLM
judgment. This is the only option where "is this number comparable?" has a real answer.
*Cost: ~0. Risk: coverage — these cover famous benchmarks well and domain benchmarks (BC5CDR)
poorly, so it needs (1) as a fallback.*

**4. Measure it yourself and skip the literature.**
You already run every candidate's zero-shot baseline at iteration 1. Set the threshold from
`max(baseline_f1) across the feasible pool + a required improvement margin`, rather than from
any external claim. This is self-consistent, needs no network, and cannot go stale.
*Cost: 0 extra (the baselines already run). Weakness: it answers "did fine-tuning help?" not
"is this competitive?" — so it is the wrong basis for a paper claim, and right for a stopping
rule.*

**Recommendation: (1) + (4) together, then (3) for the benchmarks it covers.** Use (4) as the
stopping rule (it is measured, local, and honest), use (1) to sanity-check that the target is
not absurd relative to published work, and add (3) only for benchmarks where a typed
leaderboard exists. Treat (2) as a last resort: free-text search is the highest-variance and
highest-injection-risk option, and it is the one whose output you can least mechanically verify.

**Wire-in point:** `agent/task_planner.py::plan_task`, before the threshold is written to
`state["stop_threshold"]` / `initial_stop_threshold`. The immutable-floor mechanism already
exists, so a researched threshold slots into it without new state.

---

## B208 — dead tool package deleted (closes B21 and B23)

- **Where:** `agent/tools/` (`bash_tool.py`, `file_tools.py`, `web_search.py`,
  `delegate_task.py`, `query_traces.py`, `__init__.py`)
- **Found:** 2026-07-29, verified by grep — zero references outside the package itself.
- **Why it mattered:** not a runtime cost, an audit liability. `web_search` owned the cost stage
  `iterate_web_search` and `delegate_task` owned `delegate_task`, so the ledger's stage
  vocabulary implied coverage of paths that could never fire. `query_traces` was
  production-mode-only.
- **Status:** 🟢 **deleted 2026-07-29.** Closes **B21** (delegate_task has no call sites) and
  **B23** (four `@tool`-decorated tools never invoked) by removal rather than by wiring. If
  tool use returns, `web_search.py` is the template for B32 option 2 — recoverable from git
  history.

---

## B113 (result) — did NOT reproduce; cache is warm and the model builds fine

- **When:** 2026-07-29, slurm job **37905779** (`tests/pipeline/measure_pool_sizes_l40s.slurm`).
- **What was run:** the pool-measurement sweep sequenced `Qwen/Qwen3.5-2B` last and continued
  past failures. I claimed this "doubled as a live reproduction test." **It did not.**
- **Result:** `Qwen/Qwen3.5-2B` converted and quantized cleanly —
  `Q4_K_M = 1251.4 MB`, `Q8_0 = 1980.5 MB`, `bf16 = 4337.5 MB` (**one** 4.3 GB safetensors
  shard; an earlier version of this entry said "13 shards", which was wrong — 13 is the total
  file count, including tokenizer and preprocessor configs).
- **Why it is not a reproduction test:** `resolve_hf_snapshot` found the complete local
  snapshot and **never contacted the Hub**. The download path — the thing that failed — was
  not executed. A test of B113 requires an empty or evicted cache entry.
- **Why:** the shared `HF_HOME` cache at
  `/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache` is now warm for this repo, which is
  fix **(a)** from the analysis above — already effectively in place. `resolve_hf_snapshot`
  finds a complete local snapshot and never calls the Hub.
- **Status:** ⚫ superseded by the cold-cache test below. (Original note retained:) latent, mitigated by a warm cache. The root cause (file count × xet ×
  partially-warm cache) is unchanged, so a cache eviction or a new multimodal pool entry can
  bring it back. Fix **(c)** — a bounded download-with-retry that re-drives `snapshot_download`
  until the snapshot passes a completeness check — remains the durable answer and is still not
  implemented. Downgraded from 🔴 because it is not blocking anything today.

## B209 — pool size arithmetic verified across all six families (negative result)

- **Where:** `config/android_pool.py::ModelSpec.size_mb`, `config/measured_metrics.json`
- **When:** 2026-07-29, slurm job 37905779.
- **Why it was checked:** the pool falls back to bytes-per-parameter arithmetic for any variant
  without a real measurement, and that arithmetic had been wrong by 20% exactly once before
  (Qwen3.5-4B Q4_K_M: predicted 2200 MB, measured 2654.5 MB) — enough to move a tier. Only one
  family had ever been measured, so the other five were gated on an unverified constant.
- **Result — 15 new measurements, 18/18 variants now real:**

  | Quant | measured vs arithmetic | tier changes |
  |---|---|---|
  | Q4_K_M | **+4.7 … +4.9%** (uniform) | 0 |
  | bf16 | **+4.8 … +4.9%** (uniform) | 0 |
  | Q8_0 | **−1.1%** (uniform) | 0 |

- **Interpretation:** the original 20% error was real *at the time* and was fixed by raising the
  Q4_K_M constant to 0.55 GB/1B params. This sweep confirms that correction **generalizes to
  every family** rather than being specific to the 4B. The residual bias (~4.8% light on
  Q4_K_M/bf16, ~1.1% heavy on Q8_0) sits well inside the 250–750 MB tier bands, which is why
  nothing moved. **Every hardware gate and every tier assignment in both completed runs was
  already correct.**
- **What is still unmeasured:** runtime metrics (peak RSS, tok/s, TTFT). `llama-cli` is not on
  the venv PATH — only `llama-quantize` is — so `measure_llama_cpp` could not run. Those
  fields stay **absent**, which makes consumers correctly report "unmeasured" instead of
  trusting a guess. To fill them: build `llama-cli` via `scripts/build_llamacpp_cuda.sh` and
  re-run `python hardware_eval/measure_pool.py --force`.
- **Status:** 🟢 verified. Recorded in `config/measured_metrics.json::_sweep_findings` so the
  next person does not have to re-derive it.

---

## B32 (revised design) — local registry, else first-eval baseline + orchestrator headroom

Supersedes the four-option list above. This is the approach Evan proposed on 2026-07-30, with
the failure modes it has to defend against.

**The proposal.** (1) Keep a local, versioned database of published metrics. (2) If the
benchmark is in it, calibrate the threshold from there. (3) If not, run the first fine-tuning
pass, use *that* eval as the baseline, and have the orchestrator set the goal a little higher
based on how much headroom it judges to exist.

**Assessment: this is the right shape.** It puts a measured number at the centre, keeps the
literature as a sanity check rather than an oracle, and never blocks on the network. Three
concrete hazards, each with a cheap guard.

**Hazard 1 — anchoring on a bad first config.** Iteration 1 uses `_DEFAULT_CONFIG`
(r16/α32/wd 0.01/lr 2e-4/3 epochs). If that config happens to be poor for the task, the
baseline is artificially low, so `baseline + headroom` sets a goal the run clears immediately
and terminates having learned almost nothing.
*Guard:* anchor on `max(zero_shot_baseline, first_finetune_score)`. The zero-shot baseline is
already measured at iteration 1 for free and is config-independent, so it floors the anchor
against a bad hyperparameter draw.

**Hazard 2 — the threshold becomes unfalsifiable.** A goal derived from your own first result
can always be met by lowering it, and `iterate` can already lower `stop_threshold` at runtime.
Combined, "did we hit the target?" stops being a real question.
*Guard:* `initial_stop_threshold` already exists as an immutable floor — set it from this
calibration **once**, and keep the existing rule that runtime adjustment can only move
`stop_threshold` down toward that floor and never below it. Also log the anchor, the headroom
the orchestrator asked for, and its stated reason, so the target's provenance is auditable.

**Hazard 3 — "how much headroom" is exactly the judgment an LLM is worst at.** Asking for a
number invites a confident guess. Asking for a *bounded* choice with evidence does not.
*Guard:* give it a small ordinal set — e.g. `{+0.02, +0.05, +0.10, +0.15}` — snapped like every
other bounded field in this repo, and require a one-sentence reason citing the per-difficulty
report (a large easy/hard gap implies real headroom; uniformly weak buckets imply little).

**Recommended shape:**

```
1. registry hit?          -> threshold = registry_value adjusted by a bounded, cited delta
2. no registry entry?     -> anchor = max(zero_shot, first_finetune)
                             threshold = anchor + orchestrator_headroom  (bounded set)
3. write once to initial_stop_threshold (immutable floor)
4. log anchor / headroom / reason / source into the run manifest
```

**Why this beats free-text web search:** the registry is reviewable and diffable, the anchor is
measured on *your* eval set with *your* prompt, and neither can be moved by a prompt injection
in a scraped page. The one thing it gives up is currency — a registry goes stale — which is why
step 1 keeps a bounded, cited adjustment rather than treating the stored value as fixed.

**Cost:** zero extra API calls (the zero-shot baseline and the first fine-tune already run; the
headroom choice rides along in the existing `task_analysis` call).

**Wire-in point:** `agent/task_planner.py::plan_task` for step 1; a new post-first-eval
calibration hook in `evaluate_node` (or `iterate` on `iteration == 1`) for step 2. Note this
makes the threshold *late-bound* for uncatalogued tasks, so `initial_stop_threshold` must
tolerate being set on the first evaluation instead of at plan time.

---

## B113 (cold-cache reproduction test) — did NOT reproduce; downgraded to closed-unreproducible

- **When:** 2026-07-30, slurm job **37911562**
  (`tests/pipeline/b113_cold_download_l40s.slurm`), node g3102.
- **Why a new job was needed:** job 37905779 built 3.5-2B successfully and I called that a
  reproduction test. It was not — the shared cache already held a complete snapshot, so
  `resolve_hf_snapshot` never contacted the Hub. This job points `HF_HOME` at an empty
  `/tmp` directory, so the download path actually executes. The shared `.hf-cache` is never
  read or modified.
- **Design:** the suspect plus **two controls** in the same job on the same node, because the
  2026-07-30 cache inventory had already falsified the "high file count" theory (3.5-2B has
  **one** weight shard; the controls have 2 and 3 at up to 1.8× the size). If all three failed
  it would be the node; if only the suspect failed it would be model-specific.

| Model | role | shards | download | config+tokenizer load | outcome |
|---|---|---|---|---|---|
| **Qwen/Qwen3.5-2B** | **suspect** | 1 | **46.6 s** | **63.0 s** | ✅ downloaded + loaded |
| Qwen/Qwen3-1.7B | control | 2 | 36.6 s | 0.6 s | ✅ downloaded + loaded |
| Qwen/Qwen3-4B-Instruct-2507 | control | 3 | 68.7 s | 0.5 s | ✅ downloaded + loaded |

Snapshot integrity for all three: **0 broken symlinks, 0 `.incomplete` blobs.** 16 GB fetched
in total. `HF_HUB_DISABLE_XET=1` was set, matching the 2026-07-10 conditions.

- **One incidental observation:** 3.5-2B's config+tokenizer load took **63 s** versus 0.5–0.6 s
  for both controls — ~100× slower. That is its 248,077-token multimodal vocabulary and
  video/image preprocessor configs, not a fault, but it is the one dimension where this model
  really is an outlier. Worth remembering if a first-iteration timeout ever looks mysterious.
- **Verdict:** B113 **does not reproduce** on a cold cache on this node. Combined with the
  2026-07-10 blob timestamps (11:32–11:33, i.e. the download succeeded later the same day as
  the failure), the evidence points to a **transient compute-node network fault that
  self-resolved**, not a property of the model or of xet.
- **Status:** ⚫ **closed — unreproducible.** Not "fixed": no code changed, and a transient
  network fault can recur. If it does, the durable answer is still fix **(c)** — a bounded
  download-with-retry gated on a snapshot completeness check — and
  `tests/pipeline/b113_cold_download_l40s.slurm` is now the harness to confirm it.
  The earlier "why only this model" theory in **B113 (analysis)** stays retracted; nothing in
  this test supports a model-specific cause.

---

## B32 (IMPLEMENTED 2026-07-30) — sourced registry, else measured anchor + bounded headroom

Implements the revised design above. **What was replaced is listed first, because the previous
mechanism was invisible unless you read the planner prompt.**

### What the threshold used to be, and what was deleted

| # | Old mechanism | Where | Fate |
|---|---|---|---|
| 1 | Prompt: *"PRIMARY RULE: anchor it to the PUBLISHED STATE-OF-THE-ART for this task's benchmark at this model-size class ... set stop_threshold at or just below that SOTA (roughly SOTA − 2 to 5 points)"* — pure LLM recall, evidence optional | `task_planner.py::_PLANNER_PROMPT` | **DELETED** |
| 2 | Schema field `"stop_threshold": float in [0,1] anchored to published SOTA` | `_PLANNER_PROMPT` | **DELETED**, replaced by `"threshold_headroom"` |
| 3 | `plan.setdefault("stop_threshold", 0.96)` — silent 0.96 whenever the LLM declined | `task_planner.py::plan_task` | **DELETED**; `plan.pop("stop_threshold")` now discards any number a model emits anyway |
| 4 | `if not state.get("stop_threshold"): state["stop_threshold"] = 0.96` | `task_analysis.py` | **DELETED** |
| 5 | Inline `SLM_STOP_THRESHOLD` override block | `task_analysis.py` | **MOVED** into `_calibrate_stop_threshold`; behavior unchanged (still pins both values and now also disables calibration) |
| 6 | Prompt preamble: *"calibrate stop_threshold against THESE specific models' published benchmark scores"* | `_PLANNER_PROMPT` | **REWRITTEN** — the pool is now context for data-size and headroom choices only |

`config.DEFAULT_STOP_THRESHOLD` (0.96) survives **only** as the pre-graph placeholder in
`run.py`'s initial state; `task_analysis` overwrites it before any training happens.

### What replaced it

**New files:** `config/benchmark_baselines.md` (the registry) and `agent/threshold.py` (the
calibration logic). **New state field:** `threshold_calibration` — the audit trail.

Source 1 — **registry**, `agent/threshold.py::registry_lookup`. A row calibrates only when its
`metric` matches `eval/harness.py::TASK_METRIC_NAMES[task_type]`. Verified behavior:

| Benchmark | registry metric | our metric | calibrates? |
|---|---|---|---|
| GSM8K | `exact_match` | `exact_match` | ✅ → threshold **0.8479** (0.8779 − 0.03) |
| i2b2-2014 de-id | `span_f1` | `span_f1` | ✅ → threshold **0.9485** |
| BANKING77 | `accuracy` | `macro_f1` | ❌ informational only |
| MedQA | `accuracy` | `macro_f1` | ❌ informational only |
| BC5CDR, SMS Spam | `n/a` | — | ❌ no comparable figure recorded |

Source 2 — **measured anchor**, late-bound. No usable row → `task_analysis` parks
`stop_threshold` at `UNREACHABLE_PENDING_THRESHOLD = 1.0` (above the 0.99 ceiling, so nothing
can converge before the anchor exists) and sets `pending: True`. `evaluate_node` then
calibrates at the end of iteration 1:

```
anchor    = max(zero_shot_baseline, first_finetune_score)
threshold = min(0.99, anchor + headroom)
```

and writes `initial_stop_threshold` **once**.

### The three hazards, and the guard for each

1. **Bad first hyperparameter draw.** Iteration 1 uses `_DEFAULT_CONFIG`; a poor draw would
   depress the anchor and set a goal the run clears instantly.
   → `max(zero_shot, first_finetune)`. Zero-shot is config-independent and already measured
   for free at iteration 1. Verified: `zs=0.82, ft=0.31 → 0.87` (uses zero-shot);
   `zs=0.82, ft=0.91 → 0.96` (uses the fine-tune).
2. **An unfalsifiable target.** A goal derived from your own first result can always be met by
   lowering it, and `iterate` can already lower `stop_threshold`.
   → Written **once** to the existing immutable floor; every input recorded in
   `threshold_calibration` (`source`, `threshold`, `headroom`, `zero_shot`, `first_finetune`,
   `reason`, `registry_row`).
3. **"How much headroom" is what an LLM is worst at.**
   → `VALID_HEADROOMS = (0.02, 0.05, 0.10, 0.15)`, snapped like every other bounded field,
   with per-rung guidance in the prompt and a required justification in `rationale`.

**Edge cases handled:** a failed baseline records `None` (not 0.0) and calibration proceeds on
the fine-tune alone; if *neither* number exists, calibration stays pending and retries rather
than inventing a target; the 0.99 ceiling prevents targeting a perfect score, which label noise
makes unreachable anyway.

**Cost:** zero extra API calls. The zero-shot baseline and first fine-tune already run; the
headroom rides along in the existing `task_analysis` call.

- **Status:** 🟢 implemented with 28 tests in `tests/test_threshold_calibration.py`; full suite
  green (792 → 820 passed).

## B210 — the iterate prompt masked the union rule instead of stating it

- **Where:** `agent/nodes/iterate.py::_ITERATE_SYSTEM`, `_validate_decision_json`
- **Found:** 2026-07-30, from Evan's observation that stripping `hyperparams` hides the problem
  rather than fixing it.
- **The problem with the strip alone:** `validated.pop("hyperparams", None)` removes the key
  from the decision dict. It stops the invariant violation, but the orchestrator is never told
  it did anything wrong, and the run log said nothing — so a persistently confused orchestrator
  looked identical to a compliant one. The old prompt only said *"choose exactly one of these
  two branches and never merge their payloads"* in a single sentence, which demonstrably did
  not work: **65 of 65 data_rebuild decisions in the NER run carried a `hyperparams` block.**
- **Fix, two parts:**
  1. **The prompt now states the contract as its own block** — a `CHOOSE EXACTLY ONE` section
     naming, per branch, the REQUIRED key and the **FORBIDDEN** key; why one change per
     iteration is non-negotiable (two simultaneous changes make score movement unattributable);
     and exactly what the system will do if the rule is broken (discard the field, log the
     correction, execute the data plan with the current best config).
  2. **The strip is now visible.** `_validate_decision_json` records
     `_dropped_fields: ["hyperparams"]`, and `iterate_node` logs
     `NOTE — dropped hyperparams from this data_rebuild decision: …`. The strip is retained as
     defense-in-depth — raising is what wasted two paid calls per iteration and discarded 65
     data plans — but it is no longer silent.
- **Status:** 🟢 fixed, with prompt-contract and strip-visibility assertions in
  `tests/test_threshold_calibration.py`.

## B211 — curate design cleanups: no upper truncation, resample gating, dead-code removal

- **Where:** `agent/nodes/curate.py`, `agent/data_rebuild.py`, `agent/nodes/iterate.py`,
  `data/curriculum.py`, `agent/state.py`
- **Found:** 2026-08-02, from Evan's review of the data-curation redesign (see
  [Evan's Notes 2026-08-01](Evan's%20Notes/08-01-pipeline-deep-dive.md)).
- **Three changes:**
  1. **`target_rows` is now a floor, not a cap.** curate previously ended with
     `dataset = dataset[:target_rows]`, hard-truncating the curriculum. That line is removed:
     `target_rows` only drives `_synth_fill_to_target` (top up when short). The allocation loop
     still stops at `target_rows` in the common case, but a legitimate `acquire`/`synthesize`
     overshoot now keeps its extra rows rather than discarding real signal to hit an exact count.
     Rationale: never train on *too few* rows; too many is fine.
  2. **`resample` is gated when the pool is exhausted.** New
     `resample_pool_exhausted(pool_texts, curriculum_texts)` (subset test) +
     `resample_available_for_state(state)` in `data_rebuild.py`. curate computes
     `resample_available` from the eval-decontaminated pool vs the previous artifact **before**
     plan resolution and threads it into `normalize_data_rebuild_plan` (redirects
     `resample → synthesize`) and the fallback planner (drops the `resample` weight). iterate
     sets `state["resample_available"]` and threads it into the validator, fallback, system
     prompt, and a per-turn "resample unavailable" note. Reshuffling a pool that is already
     wholly in the curriculum adds no novelty, so it is taken off the menu.
  3. **Dead code deleted.** `_difficulty_sample`, `_with_train_difficulty`, and the
     curate-local `_DIFFICULTY_BUCKETS` (zero call sites since the 2026-07-31 redesign) removed
     from `curate.py`; `build_initial_curriculum` (no live callers — it predates the declarative
     `curate_node` path; historical refs in B34/B502/B684/B1314/B1316 describe its *former*
     behavior) removed from `curriculum.py` along with its now-unused `EvalSet`/`normalize_text`/
     `_infer_pos_label`/`_infer_neg_label` imports.
- **Status:** 🟢 implemented; all five edited modules parse; `resample_available` added to
  `AgentState`.

---

## B212 — removed the vestigial pos/neg/boundary eval slices (no functional effect)

- **Where:** `data/eval_set.py`, `eval/metrics.py`, `eval/scorers/{classification,ner,generation,function_call,diff}.py`,
  `eval/harness.py`, `eval/endpoint_eval.py`, `data/curation_log.py`, `agent/state_codec.py`,
  `agent/checkpoint.py`, `agent/nodes/cold_start/eval_setup.py`, `hardware_eval/quant_accuracy_eval.py`,
  `agent/nodes/cold_start/model_selection/interpolation.py`, `scripts/prepare_shared_dataset.py`
- **Found:** 2026-08-02, from Evan's question "does pos/neg/boundary have any effect on anything?"
  (see [Evan's Notes 2026-08-02 Q4](Evan's%20Notes/08-02-firewall-test-agent.md)).
- **The finding.** The eval set carried three slices — `pos`, `neg`, `boundary` — with **no
  functional effect**. Tracing every read: the slices were only ever recombined into `.all` (what
  all eval, prompting, difficulty labeling, and firewall matching use); the per-slice scores
  (`per_slice_scores` → `EvalResult.pos_score/neg_score/boundary_score`) flowed to exactly one
  place — a cosmetic `Epos | Eneg | Eboundary` line in the curation log. **Nothing branched on
  them.** For the NER/generation families the split was a meaningless random 3-way partition anyway.
- **The change.** Deleted the slices end to end: `EvalSet` is now a single flat `.all` sample;
  `per_slice_scores` and the three `EvalResult` score fields are gone; the five scorers no longer
  emit a `"slices"` key; the curation-log line is removed. `build_eval_set(examples, task_type,
  target=…)` now returns a target-sized sample — **label-coverage stratification (round-robin
  across labels) is retained for multi-class classification**, everything else is a shuffled top-N.
  `_eval_split_sizes` → `_eval_target` (single int, min-30 floor). `eval_set.json` shrank to
  `counts={"total": …}` + `examples=[…]`.
- **Back-compat.** `EvalSet.from_serialized` folds legacy `pos+neg+boundary` into `all`, so old
  checkpoints and artifacts still load. Serialization (`state_codec`, sqlite `checkpoint`) now
  emits/reads `all` and decodes via `from_serialized`.
- **Status:** 🟢 implemented. No behavior change to any decision path (the removed data drove
  nothing). Only visible changes: the curation-log line and `eval_set.json` schema shrink, and
  multi-class eval sampling is now explicit label round-robin instead of implicit-via-slices. All
  changed production files parse; the affected test subset matches the pre-change pass/fail baseline
  (remaining failures are pre-existing: missing `langgraph-checkpoint-sqlite`, Windows cp1252
  `read_text()`, and absent local dataset bundles).

---

## B213 — CLINC150 loader's bare `clinc_oos` repo id crashed `eval_setup` under huggingface_hub 1.x

- **Where:** `data/loaders/clinc150.py` (`HF_ID`); surfaced through
  `agent/nodes/cold_start/eval_setup.py::_load_named_benchmark`.
- **Found:** 2026-08-04, first real CLINC150 run on the dedicated L40S partition
  (`slm-clinc150-l40s-38084400`) died 300.6 s in, at the `eval_setup` node, 0 iterations.
- **Symptom.**
  ```
  huggingface_hub.errors.HfUriError: Invalid HF URI
  'hf://datasets/clinc_oos@155b9c710419136e17307b80d0a13e68cd46b4ec/.huggingface.yaml'.
  Repository id must be 'namespace/name', got 'clinc_oos'.
  ```
- **Root cause.** `HF_ID = "clinc_oos"` was a *legacy canonical* (namespace-less) dataset name.
  `huggingface_hub` 1.23.0 / `datasets` 4.3.0 resolve every dataset reference by constructing an
  `hf://datasets/<id>@<rev>/...` URI, and `parse_hf_uri` hard-rejects any repo id without a
  `namespace/name` pair. The canonical alias still resolves through `HfApi.dataset_info`
  (`clinc_oos` → `clinc/clinc_oos`), which is why the id looked valid and why nothing caught it:
  **the failure is at load time on the compute node, and the loader's unit test only covered the
  pure `convert_clinc150_rows` converter, never the repo id.** CLINC150 was the only one of the six
  curated loaders using a bare name — the other five were already namespaced.
- **Fix.** `HF_ID = "clinc/clinc_oos"`. Verified live: the `plus` config loads, `intent` is a
  151-name `ClassLabel` (150 intents + `oos`), splits 15250/3100/5500.
- **Regression guard.** New parametrized `test_curated_loader_repo_ids_are_namespaced` in
  `tests/test_benchmark_loaders.py` asserts every repo-id constant across all six curated loaders
  (`HF_ID`, `DIALOGSUM_ID`, `SAMSUM_ID`, `XLAM_ID`, `BFCL_ID`) is exactly `namespace/name`. It
  failed only on clinc150 before the fix.
- **Status:** 🟢 fixed.

---

## B214 — CLINC150 head-sliced an intent-grouped split, loading 33 of 151 intents

- **Where:** `data/loaders/clinc150.py` (`load_clinc150._conv`).
- **Found:** 2026-08-04, immediately after the B213 fix, while verifying the loader end to end.
  A 120-row smoke load returned only **2 distinct labels**, which does not happen if a split is
  shuffled.
- **Root cause.** CLINC150's HF splits are **grouped by intent** — all ~100 rows of intent 61,
  then the next intent, and so on. The loader asked for `split=f"{split_name}[:{limit}]"`, a
  *prefix* of that grouped order. At the run's real sizes this yielded:

  | split | requested | intents covered (of 151) |
  |---|---|---|
  | train | `train[:3250]` | **33** |
  | test | `test[:800]` | **27** |

  So the model would have trained on 33 intents and been scored on a partly disjoint 27, over a
  mismatched label space, with `oos` present or absent by accident of ordering. Downstream
  label-coverage stratification in `build_eval_set` cannot repair this: it round-robins over the
  labels *present in the pool*, and the missing 118 intents were never loaded.
- **Why it was invisible.** This is a silent data-quality failure, not a crash — the run would
  have burned GPU-days and converged to a meaningless macro-F1. The loader's only test covered
  the pure `convert_clinc150_rows` converter, which never sees the split.
- **Fix.** Read each split **in full**, then sample with a new pure
  `stratified_by_label(rows, limit, seed=0)` that draws round-robin across labels (seeded shuffle
  within each label bucket, so a requeued run rebuilds the identical curriculum).
- **Verified live** at the real run sizes: train 3250 rows / **151 labels**, test 800 rows /
  **151 labels**, zero test labels missing from train, `oos` present in both, 21–22 rows per label
  in train, and byte-identical output across repeated calls.
- **Regression guard.** Three tests in `tests/test_benchmark_loaders.py`: full label coverage,
  determinism/bounds, and a source guard asserting `load_clinc150` never head-slices and does call
  `stratified_by_label`.
- **Status:** 🟢 fixed.

---

## B215 — `matplotlib` declared but never installed into `.venv_gpu`, so every run skipped graphics

- **Where:** `scripts/setup_gpu_env.sh`; surfaced at the tail of `tests/pipeline/run.py`.
- **Found:** 2026-08-04, in the `slm-clinc150-l40s-38084400` log:
  `graphics: skipped (ModuleNotFoundError: No module named 'matplotlib')`.
- **Root cause.** `matplotlib>=3.7.0` is declared in both `requirements.txt` and `pyproject.toml`,
  but `setup_gpu_env.sh` installs its dependency set explicitly package-by-package and never
  included it. The GPU venv is built from that script, not from the requirements file, so the
  declaration had no effect — environment drift between the declared and the actual venv.
- **Impact.** Cosmetic only: the graphics block is deliberately fail-safe (`except Exception` after
  the run has already finished), so it logged and moved on. But no cluster run has ever produced
  its end-of-run trajectory plots.
- **Fix.** Installed `matplotlib 3.11.1` into `.venv_gpu`, and added the `uv pip install
  "matplotlib>=3.7.0"` line to `setup_gpu_env.sh` so a rebuilt venv keeps it. Verified it imports
  under the headless `Agg` backend.
- **Status:** 🟢 fixed.

---

## B216 — ⚪ the 7-day walltime silently defers every job past a cluster maintenance reservation

- **Where:** operational, not a code defect. All `tests/pipeline/run_*_l40s.slurm`
  (`#SBATCH --time=7-00:00:00`) on Hyak's `gpu-l40s` partition.
- **Found:** 2026-08-04, submitting `slm-clinc150-l40s` (job 38102039). The job sat `PENDING`
  with `Reason=Resources` and `StartTime=2026-08-12T09:00:01` — **eight days out**, and exactly
  one second after a maintenance reservation ended.
- **What happens.** `scontrol show reservation August11_Maintenance` covers
  `g[3090-3137]` — *every* L40S node — from `2026-08-11T09:00` to `2026-08-12T09:00`
  (`Flags=MAINT,FLEX,OVERLAP,IGNORE_JOBS,SPEC_NODES,ALL_NODES,MAGNETIC`). A 7-day job cannot be
  placed unless it both acquires a node *and* completes before the reservation opens. Once the
  latest feasible start slips past `reservation_start − walltime`, the backfill scheduler stops
  trying to fit the job before the window and parks it at reservation end. Nothing warns you: the
  reason stays the generic `Resources`, and the only tell is a `StartTime` sitting suspiciously
  one second after a reservation boundary.
- **Workaround used.** `scontrol update job=<id> TimeLimit=5-00:00:00`. After two scheduling
  cycles the estimate moved from `2026-08-12T09:00` to *imminent* on `g3100`. Walltime can only be
  **reduced** by a non-admin, so the recovery is in-place; raising it again requires resubmission.
- **Why the scripts still request 7 days.** That is the documented weeklong contract
  (`docs/PIPELINE.md`, asserted by `test_task_slurm_scripts.py`), and the `--requeue` + USR1
  checkpoint machinery is designed to roll a long run across segments anyway. Shortening the
  walltime is the correct *situational* response near a maintenance window, not a new default.
- **Operator check before every submission:**
  ```bash
  scontrol show reservation | grep -A2 -i maint     # any window on your nodes?
  scontrol show job <id> | grep -E "StartTime|TimeLimit"
  ```
  A `StartTime` landing exactly at a reservation's `EndTime` means the walltime, not contention,
  is the blocker — reduce `TimeLimit` until the estimate moves.
- **Status:** ⚪ documented; no code change. Recurs at every cluster maintenance window.

---

## B235 — ⚪ `synthesize` split into fill and surgical sub-strategies

- **Where:** `agent/nodes/curate.py` (`_surgical_synthesize`, `_synthesize_positive_rows`).
- **Added:** 2026-08-05 on Evan's instruction.
- **Motivation.** The orchestrator names specific confusion pairs in its hypothesis every turn
  (`change_ai_name↔change_user_name`, `change_language→translate`) and that information was
  **thrown away** — `curate_node` never passed `pattern_hint` into synthesis, so the generator had
  no idea which failure it was meant to attack.
- **Design.** `SLM_SURGICAL_SYNTH_SHARE` (default 0.20) of the plan's `synth_rows` goes to the top
  `SLM_SURGICAL_MAX_PAIRS` (5) pairs. Per-pair budget is **proportional to confusion count**
  (`round(TOTAL × count / Σcounts)`, clamped 10–100) so effort follows the evidence rather than
  being split evenly. Anchors are drawn from the **gold** class — the one the model should have
  predicted — and produce in-class GOLD rows, never contrastive negatives (removed, B231).
- **Per-pair effectiveness.** `state["surgical_pair_history"]` records the confusion count at the
  time each pair was targeted. If a pair is targeted again and its count has NOT fallen, it is
  logged `EXHAUSTED` and skipped. Without this the run keeps spending on a pair that is not
  responding — precisely the loop `slm-clinc150-cse-38155022` was stuck in (B224).
- **Naming note.** The token `surgical` was previously forbidden by
  `test_data_rebuild_routing.py` because it named a ROUTE removed in the 2026-07-31 redesign. The
  guard was relaxed deliberately: this is a synthesize sub-strategy under the data_rebuild plan,
  not a resurrection of the old route.
- **Status:** ⚪ implemented; 5 tests in `tests/nodes/test_surgical_synthesis.py`.

---

## B234 — per-task dataset-size table deleted; sizing is now deterministic and per-tier

- **Where:** `config/config.py` (`DATASET_SIZE_BY_TYPE`, removed), new `agent/data_sizing.py`,
  `agent/nodes/curate.py`, `agent/nodes/cold_start/task_analysis.py`.
- **Changed:** 2026-08-05.
- **Problem.** `DATASET_SIZE_BY_TYPE` (classification 150, NER 200, generation 600, …) was dead
  code that *looked* live: every value sat far below `CURRICULUM_SIZE_FLOOR`, so it was clamped
  away on every path. It created the impression that per-task sizes were being honoured when the
  effective target was always the floor. The target was also computed ONCE and reused across
  escalations, so a 4B model trained on a target sized for a 0.6B one.
- **Fix.** Deleted the table. `agent/data_sizing.py` computes the target from two MEASURED signals:
  ```
  novelty     = 1 − zero_shot_baseline_f1        # empirical, not an LLM guess
  size_factor = clamp(1e9 / n_params, 0.5, 2.0)  # smaller model -> more data
  target      = clamp(5000 × (0.5 + novelty) × size_factor, 5000, 25000)
  ```
  Recomputed whenever the selected model changes, hooked in `curate_node` so initial selection,
  escalation and downward regression are all covered by one call site. `SLM_CURRICULUM_SIZE`
  overrides. Worked example: Qwen3-0.6B on CLINC150 (baseline 0.3152) → 9,873 rows; the same task
  on a 4B model → 5,000.
- **Deliberately NOT an LLM decision.** The orchestrator cannot estimate novelty from a task
  description; the zero-shot baseline measures the same quantity empirically and is already
  computed every run.
- **Status:** 🟢 implemented; 8 tests in `tests/test_data_sizing.py`.

---

## B236 — downward tier regression made unconditional and strategy-independent

- **Where:** `agent/nodes/iterate.py` (convergence routing), `agent/nodes/downward_probe.py`.
- **Changed:** 2026-08-05.
- **Two gates removed.**
  1. `probes_down = strategy in ("interpolation", "orchestrator_choice")` meant a
     `smallest_first` run could never regress — even after it had ESCALATED, so it could not come
     back down when the smaller model might now succeed on an improved dataset.
  2. `_should_reexplore_downward` spent an orchestrator API call to decide whether trying a
     smaller model was "worth it" — i.e. it could decline the one thing the run exists to
     determine. Deleted (61 lines).
- **Policy now.** On meeting the goal, ALWAYS probe the next tier down. The only stopping
  conditions are structural: no lower tier exists, or every lower tier has already been tried. If
  the lower tier fails to clear the goal, the last passing tier is kept (existing behaviour).
  Under `smallest_first` the run starts at tier 0, so it still never regresses — a consequence of
  where it starts, not a rule about the strategy.
- **Status:** 🟢 implemented; `tests/nodes/test_downward_probe.py` (28 tests) updated.

---

## B231 — ⚪ contrastive hard-negative synthesis removed; gold-only + teacher verification

- **Where:** `data/curriculum.py` (`synthesize_hard_negatives`, now deleted), `synthesize_examples`.
- **Decision date:** 2026-08-05, on Evan's instruction after reviewing the generated rows.
- **Why.** The generator was asked for text that "superficially resembles class A but genuinely
  belongs to class B". For intent classification the intent IS the surface meaning, so the
  instruction is close to self-contradictory and the model produced incoherent hybrids. Of the 17
  synthetic rows that survived into training in `slm-clinc150-cse-38155022`, roughly **10 were
  mislabelled or nonsense** — `"what ingredients do i need to book a flight from new york to
  london"`, `"my dough seems to have been blocked for no reason"`, `"set the oven alarm for 30
  minutes"` filed as `recipe`. Compounding it, every negative was filed under a different label
  than its anchor, so they skewed the class histogram straight into the label-balancing control
  (B221) which then deleted them.
- **Removed:** `synthesize_hard_negatives` (235 lines incl. the classification blend prompt, the
  NER rewrite prompt, math/code guards, 2-for-1 pairing and the Claude fallback),
  `_hard_negative_ratio`, `SLM_SYNTH_HARD_NEGATIVE_RATIO`, the `hard_negative_synthesis` cost
  stage, and `tests/test_curriculum_hardneg.py`. Live docs updated; dated records left intact.
- **Replacement.** `_synthesize_new_gold` only — a generated row inherits its label from a real
  anchor, so it cannot be mislabelled the same way — followed by `verify_generated_labels`, a
  teacher pass that answers `{"valid", "reason"}` per row. Rejections are logged with the
  teacher's reason. Mechanical verification failures KEEP the row so a broken verifier can never
  empty a dataset.
- **Rejected alternative (important).** Mining hard negatives from the model's own confusion
  matrix — i.e. taking held-out eval rows it got wrong and training on them — was proposed and
  **correctly rejected by Evan as test-set leakage**. It would inflate the eval score without
  improving the model and would defeat the four-layer eval firewall. Not implemented anywhere.
- **Status:** ⚪ deliberate design change.

---

## B232 — escalation collapsed to one mechanism, measured over an append-only eval history

- **Where:** `agent/nodes/iterate.py`, `agent/nodes/evaluate.py`, `agent/nodes/escalate.py`.
- **Found/changed:** 2026-08-05.
- **Problem.** Two knobs answered the same question. `MAX_STALL_EVALS` counted *consecutive* evals
  without a new best and **reset on every improvement**, so an improvement every 14 evals could
  defer escalation indefinitely. `STAGNATION_WINDOW` read `state["scores"]`, which `rollback` pops,
  so in a mostly-regressing run the window never filled and the test never fired at all.
- **Fix.** `MAX_STALL_EVALS` deleted. Stagnation is now the single mechanism and is measured over
  new append-only `state["eval_history"]` — written by `evaluate_node` for every eval including
  rolled-back ones, reset only on a tier change. Escalate when the best score in the last
  `STAGNATION_WINDOW` (15) evals fails to beat that window's first score by more than
  `STAGNATION_MIN_DELTA` (2%). `MAX_EVALS_BEFORE_ESCALATION` (30) remains as an unconditional
  ceiling.
- **Semantics (tested):** an improvement, 9 regressions, another improvement, 4 regressions = 15
  evals; if total gain across the window is under 2% it escalates. Improvements do not reset the
  window — only cumulative progress does.
- **Status:** 🟢 implemented.

---

## B227 — SUPERSEDED by B233; see below

## B227 — rollback overwrote the eval diagnosis, so the orchestrator never saw the failure it was reacting to

- **Where:** `agent/nodes/rollback.py` (restore block), consumed by `agent/nodes/iterate.py`
  prompt assembly.
- **Found:** 2026-08-05, tracing why `Hard-bucket accuracy is 0.683` appeared 163 times in
  `slm-clinc150-cse-38155022`.
- **Root cause.** `rollback_node` restored the best DAG node's `evaluation_state` wholesale,
  including `state["test_report"]`. Iterations 3-19 all regressed, so each of them reverted the
  diagnosis to iteration 2's. The orchestrator is asked "what should we change next?" immediately
  after a regression — and was shown the report of a run that had SUCCEEDED.
- **Evidence.** The prompt's per-difficulty line read
  `easy=0.963(n=214) medium=0.894(n=444) hard=0.683(n=142)` in **18 of 19 turns**, while the test
  agent recomputed fresh values every eval (`hard=` 0.493, 0.408, 0.585, 0.310, 0.134, 0.549,
  0.197...). Log line 3704 computed `hard=0.408`; the next prompt at line 4064 showed `hard=0.683`.
  **All three difficulty buckets were frozen, not just hard.** The per-bucket scoring itself is
  correct — only the copy handed to the orchestrator was stale.
- **Second-order effect — this also explains the intervention imbalance.** The frozen report
  carried `suggested_intervention: data_rebuild`, shown in **18 of 19** prompts, while the live
  test agent emitted "Tune hyperparameters" **21 times**. The orchestrator chose `data_rebuild`
  15 times not from bias but because it was repeatedly told to.
- **Fix.** Rollback now writes the restored copy to `restored_test_report` and leaves
  `test_report` describing the eval that just ran. `agent/state.py` documents both fields.
- **Status:** 🟢 fixed.

---

## B233 — rollback diagnosis: reverted B227, added an explicit failed-attempt memo instead

- **Where:** `agent/nodes/rollback.py`, `agent/nodes/evaluate.py`, `agent/nodes/iterate.py`,
  `agent/state.py`.
- **Changed:** 2026-08-05, superseding B227.
- **Why B227 was wrong.** B227 stopped rollback from restoring `test_report`, so the orchestrator
  would see the failed attempt's numbers. But after a rollback the LIVE WEIGHTS are the restored
  best checkpoint — so the per-difficulty scores and confusion pairs that describe the current
  model genuinely are the best node's. B227 would have had the orchestrator reason about a model
  that no longer existed.
- **What was actually missing** was not fresh numbers but any signal that the previous attempt had
  been discarded, and what it was. In `slm-clinc150-cse-38155022` 17 consecutive attempts were
  rolled back and the prompt never said so once.
- **Current design.** `test_report` is restored from the best node (original behaviour). Rollback
  additionally writes `state["last_failed_attempt"]` — intervention, sub-strategy, score, delta,
  the failed attempt's difficulty profile, and its hypothesis — which is rendered at the TOP of the
  report block under `## LAST ATTEMPT WAS ROLLED BACK — do not repeat it`, with an explicit note
  that the scores below describe the restored checkpoint rather than the failure. `evaluate_node`
  clears the memo on the next eval so it can never go stale.
- **Status:** 🟢 implemented (B227 reverted).

---

## B229 — no per-row label-space validation; junk labels could enter training from any source

- **Where:** `data/curriculum.py::apply_quality_controls`, wired from `agent/nodes/curate.py`.
- **Found:** 2026-08-05, follow-up to B222.
- **Gap.** B222 added a guard at the acquisition boundary, but it was per-SOURCE and only fired on
  *total* disjointness — a source with 90% valid labels and 10% junk would pass with the junk
  included, and synthetic rows bypassed it entirely.
- **Fix.** Quality control now takes `allowed_labels`, derived from the **frozen eval set** (the
  exact classes the model is scored against), and drops any classification row whose label is not
  in that set — regardless of provenance. Logged with the rejected labels and counts.
- **Status:** 🟢 fixed.

---

## B228 — ⚪ quality control removed ~1,500 rows silently, with no reason and no under-target warning

- **Where:** `data/curriculum.py::apply_quality_controls`, `agent/nodes/curate.py`.
- **Found:** 2026-08-05.
- **Finding.** QC is the largest consumer of rows in the pipeline — it deleted roughly 1,501 of
  1,549 synthesized rows in `slm-clinc150-cse-38155022` — and reported nothing at all. A dataset
  written at 3,461 against a 5,000 target looked inexplicable without reading artifacts.
- **Fix (observability only, no behaviour change).** Each control now reports its removals and the
  reason (`schema`, `label-space`, `label-balance`, `length-outlier`, `surface-dedup`), plus a
  total, plus an explicit warning when the finished dataset is under target explaining that
  synth-fill runs *before* QC and there is no refill afterwards.
- **Deliberately NOT changed:** no post-QC refill, and no per-rebuild dataset versioning — being
  under target is acceptable as long as it is stated, and versioning every intermediate costs too
  much storage. This also withdraws the versioning recommendation in B225.
- **Status:** 🟢 fixed (reporting); under-target behaviour intentionally retained.

---

## B226 — escalation policy drifted from its documented value; window could never fire

- **Where:** `agent/nodes/iterate.py` (`STAGNATION_WINDOW`, `MAX_STALL_EVALS`), plus
  `tests/nodes/test_iterate_stall.py` and `tests/config/test_curation_config.py`.
- **Found:** 2026-08-04, investigating why `slm-clinc150-cse-38155022` ran 17 non-improving evals
  on tier 0 without ever escalating.
- **Two separate defects.**
  1. **Drift.** `tests/nodes/test_iterate_stall.py` asserted `STAGNATION_WINDOW == 50` ("raised to
     50 in B161") while the code default was `20`. Five tests had been failing continuously as a
     result — the intended policy and the running policy had silently diverged.
  2. **The window is structurally unreachable in a regressing run.** `rollback.py:40` pops the
     regressing score, so `state["scores"]` only ever retains improvements. In this run it stayed
     at **2 entries** for all 20 iterations, so `_is_stagnant` (which requires
     `len(scores) >= STAGNATION_WINDOW`) could never evaluate, and the prompt kept reporting a
     healthy `Recent chronological gain (last 2 evals): 0.0472`. Only
     `consecutive_no_improvement` — which rollback does not touch — actually bit, and it reached
     17 against a limit of 20.
- **Fix (policy set deliberately, 2026-08-04).**
  - `STAGNATION_WINDOW = 15`, `STAGNATION_MIN_DELTA = 0.02` — escalate after 15 evals that gained
    less than 2%.
  - `MAX_STALL_EVALS = 15` — escalate after 15 consecutive non-improving evals.
  - **New `MAX_EVALS_BEFORE_ESCALATION = 30`** (`SLM_MAX_EVALS_BEFORE_ESCALATION`) — an
    unconditional ceiling that counts `state["iteration"]`, i.e. evals actually performed, so it
    **cannot be defeated by rollback popping scores**. Any model that burns 30 evals without
    reaching the goal escalates.
- **Tests:** the five drifted assertions now encode the real policy, and two new cases cover the
  ceiling firing at 30 with a deliberately tiny score history, and not firing at 29.
- **Status:** 🟢 fixed.

---

## B225 — the winning dataset contained ZERO synthetic rows, and intermediates are overwritten

- **Where:** `agent/nodes/curate.py` artifact write (`dataset_v{N}.jsonl`).
- **Found:** 2026-08-04 log audit of `slm-clinc150-cse-38155022`.
- **Finding.** The final artifact `dataset_v2.jsonl` — the dataset behind the converged 0.8971
  result — contains **3,249 `train_anchor` + 160 `mined_real` + 17 untagged, and no `synthetic`
  rows at all**, even though `data_sources.json` records 790 cumulative synthesized rows. Iteration
  20 was a resample-only rebuild, so it discarded the synthetic material entirely.
- **Compounding problem.** The version counter did not advance past 2: the same
  `dataset_v2.jsonl` path was rewritten at 17+ points in the run, so every intermediate dataset
  (including the mixed-provenance ones that scored 0.86+) is **unrecoverable**. Only the first and
  last states survive on disk.
- **Why it matters.** It undercuts attribution: the headline result cannot be credited to
  synthesis, and no earlier dataset can be re-examined or re-run.
- **Status:** 🔴 open. Datasets should be versioned per rebuild (or content-hashed) rather than
  overwriting a single filename.

---

## B224 — the orchestrator's context replays stale hypotheses, freezing its diagnosis

- **Where:** `agent/nodes/iterate.py` prompt assembly (trajectory + curation-log history).
- **Found:** 2026-08-04 log audit.
- **Finding.** The phrase `Hard-bucket accuracy is 0.683` appears **163 times** in the run log.
  The test agent *did* compute fresh numbers each eval (`hard=` values across the run include
  0.683, 0.493, 0.549, 0.641, 0.761), but the orchestrator's own prior hypotheses are replayed
  verbatim into every subsequent prompt. An early figure therefore keeps reappearing long after it
  is wrong, and by late iterations the context contains dozens of restatements of one stale number
  against a single fresh one.
- **Consequence.** The orchestrator kept diagnosing the same confusion pairs and chose
  `data_rebuild/synthesize` in 15 of 19 decisions, while the test agent's own suggestion was
  `hyperparameter`. Its context was dominated by its own echo rather than by new evidence.
- **Status:** 🔴 open. Candidate fixes: summarise rather than replay old hypotheses, cap replayed
  history to the last N, or strip stale metrics from replayed text and present current metrics once
  in a dedicated block.

---

## B221 — classification hard negatives all targeted ONE label, so quality control deleted them

- **Where:** `data/curriculum.py::synthesize_hard_negatives` (classification branch).
- **Found:** 2026-08-04, auditing `dataset_v1.jsonl` from the converged CLINC150 run
  `slm-clinc150-cse-38155022`.
- **Root cause.** The target class for each hard negative was chosen as
  `target_label = target_labels[0]` — *the same label for every example in the batch*, since
  `target_labels` is rebuilt in a stable order for each source label. On a 151-intent task the
  entire synthesis budget therefore landed on one or two classes.
- **Evidence.** Of the synth-fill rows that survived into the written dataset, **39 were labelled
  `recipe` and 9 `book_flight` — 2 labels out of 151**, against a real-row distribution of a flat
  21–22 rows per label.
- **Downstream effect (the expensive part).** `apply_quality_controls` enforces "no label more
  than 3x the smallest". A pile of ~1,500 rows on one label is exactly what that rule exists to
  delete, so synth-fill added **1,549 rows toward the 5,000 target and only 48 survived** — the
  dataset was written at 3,461. The synthesis compute, and the orchestrator's repeated
  `synthesize` interventions, were largely wasted.
- **Secondary defect found alongside.** `all_labels` was built from a `set`, whose iteration order
  varies per process, so the "always the first label" choice was also **not reproducible across a
  requeue** despite the pipeline's determinism contract.
- **Fix.** `all_labels` is now `sorted(...)`, and each example draws its target from a seeded
  `random.Random(20260804)` across the full label space, so hard negatives spread over all
  classes and reproduce across a requeue.
- **Not fixed (documented):** nothing verifies that a generated hard negative actually *belongs*
  to its assigned target class. One surviving row reads
  `label=recipe, text="what is the best way to make a reservation for a table at red robin"` —
  that is a reservation query stored as a `recipe` training label, i.e. injected label noise. A
  verifier (or a round-trip check with the reference model) is the real fix.
- **Status:** 🟢 targeting fixed; ⚪ label-correctness verification still open.

---

## B222 — mined DeepPavlov/clinc150 rows carried unmapped integer labels into the curriculum

- **Where:** `data/loaders/web_acquire.py` acquisition path (`mine_additional_real_rows` →
  schema mapping), surfaced in `agent/nodes/curate.py` composition.
- **Found:** 2026-08-04, same dataset audit.
- **Symptom.** The 164 rows mined from `hf:DeepPavlov/clinc150/train` were written with labels
  **`"0"` (60), `"1"` (60), `"2"` (44)** instead of intent names — visible in the curation log's
  label histogram as three bogus classes sitting alongside the 151 real intents.
- **Root cause.** The mined source exposes its intent column as integer class ids. The curated
  `data/loaders/clinc150.py` resolves those through the HF `ClassLabel` feature names
  (`_resolve_intent_names`), but the generic acquisition path has no equivalent step, so the raw
  ids were stringified and used verbatim.
- **Impact.** 164 of 3,461 training rows (~5%) carry labels that exist in no real label space,
  and they inflate the apparent class count. They cannot match any eval label, so they are pure
  noise in the curriculum.
- **Why the existing resolution missed it.** `web_acquire._convert` *does* try
  `ds.features[lcol].names` to turn integer ids into names — but `DeepPavlov/clinc150` types its
  label column as `Value('int64')`, **not** a `ClassLabel`, so there are no `.names` to read and
  the ids are unrecoverable from the dataset itself.
- **Fix (2026-08-04).** `mine_additional_real_rows` now derives the run's established label space
  from `existing_rows` and rejects any mined classification source whose labels are **entirely
  disjoint** from it, logging the reason and the offending sample labels. Overlapping sources are
  unaffected. Two regression tests cover reject-and-accept.
- **Status:** 🟢 fixed (rejected rather than remapped — remapping needs an external id→name table
  the source does not provide).

---

## B220 — curate passed `plan_identity=""`, so the first *paid* acquisition round killed the run

- **Where:** `agent/nodes/curate.py::curate_node` (the `strategy == "acquire"` branch) →
  `data/loaders/web_acquire.py::mine_additional_real_rows` →
  `data/acquisition_budget.py::reserve_paid_acquisition`.
- **Found:** 2026-08-04, run `slm-clinc150-cse-38154619`, 594 s in:
  ```
  ValueError: plan_identity must be non-empty
  ```
- **Root cause — a contract split by a refactor.** The 2026-07-31 data-curation redesign removed
  plan-identity *dedup* from `agent/data_rebuild.py` ("no plan-identity dedup, no untried-plan
  rotation"), and curate was left passing a literal `plan_identity=""` with the comment
  "(no plan seed anymore)". But the **durable paid-acquisition ledger was never part of that
  redesign**: it still meters spend in two tiers — `MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN = 3` inside
  `MAX_PAID_ACQUIRE_ROUNDS_PER_RUN = 9` — and therefore still requires a per-plan key, rejecting
  an empty one outright. One side dropped plan identities; the other still depends on them.
- **Why it stayed hidden this long.** The empty string is inert until mining actually reaches a
  *paid* round. `mine_additional_real_rows` first tries local bundles, then curated benchmarks, and
  only falls through to paid discovery when **both** are rejected. In this run every local
  candidate was rejected in turn (`apps`, `bc5cdr`, `emotion`, `go_emotions`, `gsm8k`, `mbpp`,
  `samsum` — "no explicit benchmark or strong task+label+schema match"), so it reached the paid
  path and died. It is also **strategy-dependent**: curate picks `resample`/`acquire`/`synthesize`
  non-deterministically from an entropy seed, so only `acquire` runs can hit it. The prior run
  (38151989) never took this branch, which is why it got all the way to the GGUF merge instead.
- **Fix.** New `agent/data_rebuild.plan_budget_identity(plan)` returns a content-addressed
  `plan-<sha256[:16]>` over the canonical `_PLAN_FIELDS`; curate passes it. This restores the
  ledger's per-plan bucket **without** reintroducing dedup: re-picking an identical plan keeps
  drawing down the same allowance (which is the point of a per-plan cap), while a materially
  different plan gets a fresh one. Plans carry no seed, so equal plans always hash equally.
  `hashlib` was already imported in `data_rebuild.py` and unused — a leftover from the removed
  dedup.
- **Regression guard.** `tests/nodes/test_data_rebuild_plan.py` covers stability, key-order
  independence, content-addressing, and acceptance by the real `reserve_paid_acquisition` guard;
  new `tests/nodes/test_curate_acquire_budget.py` fails if curate ever passes an empty identity
  again.
- **Status:** 🟢 fixed.

---

## B219 — LoRA merge silently wrote nothing: QLoRA's 4-bit base can't do a `merged_16bit` merge

- **Where:** `training/lora_trainer.py::merge_for_quantization`, called from
  `agent/nodes/evaluate.py::_build_or_reuse_gguf`.
- **Found:** 2026-08-04, run `slm-clinc150-l40s-38151989` — the first run to get past `eval_setup`.
  It trained Qwen3-0.6B, measured a real zero-shot baseline (F1 0.3152), calibrated the Qwen goal
  to 0.8918, then died after 1694 s building the Q4_K_M GGUF for the fine-tuned adapter:
  ```
  QuantizationInfrastructureError: Failed to build required Q4_K_M GGUF for Qwen/Qwen3-0.6B:
  snapshot is incomplete; no nonempty model weight file or weight index
  ```
- **Root cause.** The error message points at a missing file, but the merge directory was not
  incomplete — it was **completely empty**. The real cause is one buried warning:
  ```
  unsloth_zoo/saving_utils.py:2922: UserWarning: Base model should be a 16bits or mxfp4 base
  model for a 16bit model merge. Use `save_method=forced_merged_4bit` instead
  ```
  The chain:
  1. LoRA training runs as **QLoRA** (`load_in_4bit=config.lora_rank is not None`).
  2. Unsloth transparently swaps in its own pre-quantized mirror and records *that* as the
     adapter's base — `Qwen/Qwen3-0.6B` becomes **`unsloth/qwen3-0.6b-unsloth-bnb-4bit`**. This
     string appears nowhere in the repo; it is injected by Unsloth.
  3. `merge_for_quantization` faithfully resolved the adapter's recorded base — the 4-bit mirror —
     and asked for `save_method="merged_16bit"`.
  4. `unsloth_zoo` refuses that combination by **emitting a UserWarning and returning without
     writing a single byte**. No exception. The empty directory then failed
     `verify_hf_model_snapshot` with a message describing the symptom, not the cause.
- **Why the guard didn't help.** `verify_hf_model_snapshot` is a *snapshot integrity* check, so it
  reported "no nonempty model weight file" — indistinguishable from a truncated download. Nothing
  distinguished "the writer no-opped" from "a file is missing", which is what made this expensive
  to trace.
- **Fix (two parts).**
  1. `merge_for_quantization` takes an optional `base_model_id` that **pins the merge to the
     canonical 16-bit base**, overriding whatever mirror Unsloth recorded; `evaluate.py` passes the
     `model_id` it already has. Merging QLoRA deltas into the 16-bit original is the standard QLoRA
     merge, and it is what the pipeline wants before an honest Q4_K_M quantization — merging into a
     4-bit base would double-quantize.
  2. An explicit empty-directory check after `save_pretrained_merged` raises an actionable
     `RuntimeError` naming the no-op, so this can never again present as a missing-file error.
- **Verified:** `Qwen/Qwen3-0.6B` resolves and passes `verify_hf_model_snapshot` from the shared
  cache (`model.safetensors` present), so the pinned base is available offline on the compute node.
- **Regression guard.** Three tests in `tests/training/test_lora_trainer.py`: the explicit-base pin
  resolves the canonical id (not the adapter's 4-bit mirror), the default path still honours the
  adapter record, and a no-op `save_pretrained_merged` raises the actionable error. The two
  pre-existing merge tests now write a weight file from their mocks, since a mock that writes
  nothing is exactly the failure being guarded against.
- **Status:** 🟢 fixed.

---

## B218 — ⚪ six "idle" Hyak GPU nodes are unschedulable: absent from `topology.conf`

- **Where:** cluster-side (Hyak `klone`), not this repo. Affects `gpu-l40s`, `ckpt-g2`, `ckpt-all`.
- **Found:** 2026-08-04, while `slm-clinc150` sat `PENDING` on a saturated quota although `sinfo`
  showed three fully idle L40S nodes (24 free GPUs).
- **Symptom.** `sinfo` reports the nodes `idle`, `avail=up`, `Reason=none`, `AllocTRES=` empty, with
  hardware and `Partitions=` byte-identical to nodes that work. Yet every allocation is refused —
  even a bare 1-CPU, 4 GB, 5-minute job. `srun --test-only` reports the misleading
  `Requested node configuration is not available`; only a real `sbatch` prints the true cause:
  ```
  sbatch: error: Batch job submission failed: Requested topology configuration is not available
  ```
- **Root cause.** The cluster runs `TopologyPlugin=topology/tree`. Expanding every `Nodes=` range in
  `scontrol show topology` yields 553 nodes — and **g3132–g3137 are not among them**. Slurm will not
  place a job on a node missing from the topology tree. The split is exact:

  | Nodes | In topology tree | Schedulable |
  |---|---|---|
  | g3100–g3131 | yes | yes |
  | **g3132–g3137** | **no** | **no** |

  Every node that looks idle-but-unusable is precisely the set missing from the tree. Control test:
  the same pinned submission to `g3113` (in the tree) queues normally.
- **Stranded capacity:** g3134–g3137 = **32 idle L40S**, plus g3132 = **8 idle H200**.
- **Second-order effect.** The backfill estimator does *not* apply the topology filter, so
  `StartTime`/`SchedNodeList` estimates happily point at `g3134`. Those estimates are unreachable
  and churn every scheduling cycle — do not trust a projected start on g3132–g3137.
- **Not user-fixable.** New hardware was racked and registered but `topology.conf` was never
  updated. Requires a Hyak admin to add the nodes and reconfigure. Report to Hyak support.
- **Detection one-liner:**
  ```bash
  scontrol show topology | grep -oP 'Nodes=\K\S+' | while read r; do scontrol show hostnames "$r"; done | sort -u > /tmp/topo.txt
  comm -23 <(sinfo -h -N -o "%N" | sort -u) /tmp/topo.txt   # registered but not in topology
  ```
- **Status:** ⚪ external/cluster-side. Documented so the team stops chasing phantom idle GPUs.

---

## B217 — the curated `SLM_BENCHMARK_TASK` path skipped Stage-0 decontamination, so a source-data duplicate was fatal

- **Where:** `agent/nodes/cold_start/eval_setup.py::_load_named_benchmark`.
- **Found:** 2026-08-04, run `slm-clinc150-l40s-38102039` — the resubmission after B213/B214.
  It cleared the loader (B213/B214 confirmed fixed, graphics rendered per B215) and then died at
  the same node, 288.7 s in, with a *different* error:
  ```
  ValueError: eval_setup normalized train/test overlap: 1 training row(s) match held-out eval text
  ```
- **Root cause — two things combined.**
  1. **The source data is genuinely dirty.** CLINC150 ships the utterance
     `"what's your designation"` in **both** official splits, under **two different intents**:
     `what_is_your_name` in train, `user_name` in test. That is an annotation inconsistency in
     the upstream dataset, not something the pipeline created.
  2. **The curated path had no Stage-0 filter.** There are three ways data enters `eval_setup`,
     and only two decontaminate:

     | Path | Stage-0 `remove_normalized_train_overlap`? | Result on a dirty split |
     |---|---|---|
     | autonomous `acquire_dataset` | ✅ yes (`web_acquire.py`) | drops train row, logs, continues |
     | shared bundle | ✅ yes (sealed at build time) | rejected at bundle build |
     | **curated `SLM_BENCHMARK_TASK`** | ❌ **no** | **fatal raise** |

     So the curated path fell straight through to the Layer-1 firewall — whose own comment says it
     exists to catch "mocked or future loaders that bypass bundle/Stage-0 checks". It was doing its
     job; the bypass was the bug.
- **Why it only appeared now.** B214 masked it. The old head-slice loaded 33 of 151 intents and
  happened not to include `what_is_your_name`. Reading the full split surfaced a latent defect that
  would have hit **any** of the six curated benchmarks with a duplicated row.
- **Fix.** `_load_named_benchmark` now applies the same Stage-0 `remove_normalized_train_overlap`
  as the autonomous path, logs `Stage-0 normalized overlap removal for <key>: removed N train
  row(s); official test rows unchanged`, and records `overlap_removed_from_train` in
  `acquire_meta`. **Held-out test rows are authoritative and never modified — only the train row is
  dropped.** The Layer-1 firewall raise is deliberately left intact as the backstop.
- **Verified live:** train 3250 → 3249 (1 removed), test 800 unchanged, remaining overlap **0**,
  and label coverage preserved at **151/151** in both splits with no test label missing from train.
- **Regression guard.** Two tests in `tests/nodes/test_eval_setup_dataset_meta.py`: the removal /
  meta / log path, and a clean-split case asserting `overlap_removed_from_train == 0`. The existing
  `test_eval_setup_enforces_normalized_train_test_separation` still passes, proving the backstop
  raise survives for loaders that bypass Stage-0.
- **Status:** 🟢 fixed.
## B237 — orchestrator's trajectory reported the wrong intervention for past iterations
- **Symptom.** The compacted training trajectory sent to the orchestrator labelled iterations 4–8
  of `slm-clinc150-cse-38155022` as `intervention=hyperparameter`. The DAG (`hypotheses.md`),
  which is authoritative, records all five as `data_rebuild / synthesize`.
- **Impact.** The orchestrator's memory of *what it had already tried* was factually wrong, so the
  standing instruction to avoid repeating unsuccessful interventions could not be satisfied — it
  believed it had been tuning hyperparameters while it had in fact run `synthesize` five times.
  This compounds B224 (frozen diagnosis replayed as stale prose) and helps explain the 15×
  `synthesize` imbalance, since the history understated how often that sub-strategy had been used.
- **Root cause.** `evaluate_node` computed `dag_intervention = state["last_intervention"] or
  policy["intervention"]` and used it correctly for the DAG, but then passed
  `next_intervention=policy["intervention"]` — the *score-band policy's suggestion*, not the
  executed intervention — to `CurationLog.append`. `context_manager._extract_iteration_summary`
  surfaces that field as `intervention=` in the compacted trajectory, so the guess reached the
  orchestrator as history.
- **Fix.** Pass `dag_intervention` to the curation log so the DAG and the trajectory agree on a
  single source of truth. Comment added at the call site explaining that this field is read back
  as the orchestrator's own history.
- **Status:** 🟢 fixed. 27 related tests pass.

## B238 — the orchestrator's hypothesis was truncated in five places, worst at the source
- **Symptom.** Every long hypothesis in `slm-clinc150-cse-38155022` landed at exactly **240
  characters**, ending mid-word: `...cancel→freeze_a`, `...change_langua`,
  `...account_blocked→EX`. Fourteen of the twenty iterations were cut this way.
- **Impact.** The hypothesis is the orchestrator's causal reasoning and the densest signal in the
  run. The cut consistently landed *inside the confusion-pair list* — the actionable part — so
  what survived was the generic preamble (`Hard-bucket accuracy is 0.683 (n=142)...`) and what was
  lost was the specific evidence. This is a direct contributor to the stale-`0.683` repetition
  (B224): the reusable content was destroyed and only the boilerplate was carried forward.
- **Root cause.** Five independent cuts, applied in sequence:
  1. `agent/nodes/iterate.py::_validate_decision_json` — `hypothesis.strip()[:240]`. **The source
     cut**: everything downstream (console log, `data-curation.md`, `dag.json`, the next prompt)
     inherited it. Silent — no warning was ever emitted.
  2. `agent/data_rebuild.py` — `_plain_text(hypothesis, maximum=240)`.
  3. `agent/data_rebuild.py` — combined `pattern_hint`/hypothesis clause `[:240]`.
  4. `agent/context_manager.py::_extract_iteration_summary` — a further `[:100]` on the
     already-severed text, so replayed history was a 100-char fragment.
  5. `agent/nodes/iterate.py` rollback memo — `[:200]` on the failed attempt's hypothesis.
- **Second defect, same function.** `_extract_iteration_summary` read fields with
  `re.search(rf'{label}:\s*(.+)')`. Because `\s*` matches newlines, an **empty** field captured
  the next non-empty line of the document. Iteration 1 has no orchestrator hypothesis, so its
  summary rendered as `- Iter 1: ... — ### Hardware profile (Phase 1: theoretical)` — a markdown
  heading presented to the model as its own past reasoning. Affected every field, not just
  hypothesis.
- **Fix.** Single named bound `HYPOTHESIS_MAX_CHARS = 2000` (`SLM_HYPOTHESIS_MAX_CHARS`), shared
  by iterate and data_rebuild; `pattern_hint` keeps a separate 1200-char bound because it steers
  synthesis prompts. Cuts 4 and 5 removed outright. Truncation now **logs a warning** instead of
  happening silently. Field reads are anchored with `re.MULTILINE` so an empty field stays empty.
- **Tests.** 10 in `tests/test_hypothesis_integrity.py`, including a regression test that an empty
  hypothesis never captures the following heading.
- **Status:** 🟢 fixed.

## B239 — orchestrator prompt replaced the raw curation-log dump with structured run memory
- **Symptom (not a crash — an information defect).** The trajectory pasted into every iterate call
  was a dump of `data-curation.md`. A new best (iteration 2) and a rolled-back attempt
  (iteration 6) rendered **identically**: no outcome marker, no delta, no grouping. Seventeen
  consecutive failures appeared as seventeen unrelated rows, so the actual pattern —
  `data_rebuild/synthesize` tried fifteen times and never once kept — was present but invisible.
- **Fix.** New `agent/run_memory.py`, built from `state["dag"]` (append-only; rollback marks nodes
  `pruned` rather than deleting them, so discarded attempts remain visible even though
  `state["scores"]` pops them). Four sections: **MOST RECENT ITERATION** in full detail,
  **WHAT WORKED** (kept improvements with deltas and full reasoning), **FAILED SINCE THE LAST
  IMPROVEMENT** (aggregated by intervention/sub-strategy, with an explicit "prefer a DIFFERENT
  intervention type" conclusion when one dominates), and **SURGICAL SPEND** per confusion pair.
  Bounding is by how many attempts are narrated in full, never by cutting a hypothesis.
  Falls back to the old dump when the DAG is empty (first iteration).
- **Sub-defect found while validating against the real DAG.** `pi.D.plan` persists across
  iterations, so a `hyperparameter` node still carries the plan from the last rebuild and rendered
  as `hyperparameter/acquire` — crediting a data strategy to an iteration that never touched the
  data. Same false-memory class as B237. Sub-strategy is now only shown for `data_rebuild`.
- **Tests.** 22 in `tests/test_run_memory.py`. Verified by rendering the real
  `slm-clinc150-cse-38155022` DAG at iteration 19.
- **Status:** 🟢 fixed.

## B240 — orchestrator decisions discarded because the response hit the output ceiling
- **Symptom.** The first `iterate` call of `slm-clinc150-cse-38179864` returned **exactly 1536**
  output tokens (`tokens=5360->1536`) — `_ITERATE_MAX_TOKENS` — so the JSON was cut mid-string.
  The reask returned exactly 1536 again and failed identically.
- **Impact.** The orchestrator had chosen `data_rebuild`; both attempts were discarded and the
  score-band fallback executed `hyperparameter` instead. Its actual decision never ran.
- **Root cause (regression from B238).** Removing the silent 240-char hypothesis truncation
  removed the only pressure keeping responses short, while the output ceiling stayed at 1536 and
  the prompt had never stated any length expectation. The previous run peaked at 1,154 output
  tokens and never hit the ceiling. The run-memory block was NOT a contributor — it reduced input
  from 7,836 to 5,360 tokens at the same point.
- **Second defect: the reask could not fix it.** A `max_tokens` cutoff surfaced as "no parseable
  JSON", so the reask said "return valid JSON" — a format instruction for a length problem. The
  model rewrote another over-long answer and hit the same wall.
- **Fix.** `_ITERATE_MAX_TOKENS` 1536 → 4096 (`SLM_ITERATE_MAX_TOKENS`). The prompt now states a
  ~150-word hypothesis target AND the hard output ceiling, substituted from the constants so they
  cannot drift. `_hit_output_cap()` detects `stop_reason=max_tokens` before parsing and raises a
  length-specific error; the reask branches on it and asks for a shorter answer with a concrete
  word budget. `HYPOTHESIS_MAX_CHARS` raised to 4000 and demoted to a pure runaway guard —
  length is managed by prompting, not truncation.
- **Tests.** 12 in `tests/test_output_budget.py`.
- **Status:** 🟢 fixed.

## B241 — deterministic curriculum sizing silently collapsed to the floor
- **Symptom.** `[sizing] curriculum target for Qwen/Qwen3-0.6B [Q4_K_M]: 5000 — no baseline yet,
  assuming neutral novelty 0.50; unknown params -> size factor 1.00`. Both inputs to a two-input
  formula were missing, so the result was always the floor.
- **Impact.** Reproduced the exact defect `DATASET_SIZE_BY_TYPE` was deleted for (B234): sizing
  that appears task-adaptive but always clamps to the floor.
- **Root cause 1.** `resize_curriculum_for_tier` read `getattr(model, "params")` /
  `getattr(model, "n_params")`. `ModelSpec` has neither; it exposes `est_params_b` (parameters in
  BILLIONS, derived from weight size ÷ the quant's bytes-per-param). Result was always `None`.
- **Root cause 2.** The zero-shot baseline is measured by the first `evaluate`, which runs AFTER
  the first `curate`. Sizing was keyed on model change only, so tier 0 used the neutral novelty
  0.5 forever and never revisited it once the real measurement existed.
- **Trap in the fix.** `est_params_b` is a METHOD, not a property (unlike `selector`), so a plain
  `getattr` returns a truthy bound method and `float()` on it raises — silently degrading to the
  same neutral factor. Caught only by printing the resolved value.
- **Fix.** `_params_for_model()` calls `est_params_b` when callable and converts billions →
  absolute. The curate hook is keyed on `f"{selector}|baseline={baseline_is_known(state)}"`, so
  the target is recomputed once the baseline exists. Verified against the real ModelSpec:
  0.84B params → factor 1.19; target **5,952** before the baseline, **7,052** after.
- **Status:** 🟢 fixed.

## B242 — extraction-failure sentinel consumed surgical synthesis budget
- **Symptom.** `__EXTRACTION_FAILED__` appears as a `predicted` value in the confusion pairs, so
  pairs like `alarm -> __EXTRACTION_FAILED__ (5)` became top surgical targets (observed in
  `slm-clinc150-cse-38155022`).
- **Impact.** The sentinel is not a class — it means the model emitted a string outside the label
  vocabulary entirely. Such a pair names no decision boundary to sharpen, so the budget spent on
  it teaches nothing about the confusion, and it distorts per-pair effectiveness tracking.
- **Fix.** `_surgical_synthesize` drops pairs whose `predicted` is the sentinel and logs the
  count. They remain in the report for diagnosis — this only stops them consuming targeted budget.
  The remedy for out-of-vocabulary output is format adherence, which ordinary in-class training
  already provides.
- **Status:** 🟢 fixed.

## B244 — defense-in-depth re-validation rejected the validator's own output (real cause of B223)
- **Symptom.** `LLM call failed (ValueError('hyperparameter field(s) no longer tunable:
  batch_size ..., effective_batch_size ..., gradient_accumulation_steps ..., lora_alpha ...,
  lora_dropout ..., micro_batch_size ...')); using test-agent suggestion: hyperparameter`, with
  **no preceding "Decision failed validation" line and no reask**.
- **The error message was false.** The orchestrator never proposed those six fields. They are
  exactly the keys `training.hparams.normalize_hyperparams` DERIVES from the tunable five.
- **Root cause.** `_validate_decision_json` stored the NORMALIZED hyperparams back into
  `validated["hyperparams"]`. Normalization returns the trainer's full shape, so the decision
  came back carrying all six derived keys. `iterate_node` then re-validates that decision as
  defense in depth (`allow_internal=True`), and the retired-key check rejected them. Every
  `hyperparameter` decision died this way, deterministically.
- **Why no reask ever ran.** The failure happens in `iterate_node`'s outer re-validation, which
  is *outside* the try/except in `_llm_iterate` that triggers self-correction. This is the actual
  explanation for the B223 observation of "6 validation failures and ZERO `iterate_json_reask`
  events" — widening that clause from `ValueError` to `Exception` could not help, because the
  throw was never inside the block.
- **Impact.** Whenever the orchestrator chose `hyperparameter`, its decision was silently
  replaced by the test-agent/score-band fallback. Present in `slm-clinc150-cse-38155022` and
  `slm-clinc150-cse-38180646`; the latter's convergence at 0.8952 therefore came from an
  orchestrator that was only ever permitted to run data rebuilds.
- **Fix (at the source).** The decision now carries ONLY the five knobs the orchestrator actually
  chooses — `lora_rank`, `alpha_ratio`, `weight_decay`, `learning_rate`, `nr_epochs` — via
  `_decided_hyperparams()`. Nothing downstream needs more: `train._build_config` calls
  `normalize_hyperparams` itself at the point of use, so the derived shape is rebuilt where it is
  consumed. The retired-key rejection stays unconditional, because the decision can no longer
  contain those keys. No special case, no `allow_internal` exemption.
- **Second defect fixed by the same change.** `normalize_hyperparams` drops `alpha_ratio` from its
  output, so the decision log had been printing `alpha_ratio=None`. The decision now records the
  SNAPPED ratio reconstructed from alpha/rank, i.e. what was actually applied.
- **A wrong fix that was tried and rejected.** Filtering the derived keys out after normalizing
  looks equivalent and is not: dropping `lora_alpha` while `alpha_ratio` is absent makes the next
  normalization re-derive alpha at the DEFAULT ratio of 2. Measured drift in 4 of 6 rank/ratio
  combinations (rank=32 ratio=4 gives alpha 128, becomes 64).
- **Tests.** 13 in `tests/test_revalidation_roundtrip.py`, including a parametrized guard that the
  ratio round-trips through re-validation and the trainer re-derives the exact alpha for every
  rank/ratio pair. `tests/test_expanded_lora_search.py` updated to assert the decision-facing
  shape.
- **Status:** 🟢 fixed.

## B245 — run graphics used a 0-based, fractionally-ticked iteration axis and a redundant total
- **Symptom.** All three charts labelled the x-axis "iteration" but plotted a 0-based index with
  matplotlib's default float locator, producing ticks at `-0.5, 0.0, 0.5, 1.0, 1.5 ...` — a
  negative iteration and half iterations, none of which exist. The dataset-composition chart also
  drew a dashed "total" series along the top of the stacked bars, restating the stack height.
- **Fix.** `global_idx` is now 1-based, matching the numbering used in the logs, the DAG and the
  curation log. A shared `_iteration_axis()` helper applies `MaxNLocator(integer=True)` and pins
  the limits to the data range on all three charts. The total series is removed; headroom is
  reserved above the bars so the legend no longer overlaps the first stack.
- **Status:** 🟢 fixed.

## B246 — the fixed system prompt was re-logged verbatim on every orchestrator turn
- **Symptom.** `SLM_LOG_FULL_ITERATE_PROMPT` defaulted to `1`, so each iterate call dumped the
  entire (constant) system prompt plus the user content. In `slm-clinc150-cse-38155022` that is
  most of a 23,900-line log, and it buries the per-turn content that actually differs.
- **Fix.** Default flipped to `0`. Iteration 1 still logs the complete prompt so the run log stays
  self-contained and replayable; later turns log only the changing user content. Set
  `SLM_LOG_FULL_ITERATE_PROMPT=1` to restore per-turn verbatim logging.
- **Status:** 🟢 fixed.

## B247 — the orchestrator could override the deterministic curriculum size
- **Symptom.** In `slm-clinc150-cse-38180646` the curriculum dropped from 5,758 rows to 2,998
  between iterations 2 and 3. Not QC (2 rows removed total) and not a synth-fill failure (fill
  reached 2,998 of 3,000): the orchestrator's rebuild plan asked for `target_rows: 3000` while
  `agent.data_sizing` had computed **7,053** for that tier. Its hypothesis said "synth_rows and
  target_rows are scaled down from the prior aggressive round".
- **Root cause.** Two authorities for one quantity. Curriculum size is supposed to be a
  deterministic per-tier computation from the measured zero-shot baseline and the model's
  parameter count (B234/B241), but `target_rows` remained a free field on the data_rebuild plan,
  and the plan value won. The deterministic number therefore only ever applied to the first
  curate, before any plan existed.
- **Fix.** `target_rows` is removed from `_PLAN_FIELDS` and is no longer read from the payload;
  `normalize_data_rebuild_plan` stores the caller's deterministic value verbatim. The prompt no
  longer lists it in the schema examples and states plainly that it is not the orchestrator's to
  set. A stray `target_rows` in a plan is IGNORED rather than rejected — the model may emit it
  from habit, and a hard error would cost a reask round-trip (or the decision) over a value that
  is discarded anyway. Every other unknown plan key is still an error.
- **Tests.** `test_orchestrator_cannot_override_the_deterministic_target` and
  `test_target_rows_is_not_an_orchestrator_field` in `tests/nodes/test_data_rebuild_plan.py`;
  the two tests asserting the old clamping behaviour were rewritten.
- **Status:** 🟢 fixed.

## B248 — synth-fill rows were reported as "unattributed" and were invisible in the chart
- **Symptom.** `composition=[{'strategy': 'resample', 'rows': 3248}, {'strategy': 'synthesize',
  'rows': 447}, {'strategy': 'unattributed', 'rows': 2063}]` — over a third of the iteration-1
  curriculum filed under a bucket that reads like a pipeline defect.
- **Root cause.** `_synth_fill_to_target` tags its rows `_provenance="synthetic_fill"` but never
  sets `_strategy_origin`, which is the field the composition report groups by. Untagged rows
  fall through to the `"unattributed"` default.
- **Chart consequence.** `dataset_composition.png` plots gold / generated / mined, which are
  derived from the same untagged counts, so those 2,063 rows appeared nowhere. The chart also
  drew a "total (= stack height)" line whose label was FALSE precisely because of this gap — the
  stack summed to 3,695 against a total of 5,758.
- **Fix.** The chart now renders the remainder (`total - gold - generated - mined`) as an
  explicit "synth-fill (unattributed)" band, so the stack height really is the dataset size —
  which is what makes a separate total series redundant rather than merely duplicative.
- **Status:** 🟡 chart fixed; tagging `_strategy_origin` at the synth-fill call site is still
  worth doing so the composition report names the bucket honestly.

## B249 — SAMSum dataset withdrawn from the Hub; DialogSum run died in cold start
- **Symptom.** `slm-dialogsum-samsum-cse-38186256` FAILED after 6m48s, in `eval_setup`, before a
  model was ever selected: `DatasetNotFoundError: Dataset 'Samsung/samsum' doesn't exist on the
  Hub or cannot be accessed.` DialogSum loaded fine; only the SAMSum half failed.
- **Root cause.** External, not a code defect: `Samsung/samsum` and the bare `samsum` alias were
  both withdrawn from the HuggingFace Hub. Probed and confirmed — both raise
  DatasetNotFoundError; `knkarthick/samsum` resolves.
- **Fix.** `SAMSUM_ID = "knkarthick/samsum"`, the same owner as the DialogSum mirror already in
  use. It exposes the identical `dialogue`/`summary` columns and the same split sizes (14,731
  train / 819 test), so `convert_samsum_rows` is unchanged. Verified end to end: the loader
  returns well-formed `{text, answer, label}` rows from both halves.
- **Worth noting.** The curated-benchmark loaders pull live from the Hub at run time, so any of
  the six can break this way without a code change. This one cost only 7 minutes because it fails
  in cold start, but the failure mode is worth remembering when a benchmark suddenly stops.
- **Status:** 🟢 fixed.

## B250 — generation eval prompt described the wrong task, and training used a different one
- **Symptom.** On DialogSum the model continued the conversation instead of summarizing it.
  Against gold `"Shelly is volunteering at a food shelter and asks if others do..."` it produced
  `"Shelly: How about you? Any volunteer work? Tracy: Nah. Not into that."` — a next dialogue
  turn. Same for the other sampled rows.
- **Root cause 1 — the prompt never states the task.** `eval/scorers/generation.py` wrapped every
  non-code generation row in the hardcoded constant `"Answer the following question:\n\n{text}"`.
  A dialogue transcript asks no question and the word "summarize" appears nowhere, so continuing
  the chat is the only reasonable reading. The constant is task-AGNOSTIC: one string for the
  whole generation family (summarization, math, multilingual, structured extraction), never
  derived from the task, the benchmark or the loader. It reads as though written for
  question-answering and silently became the prompt for everything else.
- **Root cause 2 — train/serve skew.** `training/lora_trainer.py::_training_turn` built the
  generation input INDEPENDENTLY and passed the **bare text with no instruction at all**. So the
  model would be fine-tuned on one input distribution and scored on another. Classification has
  no such gap: both sides call the same `build_classify_prompt`.
- **Knock-on.** The judge-calibrated threshold is set from a baseline measured with a prompt that
  actively misleads the model, so the target itself was wrong.
- **Fix.** One shared builder, `build_generation_prompt(text, instruction)`, called by BOTH the
  eval harness and the trainer — importing it rather than reproducing the format is what makes
  drift impossible. The instruction is carried on the rows as `_instruction` and resolved by
  `resolve_generation_instruction(rows)`, which each side runs over its own rows exactly as the
  trainer already derives `labels` for classification.
  - Resolved once per DATASET, not per row: synthetic rows are built fresh and carry no
    `_instruction`, so a per-row lookup would give real and synthetic rows different prompts
    inside one training set.
  - The field is underscore-prefixed so `_new_example_prompt` excludes it from the schema shown
    to the synthesis teacher, which therefore cannot invent or reword it.
  - `DEFAULT_GENERATION_INSTRUCTION` keeps the old wording, so math/QA behaviour is unchanged and
    only datasets that opt in are affected.
- **DialogSum instruction.** "Summarize the following conversation in one to three sentences.
  Write only the summary — do not continue the conversation or reply to it." The second sentence
  targets the observed failure directly.
- **Tests.** 11 in `tests/test_generation_prompt_parity.py`, including an assertion that the
  trainer and eval harness emit byte-identical prompts.
- **Status:** 🟢 fixed.

## B251 — the judge scored the chain-of-thought along with the answer
- **Symptom (latent, found while diagnosing a slow run).** CoT annotation is applied to
  `math_reasoning`, `code_generation` AND `generation`, and `_training_turn` builds the target as
  `<reasoning>...</reasoning>\n\n<answer>`. Math and code are unaffected because their extractors
  pull one specific thing (`_final_answer`, `_extract_code`). Generation's extractor was
  `raw.strip()` — so the ENTIRE string, reasoning block included, was handed to an LLM judge
  asked "how good is this summary?".
- **Impact.** A model that produced a perfect summary preceded by its reasoning would still score
  badly, because the judged text looks nothing like the reference. This depresses every
  judge-scored generation number, including the threshold calibration.
- **Fix.** `split_reasoning(raw) -> (reasoning, answer)` in `eval/scorers/generation.py`;
  `extract_predictions` now returns the ANSWER only, so the judge sees just that. Tolerant of
  case/whitespace variants and of a dropped opening tag (small models often emit only the closing
  one), and it keeps the full text when the model produced reasoning and nothing else — an empty
  prediction would score 0 and hide the real failure.
- **The reasoning is separated, not discarded.** `eval/harness.py::_attach_reasoning_to_failures`
  records it on each failure so a bad answer can be traced to bad reasoning rather than bad
  phrasing. It is never sent to the judge.
- **Tests.** 14 in `tests/test_cot_answer_split.py`.
- **Status:** 🟢 fixed.

## B252 — generation-family synthesis ran serially at 1/8 of the configured concurrency
- **Symptom.** `slm-dialogsum-samsum-cse-38186914` sat on one line —
  `SYNTH-FILL (top-up to target): have 3628 row(s), need 2678 more to reach target 6306` — for
  over two hours with no further output, looking hung. vLLM was healthy but reported
  `Running: 1 reqs, Waiting: 0 reqs` at 18 tok/s with GPU 0 at 21% utilisation.
- **Root cause.** `_synthesize_new_correct` (the generation/math/code path) was a plain `while`
  loop issuing ONE blocking `generate_fn` call at a time. Every other generator in the module —
  `_synthesize_new_gold` for classification, and `annotate_cot` — fans out over `_progress_map`
  with `_synth_concurrency` workers. The server is launched for 8 concurrent requests
  (`SLM_SYNTH_CONCURRENCY`), so this path used an eighth of capacity already paid for. For
  contrast, CoT annotation on the same server ran at `Running: 8 reqs` and 138 tok/s.
- **Second defect.** The loop emitted no progress line until it finished, so a multi-hour phase
  was indistinguishable from a hang. `_progress_map` logs every ~10%.
- **Fix.** Rewritten over `_progress_map` with `_synth_concurrency` workers, over-requesting to
  absorb rejects and trimming to `n`. Verified with an instrumented fake generator: peak
  in-flight concurrency 16, previously 1.
- **Status:** 🟢 fixed.

## B253 — `xlam_bfcl` cannot load either of its two datasets
- **Symptom (2026-08-12).** Direct `load_dataset` calls on the cluster under `datasets 4.3.0`:
  `Salesforce/xlam-function-calling-60k` raises `DatasetNotFoundError` ("is a gated dataset"), and
  `gorilla-llm/Berkeley-Function-Calling-Leaderboard` raises `DataFilesNotFoundError`
  ("No (supported) data files found").
- **How it was found.** Auditing the four never-run benchmark loaders before scheduling the next
  batch of runs. `xlam_bfcl` has no run log; it has never been executed against live HF.
- **Root cause, part 1 — gating.** xLAM is `gated: auto`, i.e. click-through license. It needs an
  accepted license on the account whose token is in the environment, and `HF_TOKEN` exported into
  the Slurm env. `tests/pipeline/_l40s_task_body.sh` does not export one.
- **Root cause, part 2 — file resolution.** BFCL ships 52 files named `BFCL_v3_simple.json`,
  `BFCL_v3_parallel.json`, `BFCL_v3_multi_turn_base.json`, … None match the split-name patterns
  `datasets` uses to auto-build a config, so `load_dataset(BFCL_ID, split="train[:N]")`
  (`data/loaders/xlam_bfcl.py:85-87`) resolves nothing.
- **Root cause, part 3 — the fallback doesn't catch it.** The guard is
  `except (ValueError, KeyError)`; `DataFilesNotFoundError` is neither, so the run dies instead of
  degrading.
- **Root cause, part 4 — schema.** BFCL v3 stores prompts and gold answers in *separate* files
  (`BFCL_v3_simple.json` vs a matching `possible_answer/` entry), and gold answers are lists of
  acceptable values, not a single call. `convert_xlam_rows` reads `answers` off the same row, so
  even a successful load would drop every row and hand `build_eval_set` an empty list.
- **Why the tests passed.** `convert_xlam_rows` is a pure function unit-tested on in-memory
  samples. Nothing exercises the `load_dataset` calls.
- **Fix.** Accept the xLAM license + plumb `HF_TOKEN`; name BFCL files explicitly via
  `load_dataset("json", data_files={...resolve/main/BFCL_v3_simple.json})`; add a BFCL converter
  that joins prompts to `possible_answer`; widen the `except` to `Exception`.
- **Status:** 🔴 open.

## B254 — `routerbench` ships only pickles, and the loader reads a column that doesn't exist
- **Symptom (2026-08-12).** `load_dataset("withmartian/routerbench", split="train[:5]")` raises
  `DataFilesNotFoundError`.
- **Root cause, part 1.** The repo contains exactly `routerbench_0shot.pkl`,
  `routerbench_5shot.pkl`, `routerbench_raw.pkl` and a README. `datasets` has no pickle reader,
  so `data/loaders/routerbench.py:77` fails on every split, and the
  `except (ValueError, KeyError)` fallback at line 83 doesn't catch it either.
- **Root cause, part 2 (latent, worse).** `_correctness` looks for a field literally named
  `small_model_correct` (`routerbench.py:22`). RouterBench stores one correctness column per
  candidate model, named after the model. With a working reader every row would still return
  `None` and be dropped, yielding an empty dataset. The loader was written against an imagined
  schema.
- **Fix.** `huggingface_hub.hf_hub_download` + `pandas.read_pickle`, then choose a real model
  column as the routing boundary and pass it as `small_model_key`. Part 2 is a schema decision —
  which model counts as "small enough to keep on device" — not a mechanical fix.
- **Status:** 🔴 open.

## B255 — `tner/bc5cdr` is script-based and dead under `datasets>=4`; only the third fallback works
- **Symptom (2026-08-12).** `load_dataset("tner/bc5cdr", ...)` raises
  `RuntimeError: Dataset scripts are no longer supported, but found bc5cdr.py`.
  `spyysalo/bc5cdr` raises `DatasetNotFoundError` — the repo is gone.
- **Impact.** Candidates 1 and 2 of the `bc5cdr` ladder in `web_acquire.py:146-162` both fail.
  Candidate 3 — the script-free raw-JSON path added for exactly this reason — **works**, verified
  returning `['tags', 'tokens']`. So NER still runs, just after two failed attempts and a slow
  cold fetch.
- **Same class, elsewhere.** `AmazonScience/massive` and `iohadrubin/smcalflow` fail identically.
  Any future loader pointed at a script-based repo is dead on arrival; check for a `.py` at the
  repo root before wiring one up.
- **Better fix than repairing the ladder.** `data/local/bc5cdr` already holds a checksummed
  frozen bundle with **5,096 train / 5,865 test** rows — more gold than the live loader's 3,403,
  which directly addresses the exhausted-plan-space crash that ended
  `slm-ner-l40s-37531245`. Set `SLM_LOCAL_DATASET_DIR` and take rung 1 of the ladder.
- **Status:** ⚪ design gap — works via fallback, but two of three candidates are permanently dead
  and the ordering wastes a cold fetch on every run.

## B259 — RouterBench acquire mines a foreign dataset and an LLM invents its labels
- **Symptom (2026-08-15).** Every RouterBench rebuild after the first acquire round has QC delete
  ~1,166 rows whose labels are `cloud` / `on_device` / `router` / `remote` — classes that do not
  exist in a task whose label space is `{local, route}`. The out-of-vocabulary set *grows* through
  the run: `cloud` only for iterations 1–9, `+on_device` at 10, `+router` at 20, `+remote` at 21.
- **Root cause, part 1.** `web_acquire._BENCHMARK_ALIASES` has no `routerbench` entry, so Stage-0
  fails on the benchmark the task is about (`No (supported) data files found in
  withmartian/routerbench`, 21×) and agentic discovery substitutes
  `anasnassar/llm-query-complexity-benchmark`.
- **Root cause, part 2 (the real defect).** That dataset has **no routing labels at all** — its
  columns are `text, source, subject, domain, ground_truth, id` and `ground_truth` is
  `LOW`/`MEDIUM`/`HIGH` query-complexity over StackExchange/MMLU/PubMedQA. The `local`/`route`
  labels are **invented by `_llm_map_dataset`** (`web_acquire.py:953-995`), which asks Claude for a
  `label_map`; `_materialize_from_mapping` then applies `lmap.get(str(lab), lab)` — **unmapped values
  pass through verbatim**. The mapping is re-requested per acquire round and is
  non-deterministic, so each round can mint a *new* hallucinated class.
- **Root cause, part 3 — the guard is per-source, not per-row.** `_labels_are_usable`
  (`web_acquire.py:627-645`, added for B222) returns `True` on **any** overlap with the known label
  space. Because Claude mapped some rows to `local`/`route`, the whole source passed and the
  hallucinated classes rode along.
- **Impact, measured.** `dataset_v10.jsonl` holds **1,155 foreign rows = 34% of the training set**,
  labelled `local`/`route` split **579/576**. RouterBench's true base rate is **30/70**. The 50/50
  split is the signature of fabricated labels. These rows survive QC because the label strings are
  spelled correctly — this is worse than anything QC removes, and it is a plausible contributor to
  the `route` mode collapse (§B261).
- **Fix.** (a) add a `routerbench` alias so acquire uses the real pickle loader; (b) drop
  out-of-vocabulary **rows** rather than passing a whole source on any overlap; (c) require an
  explicit, validated `label_map` covering every source value, and reject rather than pass through.
- **Status:** 🔴 open. Not changed mid-run (38493142 was live).

## B260 — `length-outlier` QC is relative to the dataset's own median, so contamination makes it delete real data
- **Symptom (2026-08-15).** RouterBench's `[qc] length-outlier` removals climb monotonically through
  the run — 58 → 39 → 67 → 132 → 168 → 566 → … → **813 rows per rebuild**.
- **Root cause.** The filter cuts anything longer than **3× the dataset's own median**
  (`data/curriculum.py:254-264`). The rows B259 injects are short — median **75** characters against
  real RouterBench's **715** — so they drag the median down and pull the cutoff with them:

  | Version | mined rows | median | 3× cutoff | share of real RouterBench deleted |
  |---|---|---|---|---|
  | uncontaminated | 0 | 715 | 2145 | **0.6%** (216 / 36,497) |
  | `v1` | 0 | 416 | 1248 | 9% |
  | `v5` | 958 | 295 | 885 | 47% |
  | `v10` | 1,155 | 269 | 807 | **48%** (17,667) |

  Visible in the artifacts: `train_anchor` median length falls **572 → 432 → 396** across v1/v5/v10
  while the pool it is sampled from never changes — the filter is eating the long MMLU/GSM8K/hellaswag
  prompts, which are most of the real signal.
- **Not label-biased.** Deleted rows are 32.6% `local` vs 29.9% overall, so it is near-uniform
  destruction rather than skew. The problem is volume, not bias.
- **Fix.** Compute the median over **trusted** rows only (train anchors), or make the bound absolute
  and task-aware. The 3× ratio itself is fine — on clean data it removes 0.6%.
- **Status:** 🔴 open. **The QC thresholds are not the bug**; B259 is. Fixing B259 makes this inert.

## B261 — `route` mode collapse is scored 0.0000 and iterated through
- **Symptom (2026-08-15).** Five RouterBench iterations show `easy=0.000 medium=0.000 hard=0.882`
  with `F1=0.0000` (l40s 5539-5542, 6467, 6719, 11822, 14252; cse 8818-8821). The model answers
  `route` for everything; the hard bucket is route-heavy, so aggregate difficulty looks healthy.
- **Assessment.** The metric is **correct** — minority-class F1 rightly gives no credit for a
  degenerate majority strategy, and plain accuracy would have flattered the same model with 63%. The
  defect is that nothing detects the collapse: the orchestrator spends full train → quantize → eval
  cycles on it.
- **Fix.** Detect a single-class prediction distribution at eval time and surface it as its own
  diagnosis rather than as a score of zero.
- **Status:** 🔴 open.

## B262 — `calendar_json` baseline disagrees with the iteration eval path (0.2176 vs 0.0000)
- **Symptom (2026-08-15).** `slm-calendar-json-cse-38505239.out:95` reports
  `[baseline] reference ast_arg_match=0.2176`; line 257 reports `Baseline F1 = 0.0000`, and lines
  297-301 `F1=0.0000 failures=478/478`. Two numbers for the same model on the same eval set.
- **Impact.** The orchestrator optimises against whichever zero it is shown. Compounded by the eval
  set being **478 rows against a target of 800** (line 58), so calendar scores are not comparable to
  the other tasks.
- **Fix.** Reconcile the reference-endpoint path with the GGUF/llama.cpp scoring path; investigate the
  eval-size shortfall separately.
- **Status:** 🔴 open. Distinct from the known unguessable-year gold defect.

## B263 — NER sample-prediction display never prints gold
- **Symptom (2026-08-15).** `gold :` is blank on every NER row in both the baseline and fine-tuned
  eval blocks (`slm-ner-bc5cdr-cse-38455148.out:258, 262, 266, 298, 302, 306`).
- **Root cause.** The display reads a field NER rows do not carry — they hold `entities`, not
  `label`/`answer`.
- **Impact.** Cosmetic, but it removes the only human check of gold against prediction, and it is why
  a legitimate `Baseline F1 = 0.0000` looked like instrumentation failure.
- **Status:** 🔴 open.

## B264 — orchestrator decision JSON truncated at 4,096 output tokens
- **Symptom (2026-08-15).** 20 occurrences in `slm-dialogsum-samsum-l40s-38303490.out` (e.g. 316-317)
  of `response was cut off after 4096 output tokens (stop_reason=max_tokens)`, each triggering a
  ~$0.02 reask.
- **Assessment.** The B240 error message is doing its job — it says plainly "too long, NOT malformed"
  — but a reask is the wrong remedy for an output-budget failure. `_ITERATE_MAX_TOKENS` defaults to
  20,000, so this path was running under a smaller effective cap.
- **Fix.** Raise the budget for the truncating call site rather than retrying.
- **Status:** 🔴 open.

## B265 — `_threshold_raise_asked_iteration` undeclared in `AgentState` (caught pre-merge)
- **Symptom (2026-08-15).** `test_every_key_iterate_persists_is_declared` failed as soon as the
  stretch-goal guard was added.
- **Root cause.** Exactly **B256** again, in a new key: keys not on the `AgentState` schema are
  **dropped when LangGraph merges a node's returned state**, so the flag would have read back as
  absent and the orchestrator would have been asked the stretch-goal question twice per turn — the
  same mechanism that made iterate re-log its whole system prompt on all 69 turns.
- **Fix.** Declared in `agent/state.py`.
- **Status:** 🟢 fixed. Worth noting the guard test caught this before a run ever saw it.

## B266 — `tests/nodes/test_iterate_prompt.py` had been silently failing since 2026-07-31
- **Symptom (2026-08-15).** `test_iterate_prompt_receives_complete_curation_counts` failed with
  `unsupported data_rebuild field(s): primary_strategy`.
- **Root cause.** The fixture returned `{"primary_strategy": "resample_existing"}`, a field the
  2026-07-31 curation redesign retired (the plan doc explicitly states "No `primary_strategy`"). The
  decision failed validation, the reask failed identically, and the test's real assertion — that
  curation counts reach the orchestrator prompt — had stopped running.
- **Fix.** Fixture updated to `{"strategy": "resample"}`.
- **Status:** 🟢 fixed.

## B267 — the teacher was asked to judge the label's WORDING, not the task's class
- **Symptom (2026-08-15).** RouterBench's teacher rejected 70% of generated rows (876 `REJECTED`
  lines in one run, 1,378 in another) with verdicts like:
  ```
  REJECTED [local] 'What is 15 percent of 200?' — the utterance is a math question, not related
                                                  to local services or location
  REJECTED [cloud] 'The thick fog rolled in and obscured the view' — the utterance describes fog,
                                                  not clouds
  ```
- **Root cause.** `verify_generated_labels` showed the teacher the label STRING and no task context
  whatsoever: *"Does this utterance genuinely belong to the 'local' class?"*. `local` in this task
  means "a small on-device model can answer this correctly", but with nothing to say so the teacher
  read it as the English word and judged **topic** instead of **class**. The generator prompt had the
  same flaw (`"an utterance that belongs to the 'local' class"` → weather/restaurant queries), which
  is also why `cloud` looked like a plausible sibling class for the LLM to hallucinate (B259).
- **Assessment.** The teacher was **not** being too strict. Given the prompt it was handed, those
  verdicts are correct answers to the wrong question.
- **Fix.** `data/label_space.py::label_definitions_for` supplies per-benchmark label definitions;
  `_label_context_block` injects them into BOTH the generator and verifier prompts, and the verifier
  is explicitly told not to reject on topic mismatch. Tasks whose label already describes the text
  (CLINC150 intents) get no definitions and behave exactly as before.
- **Residual.** This makes the teacher ask the right question; it does not make it good at that
  question (0.5311 on RouterBench). The real remedy is not synthesizing for such tasks at all —
  deferred, see the 08-15b note §2.1.
- **Status:** 🟢 fixed (prompting); ⚪ underlying competence question open.

## B268 — NER synthesis has never produced a single row, silently
- **Symptom (2026-08-15).** Every NER synthesis call logs `kept 0`, including
  `[synth] requested 5693 new-gold row(s) -> kept 0`. The BC5CDR curriculum ran ~7,100 rows below an
  8,929-row target with no explanation.
- **Root cause.** `_synthesize_new_gold` buckets anchors by `row.get("label")`. **NER rows have no
  `label`** — their gold is in `entities` — so `by_label` is always empty and the function returns
  `[]` at `data/curriculum.py:522` before making any API call.
- **Latent second bug the first one was hiding.** Had the path run, it returns `{text, label}` and
  never entity spans, so a produced row would either have no `entities` (dropped by the NER schema
  filter) or carry the ANCHOR sentence's spans against NEW text — fabricated gold.
- **Fix.** Kept as a deliberate no-op and made explicit: the log now states that NER rows cannot
  anchor in-class generation, that this generator cannot produce spans, and that NER curricula are
  gold-only by design with no teacher calls made. Span synthesis would need its own generator.
- **Note.** BC5CDR converged at 0.8098 gold-only in 81 minutes on a 0.6B, so real data alone was
  sufficient there. Also means `docs/DATA_CURATION_AND_CAPS.md`'s claim that NER synthesis works was
  wrong; corrected.
- **Status:** 🟢 fixed (now honest and intentional).

## B269 — generation-family synthesis is entirely unverified
- **Symptom (2026-08-15).** `new-correct synthesis: 450/450 kept`, every call, in every
  generation-family run.
- **Root cause.** `_synthesize_new_correct` only checks candidates `if verify_fn is not None`, and
  `curate._verifier_for` returns `None` for every task type. The branch has never executed.
- **Impact.** For these task types the teacher invents BOTH the input and the correct output, with no
  check. On `calendar_json` the teacher scores **0.2176** and the curriculum target was 8,929 rows
  against 3,250 real ones — ~5,700 machine-invented training targets from a model that gets the task
  right 22% of the time. Unlike classification, the anchor-label-inheritance argument does not even
  apply here, because the output is generated rather than copied.
- **Status:** 🔴 open — highest-risk remaining synthesis defect. Should be covered by whatever
  competence gate is chosen (08-15b note §2.1).

## B270 — `Baseline → Best FT` could not separate fine-tuning from the search loop
- **Symptom (2026-08-15).** The Model Improvement Report showed only endpoints, so BC5CDR's
  `+0.8098` read as one uniform gain when it was almost entirely iteration 1 (0.0000 → 0.7701, with
  four further iterations adding 0.0397). The tier-3 RouterBench row is the opposite case: its first
  fine-tune (0.2675) was WORSE than its baseline (0.5443) and the search recovered +0.4909.
- **Why it could not be derived after the fact.** When the zero-shot baseline beats the first
  fine-tune it becomes iteration 1's recorded score (B161's "baseline as candidate"), so
  `scores[0]` and `baseline_f1` are indistinguishable downstream — exactly the tier-3 case above.
- **Fix.** `evaluate_node` records `first_finetuned_f1` at the moment of measurement, before the
  baseline is added to the candidate pool; threaded through `escalation_history`,
  `build_run_progression`, and the report, which gains `First FT`, `Δ base` and `Δ search` columns.
- **Status:** 🟢 fixed.

## B271 — classification rows that are themselves instructions, plus an extractor that invented labels
- **Symptom (2026-08-16).** RouterBench zero-shot baselines came out **0.4615 / 0.1685 / 0.1701 /
  0.5443** across the 0.6B / 1.7B / 2B / 4B tiers — an ordering with no relation to model size — on
  **472 / 511 / 333 / 157** extraction failures out of 800.
- **Root cause, part 1 — prompt injection by the payload.** `CLASSIFY_PROMPT` put the row text LAST
  and unfenced. RouterBench prompts are drawn from 86 upstream benchmarks and many carry their own
  output contract (`"Print only a single choice from A or B or C or D without explanation. Answer:"`,
  or the Chinese `请仅回复楚辞名` — "reply only with the name of the Chu Ci"). The most recent
  instruction the model read was therefore the row's own, and it obeyed that: raw outputs were `A`,
  `B`, `2021`, `area`, `楚辞`, `ethical` — correct answers to the embedded question, every one of them
  `__EXTRACTION_FAILED__`. **Not a multilingual bug**; the Chinese rows only made it obvious.
- **Root cause, part 2 (the worse half) — the extractor assigned labels by accident.** The final pass
  was a bare substring scan of the whole output. `local` and `route` are ordinary English words: they
  occur inside "locally", "router", "routed", and `en route` matches `route` on a **word boundary**.
  So a model that ignored the task and wrote a paragraph was scored on whether its prose happened to
  contain a label substring — and longer answers are MORE likely to. Tier 2 parsed *more* outputs
  than tier 0 (467 vs 328) and scored far worse, which is what rules out extraction rate as the
  explanation and points at label quality.
- **Consequence.** The zero-shot baselines were never measuring capability, so no tier comparison
  drawn from them is valid. Tier 3's 0.5443 is the only trustworthy one, because Qwen3-4B-**Instruct**
  actually follows the outer instruction (157 failures).
- **Fix.** (a) The payload is fenced in `<<<MESSAGE … MESSAGE>>>`, explicitly declared to be DATA and
  not instructions, and the output contract is RESTATED after it so the last thing read is ours.
  (b) Extraction enforces that contract: the answer IS a label, or a label word appears in the
  answer's TAIL, or a label appears in a SHORT answer — otherwise it is an honest failure. Qwen's
  `<think></think>` wrapper is stripped before measuring length. Both sides of train/serve move
  together because `build_classify_prompt` is shared with the trainer.
- **Expected effect.** Chatty base models score LOWER at baseline, which is correct — they did not do
  the task. Fine-tuned models are unaffected; they emit the bare label.
- **Status:** 🟢 fixed. The stronger fix — constrained label decoding, which makes extraction failure
  structurally impossible — remains open.

## B272 — the tier ladder is confounded with data contamination
- **Symptom (2026-08-16).** Tier 2 (Qwen3.5-2B) reached a LOWER best (0.6429/0.6561) than both tier 1
  (0.6809/0.6960) and tier 0 (0.6486/0.6551), despite being the larger model and getting 16–20
  iterations.
- **Root cause.** Mined foreign rows accumulate PERMANENTLY in the train pool (B259), so curriculum
  quality degrades monotonically as the run progresses — and the tier ladder also advances
  monotonically. The two are inseparable:

  | dataset | mined foreign rows | tier that trained on it |
  |---|---|---|
  | `v1` | **0** | tier 0 |
  | `v5` | 958 | tier 0/1 |
  | `v10` | 1,155 | tier 2 |
  | `v14` | 1,155 | tier 3 |

  Tier 0 trained on the cleanest curriculum the run ever had; tier 2 on the most contaminated, and
  tier-2 entry is exactly where QC removed 1,134 out-of-vocabulary rows and landed 3,404 rows against
  a 6,181 target.
- **Ruled out.** Not undertraining (16–20 iterations, escalated on stagnation). Not a multimodal
  problem — Qwen3.5-2B trained through the standard Unsloth 16-bit LoRA path with no `FastVisionModel`
  branch and no vision warnings.
- **Consequence.** **No claim of the form "model X is worse than model Y on this task" is supported by
  these runs.** Any future tier comparison needs a curriculum held fixed across tiers.
- **Status:** ⚪ design gap. B259/B260 remove the cause going forward; the affected runs stay invalid.

## B273 — LlamaPIE decision points are any `|MARKER >`, not only `|SILENCE >`
- **Symptom (2026-08-16, caught pre-launch).** The new `proactive_listening` loader found **zero
  positives in all 7,065 `synthetic0` dialogues**, and the training positive rate came out 4.2%
  against a 33.4% eval rate.
- **Root cause, part 1.** `synthetic0` annotates speaker emotion and attaches whispers to `|ANGRY >`
  / `|NEUTRAL >` markers rather than to `|SILENCE >`. The authors' own dataset code masks on the ` >`
  token (`Active_dataset.py`: `symbol_token2 = tokenizer(" >")[...]`), so *any* `|MARKER >` is a
  decision point.
- **Root cause, part 2.** The bundle is grouped by sub-corpus, so truncating an unshuffled expansion
  to `max_train` drew entirely from whichever corpus came first.
- **Fix.** Regex `\|[A-Z_]+ >` for decision points; dialogues shuffled before expansion; `synthetic0`
  excluded by default (it carries emotion markers the held-out split has none of, which would be a
  train/serve mismatch — `SLM_PROACTIVE_SOURCES` to opt in). A test asserts train and eval positive
  rates match within 5 points.
- **Status:** 🟢 fixed before the task ever ran.

## B274 — WITHDRAWN: `calendar_json`'s "two baseline scores" are two different models
- **Claimed (2026-08-15, B262).** That `calendar_json` reported `ast_arg_match=0.2176` and
  `Baseline F1 = 0.0000` for the same model on the same eval set.
- **Actually.** Two different models, exactly as designed. `[baseline] reference …` comes from
  `eval/endpoint_eval.py::measure_endpoint_baseline` and scores the **Qwen3.6-35B teacher** to
  calibrate the accuracy goal. `Baseline F1 = …` comes from `agent/nodes/evaluate.py` and scores the
  **Qwen3-0.6B student** about to be fine-tuned. The same pattern appears in RouterBench
  (`reference macro_f1=0.5341` teacher vs `Baseline F1 = 0.4615` student).
- **Status:** ⚪ not a bug. B262 withdrawn. The separate 478-vs-800 eval-size shortfall is still open.

## B275 — empty difficulty buckets are weighted and reported as failures
- **Symptom (2026-08-16).** The BC5CDR run reported `medium=None` eight times while still handing the
  orchestrator `difficulty_buckets.medium = 0.25` in rebuild plans.
- **Root cause.** Buckets come from a two-model capability gradient (`label_difficulty`): `easy` =
  both the smallest and largest model pass, `medium` = only the largest passes, `hard` = neither. On a
  **format-bound** task neither model can produce the output contract zero-shot, so nothing lands in
  `medium` and almost everything lands in `hard`. The 182 `easy` rows are essentially the rows whose
  gold entity list is EMPTY, where emitting `[]` is correct — consistent with 705/1,785 (39.5%) of
  training rows having empty entities, and with the fine-tuned model scoring easy=0.984 / hard=0.605.
  So NER's `easy` bucket measures **abstention, not extraction**.
- **Impact.** The orchestrator allocated a quarter of its curriculum budget against an empty set, and
  `medium=n/a` reads like a measurement failure rather than a structural fact.
- **Fix (not implemented).** Drop empty buckets from the plan schema and redistribute the weight;
  report `n=0` explicitly; fall back to the length-tercile heuristic when a bucket is degenerate,
  since a zero-shot gradient carries no signal precisely on format-bound tasks.
- **Status:** 🔴 open.

## B276 — the teacher's zero-shot score was measuring format compliance, not competence
- **Measured (2026-08-17, job 38555749).** Qwen3.6-35B on 200 BC5CDR eval rows, same scorer and same
  frozen eval set the pipeline uses, demonstrations drawn from TRAIN only:

  | shots | span_f1 | empty/unparseable |
  |---|---|---|
  | 0 | **0.1131** | 131/200 |
  | 1 | 0.4910 | 101/200 |
  | 3 | 0.6736 | 90/200 |
  | 5 | **0.7190** | 84/200 |

  **6.4× from five demonstrations.**
- **What the raw output shows.** Zero-shot it emitted ` ```json ` fences, `"type": "CHEMICAL"` (wrong
  case) and `"type": "GENE_OR_PROTEIN"` (a class BC5CDR does not have). One demonstration fixed all
  three. Min et al. (EMNLP 2022, arXiv:2202.12837) account for exactly these three: demonstrations
  supply "(1) the label space, (2) the distribution of the input text, and (3) the overall format".
- **Consequence.** Every inference of the form "the teacher scores 0.0999 on NER, so it cannot generate
  NER training data" rested on the wrong number. The zero-shot score measures whether the teacher
  guessed our output contract; synthesis anchored to a real example is a different regime.
- **Not a code defect** — the threshold calibration is *supposed* to use zero-shot, since that is what
  the student is measured against. What is wrong is using the same number as a proxy for synthesis
  fitness.
- **Also worth recording:** 0.7190 few-shot is still BELOW the fine-tuned 0.6B student's 0.8098. The
  project's central claim survives and is better supported, because the comparison is no longer against
  a teacher crippled by a format artifact.
- **Status:** ⚪ measurement. The decisive follow-up (corrupt demonstration labels, hold format fixed)
  is not yet run — see the 08-17 note §5.5.

## B277 — WITHDRAWN: "systematic teacher errors are worse than random noise"
- **Claimed (2026-08-15b, 2026-08-16).** That random label noise averages out while systematic teacher
  bias is learned and amplified, citing Shumailov et al. (model collapse) and Gudibande et al.
- **Actually.** No paper supports the asymmetry, and the best controlled head-to-head reports the
  reverse. Jiang et al. (ICML 2020, arXiv:1911.09781), comparing synthetic uniform noise against
  real-world structured noise: *"DNNs generalize much better on web label noise"* and *"The real-world
  label noise from the web appears to be less harmful."* Rolnick et al. (arXiv:1705.10694) tested
  confusion-biased noise specifically: robustness held *"even when erroneous labels are biased towards
  confusing classes."*
- **The two citations were also misapplied.** Shumailov et al. study *recursive* training (generation n
  trains on n−1's output) — we do a single distillation step onto a separate student — and they name
  *statistical* (sampling) error as the primary driver, close to the reverse of the claim. Gudibande et
  al. is about *breadth*, and their own finding supports narrow single-task distillation: *"training
  exclusively on ChatGPT responses for Natural-Questions-like queries drastically improves task
  accuracy."*
- **Corrected rule.** Two variables decide synthesis safety, not three: **who produces the label**, and
  **whether an independent verifier exists**. Drop the third.
- **Status:** ⚪ withdrawn. Recorded because the claim appears in two earlier notes.

## B278 — `dialogsum_samsum` has train/eval overlap with contradictory gold
- **Found (2026-08-17) by `scripts/preflight_tasks.py`.** One eval row appears verbatim in the training
  set, and the two gold summaries disagree about the subject:
  ```
  text  : "Serena: Have you been to the doctor lately?  Jeff: No, why? …"
  eval  : "Serena's skin condition is fine now and she doesn't have to take medication…"
  train : "Jeff has a skin allergy. He doesn't take meds all the time…"
  ```
  Reading the dialogue, the TRAIN summary is correct and the EVAL gold is wrong.
- **Root cause.** DialogSum and SAMSum contain overlapping dialogues with independently written
  summaries. `load_dialogsum_samsum` draws half its rows from each and does not deduplicate across the
  two sources, so the same dialogue can land in train from one and eval from the other.
- **Impact.** One row in 300, so it did not move the reported 0.7157. But the eval set contains at least
  one unwinnable row, and this is the task where fine-tuning bought exactly **+0.0000** over 15
  iterations — an explanation of that result should not have a known data defect inside it.
- **Fix.** Deduplicate across the two sources by normalized dialogue text before splitting.
- **Status:** 🔴 open — blocks a rerun.

## B279 — `set -u` in a Slurm script breaks lmod, surfacing as a bogus nvcc permission error
- **Symptom (2026-08-17, job 38550928).** The teacher probe died after ~11 minutes of weight loading
  with `torch._inductor.exc.InductorError: PermissionError: [Errno 13] Permission denied: 'nvcc'`.
- **Root cause.** The script used `set -uo pipefail`. lmod's init references `LD_LIBRARY_PATH`
  unguarded, so under `nounset` it failed with `LD_LIBRARY_PATH: unbound variable` — `module load cuda`
  never took effect, `nvcc` was not on PATH, `CUDA_HOME` resolved to garbage, and torch-inductor's JIT
  reported a permission error rather than a missing binary. `tests/pipeline/_l40s_task_body.sh` does not
  use `set -u`, which is why the pipeline never hit this.
- **Fix.** `set -o pipefail` only, plus an explicit `command -v nvcc` guard that fails in seconds with
  an actionable message instead of minutes with a misleading one.
- **Status:** 🟢 fixed. Worth knowing for any future GPU Slurm script in this repo.

## B280 — generated format-bound rows had no exact verifier, though one was free
- **Symptom (through 2026-08-17).** `_synthesize_new_correct` invents both input and answer, and
  `curate._verifier_for` returned `None` for every task type, so the `if verify_fn is not None` branch
  had never executed. Every batch logged `450/450 kept`.
- **Why it mattered most on `function_call`.** The correctness of a generated call is *decidable by
  computation* — parse the JSON, check the name against the declared tools, check the argument keys
  against the schema — and none of it was being done. For `calendar_json` the datetimes are checkable
  too (end after start, 60-minute default, date near the request's own reference instant).
- **Fix.** `data/synth_verifiers.py` with `verify_function_call_row` (5 checks) and
  `verify_calendar_row` (those 5 + 5 datetime/summary checks), dispatched by benchmark and wired into
  `_verifier_for`. Runs **before** the model-based pass: exact, free, and a row it rejects never costs a
  teacher call. Rejections are logged grouped by reason.
- **Enabling change.** `tools` and `_instruction` are now PINNED from the anchor onto every generated
  row. A generated row without `tools` cannot be schema-checked at all, and the tool signature is the
  constraint rather than something being invented.
- **Status:** 🟢 fixed. 25 tests.

## B281 — synthesis and verification were zero-shot when few-shot is 6.4x better
- **Symptom (2026-08-17, B276).** The teacher scores 0.1131 span-F1 zero-shot on BC5CDR NER and 0.7190
  with five demonstrations. Every synthesis and verification prompt was zero-shot.
- **Why it matters for VERIFICATION as much as generation.** On `calendar_json` the conventions ARE the
  task (60-minute default, ISO-8601, resolve against the reference instant); a verifier that has to
  infer them is judging its own guess.
- **Fix.** `SYNTH_SHOTS = 5` (`SLM_SYNTH_SHOTS`) applied at four call sites: `_synthesize_new_gold`
  (5 same-class examples), `_synthesize_new_correct` (5 examples in the required JSON shape),
  `verify_generated_labels` (5 confirmed in-class examples), `verify_generated_answers` (5 real
  request/answer pairs). Min et al. (arXiv:2202.12837) is the account of why: demonstrations supply
  "(1) the label space, (2) the distribution of the input text, and (3) the overall format".
- **Status:** 🟢 fixed.

## B282 — every curated benchmark's eval set was capped at 800 rows
- **Symptom (2026-08-18).** `_load_named_benchmark` passed `eval_size_target` (default 800) as the
  loader's `max_test`, regardless of how much held-out data the benchmark shipped. BC5CDR has **5,865**
  test rows and was being scored on 800.
- **Impact.** Unnecessary variance on exactly the comparisons the project makes — tier vs tier, teacher
  vs student. RouterBench's tier ordering was being read off 800 rows.
- **Fix (revised 2026-08-19).** Cap at **1,000** rows (`_EVAL_SIZE_CAP`, override with
  `SLM_EVAL_SIZE_CAP`), not the whole split. Taking the whole split was the first attempt and it is the
  wrong trade: the eval runs every iteration, so RouterBench's 7,267 rows would be 9x the old
  per-iteration cost for a variance gain that flattens out well before that. At n=1,000 the standard
  error on a proportion is ~1.5pp, below the differences this project resolves. New sizes: 1,000 for
  routerbench / ner_bc5cdr / clinc150 / proactive_listening / xlam_bfcl; dialogsum_samsum 667 and
  calendar_json 478 (their whole splits are smaller).
- **Caveat.** Scores measured before this change are on a smaller eval set and are not strictly
  comparable to scores measured after.
- **Status:** 🟢 fixed.

## B283 — `calendar_json`'s gold required an unguessable year, and leaked the imperative into the title
- **Symptom (2026-08-13 run, root-caused 2026-08-16, fixed 2026-08-18).** 82% of eval gold answers
  required rolling the date forward to 2027, and the task scored 0.0000.
- **Root cause 1.** `reference_for` scattered the reference instant uniformly across 2026 while SGD's
  calendar dialogues are almost all set in **March**, so the reference usually fell after the event's
  month, `_parse_date`'s roll-forward rule fired, and gold landed in 2027. The model answered
  2026-03-02 for "on 2nd of March" — the more natural reading — and was marked wrong.
- **Root cause 2.** The request was built as `f"Schedule {summary} on {date}"`, so a row titled `Food`
  read "Schedule Food on March 1st" and the model extracted `summary="Schedule Food"`.
- **Fix.** `_reference_before_event` places the reference a hashed 1–21 days BEFORE the event, so no
  rollforward is needed and the year is inferable from the prompt (the offset still varies per row, so a
  fixed "today" cannot be memorised). The request now reads `Add "<title>" to my calendar on …`.
- **Measured.** Eval gold in the same year as the reference: **18% → 99%** (475/478). Rolled forward:
  **82% → 1%**. Summary containing the imperative: systematic → **0**.
- **Status:** 🟢 fixed. The verifier in B280 also catches both defects at generation time.

## B284 — `calendar_json` eval data was fetched live from GitHub and never cached
- **Symptom.** `load_sgd_calendar` read dialogue JSON from `raw.githubusercontent.com` at load time on
  every run. One upstream commit silently changes the eval set, so past scores stop being comparable and
  are not reproducible; the run also cannot start without network access.
- **Fix.** `data/local/calendar_sgd/` vendors the 1,602 Calendar-service dialogues (17 MB) with a
  manifest and sha256, the same treatment `bc5cdr` and `proactive_listening` get. The loader reads it
  first and falls back to the network with a loud non-reproducibility warning.
  `SLM_CALENDAR_SGD_DIR` overrides. Load time ~140 s → ~18 s as a side effect.
- **Status:** 🟢 fixed.

## B285 — SAMSum ships a few dialogues in both its own splits, with contradictory gold [SEVERITY CORRECTED]
- **Symptom (2026-08-18), found by `scripts/preflight_tasks.py`.** Six eval rows appeared verbatim in
  the training set, with different gold:
  ```
  text  : "Serena: Have you been to the doctor lately?  Jeff: No, why? …"
  eval  : "Serena's skin condition is fine now and she doesn't have to take medication…"
  train : "Jeff has a skin allergy. He doesn't take meds all the time…"
  ```
  Reading the dialogue, the TRAIN summary is correct and the EVAL gold is wrong.
- **Root cause, CORRECTED 2026-08-19.** I first reported this as a DialogSum/SAMSum cross-source merge
  problem. It is not. Measured on a 500+500 draw, all collisions are **samsum_train x samsum_test** —
  SAMSum's own official splits are not disjoint, and it wrote a different summary for the same dialogue
  in each. `load_dialogsum_samsum` did not deduplicate.
- **Impact — NEGLIGIBLE, and the original entry overstated it.** 2 rows out of 1,000 (~0.2%). It did not
  move the run's 0.7157, and curate's eval firewall already removed the training side before training,
  so no run was ever contaminated. The tier-3 +0.0000 result is NOT explained by this.
- **Fix.** Deduplicate by normalized dialogue text, **eval first** — an eval row is never dropped, a
  colliding training row is. Logged. Kept as cheap hygiene (the reported curriculum size is honest
  instead of being silently shrunk by the firewall), not because the defect mattered.
- **Status:** 🟢 fixed, ⚪ severity downgraded to cosmetic.

## B286 — WITHDRAWN: constrained label decoding is not worth doing
- **Claimed (2026-08-16, 2026-08-17, 2026-08-18).** That scoring the label set instead of parsing free
  text was "the single biggest remaining eval improvement", because it would make
  `__EXTRACTION_FAILED__` structurally impossible.
- **Measured (2026-08-19).** Extraction-failure rate per eval, split by baseline vs fine-tuned:

  | Run | BASELINE | FINE-TUNED |
  |---|---|---|
  | routerbench l40s | median 50.3% | median **0.0%**, mean 2.6%, 47/77 evals had ZERO |
  | routerbench cse | median 50.3% | median **0.0%**, mean 1.7%, 50/75 ZERO |
  | clinc150 | 34.0% | median **2.1%** |
  | ner_bc5cdr | 0.0% | **0.0%** on all 7 |
  | xlam_bfcl | 0.0% | **0.0%** on all 23 |
  | dialogsum_samsum | 0.0% | **0.0%** on all 33 |

- **Conclusion.** Extraction failure is a **zero-shot-only** phenomenon. After fine-tuning the median is
  exactly zero on every task and three of six tasks never had a single failure. The 50% spikes in the
  RouterBench fine-tuned column are the mode-collapse iterations (`</tool_call>` on 522/800 rows) — a
  TRAINING pathology, which a constrained decoder would convert into confident wrong labels rather than
  fix.
- **Cost/benefit.** It needs a log-probability path in BOTH inference backends (Unsloth, llama-cpp-python)
  which expose scoring differently, and it invalidates every classification baseline ever measured. That
  is a large eval-harness change for a 0–2% effect on the numbers the project reports.
- **Status:** ⚪ withdrawn. The prompt hardening from B271 is the proportionate fix. If baseline honesty
  matters later, report a FEW-SHOT baseline alongside the zero-shot one — the probe harness already
  exists and it touches no inference code.

## B287 — the teacher's few-shot gain is ~77% FORMAT, not knowledge
- **Measured (2026-08-19, job 38558845).** The Min et al. (arXiv:2202.12837) ablation, adapted to
  exact-span NER: show the teacher five demonstrations that are perfectly formatted and factually WRONG
  (JSON shape, `Chemical`/`Disease` vocabulary and span count preserved; span text replaced with entity
  names borrowed from other rows, so they are real biomedical terms absent from this sentence).

  | condition | span_f1 |
  |---|---|
  | 0-shot | 0.1147 |
  | 5-shot, correct demos | **0.7215** |
  | 5-shot, CORRUPTED demos | **0.5833** |

  Corrupted demonstrations retain **(0.5833−0.1147)/(0.7215−0.1147) = 77%** of the gain.
- **Interpretation.** The teacher's zero-shot score was mostly measuring whether it guessed our output
  contract. Visible in the raw output: zero-shot it emitted a ```json fence, `"type": "CHEMICAL"` (wrong
  case) and `"type": "GENE_OR_PROTEIN"` (a class BC5CDR does not have). Demonstrations fix all three, and
  CORRUPTED demonstrations fix them just as well — format is what survives corruption.
- **The remaining 23% is real knowledge** and should not be overstated: the 0.1382 gap between correct and
  corrupted demos is the demonstrations genuinely teaching the task.
- **Consequence.** **A zero-shot score is not a valid gate on synthesis fitness** — here it understated
  usable ability by ~5x for reasons unrelated to competence. Any synthesis gate should use the observed
  keep-rate (already computed) or a few-shot measurement instead. Retroactively justifies B281 (all
  synthesis and verification made 5-shot).
- **Limit.** One task, one model. BC5CDR is unusually format-dominated; `calendar_json`, where the
  difficulty is a date convention rather than an output shape, could split differently. `--task` makes
  that one command.
- **Status:** ⚪ measurement, decisive for the synthesis question.

## B288 — B282's eval-cap fix was a silent no-op; the cap was applied twice
- **Symptom (2026-08-16).** The first `single_model` xlam run logged
  `eval set built: 800 examples (available in the held-out split: 1000)` — the loader honoured the new
  1,000-row cap, then the eval set was truncated back to 800 anyway. B282 was recorded as fixed and
  the observable behaviour had not changed at all.
- **Root cause, two independent faults.**
  1. `eval_size_target` (default 800) is applied at **two** places: once as the loader's `max_test`,
     and again as `build_eval_set(target=...)`. B282 only changed the first. For a curated benchmark
     the loader has *already* applied `_EVAL_SIZE_CAP`, so the split IS the target and re-applying
     `eval_size_target` can only shrink it.
  2. The edit that was supposed to fix this targeted a source string **that did not exist in the
     file**. `str.replace` with no match returns the string unchanged, and nothing asserted the match,
     so the "fix" shipped as a no-op and looked applied in the diff-free sense that nothing broke.
- **Fix.** With `SLM_BENCHMARK_TASK` set, pass `len(test_examples)` as the target; `eval_size_target`
  now only governs the autonomous path, where the orchestrator genuinely chooses the eval size.
  `tests/cold_start/test_eval_size_cap.py` asserts the **behaviour** (7 tests), not the source text,
  which is what would have caught the no-op.
- **Lesson.** A string-replace edit with no verification is not a fix. Assert the new behaviour, or at
  minimum re-read the file — a passing test suite says nothing about an edit that never landed.
- **Status:** 🟢 fixed. Runs before 2026-08-16 are on 800 rows.

## B289 — a GGUF that loads but decodes to nothing was scored as a real 0.0000
> **⚠ ROOT CAUSE CORRECTED — read [B290](#b290--training-taught-the-model-to-emit-a-think-block-that-inference-never-pre-filled) first.**
> This entry originally concluded that the merge/quantize path was intermittently corrupting artifacts.
> That was **wrong**, and the reasoning error is instructive enough to keep on the record. I argued
> "final `eval_loss` was 0.1221, and a model at that loss cannot emit `</tool_call>` on 800/800 rows."
> Teacher-forced eval loss is computed with the gold prefix supplied at every position, so it says
> nothing about what the model emits *first* when generating from the prompt alone — which is exactly
> the failure mode. The real cause was a train/serve prefix skew (B290), and it was deterministic, not
> intermittent; the apparent randomness was hyperparameter-dependent severity.
> The fix below is still worth having — a load-only check genuinely cannot tell a working artifact from
> a broken one — but it is a **backstop, not the fix for these scores**.

- **Symptom (2026-08-16, `slm-xlam-single-l40s-38561204`).** Iteration 3 scored **0.0000
  (800/800 failures)** and iteration 4 **0.0887 (729/800)**, while iteration 1 scored **0.8137**. Every
  prediction in the collapsed iterations was the bare string `</tool_call>`.
- **Why this is not a data or training problem.** Three independent lines of evidence:
  1. **The data was byte-identical.** Iteration 3's "rebuild" produced `dataset_v2.jsonl` with
     `novel_rows: 0, novel_fraction: 0.0` — synthesis yielded nothing and the file was a copy of v1.
     Iteration 3 also ran the `[carry-fwd best]` config. Same data, same config, 0.8137 → 0.0000.
  2. **Training converged normally.** Final `eval_loss` 0.1218 (iter 1) vs **0.1221** (iter 3) vs
     0.1220 (iter 5); `train_loss` 0.3228 vs 0.3249. A model at loss 0.1221 cannot emit `</tool_call>`
     on 800/800 rows.
  3. **The adapters were healthy.** Decoding the safetensors directly:
     `iter1 L2=38.890 max|w|=0.3621`, `iter3 L2=38.641 max|w|=0.4281`, `iter5 L2=38.714
     max|w|=0.3540`, **zero non-finite tensors in any of them**. Iteration 3's adapter is
     statistically indistinguishable from the two that scored 0.81–0.82.
  Each iteration also wrote its own content-addressed GGUF (`3bf783c8d6bd`, `e0e6c2a57233`,
  `0b7cc0dcdbe8`, …), so it was not stale-artifact reuse either. The corruption is introduced in
  **merge → quantize → llama-cpp decode**, and it is intermittent.
- **Root cause of the *scoring* failure.** `validate_and_record_gguf` only ever asked llama.cpp to
  **open** the file. It loaded every tensor, compared sizes, hashed the bytes — and never generated a
  single token. An artifact that loads cleanly and decodes to garbage passed validation, so the eval
  ran and its 0.0000 was recorded as a measurement.
- **The expensive part was not the wasted iteration.** The fabricated 0.0000 entered the agent loop as
  evidence, and the orchestrator reasoned from it. Its iteration-4 hypothesis reads: *"the synthesize
  data_rebuild on v2 catastrophically collapsed ALL buckets to 0.000 … consistent with a structural /
  format corruption in the synthesized rows"* — about a dataset containing **zero** synthesized rows.
  A silent infrastructure fault was laundered into a confident false causal story about the data, and
  every subsequent hypothesis inherited it. That is why this run was aborted and resubmitted rather
  than allowed to finish: rollback protects the *best model*, but nothing protects the *reasoning*.
- **Fix.** Two parts:
  1. `_smoke_test_generation` in `training/quantize.py`: after load, greedy-decode 16 tokens from
     `"Hello"` and reject output that is empty or contains no alphanumeric character once XML-ish tags
     and `<|...|>` special tokens are stripped. Raises the new `GgufDegenerateOutputError`.
  2. `_build_or_reuse_gguf` rebuilds **once** on that error. The corruption is transient, so one
     rebuild recovers the iteration; a second degenerate build raises
     `QuantizationInfrastructureError`, which by existing policy stops the run rather than scoring it.
     Kept as a separate exception class deliberately — subclassing `QuantizationInfrastructureError`
     would be caught by the `except … : raise` upstream and silently disable the retry.
- **Not yet root-caused.** *Why* the merge/quantize path intermittently produces a corrupt artifact is
  still open. Candidates: non-determinism in Unsloth's `save_pretrained_merged` under memory pressure,
  or the llama.cpp conversion step. The fix makes the fault **loud and recoverable** instead of
  silent and score-shaped, which is the property that matters for trusting the loop's numbers.
- **Related gap.** Per-row eval predictions are not persisted, so this could only be diagnosed from the
  3-row sample display that happens to be printed. Worth fixing separately.
- **⚠ THE FIX WAS WRONG AND HAS BEEN DOWNGRADED (2026-08-16, same day).** As a fatal gate this check
  killed **three healthy runs** — `38569605` (routerbench), `38569606` (ner_bc5cdr, already four
  iterations in), `38569608` (calendar_json) — all with the identical message: base `Qwen/Qwen3-0.6B`
  Q4_K_M "decoded to degenerate output `'////////////////////////////////'` for the prompt 'Hello'".
  Two errors compounded:
  1. **The probe was unrepresentative.** It sent the raw string `Hello`, asking the model to *continue*
     an unformatted string — never how it is used. A 0.6B base model emits junk for that. The prompt is
     now rendered through the model's chat template (`base_model` is threaded in for this).
  2. **The severity was unjustified.** Once B290 explained the 0.0000s, this check's motivating evidence
     was gone: it had produced three false positives and zero true positives. A heuristic that can end a
     multi-hour run needs far more certainty than that. It now **prints a warning and returns**; the
     eval score is the arbiter (a genuinely broken artifact scores near zero) and rollback is the remedy.
  The rebuild-once retry and `GgufDegenerateOutputError` were removed with it — there is nothing to
  recover from when nothing fails. `validate_and_record_gguf` still **raises on a failed load**, which
  is the real gate and was never the problem.
- **Lesson.** A new fatal check is a new failure mode. This one was added to protect the integrity of
  scores and instead destroyed three runs' worth of compute, for a fault that turned out not to exist.
  Warn first; escalate to fatal only once there is a confirmed true positive.
- **Status:** 🟢 downgraded to a warning; the "intermittent corruption" it was built for is now believed
  to have been B290 all along, so ⚪ no known underlying corruption remains.

## B290 — training taught the model to emit a `<think>` block that inference never pre-filled
- **Symptom (2026-08-16, `slm-xlam-single-l40s-38561204` and `-38565344`).** Fine-tuned scores on
  xlam_bfcl scattered across **0.0000, 0.0887, 0.3010, 0.6120, 0.7887, 0.8137** while the **untrained
  baseline scored a clean 0.8010**. The loop kept concluding — correctly, given its numbers — that
  *"Best this iteration is the ZERO-SHOT base model; fine-tuning did not improve on it."*
- **The tell, and the control that settles it.** Sample predictions, same three eval rows:
  ```
  BASELINE  (no adapter):  [{"name": "create_histogram", "arguments": {...}}]      <- clean
  ITERATION 1 (fine-tuned): </tool_call> </tool_call> [{"arguments": {...}}]       <- two stray tags
  ```
  Two tag tokens, *then* valid JSON, on **every** row. The parser salvages the rows where the JSON
  survives and fails the rows where the model stops after the tags — which is the entire spread of
  scores above, from 0.0000 (stopped every time) to 0.6120 (recovered most of the time).
- **Root cause.** `FastLanguageModel.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")` does not load that
  repo. It silently redirects to `unsloth/qwen3-4b-instruct-2507-unsloth-bnb-4bit` and returns **that
  mirror's tokenizer**, whose chat template applies the hybrid-Qwen3 think-block convention to a
  thinking-free checkpoint. Verified directly against the cached tokenizers:
  ```
  official Qwen/Qwen3-4B-Instruct-2507  -> '<|im_start|>assistant\nANSWER<|im_end|>\n'
  unsloth mirror (what training loads)  -> '<|im_start|>assistant\n<think>\n\n</think>\n\nANSWER<|im_end|>\n'
  ```
  `_build_completion_only_rows` builds the prompt with `add_generation_prompt=True` (which yields the
  bare prefix even under the mirror's template) and the full text with the assistant message. So the
  `<think>\n\n</think>\n\n` lands **inside `completion_mask`** — the model was explicitly *supervised*
  to emit it. Inference, correctly following the official template, does not pre-fill it, so those
  supervised tokens come out as the first tokens of the answer.
  The tags surface in logs as `</tool_call>` rather than `</think>` because llama.cpp renders those
  GGUF special-token IDs under different names; cosmetic, and it sent me down the wrong path for hours.
- **Why the existing guard missed it.** `_build_completion_only_rows` *does* assert the prompt is a
  token prefix of the full turn — but both sides use the same (wrong) template, so it passed. Nothing
  compared either against the **inference** prompt. That is now `_assert_train_serve_prefix_alignment`.
- **Why it went to the phone, not just our harness.** `merge_for_quantization` pins the **official**
  base (`[merge] pinning merge base to 'Qwen/Qwen3-4B-Instruct-2507' (adapter recorded 'unsloth/...')`),
  so the shipped GGUF is served under the official template. Teaching *inference* to send Unsloth's
  prefix would have hidden the skew in our eval while shipping a model that misbehaves in deployment.
  Hence the fix pins the **served** template for training, not the other way round.
- **Fix.** In `training/lora_trainer.py`:
  1. `_pin_serving_chat_template` — after loading, replace the tokenizer's `chat_template` with the
     official base model's, so the training target and the deployment contract are the same object.
  2. `_assert_train_serve_prefix_alignment` — before training starts, render the training text and
     require the inference prompt to be a strict prefix with **nothing** between it and the answer.
     Raises rather than warning: an hour of GPU time producing a silently crippled adapter is worse
     than a fast, legible failure.
  `tests/training/test_train_eval_prefix_alignment.py` (19 tests) covers both, including two that run
  against the **real cached vendor tokenizers** — one asserting the official templates satisfy the
  invariant, one asserting the Unsloth mirror violates it, so if upstream ever fixes their template we
  find out instead of carrying the workaround forever.
- **Scope.** Only `Qwen/Qwen3-4B-Instruct-2507` was affected: it is the sole model with a thinking-free
  official template, so it is the only one where `_qwen_no_think_prompt` omits the block that Unsloth's
  mirror inserts. The hybrid Qwen3 tiers (0.6B / 1.7B / 8B) pre-fill it on both sides and were always
  aligned — so previously reported tier-0/1/2 numbers stand, and any run that selected the 4B Instruct
  model should be treated as measuring the base model rather than fine-tuning.
- **Lesson.** Two prompt builders can each be individually correct and still disagree, when a dependency
  swaps the template out from under one of them. The invariant worth asserting is not "is this string
  right" but "does the text training produces begin with the text inference sends".
- **Status:** 🟢 fixed.

---

## xlam `single_model` audit — 2026-08-17 (B291–B298)

Source: `slm-xlam-single3-l40s-38566712` (16 iterations, 7h42m, best `ast_arg_match` 0.8530 vs an
0.8010 untrained baseline, terminated on stagnation). Full write-up:
[Evan's Notes 08-17b-xlam-single-debug.md](Evan's%20Notes/08-17b-xlam-single-debug.md).

The run's own reports were the defect. Six of eight data rebuilds announced 250–500 rows of synthesis
and produced **zero**, silently; both mining rounds rejected every candidate for a train/test overlap
the loader manufactured itself; and the two canonical source repositories were discovered and then
dropped unprobed. The orchestrator read all of that as *"ruling out a simple data-thinness
explanation"* and spent thirteen iterations on hyperparameters. Regression tests for all of it:
`tests/test_xlam_single_debug_findings.py` (29).

## B291 — `synthesize` was a silent no-op on every format-bound task, and the exact verifiers had never run
- **Symptom (2026-08-17).** On `xlam_bfcl` the log announced synthesis six times and the curriculum
  never contained a single generated row. Every dataset report read
  `Provenance: {'train_anchor': 3235}`; every `DATA REBUILD kinds` line read `resample-fill=3235`
  and nothing else. The teacher endpoint was reachable and logged as such on the line directly above
  each announcement. 2,250 rows requested, 0 produced, no error anywhere.

  | iter | announced | produced |
  |---|---|---|
  | 1 / 3 / 7 / 9 / 11 / 14 | 500 / 400 / 350 / 450 / 300 / 250 | 0 / 0 / 0 / 0 / 0 / 0 |

- **Root cause.** `data.curriculum.synthesize_examples` dispatches on task type and ends in a bare
  `return []`. It handles `("classification", "NER")` and `_GENERATION_FAMILY`, and
  **`function_call` was in neither**. `_GENERATION_FAMILY` was
  `{math_reasoning, code_generation, generation, multilingual, structured_extraction}` — `diff` was
  missing too. So the whole `synthesize` strategy was dead on every format-bound task.
- **The larger consequence.** `curate._verifier_for` correctly returns
  `data/synth_verifiers.py:verify_function_call_row` for `function_call`, and it was being passed to
  a function that returned before using it. So `verify_function_call_row` and `verify_calendar_row`
  — the entire exact-verifier subsystem, and the substance of the 08-18 note — **had never once
  executed in production**. `_synthesize_new_correct` was plainly written for these tasks: it pins
  `tools` from the anchor, a field only a function-calling row has.
- **Scope.** `xlam_bfcl` and `calendar_json` (both `function_call`) and any `diff` task. Every
  measurement of synthetic-data value on a format-bound task was taken on a curriculum containing
  zero synthetic rows, so the verdict in
  [08-16-extraction-collapse-verdict.md](Evan's%20Notes/08-16-extraction-collapse-verdict.md) does
  not apply to them either way.
- **Fix.** `function_call` and `diff` added to `_GENERATION_FAMILY`; the fallthrough now logs
  `NO SYNTHESIS PATH for task_type=... — this is a dispatch gap, not a generation failure` instead of
  returning silently. `test_every_registered_task_type_can_synthesize` asserts every task type in
  `TASK_METRIC_NAMES` has a synthesis path, so a newly registered task cannot reintroduce this.
  Verified end to end against a stub teacher: generate → exact programmatic verify → teacher answer
  verify → keep, with `tools` pinned from the anchor and undeclared-function rows dropped.
- **Lesson.** A dispatch table with a silent default is a feature switch nobody can see is off. The
  cost was not the missing rows — it was that the orchestrator drew a conclusion from their absence.
- **Status:** 🟢 fixed.

## B292 — (folded into B291) the `allocation_fallbacks` reason was a hardcoded guess
- When synthesis produced nothing, curate recorded
  `"synthesis produced no rows (endpoint unavailable or cheap mode)"` — a cause asserted without
  checking either condition, and false on every one of the six occurrences above. It cost an hour of
  looking at vLLM. Now it points at the `[synth]` lines rather than naming a cause, and a rebuild
  that generates nothing logs `⚠ SYNTHESIS PRODUCED 0 ROWS` explicitly.
- **Status:** 🟢 fixed.

## B293 — agentic discovery rejected single-split repos for a train/test overlap it created itself
- **Symptom (2026-08-17).** Every loadable xLAM mirror was rejected with
  `normalized train/test text overlap (80 rows)`. **80 is exactly `max_test`** — i.e. *every* test
  row overlapped, which is the signature of a tautology rather than of contamination.
- **Root cause.** `_mapped_split_names` resolves `test_split` back to the **train** split when a repo
  has no test/validation split, and `_materialize_from_mapping` then loaded `train[:max_train]` and
  `train[:80]` — both from the front, so test ⊂ train by construction.
  `_validate_discovered_splits` has **zero tolerance** for overlap and rejected the source. The
  Stage-0 benchmark path *strips* train-side overlap before checking; agentic discovery got the strict
  check without the cleanup step.
- **Why it matters.** Most instruction-tuning corpora on the Hub ship one split, so this was a
  guaranteed rejection for the common case — including for `Salesforce/xlam-function-calling-60k`
  itself, which is train-only.
- **Fix.** When the mapper collapses test onto train, take a **disjoint** window
  (`train[:300]` and `train[300:380]`), so the integrity check measures real contamination again. A
  genuinely separate test split is still read from the front.
- **Status:** 🟢 fixed.

## B294 — the two canonical source repositories were discovered, logged, and never probed
- **Symptom (2026-08-17).** Exa returned eight candidates, with
  `Salesforce/xlam-function-calling-60k` and `gorilla-llm/Berkeley-Function-Calling-Leaderboard` —
  the authoritative sources for the task — at positions **7 and 8**. `_discover_worker` probes
  `candidates[:6]`. Both were dropped untouched while six broken community mirrors consumed the whole
  budget. The log printed `candidates[:8]`, implying all eight had been considered.
- **Root cause.** Per-query hit lists were **concatenated**, so the first query's entire result set
  outranked every later query's best hit; the canonical repos came from the third query.
- **Fix.** Round-robin interleave across queries, so each query's top hit lands near the front. The
  log now states both numbers: `Exa found 8 candidate HF dataset(s); probing 6`.
- **Status:** 🟢 fixed.

## B295 — the accuracy chart drew one flat threshold line, and lowered goals were never recorded
- **Symptom.** `iterate_node` both lowers the goal (capacity-limited failures, down to
  `initial_stop_threshold`) and raises it (stretch goal). `run_graphics._plot_accuracy` drew the
  **final** `stop_threshold` as a single `axhline`, so every earlier iteration was shown as having
  been held to a bar that did not yet exist — and on a run that lowered its goal, that line sits
  *below* iterations the loop judged as failures.
- **Compounding.** Only raises were audited (`threshold_raises`). A lower overwrote
  `state["stop_threshold"]` and left one log line, so the goal a run was actually held to was
  unrecoverable from artifacts.
- **Fix.** `evaluate_node` stamps the in-force threshold on each DAG node (it runs before
  `iterate_node`, so the value is the one this score was judged against); `iterate_node` appends to a
  new `state["threshold_lowers"]`, persisted to `scores.json`; the chart draws a **step** line and
  annotates each change with ▲/▼ and the new value. Records predating the field carry forward, so the
  line is never discontinuous.
- **Status:** 🟢 fixed.

## B296 — every open-ended failure was reported as the same constant, and the orchestrator reasoned on it
- **Symptom (2026-08-17).** For any task that is not classification or NER,
  `build_test_report`'s confusion pairs collapse to the single literal
  `gold_verifier → incorrect`, whose count is the failure count the orchestrator already has.
  Nothing populated `error_type` for function calling, so across a dozen iterations the orchestrator
  wrote hypotheses like *"the dominant confusion gold_verifier->incorrect (147) essentially unchanged
  since iter2"* — paragraphs of reasoning about a constant, used as evidence.
- **Fix.** `eval/scorers/function_call.py:failure_category` splits failures into categories the
  scorer already computes, which point at different interventions:
  `unparseable_output` (format / chat-template), `undeclared_function` (prompt or unwinnable row),
  `wrong_function` (tool selection), `wrong_call_count` (parallel-call handling),
  `wrong_arguments` (argument extraction). `build_test_report` already reads `error_type`, so it
  flows through unchanged.
- **Status:** 🟢 fixed.

## B297 — no benchmark alias for `xlam_bfcl` / `calendar_json`, so the local corpus is unreachable
- **Symptom.** `eval_setup` loads `train[:3250]` of `Salesforce/xlam-function-calling-60k` (from
  `curriculum_size_target × 0.65`), and `state["train_examples"]` is never re-sliced afterwards. The
  remaining ~57,000 rows sit in the local HF cache, addressable by a one-line change to the split
  expression, and are **completely unreachable by the loop**. Instead `acquire` pays Exa + Claude to
  rediscover mirrors of the same corpus, which then die at validation (B293/B294).
- **Root cause.** `_BENCHMARK_ALIASES` in `data/loaders/web_acquire.py` has no `xlam`/`bfcl` entry,
  so `load_benchmark_dataset` — mining's free Stage-0 — cannot resolve the benchmark the task is
  *about*, and mining falls straight through to paid discovery. `calendar_json` has the same gap.
  The `routerbench` entry a few lines above carries a comment describing this exact failure being
  fixed for that task (B259); xlam and calendar were never added.
- **Why it is the highest-value open item.** With B291/B293/B294 fixed, this is the only remaining
  reason a run on these tasks cannot add real data. A `no_novelty` rebuild trains a full iteration on
  a content-identical curriculum, and this run spent both of its `acquire` iterations that way.
- **Proposed fix.** Register `xlam_bfcl` and `calendar_json` in `_BENCHMARK_ALIASES` routing to their
  canonical loaders, and give the loaders an **offset** so mining serves rows the pool has not seen
  (the eval firewall and the per-row dedupe against `seen` already guarantee correctness). Costs no
  provider calls at all.
- **Status:** 🔴 open.

## B298 — an eval row the small model gets right and the large model gets wrong is bucketed as `hard`
- **Symptom.** `test_agent.label_difficulty` builds the difficulty gradient from two zero-shot
  probes. Three cases are as documented (both right → easy, only large right → medium, both wrong →
  hard); the fourth — **small right, large wrong** — falls through the `else` into `hard`. Nothing
  about such a row is hard, and `hard` is the bucket the orchestrator weights most heavily.
- **Assessment.** A real logical flaw, but the population is small (it requires the larger model to
  fail where the smaller succeeds) and it never made this run take a wrong turn. Not fixed, because
  the right answer is a judgement call: a fourth `inconsistent` bucket is more honest than folding it
  into either neighbour, and that changes the report shape the orchestrator prompt depends on.
- **Related.** Two documentation-level facts worth recording while here: the probes run at **BF16**,
  not at max/min quantisation (quant siblings are deduped before ranking), and the buckets are
  computed **once** at cold start and frozen with the eval set. `plan["difficulty_buckets"]` is
  normalised and stored but never read back to sample rows with, and no training row is ever tagged
  with an eval difficulty — so `difficulty_composition` always reports `"unassigned"`.
- **Status:** ⚪ design gap.

## B299 — quality control was a silent no-op for four of the eight benchmark tasks
- **Symptom (2026-08-18).** `apply_quality_controls` was one `if task_type == ...` chain ending in
  `else: return dataset`. Measured across the whole suite, half of it received no filtering at all
  and nothing in any log said so:

| task | `task_type` | what happened |
|---|---|---|
| `xlam_bfcl`, `calendar_json` | `function_call` | **no QC at all** — no branch matched, so the `else` returned the dataset untouched |
| `gsm8k` | `math_reasoning` | **no effective QC** — entered its branch, then filtered length and near-duplicates on a `"prompt"` key its rows do not carry |
| `dialogsum` | `generation` | same as gsm8k; a 100,000-character row survived and nothing was logged |
| `clinc150`, `routerbench`, `proactive_listening`, `ner_bc5cdr` | `classification` / `NER` | worked as documented |

- **Root cause.** Two failures of the same abstraction. The `else` branch made "this task was never
  considered" indistinguishable from "this task chose not to deduplicate"; and the field a step
  filters on was *guessed from the channel* rather than stated by the task, so the generation-family
  branch admitted a row on `("text", "answer")` and then measured it on `"prompt"`. Every row passed
  the gate, none was measurable, and the step reported nothing because it removed nothing.
- **Compounding.** The two tasks that got no QC at all are the two whose gold is a JSON payload, so
  a row whose own gold answer does not parse trained the model to emit something the scorer marks
  wrong no matter what it predicts — the case QC would most obviously have caught.
- **Fix.** Quality control is now a list of named steps each task composes explicitly
  (`TaskSpec.quality_controls`, `data/quality_controls.py`). An empty tuple is a legal, visible
  choice; falling through is impossible because there is no branch. Each step is told which field to
  filter on, and `length_outliers` **logs loudly and skips** when no row carries that field instead
  of passing silently — the exact shape that hid this for gsm8k and dialogsum. `xlam_bfcl` and
  `calendar_json` additionally gained `valid_json_answer`.
- **Status:** 🟢 fixed. Covered by `tests/data/test_quality_controls_per_task.py`.

## B300 — NER `extract_predictions` returned `[]` both for "no entities" and "did not parse"
- **Symptom.** `eval/scorers/ner.py::extract_predictions` returned an empty list when the reply
  parsed to zero spans **and** when the reply was prose that never parsed at all. A model emitting
  paragraphs was therefore scored identically to a model that correctly predicted "this passage
  contains no entities", and span-F1 alone could not tell a content problem from a format one.
- **Why it mattered here.** BC5CDR is the task where a near-zero baseline is *expected* (the base
  model cannot produce the output contract), so the one number that distinguishes "cannot format"
  from "cannot extract" was the number being discarded.
- **Fix.** `extract_predictions` now returns `None` for a parse failure and a list — possibly empty
  — for anything that parsed. `score` counts non-`None` predictions as `format_valid` and scores
  `None` as an empty prediction set, so content and format are reported separately for every
  iteration, and `failure_category_of` reports `unparseable_output` for the `None` case.
- **Status:** 🟢 fixed.

## B301 — the classification scorer reported `metric="macro_f1"` while computing a minority-class F1
- **Symptom.** `classification.score` chose its headline number implicitly by counting classes —
  more than two chose macro-F1, otherwise the minority class — but returned the literal string
  `"macro_f1"` either way. `routerbench` and `proactive_listening` are both binary, so both reported
  a **minority-class F1 under a macro-F1 label** for their entire recorded history, in every log
  line, DAG node, `scores.json` and chart.
- **Why it is not cosmetic.** The two metrics are not close on an imbalanced binary task: macro-F1
  over two classes is flattered by a model that always predicts the majority, which is exactly the
  degenerate behaviour the minority-class F1 was chosen to expose. Anyone comparing a RouterBench
  number against CLINC150's genuine 151-way macro-F1 was comparing two different quantities.
- **Fix.** The choice is no longer inferred. `score_macro_f1` and `score_minority_f1` are separate
  functions, each returning its own metric name, and the task names which one it wants
  (`TaskSpec.metric_name`, surfaced by `eval.harness.task_metric_name`). `routerbench` and
  `proactive_listening` declare `minority_f1`; `clinc150` declares `macro_f1`.
- **Status:** 🟢 fixed. **Scores recorded before 2026-08-19 for those two tasks are correctly
  valued and wrongly labelled** — the number did not change, only its name.

## B302 — eight refactor bugs that `import` could not see
- **Symptom (2026-08-19).** The task-registry refactor landed with eight real defects. Four of them
  broke every run outright. They are grouped as one entry because they are one bug class, not eight
  unrelated mistakes: every single one was a **runtime** failure invisible to module import — a
  lazily imported name, a name left unbound on one branch, or a keyword argument the callee had
  stopped accepting.

| # | Site | Effect |
|---|---|---|
| 1 | `checkpoint.py` imported a deleted constant | `runtime_config_snapshot()` raised at module scope in the runner — **every run died before the graph was built** |
| 2 | two encoders read removed `EvalSet` fields | **no checkpoint could be written and no run could resume** |
| 3 | `curate.py` used `hashlib` after its import was removed | the **eval firewall raised on the first row it blocked** — the safety mechanism killing the run at the moment it caught a leak |
| 4 | `web_acquire.py` read `_spec` after the assignment moved into another function | rung 2 of the mining ladder raised on entry |
| 5 | the same for `_closed_label_space` | the B259 per-row label guard could not run |
| 6 | `annotate_cot(task=...)` after the parameter was dropped | `TypeError` on every gsm8k rebuild once the CoT teacher was reachable |
| 7 | `fallback_data_rebuild_plan(score=, mining_available=)` | `TypeError` on the orchestrator-failure route — the path that exists precisely so a bad reply cannot stop the run |
| 8 | `accept(discovered[0], ...)` unwrapped one level too far | rung 2 could never accept a row, so every discovery round reported `no_novelty` and mining retired itself after two rounds while a usable dataset went unused |

- **Root cause.** Moving code between functions during a large refactor. A module-level import smoke
  test — the check most projects rely on — would have caught **none** of these, because a lazy
  import inside a function body, an undefined name on an unexercised branch, and a keyword mismatch
  are all deferred to call time. The 1,317-test suite caught them only where a test happened to
  exercise the branch.
- **Fix.** `scripts/check_unresolved_names.py` is a static AST scan for exactly these two shapes —
  `Name` loads that resolve to nothing bound in an enclosing scope, an import, or a builtin; and
  calls to a same-file function passing a keyword it does not accept. It is deliberately
  conservative (a scan that cries wolf gets switched off), self-tested against the shapes above, and
  currently reports zero across the production packages. Run it before committing any refactor that
  moves code between functions.
- **Status:** 🟢 fixed.

## B303 — `calendar_json`'s mining source is a placeholder that has never been exercised
- **Symptom.** `calendar_json`'s `TaskSpec.mining_sources` declares `TOPv2/reminder`, but TOPv2 ships
  as a GitHub tarball rather than a Hugging Face dataset, and no one has verified that its loader
  answers a request for a *larger* head slice the way rung 1 of the mining ladder requires.
- **Consequence if true.** `_reread_known_sources` asks each source for `consumed + 4×want` rows and
  marks it exhausted when fewer come back. A loader that cannot grow its slice returns the same rows
  every time and is marked exhausted on the first rebuild, so rung 1 is a no-op for calendar and the
  task falls straight through to paid web discovery — the same failure mode B297 fixed for xlam.
- **Why it is not yet fixed.** It needs a real load against the actual corpus, not a code change:
  the honest options are to confirm the loader supports a growing slice, to point the spec at a
  hub-hosted calendar/SGD corpus that does, or to declare `mining_sources=()` so the ladder is
  visibly empty rather than apparently populated.
- **Status:** 🔴 open.

## B304 — `allow_paid_discovery=True` on the two tasks whose labels no other corpus carries
- **Symptom.** All eight tasks set `allow_paid_discovery=True`. For `routerbench` and
  `proactive_listening` the label is **derived**, not observed: RouterBench's `local` means "a small
  model answered this correctly", and proactive listening's label is an interruption judgement. No
  other corpus on the hub carries either annotation natively, so a discovery round for those two can
  only find text whose labels an LLM must invent — which is precisely what B259 was opened for.
- **Mitigations already in place.** The per-row closed-label filter, the pinned label vocabulary, and
  `_llm_map_dataset`'s explicit prohibition on introducing a class should all catch a fabricated
  label. Discovery is also only reached once every known source is exhausted, and is retired after
  `MAX_FAILED_DISCOVERY_ROUNDS` (2) fruitless rounds.
- **Why it is recorded anyway.** Those mitigations bound the damage; they do not make the round
  worth paying for. The decision worth making explicitly is whether these two tasks should simply
  declare `allow_paid_discovery=False`, which states in the registry the thing the guards are
  currently discovering at runtime.
- **Status:** ⚪ design gap.

## B305 — `agent/data_sizing.py` computes a curriculum target nothing reads
- **Symptom.** The per-tier sizing formula (`clamp(5000 × (0.5 + novelty) × size_factor, 5000,
  25000)`, ratcheted so it never falls below the previous tier's) is intact, tested, and **has no
  production call site**: `resize_curriculum_for_tier` is referenced only by
  `tests/`, a comment in `task_analysis.py`, and this documentation set.
- **Why.** The curriculum is now cumulative and has no target. Cold start loads
  `TaskSpec.initial_train_cap` (a flat 5,000 for all eight tasks) and every rebuild adds to it;
  `MIN_CURRICULUM_ROWS = 500` is a viability floor, not a target. `curriculum_size_target` survives
  in state and is still read by the **autonomous** acquisition path in `eval_setup`
  (`gold ≈ 0.65 × curriculum_size_target`), which is why the module cannot simply be deleted without
  a decision about that path.
- **What to decide.** Either delete `data_sizing` and give the autonomous path its own explicit
  size, or reinstate per-tier sizing as something the curated path actually consumes. Leaving a
  tested, documented formula that no run reads is the state most likely to be mistaken for behaviour
  — this documentation described it as live until 2026-08-19.
- **Status:** ⚪ design gap.

## B306 — synthesis was authorised by a number measured the wrong way, and never consulted

- **Symptom:** `surgical_synthesis` was offered as an intervention on every task at every score. The
  only evidence anyone had about whether the teacher could do the task was its ZERO-SHOT score from
  `measure_endpoint_baseline`, and nothing read that number before spending the teacher's budget.
- **Why the number was wrong:** measured zero-shot, so on a format-bound task it largely reports
  whether the teacher guessed our output contract. BC5CDR measures 0.1131 zero-shot and 0.7190 with
  five demonstrations. Reading 0.1131 as "the teacher cannot do biomedical NER" is a 6.4x error about
  a model that could do it all along and had simply not been told the conventions (B276).
- **Why it matters:** a teacher below ~0.80 on a task is not a source of training targets for it, it
  is a source of labelled noise, and a student trained on that output is capped near the teacher's
  error rate. The best result this project has measured (BC5CDR, 0.8098) came from a gold-only
  curriculum.
- **Fix:** `agent/teacher_fitness.py`. The teacher is scored FIVE-SHOT on the task's own eval set,
  through the task's own scorer, once at cold start. Below 0.80, `surgical_synthesis` is removed from
  the intervention menu for the whole run: the plan validator rewrites it to `mine_new_real`, the
  orchestrator prompt states the refusal and its measured reason, and `iterate` routes to a
  hyperparameter step when neither sub-strategy can add a row. An UNMEASURED teacher is refused too —
  defaulting to allowed would mean a run that skipped the gate silently regained synthesis.
  Demonstrations come from TRAIN only, so measuring cannot leak eval rows.
- **Status:** ✅ fixed 2026-08-19.

## B307 — the label verifier judged the label WORD, because it never saw the task

- **Symptom:** `verify_generated_labels` sent the class name, the utterance and a label definition,
  but not the task description. `verify_generated_answers` had been given the brief; the label path
  was missed.
- **Consequence:** the teacher judged plausibility against its own reading of the label string rather
  than against the task. This is the mechanism behind the RouterBench rejections (B267/B269), where a
  grade-school math problem was refused for the `local` class because "the utterance is a math
  problem, not a local query" — the label means "route to the local model", which a verifier that has
  not been told the task cannot know. About 70% of generated rows were discarded for the wrong reason.
- **Fix:** `task_description` is now a REQUIRED keyword on `verify_generated_labels` and is inserted
  at the top of the prompt. Required rather than optional-with-default deliberately: a default would
  let a future call site reintroduce exactly this bug silently.
- **Status:** ✅ fixed 2026-08-19.

## B308 — nothing watched across iterations, so a run that could not learn ran for 7h42m

- **Symptom:** run 38566712 completed eight data rebuilds. All eight added zero rows. The run
  continued for 7h42m and $1.34, and terminated on stagnation as though it had explored something.
- **Root cause:** every symptom was checked in isolation and each one is survivable once — a `0 novel`
  line, a quality-control drop, a mining round that finds nothing. The failure was the REPEAT, and no
  component held state across iterations to notice it. The orchestrator, which did see the history,
  read `0 novel` as evidence that data was not the problem and chose synthesis again.
- **Fix:** `agent/run_health.py`, called at the end of every `curate_node`. Raises `RunHealthError`
  on two consecutive rebuilds adding no rows, a QC drop ≥1,000 rows or ≥25% of the curriculum, two
  consecutive total verification wipeouts, three mining attempts that saw candidates and accepted
  none, or two data-load failures. Each threshold requires a repeat or a magnitude variance cannot
  explain, because a guard that fires on noise gets switched off. The message names the mechanism,
  and `_diagnose_empty` attributes an empty rebuild to a filter, a shortage, or a rejecting verifier
  from what that rebuild recorded.
- **Status:** ✅ fixed 2026-08-19.

## B309 — a cancelled run lost its entire report

- **Symptom:** the final report was top-level module code at the end of `tests/pipeline/run.py`. A
  `scancel`, a wall-clock stop, or an exception inside any report section meant losing every section
  after it — including the score trajectory and the curriculum ledger, which is the whole
  informational value of the run.
- **Fix:** the report is now `_report_body()`, invoked through an idempotent `_emit_final_report()`
  registered with `atexit`. It prints on a normal finish, an exception, a SIGTERM, or a wall-clock
  stop, and a failure inside it logs a traceback instead of truncating the report. Only SIGKILL is
  unrecoverable, and nothing in-process can help there.
- **Status:** ✅ fixed 2026-08-19.

## B310 — the only per-row diagnostic was difficulty, which says nothing about WHAT is failing

- **Symptom:** `difficulty.png` reported easy/medium/hard accuracy. Difficulty is defined by what the
  smallest and largest base models score zero-shot, so it answers "how hard were the rows it got
  wrong" — but `surgical_synthesis` targets CATEGORIES, and nothing plotted the categories.
- **Fix:** `test_report["outcome_breakdown"]` (`agent/nodes/test_agent.py`) records correct and failed
  counts per bucket — the gold class for a task with a closed label space, the task's own failure
  category otherwise — and `label_performance.png` plots them worst-first as stacked bars. Counts
  rather than rates, because a class at 50% on four rows and one at 50% on four hundred are the same
  rate and completely different problems, and choosing a target is precisely about telling them apart.
- **Status:** ✅ fixed 2026-08-19.

## B311 — four call sites and one test fixture still spoke the deleted channel vocabulary

- **Symptom:** verification run 38656655 died 27 minutes in, past the loader, the task brief and the
  teacher-fitness gate — the most expensive possible moment to discover a `TypeError`.
- **Faults, all from the 2026-08-18 channel removal:**
  - six call sites passed `task=` or `task_type=` to `eval.harness.run_eval`, whose signature had
    dropped it (the task now comes from the `EvalSet`, so the two can no longer disagree);
  - `interpolation._probe_model` passed `task_type=state["task_type"]` to `slm_train`, which takes
    `task` — and `state["task_type"]` no longer exists, so it was a `KeyError` *inside* a
    `try/except` that reported the probe as `f1=0.0`. Every interpolation probe scored zero;
  - `orchestrator_choice` read `state.get("task_type", "classification")` and passed it to
    `_benchmark_hint`, which looks the value up in the task registry. `"classification"` is not a
    task, so the lookup missed on every run and the model-choice prompt always got the generic
    fallback hint;
  - `tests/pipeline/run.py` imported `TASK_METRIC_NAMES`, deleted with the channels — 26 lines above
    the final report, so the crash also took the whole report;
  - `tests/cold_start/test_model_selection.py`'s state fixture still supplied `task_type` and no
    `task`, which is precisely why no test caught the probe bug: the fixture provided a key that
    production state does not have.
- **Why the suite was green:** every one of these is on a GPU-only path, and the one that was not sat
  behind a fixture carrying the stale field.
- **Fix:** all call sites corrected; the fixture now carries a real registry task name. More
  importantly, `scripts/check_unresolved_names.py` gained CROSS-MODULE checks — it now verifies that
  `from <project module> import <name>` names something that module defines, and that keyword
  arguments passed to an IMPORTED project function are ones it accepts. Run against the tree it
  immediately found two further instances nobody had noticed. `hardware_eval` and
  `tests/pipeline/run.py` were added to its default roots.
- **Status:** ✅ fixed 2026-08-19.

## B312 — the plan validator rejected its own output, so `data_rebuild` never once ran

- **Symptom:** verification run 38658213 completed five iterations. The orchestrator chose
  `data_rebuild` on every one of them. Not a single rebuild happened, the curriculum stayed at 4,954
  rows, and the score fell 0.789 → 0.690 across five hyperparameter steps nobody asked for.
- **Root cause:** `normalize_data_rebuild_plan` RETURNS a dict containing `hypothesis` and `task`,
  but `_PLAN_FIELDS` — the fields it ACCEPTS — did not include them. The function could not accept
  its own output. Two callers feed that output back: `curate_node` re-normalizes the plan `iterate`
  stored, and the orchestrator, shown a schema, nested `hypothesis` inside the plan object instead of
  leaving it at the top level. Every plan raised
  `unknown field(s) ['hypothesis', 'task']`.
- **Why it was invisible:** `iterate` catches a failed decision and falls back to the test-agent
  suggestion, which was `hyperparameter`. So the log read `using test-agent suggestion:
  hyperparameter` — indistinguishable from an orchestrator that had wanted hyperparameters. The
  intent was destroyed and the destruction was not reported.
- **Fix, three parts:**
  1. `_PLAN_FIELDS` accepts `hypothesis` and `task`; their values are ignored in favour of the
     authoritative arguments, except that a plan-carried `hypothesis` survives a re-normalize with no
     argument. A genuinely unknown field still raises. The function is now idempotent, and a
     round-trip test pins that.
  2. `iterate` logs `✗ ORCHESTRATOR PLAN REJECTED` whenever it downgrades a data-rebuild decision,
     naming the validation error.
  3. `run_health.record_rejected_data_plan` counts them and stops the run at two, because a rejection
     this systematic is a schema disagreement between the prompt and the validator, not a bad sample,
     and it will not fix itself.
- **Also fixed alongside:** `curate_node` re-normalized the plan without passing `synthesis_allowed`,
  so at the execution gate the flag defaulted to True. Since mining availability is re-derived there
  and can flip to False, a `mine_new_real` plan could be rewritten to `surgical_synthesis` behind a
  teacher that had FAILED its fitness gate — spending the teacher budget on exactly the output the
  gate exists to refuse.
- **Status:** ✅ fixed 2026-08-19.

## B313 — the teacher baseline measured 0.0000 because the task name was passed as the generator

- **Symptom:** 1,000 consecutive `endpoint baseline generation failed: 'str' object is not callable`
  lines, then `[baseline] reference ast_arg_match=0.0000`, then
  `[threshold] Qwen baseline 0.0000 → goal 0.8000 (floor 0.80) — FLOOR WON: the teacher scored below
  0.80`. The run then spent its life chasing 0.80 on the strength of a measurement that never happened.
- **Root cause:** `agent/nodes/cold_start/eval_setup.py` called
  `measure_endpoint_baseline(eval_set, task, log=print)`. The signature is
  `(eval_set, generate_fn=None, *, ...)` — the task is read from the `EvalSet`, so there is no task
  parameter — and the string landed in the `generate_fn` slot. Every row's generation raised, and each
  was swallowed by the deliberate one-bad-row-must-not-abort handler. Sibling of B311: the same refactor
  dropped `task` from `run_eval`, which is why several call sites made this mistake at once.
- **What made it expensive:** the teacher's real ability was already known, from a different code path,
  in the same log — the fitness gate measured 0.8250 five-shot on the same eval set minutes earlier.
  Had that gate consulted the baseline rather than measuring for itself, it would have refused
  synthetic data for this task on the strength of pure harness noise.
- **Fix, three parts, because fixing only the call site leaves the trap:**
  1. the call site drops the argument;
  2. `measure_endpoint_baseline` raises `TypeError` on a non-callable `generate_fn` before making a
     single call — that catches the whole family, since `generate_fn` is the second POSITIONAL
     parameter and anything a caller passes there believing it is a task lands silently;
  3. `_generate_all` counts failures and raises `BaselineGenerationError` above a 25% failure rate. A
     score built from nothing but failures describes the harness, not the model, and 0.0 from a broken
     harness is indistinguishable in a report from 0.0 from an incapable teacher.
- **Note on the static scan:** `check_unresolved_names.py` cannot catch this. A string in the
  `generate_fn` slot is a positional argument of legal arity, so it is only a fault given types the
  code does not declare. That is precisely why the runtime guards exist.
- **Status:** ✅ fixed 2026-08-19.

## B314 — the answer verifier was asked to check calls against a tools list it was never shown

- **Symptom:** on run 38661753, `verify_generated_answers` rejected 79 of 330 generated xlam rows (24%)
  and 21 of 108 on the next round (19%), with reasons like "Tool name 'calculate_distance' is not
  present in the provided tools list", "Tool names and arguments are invented and not from the provided
  tools list", and "Invented argument key 'is_id' not present in tool schemas".
- **Root cause:** the verification prompt contained the task description, five reference (request,
  answer) pairs, the generated request and the generated answer — and nothing else. The row's own
  `tools` field, which *defines* what a correct call is for that row, was never rendered. The teacher
  was asked whether a function call satisfies a request without being shown the functions, and answered
  anyway by inventing the missing context.
- **Why every one of those rejections was provably wrong:** each row had already passed the
  PROGRAMMATIC verifier (`verify_function_call_row`), which checks every call's name and every argument
  key against that row's own `tools` schema. A row cannot both pass that check and use a tool absent
  from its tools list. So each rejection discarded a valid row. `is_id` is a real xLAM convention that
  the teacher second-guessed from its own prior — B267/B269 in a subtler dress.
- **Fix:** `_row_context_block` renders every non-`_`, non-question, non-answer field of the row into
  the prompt as explicit context, with the instruction that anything appearing there is valid by
  definition and must not be rejected as unprovided. Built by EXCLUSION rather than from a per-task
  list of context fields, so a task that gains a field gets it shown automatically — silent omission is
  the failure mode being fixed. Applied to `verify_generated_labels` as well.
- **Consequence for prior conclusions:** the orchestrator concluded from this run that "teacher-generated
  argument rows carry label/format noise the model overfits to rather than genuine signal", having
  watched two synthesis rounds make the dominant confusion worse. That was measured through a filter
  discarding a quarter of its input for invented reasons, so it needs re-testing.
- **Status:** ✅ fixed 2026-08-19.

## B315 — mining over-delivered fourfold, and web-discovered sources were forgotten immediately

- **Symptom:** on run 38661753 a rebuild asked for 600 rows and added 2,399; the next asked for 800 and
  added 3,161. The curriculum went 4,954 → 10,701 in three rebuilds, and the orchestrator's `rows`
  field was effectively inoperative.
- **Root causes, two:**
  1. `_reread_known_sources` asked for `consumed + want * 4` — a 4x over-fetch hedging against
     deduplication losses that do not exist, since everything past `consumed` is novel by construction —
     and then returned the ENTIRE slice, leaving downstream dedup to discover the overlap.
  2. `_discover_new_source` returned rows and never registered the dataset anywhere. A web-discovered
     corpus was drained of whatever it happened to return in one pass and forgotten, so a later rebuild
     could not read more of it without paying to rediscover it.
- **What does NOT fix (1), and was tried first:** trimming the surplus after the fact. `consumed`
  records the slice depth READ, so discarding rows advances the pointer past rows that were never used
  and silently skips them for the rest of the run. Asking for the right amount is the fix, so the
  pointer and the rows agree.
- **Fix:** `MAX_MINED_ROWS_PER_REBUILD = 1000` caps one rebuild's contribution across all sources
  together; the re-read asks for exactly what it intends to keep and returns only rows past the previous
  high-water mark; discovered datasets are written into `source_progress` with `consumed` set to what
  was actually taken, so the next rebuild re-reads them from rung 1 at the right offset.
- **Why cap at all, when real gold rows are good:** growth has to stay legible. A rebuild that adds a
  few hundred rows is an experiment whose effect can be read off the next eval; one that adds three
  thousand changes curriculum size, training time and class balance at once and the score movement
  cannot be attributed to any of them. It also stops a 60,000-row corpus being drained in twenty
  iterations, after which the ladder falls through to synthesis with most of the corpus unread.
- **Status:** ✅ fixed 2026-08-19.

## B316 — a failed orchestrator decision was replaced by a different algorithm and reported as this one

- **Symptom:** run 38658213 logged `LLM call failed (...); using test-agent suggestion: hyperparameter`
  on five consecutive iterations and carried on. The trajectory looked like five deliberate tuning
  decisions; in fact the orchestrator had asked for `data_rebuild` every time and been overruled by a
  validator bug (B312).
- **Root cause:** `iterate` caught every exception from the orchestrator call and fell back to the
  test agent's `suggested_intervention`, or failing that to a score-band rule
  (`apply_iteration_policy`). The intent was resilience. The effect was that a broken decision became
  an ordinary-looking iteration.
- **Why the fallback was the wrong idea, not merely wrongly scoped:** a score-band rule is not a
  degraded version of this loop, it is a DIFFERENT algorithm. The loop is defined as the orchestrator
  choosing an intervention from evidence; substituting a rule and reporting the result under the same
  name makes the trajectory uninterpretable, because afterwards nobody can tell which iterations were
  reasoned and which were guessed. That ambiguity is what allowed B312 to run for a whole run while
  every surface reported normal operation.
- **Fix:** the fallback is gone. `iterate` logs a `✗ FATAL: could not obtain a usable orchestrator
  decision` block — naming the error, the iteration, the score, and (when the failure was a refused
  data plan) that the prompt and `normalize_data_rebuild_plan` disagree about the schema — and raises
  `OrchestratorDecisionError`. Billing/auth/quota errors still short-circuit through `raise_if_fatal`
  first, as before. A run that cannot obtain a decision has nothing worth reporting.
- **Removed as newly dead:** the `llm_decision is None` branch in `iterate` that called
  `fallback_data_rebuild_plan` (a second, score-derived source of plans alongside the orchestrator's —
  exactly the ambiguity above), and `run_health.record_rejected_data_plan` with its
  `MAX_REJECTED_DATA_PLANS` counter, which tripped on the SECOND refusal and can therefore never fire
  now that the first one raises. `fallback_data_rebuild_plan` itself remains, reached only from
  `curate`.
- **Status:** ✅ fixed 2026-08-20.

## B317 — `surgical_synthesis` on `ner_bc5cdr` was a guaranteed no-op

- **Symptom:** every synthesized BC5CDR row was rejected with `empty answer`, and ZERO verification
  prompts were sent. 100% of the generation budget for that task produced nothing, on every rebuild.
- **Root cause:** `verify_generated_answers` read the gold as `row.get("answer") or
  row.get("response")` and returned `False` before building a prompt when both were blank. A BC5CDR
  row carries its gold in `entities` and has no `answer` key at all. So the guard meant for a row with
  no gold fired on every row of a task whose gold simply lives elsewhere.
- **Why nobody noticed:** it fails as a plausible rejection, not an error — the log reads
  `teacher validated 0/N generated answer(s)`, which is indistinguishable from a teacher that judged
  the batch and disliked it. It also contradicted `docs/PIPELINE.md`, which documents the teacher pass
  as running for spans precisely because the substring verifier cannot catch a MISSED entity.
- **Fix:** `_gold_field(spec)` reads the gold key off the spec's `required_fields`, which is
  `(input, gold)` for all eight tasks — `("text", "entities")` for BC5CDR, `("text", "answer")` for
  the rest. `_render_gold` serializes a structured gold as JSON so the verifier sees the shape the
  scorer parses rather than a Python repr. The reference examples shown to the verifier render the
  same way, and the gold field is excluded from the row-context block so the answer is not also
  presented as context for itself.
- **Found by:** writing the B314 tests. The new entity-type vocabulary was unreachable in production
  for the one task it was written for, which is what surfaced the gap.
- **Status:** ✅ fixed 2026-08-20.

## B318 — 123 calendar gold rows contradicted their own utterance, and half of all gold was 09:00

- **Symptom:** `calendar_json` gold labelled "...on Tuesday at 6" as `09:00`, "...Saturday at 2" as
  `09:00`, "...tomorrow at 5" as `09:00`. 123 of the 1,608 train rows with an explicit stated time had
  gold that ignored it, and **51% of all 4,242 gold rows started at exactly 09:00**.
- **Root cause:** `_TIME_RE` requires a meridiem and `_TIME_24_RE` requires a colon, so a BARE hour
  ("at 5") matched neither. `_parse_time` returned `None`, which `resolve_datetime` cannot distinguish
  from "no time was stated at all" — and that case legitimately takes the documented 09:00 bare-date
  default. One return value was carrying two very different meanings.
- **Why it matters beyond the 123 rows:** the model was being TAUGHT to ignore explicit bare-hour
  times, and a curriculum in which half of all targets are 09:00 teaches "default to 9am" as the
  dominant prior on a task whose entire content is resolving times correctly.
- **Fix:** `_parse_time` gained a third outcome, `AMBIGUOUS_TIME`, for "a clock time IS stated but
  cannot be resolved". `resolve_datetime` drops those rows, which is the policy the module already
  documents for `_UNRESOLVABLE` ("before what?", "this week") and states in its own docstring:
  *"anything not understood with certainty must be dropped rather than guessed"*. A bare hour that an
  adjacent daypart disambiguates is now resolved rather than dropped — "at 7 tonight" is 19:00, not
  the generic 20:00 the daypart alone gave, and not 07:00.
- **Result:** contradictions 123 → 8, at a cost of 86 rows (4,242 → 4,156, 2%). The 8 remaining are
  malformed source text ("at 11:00an", "at 6:3 am"). Rows whose gold is 09:00 *and* which state a
  non-morning clock time: 15 of 4,156, most of them false positives in the checker ("game night" is
  not a time).
- **Rejected:** guessing PM for a bare afternoon-ish hour, which is what a calendar app does. This
  builds GOLD, and a plausible guess is the same class of error as the 09:00 default, just smaller.
- **Status:** ✅ fixed 2026-08-21.

## B319 — the calendar synth verifier checked JSON shape, never whether the answer was right

- **Symptom:** a generated row whose request said "remind me to call Bob tomorrow at 3pm" and whose
  answer said `09:00` the next day was ACCEPTED, with the reason
  `well-formed calendar event with coherent datetimes`.
- **Root cause:** `verify_calendar_row` checked form exhaustively — valid JSON, exactly one
  `calendar.events.insert`, `start`/`end` parse, `end > start`, 60-minute default duration, a summary
  that is not the request's imperative, and a date within ±2 years of the reference. It never asked
  whether the resolved instant matched the request. The ±2-year bound is far too loose to catch 3pm
  becoming 9am, or even tomorrow becoming next week.
- **Why that is the whole task:** `calendar_json` is scored by exact argument match, and its content
  is resolving a relative expression against a per-row reference instant. A verifier that skips that
  is checking everything except the thing being learned.
- **Fix:** `_expected_start` re-derives the instant from the request using `resolve_datetime` — the
  SAME grammar the loader used to build the gold, so verifier and gold cannot disagree about
  conventions — and the row is rejected on mismatch. This makes it a genuinely exact verifier: no
  teacher call, no judgement. When the grammar cannot resolve the request it returns `None` and the
  check ABSTAINS rather than rejecting, so the verifier's own limits never become row rejections; the
  teacher pass still sees the row.
- **Status:** ✅ fixed 2026-08-21.

## B320 — few-shot prompts concatenated N full instruction blocks with no delimiter

- **Symptom:** demonstrations made the teacher WORSE, on every task measured. `xlam_bfcl` scored
  0.8350 five-shot against 0.8680 zero-shot; `calendar_json` 0.7350 against 0.8368. `format_valid` was
  ~1.0 in all four measurements, so this was not the output contract. On `calendar_json` the depressed
  number fell below the 0.80 gate and **synthetic data was refused for a teacher that is actually
  above the bar**.
- **Root cause:** `build_training_turn` returns a COMPLETE prompt — task instruction, tool/label
  vocabulary, the request — because that is what fine-tuning shows the student. `_demo_block`
  concatenated k of them separated by a blank line, so the teacher received the entire instruction
  k+1 times with nothing marking where one example ended, which were already answered, or which one
  was the question. `calendar_json` shows the mechanism plainly: every block carries its own
  `Current date and time`, so six different reference instants arrived with nothing saying which
  governed the answer — on a task whose entire content is resolving against that instant.
- **Fix, two parts:**
  1. every demonstration is fenced with `### Solved example N of M`, and the real question is
     introduced by a header that explicitly says the examples are already answered, are shown only for
     format, and that any date/time/reference they mention must be ignored;
  2. the gate now measures BOTH k-shot and zero-shot and reads the BETTER, logging both and recording
     `best_shots`. The gate answers "can this teacher do this task", and the honest input is the best
     measurement available rather than one arbitrary prompt shape — a prompting defect must never
     again be able to masquerade as an incapable teacher.
- **Note:** the 0.1131 → 0.7190 BC5CDR few-shot gain that motivated five-shot prompting (B276) was
  measured through this same undelimited block. It held anyway because on BC5CDR the format signal
  dominated; the effect should be larger now, not smaller.
- **Status:** ✅ fixed 2026-08-21.

## B321 — calendar_json trained on TOPv2 and scored on SGD, so the metric measured the split

- **Symptom:** run 38732020 scored `ast_arg_match=0.0084` with `format_valid=1.0000` — 474 of 478 eval
  rows wrong, every prediction perfectly formed JSON. The same signature as the earlier 0.0000 on this
  task, which had been attributed to year resolution.
- **Root cause:** `load_calendar_json` returned `(TOPv2, SGD)` as `(train, test)`. The two corpora
  describe the same activity with structurally different requests:

  | | TRAIN (TOPv2, 4,156) | EVAL (SGD, 478) |
  |---|---|---|
  | `location` argument in gold | **0.0%** | **100.0%** |
  | quoted title in the request | 0.2% | 100.0% |
  | gold start == 09:00 | 50.0% | 1.7% |

  The metric is exact argument match, so being asked for a required `location` the model had never
  once seen in 4,156 training rows pinned the score near zero however capable the model was. The
  09:00 skew compounds it: TOPv2 teaches "default to 9am", SGD almost never wants it.
- **Why it looked like a model failure:** `format_valid` was a clean 1.0000, so every surface said the
  model had learned the output contract. It had. It had also learned a different task from the one it
  was being graded on.
- **Fix:** both splits are drawn from ONE pooled, shuffled, text-deduplicated TOPv2 + SGD population,
  so they are random samples of the same distribution by construction — 10.7% location-bearing in
  train against 10.5% in eval, from 0% against 100%. Deterministic under a fixed seed over a stable
  sort, because a checkpointed run that re-derived a different split would score against rows it had
  trained on.
- **Rejected:** interleaving SGD into the training pool while leaving the eval set pure SGD. Tried
  first, and insufficient — TOPv2 outnumbers SGD nine to one, so train came out 5% location-bearing
  against a 100% eval. The mismatch shrinks and does not go away.
- **Not applicable to xlam_bfcl:** BFCL is a published leaderboard and must stay the untouched eval
  set. Neither calendar corpus is a canonical benchmark, so there was nothing that had to be preserved
  whole.
- **Also fixed here:** de-duplication by request text before the split. TOPv2 repeats utterances
  verbatim, which put 29 identical requests on both sides of the split. The eval firewall in `curate`
  would have caught them at curation time, but the loader should not produce them.
- **Status:** ✅ fixed 2026-08-21.

## B322 — the calendar prompt withheld a convention 45% of its own gold labels used

- **Symptom:** on the pooled eval set the teacher scored `0.3850`, and the synthesis gate refused
  synthetic data on the strength of it.
- **Root cause:** `CALENDAR_INSTRUCTION` stated the duration convention ("make the event 60 minutes
  long unless a duration is stated") and said nothing about the default HOUR. 45% of gold rows resolve
  to 09:00 because the request names a day and no clock time ("remind me to pack my lunch tomorrow"),
  and a further slice resolves "tonight" to 20:00. Both are OUR conventions, not facts about the
  request, and nothing in the prompt named them.
- **Consequence:** nearly half the eval set was unanswerable except by guessing an unstated house rule.
  A fine-tuned student can infer it from thousands of examples, which is why the student's score
  looked merely low rather than impossible; a zero-shot or five-shot teacher cannot, so the fitness
  gate read a task defect as an incapable teacher. Neither can a human reviewer deciding whether a
  given gold label is correct — which is how the 09:00 skew went unexamined long enough to also hide
  B318.
- **Fix:** the instruction now names both conventions. One sentence, and it makes the task well-posed
  for the teacher, the student and the reader at once.
- **The general rule this encodes:** if the label generator applies a convention, the prompt has to
  state it, or the task is scoring telepathy. Worth checking the other seven tasks against.
- **Status:** ✅ fixed 2026-08-21.

## B323 — concurrent GGUF eval killed runs with an uncatchable SIGABRT, for ~20%

- **Symptom:** with `SLM_GGUF_EVAL_CONCURRENCY=8`, runs 38734202/38734203 scored two evals normally and
  then died: `CudaWorkerError: CUDA worker 'eval' failed (exit=-6, WorkerProcessError: worker produced
  no response)`.
- **Why the OOM handling did not help:** `exit=-6` is SIGABRT. llama.cpp reports a failed device
  allocation through `GGML_ASSERT`, which calls `abort()` rather than raising, so the worker process is
  gone before `_looks_like_oom` or the halving retry can run. The failure is invisible to Python by
  construction, which is the worst possible shape for a 7-day job: it dies mid-run with no diagnosis.
- **Why the trade was not worth it anyway:** measured on calendar_json over 535 prompts, 125s
  sequential against 101s at 8-way — about 20%, not the multiple that eight copies of the weights
  implies. Eight contexts submitting to one device serialise on it, so thread concurrency buys
  queueing overlap and little else.
- **Fix:** the default is 1, i.e. sequential and byte-identical to the path it replaced, with the
  machinery kept behind `SLM_GGUF_EVAL_CONCURRENCY` for deliberate experiments.
- **What would actually work, for the next attempt:** llama.cpp's multi-sequence decode API (one
  model, one context, `n_seq_max` sequences, no duplicated weights) or routing eval through the vLLM
  server already running idle on the other GPU. Both batch properly; neither duplicates the model.
  Note that llama-cpp-python 0.3.34 wires its multi-sequence `LlamaBatch` only into `embed()`, so the
  first option means driving the low-level API and hand-rolling sampling and stop conditions — the code
  that decides every accuracy number this project reports, so it needs its own verification plan.
- **A related one-line bug found on the way:** `eval/harness.py` never passed `task` to
  `infer_batch_gguf`, so concurrency sized itself from an unresolvable spec and silently degraded to 1.
  The bf16 branch beside it always passed the task; only the GGUF branch was missed — the same
  one-sided update that produced B313. Fixed, and the harness test now requires it.
- **Status:** ✅ fixed 2026-08-21 (feature disabled by default, cause documented).

## B324 — web discovery discarded any dataset smaller than the curriculum, including a near-perfect one

- **Symptom:** on calendar run 38735780, Exa found 14 candidate datasets and all 14 were rejected, so
  `mine_new_real` retired and the run had no data intervention left. One rejection read
  `materialize Xamxl/calendar_event_parser_ds_v1 failed: Instruction "train[5929:6009]" corresponds to
  no data!`
- **Root cause, two compounding faults in `data/loaders/web_acquire.py`:**
  1. the slice for a NEVER-READ dataset was sized as `len(existing_rows) + requested * 4`. With ~5,929
     rows already in the curriculum it asked a brand-new corpus for `train[:6329]`. An offset makes
     sense when re-reading a source already partly consumed; it is meaningless for one never opened;
  2. for a single-split repo the disjoint test window was a SECOND slice at `train[max_train:...]`.
     `train[offset:offset+n]` raises outright when the dataset holds fewer than `offset` rows, and the
     surrounding `except` discards the whole candidate.
  Together: any discovered dataset smaller than the current curriculum was rejected on arithmetic.
- **What it cost.** `Xamxl/calendar_event_parser_ds_v1` is 100 rows of
  `{"input": "2026-01-01T15:41:43gym tomorrow 6 am", "output": {"title": "gym", "start":
  "2026-01-02T06:00:00", "end": "2026-01-02T07:00:00"}}` — a reference instant plus a relative request,
  resolved to absolute ISO with a 60-minute default duration. That is `calendar_json` almost exactly,
  and it is the only discovered candidate of the 14 that maps cleanly onto the task.
- **Fix:** the request is sized from what is wanted (`max(300, requested * 4)`), and the single-split
  test window is carved out of ONE materialized read in memory rather than a second offset slice —
  disjoint by construction, cannot raise on a small corpus, and one read instead of two. The reserved
  test share is also now at most a fifth rather than a flat 80 rows, because the caller states the test
  half is never used as a rejection criterion; a flat 80 took 80 of those 100 rows and left 20 usable.
  Verified: the same dataset now yields train=80 / test=20 where it previously returned None.
- **The other 13 rejections were correct**, checked against the actual data rather than the log:
  `vidhikatkoria/DA_SGD_Calendar` is dialogue-act response generation (`context` → next utterance) with
  no structured event, and is derived from the same SGD corpus as our eval set, so accepting it risks
  contamination; `nvidia/Nemotron-RL-agent-calendar_scheduling` targets a slot-constraint solver STATE
  (`min_time`/`max_time`/`duration`), not an absolute-timestamped call; `Han0716/meeting-to-json-ko` is
  Korean multi-turn transcripts mapped to a meeting-summary schema; `asu-kim/conversation-calendar` has
  one unlabelled `text` column and its first row is a customer-service chat. The rest failed to build a
  split, were gated, or shipped a deprecated loading script.
- **Worth noting about the log:** eight of the fourteen were dismissed as
  `orchestrator judged it unsuitable / unmappable`, which is too terse to audit — all eight turned out
  to be correct, but confirming that required loading each dataset by hand.
- **Status:** ✅ fixed 2026-08-21.

## B325 — the empty-rebuild budget fired before the mining ladder finished its own safeguard

- **Symptom:** calendar run 38735780 stopped with `2 consecutive data rebuilds added ZERO rows` after
  mining had exhausted TOPv2 and spent exactly its two allotted web-discovery rounds.
- **Root cause:** `MAX_CONSECUTIVE_EMPTY_REBUILDS` was 2, while the mining ladder's own
  `MAX_FAILED_DISCOVERY_ROUNDS` is also 2. Those two discovery rounds legitimately add no rows —
  probing the Hub and finding nothing usable is an answer, not a malfunction — so they alone reached
  the run-health threshold and killed the run at the exact moment the ladder was about to retire
  mining and let the loop switch to hyperparameters.
- **Fix:** the budget is 4. That leaves room for both discovery rounds plus a turn either side, while
  still catching what the counter is actually for: a loop repeatedly choosing a data intervention
  against a curriculum that never changes. The tests now derive their expectations from the constant so
  the number and the assertions cannot drift apart again.
- **Status:** ✅ fixed 2026-08-21.

## B326 — the confusion-pair placeholder made the orchestrator reason about an output that never existed

- **Symptom:** on calendar run 38735780 the orchestrator wrote about "the degenerate single-token
  'incorrect' output" **65 times**, and built hypotheses on it: "the optimizer settling into a
  degenerate single-token minimum", "collapsing to a generic 'incorrect' output rather than producing
  valid structured calls". The model never emitted the string "incorrect" once. Its actual outputs were
  `[]` and well-formed calls with wrong arguments.
- **Root cause:** `build_test_report` builds confusion entries as `(gold, predicted)` pairs. A task
  with a closed label space has two real classes. An open-ended task has only a failure CATEGORY, so
  the right-hand side was filled with the literal string `"incorrect"` to keep the tuple shape — and
  then rendered verbatim as `gold='empty_call_list' predicted='incorrect' count=266` in the
  orchestrator prompt and as `empty_call_list->incorrect (266)` in the run-memory summary. Read as
  data rather than as padding, that says the model predicted the word "incorrect" 266 times.
- **Related to B296,** which removed a different symptom of the same design: the taxonomy used to
  report the single constant pair `gold_verifier -> incorrect` for every open-ended failure. That fix
  gave the LEFT side real content and left the placeholder on the right.
- **Fix:** open-ended tasks now carry `predicted=None`, and every renderer prints a category line
  instead of a pair — `failure_category='empty_call_list' count=266` in the prompt, and
  `top failures: wrong_arguments (268), empty_call_list (266)` in run memory. The pair form is kept for
  tasks that genuinely have two classes, where `local->frontier (41)` is real information.
- **Note on what was NOT wrong:** the taxonomy itself was accurate throughout. `empty_call_list` really
  was 266 of 535 failures early in the run, correctly identifying that the model was emitting `[]` for
  half the eval set, and it fell to zero as training progressed. The data was right; only the rendering
  invited a false reading of it.
- **Status:** ✅ fixed 2026-08-21.
