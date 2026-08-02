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
benchmark research (`docs/Evan's Notes/2026-07-28-task-suite-benchmark-research.md`) found
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
