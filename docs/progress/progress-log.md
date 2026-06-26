# SLM Factory — Progress Log

## Session 3 — 2026-06-24

### Status: First real GPU runs completed; now fixing remaining gaps vs. Pioneer Agent paper

### What happened in Session 2 (execution)
- All 9 plan tasks implemented and reviewed via SDD workflow
- 32 tests passing
- First real GPU run: Qwen3-0.6B on L40, real Unsloth train + inference → F1=0.553 (6-class task)
- Many bugs found and fixed (B1–B18 in BUGS.md); key remaining open bugs: B2, B4, B5, B13

### Session 3 goals — COMPLETED
1. **B2** ✅ `iterate_node` is now LLM-driven. Calls Claude Sonnet 4.6 with full `data-curation.md`
   trajectory + current failures. LLM returns `{intervention, hypothesis, ...}` JSON. Falls back to
   score-band rules on failure. `hypothesis` flows through `state["last_hypothesis"]` into
   `evaluate_node` → `data-curation.md`.
2. **B5** ✅ `rollback_node` now restores `best_weights_ref` + `best_score` from the best non-pruned
   DAG node after marking the regressing node as pruned.
3. **B13** ✅ `curate_node` now targets `N_TOTAL=150` at 65:35 split (97 gold + 53 hard).
   `synthesize_hard_negatives` uses all labels (not just minority class) as source material.
4. **B4** ✅ `train_node` increments `state["iteration"]` before building `output_dir`, so dir name
   matches DAG/data-curation.md iteration number.

### Files changed
- `agent/nodes/iterate.py` — LLM call via `_llm_iterate()`, `apply_iteration_policy()` as fallback
- `agent/nodes/rollback.py` — restores `best_weights_ref` from DAG
- `agent/nodes/evaluate.py` — `hypothesis=state.get("last_hypothesis", "")`
- `agent/state.py` — added `last_hypothesis: str` and `llm_iterate_decision: Optional[dict]`
- `data/curriculum.py` — all-label hard-negative synthesis for classification
- `agent/nodes/curate.py` — explicit 65:35 split, surgical n scales with failures
- `agent/nodes/train.py` — increment before output_dir build

All 32 tests pass.

### Session 3b — 2026-06-26 (continued)

**B19/B20 fixed:** `train_node` now reads `llm_iterate_decision["hyperparams"]` for LLM-driven configs;
`curate_node` passes `targeted_patterns` to `synthesize_hard_negatives`.

**Deep audit completed (paper × design docs × codebase):**
- Two parallel Opus agents read all 43 pages of the paper, all 3 design docs, and every source file
- Cross-referenced every paper feature, every design doc requirement, and every line of code
- Found 18 new issues (B33–B50) not covered by previous B1–B32 entries

**Key new findings:**
- **B33 (🔴):** `train_node` never passes `task_type` → all non-classification tasks train with SMS spam prompt
- **B34 (🔴):** `curate_node` double-applies `gold_fraction` → gold count is ~63 not ~97
- **B35 (🔴):** Boundary keywords are SMS-spam-specific → other tasks get no boundary detection
- **B36:** 2-for-1 rule documented but not implemented (only the synthetic half is generated)
- **B38–B41:** Android pool / hardware logging gaps vs design doc
- **B42–B43:** Generation training and model escalation state management bugs
- **B48:** NER web-acquired data has no entity annotations at all
- **B50:** No on-device eval harness exists (the core HW-in-the-loop differentiator)

**Priority for next work:** B33 > B34 > B35 (three real bugs) > B29 (NER/gen hard negatives) >
B48 (NER data pipeline) > B36 (2-for-1 rule) > B43 (escalation state).

---

## Session 2 — 2026-06-19

### Status: Implementation plan written, ready for execution

### Files created / planned
- `config.py` — API keys, cluster settings
- `android_pool.py` — model pool + constraint filter
- `data/loaders/sms_spam.py` — UCI download
- `data/eval_set.py` — E = Epos ∪ Eneg ∪ Eboundary
- `eval/metrics.py` — binary F1, per-slice scores
- `eval/harness.py` — run_eval()
- `data/curriculum.py` — Dcold synthesis, quality controls
- `data/curation_log.py` — data-curation.md read/write
- `training/slm_helpers.py` — train() / infer() / infer_batch()
- `training/lora_trainer.py` — Unsloth LoRA loop
- `training/quantize.py` — theoretical hardware profiles
- `agent/state.py` — AgentState TypedDict
- `agent/tools/` — web_search, bash, file_tools, delegate_task
- `agent/nodes/` — all 8 state machine nodes
- `agent/graph.py` — LangGraph assembly
- `run.py` — entry point

---

## Session 1 — 2026-06-18

### Status: Design phase (brainstorming), Section 1 approved

---

### What was decided

**Project:** Replicate Pioneer Agent (arXiv:2604.09791) with Android hardware constraints baked in from the start.

**Phase 1 scope:** Agentic fine-tuning loop for intent classification on CLINC150, no hardware involved yet. Weeks 1-4.

**Architecture chosen:** Approach A — full Pioneer Agent replication (LangGraph + Claude Sonnet orchestrator + tool suite + LoRA backend). No shortcuts.

**Task:** CLINC150, 30-class subset for fast iteration. Proven path from paper (84.9% → 99.3% with GLiNER2).

**Training infrastructure:** GPU cluster with H200s, A100s, L40s.

**Model pool (Android-constrained):**
- Tier 1 (≤800MB INT4): Qwen3-0.6B, Qwen3.5-0.8B, MiniCPM5-1B, MiniCPM4-0.5B, Gemma3-1B, Llama3.2-1B
- Tier 2 (800MB–1.5GB): SmolLM2-1.7B, Qwen3-1.7B, Qwen3.5-2B, DeepSeek-R1-Distill-1.5B, Gemma3n-E2B
- Tier 3 (generous upper bound): Llama3.2-3B (Q3_K_M), Ministral-3B

---

### Key research findings

**Pioneer Agent paper (arXiv:2604.09791):**
- LangGraph state machine + Claude Sonnet 4.6 (32K thinking budget, 1M context)
- Modal sandboxes (16GB RAM, 24hr timeout) for training
- 5 core tools: bash, read_file, edit_file, query_traces, delegate_task
- Cold-start mode: up to 1,500 LangGraph turns
- Production mode: 8-stage pipeline, up to 500 turns
- Avg run: 6 hours, ~$35
- CLINC150 result: 84.9% → 99.3% (GLiNER2 base, 30-class, 2,999 inferences)
- Key failure mode: "cascading fixes" — rollback-first principle prevents stagnation
- MCGS: agent searches jointly over data composition, hyperparameters, learning strategy via a DAG of training attempts

**HRM-Text (arXiv:2605.20613, Sapient Intelligence, May 2026):**
- 1.15B params, ~0.6GB INT4 (theoretical), Apache 2.0
- Novel hierarchical recurrent arch (H-stack + L-stack, 8 iterations per forward pass)
- Pre-alignment base model — needs SFT before classification
- CAUTION: standard llama.cpp/Ollama won't load it (custom `hrm_text` arch not upstream)
- ARC Prize analysis: architecture contributes "minimal performance impact" vs plain transformer; gains from training setup
- Training on structured instruction-response pairs (not raw text) — benchmark comparisons are asymmetric
- Include in pool as "research candidate," defer on-device export

**Android deployment findings:**
- Best on-device format for Qualcomm NPUs: ONNX + QNN Execution Provider (not GGUF)
- GGUF/llama.cpp does NOT access Hexagon NPU by default
- W4A16 quantization dominant for on-device inference
- Qualcomm AI Hub confirmed: Llama3.2 1B/3B, Qwen3-4B, Phi-3.5-mini are supported; Gemma3/Phi-4-mini not yet
- MediaTek: NeuroPilot SDK, officially supports Llama3.2, Qwen3
- LiteRT (Google) + QNN Accelerator: recommended path for Android NPU

---

### Decisions still pending

- Section 2+ of design doc (agent architecture, tool layer, training backend, data pipeline, eval harness) — in progress
- What elements of Pioneer Agent to keep vs modify for hardware-in-the-loop goal

---

### Elements to consider for Phase 2 hardware integration

**Keep from Pioneer Agent:**
- LangGraph state machine structure
- Context Manager (essential for long runs)
- MCGS DAG for tracking training attempts
- Regression gate / rollback-first principle
- Taxonomy construction + fixable vs external failure classification
- Curriculum synthesis (corrected + hard negatives + contrastive + replay)

**Modify for hardware-in-the-loop:**
- Model selection: constrained to Android pool — agent scores candidates not just on accuracy but on (accuracy / on-device latency) tradeoff
- Eval harness: add an "on-device proxy eval" stage — after each training round, quantize the checkpoint to INT4 and measure simulated on-device throughput using known hardware benchmarks (tok/s per chip tier)
- Data pipeline: track not just label accuracy but also inference cost (FLOP count per example) — lightweight models that get equivalent accuracy are preferred
- AdaptFT-style staging: when adding production noise, also inject hardware diversity (same utterance routed to different chip tiers, accuracy should be consistent)
- Upgrade decision: agent escalates from Tier 1 → Tier 2 only if accuracy delta justifies the on-device latency penalty

---

### Open questions / risks

1. **Context Manager implementation complexity** — the paper mentions it but doesn't open-source it. Need to build our own or find a suitable LangGraph equivalent.
2. **HRM-Text LoRA fine-tuning** — architecture reuses parameters through recurrence; standard LoRA target modules may not apply cleanly. Test carefully.
3. **CLINC150 30-class subset selection** — which 30 classes? The paper doesn't specify. Need to decide: random, balanced by domain, or hardest (most semantically overlapping).
4. **Synthetic data generation quality** — the agent uses Claude to generate targeted examples. Need prompt engineering for this to produce genuinely hard negatives vs easy positives.
5. **Regression threshold** — paper doesn't give an exact number. Need to choose: how much regression is acceptable before rollback? (Suggest: >0.5pp drop triggers rollback.)
