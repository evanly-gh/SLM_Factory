# SLM Factory — Migration Backlog

Everything considered in design docs, paper analysis, and planning sessions that has not yet been implemented. Organized by theme.

---

## 1. Open Design Gaps (Phase 1)

### 1.1 Parallel Sub-Agent Work via `delegate_task` (B21)
`agent/tools/delegate_task.py` exists but is broken and has zero call sites. The sub-agent is given no file-writing tool, so the output file is never written. **Fix:** bind `edit_file` to the sub-agent and add call sites in e.g. `curate_node` (synthesize dataset while training runs in parallel).

### 1.2 No Actual Quantization (B28)
`quantize.py` returns a theoretical profile dict from the Android pool definition. No INT4/GGUF export, no ONNX conversion, no QNN packaging. Phase 2 scope, but the interface should be defined now. **Recommended:** define a `HardwareEvalResult` dataclass and `measure_on_device(weights_ref, model_id, chip) -> HardwareEvalResult` interface.

### 1.3 No `apply_chat_template` / Assistant-Only Loss Masking (B30)
The trainer concatenates `PROMPT + LABEL` as a single text field and computes loss over all tokens. Proper SFT applies loss only to the assistant/label portion via `DataCollatorForCompletionOnlyLM` and uses the model's chat template. For classification this is tolerable (short prompts). For NER/generation with long prompts, prompt tokens dominate the loss. **Fix:** use `tokenizer.apply_chat_template` + `DataCollatorForCompletionOnlyLM`.

### 1.4 Missing Models from Android Pool (B39)
The design doc pool lists HRM-Text-1B (`sapientinc/HRM-Text-1B`, ~600MB, Tier 1, research candidate, custom runtime) and Gemma3n-E2B (~1.3GB, Tier 2, MatFormer arch). Neither is in `ANDROID_POOL`. Gemma3n-E2B has no custom runtime caveat. HRM-Text-1B needs a `notes` flag for its llama.cpp incompatibility.

### 1.5 Generation Training Has No Prompt/Response Separator (B42)
The generation branch of `format_example` concatenates prompt+response with `\n\n` and no structural delimiter. Combined with the absence of assistant-only loss masking (1.3), generation tasks have no anchor for where generation should begin. **Fix:** use `tokenizer.apply_chat_template` with role-tagged messages, or insert an explicit `\n\nAnswer:` separator.

### 1.6 Generation Scorer Metric Name and API Call Batching (B45)
Two issues: (1) the LLM-judge score (0.0–1.0) is returned in the `f1` field — semantically wrong; (2) 100 eval examples require 100 sequential Anthropic API calls. **Fix:** rename the field to `judge_score` throughout; batch judge calls with concurrent requests.

---

## 2. Phase 2 — Production Mode Pipeline

These nodes and features exist only as stubs or are not yet started.

### 2.1 Production Mode Entry Point
The system needs a `run_production()` entry point accepting a deployed model `M0` plus judged inference traces. The production mode has a 500-turn LangGraph budget (vs. 1,500 for cold-start).

### 2.2 Trace Ingestion and Partitioning (stub exists)
`agent/nodes/production/trace_ingest.py` has a skeleton. Needs: proper trace database schema, robust JSONL parsing with validation, and per-class failure rate statistics.

### 2.3 Failure Taxonomy Construction (stub exists)
`agent/nodes/production/taxonomy.py` has a basic Claude call that clusters failures. Needs: embedding-based clustering for large trace sets, fixability classification with consistent criteria, cluster size thresholds for ignoring noise, and persistence to `data-curation.md`.

### 2.4 Live Confirmation / Probe Set Verification (stub exists)
`agent/nodes/production/live_confirm.py` uses a heuristic cluster-name match instead of actual inference on `M0`. Needs: probe set generation, inference on deployed model, and confirmation logic (>50% failure rate = systematic).

### 2.5 Parent Model Awareness and Lineage Inspection (stub exists)
`agent/nodes/production/parent_awareness.py` builds a replay buffer but reads lineage from `current_dataset_path` only. Needs: full lineage reconstruction from `data-curation.md`, history-aware proposal generation to avoid repeating failed strategies.

### 2.6 Regression Gate (epsilon=2, Absolute Count)
Paper §2.6: the new model must not introduce more than 2 new errors on the regression set `R`. Not implemented. **Fix:** evaluate new checkpoint on `R`, count newly incorrect examples, reject if count > 2.

### 2.7 Cross-Checkpoint Regression Gate
The new model must pass the regression gate against all earlier deployment checkpoints, not just the previous one. Prevents accumulated drift. Not started.

### 2.8 Production Mode Turn Budget (500 Turns)
Production mode should use a 500-turn limit. Currently only `MAX_TURNS_MAIN = 1500` exists. Need a separate constant and graph compilation path.

### 2.9 Production Tool Set (`query_traces`, `trace_analysis_subagent`)
In production mode, `web_search` is removed and two new tools are added: `query_traces` and `trace_analysis_subagent`. Neither exists.

### 2.10 Trace Analyzer Sub-Agent
A specialized sub-agent with ~100K output token limit for SQL-style trace analysis. Not started.

---

## 3. Task Types Not Yet Proven

The code routes on `task_type` but only `classification` has been validated end-to-end.

### 3.1 `math_reasoning` Task Type
Mandatory CoT supervision (DeepSeek-R1 teacher), final-answer exact match eval. Routing exists but never validated. Benchmarks: GSM8K, ARC-Challenge.

### 3.2 `code_generation` Task Type
Optional CoT (GPT-4.1 teacher), execution pass@1 eval (requires sandbox). Routing exists but no sandbox. Benchmarks: HumanEval, MBPP.

### 3.3 NER End-to-End Validation
Partially fixed but never produced a working run. Remaining: `apply_chat_template` for NER prompts, entity span-F1 scorer validation, schema-constrained NER with field-level F1. Benchmark: CoNLL-2003.

### 3.4 Generation End-to-End Validation
Multiple bugs fixed but unvalidated. Remaining: assistant-only loss masking (1.3), LLM-as-judge metric rename (1.6). Benchmarks: XSum, SAMSum.

### 3.5 CLINC150 Intent Classification
30-class intent classification target. Which 30 classes to use is undecided. Dataset loader does not exist.

---

## 4. Phase 2 Hardware Infrastructure

The core differentiator of SLM Factory vs. the Pioneer Agent paper.

### 4.1 Inference Logging Infrastructure
PostgreSQL/Supabase schema for production inference traces. Not started.

### 4.2 LLM-as-Judge Config Storage
Judge prompt template and scoring criteria must be stored in inference table metadata for reproducible scoring. Not started.

### 4.3 On-Device Eval Harness
After each training round, quantize to INT4 and measure latency/power/memory on a reference device or Qualcomm AI Hub simulator. Interface defined in `training/on_device_eval.py` but implementation is `raise NotImplementedError`.

### 4.4 INT4 / GGUF / ONNX / QNN Export Pipeline
`quantize.py` is a profile lookup only. Real deployment requires GGUF export, ONNX + QNN EP for Qualcomm NPU, W4A16 quantization, LiteRT + QNN Accelerator.

### 4.5 Qualcomm AI Hub Profiling Integration
Run best checkpoint through Qualcomm AI Hub for real latency/memory/power numbers. Not started.

### 4.6 Hardware-Aware Upgrade Decision
Escalation should factor in accuracy/latency tradeoff, not just accuracy. Requires on-device measurements (4.3).

### 4.7 HRM-Text-1B Validation
Research candidate in the pool. Known issues: `token_type_ids`, FlashAttention disable, no llama.cpp. LoRA target modules may not apply cleanly. Needs experimental validation.

---

## 5. Structural Safeguards (Paper §2.7)

### 5.1 Confidence Calibration
`calibrated = w * actual_accuracy + (1-w) * raw_confidence`. No code exists. Required for production mode confidence-based routing.

### 5.2 TF-IDF Correction Propagation
When correcting a failure, use TF-IDF similarity to find and propagate corrections to similar training examples. Not started.

### 5.3 Hard Gates for Structural Safeguards
Label balancing and rollback are implemented but not enforced as hard invariants. A pre-training validation step that rejects datasets failing quality checks should be added.
