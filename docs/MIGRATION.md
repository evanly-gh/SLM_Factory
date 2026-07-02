# SLM Factory — Migration Backlog

Everything considered in design docs, paper analysis, and planning sessions that has not yet been implemented. Organized by theme. Each entry references its source and notes the implementation gap.

---

## 1. Open Design Gaps (Phase 1 — Tracked in BUGS.md)

These were explicitly designed for Phase 1 but left as `⚪ design gap` in BUGS.md.

### 1.1 Parallel Sub-Agent Work via `delegate_task` (B21)
`agent/tools/delegate_task.py` exists but is broken and has zero call sites. The sub-agent is given no file-writing tool, so the output file is never written. The paper's context isolation pattern (sub-agents write summaries to disk; main agent reads files) is entirely unimplemented. **Fix:** bind `edit_file` to the sub-agent and add call sites in e.g. `curate_node` (synthesize dataset while training runs in parallel).

### 1.2 Context Manager / Turn Compaction (B22)
No conversation history is accumulated; there is nothing to compact. The paper's Context Manager selectively compacts older turns while preserving key decisions, eval results, and dataset lineage for 500–1,500-turn runs. Currently each node is a fresh stateless LLM call so compaction pressure is low, but this becomes critical if nodes are wired into a multi-turn ReAct loop. `data-curation.md` serves as the durable complement for now.

### 1.3 Tools Not Wired into LLM / ReAct Loop (B23)
`web_search`, `bash`, `read_file`, and `edit_file` are `@tool`-decorated functions but no `ToolNode` or `bind_tools` call connects them to any LLM. All data acquisition, file reading, and shell execution happen via direct Python calls. The paper's "4 named tools" interface (§2.3) is not replicated. **Fix:** replace deterministic node functions with a ReAct loop, wire tools into LangGraph, and bind them to the orchestrator LLM.

### 1.4 `MAX_TURNS_MAIN` Dead Config (B24)
`config.MAX_TURNS_MAIN = 1500` is defined but never read. Effective run length is controlled only by `recursion_limit` in `graph.compile()`. **Fix:** wire `MAX_TURNS_MAIN` into `graph.compile(recursion_limit=config.MAX_TURNS_MAIN)`.

### 1.5 DAG Has No Edges; `π=(D,H,S)` Not Stored Per Node (B25)
`dag_node` stores score/model/weights_ref but no parent pointer, no `D` (dataset version), no `H` (hyperparameter config), no `S` (learning strategy). The DAG is a flat append-only list. Lineage attribution (understanding why accuracy changed) requires edges and full `π` triples. Rollback can only restore weights, not reproduce the exact `(D,H,S)` that achieved a given score. **Fix:** extend `dag_node` with `parent_iteration`, `dataset_version`, `hyperparams`, `strategy` fields and a parent pointer on each `dag.append()` call.

### 1.6 Teacher Models (DeepSeek-R1 / GPT-4.1) Never Called (B26)
`DEEPSEEK_API_KEY` and `OPENAI_API_KEY` are in `.env`/`config.py` but never imported or used. The generation branch of `synthesize_hard_negatives` calls `claude-sonnet-4-6` for all task types. The design spec calls for DeepSeek-R1 for math/science CoT annotation and GPT-4.1 for code/QA hard negatives. **Fix:** add DeepSeek and OpenAI client paths in `curriculum.py`, guarded by task sub-type.

### 1.7 Three of Five Quality Controls Missing (B27)
`apply_quality_controls()` only implements label balancing. Missing:
1. **Context-length matching** — training example lengths must match the length distribution of eval/production examples.
2. **NER entity diversification** — no single entity value should appear >2–3× in the training set; overrepresented entities should be replaced with synthetic equivalents.
3. **CoT annotation for generation** — generation examples need chain-of-thought traces from the teacher model, not just answers.

### 1.8 No Actual Quantization (B28)
`quantize.py` returns a theoretical profile dict from the Android pool definition. No INT4/GGUF export, no ONNX conversion, no QNN packaging. Correctly Phase 2 scope, but the interface needs to be defined now. **Recommended:** define a `HardwareEvalResult` dataclass and `measure_on_device(weights_ref, model_id, chip) -> HardwareEvalResult` interface; implement via Qualcomm AI Hub API or ADB shell profiling in Phase 2.

### 1.9 No `apply_chat_template` / Assistant-Only Loss Masking (B30)
The trainer concatenates `PROMPT + LABEL` as a single text field and computes loss over all tokens. Proper SFT applies loss only to the assistant/label portion via `DataCollatorForCompletionOnlyLM` and uses the model's chat template for train/serve parity. For classification this is tolerable (short prompts). For NER/generation with long prompts, prompt tokens dominate the loss and degrade learning. **Fix:** use `tokenizer.apply_chat_template` + `DataCollatorForCompletionOnlyLM`.

### 1.10 Dataset Size Hardcoded at 150 (B31)
`N_TOTAL = 150` in `curate_node` ignores the paper's task-type-specific targets: 100–200 examples for classification, 300 for NER, 500–3,000 for generation. Generation tasks will underfit at 150 examples. **Fix:** `classification → 150`, `NER → 300`, `generation → 1000`.

### 1.11 Baseline / SOTA Survey Never Runs (B32)
`task_analysis_node` sets `stop_threshold = 0.96` from a hardcoded default. The design spec calls for an Exa search to find published SOTA accuracy on the target benchmark at the target model size class and calibrate `stop_threshold` accordingly. **Fix:** add an Exa `web_search` call in `task_analysis_node`.

### 1.12 2-for-1 Rule Gold Anchor Not Paired (B36)
The docstring claims the 2-for-1 rule is implemented, but only the synthetic counterexample is added to the curriculum. The original gold example that inspired the hard negative is not paired alongside it. The model sees what NOT to predict but not what TO predict for the same surface form. **Fix:** `synthesize_hard_negatives` should return both the original example and the synthetic counterexample as a pair.

### 1.13 No Surface-Pattern Diversity Enforcement (B37)
The paper requires 3–5 distinct surface-text patterns per label. No code checks or enforces this. Training data may contain many similar examples for the same label without syntactic diversity. **Fix:** add a diversity check that clusters training examples by surface pattern (TF-IDF or embedding similarity) and ensures ≥3 distinct patterns per label.

### 1.14 `filter_pool()` Ignores Latency and Power (B38)
`HardwareConstraints` has `latency_ttft_ms` and `power_watts` fields, but `filter_pool()` only filters on `storage_mb` and `memory_mb`. The design doc says latency and power should be logged (not gating) in Phase 1, but even logging is missing. The tok/s data in `ModelSpec` is never compared against the constraints. **Fix:** add a `check_all_constraints()` function that returns per-constraint PASS/FAIL status for logging in `data-curation.md`.

### 1.15 Missing Models from Android Pool (B39)
The design doc pool lists HRM-Text-1B (`sapientinc/HRM-Text-1B`, ~600MB, Tier 1, research candidate, custom runtime) and Gemma3n-E2B (~1.3GB, Tier 2, MatFormer arch). Neither is in `ANDROID_POOL`. Gemma3n-E2B has no custom runtime caveat and should be straightforward to add. HRM-Text-1B needs a `notes` flag for its llama.cpp incompatibility and deferred on-device export.

### 1.16 `data-curation.md` Missing Schema Fields (B41)
The design doc §4.3 specifies fields not present in the implementation:
1. Per-slice failure taxonomy text (description of what failed and why, per `E_pos`/`E_neg`/`E_boundary`).
2. Hardware PASS/FAIL lines: `Storage: {size}MB vs S_max={S_max}MB — PASS/FAIL` for all four constraints.
3. Escalation availability: `Escalation available: yes/no`, `Next model: {id}`, `Would pass constraints: yes/no`.
**Fix:** add `failure_taxonomy`, `hw_pass_fail` dict, and `escalation_info` dict to `write_iteration()`.

### 1.17 Generation Training Has No Prompt/Response Separator (B42)
The generation branch of `format_example` concatenates prompt+response with `\n\n` and no structural delimiter. Combined with the absence of assistant-only loss masking (B30), generation tasks have no anchor for where generation should begin. **Fix:** use `tokenizer.apply_chat_template` with role-tagged messages, or insert an explicit `\n\nAnswer:` separator that the inference prompt also uses.

### 1.18 SMS Spam Train/Test Split Not Shuffled (B44)
`data/loaders/sms_spam.py` uses sequential `examples[:split]` / `examples[split:]`. The UCI SMS Spam Collection is not randomly ordered, risking train/test distribution mismatch. **Fix:** shuffle with a fixed seed before splitting.

### 1.19 Generation Scorer Metric Name and API Call Batching (B45)
Two issues: (1) the LLM-judge score (0.0–1.0) is returned in the `f1` field — semantically wrong, confuses downstream code; (2) 100 eval examples require 100 sequential Anthropic API calls. **Fix:** rename the field to `judge_score` throughout; batch judge calls with concurrent requests.

### 1.20 Classification Extractor Falls Back to Majority Class (B46)
When the model outputs garbled text that doesn't match any known label, `extract_predictions` silently defaults to the majority (non-positive) label. A model that outputs random garbage scores as well as one that consistently predicts majority class. **Fix:** use `"UNKNOWN"` as the fallback label and track extraction failures as a separate metric.

### 1.21 Inference Model Cache Never Clears (B47)
`_inference_cache` in `slm_helpers.py` is a module-level dict that never evicts old checkpoints. Over 10+ iterations with 2 configs each, 20+ model copies accumulate in VRAM. **Fix:** implement LRU eviction (keep only the last 2–3 checkpoints) or clear the cache between iterations.

### 1.22 `messages` Field in `AgentState` Never Used (B49)
`messages: list[Any]` is initialized as `[]` but no node reads or writes to it. Becomes relevant if the architecture is refactored toward a ReAct/tool-use loop (see B23). **Fix:** remove the field to reduce confusion, or wire it into `iterate_node`'s LLM call to accumulate a conversation transcript.

### 1.23 Cost Tracking and Budget Enforcement
Design doc Appendix B specifies a <$50 budget per Phase 1 run with tracked components: Claude Sonnet API token usage, GPU compute time, and Exa API calls. No cost tracking code exists anywhere. **Fix:** add token usage tracking via Anthropic SDK usage fields, GPU-time estimation from training duration, and an Exa call counter; log running total and alert at approach to $50.

---

## 2. Phase 2 — Production Mode Pipeline

These nodes and features exist only as stubs or are not yet started. Production mode activates when a deployed model + judged inference traces are provided as input instead of a task description.

### 2.1 Production Mode Entry Point
The system needs a `run_production()` entry point accepting a deployed model `M0` plus a set of judged inference traces `T = {(input, prediction, corrected_output, verdict, reasoning)}`. The structured trace format and input schema need finalizing. The production mode has a 500-turn LangGraph budget (vs. 1,500 for cold-start).

### 2.2 Trace Ingestion and Partitioning (stub exists)
`agent/nodes/production/trace_ingest.py` has a skeleton. Needs: proper trace database schema, integration with the inference logging infrastructure (see §5.1), robust JSONL parsing with validation that all required fields are present, and per-class failure rate statistics.

### 2.3 Failure Taxonomy Construction (stub exists)
`agent/nodes/production/taxonomy.py` has a basic Claude call that clusters failures. Needs: embedding-based clustering as an alternative to LLM-only clustering for large trace sets, fixability classification (`fixable` vs. `external`) with consistent criteria, cluster size thresholds for ignoring noise clusters, and persistence of the taxonomy to `data-curation.md`.

### 2.4 Live Confirmation / Probe Set Verification (stub exists)
`agent/nodes/production/live_confirm.py` uses a heuristic cluster-name match instead of actual inference on `M0`. Needs: actual probe set generation (new examples designed to test each identified weakness), inference of the currently deployed model on the probe set, and confirmation logic (>50% failure rate on probes = systematic weakness).

### 2.5 Parent Model Awareness and Lineage Inspection (stub exists)
`agent/nodes/production/parent_awareness.py` builds a replay buffer but reads lineage from `current_dataset_path` only. Needs: full lineage reconstruction from `data-curation.md`, recovery of `D_parent` composition and training config, history-aware proposal generation to avoid repeating failed strategies, and the regression set `R` (examples the deployed model currently answers correctly that must not be broken).

### 2.6 Regression Gate (epsilon=2, Absolute Count)
Paper §2.6: the new model must not introduce more than 2 new errors on the regression set `R`. This is an absolute count (not a percentage), making it strict for small eval sets. The gate must be a hard requirement — no model passes to deployment without clearing it. Not implemented at all. **Fix:** evaluate new checkpoint on `R`, count newly incorrect examples, reject if count > 2.

### 2.7 Cross-Checkpoint Regression Gate
The new model must pass the regression gate against not just the immediately previous checkpoint but all earlier deployment checkpoints. Prevents accumulated drift where `A→B→C` is fine pairwise but `A→C` has regressed. Requires: historical eval set persistence across deployment stages, multi-set evaluation, and a composite gate. Not started.

### 2.8 Production Mode Turn Budget (500 Turns)
Production mode uses a 500-turn limit vs. 1,500 for cold-start. This needs to be wired into the production graph's `compile(recursion_limit=...)`. Config constant exists (`MAX_TURNS_MAIN = 1500`) but no production-specific constant or separate graph compilation exists.

### 2.9 Production Tool Set (`query_traces`, `trace_analysis_subagent`)
In production mode, `web_search` is removed and two new tools are added: `query_traces` (SQL + bash pipeline for querying production inference logs) and `trace_analysis_subagent` (spawns a specialized sub-agent with ~100K output token limit for analyzing trace data). Neither exists. Requires: `query_traces` tool with SQL parser and trace database schema; sub-agent definition with elevated output limit; mode-aware tool registration that swaps tool sets.

### 2.10 Trace Analyzer Sub-Agent
A specialized sub-agent with ~100K output token limit dedicated to SQL-style analysis of production inference traces. Performs complex analytical queries to identify failure patterns, cluster errors, and compute statistics. Not started.

### 2.11 Replay Buffer Integration into `curate_node`
`curate_node` has a comment in PIPELINE.md that it "mixes the `replay_buffer`" in production mode, but the actual code in `curate.py` does not read `state["replay_buffer"]`. The replay buffer (10–20% of `D_parent`) must be included in the dataset composition to prevent catastrophic forgetting. Not integrated.

---

## 3. MCGS — Full Monte Carlo Graph Search

Phase 1 uses sequential greedy DAG only. Full MCGS is deferred but was fully designed.

### 3.1 UCT Selection with Time-Decaying Exploration
`UCT(v) = f_bar(v) + c(t) * sqrt(ln(N) / n_i)` where `c(t)` decays from exploration to exploitation over iterations. Requires: visit count tracking per DAG node, `c(t)` schedule, UCT computation over all leaf-adjacent nodes. Not started.

### 3.2 Top-K Exploitation
In later iterations (when `c(t)` has decayed), switch from UCT to selecting from the K highest-scoring nodes. Requires: configurable transition point (iteration number or score threshold), Top-K selection function. Not started.

### 3.3 FUSION Operator (Cross-Branch Synthesis)
Merges complementary strategies from independent branches of the search DAG. If one branch discovered effective data composition and another found optimal hyperparameters, FUSION creates a node combining both. Requires: branch comparison logic, merge function for `(D, H, S)` tuples, multi-parent edge support in the DAG, conflict resolution. Not started.

### 3.4 Stagnation Recovery via Evolution
When stagnated, `evolution` generates mutated configs with awareness of the full trajectory — larger, more exploratory changes than normal EXPAND. Currently stagnation only triggers model escalation (Node 8); evolution as an alternative recovery is not implemented.

### 3.5 Score Backpropagation Through Ancestors
After each training attempt, backpropagate the score through ancestor nodes in the DAG, updating running averages and visit counts at each ancestor (analogous to MCTS backpropagation). Currently scores are only stored at the leaf node. Not started.

---

## 4. Task Types Not Yet Proven

The code routes on `task_type` but only `classification` has been validated end-to-end.

### 4.1 `math_reasoning` Task Type
Design doc specifies: mandatory CoT supervision (DeepSeek-R1 teacher), final-answer exact match via regex extraction as eval metric, `E_boundary` = multi-step/edge-case problems. Not in original plan's three types (`classification`, `NER`, `generation`) — added in design doc §1. The task type enum needs to be expanded and routing validated. Benchmarks: GSM8K, ARC-Challenge, TriviaQA.

### 4.2 `code_generation` Task Type
Design doc specifies: optional CoT (GPT-4.1 teacher), execution pass@1 as eval metric (requires sandbox execution against unit tests), `E_boundary` = multi-step/edge-case problems, logic-error hard negatives. Not in original plan. Requires a code execution sandbox (currently no isolation exists). Benchmarks: HumanEval, MBPP, SQL generation.

### 4.3 NER End-to-End Validation
NER training has been partially fixed (B29, B48) but has never produced a working training run. Remaining gaps: `apply_chat_template` for NER prompt format, entity span-F1 scorer validation, entity diversification quality control (B27), schema-constrained NER with field-level F1 (design doc flag `schema:dict`). Benchmark: CoNLL-2003.

### 4.4 Generation End-to-End Validation
Generation has three compounding bugs fixed (B29, B33, B42) but remains unvalidated. Remaining: teacher model CoT annotation (B26), assistant-only loss masking (B30), LLM-as-judge metric rename (B45), task-type-specific dataset size (B31). Benchmarks: XSum, SAMSum.

### 4.5 CLINC150 Intent Classification
Session 1 identified CLINC150 (30-class intent classification) as a target benchmark. Which 30 classes out of 150 to use is undecided — options are: random sample, balanced-by-domain, or hardest (most semantically overlapping intents). The dataset loader for CLINC150 does not exist.

---

## 5. Phase 2 Hardware Infrastructure

The core differentiator of SLM Factory vs. the Pioneer Agent paper. Nothing from this section is started.

### 5.1 Inference Logging Infrastructure
Design doc §7 specifies a PostgreSQL/Supabase schema for recording production inference traces: `(input, prediction, corrected_output, verdict, judge_reasoning, judge_metadata)` with per-user row-level security. Must be designed alongside Phase 1 to be ready for Phase 2. Schema not started.

### 5.2 LLM-as-Judge Config Storage
The judge prompt template and scoring criteria must be stored in the inference table metadata to ensure reproducible scoring across deployment cycles. Not started.

### 5.3 On-Device Eval Harness
The core Phase 2 hardware-in-the-loop component. After each training round, quantize the checkpoint to INT4 and measure on a reference device (or Qualcomm AI Hub simulator):
- **Latency** (TTFT in ms, tok/s throughput)
- **Power** (average watts during sustained inference, via Android Battery Historian)
- **Memory** (peak RSS during inference)

Recommended interfaces to define now:
```python
@dataclass
class HardwareEvalResult:
    ttft_ms: float
    tok_s: float
    peak_memory_mb: int
    avg_power_watts: float
    chip: str

def measure_on_device(weights_ref: str, model_id: str, chip: str) -> HardwareEvalResult: ...
```

### 5.4 INT4 / GGUF / ONNX / QNN Export Pipeline
`quantize.py` is a profile lookup only. Real on-device deployment requires:
- GGUF export via `llama.cpp` for CPU/GPU inference
- ONNX export + QNN Execution Provider for Qualcomm Hexagon NPU (confirmed supported: Llama3.2 1B/3B, Qwen3-4B)
- W4A16 quantization (dominant for on-device)
- LiteRT + QNN Accelerator path for Android NPU

### 5.5 Qualcomm AI Hub Profiling Integration
Design doc §7 specifies running the Phase 1 best checkpoint through Qualcomm AI Hub profiling on the reference chip to get real latency/memory/power numbers to replace the theoretical values. API integration not started.

### 5.6 Hardware-Aware Upgrade Decision
Session 1: the agent should escalate from Tier 1→Tier 2 only if the accuracy delta justifies the on-device latency penalty. Currently escalation is purely accuracy-driven. Full implementation requires on-device latency measurements (§5.3) to compute the accuracy/latency tradeoff.

### 5.7 On-Device Proxy Eval After Each Training Round
Session 1: after each training round, quantize to INT4 and measure simulated on-device throughput using known hardware benchmarks (tok/s per chip tier). Currently this is theoretical lookup only; real measurement requires the on-device eval harness (§5.3).

### 5.8 AdaptFT-Style Hardware Diversity Injection
Session 1: when adding production noise (for robustness), also inject hardware diversity — same utterance routed to different chip tiers to ensure accuracy is consistent across device classes. Not designed further.

### 5.9 HRM-Text-1B Validation
HRM-Text-1B (`sapientinc/HRM-Text-1B`) is in the pool as a research candidate but has never been trained or run. Known issues: `token_type_ids` must be set, FlashAttention must be disabled, standard llama.cpp won't load it. LoRA fine-tuning with standard target modules may not apply cleanly due to its hierarchical recurrent architecture (H-stack + L-stack, 8 iterations per forward pass). Needs careful testing before inclusion in real runs.

### 5.10 MediaTek NeuroPilot SDK Integration
Session 1: MediaTek officially supports Llama3.2 and Qwen3 via NeuroPilot SDK. Not designed further. Relevant for devices using MediaTek Dimensity chips.

---

## 6. Structural Safeguards (Paper §2.7) Not Yet Enforced

These are in the paper as mandatory invariants, not optional features.

### 6.1 Confidence Calibration
Model confidence scores should be calibrated: `calibrated = w * actual_accuracy + (1-w) * raw_confidence`. The calibration weight `w` is tuned per task and updated as eval data accumulates. No code exists for this. Required for reliable production mode confidence-based routing.

### 6.2 TF-IDF Correction Propagation
When correcting a failure, use TF-IDF similarity to find other training examples similar to the corrected example and check/propagate the correction. Prevents fixing one example while leaving similar mislabeled examples intact. Requires: TF-IDF vectorizer fitted on training corpus, similarity threshold, propagation logic.

### 6.3 Hard Gates for Structural Safeguards
The paper identifies hard negatives + label balancing, rollback-first, parallel training, and regression gating as structural safeguards (not optional). Currently label balancing and rollback are implemented but not enforced as hard invariants — they can be bypassed by code paths that don't call `apply_quality_controls()`. A pre-training validation step that rejects datasets failing these checks should be added.

---

## 7. Open Questions from Session 1

Decisions that were flagged as unresolved and have not been answered since.

1. **CLINC150 30-class subset** — which 30 of 150 intent classes? Options: random, balanced-by-domain, or hardest (most semantically overlapping). Unresolved.
2. **HRM-Text LoRA fine-tuning** — architecture reuses parameters through recurrence; standard LoRA target modules may not apply. Needs experimental validation before pool inclusion.
3. **Synthetic data generation quality** — no metrics exist for evaluating whether LLM-generated hard negatives are genuinely hard (boundary-crossing) vs. trivially different. Prompt engineering for this is ad hoc.
4. **Regression threshold floor** — the paper uses `epsilon=2` (absolute count). For very small eval sets this may be too strict; for large eval sets too lenient. No per-task calibration exists.
