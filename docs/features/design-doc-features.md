# SLM Factory Phase 1 -- Complete Feature List

Extracted from `docs/superpowers/specs/2026-06-19-slm-factory-phase1-design.md`.
Each entry references the design doc section and its relationship to the Pioneer Agent paper (arXiv:2604.09791).

---

Paper-equivalent features (replicated from Pioneer Agent):

Features 1-15, 22-29, 31-35 -- core pipeline components: task specification, task type classification, LangGraph orchestrator, sub-agent delegation, context manager, data-curation.md, tool layer, all 8 state machine nodes, eval harness, lineage DAG, scoring/regression functions, convergence criteria, curriculum synthesis, quality controls, hard negative generation, web search integration, chat template tokenization, iteration trajectory, cost tracking
SLM Factory additions (not in paper) -- the hardware-aware Android extensions:

Feature 16: Android Model Pool (3-tiered, 14 models with INT4 sizes)
Feature 17: Hardware Constraint Layer (4 constraints: storage, memory, latency, power)
Feature 18: Hardware-Aware Model Escalation (constraint check before tier upgrade)
Feature 19: Hardware Profile Logging in data-curation.md
Feature 36: Phase 1 to Phase 2 Handoff Artifacts (5 artifacts including Qualcomm AI Hub profiling)
Feature 37: HRM-Text-1B Model-Specific Handling (custom runtime requirements)
Feature 38: Decoder-Only Architecture Constraint (paper used encoder GLiNER2)
Feature 41: Phase 1 Hardware Constraint Mode (log-only for latency/power, gating for storage/memory)

## 1. Task Specification Triple
**Design doc reference:** SS1, SS2.4
**Paper equivalent:** Paper SS3.1 (task specification)
**Description:** Every run begins with a task specification triple `tau = (description, M, constraints)` where `description` is a free-text task description (e.g. "fine-tune on SMS Spam binary classification"), `M` is the Android model pool, and `constraints` is a dictionary containing `target_metric`, `hardware_thresholds` (storage_mb, memory_mb, latency_ttft_ms, power_watts), and `target_chip`. The hardware thresholds are seeded but not gating in Phase 1. This triple is the sole input to the system.
**Implementation requirements:** A data class or typed dict for the task specification; parser that validates the triple; default values for hardware thresholds; integration with the LangGraph state so all nodes can access the spec.

---

## 2. Task Type Classification System
**Design doc reference:** SS1 (task abstraction table), SS2.4 Node 1
**Paper equivalent:** Paper SS4.2 (task-type routing)
**Description:** The system classifies every task into one of three types: `classification`, `NER`, or `generation`. This single decision, made in `task_analysis`, locks in three downstream choices for the entire run: supervision format (direct labels vs. chain-of-thought), eval metric (accuracy/F1 vs. entity F1 vs. LLM-as-judge), and curation strategy (hard negatives vs. entity diversification vs. teacher model annotation). The classification covers all nine paper benchmarks: SMS Spam and CLINC150 (classification), CoNLL-2003 (NER), and ARC-Challenge, GSM8K, TriviaQA, HumanEval, XSum, SAMSum (generation).
**Implementation requirements:** Task type enum with three values; a classifier function (likely LLM-driven) that parses the free-text description; a routing table that maps task type to (supervision_format, eval_metric, curation_strategy); tests that known benchmark names route correctly.

---

## 3. LangGraph State Machine Orchestrator
**Design doc reference:** SS2.1
**Paper equivalent:** Paper SS3.2 (orchestration layer)
**Description:** The main agent runs as a LangGraph state machine with Claude Sonnet 4.6 (1M context window, 32K extended thinking budget). It drives the full fine-tune, evaluate, diagnose, curate, retrain loop. Supports up to 1,500 LangGraph turns for cold-start mode. Each turn consists of one LLM reasoning step plus associated tool calls.
**Implementation requirements:** LangGraph graph definition with all eight nodes wired; state schema carrying current iteration data, eval results, and configuration; turn counter with 1,500-turn termination; Claude Sonnet 4.6 as the LLM backend; extended thinking configuration.

---

## 4. Hierarchical Agent Structure with Sub-Agent Delegation
**Design doc reference:** SS2.2
**Paper equivalent:** Paper SS3.2 (sub-agent spawning)
**Description:** The system uses a hierarchical agent structure. The main agent orchestrates the full pipeline. Independent sub-agents are spawned via `delegate_task` for parallel work (e.g., synthesize the next dataset while a training job runs). Sub-agents run the same Claude Sonnet 4.6 model, capped at 1,000 turns. Sub-agents share the filesystem with the main agent -- they write summaries to disk; the main agent reads those files without loading raw data into its own context window. This is the core context isolation pattern.
**Implementation requirements:** `delegate_task` mechanism that spawns sub-agents; filesystem-based communication protocol (sub-agents write, main agent reads); sub-agent turn limit of 1,000; shared filesystem access; sub-agent lifecycle management.

---

## 5. Context Manager
**Design doc reference:** SS2.2
**Paper equivalent:** Paper SS3.3 (context management)
**Description:** A custom module that monitors conversation state and selectively compacts older turns while preserving key decisions, evaluation results, and dataset lineage. This is critical for sustaining 500-1,500 turn runs without losing track of prior context. Works in conjunction with `data-curation.md` as the durable complement -- the Context Manager handles in-memory compaction while `data-curation.md` provides persistent lineage that survives compaction cycles.
**Implementation requirements:** Turn history tracker; compaction logic that identifies which turns to summarize vs. preserve verbatim; preservation rules for key decisions, eval results, and dataset lineage; integration with LangGraph state; re-read capability for `data-curation.md` to reconstruct context after compaction.

---

## 6. Data Curation Markdown (`data-curation.md`)
**Design doc reference:** SS2.2, SS4.3
**Paper equivalent:** Paper SS4.3 (lineage tracking)
**Description:** A structured markdown file written to disk every iteration. Records dataset versions, composition ratios, quality-control decisions, per-iteration evaluation results, training configurations, failure taxonomies, iteration policy decisions, and hardware profiles. This is the primary lineage artifact that survives context compaction cycles. The agent re-reads it at the start of each round to reconstruct full prior context. It is what makes multi-round iteration reliable. The schema includes sections for: dataset (version, total examples, Dgold/Dhard breakdown, label distribution), training config (base model, LoRA rank, LR, epochs, batch size, parallel configs, best config selected), eval results (task type, metric, f(pi), per-class breakdown, per-slice scores, remaining failures, failure taxonomy), iteration policy decision (score band, next intervention, hypothesis), and hardware profile (model size, tier, reference chip, all four constraint checks, escalation status).
**Implementation requirements:** Writer function that produces the specified markdown schema; reader/parser that can extract structured data from existing `data-curation.md`; append-per-iteration logic; hardware profile section (theoretical values in Phase 1); integration with every node that produces results.

---

## 7. Tool Layer -- Four Named Tools
**Design doc reference:** SS2.3
**Paper equivalent:** Paper SS3.4 (tool layer)
**Description:** The agent has four named tools for cold-start mode: (1) `web_search` -- Exa deep research API for data acquisition and baseline survey only; (2) `bash` -- unrestricted shell with pre-loaded `slm_helpers.py`; (3) `read_file` -- reads datasets, configs, eval results, `data-curation.md`; (4) `edit_file` -- edits training scripts, configs, dataset files. `delegate_task` is a separate sub-agent spawning mechanism, not one of the four named tools. Production mode adds `query_traces` and `trace_analysis_subagent` and removes `web_search`.
**Implementation requirements:** LangGraph tool definitions for all four tools; Exa API integration for `web_search`; sandboxed bash execution with `slm_helpers.py` pre-loaded; file read/write tools with path access controls; tool schema definitions for the LLM.

---

## 8. Node 1 -- Task Analysis (Three Sub-Stages)
**Design doc reference:** SS2.4 Node 1
**Paper equivalent:** Paper SS4.1 (task analysis)
**Description:** Three sequential sub-stages executed before any data is touched: (1) Task classification -- parses `description` to determine task type, selects starting model from Android pool (always Tier 1 first -- smallest feasible model); (2) Data acquisition -- runs `web_search` to locate the relevant dataset, downloads actual benchmark data for known benchmarks, uses teacher model synthesis for custom tasks; (3) Baseline survey -- runs `web_search` to find published SOTA for the target task at the target model size class, calibrates accuracy target relative to published numbers, identifies known challenges that inform data curation (e.g., SMS Spam 87:13 class imbalance, ARC-Challenge format learning, CoNLL-2003 precision vs. recall asymmetry). Writes all decisions to `data-curation.md` before any training.
**Implementation requirements:** Task classification function; model selection logic that starts at Tier 1; Exa web_search integration for data acquisition; dataset download/parsing for known benchmarks (UCI SMS Spam, CoNLL-2003, GSM8K, ARC-Challenge, etc.); teacher model synthesis fallback; SOTA lookup via web search; accuracy target calibration; `data-curation.md` initial write.

---

## 9. Node 2 -- Eval Setup (Held-Out Evaluation Set Construction)
**Design doc reference:** SS2.4 Node 2, SS4.1
**Paper equivalent:** Paper SS4.2 (eval set construction)
**Description:** Constructs the held-out evaluation set `E = Epos U Eneg U Eboundary` before any training. Fixed throughout all iterations, never included in training data. The three slices are always disjoint and their meaning varies by task type. For classification: Epos = clear positive examples, Eneg = clear negative examples, Eboundary = confusable pairs at the class boundary. For NER: Epos = entity-rich passages, Eneg = entity-free passages (hallucination test), Eboundary = near-miss spans/overlapping types. For generation: Epos = well-formed problems, Eneg = adversarial/ill-posed inputs, Eboundary = multi-step/edge-case problems. Default stopping criterion: `f(pi) >= 0.96` on E, with agent option to lower if genuine capacity plateau detected.
**Implementation requirements:** Eval set builder per task type; stratified splitting logic; slice labeling (Epos/Eneg/Eboundary); immutability enforcement (eval set never modified after construction); configurable stopping threshold (default 0.96); plateau detection logic for threshold adjustment.

---

## 10. Node 3 -- Parallel Training
**Design doc reference:** SS2.4 Node 3
**Paper equivalent:** Paper SS4.4 (parallel training)
**Description:** Fires at least two configurations in parallel every iteration -- e.g., full fine-tune vs. LoRA, or two different base models. This is not optional; the paper does this on every iteration. Training always restarts from the base foundation model, never from a prior fine-tuned checkpoint. This makes rollback clean: reverting to a previous dataset fully restores the corresponding model behavior without needing to untangle accumulated fine-tuning steps.
**Implementation requirements:** Parallel training execution (at least 2 configs per iteration); configuration generation logic (vary LoRA vs. full fine-tune, different base models, different hyperparameters); always-from-base-model enforcement (no checkpoint chaining); parallel job management; result collection from multiple training runs.

---

## 11. Node 4 -- Evaluate
**Design doc reference:** SS2.4 Node 4
**Paper equivalent:** Paper SS4.5 (evaluation)
**Description:** Scores each trained configuration against held-out E. Records per-class F1 breakdown. Logs the node to the lineage DAG as `v = (pi, f(pi))` where `pi = (D, H, S)`. Selects the best-scoring configuration from parallel runs. Writes iteration result to `data-curation.md`. Logs hardware profile (theoretical in Phase 1). Outputs: overall metric score, per-class/per-entity breakdown, per-slice scores (Epos/Eneg/Eboundary).
**Implementation requirements:** Scoring function that runs `infer_batch` on eval set E; per-class/per-entity metric computation; per-slice (Epos/Eneg/Eboundary) breakdown; best-config selection from parallel runs; DAG node creation with lineage tracking; `data-curation.md` update; hardware profile logging (theoretical).

---

## 12. Node 5 -- Iterate (Diagnostic Policy)
**Design doc reference:** SS2.4 Node 5
**Paper equivalent:** Paper SS4.6 (iteration policy)
**Description:** Applies a shared iteration policy based on `f(pi)` score bands: (1) `< 0.80` = data problem -- rebuild dataset, analyze failures, identify gaps; (2) `0.80-0.95` = optimization problem -- tune hyperparameters (epochs, LR, LoRA rank, base model), hold dataset fixed; (3) `>= 0.95` = capacity/surgical -- add 2-3 targeted examples per remaining failure pattern only; (4) `f(pi+1) < f(pi)` = regression -- roll back immediately, do not compensate. After each iteration the agent analyzes remaining failures on E to determine whether the next intervention targets D (data), H (hyperparameters), or S (model selection).
**Implementation requirements:** Score band classifier; failure analysis function that inspects incorrect predictions on E; intervention router (data rebuild vs. hyperparameter tuning vs. surgical augmentation vs. rollback); failure taxonomy generator; hypothesis formulation for next modification; `data-curation.md` update with iteration policy decision.

---

## 13. Node 6 -- Curate (Curriculum Synthesis)
**Design doc reference:** SS2.4 Node 6, SS4.2
**Paper equivalent:** Paper SS4.3 (curriculum synthesis)
**Description:** Builds `Dcold = Dgold U Dhard` at 65:35 ratio. No replay buffer in cold-start (Dparent = empty; replay allocation redistributed to gold examples). Five quality controls applied to every dataset version with task-type-specific behavior: (1) 2-for-1 rule -- for each boundary example, one gold + one hard negative; (2) Label balancing -- no label exceeds 3x count of any other; (3) Context-length matching -- training lengths match realistic input distribution; (4) Entity diversification -- no single entity value appears >2-3x (NER only); (5) CoT annotation -- teacher model generates step-by-step reasoning chains (generation only, DeepSeek-R1 for math/science, GPT-4.1 for code/QA). Each label/entity type requires 3-5 distinct surface-text patterns. Target dataset sizes: 100-200 examples for classification/NER; 500-3,000 for generation.
**Implementation requirements:** Dataset builder with Dgold/Dhard composition at 65:35; 2-for-1 rule implementation per task type; label balance checker and enforcer; context-length distribution analyzer and matcher; entity diversification logic (NER); CoT annotation pipeline with teacher model integration (generation); Claude API integration for hard negative generation (classification/NER); surface-text pattern diversity checker; dataset size targets per task type.

---

## 14. Node 7 -- Rollback Check
**Design doc reference:** SS2.4 Node 7, SS5.3
**Paper equivalent:** Paper SS5.2 (rollback mechanism)
**Description:** Cold-start uses simple rollback only -- no dual-gate (that is production mode). If `f(pi+1) < f(pi)`, immediately revert to previous dataset and configuration. Do not compensate with more data. Cascading fixes that each introduce new failure modes are the primary cause of stagnation. The paper's SMS Spam run demonstrates this: V2 (17 additional examples) regressed from 0.9834 to a lower score; system rolled back to V1.
**Implementation requirements:** Score comparison between current and previous iteration; dataset version management (ability to revert to any prior version); configuration snapshot/restore; rollback logging in `data-curation.md`; no-compensation enforcement (rollback is final for that iteration, not a starting point for additional fixes).

---

## 15. Node 8 -- Escalate or Terminate
**Design doc reference:** SS2.4 Node 8
**Paper equivalent:** Paper SS5.3 (termination) -- escalation is SLM Factory addition
**Description:** Four termination/escalation conditions in priority order: (1) Terminate if `f(pi) >= 0.96` (or agent-adjusted threshold); (2) Terminate early if agent diagnoses a genuine plateau (capacity ceiling or irreducible ambiguity); (3) Escalate model if plateau persists >2 rounds without capacity diagnosis -- move to next Android pool tier, restart `task_analysis` with new model, carry forward best dataset version; (4) Terminate if 1,500 LangGraph turns exhausted. Escalation is hardware-aware even in Phase 1: agent checks theoretical latency/memory for the next tier model against `constraints.hardware_thresholds` before escalating. In Phase 1 this check is informational (logged but not blocking). In Phase 2 it becomes a hard gate.
**Implementation requirements:** Threshold check against target metric; plateau detection (2+ rounds without improvement); capacity ceiling diagnosis logic; model escalation function (select next tier model, restart task_analysis, preserve best dataset); hardware constraint check for escalation candidates (theoretical in Phase 1); turn counter with 1,500-turn limit; termination handler that produces final artifacts.

---

## 16. Android Model Pool (Tiered)
**Design doc reference:** SS6.4
**Paper equivalent:** SLM Factory addition -- not in paper
**Description:** A bounded, tiered set of models constrained to Android-deployable sizes. Tier 1 (fits anywhere, <=800MB INT4): Qwen3-0.6B (~397MB), Qwen3.5-0.8B (~500MB), MiniCPM5-1B (~500MB), MiniCPM4-0.5B (~300MB), Gemma3-1B (~529MB QAT), Llama3.2-1B (808MB), HRM-Text-1B (~600MB, research candidate, custom runtime). Tier 2 (fits most devices, 800MB-1.5GB INT4): SmolLM2-1.7B (~1.06GB), Qwen3-1.7B (~1.19GB), Qwen3.5-2B (~1.2GB), DeepSeek-R1-Distill-1.5B (~1.29GB), Gemma3n-E2B (~1.3GB). Tier 3 (generous upper bound, flagship only): Llama3.2-3B Q3_K_M (~1.4GB), Ministral-3B (<2GB). Agent starts at Tier 1 (smallest feasible model) and escalates only if accuracy target is unmet and hardware constraints permit.
**Implementation requirements:** Model pool data structure with tier assignments, INT4 sizes, and notes; model selection function that starts at Tier 1; tier escalation logic; hardware constraint filtering at model selection time; special-case handling for HRM-Text-1B (custom runtime, no llama.cpp).

---

## 17. Hardware Constraint Layer (Four Constraints)
**Design doc reference:** SS6.1, SS6.2, SS6.3
**Paper equivalent:** SLM Factory addition -- not in paper
**Description:** Four hardware constraints define the feasibility boundary for on-device Android deployment. (1) Storage `S(model) <= S_max`: INT4 quantized file size on disk, measurable before inference via `S ~ param_count x 4bits / 8`, enforced at model selection time. (2) Memory `M(model, chip) <= M_max`: peak RAM during inference `M ~ INT4_size x 1.2 + KV_cache(context_length)`, Android OOM is hard failure, M_max = device RAM minus ~1.5GB OS overhead, enforced at model selection time. (3) Latency `L(model, chip) <= L_max`: TTFT for interactive use, varies across chip generations, common threshold TTFT <= 2 seconds, Phase 1 theoretical from benchmarks. (4) Power `P(model, chip) <= P_max`: average watts during sustained inference, determines battery life impact, Phase 1 proxy from model size. The `target_chip` in constraints is the worst-case reference device.
**Implementation requirements:** Storage calculator (`param_count x 4bits / 8`); memory estimator (`INT4_size x 1.2 + KV_cache`); latency lookup table from benchmark data; power proxy from model size; constraint checker that evaluates all four against thresholds; `target_chip` reference device specification; Phase 1 logging (informational, not gating) for latency and power; Phase 1 gating for storage and memory at model selection.

---

## 18. Hardware-Aware Model Escalation
**Design doc reference:** SS6.3, SS2.4 Node 8
**Paper equivalent:** SLM Factory addition -- not in paper
**Description:** Before escalating to the next model tier, the system checks theoretical latency and power for that model on the reference chip against `constraints.hardware_thresholds`. In Phase 1 this check is informational (logged but not blocking). In Phase 2 it becomes a hard gate -- cannot escalate if any constraint would be violated. This enters the system at two points: (1) at model selection (Node 1) -- filter Android pool by S_max and theoretical M_max before the run begins; (2) at model escalation (Node 8) -- check theoretical latency and power for the candidate model.
**Implementation requirements:** Pre-escalation constraint evaluation function; theoretical hardware profile lookup for all pool models; logging of pass/fail per constraint; Phase 1: log-only mode; Phase 2: hard gate mode; integration with both Node 1 (initial selection) and Node 8 (escalation).

---

## 19. Hardware Profile Logging in data-curation.md
**Design doc reference:** SS4.3 (hardware profile section), SS6.5
**Paper equivalent:** SLM Factory addition -- not in paper
**Description:** Every iteration's `data-curation.md` entry includes a hardware profile section recording: model ID, INT4 size, tier, reference chip, and pass/fail status for all four constraints (storage, peak memory, TTFT, power) against their respective thresholds. Also records whether escalation is available, the next model candidate, and whether that candidate would pass constraints. Phase 1 uses theoretical/proxy values; Phase 2 replaces with measured values.
**Implementation requirements:** Hardware profile data structure; theoretical value calculators for all four constraints; pass/fail evaluator against thresholds; escalation feasibility checker; `data-curation.md` writer that includes the hardware profile section; Phase 1 theoretical value sources; Phase 2 measured value integration points.

---

## 20. `slm_helpers.py` -- Training Helper Library
**Design doc reference:** SS3.1
**Paper equivalent:** Paper SS3.4 (`tinker_helpers.py`)
**Description:** The project's equivalent of the paper's `tinker_helpers.py`. Three core functions pre-loaded into the agent's bash environment: (1) `train(dataset_path, base_model, nr_epochs, learning_rate, batch_size, lora_rank)` -- executes full LoRA loop via Unsloth (decoder) or HF PEFT (fallback), tokenizes with `apply_chat_template` for train/serve parity, loss computed on assistant tokens only via per-token loss weights, EOS truncation handled for base models, returns `weights_ref` string; (2) `infer(prompt, weights_ref, base_model, ...)` -- loads checkpoint, generates text, no deployment step required; (3) `infer_batch(prompts, weights_ref, ..., max_workers=20)` -- parallel inference via ThreadPoolExecutor, used by eval harness for concurrent evaluation of all held-out E examples.
**Implementation requirements:** `train()` function with Unsloth integration, HF PEFT fallback, chat template tokenization, assistant-token-only loss, EOS handling; `infer()` function with checkpoint loading and text generation; `infer_batch()` with ThreadPoolExecutor (max_workers=20); weights_ref return/consumption protocol; pre-loading into bash environment.

---

## 21. Training Backend (Multi-Framework)
**Design doc reference:** SS3.2
**Paper equivalent:** Paper SS3.4 (training SDK -- Tinker SDK replaced with open source)
**Description:** Three training frameworks: (1) Unsloth for decoder LoRA (Qwen3, Llama3.2, MiniCPM, Gemma3) on L40; (2) HF PEFT + Transformers as fallback and for HRM-Text (custom architecture); (3) Full fine-tune path when agent selects over LoRA, on A100 if needed. Target training time: 10-30 min per run for decoder models. L40 for Phase 1 to preserve H200s/A100s. LoRA hyperparameter search space: lora_rank (4, 8, 16, 32, 64), learning_rate (1e-4 knowledge-intensive or 2e-4 format-learning), nr_epochs (1-8 with train vs. validation divergence monitoring), batch_size (agent-selected, paper found 16 optimal for some tasks), target_modules (all-linear for decoders).
**Implementation requirements:** Unsloth LoRA training integration; HF PEFT fallback path; full fine-tune path; GPU resource selection logic (L40 vs. A100); hyperparameter search space definition; training time monitoring; train vs. validation divergence detection for epoch selection.

---

## 22. Task-Type-Aware Eval Harness
**Design doc reference:** SS3.3
**Paper equivalent:** Paper Appendix D.2 (agent writes own eval code)
**Description:** The agent writes its own eval code per task (per paper Appendix D.2). The harness is task-type-aware: classification uses binary F1 or accuracy via exact match on predicted label string; NER uses entity F1 via span extraction + entity-type match; generation uses LLM-as-judge score [0,1] via Claude API judging prediction against gold answer. All three types output: overall metric score, per-class/per-entity breakdown, per-slice scores (Epos/Eneg/Eboundary).
**Implementation requirements:** Eval harness framework that the agent can extend per task; exact-match label extraction for classification; span extraction + entity-type matching for NER; Claude API LLM-as-judge integration for generation; per-class and per-slice metric computation; standardized output format across all task types; `infer_batch` integration for running eval set.

---

## 23. Lineage DAG (Sequential Greedy)
**Design doc reference:** SS2.5
**Paper equivalent:** Paper SS5.1 (MCGS -- simplified to sequential greedy for Phase 1)
**Description:** Each iteration is a node `v_i = (pi_i, f(pi_i))` with edges `(v_i, v_j)` indicating pi_j was derived from pi_i via a targeted modification. The agent selects the best current node, proposes a hypothesis-driven modification (EXPAND operator), trains it, evaluates it. The DAG records lineage so the agent can attribute accuracy changes to specific interventions (data composition, hyperparameters, learning strategy). Phase 1 uses sequential greedy -- not full MCGS (branching, UCT selection, cross-branch fusion deferred to Phase 2+ for harder tasks).
**Implementation requirements:** DAG data structure for iteration nodes; node creation with (pipeline_config, score) tuples; edge tracking for lineage; EXPAND operator (hypothesis-driven modification proposal); attribution logic linking accuracy changes to specific interventions; sequential greedy traversal; persistence to disk for cross-compaction survival.

---

## 24. Scoring Function `f(pi)`
**Design doc reference:** SS2.5, SS5.1
**Paper equivalent:** Paper SS5.1 (scoring function)
**Description:** Maps pipeline `pi = (D, H, S)` to held-out validation performance by executing the full train-then-evaluate loop. For SMS Spam: binary F1. No surrogates -- every node score requires actual training and inference on E. The optimization objective for cold-start is `pi* = argmax f(pi)` (unconstrained). Production mode adds the regression constraint `r(pi; R) <= epsilon` where epsilon = 2 (absolute count, not relative rate).
**Implementation requirements:** Scoring function that orchestrates train-then-evaluate; no surrogate/approximation shortcuts; F1/accuracy computation for classification; entity F1 for NER; LLM-as-judge for generation; score storage in DAG nodes.

---

## 25. Regression Function `r(pi)`
**Design doc reference:** SS2.5
**Paper equivalent:** Paper SS5.1 (regression function)
**Description:** Counts previously correct examples the new model answers incorrectly on held-out regression set `R subset X`. In production mode (Phase 2+), the optimization objective includes the constraint `r(pi; R) <= epsilon` where epsilon = 2. Phase 1 uses simple rollback instead of formal regression gating. The regression function is defined but not enforced as a hard constraint in Phase 1.
**Implementation requirements:** Regression set construction (subset of eval set or separate); per-example tracking of correct/incorrect across iterations; regression count computation; Phase 1: simple score comparison for rollback; Phase 2: formal epsilon=2 constraint enforcement.

---

## 26. Convergence Criteria (Priority-Ordered)
**Design doc reference:** SS2.5, SS5.4, SS5.5
**Paper equivalent:** Paper SS5.3 (convergence)
**Description:** Four convergence criteria in priority order: (1) `f(pi) >= 0.96` on E -- promote checkpoint; (2) Irreducible plateau detected -- lower target, terminate early; (3) 1,500 LangGraph turns exhausted -- terminate with best checkpoint; (4) All Android pool tiers exhausted without meeting target -- report capacity ceiling. Success requires: metric improves across >=3 consecutive rounds, at least one rollback occurs and is handled correctly, full run <8 hours at <$50, all models within Android pool bounds, complete `data-curation.md` lineage.
**Implementation requirements:** Threshold checker (configurable, default 0.96); plateau detector (2+ rounds without improvement); turn counter with 1,500 limit; tier exhaustion detector; success criteria validator; cost tracking; time tracking; rollback occurrence tracker.

---

## 27. Stagnation Detection and Handling
**Design doc reference:** SS5.4
**Paper equivalent:** Paper SS5.3 (stagnation handling)
**Description:** Stagnation detected when 2+ consecutive rounds show no improvement. Two responses: (1) Terminate early if agent diagnoses capacity ceiling (remaining failures reflect model size limitations, not fixable training gaps); (2) Model escalation -- upgrade to next Android pool tier, restart `task_analysis` with new model, carry forward best dataset version. The capacity ceiling diagnosis is an LLM reasoning step where the agent examines failure patterns and determines whether they are addressable with more/better data or fundamentally beyond the model's capacity.
**Implementation requirements:** Consecutive no-improvement counter; capacity ceiling diagnosis prompt/logic; escalation trigger; dataset version carry-forward during escalation; early termination handler.

---

## 28. Held-Out Eval Set (Three-Slice Structure)
**Design doc reference:** SS4.1
**Paper equivalent:** Paper SS4.2 (eval set)
**Description:** `E = Epos U Eneg U Eboundary` with task-type-specific semantics. For classification (e.g. SMS Spam): Epos = genuine spam stratified across surface-form range (UCI test split), Eneg = clear legitimate messages with no label ambiguity (UCI ham split), Eboundary = confusable pairs at the boundary (promotional messages, marketing, appointment reminders with links). For NER (e.g. CoNLL-2003): Epos = entity-rich passages with gold span annotations, Eneg = entity-free passages (hallucination test), Eboundary = overlapping or near-miss entity types. For generation (e.g. ARC-Challenge, GSM8K): Epos = well-formed problems with unambiguous answers, Eneg = adversarial/ill-posed inputs, Eboundary = multi-step or edge-case problems. Fixed before training, never modified.
**Implementation requirements:** Per-task-type eval set construction logic; UCI SMS Spam dataset integration (test/ham splits); CoNLL-2003 dataset integration; ARC-Challenge/GSM8K dataset integration; boundary example identification/generation; immutability enforcement; slice-level metric computation.

---

## 29. Curriculum Composition (Dgold/Dhard at 65:35)
**Design doc reference:** SS4.2
**Paper equivalent:** Paper SS4.3 (curriculum synthesis)
**Description:** Training data `Dcold = Dgold U Dhard` at 65:35 ratio. Dgold = correct labeled examples from downloaded benchmark data, balanced across labels/entity types (no label >3x any other), 3-5 distinct surface-text patterns per label. Dhard = hard negatives generated by agent using 2-for-1 rule with method varying by task type: classification uses Claude API to generate counterexample with similar surface form but opposite label; NER uses Claude API for near-miss spans; generation uses DeepSeek-R1 (math/science) or GPT-4.1 (code/QA) for plausible but incorrect answers. No replay buffer in cold-start (Dparent = empty). Target sizes: 100-200 examples for classification/NER; 500-3,000 for generation.
**Implementation requirements:** Dataset composition logic with 65:35 ratio enforcement; Dgold construction from benchmark data; label balance checker (no label >3x); surface-text pattern diversity checker (3-5 per label); Claude API integration for hard negative generation (classification/NER); DeepSeek-R1 and GPT-4.1 integration for generation hard negatives; dataset size management per task type.

---

## 30. Five Quality Controls for Dataset Curation
**Design doc reference:** SS2.4 Node 6
**Paper equivalent:** Paper SS4.3 (quality controls)
**Description:** Five quality controls applied to every dataset version: (1) 2-for-1 rule -- for each boundary example, include one gold example + one hard negative with similar input but different correct label/span/answer; (2) Label balancing -- no label exceeds 3x count of any other (classification/NER); (3) Context-length matching -- training example lengths match realistic input distribution for the task; (4) Entity diversification -- no single entity value appears >2-3x, replace with synthetic equivalents (NER only); (5) CoT annotation -- teacher model generates step-by-step reasoning chains using DeepSeek-R1 for math/science or GPT-4.1 for code/QA (generation only).
**Implementation requirements:** 2-for-1 rule enforcer with hard negative generator; label distribution analyzer and rebalancer; context-length distribution analyzer and filter; entity frequency counter and synthetic replacement generator; CoT annotation pipeline with teacher model selection; quality control validation that runs all five checks before finalizing any dataset version.

---

## 31. Hard Negative Generation (Task-Type-Specific)
**Design doc reference:** SS4.2
**Paper equivalent:** Paper SS4.3 (hard negatives)
**Description:** Hard negatives are generated differently per task type, always using the 2-for-1 rule. Classification: Claude API generates a counterexample with similar surface form but opposite label (no CoT needed). NER: Claude API generates near-miss span with same surface form but different or absent entity type. Generation: teacher model generates plausible but incorrect answer (wrong reasoning path) using DeepSeek-R1 for math/science and GPT-4.1 for code/QA. Hard negatives compose 35% of the training set (the Dhard portion).
**Implementation requirements:** Claude API prompt templates for classification hard negatives; Claude API prompt templates for NER hard negatives; DeepSeek-R1 integration for math/science hard negatives; GPT-4.1 integration for code/QA hard negatives; task-type router for hard negative generation method; quality validation of generated hard negatives.

---

## 32. Exa Web Search Integration
**Design doc reference:** SS2.3, SS2.4 Node 1
**Paper equivalent:** Paper SS3.4 (web search tool)
**Description:** Exa deep research API used for two purposes in cold-start: (1) data acquisition -- locating and downloading relevant datasets for the task; (2) baseline survey -- finding published SOTA for the target task at the target model size class to calibrate accuracy targets. Removed in production mode (replaced by `query_traces`).
**Implementation requirements:** Exa API client; search query formulation for dataset discovery; search query formulation for SOTA lookup; result parsing and dataset download; API key management; rate limiting.

---

## 33. Chat Template Tokenization (Train/Serve Parity)
**Design doc reference:** SS3.1
**Paper equivalent:** Paper SS3.4 (training details)
**Description:** The `train()` function tokenizes with `apply_chat_template` to ensure train/serve parity -- the same tokenization format used during training is used during inference. Loss is computed on assistant tokens only via per-token loss weights (the model is not penalized for the prompt/system tokens). EOS truncation is handled for base models that do not naturally stop at end-of-turn markers.
**Implementation requirements:** `apply_chat_template` integration in training pipeline; per-token loss weight computation (mask non-assistant tokens); EOS truncation detection and handling for base models; consistency enforcement between train-time and inference-time tokenization.

---

## 34. Expected Iteration Trajectory
**Design doc reference:** SS5.2
**Paper equivalent:** Paper SS5.2 (iteration trajectory)
**Description:** From the paper: the first iteration captures only 40-70% of achievable improvement. Remaining gains come from diagnosing failure modes and applying targeted interventions. The diagnosis phase consumes more agent reasoning turns than the training phase. Expected SMS Spam trajectory: Round 0: ~0.70-0.85 F1 (2+ parallel initial configs, data rebuild or hyperparameter tuning); Round 1: ~0.95-0.98 (surgical, identify recall vs. precision failures); Round 2: ~0.98-0.99 (precision repair via hard ham examples); Round 3: ~0.997 (targeted augmentation, regression may trigger rollback); Final: >=0.96 or rollback.
**Implementation requirements:** This is a behavioral expectation, not a code component. However, it informs: progress tracking against expected trajectory; logging of actual vs. expected scores per round; alert if trajectory deviates significantly from expectations.

---

## 35. Cost Tracking and Budget Enforcement
**Design doc reference:** Appendix B, SS1 (success criteria)
**Paper equivalent:** Paper Table 9 (cost breakdown)
**Description:** Total cost per Phase 1 run must be <$50. Estimated breakdown: Claude Sonnet 4.6 API (4M tokens per 12hr run, 75/25 input/output) ~$24; training compute (L40, 4-5 training runs per full loop) ~$5-15; Exa web_search API ~$1-2. Total estimated: ~$30-40. SMS Spam expected closer to $30 than $50 (simpler than ARC-Challenge at $36; CLINC150 production run was $3.48).
**Implementation requirements:** Token usage tracker for Claude API calls; compute cost tracker for GPU time; Exa API call counter; running cost total; budget threshold alert at approach to $50; cost logging in run summary.

---

## 36. Phase 1 to Phase 2 Handoff Artifacts
**Design doc reference:** SS7
**Paper equivalent:** SLM Factory addition -- not in paper (Phase 2 transition planning)
**Description:** Five artifacts Phase 1 must produce for Phase 2: (1) Deployable model artifact -- best checkpoint quantized to INT4, represented as `(best_dataset_version, base_model_id, training_config)` not just weights, becomes Phase 2's deployed model M0; (2) Complete `data-curation.md` lineage -- Phase 2's parent model awareness stage reads this to recover D_parent; (3) Inference logging infrastructure -- PostgreSQL/Supabase schema with per-user RLS, record format `(input, prediction, corrected_output, verdict, judge_reasoning, judge_metadata)`, designed alongside Phase 1; (4) LLM-as-judge config -- fixed prompt template and criteria stored in inference table metadata, must be reproducible; (5) Hardware profiling baseline -- run Phase 1 best checkpoint through Qualcomm AI Hub profiling on reference chip for real latency/memory/power numbers.
**Implementation requirements:** Checkpoint export pipeline (best weights + full recipe); `data-curation.md` completeness validator; inference database schema design (PostgreSQL/Supabase); LLM-as-judge prompt template storage; Qualcomm AI Hub profiling integration (post-Phase 1).

---

## 37. Model-Specific Handling: HRM-Text-1B
**Design doc reference:** SS6.4 (Tier 1 table)
**Paper equivalent:** SLM Factory addition -- not in paper
**Description:** HRM-Text-1B (sapientinc/HRM-Text-1B) is in the Android model pool as a research candidate at ~600MB INT4 in Tier 1. Requires custom handling: `token_type_ids` must be set, FlashAttention must be disabled at inference, standard llama.cpp/Ollama will not load it. On-device export deferred to Phase 2+. Training uses HF PEFT + Transformers (not Unsloth) due to its custom architecture.
**Implementation requirements:** HRM-Text-1B special-case in model pool metadata; `token_type_ids` setting in inference pipeline; FlashAttention disable flag; HF PEFT training path (not Unsloth); on-device export exclusion in Phase 1; documentation of known limitations.

---

## 38. Decoder-Only Architecture Constraint
**Design doc reference:** SS1
**Paper equivalent:** SLM Factory addition -- paper used GLiNER2 encoder
**Description:** Phase 1 uses decoder models from the Android pool exclusively. The paper used GLiNER2 (an encoder model) for SMS Spam, but GLiNER2 is not in the Android pool. All models in the pool are decoder-only architectures. This means classification tasks that the paper solved with encoder models must be solved with decoder models, which may require different prompting strategies (e.g., instruct-format prompts with explicit label extraction rather than direct encoder classification).
**Implementation requirements:** Instruct-format prompt templates for classification tasks; label extraction from decoder model output (exact match on predicted label string); no encoder model paths in Phase 1; prompt engineering for classification performance with decoder models.

---

## 39. Filesystem-Based Context Isolation
**Design doc reference:** SS2.2
**Paper equivalent:** Paper SS3.3 (context isolation)
**Description:** Sub-agents share the filesystem with the main agent. They write summaries to disk; the main agent reads those files without loading raw data into its own context window. This is the core context isolation pattern -- it prevents the main agent's context window from being consumed by raw training data, eval results, or intermediate computations. Combined with the Context Manager's compaction and `data-curation.md`'s persistent lineage, this enables multi-hundred-turn runs.
**Implementation requirements:** Defined filesystem locations for sub-agent outputs; summary file format specification; main agent file reading integration; context window budget tracking; sub-agent output cleanup/archival.

---

## 40. Parallel Configuration Comparison
**Design doc reference:** SS2.4 Node 3
**Paper equivalent:** Paper SS4.4 (parallel configs)
**Description:** Every training iteration fires at least two configurations in parallel. This is mandatory, not optional. Example configurations to compare: full fine-tune vs. LoRA, two different base models, different LoRA ranks, different learning rates. The evaluate node then selects the best-scoring configuration from the parallel runs. This ensures the agent explores the configuration space efficiently rather than committing to a single hypothesis per iteration.
**Implementation requirements:** Parallel job launcher for multiple training configs; configuration generation logic (systematic variation of one or more hyperparameters); result collection and comparison; best-config selection; logging of all configs tried (not just the winner) in `data-curation.md`.

---

## 41. Phase 1 Hardware Constraint Mode (Log-Only)
**Design doc reference:** SS6.5
**Paper equivalent:** SLM Factory addition -- not in paper
**Description:** Phase 1 seeds all four hardware constraint values (storage, memory, latency, power) in `data-curation.md` every iteration but operates in two modes: (1) Storage and memory are calculated/estimated and gate model selection (models that exceed S_max or M_max are excluded); (2) Latency and power use theoretical/proxy values that are logged but not gating. Phase 2 replaces theoretical values with measured values and activates all four as hard gates.
**Implementation requirements:** Storage calculation (gates model selection in Phase 1); memory estimation (gates model selection in Phase 1); latency lookup from benchmarks (log-only in Phase 1); power proxy from model size (log-only in Phase 1); dual-mode constraint system (gating vs. informational); `data-curation.md` integration for all four values every iteration.
