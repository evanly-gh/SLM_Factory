# Pioneer Agent Paper Features

Exhaustive feature list extracted from "Pioneer Agent: Continual Improvement of Small Language Models in Production" (arXiv:2604.09791, Fastino Labs, April 2026).

---

## 1. LangGraph State Machine Orchestration
**Paper reference:** Section 2.1 (Architecture)
**Description:** The entire fine-tuning loop is orchestrated as a LangGraph state machine driven by Claude Sonnet 4.6 with a 1M token context window and 32K extended thinking budget. The state machine encodes the full lifecycle: data curation, training, evaluation, diagnosis, and iteration. Each node in the graph corresponds to a distinct phase of the pipeline, and transitions are governed by the agent's decisions based on eval results and diagnostic signals.
**Implementation requirements:** A LangGraph graph definition with typed state, node functions for each pipeline phase (curate, train, eval, diagnose), conditional edges for routing based on scores, and Claude Sonnet as the LLM backend with extended thinking enabled.

## 2. Modal Sandboxes for Isolated Execution
**Paper reference:** Section 2.1 (Architecture)
**Description:** Each training/eval run executes inside a Modal sandbox provisioned with 16GB of memory and a 24-hour timeout. This provides reproducible, isolated compute environments that prevent state leakage between runs and allow the agent to safely execute arbitrary code (training scripts, eval harnesses) without affecting the host system.
**Implementation requirements:** Modal SDK integration, sandbox configuration (memory limits, timeouts, GPU attachment), a wrapper that submits jobs to Modal and retrieves results, error handling for sandbox timeouts and OOM conditions.

## 3. Dual Model Family Support (Encoder + Decoder)
**Paper reference:** Section 2.1 (Architecture)
**Description:** The system supports two distinct model families: GLiNER2 (encoder-based, for classification and NER tasks) and Qwen/Llama (decoder-based, for generation tasks). The agent selects the appropriate family based on task classification. This dual-track design means the training pipeline, data formatting, and eval harness must handle fundamentally different model architectures.
**Implementation requirements:** Model registry with family metadata, task-to-family mapping logic, separate training pipeline branches for encoder vs. decoder models, architecture-aware data formatting (e.g., chat templates for decoders, span annotations for encoders).

## 4. Tinker SDK for LoRA Fine-Tuning with Instant Inference
**Paper reference:** Section 2.1 (Architecture)
**Description:** Fine-tuning uses the Tinker SDK, which provides LoRA (Low-Rank Adaptation) training with an "instant inference" capability -- the ability to run inference on a freshly trained adapter without a separate deployment step. This tight train-then-infer loop is critical for the agent's rapid iteration cycle.
**Implementation requirements:** Tinker SDK client, LoRA configuration management (rank, alpha, target modules), training launch function, inference function that loads base model + adapter on the fly, result parsing.

## 5. Hierarchical Agent Architecture (Main Agent + Sub-Agents)
**Paper reference:** Section 2.1 (Architecture)
**Description:** The system uses a hierarchical agent design: a main orchestrator agent that can spawn sub-agents via the `delegate_task` tool. Sub-agents share the filesystem with the main agent and write their results to disk as summaries. This enables parallel execution of independent tasks (e.g., training two configurations simultaneously) and offloading of specialized analysis.
**Implementation requirements:** `delegate_task` tool implementation, sub-agent lifecycle management (spawn, monitor, collect results), shared filesystem conventions, summary file format, concurrency control.

## 6. Trace Analyzer Sub-Agent
**Paper reference:** Section 2.1 (Architecture)
**Description:** A specialized sub-agent with an approximately 100K output-token limit dedicated to SQL-style analysis of production inference traces. It receives trace data and performs complex analytical queries to identify failure patterns, cluster errors, and compute statistics that inform the main agent's diagnostic decisions.
**Implementation requirements:** Sub-agent definition with elevated output token limit, SQL query generation capability, trace data schema definition, result summarization logic, integration with the `query_traces` tool.

## 7. Context Manager (Conversation Compaction)
**Paper reference:** Section 2.1 (Architecture)
**Description:** A custom module that monitors the conversation state and compacts older turns while preserving key decisions, eval results, and dataset lineage information. This is critical for sustaining 500-1,500 turn runs without the agent losing track of its history. The compaction must be selective: routine tool outputs can be summarized, but score progressions, dataset composition decisions, and rollback events must be preserved verbatim.
**Implementation requirements:** Turn-level importance scoring, selective compaction algorithm, preservation rules for critical information types (scores, decisions, lineage), token budget tracking, compacted summary generation, integration with LangGraph state.

## 8. Persistent Structured Log (data-curation.md)
**Paper reference:** Section 2.1 (Architecture)
**Description:** A persistent structured markdown file (`data-curation.md`) that the agent writes to every iteration, recording dataset composition, curation decisions, and rationale. This log survives context compaction cycles, serving as the agent's long-term memory for data lineage. The agent reads it at the start of each iteration to reconstruct its data history.
**Implementation requirements:** Markdown template with structured sections (dataset composition, changes, rationale), read/write logic integrated into the curation node, append-friendly format, parsing logic to extract structured data from the log.

## 9. Cold-Start Tool Set (4 Tools)
**Paper reference:** Section 2.5 (Tools)
**Description:** In cold-start mode, the agent has access to exactly four tools: `web_search` (via Exa API for finding datasets and baselines), `bash` (with pre-loaded helper functions for training and eval), `read_file` (for reading datasets, configs, and eval results), and `edit_file` (for modifying configs and data files). This minimal tool set forces the agent to operate through composable primitives.
**Implementation requirements:** Exa API client wrapped as `web_search` tool, bash tool with pre-loaded `tinker_helpers.py`, `read_file` tool with path validation, `edit_file` tool with diff-based editing, tool registration with LangGraph.

## 10. Production Tool Set (Adds query_traces + trace_analysis_subagent, Removes web_search)
**Paper reference:** Section 2.5 (Tools)
**Description:** In production mode, the tool set changes: `web_search` is removed (the agent should not be searching the web during production improvement cycles), and two new tools are added: `query_traces` (a SQL + bash pipeline for querying production inference logs) and `trace_analysis_subagent` (spawns the specialized Trace Analyzer sub-agent). This reflects the shift from data acquisition to failure analysis.
**Implementation requirements:** `query_traces` tool implementation (SQL parser, trace database schema, bash pipeline for complex queries), trace_analysis_subagent spawner, mode-aware tool registration that swaps tool sets based on operational mode.

## 11. delegate_task Tool (Sub-Agent Spawning)
**Paper reference:** Section 2.5 (Tools)
**Description:** A tool that spawns sub-agents which share the main agent's filesystem. Sub-agents execute independently and write summaries to disk for the main agent to consume. This enables parallel execution of training configurations, concurrent eval runs, and offloading of analysis tasks without blocking the main agent's control flow.
**Implementation requirements:** Sub-agent spawn function with task description parameter, filesystem path conventions for summary output, polling or callback mechanism for completion detection, result file parsing, error handling for sub-agent failures.

## 12. Training Pipeline Tuple (D, H, S)
**Paper reference:** Section 2.2 (Search Procedure)
**Description:** Each training configuration is formalized as a pipeline tuple pi = (D, H, S) where D is the dataset specification (composition, size, mixing ratios), H is the hyperparameter configuration (learning rate, epochs, batch size, LoRA rank), and S is the learning strategy (e.g., curriculum ordering, multi-stage training, CoT supervision). This decomposition allows the agent to reason about and modify each component independently.
**Implementation requirements:** Data classes or typed dicts for D, H, and S components, serialization/deserialization for persistence, comparison operators for diffing configurations, a unified pipeline executor that accepts (D, H, S) tuples.

## 13. Full Train-Then-Evaluate Scoring (No Surrogates)
**Paper reference:** Section 2.2 (Search Procedure)
**Description:** The scoring function f(pi) is computed by actually training the model with configuration pi and then evaluating it on the held-out eval set. There are no surrogate models, early stopping heuristics, or proxy metrics -- every score is the real thing. This is expensive but eliminates the risk of optimizing for a proxy that diverges from true performance.
**Implementation requirements:** End-to-end training execution, eval harness that runs on the held-out set, score computation (accuracy, F1, or task-specific metric), result storage with full provenance (which config, which eval set, which checkpoint).

## 14. Monte Carlo Graph Search (MCGS) DAG
**Paper reference:** Section 2.2 (Search Procedure)
**Description:** Training attempts are organized as a directed acyclic graph (DAG) G=(V,E) where each node v=(pi, f(pi)) represents a configuration and its score, and edges represent derivation relationships (child was derived from parent by modifying some component). This DAG structure enables the agent to attribute accuracy changes to specific interventions and to select promising branches for further exploration using tree search principles.
**Implementation requirements:** DAG data structure with node and edge types, node creation on each training attempt, edge creation linking child to parent, traversal algorithms (path to root, subtree enumeration), serialization for persistence across compaction cycles.

## 15. EXPAND Operator (Hypothesis-Driven Child Generation)
**Paper reference:** Section 2.2 (Search Procedure)
**Description:** The EXPAND operator uses the LLM to propose a new child configuration from a parent node's trajectory. The proposal is hypothesis-driven: the agent examines the parent's eval results, identifies specific failure patterns, and generates a targeted modification to the (D, H, S) tuple that addresses those failures. This is not random perturbation -- it is reasoned intervention.
**Implementation requirements:** Prompt template that presents parent trajectory (config, scores, failure analysis) to the LLM, structured output parsing for the proposed modification, validation that the proposed config differs from existing nodes, node creation and edge insertion into the MCGS DAG.

## 16. UCT-Like Selection with Time-Decaying Exploration
**Paper reference:** Section 2.2 (Search Procedure)
**Description:** Node selection uses a UCT (Upper Confidence Bound for Trees) formula: UCT(v) = f_bar(v) + c(t) * sqrt(ln(N) / n_i), where f_bar(v) is the average score of node v's subtree, N is total visits, n_i is visits to node i, and c(t) is a time-decaying exploration coefficient. Early iterations favor exploration (high c), while later iterations converge toward exploitation of the best-performing branches.
**Implementation requirements:** UCT score computation function, visit count tracking per node, exploration coefficient schedule (c as a function of iteration number t), node selection algorithm that computes UCT for all leaf-adjacent nodes and selects the maximum.

## 17. Top-K Exploitation in Later Iterations
**Paper reference:** Section 2.2 (Search Procedure)
**Description:** In later iterations (when the exploration coefficient has decayed), the agent shifts to a Top-K exploitation strategy: it selects from the K highest-scoring nodes rather than using UCT. This focuses compute on refining the most promising configurations rather than exploring new territory, which is appropriate when the search space has been sufficiently explored.
**Implementation requirements:** Configurable transition point (iteration number or score threshold) from UCT to Top-K, Top-K selection function, K parameter tuning, integration with the main iteration loop.

## 18. Cross-Branch FUSION Operator
**Paper reference:** Section 2.2 (Search Procedure)
**Description:** The FUSION operator merges complementary strategies from independent branches of the search DAG. For example, if one branch discovered an effective data composition and another found optimal hyperparameters, FUSION creates a new node that combines both. This enables the agent to synthesize insights across parallel exploration paths.
**Implementation requirements:** Branch comparison logic that identifies complementary strengths, merge function for (D, H, S) tuples that combines components from different parents, multi-parent edge support in the DAG, conflict resolution when merged components are incompatible.

## 19. Stagnation Recovery (Evolution or Fusion)
**Paper reference:** Section 2.2 (Search Procedure)
**Description:** When the search stagnates (no improvement over recent iterations), the agent triggers recovery mechanisms: either evolution (trajectory-aware mutation that makes larger, more exploratory changes informed by the full history of what has been tried) or fusion (combining strategies from different branches). This prevents the search from getting stuck in local optima.
**Implementation requirements:** Stagnation detection (e.g., no score improvement over N iterations), evolution operator that generates mutated configs with awareness of the full trajectory, fusion operator (see feature 18), selection logic to choose between evolution and fusion, integration with the main iteration loop.

## 20. Score Backpropagation Through Ancestors
**Paper reference:** Section 2.2 (Search Procedure)
**Description:** After each training attempt, the resulting score is backpropagated through ancestor nodes in the DAG, updating their aggregate statistics (average score, visit count). This is analogous to MCTS backpropagation and ensures that the UCT selection formula has accurate statistics for all nodes, not just leaf nodes.
**Implementation requirements:** Backpropagation function that traverses from a newly scored node to the root, updating running averages and visit counts at each ancestor, thread-safe updates if parallel training is in use.

## 21. Dataset Composition Ratios (Gold / Hard Negatives / Replay)
**Paper reference:** Section 2.3 (Data Curation)
**Description:** Training datasets are composed of three categories with target ratios: Gold examples (40-60% of the dataset, clean correctly-labeled examples), Hard negatives (25-35%, examples near decision boundaries or from confusable categories), and Replay buffer (10-20%, sampled from the parent model's training data, production mode only). These ratios are tunable parameters in the pipeline.
**Implementation requirements:** Dataset composer that accepts examples tagged by category (gold, hard_negative, replay), ratio enforcement logic, sampling/padding to hit target ratios, validation that ratios sum to 100%.

## 22. Cold-Start Dataset Composition (No Replay)
**Paper reference:** Section 2.3 (Data Curation)
**Description:** In cold-start mode, the dataset formula simplifies to D_cold = D_gold union D_hard at approximately 65:35 ratio. There is no replay buffer because there is no parent model (D_parent = empty set). This simpler composition reflects the fact that cold-start is building from scratch rather than improving an existing deployment.
**Implementation requirements:** Mode-aware dataset composition logic that omits replay sampling in cold-start mode, adjusted ratio targets (65:35 instead of the three-way split), validation that replay buffer is empty in cold-start.

## 23. 2-for-1 Rule (Boundary Case Pairing)
**Paper reference:** Section 2.3 (Data Curation)
**Description:** For each boundary case identified in evaluation, the agent creates a pair: one gold example (clearly correct) and one hard negative (the confusable alternative). This ensures the model sees both what something IS and what it is NOT for every difficult distinction. The pairing is systematic, not ad hoc.
**Implementation requirements:** Boundary case identification from eval failures, gold example generation or retrieval for each boundary case, hard negative generation (e.g., by modifying the boundary case to change the correct label), pair tracking to ensure 1:1 correspondence.

## 24. Label Balancing Constraint
**Paper reference:** Section 2.3 (Data Curation)
**Description:** No label in the training dataset may exceed 3x the count of any other label. This prevents the model from learning frequency-based shortcuts and ensures adequate representation of all classes. The agent must actively monitor and enforce this constraint during data curation.
**Implementation requirements:** Label frequency counter, imbalance detection (max_count > 3 * min_count), rebalancing strategies (oversample minority labels, undersample majority labels, or generate synthetic examples for underrepresented labels), validation check before training.

## 25. Context-Length Matching
**Paper reference:** Section 2.3 (Data Curation)
**Description:** Training example lengths must match the realistic input distribution expected at inference time. If production inputs are typically 50-200 tokens, training examples should span that same range, not be uniformly short or uniformly long. This prevents train-serve skew in the model's length-dependent behavior.
**Implementation requirements:** Input length distribution analysis (from production traces or representative samples), length histogram computation for training data, resampling or generation to match the target distribution, length distribution comparison metric.

## 26. Entity Diversification (NER-Specific)
**Paper reference:** Section 2.3 (Data Curation)
**Description:** For Named Entity Recognition tasks, no single entity value should appear more than 2-3x as frequently as others. If "John Smith" appears 50 times but other person names only appear 5 times, the model may memorize "John Smith" as an entity rather than learning the pattern. The agent replaces overrepresented entities with synthetic equivalents to ensure diversity.
**Implementation requirements:** Entity frequency counter per entity type, overrepresentation detection, entity replacement logic (swap "John Smith" with other plausible person names), replacement tracking to maintain annotation consistency, NER-specific data augmentation pipeline.

## 27. Chain-of-Thought Annotation (Generation Tasks)
**Paper reference:** Section 2.3 (Data Curation)
**Description:** For generation tasks, a teacher model generates step-by-step chain-of-thought (CoT) reasoning that is prepended to the target output. The teacher model is selected by domain: DeepSeek-R1 for math and science reasoning, GPT-4.1 for code and QA tasks. This CoT supervision teaches the student model to reason before answering.
**Implementation requirements:** Teacher model API clients (DeepSeek-R1, GPT-4.1), domain classification to select the appropriate teacher, CoT generation prompts per domain, CoT+answer concatenation for training targets, quality filtering for generated CoT (reject incoherent reasoning).

## 28. Surface-Text Pattern Diversity (3-5 Patterns Per Label)
**Paper reference:** Section 2.3 (Data Curation)
**Description:** Each label in the training data must have 3-5 distinct surface-text patterns (phrasings, wordings, sentence structures) to prevent the model from associating a label with a single syntactic pattern. For example, an "order_status" intent should include "Where is my order?", "Can you check on my delivery?", "I want to track my package", etc.
**Implementation requirements:** Surface pattern tracking per label, pattern diversity metric, synthetic example generation to cover missing patterns, template-based or LLM-based paraphrase generation, validation that each label meets the 3-5 pattern minimum.

## 29. Dataset Sizing Guidelines
**Paper reference:** Section 2.3 (Data Curation)
**Description:** Target dataset sizes depend on task type: 100-200 examples for classification and NER tasks, 500-3,000 examples for generation tasks. These ranges reflect the data efficiency of LoRA fine-tuning (classification tasks need fewer examples because the decision boundary is simpler) while ensuring generation tasks have enough diversity.
**Implementation requirements:** Task-type-aware size targets, size validation before training, warnings or automatic expansion if dataset falls below minimum, upper bound enforcement to prevent training on unnecessarily large datasets.

## 30. Low-Score Diagnostic: Data Problem (Score < 0.80)
**Paper reference:** Section 2.4 (Iteration Policy)
**Description:** When the eval score is below 0.80, the agent diagnoses a data problem. The appropriate response is to rebuild the dataset from scratch: re-examine the task definition, re-curate examples, adjust the gold/hard-negative ratio, and potentially change the data sources. Hyperparameter tuning at this stage would be premature optimization.
**Implementation requirements:** Score threshold check (< 0.80), routing logic that triggers dataset rebuild, dataset rebuild procedure (re-run curation pipeline with different parameters), logging of the diagnostic decision and rationale.

## 31. Mid-Score Diagnostic: Optimization Problem (Score 0.80-0.95)
**Paper reference:** Section 2.4 (Iteration Policy)
**Description:** When the eval score is between 0.80 and 0.95, the agent diagnoses an optimization problem. The dataset is adequate but the model is not fully learning from it. The response is to tune hyperparameters (learning rate, epochs, batch size, LoRA rank/alpha) while holding the dataset fixed. This prevents confounding dataset changes with hyperparameter effects.
**Implementation requirements:** Score range check (0.80-0.95), routing logic that triggers hyperparameter tuning, hyperparameter search space definition, dataset freezing (prevent modifications), hyperparameter modification proposal via LLM.

## 32. High-Score Diagnostic: Surgical Intervention (Score >= 0.95)
**Paper reference:** Section 2.4 (Iteration Policy)
**Description:** When the eval score is at or above 0.95, the agent performs surgical intervention. The model is performing well overall but has specific failure patterns. The response is to add 2-3 targeted examples per identified failure pattern (hard negatives, boundary cases) without disturbing the rest of the dataset. This minimizes regression risk while addressing the remaining errors.
**Implementation requirements:** Score threshold check (>= 0.95), failure pattern extraction from eval results, targeted example generation (2-3 per pattern), dataset append logic (add without modifying existing examples), regression monitoring.

## 33. Rollback-First Policy on Regression
**Paper reference:** Section 2.4 (Iteration Policy)
**Description:** If a training iteration produces a regression from the previous best score, the agent immediately rolls back to the prior checkpoint. It does not attempt to compensate for the regression by adding more data or adjusting hyperparameters on the regressed model. The rollback is unconditional and immediate. The agent then tries a different approach from the pre-regression state.
**Implementation requirements:** Score comparison with previous best, rollback trigger on any score decrease, checkpoint management (save/restore), state reset to pre-regression configuration, alternative approach generation after rollback.

## 34. Cold-Start Input Specification
**Paper reference:** Section 2.5 (Cold-Start Mode)
**Description:** Cold-start mode accepts a task specification tuple tau = (description, M, constraints) where description is a natural language task description, M is the model to fine-tune (or a model selection policy), and constraints are optional requirements (accuracy targets, latency limits, size limits). This structured input ensures the agent has all necessary information to begin.
**Implementation requirements:** Input schema definition and validation, natural language description parsing, model resolution (name to HuggingFace ID), constraint parsing and storage, defaults for optional fields.

## 35. Task Classification (classification / NER / generation)
**Paper reference:** Section 2.5 (Cold-Start Mode)
**Description:** The agent classifies the input task into one of three categories: classification (mapping inputs to discrete labels), NER (extracting named entities from text), or generation (producing free-form text output). This classification determines the model family (encoder vs. decoder), data format, training approach, and eval metrics used throughout the pipeline.
**Implementation requirements:** Task classifier (LLM-based or rule-based), task type enum, downstream configuration selection based on task type (model family, data format, eval metric), override mechanism if the agent's classification is wrong.

## 36. Data Acquisition via Web Research
**Paper reference:** Section 2.5 (Cold-Start Mode)
**Description:** In cold-start mode, the agent uses the `web_search` tool (Exa API) to locate relevant datasets for the task. It searches for existing benchmarks, public datasets, and academic papers, then downloads actual benchmark data when named datasets are found. This is the agent's primary mechanism for bootstrapping training data from scratch.
**Implementation requirements:** Exa API search queries tuned for dataset discovery, result parsing and relevance ranking, dataset download logic (HuggingFace datasets, direct URLs, etc.), format detection and conversion, data quality validation.

## 37. Baseline SOTA Survey
**Paper reference:** Section 2.5 (Cold-Start Mode)
**Description:** Before training, the agent surveys state-of-the-art results on the task to calibrate its stopping threshold. If published SOTA for a 1B model on this task is 0.85, setting the stop threshold at 0.96 may be unrealistic. The agent adjusts the stop_threshold relative to published numbers to set achievable goals.
**Implementation requirements:** Web search queries for SOTA results on the task, result parsing for accuracy numbers, model size filtering (only consider comparable model sizes), threshold calibration logic, documentation of the calibration rationale.

## 38. Eval Set Construction (Positive + Negative + Boundary)
**Paper reference:** Section 2.5 (Cold-Start Mode)
**Description:** The held-out evaluation set is constructed as E = E_pos union E_neg union E_boundary, containing positive examples (clearly correct), negative examples (clearly wrong), and boundary examples (ambiguous or near decision boundaries). This eval set is fixed once constructed and never included in training data. It provides a comprehensive assessment of model capability.
**Implementation requirements:** Eval set partitioning logic, boundary case identification (e.g., examples where multiple labels are plausible), data leak prevention (hash-based deduplication between train and eval), eval set persistence and immutability enforcement.

## 39. Stopping Criterion (Default 0.96, Agent-Adjustable)
**Paper reference:** Section 2.5 (Cold-Start Mode)
**Description:** The default stopping criterion is f(pi) >= 0.96 (96% accuracy or equivalent metric). However, the agent can adjust this threshold downward if it detects a genuine plateau -- sustained inability to improve despite diverse interventions. This prevents infinite loops on tasks where 96% is not achievable with the given model and data.
**Implementation requirements:** Configurable stop threshold with 0.96 default, plateau detection algorithm (e.g., no improvement over K consecutive iterations), threshold adjustment logic with minimum floor, logging of threshold changes and justification.

## 40. Parallel Configuration Training (>= 2 Per Iteration)
**Paper reference:** Section 2.5 (Cold-Start Mode)
**Description:** Each iteration trains at least 2 configurations in parallel. This doubles the information gained per iteration and enables direct comparison between alternative approaches. The agent uses `delegate_task` to spawn parallel training sub-agents, then compares results to select the best configuration and inform the next iteration's proposals.
**Implementation requirements:** Parallel training launch via delegate_task, configuration differentiation logic (ensure parallel configs are meaningfully different), result collection and comparison, best-config selection, parallel resource management (GPU allocation).

## 41. Train from Base Model Every Time (No Checkpoint Continuation)
**Paper reference:** Section 2.5 (Cold-Start Mode)
**Description:** Every training run starts from the base model, never from a prior fine-tuned checkpoint. This prevents error accumulation across iterations and ensures each configuration is evaluated on its own merits. While this is more expensive (each run trains from scratch), it eliminates the confound of inheriting artifacts from prior training runs.
**Implementation requirements:** Base model caching (download once, reuse), checkpoint isolation (prior adapters are not loaded), training script parameter that forces base model initialization, validation that no prior adapter weights are being loaded.

## 42. LangGraph Turn Budget (1,500 for Cold-Start)
**Paper reference:** Section 2.5 (Cold-Start Mode)
**Description:** Cold-start mode has a hard budget of 1,500 LangGraph turns. This prevents runaway loops and bounds the cost of any single cold-start run. The agent must plan its iteration strategy to make productive use of this budget, including reserving turns for final evaluation and documentation.
**Implementation requirements:** Turn counter in LangGraph state, budget check at each turn, graceful termination when budget is approached (e.g., trigger final eval at turn 1400), budget usage logging, early termination path that saves the best result so far.

## 43. Production Mode Input (Deployed Model + Judged Traces)
**Paper reference:** Section 2.6 (Production Mode)
**Description:** Production mode accepts a deployed model M0 plus a set of judged inference traces T = {(x_i, y_hat_i, y_star_i, v_i, r_i)} where x_i is the input, y_hat_i is the model's prediction, y_star_i is the correct label (human-judged), v_i is a confidence/vote value, and r_i is additional metadata (e.g., the reason for failure). This structured input provides the agent with concrete evidence of where the deployed model fails.
**Implementation requirements:** Trace data schema (input, prediction, ground truth, confidence, metadata), trace ingestion pipeline, data validation (ensure all fields present), trace storage (database or structured files), model reference resolution.

## 44. Trace Ingestion and Partitioning (T_fail + T_pass)
**Paper reference:** Section 2.6 (Production Mode)
**Description:** The first step in production mode is partitioning the inference traces into T_fail (where the model's prediction was wrong) and T_pass (where it was correct). This partition is the basis for all subsequent analysis: failures inform what to fix, passes inform what to preserve (via the replay buffer).
**Implementation requirements:** Comparison function (y_hat vs y_star), partition logic, failure/pass set storage, statistics computation (failure rate, per-class failure rates), integration with the trace analysis pipeline.

## 45. Failure Taxonomy Construction
**Paper reference:** Section 2.6 (Production Mode)
**Description:** The agent clusters failure traces into K categories based on the nature of the error (e.g., "confuses intent A with intent B", "fails on long inputs", "misses entities in lists"). Each category is labeled as either fixable (the model can learn to handle it with better training data) or external (the error is due to factors outside the model's control, like ambiguous inputs or labeling errors). Only fixable categories are targeted for improvement.
**Implementation requirements:** Failure clustering algorithm (LLM-based or embedding-based), category labeling (fixable vs. external), cluster quality validation, taxonomy serialization, integration with the data curation pipeline (only fixable categories generate training data).

## 46. Live Confirmation (Probe Set Verification)
**Paper reference:** Section 2.6 (Production Mode)
**Description:** After constructing the failure taxonomy, the agent synthesizes a probe set -- new examples designed to test whether each identified weakness is systematic (the model consistently fails) or spurious (the original failures were edge cases). The current deployed model is run on the probe set, and only weaknesses confirmed as systematic are targeted for remediation.
**Implementation requirements:** Probe example generation per failure category, probe set construction, inference on deployed model with probe set, confirmation logic (e.g., >50% failure rate on probes = systematic), filtering of non-systematic categories.

## 47. Parent Model Awareness and Lineage Inspection
**Paper reference:** Section 2.6 (Production Mode)
**Description:** The agent inspects the deployed model's training lineage: what data it was trained on (D_parent), what hyperparameters were used, and what its known strengths and weaknesses are. This prevents the agent from repeating failed strategies and enables informed construction of the replay buffer.
**Implementation requirements:** Model lineage metadata storage (training config, dataset composition, eval results), lineage retrieval function, D_parent access, history-aware proposal generation (avoid configurations that have already been tried and failed).

## 48. Replay Buffer Construction
**Paper reference:** Section 2.6 (Production Mode)
**Description:** A replay buffer D_replay is sampled from the parent model's training data D_parent, with |D_replay| approximately 10-20% of |D_parent|. The replay buffer is included in the new training dataset to prevent catastrophic forgetting of capabilities the current model already has. Examples are sampled to cover the breadth of the parent dataset's distribution.
**Implementation requirements:** D_parent access and sampling, representative sampling strategy (stratified by label, proportional to class distribution), size computation (10-20% of parent), integration with the dataset composition pipeline (as the replay component of the three-way split).

## 49. Regression Gate (Absolute Count Threshold)
**Paper reference:** Section 2.6 (Production Mode)
**Description:** The regression gate checks whether the new model introduces regressions on previously correct examples. The gate is defined as r(pi; R) <= epsilon where R is the regression set (examples the old model got right) and epsilon = 2 (allowing at most 2 new errors on previously correct examples). This is an absolute count, not a percentage, making it strict for small eval sets.
**Implementation requirements:** Regression set computation (intersection of eval set with T_pass), new model evaluation on regression set, regression count computation, gate check (count <= 2), gate failure handling (reject the new model, trigger rollback).

## 50. Cross-Checkpoint Regression Gate
**Paper reference:** Section 2.6 (Production Mode)
**Description:** The new model must pass regression gates against not just the immediately previous checkpoint but also earlier checkpoints. This prevents a scenario where model A -> B -> C is fine pairwise but A -> C has accumulated regressions. The new model is evaluated on eval sets from all previous deployment stages.
**Implementation requirements:** Historical eval set storage (one per deployment stage), multi-set evaluation of new model, per-set regression computation, composite gate (must pass all individual gates), historical checkpoint metadata management.

## 51. Production Mode Turn Budget (500 Turns)
**Paper reference:** Section 2.6 (Production Mode)
**Description:** Production mode has a tighter turn budget of 500 LangGraph turns (vs. 1,500 for cold-start). This reflects the expectation that production improvements are more targeted: the agent starts with a working model and concrete failure traces, so it should need fewer iterations to diagnose and fix specific issues.
**Implementation requirements:** Turn counter with 500 limit for production mode, mode-aware budget selection, graceful termination at budget approach, best-result preservation on budget exhaustion.

## 52. Hard Negatives + Label Balancing as Structural Safeguard
**Paper reference:** Section 2.7 (Structural Safeguards)
**Description:** The combination of hard negative examples (feature 23) and label balancing (feature 24) serves as a structural safeguard against the model learning shortcuts or developing blind spots. These are not optional enhancements but mandatory components of every training dataset, enforced by validation checks before training begins.
**Implementation requirements:** Pre-training validation that checks hard negative presence and label balance, training rejection if validation fails, automated rebalancing as a pre-training step.

## 53. Rollback-First Iteration as Structural Safeguard
**Paper reference:** Section 2.7 (Structural Safeguards)
**Description:** The rollback-first policy (feature 33) is elevated to a structural safeguard: it is not a suggestion but an enforced invariant. The system must never attempt to recover from a regression by compensating -- it must always roll back first, then try a different approach. This prevents cascading errors from compounding.
**Implementation requirements:** Invariant enforcement in the iteration loop (cannot proceed without rollback on regression), audit logging of all rollback events, regression detection that cannot be bypassed or overridden.

## 54. Parallel Training as Structural Safeguard (2-3 Configs)
**Paper reference:** Section 2.7 (Structural Safeguards)
**Description:** Training 2-3 configurations in parallel is a structural safeguard, not just an efficiency measure. It ensures that the agent never puts all its compute into a single configuration that might fail. By always maintaining alternatives, the system is robust to individual configuration failures.
**Implementation requirements:** Minimum parallel configuration count enforcement (>= 2), configuration diversity check (parallel configs must differ meaningfully), result comparison and selection, resource allocation for parallel training.

## 55. Production Mode Regression Gating as Structural Safeguard
**Paper reference:** Section 2.7 (Structural Safeguards)
**Description:** The regression gate (feature 49) is identified as a key structural safeguard for production mode. It prevents deploying a model that fixes new problems at the cost of re-introducing old ones. The gate is non-negotiable: no model passes to deployment without clearing it.
**Implementation requirements:** Gate enforcement as a hard requirement (not a warning), deployment pipeline integration (gate must pass before checkpoint promotion), gate failure alerting and logging.

## 56. Cross-Checkpoint Regression Gate as Structural Safeguard
**Paper reference:** Section 2.7 (Structural Safeguards)
**Description:** The cross-checkpoint regression gate (feature 50) is a structural safeguard that prevents accumulated drift across multiple deployment cycles. Even if each individual update is safe pairwise, the cumulative effect can degrade performance on early capabilities. This gate catches such drift.
**Implementation requirements:** Historical eval set persistence across deployment cycles, cumulative regression tracking, drift alerting when regression counts approach the threshold even if they haven't crossed it.

## 57. Confidence Calibration
**Paper reference:** Section 2.7 (Structural Safeguards)
**Description:** Model confidence scores are calibrated using a weighted blend: calibrated = weight * actual_accuracy + (1 - weight) * raw_confidence. This prevents the model from being overconfident (raw confidence too high relative to actual accuracy) or underconfident. The calibration weight is tuned per task and updated as more eval data accumulates.
**Implementation requirements:** Confidence extraction from model outputs, actual accuracy computation per confidence bin, weight parameter tuning (grid search or optimization), calibrated confidence computation, calibration curve visualization, re-calibration trigger when accuracy distribution shifts.

## 58. TF-IDF Similarity for Correction Propagation
**Paper reference:** Section 2.7 (Structural Safeguards)
**Description:** When the agent corrects a failure, it uses TF-IDF similarity to find other examples in the training set that are textually similar to the corrected example. These similar examples are then checked and potentially corrected as well. This propagation prevents the situation where a systematic error is fixed in one example but persists in similar ones.
**Implementation requirements:** TF-IDF vectorizer fitted on the training corpus, similarity computation between corrected example and all training examples, similarity threshold for propagation candidates, correction application to similar examples, validation that propagated corrections are accurate.

## 59. Chain-of-Thought Supervision Discovery (Emergent)
**Paper reference:** Section 5.1 (Emergent Strategies)
**Description:** The agent independently discovered that adding chain-of-thought supervision improves performance on reasoning tasks. This was not pre-programmed -- the agent observed that reasoning tasks benefited from intermediate steps and began generating CoT annotations. This emergent behavior validates the agent's ability to discover effective training strategies beyond its initial programming.
**Implementation requirements:** Agent freedom to modify the training data format (not locked to a fixed template), CoT generation capability, detection of reasoning tasks, before/after comparison to validate CoT benefit, automatic CoT application when benefit is confirmed.

## 60. System Prompt as Tunable Configuration Component
**Paper reference:** Section 5.1 (Emergent Strategies)
**Description:** The agent discovered that the system prompt (the instruction prepended to each inference input) is an effective tuning knob. By modifying the system prompt's wording, emphasis, and structure, the agent can influence model behavior without retraining. This treats the system prompt as part of the pipeline configuration, not a fixed constant.
**Implementation requirements:** System prompt parameterization in the pipeline config, system prompt variation generation, eval with different system prompts (prompt-only ablation), system prompt optimization loop (potentially independent of training), system prompt versioning.

## 61. Task-Specific Epoch Count Selection
**Paper reference:** Section 5.1 (Emergent Strategies)
**Description:** The agent learned to select different epoch counts based on task characteristics. Classification tasks with clear decision boundaries converge faster (fewer epochs), while generation tasks with complex output distributions require more epochs. The agent adjusts epoch count based on observed training loss curves and eval performance trajectories.
**Implementation requirements:** Epoch count as a tunable hyperparameter (not fixed), task-type-based initial epoch selection, training loss monitoring, early stopping or epoch extension based on loss curves, epoch count recording in configuration metadata.

## 62. Learning Rate Selection by Task Type
**Paper reference:** Section 5.1 (Emergent Strategies)
**Description:** The agent learned to select learning rates based on task type: 1e-4 for knowledge-intensive tasks (where the model needs to carefully integrate new knowledge without overwriting existing capabilities) and 2e-4 for format-learning tasks (where the model primarily needs to learn an output format or style). This task-aware LR selection emerged from observing training dynamics across multiple tasks.
**Implementation requirements:** Learning rate as a tunable hyperparameter, task-type-to-LR mapping (knowledge-intensive -> 1e-4, format-learning -> 2e-4), training dynamics monitoring to validate LR selection, LR override capability for the agent.

## 63. Quality-Over-Quantity Data Curation
**Paper reference:** Section 5.1 (Emergent Strategies)
**Description:** The agent converged on a quality-over-quantity strategy: smaller, carefully curated datasets consistently outperformed larger, noisier ones. This manifested as aggressive filtering of ambiguous examples, preference for manually verified gold data, and willingness to train on fewer examples if quality was higher. This is a meta-strategy that informs all data curation decisions.
**Implementation requirements:** Data quality scoring (confidence of label correctness), quality-based filtering thresholds, dataset size vs. quality tradeoff analysis, quality metrics in the data-curation.md log, preference for quality in the EXPAND operator's proposals.

## 64. Self-Generated Labels Preferred Over External Model Outputs
**Paper reference:** Section 5.1 (Emergent Strategies)
**Description:** The agent found that labels it generated itself (through careful analysis of the task and examples) were more reliable than labels produced by external models (including larger LLMs). External model outputs sometimes introduced systematic biases or errors that degraded training. Self-generated labels, while more expensive, maintained consistency with the agent's understanding of the task.
**Implementation requirements:** Label generation capability in the agent, external label quality validation, preference weighting in label source selection, comparison studies (self-generated vs. external labels), fallback to external labels only when self-generation is not feasible.

## 65. tinker_helpers.py (train, infer, infer_batch)
**Paper reference:** Appendix D (Infrastructure)
**Description:** A helper module `tinker_helpers.py` pre-loaded in the bash environment that provides three core functions: `train()` for launching LoRA fine-tuning, `infer()` for single-example inference, and `infer_batch(max_workers=20)` for parallel inference across up to 20 workers. These helpers abstract away the Tinker SDK complexity and provide a clean interface for the agent's bash commands.
**Implementation requirements:** `train()` function wrapping Tinker SDK training with config parameters, `infer()` function for single-input inference with adapter loading, `infer_batch()` with thread pool (max_workers=20) for parallel inference, error handling and retry logic, result formatting for agent consumption.

## 66. apply_chat_template for Train/Serve Parity
**Paper reference:** Appendix D (Infrastructure)
**Description:** The `apply_chat_template` function ensures that training data is formatted with exactly the same chat template that will be used at inference time. This prevents train-serve skew where the model sees differently formatted inputs during training vs. deployment, which can cause silent accuracy degradation. The template is determined by the model's tokenizer configuration.
**Implementation requirements:** Chat template extraction from tokenizer config, template application function that formats (system, user, assistant) turns, training data preprocessor that applies the template, inference preprocessor that applies the same template, validation that train and serve templates match.

## 67. Per-Token Loss Weights (Assistant Tokens Only)
**Paper reference:** Appendix D (Infrastructure)
**Description:** During training, the loss is computed only on assistant tokens (the model's response), not on system or user tokens. This is implemented via per-token loss weights: tokens in the system/user turns have weight 0, tokens in the assistant turn have weight 1. This focuses the model's learning on generating correct outputs rather than memorizing input patterns.
**Implementation requirements:** Token-level role assignment (which tokens belong to system/user/assistant), loss weight mask generation (0 for non-assistant, 1 for assistant), integration with the training loss function, validation that the mask correctly identifies token boundaries.

## 68. EOS Truncation for Base Models
**Paper reference:** Appendix D (Infrastructure)
**Description:** For base models (not instruct-tuned), generated outputs are truncated at the first EOS (end-of-sequence) token. Base models may not reliably stop generating at a natural endpoint, so explicit truncation prevents garbage output from contaminating eval results or downstream processing.
**Implementation requirements:** EOS token ID lookup from tokenizer, post-generation truncation logic, truncation for base models only (instruct models handle their own stopping), truncation logging for debugging.

## 69. Agent-Written Eval Code Per Task Type
**Paper reference:** Appendix D (Infrastructure)
**Description:** The agent writes custom evaluation code tailored to each task type rather than using a generic eval harness. For classification, it writes accuracy/F1 computation. For NER, it writes span-level matching. For generation, it writes task-specific rubrics. This per-task customization ensures that the eval metric faithfully measures what matters for the specific task.
**Implementation requirements:** Eval code generation capability (agent writes Python eval scripts), per-task-type eval templates (classification, NER, generation), eval script execution and result parsing, eval code versioning (track changes across iterations), validation that eval code produces consistent results on fixed inputs.
