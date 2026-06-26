# SLM Factory Phase 1 -- Plan Feature List

Every unique feature, component, interface, data structure, function signature, test case, and global constraint extracted from the implementation plan (`docs/superpowers/plans/2026-06-19-slm-factory-phase1.md`).

---

## 1. Global Constraint: Android Model Pool Bound
**Plan reference:** Global Constraints
**Design doc equivalent:** Section 6.4
**Description:** All selected models must fit within the Android pool, capped at approximately 1.5 GB INT4. The agent starts at the smallest feasible Tier 1 model and escalates only if accuracy targets are unmet.
**Code location:** `android_pool.py`
**Implementation details:** Enforced by `filter_pool(constraints: HardwareConstraints) -> list[ModelSpec]`, which filters `ANDROID_POOL` by `storage_mb` and `memory_mb`, returning results sorted by `(tier, int4_size_mb)`.

---

## 2. Global Constraint: Always Train from Base Model
**Plan reference:** Global Constraints
**Design doc equivalent:** Section 2.4, Node 3
**Description:** Training always restarts from the base foundation model, never from a prior fine-tuned checkpoint. This makes rollback clean by fully restoring model behavior via dataset reversion alone.
**Code location:** `training/slm_helpers.py`, `agent/nodes/train.py`
**Implementation details:** `train()` accepts `base_model: str` (a HuggingFace model ID from the Android pool) and always loads weights from the pretrained source. No checkpoint-continuation path exists.

---

## 3. Global Constraint: Two Parallel Training Configs Per Iteration
**Plan reference:** Global Constraints
**Design doc equivalent:** Section 2.4, Node 3
**Description:** At least two training configurations must be fired in parallel every iteration (e.g., LoRA vs. full fine-tune, or two different hyperparameter sets). This is not optional.
**Code location:** `agent/nodes/train.py`
**Implementation details:** `train_node()` defines a list of two config dicts (Config A: LoRA r=8, lr=2e-4, epochs=3; Config B: full_ft, lr=1e-4, epochs=5) and submits them to a `ThreadPoolExecutor(max_workers=2)`. Both `weights_ref` results are stored in `state["_pending_weights_refs"]`.

---

## 4. Global Constraint: data-curation.md Written Every Iteration
**Plan reference:** Global Constraints
**Design doc equivalent:** Section 2.2, Section 4.3
**Description:** A structured markdown file (`data-curation.md`) is written to disk every iteration before the next training round begins. Records dataset versions, composition ratios, quality-control decisions, and per-iteration evaluation results. This is the primary lineage artifact.
**Code location:** `data/curation_log.py`, `agent/nodes/evaluate.py`
**Implementation details:** `CurationLog` class with `write_iteration(...)` and `read_latest() -> str`. Called by `evaluate_node()` after scoring. Schema includes: iteration number, timestamp, task type, dataset version, Dgold/Dhard counts with percentages, label distribution, training config details (A/B), best config selection, eval results (f1, per_class, per-slice), iteration policy decision (band, intervention, hypothesis), and hardware profile.

---

## 5. Global Constraint: No MCGS in Phase 1
**Plan reference:** Global Constraints
**Design doc equivalent:** Section 2.5, Section 5.2
**Description:** Phase 1 uses sequential greedy DAG iteration only, not full Monte Carlo Graph Search. Each iteration is a single node `v_i = (pi_i, f(pi_i))` connected by edges representing targeted modifications.
**Code location:** `agent/graph.py`, `agent/nodes/evaluate.py`
**Implementation details:** DAG is stored as `state["dag"]: list[dict]` where each dict has keys: `iteration`, `model_id`, `weights_ref`, `score`, `best_config`, `intervention`, `failures`, `pruned`.

---

## 6. Global Constraint: Simple Rollback Only
**Plan reference:** Global Constraints
**Design doc equivalent:** Section 2.4, Node 7; Section 5.3
**Description:** Cold-start uses simple rollback: if `f(pi_{i+1}) < f(pi_i)`, revert immediately. No dual-gate regression constraint (that is production mode only). No compensation with more data.
**Code location:** `agent/nodes/rollback.py`
**Implementation details:** `should_rollback(state) -> bool` checks `scores[-1] < scores[-2]`. `rollback_node()` pops the last score, marks the DAG node as pruned, and increments `consecutive_no_improvement`.

---

## 7. Global Constraint: No Replay Buffer in Phase 1
**Plan reference:** Global Constraints
**Design doc equivalent:** Section 4.2
**Description:** Cold-start has no replay buffer (`Dparent = {}`, replay allocation redistributed to gold examples). The curriculum is `Dcold = Dgold + Dhard` at 65:35 ratio.
**Code location:** `data/curriculum.py`
**Implementation details:** `build_initial_curriculum()` uses `gold_fraction=0.65` with the remaining 35% allocated to hard negatives.

---

## 8. Global Constraint: Hardware Constraints Logged But Not Enforced
**Plan reference:** Global Constraints
**Design doc equivalent:** Section 6.5
**Description:** Hardware constraints are logged every iteration as theoretical estimates but are not enforced as hard gates until Phase 2. Storage and memory still gate model selection.
**Code location:** `training/quantize.py`, `agent/nodes/evaluate.py`
**Implementation details:** `theoretical_hardware_profile(model_id) -> dict` returns estimates. `evaluate_node()` calls this and passes the result to `CurationLog.write_iteration()`.

---

## 9. Global Constraint: Task-Type Abstraction Is Mandatory
**Plan reference:** Global Constraints (revision note)
**Design doc equivalent:** Section 1 (Task abstraction), Section 2.4 Node 1
**Description:** All data/eval/curation logic must branch on `task_type in {"classification", "NER", "generation"}`. No SMS-Spam-specific hardcoding outside of the dataset loader. Passing a different task description must route correctly through the same graph without code changes.
**Code location:** All modules consume `task_type`; `data/eval_set.py`, `eval/harness.py`, `eval/scorers/`, `data/curriculum.py`, `data/curation_log.py`
**Implementation details:** `EvalSet` has a `task_type: str` field validated against `TASK_TYPES = {"classification", "NER", "generation"}`. `build_eval_set()` requires `task_type` argument. `run_eval()` dispatches to `eval/scorers/{task_type}.py`. `CurationLog.write_iteration()` requires `task_type` argument.

---

## 10. Global Constraint: Orchestrator and Sub-Agent Turn Limits
**Plan reference:** Global Constraints
**Design doc equivalent:** Section 2.1, Section 2.2
**Description:** Claude Sonnet 4.6 is the orchestrator LLM. Main agent turn limit is 1,500 LangGraph turns; sub-agent turn limit is 1,000.
**Code location:** `config.py`
**Implementation details:** `ORCHESTRATOR_MODEL = "claude-sonnet-4-6"`, `MAX_TURNS_MAIN = 1500`, `MAX_TURNS_SUBAGENT = 1000`.

---

## 11. ModelSpec Dataclass
**Plan reference:** Task 1, Step 4
**Design doc equivalent:** Section 6.4
**Description:** Data class representing a model in the Android pool with its hardware characteristics.
**Code location:** `android_pool.py`
**Implementation details:**
```python
@dataclass
class ModelSpec:
    model_id: str          # HuggingFace model ID
    int4_size_mb: int      # theoretical INT4 size in MB
    tier: int              # 1, 2, or 3
    tok_s_snapdragon_660: float
    tok_s_snapdragon_778g: float
    tok_s_snapdragon_8gen3: float
    peak_memory_mb: int
    notes: str = ""
```

---

## 12. HardwareConstraints Dataclass
**Plan reference:** Task 1, Step 4
**Design doc equivalent:** Section 6.1, Section 6.2
**Description:** Defines the feasibility boundary for on-device Android deployment: storage, memory, latency (TTFT), power, and target chip.
**Code location:** `android_pool.py`
**Implementation details:**
```python
@dataclass
class HardwareConstraints:
    storage_mb: int
    memory_mb: int
    latency_ttft_ms: int
    power_watts: float
    target_chip: str = "snapdragon_778g"
```

---

## 13. ANDROID_POOL Constant
**Plan reference:** Task 1, Step 4
**Design doc equivalent:** Section 6.4
**Description:** The complete list of 12 `ModelSpec` instances across three tiers. Tier 1 (6 models, up to 808 MB): Qwen3-0.6B, Qwen3.5-0.8B, MiniCPM4-0.5B, MiniCPM5-1B, Gemma3-1B, Llama3.2-1B. Tier 2 (4 models, 800 MB to 1.5 GB): SmolLM2-1.7B, Qwen3-1.7B, Qwen3.5-2B, DeepSeek-R1-Distill-1.5B. Tier 3 (2 models, generous upper bound): Llama3.2-3B, Ministral-3B.
**Code location:** `android_pool.py`
**Implementation details:** `ANDROID_POOL: list[ModelSpec]` with all 12 entries, each specifying `model_id`, `int4_size_mb`, `tier`, tok/s across three Snapdragon chips, and `peak_memory_mb`.

---

## 14. filter_pool Function
**Plan reference:** Task 1, Step 4
**Design doc equivalent:** Section 6.3
**Description:** Filters the Android pool by storage and memory constraints, returning feasible models sorted by tier ascending then size ascending.
**Code location:** `android_pool.py`
**Implementation details:**
```python
def filter_pool(constraints: HardwareConstraints) -> list[ModelSpec]:
    feasible = [
        m for m in ANDROID_POOL
        if m.int4_size_mb <= constraints.storage_mb
        and m.peak_memory_mb <= constraints.memory_mb
    ]
    return sorted(feasible, key=lambda m: (m.tier, m.int4_size_mb))
```

---

## 15. Config Module
**Plan reference:** Task 1, Step 3
**Design doc equivalent:** Section 2.1, Appendix B
**Description:** Central configuration for API keys, orchestrator model, turn limits, default thresholds, and cluster settings.
**Code location:** `config.py`
**Implementation details:** Environment variables: `ANTHROPIC_API_KEY`, `EXA_API_KEY`, `OPENAI_API_KEY` (optional), `DEEPSEEK_API_KEY` (optional). Constants: `ORCHESTRATOR_MODEL = "claude-sonnet-4-6"`, `MAX_TURNS_MAIN = 1500`, `MAX_TURNS_SUBAGENT = 1000`, `DEFAULT_STOP_THRESHOLD = 0.96`, `DEFAULT_STAGNATION_ROUNDS = 2`, `CLUSTER_GPU = "L40"`, `DATA_DIR = "data_cache"`, `ARTIFACTS_DIR = "artifacts"`.

---

## 16. SMS Spam Data Loader
**Plan reference:** Task 2 (complete), Task 3 Step 3 (move)
**Design doc equivalent:** Section 1 (Phase 1 first task), Section 4.1
**Description:** Downloads and parses the UCI SMS Spam Collection. Returns train/test split. Moved from `data/sms_spam.py` to `data/loaders/sms_spam.py` during Task 3 restructuring.
**Code location:** `data/loaders/sms_spam.py`
**Implementation details:**
```python
def download_sms_spam() -> tuple[list[dict], list[dict]]:
    # Returns (train_examples, test_examples)
    # Each example: {"text": str, "label": str} with label in {"spam", "ham"}
```

---

## 17. EvalSet Dataclass (Task-Type-Aware)
**Plan reference:** Task 2, Task 3 Step 4
**Design doc equivalent:** Section 4.1
**Description:** Represents the held-out evaluation set `E = Epos + Eneg + Eboundary`. Three disjoint slices whose meaning varies by task type. Fixed throughout all iterations, never included in training data.
**Code location:** `data/eval_set.py`
**Implementation details:**
```python
@dataclass
class EvalSet:
    pos: list[dict]
    neg: list[dict]
    boundary: list[dict]
    task_type: str          # validated against TASK_TYPES
    all: list[dict] = field(init=False)  # computed as pos + neg + boundary
```
Raises `ValueError` if `task_type` not in `{"classification", "NER", "generation"}`.

---

## 18. build_eval_set Function (Task-Type-Aware)
**Plan reference:** Task 2, Task 3 Step 4
**Design doc equivalent:** Section 4.1
**Description:** Constructs E from test examples. For classification: infers pos/neg labels (minority=positive), selects boundary examples by keyword matching (www, http, free, win, etc.), ensures all three slices are disjoint. For NER and generation: simple stratified partition.
**Code location:** `data/eval_set.py`
**Implementation details:**
```python
def build_eval_set(
    examples: list[dict],
    task_type: str,
    n_pos: int = 40,
    n_neg: int = 40,
    n_boundary: int = 20,
    seed: int = 42,
) -> EvalSet
```
Helper functions: `_infer_pos_label(examples) -> str` (returns minority label), `_infer_neg_label(examples, pos_label) -> str`.

---

## 19. binary_f1 Function
**Plan reference:** Task 3, Step 6
**Design doc equivalent:** Section 3.3
**Description:** Computes binary F1 for a given positive label. Manual TP/FP/FN calculation without sklearn dependency.
**Code location:** `eval/metrics.py`
**Implementation details:**
```python
def binary_f1(predictions: list[str], labels: list[str], pos_label: str) -> float
```
Returns 0.0 when TP=0.

---

## 20. entity_f1 Function
**Plan reference:** Task 3, Step 6
**Design doc equivalent:** Section 3.3
**Description:** Computes entity-level F1 for NER tasks. Each element is a list of entity span dicts with "text" and "type" keys. Matches on exact (text, type) tuples.
**Code location:** `eval/metrics.py`
**Implementation details:**
```python
def entity_f1(predictions: list[list[dict]], labels: list[list[dict]]) -> float
```

---

## 21. per_slice_scores Function
**Plan reference:** Task 3, Step 6
**Design doc equivalent:** Section 3.3
**Description:** Computes accuracy per eval set slice (pos/neg/boundary). Splits predictions by slice boundaries derived from EvalSet ordering (pos, then neg, then boundary).
**Code location:** `eval/metrics.py`
**Implementation details:**
```python
def per_slice_scores(eval_set: EvalSet, predictions: list[str]) -> dict[str, float]
```
Returns `{"pos": float, "neg": float, "boundary": float}`.

---

## 22. Classification Scorer
**Plan reference:** Task 3, Step 7
**Design doc equivalent:** Section 3.3
**Description:** Task-type scorer for classification tasks. Builds prompts, extracts label predictions from raw model output, and computes F1 + per-class + per-slice scores.
**Code location:** `eval/scorers/classification.py`
**Implementation details:**
- `CLASSIFY_PROMPT` template: `"Classify this message. Reply with exactly one word -- the label.\n\nMessage: {text}"`
- `build_prompts(eval_set: EvalSet) -> list[str]`
- `extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]` -- label extraction with fallback to majority neg label
- `score(eval_set: EvalSet, predictions: list[str]) -> dict` -- returns `{"f1", "per_class", "slices", "failures"}`

---

## 23. NER Scorer
**Plan reference:** Task 3, Step 7
**Design doc equivalent:** Section 3.3
**Description:** Task-type scorer for NER tasks. Extracts entity spans from JSON output, computes entity F1.
**Code location:** `eval/scorers/ner.py`
**Implementation details:**
- `NER_PROMPT` template requesting JSON list of `{"text", "type"}` objects
- `build_prompts(eval_set: EvalSet) -> list[str]`
- `extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[list[dict]]` -- regex-based JSON extraction with fallback to empty list
- `score(eval_set: EvalSet, predictions: list[list[dict]]) -> dict`

---

## 24. Generation Scorer (LLM-as-Judge)
**Plan reference:** Task 3, Step 7
**Design doc equivalent:** Section 3.3
**Description:** Task-type scorer for generation tasks. Uses Claude Haiku as an LLM judge to rate correctness from 0.0 to 1.0. Converts scores to pass/fail at 0.5 threshold for per-slice analysis.
**Code location:** `eval/scorers/generation.py`
**Implementation details:**
- `JUDGE_PROMPT` template with question/gold/predicted placeholders, requesting a single float
- `GENERATE_PROMPT` template: `"Answer the following question:\n\n{text}"`
- `build_prompts(eval_set: EvalSet) -> list[str]`
- `extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]` -- simple strip
- `score(eval_set: EvalSet, predictions: list[str]) -> dict` -- calls `anthropic.Anthropic` with `claude-haiku-4-5-20251001`, returns average judge score as `f1`

---

## 25. EvalResult Dataclass
**Plan reference:** Task 3, Step 9
**Design doc equivalent:** Section 2.4 Node 4
**Description:** Result of evaluating a trained model against the held-out eval set.
**Code location:** `eval/harness.py`
**Implementation details:**
```python
@dataclass
class EvalResult:
    f1: float
    per_class: dict           # {label: f1_score, ...}
    pos_score: float
    neg_score: float
    boundary_score: float
    failures: list[dict]      # examples where prediction != label
```

---

## 26. run_eval Function (Dispatcher Harness)
**Plan reference:** Task 3, Step 9
**Design doc equivalent:** Section 3.3
**Description:** Runs inference on the full eval set and computes task-type-appropriate metrics. Dispatches to the correct scorer module based on `task_type`. Raises `ValueError` for unknown task types.
**Code location:** `eval/harness.py`
**Implementation details:**
```python
def run_eval(
    eval_set: EvalSet,
    weights_ref: str,
    base_model: str,
    task_type: str,
) -> EvalResult
```
Flow: import scorer by task_type -> `scorer.build_prompts()` -> `infer_batch()` -> `scorer.extract_predictions()` -> `scorer.score()` -> construct `EvalResult`.

---

## 27. build_initial_curriculum Function
**Plan reference:** Task 4, Step 3
**Design doc equivalent:** Section 4.2
**Description:** Builds the initial cold-start curriculum `Dcold = Dgold + Dhard` at 65:35 ratio from training examples. Excludes any example that appears in the eval set. Applies label balancing (no label exceeds 3x any other).
**Code location:** `data/curriculum.py`
**Implementation details:**
```python
def build_initial_curriculum(
    train_examples: list[dict],
    eval_set: EvalSet,
    n_total: int = 150,
    gold_fraction: float = 0.65,
    seed: int = 42,
) -> list[dict]
```
Balances spam/ham counts to satisfy the 3x constraint, shuffles, and applies `apply_quality_controls()`.

---

## 28. apply_quality_controls Function
**Plan reference:** Task 4, Step 3
**Design doc equivalent:** Section 2.4 Node 6, Section 4.2
**Description:** Enforces label balancing: no label exceeds 3x the count of any other label. Trims the majority class by iterating through the dataset and capping per-label counts at `3 * min_count`.
**Code location:** `data/curriculum.py`
**Implementation details:**
```python
def apply_quality_controls(dataset: list[dict], task_type: str) -> list[dict]
```
Note: The plan implementation uses a Counter-based approach. The `task_type` parameter is listed in the interface spec but the initial implementation applies the same 3x balancing logic regardless of type.

---

## 29. synthesize_hard_negatives Function
**Plan reference:** Task 4, Step 3
**Design doc equivalent:** Section 4.2
**Description:** Generates hard negatives using the 2-for-1 rule via Claude API. For each boundary spam example, generates a ham counterexample with similar surface form but legitimate intent. Uses Claude Sonnet 4.6 for synthesis.
**Code location:** `data/curriculum.py`
**Implementation details:**
```python
def synthesize_hard_negatives(
    examples: list[dict],
    n: int,
    anthropic_client,
    task_type: str,
) -> list[dict]
```
Selects up to `n` spam examples, prompts Claude to generate a legitimate-looking counterpart, returns `{"text": generated_text, "label": "ham"}` dicts.

---

## 30. CurationLog Class
**Plan reference:** Task 4, Step 4
**Design doc equivalent:** Section 4.3
**Description:** Reads and writes `data-curation.md`, the agent's durable lineage artifact. Supports appending iteration entries and reading the full file.
**Code location:** `data/curation_log.py`
**Implementation details:**
```python
class CurationLog:
    def __init__(self, path: str = "data-curation.md")
    def write_iteration(
        self,
        iteration: int,
        task_type: str,
        dataset_version: str,
        n_gold: int, n_hard: int,
        label_dist: dict,
        config_a: str, config_b: str,
        best_config: str,
        eval_result: EvalResult,
        score_band: str,
        next_intervention: str,
        hypothesis: str,
        model_id: str,
        int4_size_mb: int,
        tier: int,
        hardware_notes: str = "Phase 1: theoretical",
    ) -> None
    def read_latest(self) -> str
```

---

## 31. TrainingConfig Dataclass
**Plan reference:** Task 5, Step 3
**Design doc equivalent:** Section 3.2
**Description:** Validates training hyperparameters before execution. LoRA rank must be in `{4, 8, 16, 32, 64}`. Learning rate must be in `(0, 1)`. `lora_rank=None` means full fine-tune.
**Code location:** `training/lora_trainer.py`
**Implementation details:**
```python
@dataclass
class TrainingConfig:
    base_model: str
    nr_epochs: int
    learning_rate: float
    batch_size: int
    lora_rank: int | None  # None = full fine-tune
```
`__post_init__` validates `lora_rank` against `VALID_LORA_RANKS = {4, 8, 16, 32, 64}` and `learning_rate` range.

---

## 32. run_lora_training Function
**Plan reference:** Task 5, Step 3
**Design doc equivalent:** Section 3.2
**Description:** Executes LoRA (or full) fine-tuning via Unsloth. Creates the output directory, delegates to `_run_unsloth_training()`, and returns the checkpoint path as `weights_ref`.
**Code location:** `training/lora_trainer.py`
**Implementation details:**
```python
def run_lora_training(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str = "artifacts",
) -> str  # returns weights_ref
```

---

## 33. _run_unsloth_training Internal Function
**Plan reference:** Task 5, Step 3
**Design doc equivalent:** Section 3.2
**Description:** The actual Unsloth training loop. Loads the base model via `FastLanguageModel.from_pretrained()`, applies PEFT if LoRA, loads JSONL dataset, formats examples with a prompt template, creates `SFTTrainer`, trains, and saves checkpoint.
**Code location:** `training/lora_trainer.py`
**Implementation details:** Uses `max_seq_length=512`, `lora_alpha = lora_rank * 2`, `lora_dropout=0.0`, `target_modules="all-linear"`, `bias="none"`. Training args: `save_strategy="epoch"`, `report_to="none"`. Prompt template: `"Classify this SMS message as spam or ham.\n\nMessage: {text}\n\nLabel: {label}"`.

---

## 34. train Function (Agent-Facing Interface)
**Plan reference:** Task 5, Step 4
**Design doc equivalent:** Section 3.1
**Description:** Agent-facing training interface called via the bash tool. Constructs a `TrainingConfig` and delegates to `run_lora_training()`. Always trains from `base_model`, never from a prior checkpoint.
**Code location:** `training/slm_helpers.py`
**Implementation details:**
```python
def train(
    dataset_path: str,
    base_model: str,
    nr_epochs: int,
    learning_rate: float,
    batch_size: int,
    lora_rank: int | None,
    output_dir: str = "artifacts",
) -> str  # returns weights_ref
```

---

## 35. infer Function
**Plan reference:** Task 5, Step 4
**Design doc equivalent:** Section 3.1
**Description:** Single-prompt inference. Loads checkpoint into a lazy cache, generates text with `do_sample=False`, returns only newly generated tokens.
**Code location:** `training/slm_helpers.py`
**Implementation details:**
```python
def infer(prompt: str, weights_ref: str, base_model: str, max_new_tokens: int = 50) -> str
```
Uses `_inference_cache: dict` keyed by `weights_ref` to avoid reloading. Loads via `AutoModelForCausalLM` with `torch.float16` and `device_map="auto"`.

---

## 36. infer_batch Function
**Plan reference:** Task 5, Step 4
**Design doc equivalent:** Section 3.1
**Description:** Parallel inference via `ThreadPoolExecutor` with configurable `max_workers` (default 20). Submits individual `infer()` calls for each prompt.
**Code location:** `training/slm_helpers.py`
**Implementation details:**
```python
def infer_batch(
    prompts: list[str],
    weights_ref: str,
    base_model: str,
    max_new_tokens: int = 50,
    max_workers: int = 20,
) -> list[str]
```

---

## 37. theoretical_hardware_profile Function
**Plan reference:** Task 5, Step 5
**Design doc equivalent:** Section 6.5
**Description:** Returns theoretical hardware estimates for a model from the Android pool. Phase 1 uses benchmark-derived estimates, not measurements. Returns a dict with `phase: "theoretical"`.
**Code location:** `training/quantize.py`
**Implementation details:**
```python
def theoretical_hardware_profile(model_id: str) -> dict
```
Looks up `model_id` in `ANDROID_POOL`. Returns `int4_size_mb`, `tier`, tok/s across three chips, `peak_memory_mb`. Returns `None` values with a note if model not in pool.

---

## 38. AgentState TypedDict
**Plan reference:** Task 6, Step 1
**Design doc equivalent:** Section 2.4
**Description:** The complete LangGraph state dictionary. Every node reads and writes to this. Contains task specification, task analysis outputs, data references, search state, last iteration results, and LangGraph messages.
**Code location:** `agent/state.py`
**Implementation details:**
```python
class AgentState(TypedDict):
    description: str
    target_metric: str
    hardware_constraints: HardwareConstraints
    task_type: str                    # "classification", "NER", or "generation"
    selected_model: Optional[ModelSpec]
    stop_threshold: float             # default 0.96
    train_examples: list[dict]
    eval_set: Optional[EvalSet]
    current_dataset_path: Optional[str]
    dataset_version: int
    best_weights_ref: Optional[str]
    best_score: float
    iteration: int
    scores: list[float]               # f(pi) per iteration
    dag: list[dict]                   # lineage DAG nodes
    consecutive_no_improvement: int
    last_eval: Optional[EvalResult]
    last_intervention: str
    next_action: str                  # "train" | "curate" | "rollback" | "escalate" | "terminate"
    messages: list[Any]
```

---

## 39. web_search Tool
**Plan reference:** Task 6, Step 2
**Design doc equivalent:** Section 2.3
**Description:** Exa deep research API wrapper. Used for locating datasets and surveying published baselines. Returns formatted search results with titles, URLs, and text excerpts.
**Code location:** `agent/tools/web_search.py`
**Implementation details:**
```python
@tool
def web_search(query: str, num_results: int = 5) -> str
```
Uses `exa_py.Exa` client with `search_and_contents()`, `use_autoprompt=True`, `text={"max_characters": 1000}`. Lazy-initializes `_exa_client` from `config.EXA_API_KEY`.

---

## 40. bash Tool
**Plan reference:** Task 6, Step 3
**Design doc equivalent:** Section 2.3
**Description:** Unrestricted shell tool with `slm_helpers.py` pre-loaded via PYTHONPATH injection. Used for running `train()`, `infer_batch()`, dataset operations, and eval scripts. 1-hour timeout for training runs.
**Code location:** `agent/tools/bash_tool.py`
**Implementation details:**
```python
@tool
def bash(command: str) -> str
```
Prepends `export PYTHONPATH=...` to every command. Uses `subprocess.run()` with `timeout=3600`, `capture_output=True`. Returns stdout + stderr combined.

---

## 41. read_file Tool
**Plan reference:** Task 6, Step 4
**Design doc equivalent:** Section 2.3
**Description:** Reads a file from disk. Used for datasets, configs, data-curation.md, and eval results. Returns file contents or an error message.
**Code location:** `agent/tools/file_tools.py`
**Implementation details:**
```python
@tool
def read_file(path: str) -> str
```

---

## 42. edit_file Tool
**Plan reference:** Task 6, Step 4
**Design doc equivalent:** Section 2.3
**Description:** Writes content to a file, creating parent directories if needed.
**Code location:** `agent/tools/file_tools.py`
**Implementation details:**
```python
@tool
def edit_file(path: str, content: str) -> str
```
Returns confirmation with character count.

---

## 43. delegate_task Function
**Plan reference:** Task 6, Step 5
**Design doc equivalent:** Section 2.2
**Description:** Sub-agent spawning mechanism. Not a named tool -- called programmatically by the orchestrator. Spawns a sub-agent that shares the filesystem with the main agent. Main agent reads output files rather than consuming raw sub-agent context (context isolation pattern).
**Code location:** `agent/tools/delegate_task.py`
**Implementation details:**
```python
def delegate_task(task_description: str, output_file: str) -> str
```
Phase 1 implementation is single-turn: creates one Claude message with a system prompt instructing the sub-agent to write its output to `output_file`. Returns file contents if written, otherwise raw response text.

---

## 44. task_analysis_node (Node 1)
**Plan reference:** Task 7, Step 5
**Design doc equivalent:** Section 2.4, Node 1
**Description:** Classifies task type, selects starting model from Android pool (always Tier 1 smallest), and sets stop threshold. Runs three sub-stages: task classification, data acquisition, baseline survey. All decisions are made before any data is touched.
**Code location:** `agent/nodes/task_analysis.py`
**Implementation details:**
```python
def task_analysis_node(state: AgentState) -> AgentState
```
Sets `state["task_type"]`, `state["selected_model"]` (first element of `filter_pool()`), and `state["stop_threshold"] = 0.96`. Raises `RuntimeError` if no models satisfy hardware constraints.

---

## 45. eval_setup_node (Node 2)
**Plan reference:** Task 7, Step 5
**Design doc equivalent:** Section 2.4, Node 2
**Description:** Downloads data and builds the held-out evaluation set `E = Epos + Eneg + Eboundary`. Eval set is built before any training and fixed throughout all iterations.
**Code location:** `agent/nodes/eval_setup.py`
**Implementation details:**
```python
def eval_setup_node(state: AgentState) -> AgentState
```
Calls `download_sms_spam()` to get train/test splits, then `build_eval_set(test_examples)`. Sets `state["train_examples"]` and `state["eval_set"]`.

---

## 46. train_node (Node 3)
**Plan reference:** Task 7, Step 5
**Design doc equivalent:** Section 2.4, Node 3
**Description:** Fires at least two parallel training configurations using `ThreadPoolExecutor(max_workers=2)`. Always trains from the base model. Stores both `weights_ref` results and config descriptions in state for the evaluate node to select the best.
**Code location:** `agent/nodes/train.py`
**Implementation details:**
```python
def train_node(state: AgentState) -> AgentState
```
Default configs: Config A (LoRA r=8, lr=2e-4, epochs=3, batch_size=8), Config B (full_ft, lr=1e-4, epochs=5, batch_size=4). Increments `state["iteration"]`. Sets `state["_pending_weights_refs"]` and `state["_pending_configs"]`. Helper: `_save_dataset(examples, version) -> str` saves JSONL to artifacts dir.

---

## 47. evaluate_node (Node 4)
**Plan reference:** Task 7, Step 5
**Design doc equivalent:** Section 2.4, Node 4
**Description:** Scores each trained configuration against the held-out eval set E. Selects the best-scoring configuration. Updates best score tracking. Records the iteration in the lineage DAG. Writes the iteration to `data-curation.md` via `CurationLog`. Logs theoretical hardware profile.
**Code location:** `agent/nodes/evaluate.py`
**Implementation details:**
```python
def evaluate_node(state: AgentState) -> AgentState
```
Iterates over `state["_pending_weights_refs"]`, calls `run_eval()` for each, selects the one with highest `f1`. Updates: `best_score`, `best_weights_ref`, `consecutive_no_improvement`, `scores`, `last_eval`, `dag`. Calls `CurationLog().write_iteration()` and `theoretical_hardware_profile()`.

---

## 48. apply_iteration_policy Function
**Plan reference:** Task 7, Step 3
**Design doc equivalent:** Section 2.4, Node 5; Section 5.2
**Description:** The shared iteration policy based on current `f(pi)`. Returns the score band, intervention type, and description. Three bands: `<0.80` (data problem, rebuild dataset), `0.80-0.95` (optimization problem, tune hyperparameters), `>=0.95` (surgical, add 2-3 targeted examples per failure pattern).
**Code location:** `agent/nodes/iterate.py`
**Implementation details:**
```python
def apply_iteration_policy(score: float) -> dict
```
Returns `{"band": str, "intervention": str, "description": str}`. Intervention values: `"data_rebuild"`, `"hyperparameter"`, `"surgical"`.

---

## 49. iterate_node (Node 5)
**Plan reference:** Task 7, Step 3
**Design doc equivalent:** Section 2.4, Node 5
**Description:** Determines the next action based on current score, stop threshold, and stagnation count. Routes to terminate (if score >= threshold), escalate (if 2+ consecutive rounds with no improvement), or curate (otherwise).
**Code location:** `agent/nodes/iterate.py`
**Implementation details:**
```python
def iterate_node(state: AgentState) -> AgentState
```
Sets `state["last_intervention"]` and `state["next_action"]` (one of `"terminate"`, `"escalate"`, `"curate"`).

---

## 50. should_rollback Function
**Plan reference:** Task 7, Step 4
**Design doc equivalent:** Section 2.4, Node 7; Section 5.3
**Description:** Cold-start simple rollback check: returns True if the latest score is less than the previous score. Returns False on first iteration (fewer than 2 scores).
**Code location:** `agent/nodes/rollback.py`
**Implementation details:**
```python
def should_rollback(state: AgentState) -> bool
```

---

## 51. rollback_node (Node 7)
**Plan reference:** Task 7, Step 4
**Design doc equivalent:** Section 2.4, Node 7
**Description:** Reverts to the previous configuration if score decreased. Removes the last score from history. Sets `last_intervention = "rollback"`. Increments `consecutive_no_improvement`. Marks the last DAG node as pruned.
**Code location:** `agent/nodes/rollback.py`
**Implementation details:**
```python
def rollback_node(state: AgentState) -> AgentState
```

---

## 52. curate_node (Node 6)
**Plan reference:** Task 7, Step 5
**Design doc equivalent:** Section 2.4, Node 6
**Description:** Builds or augments the training dataset based on the current intervention type. For `data_rebuild` or initial build: constructs fresh curriculum via `build_initial_curriculum()` + `synthesize_hard_negatives()` + `apply_quality_controls()`. For `surgical`: loads existing dataset and appends targeted hard negatives for remaining failures. For `hyperparameter`: returns state unchanged (dataset held fixed). Saves JSONL to disk and increments `dataset_version`.
**Code location:** `agent/nodes/curate.py`
**Implementation details:**
```python
def curate_node(state: AgentState) -> AgentState
```
Uses `anthropic.Anthropic` client for hard negative synthesis. Writes to `artifacts/dataset_v{N}.jsonl`. Updates `state["current_dataset_path"]` and `state["dataset_version"]`.

---

## 53. escalate_node (Node 8)
**Plan reference:** Task 7, Step 5
**Design doc equivalent:** Section 2.4, Node 8
**Description:** Escalates to the next model tier or terminates. Finds the current model's position in the feasible pool, selects the next larger model. Checks hardware constraints (storage and memory as hard gates even in Phase 1). If no larger model available or constraints violated, terminates.
**Code location:** `agent/nodes/escalate.py`
**Implementation details:**
```python
def escalate_node(state: AgentState) -> AgentState
```
Sets `state["selected_model"]` to the next feasible model, resets `consecutive_no_improvement` to 0, sets `next_action = "train"`. If no escalation possible, sets `next_action = "terminate"`.

---

## 54. LangGraph State Machine (Graph Assembly)
**Plan reference:** Task 8, Step 3
**Design doc equivalent:** Section 2.4
**Description:** The complete LangGraph state machine wiring all 8 nodes. Linear entry path: `task_analysis -> eval_setup -> curate -> train -> evaluate`. Conditional routing after evaluate (rollback or iterate). Conditional routing after iterate (train, curate, escalate, or terminate). After rollback: goes to iterate. After escalate: train or terminate.
**Code location:** `agent/graph.py`
**Implementation details:**
```python
def build_graph() -> CompiledGraph
```
Routing functions:
- `_route_after_evaluate(state)` -- returns `"rollback"` if `should_rollback()`, else `"iterate"`
- `_route_after_iterate(state)` -- returns `state["next_action"]` (one of `"train"`, `"curate"`, `"escalate"`, `"terminate"`)
- `_route_after_escalate(state)` -- returns `state["next_action"]` (one of `"train"`, `"terminate"`)

---

## 55. run_cold_start Entry Point
**Plan reference:** Task 8, Step 4
**Design doc equivalent:** Section 2.4 (input triple)
**Description:** Main entry point for Phase 1. Constructs the initial `AgentState` from the task description and hardware constraints, builds the graph, invokes it, and prints the final summary (best F1, iterations, model, weights, score trajectory).
**Code location:** `run.py`
**Implementation details:**
```python
def run_cold_start(
    description: str = "fine-tune on SMS Spam binary classification",
    hardware_constraints: HardwareConstraints | None = None,
) -> AgentState
```
Default constraints: `storage_mb=800, memory_mb=1200, latency_ttft_ms=2000, power_watts=5.0, target_chip="snapdragon_778g"`. Initial state sets: `best_score=0.0`, `iteration=0`, `dataset_version=0`, `stop_threshold=0.96`, `next_action="train"`.

---

## 56. Test: Android Pool Filter by Storage
**Plan reference:** Task 1, Step 1
**Design doc equivalent:** Section 6.3
**Description:** Verifies that `filter_pool()` returns only models with `int4_size_mb <= constraints.storage_mb` and returns at least one result for reasonable constraints.
**Code location:** `tests/test_android_pool.py`
**Implementation details:** `test_filter_pool_by_storage()` -- constraints: storage_mb=500, memory_mb=1200, latency_ttft_ms=2000, power_watts=5.0. Asserts all results have `int4_size_mb <= 500` and `len(result) > 0`.

---

## 57. Test: Android Pool Returns Tier 1 First
**Plan reference:** Task 1, Step 1
**Design doc equivalent:** Section 6.3
**Description:** Verifies that the first result from `filter_pool()` has `tier == 1` when all tiers are feasible.
**Code location:** `tests/test_android_pool.py`
**Implementation details:** `test_filter_pool_returns_tier1_first()` -- constraints: storage_mb=1600, memory_mb=2000. Asserts `result[0].tier == 1`.

---

## 58. Test: Android Pool Excludes Oversized
**Plan reference:** Task 1, Step 1
**Design doc equivalent:** Section 6.3
**Description:** Verifies that `filter_pool()` returns empty list when constraints are too tight for any model.
**Code location:** `tests/test_android_pool.py`
**Implementation details:** `test_filter_pool_excludes_oversized()` -- constraints: storage_mb=100, memory_mb=200. Asserts `len(result) == 0`.

---

## 59. Test: binary_f1 Perfect Score
**Plan reference:** Task 3, Step 1
**Design doc equivalent:** Section 3.3
**Description:** Verifies F1 = 1.0 when all predictions match labels.
**Code location:** `tests/test_metrics.py`
**Implementation details:** `test_binary_f1_perfect()` -- 4 examples, all correct. Asserts `binary_f1(...) == 1.0`.

---

## 60. Test: binary_f1 All Wrong
**Plan reference:** Task 3, Step 1
**Design doc equivalent:** Section 3.3
**Description:** Verifies F1 = 0.0 when all predictions are inverted (spam predicted as ham and vice versa).
**Code location:** `tests/test_metrics.py`
**Implementation details:** `test_binary_f1_all_wrong()` -- asserts `binary_f1(...) == 0.0`.

---

## 61. Test: binary_f1 Partial
**Plan reference:** Task 3, Step 1
**Design doc equivalent:** Section 3.3
**Description:** Verifies F1 = 0.5 for a specific half-correct prediction pattern (one TP, one FN, one FP, one TN).
**Code location:** `tests/test_metrics.py`
**Implementation details:** `test_binary_f1_partial()` -- asserts `abs(binary_f1(...) - 0.5) < 1e-6`.

---

## 62. Test: per_slice_scores
**Plan reference:** Task 3, Step 1
**Design doc equivalent:** Section 3.3
**Description:** Verifies per-slice accuracy computation returns 1.0 for all slices when all predictions are correct.
**Code location:** `tests/test_metrics.py`
**Implementation details:** `test_per_slice_scores()` -- constructs an `EvalSet` with 2 pos, 1 neg, 1 boundary. Predictions all correct. Asserts all three slices equal 1.0.

---

## 63. Test: run_eval Classification Returns EvalResult
**Plan reference:** Task 3, Step 1
**Design doc equivalent:** Section 3.3
**Description:** Verifies that `run_eval()` with mocked `infer_batch` returns a valid `EvalResult` with F1 in [0, 1] and failures as a list.
**Code location:** `tests/test_harness.py`
**Implementation details:** `test_run_eval_classification_returns_eval_result()` -- patches `infer_batch` to return `["spam", "ham", "ham"]`. Asserts `isinstance(result, EvalResult)`, `0.0 <= result.f1 <= 1.0`, `isinstance(result.failures, list)`.

---

## 64. Test: run_eval Identifies Failures
**Plan reference:** Task 3, Step 1
**Design doc equivalent:** Section 3.3
**Description:** Verifies that failures are correctly identified when predictions miss positive examples.
**Code location:** `tests/test_harness.py`
**Implementation details:** `test_run_eval_identifies_failures()` -- all predictions are `"ham"`, so the spam example is a failure. Asserts `any(f["label"] == "spam" for f in result.failures)`.

---

## 65. Test: run_eval Unknown Task Type Raises
**Plan reference:** Task 3, Step 1
**Design doc equivalent:** Section 3.3
**Description:** Verifies that passing an invalid `task_type` raises `ValueError`.
**Code location:** `tests/test_harness.py`
**Implementation details:** `test_run_eval_unknown_task_type_raises()` -- `task_type="regression"`. Uses `pytest.raises(ValueError, match="Unknown task_type")`.

---

## 66. Test: Initial Curriculum Size
**Plan reference:** Task 4, Step 1
**Design doc equivalent:** Section 4.2
**Description:** Verifies that the initial curriculum has between 100 and 200 examples.
**Code location:** `tests/test_curriculum.py`
**Implementation details:** `test_initial_curriculum_size()` -- uses 1000 fake training examples (200 spam, 800 ham), `n_total=150`. Asserts `100 <= len(dataset) <= 200`.

---

## 67. Test: Initial Curriculum Excludes Eval
**Plan reference:** Task 4, Step 1
**Design doc equivalent:** Section 4.2
**Description:** Verifies that no examples from the eval set appear in the curriculum.
**Code location:** `tests/test_curriculum.py`
**Implementation details:** `test_initial_curriculum_excludes_eval()` -- computes `eval_texts` from `FAKE_EVAL.all`, asserts none appear in the curriculum.

---

## 68. Test: Initial Curriculum Label Balance
**Plan reference:** Task 4, Step 1
**Design doc equivalent:** Section 4.2
**Description:** Verifies the 3x label balance constraint: no label exceeds 3x the count of any other.
**Code location:** `tests/test_curriculum.py`
**Implementation details:** `test_initial_curriculum_label_balance()` -- counts spam and ham in the result, asserts `len(ham) <= 3 * len(spam)` and `len(spam) <= 3 * len(ham)`.

---

## 69. Test: Quality Controls Label Balance
**Plan reference:** Task 4, Step 1
**Design doc equivalent:** Section 2.4, Node 6
**Description:** Verifies that `apply_quality_controls()` trims an imbalanced dataset (10 spam, 100 ham) to satisfy the 3x constraint.
**Code location:** `tests/test_curriculum.py`
**Implementation details:** `test_quality_controls_label_balance()` -- asserts `len(ham) <= 3 * len(spam)` after quality controls.

---

## 70. Test: CurationLog Write and Read
**Plan reference:** Task 4, Step 1
**Design doc equivalent:** Section 4.3
**Description:** Verifies that `CurationLog.write_iteration()` writes to a temp file and `read_latest()` returns content containing the iteration number, dataset version, F1 score, and task type.
**Code location:** `tests/test_curation_log.py`
**Implementation details:** `test_write_and_read_iteration()` -- creates a `CurationLog` with a temp file path, writes iteration 1, reads back, asserts `"Iteration 1"`, `"v1"`, `"0.85"`, and `"classification"` all appear in content.

---

## 71. Test: TrainingConfig Validation
**Plan reference:** Task 5, Step 1
**Design doc equivalent:** Section 3.2
**Description:** Verifies that `TrainingConfig` accepts valid LoRA ranks and learning rates.
**Code location:** `tests/test_lora_trainer.py`
**Implementation details:** `test_training_config_validation()` -- creates config with `lora_rank=8, learning_rate=2e-4`. Asserts `config.lora_rank in {4, 8, 16, 32, 64}` and `0 < config.learning_rate < 1`.

---

## 72. Test: run_lora_training Returns weights_ref
**Plan reference:** Task 5, Step 1
**Design doc equivalent:** Section 3.2
**Description:** Verifies that `run_lora_training()` returns a non-empty string `weights_ref` when Unsloth training is mocked.
**Code location:** `tests/test_lora_trainer.py`
**Implementation details:** `test_run_lora_training_returns_weights_ref()` -- creates a temp JSONL dataset (20 spam + 20 ham), patches `_run_unsloth_training` to return a mock checkpoint path, asserts the result is a non-empty string.

---

## 73. Test: Iteration Policy -- Data Rebuild Band
**Plan reference:** Task 7, Step 1
**Design doc equivalent:** Section 2.4 Node 5, Section 5.2
**Description:** Verifies that score 0.75 maps to `<0.80` band with `data_rebuild` intervention.
**Code location:** `tests/test_nodes.py`
**Implementation details:** `test_iteration_policy_data_rebuild()` -- `apply_iteration_policy(score=0.75)`. Asserts `band == "<0.80"` and `intervention == "data_rebuild"`.

---

## 74. Test: Iteration Policy -- Hyperparameter Band
**Plan reference:** Task 7, Step 1
**Design doc equivalent:** Section 2.4 Node 5, Section 5.2
**Description:** Verifies that score 0.87 maps to `0.80-0.95` band with `hyperparameter` intervention.
**Code location:** `tests/test_nodes.py`
**Implementation details:** `test_iteration_policy_hyperparameter()` -- `apply_iteration_policy(score=0.87)`. Asserts `band == "0.80-0.95"` and `intervention == "hyperparameter"`.

---

## 75. Test: Iteration Policy -- Surgical Band
**Plan reference:** Task 7, Step 1
**Design doc equivalent:** Section 2.4 Node 5, Section 5.2
**Description:** Verifies that score 0.97 maps to `>=0.95` band with `surgical` intervention.
**Code location:** `tests/test_nodes.py`
**Implementation details:** `test_iteration_policy_surgical()` -- `apply_iteration_policy(score=0.97)`. Asserts `band == ">=0.95"` and `intervention == "surgical"`.

---

## 76. Test: Rollback Triggers on Decrease
**Plan reference:** Task 7, Step 1
**Design doc equivalent:** Section 5.3
**Description:** Verifies that `should_rollback()` returns True when the latest score is lower than the previous score.
**Code location:** `tests/test_nodes.py`
**Implementation details:** `test_rollback_triggers_on_decrease()` -- `scores = [0.85, 0.82]`. Asserts `should_rollback(state) is True`.

---

## 77. Test: Rollback Does Not Trigger on Improvement
**Plan reference:** Task 7, Step 1
**Design doc equivalent:** Section 5.3
**Description:** Verifies that `should_rollback()` returns False when the latest score improved.
**Code location:** `tests/test_nodes.py`
**Implementation details:** `test_rollback_does_not_trigger_on_improvement()` -- `scores = [0.82, 0.87]`. Asserts `should_rollback(state) is False`.

---

## 78. Test: Rollback Does Not Trigger on First Iteration
**Plan reference:** Task 7, Step 1
**Design doc equivalent:** Section 5.3
**Description:** Verifies that `should_rollback()` returns False when there is only one score (no prior to compare against).
**Code location:** `tests/test_nodes.py`
**Implementation details:** `test_rollback_does_not_trigger_on_first_iteration()` -- `scores = [0.75]`. Asserts `should_rollback(state) is False`.

---

## 79. Test: Graph Builds Without Error
**Plan reference:** Task 8, Step 1
**Design doc equivalent:** Section 2.4
**Description:** Verifies that `build_graph()` returns a non-None compiled graph.
**Code location:** `tests/test_graph.py`
**Implementation details:** `test_graph_builds_without_error()` -- asserts `graph is not None`.

---

## 80. Test: Graph Has Expected Nodes
**Plan reference:** Task 8, Step 1
**Design doc equivalent:** Section 2.4
**Description:** Verifies that all 8 state machine nodes are present in the compiled graph.
**Code location:** `tests/test_graph.py`
**Implementation details:** `test_graph_has_expected_nodes()` -- checks that `graph.nodes.keys()` is a superset of `{"task_analysis", "eval_setup", "train", "evaluate", "iterate", "curate", "rollback", "escalate"}`.

---

## 81. End-to-End Smoke Test
**Plan reference:** Task 8, Step 6
**Design doc equivalent:** Section 5.5
**Description:** Full pipeline smoke test with mocked training and inference. Verifies the graph runs to completion and performs at least one iteration. No GPU required.
**Code location:** `tests/test_graph.py` (inline script in plan)
**Implementation details:** Patches `agent.nodes.train.slm_train` and `eval.harness.infer_batch` with mock returns. Calls `run_cold_start()`. Asserts `state["iteration"] > 0`.

---

## 82. File Structure and Package Layout
**Plan reference:** File Structure section
**Design doc equivalent:** Section 2 (architecture overview)
**Description:** The complete directory structure with all packages, modules, and test files.
**Code location:** Project root
**Implementation details:**
- `agent/` -- `__init__.py`, `graph.py`, `state.py`
- `agent/nodes/` -- `__init__.py`, `task_analysis.py`, `eval_setup.py`, `train.py`, `evaluate.py`, `iterate.py`, `curate.py`, `rollback.py`, `escalate.py`
- `agent/tools/` -- `__init__.py`, `web_search.py`, `bash_tool.py`, `file_tools.py`, `delegate_task.py`
- `training/` -- `__init__.py`, `slm_helpers.py`, `lora_trainer.py`, `quantize.py`
- `data/` -- `__init__.py`, `eval_set.py`, `curriculum.py`, `curation_log.py`
- `data/loaders/` -- `__init__.py`, `sms_spam.py`
- `eval/` -- `__init__.py`, `harness.py`, `metrics.py`
- `eval/scorers/` -- `__init__.py`, `classification.py`, `ner.py`, `generation.py`
- `tests/` -- `test_sms_spam.py`, `test_eval_set.py`, `test_curriculum.py`, `test_curation_log.py`, `test_metrics.py`, `test_harness.py`, `test_android_pool.py`, `test_lora_trainer.py`, `test_nodes.py`, `test_graph.py`
- Root: `android_pool.py`, `config.py`, `run.py`, `data-curation.md` (runtime artifact)

---

## 83. Requirements / Dependencies
**Plan reference:** Task 1, Step 5
**Design doc equivalent:** Architecture Overview (tech stack)
**Description:** Python package dependencies for the project.
**Code location:** `requirements.txt`
**Implementation details:** `anthropic>=0.40.0`, `langgraph>=0.2.0`, `langchain-anthropic>=0.3.0`, `unsloth>=2024.12.0`, `transformers>=4.47.0`, `peft>=0.14.0`, `datasets>=3.0.0`, `torch>=2.4.0`, `llama-cpp-python>=0.3.0`, `exa-py>=1.0.0`, `pytest>=8.0.0`, `python-dotenv>=1.0.0`.

---

## 84. Graph Routing Logic
**Plan reference:** Task 8, Step 3
**Design doc equivalent:** Section 2.4
**Description:** Three conditional routing functions that implement the state machine transitions after evaluate, iterate, and escalate nodes.
**Code location:** `agent/graph.py`
**Implementation details:**
- `_route_after_evaluate(state)` -- returns `"rollback"` if `should_rollback(state)` else `"iterate"`
- `_route_after_iterate(state)` -- returns `state.get("next_action", "curate")` with valid destinations: `"train"`, `"curate"`, `"escalate"`, `END`
- `_route_after_escalate(state)` -- returns `state.get("next_action", "terminate")` with valid destinations: `"train"`, `END`

Edge topology: `task_analysis -> eval_setup -> curate -> train -> evaluate -> (rollback | iterate)`. `rollback -> iterate`. `iterate -> (train | curate | escalate | END)`. `curate -> train`. `escalate -> (train | END)`.
