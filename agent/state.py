# agent/state.py
from typing import TypedDict, Optional
from data.eval_set import EvalSet
from eval.harness import EvalResult
from config.android_pool import ModelSpec, HardwareConstraints

class AgentState(TypedDict):
    # Task specification
    description: str
    target_metric: str
    hardware_constraints: HardwareConstraints

    # Task analysis outputs
    task_type: str                    # "classification" | "NER" | "math_reasoning" | "code_generation" | "generation"
    selected_model: Optional[ModelSpec]
    feasible_models: list[ModelSpec]       # models that passed hardware_filter (Stages 1+2), largest→smallest
    stop_threshold: float             # calibrated target; iterate_node may lower mid-run
    initial_stop_threshold: float     # set once at plan time; stop_threshold can never go below this
    task_plan: Optional[dict]         # orchestrator's autonomous plan (labels, exa_queries, ...)
    autonomous: bool                  # if True, task_analysis derives task_type/plan via LLM

    # Data
    train_examples: list[dict]
    eval_set: Optional[EvalSet]
    data_source: Optional[str]        # provenance of train/eval examples (real benchmark vs web/Exa)
    current_dataset_path: Optional[str]  # path to Dcold JSONL on disk
    dataset_version: int              # incremented each curate call

    # Search state
    best_weights_ref: Optional[str]
    best_score: float
    lifetime_best_score: float          # max best_score ever seen across all tiers/models in this run
    iteration: int
    scores: list[float]               # f(π) per iteration
    dag: list[dict]                   # lineage DAG nodes
    consecutive_no_improvement: int
    downward_probe_done: bool         # True once the one-tier-down probe has run for this terminal model

    # Last iteration results
    last_eval: Optional[EvalResult]
    last_curation: Optional[dict]     # {n_gold, n_hard, label_dist} for the dataset feeding the next train
    last_intervention: str            # "data_rebuild" | "hyperparameter" | "surgical" | "rollback"
    last_hypothesis: str              # LLM-generated causal reasoning for the next intervention
    llm_iterate_decision: Optional[dict]  # full LLM decision JSON from iterate_node
    next_action: str                  # "train" | "curate" | "rollback" | "escalate" | "terminate"

    # Baseline tracking: one entry per model tried, records zero-shot vs best fine-tuned
    model_baselines: list[dict]       # [{model_id, baseline_f1, best_finetuned_f1}, ...]

    # Phase 2 flags
    quantize_enabled: bool                # True runs INT4 quantization after eval
    hw_gating_enabled: bool               # True makes latency/power hard gates

    # Production mode (paper §2.6)
    mode: str                             # "cold_start" | "production"
    deployed_model_ref: Optional[str]     # M0 — the deployed model being improved
    traces: Optional[list[dict]]          # T — judged inference traces
    failure_taxonomy: Optional[dict]      # {category: [trace_ids]}
    regression_set: Optional[list[dict]]  # R — examples M0 gets right
    replay_buffer: Optional[list[dict]]   # D_replay ⊂ D_parent (10-20%)
    turn_budget: int                      # 1500 cold-start, 500 production

    # Internal: carry training results between train_node and evaluate_node
    _pending_weights_refs: Optional[dict]  # label -> weights_ref
    _pending_training_outputs: Optional[dict]  # label -> TrainingOutput
    _pending_configs: Optional[dict]       # label -> config dict
