# agent/state.py
from typing import TypedDict, Optional, Any
from data.eval_set import EvalSet
from eval.harness import EvalResult
from android_pool import ModelSpec, HardwareConstraints

class AgentState(TypedDict):
    # Task specification
    description: str
    target_metric: str
    hardware_constraints: HardwareConstraints

    # Task analysis outputs
    task_type: str                    # "classification", "NER", or "generation"
    selected_model: Optional[ModelSpec]
    stop_threshold: float             # default 0.96, agent may lower

    # Data
    train_examples: list[dict]
    eval_set: Optional[EvalSet]
    current_dataset_path: Optional[str]  # path to Dcold JSONL on disk
    dataset_version: int              # incremented each curate call

    # Search state
    best_weights_ref: Optional[str]
    best_score: float
    iteration: int
    scores: list[float]               # f(π) per iteration
    dag: list[dict]                   # lineage DAG nodes
    consecutive_no_improvement: int

    # Last iteration results
    last_eval: Optional[EvalResult]
    last_intervention: str            # "data_rebuild" | "hyperparameter" | "surgical" | "rollback"
    next_action: str                  # "train" | "curate" | "rollback" | "escalate" | "terminate"

    # Messages for LangGraph
    messages: list[Any]
