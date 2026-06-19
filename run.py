# run.py
"""
Entry point for SLM Factory Phase 1 cold-start loop.

Usage:
    python run.py

Environment variables required:
    ANTHROPIC_API_KEY
    EXA_API_KEY
"""
import os
from dotenv import load_dotenv
load_dotenv()

from android_pool import HardwareConstraints
from agent.graph import build_graph
from agent.state import AgentState


def _infer_task_type(description: str) -> str:
    """
    Derive task_type from the task description.
    Matches one of: "classification", "NER", "generation".
    """
    desc_lower = description.lower()
    # Check classification first (most specific keywords)
    if any(kw in desc_lower for kw in (
        "classification", "classify", "spam", "sentiment", "binary", "clinc",
    )):
        return "classification"
    # NER keywords
    if any(kw in desc_lower for kw in (
        "ner", "named entity", "entity extraction", "tagging",
        "conll", "conll-2003", "extraction",
    )):
        return "NER"
    # Generation keywords — includes common paper benchmarks
    if any(kw in desc_lower for kw in (
        "generation", "generate", "summarize", "summarization", "translation",
        "arc", "reasoning", "science",
        "gsm", "math",
        "triviaqa", "trivia",
        "humaneval", "code",
        "xsum", "samsum",
    )):
        return "generation"
    raise ValueError(
        f"Cannot infer task_type from description: {description!r}. "
        "Include a keyword like 'classification', 'NER', or 'generation'."
    )


def run_cold_start(
    description: str = "fine-tune on SMS Spam binary classification",
    hardware_constraints: HardwareConstraints | None = None,
) -> AgentState:
    if hardware_constraints is None:
        hardware_constraints = HardwareConstraints(
            storage_mb=800,
            memory_mb=1200,
            latency_ttft_ms=2000,
            power_watts=5.0,
            target_chip="snapdragon_778g",
        )

    task_type = _infer_task_type(description)

    initial_state: AgentState = {
        "description": description,
        "target_metric": "F1",
        "hardware_constraints": hardware_constraints,
        "task_type": task_type,
        "selected_model": None,
        "stop_threshold": 0.96,
        "train_examples": [],
        "eval_set": None,
        "current_dataset_path": None,
        "dataset_version": 0,
        "best_weights_ref": None,
        "best_score": 0.0,
        "iteration": 0,
        "scores": [],
        "dag": [],
        "consecutive_no_improvement": 0,
        "last_eval": None,
        "last_intervention": "",
        "next_action": "train",
        "messages": [],
        "_pending_weights_refs": None,
        "_pending_configs": None,
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    print(f"\n=== Phase 1 Complete ===")
    print(f"Best F1: {final_state['best_score']:.4f}")
    print(f"Iterations: {final_state['iteration']}")
    print(f"Best model: {final_state['selected_model'].model_id}")
    print(f"Best weights: {final_state['best_weights_ref']}")
    print(f"Score trajectory: {[f'{s:.4f}' for s in final_state['scores']]}")
    print(f"See data-curation.md for full lineage.")

    return final_state


if __name__ == "__main__":
    run_cold_start()
