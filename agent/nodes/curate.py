# agent/nodes/curate.py
import json
import os
from agent.state import AgentState
from data.curriculum import (
    build_initial_curriculum,
    synthesize_hard_negatives,
    apply_quality_controls,
)

ARTIFACTS_DIR = "artifacts"


def curate_node(state: AgentState) -> AgentState:
    """
    Node 6: build or augment Dcold = Dgold ∪ Dhard based on current intervention type.
    Writes updated dataset to disk. Increments dataset_version.
    task_type flows from state — no hardcoding.
    """
    import anthropic
    from config import ANTHROPIC_API_KEY

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    task_type = state["task_type"]
    intervention = state.get("last_intervention", "data_rebuild")
    eval_set = state["eval_set"]
    train_examples = state["train_examples"]
    failures = state["last_eval"].failures if state.get("last_eval") else []

    if intervention == "data_rebuild" or state["current_dataset_path"] is None:
        # Build fresh curriculum from gold data + hard negatives
        gold = build_initial_curriculum(train_examples, eval_set, n_total=150)
        source_examples = failures[:10] if failures else train_examples[:10]
        hard = synthesize_hard_negatives(
            source_examples,
            n=50,
            anthropic_client=client,
            task_type=task_type,
        )
        dataset = apply_quality_controls(gold + hard, task_type=task_type)

    elif intervention == "surgical":
        # Load existing dataset and add 2–3 targeted examples per failure pattern
        with open(state["current_dataset_path"]) as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        targeted = synthesize_hard_negatives(
            failures[:5],
            n=min(len(failures), 20),
            anthropic_client=client,
            task_type=task_type,
        )
        dataset = apply_quality_controls(dataset + targeted, task_type=task_type)

    else:
        # Hyperparameter intervention — hold dataset fixed, no curation needed
        return state

    # Save updated dataset to disk
    state["dataset_version"] += 1
    path = os.path.join(ARTIFACTS_DIR, f"dataset_v{state['dataset_version']}.jsonl")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    state["current_dataset_path"] = path
    return state
