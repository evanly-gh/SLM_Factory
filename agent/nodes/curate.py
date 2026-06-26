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

    # Pull targeted_pattern from the LLM's surgical decision (if available)
    llm_decision = state.get("llm_iterate_decision") or {}
    targeted_pattern = llm_decision.get("targeted_patterns") or ""

    if intervention == "data_rebuild" or state["current_dataset_path"] is None:
        # Build fresh curriculum targeting 65:35 Dgold:Dhard (paper §2.3).
        # Step 1: select gold examples (65% of target total).
        N_TOTAL = 150
        n_gold_target = int(N_TOTAL * 0.65)
        n_hard_target = N_TOTAL - n_gold_target  # ~52

        gold = build_initial_curriculum(train_examples, eval_set, n_total=n_gold_target)

        # Step 2: synthesize hard negatives from failures first, then gold examples.
        # Use the full range of labels (not just minority class) to cover all failure modes.
        source_examples = (failures[:n_hard_target] if len(failures) >= n_hard_target
                           else failures + train_examples[:n_hard_target - len(failures)])
        hard = synthesize_hard_negatives(
            source_examples,
            n=n_hard_target,
            anthropic_client=client,
            task_type=task_type,
        )
        dataset = apply_quality_controls(gold + hard, task_type=task_type)
        n_hard_added = len(hard)

    elif intervention == "surgical":
        # Load existing dataset and add targeted examples per failure pattern.
        with open(state["current_dataset_path"]) as f:
            dataset = [json.loads(line) for line in f if line.strip()]
        # For surgical: target 2–3 new examples per failure pattern, up to 20 total.
        n_surgical = min(max(len(failures) * 2, 10), 20)
        targeted = synthesize_hard_negatives(
            failures[:n_surgical],
            n=n_surgical,
            anthropic_client=client,
            task_type=task_type,
            targeted_pattern=targeted_pattern,
        )
        dataset = apply_quality_controls(dataset + targeted, task_type=task_type)
        n_hard_added = len(targeted)

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

    # Record real dataset composition so evaluate_node can log it (not placeholders).
    from collections import Counter
    n_hard = min(n_hard_added, len(dataset))
    state["last_curation"] = {
        "n_gold": len(dataset) - n_hard,
        "n_hard": n_hard,
        "label_dist": dict(Counter(ex.get("label", "?") for ex in dataset)),
    }
    return state
