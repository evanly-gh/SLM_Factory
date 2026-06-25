# agent/nodes/train.py
import os
from agent.state import AgentState
from training.slm_helpers import train as slm_train

ARTIFACTS_DIR = "artifacts"


def train_node(state: AgentState) -> AgentState:
    """
    Node 3: train >=2 configurations and select the best by f(pi) on E.
    Always trains from the base model — never from a prior checkpoint.

    The configs are trained SEQUENTIALLY, not in parallel threads: Unsloth/trl share
    process-global state (Accelerate's PartialState singleton, runtime class patching), so
    two trainers in one process race (device_map distributed-mode errors, SFTConfig pickling).
    On a single GPU parallel threads give no speedup anyway.
    """
    model_id = state["selected_model"].model_id
    dataset_path = state["current_dataset_path"]

    configs = [
        {"nr_epochs": 3, "learning_rate": 2e-4, "batch_size": 8,  "lora_rank": 8,    "label": "A: LoRA r=8"},
        {"nr_epochs": 5, "learning_rate": 1e-4, "batch_size": 4,  "lora_rank": None, "label": "B: full_ft"},
    ]

    # Increment iteration counter BEFORE building output dirs so the directory name
    # matches the iteration number recorded in the DAG and data-curation.md.
    state["iteration"] += 1

    results = {}
    for cfg in configs:
        results[cfg["label"]] = slm_train(
            dataset_path=dataset_path,
            base_model=model_id,
            nr_epochs=cfg["nr_epochs"],
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            lora_rank=cfg["lora_rank"],
            output_dir=os.path.join(
                ARTIFACTS_DIR,
                f"iter{state['iteration']}_{cfg['label'].split(':')[0].strip()}",
            ),
        )

    # Store both weights_refs; evaluate node will select best
    state["_pending_weights_refs"] = results
    state["_pending_configs"] = {c["label"]: c for c in configs}
    return state
