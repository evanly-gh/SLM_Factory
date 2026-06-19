# agent/nodes/train.py
import os
import json
from concurrent.futures import ThreadPoolExecutor
from agent.state import AgentState
from training.slm_helpers import train as slm_train

ARTIFACTS_DIR = "artifacts"


def _save_dataset(examples: list[dict], version: int) -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, f"dataset_v{version}.jsonl")
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return path


def train_node(state: AgentState) -> AgentState:
    """
    Node 3: fire ≥2 parallel training configurations.
    Always trains from base model — never from a prior checkpoint.
    Selects best config by f(π) on E.
    """
    model_id = state["selected_model"].model_id
    dataset_path = state["current_dataset_path"]

    # Two parallel configs per the paper's requirement
    configs = [
        {"nr_epochs": 3, "learning_rate": 2e-4, "batch_size": 8,  "lora_rank": 8,    "label": "A: LoRA r=8"},
        {"nr_epochs": 5, "learning_rate": 1e-4, "batch_size": 4,  "lora_rank": None, "label": "B: full_ft"},
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            cfg["label"]: executor.submit(
                slm_train,
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
            for cfg in configs
        }
        for label, future in futures.items():
            results[label] = future.result()

    # Store both weights_refs; evaluate node will select best
    state["_pending_weights_refs"] = results
    state["_pending_configs"] = {c["label"]: c for c in configs}
    state["iteration"] += 1
    return state
