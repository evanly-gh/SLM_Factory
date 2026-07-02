# agent/nodes/train.py
import os
from agent.state import AgentState
from training.slm_helpers import train as slm_train

ARTIFACTS_DIR = "artifacts"

_DEFAULT_CONFIG = {"nr_epochs": 3, "learning_rate": 2e-4, "batch_size": 8, "lora_rank": 8, "label": "LoRA r=8"}

_VALID_LORA_RANKS = {4, 8, 16, 32, 64}


def _build_config(state: AgentState) -> dict:
    """
    Build the training config for this iteration.

    If iterate_node produced a hyperparameter decision, use those values
    (clamped to valid LoRA ranks, overridden to LoRA if FFT was requested).
    Otherwise use the default config.
    """
    decision = state.get("llm_iterate_decision") or {}
    intervention = state.get("last_intervention", "")
    hyperparams = decision.get("hyperparams") if intervention == "hyperparameter" else None

    if not hyperparams:
        return dict(_DEFAULT_CONFIG)

    lora_rank = hyperparams.get("lora_rank")
    if lora_rank is None:
        lora_rank = 8
    elif lora_rank not in _VALID_LORA_RANKS:
        lora_rank = min(_VALID_LORA_RANKS, key=lambda r: abs(r - lora_rank))
    lr = float(hyperparams.get("learning_rate", 2e-4))
    epochs = max(1, int(hyperparams.get("nr_epochs", 3)))
    batch = int(hyperparams.get("batch_size", 8))

    return {
        "nr_epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch,
        "lora_rank": lora_rank,
        "label": f"LoRA r={lora_rank} lr={lr:.0e} ep={epochs}",
    }


def train_node(state: AgentState) -> AgentState:
    """
    Node 3: train a single LoRA configuration.
    Always trains from the base model — never from a prior checkpoint.
    Always produces a LoRA adapter for on-device adapter-manager deployment.
    """
    model_id = state["selected_model"].model_id
    dataset_path = state["current_dataset_path"]
    if dataset_path is None:
        raise RuntimeError("train_node called before curate_node produced a dataset")

    cfg = _build_config(state)

    state["iteration"] += 1

    task_type = state["task_type"]

    weights_ref = slm_train(
        dataset_path=dataset_path,
        base_model=model_id,
        nr_epochs=cfg["nr_epochs"],
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        lora_rank=cfg["lora_rank"],
        output_dir=os.path.join(
            ARTIFACTS_DIR,
            f"iter{state['iteration']}",
        ),
        task_type=task_type,
    )

    state["_pending_weights_refs"] = {cfg["label"]: weights_ref}
    state["_pending_configs"] = {cfg["label"]: cfg}
    return state
