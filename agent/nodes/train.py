# agent/nodes/train.py
import os
from agent.state import AgentState
from training.slm_helpers import train as slm_train

ARTIFACTS_DIR = "artifacts"

# Default LoRA config. Rank 16 (α=2r=32) is the research-backed neutral default for
# reasoning/instruction tasks (r=8 tends to underfit multi-step reasoning; r=32-64 is the
# capacity "sweet spot" but overfits small datasets). The iterate LLM can raise/lower it
# via a hyperparameter intervention. See docs (LoRA rank research, Q10/B146).
_DEFAULT_CONFIG = {"nr_epochs": 3, "learning_rate": 2e-4, "batch_size": 8, "lora_rank": 16, "label": "LoRA r=16"}

_VALID_LORA_RANKS = {4, 8, 16, 32, 64}


def _log(model_id: str, msg: str):
    print(f"[train][{model_id}] {msg}")


def _build_config(state: AgentState) -> tuple[dict, str]:
    """
    Build the training config for this iteration.
    Returns (config_dict, reasoning_string).
    """
    decision = state.get("llm_iterate_decision") or {}
    intervention = state.get("last_intervention", "")
    hyperparams = decision.get("hyperparams") if intervention == "hyperparameter" else None

    if not hyperparams:
        return dict(_DEFAULT_CONFIG), "default config (no LLM hyperparameter decision)"

    lora_rank_raw = hyperparams.get("lora_rank")
    if lora_rank_raw is None:
        lora_rank = 8
        rank_note = "LLM requested FFT → overridden to LoRA r=8 (on-device adapter deployment)"
    else:
        try:
            lora_rank = int(lora_rank_raw)
        except (TypeError, ValueError):
            lora_rank = 8
            rank_note = f"LLM returned non-numeric lora_rank={lora_rank_raw!r} → defaulting to r=8"
        else:
            if lora_rank not in _VALID_LORA_RANKS:
                original = lora_rank
                lora_rank = min(_VALID_LORA_RANKS, key=lambda r: abs(r - lora_rank))
                rank_note = f"LLM requested r={original} → clamped to valid r={lora_rank}"
            else:
                rank_note = f"LLM chose r={lora_rank}"

    lr_raw = hyperparams.get("learning_rate")
    lr = min(float(lr_raw) if lr_raw is not None else 2e-4, 0.9999)

    nr_epochs_raw = hyperparams.get("nr_epochs")
    epochs = max(1, int(nr_epochs_raw) if nr_epochs_raw is not None else 3)

    batch_size_raw = hyperparams.get("batch_size")
    batch = max(1, int(batch_size_raw) if batch_size_raw is not None else 8)

    config = {
        "nr_epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch,
        "lora_rank": lora_rank,
        "label": f"LoRA r={lora_rank} lr={lr:.0e} ep={epochs}",
    }

    hypothesis = (decision.get("hypothesis") or "no hypothesis provided")
    reasoning = (
        f"LLM hyperparameter intervention: {rank_note}, lr={lr:.0e}, "
        f"epochs={epochs}, batch={batch}. Hypothesis: {hypothesis}"
    )
    return config, reasoning


def train_node(state: AgentState) -> AgentState:
    """
    Node 4: train a single LoRA configuration.
    Always trains from the base model — never from a prior checkpoint.
    Always produces a LoRA adapter for on-device adapter-manager deployment.
    """
    selected = state["selected_model"]
    if selected is None:
        raise RuntimeError("train_node called before task_analysis selected a model")
    model_id = selected.model_id
    mlabel = selected.label  # log prefix includes quant
    dataset_path = state["current_dataset_path"]
    if dataset_path is None:
        raise RuntimeError("train_node called before curate_node produced a dataset")

    cfg, reasoning = _build_config(state)

    state["iteration"] += 1
    task_type = state["task_type"]

    _log(mlabel, f"Iteration {state['iteration']}")
    _log(mlabel, f"  Config: {cfg['label']}  (batch={cfg['batch_size']})")
    _log(mlabel, f"  Reasoning: {reasoning}")
    _log(mlabel, f"  Dataset: {os.path.basename(dataset_path)}")
    _log(mlabel, f"  Training from base model (no prior adapter)")

    training_output = slm_train(
        dataset_path=dataset_path,
        base_model=model_id,
        nr_epochs=cfg["nr_epochs"],
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        lora_rank=cfg["lora_rank"],
        output_dir=os.path.join(ARTIFACTS_DIR, f"iter{state['iteration']}"),
        task_type=task_type,
    )

    _log(mlabel, f"  Checkpoint: {training_output.weights_ref}")

    state["_pending_weights_refs"] = {cfg["label"]: training_output.weights_ref}
    state["_pending_training_outputs"] = {cfg["label"]: training_output}
    state["_pending_configs"] = {cfg["label"]: cfg}
    return state
