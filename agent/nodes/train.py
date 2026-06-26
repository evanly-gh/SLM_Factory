# agent/nodes/train.py
import os
from agent.state import AgentState
from training.slm_helpers import train as slm_train

ARTIFACTS_DIR = "artifacts"

_DEFAULT_CONFIGS = [
    {"nr_epochs": 3, "learning_rate": 2e-4, "batch_size": 8,  "lora_rank": 8,    "label": "A: LoRA r=8"},
    {"nr_epochs": 5, "learning_rate": 1e-4, "batch_size": 4,  "lora_rank": None, "label": "B: full_ft"},
]

_VALID_LORA_RANKS = {4, 8, 16, 32, 64}


def _build_configs(state: AgentState) -> list[dict]:
    """
    Build the two training configs for this iteration.

    If iterate_node produced a structured hyperparameter decision (intervention ==
    "hyperparameter"), use its "hyperparams" dict as Config A and derive a contrasting
    Config B by flipping one axis. Otherwise use the default configs.
    """
    decision = state.get("llm_iterate_decision") or {}
    intervention = state.get("last_intervention", "")
    hyperparams = decision.get("hyperparams") if intervention == "hyperparameter" else None

    if not hyperparams:
        return list(_DEFAULT_CONFIGS)

    # Validate and clamp LLM-supplied hyperparams
    lora_rank = hyperparams.get("lora_rank")
    if lora_rank is not None and lora_rank not in _VALID_LORA_RANKS:
        lora_rank = min(_VALID_LORA_RANKS, key=lambda r: abs(r - lora_rank))
    lr = float(hyperparams.get("learning_rate", 2e-4))
    epochs = max(1, int(hyperparams.get("nr_epochs", 3)))
    batch = int(hyperparams.get("batch_size", 8))

    config_a = {
        "nr_epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch,
        "lora_rank": lora_rank,
        "label": f"A: LLM-hp r={lora_rank} lr={lr:.0e} ep={epochs}",
    }

    # Config B: explore the complementary axis.
    # If A is LoRA, B tries full fine-tune; if A is full_ft, B tries LoRA r=16.
    if lora_rank is None:
        config_b = {
            "nr_epochs": epochs,
            "learning_rate": lr * 2,
            "batch_size": batch,
            "lora_rank": 16,
            "label": f"B: LoRA r=16 lr={lr*2:.0e} ep={epochs}",
        }
    else:
        config_b = {
            "nr_epochs": max(epochs - 1, 1),
            "learning_rate": lr / 2,
            "batch_size": batch,
            "lora_rank": lora_rank * 2 if lora_rank * 2 in _VALID_LORA_RANKS else lora_rank,
            "label": f"B: LoRA r={lora_rank * 2 if lora_rank * 2 in _VALID_LORA_RANKS else lora_rank} lr={lr/2:.0e} ep={max(epochs-1,1)}",
        }

    return [config_a, config_b]


def train_node(state: AgentState) -> AgentState:
    """
    Node 3: train >=2 configurations and select the best by f(pi) on E.
    Always trains from the base model — never from a prior checkpoint.

    When iterate_node set intervention="hyperparameter" and supplied structured
    "hyperparams", Config A uses the LLM-chosen values and Config B explores a
    complementary axis. Otherwise both configs are the default LoRA/full_ft pair.

    The configs are trained SEQUENTIALLY: Unsloth/trl share process-global state
    (Accelerate PartialState singleton, runtime class patching) so two trainers in
    one process race. On a single GPU sequential gives the same wall-clock anyway.
    """
    model_id = state["selected_model"].model_id
    dataset_path = state["current_dataset_path"]

    configs = _build_configs(state)

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
