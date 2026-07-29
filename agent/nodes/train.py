# agent/nodes/train.py
import os
from pathlib import Path

from agent.checkpoint import run_training_atomically
from agent.state import AgentState
from training.hparams import (
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_MICRO_BATCH_SIZE,
    DEFAULT_WEIGHT_DECAY,
    deterministic_neighbor_configs,
    hyperparameter_identity,
    normalize_hyperparams,
)
from training.slm_helpers import train as slm_train

ARTIFACTS_DIR = "artifacts"

# Fully explicit neutral default. Keeping every field in state/DAG/checkpoints prevents
# data-only interventions and rollback from silently picking up changed trainer defaults.
_DEFAULT_CONFIG, _ = normalize_hyperparams({
    "nr_epochs": DEFAULT_EPOCHS,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "lora_rank": DEFAULT_LORA_RANK,
    "lora_alpha": DEFAULT_LORA_RANK * 2,
    "lora_dropout": DEFAULT_LORA_DROPOUT,
    "weight_decay": DEFAULT_WEIGHT_DECAY,
    "micro_batch_size": DEFAULT_MICRO_BATCH_SIZE,
    "gradient_accumulation_steps": DEFAULT_GRADIENT_ACCUMULATION_STEPS,
})


def _log(model_id: str, msg: str):
    print(f"[train][{model_id}] {msg}")


def _dataset_identity(version, path) -> tuple[object, str]:
    return version, str(path or "")


def _current_dataset_identity(state: AgentState) -> tuple[object, str]:
    return _dataset_identity(
        state.get("dataset_version"),
        state.get("current_dataset_path"),
    )


def _node_config(node: dict) -> dict | None:
    hparams = (node.get("pi") or {}).get("H") or {}
    if hparams.get("lora_rank") is None:
        return None
    try:
        config, _ = normalize_hyperparams(hparams)
    except ValueError:
        return None
    return config


def _node_tried_configs(node: dict):
    """Yield each actually trained H identity recorded on a DAG node."""
    seen = set()
    primary = _node_config(node)
    if primary is not None:
        identity = hyperparameter_identity(primary)
        seen.add(identity)
        yield primary
    for record in node.get("trained_configs") or []:
        hparams = record.get("H") if isinstance(record, dict) else None
        if not isinstance(hparams, dict) or hparams.get("lora_rank") is None:
            continue
        try:
            config, _ = normalize_hyperparams(hparams)
        except ValueError:
            continue
        identity = hyperparameter_identity(config)
        if identity not in seen:
            seen.add(identity)
            yield config


def _tried_trial_identities(state: AgentState) -> set[tuple]:
    """Complete (dataset, H) identities, including regressed/pruned trials."""
    identities = set()
    for node in state.get("dag") or []:
        dataset = (node.get("pi") or {}).get("D") or {}
        for config in _node_tried_configs(node):
            identities.add((
                *_dataset_identity(
                    dataset.get("version"),
                    dataset.get("path"),
                ),
                *hyperparameter_identity(config),
            ))
    return identities


def _trial_identity(state: AgentState, config: dict) -> tuple:
    return (
        *_current_dataset_identity(state),
        *hyperparameter_identity(config),
    )


def _matching_trial_node(state: AgentState, config: dict) -> dict | None:
    target = _trial_identity(state, config)
    for node in state.get("dag") or []:
        dataset = (node.get("pi") or {}).get("D") or {}
        for node_config in _node_tried_configs(node):
            identity = (
                *_dataset_identity(
                    dataset.get("version"),
                    dataset.get("path"),
                ),
                *hyperparameter_identity(node_config),
            )
            if identity == target:
                return node
    return None


def _best_prior_config(state: AgentState) -> dict | None:
    """Best actually-trained H from non-pruned DAG history.

    A baseline-winning node has no pi.H, but its losing trained candidate still
    supplies the config that a data-only change must hold fixed.
    """
    candidates: list[tuple[float, int, dict]] = []
    order = 0
    for node in state.get("dag") or []:
        if node.get("pruned", False):
            continue
        if not str(node.get("best_config", "")).startswith("baseline"):
            primary = _node_config(node)
            if primary is not None:
                candidates.append((
                    float(node.get("score", 0.0) or 0.0),
                    order,
                    primary,
                ))
                order += 1
        for record in node.get("trained_configs") or []:
            hparams = record.get("H") if isinstance(record, dict) else None
            if not isinstance(hparams, dict):
                continue
            try:
                config, _ = normalize_hyperparams(hparams)
            except ValueError:
                continue
            candidates.append((
                float(record.get("score", 0.0) or 0.0),
                order,
                config,
            ))
            order += 1
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


_DIFF_FIELDS = (
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "weight_decay",
    "learning_rate",
    "nr_epochs",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
)


def _config_diff(old: dict | None, new: dict) -> str:
    """Human-readable field-by-field diff between the best-known config and a new
    proposal, e.g. "lora_rank 32→64, learning_rate 1e-04→5e-05". Unchanged fields are
    omitted; an empty result means the proposal is hyperparameter-identical to the best
    prior config (a pure data-only or rollback-driven retry)."""
    if old is None:
        return "no prior config (first iteration for this model)"
    changes = [
        f"{field} {old.get(field)}→{new.get(field)}"
        for field in _DIFF_FIELDS
        if old.get(field) != new.get(field)
    ]
    return ", ".join(changes) if changes else "unchanged from best prior config"


def _label_for(cfg: dict, suffix: str = "") -> str:
    return (
        f"LoRA r={cfg['lora_rank']} a={cfg['lora_alpha']} "
        f"drop={cfg['lora_dropout']:g} wd={cfg['weight_decay']:g} "
        f"lr={cfg['learning_rate']:.0e} ep={cfg['nr_epochs']} "
        f"mb={cfg['micro_batch_size']} ga="
        f"{cfg['gradient_accumulation_steps']} eb="
        f"{cfg['effective_batch_size']}{suffix}"
    )


def _finalize_config(config: dict, suffix: str = "") -> dict:
    finalized, _ = normalize_hyperparams(config)
    finalized["label"] = _label_for(finalized, suffix=suffix)
    return finalized


def _next_untried_config(state: AgentState, base: dict) -> dict:
    tried = _tried_trial_identities(state)
    for candidate in deterministic_neighbor_configs(base):
        if _trial_identity(state, candidate) not in tried:
            return _finalize_config(candidate)
    raise RuntimeError(
        "bounded LoRA search space is exhausted for the current dataset; "
        "choose data_rebuild or escalate instead of repeating a trial"
    )


def _build_config(state: AgentState) -> tuple[dict, str]:
    """
    Build the training config for this iteration.
    Returns (config_dict, reasoning_string).
    """
    decision = state.get("llm_iterate_decision") or {}
    intervention = state.get("last_intervention", "")
    hyperparams = decision.get("hyperparams") if intervention == "hyperparameter" else None

    if not hyperparams:
        # No explicit LLM hyperparameters. Previously this ALWAYS collapsed to the hardcoded
        # r=16 default, which caused two real defects (observed in run 37372065):
        #   A) every data_rebuild silently discarded the winning hyperparameters, so a
        #      rebuilt dataset was judged with WEAKER hyperparameters than the current best
        #      → it always regressed vs the best and got rolled back (the data change could
        #      never be evaluated fairly);
        #   B) a FAILED iterate LLM call fell back to intervention="hyperparameter" with no
        #      config → r=16 on the SAME dataset → a bit-identical repeat of the prior
        #      iteration (iter4≡iter5).
        # Fix: carry forward the best-known config, and for a config-less hyperparameter
        # fallback, select an UNTRIED complete identity so no axis is silently repeated.
        best = _best_prior_config(state)
        if best is None:
            if (
                intervention == "hyperparameter"
                and _tried_trial_identities(state)
            ):
                config = _next_untried_config(state, _DEFAULT_CONFIG)
                return config, (
                    "fallback hyperparameter step after the baseline won: "
                    "selected deterministic untried complete identity "
                    f"r={config['lora_rank']} a={config['lora_alpha']} "
                    f"drop={config['lora_dropout']:g} "
                    f"wd={config['weight_decay']:g}, effective batch="
                    f"{config['effective_batch_size']}; losing and pruned "
                    "training candidates both count as tried"
                )
            config = _finalize_config(_DEFAULT_CONFIG)
            return (
                config,
                "default config (first iteration; no prior best yet): "
                f"effective batch={config['effective_batch_size']} from "
                f"micro batch={config['micro_batch_size']} × gradient "
                f"accumulation={config['gradient_accumulation_steps']}",
            )

        if intervention == "hyperparameter":
            config = _next_untried_config(state, best)
            return config, (
                "fallback hyperparameter step (LLM returned no config): "
                "selected deterministic untried complete identity "
                f"r={config['lora_rank']} a={config['lora_alpha']} "
                f"drop={config['lora_dropout']:g} "
                f"wd={config['weight_decay']:g}, effective batch="
                f"{config['effective_batch_size']}; pruned configurations "
                "also count as tried"
            )

        # data_rebuild / anything else: hold hyperparameters at the current best
        # so the data (or other) change is isolated and comparable — not confounded by
        # reverting to r=16.
        config = _finalize_config(best, suffix=" [carry-fwd best]")
        return config, (
            f"carry-forward best config (r={config['lora_rank']}, "
            f"a={config['lora_alpha']}, drop="
            f"{config['lora_dropout']:g}, wd="
            f"{config['weight_decay']:g}, effective batch="
            f"{config['effective_batch_size']}) for "
            f"'{intervention or 'default'}' — every optimizer field held "
            "so the data change/rollback is isolated"
        )

    config, local_rationale = normalize_hyperparams(hyperparams)
    normalization_rationale = (
        decision.get("hyperparam_rationale") or local_rationale
    )
    matching = _matching_trial_node(state, config)
    repeat_note = ""
    if matching is not None:
        requested = config
        config = _next_untried_config(state, requested)
        repeat_note = (
            "rejected exact repeat of the same dataset and complete "
            "hyperparameter identity"
            f"{' (pruned configurations also count as tried)' if matching.get('pruned') else ''}; "
            "selected deterministic untried replacement. "
        )
    config = _finalize_config(config)

    hypothesis = (decision.get("hypothesis") or "no hypothesis provided")
    reasoning = (
        f"LLM hyperparameter intervention: {repeat_note}"
        f"LLM chose r={config['lora_rank']}, a={config['lora_alpha']}, "
        f"drop={config['lora_dropout']:g}, wd="
        f"{config['weight_decay']:g}, lr="
        f"{config['learning_rate']:.0e}, epochs="
        f"{config['nr_epochs']}, micro batch="
        f"{config['micro_batch_size']}, gradient accumulation="
        f"{config['gradient_accumulation_steps']}, effective batch="
        f"{config['effective_batch_size']}. "
        f"Validation: {normalization_rationale}. Hypothesis: {hypothesis}"
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
    config_diff = _config_diff(_best_prior_config(state), cfg)

    state["iteration"] += 1
    task_type = state["task_type"]

    _log(mlabel, f"Iteration {state['iteration']}")
    _log(mlabel, f"  Config: {cfg['label']}")
    _log(mlabel, f"  Diff vs best prior config: {config_diff}")
    _log(
        mlabel,
        f"  Batch: micro={cfg['micro_batch_size']} × "
        f"grad_accum={cfg['gradient_accumulation_steps']} → "
        f"effective={cfg['effective_batch_size']}; micro batch controls "
        "per-step peak activation memory",
    )
    _log(mlabel, f"  Reasoning: {reasoning}")
    _log(mlabel, f"  Dataset: {os.path.basename(dataset_path)}")
    _log(mlabel, f"  Training from base model (no prior adapter)")

    artifacts = Path(ARTIFACTS_DIR)
    artifacts.mkdir(parents=True, exist_ok=True)
    selector_safe = selected.selector.replace("/", "_").replace("@", "__")
    final_dir = (
        artifacts
        / "training"
        / selector_safe
        / f"iter{state['iteration']}-d{state.get('dataset_version', 0)}"
    )

    def produce(output_dir):
        return slm_train(
            dataset_path=dataset_path,
            base_model=model_id,
            nr_epochs=cfg["nr_epochs"],
            learning_rate=cfg["learning_rate"],
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            weight_decay=cfg["weight_decay"],
            micro_batch_size=cfg["micro_batch_size"],
            gradient_accumulation_steps=cfg[
                "gradient_accumulation_steps"
            ],
            effective_batch_size=cfg["effective_batch_size"],
            output_dir=output_dir,
            task_type=task_type,
        )

    training_output = run_training_atomically(final_dir, produce)

    _log(mlabel, f"  Checkpoint: {training_output.weights_ref}")

    state["_pending_weights_refs"] = {cfg["label"]: training_output.weights_ref}
    state["_pending_training_outputs"] = {cfg["label"]: training_output}
    state["_pending_configs"] = {cfg["label"]: cfg}
    return state
