# agent/nodes/evaluate.py
from agent.state import AgentState
from eval.harness import run_eval
from data.curation_log import CurationLog
from training.quantize import theoretical_hardware_profile
from agent.nodes.iterate import apply_iteration_policy


def evaluate_node(state: AgentState) -> AgentState:
    """
    Node 4: score each trained config against E, select best, log to DAG and data-curation.md.
    task_type flows from state throughout — no hardcoding.
    """
    task_type = state["task_type"]
    model_id = state["selected_model"].model_id
    eval_set = state["eval_set"]
    pending = state.get("_pending_weights_refs", {})

    # Score all configs; task_type required by run_eval dispatcher
    scored = {}
    for label, weights_ref in pending.items():
        result = run_eval(eval_set, weights_ref, model_id, task_type=task_type)
        scored[label] = (weights_ref, result)

    # Select best by F1
    best_label = max(scored, key=lambda k: scored[k][1].f1)
    best_weights_ref, best_result = scored[best_label]

    current_score = best_result.f1

    # Update state
    if current_score > state["best_score"]:
        state["best_score"] = current_score
        state["best_weights_ref"] = best_weights_ref
        state["consecutive_no_improvement"] = 0
    else:
        state["consecutive_no_improvement"] += 1

    state["scores"].append(current_score)
    state["last_eval"] = best_result

    # Log to DAG
    policy = apply_iteration_policy(current_score)
    dag_node = {
        "iteration": state["iteration"],
        "model_id": model_id,
        "weights_ref": best_weights_ref,
        "score": current_score,
        "best_config": best_label,
        "intervention": policy["intervention"],
        "failures": len(best_result.failures),
        "pruned": False,
    }
    state["dag"].append(dag_node)

    # Write data-curation.md entry
    hw_profile = theoretical_hardware_profile(model_id)
    config_descriptions = state.get("_pending_configs", {})
    config_labels = list(config_descriptions.values())
    config_a = config_labels[0]["label"] if config_labels else "Config A"
    config_b = config_labels[1]["label"] if len(config_labels) > 1 else "Config B"

    curation = state.get("last_curation") or {}
    log = CurationLog()
    log.write_iteration(
        iteration=state["iteration"],
        task_type=task_type,
        dataset_version=f"v{state['dataset_version']}",
        n_gold=curation.get("n_gold", 0),
        n_hard=curation.get("n_hard", 0),
        label_dist=curation.get("label_dist", {}),
        config_a=config_a,
        config_b=config_b,
        best_config=best_label,
        eval_result=best_result,
        score_band=policy["band"],
        next_intervention=policy["intervention"],
        hypothesis=state.get("last_hypothesis", ""),
        model_id=model_id,
        int4_size_mb=hw_profile.get("int4_size_mb") or 0,
        tier=hw_profile.get("tier") or 0,
    )

    return state
