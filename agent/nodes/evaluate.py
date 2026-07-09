# agent/nodes/evaluate.py
import os
from agent.state import AgentState
from eval.harness import run_eval
from data.curation_log import CurationLog
from training.quantize import theoretical_hardware_profile
from training.lora_trainer import merge_for_quantization
from training.quantize import quantize_from_model_spec
from agent.nodes.iterate import apply_iteration_policy
from config.android_pool import filter_pool


def _log(model_id: str, msg: str):
    print(f"[evaluate][{model_id}] {msg}")


def evaluate_node(state: AgentState) -> AgentState:
    """
    Node 5: score trained config against E, log to DAG and data-curation.md.

    On the first evaluation for a new model (iteration == 1), also runs the
    base model without any adapter to record the zero-shot baseline. This
    baseline is stored in state["model_baselines"] and printed at run end.
    """
    task_type = state["task_type"]
    model_id = state["selected_model"].model_id
    eval_set = state["eval_set"]
    if eval_set is None:
        raise RuntimeError("evaluate_node called before eval_setup_node built the eval set")
    pending = state.get("_pending_weights_refs", {})

    # --- Baseline measurement (first eval for this model) ---
    if state["iteration"] == 1:
        _log(model_id, "Measuring zero-shot baseline (base model, no adapter)...")
        try:
            baseline_result = run_eval(eval_set, model_id, model_id, task_type=task_type)
            baseline_f1 = baseline_result.f1
        except Exception as e:
            _log(model_id, f"Baseline measurement failed ({e}); recording 0.0")
            baseline_f1 = 0.0

        _log(model_id, f"Baseline F1 = {baseline_f1:.4f}")
        baselines = state.get("model_baselines") or []
        baselines.append({
            "model_id": model_id,
            "baseline_f1": baseline_f1,
            "best_finetuned_f1": 0.0,
        })
        state["model_baselines"] = baselines

    # --- Score all trained configs ---
    scored = {}
    for label, weights_ref in pending.items():
        _log(model_id, f"Evaluating config '{label}' (weights: {weights_ref})")
        quant = state["selected_model"].quant
        iteration = state["iteration"]
        gguf_path = None
        if quant is not None:
            model_id_safe = model_id.replace("/", "_")
            merged_path = merge_for_quantization(
                weights_ref,
                os.path.join("artifacts", "merged", model_id_safe, label, f"iter{iteration}"),
            )
            gguf_path = quantize_from_model_spec(
                merged_path,
                os.path.join("artifacts", "gguf", model_id_safe, label, f"iter{iteration}"),
                quant,
            )
        result = run_eval(eval_set, weights_ref, model_id, task_type=task_type, quant=quant, gguf_path=gguf_path)
        scored[label] = (weights_ref, result)
        _log(model_id, f"  → F1={result.f1:.4f}  failures={len(result.failures)}")

    best_label = max(scored, key=lambda k: scored[k][1].f1)
    best_weights_ref, best_result = scored[best_label]
    current_score = best_result.f1

    prev_best = state["best_score"]
    delta = current_score - prev_best

    # Update state
    if current_score > state["best_score"]:
        state["best_score"] = current_score
        state["best_weights_ref"] = best_weights_ref
        state["consecutive_no_improvement"] = 0
    else:
        state["consecutive_no_improvement"] += 1

    state["scores"].append(current_score)
    state["last_eval"] = best_result

    _log(model_id,
         f"Score: {current_score:.4f}  (Δ={delta:+.4f} from best {prev_best:.4f})  "
         f"failures={len(best_result.failures)}  "
         f"trajectory={[f'{s:.3f}' for s in state['scores']]}")

    # Update the model_baselines entry with the best fine-tuned score so far
    baselines = state.get("model_baselines") or []
    for entry in baselines:
        if entry["model_id"] == model_id:
            entry["best_finetuned_f1"] = max(entry.get("best_finetuned_f1", 0.0), state["best_score"])

    # Log to DAG with full π=(D,H,S) triple and parent edge
    policy = apply_iteration_policy(current_score)
    best_cfg = (state.get("_pending_configs") or {}).get(best_label, {})
    parent_iteration = state["dag"][-1]["iteration"] if state["dag"] else None
    dag_node = {
        "iteration": state["iteration"],
        "parent_iteration": parent_iteration,
        "model_id": model_id,
        "weights_ref": best_weights_ref,
        "score": current_score,
        "best_config": best_label,
        "intervention": policy["intervention"],
        "failures": len(best_result.failures),
        "pruned": False,
        "pi": {
            "D": {"version": state["dataset_version"], "path": state.get("current_dataset_path")},
            "H": {
                "lora_rank": best_cfg.get("lora_rank"),
                "learning_rate": best_cfg.get("learning_rate"),
                "nr_epochs": best_cfg.get("nr_epochs"),
                "batch_size": best_cfg.get("batch_size"),
            },
            "S": {"task_type": task_type, "supervision": "direct"},
        },
    }
    state["dag"].append(dag_node)

    # Write data-curation.md entry with hardware PASS/FAIL
    hw_profile = theoretical_hardware_profile(model_id)
    from config.android_pool import check_hardware_constraints
    hw_constraints = check_hardware_constraints(state["selected_model"], state["hardware_constraints"])
    config_descriptions = state.get("_pending_configs", {})
    config_labels = list(config_descriptions.values())
    config_a = config_labels[0]["label"] if config_labels else "N/A"
    config_b = config_labels[1]["label"] if len(config_labels) > 1 else "N/A"

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
        hw_constraints=hw_constraints,
    )

    # Downward probe: if the loop is about to terminate with success, check if a
    # lower-tier model already met stop_threshold in an earlier iteration.
    # If so, switch to the smallest model that cleared the bar (use minimum resources).
    current_selected = state.get("selected_model")
    if current_score >= state.get("stop_threshold", 0.96) and current_selected is not None:
        threshold = state.get("stop_threshold", 0.96)
        feasible = filter_pool(state["hardware_constraints"])
        current_tier = current_selected.tier
        for dag_node in state.get("dag", []):
            if dag_node.get("pruned"):
                continue
            if dag_node.get("score", 0.0) >= threshold:
                node_model_id = dag_node.get("model_id")
                # Find a lower-tier feasible model with this model_id
                smaller = next(
                    (m for m in feasible
                     if m.model_id == node_model_id and m.tier < current_tier),
                    None,
                )
                if smaller is not None:
                    _log(current_selected.model_id,
                         f"  Downward probe: {smaller.model_id} (tier {smaller.tier}) "
                         f"already cleared threshold {threshold:.3f} in iteration "
                         f"{dag_node['iteration']} — switching to smaller model")
                    state["selected_model"] = smaller
                    break

    return state
