# agent/nodes/evaluate.py
import os
import config.config as config
from agent.state import AgentState
from eval.harness import run_eval
from data.curation_log import CurationLog
from training.quantize import theoretical_hardware_profile
from training.lora_trainer import merge_for_quantization
from training.quantize import quantize_from_model_spec
from agent.nodes.iterate import apply_iteration_policy

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
    mlabel = state["selected_model"].label  # log prefix includes quant
    eval_set = state["eval_set"]
    if eval_set is None:
        raise RuntimeError("evaluate_node called before eval_setup_node built the eval set")
    pending = state.get("_pending_weights_refs") or {}

    # --- Baseline measurement (first eval for this model) ---
    if state["iteration"] == 1:
        _log(mlabel, "Measuring zero-shot baseline (base model, no adapter)...")
        try:
            baseline_result = run_eval(eval_set, model_id, model_id, task_type=task_type)
            baseline_f1 = baseline_result.f1
        except Exception as e:
            _log(mlabel, f"Baseline measurement failed ({e}); recording 0.0")
            baseline_f1 = 0.0

        _log(mlabel, f"Baseline F1 = {baseline_f1:.4f}")
        baselines = state.get("model_baselines") or []
        if not any(e["model_id"] == model_id for e in baselines):
            baselines.append({
                "model_id": model_id,
                "baseline_f1": baseline_f1,
                "best_finetuned_f1": 0.0,
            })
        state["model_baselines"] = baselines

    # --- Score all trained configs ---
    scored = {}
    for label, weights_ref in pending.items():
        _log(mlabel, f"Evaluating config '{label}' (weights: {weights_ref})")
        quant = state["selected_model"].quant
        iteration = state["iteration"]
        gguf_path = None
        # Only build a real GGUF (which requires llama.cpp: convert_hf_to_gguf +
        # llama-quantize) when we actually measure the quantized model on-device. In
        # the default "theoretical" backend the run is accuracy-only: score the HF/LoRA
        # weights via Unsloth (gguf_path=None → eval/harness.py uses infer_batch). This
        # keeps accuracy-only runs from crashing on a missing llama.cpp toolchain.
        if quant is not None and config.HW_ONDEVICE_BACKEND != "theoretical":
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
        _log(mlabel, f"  → F1={result.f1:.4f}  failures={len(result.failures)}")

    if not scored:
        raise RuntimeError("evaluate_node: no configs were scored — train_node did not populate _pending_weights_refs")
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

    # Assign a NEW list rather than appending in place: `scores` has no LangGraph
    # reducer, and an in-place mutation keeps the same object identity, so the change is
    # not reliably persisted to the channel — the list froze after ~3 entries and
    # stagnation (which reads this window) never fired, looping the run forever (BUGS B122).
    state["scores"] = list(state.get("scores") or []) + [current_score]
    state["last_eval"] = best_result

    _log(mlabel,
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
    # Prefer the LLM-driven intervention stored by iterate_node over the fallback score-band rule.
    dag_intervention = state.get("last_intervention") or policy["intervention"]
    best_cfg = (state.get("_pending_configs") or {}).get(best_label, {})
    parent_iteration = state["dag"][-1]["iteration"] if state["dag"] else None
    dag_node = {
        "iteration": state["iteration"],
        "parent_iteration": parent_iteration,
        "model_id": model_id,
        "weights_ref": best_weights_ref,
        "score": current_score,
        "best_config": best_label,
        "intervention": dag_intervention,
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
    # New list (not in-place) so the channel change persists — see B122 note above.
    state["dag"] = list(state.get("dag") or []) + [dag_node]

    # Write data-curation.md entry with hardware PASS/FAIL
    hw_profile = theoretical_hardware_profile(model_id)
    from config.android_pool import check_hardware_constraints
    hw_constraints = check_hardware_constraints(state["selected_model"], state["hardware_constraints"])
    config_descriptions = state.get("_pending_configs", {})
    config_labels = list(config_descriptions.keys())
    config_a = config_labels[0] if config_labels else "N/A"
    config_b = config_labels[1] if len(config_labels) > 1 else "N/A"

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
        size_mb=hw_profile.get("size_mb") or 0,
        tier=hw_profile.get("tier") or 0,
        hw_constraints=hw_constraints,
    )

    return state
