# agent/nodes/evaluate.py
import os
import shutil
from copy import deepcopy
from dataclasses import asdict
import config.config as config
from agent.state import AgentState
from eval.harness import run_eval
from data.curation_log import CurationLog
from training.quantize import theoretical_hardware_profile
from training.lora_trainer import (
    SFT_LOSS_CONTRACT_VERSION,
    merge_for_quantization,
)
from training.quantize import (
    QuantizationInfrastructureError,
    invalidate_gguf_cache,
    quantize_from_model_spec,
    resolve_hf_snapshot,
    validate_and_record_gguf,
    validated_gguf_cache_hit,
)
from agent.nodes.iterate import apply_iteration_policy
from training.hparams import normalize_hyperparams

def _log(model_id: str, msg: str):
    print(f"[evaluate][{model_id}] {msg}")


def _build_or_reuse_gguf(weights_ref: str, model_id: str, quant: str, mlabel: str):
    """Resolve base/adapter weights, quantize, strongly validate, and cache a GGUF.

    CACHING (B160): keyed by the exact `weights_ref` AND the quant method, so a given
    trained checkpoint is quantized to a given format at most ONCE. Both parts of the key
    are essential (B161): the quant must be in the key because the SAME base model can be
    selected at two tiers as different quant variants (e.g. 1.7B Q4_K_M then 1.7B Q8_0) and
    the checkpoint path collides across tiers (the iteration counter resets to 1 on
    escalation). Keying on weights_ref alone — and reusing ANY *.gguf in the dir — made the
    Q8_0 tier silently reuse the earlier Q4_K_M file, so Q8_0 was never actually produced
    and the two tiers scored identically. We now key on (weights_ref, quant) and check for
    the SPECIFIC `model-<method>.gguf`.

    A cached file is reusable only when its size and SHA-256 match an atomic sidecar that
    was written after a real llama.cpp model load. Raw base-model IDs bypass Unsloth's
    adapter merge and convert the immutable local Hugging Face snapshot directly.
    """
    import hashlib

    method = {"Q4_K_M": "q4_k_m", "Q8_0": "q8_0"}.get(quant, str(quant).lower())
    model_id_safe = model_id.replace("/", "_")
    wkey = hashlib.sha1(f"{weights_ref}|{quant}".encode()).hexdigest()[:12]
    gguf_dir = os.path.join("artifacts", "gguf", model_id_safe, wkey)
    expected = os.path.join(gguf_dir, f"model-{method}.gguf")
    if validated_gguf_cache_hit(expected):
        _log(mlabel, f"  Reusing cached {quant} GGUF for these weights: {expected}")
        return expected
    if os.path.exists(expected):
        _log(mlabel, f"  Invalidating unvalidated or changed GGUF cache: {expected}")
    invalidate_gguf_cache(expected)

    cleanup_path = None
    try:
        _log(mlabel, f"  Quantizing to {quant} GGUF for honest accuracy eval (llama.cpp)...")
        is_remote_base = (
            weights_ref == model_id and not os.path.exists(weights_ref)
        )
        if is_remote_base:
            source_path = resolve_hf_snapshot(model_id)
            _log(mlabel, f"  Converting immutable HF snapshot directly: {source_path}")
        else:
            source_path = merge_for_quantization(
                weights_ref,
                os.path.join("artifacts", "merged", model_id_safe, wkey),
            )
            cleanup_path = source_path
        gguf_path = quantize_from_model_spec(source_path, gguf_dir, quant)
        try:
            validate_and_record_gguf(gguf_path)
        except Exception:
            invalidate_gguf_cache(gguf_path)
            raise
        _log(
            mlabel,
            f"  GGUF built and load-validated: {gguf_path} — "
            "scoring via llama-cpp-python",
        )
        return gguf_path
    except QuantizationInfrastructureError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize toolchain/backend failures
        raise QuantizationInfrastructureError(
            f"Failed to build required {quant} GGUF for {model_id}: {exc}"
        ) from exc
    finally:
        if cleanup_path is not None:
            # Only adapter-merge output is disposable. HF snapshots are shared cache.
            shutil.rmtree(cleanup_path, ignore_errors=True)


def _reap_gguf(state, gguf_paths, keep_path, mlabel: str = "") -> None:
    """Delete the GGUFs built this iteration unless they are a retained new-best.

    The `(weights_ref, quant)` cache key is unique per iteration, so a GGUF is written,
    read once by run_eval, and then never hit again — measured 0/138 (NER) and 2/66
    (math) reuses, at 2.6 GB apiece. Retention policy: keep a GGUF only when its
    iteration set a new best for the tier, which preserves every score-improving model
    (and the run's final winner, which is a new-best by construction) while bounding
    steady-state disk.

    `keep_path` is the GGUF of this iteration's best config when it improved on the
    prior best, else None. Paths already in `state["retained_gguf_paths"]` are earlier
    new-bests and are never reaped. A rollback or downward probe that needs a reaped
    GGUF simply rebuilds it — correctness is unaffected, only a re-quantization.
    """
    retained = list(state.get("retained_gguf_paths") or [])
    retained_abs = {os.path.abspath(p) for p in retained}

    if keep_path:
        keep_abs = os.path.abspath(keep_path)
        if keep_abs not in retained_abs:
            retained.append(keep_path)
            retained_abs.add(keep_abs)

    for path in gguf_paths:
        if not path or os.path.abspath(path) in retained_abs:
            continue
        # Drop the sidecar with the file: a surviving validation record would let a
        # later build mistake a partially-written file for a warm cache hit.
        invalidate_gguf_cache(path)
        parent = os.path.dirname(path)
        try:
            if os.path.isdir(parent) and not os.listdir(parent):
                os.rmdir(parent)
        except OSError:
            pass
        _log(mlabel, f"  Reaped non-best GGUF: {path}")

    state["retained_gguf_paths"] = retained


def _build_gguf_for_eval(weights_ref: str, model_id: str, quant: str, mlabel: str):
    """Build/reuse GGUF outside the parent CUDA context when isolation is enabled."""
    from training.cuda_isolation import (
        CudaWorkerError,
        isolation_enabled,
        run_isolated,
    )

    try:
        if isolation_enabled():
            gguf_path = run_isolated(
                "build_gguf",
                {
                    "weights_ref": weights_ref,
                    "model_id": model_id,
                    "quant": quant,
                    "mlabel": mlabel,
                },
            )
        else:
            gguf_path = _build_or_reuse_gguf(weights_ref, model_id, quant, mlabel)
    except CudaWorkerError as exc:
        if exc.remote_error_type == "QuantizationInfrastructureError":
            raise QuantizationInfrastructureError(str(exc)) from exc
        raise
    if not gguf_path:
        raise QuantizationInfrastructureError(
            f"Failed to build required {quant} quantized GGUF for {model_id}; "
            "refusing to silently score BF16 under a quantized variant label"
        )
    return gguf_path


def evaluate_node(state: AgentState) -> AgentState:
    """
    Node 5: score trained config against E, log to DAG and data-curation.md.

    On the first evaluation for a new model (iteration == 1), also runs the
    base model without any adapter to record the zero-shot baseline. This
    baseline is stored in state["model_baselines"] and printed at run end.
    """
    task_type = state["task_type"]
    model_id = state["selected_model"].model_id
    selector = state["selected_model"].selector
    mlabel = state["selected_model"].label  # log prefix includes quant
    eval_set = state["eval_set"]
    if eval_set is None:
        raise RuntimeError("evaluate_node called before eval_setup_node built the eval set")
    pending = state.get("_pending_weights_refs") or {}

    # --- Baseline measurement (first eval for this model) ---
    baseline_result = None
    baseline_gguf_path = None
    if state["iteration"] == 1:
        _log(mlabel, "Measuring zero-shot baseline (base model, no adapter)...")
        try:
            baseline_quant = state["selected_model"].quant
            if baseline_quant is not None:
                baseline_gguf_path = _build_gguf_for_eval(
                    model_id,
                    model_id,
                    baseline_quant,
                    mlabel,
                )
            baseline_result = run_eval(
                eval_set,
                model_id,
                model_id,
                task_type=task_type,
                quant=baseline_quant,
                gguf_path=baseline_gguf_path,
            )
            baseline_f1 = baseline_result.f1
        except Exception as e:
            if (
                isinstance(e, QuantizationInfrastructureError)
                or getattr(e, "remote_error_type", "")
                == "QuantizationInfrastructureError"
            ):
                raise
            if task_type == "generation":
                from eval.judge_client import JudgeInfrastructureError

                remote_error_type = getattr(e, "remote_error_type", "")
                if (
                    isinstance(e, JudgeInfrastructureError)
                    or remote_error_type == "JudgeInfrastructureError"
                ):
                    raise
            # Record UNMEASURED, not 0.0. A caught exception and a genuine zero-shot
            # score of zero used to be indistinguishable downstream, so a failed
            # measurement was reported as a real baseline — the NER run's tier-2 GGUF
            # failed to load once and the final report credited fine-tuning with a
            # +0.8476 improvement over a measurement that never happened.
            _log(mlabel, f"Baseline measurement FAILED ({e}); recording n/a (not 0.0)")
            baseline_result = None
            baseline_f1 = None

        _log(
            mlabel,
            f"Baseline F1 = {baseline_f1:.4f}" if baseline_f1 is not None
            else "Baseline F1 = n/a (measurement failed)",
        )
        baselines = state.get("model_baselines") or []
        if not any(e.get("selector", e.get("model_id")) == selector for e in baselines):
            baselines.append({
                "selector": selector,
                "model_id": model_id,
                "quant": state["selected_model"].quant,
                "baseline_f1": baseline_f1,
                "best_finetuned_f1": 0.0,
            })
        state["model_baselines"] = baselines

    # --- Score all trained configs ---
    scored = {}
    gguf_by_label = {}
    for label, weights_ref in pending.items():
        _log(mlabel, f"Evaluating config '{label}' (weights: {weights_ref})")
        quant = state["selected_model"].quant
        gguf_path = None
        # Build + score the ACTUAL quantized GGUF (honest per-quant accuracy) when EITHER
        # (a) QUANT_ACCURACY_EVAL is on (default; accuracy-only, no phone needed), OR (b) a
        # real on-device backend is selected (latency/power measurement). Both require
        # llama.cpp (convert_hf_to_gguf + llama-quantize) + llama-cpp-python. If the flag is
        # off (SLM_QUANT_EVAL=0), we score the HF/LoRA weights via Unsloth (gguf_path=None).
        want_gguf_eval = config.QUANT_ACCURACY_EVAL or config.HW_ONDEVICE_BACKEND != "theoretical"
        if quant is not None and want_gguf_eval:
            gguf_path = _build_gguf_for_eval(weights_ref, model_id, quant, mlabel)
        result = run_eval(eval_set, weights_ref, model_id, task_type=task_type, quant=quant, gguf_path=gguf_path)
        gguf_by_label[label] = gguf_path
        scored[label] = (weights_ref, result)
        _log(
            mlabel,
            f"  → F1={result.f1:.4f}  failures={len(result.failures)}/{len(eval_set.all)}",
        )

    if not scored:
        raise RuntimeError("evaluate_node: no configs were scored — train_node did not populate _pending_weights_refs")

    # Count the ZERO-SHOT baseline as a candidate (B161): if fine-tuning didn't beat the
    # base model, KEEP the base model. This prevents reporting a fine-tune that is WORSE
    # than zero-shot (e.g. the 4B collapsed 0.64→0.27 and its good baseline was lost), and
    # lets a strong base model "converge" on its own when its zero-shot already meets the
    # goal. weights_ref=model_id resolves to the base model (no adapter) in run_eval.
    if state["iteration"] == 1 and baseline_result is not None:
        scored["baseline (zero-shot, no adapter)"] = (model_id, baseline_result)
        gguf_by_label["baseline (zero-shot, no adapter)"] = baseline_gguf_path
        _log(mlabel, f"  Baseline counted as a candidate (F1={baseline_result.f1:.4f}) — "
                     f"fine-tuning must beat zero-shot to be kept")

    best_label = max(scored, key=lambda k: scored[k][1].f1)
    best_weights_ref, best_result = scored[best_label]
    current_score = best_result.f1
    if best_label.startswith("baseline"):
        _log(mlabel, f"  Best this iteration is the ZERO-SHOT base model (F1={current_score:.4f}) "
                     f"— fine-tuning did not improve on it")

    prev_best = state["best_score"]
    delta = current_score - prev_best

    # Update state
    is_new_best = current_score > state["best_score"]
    if is_new_best:
        state["best_score"] = current_score
        state["best_weights_ref"] = best_weights_ref
        state["consecutive_no_improvement"] = 0
    else:
        state["consecutive_no_improvement"] += 1

    # Keep the GGUF only when this iteration improved on the prior best; every other
    # one is write-once-read-once scratch worth 2.6 GB. Earlier new-bests are protected
    # by state["retained_gguf_paths"].
    _reap_gguf(
        state,
        list(gguf_by_label.values()),
        gguf_by_label.get(best_label) if is_new_best else None,
        mlabel,
    )

    # Assign a NEW list rather than appending in place: `scores` has no LangGraph
    # reducer, and an in-place mutation keeps the same object identity, so the change is
    # not reliably persisted to the channel — the list froze after ~3 entries and
    # stagnation (which reads this window) never fired, looping the run forever (BUGS B122).
    state["scores"] = list(state.get("scores") or []) + [current_score]
    state["last_eval"] = best_result

    _log(mlabel,
         f"Score: {current_score:.4f}  (Δ={delta:+.4f} from best {prev_best:.4f})  "
         f"failures={len(best_result.failures)}/{len(eval_set.all)}  "
         f"trajectory={[f'{s:.3f}' for s in state['scores']]}")

    # Test-data agent (B161): report per-difficulty accuracy + a targeted diagnosis. This is
    # the aggregate signal iterate_node uses instead of raw failure rows.
    try:
        from agent.nodes.test_agent import build_test_report
        report = build_test_report(
            eval_set, best_result, state.get("eval_difficulty"),
            state.get("stop_threshold", 0.9), task_type,
        )
        state["test_report"] = report
        _bd = report["by_difficulty"]
        def _fmt(b):
            v = _bd.get(b, {})
            a = v.get("accuracy")
            return f"{b}={a:.3f}(n={v.get('n',0)})" if a is not None else f"{b}=n/a"
        _log(mlabel, f"  [test_agent] overall={report['overall']:.4f}  "
                     f"{_fmt('easy')}  {_fmt('medium')}  {_fmt('hard')}")
        _log(mlabel, f"  [test_agent] diagnosis: {report['diagnosis']}")
        _log(mlabel, f"  [test_agent] suggested intervention: {report['suggested_intervention']}")
    except Exception as _e:  # noqa: BLE001
        _log(mlabel, f"  [test_agent] report failed ({str(_e)[:100]})")

    # Update the model_baselines entry with the best fine-tuned score so far
    baselines = state.get("model_baselines") or []
    for entry in baselines:
        if entry.get("selector", entry.get("model_id")) == selector:
            entry["best_finetuned_f1"] = max(entry.get("best_finetuned_f1", 0.0), state["best_score"])

    # Log to DAG with full π=(D,H,S) triple and parent edge
    policy = apply_iteration_policy(current_score)
    # Prefer the LLM-driven intervention stored by iterate_node over the fallback score-band rule.
    dag_intervention = state.get("last_intervention") or policy["intervention"]
    if is_new_best:
        _log(
            mlabel,
            f"  IMPROVEMENT — iteration {state['iteration']} — {dag_intervention} — "
            f"{prev_best:.4f} → {current_score:.4f} (Δ={delta:+.4f})",
        )
    best_cfg = (state.get("_pending_configs") or {}).get(best_label, {})
    if best_cfg.get("lora_rank") is not None:
        # Checkpoints created before the expanded search carry only rank/LR/
        # epochs/batch_size. Normalize that legacy alias/default shape before
        # writing the new complete DAG identity.
        best_cfg, _ = normalize_hyperparams(best_cfg)
    parent_iteration = state["dag"][-1]["iteration"] if state["dag"] else None
    trained_configs = []
    for label in pending:
        raw_config = (state.get("_pending_configs") or {}).get(label, {})
        if raw_config.get("lora_rank") is None:
            continue
        normalized_config, _ = normalize_hyperparams(raw_config)
        trained_configs.append({
            "label": label,
            "score": scored[label][1].f1,
            "H": normalized_config,
        })
    dag_node = {
        "iteration": state["iteration"],
        "parent_iteration": parent_iteration,
        "selector": selector,
        "model_id": model_id,
        "quant": state["selected_model"].quant,
        "weights_ref": best_weights_ref,
        "score": current_score,
        "best_config": best_label,
        "intervention": dag_intervention,
        "failures": len(best_result.failures),
        "pruned": False,
        # Keep every actually-trained identity even when the zero-shot baseline
        # wins this iteration. Otherwise that losing config could be proposed
        # again because pi.H correctly belongs to the selected baseline.
        "trained_configs": trained_configs,
        "evaluation_state": {
            "last_eval": asdict(best_result),
            "test_report": deepcopy(state.get("test_report")),
        },
        "pi": {
            "D": {
                "version": state["dataset_version"],
                "path": state.get("current_dataset_path"),
                "plan": deepcopy(state.get("data_rebuild_plan")),
                "plan_identity": state.get("data_rebuild_plan_identity"),
                "config": deepcopy(
                    (state.get("last_curation") or {}).get(
                        "rebuild_config"
                    )
                ),
                "composition": deepcopy(state.get("last_curation")),
            },
            "H": {
                "lora_rank": best_cfg.get("lora_rank"),
                "lora_alpha": best_cfg.get("lora_alpha"),
                "lora_dropout": best_cfg.get("lora_dropout"),
                "weight_decay": best_cfg.get("weight_decay"),
                "learning_rate": best_cfg.get("learning_rate"),
                "nr_epochs": best_cfg.get("nr_epochs"),
                "micro_batch_size": best_cfg.get("micro_batch_size"),
                "gradient_accumulation_steps": best_cfg.get(
                    "gradient_accumulation_steps"
                ),
                "effective_batch_size": best_cfg.get(
                    "effective_batch_size"
                ),
                # Legacy checkpoint/readers still expect batch_size.
                "batch_size": best_cfg.get("batch_size"),
            },
            "S": {
                "task_type": task_type,
                "supervision": "direct",
                "loss_masking": "assistant_only",
                "loss_contract_version": SFT_LOSS_CONTRACT_VERSION,
            },
        },
    }
    # New list (not in-place) so the channel change persists — see B122 note above.
    state["dag"] = list(state.get("dag") or []) + [dag_node]

    # Write data-curation.md entry with hardware PASS/FAIL
    hw_profile = theoretical_hardware_profile(selector)
    from config.android_pool import check_hardware_constraints
    hw_constraints = check_hardware_constraints(state["selected_model"], state["hardware_constraints"])
    config_descriptions = state.get("_pending_configs", {})
    config_labels = list(config_descriptions.keys())
    config_a = config_labels[0] if config_labels else "N/A"
    config_b = config_labels[1] if len(config_labels) > 1 else "N/A"

    curation = state.get("last_curation") or {}
    log = CurationLog(state.get("curation_log_path"))
    log.write_iteration(
        iteration=state["iteration"],
        task_type=task_type,
        dataset_version=f"v{state['dataset_version']}",
        total_examples=curation.get(
            "total_examples",
            curation.get("n_gold", 0)
            + curation.get("n_hard_source", 0)
            + curation.get("n_hard_generated", curation.get("n_hard", 0))
            + curation.get("replay_count", 0),
        ),
        n_gold=curation.get("n_gold", 0),
        n_hard=curation.get("n_hard", 0),
        n_hard_source=curation.get("n_hard_source", 0),
        n_hard_generated=curation.get(
            "n_hard_generated",
            curation.get("n_hard", 0),
        ),
        replay_count=curation.get("replay_count", 0),
        rebuild_plan_identity=curation.get(
            "data_rebuild_plan_identity",
            "",
        ),
        strategy_composition=curation.get("strategy_composition", []),
        source_novelty=curation.get("source_novelty", {}),
        plan_yield=curation.get("plan_yield", {}),
        confusion_pairs=(
            (state.get("test_report") or {}).get("confusion_pairs") or []
        ),
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
        entry_id=f"{selector}:{state['iteration']}:{best_weights_ref}",
    )

    return state
