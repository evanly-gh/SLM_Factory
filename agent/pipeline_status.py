"""Pure helpers for truthful pipeline completion/failure reporting."""
import copy


def run_heading(error: BaseException | None) -> str:
    return "RUN FAILED" if error is not None else "RUN COMPLETE"


def outcome_text(
    converged: bool,
    error: BaseException | None,
    model_label: str,
) -> str:
    if error is not None:
        return f"FAILED: {type(error).__name__}: {error}"
    if converged:
        return f"CONVERGED on {model_label}"
    return "did NOT converge (escalation ceiling / step limit)"


def process_exit_code(error: BaseException | None) -> int:
    return 1 if error is not None else 0


def _first_finetuned_for(selector: str, baselines: list[dict]) -> float | None:
    """The best score this variant reached on its FIRST fine-tuned iteration.

    Recorded by evaluate_node before the zero-shot baseline is added as a candidate, so it is a
    genuine fine-tuned number even on an iteration the baseline went on to win.
    """
    return next(
        (
            entry.get("first_finetuned_f1")
            for entry in baselines
            if entry.get("selector", entry.get("model_id")) == selector
        ),
        None,
    )


def _baseline_for(selector: str, baselines: list[dict]) -> float | None:
    return next(
        (
            entry.get("baseline_f1")
            for entry in baselines
            if entry.get("selector", entry.get("model_id")) == selector
        ),
        None,
    )


def build_run_progression(
    state: dict,
    baselines: list[dict],
) -> list[dict]:
    """Build truthful per-model progression, including downward probes.

    The normal-loop ``scores``/``dag`` remain attached to the model that produced
    them. Post-convergence probe attempts become separate entries, so adopting a
    smaller model never relabels the original model's trajectory.
    """
    progression = copy.deepcopy(state.get("escalation_history") or [])
    # escalate_node stashes a finished tier WITHOUT a "kind", while every entry built below
    # carries one. Consumers that filter on kind == "model_trajectory" therefore dropped every
    # earlier tier and graphed only the final model (B255). Tag them at the source so the
    # progression is uniform whatever the producer wrote.
    for entry in progression:
        entry.setdefault("kind", "model_trajectory")
    downward = state.get("downward_probe_history") or {}
    origin = downward.get("origin")
    if origin:
        origin_entry = copy.deepcopy(origin)
        origin_entry["kind"] = "model_trajectory"
        origin_entry["baseline_f1"] = _baseline_for(
            origin_entry["selector"],
            baselines,
        )
        origin_entry["first_finetuned_f1"] = _first_finetuned_for(
            origin_entry["selector"],
            baselines,
        )
        origin_entry["best_score"] = origin_entry.get("score", 0.0)
        progression.append(origin_entry)
        for attempt in downward.get("attempts") or []:
            score = attempt.get("score")
            progression.append({
                **copy.deepcopy(attempt),
                "kind": "downward_probe",
                "baseline_f1": None,
                # A probe runs exactly one config, so its first fine-tuned score IS its best.
                "first_finetuned_f1": score,
                "best_score": score,
                "iterations": 1,
                "scores": [score] if score is not None else [],
                "dag": [],
            })
        return progression

    model = state.get("selected_model")
    if model is not None:
        progression.append({
            "kind": "model_trajectory",
            "selector": model.selector,
            "model_id": model.model_id,
            "quant": getattr(model, "quant", None),
            "tier": getattr(model, "tier", "?"),
            "baseline_f1": _baseline_for(model.selector, baselines),
            "first_finetuned_f1": _first_finetuned_for(model.selector, baselines),
            "best_score": state.get("best_score", 0.0),
            "iterations": state.get("iteration", 0),
            "scores": list(state.get("scores") or []),
            "dag": copy.deepcopy(state.get("dag") or []),
            "weights_ref": state.get("best_weights_ref"),
        })
    return progression


def _mined_source_labels(composition: dict) -> list[str]:
    """Where this rebuild's mined rows actually came from, most-used first.

    `source_usage` counts every source that contributed to the curriculum, including the run's
    original train pool, so it is filtered to entries that carry a url or a hub-style key — those
    are the ones a mining round added. Falls back to the mining report's own `source_records`,
    which is populated even when the accepted rows were later dropped by the eval firewall.
    """
    labels: list[str] = []
    for entry in composition.get("source_usage") or []:
        if not isinstance(entry, dict) or not entry.get("novel_rows"):
            continue
        label = entry.get("url") or entry.get("source")
        if label and label not in labels:
            labels.append(str(label))
    if labels:
        return labels
    for record in (composition.get("source_novelty") or {}).get("source_records") or []:
        if not isinstance(record, dict):
            continue
        label = record.get("url") or f"{record.get('kind', 'source')}:{record.get('id', '?')}"
        if label not in labels:
            labels.append(str(label))
    return labels


def format_intervention_detail(node: dict) -> str:
    """Name what a DAG node's intervention actually did, for the trajectory table.

    `data_rebuild` on its own is the least informative thing the column could say: it covers
    generating synthetic rows from train anchors and mining new real rows off the hub, which have
    different costs and different failure modes, and it hides the only number that matters —
    how many rows the rebuild actually added. Both are recorded on the node
    (``pi.D.composition``, a snapshot of ``state["last_curation"]``), so the column reports the
    sub-strategy, the row count it produced, and for mining the source it came from.
    """
    intervention = str(node.get("intervention") or "?")
    if intervention != "data_rebuild":
        return intervention
    dag_data = (node.get("pi") or {}).get("D") or {}
    composition = dag_data.get("composition") or {}
    strategy = str((dag_data.get("plan") or {}).get("strategy") or "?")
    novel = int((composition.get("plan_yield") or {}).get("novel_rows", 0) or 0)

    if strategy == "synthesize":
        generated = int(composition.get("n_synth_total", 0) or 0)
        if not generated:
            return "data_rebuild/synthesize: 0 synthetic rows kept"
        return f"data_rebuild/synthesize: +{generated} synthetic row(s) ({novel} novel)"
    if strategy == "acquire":
        mined = int(composition.get("n_hard_source", 0) or 0)
        report = composition.get("source_novelty") or {}
        if not mined:
            rejected = int(report.get("rejected_sources", 0) or 0)
            reason = str(report.get("status") or "no_novelty")
            suffix = f", {rejected} source(s) rejected" if rejected else ""
            return f"data_rebuild/mine-new-real: 0 new rows ({reason}{suffix})"
        sources = _mined_source_labels(composition)
        origin = f" from {', '.join(sources[:2])}" if sources else ""
        return f"data_rebuild/mine-new-real: +{mined} mined row(s){origin}"
    return f"data_rebuild/{strategy}"


def format_downward_probe_history(history: dict | None) -> list[str]:
    """Render exact-selector downward attempts for the final run report."""
    history = history or {}
    origin = history.get("origin")
    attempts = history.get("attempts") or []
    termination = history.get("termination")
    if not origin and not attempts and not termination:
        return []

    lines = ["  Downward Probe History:"]
    if origin:
        trajectory = " → ".join(
            f"{score:.3f}" for score in origin.get("scores") or []
        ) or "(none)"
        lines.append(
            f"    origin={origin.get('selector', '?')} "
            f"score={origin.get('score', 0.0):.4f} "
            f"weights={origin.get('weights_ref') or 'n/a'} "
            f"trajectory={trajectory} "
            f"dag_nodes={len(origin.get('dag') or [])}"
        )
    for index, attempt in enumerate(attempts, start=1):
        score = attempt.get("score")
        score_text = f"{score:.4f}" if score is not None else "n/a"
        line = (
            f"    probe[{index}] selector={attempt.get('selector', '?')} "
            f"tier={attempt.get('tier', '?')} score={score_text} "
            f"weights={attempt.get('weights_ref') or 'n/a'} "
            f"result={attempt.get('result', '?')} "
            f"adopted={'yes' if attempt.get('adopted') else 'no'}"
        )
        if attempt.get("error"):
            line += f" error={attempt['error']}"
        lines.append(line)
    if termination:
        candidates = ",".join(
            termination.get("candidate_selectors") or []
        ) or "(none)"
        lines.append(
            f"    termination stage={termination.get('stage', '?')} "
            f"result={termination.get('result', '?')} "
            f"target_tier={termination.get('target_tier')} "
            f"candidates={candidates} "
            f"reason={termination.get('reason', '?')}"
        )
    return lines
