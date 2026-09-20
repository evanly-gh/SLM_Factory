# agent/nodes/rollback.py
from copy import deepcopy

from agent.state import SKIPPED_NO_ROWS, AgentState
from eval.harness import EvalResult


def _log(model_id: str, msg: str):
    print(f"[rollback][{model_id}] {msg}")


def should_rollback(state: AgentState) -> bool:
    """
    Cold-start simple rollback: if f(π_i+1) < f(π_i), revert.
    """
    scores = state["scores"]
    if len(scores) < 2:
        return False
    return scores[-1] < scores[-2]


def rollback_node(state: AgentState) -> AgentState:
    """
    Node 6: revert to previous configuration if score decreased.
    Removes the last score from history.
    Restores best_weights_ref to the best non-pruned DAG node.
    """
    if not should_rollback(state):
        return state

    model_id = state["selected_model"].label if state.get("selected_model") else "?"
    regressed_score = state["scores"][-1]
    previous_score = state["scores"][-2] if len(state["scores"]) >= 2 else 0.0

    _log(model_id, f"REGRESSION detected: {regressed_score:.4f} < {previous_score:.4f} "
         f"(Δ={regressed_score - previous_score:+.4f})")

    # Remove the regressing score
    state["scores"].pop()
    state["last_intervention"] = "rollback"

    # Mark the last DAG node as pruned
    if state["dag"]:
        pruned_node = state["dag"][-1]
        pruned_node["pruned"] = True
        # `score` is formatted defensively for the same reason the filter below exists: a scoreless
        # node in this position would turn a rollback into a TypeError inside a log line.
        _pruned_score = pruned_node.get("score")
        _pruned_shown = (
            f"{_pruned_score:.4f}" if isinstance(_pruned_score, (int, float)) else "unmeasured"
        )
        _log(model_id, f"  Pruned DAG node: iteration={pruned_node.get('iteration', '?')}  "
             f"config={pruned_node.get('best_config', '?')}  "
             f"score={_pruned_shown}")

    # Restore best_weights_ref to the best MEASURED, non-pruned DAG node.
    #
    # `pruned` alone is not enough. `curate._record_skipped_iteration` puts a node in the DAG for a
    # rebuild that added no rows — deliberately with `score: None`, because nothing was trained or
    # evaluated and inventing a score would put a fabricated measurement into the trajectory. That
    # node is not pruned (it was never rejected, it was never measured), so it passed this filter
    # and reached the `max()` below, where comparing None against a float raises. It also carries
    # no `weights_ref` key at all, so the very next line would KeyError even with the score handled.
    #
    # This killed all three ablation runs of 2026-09-04 (39562028/29/30) at tier 1 iteration 8,
    # after 5h40m of L40S time between them. BC5CDR guarantees it: the local bundle holds 5,096
    # train rows against the 5,000 eval_setup takes, so `mine_new_real` yields +96 once and +0
    # forever after — two consecutive skipped iterations, then the first regression, then here.
    #
    # `status` is the documented marker for this and `score is None` is the belt-and-braces check,
    # because a node that cannot be compared must not be reachable by either route.
    non_pruned = [
        n for n in state["dag"]
        if not n.get("pruned", False)
        and n.get("status") != SKIPPED_NO_ROWS
        and n.get("score") is not None
    ]
    if non_pruned:
        best_node = max(non_pruned, key=lambda n: n["score"])
        state["best_weights_ref"] = best_node["weights_ref"]
        state["best_score"] = best_node["score"]
        dataset = ((best_node.get("pi") or {}).get("D") or {})
        dataset_path = dataset.get("path")
        if not isinstance(dataset_path, str) or not dataset_path:
            raise RuntimeError(
                "winning rollback DAG node has no dataset artifact path: "
                f"{dataset_path!r}"
            )
        state["current_dataset_path"] = dataset_path
        state["dataset_version"] = int(dataset.get("version", 0) or 0)
        state["last_curation"] = deepcopy(dataset.get("composition"))
        state["data_rebuild_plan"] = deepcopy(dataset.get("plan"))
        state["data_rebuild_plan_identity"] = dataset.get("plan_identity")
        evaluation_state = best_node.get("evaluation_state") or {}
        encoded_eval = evaluation_state.get("last_eval")
        state["last_eval"] = (
            EvalResult(**deepcopy(encoded_eval))
            if isinstance(encoded_eval, dict)
            else None
        )
        # The diagnosis IS rolled back, deliberately. After a rollback the live weights are the
        # restored best checkpoint, so the per-difficulty scores and confusion pairs that describe
        # the CURRENT model are the best node's — not the discarded attempt's. Judging the next
        # intervention from the failed attempt's numbers would mean reasoning about a model that
        # no longer exists.
        #
        # What the orchestrator additionally needs is a memo of what was just tried and why it
        # failed, so it does not simply repeat it. That is `last_failed_attempt` below, which is
        # NOT part of the restored state and is surfaced separately in the prompt (B227/B231).
        restored_report = deepcopy(evaluation_state.get("test_report"))
        failed_report = state.get("test_report") or {}
        state["test_report"] = restored_report
        state["last_failed_attempt"] = {
            "iteration": pruned_node.get("iteration") if state["dag"] else None,
            "intervention": pruned_node.get("intervention") if state["dag"] else None,
            "sub_strategy": (
                (((pruned_node.get("pi") or {}).get("D") or {}).get("plan") or {})
                .get("strategy")
                if state["dag"] else None
            ),
            "hypothesis": (pruned_node.get("hypothesis") if state["dag"] else "") or "",
            "score": regressed_score,
            "best_score": best_node["score"],
            "delta": round(regressed_score - best_node["score"], 4),
            # The failed attempt's own difficulty profile, kept ONLY as failure evidence.
            "by_difficulty": deepcopy(failed_report.get("by_difficulty")),
        }
        best_label = best_node.get("best_config", "restored best")
        best_hparams = dict(
            ((best_node.get("pi") or {}).get("H") or {})
        )
        if best_hparams:
            restored_config = {**best_hparams, "label": best_label}
            state["_pending_configs"] = {best_label: restored_config}
            state["_pending_weights_refs"] = {
                best_label: best_node["weights_ref"]
            }
            state["_pending_training_outputs"] = None
        _log(model_id, f"  Restored to: iteration={best_node['iteration']}  "
             f"score={best_node['score']:.4f}  "
             f"weights={best_node['weights_ref']}")
        if best_hparams.get("lora_rank") is not None:
            _log(
                model_id,
                "  Restored optimizer config: "
                f"r={best_hparams.get('lora_rank')} "
                f"alpha={best_hparams.get('lora_alpha')} "
                f"dropout={best_hparams.get('lora_dropout')} "
                f"weight_decay={best_hparams.get('weight_decay')} "
                f"lr={best_hparams.get('learning_rate')} "
                f"epochs={best_hparams.get('nr_epochs')} "
                f"micro_batch={best_hparams.get('micro_batch_size', best_hparams.get('batch_size'))} "
                f"grad_accum={best_hparams.get('gradient_accumulation_steps', 1)} "
                f"effective_batch={best_hparams.get('effective_batch_size')}",
            )
    else:
        # The message now counts the two ways a node can be unusable, because they are not the
        # same failure and the old wording ("all DAG nodes pruned") named only one of them. A
        # skipped iteration is not pruned and never had a checkpoint to begin with, so a DAG made
        # of skipped nodes plus one pruned node lands here while being nothing like "all pruned".
        _n_pruned = sum(1 for n in state["dag"] if n.get("pruned", False))
        _n_skipped = sum(1 for n in state["dag"] if n.get("score") is None)
        _log(model_id, f"  WARNING: no measured, unpruned DAG node to restore "
                       f"({_n_pruned} pruned, {_n_skipped} unmeasured of {len(state['dag'])})")
        raise RuntimeError(
            f"[rollback][{model_id}] Inconsistent state: rollback was triggered but no DAG node "
            f"holds a restorable checkpoint — {_n_pruned} of {len(state['dag'])} are pruned and "
            f"{_n_skipped} were never evaluated. scores={state['scores']}."
        )

    _log(model_id, f"  Restored best checkpoint; re-entering decision loop (iterate) to pick a "
                   f"DIFFERENT next action — a bare re-train of the same config would just regress again")

    return state
