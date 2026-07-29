# agent/graph.py
"""
LangGraph state machine for the SLM Factory fine-tuning loop.

Paper §2.1: 'Pioneer Agent is built on a LangGraph state machine orchestrated by
Claude Sonnet 4.6.' (In this codebase the orchestrator model is set in ONE place —
config.config.ORCHESTRATOR_MODEL — and every LLM node resolves it from there.)

Supports two modes (paper §2.5, §2.6):
  - cold_start: task_analysis → eval_setup → model_selection → curate → train → evaluate → iterate loop
  - production: trace_ingest → taxonomy → live_confirm → parent_awareness → curate → train loop

Model selection strategy is configurable via config.config.MODEL_SELECTION_STRATEGY:
  - "smallest_first"      — start smallest, escalate on failure
  - "largest_first"       — probe largest for feasibility, then start smallest
  - "interpolation"       — 3-probe scaling curve, pick closest to RAM target
  - "orchestrator_choice" — LLM picks based on task context
"""
import functools
from collections.abc import Mapping

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from agent.state import AgentState
from agent.timing import instrument_node
from config.config import MAX_TURNS_MAIN, MODEL_SELECTION_STRATEGY
from agent.nodes.cold_start.task_analysis import task_analysis_node
from agent.nodes.cold_start.eval_setup import eval_setup_node
from agent.nodes.cold_start.model_selection import get_model_selection_node
from agent.nodes.train import train_node
from agent.nodes.evaluate import evaluate_node
from agent.nodes.iterate import iterate_node, _wallclock_exceeded
from agent.nodes.curate import curate_node
from agent.nodes.rollback import rollback_node, should_rollback
from agent.nodes.escalate import escalate_node
from agent.nodes.downward_probe import downward_probe_step_node
from agent.checkpoint import RecursionBudgetExhausted


def guard_graph_node(
    name: str,
    node,
    *,
    max_steps: int = MAX_TURNS_MAIN,
):
    """Prevent side effects beyond the cumulative node-execution budget."""
    @functools.wraps(node)
    def guarded(state: AgentState, *args, **kwargs):
        completed = int(state.get("_graph_steps", 0) or 0)
        if completed >= int(max_steps):
            raise RecursionBudgetExhausted(
                f"cumulative LangGraph recursion budget of "
                f"{max_steps} steps is exhausted before node {name!r}"
            )
        if _wallclock_exceeded():
            update = dict(state)
            update["_graph_steps"] = completed + 1
            update["_wallclock_terminated_before"] = name
            update["next_action"] = "terminate"
            return update
        result = node(state, *args, **kwargs)
        if not isinstance(result, Mapping):
            raise TypeError(f"graph node {name!r} returned non-mapping state")
        update = dict(result)
        update["_graph_steps"] = completed + 1
        return update

    return guarded


def graph_run_config(
    thread_id: str,
    *,
    recursion_limit: int = MAX_TURNS_MAIN,
) -> dict:
    """Build the stable per-run config required by a durable checkpointer."""
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("LangGraph checkpoint thread_id must be non-empty")
    if int(recursion_limit) <= 0:
        raise ValueError("LangGraph recursion_limit must be positive")
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": int(recursion_limit),
    }


def graph_topology_descriptor(mode: str = "cold_start") -> dict:
    """Canonical graph structure used to reject unsafe checkpoint resumes."""
    if mode not in {"cold_start", "production"}:
        raise ValueError(f"Unknown graph mode: {mode!r}")
    common_nodes = {
        "train",
        "evaluate",
        "iterate",
        "curate",
        "rollback",
        "escalate",
        "downward_probe",
    }
    if mode == "cold_start":
        mode_nodes = {"task_analysis", "eval_setup", "model_selection"}
        entry_point = "task_analysis"
        mode_routes = {
            "task_analysis": {
                "eval_setup": "eval_setup",
                "terminate": "__end__",
            },
            "eval_setup": {
                "model_selection": "model_selection",
                "terminate": "__end__",
            },
            "model_selection": {
                "curate": "curate",
                "terminate": "__end__",
            },
        }
    else:
        mode_nodes = {
            "trace_ingest",
            "taxonomy_construct",
            "live_confirm",
            "parent_awareness",
        }
        entry_point = "trace_ingest"
        mode_routes = {
            "trace_ingest": {
                "taxonomy_construct": "taxonomy_construct",
                "terminate": "__end__",
            },
            "taxonomy_construct": {
                "live_confirm": "live_confirm",
                "terminate": "__end__",
            },
            "live_confirm": {
                "parent_awareness": "parent_awareness",
                "terminate": "__end__",
            },
            "parent_awareness": {
                "curate": "curate",
                "terminate": "__end__",
            },
        }
    return {
        "mode": mode,
        "entry_point": entry_point,
        "entry_routes": {
            entry_point: entry_point,
            "terminate": "__end__",
        },
        "nodes": sorted(common_nodes | mode_nodes),
        "edges": [],
        "conditional_edges": {
            **mode_routes,
            "curate": {
                "train": "train",
                "terminate": "__end__",
            },
            "train": {
                "evaluate": "evaluate",
                "terminate": "__end__",
            },
            "evaluate": {
                "rollback": "rollback",
                "iterate": "iterate",
                "terminate": "__end__",
            },
            "rollback": {
                "iterate": "iterate",
                "terminate": "__end__",
            },
            "iterate": {
                "train": "train",
                "curate": "curate",
                "escalate": "escalate",
                "downward_probe": "downward_probe",
                "terminate": "__end__",
            },
            "escalate": {
                "train": "train",
                "curate": "curate",
                "terminate": "__end__",
            },
            "downward_probe": {
                "downward_probe": "downward_probe",
                "terminate": "__end__",
            },
        },
    }


def _must_terminate(state: AgentState) -> bool:
    return (
        _wallclock_exceeded()
        or int(state.get("_graph_steps", 0) or 0) >= MAX_TURNS_MAIN
    )


def _route_after_evaluate(state: AgentState) -> str:
    if _must_terminate(state):
        return "terminate"
    if should_rollback(state):
        return "rollback"
    return "iterate"


def _route_after_iterate(state: AgentState) -> str:
    if _must_terminate(state):
        return "terminate"
    return state.get("next_action", "curate")


def _route_after_escalate(state: AgentState) -> str:
    if _must_terminate(state):
        return "terminate"
    return state.get("next_action", "terminate")


def _route_after_downward_probe(state: AgentState) -> str:
    if _must_terminate(state):
        return "terminate"
    return state.get("next_action", "terminate")


def _route_before(target: str):
    def route(state: AgentState) -> str:
        return "terminate" if _must_terminate(state) else target

    return route


def build_graph(
    mode: str = "cold_start",
    *,
    checkpointer=None,
) -> CompiledStateGraph:
    """Build the LangGraph state machine.

    Args:
        mode: "cold_start" (paper §2.5) or "production" (paper §2.6).
              Cold-start starts from a task description; production starts from
              judged inference traces of a deployed model.
    """
    graph_topology_descriptor(mode)  # validate before importing mode-specific nodes
    graph = StateGraph(AgentState)

    def durable_node(name, node):
        return guard_graph_node(name, instrument_node(name, node))

    # Common nodes (both modes share the training loop)
    graph.add_node("train", durable_node("train", train_node))
    graph.add_node("evaluate", durable_node("evaluate", evaluate_node))
    graph.add_node("iterate", durable_node("iterate", iterate_node))
    graph.add_node("curate", durable_node("curate", curate_node))
    graph.add_node("rollback", durable_node("rollback", rollback_node))
    graph.add_node("escalate", durable_node("escalate", escalate_node))
    graph.add_node(
        "downward_probe",
        durable_node("downward_probe", downward_probe_step_node),
    )

    if mode == "cold_start":
        model_selection_node = get_model_selection_node(MODEL_SELECTION_STRATEGY)

        graph.add_node(
            "task_analysis", durable_node("task_analysis", task_analysis_node)
        )
        graph.add_node("eval_setup", durable_node("eval_setup", eval_setup_node))
        graph.add_node(
            "model_selection",
            durable_node("model_selection", model_selection_node),
        )

        graph.set_conditional_entry_point(
            _route_before("task_analysis"),
            {"task_analysis": "task_analysis", "terminate": END},
        )
        graph.add_conditional_edges(
            "task_analysis",
            _route_before("eval_setup"),
            {"eval_setup": "eval_setup", "terminate": END},
        )
        graph.add_conditional_edges(
            "eval_setup",
            _route_before("model_selection"),
            {"model_selection": "model_selection", "terminate": END},
        )
        graph.add_conditional_edges(
            "model_selection",
            _route_before("curate"),
            {"curate": "curate", "terminate": END},
        )
    else:
        from agent.nodes.production.trace_ingest import trace_ingest_node
        from agent.nodes.production.taxonomy import taxonomy_construct_node
        from agent.nodes.production.live_confirm import live_confirm_node
        from agent.nodes.production.parent_awareness import parent_awareness_node

        graph.add_node(
            "trace_ingest", durable_node("trace_ingest", trace_ingest_node)
        )
        graph.add_node(
            "taxonomy_construct",
            durable_node("taxonomy_construct", taxonomy_construct_node),
        )
        graph.add_node(
            "live_confirm", durable_node("live_confirm", live_confirm_node)
        )
        graph.add_node(
            "parent_awareness",
            durable_node("parent_awareness", parent_awareness_node),
        )

        graph.set_conditional_entry_point(
            _route_before("trace_ingest"),
            {"trace_ingest": "trace_ingest", "terminate": END},
        )
        graph.add_conditional_edges(
            "trace_ingest",
            _route_before("taxonomy_construct"),
            {"taxonomy_construct": "taxonomy_construct", "terminate": END},
        )
        graph.add_conditional_edges(
            "taxonomy_construct",
            _route_before("live_confirm"),
            {"live_confirm": "live_confirm", "terminate": END},
        )
        graph.add_conditional_edges(
            "live_confirm",
            _route_before("parent_awareness"),
            {"parent_awareness": "parent_awareness", "terminate": END},
        )
        graph.add_conditional_edges(
            "parent_awareness",
            _route_before("curate"),
            {"curate": "curate", "terminate": END},
        )

    # Shared loop edges (both modes)
    graph.add_conditional_edges(
        "curate",
        _route_before("train"),
        {"train": "train", "terminate": END},
    )
    graph.add_conditional_edges(
        "train",
        _route_before("evaluate"),
        {"evaluate": "evaluate", "terminate": END},
    )

    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {"rollback": "rollback", "iterate": "iterate", "terminate": END},
    )

    # After a regression we rollback (restore the best checkpoint) and then RE-ENTER the
    # decision loop (iterate) rather than blindly re-training the identical config. A bare
    # re-train on the same dataset + hyperparameters is (near-)deterministic, so it would
    # reproduce the same regressing score and rollback again — an endless loop. Routing to
    # iterate forces a *different* next action (data_rebuild with a rotated seed, a
    # hyperparameter change, escalation, or termination via the stall guard).
    graph.add_conditional_edges(
        "rollback",
        _route_before("iterate"),
        {"iterate": "iterate", "terminate": END},
    )

    graph.add_conditional_edges(
        "iterate",
        _route_after_iterate,
        {
            "train": "train",
            "curate": "curate",
            "escalate": "escalate",
            "downward_probe": "downward_probe",
            "terminate": END,
        },
    )

    graph.add_conditional_edges(
        "downward_probe",
        _route_after_downward_probe,
        {"downward_probe": "downward_probe", "terminate": END},
    )

    graph.add_conditional_edges(
        "escalate",
        _route_after_escalate,
        {"train": "train", "curate": "curate", "terminate": END},
    )

    compiled = (
        graph.compile(checkpointer=checkpointer)
        if checkpointer is not None
        else graph.compile()
    )
    return compiled.with_config(recursion_limit=MAX_TURNS_MAIN)
