# agent/graph.py
"""
LangGraph state machine for the SLM Factory fine-tuning loop.

Paper §2.1: 'Pioneer Agent is built on a LangGraph state machine orchestrated by
Claude Sonnet 4.6.' (In this codebase the orchestrator model is set in ONE place —
config.config.ORCHESTRATOR_MODEL — and every LLM node resolves it from there.)

Supports two modes (paper §2.5, §2.6):
  - cold_start: task_analysis → eval_setup → scaling_curve → curate → train → evaluate → iterate loop
  - production: trace_ingest → taxonomy → live_confirm → parent_awareness → curate → train loop
"""
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from agent.state import AgentState
from config.config import MAX_TURNS_MAIN
from agent.nodes.cold_start.task_analysis import task_analysis_node
from agent.nodes.cold_start.eval_setup import eval_setup_node
from agent.nodes.cold_start.scaling_curve import scaling_curve_node
from agent.nodes.train import train_node
from agent.nodes.evaluate import evaluate_node
from agent.nodes.iterate import iterate_node
from agent.nodes.curate import curate_node
from agent.nodes.rollback import rollback_node, should_rollback
from agent.nodes.escalate import escalate_node
from agent.nodes.downward_probe import downward_probe_node


def _route_after_evaluate(state: AgentState) -> str:
    if should_rollback(state):
        return "rollback"
    return "iterate"


def _route_after_iterate(state: AgentState) -> str:
    return state.get("next_action", "curate")


def _route_after_escalate(state: AgentState) -> str:
    return state.get("next_action", "terminate")


def build_graph(mode: str = "cold_start") -> CompiledStateGraph:
    """Build the LangGraph state machine.

    Args:
        mode: "cold_start" (paper §2.5) or "production" (paper §2.6).
              Cold-start starts from a task description; production starts from
              judged inference traces of a deployed model.
    """
    graph = StateGraph(AgentState)

    # Common nodes (both modes share the training loop)
    graph.add_node("train", train_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("iterate", iterate_node)
    graph.add_node("curate", curate_node)
    graph.add_node("rollback", rollback_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("downward_probe", downward_probe_node)

    if mode == "cold_start":
        graph.add_node("task_analysis", task_analysis_node)
        graph.add_node("eval_setup", eval_setup_node)
        graph.add_node("scaling_curve", scaling_curve_node)

        graph.set_entry_point("task_analysis")
        graph.add_edge("task_analysis", "eval_setup")
        graph.add_edge("eval_setup", "scaling_curve")
        graph.add_edge("scaling_curve", "curate")
    else:
        from agent.nodes.production.trace_ingest import trace_ingest_node
        from agent.nodes.production.taxonomy import taxonomy_construct_node
        from agent.nodes.production.live_confirm import live_confirm_node
        from agent.nodes.production.parent_awareness import parent_awareness_node

        graph.add_node("trace_ingest", trace_ingest_node)
        graph.add_node("taxonomy_construct", taxonomy_construct_node)
        graph.add_node("live_confirm", live_confirm_node)
        graph.add_node("parent_awareness", parent_awareness_node)

        graph.set_entry_point("trace_ingest")
        graph.add_edge("trace_ingest", "taxonomy_construct")
        graph.add_edge("taxonomy_construct", "live_confirm")
        graph.add_edge("live_confirm", "parent_awareness")
        graph.add_edge("parent_awareness", "curate")

    # Shared loop edges (both modes)
    graph.add_edge("curate", "train")
    graph.add_edge("train", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {"rollback": "rollback", "iterate": "iterate"},
    )

    # After a regression we rollback (restore the best checkpoint) and then RE-ENTER the
    # decision loop (iterate) rather than blindly re-training the identical config. A bare
    # re-train on the same dataset + hyperparameters is (near-)deterministic, so it would
    # reproduce the same regressing score and rollback again — an endless loop. Routing to
    # iterate forces a *different* next action (data_rebuild with a rotated seed, a
    # hyperparameter change, escalation, or termination via the stall guard).
    graph.add_edge("rollback", "iterate")

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

    graph.add_edge("downward_probe", END)

    graph.add_conditional_edges(
        "escalate",
        _route_after_escalate,
        {"train": "train", "curate": "curate", "terminate": END},
    )

    return graph.compile().with_config(recursion_limit=MAX_TURNS_MAIN)
