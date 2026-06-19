# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from agent.state import AgentState
from agent.nodes.task_analysis import task_analysis_node
from agent.nodes.eval_setup import eval_setup_node
from agent.nodes.train import train_node
from agent.nodes.evaluate import evaluate_node
from agent.nodes.iterate import iterate_node
from agent.nodes.curate import curate_node
from agent.nodes.rollback import rollback_node, should_rollback
from agent.nodes.escalate import escalate_node


def _route_after_evaluate(state: AgentState) -> str:
    """Route after evaluation: rollback if score decreased, else iterate."""
    if should_rollback(state):
        return "rollback"
    return "iterate"


def _route_after_iterate(state: AgentState) -> str:
    """Route based on next_action set by iterate_node."""
    return state.get("next_action", "curate")


def _route_after_escalate(state: AgentState) -> str:
    return state.get("next_action", "terminate")


def build_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("task_analysis", task_analysis_node)
    graph.add_node("eval_setup", eval_setup_node)
    graph.add_node("train", train_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("iterate", iterate_node)
    graph.add_node("curate", curate_node)
    graph.add_node("rollback", rollback_node)
    graph.add_node("escalate", escalate_node)

    # Linear entry path
    graph.set_entry_point("task_analysis")
    graph.add_edge("task_analysis", "eval_setup")
    graph.add_edge("eval_setup", "curate")   # build initial dataset before first train
    graph.add_edge("curate", "train")
    graph.add_edge("train", "evaluate")

    # Conditional routing after evaluate
    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {"rollback": "rollback", "iterate": "iterate"},
    )

    # After rollback: re-evaluate iteration decision
    graph.add_edge("rollback", "iterate")

    # Conditional routing after iterate
    graph.add_conditional_edges(
        "iterate",
        _route_after_iterate,
        {
            "train": "train",         # hyperparameter change, same data
            "curate": "curate",       # rebuild or augment data
            "escalate": "escalate",
            "terminate": END,
        },
    )

    # After escalate: train with new model or terminate
    graph.add_conditional_edges(
        "escalate",
        _route_after_escalate,
        {"train": "train", "terminate": END},
    )

    return graph.compile()
