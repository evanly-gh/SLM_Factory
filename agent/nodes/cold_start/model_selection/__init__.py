# agent/nodes/cold_start/model_selection/__init__.py
"""
Model selection strategies — interchangeable approaches for choosing the
initial model from the hardware-feasible set.

All strategies share the same interface:
    node(state: AgentState) -> AgentState
    Sets state["selected_model"] from state["feasible_models"].

Available strategies (set MODEL_SELECTION_STRATEGY in config.py or env):

    "smallest_first"      Start with the smallest model, escalate on failure.
                          No probing overhead; most resource-conservative.

    "largest_first"       Probe with the largest model to test feasibility,
                          then drop to smallest and escalate from there.

    "interpolation"       Probe 3 models, fit a scaling curve, pick the model
                          closest to the RAM budget that meets the accuracy goal.
                          (Original scaling_curve approach from the paper.)

    "orchestrator_choice" Let the orchestrator LLM pick the starting model
                          based on task context and benchmark affinities.
                          No probing; single API call.
"""
from agent.nodes.cold_start.model_selection.smallest_first import smallest_first_node
from agent.nodes.cold_start.model_selection.largest_first import largest_first_node
from agent.nodes.cold_start.model_selection.interpolation import interpolation_node
from agent.nodes.cold_start.model_selection.orchestrator_choice import orchestrator_choice_node

STRATEGIES = {
    "smallest_first": smallest_first_node,
    "largest_first": largest_first_node,
    "interpolation": interpolation_node,
    "orchestrator_choice": orchestrator_choice_node,
}


def get_model_selection_node(strategy: str):
    """Return the model selection node function for the given strategy name."""
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown MODEL_SELECTION_STRATEGY: {strategy!r}. "
            f"Valid options: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[strategy]
