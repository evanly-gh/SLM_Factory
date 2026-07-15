# agent/nodes/cold_start/model_selection/base.py
"""
Base interface for model selection strategies.

All strategies share the same contract:
  Input:  AgentState with feasible_models, stop_threshold, eval_set, task_type, etc.
  Output: AgentState with selected_model set.

Each strategy is a function: (AgentState) -> AgentState
"""
import logging
from agent.state import AgentState
from config.android_pool import ModelSpec

logger = logging.getLogger(__name__)


def select_smallest(models: list[ModelSpec]) -> ModelSpec:
    """Return the model with the lowest peak_memory_mb."""
    return min(models, key=lambda m: m.peak_memory_mb)


def select_largest(models: list[ModelSpec]) -> ModelSpec:
    """Return the model with the highest peak_memory_mb."""
    return max(models, key=lambda m: m.peak_memory_mb)


def select_by_ram_target(models: list[ModelSpec], target_mb: int) -> ModelSpec:
    """Return the model whose peak_memory_mb is closest to target_mb."""
    return min(models, key=lambda m: abs(m.peak_memory_mb - target_mb))
