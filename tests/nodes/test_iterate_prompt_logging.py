"""B256: the orchestrator system prompt must be logged ONCE per run, not every turn.

`_llm_iterate` sets `state["_iterate_prompt_logged"] = True` after logging, but the key was not
declared on AgentState. LangGraph merges a node's returned state against that TypedDict schema
and drops undeclared keys, so the flag never survived to the next call and the full system
prompt was re-emitted on all 69 iterate turns of run 38303490.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")


def test_prompt_logged_flag_is_declared_on_the_state_schema():
    from agent.state import AgentState

    annotations = getattr(AgentState, "__annotations__", {})
    assert "_iterate_prompt_logged" in annotations, (
        "an undeclared key is dropped on the LangGraph state merge, so the guard resets "
        "every turn and the system prompt is logged repeatedly"
    )


def test_every_key_iterate_persists_is_declared():
    """Guard the whole class of bug: anything iterate writes for a LATER turn to read must be
    on the schema, or it silently reads back as absent."""
    import re

    from agent.state import AgentState

    source = open("agent/nodes/iterate.py", encoding="utf-8").read()
    written = set(re.findall(r'state\[[\'"](_[a-z_]+)[\'"]\]\s*=', source))
    annotations = set(getattr(AgentState, "__annotations__", {}))
    undeclared = sorted(written - annotations)
    assert undeclared == [], (
        f"iterate writes these private state keys that AgentState does not declare, so they "
        f"are dropped on the state merge: {undeclared}"
    )
