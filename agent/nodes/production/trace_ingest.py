# agent/nodes/production/trace_ingest.py
"""
Production Node 1: Trace Ingestion (paper §2.6, Eq. 12-13).

Loads judged inference traces T from disk, partitions into T_fail and T_pass.
Each trace is: t_i = (input, prediction, corrected_output, verdict, reasoning)
"""
import json
from agent.state import AgentState


def trace_ingest_node(state: AgentState) -> AgentState:
    """Load judged inference traces and partition into failure/passing sets.

    Paper §2.6: 'T_fail = {t_i | v_i = fail}, T_pass = {t_i | v_i = pass}'

    Traces are read from state["traces"] if already loaded, otherwise from the path
    stored in state["description"] is not used — the caller must pre-populate
    state["traces"] or ensure a "traces.jsonl" file exists in the working directory.
    """
    state["mode"] = "production"

    traces = []
    if state.get("traces"):
        traces = state["traces"]
    else:
        traces_path = "traces.jsonl"
        try:
            with open(traces_path) as f:
                traces = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            print(f"[trace_ingest] Traces file not found: {traces_path}")

    t_fail = [t for t in traces if t.get("verdict") == "fail"]
    t_pass = [t for t in traces if t.get("verdict") == "pass"]

    state["traces"] = traces
    state["train_examples"] = [
        {"text": t.get("input", ""), "label": t.get("corrected_output", "")}
        for t in t_fail
    ]

    print(f"[trace_ingest] Loaded {len(traces)} traces: {len(t_fail)} fail, {len(t_pass)} pass")
    return state
