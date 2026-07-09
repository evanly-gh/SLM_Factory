# agent/nodes/production/live_confirm.py
"""
Production Node 3: Live Confirmation (paper §2.6).

Verifies that identified weaknesses are systematic rather than sampling artifacts.
Re-runs failing inputs through M0 to confirm they still fail.
"""
from agent.state import AgentState


def live_confirm_node(state: AgentState) -> AgentState:
    """Confirm failures are systematic by re-running M0 on failing inputs.

    Paper §2.6: 'Before constructing training data, the agent first verifies that
    each identified weakness is systematic rather than an artifact of sampling noise.'

    Step 1 (pre-screen): filter failures to those whose cluster was labelled fixable
    by taxonomy_construct_node. Traces tagged with cluster=None (outside the taxonomy
    sample window) are treated as unclassified and kept for re-inference.

    Step 2 (re-inference): load M0 from state['deployed_model_ref'] and re-run each
    candidate failure's input through the model. Only retain failures where M0 still
    produces an incorrect output (i.e. the output does not match corrected_output).
    """
    taxonomy = state.get("failure_taxonomy", {})
    traces = state.get("traces", [])
    failures = [t for t in traces if t.get("verdict") == "fail"]

    # --- Step 1: taxonomy pre-screen (bug 1 fix) ---
    # Filter by the 'cluster' key written by taxonomy_construct_node, not by raw
    # string-matching the cluster name against the entire trace repr.
    fixable_clusters = {c["name"] for c in taxonomy.get("clusters", []) if c.get("fixable")}
    external_clusters = {c["name"] for c in taxonomy.get("clusters", []) if not c.get("fixable")}

    if fixable_clusters or external_clusters:
        # Keep traces whose cluster is fixable, or whose cluster is unassigned (None).
        # Traces explicitly assigned to an external cluster are excluded here.
        prescreened = [
            t for t in failures
            if t.get("cluster") not in external_clusters
        ]
        external_prescreen = [t for t in failures if t.get("cluster") in external_clusters]
    else:
        # No taxonomy available — pass all failures through to re-inference.
        prescreened = failures
        external_prescreen = []

    # --- Step 2: M0 re-inference (bug 2 fix) ---
    # Re-run each candidate failure through the deployed model M0 to confirm the
    # failure is systematic and not a sampling artifact.
    model_ref = state.get("deployed_model_ref")
    confirmed = []
    sampling_artifacts = []

    if model_ref:
        from training.slm_helpers import infer

        for t in prescreened:
            prompt = t.get("input", "")
            expected = t.get("corrected_output", "")
            try:
                prediction = infer(prompt, model_ref, model_ref)
            except Exception as e:
                # If inference fails for a trace, conservatively treat it as confirmed
                # so the failure is not silently dropped.
                print(f"[live_confirm] inference error on trace (kept): {e}")
                confirmed.append(t)
                continue

            # Failure is confirmed if M0 still does not produce the correct output.
            if prediction.strip() != expected.strip():
                confirmed.append(t)
            else:
                sampling_artifacts.append(t)
    else:
        # No deployed model reference available; skip re-inference and keep all
        # prescreened failures. Log a warning so operators are aware.
        print(
            "[live_confirm] WARNING: state['deployed_model_ref'] is not set — "
            "skipping M0 re-inference. All taxonomy-prescreened failures are "
            "treated as confirmed. Set deployed_model_ref for full paper §2.6 "
            "confirmation semantics."
        )
        confirmed = prescreened

    state["train_examples"] = [
        {"text": t.get("input", ""), "label": t.get("corrected_output", "")}
        for t in confirmed
    ]

    print(
        f"[live_confirm] {len(confirmed)} confirmed fixable, "
        f"{len(external_prescreen)} external (taxonomy-excluded), "
        f"{len(sampling_artifacts)} sampling artifacts (M0 now passes)"
    )
    return state
