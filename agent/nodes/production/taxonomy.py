# agent/nodes/production/taxonomy.py
"""
Production Node 2: Failure Taxonomy Construction (paper §2.6, Eq. 14).

Clusters failures into K categories, labels each as fixable or external.
Paper: 'The agent partitions T_fail into K failure clusters {C_1,...,C_K}
such that union(C_k) = T_fail.'
"""
import json
from agent.state import AgentState


def taxonomy_construct_node(state: AgentState) -> AgentState:
    """Cluster failures into actionable categories via the orchestrator LLM.

    Paper §2.6: For each cluster C_k, records size, dominant input characteristics,
    and a fixability label phi_k in {fixable, external}.
    """
    import anthropic
    from config.config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL

    traces = state.get("traces", [])
    failures = [t for t in traces if t.get("verdict") == "fail"]

    if not failures:
        state["failure_taxonomy"] = {"clusters": [], "summary": "No failures to classify"}
        return state

    # Single consistent sample window: the LLM sees exactly the traces it is asked to
    # assign, and only those get a cluster tag. (Previously it saw 20 but the code
    # tagged against a 50-slice, leaving a silent mismatch.)
    SAMPLE_N = 40
    sample = failures[:SAMPLE_N]
    sample_text = json.dumps(
        [{"idx": i, **t} for i, t in enumerate(sample)],
        indent=2,
        default=str,
    )
    unsampled = len(failures) - len(sample)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=ORCHESTRATOR_MODEL,
        max_tokens=2048,
        system=(
            "You are analyzing failure patterns in a deployed model's inference logs. "
            "Cluster the sampled failures into a small number of categories (aim for 3-8). "
            "For each cluster, determine:\n"
            "1. name: a short descriptive cluster name\n"
            "2. count: how many of the sampled traces fall in this cluster\n"
            "3. root_cause: one sentence on the shared cause\n"
            "4. fixable: a JSON boolean — true if more/better TRAINING DATA would fix it "
            "(the model can learn the pattern); false if it is EXTERNAL (prompt design, "
            "schema mismatch, genuinely ambiguous input, or beyond model capacity)\n"
            "5. trace_indices: the list of 'idx' values belonging to this cluster\n\n"
            "Output STRICT JSON only:\n"
            "{\"clusters\": [{\"name\": \"...\", \"count\": 0, \"root_cause\": \"...\", "
            "\"fixable\": true, \"trace_indices\": [0, 1]}], \"summary\": \"...\"}"
        ),
        messages=[{"role": "user", "content": (
            f"Total failures observed: {len(failures)} "
            f"({'showing all' if unsampled <= 0 else f'showing a sample of {len(sample)}; {unsampled} more not shown'}).\n\n"
            f"Sampled failures (each has an 'idx' field):\n{sample_text}\n\n"
            f"Cluster ONLY the sampled traces above. Assign each to exactly one cluster via "
            f"its 'idx'. Every idx 0-{len(sample) - 1} must appear in exactly one cluster's "
            f"trace_indices list. 'fixable' MUST be a JSON boolean (true/false), not a string."
        )}],
    )

    raw = response.content[0].text.strip()
    try:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        taxonomy = json.loads(match.group()) if match else {"clusters": [], "summary": raw}
    except Exception:
        taxonomy = {"clusters": [], "summary": raw}

    # Build a mapping from sample index to cluster name, then tag each trace.
    # Traces beyond the sample window (index >= 20) are left untagged (cluster=None).
    idx_to_cluster: dict[int, str] = {}
    for cluster in taxonomy.get("clusters", []):
        for idx in cluster.get("trace_indices", []):
            idx_to_cluster[idx] = cluster["name"]

    for i, t in enumerate(failures):
        t["cluster"] = idx_to_cluster.get(i)  # None for out-of-sample traces

    state["failure_taxonomy"] = taxonomy
    fixable_count = sum(c.get("count", 0) for c in taxonomy.get("clusters", []) if c.get("fixable"))
    print(f"[taxonomy] {len(taxonomy.get('clusters', []))} clusters, {fixable_count} fixable failures")
    return state
