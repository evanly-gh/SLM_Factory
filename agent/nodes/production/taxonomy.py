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

    sample = failures[:50]
    sample_text = json.dumps(sample[:20], indent=2, default=str)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=ORCHESTRATOR_MODEL,
        max_tokens=1024,
        system=(
            "You are analyzing failure patterns in a deployed model's inference logs. "
            "Cluster the failures into categories. For each category, determine:\n"
            "1. Category name\n"
            "2. Count of failures in this category\n"
            "3. Root cause description\n"
            "4. Fixability: 'fixable' (can be addressed by training) or 'external' "
            "(prompt design, schema mismatch, ambiguous input)\n\n"
            "Output JSON: {\"clusters\": [{\"name\": ..., \"count\": ..., "
            "\"root_cause\": ..., \"fixable\": true/false}], \"summary\": \"...\"}"
        ),
        messages=[{"role": "user", "content": (
            f"Total failures: {len(failures)}\n\n"
            f"Sample failures (first {len(sample)}):\n{sample_text}\n\n"
            f"Classify these into failure categories."
        )}],
    )

    raw = response.content[0].text.strip()
    try:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        taxonomy = json.loads(match.group()) if match else {"clusters": [], "summary": raw}
    except Exception:
        taxonomy = {"clusters": [], "summary": raw}

    state["failure_taxonomy"] = taxonomy
    fixable_count = sum(c.get("count", 0) for c in taxonomy.get("clusters", []) if c.get("fixable"))
    print(f"[taxonomy] {len(taxonomy.get('clusters', []))} clusters, {fixable_count} fixable failures")
    return state
