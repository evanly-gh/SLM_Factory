# agent/tools/query_traces.py
"""
Production mode tool: query judged inference traces (paper §2.6, Listing 1).

Paper: 'The query_traces tool accepts both a SQL query and an optional bash_pipeline
command, piping the query result directly to the bash command stdin in a single
invocation. This design allows the agent to filter, aggregate, and persist large
result sets on disk without loading raw rows into its context window.'

Our implementation uses a simplified filter/aggregate pipeline over JSONL files
rather than SQL, since we don't have a PostgreSQL backend in Phase 2.
"""
import json
from langchain_core.tools import tool


@tool
def query_traces(query: str, traces_path: str = "traces.jsonl") -> str:
    """Query judged inference traces from a JSONL file.

    Supported queries:
    - 'failures': return all failed traces
    - 'passing': return all passing traces
    - 'count': return counts by verdict
    - 'sample:N': return N random traces
    - 'category:NAME': return traces matching a taxonomy category
    """
    try:
        with open(traces_path) as f:
            traces = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return f"[ERROR] Traces file not found: {traces_path}"

    query = query.strip().lower()

    if query == "failures":
        results = [t for t in traces if t.get("verdict") == "fail"]
        return json.dumps(results[:50], indent=2, default=str)

    elif query == "passing":
        results = [t for t in traces if t.get("verdict") == "pass"]
        return json.dumps(results[:50], indent=2, default=str)

    elif query == "count":
        from collections import Counter
        counts = Counter(t.get("verdict", "unknown") for t in traces)
        return json.dumps(dict(counts))

    elif query.startswith("sample:"):
        import random
        n = int(query.split(":")[1])
        sample = random.sample(traces, min(n, len(traces)))
        return json.dumps(sample, indent=2, default=str)

    elif query.startswith("category:"):
        category = query.split(":", 1)[1]
        results = [t for t in traces if category in str(t).lower()]
        return json.dumps(results[:50], indent=2, default=str)

    else:
        return f"Unknown query: {query}. Supported: failures, passing, count, sample:N, category:NAME"
