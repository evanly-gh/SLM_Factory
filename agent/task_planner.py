# agent/task_planner.py
"""
Autonomous task-analysis stage (Pioneer Agent cold-start, arXiv:2604.09791v1 §2.5).

Given ONLY a natural-language task description, the orchestrator LLM (Claude Sonnet)
decides:
  - task_type        : classification | NER | generation
  - labels           : class names (classification) / entity types (NER) / [] (generation)
  - exa_queries      : web-search query per label/topic, used to autonomously acquire data
  - benchmark        : a known public benchmark if one fits, else null
  - stop_threshold   : calibrated target on the held-out eval set
This replaces hardcoded, task-specific routing so the same code handles ANY task.
"""
import json
import re

PLANNER_MODEL = "claude-sonnet-4-6"

_PLANNER_PROMPT = """You are the task-analysis stage of an autonomous fine-tuning agent that \
adapts a small on-device language model to a user's task (Pioneer Agent cold-start).

Given the user's task description, decide how to bootstrap the model from scratch. \
Reply with ONLY a JSON object (no prose, no code fences) with these keys:

- "task_type": one of "classification", "NER", "generation".
- "task_name": a short slug for the task.
- "labels": list of class names (for classification), entity types (for NER), or [] (for generation). \
For classification use 2-6 mutually-exclusive, clearly-separable classes.
- "exa_queries": object mapping each label (or, for generation/NER, a few topic names) to a \
concrete web-search query that will retrieve REAL example texts of that class from the open web.
- "benchmark": the name of a well-known public benchmark that matches this task, or null.
- "stop_threshold": a float in [0,1] — the target metric on the held-out eval set \
(default 0.96; lower it if the task is inherently hard or ambiguous).
- "rationale": one sentence explaining the choices.

User task description:
\"\"\"{description}\"\"\"
"""


def _extract_json(text: str) -> dict:
    """Parse the first JSON object out of an LLM reply, tolerating stray prose/fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"Planner did not return JSON: {text[:200]!r}")
        return json.loads(m.group())


_VALID = {"classification", "NER", "generation"}


def plan_task(description: str, anthropic_client=None, log=print) -> dict:
    """Call Claude Sonnet to produce a structured task plan. Returns the parsed dict."""
    if anthropic_client is None:
        import anthropic
        from config import ANTHROPIC_API_KEY
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    resp = anthropic_client.messages.create(
        model=PLANNER_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": _PLANNER_PROMPT.format(description=description)}],
    )
    plan = _extract_json(resp.content[0].text)

    # Validate / normalize
    if plan.get("task_type") not in _VALID:
        raise ValueError(f"Planner returned invalid task_type: {plan.get('task_type')!r}")
    plan.setdefault("labels", [])
    plan.setdefault("exa_queries", {})
    plan.setdefault("benchmark", None)
    plan.setdefault("stop_threshold", 0.96)
    plan.setdefault("task_name", "task")

    log(f"      [planner] task_type={plan['task_type']}  labels={plan['labels']}  "
        f"benchmark={plan['benchmark']}  stop_threshold={plan['stop_threshold']}")
    log(f"      [planner] rationale: {plan.get('rationale','')}")
    return plan
