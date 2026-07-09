# agent/task_planner.py
"""
Autonomous task-analysis stage (Pioneer Agent cold-start, arXiv:2604.09791v1 §2.5).

Given ONLY a natural-language task description, the orchestrator LLM (Claude Sonnet)
decides:
  - task_type      : classification | NER | math_reasoning | code_generation | generation
  - flags          : multi_label, schema, multilingual (on task_plan dict)
  - labels         : class names / entity types / schema fields / [] for generation types
  - exa_queries    : web-search query per label/topic
  - benchmark      : a known public benchmark if one fits, else null
  - stop_threshold : calibrated target on the held-out eval set
This replaces hardcoded, task-specific routing so the same code handles ANY task.
"""
import json
import re

_PLANNER_PROMPT = """You are the task-analysis stage of an autonomous fine-tuning agent that \
adapts a small on-device language model to a user's task (Pioneer Agent cold-start).

The target model is in the {param_range} parameter range, constrained to run on Android \
hardware. This size class has known capability ceilings.

TASK TYPE — choose the MOST SPECIFIC type that fits. There are 5 types; \
use flags to express variants within a type:

- "classification"
  Binary or multi-class argmax label prediction.
  Examples: spam detection, sentiment, intent routing (any number of classes).
  Eval: accuracy or macro-F1.
  Stop threshold: 0.90–0.96 for binary; 0.80–0.90 for 10–30 classes.
  Flags: set "multi_label": true for tasks where multiple labels apply simultaneously \
  (content moderation, product tagging, symptom classification). \
  Eval then becomes per-label micro-F1; threshold 0.70–0.85.
  Set "multilingual": true if the input text is non-English or code-switching.

- "NER"
  Typed span extraction, OR schema-constrained JSON from unstructured text.
  Examples: person/org/location tagging, medical entities, \
  parsing receipts into {vendor, total, date}, slot filling, function call argument parsing.
  Eval: entity-level span-F1 by default; field-level F1 when "schema" is set.
  Stop threshold: 0.70–0.85.
  Flags: set "schema": {"field": "description", ...} for structured JSON extraction tasks. \
  Set "multilingual": true for non-English entity extraction.

- "math_reasoning"
  Arithmetic, algebra, word problems, step-by-step derivations.
  Mandatory chain-of-thought; teacher model = DeepSeek-R1.
  Eval: final-answer exact match (NOT LLM-as-judge).
  Stop threshold: 0.40–0.65 at 0.5–2B scale (GSM8K tops out ~75% at 1.7B after fine-tuning).
  No flags.

- "code_generation"
  Function synthesis, completion, bug-fix, SQL generation.
  Eval: execution pass@1 against unit tests.
  Stop threshold: 0.55–0.80 depending on problem complexity.
  No flags.

- "generation"
  Open-ended summarization, open-domain QA, dialogue, translation, instruction following.
  Use this when none of the above types fit.
  Eval: LLM-as-judge [0,1].
  Stop threshold: 0.75–0.90.
  Set "multilingual": true for translation or non-English generation tasks.

LABELS field:
- classification: list of class names (2–30). Empty list [] is invalid for classification.
- NER: list of entity type names. Empty list [] is invalid for NER.
- NER with schema: list of JSON field names (keys in the output schema).
- math_reasoning, code_generation, generation: [] (empty).

STOP THRESHOLD — calibrate relative to {param_range}-scale SOTA:
- Binary classification: 0.90–0.96
- Multi-class (10–30 classes): 0.80–0.90
- Multi-label classification: 0.70–0.85
- NER (span-F1): 0.70–0.85
- NER with schema (field-F1): 0.75–0.90
- Math reasoning (0.5–2B): 0.40–0.65
- Code generation: 0.55–0.80
- Generation / translation: 0.75–0.90
- Any multilingual task: subtract 5–10pp from the above range

Reply with ONLY a JSON object (no prose, no code fences) with these keys:
- "task_type": one of "classification", "NER", "math_reasoning", "code_generation", "generation"
- "task_name": short slug
- "labels": list as described above
- "multi_label": true | false (default false; classification only)
- "schema": object | null (NER structured-extraction variant only)
- "multilingual": true | false (default false)
- "exa_queries": object mapping each label/field/topic to a web-search query for REAL examples
- "benchmark": well-known public benchmark name, or null
- "stop_threshold": float in [0,1] calibrated as above
- "rationale": one sentence: task_type + flag choices + how stop_threshold was calibrated

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


_VALID = {
    "classification",
    "NER",
    "math_reasoning",
    "code_generation",
    "generation",
}


def _param_range_label(model_pool) -> str:
    """Return a human-readable parameter range string from the Android pool."""
    if not model_pool:
        return "0.5B–2B"
    try:
        sizes_b = sorted(
            m.int4_size_mb * 2 / 1000  # params_b: Q4_K_M ~0.5 bytes/param → MB × 2 / 1000 ≈ billions
            for m in model_pool
            if m.quant is None  # use base models only to avoid double-counting siblings
        )
        lo = f"{sizes_b[0]:.1f}B" if sizes_b[0] >= 0.1 else f"{sizes_b[0]*1000:.0f}M"
        hi = f"{sizes_b[-1]:.1f}B" if sizes_b[-1] >= 0.1 else f"{sizes_b[-1]*1000:.0f}M"
        return f"{lo}–{hi}"
    except Exception:
        return "0.5B–2B"


def plan_task(description: str, anthropic_client=None, log=print, model_pool=None) -> dict:
    """Call Claude Sonnet to produce a structured task plan. Returns the parsed dict."""
    if anthropic_client is None:
        import anthropic
        from config.config import ANTHROPIC_API_KEY
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    from config.config import ORCHESTRATOR_MODEL
    param_range = _param_range_label(model_pool)
    prompt = _PLANNER_PROMPT.format(description=description, param_range=param_range)

    resp = anthropic_client.messages.create(
        model=ORCHESTRATOR_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    plan = _extract_json(resp.content[0].text)

    if plan.get("task_type") not in _VALID:
        raise ValueError(f"Planner returned invalid task_type: {plan.get('task_type')!r}")

    plan.setdefault("labels", [])
    plan.setdefault("exa_queries", {})
    plan.setdefault("benchmark", None)
    plan.setdefault("stop_threshold", 0.96)
    plan.setdefault("task_name", "task")
    plan.setdefault("multi_label", False)
    plan.setdefault("schema", None)
    plan.setdefault("multilingual", False)

    log(
        f"      [planner] task_type={plan['task_type']}  "
        f"multi_label={plan['multi_label']}  schema={'set' if plan['schema'] else 'null'}  "
        f"multilingual={plan['multilingual']}  labels={plan['labels']}  "
        f"benchmark={plan['benchmark']}  stop_threshold={plan['stop_threshold']}"
    )
    log(f"      [planner] rationale: {plan.get('rationale', '')}")
    return plan
