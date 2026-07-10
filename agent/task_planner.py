# agent/task_planner.py
"""
Autonomous task-analysis stage (Pioneer Agent cold-start, arXiv:2604.09791v1 §2.5).

Given ONLY a natural-language task description, the orchestrator LLM
(config.config.ORCHESTRATOR_MODEL) decides:
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
adapts a small on-device language model to a user's task.

The target model is in the {param_range} parameter range, constrained to run on Android \
hardware. This size class has known capability ceilings.

Candidate models available for this run (calibrate stop_threshold against THESE \
specific models' published benchmark scores, not a generic size bucket):
{pool_summary}

TASK TYPE — choose the MOST SPECIFIC type that fits. There are 5 types; \
use flags to express variants within a type:

- "classification"
  Binary or multi-class argmax label prediction.
  Examples: spam detection, sentiment, intent routing (any number of classes).
  Eval: accuracy or macro-F1.
  Stop threshold: anchor to SOTA (see STOP THRESHOLD section).
  Flags: set "multi_label": true for tasks where multiple labels apply simultaneously \
  (content moderation, product tagging, symptom classification). \
  Eval then becomes per-label micro-F1; threshold 0.70–0.85.
  Set "multilingual": true if the input text is non-English or code-switching.

- "NER"
  Typed span extraction, OR schema-constrained JSON from unstructured text.
  Examples: person/org/location tagging, medical entities, \
  parsing receipts into {{vendor, total, date}}, slot filling, function call argument parsing.
  Eval: entity-level span-F1 by default; field-level F1 when "schema" is set.
  Stop threshold: anchor to SOTA (see STOP THRESHOLD section).
  Flags: set "schema": {{"field": "description", ...}} for structured JSON extraction tasks. \
  Set "multilingual": true for non-English entity extraction.

- "math_reasoning"
  Arithmetic, algebra, word problems, step-by-step derivations.
  Mandatory chain-of-thought; teacher model = DeepSeek-R1.
  Eval: final-answer exact match (NOT LLM-as-judge).
  Stop threshold: anchor to GSM8K/benchmark SOTA at this size (see STOP THRESHOLD section).
  No flags.

- "code_generation"
  Function synthesis, completion, bug-fix, SQL generation.
  Eval: execution pass@1 against unit tests.
  Stop threshold: anchor to HumanEval/MBPP pass@1 SOTA at this size (see STOP THRESHOLD section).
  No flags.

- "generation"
  Open-ended summarization, open-domain QA, dialogue, translation, instruction following.
  Use this when none of the above types fit.
  Eval: LLM-as-judge [0,1].
  Stop threshold: anchor to SOTA (see STOP THRESHOLD section).
  Set "multilingual": true for translation or non-English generation tasks.

LABELS field:
- classification: list of class names (2–30). Empty list [] is invalid for classification.
- NER: list of entity type names. Empty list [] is invalid for NER.
- NER with schema: list of JSON field names (keys in the output schema).
- math_reasoning, code_generation, generation: [] (empty).

STOP THRESHOLD — the accuracy target the fine-tuning loop must reach before it stops.
PRIMARY RULE: anchor it to the PUBLISHED STATE-OF-THE-ART for this task's benchmark at
this model-size class. In other words: "what does a well-fine-tuned model of ~{param_range}
parameters actually achieve on this benchmark today?" — set stop_threshold at or just below
that SOTA (roughly SOTA − 2 to 5 points to leave headroom for a task-specific dataset).
Use the candidate models' published gsm8k/mmlu scores above, the named "benchmark", and
your knowledge of current small-model leaderboard results to estimate that SOTA. Name the
SOTA figure you anchored to in "rationale" (e.g. "GSM8K SOTA for ~1.7B fine-tunes ≈ 0.75,
so target 0.72").

Do NOT default to a low, generic number — a too-low target makes the loop stop before the
model is actually good. The ranges below are only SANITY BOUNDS / fallbacks for when you
cannot estimate the SOTA; the SOTA anchor takes precedence:
- Binary classification: 0.90–0.97
- Multi-class (10–30 classes): 0.82–0.92
- Multi-label classification: 0.72–0.88
- NER (span-F1): 0.72–0.88
- NER with schema (field-F1): 0.78–0.92
- Math reasoning: anchor to GSM8K/benchmark SOTA at this size (often 0.55–0.80 for ~1.5–3B fine-tunes)
- Code generation: anchor to HumanEval/MBPP pass@1 SOTA at this size (often 0.55–0.80)
- Generation / translation: 0.78–0.92
- Any multilingual task: subtract 5–10pp from the above

EXA_QUERIES — one web-search query per label/field/topic that will retrieve REAL,
labelled-in-context example documents (not dataset landing pages). Make each query
specific enough that the top results ARE examples of that label. Good queries name
the phenomenon and the medium; weak queries just repeat the label word.

Reply with ONLY a JSON object (no prose, no code fences) with these keys:
- "task_type": one of "classification", "NER", "math_reasoning", "code_generation", "generation"
- "task_name": short slug
- "labels": list as described above
- "multi_label": true | false (default false; classification only)
- "schema": object | null (NER structured-extraction variant only)
- "multilingual": true | false (default false)
- "exa_queries": object mapping each label/field/topic to a web-search query for REAL examples
- "benchmark": well-known public benchmark name, or null
- "stop_threshold": float in [0,1] anchored to published SOTA for this benchmark at ~{param_range} scale
- "rationale": one sentence: task_type + flag choices + the SOTA figure stop_threshold was anchored to

EXAMPLE (task: "detect spam vs legitimate SMS on a Pixel 8"):
{{
  "task_type": "classification",
  "task_name": "sms-spam",
  "labels": ["spam", "ham"],
  "multi_label": false,
  "schema": null,
  "multilingual": false,
  "exa_queries": {{
    "spam": "examples of scam and promotional SMS text messages people received",
    "ham": "examples of normal everyday personal SMS text message conversations"
  }},
  "benchmark": "SMS Spam Collection",
  "stop_threshold": 0.95,
  "rationale": "Binary classification; 0.95 target since SMS spam is near-solved at the ~1B scale."
}}

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


def _params_b(spec) -> float:
    """Estimate parameter count (billions) from a variant's on-disk weight size.
    Bytes/param by quant: Q4_K_M 0.55, Q8_0 1.0, BF16 (quant=None) 2.0."""
    quant = getattr(spec, "quant", None)
    bytes_per_param = 0.55 if quant == "Q4_K_M" else 1.0 if quant == "Q8_0" else 2.0
    return spec.size_mb / 1000 / bytes_per_param


def _param_range_label(model_pool) -> str:
    """Return a human-readable parameter range string from the Android pool."""
    if not model_pool:
        return "0.5B–2B"
    try:
        sizes_b = sorted(_params_b(m) for m in model_pool)
        lo = f"{sizes_b[0]:.1f}B" if sizes_b[0] >= 0.1 else f"{sizes_b[0]*1000:.0f}M"
        hi = f"{sizes_b[-1]:.1f}B" if sizes_b[-1] >= 0.1 else f"{sizes_b[-1]*1000:.0f}M"
        return f"{lo}–{hi}"
    except Exception:
        return "0.5B–2B"


def _pool_summary(model_pool, max_rows: int = 12) -> str:
    """One line per distinct model with the benchmarks the planner needs to calibrate
    a realistic stop_threshold against the ACTUAL candidate pool. Deduplicates the
    three quant variants of each model (benchmarks are shared across variants)."""
    if not model_pool:
        return "  (pool unavailable — calibrate against generic size-class SOTA)"
    seen = {}
    for m in model_pool:
        # Keep one representative variant per model_id (params estimated from it).
        if m.model_id not in seen:
            seen[m.model_id] = m
    reps = sorted(seen.values(), key=_params_b)[:max_rows]
    rows = []
    for m in reps:
        rows.append(
            f"  - {m.model_id} (~{_params_b(m):.1f}B, gsm8k={m.gsm8k:.2f}, mmlu={m.mmlu:.2f})"
        )
    return "\n".join(rows) if rows else "  (no base models in pool)"


def plan_task(description: str, anthropic_client=None, log=print, model_pool=None) -> dict:
    """Call the orchestrator LLM to produce a structured task plan. Returns the parsed dict."""
    if anthropic_client is None:
        import anthropic
        from config.config import ANTHROPIC_API_KEY
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    from config.config import ORCHESTRATOR_MODEL
    param_range = _param_range_label(model_pool)
    pool_summary = _pool_summary(model_pool)
    prompt = _PLANNER_PROMPT.format(
        description=description, param_range=param_range, pool_summary=pool_summary
    )

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
