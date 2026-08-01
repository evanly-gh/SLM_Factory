# agent/task_planner.py
"""
Autonomous task-analysis stage (Pioneer Agent cold-start, arXiv:2604.09791v1 §2.5).

Given ONLY a natural-language task description, the orchestrator LLM
(config.config.ORCHESTRATOR_MODEL) decides:
  - task_type      : classification | NER | math_reasoning | code_generation | generation | function_call | diff
  - flags          : multi_label, schema, multilingual (on task_plan dict)
  - labels         : class names / entity types / schema fields / [] for generation types
  - exa_queries    : web-search query per label/topic
  - benchmark      : a known public benchmark if one fits, else null
  - threshold_headroom : bounded rung used by agent/threshold.py to calibrate the target
This replaces hardcoded, task-specific routing so the same code handles ANY task.
"""
import json
import re
from agent.cost import tracked_anthropic_messages_create
from config.android_pool import (
    METRIC_COMPARABILITY_CAVEAT,
    format_capability_metrics,
)

_PLANNER_PROMPT = """You are the task-analysis stage of an autonomous fine-tuning agent that \
adapts a small on-device language model to a user's task.

The target model is in the {param_range} parameter range, constrained to run on Android \
hardware. This size class has known capability ceilings.

Candidate models available for this run (context for your data-size and headroom \
choices; you do NOT set an accuracy number from these):
{pool_summary}

METRIC COMPARABILITY CONTRACT: {metric_caveat}

TASK TYPE — choose the MOST SPECIFIC type that fits. There are 7 types; \
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
  Mandatory chain-of-thought; cloud fallback teacher = DeepSeek V4 Flash (thinking mode).
  Eval: final-answer exact match (NOT LLM-as-judge).
  Stop threshold: anchor to GSM8K/benchmark SOTA at this size (see STOP THRESHOLD section).
  No flags.

- "code_generation"
  Function synthesis, completion, bug-fix, SQL generation.
  Eval: execution pass@1 against unit tests.
  Stop threshold: anchor to APPS introductory pass@1 SOTA at this size (see STOP THRESHOLD section).
  No flags.

- "generation"
  Open-ended summarization, open-domain QA, dialogue, translation, instruction following.
  Use this when none of the format-bound types below fit and none of the above fit.
  Eval: LLM-as-judge [0,1].
  Stop threshold: anchor to SOTA (see STOP THRESHOLD section).
  Set "multilingual": true for translation or non-English generation tasks.

- "function_call"
  Map a natural-language request to a structured tool/API call (name + arguments).
  Examples: intent → app action, assistant function calling, API-call synthesis.
  Eval: BFCL-style AST argument match (right function name from the allowed set, all
  required args present, values equal gold with type coercion). Judge-free.
  No flags.

- "diff"
  Edit a given source text and express the change as a UNIFIED DIFF (not the full rewrite).
  Examples: prose copy-editing, grammar/style fixes, config patches.
  Eval: `git apply --check` for format validity + exact match of the applied result. Judge-free.
  No flags.

LABELS field:
- classification: list of class names (2–30). Empty list [] is invalid for classification.
- NER: list of entity type names. Empty list [] is invalid for NER.
- NER with schema: list of JSON field names (keys in the output schema).
- math_reasoning, code_generation, generation, function_call, diff: [] (empty).

ACCURACY TARGET — you do NOT set a number.

The stop threshold is calibrated by the system, from one of two sources, neither of which is
your recall of leaderboards:
  1. config/benchmark_baselines.md — a sourced, human-verified registry, used only when its
     metric name matches what this pipeline actually measures for the task type.
  2. Otherwise, the pipeline's OWN first measurement: max(zero-shot baseline, first fine-tune)
     on the held-out eval set, plus a bounded headroom.

Your job is only to (a) name the "benchmark" accurately so the registry can be searched, and
(b) choose "threshold_headroom" — how much ABOVE the first measured score the target should sit
if the registry has no usable row.

"threshold_headroom" must be exactly one of 0.02, 0.05, 0.10, 0.15. Choose it from expected
headroom, and justify it in "rationale":
  0.02 — near a known ceiling, or a noisy/small-label task where more is unreachable
  0.05 — default; ordinary fine-tuning gain over a reasonable first config
  0.10 — the base model is clearly underfitting this format and should improve a lot
  0.15 — the task is far outside the base model's behaviour (rare notation, closed vocabulary)
Do NOT invent a SOTA figure, and do NOT report a remembered leaderboard score as fact.

Do NOT default to a low, generic number — a too-low target makes the loop stop before the
model is actually good. The ranges below are only SANITY BOUNDS / fallbacks for when you
cannot estimate the SOTA; the SOTA anchor takes precedence:
- Binary classification: 0.90–0.97
- Multi-class (10–30 classes): 0.82–0.92
- Multi-label classification: 0.72–0.88
- NER (span-F1): 0.72–0.88
- NER with schema (field-F1): 0.78–0.92
- Math reasoning: anchor to GSM8K/benchmark SOTA at this size (often 0.55–0.80 for ~1.5–3B fine-tunes)
- Code generation: anchor to APPS introductory pass@1 SOTA at this size
- Generation / translation: 0.78–0.92
- Any multilingual task: subtract 5–10pp from the above

DATA SIZE — choose how many examples to build for the CURRICULUM (training) and the
held-out EVAL set. Ground this in fine-tuning sample-size research and two factors:
  1. Task complexity / distance from pretraining: obscure, niche, or specialized benchmarks
     (little public data, unlikely to be well-covered in pretraining) need MORE data to
     instill the behavior. Widely-known, popular benchmarks (heavily represented in
     pretraining) need less — the base model already has the capability, so fine-tuning
     mostly selects the format.
  2. On-device small models (this pool is sub-4B) sit in the "instillation" regime and need
     MORE data than an 8B would for the same task.
Bias UP for obscure/less-popular tasks. Give integer counts:
- "curriculum_size": total training examples to curate (gold + hard negatives).
- "eval_size": held-out evaluation examples (bigger = statistically more reliable macro-F1).
Name your popularity/complexity judgment in "rationale" (e.g. "niche biomedical NER, little
public data → large curriculum 3000"). The system clamps both to safe floors/ceiling.

EXA_QUERIES — one web-search query per label/field/topic that will retrieve REAL,
labelled-in-context example documents (not dataset landing pages). Make each query
specific enough that the top results ARE examples of that label. Good queries name
the phenomenon and the medium; weak queries just repeat the label word.

Reply with ONLY a JSON object (no prose, no code fences) with these keys:
- "task_type": one of "classification", "NER", "math_reasoning", "code_generation", "generation", "function_call", "diff"
- "task_name": short slug
- "labels": list as described above
- "multi_label": true | false (default false; classification only)
- "schema": object | null (NER structured-extraction variant only)
- "multilingual": true | false (default false)
- "exa_queries": object mapping each label/field/topic to a web-search query for REAL examples
- "benchmark": well-known public benchmark name, or null
- "threshold_headroom": one of 0.02 | 0.05 | 0.10 | 0.15 (how far above the first measured score to target)
- "curriculum_size": integer — total training examples to curate (bias up for obscure tasks)
- "eval_size": integer — held-out eval examples (bigger = more reliable metrics)
- "rationale": one sentence: task_type + flag choices + why that threshold_headroom + your data-size reasoning

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
  "threshold_headroom": 0.05,
  "curriculum_size": 1200,
  "eval_size": 800,
  "rationale": "Binary classification; 0.05 headroom since SMS spam is near-solved so a first config should already be close; popular benchmark so a moderate 1200-example curriculum suffices."
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
    "function_call",
    "diff",
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
    """One line per model with explicitly named, optional capability measurements."""
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
        source = getattr(m, "benchmark_source", None) or "not recorded"
        gsm8k_source = getattr(m, "gsm8k_source", None) or "not reported"
        rows.append(
            f"  - {m.model_id} (~{_params_b(m):.1f}B; "
            f"{format_capability_metrics(m)}; knowledge source: {source}; "
            f"GSM8K source: {gsm8k_source})"
        )
    return "\n".join(rows) if rows else "  (no base models in pool)"


def plan_task(description: str, anthropic_client=None, log=print, model_pool=None) -> dict:
    """Call the orchestrator LLM to produce a structured task plan. Returns the parsed dict."""
    if anthropic_client is None:
        import anthropic
        from config.config import ANTHROPIC_API_KEY, orchestrator_client_kwargs
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs())

    from config.config import ORCHESTRATOR_MODEL
    param_range = _param_range_label(model_pool)
    pool_summary = _pool_summary(model_pool)
    prompt = _PLANNER_PROMPT.format(
        description=description,
        param_range=param_range,
        pool_summary=pool_summary,
        metric_caveat=METRIC_COMPARABILITY_CAVEAT,
    )

    resp = tracked_anthropic_messages_create(
        anthropic_client.messages,
        stage="task_analysis",
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
    # No stop_threshold default. The planner no longer proposes one — see agent/threshold.py
    # for why a recalled SOTA number was removed. `threshold_headroom` is snapped to the
    # bounded rung set; an absent or nonsense value falls back to the middle rung.
    from agent.threshold import snap_headroom

    plan["threshold_headroom"] = snap_headroom(plan.get("threshold_headroom"))
    plan.pop("stop_threshold", None)
    plan.setdefault("task_name", "task")
    plan.setdefault("multi_label", False)
    plan.setdefault("schema", None)
    plan.setdefault("multilingual", False)
    plan.setdefault("curriculum_size", None)   # None → task_analysis falls back to config default
    plan.setdefault("eval_size", None)

    log(
        f"      [planner] task_type={plan['task_type']}  "
        f"multi_label={plan['multi_label']}  schema={'set' if plan['schema'] else 'null'}  "
        f"multilingual={plan['multilingual']}  labels={plan['labels']}  "
        f"benchmark={plan['benchmark']}  threshold_headroom={plan['threshold_headroom']}"
    )
    log(f"      [planner] data targets (pre-clamp): curriculum={plan['curriculum_size']}  "
        f"eval={plan['eval_size']}")
    log(f"      [planner] rationale: {plan.get('rationale', '')}")
    return plan
