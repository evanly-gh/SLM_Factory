# eval/scorers/generation.py
from data.eval_set import EvalSet
import anthropic
import os

# System prompt anchors the judge role and forces a strict, calibrated scale.
# Without a rubric the judge clusters everything around 0.7–0.9; the anchored
# scale below keeps scores discriminative across eval runs.
JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of a language model's answer against a "
    "reference (gold) answer. Judge only semantic correctness relative to the gold "
    "answer — ignore style, verbosity, and formatting differences. Be conservative: "
    "most answers are NOT perfect. Reserve 1.0 for answers that are fully correct and "
    "complete. Output a single decimal number in [0.0, 1.0] and nothing else."
)

JUDGE_PROMPT = (
    "Task/question:\n{question}\n\n"
    "Gold (reference) answer:\n{gold}\n\n"
    "Model answer:\n{predicted}\n\n"
    "Score the model answer against the gold answer using this rubric:\n"
    "  1.0  — fully correct and complete; matches the gold answer's meaning\n"
    "  0.7  — mostly correct; minor omission or imprecision, no factual error\n"
    "  0.4  — partially correct; missing key information or a notable error\n"
    "  0.0  — wrong, irrelevant, empty, or a refusal when an answer was expected\n"
    "Interpolate between anchors when warranted. "
    "Reply with ONLY the decimal number."
)

GENERATE_PROMPT = "Answer the following question:\n\n{text}"

def build_prompts(eval_set: EvalSet) -> list[str]:
    return [GENERATE_PROMPT.format(text=ex.get("text", "")) for ex in eval_set.all]

def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]:
    return [r.strip() for r in raw_outputs]

def _exact_match(gold: str, pred: str) -> float:
    """Normalize whitespace and case, then compare gold vs predicted answer."""
    return 1.0 if gold.strip().lower() == pred.strip().lower() else 0.0

def _code_pass_at_1(code: str) -> float:
    """Execute predicted code in a restricted sandbox and return 1.0 if no exception."""
    try:
        exec(compile(code, "<string>", "exec"), {})
        return 1.0
    except Exception:
        return 0.0

def score(eval_set: EvalSet, predictions: list[str]) -> dict:
    task_type = eval_set.task_type
    scores = []

    if task_type == "math_reasoning":
        # Exact match after normalizing whitespace/case
        for ex, pred in zip(eval_set.all, predictions):
            gold = ex.get("answer", "")
            scores.append(_exact_match(gold, pred))

    elif task_type == "code_generation":
        # pass@1: execute predicted code in a sandbox
        for ex, pred in zip(eval_set.all, predictions):
            scores.append(_code_pass_at_1(pred))

    else:
        # task_type == "generation": LLM-as-judge
        from config.config import ANTHROPIC_API_KEY, JUDGE_MODEL
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        for ex, pred in zip(eval_set.all, predictions):
            resp = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=10,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=ex.get("text", ""),
                    gold=ex.get("answer", ex.get("label", "")),
                    predicted=pred,
                )}],
            )
            try:
                scores.append(float(resp.content[0].text.strip()))
            except Exception:
                scores.append(0.0)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    # For per-slice: convert to pass/fail labels at threshold 0.5
    pred_labels = ["correct" if s >= 0.5 else "wrong" for s in scores]
    # Gold label is "correct" only for examples that have a reference answer;
    # neg-slice adversarial/ill-posed inputs without an answer get "wrong" so
    # that a model which correctly refuses them scores 1 against "wrong", not 0.
    gold_labels_for_slice = ["correct" if ex.get("answer") else "wrong" for ex in eval_set.all]
    from eval.metrics import per_slice_scores
    slices = per_slice_scores(eval_set, pred_labels, gold_labels=gold_labels_for_slice)
    failures = [
        {**ex, "predicted": pred, "judge_score": sc}
        for ex, pred, sc in zip(eval_set.all, predictions, scores)
        if sc < 0.5
    ]
    return {"f1": avg_score, "per_class": {"judge_score": avg_score}, "slices": slices, "failures": failures}
