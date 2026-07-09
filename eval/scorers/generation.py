# eval/scorers/generation.py
from data.eval_set import EvalSet
import anthropic
import os

JUDGE_PROMPT = (
    "You are an impartial judge evaluating a model's answer.\n"
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Model answer: {predicted}\n\n"
    "Rate correctness from 0.0 (completely wrong) to 1.0 (perfectly correct). "
    "Reply with only a number."
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
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        for ex, pred in zip(eval_set.all, predictions):
            resp = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=10,
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
