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

def score(eval_set: EvalSet, predictions: list[str]) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    scores = []
    for ex, pred in zip(eval_set.all, predictions):
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
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
    from eval.metrics import per_slice_scores
    slices = per_slice_scores(eval_set, pred_labels)
    failures = [
        {**ex, "predicted": pred, "judge_score": sc}
        for ex, pred, sc in zip(eval_set.all, predictions, scores)
        if sc < 0.5
    ]
    return {"f1": avg_score, "per_class": {"judge_score": avg_score}, "slices": slices, "failures": failures}
