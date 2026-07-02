# eval/harness.py
from dataclasses import dataclass
from data.eval_set import EvalSet

from training.slm_helpers import infer_batch

@dataclass
class EvalResult:
    f1: float
    per_class: dict
    pos_score: float
    neg_score: float
    boundary_score: float
    failures: list[dict]

def run_eval(
    eval_set: EvalSet,
    weights_ref: str,
    base_model: str,
    task_type: str,
) -> EvalResult:
    """
    Run inference on E and compute task-type-appropriate metrics.
    Dispatches to eval/scorers/{task_type}.py for prompting, extraction, and scoring.
    """
    # Dispatch to the appropriate scorer module.
    # classification family: argmax label prediction (binary, multi-class).
    #   multi_label flag on eval_set switches scorer to per-label threshold mode.
    # NER family: span extraction with entity F1.
    # generation family: math_reasoning → exact match; code_generation → pass@1;
    #   generation → LLM-as-judge. The generation scorer inspects task_type internally.
    if task_type == "classification":
        from eval.scorers import classification as scorer
    elif task_type == "NER":
        from eval.scorers import ner as scorer
    elif task_type in ("math_reasoning", "code_generation", "generation"):
        from eval.scorers import generation as scorer
    else:
        raise ValueError(
            f"Unknown task_type: {task_type!r}. "
            f"Must be one of: classification, NER, math_reasoning, code_generation, generation."
        )

    prompts = scorer.build_prompts(eval_set)
    raw_outputs = infer_batch(prompts, weights_ref, base_model, max_workers=20)
    predictions = scorer.extract_predictions(raw_outputs, eval_set)
    result = scorer.score(eval_set, predictions)

    return EvalResult(
        f1=result["f1"],
        per_class=result["per_class"],
        pos_score=result["slices"]["pos"],
        neg_score=result["slices"]["neg"],
        boundary_score=result["slices"]["boundary"],
        failures=result["failures"],
    )
