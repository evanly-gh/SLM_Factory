# eval/harness.py
import os
from dataclasses import dataclass, field
from data.eval_set import EvalSet
from training.slm_helpers import infer_batch, infer_batch_gguf


_EVAL_OUTPUT_TOKEN_SETTINGS = {
    "classification": ("SLM_EVAL_MAX_NEW_TOKENS_CLASSIFICATION", 50),
    "NER": ("SLM_EVAL_MAX_NEW_TOKENS_NER", 512),
    "math_reasoning": ("SLM_EVAL_MAX_NEW_TOKENS_MATH", 512),
    "generation": ("SLM_EVAL_MAX_NEW_TOKENS_GENERATION", 512),
    "code_generation": ("SLM_EVAL_MAX_NEW_TOKENS_APPS", 1024),
    # Format-bound types (2026-08-01). A single tool call is short; a unified diff needs
    # room for a few hunks.
    "function_call": ("SLM_EVAL_MAX_NEW_TOKENS_FUNCTION_CALL", 256),
    "diff": ("SLM_EVAL_MAX_NEW_TOKENS_DIFF", 512),
}


def eval_output_token_reserve(
    task_type: str,
    *,
    max_seq_length: int | None = None,
) -> int:
    """Return a positive task reserve that leaves prompt context available."""
    try:
        setting, default = _EVAL_OUTPUT_TOKEN_SETTINGS[task_type]
    except KeyError as exc:
        raise ValueError(f"Unknown task_type: {task_type!r}") from exc
    raw_value = os.environ.get(setting, str(default))
    try:
        reserve = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{setting} must be a positive integer, got {raw_value!r}."
        ) from exc
    if reserve < 1:
        raise ValueError(
            f"{setting} must be a positive integer, got {raw_value!r}."
        )
    if max_seq_length is None:
        from training.slm_helpers import _inference_max_seq_length

        max_seq_length = _inference_max_seq_length()
    if reserve >= max_seq_length:
        raise ValueError(
            f"task_type={task_type} reserves {reserve} output tokens, leaving "
            f"no prompt budget inside max sequence length {max_seq_length}. "
            "Increase SLM_MAX_SEQ_LENGTH or lower the task-specific output "
            f"reserve {setting}."
        )
    return reserve


# What the comparison scalar actually measures, per task type. Used to label reports so the
# `f1` field name cannot be mistaken for a real F1 on tasks that do not compute one.
TASK_METRIC_NAMES = {
    "classification": "macro_f1",
    "NER": "span_f1",
    "math_reasoning": "exact_match",
    "code_generation": "execution_pass@1",
    "generation": "judge_mean_0_1",
    # Format-bound: the comparison scalar is content-correctness; format_valid rides
    # alongside in per_class (see eval/scorers/function_call.py, eval/scorers/diff.py).
    "function_call": "ast_arg_match",
    "diff": "apply_match",
}


@dataclass
class EvalResult:
    # `f1` is the pipeline's universal comparison scalar: it drives best_score, the
    # stagnation window, rollback, and every DAG node. Its NAME is a historical artifact —
    # only classification and NER actually compute an F1. `metric` records what the number
    # really is so reports and papers cannot misattribute it. Never rename `f1` itself;
    # checkpoints and DAG replay depend on the field name.
    f1: float
    per_class: dict
    pos_score: float
    neg_score: float
    boundary_score: float
    failures: list[dict]
    execution_diagnostics: list[dict] = field(default_factory=list)
    metric: str = "f1"


def run_eval(
    eval_set: EvalSet,
    weights_ref: str,
    base_model: str,
    task_type: str,
    quant: str | None = None,
    gguf_path: str | None = None,
) -> EvalResult:
    """Run one complete evaluation, isolated in a disposable process when enabled."""
    from training.cuda_isolation import isolation_enabled, run_isolated

    if isolation_enabled():
        from training.slm_helpers import clear_inference_cache

        payload = {
            "eval_set": eval_set,
            "weights_ref": weights_ref,
            "base_model": base_model,
            "task_type": task_type,
            "quant": quant,
            "gguf_path": gguf_path,
        }
        clear_inference_cache()
        try:
            return run_isolated("eval", payload)
        finally:
            clear_inference_cache()

    return _run_eval_local(
        eval_set, weights_ref, base_model, task_type, quant=quant, gguf_path=gguf_path,
    )


def _run_eval_local(
    eval_set: EvalSet,
    weights_ref: str,
    base_model: str,
    task_type: str,
    quant: str | None = None,
    gguf_path: str | None = None,
) -> EvalResult:
    """
    Run inference on E and compute task-type-appropriate metrics.

    When gguf_path is provided (quantized model path), inference uses
    llama-cpp-python (infer_batch_gguf) to get honest on-device accuracy.
    When gguf_path is None (base/BF16 model), uses Unsloth (infer_batch).

    Dispatches to eval/scorers/{task_type}.py for prompting, extraction, scoring.
    Scorers are inference-backend agnostic — they receive list[str] predictions.
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
    elif task_type == "function_call":
        from eval.scorers import function_call as scorer
    elif task_type == "diff":
        from eval.scorers import diff as scorer
    else:
        raise ValueError(
            f"Unknown task_type: {task_type!r}. Must be one of: classification, NER, "
            "math_reasoning, code_generation, generation, function_call, diff."
        )

    # Validate the task reserve against the configured context before loading a
    # model. This prevents the historical 512-context/512-output zero prompt
    # budget while retaining task-specific completion room.
    max_new_tokens = eval_output_token_reserve(task_type)
    prompts = scorer.build_prompts(eval_set)

    if gguf_path is not None:
        raw_outputs = infer_batch_gguf(
            prompts,
            gguf_path,
            max_new_tokens=max_new_tokens,
            base_model=base_model,
        )
    else:
        raw_outputs = infer_batch(
            prompts,
            weights_ref,
            base_model,
            max_workers=20,
            max_new_tokens=max_new_tokens,
            task_type=task_type,
        )

    predictions = scorer.extract_predictions(raw_outputs, eval_set)
    result = scorer.score(eval_set, predictions)

    return EvalResult(
        f1=result["f1"],
        per_class=result["per_class"],
        pos_score=result["slices"]["pos"],
        neg_score=result["slices"]["neg"],
        boundary_score=result["slices"]["boundary"],
        failures=result["failures"],
        execution_diagnostics=result.get("execution_diagnostics", []),
        metric=result.get("metric", TASK_METRIC_NAMES.get(task_type, "f1")),
    )
