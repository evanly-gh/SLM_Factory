"""DialogSum scoring: multi-reference ROUGE to select, ROUGE plus BERTScore to publish.

WHY NOT THE LLM JUDGE, WHICH THIS TASK USED TO USE
    Three reasons, none of them about cost alone. A judge score is comparable to no published
    number, so it cannot be read against DialogSum's baselines or its human ceiling. It is
    non-deterministic, so a re-run moves it. And it costs money on every eval, in a loop that
    evaluates dozens of times per run. ROUGE against three references is comparable, repeatable
    and free. The judge remains available as an occasional spot-check and is still the DEFINED
    metric for `toolbench`, whose pass rate is a judged vote by construction.

WHY THREE REFERENCES CHANGE THE MEASUREMENT
    A summary has many correct forms, so single-reference ROUGE is partly a lottery about whose
    phrasing the model matched. DialogSum's test split ships three independent human summaries per
    dialogue; taking the best-matching one per example and averaging over examples is the standard
    multi-reference protocol. It is also what makes the ceiling legible: one annotator scored
    against the other two reaches ROUGE-1 53.35 / ROUGE-2 26.72 / ROUGE-L 50.84, and the best
    fine-tuned models sit near 47 ROUGE-1. So 47 is ~88% of human, not 47% of perfect.

WHY BERTSCORE IS SECONDARY AND NOT THE HEADLINE
    ROUGE counts word overlap, so it punishes a correct summary worded differently, and BERTScore
    catches exactly those. But its floor is high and model-dependent: measured here, a summary
    about an unrelated subject scores 0.854 against these references where a perfect one scores
    1.000. A metric with 85% of its range unreachable discriminates poorly on its own, and it is
    not comparable across papers unless the scoring model and rescaling match. So it is a
    diagnostic for the gap between "wrong" and "worded differently" — a real and useful
    distinction — and nothing more.

    It is also SLOW: roberta-large on CPU, so the full 500-row split at three references each is
    on the order of ten minutes. Acceptable because it runs once per run from
    `scripts/report_eval.py` and never inside the loop, which is the same reason it can afford to
    live in a separate CPU-only venv.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile

from data.eval_set import EvalSet
from eval.metrics import multi_reference_rouge

METRICS_VENV_ENV = "METRICS_VENV"
DEFAULT_METRICS_VENV = ".venv_metrics"

# Pinned with the scorer, because BERTScore's number is a property of the model that computes it.
# roberta-large layer 17 is the library's own default for English and what published BERTScore
# figures use.
BERTSCORE_MODEL = "roberta-large"
BERTSCORE_LAYERS = 17

# Continuing the conversation instead of summarizing it was B250, and the row-level `_instruction`
# is the fix. This is how the failure is DETECTED if it returns.
#
# THE COLON IS THE WHOLE TEST, and getting this wrong is a real trap: DialogSum's gold summaries
# REFER to the speakers by tag — "Ms. Dawson helps #Person1# to write a memo" — and 78% of the
# 1,500 test references do so. A bare `#Person1#` is therefore correct output, and matching on it
# scored the ORACLE at format_valid 0.15, condemning the reference summaries as continuations.
#
# What never appears in a summary is the TURN LABEL form `#Person1#:`, which is how the transcript
# marks who is speaking: 0 of 1,500 gold references contain it. So the colon separates "mentions a
# participant" from "is transcript text".
_TURN_LABEL_RE = re.compile(r"#person\d+#\s*:", re.IGNORECASE)


def build_summarization_prompt(text: str, instruction: str) -> str:
    """The ONE prompt. Used by the eval harness AND the trainer."""
    return f"{instruction}\n\n{text}"


def resolve_instruction(rows) -> str:
    """The single instruction shared by every row of the dataset.

    Resolved ONCE per dataset rather than per row, deliberately: synthetic rows are built fresh
    and carry no `_instruction`, so a per-row lookup would silently give real and generated rows
    different prompts inside one training set.
    """
    from data.loaders.dialogsum import SUMMARIZATION_INSTRUCTION

    for row in rows or []:
        if isinstance(row, dict):
            instruction = str(row.get("_instruction") or "").strip()
            if instruction:
                return instruction
    return SUMMARIZATION_INSTRUCTION


def build_prompts(eval_set: EvalSet) -> list[str]:
    instruction = resolve_instruction(eval_set.all)
    return [
        build_summarization_prompt(example.get("text", ""), instruction)
        for example in eval_set.all
    ]


def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]:
    """The summary text, with any `<reasoning>` block stripped.

    Reuses `eval.scorers.generation.split_reasoning` so a CoT-trained model's reasoning is not
    scored as part of its summary — the same reason that split exists for the judge path (B251).
    """
    from eval.scorers.generation import split_reasoning

    return [split_reasoning(raw)[1] for raw in raw_outputs]


def _references(example: dict) -> list[str]:
    """Every reference for one row: the three test summaries, or the single train/dev one."""
    references = example.get("references")
    if isinstance(references, list) and references:
        return [str(r) for r in references if str(r).strip()]
    answer = str(example.get("answer") or "").strip()
    return [answer] if answer else []


def _bertscore(predictions: list[str], references: list[list[str]]) -> dict[str, float] | None:
    """BERTScore F1, max over references then mean, from its own venv. None when unavailable.

    Isolated in a subprocess for the same reason ERRANT is: `bert-score` predates transformers 5.x
    and imports AutoModel directly, and a resolver reconciling that with the training stack does
    not fail at install time — it fails inside a training run.

    Returns None rather than raising when the venv is absent. This is a SECONDARY metric, so a
    missing one must not take down a report that has a valid headline in it; the headline is
    ROUGE, which needs nothing outside `.venv_gpu`.
    """
    root = os.environ.get(METRICS_VENV_ENV) or DEFAULT_METRICS_VENV
    python = os.path.join(root, "bin", "python")
    if not os.path.exists(python) or not predictions:
        return None

    # One (prediction, reference) pair per reference, scored flat and then maxed per example.
    # `bert_score` has no multi-reference entry point, so the grouping is done here.
    flat_predictions: list[str] = []
    flat_references: list[str] = []
    owners: list[int] = []
    for index, (prediction, refs) in enumerate(zip(predictions, references)):
        for reference in refs or []:
            flat_predictions.append(str(prediction or ""))
            flat_references.append(str(reference))
            owners.append(index)
    if not flat_predictions:
        return None

    payload = {
        "candidates": flat_predictions,
        "references": flat_references,
        "model_type": BERTSCORE_MODEL,
        "num_layers": BERTSCORE_LAYERS,
    }
    with tempfile.TemporaryDirectory(prefix="bertscore-") as tmp:
        request = os.path.join(tmp, "request.json")
        with open(request, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        code = (
            "import json,sys;import bert_score;"
            "d=json.load(open(sys.argv[1]));"
            "P,R,F=bert_score.score(d['candidates'],d['references'],lang='en',"
            "model_type=d['model_type'],num_layers=d['num_layers'],batch_size=32,verbose=False);"
            "print(json.dumps([float(x) for x in F]))"
        )
        try:
            completed = subprocess.run(
                [python, "-c", code, request],
                capture_output=True, text=True, timeout=3600,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
    if completed.returncode != 0:
        return None
    try:
        # The library writes progress and checkpoint warnings to stdout too, so take the last
        # line that parses as JSON rather than assuming the payload is alone.
        scores = None
        for line in reversed(completed.stdout.strip().splitlines()):
            try:
                scores = json.loads(line)
                break
            except ValueError:
                continue
        if not isinstance(scores, list) or len(scores) != len(owners):
            return None
    except ValueError:
        return None

    best: dict[int, float] = {}
    for owner, value in zip(owners, scores):
        best[owner] = max(best.get(owner, float("-inf")), float(value))
    if not best:
        return None
    return {
        "bertscore_f1": sum(best.values()) / len(predictions),
        "bertscore_model": BERTSCORE_MODEL,
    }


def _rouge_result(eval_set: EvalSet, predictions: list[str], metric_name: str,
                  with_bertscore: bool) -> dict:
    rows = eval_set.all
    if not rows:
        return {
            "f1": 0.0,
            "metric": metric_name,
            "per_class": {metric_name: 0.0, "format_valid": 0.0},
            "failures": [],
            "format_valid": 0.0,
        }

    references = [_references(example) for example in rows]
    cleaned = [str(prediction or "").strip() for prediction in predictions]

    # "Format" for a summarization task can only mean the model produced a summary rather than
    # something else. A reply carrying a transcript turn label is a CONTINUATION of the dialogue,
    # which is the specific B250 failure this task has already had, so it counts as unusable
    # rather than merely low-scoring.
    flags = [
        1.0 if prediction and not _looks_like_continuation(prediction) else 0.0
        for prediction in cleaned
    ]
    format_valid = sum(flags) / len(flags) if flags else 0.0

    # An unusable reply SCORES ZERO rather than earning partial credit, which is the same
    # substitution the span scorers make when they replace unparseable output with `[]`.
    #
    # It matters more than it looks. A continuation is made of the transcript's own words, so it
    # shares vocabulary with the reference by construction — measured at ROUGE-L 0.14 on the
    # contract fixture. Left in, a model that merely echoes the conversation collects a free floor
    # of over a tenth of the metric's range, which inflates every zero-shot baseline on precisely
    # the task where contamination already inflates them.
    scoreable = [
        prediction if flag else "" for prediction, flag in zip(cleaned, flags)
    ]
    rouge = multi_reference_rouge(scoreable, references)

    per_class = {
        "rouge_1": rouge["rouge1"],
        "rouge_2": rouge["rouge2"],
        "rouge_l": rouge["rougeL"],
        # Published for calibration, so a 47 is read against 53 rather than against 100.
        "human_ceiling_rouge_1": 53.35 / 100,
        "human_ceiling_rouge_2": 26.72 / 100,
        "human_ceiling_rouge_l": 50.84 / 100,
        "references_per_row": (
            sum(len(refs) for refs in references) / len(references) if references else 0.0
        ),
        "format_valid": format_valid,
    }
    if with_bertscore:
        bert = _bertscore(scoreable, references)
        if bert:
            per_class.update(bert)
        else:
            per_class["bertscore_unavailable"] = (
                "bert-score venv not found or failed; ROUGE (the headline) is unaffected. "
                "Build it with `bash scripts/setup_metric_envs.sh`."
            )

    headline = rouge["rougeL"] if metric_name == "rouge_l" else rouge["rougeL"]
    per_class[metric_name] = headline
    failures = [
        {**example, "predicted": prediction,
         "error_type": failure_category_of({**example, "predicted": prediction})}
        for example, prediction in zip(rows, cleaned)
        if _row_rouge_l(prediction, _references(example)) < 0.3
    ]
    return {
        "f1": headline,
        "metric": metric_name,
        "per_class": per_class,
        "failures": failures,
        "format_valid": format_valid,
    }


def _row_rouge_l(prediction: str, references: list[str]) -> float:
    """ROUGE-L for one row, used only to decide whether it is a failure worth recording.

    Per-row ROUGE is NOT a measurement and is never reported: human agreement with these metrics
    is 0.3-0.4 at the sentence level against above 0.9 at the system level. It is used here purely
    as a threshold for which rows to hand the orchestrator as examples.
    """
    if not prediction or not references:
        return 0.0
    return multi_reference_rouge([prediction], [references], rouge_types=("rougeL",))["rougeL"]


def _looks_like_continuation(prediction: str) -> bool:
    """True when the reply contains a transcript TURN LABEL rather than a summary.

    See `_TURN_LABEL_RE`: mentioning `#Person1#` is what a correct summary does, and matching on
    that flagged every gold reference.
    """
    return bool(_TURN_LABEL_RE.search(prediction))


def score(eval_set: EvalSet, predictions: list[str]) -> dict:
    """SELECTION scoring: multi-reference ROUGE-L. Cheap, deterministic, runs every iteration."""
    return _rouge_result(eval_set, predictions, "rouge_l", with_bertscore=False)


def score_report(eval_set: EvalSet, predictions: list[str]) -> dict:
    """REPORT scoring: ROUGE-1/2/L plus BERTScore. Runs once.

    The comparison scalar stays ROUGE-L so the reported headline is on the same scale as the
    selection metric; ROUGE-1, ROUGE-2 and BERTScore ride alongside in `per_class`, which is
    where the writeup decomposes them.
    """
    return _rouge_result(eval_set, predictions, "rouge_1_2_l_bertscore", with_bertscore=True)


def failure_category_of(failure: dict) -> str:
    """Why this summary scored poorly, in categories that suggest different fixes.

    `continued_the_conversation` is named because it is the failure this task has actually had:
    with no row-level instruction the model treated the transcript as something to reply to
    (B250). If it ever returns, it must be visible as a prompt problem rather than averaged into
    generic low overlap.
    """
    predicted = str(failure.get("predicted") or "").strip()
    if not predicted:
        return "empty_output"
    if _looks_like_continuation(predicted):
        return "continued_the_conversation"
    references = _references(failure)
    # A summary far longer than every reference is usually the model copying the transcript
    # rather than abstracting it, which is the characteristic failure on a corpus whose
    # conversations are long and whose summaries are short.
    longest = max((len(r.split()) for r in references), default=0)
    if longest and len(predicted.split()) > 3 * longest:
        return "copied_the_transcript"
    return "low_overlap"
