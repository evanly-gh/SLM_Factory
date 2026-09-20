"""GEC scoring through the real ERRANT CLI, in its own interpreter.

WHY A SUBPROCESS AND NOT AN IMPORT
    ERRANT requires `spacy<4` and a pinned `en_core_web_sm`, and `.venv_gpu` is the training stack
    (unsloth, transformers 5.5.0, numpy 2.4.6). A resolver that downgrades numpy or pydantic
    underneath Unsloth does not fail at install time — it fails inside a training run. ERRANT's
    native interface is text files, so it costs nothing to keep it behind a process boundary.
    `scripts/setup_metric_envs.sh` builds the venv; `$ERRANT_VENV` locates it.

WHY THE REAL SCORER AND NOT A REIMPLEMENTATION
    ERRANT's number IS its implementation: the edit extraction aligns source against hypothesis
    using en_core_web_sm's POS tags and lemmas, and the 25 error types are assigned by rules over
    those tags. A hand-rolled edit-based F0.5 would be a different metric wearing the same name,
    comparable to none of the published figures — which is the only reason to use F0.5 at all.
    The spaCy MODEL version is therefore pinned too, and recorded in the score.

PRECISION AND RECALL ARE ALWAYS REPORTED SEPARATELY
    Non-negotiable for this task. Fine-tuned small models under-correct (high precision, low
    recall) while large untuned models over-correct (low precision, high recall), and both land in
    the same middling F0.5. Without the pair you cannot tell which failure you have, and the two
    want opposite interventions.

THE ONE-LINE-PER-SENTENCE CONTRACT
    ERRANT reads one tokenized sentence per line and aligns file lines positionally. A model that
    merges two sentences, splits one, or emits a preamble shifts every subsequent line and scores
    near zero for reasons that have nothing to do with grammar. Rather than constrained decoding —
    which this repo weighed and rejected for classification (B286) — the extractor takes the first
    non-empty line and records a multi-line reply as a FORMAT failure, so `format_valid` carries
    it and the collapse is diagnosable instead of mysterious.

THE ORACLE CEILING IS ABOUT 0.89, NOT 1.00 — CALIBRATE AGAINST THIS, NOT AGAINST 100
    Measured on 120 dev sentences (2026-09-06): feeding the GOLD correction back in as the
    hypothesis scores F0.5 0.8934, with P 0.8925 / R 0.8967 and 23 false positives and 22 false
    negatives out of 213 gold edits.

    That gap is structural and is not a bug. The reference edits are the annotator's own, taken
    from the shipped M2 file; the hypothesis edits have to be DERIVED from the corrected string by
    `errant_parallel`'s alignment rules. Where a human merged two adjacent corrections into one
    edit and the rules split them, or vice versa, the edit lists differ even though the sentences
    are identical. Regenerating the reference side the same way would make the oracle score 1.00
    and would be the wrong choice: it would replace the corpus's gold annotation with our scorer's
    opinion of it, and change the number every published figure is compared against.

    So a system scoring 0.75 here is at ~84% of the achievable maximum, not 75%. Any claim that
    reads our F0.5 as a percentage of perfect is wrong by about eleven points.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

from data.eval_set import EvalSet

GEC_PROMPT = (
    "Correct the grammatical errors in the sentence below. Change as little as possible: fix only "
    "what is wrong and leave everything else exactly as written. If the sentence is already "
    "correct, repeat it unchanged.\n"
    "Reply with only the corrected sentence, on a single line.\n\n"
    "Sentence: {text}"
)

ERRANT_VENV_ENV = "ERRANT_VENV"
DEFAULT_ERRANT_VENV = ".venv_errant"

# `errant_compare -cat 2` groups by the 25 error TYPES (VERB:TENSE, DET, SPELL). `-cat 1` would
# group by operation (M/R/U) and `-cat 3` by both. The types are the published breakdown and the
# task's second episode axis.
_CATEGORY_LEVEL = "2"

# The summary table errant_compare prints, e.g. "1\t0\t1\t1.0\t0.5\t0.8333".
_SUMMARY_ROW = re.compile(
    r"^(\d+)\t(\d+)\t(\d+)\t([\d.]+)\t([\d.]+)\t([\d.]+)\s*$", re.MULTILINE
)


class ErrantUnavailable(RuntimeError):
    """ERRANT could not be run. Raised rather than scored as zero, on purpose.

    A missing scorer is not a bad model. Scoring zero would look exactly like a catastrophic
    regression, and the loop would spend hours of L40S time rebuilding data to chase it — the same
    reasoning behind `TaskSpec.needs_judge` failing loudly on a judge outage.
    """


def build_prompts(eval_set: EvalSet) -> list[str]:
    return [GEC_PROMPT.format(text=example.get("text", "")) for example in eval_set.all]


def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str | None]:
    """One single-line correction per reply, or None when the reply is not one line of text.

    A reply whose extra lines are blank or a fenced code block is still one sentence, and is
    accepted. A reply with genuine extra prose is not: ERRANT would align it against the next
    source sentence and corrupt every score after it.
    """
    out: list[str | None] = []
    for raw in raw_outputs:
        lines = [line.strip() for line in str(raw or "").splitlines()]
        content = [line.strip("`").strip() for line in lines if line.strip().strip("`")]
        if not content:
            out.append(None)
            continue
        if len(content) > 1:
            # More than one line of substance. Recorded as unusable rather than silently
            # truncated to the first: the model did not answer the question that was asked, and
            # `format_valid` is where that belongs.
            out.append(None)
            continue
        out.append(content[0])
    return out


def errant_bin(name: str) -> str:
    """Absolute path to an ERRANT CLI tool, or a clear failure.

    `$ERRANT_VENV` rather than a literal path so the `_l40s` and `_cse` launchers for this task
    stay byte-identical apart from their SBATCH headers, which is an invariant the SLURM tests
    enforce.
    """
    root = os.environ.get(ERRANT_VENV_ENV) or DEFAULT_ERRANT_VENV
    candidate = os.path.join(root, "bin", name)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise ErrantUnavailable(
        f"{name} not found in {root}/bin or on PATH. Build the scorer venv with "
        f"`bash scripts/setup_metric_envs.sh` and export {ERRANT_VENV_ENV}=<repo>/.venv_errant."
    )


def _run(argv: list[str]) -> str:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError as exc:
        raise ErrantUnavailable(f"could not execute {argv[0]}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ErrantUnavailable(f"{argv[0]} timed out after an hour") from exc
    if completed.returncode != 0:
        raise ErrantUnavailable(
            f"{os.path.basename(argv[0])} exited {completed.returncode}: "
            f"{(completed.stderr or completed.stdout)[-800:]}"
        )
    return completed.stdout


def _parse_summary(output: str) -> dict[str, float]:
    """TP/FP/FN/P/R/F0.5 out of errant_compare's summary table."""
    match = None
    for match in _SUMMARY_ROW.finditer(output):
        pass  # the LAST such row is the overall summary; -cat also prints per-category rows
    if match is None:
        raise ErrantUnavailable(
            f"could not find a summary row in errant_compare output:\n{output[-800:]}"
        )
    tp, fp, fn, precision, recall, f05 = match.groups()
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "precision": float(precision), "recall": float(recall), "f0_5": float(f05),
    }


def _parse_categories(output: str) -> dict[str, float]:
    """Per-error-type F0.5 out of the `-cat 2` table. Free from ERRANT, and the episode axis."""
    per_type: dict[str, float] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 7 or fields[0] in ("Category", "TP"):
            continue
        name, _tp, _fp, _fn, _p, _r, f05 = fields
        try:
            per_type[f"f0_5_{name}"] = float(f05)
        except ValueError:
            continue
    return per_type


_VERSION_CACHE: dict[str, str] | None = None


def _scorer_versions() -> dict[str, str]:
    """errant, spacy and en_core_web_sm versions, recorded with the score.

    ERRANT's edit typing runs off the spaCy model's POS tags, so a model version bump moves F0.5
    with no change to the system under test. Recording the versions is what makes two of our own
    numbers comparable, and what makes an unexplained shift attributable.

    Cached for the process: it costs a subprocess and a spaCy import, and the answer cannot change
    while the interpreter is alive.
    """
    global _VERSION_CACHE
    if _VERSION_CACHE is not None:
        return _VERSION_CACHE
    _VERSION_CACHE = _read_scorer_versions()
    return _VERSION_CACHE


def _read_scorer_versions() -> dict[str, str]:
    python = errant_bin("python")
    code = (
        "import errant, spacy, en_core_web_sm; "
        "print(errant.__version__, spacy.__version__, en_core_web_sm.__version__)"
    )
    try:
        parts = _run([python, "-c", code]).split()
    except ErrantUnavailable:
        return {}
    if len(parts) != 3:
        return {}
    return {"errant": parts[0], "spacy": parts[1], "en_core_web_sm": parts[2]}


def _empty(metric_name: str) -> dict:
    return {
        "f1": 0.0,
        "metric": metric_name,
        "per_class": {metric_name: 0.0, "precision": 0.0, "recall": 0.0, "format_valid": 0.0},
        "failures": [],
        "format_valid": 0.0,
    }


def _score(eval_set: EvalSet, predictions: list[str | None], metric_name: str) -> dict:
    rows = eval_set.all
    if not rows:
        return _empty(metric_name)

    readable = sum(1 for prediction in predictions if prediction is not None)
    format_valid = readable / len(predictions) if predictions else 0.0

    # An unusable reply becomes the SOURCE sentence, i.e. "corrected nothing". That is the honest
    # substitution: it yields no true positives and no false positives, so a format failure costs
    # recall without being rewarded on precision. Dropping the row instead would shorten the file
    # and misalign every line after it, and inserting a blank line would make ERRANT read the
    # whole sentence as one enormous deletion — a false positive the model never proposed.
    hypotheses = [
        prediction if prediction is not None else str(row.get("text", ""))
        for row, prediction in zip(rows, predictions)
    ]

    with tempfile.TemporaryDirectory(prefix="gec-errant-") as tmp:
        source_path = os.path.join(tmp, "src.txt")
        hypothesis_path = os.path.join(tmp, "hyp.txt")
        gold_path = os.path.join(tmp, "gold.m2")
        hypothesis_m2 = os.path.join(tmp, "hyp.m2")

        # The SOURCE goes out untouched: it must stay byte-identical to the `S` lines of the gold
        # M2, which is what the reference edit offsets are relative to.
        _write_lines(source_path, [str(row.get("text", "")) for row in rows])
        # The HYPOTHESIS is re-tokenized to the corpus convention first. See `retokenize` — without
        # this a perfect answer scores 0.31 instead of 0.87, purely on spacing.
        _write_lines(hypothesis_path, retokenize(hypotheses))
        # The gold reference is the corpus's OWN M2 blocks, reassembled for this subset — never
        # regenerated from the target string, which would re-segment the annotator's edits by rule
        # and quietly change the reference the score is computed against.
        with open(gold_path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(str(row.get("m2", "")).rstrip("\n") + "\n\n")

        _run([
            errant_bin("errant_parallel"),
            "-orig", source_path, "-cor", hypothesis_path, "-out", hypothesis_m2,
        ])
        summary = _parse_summary(
            _run([errant_bin("errant_compare"), "-hyp", hypothesis_m2, "-ref", gold_path])
        )
        categories = _parse_categories(
            _run([
                errant_bin("errant_compare"), "-hyp", hypothesis_m2, "-ref", gold_path,
                "-cat", _CATEGORY_LEVEL,
            ])
        )

    failures = [
        {**row, "predicted": prediction,
         "error_type": failure_category_of({**row, "predicted": prediction})}
        for row, prediction in zip(rows, predictions)
        if prediction is None or prediction.strip() != str(row.get("answer", "")).strip()
    ]
    return {
        "f1": summary["f0_5"],
        "metric": metric_name,
        "per_class": {
            metric_name: summary["f0_5"],
            # ALWAYS beside the F0.5. See the module docstring.
            "precision": summary["precision"],
            "recall": summary["recall"],
            "tp": summary["tp"],
            "fp": summary["fp"],
            "fn": summary["fn"],
            # Single-reference, so measurable recall is capped and the number is not comparable to
            # an F0.5 computed against a multi-reference set. Carried in the result so a report
            # cannot lose it.
            "references_per_sentence": 1,
            **{f"scorer_{k}": v for k, v in _scorer_versions().items()},
            **categories,
            "format_valid": format_valid,
        },
        "failures": failures,
        "format_valid": format_valid,
    }


def _write_lines(path: str, lines: list[str]) -> None:
    """One line per row, blanks replaced. See `_score` on why a blank line is not acceptable."""
    with open(path, "w", encoding="utf-8") as handle:
        for line in lines:
            collapsed = " ".join(str(line or "").split())
            handle.write((collapsed or ".") + "\n")


# One spaCy call for the whole file, inside the ERRANT venv so the tokenizer is BY CONSTRUCTION
# the same `en_core_web_sm` that ERRANT's own alignment uses. Using a different tokenizer here —
# even a good one — would reintroduce the mismatch this exists to remove.
_TOKENIZE_SCRIPT = (
    "import sys, spacy;"
    "nlp = spacy.load('en_core_web_sm', disable=['parser','tagger','ner','lemmatizer']);"
    "lines = open(sys.argv[1], encoding='utf-8').read().split(chr(10));"
    "out = [' '.join(t.text for t in nlp(l) if not t.is_space) for l in lines];"
    "open(sys.argv[2], 'w', encoding='utf-8').write(chr(10).join(out))"
)

# Re-join clitics before tokenizing, so the step is IDEMPOTENT on already-tokenized input.
#
# Without this, text that is already in the corpus convention gets tokenized a second time and
# comes out WORSE: the corpus writes `I 'm`, spaCy sees a bare apostrophe followed by `m` and
# splits it again into `I ' m`. Measured on 200 dev sentences, that was 5 of 8 disagreements and
# cost 1.5 F0.5 points against a perfectly tokenized answer — a penalty aimed squarely at the
# fine-tuned student, which learns the corpus convention from its training targets and is
# therefore the one system that arrives already tokenized.
#
# Canonicalizing to natural text first means both a natural `I'm` and a pre-split `I 'm` reach
# spaCy as `I'm` and leave as `I 'm`.
_CLITICS = re.compile(r"\s+(n't|'(?:s|re|ve|ll|d|m|t))\b", re.IGNORECASE)


def retokenize(lines: list[str]) -> list[str]:
    """Re-tokenize model output to the corpus's own word tokenization.

    WHY THIS IS NOT OPTIONAL, MEASURED
        ERRANT scores EDITS derived by aligning token sequences, and W&I+LOCNESS is distributed
        WORD-TOKENIZED — a space before every period, `do n't` split in two. A generative model
        emits natural text. So the same correction expressed with natural spacing produces
        completely different edit spans from the gold's.

        Measured on 300 dev sentences by scoring the GOLD answer against its own reference:

            gold, corpus tokenization    F0.5 0.8726   P 0.8698  R 0.8838   fp 66
            gold, natural spacing        F0.5 0.3102   P 0.2776  R 0.5852   fp 760

        A PERFECT answer loses 56 points and gains 11x the false positives, purely from spacing.
        That is what produced the 0.1569 the local teacher first measured on this task — a number
        that then refused synthesis and pinned the run's goal to the floor. It was not a
        measurement of correction quality at all.

        The concrete case, from run 39707195: for the source `... on the road .` with gold
        `... on the road ?`, the model answered `... on the road?` — the right correction — and
        ERRANT compared "replace `.` with `?`" against "rewrite `road` as `road?` and delete `.`",
        scoring one false positive and one false negative on a correct answer.

    ONLY THE HYPOTHESIS IS RE-TOKENIZED. The source must stay byte-identical to the `S` lines of
    the gold M2 file, because that is what the reference edit offsets are relative to; running it
    through a tokenizer would break the reference side to fix the hypothesis side.

    Idempotent on text that is already tokenized, so a fine-tuned student that has learned the
    corpus convention is unaffected. Falls back to the input unchanged if the venv is unavailable
    — a degraded score is better than no score, and `ErrantUnavailable` will surface anyway when
    the actual scoring runs.
    """
    if not lines:
        return []
    python = errant_bin("python")
    with tempfile.TemporaryDirectory(prefix="gec-tok-") as tmp:
        raw_path = os.path.join(tmp, "raw.txt")
        out_path = os.path.join(tmp, "tok.txt")
        with open(raw_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(
                _CLITICS.sub(r"\1", " ".join(str(line or "").split())) for line in lines
            ))
        try:
            _run([python, "-c", _TOKENIZE_SCRIPT, raw_path, out_path])
            with open(out_path, encoding="utf-8") as handle:
                tokenized = handle.read().split("\n")
        except ErrantUnavailable:
            return [" ".join(str(line or "").split()) for line in lines]
    if len(tokenized) != len(lines):
        # A line count mismatch would silently shift every subsequent sentence, which is the exact
        # misalignment this function exists to prevent. Refuse rather than score garbage.
        raise ErrantUnavailable(
            f"re-tokenization returned {len(tokenized)} line(s) for {len(lines)} input(s)"
        )
    return tokenized


def score(eval_set: EvalSet, predictions: list[str | None]) -> dict:
    """ERRANT F0.5, corpus level. Selection and report scoring are the same computation.

    Corpus level only, never per sentence: sentence-level agreement between humans on these
    metrics is 0.3-0.4 while system-level is above 0.9, so a per-sentence F0.5 is not a
    measurement of anything.
    """
    return _score(eval_set, predictions, "errant_f05")


def failure_category_of(failure: dict) -> str:
    """Under- versus over-correction, which is the distinction this task is about.

    The two are opposite failures wanting opposite fixes, and F0.5 alone reports them as the same
    middling number. `missed_correction` is the fine-tuned small model's signature;
    `overcorrected_correct_sentence` is the frontier model's, and it is the expensive one — a
    wrong "fix" to a sentence the user wrote correctly is what F0.5 weights double.
    """
    predicted = failure.get("predicted")
    if predicted is None:
        return "unusable_output"
    source = " ".join(str(failure.get("text", "")).split())
    gold = " ".join(str(failure.get("answer", "")).split())
    prediction = " ".join(str(predicted).split())
    if not prediction:
        return "empty_output"
    if prediction == source and gold != source:
        return "missed_correction"
    if source == gold and prediction != source:
        return "overcorrected_correct_sentence"
    if prediction == source:
        return "no_edit_attempted"
    return "wrong_correction"
