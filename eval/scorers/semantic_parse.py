"""TOPv2 scoring: exact match on the parse string, decomposed by target domain.

WHY THE COMPARISON SCALAR IS A MEAN OF TWO DOMAINS
    `weather` is flat — 7 intents, 11 slots, 0.1% of parses with more than one intent — and will
    saturate. `reminder` is compositional at 21.5% and will not. Averaging them is the right LOOP
    signal precisely because of that asymmetry: once weather tops out, all remaining optimization
    pressure falls on the compositional domain, which is where it should be. But an unlabelled
    average hides which half moved, so both components are always reported beside it and the
    writeup decomposes them.

WHY EXACT MATCH AND NOT A TREE-EDIT SCORE
    EM is what the TOPv2 literature reports, so it is the only number with anything to compare
    against. It is also brutally sensitive to serialization, which is the documented watch-out for
    this task: our target is the mirror's `semantic_parse` string verbatim, so nothing here
    normalizes the tree. The only normalization applied is collapsing runs of whitespace, and that
    is applied identically to gold and prediction — a model that emits two spaces between brackets
    has a formatting quirk, not a parsing error, and EM should not be measuring the tokenizer.
"""
import re

from data.eval_set import EvalSet

TOPV2_PROMPT = (
    "Parse the command into a nested intent and slot tree.\n\n"
    "Intents: {intents}\n\n"
    "Slots: {slots}\n\n"
    "Rules: the tree starts with an intent, written [IN:NAME ... ]. Slots inside it are written "
    "[SL:NAME ... ]. A slot may contain a nested intent. Every bracket is separated from its "
    "neighbours by a single space, every closing bracket has a space before it, and the words in "
    "the tree reproduce the command exactly, in order, adding and omitting nothing.\n\n"
    "Example command: Set alarm for 6 am every day\n"
    "Example parse: [IN:CREATE_ALARM Set alarm [SL:DATE_TIME_RECURRING for 6 am every day ] ]\n\n"
    "Reply with only the parse for the command below, on one line.\n\n"
    "Command: {text}"
)


def build_prompt(text: str) -> str:
    """The ONE prompt. Imported by the training turn builder so the two cannot drift (B250).

    WHY THERE IS A WORKED EXAMPLE AND NOT A FORMAT TEMPLATE
        This prompt used to show the shape as `Format: [IN:INTENT_NAME words [SL:SLOT_NAME words ]
        ]`. That placeholder is itself a syntactically valid answer, so a weak model copied it
        verbatim — and the extractor accepted it, because `INTENT_NAME` matches the label pattern
        and the brackets balance. Observed on run 39719567, the base model's answer to
        "Remind me to hook up the DVD player in the bedroom next week" was, in full,
        `[IN:INTENT_NAME words [SL:SLOT_NAME words ] ]`.

        A concrete example with REAL labels cannot be copied into a passing answer, because the
        spans have to reproduce the model's own command and the example's do not.
    """
    from data.loaders.topv2 import INTENTS, SLOTS

    return TOPV2_PROMPT.format(
        intents=", ".join(INTENTS), slots=", ".join(SLOTS), text=text,
    )

# The first balanced `[IN:...]` tree in the reply. A model that prefaces its answer with prose is
# still answering; a model that emits no tree at all is not, and the two must score differently
# (`format_valid` vs a wrong parse) or a prompt problem is indistinguishable from a data problem.
_TREE_START = re.compile(r"\[IN:")
_INTENT_RE = re.compile(r"\[(IN:[A-Z0-9_]+)")
_SLOT_RE = re.compile(r"\[(SL:[A-Z0-9_]+)")


def build_prompts(eval_set: EvalSet) -> list[str]:
    return [build_prompt(example.get("text", "")) for example in eval_set.all]


def normalize_parse(parse: str) -> str:
    """Collapse whitespace runs. Applied to gold AND prediction, or it would be a thumb on the scale."""
    return " ".join(str(parse or "").split())


def _extract_tree(raw: str) -> str | None:
    """The first balanced bracket tree in `raw`, or None when there is not one.

    Scanning for balance rather than taking the first line: a model that wraps its answer in
    markdown fences or trails a sentence after the tree has produced a usable parse, and throwing
    it away would report a content failure for a formatting habit. An unbalanced tree IS a format
    failure — there is no parse to score — and returns None.
    """
    text = str(raw or "")
    match = _TREE_START.search(text)
    if not match:
        return None
    depth = 0
    for index in range(match.start(), len(text)):
        if text[index] == "[":
            depth += 1
        elif text[index] == "]":
            depth -= 1
            if depth == 0:
                return normalize_parse(text[match.start():index + 1])
    return None


def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str | None]:
    """One normalized parse per reply, or None where no balanced tree could be found."""
    return [_extract_tree(raw) for raw in raw_outputs]


def _labels_outside_vocabulary(parse: str) -> set[str]:
    """Labels in `parse` that the task's declared vocabulary does not contain."""
    from data.loaders.topv2 import INTENTS, SLOTS

    used = set(_INTENT_RE.findall(parse)) | set(_SLOT_RE.findall(parse))
    declared = {f"IN:{name}" for name in INTENTS} | {f"SL:{name}" for name in SLOTS}
    return used - declared


def _domain_of(example: dict) -> str:
    return str(example.get("domain") or "unknown")


def score(eval_set: EvalSet, predictions: list[str | None]) -> dict:
    """Exact match, with the comparison scalar averaged over target domains.

    The headline is the mean of the PER-DOMAIN exact-match rates, not the pooled rate. The two
    differ whenever the eval draw is not perfectly balanced, and the per-domain mean is the one
    that does not let a 60/40 sampling accident move the score.
    """
    rows = eval_set.all
    per_domain_hits: dict[str, int] = {}
    per_domain_total: dict[str, int] = {}
    failures = []
    readable = 0

    for example, prediction in zip(rows, predictions):
        domain = _domain_of(example)
        per_domain_total[domain] = per_domain_total.get(domain, 0) + 1
        if prediction is not None:
            readable += 1
        gold = normalize_parse(example.get("answer", ""))
        if prediction is not None and prediction == gold:
            per_domain_hits[domain] = per_domain_hits.get(domain, 0) + 1
            continue
        failures.append({
            **example,
            "predicted": prediction,
            "error_type": failure_category_of({**example, "predicted": prediction}),
        })

    per_domain_em = {
        f"em_{domain}": per_domain_hits.get(domain, 0) / total
        for domain, total in sorted(per_domain_total.items())
        if total
    }
    # Mean over the domains PRESENT. A draw that happens to contain only one target domain still
    # gets a defined score, and the `n_` counts below say the average was over one thing.
    em = sum(per_domain_em.values()) / len(per_domain_em) if per_domain_em else 0.0
    format_valid = readable / len(rows) if rows else 0.0
    return {
        "f1": em,
        "metric": "exact_match",
        "per_class": {
            "exact_match": em,
            **per_domain_em,
            **{f"n_{domain}": total for domain, total in sorted(per_domain_total.items())},
            # The pooled rate, kept as a diagnostic so the gap between it and the headline shows
            # when the eval draw is unbalanced.
            "em_pooled": (
                sum(per_domain_hits.values()) / len(rows) if rows else 0.0
            ),
            "format_valid": format_valid,
        },
        "failures": failures,
        "format_valid": format_valid,
    }


def failure_category_of(failure: dict) -> str:
    """Why this parse is wrong, in a category that points at a different fix.

    Intent errors and slot errors want different data: the first is a classification problem over
    a small label set, the second is a span problem. Reporting both as "wrong parse" told the
    orchestrator nothing it could act on, which is the B296 shape.
    """
    predicted = failure.get("predicted")
    if predicted is None:
        return "unparseable_output"
    gold = normalize_parse(failure.get("answer", ""))
    if predicted == gold:
        return "correct"
    # A label outside the declared vocabulary is definitionally wrong, and it is a LABEL-SPACE
    # problem rather than a parsing one — the prompt listed the names and the model used another.
    # Called out separately for the same reason `multiconer` separates `type_outside_taxonomy`:
    # it wants a different fix from choosing the wrong name from the right list. It also catches
    # a model echoing a placeholder, which is how run 39719567's base model answered every row.
    if _labels_outside_vocabulary(predicted):
        return "label_outside_vocabulary"
    if _INTENT_RE.findall(predicted) != _INTENT_RE.findall(gold):
        return "wrong_intent"
    if _SLOT_RE.findall(predicted) != _SLOT_RE.findall(gold):
        return "wrong_slot_set"
    # Same intents in the same order and the same slots in the same order, so what differs is the
    # text inside the spans — a boundary or a copying error, not a structural one.
    return "wrong_span_text"
