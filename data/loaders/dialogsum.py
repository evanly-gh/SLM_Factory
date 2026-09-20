"""DialogSum — summarize a real-life spoken conversation in a sentence or two.

WHY THIS REPLACED THE OLD `dialogsum_samsum` LOADER (2026-09-06)
    The previous loader concatenated DialogSum and SAMSum 50/50 and the task was scored by the LLM
    judge. Both are gone, for reasons that are properties of the data rather than preferences:

    1. THE THREE REFERENCES ARE THE POINT, AND THE OLD SOURCE CANNOT SUPPLY THEM.
       DialogSum's test split has three independent human summaries per dialogue. Scoring against
       all three and taking the best match per example is what stops the number being a lottery
       about whose phrasing the model happened to match, and it is what makes the published human
       ceiling meaningful — one annotator scored against the other two reaches ROUGE-1 53.35 /
       ROUGE-2 26.72 / ROUGE-L 50.84, against about 47 for the best fine-tuned models.

       `knkarthick/dialogsum`, which the old loader used, DESTROYS that structure. Its `test.csv`
       is 1,500 rows with a single `summary` column: the three references have been flattened into
       three separate rows of the same dialogue. Verified 2026-09-06. Nothing downstream can
       recover the grouping, and scoring it as 1,500 independent examples would count each
       dialogue three times and compare each copy against one arbitrary annotator.

       The original release does carry them, so that is what this loads: `cylnlp/dialogsum` ships
       `dialogsum.{train,dev,test}.jsonl` at 12,460 / 500 / 500, with `summary1/2/3` on test.

    2. SAMSUM IS OUT. Same task, single reference, easier, and it was diluting the training set
       with a second summary style while contributing nothing the metric could use.

    3. THE JUDGE IS OUT AS THE HEADLINE. An LLM judge score is comparable to no published number,
       is non-deterministic, and costs money on every eval. ROUGE against three references plus
       BERTScore is comparable, free and repeatable. The judge machinery still exists for
       `toolbench`, whose pass rate is DEFINED as a judged vote.
"""
from __future__ import annotations

import json
import os
import statistics

# The original release, not a mirror. GitHub raw rather than the Hub because no Hub mirror carries
# the three test references — which is the entire reason this task was reworked.
RAW_BASE = "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data"
JSONL_FILES = {
    "train": "dialogsum.train.jsonl",
    "dev": "dialogsum.dev.jsonl",
    "test": "dialogsum.test.jsonl",
}

LOCAL_BUNDLE = "data/local/dialogsum"
BUNDLE_ENV = "SLM_DIALOGSUM_DIR"

# Published counts, checked at load so a truncated fetch reads as an error rather than as a
# quietly worse score.
EXPECTED = {"train": 12_460, "dev": 500, "test": 500}

# Carried on every row and read by `eval.scorers.summarization`, which the eval harness AND the
# trainer both call, so the model is asked to do the same thing in training and at eval.
#
# Kept VERBATIM from the old loader. It is the B250 fix and it is still exactly right: without it
# both paths fell back to the generation family's default, "Answer the following question:", which
# is wrong here in a way that destroys the task — a dialogue transcript asks no question, so the
# model continued the conversation instead of summarizing it. Observed in run
# slm-dialogsum-samsum-cse-38186375, e.g. the reply "Shelly: How about you? Any volunteer work?
# Tracy: Nah. Not into that." against a gold summary. The second sentence is aimed at that.
SUMMARIZATION_INSTRUCTION = (
    "Summarize the following conversation in one to three sentences. "
    "Write only the summary — do not continue the conversation or reply to it."
)


def convert_dialogsum_rows(dataset) -> list[dict]:
    """Original-release rows into `{text, answer, references, topic}`.

    `references` is the list the scorer grades against: all three summaries on test, and the
    single one on train and dev. `answer` is `references[0]` — the training target, and the field
    `dataset_integrity.validate_rows` type-checks — so a one-reference row and a three-reference
    row have the same shape and nothing downstream needs to know which split it came from.
    """
    out: list[dict] = []
    for example in dataset:
        dialogue = str(example.get("dialogue") or "").strip()
        if not dialogue:
            continue
        references = [
            str(example[key]).strip()
            for key in ("summary", "summary1", "summary2", "summary3")
            if str(example.get(key) or "").strip()
        ]
        if not references:
            continue
        topics = [
            str(example[key]).strip()
            for key in ("topic", "topic1", "topic2", "topic3")
            if str(example.get(key) or "").strip()
        ]
        out.append({
            "text": dialogue,
            "answer": references[0],
            "references": references,
            "topic": topics[0] if topics else "",
            "_instruction": SUMMARIZATION_INSTRUCTION,
        })
    return out


def _bundle_dir() -> str:
    """Where the vendored release lives.

    Resolved from `__file__` rather than through `config.config`, which reads
    `os.environ["ANTHROPIC_API_KEY"]` at import — so a loader touching it could not run offline or
    under a unit test without a key. Same choice `data/loaders/ner_bc5cdr.py` makes.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.environ.get(BUNDLE_ENV) or os.path.join(repo_root, LOCAL_BUNDLE)


def _read_jsonl(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _bundle_rows(split: str) -> list[dict] | None:
    """One split of the frozen offline bundle, ALREADY in `{text, answer, references}` form.

    `scripts/download_datasets.py` writes bundles in the task's row schema, not the source's, and
    it does so by calling `convert_dialogsum_rows` — so these rows have been through the same
    converter a live fetch would use. Reading them back through the converter a second time would
    be a no-op at best and a divergence the moment the two shapes differ, which is the
    bundle-versus-loader drift `data/local/bc5cdr` exists to avoid.
    """
    path = os.path.join(_bundle_dir(), f"{split}.jsonl")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        rows = _read_jsonl(handle.read())
    return [row for row in rows if row.get("text") and row.get("references")] or None


def _release_rows(split: str, log=print) -> list[dict]:
    """One split fetched from the original release, in the SOURCE format."""
    import urllib.request

    url = f"{RAW_BASE}/{JSONL_FILES[split]}"
    log(f"      [dialogsum] no local bundle; fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 - pinned https URL
        return convert_dialogsum_rows(_read_jsonl(response.read().decode("utf-8")))


def _load_split(split: str, log=print) -> list[dict]:
    """One split as task rows, from the vendored bundle if present and otherwise the release."""
    rows = _bundle_rows(split)
    if rows is not None:
        return rows
    return _release_rows(split, log=log)


def load_dialogsum(
    max_train: int = 5000, max_test: int = 1000, log=print
) -> tuple[list[dict], list[dict]]:
    """Return `(train, test)`. Test is the 500-dialogue split carrying three references each.

    The 500-row test split is the smallest eval in the suite, and that is acceptable here in a way
    it would not be for a binary metric: ROUGE is continuous per example rather than 0/1, so the
    confidence interval lands near +/-1.3 points. It is reported, not glossed over.
    """
    train_rows = _load_split("train", log=log)
    test_rows = _load_split("test", log=log)

    if not train_rows or not test_rows:
        raise RuntimeError(
            f"DialogSum not found. Vendor the bundle into {_bundle_dir()} with "
            f"`python scripts/download_datasets.py --only dialogsum`, or set {BUNDLE_ENV}."
        )
    for name, rows in (("train", train_rows), ("test", test_rows)):
        # The bundle can legitimately be a row or two short of the release: the bundle writer runs
        # normalized train/test decontamination and removes any training row whose dialogue also
        # appears in test. So this warns on a real shortfall and tolerates that.
        if len(rows) > EXPECTED[name] or len(rows) < EXPECTED[name] - 10:
            log(f"      [dialogsum] WARNING: {name} has {len(rows)} rows, expected about "
                f"{EXPECTED[name]}; the source may be incomplete")

    # THE PROPERTY THE REWORK EXISTS FOR, asserted at load rather than assumed. A mirror that
    # flattens the three references into separate rows — which `knkarthick/dialogsum` does — would
    # pass every other check and silently reduce the metric to single-reference ROUGE.
    reference_counts = {len(row["references"]) for row in test_rows}
    if reference_counts != {3}:
        raise RuntimeError(
            f"DialogSum test rows carry {sorted(reference_counts)} reference(s), expected exactly "
            "3. The multi-reference structure is this task's whole premise; a source that "
            "flattens it (knkarthick/dialogsum's test.csv is 1,500 rows with one `summary` "
            "column) cannot be used."
        )

    dialogue_tokens = statistics.median(len(row["text"].split()) for row in test_rows)
    summary_tokens = statistics.median(
        len(reference.split()) for row in test_rows for reference in row["references"]
    )
    log(f"      [dialogsum] train={len(train_rows)} test={len(test_rows)} "
        f"(3 references per test dialogue)")
    log(f"      [dialogsum] median dialogue {dialogue_tokens:.0f} words, "
        f"summary {summary_tokens:.0f} words — shorter summaries over longer conversations than "
        f"SAMSum, which is what makes it demand real abstraction")

    # No cross-source dedup pass. The old loader needed one because SAMSum ships a handful of
    # dialogues in both its own train and test splits with different gold summaries (B285);
    # DialogSum is one corpus with disjoint official splits, so there is nothing to reconcile.
    return train_rows[:max_train], test_rows[:max_test]
