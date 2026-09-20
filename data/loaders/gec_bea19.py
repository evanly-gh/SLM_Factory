"""W&I+LOCNESS (BEA-2019) — correct the grammar of one sentence.

WHY THE GOLD M2 TRAVELS ON EVERY ROW
    ERRANT scores EDITS, not strings: it diffs source against your output to get your edit list,
    diffs source against gold to get the reference list, and compares those. The reference side is
    already published — BEA-2019 ships `ABCN.dev.gold.bea19.m2` with the original annotator's
    edits — and regenerating it from the corrected sentence with `errant_parallel` would re-segment
    those edits by rule. Same sentences, slightly different reference edit list, silently different
    F0.5.

    So each row carries its own raw M2 block, and the scorer reassembles a gold M2 file for
    whatever subset it was handed. That is also what makes a subsampled eval possible at all: the
    official file is one fixed 4,384-sentence artifact, and the in-loop eval scores 1,000 rows.

    F0.5 IS REFERENCE-COUNT DEPENDENT, AND THE EFFECT IS LARGE. CoNLL-14 with two references puts
    top systems near 68; re-scored against a 10-annotator extension the same systems reach 80-81
    against a human 72.58. W&I+LOCNESS train and dev are SINGLE-reference, which caps measurable
    recall and is a property of our number that has to be stated whenever it is compared.

THE PROFICIENCY LEVELS ARE THE EPISODE AXIS, AND THEY COME FROM THE FILE NAMES
    The release splits both train and dev by CEFR level — A beginner, B intermediate, C advanced —
    in separate M2 files, and LOCNESS N (native) appears in dev ONLY. So the level is recovered by
    reading the per-level files rather than the combined one, which also reconstructs the official
    combined dev exactly: 1,037 + 1,290 + 1,069 + 988 = 4,384, and 10,493 + 13,032 + 10,783 =
    34,308 on train.

    That gives leave-one-level-out over A/B/C with N as a free zero-shot out-of-domain slice. The
    official TEST set is deliberately combined with no level labels, because a real keyboard does
    not know the user's proficiency — and it is blind (CodaLab submission only), which is why dev
    is what gets reported.
"""
from __future__ import annotations

import os
import tarfile
import tempfile
from collections import Counter

from data.loaders.dataset_integrity import remove_normalized_train_overlap

# The official release. A tarball off a university web server rather than anything on the Hub, and
# the reason this task gets a vendored `data/local` bundle: a compute node reaching out to
# cl.cam.ac.uk mid-run is the fragility the checksummed-bundle pattern exists to remove.
BEA19_URL = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"
LOCAL_BUNDLE = "data/local/bea19"
BUNDLE_ENV = "SLM_BEA19_DIR"

# Level -> (train file, dev file). N is native LOCNESS and has no training half, which is the
# point of it: an out-of-domain slice no training data can have leaked into.
M2_FILES = {
    "A": ("m2/A.train.gold.bea19.m2", "m2/A.dev.gold.bea19.m2"),
    "B": ("m2/B.train.gold.bea19.m2", "m2/B.dev.gold.bea19.m2"),
    "C": ("m2/C.train.gold.bea19.m2", "m2/C.dev.gold.bea19.m2"),
    "N": (None, "m2/N.dev.gold.bea19.m2"),
}

# Published counts, asserted at load. A silently truncated download would otherwise present as a
# quietly worse score rather than as an error.
EXPECTED_TRAIN = 34_308
EXPECTED_DEV = 4_384

INSTRUCTION = "Correct the grammatical errors in the sentence."


def parse_m2(text: str) -> list[dict]:
    """An M2 file into `[{"source", "target", "m2"}]`.

    `source` is the tokenized original, `target` is it with the annotator's edits applied, and
    `m2` is the sentence's own raw block so the scorer can rebuild an exact reference file.

    Only annotator 0 is applied. W&I+LOCNESS train and dev are single-reference, so there is only
    one; being explicit means a multi-reference file cannot silently have its annotators' edits
    merged into one incoherent target.
    """
    blocks: list[dict] = []
    source: str | None = None
    lines: list[str] = []

    def flush() -> None:
        nonlocal source
        if source is not None:
            edits = [_parse_edit(line) for line in lines]
            blocks.append({
                "source": source,
                "target": apply_m2_edits(source, [e for e in edits if e is not None]),
                "m2": "\n".join([f"S {source}"] + lines),
            })
        source = None
        lines.clear()

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("S "):
            flush()
            source = line[2:]
        elif line.startswith("A ") and source is not None:
            lines.append(line)
    flush()
    return blocks


def _parse_edit(line: str) -> tuple[int, int, str] | None:
    """One `A` line into `(start, end, replacement)`, or None for a noop."""
    fields = line[2:].split("|||")
    if len(fields) < 3:
        return None
    span, error_type, replacement = fields[0], fields[1], fields[2]
    if error_type == "noop":
        return None
    try:
        start_text, end_text = span.split()
        start, end = int(start_text), int(end_text)
    except (ValueError, IndexError):
        return None
    if start < 0 or end < 0:
        return None
    # Only annotator 0. See parse_m2.
    if len(fields) >= 6 and fields[5].strip() not in ("0", ""):
        return None
    return start, end, ("" if replacement == "-NONE-" else replacement)


def apply_m2_edits(source: str, edits: list[tuple[int, int, str]]) -> str:
    """Apply token-span edits to a tokenized sentence.

    Applied HIGHEST START FIRST, so replacing a span never shifts the indices of an edit that has
    not been applied yet. Doing it left to right requires tracking a running offset, which is the
    same computation with a place to get it wrong.
    """
    tokens = source.split()
    for start, end, replacement in sorted(edits, key=lambda e: (-e[0], -e[1])):
        if start > len(tokens):
            continue
        tokens[start:end] = replacement.split() if replacement else []
    return " ".join(tokens)


def _bundle_dir() -> str:
    """Where the vendored release lives.

    Resolved from `__file__` rather than through `config.config`, which is the same choice
    `data/loaders/ner_bc5cdr.py` makes and for the same reason: importing `config.config` reads
    `os.environ["ANTHROPIC_API_KEY"]` at module scope, so a loader that touched it could not be
    unit-tested or run offline without a key in the environment.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.environ.get(BUNDLE_ENV) or os.path.join(repo_root, LOCAL_BUNDLE)


def _read_m2_files(root: str) -> dict[str, str]:
    """The eight per-level M2 files, keyed by their archive-relative path."""
    wanted = {path for pair in M2_FILES.values() for path in pair if path}
    found: dict[str, str] = {}
    for path in wanted:
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            with open(candidate, encoding="utf-8") as handle:
                found[path] = handle.read()
    return found


def _fetch_m2_files(log=print) -> dict[str, str]:
    """Download the release tarball and read the per-level M2 files out of it.

    Extracted into a temporary directory and not persisted: the durable copy is the vendored
    `data/local/bea19` bundle written by `scripts/download_datasets.py`, and a loader that
    silently populated the offline bundle would make the bundle's checksums meaningless.
    """
    import urllib.request

    log(f"      [gec_bea19] no local bundle; fetching {BEA19_URL}")
    with tempfile.TemporaryDirectory(prefix="bea19-") as tmp:
        archive = os.path.join(tmp, "wi_locness.tar.gz")
        urllib.request.urlretrieve(BEA19_URL, archive)  # noqa: S310 - pinned https URL
        with tarfile.open(archive) as tar:
            # `filter="data"` refuses absolute paths and symlinks escaping the destination.
            tar.extractall(tmp, filter="data")
        return _read_m2_files(os.path.join(tmp, "wi+locness"))


def _load_m2_files(log=print) -> dict[str, str]:
    root = _bundle_dir()
    found = _read_m2_files(root)
    if found:
        log(f"      [gec_bea19] local bundle {root}: {len(found)} M2 file(s)")
        return found
    return _fetch_m2_files(log=log)


def to_rows(blocks: list[dict], cefr: str) -> list[dict]:
    """M2 blocks into task rows, tagged with the proficiency level they came from."""
    rows: list[dict] = []
    for block in blocks:
        source = block["source"].strip()
        if not source:
            continue
        rows.append({
            "text": source,
            # An unedited sentence's target IS its source. Those rows are not noise to be dropped:
            # roughly a fifth of the corpus needs no correction, and a model that "fixes" correct
            # text is the precision failure F0.5 weights double.
            "answer": block["target"].strip() or source,
            "cefr": cefr,
            "m2": block["m2"],
            "_instruction": INSTRUCTION,
        })
    return rows


def load_gec_bea19(
    max_train: int = 5000, max_test: int = 1000, log=print
) -> tuple[list[dict], list[dict]]:
    """Return `(train, dev)` as tokenized sentence pairs carrying their gold M2 and CEFR level.

    Dev is the official BEA-2019 dev set — the A, B, C and N files, which reassemble to the
    published 4,384 — because the 4,477-sentence test set is blind and scoreable only through
    CodaLab.
    """
    files = _load_m2_files(log=log)
    train: list[dict] = []
    dev: list[dict] = []
    for level, (train_path, dev_path) in sorted(M2_FILES.items()):
        if train_path and train_path in files:
            train.extend(to_rows(parse_m2(files[train_path]), level))
        if dev_path and dev_path in files:
            dev.extend(to_rows(parse_m2(files[dev_path]), level))

    if not train or not dev:
        raise RuntimeError(
            f"BEA-2019 M2 files not found. Vendor the bundle into {_bundle_dir()} with "
            f"`python scripts/download_datasets.py --only bea19`, or set {BUNDLE_ENV}."
        )
    # Loud rather than quiet: a truncated download would otherwise read as a worse model.
    if len(train) != EXPECTED_TRAIN or len(dev) != EXPECTED_DEV:
        log(f"      [gec_bea19] WARNING: got train={len(train)} dev={len(dev)}, expected "
            f"{EXPECTED_TRAIN}/{EXPECTED_DEV}; the release may be incomplete")

    # Learner corpora repeat short sentences heavily — "Thank you .", "I like it very much ." —
    # so a few appear verbatim in both train and dev. `curate`'s eval firewall would drop them
    # before training, but the reported curriculum size would then shrink silently; removing them
    # here keeps the count honest. Dev is kept intact: the official eval split never gives way.
    train, removed = remove_normalized_train_overlap(train, dev)

    by_level = Counter(row["cefr"] for row in dev)
    unchanged = sum(1 for row in dev if row["text"] == row["answer"])
    log(f"      [gec_bea19] train={len(train)} dev={len(dev)} dev levels={dict(sorted(by_level.items()))}")
    if removed:
        log(f"      [gec_bea19] dropped {removed} train sentence(s) that also appear in dev")
    log(f"      [gec_bea19] {unchanged} of {len(dev)} dev sentences need no correction "
        f"({100 * unchanged / max(1, len(dev)):.0f}%); N (native) is dev-only and has no train half")
    # Interleaved so the caller's cap keeps every level represented. The per-level files are
    # concatenated in order, so a prefix would otherwise be all level A — the same truncation trap
    # TOPv2's domain-ordered test split had.
    return _interleave_by_level(train)[:max_train], _interleave_by_level(dev)[:max_test]


def _interleave_by_level(rows: list[dict]) -> list[dict]:
    """Round-robin across CEFR levels so ANY prefix covers every level."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["cefr"], []).append(row)
    out: list[dict] = []
    for index in range(max((len(bucket) for bucket in buckets.values()), default=0)):
        for level in sorted(buckets):
            if index < len(buckets[level]):
                out.append(buckets[level][index])
    return out
