"""GoEmotions — multi-label emotion over 27 emotions plus neutral, on short Reddit comments.

WHY THE LABEL SPACE HAS THREE RESOLUTIONS
    The corpus ships 28 fine labels, and the original paper also evaluates them grouped into
    Ekman's 6 basic emotions plus neutral, and into 4 sentiment classes. That is not redundancy —
    the three resolutions are what make this task usable at all:

      * The 28-label macro-F1 is a thresholding artifact and its tail is statistically empty. Test
        support: grief 6, relief 11, pride 16, nervousness 23, against neutral 1,787.
      * The Ekman-7 grouping has real support in every class, so it is the stable number and the
        one the loop selects on.
      * The published tail improvements are largely noise. One paper reports grief going 0.00 ->
        0.57 F1, which is +2 macro-F1 points earned on six test examples.

    So the headline is threshold-free macro AUPRC over the 28 labels, selection is Ekman-7
    macro-F1, and per-label numbers are reported in frequency BANDS rather than individually.

THE LABEL NAMES, NOT THE INDICES
    The corpus stores labels as integer ids into a fixed list. A generative model has to emit
    words, so rows carry names and the id order is pinned here — the ordering is the dataset's
    own and `neutral` is index 27, which several downstream groupings depend on.
"""
from __future__ import annotations

from collections import Counter
from data.loaders.dataset_integrity import remove_normalized_train_overlap

GOEMOTIONS_ID = "google-research-datasets/go_emotions"
CONFIG = "simplified"

PARQUET_FILES = {
    "train": "simplified/train-00000-of-00001.parquet",
    "validation": "simplified/validation-00000-of-00001.parquet",
    "test": "simplified/test-00000-of-00001.parquet",
}

# The dataset's own label order. Index 27 is `neutral`.
EMOTIONS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
)

# Ekman's six basic emotions plus neutral, as the GoEmotions paper groups them. This is the
# SELECTION taxonomy: every one of the seven has real support in any plausible eval draw, which
# the 28-label space emphatically does not.
EKMAN_GROUPS = {
    "anger": ("anger", "annoyance", "disapproval"),
    "disgust": ("disgust",),
    "fear": ("fear", "nervousness"),
    "joy": (
        "admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude",
        "joy", "love", "optimism", "pride", "relief",
    ),
    "sadness": ("disappointment", "embarrassment", "grief", "remorse", "sadness"),
    "surprise": ("confusion", "curiosity", "realization", "surprise"),
    "neutral": ("neutral",),
}

EKMAN_OF = {
    emotion: group for group, members in EKMAN_GROUPS.items() for emotion in members
}

# Frequency bands for the per-label diagnostic, by GOLD support in the scored split. Reporting
# bands rather than 28 individual F1s is the mechanism that stops six rows carrying a headline.
BAND_HEAD_MIN = 300
BAND_MID_MIN = 50

INSTRUCTION = "Label the emotions expressed in the comment."


def band_of(support: int) -> str:
    """Which frequency band a label's support falls in."""
    if support >= BAND_HEAD_MIN:
        return "head"
    if support >= BAND_MID_MIN:
        return "mid"
    return "tail"


def convert_goemotions_rows(dataset) -> list[dict]:
    """Rows carrying `text` and integer `labels` into `{text, labels, label}`.

    `labels` is the sorted list of emotion NAMES. `label` is the same thing joined by ", " — the
    single string a generative model is trained to emit and scored against, kept as a separate
    field so nothing has to re-derive the serialization in two places.
    """
    out: list[dict] = []
    for example in dataset:
        text = str(example.get("text") or "").strip()
        raw = example.get("labels")
        if not text or raw is None:
            continue
        names = sorted(
            EMOTIONS[int(index)] for index in raw if 0 <= int(index) < len(EMOTIONS)
        )
        if not names:
            continue
        out.append({
            "text": text,
            "labels": names,
            "label": ", ".join(names),
            "_instruction": INSTRUCTION,
        })
    return out


def _read_split(split: str):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(GOEMOTIONS_ID, PARQUET_FILES[split], repo_type="dataset")
    import pandas as pd

    return pd.read_parquet(path).to_dict("records")


def load_goemotions(
    max_train: int = 5000, max_test: int = 1000, log=print
) -> tuple[list[dict], list[dict]]:
    """Return `(train, test)`. Test is the official 5,427-row split.

    The official validation split is not used: early stopping already carves a validation slice
    out of train (`training/lora_trainer.py`), and the test split is what the published numbers
    are computed on. Loading validation as the eval set would report a number nothing else does.
    """
    train = convert_goemotions_rows(_read_split("train"))
    test = convert_goemotions_rows(_read_split("test"))

    support = Counter(name for row in test for name in row["labels"])
    tail = {name: support.get(name, 0) for name in EMOTIONS if support.get(name, 0) < BAND_MID_MIN}
    multi = sum(1 for row in test if len(row["labels"]) > 1)
    log(f"      [goemotions] train={len(train)} test={len(test)} "
        f"({multi} test row(s) carry more than one label)")
    log(f"      [goemotions] tail labels under {BAND_MID_MIN} test mentions: {tail}")

    # A handful of Reddit comments appear in both official splits — the corpus is scraped, and
    # short comments recur. Same reasoning as `multiconer` and `topv2`: the eval firewall in
    # `curate` catches them, but a loader that knowingly ships leakage makes the curriculum count
    # a lie and leaves the guarantee resting on a downstream net. TEST IS KEPT INTACT.
    #
    # Deduped BEFORE the caps so the row the cap drops is not the row the dedupe would have.
    train, leaked = remove_normalized_train_overlap(train, test)
    if leaked:
        log(f"      [goemotions] dropped {leaked} train row(s) whose text also appears in test")
    return train[:max_train], test[:max_test]
