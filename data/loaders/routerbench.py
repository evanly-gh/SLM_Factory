"""RouterBench routing-decision loader (2026-08-01).

RouterBench records, per prompt, whether each candidate model answers correctly. We reframe it
as a binary ``classification`` task: given the prompt, decide whether a small on-device model
can handle it (``local``) or it should be escalated (``route``). The label is derived from the
designated small-model's recorded correctness on that prompt.

The pure ``convert_routerbench_rows`` is unit-tested on an in-memory sample; ``load_routerbench``
performs the live HF pull on the cluster.
"""
from __future__ import annotations

from collections.abc import Iterable

HF_ID = "withmartian/routerbench"

LABEL_LOCAL = "local"   # the small model is correct → keep on device.
LABEL_ROUTE = "route"   # the small model fails → escalate to a larger model.

# Default column whose recorded correctness stands in for the on-device model. Overridable so a
# different small model in the benchmark can define the routing boundary.
_DEFAULT_SMALL_MODEL_KEY = "small_model_correct"


def _correctness(ex: dict, small_model_key: str, threshold: float):
    """Extract the small model's correctness for a row, or None if unavailable."""
    value = ex.get(small_model_key)
    if value is None:
        perf = ex.get("performance") or ex.get("scores")
        if isinstance(perf, dict):
            value = perf.get(small_model_key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        return float(value) >= threshold
    except (TypeError, ValueError):
        return None


def convert_routerbench_rows(
    dataset: Iterable[dict],
    *,
    small_model_key: str = _DEFAULT_SMALL_MODEL_KEY,
    threshold: float = 0.5,
) -> list[dict]:
    """Map raw RouterBench rows to ``{text, label}`` routing decisions.

    Rows already carrying an explicit ``label`` (``local``/``route``) are used verbatim. Rows
    without a usable prompt or a resolvable correctness signal are dropped.
    """
    out: list[dict] = []
    for ex in dataset:
        text = str(ex.get("prompt") or ex.get("text") or ex.get("question") or "").strip()
        if not text:
            continue
        label = ex.get("label")
        if label in (LABEL_LOCAL, LABEL_ROUTE):
            out.append({"text": text, "label": label})
            continue
        correct = _correctness(ex, small_model_key, threshold)
        if correct is None:
            continue
        out.append({"text": text, "label": LABEL_LOCAL if correct else LABEL_ROUTE})
    return out


def load_routerbench(
    max_train: int = 2000, max_test: int = 800, *,
    small_model_key: str = _DEFAULT_SMALL_MODEL_KEY,
) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` RouterBench routing decisions as ``{text, label}`` rows."""
    from datasets import load_dataset

    def _conv(split_name: str, limit: int) -> list[dict]:
        raw = load_dataset(HF_ID, split=f"{split_name}[:{limit}]", trust_remote_code=True)
        return convert_routerbench_rows(raw, small_model_key=small_model_key)

    try:
        train = _conv("train", max_train)
        test = _conv("test", max_test)
    except (ValueError, KeyError):
        # Single-split benchmark: partition one split into train/test deterministically.
        raw = load_dataset(HF_ID, split=f"train[:{max_train + max_test}]", trust_remote_code=True)
        rows = convert_routerbench_rows(raw, small_model_key=small_model_key)
        train, test = rows[:max_train], rows[max_train:max_train + max_test]
    return train, test
