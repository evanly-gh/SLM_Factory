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
PICKLE_FILE = "routerbench_0shot.pkl"

LABEL_LOCAL = "local"   # the small model is correct → keep on device.
LABEL_ROUTE = "route"   # the small model fails → escalate to a larger model.

# Non-model columns in the pickle, excluded when reporting available candidate models.
_META_COLUMNS = frozenset({"sample_id", "prompt", "eval_name", "oracle_model_to_route_to"})

# The column whose recorded correctness stands in for the on-device model. RouterBench holds one
# correctness column per candidate model, named after the model; there is no
# `small_model_correct` column, and assuming one is what made the previous loader return an empty
# dataset even where the file could be read (B254).
#
# mistral-7b-chat is the SMALLEST model in the benchmark and therefore the closest available
# analogue to something that would actually run on an S24 Ultra. It answers 29.9% of prompts
# correctly, so `local` is the minority class — which is what `eval/scorers/classification.py`
# reports F1 on for a binary task.
_DEFAULT_SMALL_MODEL_KEY = "mistralai/mistral-7b-chat"


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


def _prompt_text(value: object) -> str:
    """Recover the readable prompt from RouterBench's ``prompt`` column.

    The column is not the prompt — it is a **string containing a Python list literal** of the
    message turns, e.g. ``"['You are a helpful assistant.', 'What is 2+2?']"``. Passing it through
    ``str()`` leaks brackets and quote marks into every single eval prompt, which is both ugly and
    a genuine distribution shift away from anything the model will see in deployment. Parse the
    literal and join the turns; fall back to the raw text when it is not a list literal.
    """
    import ast

    if isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        text = str(value or "").strip()
        if not (text.startswith("[") and text.endswith("]")):
            return text
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return text
        if not isinstance(parsed, (list, tuple)):
            return text
        parts = list(parsed)
    return "\n\n".join(str(p).strip() for p in parts if str(p).strip()).strip()


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
        text = _prompt_text(
            ex.get("prompt") or ex.get("text") or ex.get("question") or "")
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
    log=print,
) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` RouterBench routing decisions as ``{text, label}`` rows.

    **RouterBench is not `load_dataset`-able (B254).** The repo ships only
    ``routerbench_{0shot,5shot,raw}.pkl`` and `datasets` has no pickle reader, so any
    ``load_dataset(HF_ID, ...)`` raises ``DataFilesNotFoundError``. The file is fetched through
    ``hf_hub_download`` (cached under ``HF_HOME``) and read with ``pandas.read_pickle``.

    The benchmark ships **no train/test split**, so one is made here: rows are partitioned by a
    stable hash of ``sample_id``, which keeps the split identical across runs and machines
    without depending on row order.
    """
    import hashlib

    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_ID, PICKLE_FILE, repo_type="dataset")
    frame = pd.read_pickle(path)
    if small_model_key not in frame.columns:
        raise ValueError(
            f"RouterBench has no correctness column {small_model_key!r}; available candidate "
            f"models are {sorted(c for c in frame.columns if '|' not in c and c not in _META_COLUMNS)}"
        )
    log(f"      [routerbench] {len(frame)} rows; routing boundary = {small_model_key!r}")

    records = frame[["sample_id", "prompt", small_model_key]].to_dict("records")
    rows = convert_routerbench_rows(records, small_model_key=small_model_key)
    by_id = {r["text"]: rec for r, rec in zip(rows, records)}

    def is_test(row: dict) -> bool:
        # Deterministic ~20% holdout, stable across runs. Hashing the sample id (not the row
        # index) means the split does not move if the upstream file is ever reordered.
        sample_id = str(by_id.get(row["text"], {}).get("sample_id", row["text"]))
        return hashlib.sha256(sample_id.encode("utf-8")).digest()[0] < 51

    train = [r for r in rows if not is_test(r)][:max_train]
    test = [r for r in rows if is_test(r)][:max_test]
    log(f"      [routerbench] train={len(train)} test={len(test)}")
    if not test:
        raise RuntimeError(
            "RouterBench produced zero eval rows — refusing to proceed with an empty held-out "
            "set."
        )
    return train, test
