"""The length-outlier bound is anchored on trusted rows, not on the contaminated dataset (B260).

The filter drops rows longer than 3x the median. Because the bound is RELATIVE, whatever sets the
median sets the cutoff — so injecting short foreign rows made it delete the real benchmark data it
exists to protect. Measured on RouterBench: real rows have a median of 715 characters, the mined
foreign rows 75; the dataset median fell to 269 and the cutoff from 2145 to 807, a bound that removes
48% of the real benchmark instead of 0.6%.
"""
from data.curriculum import _filter_length_outliers, apply_quality_controls

# RouterBench-shaped: real prompts are long and vary a lot, mined foreign rows are short.
REAL_LENGTHS = [200, 400, 715, 715, 900, 1200, 1800, 2000]
MINED_LENGTH = 75


def _real(n_chars, i=0):
    return {"text": "x" * n_chars, "label": "route", "_provenance": "train_anchor"}


def _mined(i=0):
    return {"text": "y" * MINED_LENGTH, "label": "local", "_provenance": "mined_real"}


def test_short_mined_rows_cannot_pull_the_cutoff_down_onto_real_data():
    real = [_real(n, i) for i, n in enumerate(REAL_LENGTHS)]
    contaminated = real + [_mined(i) for i in range(200)]

    kept = _filter_length_outliers(contaminated)
    kept_real = [r for r in kept if r["_provenance"] == "train_anchor"]

    # Every real row survives: the median is taken over real rows (715), so the cutoff is ~2145 and
    # only a genuine outlier relative to the TASK's own distribution would be cut.
    assert len(kept_real) == len(real), (
        "mined rows moved the goalposts and deleted real data"
    )


def test_the_old_behaviour_really_would_have_deleted_them():
    """Guard the guard: without provenance tags the dataset median wins and real rows die.

    This is what the RouterBench runs actually did, and it is the reason the fix is anchoring rather
    than raising the 3x ratio.
    """
    untagged = (
        [{"text": "x" * n, "label": "route"} for n in REAL_LENGTHS]
        + [{"text": "y" * MINED_LENGTH, "label": "local"} for _ in range(200)]
    )
    kept = _filter_length_outliers(untagged)
    long_rows_kept = [r for r in kept if len(r["text"]) > 3 * MINED_LENGTH]
    assert long_rows_kept == [], "expected the contaminated median to delete the long real rows"


def test_genuine_outliers_are_still_removed():
    """The filter must keep working — anchoring is not the same as disabling."""
    real = [_real(n) for n in (700, 710, 715, 720, 730)]
    outlier = _real(50_000)
    kept = _filter_length_outliers(real + [outlier])
    assert len(kept) == len(real)
    assert all(len(r["text"]) < 50_000 for r in kept)


def test_falls_back_to_all_rows_when_nothing_is_tagged():
    rows = [{"text": "x" * n} for n in (10, 10, 10, 10_000)]
    kept = _filter_length_outliers(rows)
    assert len(kept) == 3


def test_empty_input_is_safe():
    assert _filter_length_outliers([]) == []


def test_end_to_end_through_quality_controls():
    """The same property via the real entry point, with the label-space filter also active."""
    real = [
        {"text": "x" * n, "label": lab, "_provenance": "train_anchor"}
        for n in REAL_LENGTHS for lab in ("local", "route")
    ]
    mined = [
        {"text": f"y{i}" + "y" * MINED_LENGTH, "label": "local", "_provenance": "mined_real"}
        for i in range(100)
    ]
    kept = apply_quality_controls(
        real + mined, task_type="classification", allowed_labels={"local", "route"}
    )
    survived_long = [r for r in kept if len(r["text"]) >= 1800]
    assert survived_long, "long real rows must survive QC on a contaminated dataset"
