"""The per-label logprob pass that makes a threshold-free metric computable.

The arithmetic here is easy to get subtly wrong and impossible to notice afterwards: an
off-by-one in the logits shift, or padding on the wrong side, produces scores that are still
finite, still ordered plausibly, and still yield an AUPRC somebody would put in a table. So these
tests check the numbers against an independent computation rather than checking that the function
returns floats.

Driven through a hand-built stub model, not a real checkpoint. The thing under test is index
arithmetic over logits, and a real model would make the expected values unknowable — which is
exactly the position that lets a shift bug survive.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from training.slm_helpers import _label_score_rows, _score_label_batch


class _StubTokenizer:
    """Character-level tokenizer: id = ord(char). Deterministic and trivially invertible."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


class _StubModel:
    """Returns fixed logits so the expected logprob of any sequence is computable by hand.

    Every position emits the same distribution: `favoured` gets logit 1.0 and everything else
    0.0. That makes the per-token logprob of a favoured token differ from an unfavoured one by a
    known amount, so a shift error moves the result away from both.
    """

    device = None
    vocab = 256

    def __init__(self, favoured: int):
        self.favoured = favoured

    def __call__(self, input_ids, attention_mask):
        batch, width = input_ids.shape
        logits = torch.zeros(batch, width, self.vocab)
        logits[:, :, self.favoured] = 1.0
        return type("Out", (), {"logits": logits})()


def _expected_mean_logprob(label: str, favoured: int) -> float:
    """The same number computed independently of the batching, padding and shift code."""
    favoured_logprob = 1.0 - math.log(math.exp(1.0) + 255 * math.exp(0.0))
    other_logprob = 0.0 - math.log(math.exp(1.0) + 255 * math.exp(0.0))
    per_token = [
        favoured_logprob if ord(c) == favoured else other_logprob for c in label
    ]
    return sum(per_token) / len(per_token)


def test_the_scored_span_is_the_label_and_not_the_prompt():
    """A shift or slice error would fold prompt tokens into the label's score.

    The two labels here differ only in whether their characters are the favoured token, so the
    expected values are far apart. Scoring the prompt as well would drag both toward the same
    number, which is the shape a boundary bug takes.
    """
    tokenizer = _StubTokenizer()
    favoured = ord("a")
    model = _StubModel(favoured)

    rows = _label_score_rows(tokenizer, "PROMPT TEXT", ["aaa", "zzz"])
    scores = _score_label_batch(model, tokenizer, rows, torch)

    assert scores[0] == pytest.approx(_expected_mean_logprob("aaa", favoured), abs=1e-5)
    assert scores[1] == pytest.approx(_expected_mean_logprob("zzz", favoured), abs=1e-5)
    assert scores[0] > scores[1]


def test_padding_a_short_label_beside_a_long_one_does_not_change_either_score():
    """Right padding must be masked out of the scored span.

    Labels of different token lengths share a batch on every real call — GoEmotions has both
    `joy` and `disappointment` — so if padding leaked into the slice, the score of every short
    label would depend on which other labels happened to be in its chunk.
    """
    tokenizer = _StubTokenizer()
    favoured = ord("a")
    model = _StubModel(favoured)

    alone = _score_label_batch(
        model, tokenizer, _label_score_rows(tokenizer, "P", ["aa"]), torch,
    )
    together = _score_label_batch(
        model, tokenizer, _label_score_rows(tokenizer, "P", ["aa", "aaaaaaaaaaaa"]), torch,
    )
    assert together[0] == pytest.approx(alone[0], abs=1e-6)


def test_the_score_is_length_normalized_so_labels_are_comparable_to_each_other():
    """A raw sum would make a one-token label beat a ten-token one for free.

    This cannot change any label's average precision — AP ranks EXAMPLES for a fixed label, and
    the divisor is constant per label — but unnormalized scores would be meaningless to read side
    by side in the per-label diagnostics, which is what they are published for.
    """
    tokenizer = _StubTokenizer()
    favoured = ord("a")
    model = _StubModel(favoured)

    rows = _label_score_rows(tokenizer, "P", ["a", "aaaaaaaa"])
    short, long = _score_label_batch(model, tokenizer, rows, torch)
    assert short == pytest.approx(long, abs=1e-6)


def test_a_label_that_tokenizes_to_nothing_scores_negative_infinity():
    """It must not average over an empty slice and it must not silently rank mid-field.

    -inf sorts last in `average_precision`'s ranking, which is the honest position for a label the
    model was never actually asked about.
    """
    tokenizer = _StubTokenizer()
    model = _StubModel(ord("a"))

    rows = _label_score_rows(tokenizer, "P", ["", "a"])
    scores = _score_label_batch(model, tokenizer, rows, torch)
    assert scores[0] == float("-inf")
    assert math.isfinite(scores[1])


def test_label_rows_carry_the_prompt_prefix_exactly_once():
    """`add_special_tokens=False` on the label half; the prompt is already fully rendered."""
    tokenizer = _StubTokenizer()
    rows = _label_score_rows(tokenizer, "abc", ["de"])
    ids, n_label = rows[0]
    assert ids == [ord("a"), ord("b"), ord("c"), ord("d"), ord("e")]
    assert n_label == 2


def test_the_isolated_worker_can_dispatch_the_label_scoring_operation():
    """A CUDA-holding operation that the worker cannot dispatch would fail only under isolation,
    which is how every pipeline run is configured (`SLM_CUDA_ISOLATION=1`)."""
    from pathlib import Path

    source = Path("training/cuda_worker.py").read_text(encoding="utf-8")
    assert 'operation == "infer_label_scores"' in source
    assert "infer_label_scores_batch(**payload)" in source
