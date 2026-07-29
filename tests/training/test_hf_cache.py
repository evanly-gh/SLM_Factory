import json

import pytest

from training.hf_cache import verify_hf_model_snapshot


def _write_weights(snapshot):
    (snapshot / "model.safetensors").write_bytes(b"weights")


def test_snapshot_accepts_qwen_bpe_tokenizer(tmp_path):
    snapshot = tmp_path / "qwen-bpe"
    snapshot.mkdir()
    _write_weights(snapshot)
    (snapshot / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE"}}),
        encoding="utf-8",
    )
    (snapshot / "vocab.json").write_text(
        json.dumps({"hello": 0, "world": 1}),
        encoding="utf-8",
    )
    (snapshot / "merges.txt").write_text(
        "#version: 0.2\nh e\n",
        encoding="utf-8",
    )

    verify_hf_model_snapshot(snapshot)


def test_snapshot_accepts_sentencepiece_tokenizer(tmp_path):
    snapshot = tmp_path / "sentencepiece"
    snapshot.mkdir()
    _write_weights(snapshot)
    (snapshot / "tokenizer.model").write_bytes(b"sentencepiece-protobuf")

    verify_hf_model_snapshot(snapshot)


def test_snapshot_rejects_missing_tokenizer_representation(tmp_path):
    snapshot = tmp_path / "missing-tokenizer"
    snapshot.mkdir()
    _write_weights(snapshot)

    with pytest.raises(
        ValueError,
        match=(
            r"snapshot is incomplete; no valid tokenizer representation.*"
            r"tokenizer\.json.*tokenizer\.model.*vocab\.json.*merges\.txt"
        ),
    ):
        verify_hf_model_snapshot(snapshot)
