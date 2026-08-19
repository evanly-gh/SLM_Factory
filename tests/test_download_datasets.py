import hashlib
import json

import pytest

from scripts import download_datasets


def _spec(name):
    return download_datasets.DATASETS_BY_NAME[name]


def test_bc5cdr_converter_builds_exact_entity_spans():
    rows, labels = download_datasets._convert(
        [
            {
                "tokens": ["Aspirin", "can", "cause", "renal", "failure", "."],
                "tags": [1, 0, 0, 2, 3, 0],
            }
        ],
        _spec("bc5cdr"),
    )

    assert rows == [
        {
            "text": "Aspirin can cause renal failure .",
            "entities": [
                {"text": "Aspirin", "type": "Chemical"},
                {"text": "renal failure", "type": "Disease"},
            ],
        }
    ]
    assert labels == ["Chemical", "Disease"]


def test_gsm8k_converter_preserves_gold_reasoning_and_final_answer():
    """Shaped by the LIVE loader's own converter, not a copy of it.

    The bundle used to reproduce the `####` split itself, and the two drifted: the bundle wrote
    `label="math_reasoning"` and no `_instruction`, so a run served from the offline copy silently
    fell back to the "Answer the following question:" default while a run served from the loader
    got GSM8K's real instruction — the B250 failure mode.
    """
    from tasks.gsm8k import INSTRUCTION

    rows, labels = download_datasets._convert(
        [
            {
                "question": "How many widgets remain?",
                "answer": "Ten minus three is seven.\n#### 7",
            }
        ],
        _spec("gsm8k"),
    )

    assert rows == [
        {
            "text": "How many widgets remain?",
            "answer": "7",
            "cot_reasoning": "Ten minus three is seven.",
            "label": "gsm8k",
            "_instruction": INSTRUCTION,
        }
    ]
    assert labels == ["gsm8k"]


def test_samsum_converter_preserves_dialogue_summary_pair():
    from data.loaders.dialogsum_samsum import SUMMARIZATION_INSTRUCTION

    rows, labels = download_datasets._convert(
        [{"id": "chat-1", "dialogue": "A: Hi\nB: Hello", "summary": "A greets B."}],
        _spec("samsum"),
    )

    assert rows == [
        {
            "text": "A: Hi\nB: Hello",
            "answer": "A greets B.",
            "label": "generation",
            "_instruction": SUMMARIZATION_INSTRUCTION,
        }
    ]
    assert labels == ["generation"]


def test_manifest_records_schema_counts_provenance_and_eval_ban():
    spec = _spec("gsm8k")
    source = spec["sources"][0]
    manifest = download_datasets._build_manifest(
        spec,
        source,
        train_rows=[{"text": "train", "answer": "1", "label": "gsm8k"}],
        test_rows=[{"text": "test", "answer": "2", "label": "gsm8k"}],
        labels=["gsm8k"],
        removed_train_overlap=0,
        source_revision="abc123",
        content_hashes={"train.jsonl": "a" * 64, "test.jsonl": "b" * 64},
    )

    assert manifest["schema_version"] == 2
    # The manifest names the REGISTRY TASK the bundle supplies rows for. It used to name an
    # abstract `task_type`, which several bundles shared, so `load_local_dataset` could not tell
    # which task a bundle was actually for.
    assert manifest["task"] == "gsm8k"
    assert manifest["hf_id"] == "openai/gsm8k"
    assert manifest["config"] == "main"
    assert manifest["source_revision"] == "abc123"
    assert manifest["source_splits"] == {"train": "train", "test": "test"}
    assert manifest["counts"] == {"train": 1, "test": 1, "total": 2}
    assert manifest["provenance"]["official_splits_only"] is True
    assert manifest["provenance"]["records"][0]["split"] == "train"
    assert manifest["eval_ban"][0]["split"] == "test"
    assert manifest["overlap"] == {
        "field": "text",
        "normalization": "nfkc_casefold_whitespace_v1",
        "normalized_count": 0,
        "removed_from_train": 0,
    }
    assert manifest["integrity"]["files"]["train.jsonl"] == "a" * 64


def test_write_bundle_rejects_normalized_train_test_contamination(tmp_path):
    train = {"text": " Same\n  Prompt ", "answer": "gold", "label": "generation"}
    test = {"text": "same prompt", "answer": "gold", "label": "generation"}

    with pytest.raises(ValueError, match="normalized train/test text overlap"):
        download_datasets._write_bundle(
            tmp_path,
            _spec("samsum"),
            _spec("samsum")["sources"][0],
            [train],
            [test],
            labels=["generation"],
        )


def test_normalized_decontamination_keeps_official_test_rows():
    """Decontamination is `dataset_integrity.remove_normalized_train_overlap` called directly.

    The local `_drop_train_overlap` wrapper existed only to run APPS's URL/solution fingerprint
    pass first; with APPS gone it was a second name for one function, which is the kind of
    duplicate that drifts.
    """
    from data.loaders.dataset_integrity import remove_normalized_train_overlap

    train = [
        {"text": "unique", "label": "generation"},
        {"text": " Duplicate\n Text ", "label": "generation"},
    ]
    test = [{"text": "duplicate text", "label": "generation"}]

    clean, removed = remove_normalized_train_overlap(train, test)

    assert clean == [train[0]]
    assert removed == 1
    assert test == [{"text": "duplicate text", "label": "generation"}]


def test_write_bundle_creates_hashed_jsonl_and_manifest(tmp_path):
    train = [{"text": "train dialogue", "answer": "train summary", "label": "generation"}]
    test = [{"text": "test dialogue", "answer": "test summary", "label": "generation"}]

    manifest = download_datasets._write_bundle(
        tmp_path,
        _spec("samsum"),
        _spec("samsum")["sources"][0],
        train,
        test,
        labels=["generation"],
        removed_train_overlap=42,
        source_revision="deadbeef",
    )

    bundle = tmp_path / "samsum"
    assert json.loads((bundle / "train.jsonl").read_text()) == train[0]
    assert json.loads((bundle / "test.jsonl").read_text()) == test[0]
    assert json.loads((bundle / "manifest.json").read_text()) == manifest
    assert manifest["hf_id"] == "knkarthick/samsum"
    assert manifest["overlap"]["removed_from_train"] == 42
    assert manifest["source_revision"] == "deadbeef"
    checksums = {}
    for line in (bundle / "checksums.sha256").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        checksums[filename] = digest
    assert set(checksums) == {"train.jsonl", "test.jsonl", "manifest.json"}
    for filename, expected in checksums.items():
        actual = hashlib.sha256((bundle / filename).read_bytes()).hexdigest()
        assert actual == expected
    assert manifest["integrity"]["files"] == {
        "train.jsonl": checksums["train.jsonl"],
        "test.jsonl": checksums["test.jsonl"],
    }


def test_cached_hf_revision_reads_hub_ref_without_network(tmp_path, monkeypatch):
    ref = tmp_path / "datasets--openai--gsm8k" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("740312add88f781978c0658806c59bc2815b9866\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    assert (
        download_datasets._cached_hf_revision("openai/gsm8k")
        == "740312add88f781978c0658806c59bc2815b9866"
    )


def test_only_cli_downloads_selected_bundles(tmp_path, monkeypatch):
    downloaded = []

    def fake_download(spec, output_dir, load_dataset_fn=None):
        downloaded.append((spec["name"], output_dir))

    monkeypatch.setattr(download_datasets, "_download_dataset", fake_download)
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    assert download_datasets.main(["--only", "gsm8k", "bc5cdr", "samsum"]) == 0
    assert downloaded == [
        ("gsm8k", tmp_path),
        ("bc5cdr", tmp_path),
        ("samsum", tmp_path),
    ]


def test_only_cli_accepts_comma_separated_names_and_rejects_unknown(tmp_path, monkeypatch):
    downloaded = []
    monkeypatch.setattr(
        download_datasets,
        "_download_dataset",
        lambda spec, output_dir, load_dataset_fn=None: downloaded.append(spec["name"]),
    )
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    assert download_datasets.main(["--only", "bc5cdr,samsum"]) == 0
    assert downloaded == ["bc5cdr", "samsum"]
    with pytest.raises(SystemExit):
        download_datasets.main(["--only", "not-a-dataset"])


def test_list_cli_reports_supported_task_aware_catalog(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SLM_LOCAL_DATASET_DIR", str(tmp_path))

    assert download_datasets.main(["--list"]) == 0
    listed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    by_name = {item["name"]: item for item in listed}

    # Every catalogued bundle names a REGISTRY TASK. A bundle whose `task` is not a registry key
    # is unreachable — `load_local_dataset` matches on it — which is why the APPS, MBPP, emotion
    # and go_emotions bundles went with the channels they belonged to.
    from tasks import TASKS

    assert set(by_name) == {"bc5cdr", "gsm8k", "samsum"}
    assert by_name["bc5cdr"]["task"] == "ner_bc5cdr"
    assert by_name["gsm8k"]["task"] == "gsm8k"
    assert by_name["samsum"]["task"] == "dialogsum"
    assert all(item["task"] in TASKS for item in listed)
    assert by_name["samsum"]["installed"] is False
