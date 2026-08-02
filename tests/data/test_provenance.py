"""Tests for data/provenance.py — data-source provenance logging (2026-08-01)."""
from data.provenance import (
    aggregate_data_sources,
    build_source_usage,
    format_run_data_sources,
    source_key,
)


def test_source_key_prefers_explicit_source_tag():
    assert source_key({"_source": "hf:X/train"}) == "hf:X/train"


def test_source_key_derives_from_record_when_no_tag():
    row = {"_source_record": {"kind": "hf", "id": "X", "split": "train"}}
    assert source_key(row) == "hf:X/train"


def test_source_key_falls_back_to_default():
    assert source_key({}, default="existing pool") == "existing pool"


def test_build_source_usage_joins_counts_with_urls():
    dataset = [
        {"_source": "hf:Salesforce/xlam/train",
         "_source_record": {"kind": "hf", "id": "Salesforce/xlam", "split": "train",
                            "url": "https://huggingface.co/datasets/Salesforce/xlam",
                            "role": "curriculum"}},
        {"_source": "hf:Salesforce/xlam/train",
         "_source_record": {"kind": "hf", "id": "Salesforce/xlam", "split": "train",
                            "url": "https://huggingface.co/datasets/Salesforce/xlam"}},
    ]
    usage = build_source_usage(dataset)
    assert len(usage) == 1
    entry = usage[0]
    assert entry["source"] == "hf:Salesforce/xlam/train"
    assert entry["url"] == "https://huggingface.co/datasets/Salesforce/xlam"
    assert entry["rows"] == 2


def test_build_source_usage_uses_source_records_for_url():
    dataset = [{"_source": "hf:X/train"}]
    records = [{"kind": "hf", "id": "X", "split": "train",
                "url": "https://huggingface.co/datasets/X", "role": "curriculum"}]
    usage = build_source_usage(dataset, source_records=records)
    assert usage[0]["url"] == "https://huggingface.co/datasets/X"
    assert usage[0]["role"] == "curriculum"


def test_build_source_usage_untagged_rows_are_existing_pool_without_url():
    dataset = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    usage = build_source_usage(dataset, default="existing pool")
    assert len(usage) == 1
    assert usage[0]["source"] == "existing pool"
    assert usage[0]["url"] is None
    assert usage[0]["rows"] == 3


def test_build_source_usage_counts_sum_to_dataset_size():
    dataset = (
        [{"_source": "hf:X/train"}] * 4
        + [{"_source": "web:site.com"}] * 2
        + [{"text": "untagged"}] * 3
    )
    usage = build_source_usage(dataset)
    assert sum(e["rows"] for e in usage) == len(dataset)


def test_build_source_usage_sorted_by_rows_desc():
    dataset = (
        [{"_source": "web:small.com"}] * 1
        + [{"_source": "hf:big/train"}] * 5
    )
    usage = build_source_usage(dataset)
    assert [e["source"] for e in usage] == ["hf:big/train", "web:small.com"]


def test_build_source_usage_novel_rows_from_map():
    dataset = [{"_source": "hf:X/train"}] * 3
    usage = build_source_usage(dataset, novel_by_source={"hf:X/train": 2})
    assert usage[0]["novel_rows"] == 2


def test_build_source_usage_row_level_record_url_when_records_absent():
    # Exa web-scrape case: no aggregate source_records, url lives on the row.
    dataset = [{"_source": "web:site.example",
                "_source_record": {"kind": "web", "id": "site.example",
                                   "url": "https://site.example/faq", "split": "web"}}]
    usage = build_source_usage(dataset)
    assert usage[0]["url"] == "https://site.example/faq"


def test_has_external_source_via_url():
    external = build_source_usage([{"_source": "hf:X/train",
                                    "_source_record": {"kind": "hf", "id": "X",
                                                       "split": "train", "url": "https://h/X"}}])
    internal = build_source_usage([{"text": "a"}])
    assert any(e.get("url") for e in external)
    assert not any(e.get("url") for e in internal)


def test_aggregate_data_sources_sums_across_iterations_by_url():
    usage_log = [
        {"iteration": 1, "dataset_version": "v1", "strategy": "acquire",
         "sources": [{"source": "hf:X/train", "url": "https://h/X", "rows": 100,
                      "novel_rows": 100, "role": "curriculum"}]},
        {"iteration": 7, "dataset_version": "v7", "strategy": "acquire",
         "sources": [{"source": "hf:X/train", "url": "https://h/X", "rows": 40,
                      "novel_rows": 10, "role": "curriculum"},
                     {"source": "existing pool", "url": None, "rows": 60,
                      "novel_rows": 0}]},
    ]
    agg = aggregate_data_sources(usage_log)
    by_url = {a.get("url"): a for a in agg}
    assert by_url["https://h/X"]["total_rows"] == 140
    assert by_url["https://h/X"]["total_novel_rows"] == 110
    assert by_url["https://h/X"]["iterations"] == [1, 7]
    assert by_url["https://h/X"]["dataset_versions"] == ["v1", "v7"]


def test_aggregate_data_sources_orders_external_first():
    usage_log = [{"iteration": 1, "dataset_version": "v1", "strategy": "acquire",
                  "sources": [{"source": "existing pool", "url": None, "rows": 500},
                              {"source": "hf:X/train", "url": "https://h/X", "rows": 50}]}]
    agg = aggregate_data_sources(usage_log)
    assert agg[0]["url"] == "https://h/X"  # linked source first despite fewer rows


def test_format_run_data_sources_renders_links_and_counts():
    agg = [{"source": "hf:X/train", "url": "https://h/X", "total_rows": 140,
            "total_novel_rows": 110, "iterations": [1, 7], "dataset_versions": ["v1", "v7"]}]
    block = format_run_data_sources(agg)
    assert "https://h/X" in block
    assert "140" in block
    assert "Data sources used this run" in block


def test_format_run_data_sources_empty_is_falsy_marker():
    block = format_run_data_sources([])
    assert "No external data sources" in block
