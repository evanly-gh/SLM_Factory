# data/provenance.py
"""Data-source provenance: join per-row source tags (counts) with source records (links).

Used by curate (per-iteration `data-curation.md` section) and the run summary
(`artifacts/data_sources.json` + a printed block that lands in the tee'd SLURM log). The
firewall guarantee is preserved: only source ids/urls/counts are emitted, never row text.
"""
from collections import Counter


def source_key(row: dict, default: str = "unknown") -> str:
    """Collapse a row to its source key: `_source` tag, else `kind:id/split`, else default."""
    value = row.get("_source")
    if value:
        return str(value)
    record = row.get("_source_record")
    if isinstance(record, dict):
        return (
            f"{record.get('kind', 'source')}:{record.get('id', '?')}"
            f"/{record.get('split', 'train')}"
        )
    return default


def _records_url_lookup(source_records) -> dict:
    """Map `kind:id/split` -> {url, split, role} from a list of source records."""
    lookup: dict = {}
    for record in source_records or []:
        if not isinstance(record, dict):
            continue
        key = (
            f"{record.get('kind', 'source')}:{record.get('id', '?')}"
            f"/{record.get('split', 'train')}"
        )
        lookup[key] = {
            "url": record.get("url"),
            "split": record.get("split"),
            "role": record.get("role"),
        }
    return lookup


def build_source_usage(
    dataset,
    *,
    source_records=None,
    novel_by_source=None,
    default: str = "existing pool",
) -> list[dict]:
    """One source-usage list for a single build: counts joined to links.

    Returns [{source, url, split, role, rows, novel_rows}, ...] sorted by rows desc (then
    source asc). Counts sum to len(dataset); untagged rows collapse to `default` with no url.
    URLs come from `source_records` first, then the row's own `_source_record["url"]`.
    """
    record_lookup = _records_url_lookup(source_records)
    novel_by_source = novel_by_source or {}

    counts: Counter = Counter()
    # First-seen url/split/role carried on a row, used when source_records lacks the key.
    row_meta: dict = {}
    for row in dataset:
        key = source_key(row, default=default)
        counts[key] += 1
        if key not in row_meta:
            record = row.get("_source_record")
            if isinstance(record, dict):
                row_meta[key] = {
                    "url": record.get("url"),
                    "split": record.get("split"),
                    "role": record.get("role"),
                }

    usage: list[dict] = []
    for key, rows in counts.items():
        meta = record_lookup.get(key) or row_meta.get(key) or {}
        usage.append({
            "source": key,
            "url": meta.get("url"),
            "split": meta.get("split"),
            "role": meta.get("role"),
            "rows": rows,
            "novel_rows": int(novel_by_source.get(key, 0)),
        })
    usage.sort(key=lambda e: (-e["rows"], e["source"]))
    return usage


def aggregate_data_sources(data_source_usage, base_records=None) -> list[dict]:
    """Aggregate per-build usage across the whole run, grouped by url (else source key).

    `data_source_usage` is the run-wide list of {iteration, dataset_version, strategy,
    sources:[...]}. `base_records` (e.g. state["data_sources"]) seeds linked sources that may
    have contributed before per-row tagging existed (the initial eval_setup acquisition).
    Linked (url-bearing) sources are ordered first, then by total rows desc.
    """
    agg: dict = {}

    def _slot(key, url, source, split=None, role=None):
        if key not in agg:
            agg[key] = {
                "source": source,
                "url": url,
                "split": split,
                "role": role,
                "total_rows": 0,
                "total_novel_rows": 0,
                "iterations": [],
                "dataset_versions": [],
            }
        return agg[key]

    for record in base_records or []:
        if not isinstance(record, dict) or not record.get("url"):
            continue
        key = record["url"]
        _slot(key, record.get("url"),
              f"{record.get('kind', 'source')}:{record.get('id', '?')}"
              f"/{record.get('split', 'train')}",
              split=record.get("split"), role=record.get("role"))

    for build in data_source_usage or []:
        iteration = build.get("iteration")
        version = build.get("dataset_version")
        for entry in build.get("sources") or []:
            key = entry.get("url") or entry.get("source")
            slot = _slot(key, entry.get("url"), entry.get("source"),
                         split=entry.get("split"), role=entry.get("role"))
            slot["total_rows"] += int(entry.get("rows", 0) or 0)
            slot["total_novel_rows"] += int(entry.get("novel_rows", 0) or 0)
            if iteration is not None and iteration not in slot["iterations"]:
                slot["iterations"].append(iteration)
            if version is not None and version not in slot["dataset_versions"]:
                slot["dataset_versions"].append(version)

    out = list(agg.values())
    out.sort(key=lambda a: (a.get("url") is None, -a["total_rows"], str(a["source"])))
    return out


def format_run_data_sources(aggregated) -> str:
    """Render the aggregated run-wide sources as a printed summary block (SLURM log)."""
    linked = [a for a in aggregated if a.get("url")]
    if not aggregated or not linked:
        return "=== Data sources used this run ===\n  No external data sources (no web/HF acquisition)."
    lines = ["=== Data sources used this run (links + row counts) ==="]
    for a in aggregated:
        label = a.get("url") or a.get("source")
        iters = ",".join(str(i) for i in a.get("iterations", []))
        novel = a.get("total_novel_rows", 0)
        suffix = f" (novel {novel})" if novel else ""
        where = f" [iters {iters}]" if iters else ""
        lines.append(f"  {label} — {a.get('total_rows', 0)} rows{suffix}{where}")
    return "\n".join(lines)
