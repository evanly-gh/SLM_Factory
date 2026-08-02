# Data-Source Provenance Logging — Design

**Date:** 2026-08-01
**Branch:** `data-curation-redesign`
**Author:** brainstormed with Evan

## Goal

Every time data curation pulls in **new external data** (web search / research — the `acquire`
strategy's Exa web-scrape and HuggingFace-dataset discovery, plus the initial `eval_setup`
acquisition), record **where the data came from (link)** and **how much (row counts)**:

1. In the **per-iteration curation log** (`data-curation.md`, written every curate).
2. In the **final run report** — both a machine-readable `artifacts/data_sources.json` and a
   printed block that lands in the run's tee'd stdout / SLURM log under `logs/`.

Granularity (decided): **per-source row counts + links**, not row-level identities. Counts always
sum to the dataset total (non-web origins — `resample of existing pool`, `synthesized (local
Qwen)` — are shown as unlinked rows so the arithmetic closes).

The "Data sources" section appears **only when new external data was actually fetched this build**
(the `acquire` strategy, or any row carrying an external source tag). Pure `resample` / `synthesize`
builds omit it.

## Current state (verified against code)

- **Links already captured** for HF / benchmark / local paths: `source_records[].url` / `.id`
  (`data/loaders/web_acquire.py:217-237`, `_source_records`). They survive into
  `state["data_sources"]` (appended by `eval_setup.py:304-313` and `curate.py:601-605`, accumulates
  run-wide) and into `last_curation["source_novelty"]["source_records"]`.
- **Per-row origin tags already exist** on mined rows: `_source` (a `kind:id/split` string) and
  `_source_record` (the full record incl. `url`), set in `mine_additional_real_rows.accept()`
  (`web_acquire.py:640-657`) and preserved by `_tag_mined_rows` (`curate.py:278-286`).
- **Per-source counts are derivable but not surfaced:** `last_curation["source_composition"]`
  already `Counter`s per-row `_source` (`curate.py:733-736`), but it is keyed by `kind:id/split`
  (not URL), is never joined to the URL-bearing records, and is **never passed to the curation log**
  (call site `evaluate.py:485-491` forwards only `strategy_composition`, `source_novelty`,
  `plan_yield`).
- **Two real gaps:**
  1. The raw Exa web-scrape fallback `_exa_round` discards `x.url` (`web_acquire.py:804-812`), so
     scraped web docs carry no link and no per-row tag.
  2. The final run report aggregates **no** provenance. `state["data_sources"]` is the only run-wide
     structure but holds records **without counts**, and the run-summary section of
     `tests/pipeline/run.py` never reads it.

## Design

The organizing principle: **make every acquisition path emit the same per-row `_source` /
`_source_record` tags (with a `url`)**, so the existing counting + lineage machinery works uniformly
and both logs become a join of *counts × URLs*. No per-path special-casing downstream.

### Unit 1 — `_exa_round` URL capture (`data/loaders/web_acquire.py`)

Tag each accepted scraped doc with its source URL so it flows through the same pipeline as HF rows.

- In `_exa_round` (`web_acquire.py:804-812`), attach to each emitted row:
  `_source = "web:" + registrable_domain_or_url`, and
  `_source_record = {"kind": "web", "id": <domain>, "url": x.url, "split": "web", "role": "curriculum"}`.
- Rows are grouped by URL downstream by the existing `_source_key` / `source_composition` counter, so
  each distinct page becomes one counted, linked source.
- `mine_additional_real_rows` already spreads existing row keys when tagging
  (`_tag_mined_rows`, `**row`), so these tags survive into `curate`.
- No new count field is needed in `source_records`: counts come from the per-row `Counter`.

### Unit 2 — `build_source_usage` (new pure helper)

A single, independently testable function (home: `data/provenance.py`, a new small module — keeps
`curate.py` from growing another responsibility):

```python
def build_source_usage(dataset, *, source_records, origin_novelty=None, default="existing pool") -> list[dict]:
    """Join per-row source tags (counts) with URL-bearing records (links).
    Returns [{"source","url","split","role","rows","novel_rows"}, ...] sorted by rows desc,
    counts summing to len(dataset). Untagged rows fall into `default` (no url)."""
```

- Counts come from `Counter(_source_key(row) for row in dataset)` (reuse `curate._source_key`).
- URLs/splits/roles come from `source_records` (run lineage + this build's mining records), matched
  on `kind:id/split`. A row's own `_source_record["url"]` is used when present (covers Exa web rows
  whose URL isn't in `source_records`).
- `novel_rows` per source from the mining report where available; else omitted.
- Non-external origins (`resample`, `synthesize`, untagged initial pool) appear as unlinked entries
  so the list is exhaustive.

Called in `curate_node`, result stored on `last_curation["source_usage"]` and also appended to the
run-wide accumulator (Unit 4).

### Unit 3 — per-curate log section (`data/curation_log.py`)

- Add `source_usage: list[dict] | None = None` and `dataset_version` (already passed) to
  `write_iteration`.
- Render a new section, emitted **only** when `source_usage` contains ≥1 entry with a `url`
  (i.e. external data was fetched this build):

```
### Data sources (iteration 7, dataset v7)
- https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k (train) — 240 rows (novel: 210)
- https://site-a.example/faq (web) — 8 rows
- resample of existing pool — 2760 rows
- synthesized (local Qwen) — 500 rows
```

- Thread `source_usage` through the existing call site `agent/nodes/evaluate.py:485-491`
  (read from `state["last_curation"]["source_usage"]`).

### Unit 4 — run-wide aggregation + final report

- **New run-durable state field** `data_source_usage: list[dict]` in `agent/state.py`, initialized
  `[]` in the run-state template (`tests/pipeline/run.py:~639`), appended in `curate_node` as a
  **new list** (`state["data_source_usage"] = list(old) + [entry]`) — not in-place, to survive the
  channel-persistence issue that froze `scores`/`dag` (B122). Each entry:
  `{"iteration", "dataset_version", "strategy", "sources": <source_usage>}`.
- **`artifacts/data_sources.json`** written in the run-summary phase of `tests/pipeline/run.py`
  (next to `scores.json`/`baselines.json`, ~`run.py:1014-1047`), aggregated by URL across the run:
  `[{"source","url","total_rows","total_novel_rows","iterations":[…],"dataset_versions":[…]}]`,
  seeded from `state["data_sources"]` (so the initial `eval_setup` acquisition is included even if
  its rows weren't per-row tagged) and summed from `state["data_source_usage"]`.
- **Printed block** in the human-readable summary near the cost block (`run.py:~1225-1240`) — goes to
  stdout, captured by the phase-1 tee logger into the run log / SLURM `--output` under `logs/`:

```
=== Data sources used this run (links + row counts) ===
  https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k — 480 rows (novel 300) [iters 1,7]
  https://site-a.example/faq — 8 rows [iter 7]
  (resample/synth/local rows are per-iteration; see artifacts/data_sources.json)
```

## Components / data flow

```
_exa_round ─┐
mine_*      ─┼─► rows tagged _source/_source_record(url)
eval_setup  ─┘            │
                          ▼
                curate_node ── build_source_usage(dataset, source_records)
                          │        └─► last_curation["source_usage"]
                          ├─► state["data_source_usage"] += [entry]   (run-wide, durable)
                          └─► data_sources (unique URL lineage; unchanged)
                          ▼
   evaluate ─► CurationLog.write_iteration(..., source_usage) ─► data-curation.md "### Data sources"
                          ▼
   run.py summary ─► artifacts/data_sources.json  +  printed block → tee'd SLURM log
```

## Error handling / edge cases

- **No external data this build** (resample/synthesize only): section + JSON entry omitted / marked
  non-external; counts still recorded internally but no "Data sources" section is printed.
- **Untagged initial pool** rows collapse to `existing pool` (no link) in per-curate; the run-level
  report still surfaces the initial `eval_setup` sources via `state["data_sources"]`.
- **Firewall:** provenance carries URLs/ids/counts only — never raw eval text. `build_source_usage`
  emits no row text, so no new leak surface.
- **Counts must sum** to `len(dataset)`; a unit test asserts this invariant.
- Missing/malformed `source_records` → entries fall back to the row's own `_source_record["url"]`,
  else appear unlinked. Never raises inside curate/summary (provenance is best-effort logging).

## Testing

- `build_source_usage`: counts sum to total; URL join by `kind:id/split`; row-level `url` fallback;
  non-external origins included; empty/edge inputs.
- `_exa_round`: accepted rows carry `_source_record["url"]`; grouping by URL yields per-page counts.
- `CurationLog.write_iteration`: renders the section when external sources present; omits it
  otherwise; no raw text leak.
- Run-level aggregation: by-URL sum across multiple iterations/dataset versions; JSON shape;
  `data_source_usage` appended as a new list.

## Out of scope (YAGNI)

- Row-level provenance manifest (per-row hash → URL sidecar) — explicitly declined.
- Tagging every initial `acquire_dataset` row with `_source_record` — optional future enhancement so
  the *first* curriculum's rows carry links per-row; the run-level report already covers those
  sources via `state["data_sources"]`, so not required now.
- Any change to what the orchestrator sees at `iterate` (provenance is logging only).

## Files touched

| File | Change |
|---|---|
| `data/loaders/web_acquire.py` | Unit 1: tag Exa-scraped rows with `_source`/`_source_record(url)` |
| `data/provenance.py` (new) | Unit 2: `build_source_usage` |
| `agent/nodes/curate.py` | call `build_source_usage`; store on `last_curation`; append `data_source_usage` |
| `data/curation_log.py` | Unit 3: new `### Data sources` section |
| `agent/nodes/evaluate.py` | thread `source_usage` into `write_iteration` |
| `agent/state.py` | add `data_source_usage: list[dict]` |
| `tests/pipeline/run.py` | init field; write `artifacts/data_sources.json`; printed block |
| `tests/…` | unit tests per the Testing section |
