"""Exa web-scrape rows must carry their source URL for provenance logging (2026-08-01)."""
import data.loaders.web_acquire as wa


class _Hit:
    def __init__(self, url, text, title=""):
        self.url = url
        self.text = text
        self.title = title


class _Result:
    def __init__(self, hits):
        self.results = hits


def test_exa_round_tags_rows_with_source_url(monkeypatch):
    long_text = "This is a sufficiently long scraped document. " * 6  # > 180 chars
    hit = _Hit(url="https://site.example/faq", text=long_text, title="FAQ")
    monkeypatch.setattr(wa, "_exa_search", lambda exa, query, n: _Result([hit]))

    rows = wa._exa_round(
        exa=object(),
        task_type="classification",
        plan={"labels": ["spam"], "exa_queries": {"spam": "spam example texts"}},
        description="spam detection",
        n_per_label=5,
        round_idx=0,
        seen_texts=set(),
    )

    assert rows, "expected at least one scraped row"
    row = rows[0]
    assert row["_source_record"]["url"] == "https://site.example/faq"
    assert row["_source_record"]["kind"] == "web"
    assert row["_source"].startswith("web:")


def test_exa_round_groups_rows_by_page_via_source_key(monkeypatch):
    from data.provenance import build_source_usage

    long_a = "Document A content that is long enough to be useful here. " * 5
    long_b = "Document B content that is long enough to be useful here. " * 5
    hits = [
        _Hit("https://a.example/x", long_a, "A"),
        _Hit("https://b.example/y", long_b, "B"),
    ]
    monkeypatch.setattr(wa, "_exa_search", lambda exa, query, n: _Result(hits))

    rows = wa._exa_round(
        exa=object(),
        task_type="generation",
        plan={"exa_queries": {"general": "some query"}},
        description="general",
        n_per_label=5,
        round_idx=0,
        seen_texts=set(),
    )
    usage = build_source_usage(rows)
    urls = {e["url"] for e in usage}
    assert "https://a.example/x" in urls
    assert "https://b.example/y" in urls
    assert sum(e["rows"] for e in usage) == len(rows)
