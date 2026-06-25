# data/loaders/research_papers.py
"""
Autonomous research-paper acquisition for the "classify a paper by general area
of study" task (Pioneer Agent cold-start, arXiv:2604.09791v1, Node 1/2 data
acquisition). Uses the Exa API to locate real paper abstracts on the web and
label each by the area it was retrieved for. No local benchmark file required —
the agent builds its own dataset from the web, then eval_setup builds the held-out
eval set and curate builds the curriculum from it.
"""
import os
import time

# General areas of study the agent classifies papers into (top-level arXiv groups).
DEFAULT_AREAS = ["Computer Science", "Physics", "Mathematics", "Biology", "Economics"]

# Specific subtopic queries so Exa returns actual papers, not category listing pages.
AREA_QUERIES = {
    "Computer Science": "transformer neural network deep learning research paper abstract",
    "Physics":          "quantum field theory condensed matter physics research paper abstract",
    "Mathematics":      "algebraic topology theorem proof mathematics research paper abstract",
    "Biology":          "CRISPR gene expression molecular biology research paper abstract",
    "Economics":        "macroeconomic policy inflation econometrics research paper abstract",
}

_SKIP_URL_MARKERS = ("/archive/", "/list/", "/find/", "arxiv.org/a/")


def _looks_like_paper(url: str, text: str) -> bool:
    if not text or len(text) < 200:
        return False
    if any(m in url for m in _SKIP_URL_MARKERS):
        return False
    return True


def survey_baseline(exa_client, log=print) -> str:
    """Node-1 baseline survey: one Exa call to ground the task. Logged, non-fatal."""
    try:
        r = exa_client.search_and_contents(
            "standard benchmark and accuracy for classifying research papers by field of study",
            num_results=3,
            type="auto",
            text={"max_characters": 400},
        )
        lines = [f"  - {x.title[:80]} ({x.url})" for x in r.results]
        survey = "Baseline survey (Exa):\n" + "\n".join(lines)
    except Exception as e:
        survey = f"Baseline survey skipped (Exa error: {e})"
    log(survey)
    return survey


def acquire_research_papers(
    areas: list[str] | None = None,
    n_per_area: int = 20,
    test_fraction: float = 0.3,
    log=print,
) -> tuple[list[dict], list[dict]]:
    """
    Acquire labeled paper abstracts from the web via Exa, one search per area.
    Returns (train_examples, test_examples), each item {"text": abstract, "label": area}.
    """
    from config import EXA_API_KEY
    from exa_py import Exa

    areas = areas or DEFAULT_AREAS
    exa = Exa(api_key=EXA_API_KEY)

    survey_baseline(exa, log=log)

    all_examples: list[dict] = []
    for area in areas:
        query = AREA_QUERIES.get(area, f"{area} research paper abstract")
        try:
            r = exa.search_and_contents(
                query,
                num_results=n_per_area,
                type="auto",
                text={"max_characters": 800},
                include_domains=["arxiv.org"],
            )
            kept = 0
            for x in r.results:
                text = (x.text or "").strip().replace("\n", " ")
                if _looks_like_paper(x.url, text):
                    # use title + abstract snippet as the document
                    doc = f"{(x.title or '').strip()}. {text}"[:700]
                    all_examples.append({"text": doc, "label": area})
                    kept += 1
            log(f"  [{area}] Exa returned {len(r.results)} -> kept {kept} paper-like results")
        except Exception as e:
            log(f"  [{area}] Exa error: {e}")
        time.sleep(0.2)  # gentle pacing

    if not all_examples:
        raise RuntimeError("Exa acquisition returned no usable paper examples.")

    # Deterministic split
    import random
    rng = random.Random(42)
    rng.shuffle(all_examples)
    split = int(len(all_examples) * (1 - test_fraction))
    train, test = all_examples[:split], all_examples[split:]
    log(f"  acquired total={len(all_examples)}  train={len(train)}  test={len(test)}  "
        f"areas={sorted(set(e['label'] for e in all_examples))}")
    return train, test
