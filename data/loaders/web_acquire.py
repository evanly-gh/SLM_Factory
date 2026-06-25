# data/loaders/web_acquire.py
"""
General autonomous data acquisition from the web via Exa, driven by the orchestrator's
task plan (agent/task_planner.py). Works for ANY task — no per-task hardcoding.

For classification: one search per class label; each retrieved document is labeled with
that class. For NER/generation: searches the plan's topic queries and returns documents
as raw text examples (supervision is synthesized later by the curate node).
"""
import time

_SKIP_URL_MARKERS = ("/archive/", "/list/", "/find/")
MAX_LABELS = 6          # bound Exa spend
DEFAULT_N_PER_LABEL = 16


def _looks_useful(url: str, text: str) -> bool:
    if not text or len(text) < 180:
        return False
    if any(m in url for m in _SKIP_URL_MARKERS):
        return False
    return True


def _exa_search(exa, query, n):
    """Single Exa call. search_and_contents returns text by default."""
    return exa.search_and_contents(
        query, num_results=n, type="auto", text={"max_characters": 800},
    )


def survey_baseline(exa, description, log=print) -> str:
    try:
        r = _exa_search(exa, f"state of the art benchmark accuracy for: {description}", 3)
        lines = [f"        - {x.title[:80]} ({x.url})" for x in r.results]
        log("      [acquire] baseline survey (Exa):\n" + "\n".join(lines))
    except Exception as e:
        log(f"      [acquire] baseline survey skipped: {e}")
    return ""


def acquire_dataset(plan: dict, description: str = "", n_per_label: int = DEFAULT_N_PER_LABEL,
                    test_fraction: float = 0.3, log=print):
    """
    Acquire a labeled dataset from the web per the task plan.
    Returns (train_examples, test_examples) as lists of {"text", "label"} dicts
    (classification) or {"text"} dicts (NER/generation).
    """
    from config import EXA_API_KEY
    from exa_py import Exa

    exa = Exa(api_key=EXA_API_KEY)
    task_type = plan["task_type"]
    queries: dict = plan.get("exa_queries") or {}

    survey_baseline(exa, description or plan.get("task_name", ""), log=log)

    examples: list[dict] = []
    if task_type == "classification":
        labels = (plan.get("labels") or list(queries.keys()))[:MAX_LABELS]
        for label in labels:
            query = queries.get(label, f"{label} example text")
            try:
                r = _exa_search(exa, query, n_per_label)
                kept = 0
                for x in r.results:
                    text = (x.text or "").strip().replace("\n", " ")
                    if _looks_useful(x.url, text):
                        doc = f"{(x.title or '').strip()}. {text}"[:700]
                        examples.append({"text": doc, "label": label})
                        kept += 1
                log(f"      [acquire] {label!r}: {len(r.results)} hits -> kept {kept}")
            except Exception as e:
                log(f"      [acquire] {label!r}: Exa error: {e}")
            time.sleep(0.2)
    else:
        topics = list(queries.items())[:MAX_LABELS] or [("general", description)]
        for topic, query in topics:
            try:
                r = _exa_search(exa, query, n_per_label)
                for x in r.results:
                    text = (x.text or "").strip().replace("\n", " ")
                    if _looks_useful(x.url, text):
                        examples.append({"text": f"{(x.title or '').strip()}. {text}"[:700],
                                         "label": topic})
                log(f"      [acquire] topic {topic!r}: collected")
            except Exception as e:
                log(f"      [acquire] topic {topic!r}: Exa error: {e}")
            time.sleep(0.2)

    if not examples:
        raise RuntimeError("Exa acquisition returned no usable examples for this task.")

    import random
    rng = random.Random(42)
    rng.shuffle(examples)
    split = int(len(examples) * (1 - test_fraction))
    train, test = examples[:split], examples[split:]
    log(f"      [acquire] total={len(examples)} train={len(train)} test={len(test)} "
        f"labels={sorted(set(e['label'] for e in examples))}")
    return train, test
