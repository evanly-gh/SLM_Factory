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


# ---------------------------------------------------------------------------
# Real benchmark datasets (BUGS B119). For well-known public benchmarks the plan
# names, load the ACTUAL dataset (real questions + gold answers) instead of
# Exa-scraping webpage/repo metadata. Falls back to Exa for unknown benchmarks.
# ---------------------------------------------------------------------------
_BENCHMARK_ALIASES = {
    "gsm8k": "gsm8k",
    "financialphrasebank": "fpb", "fpb": "fpb",
    "arcchallenge": "arc", "arc": "arc", "ai2arc": "arc", "arcc": "arc",
}


def _norm_bench(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def load_benchmark_dataset(plan: dict, log=print, max_train: int = 150, max_test: int = 25):
    """Return (train, test) from the real benchmark dataset, or None if unknown/failed."""
    import re as _re
    key = _BENCHMARK_ALIASES.get(_norm_bench(plan.get("benchmark")))
    if not key:
        return None
    name = plan.get("benchmark")
    try:
        from datasets import load_dataset
        if key == "gsm8k":
            def conv(ds):
                out = []
                for ex in ds:
                    ans_full = ex["answer"]
                    m = _re.search(r'####\s*(-?[\d,]+)', ans_full)
                    final = m.group(1).replace(",", "") if m else ans_full.strip()
                    cot = _re.sub(r'####.*$', '', ans_full, flags=_re.DOTALL).strip()
                    out.append({"text": ex["question"], "answer": final,
                                "cot_reasoning": cot, "label": "math_reasoning"})
                return out
            train = conv(load_dataset("openai/gsm8k", "main", split=f"train[:{max_train}]"))
            test = conv(load_dataset("openai/gsm8k", "main", split=f"test[:{max_test}]"))
        elif key == "fpb":
            ds = load_dataset("ChanceFocus/flare-fpb", split="train")
            rows = [{"text": ex["text"], "label": ex["answer"]}
                    for ex in ds if ex.get("text") and ex.get("answer")]
            import random as _rnd
            _rnd.Random(42).shuffle(rows)
            train, test = rows[:max_train], rows[max_train:max_train + max_test]
        elif key == "arc":
            def conv(ds):
                out = []
                for ex in ds:
                    ch = ex["choices"]
                    ans = ch["text"][ch["label"].index(ex["answerKey"])] \
                        if ex["answerKey"] in ch["label"] else ""
                    out.append({"text": ex["question"], "answer": ans, "label": "generation"})
                return out
            train = conv(load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f"train[:{max_train}]"))
            test = conv(load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f"test[:{max_test}]"))
        else:
            return None
    except Exception as e:
        log(f"      [acquire] real-benchmark loader failed for {name!r}: {e}; falling back to Exa")
        return None
    if not train or not test:
        return None
    log(f"      [acquire] loaded REAL benchmark {name!r}: train={len(train)} test={len(test)}")
    return train, test


def acquire_dataset(plan: dict, description: str = "", n_per_label: int = DEFAULT_N_PER_LABEL,
                    test_fraction: float = 0.3, log=print):
    """
    Acquire a labeled dataset per the task plan.
    Prefers the real public benchmark dataset when the plan names a known one
    (B119); otherwise acquires from the web via Exa.
    Returns (train_examples, test_examples) as lists of {"text", "label", ...} dicts
    (classification/generation) or {"text", "entities"} dicts (NER).
    """
    real = load_benchmark_dataset(plan, log=log)
    if real is not None:
        return real

    from config.config import EXA_API_KEY
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

    # For NER tasks, acquired documents lack entity annotations. Use Claude to
    # extract gold entity spans from each passage. (B48 fix; paper §2.5 data acquisition)
    if task_type == "NER":
        examples = _annotate_ner_entities(examples, log=log)

    import random
    rng = random.Random(42)
    rng.shuffle(examples)
    split = int(len(examples) * (1 - test_fraction))
    train, test = examples[:split], examples[split:]
    log(f"      [acquire] total={len(examples)} train={len(train)} test={len(test)} "
        f"labels={sorted(set(e.get('label', '?') for e in examples))}")
    return train, test


def _annotate_ner_entities(examples: list[dict], log=print) -> list[dict]:
    """Add gold entity annotations to web-acquired NER passages via Claude."""
    import anthropic, json as _json, re as _re
    from config.config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    annotated = []
    for i, ex in enumerate(examples):
        try:
            response = client.messages.create(
                model=ORCHESTRATOR_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": (
                    "Extract all named entities from this text. Return a JSON list of "
                    "objects with \"text\" (the exact span as it appears in the text) and "
                    "\"type\" (PER, ORG, LOC, MISC, or other standard NER types). "
                    "Rules for consistency: use the EXACT substring from the text for each "
                    "span; for overlapping candidates prefer the LONGEST span; do NOT emit "
                    "nested or duplicated spans; assign each span exactly one type. "
                    "Return [] if there are no entities.\n\n"
                    f"Text: {ex['text'][:500]}\n\n"
                    "Reply with JSON only, no explanation."
                )}],
            )
            raw = response.content[0].text.strip()
            match = _re.search(r'\[.*\]', raw, _re.DOTALL)
            entities = _json.loads(match.group()) if match else []
            entities = [e for e in entities if "text" in e and "type" in e]
        except Exception:
            entities = []
        annotated.append({"text": ex["text"], "entities": entities})
    log(f"      [acquire] annotated {len(annotated)} NER passages with entities")
    return annotated
