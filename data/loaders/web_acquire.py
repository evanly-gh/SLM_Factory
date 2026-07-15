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
DEFAULT_N_PER_LABEL = 24  # docs requested per label/topic (Exa fallback path)


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
    # NER benchmarks: load real token+BIO-tag data and convert to entity spans.
    "bc5cdr": "bc5cdr", "bc5cdrner": "bc5cdr",
    "conll": "conll", "conll2003": "conll", "conll03": "conll",
}


def _bio_to_spans(tokens: list[str], tags: list, names: list[str] | None):
    """Convert a token sequence + BIO tag sequence into (text, entity spans).

    `names` maps integer tag ids → label strings (e.g. from a ClassLabel feature).
    Returns (text, [{"text","type"}, ...]) with contiguous B-/I- runs merged.
    """
    text = " ".join(tokens)
    spans: list[dict] = []
    cur_toks: list[str] = []
    cur_type: str | None = None

    def _flush():
        nonlocal cur_toks, cur_type
        if cur_toks and cur_type:
            spans.append({"text": " ".join(cur_toks), "type": cur_type})
        cur_toks, cur_type = [], None

    for tok, tag in zip(tokens, tags):
        label = names[tag] if names is not None and isinstance(tag, int) else str(tag)
        if label in ("O", "0") or not label:
            _flush()
            continue
        prefix, _, etype = label.partition("-")
        etype = etype or label
        if prefix == "B" or etype != cur_type:
            _flush()
            cur_toks, cur_type = [tok], etype
        else:  # "I-" continuation of the same type
            cur_toks.append(tok)
    _flush()
    return text, spans


def _load_ner_benchmark(key: str, max_train: int, max_test: int, log=print):
    """Load a real NER benchmark (BC5CDR / CoNLL-2003) as {"text","entities"} dicts.

    Tries a few known HuggingFace dataset ids; returns (train, test) or None if none
    load (caller then falls back to Exa). Defensive by design — an unreachable dataset
    must not crash the run.
    """
    from datasets import load_dataset

    # (dataset_id, config, tokens_key, tags_key) candidates, most-canonical first.
    candidates = {
        "conll": [("eriktks/conll2003", None, "tokens", "ner_tags"),
                  ("conll2003", None, "tokens", "ner_tags")],
        "bc5cdr": [("tner/bc5cdr", None, "tokens", "tags"),
                   ("spyysalo/bc5cdr", None, "tokens", "ner_tags")],
    }.get(key, [])
    # Fallback label map for datasets whose tags are bare ints (no ClassLabel names).
    _FALLBACK_NAMES = {
        "bc5cdr": ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"],
    }

    for ds_id, cfg, tok_key, tag_key in candidates:
        try:
            def _split(split, n):
                ds = (load_dataset(ds_id, cfg, split=f"{split}[:{n}]", trust_remote_code=True)
                      if cfg else
                      load_dataset(ds_id, split=f"{split}[:{n}]", trust_remote_code=True))
                feat = ds.features.get(tag_key)
                names = getattr(getattr(feat, "feature", None), "names", None) or _FALLBACK_NAMES.get(key)
                out = []
                for ex in ds:
                    text, spans = _bio_to_spans(ex[tok_key], ex[tag_key], names)
                    if text.strip():
                        out.append({"text": text, "entities": spans})
                return out
            train = _split("train", max_train)
            # CoNLL uses "validation"; BC5CDR uses "test" — try both.
            try:
                test = _split("validation", max_test)
            except Exception:
                test = _split("test", max_test)
            if train and test:
                log(f"      [acquire] loaded REAL NER benchmark via {ds_id!r}: "
                    f"train={len(train)} test={len(test)}")
                return ds_id, train, test
        except Exception as e:
            log(f"      [acquire] NER benchmark {ds_id!r} unavailable ({str(e)[:80]}); trying next")
    return None


def _norm_bench(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def load_benchmark_dataset(plan: dict, log=print, max_train: int = 300, max_test: int = 80,
                           meta: dict | None = None):
    """Return (train, test) from the real benchmark dataset, or None if unknown/failed.

    When `meta` is provided it is populated with a human-readable `source` string so
    callers can report data provenance downstream (e.g. in the curate log).
    """
    import re as _re
    key = _BENCHMARK_ALIASES.get(_norm_bench(plan.get("benchmark")))
    if not key:
        return None
    name = plan.get("benchmark")

    # NER benchmarks are token/BIO-tagged, not question/answer — handle separately.
    if key in ("bc5cdr", "conll"):
        ner = _load_ner_benchmark(key, max_train, max_test, log=log)
        if ner is None:
            return None
        ds_id, train, test = ner
        if meta is not None:
            meta["source"] = (
                f"real NER benchmark {name!r} via HuggingFace {ds_id!r} "
                f"(token/BIO tags → entity spans; train={len(train)}/test={len(test)})"
            )
        return train, test

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
    if meta is not None:
        meta["source"] = (
            f"real benchmark dataset {name!r} (HuggingFace datasets.load_dataset; "
            f"capped at max_train={max_train}/max_test={max_test})"
        )
    return train, test


# Data-acquisition budget/floors (B139). Acquisition ladder: real benchmark → bounded,
# diversified Exa rounds → verified LLM synthesis. Quality-over-quantity: `target` is an
# UPPER bound, `floor` is the minimum we insist on before falling back to synthesis.
MAX_ACQUIRE_ROUNDS = 3           # bounded Exa retries (each round diversifies the queries)
DEFAULT_TARGET_EXAMPLES = 120    # desired acquired pool size (train+test), a ceiling
MIN_VIABLE_FRACTION = 0.5        # below target*this → try more rounds, then synthesis


def _diversify_query(base_query: str, round_idx: int) -> str:
    """Rephrase a query per round so re-runs fetch NEW documents, not the same top hits.

    Exa is roughly deterministic for a fixed query, so blindly re-running returns
    duplicates. Rotating the phrasing surfaces different parts of the web.
    """
    if round_idx == 0:
        return base_query
    templates = [
        "real-world examples of {q}",
        "labeled dataset or corpus of {q}",
        "annotated {q} samples with ground-truth labels",
    ]
    return templates[(round_idx - 1) % len(templates)].format(q=base_query)


def _exa_round(exa, task_type, plan, description, n_per_label, round_idx,
               seen_texts: set, log=print) -> list[dict]:
    """One Exa acquisition round across all labels/topics; dedups against seen_texts."""
    queries: dict = plan.get("exa_queries") or {}
    out: list[dict] = []
    if task_type == "classification":
        labels = (plan.get("labels") or list(queries.keys()))[:MAX_LABELS]
        items = [(lbl, queries.get(lbl, f"{lbl} example text")) for lbl in labels]
    else:
        items = list(queries.items())[:MAX_LABELS] or [("general", description)]
    for label, base_query in items:
        query = _diversify_query(base_query, round_idx)
        try:
            r = _exa_search(exa, query, n_per_label)
            kept = 0
            for x in r.results:
                text = (x.text or "").strip().replace("\n", " ")
                if not _looks_useful(x.url, text):
                    continue
                doc = f"{(x.title or '').strip()}. {text}"[:700]
                if doc in seen_texts:
                    continue
                seen_texts.add(doc)
                out.append({"text": doc, "label": label})
                kept += 1
            log(f"      [acquire] round {round_idx} {label!r} q={query!r}: kept {kept} new")
        except Exception as e:
            log(f"      [acquire] round {round_idx} {label!r}: Exa error: {e}")
        time.sleep(0.2)
    return out


def acquire_dataset(plan: dict, description: str = "", n_per_label: int = DEFAULT_N_PER_LABEL,
                    test_fraction: float = 0.3, target_examples: int = DEFAULT_TARGET_EXAMPLES,
                    log=print, meta: dict | None = None):
    """
    Acquire a labeled dataset per the task plan, following the acquisition ladder (B139):

      1. REAL benchmark dataset if the plan names a known one (highest quality; B119).
      2. Otherwise BOUNDED + DIVERSIFIED Exa rounds: up to MAX_ACQUIRE_ROUNDS, each round
         rephrases the queries so re-runs fetch NEW documents (not duplicates), deduping,
         stopping once `target_examples` is reached.
      3. If still below `target_examples * MIN_VIABLE_FRACTION`, TOP UP with verified LLM
         synthesis (`synthesize_seed_examples`) — deduped and label-validated.

    `target_examples` is an UPPER bound (quality-over-quantity); we do not chase it past
    what clean sources provide, but we do insist on the viability floor before proceeding.

    Returns (train, test). `meta["source"]` records provenance (benchmark / web+synth mix).
    """
    real = load_benchmark_dataset(plan, log=log, meta=meta)
    if real is not None:
        return real

    from config.config import EXA_API_KEY
    from exa_py import Exa

    exa = Exa(api_key=EXA_API_KEY)
    task_type = plan["task_type"]
    survey_baseline(exa, description or plan.get("task_name", ""), log=log)

    # --- Stage 2: bounded, diversified Exa rounds -------------------------------------
    examples: list[dict] = []
    seen_texts: set = set()
    floor = max(1, int(target_examples * MIN_VIABLE_FRACTION))
    for round_idx in range(MAX_ACQUIRE_ROUNDS):
        examples.extend(_exa_round(exa, task_type, plan, description, n_per_label,
                                   round_idx, seen_texts, log=log))
        if len(examples) >= target_examples:
            break
        if round_idx + 1 < MAX_ACQUIRE_ROUNDS:
            log(f"      [acquire] have {len(examples)}/{target_examples} — diversifying and retrying")
    n_web = len(examples)

    # --- Stage 3: verified synthesis fallback (only if below the viability floor) ------
    n_synth = 0
    if len(examples) < floor:
        needed = target_examples - len(examples)
        log(f"      [acquire] ⚠ web acquisition returned {len(examples)} < floor {floor}; "
            f"synthesizing up to {needed} gold examples via the orchestrator (verified/deduped)")
        synth = synthesize_seed_examples(plan, task_type, needed,
                                         existing_texts=seen_texts, log=log)
        n_synth = len(synth)
        examples.extend(synth)

    if not examples:
        raise RuntimeError(
            "Data acquisition returned no usable examples (real benchmark, Exa rounds, "
            "and synthesis all failed). Provide a known benchmark or check API keys."
        )

    # For NER, acquired documents lack entity annotations — annotate via Claude (B48).
    # (Synthesized NER examples already carry entities and are skipped inside the annotator.)
    if task_type == "NER":
        examples = _annotate_ner_entities(examples, log=log)

    import random
    rng = random.Random(42)
    rng.shuffle(examples)
    split = int(len(examples) * (1 - test_fraction))
    train, test = examples[:split], examples[split:]
    log(f"      [acquire] total={len(examples)} (web={n_web}, synthesized={n_synth}) "
        f"train={len(train)} test={len(test)} "
        f"labels={sorted(set(e.get('label', '?') for e in examples))}")
    if meta is not None:
        parts = []
        if n_web:
            parts.append(f"{n_web} web docs via Exa ({MAX_ACQUIRE_ROUNDS}-round diversified)")
        if n_synth:
            parts.append(f"{n_synth} LLM-synthesized+verified examples")
        meta["source"] = "; ".join(parts) or "web/synthesis"
    return train, test


def synthesize_seed_examples(plan: dict, task_type: str, n_needed: int,
                             existing_texts: set | None = None, log=print) -> list[dict]:
    """Last-resort GOLD synthesis via the orchestrator, deduped + label-validated.

    Only used when real-benchmark and bounded web acquisition can't reach the viability
    floor. Kept deliberately conservative: the orchestrator generates task-appropriate
    labeled examples, we drop duplicates and anything that fails a basic validity check
    (label in the allowed set / required fields present). NOTE: synthetic gold for
    knowledge-heavy tasks can encode the teacher's errors — this is a fallback, not the
    preferred source; the low-data warning still fires downstream.
    """
    if n_needed <= 0:
        return []
    import json as _json
    import re as _re
    import anthropic
    from config.config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL

    existing_texts = existing_texts or set()
    labels = plan.get("labels") or []
    task_name = plan.get("task_name", task_type)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if task_type == "classification":
        schema = '{"examples": [{"text": "<input text>", "label": "<one of LABELS>"}]}'
        label_line = f"LABELS = {labels}. Balance examples across all labels."
    elif task_type == "NER":
        schema = ('{"examples": [{"text": "<passage>", '
                  '"entities": [{"text": "<span>", "type": "<TYPE>"}]}]}')
        label_line = f"Entity types = {labels}. Every entity span must be an exact substring of text."
    else:  # math_reasoning / code_generation / generation
        schema = '{"examples": [{"text": "<problem/prompt>", "answer": "<correct answer>"}]}'
        label_line = "Each answer must be correct and verifiable."

    prompt = (
        f"Generate {n_needed} diverse, realistic training examples for this task.\n"
        f"Task: {task_name} (type: {task_type}).\n{label_line}\n"
        f"Return STRICT JSON only, no prose: {schema}"
    )
    try:
        resp = client.messages.create(
            model=ORCHESTRATOR_MODEL, max_tokens=4096, temperature=1.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        parsed = _json.loads(m.group()) if m else {}
        candidates = parsed.get("examples", []) if isinstance(parsed, dict) else []
    except Exception as e:
        log(f"      [acquire] synthesis failed: {e}")
        return []

    valid: list[dict] = []
    for ex in candidates:
        if not isinstance(ex, dict):
            continue
        text = str(ex.get("text", "")).strip()
        if not text or text in existing_texts:
            continue
        if task_type == "classification":
            if labels and ex.get("label") not in labels:
                continue
            valid.append({"text": text, "label": ex["label"]})
        elif task_type == "NER":
            ents = [e for e in ex.get("entities", [])
                    if isinstance(e, dict) and e.get("text") and e.get("text") in text]
            valid.append({"text": text, "entities": ents})
        else:
            ans = str(ex.get("answer", "")).strip()
            if not ans:
                continue
            valid.append({"text": text, "answer": ans, "label": task_type})
        existing_texts.add(text)
    log(f"      [acquire] synthesized {len(valid)}/{len(candidates)} valid examples "
        f"(deduped + validated)")
    return valid


def _annotate_ner_entities(examples: list[dict], log=print) -> list[dict]:
    """Add gold entity annotations to web-acquired NER passages via Claude."""
    import anthropic, json as _json, re as _re
    from config.config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    annotated = []
    for i, ex in enumerate(examples):
        # Skip examples that already carry entities (e.g. synthesized seeds) — don't
        # re-annotate and clobber known-good gold.
        if ex.get("entities"):
            annotated.append(ex)
            continue
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
