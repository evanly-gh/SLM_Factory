# data/loaders/web_acquire.py
"""
General autonomous data acquisition from the web via Exa, driven by the orchestrator's
task plan (agent/task_planner.py). Works for ANY task — no per-task hardcoding.

For classification: one search per class label; each retrieved document is labeled with
that class. For NER/generation: searches the plan's topic queries and returns documents
as raw text examples (supervision is synthesized later by the curate node).
"""
import os
import time
from pathlib import Path

from agent.cost import tracked_anthropic_messages_create, tracked_exa_call
from agent.llm_text import MIN_THINKING_SAFE_MAX_TOKENS, response_text
from data.loaders.dataset_integrity import (
    normalize_text,
    normalized_text_overlap,
    remove_normalized_train_overlap,
    required_fields_for_task,
    validate_rows,
    verify_bundle_checksums,
)

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
    return tracked_exa_call(
        exa.search_and_contents,
        query,
        stage="acquire_exa",
        model="search-and-contents",
        num_results=n,
        type="auto",
        text={"max_characters": 800},
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
    "gradeschoolmath8k": "gsm8k",
    "financialphrasebank": "fpb", "fpb": "fpb",
    "smsspam": "sms_spam",
    "smsspamcollection": "sms_spam",
    "ucismsspamcollection": "sms_spam",
    "arcchallenge": "arc", "arc": "arc", "ai2arc": "arc", "arcc": "arc",
    # NER benchmarks: load real token+BIO-tag data and convert to entity spans.
    "bc5cdr": "bc5cdr", "bc5cdrner": "bc5cdr",
    "biocreativevchemicaldiseaserelation": "bc5cdr",
    "conll": "conll", "conll2003": "conll", "conll03": "conll",
    "apps": "apps", "codeparrotapps": "apps",
    "mbpp": "mbpp", "mostlybasicpythonproblems": "mbpp",
    "samsum": "samsum", "samsumdialoguesummarization": "samsum",
    # RouterBench. Without this entry Stage-0 had no alias for the benchmark the task is ABOUT, so
    # `load_dataset("withmartian/routerbench")` failed (the repo ships only pickles), mining fell
    # through to agentic discovery, and a completely different corpus was substituted 21 times in
    # one run — with its labels invented by an LLM (B259). Routes to the real pickle loader.
    "routerbench": "routerbench", "withmartianrouterbench": "routerbench",
    "routerbench0shot": "routerbench",
}

# The planner often emits a composite benchmark label ("HumanEval / MBPP", "Biomedical
# NER (BC5CDR)") rather than an exact catalog key. These distinctive canonical markers are
# safe to recognize inside a normalized composite name.
_COMPOSITE_BENCHMARK_MARKERS = {
    "bc5cdr": "bc5cdr",
    "routerbench": "routerbench",
    "gsm8k": "gsm8k",
    "codeparrotapps": "apps",
    "appsintroductory": "apps",
    "mbpp": "mbpp",
    "samsum": "samsum",
    "smsspam": "sms_spam",
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

    Tries a few known HuggingFace dataset ids; returns (source, train, test) or None if none
    load (caller then falls back to Exa). Defensive by design — an unreachable dataset
    must not crash the run.
    """
    from datasets import load_dataset

    candidates = {
        "conll": [
            {"id": "eriktks/conll2003", "config": None, "tokens": "tokens",
             "tags": "ner_tags", "splits": {"train": "train", "test": "validation"}},
            {"id": "conll2003", "config": None, "tokens": "tokens",
             "tags": "ner_tags", "splits": {"train": "train", "test": "validation"}},
        ],
        "bc5cdr": [
            {"id": "tner/bc5cdr", "config": None, "tokens": "tokens",
             "tags": "tags", "splits": {"train": "train", "test": "test"}},
            {"id": "spyysalo/bc5cdr", "config": None, "tokens": "tokens",
             "tags": "ner_tags", "splits": {"train": "train", "test": "test"}},
            # datasets>=4 refuses script-based repos. Read T-NER's public official JSON
            # files directly as a script-free compatibility path.
            {"id": "tner/bc5cdr", "config": None, "loader": "json",
             "tokens": "tokens", "tags": "tags",
             "splits": {"train": "train", "test": "test"},
             "data_files": {
                 "train": ("https://huggingface.co/datasets/tner/bc5cdr/"
                           "resolve/main/dataset/train.json"),
                 "test": ("https://huggingface.co/datasets/tner/bc5cdr/"
                          "resolve/main/dataset/test.json"),
             }},
        ],
    }.get(key, [])
    _FALLBACK_NAMES = {
        "bc5cdr": ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"],
    }

    for source in candidates:
        ds_id, cfg = source["id"], source.get("config")
        tok_key, tag_key = source["tokens"], source["tags"]
        try:
            def _split(our_split, n):
                split = source["splits"][our_split]
                if source.get("loader") == "json":
                    ds = load_dataset(
                        "json", data_files={split: source["data_files"][our_split]},
                        split=f"{split}[:{n}]",
                    )
                elif cfg is not None:
                    ds = load_dataset(ds_id, cfg, split=f"{split}[:{n}]")
                else:
                    ds = load_dataset(ds_id, split=f"{split}[:{n}]")
                features = getattr(ds, "features", {})
                feat = features.get(tag_key)
                names = getattr(getattr(feat, "feature", None), "names", None) or _FALLBACK_NAMES.get(key)
                out = []
                for ex in ds:
                    text, spans = _bio_to_spans(ex[tok_key], ex[tag_key], names)
                    if text.strip():
                        out.append({"text": text, "entities": spans})
                return out
            train = _split("train", max_train)
            test = _split("test", max_test)
            if train and test:
                log(f"      [acquire] loaded REAL NER benchmark via {ds_id!r}: "
                    f"train={len(train)} test={len(test)}")
                return source, train, test
        except Exception as e:
            log(f"      [acquire] NER benchmark {ds_id!r} unavailable ({str(e)[:80]}); trying next")
    return None


def _norm_bench(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def _resolve_benchmark_key(name: str | None) -> str | None:
    normalized = _norm_bench(name)
    exact = _BENCHMARK_ALIASES.get(normalized)
    if exact:
        return exact
    for marker, key in _COMPOSITE_BENCHMARK_MARKERS.items():
        if marker in normalized:
            return key
    return None


def _source_records(
    source_id: str,
    config: str | None,
    splits: dict[str, str],
    revision: str | None = None,
) -> list[dict]:
    url = f"https://huggingface.co/datasets/{source_id}"
    records = [
        {
            "kind": "hf", "id": source_id, "config": config,
            "split": splits["train"], "url": url, "role": "curriculum",
        },
        {
            "kind": "hf", "id": source_id, "config": config,
            "split": splits["test"], "url": url, "role": "eval",
        },
    ]
    if revision:
        for record in records:
            record["revision"] = revision
    return records


def _set_benchmark_meta(
    meta,
    name,
    source_id,
    config,
    splits,
    detail="",
    revision: str | None = None,
):
    if meta is None:
        return
    suffix = f"; {detail}" if detail else ""
    meta["source"] = (
        f"real benchmark dataset {name!r} via HuggingFace {source_id!r} "
        f"(config={config!r}, official train/test splits{suffix})"
    )
    records = _source_records(
        source_id,
        config,
        splits,
        revision=revision,
    )
    meta["source_records"] = records
    meta["eval_ban"] = [dict(records[1])]


def _clean_stage0_splits(train, test, name, log=print, meta=None):
    """Apply the same normalized train/test decontamination as offline bundles."""
    fingerprint_removed = 0
    if _resolve_benchmark_key(name) == "apps":
        from data.loaders.apps import remove_apps_train_fingerprint_overlap

        train, fingerprint_removed = remove_apps_train_fingerprint_overlap(
            train,
            test,
        )
        if fingerprint_removed:
            log(
                f"      [acquire] Stage-0 APPS URL/solution overlap removal: "
                f"removed {fingerprint_removed} train row(s)"
            )
    clean_train, removed = remove_normalized_train_overlap(train, test)
    if removed:
        log(
            f"      [acquire] Stage-0 normalized overlap removal for {name!r}: "
            f"removed {removed} train row(s); official test rows unchanged"
        )
    remaining = normalized_text_overlap(clean_train, test)
    if remaining:
        raise ValueError(
            f"{name}: normalized train/test text overlap remains ({len(remaining)})"
        )
    if meta is not None:
        meta["overlap_removed_from_train"] = removed
        if _resolve_benchmark_key(name) == "apps":
            meta["fingerprint_overlap_removed_from_train"] = (
                fingerprint_removed
            )
    return clean_train, test


def load_benchmark_dataset(plan: dict, log=print, max_train: int = 300, max_test: int = 80,
                           meta: dict | None = None):
    """Return (train, test) from the real benchmark dataset, or None if unknown/failed.

    When `meta` is provided it is populated with a human-readable `source` string so
    callers can report data provenance downstream (e.g. in the curate log).
    """
    key = _resolve_benchmark_key(plan.get("benchmark"))
    if not key:
        return None
    name = plan.get("benchmark")

    # RouterBench cannot go through `load_dataset` at all — the repo ships only
    # `routerbench_{0shot,5shot,raw}.pkl` and `datasets` has no pickle reader (B254). It needs its
    # dedicated loader, which fetches via hf_hub_download + pandas and derives the routing label
    # from a real per-model correctness column. Handled before the load_dataset block below.
    if key == "routerbench":
        try:
            from data.loaders.routerbench import HF_ID as _RB_ID, load_routerbench

            train, test = load_routerbench(
                max_train=max_train, max_test=max_test, log=log
            )
        except Exception as e:  # noqa: BLE001 - fall through to discovery, as other loaders do
            log(f"      [acquire] routerbench loader failed: {e}; falling back to Exa")
            return None
        _set_benchmark_meta(
            meta, name, _RB_ID, None,
            {"train": "hash-split 80%", "test": "hash-split 20%"},
            detail="per-model correctness column → local/route routing decision",
        )
        train, test = _clean_stage0_splits(train, test, name, log=log, meta=meta)
        if not train or not test:
            return None
        log(f"      [acquire] loaded REAL benchmark {name!r}: "
            f"train={len(train)} test={len(test)}")
        return train, test

    # NER benchmarks are token/BIO-tagged, not question/answer — handle separately.
    if key in ("bc5cdr", "conll"):
        ner = _load_ner_benchmark(key, max_train, max_test, log=log)
        if ner is None:
            return None
        source, train, test = ner
        _set_benchmark_meta(
            meta, name, source["id"], source.get("config"), source["splits"],
            detail="token/BIO tags → entity spans",
        )
        train, test = _clean_stage0_splits(train, test, name, log=log, meta=meta)
        if not train or not test:
            return None
        return train, test

    try:
        from datasets import load_dataset
        if key == "sms_spam":
            from data.loaders.sms_spam import (
                SMS_SPAM_URL,
                download_sms_spam,
            )

            train, test = download_sms_spam()
            train = _stratified_take(train, max_train)
            test = _stratified_take(test, max_test, seed=7)
            if meta is not None:
                records = [
                    {
                        "kind": "url",
                        "id": "uci-sms-spam",
                        "split": "train",
                        "url": SMS_SPAM_URL,
                        "role": "curriculum",
                    },
                    {
                        "kind": "url",
                        "id": "uci-sms-spam",
                        "split": "test",
                        "url": SMS_SPAM_URL,
                        "role": "eval",
                    },
                ]
                meta["source"] = "UCI SMS Spam Collection"
                meta["source_records"] = records
                meta["eval_ban"] = [dict(records[1])]
        elif key == "gsm8k":
            def conv(ds):
                out = []
                for ex in ds:
                    question = str(ex.get("question") or "").strip()
                    raw = str(ex.get("answer") or "")
                    if not question or not raw:
                        continue
                    if "####" in raw:
                        cot, _, final = raw.rpartition("####")
                    else:
                        cot, final = "", raw
                    out.append({
                        "text": question, "answer": final.strip(),
                        "cot_reasoning": cot.strip(), "label": "math_reasoning",
                    })
                return out
            source_id, config = "openai/gsm8k", "main"
            splits = {"train": "train", "test": "test"}
            train = conv(load_dataset(source_id, config, split=f"train[:{max_train}]"))
            test = conv(load_dataset(source_id, config, split=f"test[:{max_test}]"))
            _set_benchmark_meta(meta, name, source_id, config, splits, detail="gold CoT preserved")
        elif key == "apps":
            from data.loaders.apps import (
                APPS_SOURCE_REVISION,
                convert_apps_rows,
            )

            source_id, config = "codeparrot/apps", "introductory"
            splits = {"train": "train", "test": "test"}
            data_files = {
                split: (
                    "https://huggingface.co/datasets/codeparrot/apps/"
                    f"resolve/{APPS_SOURCE_REVISION}/{split}.jsonl"
                )
                for split in splits.values()
            }

            def load_apps(split, limit):
                raw = load_dataset(
                    "json",
                    data_files={split: data_files[split]},
                    split=split,
                    streaming=True,
                )
                from eval.scorers.generation import _run_apps_tests

                def gold_validator(solution, row):
                    return _run_apps_tests(solution, row)

                conversion_stats: dict = {}
                rows, _ = convert_apps_rows(
                    raw,
                    split=split,
                    limit=limit,
                    gold_validator=gold_validator,
                    conversion_stats=conversion_stats,
                    skip_runner_incompatible=split == "test",
                )
                if meta is not None:
                    meta.setdefault(
                        "apps_conversion_stats",
                        {},
                    )[split] = conversion_stats
                return rows

            train = load_apps("train", max_train)
            test = load_apps("test", max_test)
            _set_benchmark_meta(
                meta,
                name,
                source_id,
                config,
                splits,
                detail=(
                    "introductory only; solutions/starter code and call-based/"
                    "stdin tests preserved"
                ),
                revision=APPS_SOURCE_REVISION,
            )
        elif key == "mbpp":
            def conv(ds):
                out = []
                for ex in ds:
                    prompt = str(ex.get("prompt") or "").strip()
                    code = str(ex.get("code") or "").strip()
                    tests = ex.get("test_list")
                    if not prompt or not code or not isinstance(tests, (list, tuple)):
                        continue
                    out.append({
                        "text": prompt, "answer": code, "code": code,
                        "test_imports": list(ex.get("test_imports") or []),
                        "test_list": list(tests), "task_id": ex.get("task_id"),
                        "label": "code_generation",
                    })
                return out
            source_id, config = "google-research-datasets/mbpp", "sanitized"
            splits = {"train": "train", "test": "test"}
            train = conv(load_dataset(source_id, config, split=f"train[:{max_train}]"))
            test = conv(load_dataset(source_id, config, split=f"test[:{max_test}]"))
            _set_benchmark_meta(meta, name, source_id, config, splits, detail="code/tests preserved")
        elif key == "samsum":
            def conv(ds):
                out = []
                for ex in ds:
                    dialogue = str(ex.get("dialogue") or "").strip()
                    summary = str(ex.get("summary") or "").strip()
                    if dialogue and summary:
                        out.append({
                            "text": dialogue, "answer": summary, "label": "generation",
                        })
                return out
            train = test = None
            source_id = config = None
            splits = {"train": "train", "test": "test"}
            errors = []
            for candidate in ("samsum", "knkarthick/samsum"):
                try:
                    candidate_train = conv(
                        load_dataset(candidate, split=f"train[:{max_train}]")
                    )
                    candidate_test = conv(
                        load_dataset(candidate, split=f"test[:{max_test}]")
                    )
                    if candidate_train and candidate_test:
                        source_id, train, test = candidate, candidate_train, candidate_test
                        break
                except Exception as error:
                    errors.append(f"{candidate}: {error}")
                    log(
                        f"      [acquire] SAMSum source {candidate!r} unavailable "
                        f"({str(error)[:80]}); trying mirror"
                    )
            if not source_id:
                raise RuntimeError("; ".join(errors) or "no usable SAMSum rows")
            _set_benchmark_meta(meta, name, source_id, config, splits)
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
    train, test = _clean_stage0_splits(train, test, name, log=log, meta=meta)
    if not train or not test:
        return None
    log(f"      [acquire] loaded REAL benchmark {name!r}: train={len(train)} test={len(test)}")
    if meta is not None and "source" not in meta:
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


def _source_record_is_banned(record: dict, bans: list[dict]) -> bool:
    """Match source restrictions, respecting split-specific held-out bans."""
    for ban in bans:
        if not isinstance(ban, dict):
            continue
        comparable = ("kind", "id", "url", "config")
        supplied = [key for key in comparable if ban.get(key) is not None]
        if not supplied or any(record.get(key) != ban.get(key) for key in supplied):
            continue
        banned_split = ban.get("split")
        if banned_split is None or record.get("split") == banned_split:
            return True
    return False


def mine_additional_real_rows(
    *,
    task_plan: dict,
    description: str,
    task_type: str,
    existing_rows: list[dict],
    eval_rows: list[dict],
    eval_source_ban: list[dict],
    requested_rows: int,
    max_paid_rounds: int,
    query_variant: int,
    plan_identity: str,
    label_space: set[str] | None = None,
    log=print,
) -> tuple[list[dict], dict]:
    """Mine bounded, novel real-source training rows without touching eval data.

    The acquisition order reuses the existing clean local loader and deterministic
    benchmark loader before the process-isolated Exa/HuggingFace discovery path.
    It deliberately does not call the seed-synthesis fallback: rows returned by
    this function must come from a real source.
    """
    requested = max(0, min(500, int(requested_rows or 0)))
    paid_limit = max(0, min(MAX_ACQUIRE_ROUNDS, int(max_paid_rounds or 0)))
    existing_texts = {
        normalize_text(row.get("text", row.get("prompt", "")))
        for row in existing_rows
        if isinstance(row, dict)
    }
    eval_texts = {
        normalize_text(row.get("text", row.get("prompt", "")))
        for row in eval_rows
        if isinstance(row, dict)
    }
    existing_texts.discard("")
    eval_texts.discard("")
    seen = set(existing_texts) | set(eval_texts)
    novel: list[dict] = []
    source_records: list[dict] = []
    candidate_rows = 0
    rejected_sources = 0
    paid_rounds_used = 0
    paid_budget_exhausted = False

    # The task's CLOSED label vocabulary. Prefer the pinned space from the frozen eval set (that is
    # what the model is scored against); fall back to the labels already present in the run's own
    # rows when no pinned space was supplied.
    #
    # Two guards, and the distinction matters (B259). The ORIGINAL guard was per-source and passed
    # a source on ANY overlap, so a source whose labels were *partly* right admitted all of its
    # rows — that is precisely how `cloud`/`on_device`/`router`/`remote` entered a two-class
    # RouterBench run and then got deleted by quality control on every rebuild for the rest of the
    # run. It also still catches the case it was written for: a mined dataset storing its intent
    # column as a plain integer rather than a ClassLabel (`DeepPavlov/clinc150` types `label` as
    # `Value('int64')`, so raw ids "0"/"1"/"2" were written straight into training data, B222).
    # Whether the vocabulary is AUTHORITATIVE decides which of the two rules applies, and getting
    # this wrong in either direction is harmful. A pinned space (from the frozen eval set) or a
    # plan-declared label list is complete by construction, so a source carrying anything outside it
    # is genuinely unusable and the whole source is rejected. A space merely *inferred* from the rows
    # the run happens to hold is NOT complete — a small or skewed pool can be missing real classes —
    # so applying the strict rule there would reject perfectly good sources for classes the pool
    # simply had not seen yet. In that case fall back to the original any-overlap check, which still
    # catches what it was written for: raw integer class ids from a non-ClassLabel column
    # (`DeepPavlov/clinc150` types `label` as `Value('int64')`, so "0"/"1"/"2" reached training, B222).
    _pinned_labels = {str(v) for v in (label_space or ()) if str(v).strip()}
    _plan_labels = {
        str(v) for v in (task_plan.get("labels") or []) if str(v).strip()
    }
    _row_labels = {
        str(row.get("label"))
        for row in (existing_rows or [])
        if isinstance(row, dict) and row.get("label") is not None
    }
    _known_labels = _pinned_labels or (_plan_labels | _row_labels)
    _authoritative = bool(_pinned_labels or _plan_labels)

    def _labels_are_usable(train: list, stage: str) -> bool:
        """Reject a source whose labels fall outside the task's vocabulary.

        Strict subset when the vocabulary is authoritative; any-overlap when it was inferred.
        """
        if task_type != "classification" or not _known_labels:
            return True
        from data.label_space import describe_rejected_labels

        mined_labels = {
            str(row.get("label"))
            for row in train
            if isinstance(row, dict) and row.get("label") is not None
        }
        if not mined_labels:
            return True
        foreign = mined_labels - _known_labels
        if not foreign:
            return True
        if not _authoritative and foreign != mined_labels:
            # Inferred vocabulary, partial overlap: cannot distinguish "a class the pool has not
            # seen" from "a foreign class", so admit and let the per-row filter and quality control
            # decide. This is the pre-B259 behaviour, retained only for the inferred case.
            return True
        counts: dict[str, int] = {}
        for row in train:
            if isinstance(row, dict):
                key = str(row.get("label"))
                if key in foreign:
                    counts[key] = counts.get(key, 0) + 1
        log(
            f"      [mine] REJECTED {stage} source: {len(foreign)} of its {len(mined_labels)} "
            f"label(s) are NOT in the task's "
            f"{'pinned' if _authoritative else 'established'} {len(_known_labels)}-class "
            f"vocabulary [{describe_rejected_labels(counts)}]. The label space is closed — a "
            "source is used only when every label it carries already exists, because a class "
            "absent from the eval set cannot be scored (B259/B222)."
        )
        return False

    def accept(result, meta: dict, stage: str) -> int:
        nonlocal candidate_rows, rejected_sources
        if not result:
            return 0
        train = result[0] if isinstance(result, (tuple, list)) else None
        if not isinstance(train, list):
            return 0
        records = [
            dict(record)
            for record in (meta.get("source_records") or [])
            if isinstance(record, dict)
            and record.get("role") != "eval"
        ]
        if any(
            _source_record_is_banned(record, eval_source_ban)
            for record in records
        ):
            rejected_sources += 1
            log(
                f"      [mine] rejected {stage} source due to held-out "
                "source restriction"
            )
            return 0

        candidate_rows += len(train)
        if not _labels_are_usable(train, stage):
            rejected_sources += 1
            return 0
        # Per-ROW backstop, only against an AUTHORITATIVE vocabulary — against an inferred one it
        # would drop rows for classes the pool merely had not seen yet. The source-level check above
        # already rejects a source carrying a foreign label, so reaching here with one means a
        # converter produced it after validation; dropping the row is then the only safe response,
        # because a class outside a closed space can never be scored.
        if task_type == "classification" and _authoritative and _known_labels:
            from data.label_space import describe_rejected_labels, partition_rows_by_label

            train, _row_rejected = partition_rows_by_label(train, _known_labels)
            if _row_rejected:
                log(
                    f"      [mine] dropped {sum(_row_rejected.values())} row(s) from {stage} with "
                    f"out-of-vocabulary labels [{describe_rejected_labels(_row_rejected)}]"
                )
            if not train:
                return 0
        accepted = 0
        source_id = (
            f"{records[0].get('kind', 'source')}:{records[0].get('id', '?')}"
            f"/{records[0].get('split', 'train')}"
            if records
            else str(meta.get("source") or stage)
        )
        for row in train:
            if len(novel) >= requested or not isinstance(row, dict):
                break
            text = normalize_text(row.get("text", row.get("prompt", "")))
            if not text or text in seen:
                continue
            seen.add(text)
            tagged = dict(row)
            tagged.setdefault("_source", source_id)
            if records:
                tagged["_source_record"] = dict(records[0])
            novel.append(tagged)
            accepted += 1
        for record in records:
            if record not in source_records:
                source_records.append(record)
        log(
            f"      [mine] {stage}: candidates={len(train)} "
            f"novel={accepted}"
        )
        return accepted

    if requested:
        local_meta: dict = {}
        local = load_local_dataset(
            task_plan,
            task_type,
            max(300, len(existing_rows) + requested * 4),
            80,
            log=log,
            meta=local_meta,
        )
        if accept(local, local_meta, "local"):
            paid_limit = 0
        else:
            benchmark_meta: dict = {}
            benchmark = load_benchmark_dataset(
                task_plan,
                log=log,
                max_train=max(300, len(existing_rows) + requested * 4),
                max_test=80,
                meta=benchmark_meta,
            )
            if accept(benchmark, benchmark_meta, "benchmark"):
                paid_limit = 0

    for round_index in range(paid_limit):
        if len(novel) >= requested:
            break
        from agent.data_rebuild import MAX_PAID_ACQUIRE_ROUNDS_PER_RUN
        from data.acquisition_budget import (
            acquisition_budget_snapshot,
            reconcile_paid_acquisition,
            reserve_paid_acquisition,
        )

        reservation = reserve_paid_acquisition(
            plan_identity=plan_identity,
            per_plan_limit=paid_limit,
            run_limit=MAX_PAID_ACQUIRE_ROUNDS_PER_RUN,
        )
        if reservation is None:
            paid_budget_exhausted = True
            log(
                "      [mine] durable paid-acquisition budget exhausted; "
                "no provider call made"
            )
            break
        paid_rounds_used += 1
        variant = (
            int(query_variant or 0)
            + int(reservation["plan_round"])
        ) % 8
        variant_plan = dict(task_plan)
        base_name = str(
            task_plan.get("task_name")
            or task_plan.get("benchmark")
            or task_type
        )
        variant_plan["task_name"] = _diversify_query(base_name, variant)
        queries = task_plan.get("exa_queries") or {}
        variant_plan["exa_queries"] = {
            key: _diversify_query(str(query), variant)
            for key, query in queries.items()
        }
        paid_meta: dict = {}
        try:
            result = discover_and_load_hf_dataset(
                variant_plan,
                _diversify_query(description or base_name, variant),
                task_type,
                max(300, len(existing_rows) + requested * 4),
                80,
                log=log,
                meta=paid_meta,
            )
        except Exception as error:
            reconcile_paid_acquisition(
                reservation,
                status="failed",
                detail=f"{type(error).__name__}: {error}",
            )
            log(
                f"      [mine] paid round failed after reservation: "
                f"{type(error).__name__}: {error}"
            )
            continue
        reconcile_paid_acquisition(
            reservation,
            status="completed",
            detail="dataset discovered" if result else "no dataset discovered",
        )
        accept(result, paid_meta, f"paid-round-{round_index + 1}")

    run_paid_rounds_spent = 0
    if paid_rounds_used or paid_budget_exhausted:
        run_paid_rounds_spent = acquisition_budget_snapshot()["run_spent"]

    status = "novel" if novel else "no_novelty"
    report = {
        "requested": requested,
        "candidate_rows": candidate_rows,
        "novel_rows": len(novel),
        "novel_fraction": (
            round(len(novel) / candidate_rows, 4)
            if candidate_rows
            else 0.0
        ),
        "paid_rounds_used": paid_rounds_used,
        "run_paid_rounds_spent": run_paid_rounds_spent,
        "paid_budget_exhausted": paid_budget_exhausted,
        "status": status,
        "source_records": source_records,
        "rejected_sources": rejected_sources,
    }
    if not novel:
        log(
            "      [mine] no novel non-eval real rows found; "
            "marking plan yield as no_novelty"
        )
    return novel[:requested], report


def _web_source_id(url: str) -> str:
    """Registrable-ish source id for a scraped page: its host, else the raw url."""
    from urllib.parse import urlparse

    try:
        host = urlparse(url).netloc
    except Exception:
        host = ""
    return host or (url or "web")


def _exa_round(exa, task_type, plan, description, n_per_label, round_idx,
               seen_texts: set, log=print) -> list[dict]:
    """One Exa acquisition round across all labels/topics; dedups against seen_texts.

    Each accepted doc is tagged with its source URL (`_source`/`_source_record`) so the
    provenance logger can report where web-scraped rows came from and how many per page.
    """
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
                host = _web_source_id(x.url)
                out.append({
                    "text": doc,
                    "label": label,
                    "_source": f"web:{host}",
                    "_source_record": {
                        "kind": "web",
                        "id": host,
                        "url": x.url,
                        "split": "web",
                        "role": "curriculum",
                    },
                })
                kept += 1
            log(f"      [acquire] round {round_idx} {label!r} q={query!r}: kept {kept} new")
        except Exception as e:
            log(f"      [acquire] round {round_idx} {label!r}: Exa error: {e}")
        time.sleep(0.2)
    return out


# ---------------------------------------------------------------------------
# Agentic HuggingFace-dataset discovery (matches the paper's "web research to LOCATE
# datasets → download the ACTUAL data", §6.1). Exa finds candidate HF dataset repos, the
# orchestrator picks the best one and maps its columns to our schema, then we load it with
# datasets.load_dataset(). This replaces "scrape web pages and call the text a dataset".
# ---------------------------------------------------------------------------

def _extract_hf_dataset_ids(text: str) -> list[str]:
    import re as _re
    return _re.findall(r"huggingface\.co/datasets/([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)", text or "")


def _exa_find_hf_dataset_ids(exa, query: str, log=print, n: int = 6) -> list[str]:
    """Search Exa for HuggingFace dataset repos matching the task; return candidate ids."""
    ids: list[str] = []
    try:
        r = tracked_exa_call(
            exa.search_and_contents,
            f"HuggingFace dataset for {query} site:huggingface.co/datasets",
            stage="acquire_dataset_discovery",
            model="search-and-contents",
            num_results=n, type="auto", text={"max_characters": 400},
        )
        for x in r.results:
            ids += _extract_hf_dataset_ids(x.url or "")
            ids += _extract_hf_dataset_ids(getattr(x, "text", "") or "")
    except Exception as e:
        log(f"      [acquire] Exa HF-dataset search failed: {e}")
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _peek_hf_dataset(hf_id: str, log=print):
    """Load 2 rows to expose (config, split_names, columns, sample_row). None on failure.

    Fully defensive (B159): EVERY datasets call — INCLUDING config-name discovery — is
    guarded. An unloadable candidate (a script-based dataset the current `datasets` lib
    refuses, a private/removed/renamed repo, a network hiccup) must be SKIPPED by returning
    None, never crash the run. The previous version called get_dataset_config_names() in the
    for-loop header, outside the try/except, so a single bad Exa candidate (e.g. the
    script-based SemEvalWorkshop/sem_eval_2018_task_1) raised FileNotFoundError and aborted
    the whole pipeline in eval_setup.
    """
    from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
    try:
        extra_cfgs = get_dataset_config_names(hf_id)[:1]
    except Exception as e:
        log(f"      [acquire] peek {hf_id} config-name lookup failed ({str(e)[:80]}); trying default config only")
        extra_cfgs = []
    for cfg in [None] + extra_cfgs:
        try:
            splits = get_dataset_split_names(hf_id, cfg) if cfg is not None else get_dataset_split_names(hf_id)
            train_split = "train" if "train" in splits else splits[0]
            ds = (load_dataset(hf_id, cfg, split=f"{train_split}[:2]")
                  if cfg else load_dataset(hf_id, split=f"{train_split}[:2]"))
            sample = {k: (str(v)[:200]) for k, v in ds[0].items()}
            return cfg, list(splits), ds.column_names, sample, ds.features
        except Exception as e:
            log(f"      [acquire] peek {hf_id} (config={cfg}) failed: {str(e)[:80]}")
            continue
    return None


def _llm_map_dataset(hf_id, cfg, splits, columns, sample_row, task_type, plan, log=print,
                     label_space=None):
    """Ask the orchestrator to map this dataset's columns to our schema. dict or None.

    `label_space` is the task's CLOSED vocabulary. When supplied, the model is told the exact
    permitted target labels and that inventing one is forbidden; `_materialize_from_mapping`
    additionally strips any entry that targets a label outside the space, so a hallucinated class
    cannot survive even if the model ignores the instruction (B259).
    """
    import anthropic
    from config.config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL, orchestrator_client_kwargs
    allowed = sorted(label_space or ())
    if task_type == "classification":
        want = ('{"train_split","test_split","text_col","label_col",'
                '"label_map": {"<raw>":"<one of the task labels>"} (optional)}')
    elif task_type == "NER":
        want = ('{"train_split","test_split","tokens_col" (list of tokens) + "tags_col" '
                '(list of BIO tag ids) OR "text_col" + "entities_col"}')
    else:  # math_reasoning / code_generation / generation
        want = '{"train_split","test_split","question_col","answer_col","cot_col" (optional)}'
    # State the closed vocabulary explicitly. Without it the model was free to invent plausible-
    # sounding classes, and it did: a RouterBench run (two classes, `local`/`route`) accumulated
    # `cloud`, `on_device`, `router` and `remote` — one new hallucinated class per acquire round,
    # because the mapping is re-requested each round and is non-deterministic (B259).
    label_rules = ""
    if task_type == "classification" and allowed:
        label_rules = (
            f"\nThe task's label space is CLOSED and consists of EXACTLY these "
            f"{len(allowed)} labels: {allowed}.\n"
            f"Every value on the right-hand side of `label_map` MUST be one of those labels "
            f"verbatim. You may NOT invent, rename, split, or add a label, however sensible a new "
            f"one would seem. If this dataset's classes do not map cleanly onto that exact set, "
            f'reply {{"suitable": false}} — a partial or approximate mapping is worse than no '
            f"dataset, because a class outside the space cannot be scored and becomes training "
            f"noise. Omit any source value you cannot map; omitted values are discarded.\n"
        )
    prompt = (
        f"We want to fine-tune for task_type={task_type} "
        f"(name={plan.get('task_name', task_type)}, labels={allowed or plan.get('labels', [])}).\n"
        f"HuggingFace dataset: {hf_id} (config={cfg}); splits={splits}; columns={columns}\n"
        f"One sample row: {sample_row}\n"
        f"{label_rules}\n"
        f"If this dataset is a GOOD fit, reply with STRICT JSON mapping its columns to our "
        f"schema: {want}. If it is NOT a suitable dataset for this task, reply exactly "
        f'{{"suitable": false}}. JSON only, no prose.'
    )
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs())
        resp = tracked_anthropic_messages_create(
            client.messages,
            stage="acquire_schema_mapping",
            model=ORCHESTRATOR_MODEL,
            max_tokens=MIN_THINKING_SAFE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response_text(resp)
        import json as _json, re as _re
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        obj = _json.loads(m.group()) if m else {}
        if not isinstance(obj, dict) or obj.get("suitable") is False:
            return None
        return obj if (obj.get("text_col") or obj.get("question_col") or obj.get("tokens_col")) else None
    except Exception as e:
        from agent.llm_errors import raise_if_fatal
        raise_if_fatal(e, "acquire")
        log(f"      [acquire] column-mapping LLM call failed for {hf_id}: {str(e)[:80]}")
        return None


def _mapped_split_names(splits, mapping):
    tr = mapping.get("train_split") or ("train" if "train" in splits else splits[0])
    te = mapping.get("test_split") or ("test" if "test" in splits else
                                       ("validation" if "validation" in splits else tr))
    return tr, te


def _materialize_from_mapping(hf_id, cfg, splits, mapping, task_type, max_train, max_test,
                              log=print, label_space=None):
    """Load train/test with the LLM's column mapping and convert to our example dicts.

    When `label_space` is supplied the mapping is sanitized first (entries targeting a label
    outside the space are discarded) and any row whose resulting label is still outside the space
    is dropped. The old code applied `lmap.get(str(lab), lab)`, so an unmapped source value passed
    through VERBATIM into training data — the mechanism by which `LOW`/`MEDIUM`/`HIGH` complexity
    tiers and four invented routing classes reached a two-class task (B259).
    """
    from datasets import load_dataset
    from data.label_space import describe_rejected_labels, sanitize_label_map

    allowed = set(label_space or ())
    tr, te = _mapped_split_names(splits, mapping)

    def _load(split, n):
        return (load_dataset(hf_id, cfg, split=f"{split}[:{n}]")
                if cfg else load_dataset(hf_id, split=f"{split}[:{n}]"))

    def _convert(ds):
        out = []
        if task_type == "classification":
            tcol, lcol = mapping.get("text_col"), mapping.get("label_col")
            lmap, dropped_targets = sanitize_label_map(mapping.get("label_map"), allowed)
            if dropped_targets:
                log(
                    f"      [acquire] IGNORED {len(dropped_targets)} label_map target(s) that are "
                    f"not in the task's label space: {dropped_targets}. The orchestrator proposed "
                    "classes this task does not have; they are discarded rather than trained on."
                )
            label_names = ds.features[lcol].names if hasattr(ds.features.get(lcol), "names") else None
            out_of_vocab: dict[str, int] = {}
            for ex in ds:
                lab = ex[lcol]
                if isinstance(lab, int) and label_names:
                    lab = label_names[lab]
                lab = str(lmap.get(str(lab), lab))
                if allowed and lab not in allowed:
                    # Unmapped or mis-mapped: discard. Passing it through is what put four
                    # invented classes into a two-class curriculum.
                    out_of_vocab[lab] = out_of_vocab.get(lab, 0) + 1
                    continue
                if ex.get(tcol):
                    out.append({"text": str(ex[tcol]), "label": lab})
            if out_of_vocab:
                log(
                    f"      [acquire] dropped {sum(out_of_vocab.values())} row(s) whose label is "
                    f"outside the task's closed vocabulary [{describe_rejected_labels(out_of_vocab)}]"
                )
        elif task_type == "NER":
            if mapping.get("tokens_col"):
                tok, tag = mapping["tokens_col"], mapping["tags_col"]
                names = getattr(getattr(ds.features.get(tag), "feature", None), "names", None)
                for ex in ds:
                    text, spans = _bio_to_spans(ex[tok], ex[tag], names)
                    if text.strip():
                        out.append({"text": text, "entities": spans})
            else:
                tcol, ecol = mapping.get("text_col"), mapping.get("entities_col")
                for ex in ds:
                    if ex.get(tcol):
                        ents = ex.get(ecol) or []
                        out.append({"text": str(ex[tcol]), "entities": ents if isinstance(ents, list) else []})
        else:
            qcol, acol, ccol = mapping.get("question_col"), mapping.get("answer_col"), mapping.get("cot_col")
            for ex in ds:
                if ex.get(qcol) and ex.get(acol) is not None:
                    row = {"text": str(ex[qcol]), "answer": str(ex[acol]), "label": task_type}
                    if ccol and ex.get(ccol):
                        row["cot_reasoning"] = str(ex[ccol])
                    out.append(row)
        return out

    try:
        train = _convert(_load(tr, max_train))
        test = _convert(_load(te, max_test))
    except Exception as e:
        log(f"      [acquire] materialize {hf_id} failed: {str(e)[:100]}")
        return None
    return (train, test) if train and test else None


_KNOWN_DISCOVERED_BENCHMARKS = {
    "apps": ("apps", "code_generation", "APPS introductory"),
    "humaneval": ("humaneval", "code_generation", "HumanEval"),
    "mbpp": ("mbpp", "code_generation", "MBPP"),
    "bc5cdr": ("bc5cdr", "NER", "BC5CDR"),
    "gsm8k": ("gsm8k", "math_reasoning", "GSM8K"),
    "samsum": ("samsum", "generation", "SAMSum"),
}


def _known_discovered_benchmark(hf_id: str):
    """Return the converter route for a recognized benchmark repository."""
    repo = str(hf_id or "").rstrip("/").rsplit("/", 1)[-1].lower()
    normalized_repo = "".join(ch for ch in repo if ch.isalnum())
    return _KNOWN_DISCOVERED_BENCHMARKS.get(normalized_repo)


def _validate_discovered_splits(
    result,
    task_type: str,
    *,
    source: str,
) -> tuple[list[dict], list[dict]]:
    """Validate an agent-discovered result before it can outrank local data."""
    if (
        not isinstance(result, (list, tuple))
        or len(result) != 2
        or not isinstance(result[0], list)
        or not isinstance(result[1], list)
        or not result[0]
        or not result[1]
    ):
        raise ValueError(f"{source}: discovered result must contain non-empty train/test lists")
    train, test = result
    required = required_fields_for_task(task_type)
    validate_rows(train, required, bundle_name=source, split="train")
    validate_rows(test, required, bundle_name=source, split="test")

    overlap = normalized_text_overlap(train, test)
    if overlap:
        raise ValueError(
            f"{source}: normalized train/test text overlap ({len(overlap)} rows)"
        )

    if task_type == "classification":
        distinct = {str(row.get("label")) for row in train}
        if len(distinct) < 2:
            raise ValueError(
                f"{source}: classification train split has only "
                f"{len(distinct)} distinct label(s)"
            )
    return train, test


def _accept_discovered_result(
    result,
    task_type: str,
    log=print,
    source="agentic",
    requested_benchmark: str | None = None,
):
    if result is None:
        return None
    try:
        requested_key = _resolve_benchmark_key(requested_benchmark)
        if (
            requested_key == "apps"
            and isinstance(result, (list, tuple))
            and len(result) == 2
            and isinstance(result[0], list)
            and isinstance(result[1], list)
        ):
            from data.loaders.apps import (
                remove_apps_train_fingerprint_overlap,
            )

            train, test = result
            train, fingerprint_removed = (
                remove_apps_train_fingerprint_overlap(train, test)
            )
            train, text_removed = remove_normalized_train_overlap(
                train,
                test,
            )
            test = [
                row
                for row in test
                if row.get("runner_compatible", True) is not False
            ]
            if fingerprint_removed or text_removed:
                log(
                    f"      [acquire] {source} APPS decontamination removed "
                    f"{fingerprint_removed} URL/solution and {text_removed} "
                    "normalized-text train row(s)"
                )
            result = (train, test)
        train, test = _validate_discovered_splits(
            result,
            task_type,
            source=source,
        )
        if requested_key == "apps" and any(
            not isinstance(row.get("input_output"), dict)
            for row in train + test
        ):
            raise ValueError(
                f"{source}: requested APPS but discovered rows do not preserve "
                "the canonical APPS input_output schema"
            )
        if requested_key == "mbpp" and any(
            not isinstance(row.get("test_list"), list)
            for row in train + test
        ):
            raise ValueError(
                f"{source}: requested MBPP but discovered rows do not preserve "
                "the canonical MBPP test_list schema"
            )
        return train, test
    except (TypeError, ValueError) as error:
        log(
            f"      [acquire] {source} dataset REJECTED by schema/integrity/"
            f"overlap validation ({error}); falling back"
        )
        return None


# Overall wall-clock budget for the entire agentic-discovery phase (Exa search + candidate
# peeks + one download). Ample for a healthy hub (~30-90s); the worker is killed past this
# so a transient hub outage (HTTP 504 retry storms) can't stall the run — acquire_dataset
# then degrades to the web-scrape+synthesis ladder.
DISCOVERY_TIMEOUT_S = 600


def _discover_worker(plan, description, task_type, max_train, max_test, q):
    """Run the Exa→peek→map→download pipeline and return a result dict via `q`:
    {"train","test","source","source_records","eval_ban","logs"}.

    Executed in a CHILD process so a stuck HF-hub call can be bounded by KILLING the child,
    not by interrupting this interpreter (B159). The earlier in-process SIGALRM timeout
    fired mid-import of `datasets`, leaving it "partially initialized" — which broke every
    later dataset peek AND cascaded into the unsloth/torch `_inductor` import at train time
    ('torch has no attribute _utils'). A separate process cannot corrupt the parent's
    import state, so terminating it is always safe.
    """
    logs: list[str] = []

    def log(m):
        logs.append(m)

    out = {
        "train": None, "test": None, "source": None,
        "source_records": [], "eval_ban": [], "logs": logs,
    }
    try:
        from config.config import EXA_API_KEY
        from exa_py import Exa
        exa = Exa(api_key=EXA_API_KEY)
    except Exception as e:  # noqa: BLE001
        log(f"      [acquire] Exa init failed: {str(e)[:80]}")
        q.put(out)
        return

    # Search a few COMPLEMENTARY queries so a mismatched/obscure benchmark name doesn't
    # starve the candidate pool of the canonical, loadable dataset (B159). E.g. the planner
    # named "SemEval-2018 Task 1" for a generic 6-class emotion task, whose only Exa hits
    # were script-based repos that no longer load; querying the label set as well surfaces
    # the clean single-label dataset. Queries merged in priority order, deduped.
    seen_q, queries = set(), []
    for qq in (plan.get("benchmark"), plan.get("task_name"),
               " ".join(str(x) for x in (plan.get("labels") or [])[:6]) or None,
               description or None):
        qs = (str(qq).strip() if qq else "")
        if qs and qs.lower() not in seen_q:
            seen_q.add(qs.lower())
            queries.append(qs)
    if not queries:
        queries = [task_type]
    candidates, seen_c = [], set()
    for qq in queries[:3]:
        for hf_id in _exa_find_hf_dataset_ids(exa, f"{qq} {task_type}", log=log):
            if hf_id not in seen_c:
                seen_c.add(hf_id)
                candidates.append(hf_id)
    if not candidates:
        log("      [acquire] no candidate HF datasets found via Exa")
        q.put(out)
        return
    log(f"      [acquire] Exa-discovered candidate HF datasets: {candidates[:8]}")
    for hf_id in candidates[:6]:
        known = _known_discovered_benchmark(hf_id)
        if known is not None:
            _key, expected_task_type, display_name = known
            requested_key = _resolve_benchmark_key(plan.get("benchmark"))
            if requested_key is not None and _key != requested_key:
                log(
                    f"      [acquire] {hf_id}: REJECTED — requested canonical "
                    f"benchmark {requested_key!r}, discovered {_key!r}"
                )
                continue
            if task_type != expected_task_type:
                log(
                    f"      [acquire] {hf_id}: REJECTED — known {display_name} "
                    f"schema is {expected_task_type}, not requested {task_type}"
                )
                continue
            benchmark_meta: dict = {}
            converted = load_benchmark_dataset(
                {"benchmark": display_name, "task_type": task_type},
                log=log,
                max_train=max_train,
                max_test=max_test,
                meta=benchmark_meta,
            )
            accepted = _accept_discovered_result(
                converted,
                task_type,
                log=log,
                source=f"agentic {display_name}",
                requested_benchmark=plan.get("benchmark"),
            )
            if accepted is None:
                continue
            train, test = accepted
            log(
                f"      [acquire] routed discovered {hf_id!r} through the "
                f"task-aware {display_name} converter"
            )
            out["train"], out["test"] = train, test
            out["source"] = benchmark_meta.get(
                "source",
                f"agentic {display_name} benchmark via {hf_id!r}",
            )
            out["source_records"] = list(
                benchmark_meta.get("source_records") or []
            )
            out["eval_ban"] = list(benchmark_meta.get("eval_ban") or [])
            q.put(out)
            return

        peek = _peek_hf_dataset(hf_id, log=log)
        if peek is None:
            continue
        cfg, splits, columns, sample, _features = peek
        # `plan["labels"]` carries the run's CLOSED label vocabulary. curate writes the pinned
        # space (from the frozen eval set) into it before mining, and the plan is the one object
        # that already crosses this subprocess boundary — so it is the carrier rather than a new
        # parameter that would have to be pickled separately.
        _plan_labels = {str(v) for v in (plan.get("labels") or []) if str(v).strip()}
        mapping = _llm_map_dataset(hf_id, cfg, splits, columns, sample, task_type, plan, log=log,
                                   label_space=_plan_labels)
        if not mapping:
            log(f"      [acquire] {hf_id}: orchestrator judged it unsuitable / unmappable")
            continue
        result = _materialize_from_mapping(hf_id, cfg, splits, mapping, task_type, max_train,
                                          max_test, log=log, label_space=_plan_labels)
        if result is not None:
            accepted = _accept_discovered_result(
                result,
                task_type,
                log=log,
                source=f"agentic {hf_id}",
                requested_benchmark=plan.get("benchmark"),
            )
            if accepted is None:
                continue
            train, test = accepted
            log(f"      [acquire] loaded AGENTIC HF dataset {hf_id!r} (config={cfg}): "
                f"train={len(train)} test={len(test)}")
            out["train"], out["test"] = train, test
            out["source"] = f"agentic HF dataset {hf_id!r} (Exa-discovered, LLM-mapped, load_dataset)"
            train_split, test_split = _mapped_split_names(splits, mapping)
            records = _source_records(
                hf_id, cfg, {"train": train_split, "test": test_split}
            )
            out["source_records"] = records
            out["eval_ban"] = [dict(records[1])]
            q.put(out)
            return
    q.put(out)


def discover_and_load_hf_dataset(plan, description, task_type, max_train, max_test,
                                 log=print, meta=None):
    """Agentic path: Exa → candidate HF datasets → LLM picks+maps → load_dataset.

    Runs in a killable `fork` subprocess with a hard wall-clock timeout (B159). Fork (not
    spawn) is required because the pipeline entrypoint run.py is not import-safe; fork also
    inherits the parent's env/clients cleanly. Fork is safe HERE because acquisition runs in
    eval_setup, before any torch/CUDA initialization. Returns (train, test) or None; on
    timeout or any failure it returns None and the caller falls back to the scrape ladder.
    """
    import multiprocessing as _mp
    try:
        ctx = _mp.get_context("fork")
    except ValueError:
        # No fork (non-POSIX): run inline. Defensive try/except in the callees still
        # prevents crashes; only the hard hang-backstop is unavailable.
        q_inline: list = []

        class _Q:
            def put(self, v):
                q_inline.append(v)
        _discover_worker(plan, description, task_type, max_train, max_test, _Q())
        result = q_inline[0] if q_inline else None
    else:
        q = ctx.Queue()
        p = ctx.Process(target=_discover_worker,
                        args=(plan, description, task_type, max_train, max_test, q),
                        daemon=True)
        p.start()
        # get() BEFORE join() to avoid the feeder-thread deadlock; the timeout doubles as
        # the hang backstop.
        try:
            result = q.get(timeout=DISCOVERY_TIMEOUT_S)
            timed_out = False
        except Exception:  # queue.Empty on timeout, or worker died without putting
            result = None
            timed_out = True
        if timed_out and p.is_alive():
            log(f"      [acquire] agentic discovery exceeded {DISCOVERY_TIMEOUT_S}s "
                f"(flaky/unreachable hub); terminating and falling back to scrape ladder")
            p.terminate()
        p.join(5)
        if p.is_alive():  # still finishing/flushing after a successful get — reap quietly
            p.terminate()
            p.join(5)

    if not result:
        return None
    for line in result.get("logs", []):
        log(line)
    train, test = result.get("train"), result.get("test")
    if train and test:
        accepted = _accept_discovered_result(
            (train, test),
            task_type,
            log=log,
            source="agentic discovery",
            requested_benchmark=plan.get("benchmark"),
        )
        if accepted is None:
            return None
        train, test = accepted
        if meta is not None:
            if result.get("source"):
                meta["source"] = result["source"]
            meta["source_records"] = list(result.get("source_records") or [])
            meta["eval_ban"] = list(result.get("eval_ban") or [])
        return train, test
    return None


def _stratified_take(rows: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Take up to n rows, balanced across labels (round-robin by class) so a capped subset
    stays class-balanced instead of inheriting the source imbalance (B161 stratification)."""
    import random as _random
    if n <= 0 or len(rows) <= n:
        return rows
    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(str(r.get("label", r.get("type", "_"))), []).append(r)
    rng = _random.Random(seed)
    for lst in by_label.values():
        rng.shuffle(lst)
    out, labels = [], list(by_label.keys())
    i = 0
    while len(out) < n and any(by_label.values()):
        lbl = labels[i % len(labels)]
        if by_label[lbl]:
            out.append(by_label[lbl].pop())
        i += 1
        if i > n * len(labels) + len(labels):  # safety
            break
    return out[:n]


def _local_dataset_dir() -> Path:
    """Resolve local bundles without importing API-key-bearing project config."""
    project_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("SLM_LOCAL_DATASET_DIR", project_root / "data" / "local"))


def _local_manifest_match(
    plan: dict,
    task_type: str,
    bundle_name: str,
    manifest: dict,
) -> tuple[int, float] | None:
    """Return a strong-match score, or reject an unrelated same-task bundle."""
    if manifest.get("task_type") != task_type:
        return None
    requested_key = _resolve_benchmark_key(plan.get("benchmark"))
    manifest_key = (
        _resolve_benchmark_key(manifest.get("name") or bundle_name)
        or _resolve_benchmark_key(manifest.get("hf_id"))
    )
    if requested_key is not None:
        return (2, 1.0) if manifest_key == requested_key else None

    requested_labels = {
        str(value).strip().casefold()
        for value in (plan.get("labels") or [])
        if str(value).strip()
    }
    manifest_labels = {
        str(value).strip().casefold()
        for value in (manifest.get("labels") or [])
        if str(value).strip()
    }
    if not requested_labels or not manifest_labels:
        return None
    intersection = len(requested_labels & manifest_labels)
    union = len(requested_labels | manifest_labels)
    jaccard = intersection / union if union else 0.0
    requested_coverage = intersection / len(requested_labels)
    required = set(required_fields_for_task(task_type))
    declared_required = set(
        (manifest.get("row_schema") or {}).get("required") or []
    )
    schema_matches = (
        required <= declared_required
        if declared_required
        else requested_labels == manifest_labels
    )
    if (
        not schema_matches
        or jaccard < 0.80
        or requested_coverage < 0.80
    ):
        return None
    return (1 if requested_labels == manifest_labels else 0, jaccard)


def load_local_dataset(plan, task_type, max_train, max_test, log=print, meta=None):
    """Load a clean OFFLINE dataset from ``SLM_LOCAL_DATASET_DIR`` that matches the task
    (B161 local fallback). Known benchmark aliases require the exact bundle; otherwise a
    bundle must pass a strong task+label+schema match. Unrelated same-task bundles are
    rejected so the caller can continue its acquisition ladder.

    The complete local train/test files are checksum/schema/count checked and rejected on
    normalized text overlap before caps are applied. The default path is derived from this
    module, so local loading never imports config.config or requires paid-API keys.
    """
    import json as _json
    import os as _os
    LOCAL_DATASET_DIR = _local_dataset_dir()
    if not _os.path.isdir(LOCAL_DATASET_DIR):
        return None

    best = None  # (match score, name, manifest)
    for name in sorted(_os.listdir(LOCAL_DATASET_DIR)):
        man_path = _os.path.join(LOCAL_DATASET_DIR, name, "manifest.json")
        if not _os.path.exists(man_path):
            continue
        try:
            with open(man_path, encoding="utf-8") as handle:
                man = _json.load(handle)
        except Exception:
            continue
        score = _local_manifest_match(plan, task_type, name, man)
        if score is None:
            log(
                f"      [acquire] LOCAL candidate {name!r} rejected: "
                "no explicit benchmark or strong task+label+schema match"
            )
            continue
        if best is None or score > best[0]:
            best = (score, name, man)
    if best is None:
        return None

    _, name, man = best
    ddir = _os.path.join(LOCAL_DATASET_DIR, name)
    schema_version = man.get("schema_version")
    checksums_verified = verify_bundle_checksums(ddir)
    if schema_version == 2 and not checksums_verified:
        raise ValueError(
            f"{name}: schema-v2 bundle requires checksums.sha256; refusing unverified data"
        )
    if schema_version == 1 and not checksums_verified:
        log(
            f"      [acquire] legacy schema-v1 local bundle {name!r} loaded "
            "without checksums (compatibility mode)"
        )
    if schema_version not in (1, 2):
        raise ValueError(
            f"{name}: missing or unsupported manifest schema_version={schema_version!r}; "
            "only explicit legacy schema-v1 or verified schema-v2 bundles are accepted"
        )

    def _read(split):
        p = _os.path.join(ddir, f"{split}.jsonl")
        if not _os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as f:
            return [_json.loads(ln) for ln in f if ln.strip()]

    all_train, all_test = _read("train"), _read("test")
    if not all_train or not all_test:
        return None

    counts = man.get("counts") or {}
    if counts and (
        counts.get("train") != len(all_train) or counts.get("test") != len(all_test)
    ):
        raise ValueError(
            f"{name}: manifest counts do not match local JSONL "
            f"(manifest={counts}, actual train={len(all_train)} test={len(all_test)})"
        )

    required = set((man.get("row_schema") or {}).get("required") or [])
    if required:
        for split, rows in (("train", all_train), ("test", all_test)):
            for index, row in enumerate(rows):
                missing = required - set(row)
                if missing:
                    raise ValueError(
                        f"{name}: {split} row {index} missing schema fields {sorted(missing)}"
                    )
    canonical_required = required_fields_for_task(task_type)
    validate_rows(
        all_train,
        canonical_required,
        bundle_name=name,
        split="train",
    )
    validate_rows(
        all_test,
        canonical_required,
        bundle_name=name,
        split="test",
    )

    overlap = normalized_text_overlap(all_train, all_test)
    if overlap:
        raise ValueError(
            f"{name}: normalized train/test text overlap ({len(overlap)} rows), "
            f"sample={sorted(overlap)[:3]!r}"
        )
    if name == "apps":
        from data.loaders.apps import apps_fingerprint_overlap

        fingerprint_overlap = apps_fingerprint_overlap(
            all_train,
            all_test,
        )
        if fingerprint_overlap:
            raise ValueError(
                f"{name}: train/test URL or solution fingerprint overlap "
                f"({len(fingerprint_overlap)} fingerprints)"
            )

    eligible_test = all_test
    if name == "apps":
        eligible_test = [
            row
            for row in all_test
            if row.get("runner_compatible", True) is not False
        ]
        skipped = len(all_test) - len(eligible_test)
        if skipped:
            log(
                f"      [acquire] APPS runner compatibility: skipped "
                f"{skipped} explicitly marked eval row(s)"
            )
    train = _stratified_take(all_train, max_train)
    test = _stratified_take(eligible_test, max_test, seed=7)
    if not train or not test:
        return None

    if task_type == "NER":
        label_values = sorted({
            str(entity.get("type")) for row in train for entity in row.get("entities", [])
            if entity.get("type")
        })
    else:
        label_values = sorted({str(row.get("label")) for row in train if row.get("label")})
    log(f"      [acquire] LOCAL fallback: loaded {name!r} ({man.get('hf_id')}) — "
        f"train={len(train)} test={len(test)}  labels={label_values}")
    if meta is not None:
        meta["source"] = f"local dataset {name!r} ({man.get('hf_id')}, offline copy)"
        records = list((man.get("provenance") or {}).get("records") or [])
        if not records:
            records = [{
                "kind": "hf", "id": man.get("hf_id"), "config": man.get("config"),
                "split": "train+test(local)", "url": man.get("source_url"),
                "role": "source",
            }]
        meta["source_records"] = records
        # Only explicit held-out split restrictions are forwarded. Provenance records are
        # not themselves enforcement rules.
        meta["eval_ban"] = list(man.get("eval_ban") or [])
    return train, test


def acquire_dataset(plan: dict, description: str = "", n_per_label: int = DEFAULT_N_PER_LABEL,
                    test_fraction: float = 0.3, target_examples: int = DEFAULT_TARGET_EXAMPLES,
                    benchmark_max_train: int = 300, benchmark_max_test: int = 80,
                    log=print, meta: dict | None = None):
    """
    Acquire a labeled dataset per the task plan, following the acquisition ladder (B139):

      1. A clean schema-aware local offline bundle.
      2. Deterministic remote Stage-0 for a known benchmark.
      3. Agentic Exa+orchestrator HF discovery for unknown/unavailable datasets.
      4. Otherwise BOUNDED + DIVERSIFIED Exa rounds: up to MAX_ACQUIRE_ROUNDS, each round
         rephrases the queries so re-runs fetch NEW documents (not duplicates), deduping,
         stopping once `target_examples` is reached.
      5. If still below `target_examples * MIN_VIABLE_FRACTION`, TOP UP with verified LLM
         synthesis (`synthesize_seed_examples`) — deduped and label-validated.

    Readiness tests may set ``SLM_AGENT_FIRST_DATASET_DISCOVERY=1`` to exercise the
    paid discovery path first. In that mode the fallback order is local, then deterministic
    Stage-0. Without the flag, no paid discovery happens before local and Stage-0 fail.

    `target_examples` is an UPPER bound (quality-over-quantity); we do not chase it past
    what clean sources provide, but we do insist on the viability floor before proceeding.

    Returns (train, test). `meta["source"]` records provenance (benchmark / web+synth mix).
    """
    task_type = plan["task_type"]
    agent_first = os.environ.get("SLM_AGENT_FIRST_DATASET_DISCOVERY", "0") == "1"

    if agent_first:
        log("      [acquire] readiness mode: trying agentic dataset discovery first")
        prior_meta = dict(meta) if meta is not None else None
        disc = discover_and_load_hf_dataset(
            plan, description, task_type, benchmark_max_train, benchmark_max_test,
            log=log, meta=meta,
        )
        accepted = _accept_discovered_result(
            disc,
            task_type,
            log=log,
            source="agent-first discovery",
            requested_benchmark=plan.get("benchmark"),
        )
        if accepted is not None:
            return accepted
        if meta is not None and prior_meta is not None:
            meta.clear()
            meta.update(prior_meta)

    # Clean local bundles are the default and the first non-paid fallback in readiness mode.
    local = load_local_dataset(plan, task_type, benchmark_max_train, benchmark_max_test,
                               log=log, meta=meta)
    if local is not None:
        return local

    # Deterministic remote Stage-0 is free of Exa/Anthropic calls.
    real = load_benchmark_dataset(plan, log=log, meta=meta,
                                  max_train=benchmark_max_train, max_test=benchmark_max_test)
    if real is not None:
        return real

    if not agent_first:
        # Only now may the default path make paid Exa/orchestrator discovery calls.
        prior_meta = dict(meta) if meta is not None else None
        disc = discover_and_load_hf_dataset(
            plan, description, task_type, benchmark_max_train, benchmark_max_test,
            log=log, meta=meta,
        )
        accepted = _accept_discovered_result(
            disc,
            task_type,
            log=log,
            source="agentic discovery",
            requested_benchmark=plan.get("benchmark"),
        )
        if accepted is not None:
            return accepted
        if meta is not None and prior_meta is not None:
            meta.clear()
            meta.update(prior_meta)

    # Stage 2 (LAST-RESORT fallback): only if no real dataset could be located/loaded do we
    # fall back to the bounded, diversified web-scrape + verified-synthesis ladder below.
    log("      [acquire] no downloadable dataset found — falling back to web-scrape + synthesis ladder")
    from config.config import EXA_API_KEY
    from exa_py import Exa

    exa = Exa(api_key=EXA_API_KEY)
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
        # Prefer the task plan's declared entity types over the CoNLL default, so a
        # domain task (e.g. BC5CDR CHEMICAL/DISEASE) allow-lists its own schema instead of
        # having every domain span rejected as a bad type.
        _plan_types = frozenset(
            str(label).strip().upper()
            for label in (plan.get("labels") or [])
            if str(label).strip()
        ) or None
        examples = _annotate_ner_entities(
            examples,
            log=log,
            allowed_types=_plan_types,
        )

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
    from config.config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL, orchestrator_client_kwargs

    existing_texts = existing_texts or set()
    labels = plan.get("labels") or []
    task_name = plan.get("task_name", task_type)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs())

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
        resp = tracked_anthropic_messages_create(
            client.messages,
            stage="acquire_seed_synthesis",
            model=ORCHESTRATOR_MODEL, max_tokens=4096, temperature=1.0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response_text(resp)
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


# The window that is BOTH annotated and stored. Previously the annotator saw text[:500] while
# the row kept the FULL passage, so every entity past character 500 became an unlabeled span —
# i.e. a silent false negative teaching the model that those spans are not entities. The
# annotated window and the stored text must be the same string; this is that string's length.
_NER_ANNOTATION_WINDOW_CHARS = int(
    os.environ.get("SLM_NER_ANNOTATION_WINDOW", "500")
)

# Standard CoNLL-style types plus the coarse catch-all. Anything outside this set (and outside
# the task plan's declared types) is dropped rather than admitted as a novel gold type.
_DEFAULT_NER_TYPES = frozenset({
    "PER", "PERSON", "ORG", "ORGANIZATION", "LOC", "LOCATION", "GPE", "MISC",
})


def _validate_ner_annotation(
    raw_entities,
    annotated_text: str,
    allowed_types: frozenset[str],
) -> tuple[list[dict], dict[str, int]]:
    """Keep only spans that are exact substrings of `annotated_text` with an allowed type.

    Returns (kept, rejection_counts). Rejections are counted by reason so acquisition noise is
    visible instead of silently shrinking the label set.
    """
    rejected = {"malformed": 0, "not_substring": 0, "bad_type": 0, "duplicate": 0}
    kept: list[dict] = []
    seen: set[tuple[str, str]] = set()
    if not isinstance(raw_entities, list):
        return [], {**rejected, "malformed": 1}
    for entity in raw_entities:
        if not isinstance(entity, dict):
            rejected["malformed"] += 1
            continue
        span = entity.get("text")
        etype = entity.get("type")
        if not isinstance(span, str) or not span.strip() or not isinstance(etype, str):
            rejected["malformed"] += 1
            continue
        span = span.strip()
        normalized_type = etype.strip().upper()
        if span not in annotated_text:
            # The prompt demands an exact substring. A span that is not present is either a
            # hallucination or a normalization drift; either way it cannot be a gold span.
            rejected["not_substring"] += 1
            continue
        if allowed_types and normalized_type not in allowed_types:
            rejected["bad_type"] += 1
            continue
        key = (span, normalized_type)
        if key in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(key)
        kept.append({"text": span, "type": normalized_type})
    return kept, rejected


def _annotate_ner_entities(
    examples: list[dict],
    log=print,
    allowed_types: frozenset[str] | None = None,
    max_attempts: int = 2,
) -> list[dict]:
    """Add gold entity annotations to web-acquired NER passages via the orchestrator.

    Five defects are fixed here relative to the original implementation, all of which turned
    acquisition noise into training signal:

    1. **Window mismatch.** The annotator saw ``text[:500]`` while the emitted row kept the
       FULL passage, so entities beyond char 500 were unlabeled — systematic false negatives
       on exactly the long passages a NER model finds hardest. The row now stores precisely
       the window that was annotated.
    2. **Silent failure became a negative.** ``except Exception: entities = []`` emitted an
       empty-entity gold row, indistinguishable from a genuine negative. Failed rows are now
       DROPPED and counted, never emitted as gold.
    3. **No span validation.** Spans are now required to be exact substrings of the annotated
       window, and types must be in the allowed set.
    4. **API errors were swallowed.** ``raise_if_fatal`` now aborts the run on an
       auth/quota/billing failure, matching every other orchestrator call site. Otherwise a
       dead API key silently produced an entirely unlabeled NER corpus.
    5. **No retry.** A malformed reply now gets one bounded retry before the row is dropped.
    """
    import anthropic, json as _json, re as _re
    from agent.llm_errors import raise_if_fatal
    from config.config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL, orchestrator_client_kwargs

    if allowed_types is None:
        allowed_types = _DEFAULT_NER_TYPES

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, **orchestrator_client_kwargs())
    annotated: list[dict] = []
    stats = {
        "input": len(examples),
        "preannotated": 0,
        "annotated": 0,
        "dropped_call_error": 0,
        "dropped_unparseable": 0,
        "with_entities": 0,
        "empty_after_validation": 0,
        "retried": 0,
    }
    rejections = {"malformed": 0, "not_substring": 0, "bad_type": 0, "duplicate": 0}

    for ex in examples:
        # Skip examples that already carry entities (e.g. synthesized seeds) — don't
        # re-annotate and clobber known-good gold.
        if ex.get("entities"):
            annotated.append(ex)
            stats["preannotated"] += 1
            continue

        # The annotated window IS the stored text. Never annotate a prefix and keep the whole.
        window = str(ex.get("text", ""))[:_NER_ANNOTATION_WINDOW_CHARS]
        if not window.strip():
            stats["dropped_unparseable"] += 1
            continue

        prompt = (
            "Extract all named entities from this text. Return a JSON list of "
            "objects with \"text\" (the exact span as it appears in the text) and "
            f"\"type\" (one of: {', '.join(sorted(allowed_types))}). "
            "Rules for consistency: use the EXACT substring from the text for each "
            "span; for overlapping candidates prefer the LONGEST span; do NOT emit "
            "nested or duplicated spans; assign each span exactly one type. "
            "Return [] if there are no entities.\n\n"
            f"Text: {window}\n\n"
            "Reply with JSON only, no explanation."
        )

        entities = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = tracked_anthropic_messages_create(
                    client.messages,
                    stage="acquire_ner_annotation",
                    model=ORCHESTRATOR_MODEL,
                    max_tokens=MIN_THINKING_SAFE_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:  # noqa: BLE001
                # Auth/quota/billing failures recur on every subsequent call; aborting is the
                # only honest outcome, and matches iterate/escalate/downward_probe.
                raise_if_fatal(exc, "acquire_ner_annotation")
                if attempt >= max_attempts:
                    log(
                        f"      [acquire] NER annotation call failed after {attempt} "
                        f"attempt(s) ({type(exc).__name__}: {str(exc)[:120]}); "
                        "DROPPING this passage rather than emitting it as a negative"
                    )
                    stats["dropped_call_error"] += 1
                    break
                stats["retried"] += 1
                continue

            raw = response_text(response)
            match = _re.search(r"\[.*\]", raw, _re.DOTALL)
            if not match:
                if attempt >= max_attempts:
                    log(
                        "      [acquire] NER annotation returned no JSON list after "
                        f"{attempt} attempt(s); DROPPING this passage"
                    )
                    stats["dropped_unparseable"] += 1
                    break
                stats["retried"] += 1
                continue
            try:
                entities = _json.loads(match.group())
            except _json.JSONDecodeError:
                if attempt >= max_attempts:
                    log(
                        "      [acquire] NER annotation JSON was invalid after "
                        f"{attempt} attempt(s); DROPPING this passage"
                    )
                    stats["dropped_unparseable"] += 1
                    break
                stats["retried"] += 1
                continue
            break

        if entities is None:
            continue

        kept, row_rejections = _validate_ner_annotation(
            entities,
            window,
            allowed_types,
        )
        for reason, count in row_rejections.items():
            rejections[reason] += count
        stats["annotated"] += 1
        if kept:
            stats["with_entities"] += 1
        else:
            # A validated-empty row IS a legitimate negative: the call succeeded, the reply
            # parsed, and no span survived validation. That is different from a failure, and
            # it is kept — negatives are necessary training signal.
            stats["empty_after_validation"] += 1
        annotated.append({"text": window, "entities": kept})

    dropped = stats["dropped_call_error"] + stats["dropped_unparseable"]
    log(
        f"      [acquire] NER annotation: {stats['annotated']} annotated "
        f"({stats['with_entities']} with entities, "
        f"{stats['empty_after_validation']} validated-empty), "
        f"{stats['preannotated']} pre-annotated, {dropped} DROPPED "
        f"({stats['dropped_call_error']} call errors, "
        f"{stats['dropped_unparseable']} unparseable), "
        f"{stats['retried']} retries — of {stats['input']} input passages"
    )
    if any(rejections.values()):
        log(
            "      [acquire] NER span rejections: "
            f"not_substring={rejections['not_substring']} "
            f"bad_type={rejections['bad_type']} "
            f"malformed={rejections['malformed']} "
            f"duplicate={rejections['duplicate']}"
        )
    if dropped:
        log(
            f"      [acquire] WARNING — {dropped}/{stats['input']} passage(s) were dropped "
            "rather than emitted as entity-free gold. Previously these became negatives, "
            "which trains the model to predict 'no entities'."
        )
    return annotated
