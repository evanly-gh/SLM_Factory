"""Execute bounded declarative dataset rebuild plans."""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from collections.abc import Callable

from agent.checkpoint import atomic_write_jsonl
from agent.data_rebuild import (
    fallback_data_rebuild_plan,
    normalize_data_rebuild_plan,
    plan_budget_identity,
    remaining_paid_acquire_rounds,
    resample_pool_exhausted,
)
from agent.state import AgentState
from data.curriculum import (
    annotate_cot,
    apply_quality_controls,
    synthesize_examples,
)
from data.loaders.dataset_integrity import normalize_text
from data.loaders.web_acquire import mine_additional_real_rows
from data.provenance import build_source_usage, source_key

# Per-row firewall reporting is bounded: a pathological rebuild could otherwise blocklist
# thousands of rows and bury the log in the very noise this reporting exists to replace.
_FIREWALL_LOG_LIMIT = int(os.environ.get("SLM_FIREWALL_LOG_LIMIT", "20"))


ARTIFACTS_DIR = "artifacts"


def _log(model_id: str, message: str) -> None:
    print(f"[curate][{model_id}] {message}")


def _entropy_seed() -> int:
    """A fresh, non-deterministic seed per sampler call (redesign 2026-07-31).

    Curation is deliberately NON-deterministic: each reshuffle/synthesis draw uses real
    OS entropy so repeated rebuilds genuinely vary run to run. There is no plan-identity
    seed and no reproducibility guarantee across checkpoint resumes.
    """
    return int.from_bytes(os.urandom(8), "big")


def _verifier_for(task_type: str, state: AgentState):
    """Return a correctness verifier for synthetic rows, or None if none applies.

    Enhanced in Task 5 with real math/code verifiers; for now generation-family rows
    are kept after standard quality controls (verify_fn=None) and classification/NER rows
    inherit their anchor's label, so they need no verifier here — they get a separate
    teacher label-verification pass instead.
    """
    return None


def _row_text(row: dict) -> object:
    return row.get("text", row.get("prompt", ""))


def _normalized_eval_texts(eval_set) -> set[str]:
    if eval_set is None:
        return set()
    rows = getattr(eval_set, "all", [])
    if not isinstance(rows, (list, tuple)):
        return set()
    texts = {
        normalize_text(_row_text(row))
        for row in rows
        if isinstance(row, dict)
    }
    texts.discard("")
    return texts


def _exclude_eval_rows(
    rows: list[dict],
    eval_set,
    *,
    tally: dict | None = None,
    layer: str | None = None,
    log=None,
) -> tuple[list[dict], int]:
    """Apply the normalized-text eval firewall to candidate training rows.

    When ``tally`` is provided, the number of rows removed at this ``layer`` is accumulated into
    it (creating the key at 0 even when nothing is removed) so a run can durably report firewall
    activity per iteration — including the healthy zero-drop case.

    When ``log`` is provided, each blocked row is reported with its reason and a short redacted
    fingerprint of the offending text. There is exactly one reason a row is blocked here — its
    normalized text is byte-identical to a held-out eval row — but stating it per row means an
    operator can see WHICH rows leaked instead of only a count. The text is truncated and the
    match is reported as a hash so held-out eval content never lands in the run log.
    """
    eval_texts = _normalized_eval_texts(eval_set)
    if tally is not None and layer is not None:
        tally.setdefault(layer, 0)
    if not eval_texts:
        return list(rows), 0
    clean = []
    blocked = []
    for row in rows:
        if isinstance(row, dict) and normalize_text(_row_text(row)) in eval_texts:
            blocked.append(row)
        else:
            clean.append(row)
    removed = len(blocked)
    if tally is not None and layer is not None:
        tally[layer] = tally.get(layer, 0) + removed
    if log and blocked:
        for row in blocked[:_FIREWALL_LOG_LIMIT]:
            normalized = normalize_text(_row_text(row))
            digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8]
            log(
                f"    [firewall:{layer or 'unknown'}] BLOCKED row "
                f"(provenance={row.get('_provenance') or 'unknown'}, "
                f"label={row.get('label') or 'n/a'}, len={len(normalized)} chars, "
                f"text_sha8={digest}): normalized text exactly matches a held-out eval row, "
                "so training on it would leak the eval set"
            )
        if removed > _FIREWALL_LOG_LIMIT:
            log(
                f"    [firewall:{layer or 'unknown'}] ... and "
                f"{removed - _FIREWALL_LOG_LIMIT} more blocked for the same reason"
            )
    return clean, removed


def _read_jsonl(path: str | None) -> list[dict]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as source:
        return [
            value
            for line in source
            if line.strip()
            for value in [json.loads(line)]
            if isinstance(value, dict)
        ]


def _normalized_texts(rows: list[dict]) -> set[str]:
    values = {
        normalize_text(_row_text(row))
        for row in rows
        if isinstance(row, dict)
    }
    values.discard("")
    return values


def _label_key(row: dict, task_type: str) -> str:
    if task_type == "NER":
        entity_types = [
            str(entity.get("type", "?"))
            for entity in row.get("entities", [])
            if isinstance(entity, dict)
        ]
        return entity_types[0] if entity_types else "no_entity"
    return str(row.get("label", row.get("type", "?")))


def _source_key(row: dict, default: str = "unknown") -> str:
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


def _round_robin_sample(
    rows: list[dict],
    *,
    count: int,
    key: Callable[[dict], str],
    seed: int,
) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    names = sorted(groups)
    selected: list[dict] = []
    while len(selected) < count and any(groups.values()):
        for name in names:
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
    return selected


def _balanced_sample(
    rows: list[dict],
    *,
    count: int,
    task_type: str,
    seed: int,
) -> list[dict]:
    if count <= 0:
        return []
    if task_type == "classification":
        return _round_robin_sample(
            rows,
            count=min(count, len(rows)),
            key=lambda row: _label_key(row, task_type),
            seed=seed,
        )
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:count]


def _tag_train_rows(
    rows: list[dict],
    *,
    default_source: str,
) -> list[dict]:
    tagged = []
    for row in rows:
        value = dict(row)
        value.setdefault("_provenance", "train_anchor")
        value.setdefault("_source", default_source)
        tagged.append(value)
    return tagged


def _tag_mined_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            **row,
            "_provenance": "mined_real",
            "_strategy_origin": "mine_new_real_source",
        }
        for row in rows
    ]


def _merge_persistent_train_rows(
    existing_rows: list[dict],
    mined_rows: list[dict],
    eval_set,
    *,
    tally: dict | None = None,
) -> list[dict]:
    """Retain novel real rows for every later rebuild and checkpoint."""
    candidates, _ = _exclude_eval_rows(
        [*existing_rows, *mined_rows],
        eval_set,
        tally=tally,
        layer="persistent_merge",
    )
    merged = []
    seen = set()
    for row in candidates:
        if not isinstance(row, dict):
            continue
        text = normalize_text(_row_text(row))
        if not text or text in seen:
            continue
        seen.add(text)
        merged.append(dict(row))
    return merged


def _synthesize_positive_rows(
    state: AgentState,
    plan: dict,
    train_rows: list[dict],
    *,
    model_id: str,
    seed: int,
    tally: dict | None = None,
) -> list[dict]:
    task_type = state["task_type"]
    # Ungated (redesign 2026-07-31): synthesis is available for EVERY task type and score.
    if os.environ.get("SLM_CHEAP") == "1" or not train_rows:
        _log(model_id, "  Synthesis skipped (cheap mode or no anchors)")
        return []
    from data.synth_client import get_generate_fn, wait_until_available

    logger = lambda message: _log(model_id, message)
    # Synthesis is no longer a hard requirement. If the endpoint does not become reachable
    # within the bounded wait, degrade GRACEFULLY (return no synthetic rows). The caller
    # records an allocation fallback and resample-fill covers the remainder — never a crash.
    if not wait_until_available(log=logger):
        _log(
            model_id,
            "  Synthesis endpoint unavailable — degrading (no synthetic rows this pass)",
        )
        return []
    generate = get_generate_fn(log=logger)

    # SURGICAL portion: spend part of the budget on the classes the model is actually confusing,
    # with the per-pair budget PROPORTIONAL to how often each pair is confused. A pair confused 7
    # times deserves more than one confused twice. Pairs that were targeted before and did not
    # improve are skipped, so the run stops pouring data at something that is not responding.
    surgical_rows: list[dict] = []
    if task_type == "classification":
        surgical_rows = _surgical_synthesize(
            state,
            plan,
            train_rows,
            generate_fn=generate,
            model_id=model_id,
            seed=seed,
        )

    fill_budget = max(0, int(plan["synth_rows"]) - len(surgical_rows))
    if fill_budget <= 0:
        return surgical_rows
    _log(
        model_id,
        f"  ▶ FILL SYNTHESIS: {fill_budget} row(s) balanced across the label space "
        f"(surgical already produced {len(surgical_rows)})",
    )
    anchors = _balanced_sample(
        train_rows,
        count=min(fill_budget, len(train_rows)),
        task_type=task_type,
        seed=seed,
    )
    candidates = synthesize_examples(
        anchors,
        task_type=task_type,
        n=len(anchors),
        generate_fn=generate,
        verify_fn=_verifier_for(task_type, state),
        log=logger,
    )
    generated = [
        {
            **row,
            "_provenance": "synthetic",
            "_strategy_origin": "synthesize",
        }
        for row in candidates
        if isinstance(row, dict)
        and (
            str(row.get("_source", "")).startswith("synth")
            or row.get("_provenance") == "synthetic_positive"
        )
    ]
    generated, removed = _exclude_eval_rows(
        generated, state.get("eval_set"), tally=tally, layer="synthesis",
        log=lambda m: _log(model_id, m))
    if removed:
        _log(model_id, f"  Synthesis eval firewall removed {removed} row(s)")
    return (surgical_rows + generated)[:plan["synth_rows"]]


# Share of a `synthesize` plan's budget spent on the confused classes rather than on balanced
# fill. Kept a minority: surgical rows concentrate on the hardest, most confusable classes, which
# is exactly where a generator is most likely to produce something wrong.
SURGICAL_SYNTH_SHARE = float(os.environ.get("SLM_SURGICAL_SYNTH_SHARE", "0.20"))
SURGICAL_MAX_PAIRS = int(os.environ.get("SLM_SURGICAL_MAX_PAIRS", "5"))
SURGICAL_MIN_ROWS_PER_PAIR = 10
SURGICAL_MAX_ROWS_PER_PAIR = 100


def _pair_key(pair: dict) -> str:
    return f"{pair.get('gold')}->{pair.get('predicted')}"


def _surgical_synthesize(
    state: AgentState,
    plan: dict,
    train_rows: list[dict],
    *,
    generate_fn,
    model_id: str,
    seed: int,
) -> list[dict]:
    """Generate extra GOLD rows for the classes involved in the top confusion pairs.

    Budget per pair is proportional to that pair's confusion count, so effort follows the
    evidence. A pair that was targeted before and did not improve is marked EXHAUSTED and
    skipped — without that check the run keeps spending on a pair that is not responding, which
    is exactly the loop the CLINC150 run got stuck in (B224).

    Note this generates in-class gold rows for the *confused* classes, and it never touches
    held-out eval rows.
    """
    report = state.get("test_report") or {}
    pairs = [p for p in (report.get("confusion_pairs") or []) if isinstance(p, dict)]
    # `__EXTRACTION_FAILED__` is not a class — it is the sentinel for "the model emitted a string
    # that is not in the label vocabulary at all". A pair like `alarm -> __EXTRACTION_FAILED__`
    # therefore names no decision boundary to sharpen, and spending surgical budget on it teaches
    # nothing about the confusion; the remedy for out-of-vocabulary output is format adherence,
    # which ordinary in-class training already provides. It stays in the report for diagnosis —
    # this only stops it from consuming targeted budget (B242).
    _dropped = [p for p in pairs if str(p.get("predicted")) == "__EXTRACTION_FAILED__"]
    if _dropped:
        pairs = [p for p in pairs if str(p.get("predicted")) != "__EXTRACTION_FAILED__"]
        _log(
            model_id,
            f"  [surgical] ignoring {len(_dropped)} extraction-failure pair(s) "
            "(no class boundary to target; format adherence, not confusion)",
        )
    if not pairs:
        return []

    history = dict(state.get("surgical_pair_history") or {})
    eligible, skipped = [], []
    for pair in sorted(pairs, key=lambda p: -int(p.get("count", 0) or 0)):
        key = _pair_key(pair)
        count = int(pair.get("count", 0) or 0)
        previous = history.get(key)
        # Skip if we targeted this pair before and its confusion count did not fall.
        if previous is not None and count >= int(previous.get("count_when_targeted", 0)):
            skipped.append((key, previous.get("count_when_targeted"), count))
            continue
        eligible.append((key, pair, count))
        if len(eligible) >= SURGICAL_MAX_PAIRS:
            break

    for key, before, now in skipped:
        _log(
            model_id,
            f"  [surgical] SKIP {key}: targeted before at count={before}, still {now} — "
            "EXHAUSTED, spending elsewhere",
        )
    if not eligible:
        return []

    total_budget = int(round(int(plan["synth_rows"]) * SURGICAL_SYNTH_SHARE))
    total_count = sum(count for _k, _p, count in eligible) or 1
    by_label: dict[str, list[dict]] = {}
    for row in train_rows:
        if isinstance(row, dict) and row.get("label") is not None:
            by_label.setdefault(str(row["label"]), []).append(row)

    out: list[dict] = []
    for key, pair, count in eligible:
        share = int(round(total_budget * count / total_count))
        share = max(
            SURGICAL_MIN_ROWS_PER_PAIR, min(share, SURGICAL_MAX_ROWS_PER_PAIR)
        )
        # Anchor on the GOLD class — the one the model should have predicted.
        anchors_pool = by_label.get(str(pair.get("gold")), [])
        if not anchors_pool:
            continue
        anchors = _balanced_sample(
            anchors_pool,
            count=min(share, len(anchors_pool)),
            task_type=state["task_type"],
            seed=seed,
        )
        _log(
            model_id,
            f"  ▶ SURGICAL SYNTHESIS {key}: confused {count}x → generating "
            f"{len(anchors)} gold row(s) for {pair.get('gold')!r}",
        )
        rows = synthesize_examples(
            anchors,
            task_type=state["task_type"],
            n=len(anchors),
            generate_fn=generate_fn,
            log=lambda message: _log(model_id, message),
        )
        out.extend(
            {**row, "_provenance": "synthetic", "_strategy_origin": "surgical_synthesize"}
            for row in rows
            if isinstance(row, dict) and str(row.get("_source", "")).startswith("synth")
        )
        history[key] = {
            "count_when_targeted": count,
            "iteration": int(state.get("iteration", 0) or 0),
            "rows_generated": len(anchors),
        }

    state["surgical_pair_history"] = history
    return out


def _synth_fill_to_target(
    dataset: list[dict],
    *,
    target_rows: int,
    task_type: str,
    generate_fn,
    state: AgentState,
    model_id: str,
    fallbacks: list[dict] | None = None,
    tally: dict | None = None,
) -> list[dict]:
    """Top up the dataset with task-adaptive synthesis up to ``target_rows``.

    Covers the initial curriculum (the first curate pass) and every later rebuild: when
    real data + the chosen strategy fall short of the target, synthesize the remainder in
    the same format (task-adaptive — new in-class gold for classification/NER, new-correct
    examples for generation-family). Non-fatal: if synthesis is unavailable (no endpoint
    or cheap mode) the dataset is left as-is and an honest fallback is recorded — never a
    crash.
    """
    deficit = target_rows - len(dataset)
    if deficit <= 0:
        return dataset
    # Announce the top-up explicitly. Synth-fill and the plan's `synthesize` strategy both call
    # the same generator, so without a label the log cannot tell "the orchestrator asked for
    # targeted rows" apart from "the dataset came up short and is being padded to target".
    _log(
        model_id,
        f"  ▶ SYNTH-FILL (top-up to target): have {len(dataset)} row(s), "
        f"need {deficit} more to reach target {target_rows}",
    )
    if os.environ.get("SLM_CHEAP") == "1" or generate_fn is None:
        if fallbacks is not None:
            fallbacks.append({
                "policy": "synth_unavailable_degrade",
                "reason": "synthesis endpoint unavailable or cheap mode",
                "from": "synthesize",
                "to": "base_fill",
                "unfilled_rows": deficit,
            })
        _log(
            model_id,
            f"  Synth-fill unavailable; leaving {len(dataset)} rows "
            f"({deficit} short of {target_rows})",
        )
        return dataset
    extra = synthesize_examples(
        dataset,
        task_type=task_type,
        n=deficit,
        generate_fn=generate_fn,
        verify_fn=_verifier_for(task_type, state),
        log=lambda message: _log(model_id, message),
    )
    # Defensive: admit only rows the generator actually tagged as synthetic, so an anchor row
    # echoed back by a generator could never be appended as a duplicate of what is already here.
    extra = [
        row
        for row in extra
        if isinstance(row, dict)
        and (
            str(row.get("_source", "")).startswith("synth")
            or row.get("_provenance") == "synthetic_positive"
        )
    ]
    # Tag provenance explicitly. A generator that sets only `_source` leaves these rows
    # untagged, so they used to reach the composition report counted as "unknown" — which read
    # like a pipeline defect in the dataset summary when they are simply synth-fill rows.
    # A distinct tag also keeps them separable from the plan's `synthesize` strategy output.
    extra = [
        {**row, "_provenance": row.get("_provenance") or "synthetic_fill"}
        for row in extra
    ]
    extra, _ = _exclude_eval_rows(
        extra, state.get("eval_set"), tally=tally, layer="synth_fill",
        log=lambda m: _log(model_id, m))
    if not extra and fallbacks is not None:
        fallbacks.append({
            "policy": "synth_fill_empty",
            "reason": "synthesis produced no usable rows",
            "from": "synthesize",
            "to": "base_fill",
            "unfilled_rows": deficit,
        })
    _log(model_id, f"  Synth-fill added {len(extra)} row(s) toward {target_rows}")
    return dataset + extra[:deficit]


def _cot_applies(task_type: str) -> bool:
    """Whether a teacher-authored chain of thought earns its one call per row.

    CoT pays for itself when the gold answer is the END of a multi-step derivation the model
    has to reason its way to: `math_reasoning` and `code_generation` qualify by definition.

    The `generation` family does NOT qualify by default, because it also covers single-step
    rewrites. Dialogue summarization is the case that forced this distinction: the summary is
    a compression of text already sitting in the prompt, not the conclusion of an argument, so
    a reasoning chain adds nothing the model cannot read off its own input — while costing one
    teacher call for EVERY row, which was over half the synthesis budget of a DialogSum/SAMSum
    curate pass. Set SLM_COT_GENERATION=1 for a genuinely multi-step `generation` dataset such
    as open-domain QA.
    """
    if task_type in ("math_reasoning", "code_generation"):
        return True
    if task_type == "generation":
        return os.environ.get("SLM_COT_GENERATION") == "1"
    return False


def _annotate_generation_cot(
    rows: list[dict],
    state: AgentState,
    *,
    model_id: str,
) -> list[dict]:
    task_type = state["task_type"]
    if not _cot_applies(task_type):
        if task_type == "generation":
            _log(
                model_id,
                "  Skipping CoT annotation: single-step generation task "
                "(set SLM_COT_GENERATION=1 if this dataset needs reasoning chains)",
            )
        return rows
    if os.environ.get("SLM_CHEAP") == "1":
        _log(model_id, "  CHEAP MODE: skipping CoT annotation")
        return rows
    from config.config import SYNTH_MODEL
    from data.synth_client import get_generate_fn, is_available

    logger = lambda message: _log(model_id, message)
    generate = get_generate_fn(log=logger) if is_available(log=logger) else None
    _log(
        model_id,
        f"  CoT annotation: teacher=LOCAL {SYNTH_MODEL} "
        f"({'available' if generate is not None else 'unavailable — skipping CoT'})",
    )
    return annotate_cot(
        rows,
        task_type=task_type,
        generate_fn=generate,
        log=logger,
    )


def _log_dataset_report(
    model_id: str,
    dataset: list[dict],
    *,
    task_type: str,
    provenance: dict[str, int],
) -> None:
    label_dist = Counter(_label_key(row, task_type) for row in dataset)
    lengths = [len(str(_row_text(row))) for row in dataset]
    mean_length = sum(lengths) / len(lengths) if lengths else 0
    median_length = sorted(lengths)[len(lengths) // 2] if lengths else 0
    _log(model_id, "  ┌─ Dataset report ─────────────────────────")
    _log(model_id, f"  │ Total examples : {len(dataset)}")
    _log(model_id, f"  │ Provenance     : {dict(provenance)}")
    _log(model_id, f"  │ Labels         : {dict(label_dist)}")
    _log(
        model_id,
        f"  │ Text length    : mean={mean_length:.0f} median={median_length}",
    )
    _log(model_id, "  └────────────────────────────────────────────")


def curate_node(state: AgentState) -> AgentState:
    """Build one dataset artifact from the current declarative rebuild plan."""
    task_type = state["task_type"]
    selected = state.get("selected_model")
    if selected is None:
        raise RuntimeError("curate_node called before a model was selected")
    model_id = selected.label
    intervention = state.get("last_intervention", "data_rebuild")
    if intervention != "data_rebuild":
        _log(model_id, f"SKIP: intervention={intervention} — dataset held fixed")
        return state
    eval_set = state.get("eval_set")
    if eval_set is None:
        raise RuntimeError(
            "curate_node requires a fixed eval_set before rebuilding data"
        )

    # Re-size the curriculum whenever the model changes tier. Doing it here — rather than at each
    # of the ten places that assign `selected_model` — means initial selection, escalation and
    # downward regression are all covered by one hook, and the target is always recomputed before
    # the plan that consumes it is built.
    _selector = (
        getattr(state.get("selected_model"), "selector", None)
        or getattr(state.get("selected_model"), "model_id", None)
    )
    if _selector:
        from agent.data_sizing import resize_curriculum_for_tier, baseline_is_known

        # Keyed on the baseline's AVAILABILITY as well as the model. The zero-shot baseline is
        # measured by the first evaluate, which runs AFTER the first curate — so sizing on model
        # change alone permanently used the "no baseline yet" neutral novelty of 0.5 for tier 0
        # and never revisited it. Re-sizing once the measurement exists is the whole point of
        # deriving novelty empirically (B241).
        _sizing_key = f"{_selector}|baseline={baseline_is_known(state)}"
        if state.get("_sized_for_selector") != _sizing_key:
            resize_curriculum_for_tier(state, log=lambda m: _log(model_id, m))
            state["_sized_for_selector"] = _sizing_key

    # Per-layer eval-firewall accounting for this rebuild. Every _exclude_eval_rows call records
    # into this dict (even zero-drop layers), so firewall activity is durably observable — see the
    # always-on summary line below and the persisted eval_firewall block in last_curation.
    firewall_tally: dict[str, int] = {}

    train_rows, excluded_train = _exclude_eval_rows(
        list(state.get("train_examples") or []),
        eval_set,
        tally=firewall_tally,
        layer="train_anchor",
    )
    if excluded_train:
        _log(
            model_id,
            f"  Excluded {excluded_train} normalized train/eval overlap row(s)",
        )
    hypothesis = (
        state.get("last_hypothesis")
        or "initial balanced train-only data rebuild"
    )

    # The current curriculum (previous artifact) and the decontaminated pool decide whether
    # `resample` can still add novel rows this turn. If the entire pool is already in the
    # curriculum, reshuffling is a no-op — so resample is taken off the menu here (the plan
    # normalizer/fallback redirect it to synthesize). Computed BEFORE plan resolution so the
    # gate applies to both the orchestrator's plan and the fallback plan.
    previous_rows = _read_jsonl(state.get("current_dataset_path"))
    previous_texts = _normalized_texts(previous_rows)
    resample_available = not resample_pool_exhausted(
        _normalized_texts(train_rows),
        previous_texts,
    )
    if not resample_available:
        _log(
            model_id,
            "  resample unavailable (whole train pool already in the curriculum) — "
            "any resample plan is redirected to synthesize",
        )

    plan = state.get("data_rebuild_plan")
    if not isinstance(plan, dict):
        plan = fallback_data_rebuild_plan(
            state,
            hypothesis=hypothesis,
            score=(state.get("scores") or [0.0])[-1],
            resample_available=resample_available,
        )
    else:
        plan = normalize_data_rebuild_plan(
            plan,
            task_type=task_type,
            hypothesis=hypothesis,
            target_rows=int(
                state.get("curriculum_size_target", 3000) or 3000
            ),
            default_dataset_version=int(
                state.get("dataset_version", 0) or 0
            ),
            remaining_acquire_rounds=remaining_paid_acquire_rounds(state),
            forbidden_eval_texts=_normalized_eval_texts(eval_set),
            resample_available=resample_available,
        )
    # Non-deterministic redesign (2026-07-31): no plan-identity dedup, no untried-plan
    # rotation, no plan-space exhaustion. The orchestrator freely re-picks a strategy each
    # turn; escalation-on-no-improvement is the sole stuck-run backstop.
    state["data_rebuild_plan"] = plan

    strategy = plan["strategy"]
    seed = _entropy_seed()
    # Spell out what the chosen sub-strategy actually DOES. "data_rebuild" alone is ambiguous in
    # the trajectory — reshuffling the existing pool, mining new real rows, and generating
    # synthetic rows are three very different interventions with very different expected effects.
    _SUBSTRATEGY_EFFECT = {
        "resample": "reshuffle/re-draw from the existing row pool (no new material)",
        "acquire": "mine NEW REAL rows from local bundles / HF / paid discovery",
        "synthesize": "generate NEW SYNTHETIC rows from train anchors",
    }
    _log(
        model_id,
        f"DATA REBUILD: strategy={strategy} ({_SUBSTRATEGY_EFFECT.get(strategy, 'unknown')}) "
        f"target_rows={plan['target_rows']} seed={seed}",
    )

    mining_report = {
        "requested": 0,
        "candidate_rows": 0,
        "novel_rows": 0,
        "novel_fraction": 0.0,
        "paid_rounds_used": 0,
        "run_paid_rounds_spent": int(
            state.get("source_acquire_rounds_used", 0) or 0
        ),
        "paid_budget_exhausted": False,
        "status": "not_requested",
        "source_records": [],
        "rejected_sources": 0,
    }
    mined_rows: list[dict] = []
    if strategy == "acquire":
        eval_rows = getattr(eval_set, "all", [])
        if not isinstance(eval_rows, (list, tuple)):
            eval_rows = []
        task_plan = dict(state.get("task_plan") or {})
        task_plan.setdefault("task_type", task_type)
        task_plan.setdefault("task_name", state.get("description") or task_type)
        mined_rows, mining_report = mine_additional_real_rows(
            task_plan=task_plan,
            description=str(state.get("description") or ""),
            task_type=task_type,
            existing_rows=train_rows + previous_rows,
            eval_rows=list(eval_rows),
            eval_source_ban=list(state.get("eval_source_ban") or []),
            requested_rows=plan["new_real_rows"],
            max_paid_rounds=plan["max_acquire_rounds"],
            query_variant=seed % 8,  # entropy-driven search variance (no plan seed anymore)
            # Content-addressed, NOT a dedup key: the durable ledger meters paid rounds per
            # plan and rejects an empty identity outright (B220).
            plan_identity=plan_budget_identity(plan),
            log=lambda message: _log(model_id, message),
        )
        mined_rows, removed = _exclude_eval_rows(
            mined_rows, eval_set, tally=firewall_tally, layer="mined",
            log=lambda m: _log(model_id, m))
        if removed:
            _log(model_id, f"  Mining eval firewall removed {removed} row(s)")
        state["source_acquire_rounds_used"] = max(
            int(state.get("source_acquire_rounds_used", 0) or 0)
            + int(mining_report.get("paid_rounds_used", 0) or 0),
            int(mining_report.get("run_paid_rounds_spent", 0) or 0),
        )
        lineage = list(state.get("data_sources") or [])
        for record in mining_report.get("source_records") or []:
            if record not in lineage:
                lineage.append(dict(record))
        state["data_sources"] = lineage
        mined_rows = _tag_mined_rows(mined_rows)
        state["train_examples"] = _merge_persistent_train_rows(
            train_rows,
            mined_rows,
            eval_set,
            tally=firewall_tally,
        )
        train_rows = list(state["train_examples"])

    default_source = str(state.get("data_source") or "train_examples")
    tagged_train = _tag_train_rows(
        train_rows,
        default_source=default_source,
    )
    target_rows = int(plan["target_rows"])
    allocation_fallbacks: list[dict] = []
    if strategy == "acquire" and not mined_rows:
        allocation_fallbacks.append({
            "policy": "rewrite_noop_strategy",
            "reason": "source mining produced no novel rows",
            "from": "acquire",
            "to": "base_fill",
        })
    generated_rows: list[dict] = []
    if strategy == "synthesize":
        generated_rows = _synthesize_positive_rows(
            state,
            plan,
            train_rows,
            model_id=model_id,
            seed=seed,
            tally=firewall_tally,
        )
        if not generated_rows:
            allocation_fallbacks.append({
                "policy": "rewrite_noop_strategy",
                "reason": "synthesis produced no rows (endpoint unavailable or cheap mode)",
                "from": "synthesize",
                "to": "base_fill",
            })

    selected_rows: list[dict] = []
    selected_texts: set[str] = set()

    def allocate(rows: list[dict], budget: int) -> None:
        for row in rows:
            if len(selected_rows) >= target_rows or budget <= 0:
                break
            text = normalize_text(_row_text(row))
            if text and text in selected_texts:
                continue
            selected_rows.append(dict(row))
            if text:
                selected_texts.add(text)
            budget -= 1

    # Reserve the chosen material strategy's rows first (acquire/synthesize), then
    # resample-fill the remainder from the existing pool (the universal filler for all
    # three strategies). Each sampler draws its own entropy seed — no reproducibility.
    if strategy == "acquire":
        allocate(mined_rows, min(plan["new_real_rows"], target_rows))
    elif strategy == "synthesize":
        allocate(generated_rows, min(plan["synth_rows"], target_rows))

    working_budget = max(0, target_rows - len(selected_rows))
    working = _balanced_sample(
        list(tagged_train),
        count=min(working_budget, len(tagged_train)),
        task_type=task_type,
        seed=_entropy_seed(),
    )
    working = [
        {**row, "_strategy_origin": "resample"}
        for row in working
    ]
    allocate(working, working_budget)

    dataset = selected_rows

    # Synth-fill to target: when real data + the chosen strategy fall short, top up with
    # task-adaptive synthesis (initial curriculum and every rebuild). Uses the local synth
    # endpoint when reachable; degrades gracefully otherwise (recorded in fallbacks).
    _fill_generate = None
    if os.environ.get("SLM_CHEAP") != "1" and len(dataset) < target_rows:
        from data.synth_client import get_generate_fn, is_available

        _fill_logger = lambda message: _log(model_id, message)
        if is_available(log=_fill_logger):
            _fill_generate = get_generate_fn(log=_fill_logger)
    dataset = _synth_fill_to_target(
        dataset,
        target_rows=target_rows,
        task_type=task_type,
        generate_fn=_fill_generate,
        state=state,
        model_id=model_id,
        fallbacks=allocation_fallbacks,
        tally=firewall_tally,
    )

    dataset = _annotate_generation_cot(
        dataset,
        state,
        model_id=model_id,
    )
    # The task's established label space comes from the frozen eval set — those are exactly the
    # classes the model will be scored against, so any training row outside them is unusable.
    _allowed_labels = None
    if task_type == "classification":
        _allowed_labels = {
            str(row.get("label"))
            for row in (getattr(eval_set, "all", None) or [])
            if isinstance(row, dict) and row.get("label") is not None
        } or None
    _pre_qc = len(dataset)
    dataset = apply_quality_controls(
        dataset,
        task_type=task_type,
        allowed_labels=_allowed_labels,
        log=lambda message: _log(model_id, message),
    )
    if len(dataset) < _pre_qc:
        _log(
            model_id,
            f"  Quality control removed {_pre_qc - len(dataset)} row(s) total "
            f"({_pre_qc} → {len(dataset)})",
        )
    # No upper truncation: target_rows is a floor for synth-fill, not a cap. Extra rows
    # above target (e.g. acquire/synthesize overshoot) are kept — we only guard against
    # too FEW rows, never too many.
    dataset, removed_final = _exclude_eval_rows(
        dataset, eval_set, tally=firewall_tally, layer="final",
        log=lambda m: _log(model_id, m))
    if removed_final:
        _log(
            model_id,
            f"  Final eval firewall removed {removed_final} row(s)",
        )

    # Always-on firewall summary (even when nothing was dropped) so every run's log confirms the
    # eval firewall ran and by how much. The per-layer breakdown is persisted in last_curation.
    firewall_total = sum(firewall_tally.values())
    _log(
        model_id,
        f"  Eval firewall: {firewall_total} row(s) removed across "
        f"{len(firewall_tally)} checkpoint(s) "
        f"[{', '.join(f'{k}={v}' for k, v in firewall_tally.items())}]",
    )

    # Under-target is ACCEPTABLE — synth-fill runs before quality control and there is no second
    # fill afterwards, so QC legitimately lands the dataset below target. It must not be silent
    # though: a run training on 3,461 rows against a 5,000-row target should say so plainly
    # rather than leaving the gap to be discovered by reading artifacts later (B228).
    if len(dataset) < target_rows:
        _log(
            model_id,
            f"  ⚠ PROCEEDING BELOW DATA TARGET: {len(dataset)} row(s) vs target {target_rows} "
            f"(short by {target_rows - len(dataset)}). Synth-fill runs BEFORE quality control "
            "and there is no refill afterwards, so QC removals land the final dataset under "
            "target. See the [qc] lines above for exactly what was removed and why.",
        )

    next_version = int(state.get("dataset_version", 0) or 0) + 1
    for row in dataset:
        row["_dataset_version"] = next_version
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, f"dataset_v{next_version}.jsonl")
    atomic_write_jsonl(path, dataset)
    state["dataset_version"] = next_version
    state["current_dataset_path"] = path

    provenance = Counter(
        str(row.get("_provenance") or "unknown")
        for row in dataset
    )
    source_composition = Counter(
        _source_key(row, default_source)
        for row in dataset
    )
    difficulty_composition = Counter(
        str(row.get("_difficulty") or "unassigned")
        for row in dataset
    )
    final_texts = _normalized_texts(dataset)
    novel_rows = len(final_texts - previous_texts)
    yield_status = "novel" if novel_rows else "no_novelty"
    if (
        strategy == "acquire"
        and mining_report.get("status") == "no_novelty"
    ):
        _log(
            model_id,
            "  Source mining produced no novelty; resample-fill covered the remainder",
        )
    plan_yield = {
        "status": yield_status,
        "previous_rows": len(previous_rows),
        "final_rows": len(dataset),
        "novel_rows": novel_rows,
        "novel_fraction": (
            round(novel_rows / len(final_texts), 4)
            if final_texts
            else 0.0
        ),
    }

    origin_composition = Counter(
        str(row.get("_strategy_origin") or "unattributed")
        for row in dataset
    )
    origin_novelty = Counter(
        str(row.get("_strategy_origin") or "unattributed")
        for row in dataset
        if normalize_text(_row_text(row)) not in previous_texts
    )
    strategy_composition = [
        {
            "strategy": origin,
            "rows": origin_composition.get(origin, 0),
            "novel_rows": origin_novelty.get(origin, 0),
        }
        for origin in sorted(origin_composition)
        if origin_composition.get(origin, 0)
    ]
    # Per-source usage for provenance logging: join per-row source tags (counts) with the
    # url-bearing records (this build's mining records + the run-wide lineage). Web-scraped
    # rows carry their url on the row itself, so they are covered too.
    novel_by_source = Counter(
        source_key(row, default_source)
        for row in dataset
        if normalize_text(_row_text(row)) not in previous_texts
    )
    source_usage = build_source_usage(
        dataset,
        source_records=(
            list(mining_report.get("source_records") or [])
            + list(state.get("data_sources") or [])
        ),
        novel_by_source=dict(novel_by_source),
        default=default_source,
    )

    label_dist = Counter(_label_key(row, task_type) for row in dataset)
    state["last_curation"] = {
        "total_examples": len(dataset),
        "n_gold": (
            provenance.get("train_anchor", 0)
            + provenance.get("resample", 0)
        ),
        "n_hard": provenance.get("synthetic", 0),
        "n_hard_generated": provenance.get("synthetic", 0),
        "n_hard_source": provenance.get("mined_real", 0),
        "label_dist": dict(label_dist),
        "strategy": strategy,
        "data_rebuild_plan": plan,
        "rebuild_config": {
            "target_rows": plan["target_rows"],
            "resample_fraction": plan["resample_fraction"],
            "seed": seed,
        },
        "strategy_composition": strategy_composition,
        "provenance_composition": dict(provenance),
        "source_composition": dict(source_composition),
        "difficulty_composition": dict(difficulty_composition),
        "source_novelty": dict(mining_report),
        "source_usage": source_usage,
        "plan_yield": plan_yield,
        "allocation_fallbacks": allocation_fallbacks,
        # Durable eval-firewall audit: per-checkpoint drop counts + total for this rebuild, so
        # contamination filtering is observable from artifacts (not only run.log grepping).
        "eval_firewall": {"total": firewall_total, "by_layer": dict(firewall_tally)},
    }
    # Run-wide provenance accumulator (survives across iterations/checkpoints). Assigned as a
    # NEW list, never mutated in place, so it persists to the LangGraph channel (cf. B122).
    state["data_source_usage"] = list(state.get("data_source_usage") or []) + [{
        "iteration": int(state.get("iteration", 0) or 0),
        "dataset_version": f"v{next_version}",
        "strategy": strategy,
        "sources": source_usage,
    }]
    _log_dataset_report(
        model_id,
        dataset,
        task_type=task_type,
        provenance=dict(provenance),
    )
    _log(
        model_id,
        f"  Plan yield: {plan_yield}; composition={strategy_composition}",
    )
    _log(model_id, f"  Saved: {path}")
    return state
