"""Execute bounded declarative dataset rebuild plans."""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from collections.abc import Callable

from agent.checkpoint import atomic_write_jsonl
from agent.data_rebuild import (
    fallback_data_rebuild_plan,
    normalize_data_rebuild_plan,
    remaining_paid_acquire_rounds,
)
from agent.state import AgentState
from data.curriculum import (
    annotate_cot,
    apply_quality_controls,
    get_cot_fallbacks,
    synthesize_examples,
)
from data.loaders.dataset_integrity import normalize_text
from data.loaders.web_acquire import mine_additional_real_rows


ARTIFACTS_DIR = "artifacts"
_DIFFICULTY_BUCKETS = ("easy", "medium", "hard")


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
    are kept after standard quality controls (verify_fn=None) and classification/NER use
    the contrastive path which needs no verifier.
    """
    return None


def _cot_benchmark(plan: dict) -> str:
    return plan.get("benchmark") or plan.get("task_name", "")


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
) -> tuple[list[dict], int]:
    """Apply the normalized-text eval firewall to candidate training rows."""
    eval_texts = _normalized_eval_texts(eval_set)
    if not eval_texts:
        return list(rows), 0
    clean = [
        row
        for row in rows
        if not isinstance(row, dict)
        or normalize_text(_row_text(row)) not in eval_texts
    ]
    return clean, len(rows) - len(clean)


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


def _with_train_difficulty(rows: list[dict]) -> list[dict]:
    """Attach train-only length-tercile difficulty where metadata is absent."""
    copied = [dict(row) for row in rows]
    missing = [
        index
        for index, row in enumerate(copied)
        if row.get("_difficulty") not in _DIFFICULTY_BUCKETS
    ]
    ordered = sorted(
        missing,
        key=lambda index: (len(str(_row_text(copied[index]))), index),
    )
    size = len(ordered)
    for rank, index in enumerate(ordered):
        if rank < size // 3:
            bucket = "easy"
        elif rank < (2 * size) // 3:
            bucket = "medium"
        else:
            bucket = "hard"
        copied[index]["_difficulty"] = bucket
    return copied


def _difficulty_sample(
    rows: list[dict],
    *,
    count: int,
    weights: dict[str, float],
    seed: int,
) -> tuple[list[dict], int]:
    candidates = _with_train_difficulty(rows)
    groups = {
        bucket: [
            row
            for row in candidates
            if row.get("_difficulty") == bucket
        ]
        for bucket in _DIFFICULTY_BUCKETS
    }
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)

    raw_quotas = {
        bucket: count * float(weights.get(bucket, 0.0))
        for bucket in _DIFFICULTY_BUCKETS
    }
    quotas = {
        bucket: min(len(groups[bucket]), int(raw_quotas[bucket]))
        for bucket in _DIFFICULTY_BUCKETS
    }
    remaining = count - sum(quotas.values())
    order = sorted(
        [
            bucket for bucket in _DIFFICULTY_BUCKETS
            if float(weights.get(bucket, 0.0)) > 0
        ],
        key=lambda bucket: (
            -(raw_quotas[bucket] - int(raw_quotas[bucket])),
            _DIFFICULTY_BUCKETS.index(bucket),
        ),
    )
    while remaining > 0:
        progressed = False
        for bucket in order:
            if quotas[bucket] < len(groups[bucket]):
                quotas[bucket] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    selected = [
        row
        for bucket in _DIFFICULTY_BUCKETS
        for row in groups[bucket][:quotas[bucket]]
    ]
    rng.shuffle(selected)
    selected = selected[:count]
    return selected, max(0, count - len(selected))


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
) -> list[dict]:
    """Retain novel real rows for every later rebuild and checkpoint."""
    candidates, _ = _exclude_eval_rows(
        [*existing_rows, *mined_rows],
        eval_set,
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
    anchors = _balanced_sample(
        train_rows,
        count=min(plan["synth_rows"], len(train_rows)),
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
    generated, removed = _exclude_eval_rows(generated, state.get("eval_set"))
    if removed:
        _log(model_id, f"  Synthesis eval firewall removed {removed} row(s)")
    return generated[:plan["synth_rows"]]


def _annotate_generation_cot(
    rows: list[dict],
    state: AgentState,
    *,
    model_id: str,
) -> list[dict]:
    task_type = state["task_type"]
    if task_type not in ("math_reasoning", "code_generation", "generation"):
        return rows
    if os.environ.get("SLM_CHEAP") == "1":
        _log(model_id, "  CHEAP MODE: skipping CoT annotation")
        return rows
    task_plan = state.get("task_plan") or {}
    fallbacks = get_cot_fallbacks(task_type, _cot_benchmark(task_plan))
    from config.config import SYNTH_MODEL
    from data.synth_client import get_generate_fn, is_available

    logger = lambda message: _log(model_id, message)
    generate = get_generate_fn(log=logger) if is_available(log=logger) else None
    _log(
        model_id,
        f"  CoT annotation: primary=LOCAL {SYNTH_MODEL}; "
        f"fallbacks={[model for _, model in fallbacks] or ['none']}",
    )
    return annotate_cot(
        rows,
        task_type=task_type,
        generate_fn=generate,
        fallback_teachers=fallbacks,
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

    train_rows, excluded_train = _exclude_eval_rows(
        list(state.get("train_examples") or []),
        eval_set,
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
    plan = state.get("data_rebuild_plan")
    if not isinstance(plan, dict):
        plan = fallback_data_rebuild_plan(
            state,
            hypothesis=hypothesis,
            score=(state.get("scores") or [0.0])[-1],
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
        )
    # Non-deterministic redesign (2026-07-31): no plan-identity dedup, no untried-plan
    # rotation, no plan-space exhaustion. The orchestrator freely re-picks a strategy each
    # turn; escalation-on-no-improvement is the sole stuck-run backstop.
    state["data_rebuild_plan"] = plan

    strategy = plan["strategy"]
    seed = _entropy_seed()
    _log(
        model_id,
        f"DATA REBUILD: strategy={strategy} target_rows={plan['target_rows']} seed={seed}",
    )

    previous_rows = _read_jsonl(state.get("current_dataset_path"))
    previous_texts = _normalized_texts(previous_rows)

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
            plan_identity="",
            log=lambda message: _log(model_id, message),
        )
        mined_rows, removed = _exclude_eval_rows(mined_rows, eval_set)
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
    dataset = _annotate_generation_cot(
        dataset,
        state,
        model_id=model_id,
    )
    dataset = apply_quality_controls(
        dataset,
        task_type=task_type,
    )
    dataset = dataset[:target_rows]
    dataset, removed_final = _exclude_eval_rows(dataset, eval_set)
    if removed_final:
        _log(
            model_id,
            f"  Final eval firewall removed {removed_final} row(s)",
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
        "plan_yield": plan_yield,
        "allocation_fallbacks": allocation_fallbacks,
    }
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
