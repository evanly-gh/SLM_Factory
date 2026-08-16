# agent/nodes/cold_start/eval_setup.py
import os
import json
import time
from agent.checkpoint import atomic_write_json
from agent.timing import TimingEvent, record_timing_event
from agent.state import AgentState
from data.eval_set import build_eval_set
from eval.endpoint_eval import measure_endpoint_baseline
from data.loaders.dataset_integrity import (
    normalize_text,
    normalized_text_overlap,
    remove_normalized_train_overlap,
    required_fields_for_task,
    validate_rows,
    verify_checksum_sidecar,
    verify_manifest_hashes,
)

ARTIFACTS_DIR = "artifacts"
SHARED_CONTENT_FILES = (
    "train.jsonl", "test.jsonl", "difficulty.json", "sources.json",
    "eval_ban.json", "plan.json",
)
SHARED_CHECKSUM_FILES = SHARED_CONTENT_FILES + ("manifest.json",)


# Single source of truth for the curated benchmarks' task types + human labels, keyed by the
# SLM_BENCHMARK_TASK value. Kept free of loader imports so callers (e.g. the driver deciding the
# initial task_type) can read it without pulling `datasets` or any optional loader dependency.
#
# The suite is organised by what fine-tuning is expected to DO for each task:
#   in-distribution — the base model already does the task; FT buys format discipline and modest
#     accuracy. Good baseline, small delta.
#   format-bound    — the base model knows the content but cannot produce the exact output
#     contract. Near-zero baseline, large delta (BC5CDR: 0.0000 → 0.8098).
#   out-of-distribution — the label is not a property of the input's surface form, so the base
#     model has nothing to pattern-match. Low/noisy baseline, delta bounded by label noise.
#
# `coedit` and `medqa` were removed 2026-08-15 by decision — neither had ever produced a run.
NAMED_BENCHMARK_TASK_TYPES: dict[str, tuple[str, str]] = {
    # in-distribution
    "dialogsum_samsum": ("generation", "DialogSum + SAMSum"),
    # format-bound
    "xlam_bfcl": ("function_call", "xLAM-60k / BFCL"),
    "calendar_json": ("function_call", "Calendar NL→JSON (TOPv2 reminder / SGD Calendar_1)"),
    "ner_bc5cdr": ("NER", "BC5CDR (Chemical/Disease spans)"),
    # out-of-distribution
    "clinc150": ("classification", "CLINC150 (clinc_oos/plus)"),
    "routerbench": ("classification", "RouterBench"),
    "proactive_listening": ("classification", "Proactive listening (LlamaPIE interrupt/wait)"),
}


def _named_benchmark_loaders() -> dict:
    """Registry of the curated benchmark loaders, selectable via SLM_BENCHMARK_TASK on the
    non-autonomous path. Each entry is (loader_callable, task_type, source_label). Imports are
    lazy so a missing optional dependency only breaks the benchmark that needs it. Task types /
    labels come from NAMED_BENCHMARK_TASK_TYPES so there is one source of truth."""
    from data.loaders.clinc150 import load_clinc150
    from data.loaders.dialogsum_samsum import load_dialogsum_samsum
    from data.loaders.xlam_bfcl import load_xlam_bfcl
    from data.loaders.routerbench import load_routerbench
    from data.loaders.ner_bc5cdr import load_ner_bc5cdr
    from data.loaders.calendar_json import load_calendar_json
    from data.loaders.proactive_listening import load_proactive_listening
    loaders = {
        "clinc150": load_clinc150,
        "dialogsum_samsum": load_dialogsum_samsum,
        "xlam_bfcl": load_xlam_bfcl,
        "routerbench": load_routerbench,
        "ner_bc5cdr": load_ner_bc5cdr,
        "calendar_json": load_calendar_json,
        "proactive_listening": load_proactive_listening,
    }
    return {
        key: (loaders[key], task_type, label)
        for key, (task_type, label) in NAMED_BENCHMARK_TASK_TYPES.items()
    }


def _load_named_benchmark(name: str, state: AgentState, acquire_meta: dict):
    """Load one of the six curated benchmarks by SLM_BENCHMARK_TASK key, sized to the run's
    curriculum/eval targets. Raises ValueError on an unknown key or a task_type mismatch."""
    registry = _named_benchmark_loaders()
    key = str(name).strip().lower()
    if key not in registry:
        raise ValueError(
            f"SLM_BENCHMARK_TASK={name!r} is not a known benchmark; choose one of "
            f"{sorted(registry)}"
        )
    loader, expected_task, source_label = registry[key]
    task_type = state["task_type"]
    if task_type != expected_task:
        raise ValueError(
            f"SLM_BENCHMARK_TASK={key!r} produces task_type={expected_task!r} but the run's "
            f"task_type is {task_type!r}; set them consistently"
        )
    max_train = int(int(state.get("curriculum_size_target") or 1000) * 0.65)
    max_test = int(state.get("eval_size_target") or 800)
    print(f"      [eval_setup] loading named benchmark {key!r} ({source_label}): "
          f"train≤{max_train} test≤{max_test}")
    train_examples, test_examples = loader(max_train=max(max_train, 60), max_test=max(max_test, 60))
    # Stage-0 decontamination, matching the autonomous acquire_dataset path. Official benchmark
    # splits are not guaranteed disjoint (CLINC150 ships "what's your designation" in both splits
    # under two different intents), and this path never passes through web_acquire, so without
    # this the eval_setup overlap firewall would turn a source-data quirk into a fatal raise.
    # The held-out test rows are authoritative and never modified; the train row is dropped.
    train_examples, overlap_removed = remove_normalized_train_overlap(
        train_examples, test_examples)
    if overlap_removed:
        print(f"      [eval_setup] Stage-0 normalized overlap removal for {key!r}: "
              f"removed {overlap_removed} train row(s); official test rows unchanged")
    acquire_meta["overlap_removed_from_train"] = overlap_removed
    acquire_meta["source"] = source_label
    acquire_meta["source_records"] = [
        {"kind": "hf", "id": key, "split": "train", "role": "curriculum"},
        {"kind": "hf", "id": key, "split": "test", "role": "eval"},
    ]
    acquire_meta["eval_ban"] = [{"kind": "hf", "id": key, "split": "test", "role": "eval"}]
    return train_examples, test_examples


class QwenBaselineUnavailableError(RuntimeError):
    """The Qwen-3.6 reference endpoint could not be measured, so no accuracy goal exists.

    This is FATAL by design: the Qwen baseline is the sole accuracy target, so an unreachable
    endpoint has no honest fallback. Raising breaks the pipeline loop rather than degrading to a
    guessed threshold.
    """


def _pin_label_space(state: AgentState, eval_set) -> None:
    """Close the task's label vocabulary against the frozen eval set, once, before any curation.

    Everything downstream (mining, LLM column mapping, synthesis, quality control) treats this as
    authoritative and may only REMOVE rows that fall outside it — never extend it. See
    `data/label_space.py` for why (B259: four hallucinated classes entered a two-class task).
    """
    from data.label_space import label_definitions_for, label_space_from_eval_set

    labels = label_space_from_eval_set(eval_set, state["task_type"])
    if not labels:
        state["task_label_space"] = None
        return
    benchmark = os.environ.get("SLM_BENCHMARK_TASK") or ""
    definitions = label_definitions_for(benchmark)
    state["task_label_space"] = {
        "labels": sorted(labels),
        "definitions": definitions,
        "benchmark": benchmark or None,
        "source": "frozen_eval_set",
    }
    shown = sorted(labels)
    print(
        f"      [label-space] PINNED {len(shown)} class(es) from the frozen eval set: "
        + (", ".join(repr(label) for label in shown[:12])
           + (f", … (+{len(shown) - 12} more)" if len(shown) > 12 else ""))
    )
    print(
        "      [label-space] this vocabulary is CLOSED — mined sources whose labels are not a "
        "subset are rejected, and no LLM may introduce a new class"
    )
    if definitions:
        print(
            f"      [label-space] {len(definitions)} label definition(s) available for synthesis "
            "and verification prompts (the teacher is told what each class MEANS, not just its name)"
        )


def _calibrate_qwen_goal_if_pending(state: AgentState, eval_set) -> None:
    """Complete the Qwen-3.6-baseline accuracy goal once the frozen E exists.

    task_analysis parks the goal as ``pending_qwen_baseline`` because E is not built until this
    node. Here we score the hosted reference model zero-shot on E and set the goal to
    ``min(0.99, max(measured, floor))`` via agent.threshold.threshold_from_endpoint_baseline.
    ``initial_stop_threshold`` (the immutable floor) is written ONCE, here.

    The Qwen baseline is the ONLY accuracy target. If the endpoint is unreachable or the
    measurement errors, this RAISES ``QwenBaselineUnavailableError`` and the run stops — there
    is deliberately no fallback to a guessed threshold. A successfully-measured-but-weak score
    is still floored at 0.8.
    """
    calibration = state.get("threshold_calibration") or {}
    if calibration.get("source") != "pending_qwen_baseline":
        return

    from agent.threshold import threshold_from_endpoint_baseline

    floor = float(calibration.get("floor", 0.8))
    task_type = state["task_type"]
    try:
        baseline = measure_endpoint_baseline(eval_set, task_type, log=print)
    except Exception as error:  # noqa: BLE001 - re-raised as a fatal calibration failure
        raise QwenBaselineUnavailableError(
            f"Qwen-3.6 baseline measurement failed ({str(error)[:160]}); the reference "
            "endpoint is the sole accuracy target, so the run cannot continue"
        ) from error

    if baseline is None:
        raise QwenBaselineUnavailableError(
            "Qwen-3.6 baseline endpoint is unreachable (measure_endpoint_baseline returned "
            "None); the reference endpoint is the sole accuracy target, so the run cannot "
            "continue. Set SYNTH_ENDPOINT/SYNTH_MODEL to a reachable Qwen-3.6 server."
        )

    threshold, reason = threshold_from_endpoint_baseline(baseline.f1, floor=floor)
    # Whether the FLOOR or the teacher's own measurement set the goal. A floored goal and a
    # teacher-set goal print the same number, so without this flag a run that converged against a
    # floor the teacher never reached is indistinguishable in the summary from one that matched a
    # strong teacher. Recorded here, reported at end of run.
    floored = float(baseline.f1) < float(floor)
    print(f"      [threshold] Qwen baseline {baseline.f1:.4f} → goal {threshold:.4f} "
          f"(floor {floor:.2f}) — "
          + (
              f"FLOOR WON: the teacher scored below {floor:.2f}, so the goal is the floor, "
              f"not the teacher's {baseline.f1:.4f}"
              if floored
              else f"teacher's own score set the goal (above the {floor:.2f} floor)"
          ))

    state["stop_threshold"] = threshold
    state["initial_stop_threshold"] = threshold
    state["threshold_calibration"] = {
        "source": "qwen_baseline",
        "threshold": threshold,
        "floor": floor,
        "floored": floored,
        "reason": reason,
        "pending": False,
        "measured_qwen": round(float(baseline.f1), 4),
        "measured_metric": baseline.metric,
        "endpoint": "reachable",
    }


def _eval_target(target: int) -> int:
    """Clamp the ORCHESTRATOR's eval-size target to a sane floor (B161).
    build_eval_set defaulted to 100 total, which silently capped the eval set no matter how
    many test rows were acquired (the eval-size bug). Honor the requested size with a min-30
    floor so a tiny target can never starve the eval set."""
    return max(int(target or 0), 30)


def _load_shared_dataset(shared_dir: str):
    """Load a schema-checked, checksum-verified frozen train/eval bundle."""
    if not shared_dir:
        return None
    if not os.path.isdir(shared_dir):
        raise ValueError(f"shared dataset directory does not exist: {shared_dir}")

    manifest_path = os.path.join(shared_dir, "manifest.json")
    checksum_path = os.path.join(shared_dir, "checksums.sha256")
    if not os.path.exists(manifest_path) or not os.path.exists(checksum_path):
        raise ValueError(
            "shared dataset requires manifest.json and checksums.sha256; "
            "refusing unverified data"
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("bundle_type") != "shared_dataset" or manifest.get("schema_version") != 1:
        raise ValueError(
            "shared dataset manifest must declare bundle_type='shared_dataset' "
            "and schema_version=1"
        )
    if not verify_checksum_sidecar(shared_dir, SHARED_CHECKSUM_FILES):
        raise ValueError("shared dataset checksums.sha256 is required")
    verify_manifest_hashes(shared_dir, (manifest.get("integrity") or {}).get("files"))

    def _json(name):
        with open(os.path.join(shared_dir, name), encoding="utf-8") as handle:
            return json.load(handle)

    def _jsonl(name):
        with open(os.path.join(shared_dir, name), encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    train = _jsonl("train.jsonl")
    test = _jsonl("test.jsonl")
    task_type = manifest.get("task_type")
    required = tuple((manifest.get("row_schema") or {}).get("required") or ())
    canonical_required = required_fields_for_task(task_type)
    if set(required) != set(canonical_required):
        raise ValueError(
            f"shared dataset row_schema mismatch for {task_type!r}: {required}"
        )
    validate_rows(train, required, bundle_name="shared dataset", split="train")
    validate_rows(test, required, bundle_name="shared dataset", split="test")
    counts = manifest.get("counts") or {}
    if counts != {
        "train": len(train), "test": len(test), "total": len(train) + len(test)
    }:
        raise ValueError("shared dataset manifest counts do not match JSONL files")
    overlap = normalized_text_overlap(train, test)
    if overlap:
        raise ValueError(
            f"shared dataset normalized train/eval overlap ({len(overlap)} rows)"
        )

    diff = _json("difficulty.json")
    sources = _json("sources.json")
    eval_ban = _json("eval_ban.json")
    if sources != manifest.get("source_records") or eval_ban != manifest.get("eval_ban"):
        raise ValueError("shared dataset provenance files do not match manifest")
    return train, test, diff, sources, eval_ban, manifest


def eval_setup_node(state: AgentState) -> AgentState:
    """
    Node 2: download data and build the held-out eval set E.
    Eval set is built BEFORE any training. Fixed throughout all iterations.
    task_type flows from state — no hardcoding.
    """
    task_type = state["task_type"]
    plan = state.get("task_plan")

    # Shared-dataset harness (B161): for a FAIR strategy comparison, all runs can load ONE
    # frozen dataset + eval set (built once by scripts/prepare_shared_dataset.py) instead of
    # each independently re-acquiring — acquisition nondeterminism previously gave each
    # strategy different data, confounding the comparison. Enabled via SLM_SHARED_DATASET_DIR.
    _shared = _load_shared_dataset(os.environ.get("SLM_SHARED_DATASET_DIR", ""))
    if _shared is not None:
        (train_examples, test_examples, _shared_diff, _shared_sources,
         _shared_eval_ban, _shared_manifest) = _shared
        if _shared_manifest["task_type"] != task_type:
            raise ValueError(
                f"shared dataset task_type={_shared_manifest['task_type']!r} "
                f"does not match run task_type={task_type!r}"
            )
        print(f"      [eval_setup] using SHARED frozen dataset: train={len(train_examples)} "
              f"test={len(test_examples)} (SLM_SHARED_DATASET_DIR) — identical across strategy runs")
        state["train_examples"] = train_examples
        state["data_source"] = "shared frozen dataset"
        state["data_sources"] = list(_shared_sources or [])
        state["eval_source_ban"] = list(_shared_eval_ban or [])
        eval_set = build_eval_set(
            test_examples, task_type=task_type,
            target=_eval_target(state.get("eval_size_target", 800)),
            multi_label=plan.get("multi_label", False) if plan else False,
            schema=(plan.get("schema") if plan else None),
            multilingual=plan.get("multilingual", False) if plan else False,
        )
        state["eval_set"] = eval_set
        state["eval_difficulty"] = _shared_diff
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        atomic_write_json(
            os.path.join(ARTIFACTS_DIR, "eval_set.json"),
            {
                "task_type": eval_set.task_type,
                "counts": {"total": len(eval_set.all)},
                "difficulty_counts": ({k: len(v) for k, v in _shared_diff.items()} if _shared_diff else {}),
                "examples": eval_set.all,
                "difficulty": _shared_diff or {},
            },
        )
        _pin_label_space(state, eval_set)
        _calibrate_qwen_goal_if_pending(state, eval_set)
        return state

    acquire_meta: dict = {}
    if plan is not None:
        # Autonomous, general path: acquire the dataset per the orchestrator's plan, sized to
        # the ORCHESTRATOR-CHOSEN data targets (B161): curriculum_size_target (gold ≈ 65% of
        # it) and eval_size_target. These are clamped to config floors/ceiling in task_analysis.
        from config.config import DATA_SIZE_CEILING
        _curriculum = int(state.get("curriculum_size_target") or 1000)
        _gold_target = int(_curriculum * 0.65)
        # Headroom (×1.15 + 40) covers eval-overlap removal + quality-control drops.
        _bench_train = min(int(_gold_target * 1.15) + 40, DATA_SIZE_CEILING)
        _bench_test = int(state.get("eval_size_target") or 800)
        print(f"      [eval_setup] acquiring: curriculum_target={_curriculum} "
              f"(gold≈{_gold_target}, request train≤{_bench_train})  eval_target={_bench_test}")
        from data.loaders.web_acquire import acquire_dataset
        train_examples, test_examples = acquire_dataset(
            plan, description=state.get("description", ""),
            target_examples=max(_gold_target, 120),
            benchmark_max_train=_bench_train, benchmark_max_test=_bench_test,
            meta=acquire_meta,
        )
    elif os.environ.get("SLM_BENCHMARK_TASK"):
        # Curated non-autonomous path: load one of the six benchmark loaders by env key. This
        # feeds the same build_eval_set + overlap-firewall path the classification branch uses.
        train_examples, test_examples = _load_named_benchmark(
            os.environ["SLM_BENCHMARK_TASK"], state, acquire_meta)
    elif task_type == "classification":
        from data.loaders.sms_spam import download_sms_spam
        train_examples, test_examples = download_sms_spam()
        acquire_meta["source"] = "bundled SMS Spam dataset (UCI)"
    else:
        raise NotImplementedError(
            f"eval_setup_node does not yet have a data loader for task_type={task_type!r}. "
            "Add a loader branch here when NER or generation data sources are available."
        )

    state["train_examples"] = train_examples
    state["data_source"] = acquire_meta.get("source", "unknown")

    # --- Data provenance + held-out split restriction metadata (B161) ---
    # Keep full lineage in data_sources, but only copy explicitly declared eval restrictions
    # into eval_source_ban. The field is metadata for downstream acquisition decisions; this
    # node's concrete enforcement is the normalized train/test overlap assertion below.
    _sources = acquire_meta.get("source_records") or [{
        "kind": "unknown", "id": acquire_meta.get("source", "unknown"), "role": "source",
    }]
    _eval_bans = list(acquire_meta.get("eval_ban") or [])
    state["data_sources"] = list(state.get("data_sources") or []) + _sources
    state["eval_source_ban"] = list(state.get("eval_source_ban") or []) + _eval_bans
    print(f"      [eval_setup] data provenance: {acquire_meta.get('source', 'unknown')}")
    print(f"      [eval_setup] eval source restrictions recorded: "
          f"{[(s.get('id'), s.get('split')) for s in _eval_bans]}")
    # Enforce official split separation after all acquisition paths, including mocked or
    # future loaders that bypass bundle/Stage-0 checks.
    _eval_texts = {
        normalize_text(e.get("text")) for e in test_examples
        if isinstance(e.get("text"), str)
    }
    _leaks = [
        e for e in train_examples
        if isinstance(e.get("text"), str) and normalize_text(e.get("text")) in _eval_texts
    ]
    if _leaks:
        raise ValueError(
            f"eval_setup normalized train/test overlap: {len(_leaks)} training row(s) "
            "match held-out eval text"
        )
    print("      [eval_setup] official train/test separation: normalized overlap=0")

    # Forward planner flags so the eval set carries multi_label/schema/multilingual
    # context for downstream scorer dispatch.
    plan = state.get("task_plan") or {}
    eval_set = build_eval_set(
        test_examples,
        task_type=task_type,
        target=_eval_target(state.get("eval_size_target", 800)),
        multi_label=plan.get("multi_label", False),
        schema=plan.get("schema", None),
        multilingual=plan.get("multilingual", False),
    )
    state["eval_set"] = eval_set
    _target = int(state.get("eval_size_target", 800) or 800)
    print(f"      [eval_setup] eval set built: {len(eval_set.all)} examples (target {_target})")
    # A short eval set is acceptable — some loaders simply have less held-out data than we asked
    # for — but it must be stated, because it changes how the score should be read: fewer rows means
    # more variance, and scores are then not directly comparable across tasks. `calendar_json` ran on
    # 478 rows against a target of 800 and nothing said so.
    if len(eval_set.all) < _target:
        _short = _target - len(eval_set.all)
        print(f"      [eval_setup] ⚠ EVAL SET IS SHORT: {len(eval_set.all)}/{_target} rows "
              f"(short by {_short}, {len(eval_set.all) / _target:.0%} of target). Accepted — the "
              f"loader's held-out split is the limit — but scores carry more variance than a "
              f"full-size eval set and are not directly comparable to tasks that reached target.")
    _pin_label_space(state, eval_set)

    # Difficulty-stratify the eval set for the test-data agent (B161): label each held-out
    # example easy/medium/hard by the base-model zero-shot capability gradient (smallest vs
    # largest feasible model), so the test agent can report per-difficulty accuracy and give
    # targeted diagnoses. Guarded — falls back to a length heuristic on any failure.
    _difficulty_t0 = time.perf_counter()
    _difficulty_status = "success"
    try:
        from agent.nodes.test_agent import label_difficulty
        difficulty = label_difficulty(
            eval_set, state.get("feasible_models") or [], task_type, log=print)
    except Exception as _e:  # noqa: BLE001
        _difficulty_status = "error"
        print(f"      [eval_setup] difficulty labeling failed ({str(_e)[:100]}); skipping")
        difficulty = None
    record_timing_event(TimingEvent(
        kind="phase",
        name="difficulty_labeling",
        duration_ms=(time.perf_counter() - _difficulty_t0) * 1000,
        status=_difficulty_status,
        metadata={"eval_examples": len(eval_set.all)},
    ))
    state["eval_difficulty"] = difficulty

    # Persist the held-out eval set as a durable artifact (it is otherwise only in state).
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    atomic_write_json(
        os.path.join(ARTIFACTS_DIR, "eval_set.json"),
        {
            "task_type": eval_set.task_type,
            "counts": {"total": len(eval_set.all)},
            "difficulty_counts": ({k: len(v) for k, v in difficulty.items()} if difficulty else {}),
            "examples": eval_set.all,
            "difficulty": difficulty or {},
        },
    )
    _calibrate_qwen_goal_if_pending(state, eval_set)
    return state
