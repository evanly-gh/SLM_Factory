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
# initial task) can read it without pulling `datasets` or any optional loader dependency.
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
# The benchmark registry moved to `tasks/` on 2026-08-18. It used to live here as
# NAMED_BENCHMARK_TASK_TYPES (name -> task_type, label) plus a parallel loader dict that had to be
# kept in lockstep by a test — two of the five hand-synchronised side registries the task specs
# replaced.


def _load_named_benchmark(name: str, state: AgentState, acquire_meta: dict):
    """Load one curated benchmark by name, sized to the run's curriculum and eval targets."""
    from tasks import get_task

    spec = get_task(name)
    if state["task"] != spec.name:
        raise ValueError(
            f"SLM_BENCHMARK_TASK={spec.name!r} does not match the run's task "
            f"{state['task']!r}; set them consistently"
        )
    # As many gold rows as the source has, up to the task's cap. There is no fraction and no
    # split: the number used to be `curriculum_size_target × 0.65`, which is where the mystery
    # 3,250 came from — 65% of a 5,000-row "target" the curriculum was then never allowed to
    # reach, because the only thing that could have filled the gap was re-drawing rows it already
    # had. The curriculum starts at this size and GROWS from here (see agent/data_rebuild.py).
    max_train = spec.initial_train_cap
    # A bigger eval set is strictly better for variance — it costs one extra inference pass per
    # iteration and buys precision on exactly the comparisons this project makes — but the eval runs
    # EVERY iteration, so an unbounded split makes each turn proportionally slower.
    _cap = os.environ.get("SLM_EVAL_SIZE_CAP")
    max_test = int(_cap) if _cap else spec.eval_cap
    print(f"      [eval_setup] loading {spec.name!r} ({spec.title}): "
          f"train\u2264{max_train} test\u2264{max_test}"
          + ("" if _cap else " (task default; override with SLM_EVAL_SIZE_CAP)"))
    train_examples, test_examples = spec.load(max_train=max_train, max_test=max_test)
    print(f"      [eval_setup] {spec.name}: loaded {len(train_examples)} gold train rows "
          f"(asked for {max_train}) and {len(test_examples)} eval rows (asked for {max_test})")
    # Record what we consumed per source, so a later `mine_new_real` knows which corpora still have
    # rows we have not taken. Without this the loop had no way to tell "this dataset is exhausted"
    # from "we only ever asked for the first 3,250 rows of it" (B297).
    state["source_progress"] = {
        source.hf_id: {
            "consumed": len(train_examples),
            "asked_for": max_train,
            "url": source.url,
            # Exhausted only when the loader returned FEWER rows than we asked for; that is the
            # only reliable evidence a head slice has reached the end of the split.
            "exhausted": len(train_examples) < max_train,
        }
        for source in spec.mining_sources
    }
    # Stage-0 decontamination. Official benchmark splits are not guaranteed disjoint (CLINC150
    # ships "what's your designation" in both splits under two different intents), so without this
    # the eval_setup overlap firewall would turn a source-data quirk into a fatal raise. The
    # held-out test rows are authoritative and never modified; the train row is dropped.
    train_examples, overlap_removed = remove_normalized_train_overlap(
        train_examples, test_examples)
    if overlap_removed:
        print(f"      [eval_setup] Stage-0 normalized overlap removal for {spec.name!r}: "
              f"removed {overlap_removed} train row(s); official test rows unchanged")
    acquire_meta["overlap_removed_from_train"] = overlap_removed
    acquire_meta["source"] = spec.title
    acquire_meta["source_records"] = [
        {"kind": "hf", "id": spec.name, "split": "train", "role": "curriculum"},
        {"kind": "hf", "id": spec.name, "split": "test", "role": "eval"},
    ]
    acquire_meta["eval_ban"] = [
        {"kind": "hf", "id": spec.name, "split": "test", "role": "eval"}
    ]
    return train_examples, test_examples


# The eval-set cap is per task (`TaskSpec.eval_cap`). Bigger is statistically better — the standard
# error on a proportion at n=1000 is about 1.5 percentage points, well below the score differences
# this project cares about — but the eval runs EVERY iteration, so an unbounded split makes each
# loop turn proportionally slower: RouterBench's whole split is 7,267 rows. Each task picks the
# point on that trade-off that suits its own split size.


class QwenBaselineUnavailableError(RuntimeError):
    """The Qwen-3.6 reference endpoint could not be measured, so no accuracy goal exists.

    This is FATAL by design: the Qwen baseline is the sole accuracy target, so an unreachable
    endpoint has no honest fallback. Raising breaks the pipeline loop rather than degrading to a
    guessed threshold.
    """


def _gate_synthetic_data(state: AgentState, eval_set, train_rows: list[dict]) -> None:
    """Decide, once, whether this run may use synthetic data at all.

    Measured five-shot through the task's own scorer, because that is how synthesis prompts the
    teacher; a zero-shot number on a format-bound task mostly reports whether it guessed our output
    contract (B276). Below the gate, `surgical_synthesis` is removed from the intervention menu for
    the rest of the run — see agent/teacher_fitness.py.
    """
    from agent.teacher_fitness import measure_teacher_fitness
    from tasks import get_task

    spec = get_task(state["task"])
    state["teacher_fitness"] = measure_teacher_fitness(
        spec, eval_set, list(train_rows or []), log=print,
    )


def _author_task_brief(state: AgentState, train_rows: list[dict]) -> None:
    """Have the orchestrator describe this benchmark, once, before any teacher call.

    Every synthesis and verification prompt is built from this text, so it is authored here — after
    the real data is loaded, from real rows — rather than hardcoded per task type. See
    `agent/task_brief.py` for what that hardcoded table cost.
    """
    from agent.task_brief import build_task_brief
    from tasks import get_task

    spec = get_task(state["task"])
    state["task_brief"] = build_task_brief(spec, list(train_rows or []), log=print)


def _pin_label_space(state: AgentState, eval_set) -> None:
    """Close the task's label vocabulary against the frozen eval set, once, before any curation.

    Everything downstream (mining, LLM column mapping, synthesis, quality control) treats this as
    authoritative and may only REMOVE rows that fall outside it — never extend it. See
    `data/label_space.py` for why (B259: four hallucinated classes entered a two-class task).
    """
    from data.label_space import label_definitions_for, label_space_from_eval_set

    task = state["task"]
    labels = label_space_from_eval_set(eval_set, task)
    if not labels:
        state["task_label_space"] = None
        return
    benchmark = task
    definitions = label_definitions_for(task)
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
    try:
        # The task comes from the EVAL SET, which is why there is no task argument here. Passing one
        # put the task NAME in the `generate_fn` slot, so every one of the 1,000 eval rows failed with
        # `'str' object is not callable`, the baseline measured 0.0000, and the accuracy goal silently
        # fell back to the 0.80 floor while reporting that the teacher had scored zero (B313).
        baseline = measure_endpoint_baseline(eval_set, log=print)
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
    task = manifest.get("task")
    required = tuple((manifest.get("row_schema") or {}).get("required") or ())
    canonical_required = required_fields_for_task(task)
    if set(required) != set(canonical_required):
        raise ValueError(
            f"shared dataset row_schema mismatch for {task!r}: {required}"
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
    task flows from state — no hardcoding.
    """
    task = state["task"]
    plan = state.get("task_plan")

    # Shared-dataset harness (B161): for a FAIR strategy comparison, all runs can load ONE
    # frozen dataset + eval set (built once by scripts/prepare_shared_dataset.py) instead of
    # each independently re-acquiring — acquisition nondeterminism previously gave each
    # strategy different data, confounding the comparison. Enabled via SLM_SHARED_DATASET_DIR.
    _shared = _load_shared_dataset(os.environ.get("SLM_SHARED_DATASET_DIR", ""))
    if _shared is not None:
        (train_examples, test_examples, _shared_diff, _shared_sources,
         _shared_eval_ban, _shared_manifest) = _shared
        if _shared_manifest["task"] != task:
            raise ValueError(
                f"shared dataset task={_shared_manifest['task']!r} "
                f"does not match run task={task!r}"
            )
        print(f"      [eval_setup] using SHARED frozen dataset: train={len(train_examples)} "
              f"test={len(test_examples)} (SLM_SHARED_DATASET_DIR) — identical across strategy runs")
        state["train_examples"] = train_examples
        state["data_source"] = "shared frozen dataset"
        state["data_sources"] = list(_shared_sources or [])
        state["eval_source_ban"] = list(_shared_eval_ban or [])
        eval_set = build_eval_set(
            test_examples, task=task, target=_eval_target(len(test_examples)),
        )
        state["eval_set"] = eval_set
        state["eval_difficulty"] = _shared_diff
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        atomic_write_json(
            os.path.join(ARTIFACTS_DIR, "eval_set.json"),
            {
                "task": eval_set.task,
                "counts": {"total": len(eval_set.all)},
                "difficulty_counts": ({k: len(v) for k, v in _shared_diff.items()} if _shared_diff else {}),
                "examples": eval_set.all,
                "difficulty": _shared_diff or {},
            },
        )
        _pin_label_space(state, eval_set)
        _author_task_brief(state, train_examples)
        _gate_synthetic_data(state, eval_set, train_examples)
        _calibrate_qwen_goal_if_pending(state, eval_set)
        return state

    # The task's own loader, named on its spec. There is no branch here: every run names a task from
    # the registry, every registered task has a loader, and `get_task` already raised if it did not.
    #
    # What used to sit here was an "autonomous" path that asked the orchestrator to plan a task TYPE
    # and then went looking for a dataset to fit it, sized to `curriculum_size_target × 0.65`. That
    # is where the mystery 3,250 came from. It is gone because a task with no loader has no scorer,
    # no training prompt and no synthesis path either — the silent no-ops of B291 and B299 were all
    # downstream of pretending otherwise. Web research survives where it is useful: as rung 2 of the
    # `mine_new_real` ladder, finding a corpus to GROW a curriculum that already exists.
    acquire_meta: dict = {}
    train_examples, test_examples = _load_named_benchmark(task, state, acquire_meta)

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

    # The loader already applied the task's own eval cap, so the split IS the target — re-applying
    # a separate `eval_size_target` here would cap it a second time and undo the cap (B288).
    eval_set = build_eval_set(
        test_examples,
        task=task,
        target=_eval_target(len(test_examples)),
    )
    state["eval_set"] = eval_set
    # The "target" is the size of the held-out split the loader returned.
    _target = len(test_examples)
    print(f"      [eval_setup] eval set built: {len(eval_set.all)} examples "
          f"(available in the held-out split: {_target})")
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
    _author_task_brief(state, train_examples)
    _gate_synthetic_data(state, eval_set, train_examples)

    # Difficulty-stratify the eval set for the test-data agent (B161): label each held-out
    # example easy/medium/hard by the base-model zero-shot capability gradient (smallest vs
    # largest feasible model), so the test agent can report per-difficulty accuracy and give
    # targeted diagnoses. Guarded — falls back to a length heuristic on any failure.
    _difficulty_t0 = time.perf_counter()
    _difficulty_status = "success"
    try:
        from agent.nodes.test_agent import label_difficulty
        difficulty = label_difficulty(
            eval_set, state.get("feasible_models") or [], task, log=print)
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
            "task": eval_set.task,
            "counts": {"total": len(eval_set.all)},
            "difficulty_counts": ({k: len(v) for k, v in difficulty.items()} if difficulty else {}),
            "examples": eval_set.all,
            "difficulty": difficulty or {},
        },
    )
    _calibrate_qwen_goal_if_pending(state, eval_set)
    return state
