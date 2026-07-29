# agent/nodes/cold_start/eval_setup.py
import os
import json
import time
from agent.checkpoint import atomic_write_json
from agent.timing import TimingEvent, record_timing_event
from agent.state import AgentState
from data.eval_set import build_eval_set
from data.loaders.dataset_integrity import (
    normalize_text,
    normalized_text_overlap,
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


def _eval_split_sizes(target: int) -> dict:
    """Scale the pos/neg/boundary eval slices to the ORCHESTRATOR's eval-size target (B161).
    build_eval_set defaulted to 40/40/20 = 100 total, which silently capped the eval set to
    100 no matter how many test rows were acquired (the eval-size bug). Keep the 0.4/0.4/0.2
    ratio at the requested scale."""
    t = max(int(target or 0), 30)
    return {"n_pos": int(round(t * 0.4)), "n_neg": int(round(t * 0.4)),
            "n_boundary": int(round(t * 0.2))}


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
    Node 2: download data and build E = Epos ∪ Eneg ∪ Eboundary.
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
            **_eval_split_sizes(state.get("eval_size_target", 800)),
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
                "counts": {"pos": len(eval_set.pos), "neg": len(eval_set.neg),
                           "boundary": len(eval_set.boundary), "total": len(eval_set.all)},
                "difficulty_counts": ({k: len(v) for k, v in _shared_diff.items()} if _shared_diff else {}),
                "pos": eval_set.pos, "neg": eval_set.neg, "boundary": eval_set.boundary,
                "difficulty": _shared_diff or {},
            },
        )
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
        **_eval_split_sizes(state.get("eval_size_target", 800)),
        multi_label=plan.get("multi_label", False),
        schema=plan.get("schema", None),
        multilingual=plan.get("multilingual", False),
    )
    state["eval_set"] = eval_set
    print(f"      [eval_setup] eval set built: {len(eval_set.all)} examples "
          f"(target {state.get('eval_size_target', 800)}; pos={len(eval_set.pos)} "
          f"neg={len(eval_set.neg)} boundary={len(eval_set.boundary)})")

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
            "counts": {"pos": len(eval_set.pos), "neg": len(eval_set.neg),
                       "boundary": len(eval_set.boundary), "total": len(eval_set.all)},
            "difficulty_counts": ({k: len(v) for k, v in difficulty.items()} if difficulty else {}),
            "pos": eval_set.pos,
            "neg": eval_set.neg,
            "boundary": eval_set.boundary,
            "difficulty": difficulty or {},
        },
    )
    return state
