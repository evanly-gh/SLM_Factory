"""A checkpoint must survive a round trip, or a killed run cannot be resumed.

WHY THIS FILE EXISTS
    Runs are hours long on a preemptible allocation, so resume is not a nice-to-have — it is how a
    run finishes at all. The invariant is narrow and unforgiving: whatever `save_checkpoint` writes,
    `load_checkpoint` must give back, with domain objects rehydrated and no live runtime object
    pickled along the way.

    Two properties beyond the round trip matter as much:

      * The checkpoint stores a model SELECTOR, not a `ModelSpec`. Resuming resolves the selector
        against the CURRENT pool, so a pool edit is caught as an incompatibility rather than
        silently replaying stale hardware metadata.
      * A referenced artifact is fingerprinted, so a resumed run that points at weights which have
        changed underneath it fails instead of evaluating something else.
"""
from __future__ import annotations

import json

import pytest

from agent.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointArtifactError,
    CheckpointCompatibilityError,
    CheckpointCorruptError,
    RecursionBudgetExhausted,
    atomic_write_json,
    atomic_write_jsonl,
    capture_artifacts,
    checkpoint_compatibility,
    graph_topology_fingerprint,
    load_checkpoint,
    model_pool_fingerprint,
    remaining_recursion_limit,
    runtime_config_snapshot,
    save_checkpoint,
    stable_thread_id,
    validate_artifacts,
)
from agent.state_codec import StateCodecError, decode_state, encode_state
from config.android_pool import ANDROID_POOL, HardwareConstraints
from data.eval_set import EvalSet
from eval.harness import EvalResult

COMPATIBILITY = {
    "mode": "cold_start",
    "pool_fingerprint": "pool-abc",
    "topology_fingerprint": "topo-abc",
    "config_fingerprint": "config-abc",
}


def _state(**overrides):
    state = {
        "task": "clinc150",
        "selected_model": ANDROID_POOL[0],
        "feasible_models": [ANDROID_POOL[0], ANDROID_POOL[1]],
        "hardware_constraints": HardwareConstraints(1000, 800, 2000),
        "iteration": 4,
        "scores": [0.31, 0.44],
        "eval_history": [0.31, 0.44],
        "best_score": 0.44,
        "stop_threshold": 0.9,
        "last_eval": EvalResult(f1=0.44, per_class={"a": 0.4}, failures=[],
                                metric="macro_f1", format_valid=0.97),
        "source_progress": {"clinc/clinc_oos": {"consumed": 3250}},
        "failed_discovery_rounds": 1,
    }
    state.update(overrides)
    return state


def _save(path, state, **overrides):
    kwargs = {
        "thread_id": "slm-thread",
        "compatibility": COMPATIBILITY,
        "graph_steps": 12,
        "last_node": "evaluate",
        "next_nodes": ["iterate"],
        "cumulative_wall_time_s": 1234.5,
    }
    kwargs.update(overrides)
    return save_checkpoint(path, state, **kwargs)


# --------------------------------------------------------------------------
# The state codec
# --------------------------------------------------------------------------


def test_a_state_round_trips_and_rehydrates_its_domain_objects():
    decoded = decode_state(encode_state(_state()))

    assert decoded["selected_model"] is ANDROID_POOL[0]
    assert decoded["feasible_models"] == [ANDROID_POOL[0], ANDROID_POOL[1]]
    assert decoded["hardware_constraints"] == HardwareConstraints(1000, 800, 2000)
    assert decoded["last_eval"] == EvalResult(
        f1=0.44, per_class={"a": 0.4}, failures=[], metric="macro_f1", format_valid=0.97,
    )
    assert decoded["source_progress"] == {"clinc/clinc_oos": {"consumed": 3250}}


def test_a_state_carrying_an_eval_set_round_trips_through_the_json_codec():
    """Encoding then decoding must return an equal `EvalSet`, so a resumed run evaluates against
    byte-identical held-out rows — a resume that silently re-drew E would make every score before
    and after it incomparable.

    Asserted separately from the rest of the state because the eval set is the field the codec has
    most often disagreed with: `EvalSet` lost three fields the encoder still read, which raised
    AttributeError as soon as an eval set was on the state, i.e. from `eval_setup` onward. Nothing
    imports its way into that failure — it needs a state with a real eval set on it, which is what
    this builds.
    """
    eval_set = EvalSet(all=[{"text": "held out", "label": "local"}], task="routerbench")
    encoded = encode_state(_state(task="routerbench", eval_set=eval_set))
    assert json.loads(json.dumps(encoded)) == encoded, "the encoded state must be plain JSON"

    decoded = decode_state(encoded)
    assert decoded["eval_set"] == eval_set


def test_the_encoded_state_is_plain_json():
    """It is written to a file and read by other tools; anything that needs a pickle to survive is
    a live runtime object that should not have been in the checkpoint."""
    encoded = encode_state(_state())
    assert json.loads(json.dumps(encoded)) == encoded


def test_the_format_score_survives_the_round_trip():
    """`format_valid` is half of the diagnosis (B290). A resume that lost it would leave the
    orchestrator unable to tell a format problem from a content one for the rest of the run."""
    decoded = decode_state(encode_state(_state()))
    assert decoded["last_eval"].format_valid == 0.97


def test_a_model_is_stored_as_a_selector_not_as_a_spec():
    """So resume resolves against the CURRENT pool. Storing the spec would silently replay stale
    hardware metadata after a pool edit."""
    encoded = encode_state(_state())
    assert encoded["selected_model"] == ANDROID_POOL[0].selector
    assert isinstance(encoded["selected_model"], str)


def test_a_live_training_object_is_refused_rather_than_pickled():
    with pytest.raises(StateCodecError, match="runtime object"):
        encode_state({"trainer": object()})


def test_pending_training_outputs_are_dropped_and_reset_on_load():
    """The live objects the next node would have used are not restartable; the PATHS it needs are
    kept separately in `_pending_weights_refs`."""
    encoded = encode_state(_state(_pending_training_outputs=[object()],
                                  _pending_weights_refs=["/weights/iter4"]))
    assert "_pending_training_outputs" not in encoded
    assert encoded["_pending_weights_refs"] == ["/weights/iter4"]
    assert decode_state(encoded)["_pending_training_outputs"] is None


def test_a_non_finite_score_is_refused():
    """A NaN survives `json.dumps` as invalid JSON that a reader may or may not accept. Refusing at
    the boundary means the corruption is reported where it happened."""
    with pytest.raises(StateCodecError, match="non-finite"):
        encode_state({"best_score": float("nan")})


def test_decoding_requires_an_exact_model_selector():
    """A bare model id can match several quantized variants, and picking one would resume a
    different model than the run was training."""
    with pytest.raises(StateCodecError, match="exact model selector"):
        decode_state({"selected_model": "Qwen/Qwen3-0.6B"})


def test_decoding_a_selector_the_pool_no_longer_has_is_an_error():
    with pytest.raises(StateCodecError, match="absent or ambiguous"):
        decode_state({"selected_model": "Nobody/Model@Q4_K_M"})


def test_a_selected_model_of_none_round_trips():
    """Cold start, before model selection has run."""
    assert decode_state(encode_state({"selected_model": None}))["selected_model"] is None


# --------------------------------------------------------------------------
# The checkpoint file
# --------------------------------------------------------------------------


def test_a_saved_checkpoint_loads_back_with_its_state_and_progress(tmp_path):
    path = tmp_path / "checkpoint.json"
    _save(path, _state())
    loaded = load_checkpoint(path, expected_compatibility=COMPATIBILITY)

    assert loaded["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert loaded["thread_id"] == "slm-thread"
    assert loaded["progress"]["graph_steps"] == 12
    assert loaded["progress"]["last_node"] == "evaluate"
    assert loaded["progress"]["next_nodes"] == ["iterate"]
    assert loaded["progress"]["cumulative_wall_time_s"] == pytest.approx(1234.5)
    assert loaded["state"]["selected_model"] is ANDROID_POOL[0]
    assert loaded["state"]["iteration"] == 4


def test_saving_twice_preserves_the_original_creation_time(tmp_path):
    """A resumed run keeps one checkpoint file; losing `created_at` would make the run's own age
    unrecoverable from its artifacts."""
    path = tmp_path / "checkpoint.json"
    first = _save(path, _state())
    second = _save(path, _state(iteration=5))

    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]
    assert load_checkpoint(path)["state"]["iteration"] == 5


def test_a_checkpoint_at_another_schema_version_is_rejected(tmp_path):
    path = tmp_path / "checkpoint.json"
    _save(path, _state())
    payload = json.loads(path.read_text())
    payload["schema_version"] = CHECKPOINT_SCHEMA_VERSION - 1
    atomic_write_json(path, payload)

    with pytest.raises(CheckpointCompatibilityError, match="schema version"):
        load_checkpoint(path)


@pytest.mark.parametrize("key,label", [
    ("pool_fingerprint", "pool fingerprint"),
    ("topology_fingerprint", "topology fingerprint"),
    ("config_fingerprint", "config fingerprint"),
])
def test_a_drifted_fingerprint_refuses_the_resume(tmp_path, key, label):
    """Resuming across a changed model pool, graph topology or runtime config would mean the
    trajectory before and after the resume were produced by two different systems."""
    path = tmp_path / "checkpoint.json"
    _save(path, _state())

    with pytest.raises(CheckpointCompatibilityError, match=label):
        load_checkpoint(path, expected_compatibility={**COMPATIBILITY, key: "changed"})


def test_a_truncated_checkpoint_is_reported_as_corrupt_not_as_a_drift(tmp_path):
    """Different causes, different fixes: a drift means "start fresh deliberately", corruption means
    "the last write was interrupted"."""
    path = tmp_path / "checkpoint.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(CheckpointCorruptError, match="not valid JSON"):
        load_checkpoint(path)


def test_a_checkpoint_missing_a_required_section_is_corrupt(tmp_path):
    path = tmp_path / "checkpoint.json"
    _save(path, _state())
    payload = json.loads(path.read_text())
    del payload["progress"]
    atomic_write_json(path, payload)

    with pytest.raises(CheckpointCorruptError, match="progress"):
        load_checkpoint(path)


def test_an_absent_checkpoint_is_reported_as_unreadable(tmp_path):
    with pytest.raises(CheckpointCorruptError, match="cannot be read"):
        load_checkpoint(tmp_path / "nothing.json")


def test_a_pool_edit_is_visible_in_the_fingerprint():
    """The fingerprint covers the runtime-relevant metadata, not just the ids, so an edited RAM
    figure is a drift rather than a silent change of what "tier 0" means."""
    import dataclasses

    edited = [dataclasses.replace(ANDROID_POOL[0], notes="edited"), *ANDROID_POOL[1:]]
    assert model_pool_fingerprint(edited) != model_pool_fingerprint(ANDROID_POOL)
    assert model_pool_fingerprint(ANDROID_POOL) == model_pool_fingerprint()


def test_the_topology_fingerprint_is_stable_and_mode_scoped():
    assert graph_topology_fingerprint("cold_start") == graph_topology_fingerprint("cold_start")
    with pytest.raises(ValueError, match="production mode was removed"):
        graph_topology_fingerprint("production")


def test_the_resume_config_snapshot_can_be_taken():
    """`tests/pipeline/run.py` calls this at MODULE scope (`_EFFECTIVE_CONFIG`), so it is the very
    first thing a run does — before the graph is built, before a model is chosen, before anything
    that could be caught and reported. It gathers its settings by importing from half the codebase,
    so one stale name in it kills every run outright: an import of a constant the data-rebuild
    refactor had removed did exactly that, and no module-level smoke test saw it because the failure
    is inside the function body.
    """
    snapshot = runtime_config_snapshot("cold_start")
    assert snapshot["mode"] == "cold_start"
    assert "DATA_REBUILD_SCHEMA_VERSION" in snapshot


def test_checkpoint_compatibility_can_be_computed():
    """The three fingerprints a resume is validated against, plus the mode they were taken under.
    Computing them is what turns "the pool/topology/config changed" into a refused resume rather
    than a trajectory produced by two different systems."""
    assert set(checkpoint_compatibility(mode="cold_start")) == set(COMPATIBILITY)


def test_the_thread_id_is_derived_from_the_run_directory_and_is_not_a_secret(tmp_path):
    """LangGraph keys its SQLite state on the thread id, so it must be stable across restarts of the
    same run and different between two runs."""
    first = stable_thread_id(tmp_path / "run-a")
    assert first == stable_thread_id(tmp_path / "run-a")
    assert first != stable_thread_id(tmp_path / "run-b")
    assert first.startswith("slm-")
    assert str(tmp_path) not in first


# --------------------------------------------------------------------------
# Artifacts referenced by a checkpoint
# --------------------------------------------------------------------------


def test_a_referenced_dataset_is_fingerprinted_and_drift_is_caught(tmp_path):
    """A resumed run pointing at a dataset that changed underneath it would train on something else
    while reporting the original trajectory."""
    dataset = tmp_path / "dataset_v1.jsonl"
    atomic_write_jsonl(dataset, [{"text": "a", "label": "x"}])

    artifacts = capture_artifacts({"current_dataset_path": str(dataset)})
    assert artifacts, "the dataset path was not captured as an artifact"
    validate_artifacts(artifacts)

    atomic_write_jsonl(dataset, [{"text": "a", "label": "x"}, {"text": "b", "label": "y"}])
    with pytest.raises(CheckpointArtifactError):
        validate_artifacts(artifacts)


def test_a_referenced_artifact_that_has_been_deleted_is_caught(tmp_path):
    dataset = tmp_path / "dataset_v1.jsonl"
    atomic_write_jsonl(dataset, [{"text": "a", "label": "x"}])
    artifacts = capture_artifacts({"current_dataset_path": str(dataset)})

    dataset.unlink()
    with pytest.raises(CheckpointArtifactError):
        validate_artifacts(artifacts)


def test_artifact_validation_can_be_skipped_for_an_inspection_load(tmp_path):
    """Reading a checkpoint to report on a finished run must not require its weights to still exist."""
    dataset = tmp_path / "dataset_v1.jsonl"
    atomic_write_jsonl(dataset, [{"text": "a", "label": "x"}])
    path = tmp_path / "checkpoint.json"
    _save(path, _state(current_dataset_path=str(dataset)))
    dataset.unlink()

    loaded = load_checkpoint(path, validate_artifact_paths=False)
    assert loaded["state"]["current_dataset_path"] == str(dataset)


# --------------------------------------------------------------------------
# The recursion budget
# --------------------------------------------------------------------------


def test_the_recursion_budget_is_what_is_left_of_the_run_not_a_fresh_allowance():
    """A resumed run that reset the budget could loop forever across enough restarts.

    The `+1` is LangGraph's: it raises when `recursion_limit` equals the number of node supersteps
    even if the final node routed to END, so one extra superstep is needed to OBSERVE termination.
    It cannot execute another node.
    """
    assert remaining_recursion_limit(0, total=1500) == 1501
    assert remaining_recursion_limit(1200, total=1500) == 301
    # The budget shrinks by exactly the work already done.
    assert (remaining_recursion_limit(0, total=1500)
            - remaining_recursion_limit(1200, total=1500)) == 1200


def test_the_last_step_of_the_budget_can_still_observe_termination():
    """At exactly the cap the run may finish, but it may not run another node."""
    assert remaining_recursion_limit(1500, total=1500) == 1


def test_an_overspent_recursion_budget_refuses_to_resume():
    with pytest.raises(RecursionBudgetExhausted, match="exhausted"):
        remaining_recursion_limit(1501, total=1500)


# --------------------------------------------------------------------------
# Atomic writes
# --------------------------------------------------------------------------


def test_an_atomic_write_replaces_the_file_rather_than_truncating_it(tmp_path):
    """A checkpoint half-written when the allocation is preempted is worse than no checkpoint: the
    run has a file it will try to resume from and cannot."""
    path = tmp_path / "checkpoint.json"
    atomic_write_json(path, {"a": 1})
    atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    assert list(tmp_path.iterdir()) == [path], "a temporary file was left behind"


def test_an_atomic_jsonl_write_leaves_one_row_per_line(tmp_path):
    path = tmp_path / "dataset.jsonl"
    atomic_write_jsonl(path, [{"text": "a"}, {"text": "b"}])
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert lines == [{"text": "a"}, {"text": "b"}]


# ── Adding a fingerprint key must not brick runs already in flight ───────────────────────────
#
# clinc150 run 40105479 died on an infrastructure error at 1d13h and then could NOT be resumed,
# because SLM_ABLATION_TRAIN_CAP and SLM_ABLATION_DISALLOW_MINING had been ADDED to the resume
# fingerprint while it was running. Its manifest had no entry for either, the live snapshot had
# both at their defaults, and the guard read `stored=None, expected='0'` as somebody changing the
# configuration mid-run. Absence and default say the same thing about that run — the switch was off
# then because it did not exist, and it is off now — so the guard has to forgive exactly that case
# and no more.


def _manifest_fixture(tmp_path, stored_config, compatibility):
    from agent.checkpoint import create_run_manifest

    return create_run_manifest(
        tmp_path / "run-manifest.json",
        run_dir=tmp_path,
        description="a task",
        force_model="",
        mode="cold_start",
        compatibility=compatibility,
        effective_config=stored_config,
    )


def _compat_for(config):
    from agent.checkpoint import _canonical_hash

    return {
        "mode": "cold_start",
        "pool_fingerprint": "pool",
        "topology_fingerprint": "topology",
        "config_fingerprint": _canonical_hash(config),
    }


def test_manifest_written_before_a_key_existed_still_resumes(tmp_path):
    """The key is absent from the manifest and switched off now: same run, so resume is allowed."""
    from agent.checkpoint import load_run_manifest

    stored = {"mode": "cold_start", "SLM_STAGNATION_WINDOW": "15"}
    _manifest_fixture(tmp_path, stored, _compat_for(stored))

    # The live snapshot gained the ablation-4 pair, both at their declared defaults.
    current = dict(stored, SLM_ABLATION_TRAIN_CAP="", SLM_ABLATION_DISALLOW_MINING="0")
    payload = load_run_manifest(
        tmp_path / "run-manifest.json",
        expected_description="a task",
        expected_force_model="",
        expected_mode="cold_start",
        expected_compatibility=_compat_for(current),
        expected_effective_config=current,
    )
    assert payload["effective_config"] == stored


def test_a_new_key_that_is_switched_on_still_blocks_resume(tmp_path):
    """The whole point of fingerprinting the cap: a full-data run must not resume into a capped one."""
    import pytest

    from agent.checkpoint import CheckpointCompatibilityError, load_run_manifest

    stored = {"mode": "cold_start", "SLM_STAGNATION_WINDOW": "15"}
    _manifest_fixture(tmp_path, stored, _compat_for(stored))

    current = dict(stored, SLM_ABLATION_TRAIN_CAP="151")
    with pytest.raises(CheckpointCompatibilityError, match="SLM_ABLATION_TRAIN_CAP"):
        load_run_manifest(
            tmp_path / "run-manifest.json",
            expected_description="a task",
            expected_force_model="",
            expected_mode="cold_start",
            expected_compatibility=_compat_for(current),
            expected_effective_config=current,
        )


def test_a_key_present_in_both_but_changed_still_blocks_resume(tmp_path):
    """Forgiveness is for keys that did not exist, not for keys whose value moved."""
    import pytest

    from agent.checkpoint import CheckpointCompatibilityError, load_run_manifest

    stored = {"mode": "cold_start", "SLM_ABLATION_TRAIN_CAP": "100"}
    _manifest_fixture(tmp_path, stored, _compat_for(stored))

    current = {"mode": "cold_start", "SLM_ABLATION_TRAIN_CAP": "151"}
    with pytest.raises(CheckpointCompatibilityError, match="SLM_ABLATION_TRAIN_CAP"):
        load_run_manifest(
            tmp_path / "run-manifest.json",
            expected_description="a task",
            expected_force_model="",
            expected_mode="cold_start",
            expected_compatibility=_compat_for(current),
            expected_effective_config=current,
        )


def test_switching_quantization_backend_blocks_resume(tmp_path):
    """Every score in the checkpoint was measured through ONE runtime on ONE toolchain's artifact.

    A resume that changed `SLM_QUANT_BACKEND` would go on comparing new MNN numbers against stored
    llama.cpp ones — best_score, the stagnation window, the rollback comparisons — and call the
    difference progress or regression. Measured, not hypothetical: the same weights score 0.7511
    through llama.cpp/GGUF and 0.7367 through MNN.
    """
    import pytest

    from agent.checkpoint import CheckpointCompatibilityError, load_run_manifest

    stored = {"mode": "cold_start", "SLM_QUANT_BACKEND": "llama_cpp"}
    _manifest_fixture(tmp_path, stored, _compat_for(stored))

    current = {"mode": "cold_start", "SLM_QUANT_BACKEND": "mnn"}
    with pytest.raises(CheckpointCompatibilityError, match="SLM_QUANT_BACKEND"):
        load_run_manifest(
            tmp_path / "run-manifest.json",
            expected_description="a task",
            expected_force_model="",
            expected_mode="cold_start",
            expected_compatibility=_compat_for(current),
            expected_effective_config=current,
        )


def test_a_run_started_before_the_backend_flag_existed_still_resumes(tmp_path):
    """Absent and default mean the same thing — otherwise adding the key bricks runs in flight.

    This is the `_keys_added_since` contract that clinc150 run 40105479 was lost to when the
    ablation keys were added mid-run.
    """
    from agent.checkpoint import load_run_manifest

    stored = {"mode": "cold_start"}
    _manifest_fixture(tmp_path, stored, _compat_for(stored))

    current = {"mode": "cold_start", "SLM_QUANT_BACKEND": "llama_cpp"}
    manifest = load_run_manifest(
        tmp_path / "run-manifest.json",
        expected_description="a task",
        expected_force_model="",
        expected_mode="cold_start",
        expected_compatibility=_compat_for(current),
        expected_effective_config=current,
    )
    assert manifest["compatibility"] == _compat_for(stored)


def test_unknown_keys_are_not_forgiven(tmp_path):
    """Only keys declared in `_RESUME_ENV_DEFAULTS` have a knowable 'off' value."""
    import pytest

    from agent.checkpoint import CheckpointCompatibilityError, load_run_manifest

    stored = {"mode": "cold_start"}
    _manifest_fixture(tmp_path, stored, _compat_for(stored))

    current = {"mode": "cold_start", "SLM_SOMETHING_NOBODY_DECLARED": ""}
    with pytest.raises(CheckpointCompatibilityError, match="SLM_SOMETHING_NOBODY_DECLARED"):
        load_run_manifest(
            tmp_path / "run-manifest.json",
            expected_description="a task",
            expected_force_model="",
            expected_mode="cold_start",
            expected_compatibility=_compat_for(current),
            expected_effective_config=current,
        )
