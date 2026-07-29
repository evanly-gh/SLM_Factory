"""Regression tests for the two defects found in run 37372065:

1. iterate._parse_decision_json crashed with a raw JSONDecodeError when the Anthropic
   response came back as a LIST of content blocks (str(list) → single-quoted repr).
2. train._build_config collapsed to the r=16 default for every data_rebuild and every
   config-less hyperparameter fallback — causing data rebuilds to auto-regress and a
   failed iterate LLM call to reproduce the previous iteration bit-for-bit (iter4≡iter5).
"""
import json
import pytest

from agent.nodes.iterate import _parse_decision_json, _coerce_to_text
from agent.nodes.train import _best_prior_config, _build_config, _config_diff, _DEFAULT_CONFIG

_VALID_DATA_REBUILD = (
    '{"intervention":"data_rebuild",'
    '"hypothesis":"rebalance aggregate training coverage",'
    '"data_rebuild":{"primary_strategy":"resample_existing"}}'
)


# --------------------------------------------------------------------------- #
# 1) JSON parsing robustness
# --------------------------------------------------------------------------- #
def test_parse_plain_json():
    out = _parse_decision_json(_VALID_DATA_REBUILD)
    assert out["intervention"] == "data_rebuild"


def test_parse_content_block_list_root_cause():
    # The exact shape that produced "Expecting property name ... line 1 column 2 (char 1)":
    # langchain-anthropic returns content as a list of blocks.
    raw = [{"type": "text", "text": (
        '{"intervention":"hyperparameter",'
        '"hypothesis":"increase adapter capacity",'
        '"hyperparams":{"lora_rank":32}}'
    )}]
    out = _parse_decision_json(raw)
    assert out["intervention"] == "hyperparameter"
    assert out["hyperparams"]["lora_rank"] == 32


def test_parse_content_block_list_multiple_blocks():
    raw = [
        {"type": "text", "text": "Here is my decision:\n"},
        {"type": "text", "text": (
            '{"intervention": "data_rebuild", '
            '"hypothesis": "remaining classes are imbalanced", '
            '"data_rebuild": {"primary_strategy": "resample_existing"}}'
        )},
    ]
    assert _parse_decision_json(raw)["intervention"] == "data_rebuild"


def test_parse_code_fenced():
    raw = f"```json\n{_VALID_DATA_REBUILD}\n```"
    assert _parse_decision_json(raw)["intervention"] == "data_rebuild"


def test_parse_prose_wrapped():
    raw = f"I think we should rebuild. {_VALID_DATA_REBUILD} done."
    assert _parse_decision_json(raw)["intervention"] == "data_rebuild"


def test_parse_single_quoted_dict_is_rejected_as_non_json():
    raw = (
        "{'intervention': 'hyperparameter', "
        "'hypothesis': 'increase adapter capacity', "
        "'hyperparams': {'lora_rank': 64}}"
    )
    with pytest.raises(ValueError, match="parseable JSON"):
        _parse_decision_json(raw)


def test_parse_garbage_raises_valueerror_not_jsondecode():
    # The caller catches Exception then raise_if_fatal; the important guarantee is that we
    # NEVER escape with a bare JSONDecodeError from the second json.loads.
    with pytest.raises(ValueError):
        _parse_decision_json("no json here at all")


def test_coerce_to_text_handles_list_and_str():
    assert _coerce_to_text("hi") == "hi"
    assert "abc" in _coerce_to_text([{"type": "text", "text": "abc"}])


# --------------------------------------------------------------------------- #
# 2) train._build_config carry-forward
# --------------------------------------------------------------------------- #
def _dag_node(rank, score, lr=5e-4, epochs=5, pruned=False, label=None):
    return {
        "pruned": pruned,
        "score": score,
        "best_config": label or f"LoRA r={rank}",
        "pi": {"H": {"lora_rank": rank, "learning_rate": lr, "nr_epochs": epochs, "batch_size": 8}},
    }


def test_first_iteration_uses_default():
    cfg, reason = _build_config({"dag": [], "last_intervention": "", "llm_iterate_decision": None})
    assert cfg["lora_rank"] == _DEFAULT_CONFIG["lora_rank"] == 16
    assert "first iteration" in reason


def test_data_rebuild_carries_forward_best_config():
    # Best node is r=32 @ 0.723. A data_rebuild must train at r=32, NOT revert to r=16.
    state = {
        "dag": [_dag_node(16, 0.576), _dag_node(32, 0.723)],
        "last_intervention": "data_rebuild",
        "llm_iterate_decision": {"intervention": "data_rebuild"},
    }
    cfg, reason = _build_config(state)
    assert cfg["lora_rank"] == 32
    assert cfg["learning_rate"] == pytest.approx(5e-4)
    assert "carry-forward" in reason


def test_config_diff_reports_only_changed_fields():
    old = {"lora_rank": 32, "learning_rate": 5e-4, "nr_epochs": 5, "lora_alpha": 64}
    new = {"lora_rank": 64, "learning_rate": 5e-4, "nr_epochs": 5, "lora_alpha": 64}
    assert _config_diff(old, new) == "lora_rank 32→64"


def test_config_diff_reports_multiple_changed_fields_in_field_order():
    old = {"lora_rank": 32, "learning_rate": 5e-4}
    new = {"lora_rank": 64, "learning_rate": 1e-4}
    assert _config_diff(old, new) == "lora_rank 32→64, learning_rate 0.0005→0.0001"


def test_config_diff_no_prior_config():
    assert _config_diff(None, {"lora_rank": 16}) == "no prior config (first iteration for this model)"


def test_config_diff_identical_configs():
    cfg = {"lora_rank": 32, "learning_rate": 5e-4}
    assert _config_diff(cfg, dict(cfg)) == "unchanged from best prior config"


def test_best_prior_config_feeds_diff_end_to_end():
    # Best node is r=32 @ 0.723 (see test_data_rebuild_carries_forward_best_config above).
    # The untried-rank fallback steps rank 32→64 AND scales alpha alongside it (2x rank
    # convention) — both fields must show up in the diff, and nothing else.
    state = {"dag": [_dag_node(16, 0.576), _dag_node(32, 0.723)]}
    prior = _best_prior_config(state)
    new_cfg, _ = _build_config({**state, "last_intervention": "hyperparameter", "llm_iterate_decision": None})
    assert _config_diff(prior, new_cfg) == "lora_rank 32→64, lora_alpha 64→128"


def test_failed_hyperparameter_fallback_steps_to_untried_rank():
    # LLM failed → intervention=hyperparameter but no decision/hyperparams. Best is r=32,
    # tried {16, 32}. Must step UP to r=64 (untried) — never repeat a tried config.
    state = {
        "dag": [_dag_node(16, 0.576), _dag_node(32, 0.723)],
        "last_intervention": "hyperparameter",
        "llm_iterate_decision": None,
    }
    cfg, reason = _build_config(state)
    assert cfg["lora_rank"] == 64
    assert "untried" in reason


def test_failed_fallback_holds_best_when_ladder_exhausted():
    state = {
        "dag": [_dag_node(4, 0.4), _dag_node(8, 0.5), _dag_node(16, 0.55),
                _dag_node(32, 0.72), _dag_node(64, 0.70)],
        "last_intervention": "hyperparameter",
        "llm_iterate_decision": None,
    }
    cfg, _ = _build_config(state)
    # All ranks tried → hold the best (r=32), do not crash.
    assert cfg["lora_rank"] == 32


def test_pruned_nodes_still_count_as_tried():
    # r=64 was tried but pruned (regressed). Best non-pruned is r=32. A fallback must skip
    # the pruned-but-tried r=64 and hold best (no untried rank left above 32).
    state = {
        "dag": [_dag_node(16, 0.576), _dag_node(32, 0.723), _dag_node(64, 0.60, pruned=True)],
        "last_intervention": "hyperparameter",
        "llm_iterate_decision": None,
    }
    cfg, _ = _build_config(state)
    assert cfg["lora_rank"] == 32  # 64 already tried (pruned), nothing higher untried


def test_explicit_hyperparams_still_respected():
    state = {
        "dag": [_dag_node(16, 0.576)],
        "last_intervention": "hyperparameter",
        "llm_iterate_decision": {"intervention": "hyperparameter", "hyperparams": {"lora_rank": 32, "learning_rate": 5e-4, "nr_epochs": 5}},
    }
    cfg, reason = _build_config(state)
    assert cfg["lora_rank"] == 32
    assert "LLM chose r=32" in reason


def test_baseline_best_node_not_carried():
    # If the only non-pruned node is the zero-shot baseline, there is no adapter config to
    # carry → fall back to default for a data_rebuild.
    base = _dag_node(16, 0.012, label="baseline (zero-shot, no adapter)")
    base["pi"]["H"] = {"lora_rank": None, "learning_rate": None, "nr_epochs": None, "batch_size": None}
    state = {"dag": [base], "last_intervention": "data_rebuild", "llm_iterate_decision": {"intervention": "data_rebuild"}}
    cfg, _ = _build_config(state)
    assert cfg["lora_rank"] == 16
