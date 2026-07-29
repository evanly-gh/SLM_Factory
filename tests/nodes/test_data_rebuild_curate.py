import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.data_rebuild import (
    data_rebuild_plan_identity,
    normalize_data_rebuild_plan,
)
from agent.nodes.curate import curate_node
from config.android_pool import ANDROID_POOL
from data.eval_set import EvalSet
from data.loaders.dataset_integrity import normalize_text


EVAL_SECRET = "held out evaluation secret 7319"


def _eval_set():
    return EvalSet(
        pos=[{"text": EVAL_SECRET, "label": "a"}],
        neg=[],
        boundary=[],
        task_type="classification",
    )


def _raw_plan(primary, supports=(), **overrides):
    raw = {
        "primary_strategy": primary,
        "support_strategies": list(supports),
        "target_rows": 16,
        "resample_fraction": 1.0,
        "preserve_elite_fraction": 0.25,
        "new_real_rows": 5,
        "synth_rows": 5,
        "max_acquire_rounds": 1,
        "query_variant": 2,
        "difficulty_buckets": {
            "easy": 0.1,
            "medium": 0.1,
            "hard": 0.8,
        },
        "confusion_pairs": [
            {"gold": "a", "predicted": "b", "count": 5},
        ],
        "pattern_hint": "aggregate a to b confusion",
        "elite": {
            "provenance": "current_dataset",
            "dataset_version": 2,
        },
    }
    raw.update(overrides)
    return raw


def _state(plan, rows, *, current_path=None, version=2, hypothesis="hard gap"):
    return {
        "task_type": "classification",
        "selected_model": ANDROID_POOL[0],
        "last_intervention": "data_rebuild",
        "last_hypothesis": hypothesis,
        "eval_set": _eval_set(),
        "train_examples": rows,
        "last_eval": SimpleNamespace(
            failures=[{
                "text": EVAL_SECRET,
                "label": "a",
                "predicted": "b",
            }],
        ),
        "test_report": {
            "confusion_pairs": [
                {"gold": "a", "predicted": "b", "count": 5},
            ],
        },
        "current_dataset_path": current_path,
        "curriculum_size_target": 16,
        "dataset_version": version,
        "data_source": "fixture",
        "data_sources": [],
        "eval_source_ban": [],
        "mode": "cold_start",
        "replay_buffer": [],
        "source_acquire_rounds_used": 0,
        "data_rebuild_plan": plan,
        "data_rebuild_plan_identity": data_rebuild_plan_identity(plan),
        "llm_iterate_decision": {
            "intervention": "data_rebuild",
            "hypothesis": hypothesis,
            "data_rebuild": plan,
            "data_rebuild_plan_identity": data_rebuild_plan_identity(plan),
        },
        "dag": [],
    }


def _rows(path):
    return [
        json.loads(line)
        for line in open(path, encoding="utf-8")
        if line.strip()
    ]


def _normal_texts(rows):
    return {
        normalize_text(row.get("text", row.get("prompt", "")))
        for row in rows
    }


def _plan(primary, supports=(), hypothesis="hard gap", **overrides):
    return normalize_data_rebuild_plan(
        _raw_plan(primary, supports, **overrides),
        task_type="classification",
        hypothesis=hypothesis,
        target_rows=16,
        default_dataset_version=2,
    )


def test_resample_existing_executes_declared_plan_and_records_yield(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "text": f"train row {index} unique words {index * 13}",
            "label": "a" if index % 2 == 0 else "b",
        }
        for index in range(30)
    ]
    plan = _plan("resample_existing")
    state = _state(plan, train)

    with patch("data.synth_client.is_available", return_value=False):
        out = curate_node(state)

    saved = _rows(out["current_dataset_path"])
    assert len(saved) == 16
    assert out["last_curation"]["data_rebuild_plan"] == plan
    assert (
        out["last_curation"]["data_rebuild_plan_identity"]
        == data_rebuild_plan_identity(plan)
    )
    assert out["last_curation"]["strategy_composition"][0]["strategy"] == (
        "resample_existing"
    )
    assert out["last_curation"]["plan_yield"]["status"] == "novel"
    assert out["last_curation"]["plan_yield"]["final_rows"] == 16
    assert all(row.get("_provenance") == "train_anchor" for row in saved)


def test_curate_revalidates_and_clamps_plan_at_execution_boundary(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    raw = _raw_plan(
        "resample_existing",
        target_rows=99999,
        resample_fraction=0.63,
        max_acquire_rounds=99,
    )
    state = _state(
        _plan("resample_existing"),
        [
            {
                "text": f"boundary validation row {index}",
                "label": "a" if index % 2 == 0 else "b",
            }
            for index in range(20)
        ],
    )
    state["data_rebuild_plan"] = raw
    state["data_rebuild_plan_identity"] = "untrusted"

    with patch("data.synth_client.is_available", return_value=False):
        out = curate_node(state)

    assert out["data_rebuild_plan"]["target_rows"] == 2000
    assert out["data_rebuild_plan"]["resample_fraction"] == 0.65
    assert out["data_rebuild_plan"]["max_acquire_rounds"] == 3
    assert out["data_rebuild_plan_identity"] != "untrusted"


def test_preserve_elite_resample_keeps_rows_from_declared_version(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    prior = tmp_path / "prior.jsonl"
    elite = [
        {
            "text": f"elite row {index} distinct",
            "label": "a" if index % 2 == 0 else "b",
            "_provenance": "train_anchor",
            "_dataset_version": 2,
        }
        for index in range(6)
    ]
    prior.write_text(
        "".join(json.dumps(row) + "\n" for row in elite),
        encoding="utf-8",
    )
    train = [
        {
            "text": f"fresh row {index} distinct",
            "label": "a" if index % 2 == 0 else "b",
        }
        for index in range(30)
    ]
    plan = _plan(
        "preserve_elite_resample",
        preserve_elite_fraction=0.25,
    )
    state = _state(plan, train, current_path=str(prior))

    with patch("data.synth_client.is_available", return_value=False):
        out = curate_node(state)

    saved = _rows(out["current_dataset_path"])
    preserved = [
        row for row in saved
        if row.get("_provenance") == "elite"
    ]
    assert len(preserved) == 4
    assert {
        row["_elite_dataset_version"] for row in preserved
    } == {2}
    assert {
        row["_elite_provenance"] for row in preserved
    } == {"current_dataset"}
    assert {row["text"] for row in preserved} <= {
        row["text"] for row in elite
    }


def test_elite_rows_obey_new_dataset_balance_controls(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    prior = tmp_path / "elite-heavy.jsonl"
    elite = [
        {
            "text": f"winning elite evidence {index} distinct tokens {index * 17}",
            "label": "a",
            "_provenance": "train_anchor",
            "_dataset_version": 2,
        }
        for index in range(13)
    ]
    prior.write_text(
        "".join(json.dumps(row) + "\n" for row in elite),
        encoding="utf-8",
    )
    train = [
        {"text": f"new class b row {index}", "label": "b"}
        for index in range(20)
    ]
    plan = _plan(
        "preserve_elite_resample",
        preserve_elite_fraction=0.8,
    )

    with patch("data.synth_client.is_available", return_value=False):
        out = curate_node(
            _state(plan, train, current_path=str(prior))
        )

    saved = _rows(out["current_dataset_path"])
    elite_kept = sum(
        row.get("_provenance") == "elite" for row in saved
    )
    fresh_kept = sum(
        row.get("_provenance") == "train_anchor" for row in saved
    )
    assert elite_kept == 9
    assert elite_kept <= 3 * fresh_kept


def test_elite_selection_is_quality_ranked_deterministic_and_qc_gated(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    prior = tmp_path / "elite-quality.jsonl"
    elite = [
        {
            "text": (
                "outlier " + ("verylong " * 200)
                if index == 9
                else f"quality elite row {index} distinct token {index * 31}"
            ),
            "label": "a" if index % 2 == 0 else "b",
            "_quality_score": index,
            "_dataset_version": 2,
        }
        for index in range(10)
    ]
    prior.write_text(
        "".join(json.dumps(row) + "\n" for row in elite),
        encoding="utf-8",
    )
    train = [
        {
            "text": f"balanced fresh row {index}",
            "label": "a" if index % 2 == 0 else "b",
        }
        for index in range(30)
    ]
    plan = _plan(
        "preserve_elite_resample",
        preserve_elite_fraction=0.25,
    )

    with patch("data.synth_client.is_available", return_value=False):
        out = curate_node(_state(plan, train, current_path=str(prior)))

    selected = [
        row for row in _rows(out["current_dataset_path"])
        if row.get("_provenance") == "elite"
    ]
    assert [row["_quality_score"] for row in selected] == [8, 7, 6, 5]
    assert not any(row["text"].startswith("outlier") for row in selected)


def test_source_diversification_balances_train_source_provenance(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "text": f"dominant source row {index}",
            "label": "a" if index % 2 == 0 else "b",
            "_source": "source-a",
        }
        for index in range(30)
    ] + [
        {
            "text": f"rare source row {index}",
            "label": "a" if index % 2 == 0 else "b",
            "_source": "source-b",
        }
        for index in range(6)
    ]
    plan = _plan("source_diversification")

    with patch("data.synth_client.is_available", return_value=False):
        out = curate_node(_state(plan, train))

    saved = _rows(out["current_dataset_path"])
    by_source = out["last_curation"]["source_composition"]
    assert by_source["source-b"] == 6
    assert by_source["source-a"] == 10
    assert {row["_source"] for row in saved} == {"source-a", "source-b"}


def test_source_diversification_rewrites_single_source_noop(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "text": f"single source row {index}",
            "label": "a" if index % 2 == 0 else "b",
            "_source": "only-source",
        }
        for index in range(24)
    ]
    plan = _plan("source_diversification")

    with patch("data.synth_client.is_available", return_value=False):
        out = curate_node(_state(plan, train))

    composition = out["last_curation"]["strategy_composition"]
    assert composition[0]["strategy"] == "source_diversification"
    assert composition[0]["rows"] == 0
    assert composition[1]["strategy"] == "resample_existing_fallback"
    assert out["last_curation"]["allocation_fallbacks"] == [{
        "policy": "rewrite_noop_strategy",
        "reason": "source diversification requires at least two sources",
        "from": "source_diversification",
        "to": "resample_existing_fallback",
    }]


def test_difficulty_weighting_uses_train_metadata_without_eval_text(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "text": f"easy train row {index}",
            "label": "a" if index % 2 == 0 else "b",
            "_difficulty": "easy",
        }
        for index in range(30)
    ] + [
        {
            "text": f"hard train row {index} with longer distinct context",
            "label": "a" if index % 2 == 0 else "b",
            "_difficulty": "hard",
        }
        for index in range(10)
    ]
    plan = _plan("difficulty_weighted_sampling")

    with patch("data.synth_client.is_available", return_value=False):
        out = curate_node(_state(plan, train))

    saved = _rows(out["current_dataset_path"])
    difficulty = out["last_curation"]["difficulty_composition"]
    assert difficulty["hard"] >= 10
    assert difficulty["hard"] > difficulty["easy"]
    assert normalize_text(EVAL_SECRET) not in _normal_texts(saved)


def test_zero_weight_difficulty_buckets_never_backfill(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "text": f"easy row {index}",
            "label": "a" if index % 2 == 0 else "b",
            "_difficulty": "easy",
        }
        for index in range(20)
    ] + [
        {
            "text": f"hard row {index}",
            "label": "a" if index % 2 == 0 else "b",
            "_difficulty": "hard",
        }
        for index in range(2)
    ]
    plan = _plan(
        "difficulty_weighted_sampling",
        difficulty_buckets={"easy": 0, "medium": 0, "hard": 1},
    )

    with patch("data.synth_client.is_available", return_value=False):
        out = curate_node(_state(plan, train))

    saved = _rows(out["current_dataset_path"])
    assert len(saved) == 2
    assert {row["_difficulty"] for row in saved} == {"hard"}
    assert out["last_curation"]["allocation_fallbacks"] == [{
        "policy": "nonzero_buckets_only",
        "reason": "difficulty quota unavailable",
        "unfilled_rows": 14,
    }]


def test_targeted_positive_synthesis_uses_only_train_anchors_and_hypothesis(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    hypothesis = "aggregate a to b confusion comes from negation"
    train = [
        {
            "text": f"safe synthesis anchor {index}",
            "label": "a" if index % 2 == 0 else "b",
        }
        for index in range(20)
    ]
    plan = _plan(
        "targeted_synth_positive",
        hypothesis=hypothesis,
    )
    captured = {}

    def synthesize(anchors, **kwargs):
        captured["anchors"] = list(anchors)
        captured["pattern_hint"] = kwargs["pattern_hint"]
        return [
            *anchors,
            {
                "text": "new verified positive boundary example",
                "label": "b",
                "_source": "synth:test-model",
            },
        ]

    with (
        patch(
            "agent.nodes.curate.synthesize_hard_negatives",
            side_effect=synthesize,
        ),
        patch("data.synth_client.is_available", return_value=True),
        patch("data.synth_client.get_generate_fn", return_value=MagicMock()),
        patch("config.config.SYNTH_MODEL", "test-model"),
    ):
        out = curate_node(
            _state(plan, train, hypothesis=hypothesis)
        )

    saved = _rows(out["current_dataset_path"])
    assert normalize_text(EVAL_SECRET) not in _normal_texts(captured["anchors"])
    assert "causal hypothesis" in captured["pattern_hint"]
    assert "negation" in captured["pattern_hint"]
    assert any(
        row.get("_provenance") == "targeted_synth_positive"
        for row in saved
    )
    assert (
        out["last_curation"]["provenance_composition"][
            "targeted_synth_positive"
        ]
        == 1
    )


def test_unavailable_positive_synthesis_is_explicitly_rewritten(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "text": f"fallback synthesis anchor {index}",
            "label": "a" if index % 2 == 0 else "b",
        }
        for index in range(20)
    ]
    plan = _plan("targeted_synth_positive")

    with patch("data.synth_client.is_available", return_value=False):
        out = curate_node(_state(plan, train))

    composition = out["last_curation"]["strategy_composition"]
    assert composition[0]["strategy"] == "targeted_synth_positive"
    assert composition[0]["rows"] == 0
    assert any(
        item["strategy"] == "base_fill" and item["rows"] > 0
        for item in composition
    )
    assert out["last_curation"]["allocation_fallbacks"] == [{
        "policy": "rewrite_noop_strategy",
        "reason": "positive synthesis produced no verified rows",
        "from": "targeted_synth_positive",
        "to": "base_fill",
    }]


def test_final_target_cap_applies_after_elite_synth_and_replay(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    prior = tmp_path / "prior-cap.jsonl"
    prior.write_text(
        "".join(
            json.dumps({
                "text": f"elite cap row {index}",
                "label": "a" if index % 2 == 0 else "b",
                "_dataset_version": 2,
            }) + "\n"
            for index in range(8)
        ),
        encoding="utf-8",
    )
    train = [
        {
            "text": f"cap anchor {index}",
            "label": "a" if index % 2 == 0 else "b",
        }
        for index in range(30)
    ]
    plan = _plan(
        "targeted_synth_positive",
        supports=("preserve_elite_resample",),
        synth_rows=10,
    )
    state = _state(plan, train, current_path=str(prior))
    state["mode"] = "production"
    state["replay_buffer"] = [
        {
            "text": f"replay cap row {index}",
            "label": "a" if index % 2 == 0 else "b",
        }
        for index in range(20)
    ]

    def synthesize(anchors, **_kwargs):
        return [
            *anchors,
            *[
                {
                    "text": f"generated cap row {index}",
                    "label": "a" if index % 2 == 0 else "b",
                    "_source": "synth:test-model",
                }
                for index in range(10)
            ],
        ]

    with (
        patch(
            "agent.nodes.curate.synthesize_hard_negatives",
            side_effect=synthesize,
        ),
        patch("data.synth_client.is_available", return_value=True),
        patch("data.synth_client.get_generate_fn", return_value=MagicMock()),
        patch("config.config.SYNTH_MODEL", "test-model"),
    ):
        out = curate_node(state)

    saved = _rows(out["current_dataset_path"])
    assert len(saved) == plan["target_rows"] == 16
    assert out["last_curation"]["total_examples"] == 16
    assert sum(
        out["last_curation"]["provenance_composition"].values()
    ) == 16
    composition = out["last_curation"]["provenance_composition"]
    assert composition.get("elite", 0) <= round(
        16 * plan["preserve_elite_fraction"]
    )
    assert composition.get("targeted_synth_positive", 0) <= plan["synth_rows"]
    assert composition.get("replay", 0) <= round(16 * 0.20)


def test_mining_merges_only_novel_non_eval_real_rows_and_accounts_rounds(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "text": f"existing train row {index}",
            "label": "a" if index % 2 == 0 else "b",
        }
        for index in range(20)
    ]
    plan = _plan("mine_new_real_source")
    mined = [
        {
            "text": "novel mined real row",
            "label": "b",
            "_source": "hf:new/train",
        },
        {"text": EVAL_SECRET, "label": "a", "_source": "hf:new/train"},
    ]
    mining_report = {
        "requested": 5,
        "candidate_rows": 2,
        "novel_rows": 1,
        "novel_fraction": 0.5,
        "paid_rounds_used": 1,
        "status": "novel",
        "source_records": [{
            "kind": "hf",
            "id": "new/source",
            "split": "train",
            "role": "curriculum",
        }],
    }

    with (
        patch(
            "agent.nodes.curate.mine_additional_real_rows",
            return_value=(mined, mining_report),
        ) as mine,
        patch("data.synth_client.is_available", return_value=False),
    ):
        out = curate_node(_state(plan, train))

    mine.assert_called_once()
    saved = _rows(out["current_dataset_path"])
    assert "novel mined real row" in {row["text"] for row in saved}
    assert normalize_text(EVAL_SECRET) not in _normal_texts(saved)
    assert out["source_acquire_rounds_used"] == 1
    assert out["last_curation"]["source_novelty"]["novel_rows"] == 1
    assert out["data_sources"][-1]["id"] == "new/source"


def test_mined_real_rows_persist_in_train_state_with_dedup_and_provenance(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "text": f"persistent train row {index}",
            "label": "a" if index % 2 == 0 else "b",
        }
        for index in range(20)
    ]
    plan = _plan("mine_new_real_source")
    source_record = {
        "kind": "hf",
        "id": "persistent/source",
        "split": "train",
        "role": "curriculum",
    }
    mined = [
        {
            "text": "newly mined durable row",
            "label": "b",
            "_source": "hf:persistent/source/train",
            "_source_record": source_record,
        },
        {
            "text": "  newly   mined durable row ",
            "label": "b",
            "_source": "duplicate",
        },
        {
            "text": EVAL_SECRET,
            "label": "a",
            "_source": "hf:persistent/source/train",
        },
    ]
    report = {
        "requested": 5,
        "candidate_rows": 3,
        "novel_rows": 1,
        "novel_fraction": 1 / 3,
        "paid_rounds_used": 0,
        "status": "novel",
        "source_records": [source_record],
    }

    with (
        patch(
            "agent.nodes.curate.mine_additional_real_rows",
            return_value=(mined, report),
        ),
        patch("data.synth_client.is_available", return_value=False),
    ):
        out = curate_node(_state(plan, train))

    persisted = out["train_examples"]
    normalized = _normal_texts(persisted)
    assert len(persisted) == len(train) + 1
    assert normalize_text(EVAL_SECRET) not in normalized
    assert sum(
        normalize_text(row["text"])
        == normalize_text("newly mined durable row")
        for row in persisted
    ) == 1
    durable = next(
        row for row in persisted
        if normalize_text(row["text"])
        == normalize_text("newly mined durable row")
    )
    assert durable["_provenance"] == "mined_real"
    assert durable["_strategy_origin"] == "mine_new_real_source"
    assert durable["_source_record"] == source_record


def test_mining_no_novelty_does_not_override_other_strategy_novelty(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    prior = tmp_path / "prior-novelty.jsonl"
    prior.write_text(
        "".join(
            json.dumps({
                "text": f"existing row {index}",
                "label": "a" if index % 2 == 0 else "b",
                "_dataset_version": 2,
            }) + "\n"
            for index in range(5)
        ),
        encoding="utf-8",
    )
    train = [
        {
            "text": f"existing row {index}",
            "label": "a" if index % 2 == 0 else "b",
        }
        for index in range(20)
    ]
    plan = _plan(
        "resample_existing",
        supports=("mine_new_real_source",),
    )
    report = {
        "requested": 5,
        "candidate_rows": 3,
        "novel_rows": 0,
        "novel_fraction": 0.0,
        "paid_rounds_used": 1,
        "status": "no_novelty",
        "source_records": [],
    }

    with (
        patch(
            "agent.nodes.curate.mine_additional_real_rows",
            return_value=([], report),
        ),
        patch("data.synth_client.is_available", return_value=False),
    ):
        out = curate_node(_state(plan, train, current_path=str(prior)))

    assert out["last_curation"]["source_novelty"]["status"] == "no_novelty"
    assert out["last_curation"]["plan_yield"]["status"] == "novel"
    assert out["last_curation"]["plan_yield"]["novel_rows"] > 0
    assert (
        out["last_curation"]["data_rebuild_plan_identity"]
        == data_rebuild_plan_identity(plan)
    )


def test_composed_plan_records_each_executed_strategy(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "text": f"composed train row {index}",
            "label": "a" if index % 2 == 0 else "b",
            "_source": f"source-{index % 3}",
            "_difficulty": "hard" if index % 3 == 0 else "easy",
        }
        for index in range(30)
    ]
    plan = _plan(
        "source_diversification",
        supports=("mine_new_real_source",),
    )
    mining_report = {
        "requested": 5,
        "candidate_rows": 2,
        "novel_rows": 2,
        "novel_fraction": 1.0,
        "paid_rounds_used": 0,
        "status": "novel",
        "source_records": [],
    }
    with (
        patch(
            "agent.nodes.curate.mine_additional_real_rows",
            return_value=([
                {"text": "causal mined one", "label": "a", "_source": "new"},
                {"text": "causal mined two", "label": "b", "_source": "new"},
            ], mining_report),
        ),
        patch("data.synth_client.is_available", return_value=False),
    ):
        out = curate_node(_state(plan, train))

    composition = out["last_curation"]["strategy_composition"]
    assert [item["strategy"] for item in composition] == [
        "source_diversification",
        "mine_new_real_source",
    ]
    saved = _rows(out["current_dataset_path"])
    assert {
        item["strategy"]: item["rows"] for item in composition
    } == dict(Counter(row["_strategy_origin"] for row in saved))
    assert sum(item["rows"] for item in composition) == len(saved)


def test_curation_trajectory_versions_each_distinct_plan(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    train = [
        {
            "text": f"trajectory row {index}",
            "label": "a" if index % 2 == 0 else "b",
            "_source": f"source-{index % 2}",
        }
        for index in range(24)
    ]
    first_plan = _plan("resample_existing")
    state = _state(first_plan, train, version=0)

    with patch("data.synth_client.is_available", return_value=False):
        first = curate_node(state)
        first_path = first["current_dataset_path"]
        first_identity = first["data_rebuild_plan_identity"]
        second_plan = _plan(
            "source_diversification",
            hypothesis="source coverage is narrow",
            query_variant=6,
        )
        first["last_hypothesis"] = "source coverage is narrow"
        first["data_rebuild_plan"] = second_plan
        first["data_rebuild_plan_identity"] = data_rebuild_plan_identity(
            second_plan
        )
        second = curate_node(first)

    assert first_path.endswith("dataset_v1.jsonl")
    assert (tmp_path / first_path).is_file()
    assert second["current_dataset_path"].endswith("dataset_v2.jsonl")
    assert second["dataset_version"] == 2
    assert second["data_rebuild_plan_identity"] != first_identity
    assert (
        second["last_curation"]["strategy_composition"][0]["strategy"]
        == "source_diversification"
    )
