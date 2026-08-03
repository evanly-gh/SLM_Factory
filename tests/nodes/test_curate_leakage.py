import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from config.android_pool import ANDROID_POOL
from data.eval_set import EvalSet
from data.loaders.dataset_integrity import normalize_text


EVAL_TEXT = "Held-Out\nEvaluation Secret"


def _eval_set():
    return EvalSet(
        all=[{"text": EVAL_TEXT, "label": "a"}],
        task_type="classification",
    )


def _saved_rows(state):
    with open(state["current_dataset_path"], encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalized_training_texts(rows):
    return {
        normalize_text(row.get("text", row.get("prompt", "")))
        for row in rows
    }


def test_data_rebuild_never_seeds_or_saves_normalized_eval_text(
    tmp_path,
    monkeypatch,
):
    from agent.nodes.curate import curate_node
    from agent.data_rebuild import normalize_data_rebuild_plan

    monkeypatch.chdir(tmp_path)
    captured_seeds = []
    train = [
        {"text": "safe train a", "label": "a"},
        {"text": "safe train b", "label": "b"},
        {"text": "another safe train a", "label": "a"},
        {"text": "  held-out evaluation   secret ", "label": "b"},
    ]
    failure = {"text": EVAL_TEXT, "label": "a", "predicted": "b"}
    state = {
        "task_type": "classification",
        "selected_model": ANDROID_POOL[0],
        "last_intervention": "data_rebuild",
        "eval_set": _eval_set(),
        "train_examples": train,
        "last_eval": SimpleNamespace(failures=[failure]),
        "current_dataset_path": None,
        "curriculum_size_target": 6,
        "dataset_version": 0,
        # Force the synthesize strategy so the eval firewall on synthesis seeds is exercised
        # (the fallback strategy is now non-deterministic).
        "data_rebuild_plan": normalize_data_rebuild_plan(
            {"strategy": "synthesize", "synth_rows": 5, "target_rows": 6},
            task_type="classification",
            hypothesis="firewall check",
        ),
    }

    def synthesize(seeds, **_kwargs):
        captured_seeds.extend(seeds)
        return list(seeds)

    with (
        patch(
            "agent.nodes.curate.synthesize_examples",
            side_effect=synthesize,
        ),
        patch("data.synth_client.wait_until_available", return_value=True),
        patch("data.synth_client.is_available", return_value=True),
        patch("data.synth_client.get_generate_fn", return_value=MagicMock()),
    ):
        out = curate_node(state)

    eval_normalized = normalize_text(EVAL_TEXT)
    assert eval_normalized not in _normalized_training_texts(captured_seeds)
    assert eval_normalized not in _normalized_training_texts(_saved_rows(out))

