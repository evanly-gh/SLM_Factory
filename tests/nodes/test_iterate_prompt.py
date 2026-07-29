import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from config.android_pool import ANDROID_POOL


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_iterate_prompt_receives_complete_curation_counts():
    from agent.nodes.iterate import _llm_iterate

    trajectory = """### Dataset
- Total examples: 12
- Initial gold: 5
- Source anchors: 2
- Generated hard rows: 3
- Replay rows: 1
- Rebuild plan identity: plan-1
- Strategy composition: [{"strategy": "resample_existing", "rows": 12}]
"""
    captured = {}

    def fake_invoke(_llm, messages, **_kwargs):
        captured["prompt"] = messages[1].content
        return MagicMock(
            tool_calls=[],
            content=(
                '{"intervention":"data_rebuild","hypothesis":"counts verified",'
                '"data_rebuild":{"primary_strategy":"resample_existing"},'
                '"threshold_adjustment":{"new_threshold":null,"reason":""}}'
            ),
        )

    chat = MagicMock()
    state = {
        "task_type": "classification",
        "selected_model": ANDROID_POOL[0],
        "iteration": 1,
        "scores": [0.5],
        "best_score": 0.5,
        "stop_threshold": 0.9,
        "initial_stop_threshold": 0.9,
        "last_eval": None,
        "dag": [],
        "test_report": None,
    }
    with (
        patch("langchain_anthropic.ChatAnthropic", return_value=chat),
        patch("data.curation_log.CurationLog.read_latest", return_value=trajectory),
        patch("agent.context_manager.should_compact", return_value=False),
        patch(
            "agent.nodes.iterate.tracked_chat_anthropic_invoke",
            side_effect=fake_invoke,
        ),
    ):
        decision = _llm_iterate(state)

    assert decision["hypothesis"] == "counts verified"
    for line in trajectory.strip().splitlines():
        assert line in captured["prompt"]
    chat.bind_tools.assert_not_called()


def test_concurrent_iterate_runs_read_only_their_explicit_curation_logs(
    tmp_path,
    monkeypatch,
):
    from agent.nodes.iterate import _llm_iterate

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data-curation.md").write_text(
        "STALE_PROJECT_ROOT_HISTORY",
        encoding="utf-8",
    )
    paths = []
    for name, marker in (("run-a", "CURRENT_RUN_A"), ("run-b", "CURRENT_RUN_B")):
        run_dir = tmp_path / name
        run_dir.mkdir()
        path = run_dir / "data-curation.md"
        path.write_text(marker, encoding="utf-8")
        paths.append(path)

    prompts = []

    def fake_invoke(_llm, messages, **_kwargs):
        prompts.append(messages[1].content)
        return MagicMock(
            tool_calls=[],
            content=(
                '{"intervention":"hyperparameter",'
                '"hypothesis":"use a distinct bounded optimizer setting",'
                '"hyperparams":{"lora_rank":4,"alpha_ratio":2,'
                '"weight_decay":0.01,'
                '"learning_rate":0.0002,"nr_epochs":1},'
                '"threshold_adjustment":{"new_threshold":null,"reason":""}}'
            ),
        )

    def state(path):
        return {
            "task_type": "classification",
            "selected_model": ANDROID_POOL[0],
            "iteration": 1,
            "scores": [0.5],
            "best_score": 0.5,
            "stop_threshold": 0.9,
            "initial_stop_threshold": 0.9,
            "dag": [],
            "test_report": None,
            "curation_log_path": str(path),
        }

    with (
        patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
            clear=False,
        ),
        patch("langchain_anthropic.ChatAnthropic", return_value=MagicMock()),
        patch("agent.context_manager.should_compact", return_value=False),
        patch(
            "agent.nodes.iterate.tracked_chat_anthropic_invoke",
            side_effect=fake_invoke,
        ),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            decisions = list(executor.map(_llm_iterate, map(state, paths)))

    assert len(decisions) == 2
    assert len(prompts) == 2
    for own, other in (
        ("CURRENT_RUN_A", "CURRENT_RUN_B"),
        ("CURRENT_RUN_B", "CURRENT_RUN_A"),
    ):
        matching = [prompt for prompt in prompts if own in prompt]
        assert len(matching) == 1
        assert other not in matching[0]
        assert "STALE_PROJECT_ROOT_HISTORY" not in matching[0]
