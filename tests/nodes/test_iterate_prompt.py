import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from config.android_pool import ANDROID_POOL


def _stub_chat_anthropic(monkeypatch, chat):
    """Install a stub `langchain_anthropic` rather than patching the real one.

    `patch("langchain_anthropic.ChatAnthropic", ...)` has to IMPORT the package to patch it, and on
    this cluster's shared filesystem that import costs about a minute of wall clock — for a class
    every one of these tests then replaces. `iterate` imports it lazily inside the function, so a
    stub module is indistinguishable from the patch and the assertions are unchanged.
    """
    import sys
    from types import ModuleType

    module = ModuleType("langchain_anthropic")
    module.ChatAnthropic = lambda *_args, **_kwargs: chat
    monkeypatch.setitem(sys.modules, "langchain_anthropic", module)


@patch.dict(
    "os.environ",
    {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
    clear=False,
)
def test_iterate_prompt_receives_complete_curation_counts(monkeypatch):
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
                # `strategy`, not the retired `primary_strategy`: the 2026-07-31 curation redesign
                # replaced the primary/support strategy pair with one strategy field, and this
                # fixture was never updated — so the decision failed validation, the reask failed
                # identically, and this test's real assertion (that the curation counts reach the
                # prompt) had stopped running.
                '{"intervention":"data_rebuild","hypothesis":"counts verified",'
                '"data_rebuild":{"strategy":"surgical_synthesis"},'
                '"threshold_adjustment":{"new_threshold":null,"reason":""}}'
            ),
        )

    chat = MagicMock()
    _stub_chat_anthropic(monkeypatch, chat)
    state = {
        "task": "clinc150",
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
            "task": "clinc150",
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

    _stub_chat_anthropic(monkeypatch, MagicMock())
    with (
        patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "fake", "EXA_API_KEY": "fake"},
            clear=False,
        ),
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
