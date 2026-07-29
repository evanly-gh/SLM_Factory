import ast
from pathlib import Path
from types import SimpleNamespace


def test_pipeline_error_is_reported_as_failure_not_budget_exhaustion():
    from agent.pipeline_status import outcome_text, process_exit_code, run_heading

    error = RuntimeError("CUDA out of memory")

    assert run_heading(error) == "RUN FAILED"
    outcome = outcome_text(False, error, "Qwen3-1.7B")
    assert "FAILED: RuntimeError: CUDA out of memory" == outcome
    assert "budget" not in outcome.lower()
    assert process_exit_code(error) == 1


def test_successful_pipeline_status_is_unchanged():
    from agent.pipeline_status import outcome_text, process_exit_code, run_heading

    assert run_heading(None) == "RUN COMPLETE"
    assert outcome_text(True, None, "Qwen3-1.7B") == "CONVERGED on Qwen3-1.7B"
    assert process_exit_code(None) == 0


def _downward_state():
    original_dag = [
        {
            "iteration": 3,
            "model_id": "test/Original",
            "score": 0.95,
        }
    ]
    return {
        "selected_model": SimpleNamespace(
            selector="test/Middle@Q8_0",
            model_id="test/Middle",
            quant="Q8_0",
            tier=1,
        ),
        # These remain the original model's normal-loop trajectory.
        "scores": [0.80, 0.90, 0.95],
        "dag": original_dag,
        "iteration": 3,
        "best_score": 0.92,
        "escalation_history": [],
        "downward_probe_history": {
            "origin": {
                "selector": "test/Original@bf16",
                "model_id": "test/Original",
                "quant": None,
                "tier": 2,
                "score": 0.95,
                "weights_ref": "/original/ckpt",
                "iterations": 3,
                "scores": [0.80, 0.90, 0.95],
                "dag": original_dag,
            },
            "attempts": [
                {
                    "selector": "test/Middle@Q8_0",
                    "model_id": "test/Middle",
                    "quant": "Q8_0",
                    "tier": 1,
                    "score": 0.92,
                    "weights_ref": "/middle/ckpt",
                    "result": "adopted",
                    "adopted": True,
                    "error": None,
                },
                {
                    "selector": "test/Small@Q4_K_M",
                    "model_id": "test/Small",
                    "quant": "Q4_K_M",
                    "tier": 0,
                    "score": 0.87,
                    "weights_ref": "/small/ckpt",
                    "result": "rejected",
                    "adopted": False,
                    "error": None,
                },
            ],
        },
    }


def test_progression_preserves_origin_identity_and_every_downward_probe():
    import agent.pipeline_status as status

    build = getattr(status, "build_run_progression", lambda *_args: [])
    progression = build(
        _downward_state(),
        [{"selector": "test/Original@bf16", "baseline_f1": 0.70}],
    )

    assert [entry["selector"] for entry in progression] == [
        "test/Original@bf16",
        "test/Middle@Q8_0",
        "test/Small@Q4_K_M",
    ]
    assert progression[0]["scores"] == [0.80, 0.90, 0.95]
    assert progression[0]["dag"][0]["model_id"] == "test/Original"
    assert progression[1]["scores"] == [0.92]
    assert progression[1]["dag"] == []
    assert progression[1]["result"] == "adopted"
    assert progression[2]["result"] == "rejected"
    assert not any(
        entry["selector"] == "test/Middle@Q8_0"
        and entry["scores"] == [0.80, 0.90, 0.95]
        for entry in progression
    )


def test_downward_report_lines_display_intermediate_exact_probes():
    import agent.pipeline_status as status

    format_history = getattr(
        status,
        "format_downward_probe_history",
        lambda _history: [],
    )
    lines = format_history(_downward_state()["downward_probe_history"])
    rendered = "\n".join(lines)

    assert "origin=test/Original@bf16" in rendered
    assert "test/Middle@Q8_0" in rendered
    assert "score=0.9200" in rendered
    assert "weights=/middle/ckpt" in rendered
    assert "result=adopted" in rendered
    assert "test/Small@Q4_K_M" in rendered
    assert "result=rejected" in rendered


def test_downward_report_displays_optional_probe_skip_reason():
    from agent.pipeline_status import format_downward_probe_history

    history = _downward_state()["downward_probe_history"]
    history["termination"] = {
        "stage": "model_chooser",
        "result": "skipped_error",
        "target_tier": 0,
        "candidate_selectors": ["test/Small@Q4_K_M"],
        "reason": "FatalLLMError: provider permission denied",
    }

    rendered = "\n".join(format_downward_probe_history(history))

    assert "stage=model_chooser" in rendered
    assert "result=skipped_error" in rendered
    assert "target_tier=0" in rendered
    assert "FatalLLMError: provider permission denied" in rendered


def test_pipeline_runner_uses_shared_progression_and_probe_formatters():
    source = (
        Path(__file__).parent / "pipeline" / "run.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "agent.pipeline_status"
        for alias in node.names
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "build_run_progression" in imported
    assert "format_downward_probe_history" in imported
    assert "build_run_progression" in calls
    assert "format_downward_probe_history" in calls
    assert 'log(f"  iterations: {last_state.get(' not in source
    assert "_final_entry.get('iterations', 0)" in source
    assert '"trajectory_selector"' in source
    assert '"final_selector"' in source
