import json
import multiprocessing
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest


def _append_local_events(start: int, count: int) -> None:
    from agent.cost import CostEvent, record_cost_event

    for index in range(start, start + count):
        record_cost_event(
            CostEvent(
                provider="local",
                model="Qwen/Qwen3.6-35B",
                stage="multiprocess-test",
                callsite="tests.worker",
                status="success",
                input_tokens=index,
                output_tokens=1,
                latency_ms=1.0,
                estimated_usd=0.0,
            ),
        )


def _usage(**values):
    return SimpleNamespace(**values)


def _openai_response(
    *,
    response_id="chatcmpl-test",
    prompt_tokens=100,
    completion_tokens=20,
    cached_tokens=0,
    cache_hit_tokens=None,
    cache_miss_tokens=None,
):
    values = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tokens_details": SimpleNamespace(cached_tokens=cached_tokens),
    }
    if cache_hit_tokens is not None:
        values["prompt_cache_hit_tokens"] = cache_hit_tokens
    if cache_miss_tokens is not None:
        values["prompt_cache_miss_tokens"] = cache_miss_tokens
    return SimpleNamespace(id=response_id, usage=_usage(**values))


class _FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def create(self, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.response


class _FakeOpenAI:
    def __init__(self, base_url, response=None, error=None):
        self.base_url = base_url
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(response=response, error=error)
        )


def test_jsonl_ledger_is_process_safe_and_append_only(tmp_path, monkeypatch):
    from agent.cost import CostLedger

    event_path = tmp_path / "cost-events.jsonl"
    monkeypatch.setenv("SLM_COST_EVENT_PATH", str(event_path))
    ctx = multiprocessing.get_context("fork")
    workers = [
        ctx.Process(
            target=_append_local_events,
            args=(worker * 25, 25),
        )
        for worker in range(4)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)
        assert worker.exitcode == 0

    raw_lines = event_path.read_text().splitlines()
    assert len(raw_lines) == 100
    events = [json.loads(line) for line in raw_lines]
    assert {event["input_tokens"] for event in events} == set(range(100))

    snapshot = CostLedger(event_path).snapshot()
    assert snapshot["total_calls"] == 100
    assert snapshot["by_provider"]["local"]["calls"] == 100
    assert snapshot["total_cost_usd"] == 0.0


def test_official_pricing_accounts_for_anthropic_cache_tokens():
    from agent.cost import estimate_cost_usd, pricing_registry

    sonnet = estimate_cost_usd(
        "anthropic",
        "claude-sonnet-4-6-20260601",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    haiku = estimate_cost_usd(
        "anthropic",
        "claude-haiku-4-5",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )

    assert sonnet == pytest.approx(3 + 15 + 0.30 + 3.75)
    assert haiku == pytest.approx(1 + 5 + 0.10 + 1.25)
    registry = pricing_registry()
    assert registry["effective_date"] == "2026-07-21"
    # Cloud non-Anthropic teachers (gpt-4.1 / deepseek-v4-flash) were removed — the only
    # priced models are Anthropic tiers; the local Qwen synth endpoint is charged $0.
    assert "gpt-4.1" not in registry["models"]
    assert "deepseek-v4-flash" not in registry["models"]
    assert registry["models"]["claude-sonnet-5"]["input_per_mtok"] == 2.0
    assert registry["models"]["claude-sonnet-5"]["output_per_mtok"] == 10.0
    assert registry["models"]["claude-sonnet-5"]["valid_through"] == "2026-08-31"
    assert registry["models"]["claude-opus-4-8"]["input_per_mtok"] == 5.0
    assert "claude-fable-5" not in registry["models"]


def test_unknown_paid_model_is_explicit_and_can_be_strict_or_overridden(
    tmp_path, monkeypatch
):
    from agent.cost import (
        CostLedger,
        UnknownPricingError,
        UnknownPricingWarning,
        estimate_cost_usd,
        tracked_anthropic_messages_create,
    )

    with pytest.warns(UnknownPricingWarning, match="claude-fable-5"):
        assert estimate_cost_usd(
            "anthropic",
            "claude-fable-5",
            input_tokens=1_000,
            output_tokens=100,
        ) == 0.0
    with pytest.raises(UnknownPricingError, match="claude-fable-5"):
        estimate_cost_usd(
            "anthropic",
            "claude-fable-5",
            input_tokens=1_000,
            strict=True,
        )

    event_path = tmp_path / "events.jsonl"
    response = SimpleNamespace(
        id="msg_unknown",
        usage=_usage(input_tokens=1_000, output_tokens=100),
    )
    with pytest.warns(UnknownPricingWarning):
        tracked_anthropic_messages_create(
            _FakeCompletions(response=response),
            stage="task_analysis",
            event_path=event_path,
            model="claude-fable-5",
            messages=[],
        )
    event = CostLedger(event_path).events()[0]
    assert event["pricing_status"] == "unknown"
    snapshot = CostLedger(event_path).snapshot()
    assert snapshot["pricing_complete"] is False
    assert snapshot["unknown_pricing_calls"] == 1
    assert snapshot["unknown_pricing"][0]["model"] == "claude-fable-5"

    monkeypatch.setenv(
        "SLM_PRICING_OVERRIDES",
        json.dumps(
            {
                "models": {
                    "claude-fable-5": {
                        "provider": "anthropic",
                        "input_per_mtok": 9.0,
                        "output_per_mtok": 45.0,
                    }
                }
            }
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert estimate_cost_usd(
            "anthropic",
            "claude-fable-5",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
        ) == pytest.approx(54.0)
    assert caught == []


def test_pricing_registry_accepts_environment_overrides(monkeypatch):
    from agent.cost import estimate_cost_usd

    monkeypatch.setenv(
        "SLM_PRICING_OVERRIDES",
        json.dumps(
            {
                "models": {
                    "claude-opus-4-8": {
                        "input_per_mtok": 4.0,
                        "output_per_mtok": 10.0,
                    }
                }
            }
        ),
    )
    assert estimate_cost_usd(
        "anthropic", "claude-opus-4-8", input_tokens=1_000_000, output_tokens=1_000_000
    ) == pytest.approx(14.0)


def test_anthropic_wrapper_records_success_and_failure(tmp_path, capsys):
    from agent.cost import CostLedger, tracked_anthropic_messages_create

    event_path = tmp_path / "events.jsonl"
    response = SimpleNamespace(
        id="msg_123",
        usage=_usage(
            input_tokens=100,
            output_tokens=20,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=10,
        ),
    )
    messages = _FakeCompletions(response=response)
    assert (
        tracked_anthropic_messages_create(
            messages,
            stage="task_analysis",
            event_path=event_path,
            model="claude-sonnet-4-6",
            messages=[],
        )
        is response
    )

    failing = _FakeCompletions(error=RuntimeError("offline failure"))
    with pytest.raises(RuntimeError, match="offline failure"):
        tracked_anthropic_messages_create(
            failing,
            stage="task_analysis",
            event_path=event_path,
            model="claude-sonnet-4-6",
            messages=[],
        )

    events = CostLedger(event_path).events()
    assert [event["status"] for event in events] == ["success", "error"]
    assert events[0]["response_id"] == "msg_123"
    assert events[0]["cache_read_tokens"] == 30
    assert events[0]["cache_write_tokens"] == 10
    assert capsys.readouterr().out.count("[cost]") == 2


def test_anthropic_one_hour_cache_creation_is_not_double_charged(tmp_path):
    from agent.cost import CostLedger, tracked_anthropic_messages_create

    event_path = tmp_path / "events.jsonl"
    pure_one_hour = SimpleNamespace(
        id="msg_1h",
        usage=_usage(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=1_000_000,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=0,
                ephemeral_1h_input_tokens=1_000_000,
            ),
        ),
    )
    mixed = SimpleNamespace(
        id="msg_mixed",
        usage=_usage(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=1_000_000,
            cache_creation=SimpleNamespace(
                ephemeral_5m_input_tokens=400_000,
                ephemeral_1h_input_tokens=600_000,
            ),
        ),
    )
    for response in (pure_one_hour, mixed):
        tracked_anthropic_messages_create(
            _FakeCompletions(response=response),
            stage="cache_test",
            event_path=event_path,
            model="claude-sonnet-4-6",
            messages=[],
        )

    one_hour, mixed_event = CostLedger(event_path).events()
    assert one_hour["cache_write_tokens"] == 0
    assert one_hour["cache_write_1h_tokens"] == 1_000_000
    assert one_hour["estimated_usd"] == pytest.approx(6.0)
    assert mixed_event["cache_write_tokens"] == 400_000
    assert mixed_event["cache_write_1h_tokens"] == 600_000
    assert mixed_event["estimated_usd"] == pytest.approx(5.1)


def test_chat_anthropic_invoke_is_recorded_exactly_once(tmp_path):
    from agent.cost import CostLedger, tracked_chat_anthropic_invoke

    response = SimpleNamespace(
        id="lc-msg",
        usage_metadata={
            "input_tokens": 80,
            "output_tokens": 12,
            "input_token_details": {"cache_read": 20, "cache_creation": 5},
        },
        response_metadata={"model_name": "claude-sonnet-4-6"},
    )

    class FakeChat:
        def invoke(self, _messages):
            return response

    event_path = tmp_path / "events.jsonl"
    assert tracked_chat_anthropic_invoke(
        FakeChat(),
        [],
        stage="iterate",
        model="claude-sonnet-4-6",
        event_path=event_path,
    ) is response

    events = CostLedger(event_path).events()
    assert len(events) == 1
    assert events[0]["stage"] == "iterate"
    assert events[0]["input_tokens"] == 80
    assert events[0]["cache_read_tokens"] == 20


def test_local_vllm_openai_endpoint_is_never_charged(tmp_path):
    from agent.cost import CostLedger, tracked_openai_chat_create

    event_path = tmp_path / "events.jsonl"
    client = _FakeOpenAI(
        "http://127.0.0.1:8000/v1",
        response=_openai_response(prompt_tokens=50_000, completion_tokens=10_000),
    )
    tracked_openai_chat_create(
        client,
        stage="local_synthesis",
        event_path=event_path,
        model="Qwen/Qwen3.6-35B-A3B",
        messages=[],
    )

    event = CostLedger(event_path).events()[0]
    assert event["provider"] == "local"
    assert event["estimated_usd"] == 0.0
    assert event["input_tokens"] == 50_000
    assert event["output_tokens"] == 10_000


def test_local_preflight_call_records_latency_at_zero_api_cost(tmp_path):
    from agent.cost import CostLedger, tracked_local_call

    event_path = tmp_path / "events.jsonl"
    result = tracked_local_call(
        lambda: {"data": [{"id": "Qwen/Qwen3.6-35B-A3B"}]},
        stage="synth_preflight",
        model="Qwen/Qwen3.6-35B-A3B",
        operation="models.list",
        event_path=event_path,
    )

    assert result["data"][0]["id"].startswith("Qwen/")
    event = CostLedger(event_path).events()[0]
    assert event["provider"] == "local"
    assert event["estimated_usd"] == 0.0
    assert event["metadata"]["operation"] == "models.list"
    assert event["latency_ms"] >= 0.0


def test_exa_uses_actual_response_cost_and_records_failure(tmp_path):
    from agent.cost import CostLedger, tracked_exa_call

    event_path = tmp_path / "events.jsonl"

    def search():
        return {"requestId": "exa-1", "costDollars": {"total": 0.0123}}

    tracked_exa_call(
        search,
        stage="acquire",
        event_path=event_path,
        model="search-and-contents",
    )

    def fail():
        raise TimeoutError("exa timeout")

    with pytest.raises(TimeoutError):
        tracked_exa_call(
            fail,
            stage="acquire",
            event_path=event_path,
            model="search-and-contents",
        )

    success, failure = CostLedger(event_path).events()
    assert success["provider"] == "exa"
    assert success["estimated_usd"] == pytest.approx(0.0123)
    assert success["response_id"] == "exa-1"
    assert failure["status"] == "error"


def test_cost_snapshot_has_provider_model_and_stage_summaries(tmp_path):
    from agent.cost import CostEvent, CostLedger

    ledger = CostLedger(tmp_path / "events.jsonl")
    ledger.append(
        CostEvent(
            provider="local",
            model="Qwen/Qwen3.6-35B-A3B",
            stage="generation_judge",
            callsite="eval.scorers.generation.score",
            status="success",
            input_tokens=10,
            output_tokens=2,
            latency_ms=7.5,
            estimated_usd=0.0,
            pricing_status="not_applicable",
        )
    )
    snapshot = ledger.snapshot()

    assert snapshot["by_provider"]["local"]["calls"] == 1
    assert snapshot["by_model"]["Qwen/Qwen3.6-35B-A3B"]["calls"] == 1
    assert snapshot["by_stage"]["generation_judge"]["calls"] == 1
    assert snapshot["total_cost_usd"] == 0.0
    assert snapshot["pricing"]["effective_date"] == "2026-07-21"


def test_event_paths_are_absolute_exported_and_required_for_pipeline(
    tmp_path, monkeypatch
):
    from agent.cost import (
        MissingEventPathError,
        install_cost_tracking,
    )
    from agent.timing import MissingTimingPathError, install_timing_tracking

    monkeypatch.chdir(tmp_path)
    install_cost_tracking("relative/cost-events.jsonl", required=True)
    install_timing_tracking("relative/timing-events.jsonl", required=True)
    assert os.environ["SLM_COST_EVENT_PATH"] == str(
        (tmp_path / "relative/cost-events.jsonl").resolve()
    )
    assert os.environ["SLM_TIMING_EVENT_PATH"] == str(
        (tmp_path / "relative/timing-events.jsonl").resolve()
    )

    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, os; "
                "print(json.dumps([os.environ['SLM_COST_EVENT_PATH'], "
                "os.environ['SLM_TIMING_EVENT_PATH']]))"
            ),
        ],
        env=dict(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(child.stdout) == [
        os.environ["SLM_COST_EVENT_PATH"],
        os.environ["SLM_TIMING_EVENT_PATH"],
    ]

    monkeypatch.delenv("SLM_COST_EVENT_PATH")
    monkeypatch.delenv("SLM_TIMING_EVENT_PATH")
    monkeypatch.setenv("SLM_OBSERVABILITY_REQUIRED", "1")
    with pytest.raises(MissingEventPathError):
        install_cost_tracking()
    with pytest.raises(MissingTimingPathError):
        install_timing_tracking()


def test_observability_artifacts_are_atomic_and_include_early_exit_metadata(tmp_path):
    from agent.cost import CostEvent, CostLedger
    from agent.observability import write_observability_artifacts
    from agent.timing import TimingEvent, TimingLedger

    cost_ledger = CostLedger(tmp_path / "cost-events.jsonl")
    cost_ledger.append(
        CostEvent(
            provider="anthropic",
            model="claude-sonnet-4-6",
            stage="task_analysis",
            latency_ms=12.0,
            estimated_usd=0.001,
            pricing_status="known",
        )
    )
    cost_ledger.append(
        CostEvent(
            provider="local",
            model="Qwen/Qwen3.6-35B-A3B",
            stage="generation_judge",
            latency_ms=8.0,
            estimated_usd=0.0,
            pricing_status="not_applicable",
        )
    )
    timing_ledger = TimingLedger(tmp_path / "timing-events.jsonl")
    timing_ledger.append(
        TimingEvent(
            kind="phase",
            name="synth_preflight",
            duration_ms=20.0,
            status="error",
        )
    )
    run_dir = tmp_path / "run"

    cost, timings = write_observability_artifacts(
        run_dir,
        cost_ledger,
        timing_ledger,
        run_status="preflight_error",
        exit_code=2,
        reason="missing synth endpoint",
    )

    assert json.loads((run_dir / "cost.json").read_text()) == cost
    assert json.loads((run_dir / "timings.json").read_text()) == timings
    assert timings["run_exit"]["status"] == "preflight_error"
    assert timings["run_exit"]["exit_code"] == 2
    assert timings["api_calls"]["anthropic_api_latency_ms"] == 12.0
    assert timings["api_calls"]["anthropic_by_stage"]["task_analysis"][
        "latency_ms"
    ] == 12.0
    assert "generation_judge" not in timings["api_calls"]["anthropic_by_stage"]
    assert timings["api_calls"]["by_provider"]["local"]["latency_ms"] == 8.0
    assert "orchestrator_api_latency_ms" not in timings["api_calls"]
    assert list(run_dir.glob("*.tmp*")) == []


def test_timing_ledger_summarizes_graph_nodes_and_worker_ops(tmp_path):
    from agent.timing import TimingEvent, TimingLedger

    ledger = TimingLedger(tmp_path / "timing-events.jsonl")
    ledger.append(
        TimingEvent(
            kind="graph_node",
            name="curate",
            duration_ms=125.0,
            status="success",
        )
    )
    ledger.append(
        TimingEvent(
            kind="worker_op",
            name="eval",
            duration_ms=75.0,
            status="success",
        )
    )
    ledger.append(
        TimingEvent(
            kind="worker_dispatch",
            name="eval",
            duration_ms=50.0,
            status="success",
        )
    )
    summary = ledger.snapshot()

    assert summary["by_kind"]["graph_node"]["calls"] == 1
    assert summary["by_name"]["curate"]["duration_ms"] == 125.0
    assert summary["by_name"]["eval"]["duration_ms"] == 125.0
    assert summary["graph_nodes"]["curate"]["duration_ms"] == 125.0
    assert summary["worker_ops"]["eval"]["duration_ms"] == 75.0
    assert summary["worker_dispatch"]["eval"]["duration_ms"] == 50.0
    assert summary["worker_overhead"]["eval"]["duration_ms"] == 25.0


def test_timing_distinguishes_wall_clock_from_nested_event_sum(tmp_path):
    from agent.timing import TimingEvent, TimingLedger

    ledger = TimingLedger(tmp_path / "timing-events.jsonl")
    ledger.append(
        TimingEvent(
            kind="run",
            name="graph_pipeline",
            duration_ms=100.0,
            timestamp="2026-07-21T22:00:00.100Z",
        )
    )
    ledger.append(
        TimingEvent(
            kind="graph_node",
            name="curate",
            duration_ms=80.0,
            timestamp="2026-07-21T22:00:00.100Z",
        )
    )
    snapshot = ledger.snapshot()

    assert snapshot["summed_event_duration_ms"] == 180.0
    assert snapshot["wall_clock_span_ms"] == 100.0
    assert "total_duration_ms" not in snapshot


def test_instrumented_node_records_accurate_failure_name(tmp_path):
    from agent.timing import TimingLedger, instrument_node

    timing_path = tmp_path / "timing-events.jsonl"

    def fail(_state):
        raise RuntimeError("curate failed")

    wrapped = instrument_node("curate", fail, path=timing_path)
    with pytest.raises(RuntimeError, match="curate failed"):
        wrapped({})

    event = TimingLedger(timing_path).events()[0]
    assert event["kind"] == "graph_node"
    assert event["name"] == "curate"
    assert event["status"] == "error"
    assert event["metadata"]["error_type"] == "RuntimeError"


def test_disposable_worker_contributes_to_shared_timing_ledger(tmp_path, monkeypatch):
    from agent.timing import TimingLedger
    from training.cuda_isolation import run_isolated

    timing_path = tmp_path / "timing-events.jsonl"
    monkeypatch.setenv("SLM_TIMING_EVENT_PATH", str(timing_path))
    monkeypatch.setenv(
        "SLM_COST_EVENT_PATH", str(tmp_path / "cost-events.jsonl")
    )

    result = run_isolated("ping", {"source": "test"})

    assert result["source"] == "test"
    assert result["cost_event_path"] == str(
        (tmp_path / "cost-events.jsonl").resolve()
    )
    assert result["timing_event_path"] == str(timing_path.resolve())
    events = TimingLedger(timing_path).events()
    assert {(event["kind"], event["name"], event["status"]) for event in events} == {
        ("worker_dispatch", "ping", "success"),
        ("worker_op", "ping", "success"),
    }


def test_runner_and_cuda_worker_install_shared_instrumentation():
    root = Path(__file__).parents[1]
    runner = (root / "tests" / "pipeline" / "run.py").read_text()
    worker = (root / "training" / "cuda_worker.py").read_text()
    graph = (root / "agent" / "graph.py").read_text()
    observability = (root / "agent" / "observability.py").read_text()

    assert "SLM_COST_EVENT_PATH" in runner
    assert "SLM_TIMING_EVENT_PATH" in runner
    assert "install_cost_tracking" in runner
    assert "write_observability_artifacts" in runner
    assert "atexit.register" in runner
    assert "orchestrator_api_latency_ms" not in runner
    assert "timings.json" in observability
    assert "anthropic_by_stage" in observability
    assert "orchestrator_api_latency_ms" not in observability
    assert "install_cost_tracking" in worker
    assert "instrument_node" in graph
    assert 'kind="graph_node"' not in runner


def test_paid_callsites_use_central_tracking_wrappers():
    root = Path(__file__).parents[1]
    paid_paths = [
        *sorted((root / "agent").rglob("*.py")),
        *sorted((root / "data").rglob("*.py")),
        *sorted((root / "eval").rglob("*.py")),
    ]
    paid_paths = [path for path in paid_paths if path != root / "agent" / "cost.py"]

    direct_anthropic = []
    direct_openai = []
    direct_exa = []
    for path in paid_paths:
        source = path.read_text()
        if ".messages.create(" in source:
            direct_anthropic.append(str(path.relative_to(root)))
        if ".chat.completions.create(" in source:
            direct_openai.append(str(path.relative_to(root)))
        if re.search(
            r"\b(?:exa|_exa_client)\.(?:search|search_and_contents)\(",
            source,
        ):
            direct_exa.append(str(path.relative_to(root)))

    assert direct_anthropic == []
    assert direct_openai == []
    assert direct_exa == []

    iterate = (root / "agent" / "nodes" / "iterate.py").read_text()
    assert "llm.invoke(" not in iterate
    assert "llm_final.invoke(" not in iterate
    assert iterate.count("tracked_chat_anthropic_invoke(") == 2


def test_no_cloud_cot_teacher_config_or_curriculum_residue():
    """The CoT teacher is the local Qwen3.6 synth model only — no DeepSeek/OpenAI cloud
    teacher constants or thinking-mode plumbing may remain in config or curriculum."""
    root = Path(__file__).parents[1]
    config_source = (root / "config" / "config.py").read_text()
    curriculum_source = (root / "data" / "curriculum.py").read_text()

    assert "deepseek" not in config_source.lower()
    assert "TEACHER_MODEL_GPT" not in config_source
    assert "TEACHER_MODEL_DEEPSEEK" not in config_source
    assert "get_cot_fallbacks" not in curriculum_source
    assert 'reasoning_effort="high"' not in curriculum_source
