from unittest.mock import patch


def _call(**overrides):
    from data.loaders.web_acquire import mine_additional_real_rows

    values = {
        "task_plan": {
            "task_type": "classification",
            "task_name": "sentiment",
            "benchmark": "unknown",
            "labels": ["a", "b"],
            "exa_queries": {"a": "a samples", "b": "b samples"},
        },
        "description": "classify sentiment",
        "task_type": "classification",
        "existing_rows": [{"text": "already present", "label": "a"}],
        "eval_rows": [{"text": "held out secret", "label": "b"}],
        "eval_source_ban": [],
        "requested_rows": 3,
        "max_paid_rounds": 2,
        "query_variant": 1,
        "plan_identity": "test-plan",
        "log": lambda _message: None,
    }
    values.update(overrides)
    return mine_additional_real_rows(**values)


def test_source_mining_uses_clean_local_rows_before_paid_discovery():
    meta = {}

    def local(_plan, _task_type, _max_train, _max_test, log, meta):
        meta.update({
            "source": "local fixture",
            "source_records": [{
                "kind": "hf",
                "id": "fixture/source",
                "split": "train",
                "role": "curriculum",
            }],
        })
        return (
            [
                {"text": "already present", "label": "a"},
                {"text": "held   out secret", "label": "b"},
                {"text": "novel local one", "label": "a"},
                {"text": "novel local two", "label": "b"},
            ],
            [{"text": "local test", "label": "a"}],
        )

    with (
        patch(
            "data.loaders.web_acquire.load_local_dataset",
            side_effect=local,
        ),
        patch(
            "data.loaders.web_acquire.load_benchmark_dataset",
        ) as benchmark,
        patch(
            "data.loaders.web_acquire.discover_and_load_hf_dataset",
        ) as paid,
    ):
        rows, report = _call()

    benchmark.assert_not_called()
    paid.assert_not_called()
    assert {row["text"] for row in rows} == {
        "novel local one",
        "novel local two",
    }
    assert report["paid_rounds_used"] == 0
    assert report["novel_rows"] == 2
    assert report["status"] == "novel"
    assert report["source_records"][0]["id"] == "fixture/source"


def test_source_mining_bounds_paid_process_safe_discovery_rounds_and_variants(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SLM_RUN_DIR", str(tmp_path))
    calls = []

    def discover(plan, description, task_type, max_train, max_test, log, meta):
        calls.append((plan["task_name"], description))
        meta.update({
            "source": f"paid-{len(calls)}",
            "source_records": [{
                "kind": "hf",
                "id": f"paid/source-{len(calls)}",
                "split": "train",
                "role": "curriculum",
            }],
        })
        if len(calls) == 1:
            return (
                [{"text": "already present", "label": "a"}],
                [{"text": "paid test one", "label": "a"}],
            )
        return (
            [
                {"text": "paid novel one", "label": "a"},
                {"text": "paid novel two", "label": "b"},
                {"text": "paid novel three", "label": "a"},
            ],
            [{"text": "paid test two", "label": "b"}],
        )

    with (
        patch(
            "data.loaders.web_acquire.load_local_dataset",
            return_value=None,
        ),
        patch(
            "data.loaders.web_acquire.load_benchmark_dataset",
            return_value=None,
        ),
        patch(
            "data.loaders.web_acquire.discover_and_load_hf_dataset",
            side_effect=discover,
        ),
    ):
        rows, report = _call(max_paid_rounds=2, requested_rows=3)

    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert len(rows) == 3
    assert report["paid_rounds_used"] == 2
    assert report["novel_rows"] == 3


def test_source_mining_rejects_whole_source_eval_ban():
    def local(_plan, _task_type, _max_train, _max_test, log, meta):
        meta.update({
            "source_records": [{
                "kind": "hf",
                "id": "banned/source",
                "split": "train",
                "role": "curriculum",
            }],
        })
        return (
            [{"text": "would otherwise be novel", "label": "a"}],
            [{"text": "test", "label": "b"}],
        )

    with (
        patch(
            "data.loaders.web_acquire.load_local_dataset",
            side_effect=local,
        ),
        patch(
            "data.loaders.web_acquire.load_benchmark_dataset",
            return_value=None,
        ),
    ):
        rows, report = _call(
            max_paid_rounds=0,
            eval_source_ban=[{
                "kind": "hf",
                "id": "banned/source",
            }],
        )

    assert rows == []
    assert report["novel_rows"] == 0
    assert report["status"] == "no_novelty"
    assert report["rejected_sources"] == 1


def test_paid_round_is_durably_reserved_before_provider_call(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SLM_RUN_DIR", str(tmp_path))
    observed = {}

    def discover(*_args, **_kwargs):
        ledger = tmp_path / "acquisition-reservations.jsonl"
        events = [
            __import__("json").loads(line)
            for line in ledger.read_text(encoding="utf-8").splitlines()
        ]
        observed["events_during_call"] = events
        return None

    with (
        patch(
            "data.loaders.web_acquire.load_local_dataset",
            return_value=None,
        ),
        patch(
            "data.loaders.web_acquire.load_benchmark_dataset",
            return_value=None,
        ),
        patch(
            "data.loaders.web_acquire.discover_and_load_hf_dataset",
            side_effect=discover,
        ),
    ):
        _, report = _call(max_paid_rounds=1)

    assert [
        event["event"] for event in observed["events_during_call"]
    ] == ["reserved"]
    events = [
        __import__("json").loads(line)
        for line in (
            tmp_path / "acquisition-reservations.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "reserved",
        "completed",
    ]
    assert report["paid_rounds_used"] == 1
    assert report["run_paid_rounds_spent"] == 1


def test_failed_or_crashed_reservation_cannot_repeat_on_resume(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("SLM_RUN_DIR", str(tmp_path))
    with (
        patch(
            "data.loaders.web_acquire.load_local_dataset",
            return_value=None,
        ),
        patch(
            "data.loaders.web_acquire.load_benchmark_dataset",
            return_value=None,
        ),
        patch(
            "data.loaders.web_acquire.discover_and_load_hf_dataset",
            side_effect=RuntimeError("provider failed"),
        ) as discover,
    ):
        _, first = _call(max_paid_rounds=1, plan_identity="same-plan")
        _, second = _call(max_paid_rounds=1, plan_identity="same-plan")

    assert discover.call_count == 1
    assert first["paid_rounds_used"] == 1
    assert second["paid_rounds_used"] == 0
    assert second["paid_budget_exhausted"] is True
