import agent.nodes.curate as curate


def test_synth_fill_tops_up_to_target(monkeypatch):
    monkeypatch.delenv("SLM_CHEAP", raising=False)
    rows = [{"text": f"r{i}", "label": "a"} for i in range(10)]
    filled = curate._synth_fill_to_target(
        rows,
        target_rows=20,
        task_type="classification",
        generate_fn=lambda p, *a, **k: "brand new synthetic text",
        state={"task_type": "classification"},
        model_id="m",
    )
    assert len(filled) >= 20


def test_synth_fill_degrades_when_unavailable():
    rows = [{"text": "r", "label": "a"}]
    fallbacks = []
    out = curate._synth_fill_to_target(
        rows,
        target_rows=50,
        task_type="classification",
        generate_fn=None,
        state={"task_type": "classification"},
        model_id="m",
        fallbacks=fallbacks,
    )
    assert out == rows  # unchanged, no crash
    assert fallbacks and fallbacks[0]["from"] == "synthesize"


def test_synth_fill_noop_when_at_target():
    rows = [{"text": f"r{i}", "label": "a"} for i in range(30)]
    out = curate._synth_fill_to_target(
        rows,
        target_rows=20,
        task_type="classification",
        generate_fn=lambda p, *a, **k: "x",
        state={"task_type": "classification"},
        model_id="m",
    )
    assert out == rows
