"""Unit tests for the consolidated on-device hardware evaluation module.

These cover the pure logic (no device, no adb): the backend dispatcher's
theoretical fallback, mA→W power conversion, logcat parsing, per-question
summarization, and the HardwareEvalResult→measured mapping consumed by
check_hardware_constraints.
"""
from hardware_eval.on_device_eval import (
    HardwareEvalResult,
    run_on_device_eval,
    unmeasured_profile,
    ma_to_watts,
    parse_run_lines,
    summarize_smolchat,
    _build_metrics,
)
from config.android_pool import CapabilityMeasurement, ModelSpec, HardwareConstraints


def _model(quant=None):
    return ModelSpec(
        model_id="test/Model-1B", size_mb=700, tier=1,
        capability_measurements=(
            CapabilityMeasurement(
                metric="MMLU", value=0.5, artifact="test/Model-1B",
                mode=None, protocol="test", source="https://example.test",
            ),
        ),
        quant=quant,
    )


def _constraints():
    return HardwareConstraints(
        storage_mb=5000, memory_mb=5000, latency_ttft_ms=3000,
        power_watts=5.0, target_chip="snapdragon_778g", min_tok_s=0.0,
    )


# --- Dispatcher --------------------------------------------------------------

def test_dispatcher_defaults_to_unmeasured_and_invents_nothing():
    """Default path must return all-None, not a modelled profile.

    This used to assert tok_per_s == 14.0 and peak_memory_mb == 1100, both read off a
    ModelSpec estimate table. Those fields are gone; the honest default is "unknown".
    """
    res = run_on_device_eval(_model(), _constraints())
    assert res.success is True
    assert res.eval_method == "unmeasured"
    assert res.tok_per_s is None
    assert res.ttft_ms is None
    assert res.peak_memory_mb is None
    assert res.avg_watts is None


def test_dispatcher_reports_unmeasured_without_gguf_even_if_backend_set():
    # A real backend requires a GGUF; without one we must not attempt adb/llama, and we
    # must not substitute an estimate either.
    res = run_on_device_eval(_model(), _constraints(), backend="smolchat", gguf_path=None)
    assert res.eval_method == "unmeasured"
    assert res.success is True
    assert res.tok_per_s is None


def test_dispatcher_unknown_backend_degrades_gracefully():
    res = run_on_device_eval(_model(), _constraints(), backend="does_not_exist",
                             gguf_path="/tmp/x.gguf")
    assert res.eval_method == "unmeasured"
    assert "unknown backend" in (res.error or "")


def test_legacy_theoretical_backend_name_yields_unmeasured():
    """Old configs/env values naming "theoretical" must not resurrect estimates."""
    res = run_on_device_eval(_model(), _constraints(), backend="theoretical")
    assert res.eval_method == "unmeasured"
    assert res.tok_per_s is None


def test_unmeasured_profile_emits_no_derived_ttft():
    """TTFT used to be synthesised as 1/tok_s*1000 — an estimate built on an estimate."""
    res = unmeasured_profile(_model(), _constraints())
    assert res.ttft_ms is None
    assert res.tok_per_s is None


# --- Power conversion --------------------------------------------------------

def test_ma_to_watts_basic():
    # 1500 mA at 3.85 V = 5.775 W
    assert ma_to_watts(1500, 3.85) == 5.775


def test_ma_to_watts_uses_absolute_value_for_discharge():
    # Some OEMs report discharge current as negative.
    assert ma_to_watts(-1000, 4.0) == 4.0


def test_ma_to_watts_none_passthrough():
    assert ma_to_watts(None, 3.85) is None


# --- to_measured mapping -----------------------------------------------------

def test_to_measured_includes_only_present_fields():
    r = HardwareEvalResult(model_id="m", success=True, ttft_ms=100.0, tok_per_s=12.0)
    m = r.to_measured()
    assert m["ttft_ms"] == 100.0
    assert m["tok_per_s"] == 12.0
    assert "avg_watts" not in m          # None → omitted so the gate falls back
    assert "peak_memory_mb" not in m


def test_to_measured_full():
    r = HardwareEvalResult(model_id="m", success=True, ttft_ms=100.0, tok_per_s=12.0,
                           avg_watts=4.2, peak_memory_mb=900, device="phone")
    m = r.to_measured()
    assert m == {"ttft_ms": 100.0, "tok_per_s": 12.0, "avg_watts": 4.2,
                 "peak_memory_mb": 900, "device": "phone"}


# --- logcat parsing ----------------------------------------------------------

def test_parse_run_lines_matches_run_id_and_tags():
    logcat = (
        "01-01 TTFT: run_id=run_1_123 value=250\n"
        "01-01 TPS: run_id=run_1_123 value=11.5\n"
        "01-01 TTFT: run_id=run_2_999 value=800\n"   # different run — ignored
        "01-01 RUN_DONE: run_id=run_1_123 response=Paris is the capital\n"
    )
    lines = parse_run_lines(logcat, "run_1_123")
    assert set(lines.keys()) == {"TTFT", "TPS", "RUN_DONE"}
    assert "value=250" in lines["TTFT"]
    assert "run_2_999" not in lines["TTFT"]


def test_build_metrics_casts_types_and_tolerates_missing():
    lines = {
        "TTFT": "x run_id=r value=250",
        "TPS": "x run_id=r value=11.5",
        "MEMORY": "x run_id=r value=880000",
        "THERMAL": "x run_id=r value=NOMINAL",
        # POWER + COLD_LOAD absent
    }
    m = _build_metrics(lines)
    assert m["ttft_ms"] == 250 and isinstance(m["ttft_ms"], int)
    assert m["tps"] == 11.5
    assert m["memory_kb"] == 880000
    assert m["thermal"] == "NOMINAL"
    assert m["power_ma"] is None
    assert m["cold_load_ms"] is None


# --- summarization -----------------------------------------------------------

def _q(n, ttft, tps, mem_kb, power_ma, cold=None, thermal="NOMINAL"):
    return {"question_number": n, "status": "success",
            "metrics": {"ttft_ms": ttft, "tps": tps, "memory_kb": mem_kb,
                        "power_ma": power_ma, "cold_load_ms": cold, "thermal": thermal}}


def test_summarize_uses_peak_rss_and_mean_power():
    results = [
        _q(1, 200, 12.0, 800_000, 1000, cold=1500),
        _q(2, 300, 10.0, 900_000, 2000),           # higher RSS → peak
    ]
    summary = summarize_smolchat(results, voltage_v=4.0)
    meas = summary["measured"]
    # peak RSS = max(800k, 900k) / 1024 ≈ 878 MB
    assert meas["peak_memory_mb"] == int(900_000 / 1024)
    # mean power = 1500 mA @ 4.0 V = 6.0 W
    assert meas["avg_watts"] == 6.0
    assert meas["ttft_ms"] == 250.0            # mean of 200, 300
    assert summary["first_question_cold_load_ms"] == 1500


def test_summarize_ignores_failed_questions():
    results = [
        _q(1, 200, 12.0, 800_000, 1000),
        {"question_number": 2, "status": "failed", "metrics": None},
    ]
    summary = summarize_smolchat(results, voltage_v=3.85)
    assert summary["ttft_ms"]["mean"] == 200
    assert summary["measured"]["peak_memory_mb"] == int(800_000 / 1024)


def test_summarize_all_failed_yields_none_metrics():
    results = [{"question_number": 1, "status": "failed", "metrics": None}]
    summary = summarize_smolchat(results, voltage_v=3.85)
    assert summary["measured"]["ttft_ms"] is None
    assert summary["measured"]["avg_watts"] is None
    assert summary["measured"]["peak_memory_mb"] is None
