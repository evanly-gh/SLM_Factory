"""The pool must never fabricate throughput or peak RAM.

It used to: per-model tok/s for three reference chips, a CHIP_SCALE_FACTORS table that
interpolated a decode rate for any other chipset, and a peak_memory_mb derived by adding
a fixed overhead to a bytes-per-parameter size estimate. All three were rendered in the
same shape as measurements, so hardware gating and the model-selection prompt consumed
them as facts. The single modelled number ever checked against reality was 20% off
(Qwen3.5-4B Q4_K_M: modelled 2200 MB, actual GGUF 2654.5 MB).

These tests exist so that contract cannot silently regress.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")

import config.android_pool as pool  # noqa: E402
from config.android_pool import (  # noqa: E402
    ANDROID_POOL,
    HardwareConstraints,
    check_hardware_constraints,
    filter_pool,
)


def _constraints(**kw):
    base = dict(
        storage_mb=200_000,
        memory_mb=10_240,
        latency_ttft_ms=2_000,
        target_chip="snapdragon_8gen3",
    )
    base.update(kw)
    return HardwareConstraints(**base)


def test_modelspec_has_no_modelled_throughput_or_peak_ram_fields():
    spec = ANDROID_POOL[0]
    for banned in (
        "tok_s_snapdragon_660",
        "tok_s_snapdragon_778g",
        "tok_s_snapdragon_8gen3",
        "tok_s_for_chip",
        "peak_memory_mb",
    ):
        assert not hasattr(spec, banned), (
            f"ModelSpec.{banned} is back — that is a modelled metric, not a measurement"
        )


def test_chip_scale_factors_table_is_gone():
    assert not hasattr(pool, "CHIP_SCALE_FACTORS"), (
        "CHIP_SCALE_FACTORS invented a decode multiplier per chipset; it must stay deleted"
    )
    # The chip NAMES are fine to keep — a vocabulary carries no performance claim.
    assert "snapdragon_8gen3" in pool.KNOWN_CHIPS


def test_unmeasured_runtime_metrics_report_unmeasured_and_do_not_gate():
    spec = next(m for m in ANDROID_POOL if m.selector == "Qwen/Qwen3.5-4B@Q4_K_M")
    result = check_hardware_constraints(spec, _constraints(min_tok_s=6.0))

    assert result["latency"]["measured"] is False
    assert result["latency"]["measured_tok_s"] is None
    assert result["latency"]["measured_ttft_ms"] is None
    # A 6 tok/s floor must NOT eliminate a candidate we have never measured.
    assert result["latency"]["pass"] is True
    assert result["power"]["measured"] is False
    assert result["power"]["value_watts"] is None
    assert result["power"]["pass"] is True


def test_memory_gate_uses_real_weight_size_as_a_lower_bound():
    spec = next(m for m in ANDROID_POOL if m.selector == "Qwen/Qwen3.5-4B@bf16")
    memory = check_hardware_constraints(spec, _constraints(memory_mb=10_240))["memory"]
    assert memory["measured"] is False
    assert memory["value_mb"] is None, "unmeasured peak RAM must not be filled in"
    assert memory["weight_floor_mb"] == spec.size_mb

    # Weights alone exceeding RAM is a physical impossibility, so it must still fail.
    too_small = check_hardware_constraints(spec, _constraints(memory_mb=1_000))["memory"]
    assert too_small["pass"] is False


def test_recorded_measurement_is_used_and_gates(monkeypatch):
    spec = next(m for m in ANDROID_POOL if m.selector == "Qwen/Qwen3.5-4B@Q4_K_M")
    monkeypatch.setattr(
        pool,
        "measured_metrics_for",
        lambda model_id, quant, chip: (
            {"peak_memory_mb": 99_000, "tok_per_s": 1.5, "ttft_ms": 90_000}
            if (model_id, quant, chip)
            == (spec.model_id, spec.quant, "snapdragon_8gen3")
            else None
        ),
    )
    result = check_hardware_constraints(spec, _constraints(min_tok_s=6.0))
    assert result["memory"]["measured"] is True
    assert result["memory"]["value_mb"] == 99_000
    assert result["memory"]["pass"] is False, "measured peak over budget must fail"
    assert result["latency"]["measured"] is True
    assert result["latency"]["throughput_pass"] is False, "1.5 tok/s is below the 6 floor"


def test_measurements_never_cross_chips_or_quants():
    """A number from different silicon is not a measurement of this configuration."""
    spec = next(m for m in ANDROID_POOL if m.selector == "Qwen/Qwen3.5-4B@Q4_K_M")
    assert spec.measured("a_chip_never_measured") is None


def test_tier_buckets_real_weight_size():
    for spec in ANDROID_POOL:
        expected = pool._size_tier(spec.size_mb)
        assert spec.tier == expected, (
            f"{spec.selector} tier {spec.tier} does not match its size bucket {expected}"
        )


def test_filter_pool_does_not_eliminate_on_an_unmeasured_throughput_floor():
    with_floor = filter_pool(_constraints(min_tok_s=6.0))
    without_floor = filter_pool(_constraints(min_tok_s=0.0))
    assert [m.selector for m in with_floor] == [m.selector for m in without_floor]


def test_measured_sizes_override_bytes_per_parameter_arithmetic():
    """The real GGUF is the authority on its own size."""
    spec = next(m for m in ANDROID_POOL if m.selector == "Qwen/Qwen3.5-4B@Q4_K_M")
    recorded = pool.measured_size_mb("Qwen/Qwen3.5-4B", "Q4_K_M")
    if recorded is None:
        return  # nothing measured yet in this checkout
    assert spec.size_mb == recorded
    assert spec.size_mb != 2200, (
        "2200 MB was the bytes-per-parameter estimate; the measured file is larger"
    )
