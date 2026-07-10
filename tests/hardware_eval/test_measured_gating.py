"""check_hardware_constraints must prefer MEASURED values over theoretical ones
when a `measured` dict is supplied (post-convergence on-device verification path)."""
from config.android_pool import ModelSpec, HardwareConstraints, check_hardware_constraints


def _model():
    # Theoretical peak_memory_mb=1100 — comfortably under the 2000MB limit below.
    return ModelSpec(
        model_id="test/M", size_mb=700, tier=1,
        tok_s_snapdragon_660=8.0, tok_s_snapdragon_778g=14.0, tok_s_snapdragon_8gen3=38.0,
        peak_memory_mb=1100, gsm8k=0.6, mmlu=0.5, quant="Q4_K_M",
    )


def _constraints(memory_mb=2000):
    return HardwareConstraints(
        storage_mb=5000, memory_mb=memory_mb, latency_ttft_ms=3000,
        power_watts=5.0, target_chip="snapdragon_778g", min_tok_s=0.0,
    )


def test_measured_memory_overrides_theoretical_and_can_fail_gate():
    # Theoretical peak (1100) passes, but the device actually used 2600MB → must FAIL.
    check = check_hardware_constraints(
        _model(), _constraints(memory_mb=2000),
        measured={"peak_memory_mb": 2600},
    )
    assert check["memory"]["measured"] is True
    assert check["memory"]["value_mb"] == 2600
    assert check["memory"]["pass"] is False


def test_measured_memory_within_limit_passes():
    check = check_hardware_constraints(
        _model(), _constraints(memory_mb=2000),
        measured={"peak_memory_mb": 1200},
    )
    assert check["memory"]["measured"] is True
    assert check["memory"]["value_mb"] == 1200
    assert check["memory"]["pass"] is True


def test_falls_back_to_theoretical_when_unmeasured():
    check = check_hardware_constraints(_model(), _constraints(memory_mb=2000))
    assert check["memory"]["measured"] is False
    assert check["memory"]["value_mb"] == 1100     # ModelSpec theoretical peak
    assert check["memory"]["pass"] is True


def test_measured_power_gate():
    # 6W measured exceeds the 5W budget → power gate fails.
    check = check_hardware_constraints(
        _model(), _constraints(),
        measured={"avg_watts": 6.0},
    )
    assert check["power"]["pass"] is False
    # Under budget passes.
    check2 = check_hardware_constraints(
        _model(), _constraints(),
        measured={"avg_watts": 3.0},
    )
    assert check2["power"]["pass"] is True
