import pytest
from config.android_pool import HardwareConstraints, ANDROID_POOL
from agent.nodes.cold_start.hardware_filter import run_hardware_filter


@pytest.fixture
def loose_constraints():
    # Large enough to admit every variant, including the biggest BF16 (Phi-4-mini
    # BF16 peaks ~10.7GB), so the "all pass" invariant holds.
    return HardwareConstraints(
        storage_mb=20000,
        memory_mb=20000,
        latency_ttft_ms=5000,
        power_watts=20.0,
        target_chip="snapdragon_778g",
        min_tok_s=0.0,
    )


@pytest.fixture
def tight_constraints():
    return HardwareConstraints(
        storage_mb=500,   # only sub-0.5B models fit
        memory_mb=800,
        latency_ttft_ms=1000,
        power_watts=5.0,
        target_chip="snapdragon_778g",
        min_tok_s=0.0,
    )


def test_returns_list(loose_constraints):
    result = run_hardware_filter(loose_constraints)
    assert isinstance(result, list)


def test_all_pass_loose_constraints(loose_constraints):
    result = run_hardware_filter(loose_constraints)
    assert len(result) == len(ANDROID_POOL)


def test_tight_constraints_filters(tight_constraints):
    result = run_hardware_filter(tight_constraints)
    for m in result:
        assert m.size_mb <= tight_constraints.storage_mb
        assert m.peak_memory_mb <= tight_constraints.memory_mb


def test_order_largest_first(loose_constraints):
    result = run_hardware_filter(loose_constraints)
    sizes = [m.size_mb for m in result]
    assert sizes == sorted(sizes, reverse=True)


def test_empty_when_nothing_fits():
    constraints = HardwareConstraints(
        storage_mb=1, memory_mb=1, latency_ttft_ms=1,
        power_watts=0.1, target_chip="snapdragon_778g",
    )
    result = run_hardware_filter(constraints)
    assert result == []
