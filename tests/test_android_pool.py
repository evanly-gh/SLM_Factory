from android_pool import filter_pool, ModelSpec, HardwareConstraints

def test_filter_pool_by_storage():
    constraints = HardwareConstraints(storage_mb=500, memory_mb=1200,
                                       latency_ttft_ms=2000, power_watts=5.0)
    result = filter_pool(constraints)
    assert all(m.int4_size_mb <= 500 for m in result)
    assert len(result) > 0

def test_filter_pool_returns_tier1_first():
    constraints = HardwareConstraints(storage_mb=1600, memory_mb=2000,
                                       latency_ttft_ms=2000, power_watts=5.0)
    result = filter_pool(constraints)
    assert result[0].tier == 1

def test_filter_pool_excludes_oversized():
    constraints = HardwareConstraints(storage_mb=100, memory_mb=200,
                                       latency_ttft_ms=2000, power_watts=5.0)
    result = filter_pool(constraints)
    assert len(result) == 0
