from config.android_pool import ANDROID_POOL, _ram_tier


def test_tier_matches_ram_bucket():
    """Tier is a pure RAM bucket of the variant's own peak_memory_mb."""
    for m in ANDROID_POOL:
        assert m.tier == _ram_tier(m.peak_memory_mb), (
            f"{m.model_id} quant={m.quant} peak={m.peak_memory_mb} "
            f"tier={m.tier} but RAM bucket says {_ram_tier(m.peak_memory_mb)}"
        )


def test_ram_tier_boundaries():
    assert _ram_tier(500) == 0
    assert _ram_tier(749) == 0
    assert _ram_tier(750) == 1
    assert _ram_tier(1499) == 1
    assert _ram_tier(1500) == 2
    assert _ram_tier(2499) == 2
    assert _ram_tier(2500) == 3
    assert _ram_tier(9000) == 3


def test_three_variants_per_model():
    """Every base model expands to exactly three quant variants."""
    from collections import Counter
    by_id = Counter(m.model_id for m in ANDROID_POOL)
    for model_id, count in by_id.items():
        assert count == 3, f"{model_id} has {count} variants, expected 3 (BF16/Q8/Q4)"
    quants = {m.quant for m in ANDROID_POOL}
    assert quants == {None, "Q8_0", "Q4_K_M"}


def test_variants_ordered_by_size():
    """Within one model: Q4_K_M < Q8_0 < BF16 on both disk size and peak RAM."""
    ids = {m.model_id for m in ANDROID_POOL}
    for model_id in ids:
        vs = {m.quant: m for m in ANDROID_POOL if m.model_id == model_id}
        assert vs["Q4_K_M"].size_mb < vs["Q8_0"].size_mb < vs[None].size_mb, model_id
        assert vs["Q4_K_M"].peak_memory_mb < vs["Q8_0"].peak_memory_mb < vs[None].peak_memory_mb, model_id


def test_smaller_quant_is_same_or_lower_tier():
    """A more-compressed variant never lands in a HIGHER RAM tier than a less-compressed one."""
    ids = {m.model_id for m in ANDROID_POOL}
    for model_id in ids:
        vs = {m.quant: m for m in ANDROID_POOL if m.model_id == model_id}
        assert vs["Q4_K_M"].tier <= vs["Q8_0"].tier <= vs[None].tier, model_id


def test_quant_variants_share_benchmarks():
    """Weight quant barely changes accuracy — variants share benchmark scores."""
    ids = {m.model_id for m in ANDROID_POOL}
    for model_id in ids:
        vs = [m for m in ANDROID_POOL if m.model_id == model_id]
        assert len({m.gsm8k for m in vs}) == 1, model_id
        assert len({m.mmlu for m in vs}) == 1, model_id
