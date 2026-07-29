from config.android_pool import ANDROID_POOL, _size_tier


def test_tier_matches_size_bucket():
    """Tier buckets the variant's REAL on-disk weight size.

    It used to bucket a modelled peak_memory_mb (size estimate + a fixed overhead
    constant). Weight size is verifiable; the peak-RAM figure was not.
    """
    for m in ANDROID_POOL:
        assert m.tier == _size_tier(m.size_mb), (
            f"{m.model_id} quant={m.quant} size={m.size_mb} "
            f"tier={m.tier} but size bucket says {_size_tier(m.size_mb)}"
        )


def test_size_tier_boundaries():
    assert _size_tier(500) == 0
    assert _size_tier(749) == 0
    assert _size_tier(750) == 1
    assert _size_tier(1499) == 1
    assert _size_tier(1500) == 2
    assert _size_tier(2499) == 2
    assert _size_tier(2500) == 3
    assert _size_tier(9000) == 3


def test_three_variants_per_model():
    """Every base model expands to exactly three quant variants."""
    from collections import Counter
    by_id = Counter(m.model_id for m in ANDROID_POOL)
    for model_id, count in by_id.items():
        assert count == 3, f"{model_id} has {count} variants, expected 3 (BF16/Q8/Q4)"
    quants = {m.quant for m in ANDROID_POOL}
    assert quants == {None, "Q8_0", "Q4_K_M"}


def test_every_variant_has_a_unique_stable_selector():
    selectors = {model.selector for model in ANDROID_POOL}

    assert len(selectors) == len(ANDROID_POOL)
    for model in ANDROID_POOL:
        assert model.selector == f"{model.model_id}@{model.quant or 'bf16'}"


def test_variants_ordered_by_size():
    """Within one model: Q4_K_M < Q8_0 < BF16 on real on-disk weight size.

    The peak-RAM half of this assertion was dropped along with the modelled
    peak_memory_mb field — ordering by a fabricated number proved nothing. Size ordering
    is the real invariant and holds for both arithmetic and measured sizes.
    """
    ids = {m.model_id for m in ANDROID_POOL}
    for model_id in ids:
        vs = {m.quant: m for m in ANDROID_POOL if m.model_id == model_id}
        assert vs["Q4_K_M"].size_mb < vs["Q8_0"].size_mb < vs[None].size_mb, model_id


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
        assert len({m.knowledge_metric for m in vs}) == 1, model_id
        assert len({m.knowledge_score for m in vs}) == 1, model_id
        assert len({m.mmlu_redux for m in vs}) == 1, model_id
