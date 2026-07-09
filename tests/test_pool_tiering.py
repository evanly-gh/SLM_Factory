from config.android_pool import ANDROID_POOL


def test_minicpm4_is_tier0():
    m = next(m for m in ANDROID_POOL if "MiniCPM4-0.5B" in m.model_id and m.quant is None)
    assert m.tier == 0, f"MiniCPM4-0.5B base should be tier 0, got {m.tier}"


def test_llama_1b_is_tier1():
    m = next(m for m in ANDROID_POOL if "Llama-3.2-1B" in m.model_id and m.quant is None)
    assert m.tier == 1, f"Llama-3.2-1B base should be tier 1, got {m.tier}"


def test_gemma3_1b_is_tier2():
    m = next(m for m in ANDROID_POOL if "gemma-3-1b-it" in m.model_id and m.quant is None)
    assert m.tier == 2, f"gemma-3-1b-it base should be tier 2, got {m.tier}"


def test_llama_3b_is_tier3():
    m = next(m for m in ANDROID_POOL if "Llama-3.2-3B" in m.model_id and m.quant is None)
    assert m.tier == 3, f"Llama-3.2-3B base should be tier 3, got {m.tier}"


def test_siblings_inherit_base_tier():
    """Q4 and Q8 siblings of the same base model must have the same tier."""
    base_tiers = {
        m.model_id: m.tier
        for m in ANDROID_POOL if m.quant is None
    }
    for m in ANDROID_POOL:
        if m.quant is not None:
            assert m.tier == base_tiers[m.model_id], (
                f"{m.model_id} quant={m.quant} tier={m.tier} "
                f"but base tier={base_tiers[m.model_id]}"
            )


def test_no_tier_gaps():
    """All four tiers 0-3 must be present."""
    tiers = {m.tier for m in ANDROID_POOL}
    assert tiers == {0, 1, 2, 3}
