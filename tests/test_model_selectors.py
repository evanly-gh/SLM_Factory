import config.android_pool as android_pool


ANDROID_POOL = android_pool.ANDROID_POOL


def _resolve(models, value):
    resolver = getattr(android_pool, "resolve_model_selector", None)
    assert callable(resolver), "resolve_model_selector is not implemented"
    return resolver(models, value)


def _siblings(model_id: str):
    return [model for model in ANDROID_POOL if model.model_id == model_id]


def test_every_quant_sibling_is_selectable_by_exact_selector():
    siblings = _siblings("Qwen/Qwen3-1.7B")

    assert {model.quant for model in siblings} == {None, "Q8_0", "Q4_K_M"}
    for expected in siblings:
        assert _resolve(siblings, expected.selector) is expected


def test_bare_model_id_is_allowed_when_unambiguous():
    candidate = _siblings("Qwen/Qwen3-1.7B")[0]

    assert _resolve([candidate], candidate.model_id) is candidate


def test_ambiguous_bare_model_id_uses_documented_lowest_resource_default():
    siblings = _siblings("Qwen/Qwen3-1.7B")

    selected = _resolve(siblings, "Qwen/Qwen3-1.7B")

    assert selected.quant == "Q4_K_M"
    # Tie-break is now on REAL on-disk size; peak_memory_mb was a modelled figure and
    # has been removed pool-wide.
    assert selected.size_mb == min(model.size_mb for model in siblings)


def test_unknown_selector_returns_none():
    assert _resolve(ANDROID_POOL, "Qwen/does-not-exist@Q4_K_M") is None
