# training/quantize.py
"""
INT4 quantization and theoretical hardware profile for Android pool models.
Phase 1: theoretical estimates only. Phase 2 replaces with measured values.
"""
from android_pool import ANDROID_POOL, ModelSpec

def theoretical_hardware_profile(model_id: str) -> dict:
    """
    Return theoretical hardware estimates for a model from the Android pool.
    Phase 1 uses benchmark-derived estimates, not measurements.
    """
    spec = next((m for m in ANDROID_POOL if m.model_id == model_id), None)
    if spec is None:
        return {
            "model_id": model_id,
            "int4_size_mb": None,
            "tier": None,
            "tok_s_snapdragon_778g": None,
            "peak_memory_mb": None,
            "phase": "theoretical",
            "note": "Model not in Android pool",
        }
    return {
        "model_id": spec.model_id,
        "int4_size_mb": spec.int4_size_mb,
        "tier": spec.tier,
        "tok_s_snapdragon_660": spec.tok_s_snapdragon_660,
        "tok_s_snapdragon_778g": spec.tok_s_snapdragon_778g,
        "tok_s_snapdragon_8gen3": spec.tok_s_snapdragon_8gen3,
        "peak_memory_mb": spec.peak_memory_mb,
        "phase": "theoretical",
    }
