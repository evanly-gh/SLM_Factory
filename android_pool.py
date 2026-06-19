# android_pool.py
from dataclasses import dataclass

@dataclass
class ModelSpec:
    model_id: str          # HuggingFace model ID
    int4_size_mb: int      # theoretical INT4 size in MB
    tier: int              # 1, 2, or 3
    # Theoretical benchmarks on reference chips (tok/s)
    tok_s_snapdragon_660: float
    tok_s_snapdragon_778g: float
    tok_s_snapdragon_8gen3: float
    peak_memory_mb: int    # estimated peak RAM during inference
    notes: str = ""

@dataclass
class HardwareConstraints:
    storage_mb: int
    memory_mb: int
    latency_ttft_ms: int
    power_watts: float
    target_chip: str = "snapdragon_778g"

ANDROID_POOL: list[ModelSpec] = [
    # Tier 1 — fits anywhere (≤800MB)
    ModelSpec("Qwen/Qwen3-0.6B",         397,  1,  8.0, 15.0, 50.0, 500),
    ModelSpec("Qwen/Qwen3.5-0.8B",       500,  1,  7.0, 13.0, 45.0, 620),
    ModelSpec("openbmb/MiniCPM4-0.5B",   300,  1, 10.0, 18.0, 60.0, 380),
    ModelSpec("openbmb/MiniCPM5-1B",     500,  1,  8.0, 16.0, 55.0, 620),
    ModelSpec("google/gemma-3-1b-it",    529,  1,  9.0, 17.0, 70.0, 660),
    ModelSpec("meta-llama/Llama-3.2-1B", 808,  1,  6.0, 12.0, 40.0, 980),
    # Tier 2 — most Android devices (800MB–1.5GB)
    ModelSpec("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1060, 2, 4.0, 8.0, 30.0, 1280),
    ModelSpec("Qwen/Qwen3-1.7B",                      1190, 2, 4.0, 8.0, 30.0, 1430),
    ModelSpec("Qwen/Qwen3.5-2B",                      1200, 2, 3.5, 7.0, 28.0, 1440),
    ModelSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1290, 2, 3.0, 6.0, 25.0, 1550),
    # Tier 3 — generous upper bound
    ModelSpec("meta-llama/Llama-3.2-3B", 1400, 3, 2.0, 4.0, 15.0, 1700),
    ModelSpec("mistralai/Ministral-3B-Instruct", 1900, 3, 1.5, 3.0, 12.0, 2200),
]

def filter_pool(constraints: HardwareConstraints) -> list[ModelSpec]:
    """Return models that fit within storage and memory constraints, sorted tier asc."""
    feasible = [
        m for m in ANDROID_POOL
        if m.int4_size_mb <= constraints.storage_mb
        and m.peak_memory_mb <= constraints.memory_mb
    ]
    return sorted(feasible, key=lambda m: (m.tier, m.int4_size_mb))
