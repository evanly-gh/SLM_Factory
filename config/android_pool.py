# android_pool.py
#
# MODEL POOL DESIGN — QWEN-ONLY BRANCH
#
# This branch restricts the model pool to the Qwen architecture family for
# controlled ablation of model selection strategies. All models share the same
# tokenizer lineage and MNN-LLM runtime compatibility.
#
# Tier structure rationale:
#   Tiers are pure PEAK-RAM buckets of each deployed variant, NOT param-count bands.
#   The loop selects the smallest-RAM variant that still hits the accuracy goal, so the
#   axis that matters is on-device memory, and a single model's BF16/Q8/Q4 variants can
#   land in different tiers.
#
#     Tier 0  peak < 750 MB
#     Tier 1  peak 750–1500 MB
#     Tier 2  peak 1500–2500 MB
#     Tier 3  peak >= 2500 MB
#
#   Each base entry below stores the MEASURED Q4_K_M size + peak RAM. _variant() expands
#   it into three real deployment candidates (BF16 / Q8_0 / Q4_K_M) with honest per-variant
#   size, peak, tier, and decode speed. Weight bytes/param: BF16 2.0, Q8_0 1.0, Q4_K_M 0.55.
#
# Pool members (3 base models × 3 quant variants = 9 entries):
#   Qwen3.5-0.8B                      — Gated DeltaNet, 262K ctx, multimodal
#   DeepSeek-R1-Distill-Qwen-1.5B     — Qwen arch, R1 distilled, math/reasoning
#   Qwen3.5-2B (unsloth GGUF)         — Gated DeltaNet, 262K ctx, multimodal
#
# Android framework compatibility:
#   llama.cpp (GGUF): all models — broadest format support, CPU+Vulkan backends
#   MNN-LLM:    Qwen family — fastest CPU prefill (8.6x vs llama.cpp)
#   ExecuTorch: Qwen3 all sizes — SpinQuant INT4, KleidiAI acceleration
#
# Benchmark sources:
#   Qwen3 scores: arXiv 2505.09388 (Qwen3 Technical Report, Table 8)
#   DeepSeek-R1-Distill: arXiv 2501.12948 (DeepSeek-R1 paper, Table 4)
#   INT4 sizes: HuggingFace GGUF repos (unsloth)

from dataclasses import dataclass, field


# Throughput scaling factors for estimating decode speed on unlisted chipsets.
# These are rough per-generation multipliers (CPU-bound, Q4_K_M, 1B model).
# Used by check_hardware_constraints when target_chip is not in tok_s_by_chip.
#
# Derivation (arXiv 2410.03613 + Qualcomm AI Hub):
#   Each Snapdragon generation ≈ +50% prefill, +110% decode vs. prior gen.
#   Dimensity 9300 ≈ SD 8 Gen 3 on CPU.  8 Elite ≈ 1.4× Gen 3 on NPU path.
CHIP_SCALE_FACTORS: dict[str, float] = {
    # Format: chip_name → relative decode throughput vs. snapdragon_778g (= 1.0)
    "snapdragon_660":    0.55,   # 2017, Cortex-A73, no dotprod — very slow
    "snapdragon_730":    0.70,
    "snapdragon_750g":   0.85,
    "snapdragon_778g":   1.00,   # reference baseline
    "snapdragon_870":    1.10,
    "snapdragon_888":    1.30,
    "snapdragon_8gen1":  1.50,
    "snapdragon_8gen2":  1.80,
    "snapdragon_8gen3":  2.20,   # ~20-25 tok/s on 1B Q4_K_M (CPU)
    "snapdragon_8elite": 3.00,   # ~30-35 tok/s on 1B (CPU+NPU mixed)
    "dimensity_9300":    2.10,   # all-big-core; similar to 8 Gen 3 on CPU
    "dimensity_9400":    2.50,
    "exynos_2400":       1.80,
    "exynos_2500":       2.30,
    "tensor_g3":         1.60,   # Pixel 8 Pro; Mali GPU is slower for LLM
    "tensor_g4":         1.80,
}


@dataclass
class ModelSpec:
    model_id: str          # HuggingFace model ID
    size_mb: int           # on-disk weight size in MB for THIS variant's quant (see `quant`)
    tier: int              # RAM bucket of peak_memory_mb: 0=<0.75GB, 1=0.75-1.5GB, 2=1.5-2.5GB, 3=>=2.5GB
    # Estimated decode throughput on three well-documented reference chips (tok/s, CPU-bound Q4_K_M).
    # For any other chip, check_hardware_constraints uses CHIP_SCALE_FACTORS to interpolate.
    tok_s_snapdragon_660: float   # 2017 entry-level baseline
    tok_s_snapdragon_778g: float  # 2021 mid-range baseline (most common eval chip)
    tok_s_snapdragon_8gen3: float # 2023 flagship baseline
    peak_memory_mb: int    # estimated peak RAM during inference (file + KV cache + runtime)
    # Benchmark scores for tier-selection logic (base model unless noted)
    gsm8k: float           # GSM8K 5-shot CoT accuracy (0–1 scale)
    mmlu: float            # MMLU 5-shot accuracy (0–1 scale)
    notes: str = ""
    quant: str | None = None   # None = base (BF16/FP16). "Q4_K_M" or "Q8_0" = GGUF variant.

    def est_params_b(self) -> float:
        """Estimate parameter count (billions) from this variant's weight size.
        Uses the quant's bytes/param so all three variants of one model return the
        same param count (BF16 2.0, Q8_0 1.0, Q4_K_M 0.55 GB/1B)."""
        bpp = 0.55 if self.quant == "Q4_K_M" else 1.0 if self.quant == "Q8_0" else 2.0
        return self.size_mb / 1000 / bpp

    def tok_s_for_chip(self, chip: str) -> float:
        """Return estimated decode tok/s for any chip, not just the three hardcoded ones."""
        direct = {
            "snapdragon_660": self.tok_s_snapdragon_660,
            "snapdragon_778g": self.tok_s_snapdragon_778g,
            "snapdragon_8gen3": self.tok_s_snapdragon_8gen3,
        }
        if chip in direct:
            return direct[chip]
        # Interpolate via scale factors anchored to the 778G baseline
        if chip in CHIP_SCALE_FACTORS:
            factor = CHIP_SCALE_FACTORS[chip] / CHIP_SCALE_FACTORS["snapdragon_778g"]
            return self.tok_s_snapdragon_778g * factor
        # Unknown chip — return 778G as a conservative fallback
        return self.tok_s_snapdragon_778g


@dataclass
class HardwareConstraints:
    storage_mb: int
    memory_mb: int
    # TTFT (time-to-first-token) and decode throughput are separate concerns:
    #   latency_ttft_ms — how long until the first token appears (prompt processing time).
    #                     Dominated by model size and prefill speed.
    #   min_tok_s       — sustained decode throughput floor (hardware_metrics.md §6.5):
    #                     ≥ 6 tok/s — average English reading speed (~250 wpm ≈ 6 tok/s).
    #                     Below this, streaming output visibly lags behind reading pace.
    #                     Above it, users cannot perceptually distinguish 10 from 40 tok/s.
    #                     This threshold is CONSTANT regardless of model or task —
    #                     it's a UX floor, not a model property.
    latency_ttft_ms: int
    power_watts: float = 6.0
    target_chip: str = "snapdragon_778g"
    # Sustained decode throughput floor (hardware_metrics.md §6.5).
    # ≥ 6 tok/s — average English reading speed (~250 wpm ≈ 6 tok/s).
    # Below this, streaming output visibly lags behind reading pace.
    # Defaults to 0 (disabled) so existing callers are unaffected.
    # Set to 6.0 when the use case requires interactive streaming responses.
    # Set to 0 for batch/async workflows where latency doesn't matter.
    min_tok_s: float = 0.0


# ---------------------------------------------------------------------------
# QWEN-ONLY POOL TIERS (by peak RAM, not param count)
#
# Tier 0 — peak < 750 MB  (Qwen3.5-0.8B Q4_K_M lands here)
# Tier 1 — 750–1500 MB    (Qwen3.5-0.8B Q8_0/BF16, R1-Distill Q4_K_M)
# Tier 2 — 1500–2500 MB   (R1-Distill Q8_0/BF16, Qwen3.5-2B Q4_K_M/Q8_0)
# Tier 3 — >= 2500 MB     (Qwen3.5-2B BF16)
#
# ---------------------------------------------------------------------------
# Variant expansion: 3 base models × 3 quant variants = 9 entries.
#
# Sizing model (bytes/param, fact-checked against 2026 GGUF measurements):
#     BF16  = 2.00 GB/1B params   (Q4 × 3.64)
#     Q8_0  = 1.00 GB/1B params   (Q4 × 1.82)   near-lossless
#     Q4_K_M= 0.55 GB/1B params   (× 1.00, the measured anchor)
# Only the weight-resident bytes scale with quant; the KV-cache + runtime overhead
# (peak − weights) is quant-independent (KV cache stays FP16), so it is held
# constant per model and added back on top of each variant's weight size.
#
# Decode speed: smaller weights → less memory bandwidth per token → faster decode.
#     Q4_K_M = measured (anchor, ×1.00) · Q8_0 ≈ ×0.65 · BF16 ≈ ×0.45
# ---------------------------------------------------------------------------

_BF16_OVER_Q4 = 2.00 / 0.55   # ≈ 3.636
_Q8_OVER_Q4 = 1.00 / 0.55     # ≈ 1.818
_SPEED_FACTOR = {"Q4_K_M": 1.00, "Q8_0": 0.65, None: 0.45}  # None == BF16


def _ram_tier(peak_mb: int) -> int:
    """Tier is a pure RAM bucket of peak inference memory."""
    if peak_mb < 750:
        return 0
    if peak_mb < 1500:
        return 1
    if peak_mb < 2500:
        return 2
    return 3


def _variant(base: ModelSpec, quant: str | None) -> ModelSpec:
    """Build one deployment variant (quant=None→BF16, "Q8_0", or "Q4_K_M") from a
    base entry whose size_mb/peak_memory_mb hold the MEASURED Q4_K_M values."""
    q4_size = base.size_mb
    overhead = max(base.peak_memory_mb - q4_size, 0)  # KV cache + runtime, quant-independent
    if quant == "Q4_K_M":
        size = q4_size
    elif quant == "Q8_0":
        size = round(q4_size * _Q8_OVER_Q4)
    else:  # BF16 base
        size = round(q4_size * _BF16_OVER_Q4)
    peak = size + overhead
    speed = _SPEED_FACTOR[quant]
    return ModelSpec(
        model_id=base.model_id,
        size_mb=size,
        tier=_ram_tier(peak),
        tok_s_snapdragon_660=round(base.tok_s_snapdragon_660 * speed, 1),
        tok_s_snapdragon_778g=round(base.tok_s_snapdragon_778g * speed, 1),
        tok_s_snapdragon_8gen3=round(base.tok_s_snapdragon_8gen3 * speed, 1),
        peak_memory_mb=peak,
        gsm8k=base.gsm8k,
        mmlu=base.mmlu,
        notes=base.notes,
        quant=quant,
    )


ANDROID_POOL: list[ModelSpec] = [
    # NOTE: each entry below is a MEASURED-Q4_K_M SEED. Its size_mb / peak_memory_mb /
    # tok_s are the real Q4_K_M values, and its `tier=` field is IGNORED — _variant()
    # recomputes tier per variant from peak RAM via _ram_tier(). The "~params" in the
    # section headers just groups seeds by model scale for readability.
    #
    # QWEN-ONLY POOL: This branch restricts the pool to the Qwen model family only.
    # All Qwen models share the same tokenizer lineage, structured-output strength,
    # and MNN-LLM runtime compatibility, making them ideal for controlled ablation
    # studies of model selection strategies.

    # ── ~0.75B params ─────────────────────────────────────────────────────
    # Qwen3.5-0.8B (March 2026): Gated DeltaNet hybrid architecture, natively multimodal,
    # 262K context, Apache 2.0. Intelligence Index +2.5 pts over Qwen3-0.6B.
    # CAUTION: documented 67%→33% code generation collapse when few-shot examples are added.
    # INT4 size estimated ~500MB; source: huggingface.co/Qwen/Qwen3.5-0.8B
    ModelSpec(
        model_id="Qwen/Qwen3.5-0.8B",
        size_mb=500,
        tier=1,
        tok_s_snapdragon_660=9.0,
        tok_s_snapdragon_778g=15.0,
        tok_s_snapdragon_8gen3=42.0,
        peak_memory_mb=670,
        gsm8k=0.610,
        mmlu=0.540,
        notes="Gated DeltaNet hybrid, multimodal, 262K ctx, 201 langs; avoid few-shot code generation; replaces Qwen3-0.6B",
    ),

    # ── ~1.5B params ──────────────────────────────────────────────────────
    # DeepSeek-R1-Distill-Qwen-1.5B: Qwen-architecture model distilled from R1 671B.
    # Specialized reasoning: MATH-500 83.9%, AIME 2024 28.9%.
    # Source: arXiv 2501.12948 Table 4.
    ModelSpec(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        size_mb=958,
        tier=2,
        tok_s_snapdragon_660=4.5,
        tok_s_snapdragon_778g=8.0,
        tok_s_snapdragon_8gen3=22.0,
        peak_memory_mb=1270,
        gsm8k=0.870,
        mmlu=0.580,
        notes="Qwen arch, R1 distilled; MATH-500 83.9%, AIME 28.9%; best for math/reasoning tasks",
    ),

    # ── ~2B params ────────────────────────────────────────────────────────
    # Qwen3.5-2B (March 2026): Gated DeltaNet hybrid architecture, multimodal, 262K ctx.
    # Same family as Qwen3.5-0.8B but at 2B params. Unsloth Dynamic 2.0 GGUF confirmed
    # working with llama.cpp, Ollama.
    # Source: unsloth/Qwen3.5-2B-GGUF on HuggingFace; unsloth.ai/docs/models/qwen3.5.
    ModelSpec(
        model_id="unsloth/Qwen3.5-2B-GGUF",
        size_mb=1350,
        tier=3,
        tok_s_snapdragon_660=4.0,
        tok_s_snapdragon_778g=7.5,
        tok_s_snapdragon_8gen3=20.0,
        peak_memory_mb=1800,
        gsm8k=0.720,
        mmlu=0.610,
        notes="Gated DeltaNet hybrid, multimodal (text+image+video), 262K ctx, 201 langs; Unsloth GGUF confirmed for llama.cpp",
    ),
]

# Expand each base entry (which holds MEASURED Q4_K_M numbers) into three real,
# independently-selectable deployment variants: BF16 (quant=None), Q8_0, Q4_K_M.
# Each variant gets its own honest size/peak/tier/speed. Tier is a pure RAM bucket,
# so a single model's variants can span multiple tiers.
_BASE_MODELS = [m for m in ANDROID_POOL]  # snapshot: these carry measured Q4_K_M values
ANDROID_POOL = sorted(
    [_variant(m, None) for m in _BASE_MODELS]       # BF16
    + [_variant(m, "Q8_0") for m in _BASE_MODELS]
    + [_variant(m, "Q4_K_M") for m in _BASE_MODELS],
    key=lambda m: (m.tier, m.size_mb),
)


def filter_pool(constraints: HardwareConstraints) -> list[ModelSpec]:
    """Return variants that satisfy all hard constraints, sorted by tier then size.

    Each entry is an independent quant variant (BF16 / Q8_0 / Q4_K_M).
    Filters on:
      - storage_mb: this variant's on-disk weight file must fit
      - memory_mb: this variant's peak RAM must fit
      - min_tok_s: sustained decode throughput floor (UX gate, hardware_metrics.md §6.5)
                   Set constraints.min_tok_s=0 to disable (batch/async workflows).
    """
    feasible = [
        m for m in ANDROID_POOL
        if m.size_mb <= constraints.storage_mb
        and m.peak_memory_mb <= constraints.memory_mb
        and (constraints.min_tok_s <= 0 or m.tok_s_for_chip(constraints.target_chip) >= constraints.min_tok_s)
    ]
    return sorted(feasible, key=lambda m: (m.tier, m.size_mb))


def filter_pool_by_task(
    constraints: HardwareConstraints,
    task_type: str | None = None,
) -> list[ModelSpec]:
    """
    Return feasible models, optionally pre-sorted for a specific task type.

    With the Qwen-only pool, all models share the same architecture family.
    Sorting still applies benchmark-based ranking within the pool:
      "math" / "reasoning" — sort by GSM8K within tier
      "classification"     — sort by MMLU within tier
      "ner" / "code" / "multilingual" — sort by GSM8K within tier
      None                 — default sort by tier then size
    """
    feasible = filter_pool(constraints)

    if task_type in ("math", "reasoning"):
        return sorted(feasible, key=lambda m: (m.tier, -m.gsm8k))
    if task_type == "classification":
        return sorted(feasible, key=lambda m: (m.tier, -m.mmlu))
    if task_type in ("ner", "multilingual", "code"):
        return sorted(feasible, key=lambda m: (m.tier, -m.gsm8k))

    return feasible


def all_constraints_pass(hw_check: dict) -> bool:
    """Return True if all four hardware constraints pass."""
    return all(hw_check[k]["pass"] for k in ("storage", "memory", "latency", "power"))


def check_hardware_constraints(
    model: ModelSpec,
    constraints: HardwareConstraints,
    measured: dict | None = None,
) -> dict:
    """
    Check all four hardware constraints and return per-constraint PASS/FAIL status.

    Phase 1: latency and power use theoretical estimates and are logged but not gating.
    Phase 2: all four become hard gates. When `measured` is provided (a HardwareEvalResult
    dict), use measured values instead of theoretical estimates. (Design doc §6.2, §6.5)
    """
    chip = constraints.target_chip
    tok_s = model.tok_s_for_chip(chip)
    # TTFT estimate: time for the model to produce the first token (prefill-dominated).
    # Approximated as 1/tok_s * 1000 ms — rough, but order-of-magnitude correct for
    # short prompts on CPU-bound inference. Measured values override this.
    estimated_ttft_ms = (1.0 / max(tok_s, 0.1)) * 1000

    # Sustained throughput check: constant UX floor from hardware_metrics.md §6.5.
    # 6 tok/s = average English reading speed. Below this streaming visibly lags.
    # min_tok_s = 0 disables the check (e.g. for batch/async workflows).
    throughput_pass = (constraints.min_tok_s <= 0) or (tok_s >= constraints.min_tok_s)

    result = {
        "storage": {
            "value_mb": model.size_mb,
            "limit_mb": constraints.storage_mb,
            "pass": model.size_mb <= constraints.storage_mb,
        },
        "memory": {
            # Prefer measured peak RSS from a real device run; fall back to the
            # ModelSpec's theoretical peak_memory_mb when unmeasured.
            "value_mb": (measured["peak_memory_mb"]
                         if measured and measured.get("peak_memory_mb") is not None
                         else model.peak_memory_mb),
            "limit_mb": constraints.memory_mb,
            "measured": bool(measured and measured.get("peak_memory_mb") is not None),
            "pass": ((measured["peak_memory_mb"]
                      if measured and measured.get("peak_memory_mb") is not None
                      else model.peak_memory_mb) <= constraints.memory_mb),
        },
        "latency": {
            # TTFT sub-check: prompt processing latency
            "estimated_ttft_ms": round(estimated_ttft_ms),
            "limit_ms": constraints.latency_ttft_ms,
            "ttft_pass": estimated_ttft_ms <= constraints.latency_ttft_ms,
            # Throughput sub-check: sustained decode speed (UX floor, hardware_metrics.md §6.5)
            "tok_s": tok_s,
            "min_tok_s": constraints.min_tok_s,
            "throughput_pass": throughput_pass,
            "chip": chip,
            # latency gate passes only if both sub-checks pass
            "pass": estimated_ttft_ms <= constraints.latency_ttft_ms and throughput_pass,
        },
        "power": {
            "note": "Phase 1: proxy from model size, not measured" if not measured else "measured",
            "value_watts": measured.get("avg_watts", 0) if measured else 0,
            "limit_watts": constraints.power_watts,
            "pass": (measured["avg_watts"] <= constraints.power_watts) if measured and measured.get("avg_watts") is not None else True,
        },
    }

    if measured and measured.get("ttft_ms"):
        measured_tok_s = measured.get("tok_per_s", 0)
        measured_throughput_pass = (constraints.min_tok_s <= 0) or (measured_tok_s >= constraints.min_tok_s)
        result["latency"] = {
            "measured_ttft_ms": measured["ttft_ms"],
            "measured_tok_s": measured_tok_s,
            "limit_ms": constraints.latency_ttft_ms,
            "min_tok_s": constraints.min_tok_s,
            "ttft_pass": measured["ttft_ms"] <= constraints.latency_ttft_ms,
            "throughput_pass": measured_throughput_pass,
            "chip": measured.get("device", chip),
            "pass": measured["ttft_ms"] <= constraints.latency_ttft_ms and measured_throughput_pass,
        }

    return result
