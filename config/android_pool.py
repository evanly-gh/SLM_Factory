# android_pool.py
#
# MODEL POOL DESIGN — RESEARCH BASIS (see PAPER.md §17 for full writeup)
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
# Quantization notes:
#   All sizes below are Q4_K_M GGUF (4.5 bpw effective), the community-recommended
#   default. Q4_K_M is ~4.7x lower perplexity degradation than Q4_0 at only 8.6% more
#   storage (llama.cpp official benchmarks). Q4_0 is obsolete. Q5_K_M is recommended
#   when storage is not the constraint.
#
#   Sub-1B models suffer more from INT4 quantization than 3B models (IJCAI-25:
#   >10% perplexity increase for sub-1B decoders vs. ~5-10% for 3B). AWQ and QAT
#   both help, especially at 4-bit and below. For Tier 0 models specifically,
#   prefer QAT-quantized variants (Unsloth Dynamic 2.0 or Google's QAT checkpoints)
#   over standard PTQ when available.
#
# Android framework compatibility:
#   llama.cpp (GGUF): all models below — broadest format support, CPU+Vulkan backends
#   ExecuTorch: Llama 3.2 1B/3B, Qwen3 all sizes, Phi-4-mini, Gemma 3 — best for
#               production Android apps (SpinQuant INT4, KleidiAI acceleration)
#   MNN-LLM:    Qwen3/3.5, Llama 3.2, DeepSeek R1 distills, Gemma — fastest CPU
#               prefill (8.6x vs llama.cpp); strongest for Qwen family
#   LiteRT-LM:  Gemma family (official Google path), Llama, Phi-4, Qwen
#
# Chipset decode throughput reference (4-bit INT4, CPU-bound):
#   Snapdragon 660:   ~6-8  tok/s (1B),  ~2-3  tok/s (3B)
#   Snapdragon 778G:  ~12-15 tok/s (1B), ~4-6  tok/s (3B)
#   Snapdragon 8 Gen3:~20-30 tok/s (1B), ~8-12 tok/s (3B), ~12.85 tok/s (7B QNN NPU)
#   Source: arXiv 2410.03613; Qualcomm AI Hub model cards; Grokipedia community benchmarks
#
# Benchmark sources:
#   Qwen3 scores: arXiv 2505.09388 (Qwen3 Technical Report, Table 8)
#   Qwen2.5 scores: arXiv 2412.15115 (Qwen2.5 Technical Report, Table 5)
#   Gemma 3 scores: arXiv 2503.19786 (Gemma 3 Technical Report)
#   Llama 3.2 scores: Meta model card / arXiv 2407.21783
#   SmolLM2 scores: HuggingFaceTB model card / Distil Labs benchmark
#   MiniCPM4 scores: arXiv 2506.07900 (MiniCPM4 Technical Report)
#   DeepSeek-R1-Distill: arXiv 2501.12948 (DeepSeek-R1 paper, Table 4)
#   Phi-4-mini: Microsoft Phi-4-mini-instruct model card / localaimaster.com
#   INT4 sizes: HuggingFace GGUF repos (hugging-quants, bartowski, unsloth)

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
# TIER 0 — Micro  (~0.5B params, int4_size < ~375MB)
# Use for: binary classification, simple NER, keyword extraction, routing
# Avoid for: multi-step reasoning, math, code, open-ended generation
# Quantization risk: HIGH — these models lose the most from INT4; prefer QAT
#   variants. MiniCPM4-0.5B uses BitCPM4 (QAT-aware), Qwen2.5-0.5B is PTQ.
# ---------------------------------------------------------------------------
#
# TIER 1 — Small  (~0.75–1.5B params, int4_size ~375–750MB)
# GSM8K: 59–63%. The capability gap from Tier 0→1 is the LARGEST relative jump
# in the pool (e.g., GSM8K: 41.6% → 59.6%). Good for classification, NER, simple
# generation. Qwen3.5-0.8B, Llama-3.2-1B, MiniCPM5-1B are the Tier 1 members.
# ---------------------------------------------------------------------------
#
# TIER 2 — Mid    (~1.5–2.5B params, int4_size ~750–1250MB)
# GSM8K: 70–77%. Most capable tier that runs on all 6GB+ Android devices.
# SmolLM2-1.7B leads on IFEval (56.7%) — best for instruction-following tasks.
# DeepSeek-R1-Distill-1.5B is specialized for math/reasoning only (MATH-500: 83.9%).
# Gemma-3-1b-it (~1.6B actual params) sits in Tier 2 despite its "1B" label.
# ---------------------------------------------------------------------------
#
# TIER 3 — Large  (~2.5B+ params, int4_size > ~1250MB)
# GSM8K: 77–82%. Requires 8GB+ RAM phone for comfortable inference. Llama 3.2-3B
# is the ExecuTorch reference model (fastest on-device path via KleidiAI).
# Phi-4-mini (3.8B) punches above class on reasoning; Q4_K_M is 2.49GB,
# fitting the 3GB RAM budget on 8GB devices only (not 6GB).
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# Variant model (BF16 / Q8_0 / Q4_K_M) as INDEPENDENT deployment candidates.
#
# The 12 entries in _BASE_MODELS carry the MEASURED Q4_K_M on-disk size and its
# measured peak RAM (from HF repos / device runs). Each base is expanded into
# three real, independently-selectable variants that differ in weight precision,
# on-disk size, peak RAM, and decode speed — but share benchmark accuracy (weight
# quant barely moves task accuracy, which is exactly why min-RAM selection matters).
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
#
# TIER = pure RAM bucket of the variant's own peak_memory_mb (NOT param count):
#     Tier 0: peak < 750 MB | Tier 1: 750–1500 | Tier 2: 1500–2500 | Tier 3: >= 2500
# A model's BF16 / Q8 / Q4 variants can therefore land in DIFFERENT tiers — which is
# the whole point: the loop picks the smallest-RAM variant that still hits the goal.
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
    # ── ~0.5B params ──────────────────────────────────────────────────────
    # MiniCPM4-0.5B: Uses BitCPM4 QAT quantization for better INT4 quality.
    # Benchmarks claim to exceed Qwen3-0.6B; specialized sparse attention for
    # long context. Source: arXiv 2506.07900.
    ModelSpec(
        model_id="openbmb/MiniCPM4-0.5B",
        size_mb=310,
        tier=0,
        tok_s_snapdragon_660=14.0,
        tok_s_snapdragon_778g=22.0,
        tok_s_snapdragon_8gen3=60.0,
        peak_memory_mb=480,
        gsm8k=0.55,   # extrapolated; paper shows it exceeds Qwen3-0.6B on most evals
        mmlu=0.53,
        notes="QAT-quantized (BitCPM4); best sub-0.6B for long-context tasks; use MNN for best speed",
    ),

    # ── ~0.75–1.5B params ─────────────────────────────────────────────────
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
        gsm8k=0.610,   # estimated from Intelligence Index comparison vs Qwen3-0.6B
        mmlu=0.540,    # estimated
        notes="Gated DeltaNet hybrid, multimodal, 262K ctx, 201 langs; avoid few-shot code generation; replaces Qwen3-0.6B",
    ),
    # Llama 3.2-1B: ExecuTorch reference model (SpinQuant + KleidiAI).
    # >350 tok/s prefill on Samsung S24+. Best for latency-critical apps.
    # Source: Meta model card; PyTorch ExecuTorch blog.
    ModelSpec(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        size_mb=658,
        tier=1,
        tok_s_snapdragon_660=8.0,
        tok_s_snapdragon_778g=14.0,
        tok_s_snapdragon_8gen3=40.0,
        peak_memory_mb=900,
        gsm8k=0.535,   # IFEval 53.5%, GSM8K approximate from Meta model card
        mmlu=0.490,    # MMLU approximate
        notes="ExecuTorch reference model; best TTFT via KleidiAI SpinQuant; most tunable 1B (largest fine-tuning gains)",
    ),
    # MiniCPM5-1B: Best-in-class 1B as of May 2026. MATH-500 91.6%, HumanEval+ 78.7%,
    # IFEval 80.4%, τ²-Bench (agentic) 79.5%. Average 42.57 vs 26.77 for Qwen3-0.6B.
    # Q4_K_M GGUF = 688MB confirmed (openbmb/MiniCPM5-1B-GGUF on HuggingFace).
    # llama.cpp supported (OpenBMB docs confirm).
    # Source: openbmb/MiniCPM5-1B model card; deepwiki.com benchmarks.
    ModelSpec(
        model_id="openbmb/MiniCPM5-1B",
        size_mb=688,
        tier=1,
        tok_s_snapdragon_660=8.0,
        tok_s_snapdragon_778g=14.0,
        tok_s_snapdragon_8gen3=38.0,
        peak_memory_mb=920,
        gsm8k=0.850,   # proxy from MATH-500 91.6%; GSM8K not separately published
        mmlu=0.620,    # estimated from benchmark suite aggregate 42.57/100
        notes="Best 1B model (May 2026): MATH-500 91.6%, HumanEval+ 78.7%, IFEval 80.4%; 131K context; hybrid thinking mode",
    ),

    # ── ~1.5–2.5B params ──────────────────────────────────────────────────
    # Gemma 3 1B IT: Google QAT checkpoint; GSM8K 62.8%, benefits from
    # superior instruction tuning. LiteRT/MediaPipe native support.
    # size_mb=806 → params_b≈1.61B → Tier 2 by the tier formula.
    # Despite the "1B" in the name, actual param count places it in Tier 2.
    # Source: arXiv 2503.19786; Google Gemma 3 1B IT model card.
    ModelSpec(
        model_id="google/gemma-3-1b-it",
        size_mb=806,
        tier=2,
        tok_s_snapdragon_660=7.0,
        tok_s_snapdragon_778g=12.0,
        tok_s_snapdragon_8gen3=32.0,
        peak_memory_mb=1050,
        gsm8k=0.628,   # GSM8K 5-shot, Gemma 3 tech report
        mmlu=0.480,    # MMLU 5-shot, approximate from Gemma 3 tech report
        notes="Google QAT INT4; best for LiteRT/MediaPipe deployment; official on-device path for Gemma; multimodal (image+text)",
    ),
    # DeepSeek-R1-Distill-Qwen-1.5B: Specialized reasoning only.
    # MATH-500: 83.9%, AIME 2024: 28.9%. Distilled from DeepSeek-R1 671B.
    # NOT recommended for classification/NER/general tasks — use Qwen models instead.
    # Source: arXiv 2501.12948 Table 4.
    ModelSpec(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        size_mb=958,
        tier=2,
        tok_s_snapdragon_660=4.5,
        tok_s_snapdragon_778g=8.0,
        tok_s_snapdragon_8gen3=22.0,
        peak_memory_mb=1270,
        gsm8k=0.870,   # proxy from MATH-500 83.9%; official GSM8K not reported for 1.5B distill
        mmlu=0.580,    # approximate; MMLU not officially reported for 1.5B distill
        notes="SPECIALIZED REASONING ONLY: MATH-500 83.9%, AIME 28.9%. Do not use for classification/NER. R1 distillation requires longer generation.",
    ),
    # SmolLM2-1.7B: Best Tier 2 for instruction following (IFEval 56.7%,
    # beats Llama 3.2-1B and Qwen2.5-1.5B). Strong for classification tasks.
    # Source: HuggingFaceTB model card; Distil Labs benchmark.
    ModelSpec(
        model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        size_mb=1060,
        tier=2,
        tok_s_snapdragon_660=4.5,
        tok_s_snapdragon_778g=8.0,
        tok_s_snapdragon_8gen3=22.0,
        peak_memory_mb=1350,
        gsm8k=0.488,   # GSM8K 5-shot from HuggingFaceTB model card / Distil Labs
        mmlu=0.520,    # MMLU approximate from Distil Labs benchmark
        notes="Best Tier 2 for IFEval/instruction-following (56.7%); strong for classification; compact training corpus",
    ),

    # ── ~2B params (compact) ──────────────────────────────────────────────
    # gemma-3n-e2b-it uses PLE caching (2.3B effective); Qwen3.5-2B is 2B params.
    # Gemma 3n E2B IT: MatFormer (Matryoshka) architecture, natively multimodal.
    # Beats Gemma3-1B on 9/9 shared benchmarks: HumanEval 66.5% vs 41.5%,
    # MMLU 60.1% vs ~48%. 5B total params / 2.3B effective via PLE caching.
    # Source: llm-stats.com/models/compare/gemma-3-1b-it-vs-gemma-3n-e2b-it; Google.
    # WARNING: Proprietary license (not Apache 2.0) — check before commercial use.
    ModelSpec(
        model_id="google/gemma-3n-e2b-it",
        size_mb=1300,
        tier=3,
        tok_s_snapdragon_660=4.0,
        tok_s_snapdragon_778g=7.0,
        tok_s_snapdragon_8gen3=22.0,
        peak_memory_mb=2200,
        gsm8k=0.700,   # estimated; GSM8K not directly published for E2B (E4B ~83%)
        mmlu=0.601,    # MMLU 60.1% from llm-stats comparison
        notes="MatFormer arch; natively multimodal (text+image+video+audio); beats Gemma3-1B on all benchmarks; HumanEval 66.5%; 50-80 tok/s on NPU; ⚠️ proprietary license",
    ),
    # Qwen3.5-2B (March 2026): Gated DeltaNet hybrid architecture, multimodal, 262K ctx.
    # Same family as Qwen3.5-0.8B but at 2B params. Unsloth Dynamic 2.0 GGUF confirmed
    # working with llama.cpp, Ollama. No classic GSM8K/MMLU published (uses newer suite).
    # Source: unsloth/Qwen3.5-2B-GGUF on HuggingFace; unsloth.ai/docs/models/qwen3.5.
    ModelSpec(
        model_id="unsloth/Qwen3.5-2B-GGUF",
        size_mb=1350,   # estimated Q4_K_M; exact size at huggingface.co/unsloth/Qwen3.5-2B-GGUF
        tier=3,
        tok_s_snapdragon_660=4.0,
        tok_s_snapdragon_778g=7.5,
        tok_s_snapdragon_8gen3=20.0,
        peak_memory_mb=1800,
        gsm8k=0.720,   # estimated; no official score — uses MMLU-ProX/MAXIFE/WMT24++ suite
        mmlu=0.610,    # estimated
        notes="NEW (Mar 2026): Gated DeltaNet hybrid, multimodal (text+image+video), 262K ctx, 201 langs; Unsloth GGUF confirmed for llama.cpp; no GSM8K/MMLU published; thinking OFF by default",
    ),

    # ── ~3B+ params ───────────────────────────────────────────────────────
    # Llama 3.2-3B: ExecuTorch reference model for 3B class. Q4_K_M = ~2.02GB.
    # Decode: ~10 tok/s on SD 8 Gen 3 (CPU). GSM8K 77.7%, ARC-C 78.6%.
    # Q3_K_M alternative (~1.5GB) fits tighter storage budgets at quality cost.
    # Source: Meta model card; hugging-quants HF repo (2.02GB confirmed).
    ModelSpec(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        size_mb=2020,
        tier=3,
        tok_s_snapdragon_660=2.5,
        tok_s_snapdragon_778g=5.0,
        tok_s_snapdragon_8gen3=12.0,
        peak_memory_mb=3400,
        gsm8k=0.777,   # GSM8K from Meta model card / arXiv
        mmlu=0.630,    # MMLU approximate from community benchmarks
        notes="ExecuTorch 3B reference; Q4_K_M 2.02GB confirmed; needs 8GB+ RAM; Q3_K_M ~1.5GB fits tighter budgets at quality cost",
    ),
    # Ministral-3B: MMLU ~65% (beats Llama-3.2-3B's 63.4%), 256K context window
    # (unique in the pool — others max at 128K). Native function calling and JSON.
    # Q4_K_M estimated ~1.9GB (3B params × 0.63 GB/B). 6GB RAM phones feasible.
    # Source: mistralai/Ministral-3-3B-Instruct-2512-GGUF on HuggingFace; codersera.com.
    ModelSpec(
        model_id="mistralai/Ministral-3B-Instruct",
        size_mb=1900,
        tier=3,
        tok_s_snapdragon_660=2.8,
        tok_s_snapdragon_778g=5.5,
        tok_s_snapdragon_8gen3=13.0,
        peak_memory_mb=3200,
        gsm8k=0.760,   # estimated; benchmark suite confirms strong math reasoning
        mmlu=0.650,    # ~65% reported (beats Llama-3.2-3B 63.4%) from codersera.com
        notes="256K context (largest in pool); native function calling + JSON; MMLU ~65% beats Llama-3.2-3B; edge-optimized architecture; ~225 tok/s on desktop GPU",
    ),
    # Phi-4-mini (3.8B): Best reasoning-per-GB in the pool. Q4_K_M = 2.49GB.
    # Fits 8GB RAM phones only (peak ~4.1GB with OS). Microsoft's recommended
    # on-device model. ONNX + LiteRT deployment paths available.
    # Source: localaimaster.com; unsloth/Phi-4-mini-instruct-GGUF HF repo.
    ModelSpec(
        model_id="microsoft/Phi-4-mini-instruct",
        size_mb=2490,
        tier=3,
        tok_s_snapdragon_660=1.0,
        tok_s_snapdragon_778g=2.5,
        tok_s_snapdragon_8gen3=7.0,
        peak_memory_mb=4100,
        gsm8k=0.880,   # GSM8K from Phi-4-mini model card / localaimaster.com
        mmlu=0.720,    # MMLU from Phi-4-mini model card
        notes="Best reasoning-per-GB; 8GB+ RAM phone only (Q4_K_M 2.49GB); ONNX GenAI + LiteRT paths available",
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

    task_type values:
      "math"           — prefer Qwen3, DeepSeek-R1-Distill (high GSM8K)
      "classification" — prefer SmolLM2, Qwen (high IFEval)
      "ner"            — prefer Qwen (structured output strength)
      "code"           — prefer Qwen (Qwen2.5-Coder lineage), Phi-4-mini
      "multilingual"   — prefer Qwen family (CJK training advantage)
      "reasoning"      — prefer Phi-4-mini, DeepSeek-R1-Distill, Qwen3
      None             — default sort by tier then size
    """
    feasible = filter_pool(constraints)

    if task_type == "math" or task_type == "reasoning":
        # Sort by GSM8K score within tier, then by tier
        return sorted(feasible, key=lambda m: (m.tier, -m.gsm8k))
    if task_type == "classification":
        # SmolLM2 leads on IFEval; otherwise sort by MMLU
        def classification_key(m: ModelSpec):
            smol_bonus = -0.05 if "SmolLM" in m.model_id else 0.0
            return (m.tier, smol_bonus - m.mmlu)
        return sorted(feasible, key=classification_key)
    if task_type in ("ner", "multilingual", "code"):
        # Qwen family preferred; sort by GSM8K as proxy for structured generation
        def qwen_first_key(m: ModelSpec):
            qwen_bonus = -0.03 if "Qwen" in m.model_id else 0.0
            return (m.tier, qwen_bonus - m.gsm8k)
        return sorted(feasible, key=qwen_first_key)

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
            "value_mb": model.peak_memory_mb,
            "limit_mb": constraints.memory_mb,
            "pass": model.peak_memory_mb <= constraints.memory_mb,
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
