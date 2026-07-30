# android_pool.py
#
# MODEL POOL DESIGN — QWEN-ONLY BRANCH
#
# This branch restricts the model pool to the Qwen architecture family for
# controlled ablation of model selection strategies. All models share the same
# tokenizer lineage and MNN-LLM runtime compatibility.
#
# Tier structure rationale:
#   Tiers are ON-DISK WEIGHT SIZE buckets of each deployed variant, NOT param-count bands.
#   The loop selects the smallest variant that still hits the accuracy goal, and a single
#   model's BF16/Q8/Q4 variants can land in different tiers.
#
#     Tier 0  size < 750 MB
#     Tier 1  size 750–1500 MB
#     Tier 2  size 1500–2500 MB
#     Tier 3  size >= 2500 MB
#
#   Tier used to bucket a MODELLED peak-inference-RAM figure. It now buckets weight size,
#   which is verifiable. See "NOTE ON THROUGHPUT AND PEAK RAM" below.
#
#   Each base entry stores the Q4_K_M size. _variant() expands it into three deployment
#   candidates (BF16 / Q8_0 / Q4_K_M) at bytes/param BF16 2.0, Q8_0 1.0, Q4_K_M 0.55 —
#   overridden by a REAL measured size whenever config/measured_metrics.json has one.
#
# Pool members (6 official Qwen base models × 3 quant variants = 18 entries):
#   Qwen3-0.6B                 (text)       — Tier 0 Q4 seed
#   Qwen3-1.7B                 (text)       — Tier 1 Q4 seed
#   Qwen3-4B-Instruct-2507     (text)       — Tier 3 Q4 seed; NON-thinking
#   Qwen3.5-0.8B               (multimodal) — Tier 0 seed; text-only LoRA via FastVisionModel
#   Qwen3.5-2B                 (multimodal) — Tier 2 seed; base repo (not -GGUF, B107)
#   Qwen3.5-4B                 (multimodal) — Tier 3 Q4 seed
#
# No Qwen2.5, no distilled (DeepSeek-R1-Distill), no thinking-only (Qwen3-4B-Thinking-2507)
# models. See docs/model_pool.md for the per-model capability write-up and "worth using for".
#
# Multimodal handling (B123/B136): Qwen3.5 is a "Causal LM with Vision" and is fine-tuned
# TEXT-ONLY via Unsloth's FastVisionModel with finetune_vision_layers=False (see lora_trainer).
# Qwen3.5-* use the BASE transformers repos, never the -GGUF repo (no transformers config,
# cannot be fine-tuned, B107).
#
# Android framework compatibility:
#   llama.cpp (GGUF): all models — broadest format support, CPU+Vulkan backends
#   MNN-LLM:    Qwen family — fastest CPU prefill (8.6x vs llama.cpp)
#   ExecuTorch: Qwen3 all sizes — SpinQuant INT4, KleidiAI acceleration
#
# Capability sources (checked 2026-07-21):
#   Qwen3 report: https://arxiv.org/abs/2505.09388
#   Qwen3.5-0.8B card: https://huggingface.co/Qwen/Qwen3.5-0.8B
#   Qwen3.5-2B card: https://huggingface.co/Qwen/Qwen3.5-2B
#   Qwen3.5-4B card: https://huggingface.co/Qwen/Qwen3.5-4B
#   Qwen3-4B-Instruct-2507 card:
#     https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
#
# MMLU, MMLU-Pro, and MMLU-Redux are different evaluations. Scores retain their
# explicit metric names below and must never be compared as one generic "mmlu" value.
# Missing benchmarks remain None; in particular, no GSM8K value is inferred from a
# different model or benchmark.

import functools
import json
import os
from dataclasses import dataclass


METRIC_COMPARABILITY_CAVEAT = (
    "MMLU, MMLU-Pro, and MMLU-Redux are distinct evaluations and are not directly "
    "comparable. Compare only the same named metric under the same mode/protocol. "
    "A benchmark marked 'not reported' is unknown, not zero; never infer a missing "
    "score or rank it below a reported score."
)


# NOTE ON THROUGHPUT AND PEAK RAM (deliberate absence)
#
# This pool intentionally carries NO estimated tok/s and NO estimated peak inference RAM.
# It used to: per-model tok/s for three reference chips plus a CHIP_SCALE_FACTORS table
# that interpolated "decode throughput" for any other chipset. Those were modelled
# numbers presented in the same shape as measurements, and downstream code (hardware
# gating, model-selection prompts, run reports) consumed them as if they were facts.
#
# They were not facts. Decode speed depends on memory bandwidth, thermal state, runtime,
# context length, and batch shape; none of that is knowable from a parameter count. The
# one place we checked a modelled number against reality, it was 20% off (Qwen3.5-4B
# Q4_K_M: modelled 2200 MB on disk, actual GGUF 2654.5 MB).
#
# The rule now: throughput and true peak RAM come from MEASUREMENT or they are `None`.
#   - Real measurements live in `config/measured_metrics.json`, keyed by
#     (model_id, quant, chip), and are loaded by `measured_metrics_for()`.
#   - Absent a measurement, consumers must render "unmeasured" and MUST NOT gate on a
#     substitute. See `check_hardware_constraints`.
#   - `size_mb` (on-disk weight bytes) is kept: it is arithmetic over the weight files,
#     it is verifiable, and it is the only quantity used to ORDER models by size.
#
# Populate measurements with:  python hardware_eval/measure_model.py --help


# Chip identifiers the pipeline recognises as valid `target_chip` values. This is a
# NAMESPACE, not a performance model: it carries no throughput factors, and membership
# implies nothing about how fast a chip is. It exists so hardware_research can validate
# the chipset it inferred against a known vocabulary, and so measurements can be filed
# under a canonical key. The previous version of this list doubled as CHIP_SCALE_FACTORS,
# mapping each chip to a made-up relative decode multiplier — that is what was removed.
KNOWN_CHIPS: tuple[str, ...] = (
    "snapdragon_660",
    "snapdragon_730",
    "snapdragon_750g",
    "snapdragon_778g",
    "snapdragon_870",
    "snapdragon_888",
    "snapdragon_8gen1",
    "snapdragon_8gen2",
    "snapdragon_8gen3",
    "snapdragon_8elite",
    "dimensity_9300",
    "dimensity_9400",
    "exynos_2400",
    "exynos_2500",
    "tensor_g3",
    "tensor_g4",
    "host_cpu",   # local host measurements (never a stand-in for a phone)
)


@dataclass(frozen=True)
class CapabilityMeasurement:
    """One sourced measurement tied to the artifact and evaluation protocol used."""

    metric: str
    value: float
    artifact: str
    mode: str | None
    protocol: str | None
    source: str

    @property
    def comparison_key(self) -> tuple[str, str | None, str | None]:
        """Measurements are rank-comparable only when this key is identical."""
        return (self.metric, self.mode, self.protocol)


@dataclass
class ModelSpec:
    model_id: str          # HuggingFace model ID
    size_mb: int           # on-disk weight size in MB for THIS variant's quant (see `quant`)
    tier: int              # size bucket of size_mb: 0=<0.75GB, 1=0.75-1.5GB, 2=1.5-2.5GB, 3=>=2.5GB
    # There is deliberately no tok/s or peak_memory_mb field here — see the note above
    # ANDROID_POOL. Those come from `measured_metrics_for()` or they are unknown.
    # Optional, sourced measurements (0–1 scale). Missing means unknown, not zero.
    capability_measurements: tuple[CapabilityMeasurement, ...] = ()
    notes: str = ""
    quant: str | None = None   # None = base (BF16/FP16). "Q4_K_M" or "Q8_0" = GGUF variant.
    multimodal: bool = False   # True for image-text-to-text models (Qwen3.5, Gemma-3n).
                               # Handled via text_tokenizer() in lora_trainer/slm_helpers.

    @property
    def label(self) -> str:
        """Human-readable id for logs, including the on-device weight format (quant).
        e.g. 'Qwen/Qwen3-4B-Instruct-2507 [Q4_K_M]'. Used in node log prefixes so the
        exact variant being trained/evaluated is visible in the progress output."""
        return f"{self.model_id} [{self.quant or 'bf16'}]"

    @property
    def selector(self) -> str:
        """Stable deployment-variant identity used by prompts, overrides, and history."""
        return f"{self.model_id}@{self.quant or 'bf16'}"

    def measurement(self, metric: str) -> CapabilityMeasurement | None:
        """Return this artifact's measurement for ``metric``, if one is sourced."""
        return next(
            (item for item in self.capability_measurements if item.metric == metric),
            None,
        )

    @property
    def gsm8k(self) -> float | None:
        item = self.measurement("GSM8K")
        return item.value if item else None

    @property
    def knowledge_metric(self) -> str | None:
        item = self.measurement("MMLU-Pro") or self.measurement("MMLU")
        return item.metric if item else None

    @property
    def knowledge_score(self) -> float | None:
        item = self.measurement("MMLU-Pro") or self.measurement("MMLU")
        return item.value if item else None

    @property
    def mmlu_redux(self) -> float | None:
        item = self.measurement("MMLU-Redux")
        return item.value if item else None

    @property
    def benchmark_mode(self) -> str | None:
        item = self.measurement("MMLU-Pro") or self.measurement("MMLU")
        return item.mode if item else None

    @property
    def benchmark_source(self) -> str | None:
        item = self.measurement("MMLU-Pro") or self.measurement("MMLU")
        return item.source if item else None

    @property
    def gsm8k_source(self) -> str | None:
        item = self.measurement("GSM8K")
        return item.source if item else None

    def est_params_b(self) -> float:
        """Estimate parameter count (billions) from this variant's weight size.
        Uses the quant's bytes/param so all three variants of one model return the
        same param count (BF16 2.0, Q8_0 1.0, Q4_K_M 0.55 GB/1B)."""
        bpp = 0.55 if self.quant == "Q4_K_M" else 1.0 if self.quant == "Q8_0" else 2.0
        return self.size_mb / 1000 / bpp

    def measured(self, chip: str) -> dict | None:
        """Real measured metrics for this variant on `chip`, or None if never measured.

        Never falls back to another chip or another quant: a number measured on different
        silicon is not a measurement of this one.
        """
        return measured_metrics_for(self.model_id, self.quant, chip)


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
    #                     This threshold is CONSTANT regardless of model or task —
    #                     it's a UX floor, not a model property.
    #                     APPLIES ONLY TO MEASURED THROUGHPUT. A candidate with no
    #                     recorded measurement is never eliminated by this floor,
    #                     because there is no honest number to compare against.
    latency_ttft_ms: int
    power_watts: float = 6.0
    target_chip: str = "snapdragon_778g"
    # Sustained decode throughput floor (hardware_metrics.md §6.5), applied ONLY where a
    # real measurement exists for (model, quant, target_chip). Unmeasured candidates pass.
    # Set to 6.0 for interactive streaming; 0 for batch/async workflows.
    min_tok_s: float = 0.0


# ---------------------------------------------------------------------------
# QWEN-ONLY POOL TIERS (by ON-DISK WEIGHT SIZE, not param count and not peak RAM)
#
# Tier 0 — size < 750 MB
# Tier 1 — size 750–1500 MB
# Tier 2 — size 1500–2500 MB
# Tier 3 — size >= 2500 MB
#
# These buckets used to be cut on a MODELLED peak-inference-RAM figure, which was removed
# with the rest of the fabricated metrics. Tier is only used to order and group candidates
# by rough scale, which real weight size serves without inventing a runtime number.
#
# A tier belongs to a quantized deployment VARIANT, not to a base model or
# parameter-count class. The same base model can occupy multiple tiers.
#
# ---------------------------------------------------------------------------
# Variant expansion: 6 base models × 3 quant variants = 18 entries.
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

_QWEN35_SMALL_NONTHINKING_PROTOCOL = "qwen3.5-small-nonthinking-card-table"
_QWEN35_4B_UNSPECIFIED_PROTOCOL = "qwen3.5-4b-card-table-mode-unspecified"


_MEASURED_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "measured_metrics.json"
)


@functools.lru_cache(maxsize=1)
def _load_measured_metrics() -> dict:
    """Load the whole config/measured_metrics.json document.

    Callers index into the section they need (`_sizes` or `measurements`); returning the
    full document keeps those two namespaces distinct.
    Missing file == nothing measured yet.
    """
    try:
        with open(_MEASURED_METRICS_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:  # corrupt file must not silently read as "unmeasured"
        raise RuntimeError(
            f"{_MEASURED_METRICS_PATH} exists but could not be parsed: {exc}"
        ) from exc


def measured_metrics_key(model_id: str, quant: str | None, chip: str) -> str:
    return f"{model_id}|{quant or 'bf16'}|{chip}"


def measured_size_mb(model_id: str, quant: str | None) -> int | None:
    """Real on-disk MB from an actual build of this (model, quant), else None.

    Size is chip-independent, so this is keyed without a chip. Recorded by
    hardware_eval/measure_model.py after a real GGUF build.
    """
    sizes = _load_measured_metrics().get("_sizes", {})
    value = sizes.get(f"{model_id}|{quant or 'bf16'}")
    return round(value) if value is not None else None


def measured_metrics_for(model_id: str, quant: str | None, chip: str) -> dict | None:
    """Real measured metrics for exactly this (model, quant, chip), else None.

    Deliberately exact-match: no interpolation across chips or quants. An unmeasured
    combination reads as unknown, which is the honest answer, rather than as a number
    borrowed from a different configuration.
    """
    return _load_measured_metrics().get("measurements", {}).get(
        measured_metrics_key(model_id, quant, chip)
    )


def _size_tier(size_mb: int) -> int:
    """Tier is a bucket of on-disk weight size — the one metric we can verify.

    Previously this bucketed a MODELLED peak-inference-RAM figure. Tier is used only to
    order and group candidates by rough scale, which weight size serves equally well
    without inventing a runtime number.
    """
    if size_mb < 750:
        return 0
    if size_mb < 1500:
        return 1
    if size_mb < 2500:
        return 2
    return 3


def _variant(base: ModelSpec, quant: str | None) -> ModelSpec:
    """Build one deployment variant (quant=None→BF16, "Q8_0", or "Q4_K_M").

    `size_mb` is the only derived quantity left, and it is derived by arithmetic over
    weight bytes rather than by performance modelling: a quantized checkpoint holds the
    same parameters at a different bytes-per-parameter. A REAL measured size from an
    actual GGUF build always wins over the arithmetic — see `measured_size_mb()`.
    """
    q4_size = base.size_mb
    if quant == "Q4_K_M":
        size = q4_size
    elif quant == "Q8_0":
        size = round(q4_size * _Q8_OVER_Q4)
    else:  # BF16 base
        size = round(q4_size * _BF16_OVER_Q4)
    real = measured_size_mb(base.model_id, quant)
    if real is not None:
        size = real
    return ModelSpec(
        model_id=base.model_id,
        size_mb=size,
        tier=_size_tier(size),
        capability_measurements=base.capability_measurements,
        notes=base.notes,
        quant=quant,
        multimodal=base.multimodal,
    )


ANDROID_POOL: list[ModelSpec] = [
    # NOTE: each entry below is a Q4_K_M SEED holding that variant's on-disk size_mb.
    # Its `tier=` field is IGNORED — _variant() recomputes tier per variant from weight
    # size via _size_tier(), and substitutes a REAL measured size when one is recorded in
    # config/measured_metrics.json. The "~params" in the section headers just groups
    # seeds by model scale for readability.
    #
    # OFFICIAL QWEN POOL (Qwen3 + Qwen3.5, official Qwen/* repos only).
    # No Qwen2.5, no distilled models, no thinking-only models. The two families:
    #   - Qwen3 (text): 0.6B, 1.7B, 4B-Instruct-2507 — verified, reliable.
    #   - Qwen3.5 (multimodal "Causal LM with Vision"): 0.8B, 2B, 4B — fine-tuned
    #     TEXT-ONLY via FastVisionModel (finetune_vision_layers=False), see lora_trainer.
    # Official capability values retain the metric names and evaluation mode used by
    # their source. Sources were checked 2026-07-21; URLs are stored on each seed.

    # ── Qwen3-0.6B (text) — Tier 0 seed ───────────────────────────────────
    # Qwen3 Technical Report Table 8 values are Base/proxy measurements and are not
    # attached to this post-trained artifact.
    ModelSpec(
        model_id="Qwen/Qwen3-0.6B",
        size_mb=400,
        tier=0,
        notes="Qwen3-0.6B (official); text-only; dual-mode; use non-thinking for classification/NER",
    ),

    # ── Qwen3-1.7B (text) — Tier 1 seed ───────────────────────────────────
    # Qwen3.5 comparison card (non-thinking): MMLU-Pro 40.2, MMLU-Redux 64.4.
    # Do not attach the report's GSM8K row: it does not identify this exact
    # post-trained artifact/mode.
    ModelSpec(
        model_id="Qwen/Qwen3-1.7B",
        size_mb=1000,
        tier=1,
        capability_measurements=(
            CapabilityMeasurement(
                metric="MMLU-Pro",
                value=0.402,
                artifact="Qwen/Qwen3-1.7B",
                mode="non-thinking",
                protocol=_QWEN35_SMALL_NONTHINKING_PROTOCOL,
                source="https://huggingface.co/Qwen/Qwen3.5-0.8B",
            ),
            CapabilityMeasurement(
                metric="MMLU-Redux",
                value=0.644,
                artifact="Qwen/Qwen3-1.7B",
                mode="non-thinking",
                protocol=_QWEN35_SMALL_NONTHINKING_PROTOCOL,
                source="https://huggingface.co/Qwen/Qwen3.5-0.8B",
            ),
        ),
        notes="Qwen3-1.7B (official); text-only; Q4/Q8/BF16 variants span peak-RAM tiers 1–3",
    ),

    # ── Qwen3-4B-Instruct-2507 (text) — Tier 3 seed ───────────────────────
    # Model card: MMLU-Redux 84.2, MMLU-Pro 69.6, IFEval 83.4, non-thinking (no CoT
    # preamble → good for classification/NER/generation). 256K ctx. 4B (3.6B non-embed).
    ModelSpec(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        size_mb=2200,
        tier=3,
        capability_measurements=(
            CapabilityMeasurement(
                metric="MMLU-Pro",
                value=0.696,
                artifact="Qwen/Qwen3-4B-Instruct-2507",
                mode="non-thinking",
                protocol=_QWEN35_SMALL_NONTHINKING_PROTOCOL,
                source="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
            ),
            CapabilityMeasurement(
                metric="MMLU-Redux",
                value=0.842,
                artifact="Qwen/Qwen3-4B-Instruct-2507",
                mode="non-thinking",
                protocol=_QWEN35_SMALL_NONTHINKING_PROTOCOL,
                source="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
            ),
        ),
        notes="Qwen3-4B-Instruct-2507 (official); text-only; NON-thinking (direct answers); 256K context",
    ),

    # ── Qwen3.5-0.8B (multimodal) — Tier 0 seed ───────────────────────────
    # Multimodal "Causal LM with Vision"; fine-tuned TEXT-ONLY via FastVisionModel
    # (finetune_vision_layers=False). Card values below are non-thinking mode.
    ModelSpec(
        model_id="Qwen/Qwen3.5-0.8B",
        size_mb=500,
        tier=0,
        capability_measurements=(
            CapabilityMeasurement(
                metric="MMLU-Pro",
                value=0.297,
                artifact="Qwen/Qwen3.5-0.8B",
                mode="non-thinking",
                protocol=_QWEN35_SMALL_NONTHINKING_PROTOCOL,
                source="https://huggingface.co/Qwen/Qwen3.5-0.8B",
            ),
            CapabilityMeasurement(
                metric="MMLU-Redux",
                value=0.485,
                artifact="Qwen/Qwen3.5-0.8B",
                mode="non-thinking",
                protocol=_QWEN35_SMALL_NONTHINKING_PROTOCOL,
                source="https://huggingface.co/Qwen/Qwen3.5-0.8B",
            ),
        ),
        notes="Qwen3.5-0.8B (official, multimodal); text-only LoRA via FastVisionModel; 262K context",
        multimodal=True,
    ),

    # ── Qwen3.5-2B (multimodal) — Tier 2 seed ─────────────────────────────
    ModelSpec(
        model_id="Qwen/Qwen3.5-2B",
        size_mb=1100,
        tier=2,
        capability_measurements=(
            CapabilityMeasurement(
                metric="MMLU-Pro",
                value=0.553,
                artifact="Qwen/Qwen3.5-2B",
                mode="non-thinking",
                protocol=_QWEN35_SMALL_NONTHINKING_PROTOCOL,
                source="https://huggingface.co/Qwen/Qwen3.5-2B",
            ),
            CapabilityMeasurement(
                metric="MMLU-Redux",
                value=0.692,
                artifact="Qwen/Qwen3.5-2B",
                mode="non-thinking",
                protocol=_QWEN35_SMALL_NONTHINKING_PROTOCOL,
                source="https://huggingface.co/Qwen/Qwen3.5-2B",
            ),
        ),
        notes="Qwen3.5-2B (official, multimodal); text-only LoRA via FastVisionModel; base repo (not -GGUF, B107); 262K context",
        multimodal=True,
    ),

    # ── Qwen3.5-4B (multimodal) — Tier 3 seed ─────────────────────────────
    ModelSpec(
        model_id="Qwen/Qwen3.5-4B",
        size_mb=2200,
        tier=3,
        capability_measurements=(
            CapabilityMeasurement(
                metric="MMLU-Pro",
                value=0.791,
                artifact="Qwen/Qwen3.5-4B",
                mode=None,
                protocol=_QWEN35_4B_UNSPECIFIED_PROTOCOL,
                source="https://huggingface.co/Qwen/Qwen3.5-4B",
            ),
            CapabilityMeasurement(
                metric="MMLU-Redux",
                value=0.888,
                artifact="Qwen/Qwen3.5-4B",
                mode=None,
                protocol=_QWEN35_4B_UNSPECIFIED_PROTOCOL,
                source="https://huggingface.co/Qwen/Qwen3.5-4B",
            ),
        ),
        notes="Qwen3.5-4B (official, multimodal); text-only LoRA via FastVisionModel; 262K context",
        multimodal=True,
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


def format_capability_metrics(model: ModelSpec) -> str:
    """Render sourced scores with artifact/mode/protocol; never fabricate missing data."""

    def render(item: CapabilityMeasurement | None, metric: str) -> str:
        if item is None:
            return f"{metric}: not reported"
        mode = item.mode or "not specified by source"
        protocol = item.protocol or "not specified by source"
        return (
            f"{item.metric}: {item.value * 100:.1f} "
            f"[artifact={item.artifact}; mode={mode}; protocol={protocol}; "
            f"source={item.source}]"
        )

    return "; ".join(
        (
            render(model.measurement("GSM8K"), "GSM8K"),
            render(
                model.measurement("MMLU-Pro") or model.measurement("MMLU"),
                "knowledge benchmark",
            ),
            render(model.measurement("MMLU-Redux"), "MMLU-Redux"),
        )
    )


def _resource_sort_key(model: ModelSpec):
    return (
        model.tier,
        model.size_mb,
        model.model_id,
        model.quant or "bf16",
    )


def resolve_model_selector(
    candidates: list[ModelSpec],
    value: str,
) -> ModelSpec | None:
    """Resolve an exact selector, with deterministic bare-model compatibility.

    Exact ``model_id@quant`` selectors always identify one deployment variant. A
    legacy bare ``model_id`` is accepted when unique; when several quant siblings
    are present, it deterministically chooses the smallest-on-disk sibling (then
    tier/selector as stable tie-breakers). This preserves old overrides without
    retaining list-order/first-match ambiguity.
    """
    exact = [model for model in candidates if model.selector == value]
    if exact:
        return min(exact, key=_resource_sort_key)

    siblings = [model for model in candidates if model.model_id == value]
    if not siblings:
        return None
    return min(
        siblings,
        key=lambda model: (
            model.size_mb,
            model.tier,
            model.selector,
        ),
    )


def _sort_by_comparable_metric(feasible, metric_getter):
    """Sort a tier only when every measurement has an identical comparison key."""
    ranked = []
    for tier in sorted({model.tier for model in feasible}):
        group = sorted(
            (model for model in feasible if model.tier == tier),
            key=_resource_sort_key,
        )
        observations = [metric_getter(model) for model in group]
        comparison_keys = {
            item.comparison_key for item in observations if item is not None
        }
        if (
            group
            and len(comparison_keys) == 1
            and all(item is not None for item in observations)
        ):
            scores = {
                id(model): item.value
                for model, item in zip(group, observations)
                if item is not None
            }
            group = sorted(
                group,
                key=lambda model: (-scores[id(model)], _resource_sort_key(model)),
            )
        ranked.extend(group)
    return ranked


def filter_pool(constraints: HardwareConstraints) -> list[ModelSpec]:
    """Return variants that satisfy all hard constraints, sorted by tier then size.

    Each entry is an independent quant variant (BF16 / Q8_0 / Q4_K_M).

    Filters ONLY on quantities that are known rather than modelled:
      - storage_mb: this variant's on-disk weight file must fit on disk
      - memory_mb:  the weight bytes must fit in RAM. This is a lower bound, not the
                    true peak (which adds KV cache + runtime and needs measurement),
                    but a model whose weights alone exceed RAM cannot run.
      - a recorded real measurement, when config/measured_metrics.json has one for this
        (model, quant, target_chip), additionally gates on measured peak RAM / tok-s.

    Throughput is NOT gated from an estimate. `min_tok_s` applies only where a real
    measurement exists; unmeasured candidates are not eliminated on a guess.
    """
    feasible = [
        m for m in ANDROID_POOL
        if m.size_mb <= constraints.storage_mb
        and m.size_mb <= constraints.memory_mb
        and all_constraints_pass(check_hardware_constraints(m, constraints))
    ]
    return sorted(feasible, key=lambda m: (m.tier, m.size_mb))


def filter_pool_by_task(
    constraints: HardwareConstraints,
    task_type: str | None = None,
) -> list[ModelSpec]:
    """
    Return feasible models, optionally pre-sorted for a specific task type.

    Benchmark ordering is applied within a tier only when every candidate in that
    tier has the same explicitly named metric. If values are missing or metric names
    differ (for example MMLU vs MMLU-Pro), resource order is retained. This prevents
    unknown values from becoming numeric zero and prevents incomparable measurements
    from biasing selection.
    """
    feasible = filter_pool(constraints)

    if task_type in ("math", "reasoning", "math_reasoning"):
        return _sort_by_comparable_metric(
            feasible,
            lambda model: model.measurement("GSM8K"),
        )
    if task_type == "classification":
        return _sort_by_comparable_metric(
            feasible,
            lambda model: (
                model.measurement("MMLU-Pro") or model.measurement("MMLU")
            ),
        )

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

    Gating contract: a constraint gates ONLY on a real measurement, with one exception —
    memory additionally enforces a weight-size lower bound, since a model whose weight
    bytes exceed available RAM cannot run regardless of runtime overhead. Every other
    unmeasured constraint reports `measured: False` and passes rather than inventing a
    value. `measured` may be supplied by the caller (a live device run) or resolved from
    config/measured_metrics.json.
    """
    chip = constraints.target_chip
    # A measurement passed in by the caller (a real device/llama.cpp run) wins; otherwise
    # look for a recorded measurement for exactly this (model, quant, chip).
    measured = measured or model.measured(chip)

    def _measured(field):
        return measured.get(field) if measured else None

    peak_mb = _measured("peak_memory_mb")
    tok_s = _measured("tok_per_s")
    ttft_ms = _measured("ttft_ms")
    watts = _measured("avg_watts")

    result = {
        "storage": {
            "value_mb": model.size_mb,
            "limit_mb": constraints.storage_mb,
            "pass": model.size_mb <= constraints.storage_mb,
        },
        "memory": {
            # Two distinct checks, never conflated:
            #  - weight_floor: the weight bytes MUST fit in RAM. Arithmetic over the
            #    files, not an estimate, so it is always safe to gate on. Necessary but
            #    NOT sufficient — true peak adds KV cache + runtime, which vary with
            #    context length and cannot be derived from a parameter count.
            #  - value_mb: real peak RSS, present only when actually measured.
            "weight_floor_mb": model.size_mb,
            "value_mb": peak_mb,
            "limit_mb": constraints.memory_mb,
            "measured": peak_mb is not None,
            "note": (
                "measured peak RSS"
                if peak_mb is not None
                else "UNMEASURED — gating on weight-size lower bound only; "
                     "true peak (weights + KV cache + runtime) requires measurement"
            ),
            "pass": (
                peak_mb <= constraints.memory_mb
                if peak_mb is not None
                else model.size_mb <= constraints.memory_mb
            ),
        },
        "latency": {
            # No estimate. Previously this derived TTFT as 1/tok_s*1000 from a modelled
            # tok/s — a guess built on a guess. Unmeasured now means unmeasured, and an
            # unmeasured gate cannot fail a model.
            "measured_ttft_ms": ttft_ms,
            "measured_tok_s": tok_s,
            "limit_ms": constraints.latency_ttft_ms,
            "min_tok_s": constraints.min_tok_s,
            "measured": ttft_ms is not None or tok_s is not None,
            "chip": (measured or {}).get("device", chip),
            "note": None if measured else "UNMEASURED — not gating",
            "ttft_pass": (
                ttft_ms <= constraints.latency_ttft_ms if ttft_ms is not None else True
            ),
            "throughput_pass": (
                constraints.min_tok_s <= 0 or tok_s >= constraints.min_tok_s
                if tok_s is not None
                else True
            ),
            "pass": (
                (ttft_ms is None or ttft_ms <= constraints.latency_ttft_ms)
                and (
                    tok_s is None
                    or constraints.min_tok_s <= 0
                    or tok_s >= constraints.min_tok_s
                )
            ),
        },
        "power": {
            "value_watts": watts,
            "limit_watts": constraints.power_watts,
            "measured": watts is not None,
            "note": None if watts is not None else "UNMEASURED — not gating",
            "pass": watts <= constraints.power_watts if watts is not None else True,
        },
    }
    return result
