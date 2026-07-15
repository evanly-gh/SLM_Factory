# Model Pool — Official Qwen (Qwen3 + Qwen3.5)

The **current, authoritative** model-pool reference. (The old broad-candidate research doc is
archived at [`config/android_pool.md`](../config/android_pool.md).)

The `ANDROID_POOL` in `config/android_pool.py` is restricted to **official Qwen models from the
Qwen HuggingFace org**. No Qwen2.5, no distilled models (e.g. DeepSeek-R1-Distill), and no
thinking-only models (e.g. Qwen3-4B-Thinking-2507). Two families:

- **Qwen3 (text-only)** — verified, reliable, the backbone of the pool.
- **Qwen3.5 (multimodal, "Causal LM with Vision")** — fine-tuned **text-only** via Unsloth's
  `FastVisionModel` with `finetune_vision_layers=False`. Benchmarks are **estimated** and the
  text-only multimodal LoRA path is **best-effort / unverified** pending a GPU run.

Each base model is expanded into 3 deployment variants (`Q4_K_M`, `Q8_0`, `bf16`), so the pool is
**6 base models → 18 variants**. Tier is a **peak-RAM bucket** of each variant (`_ram_tier`:
<750 / 750–1500 / 1500–2500 / ≥2500 MB), so a model's quant variants can span tiers — this is
what lets model selection start small and escalate.

---

## The models

| Model | Params | Type | Ctx | GSM8K | MMLU | Tier span | Verified? |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 0.6B | text | 32K | 0.596 | 0.528 | 0 (Q4) → 2 (BF16) | ✅ Qwen3 report |
| `Qwen/Qwen3-1.7B` | 1.7B | text | 32K | 0.754 | 0.626 | 1 → 3 | ✅ Qwen3 report |
| `Qwen/Qwen3-4B-Instruct-2507` | 4B | text | 256K | ~0.88 | ~0.83 | 3 | ✅ model card |
| `Qwen/Qwen3.5-0.8B` | 0.8B | multimodal | 262K | ~0.61 | ~0.54 | 0 → 2 | ⚠️ estimated |
| `Qwen/Qwen3.5-2B` | 2B | multimodal | 262K | ~0.72 | ~0.61 | 2 → 3 | ⚠️ estimated |
| `Qwen/Qwen3.5-4B` | 4B | multimodal | 262K | ~0.85 | ~0.70 | 3 | ⚠️ estimated |

Every tier is populated after quant expansion, so escalation can walk tiers 0→3 on a
large-enough device.

---

## Which to use for what

**`Qwen3-0.6B` — Tier 0 default / routing.** Smallest, cheapest, loads everywhere. Good for
binary/simple classification, routing, keyword extraction. Weak at multi-step reasoning
(GSM8K 0.60). `smallest_first` starts here. Use non-thinking mode for classification/NER.

**`Qwen3-1.7B` — Tier 1 workhorse.** Big jump over 0.6B (GSM8K 0.60 → 0.75). Solid general-purpose
small model for classification, NER, and light generation. Reliable text-only.

**`Qwen3-4B-Instruct-2507` — Tier 3 general champion.** Non-thinking (direct answers, no CoT
preamble), so it does *not* suffer the reasoning-model truncation problem on classification/NER.
Strongest instruction-following (IFEval 83.4) and general knowledge (MMLU-Redux 84.2) in the pool.
Best default when a task needs the top text model. 256K context. 8 GB+ RAM (Q4).

**`Qwen3.5-0.8B` — Tier 0 multimodal.** Newer architecture, similar footprint to Qwen3-0.6B with
slightly higher estimated scores. Multimodal → text-only LoRA via the `FastVisionModel` path
(best-effort). Prefer Qwen3-0.6B until the multimodal path is verified on hardware.

**`Qwen3.5-2B` — Tier 2 multimodal.** Mid-tier multimodal option; fills the tier-2 seed. Base
transformers repo (never `-GGUF`). Estimated benchmarks.

**`Qwen3.5-4B` — Tier 3 multimodal champion.** Top-capability multimodal model; alternative to
Qwen3-4B-Instruct-2507 when latest-arch/multimodal behavior is wanted. Estimated benchmarks.

---

## Notes / caveats

- **Excluded on purpose:** Qwen2.5 (superseded by Qwen3), DeepSeek-R1-Distill (reasoning-only,
  catastrophic forgetting on general tasks), Qwen3-4B-Thinking-2507 (thinking-only — always emits
  CoT, which truncates/garbles classification & NER inside the eval token budget — the same
  failure mode we removed the distilled model for).
- **Everything ≥4B is Tier 3 only** (~2.3 GB Q4 / ~2.9 GB peak → 8 GB+ phones). Low-RAM devices
  effectively choose among the 0.6B / 0.8B / 1.7B / 2B models.
- **Multimodal text-only LoRA** (Qwen3.5): loaded with `FastVisionModel`, vision tower frozen
  (`finetune_vision_layers=False`), only language layers tuned. See `training/lora_trainer.py`
  (`is_multimodal_model` + the FastVisionModel branch) and `slm_helpers.infer`. Uses the **base**
  transformers repo, never `-GGUF` (a GGUF repo has no transformers config → cannot fine-tune).
  Implemented to Unsloth's documented Qwen3.5 recipe but **not yet verified on hardware**.
- **Qwen3.5 benchmark numbers are estimates** — replace with official numbers once available.

---

## Sources
- Qwen3 Technical Report — [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) (0.6B: MMLU 52.81 / GSM8K 59.59; 1.7B: 62.63 / 75.44; 4B: 72.99 / 87.79)
- Qwen3-4B-Instruct-2507 model card — [HF](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (MMLU-Redux 84.2, MMLU-Pro 69.6, IFEval 83.4, 256K ctx, non-thinking)
- Unsloth Qwen3.5 fine-tuning guide (FastVisionModel text-only recipe) — unsloth.ai/docs/models/qwen3.5/fine-tune
