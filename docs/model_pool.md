# Model Pool — Official Qwen (Qwen3 + Qwen3.5)

The **current, authoritative** model-pool reference. (The old broad-candidate research doc is
archived at [`config/android_pool.md`](../config/android_pool.md).)

The `ANDROID_POOL` in `config/android_pool.py` is restricted to **official Qwen models from the
Qwen HuggingFace org**. No Qwen2.5, no distilled models (e.g. DeepSeek-R1-Distill), and no
thinking-only models (e.g. Qwen3-4B-Thinking-2507). Two families:

- **Qwen3 (text-only)**.
- **Qwen3.5 (multimodal, "Causal LM with Vision")** — fine-tuned **text-only** via Unsloth's
  `FastVisionModel` with `finetune_vision_layers=False`.

Each base model is expanded into 3 deployment variants (`Q4_K_M`, `Q8_0`, `bf16`), so the pool is
**6 base models → 18 variants**. Tier is an **on-disk weight-size bucket** of each variant
(`_size_tier`: <750 / 750–1500 / 1500–2500 / ≥2500 MB), so a model's quant variants can span
tiers — this is what lets model selection start small and escalate.

> **Correction.** This previously said tier bucketed a *modelled peak-inference-RAM* figure
> (`_ram_tier`). That function is gone. Tier is used only to order and group candidates by rough
> scale, which real weight size serves equally well without inventing a runtime number — and
> inventing runtime numbers is what the 2026-07-27 change below removed everywhere else in the pool.
> The bucket boundaries are unchanged, so tier assignments did not move.

Every variant has a stable selector: `<model_id>@bf16`, `<model_id>@Q8_0`, or
`<model_id>@Q4_K_M`. Prompts, force overrides, baselines, and histories use that selector.
A legacy bare model ID deterministically resolves to its lowest-peak-RAM feasible sibling.

Capability values and sources below were checked **2026-07-21**. MMLU, MMLU-Pro, and
MMLU-Redux are different evaluations and their raw values are not interchangeable. Missing
GSM8K is reported as unknown, never as zero or an estimate.

**Which published metric ranks candidates is decided by the TASK, not by this file** (added
2026-08-19). `filter_pool_by_task` reads `TaskSpec.model_ranking_metric`: `GSM8K` for `gsm8k`,
`MMLU` (falling back to MMLU-Pro) for the three classification tasks, and **`None`** for
`xlam_bfcl`, `calendar_json`, `ner_bc5cdr` and `dialogsum` — which says plainly that no published
benchmark in this pool is a fair proxy for function calling, span extraction or summarisation, so
the deterministic resource ordering is kept rather than a proxy being invented. A metric ranks only
when every candidate carries a present measurement with an identical metric/mode/protocol key;
mixed metrics or any missing value fall back to resource order. Note the interaction with the table
above: **no pool entry reports GSM8K**, so `gsm8k`'s ranking metric currently has no effect and
candidates fall back to resource order — which is the honest outcome, not a gap to paper over.

---

## The models

| Model | Params | Type | Ctx | GSM8K (%) | Named knowledge metrics (%) | Variant tier span |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 0.6B | text | 32K | not reported | not reported (Table-8 Base proxies are not attached) | 0 (Q4) → 2 (BF16) |
| `Qwen/Qwen3-1.7B` | 1.7B | text | 32K | not attached to this artifact/mode | MMLU-Pro 40.2; MMLU-Redux 64.4 (non-thinking) | 1 → 3 |
| `Qwen/Qwen3-4B-Instruct-2507` | 4B | text | 256K | not reported | MMLU-Pro 69.6; MMLU-Redux 84.2 (non-thinking) | 3 |
| `Qwen/Qwen3.5-0.8B` | 0.8B | multimodal | 262K | not reported | MMLU-Pro 29.7; MMLU-Redux 48.5 (non-thinking) | 0 → 2 |
| `Qwen/Qwen3.5-2B` | 2B | multimodal | 262K | not reported | MMLU-Pro 55.3; MMLU-Redux 69.2 (non-thinking) | 2 → 3 |
| `Qwen/Qwen3.5-4B` | 4B | multimodal | 262K | not reported | MMLU-Pro 79.1; MMLU-Redux 88.8 (mode not specified by card table) | 3 |

Every tier is populated after quant expansion, so escalation can walk tiers 0→3 on a
large-enough device.

---

## Which to use for what

**`Qwen3-0.6B`.** Smallest text entry. Its Q4_K_M variant is Tier 0; Q8/BF16 variants are
higher. `smallest_first` starts from the smallest feasible deployment variant. Use
non-thinking mode for short classification/NER output contracts.

**`Qwen3-1.7B`.** Text-only variants span Tiers 1–3. The Qwen3.5 comparison table supplies
its like-for-like non-thinking MMLU-Pro/Redux values. A separate report GSM8K row is not
attached because it does not identify this exact post-trained artifact/mode.

**`Qwen3-4B-Instruct-2507`.** Non-thinking text model with direct answers and 256K context.
All configured variants are Tier 3. Its card does not report GSM8K.

**`Qwen3.5-0.8B`.** Multimodal entry whose variants span Tiers 0–2. This project tunes only
the language path through `FastVisionModel`.

**`Qwen3.5-2B`.** Multimodal entry spanning Tiers 2–3. Fine-tuning uses the base
Transformers repository, never a `-GGUF` repository.

**`Qwen3.5-4B`.** Largest Qwen3.5 entry in this pool. All configured variants are Tier 3;
no cross-metric claim is made against the text-only 4B entry.

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
  Implemented to Unsloth's documented Qwen3.5 recipe.
- **Tier labels are quant/peak-RAM specific.** They do not assert that every quant variant of
  a base model has the same footprint or capability.
- **Target mode is explicitly non-thinking.** HF training and inference pass
  `enable_thinking=False` into the tokenizer chat template. GGUF evaluation passes the same
  template kwarg when llama-cpp-python supports it; installed version 0.3.34 does not, so
  hybrid Qwen3/Qwen3.5 GGUFs use the verified ChatML prefix with an empty
  `<think></think>` block. Non-thinking-only Qwen3-4B-Instruct-2507 uses its plain
  assistant prefix without think tags. Unknown non-Qwen templates fail rather than silently
  changing modes.
- **Quantized identities receive quantized scores.** Q4/Q8 zero-shot baselines,
  interpolation probes, downward probes, and enabled fine-tuned quant evaluation build/reuse
  the exact GGUF and pass it to the eval harness; they are never credited a BF16 score.
- **Interpolation fits deployment footprint.** Its x-axis is log on-disk weight size,
  duplicate footprints are averaged, and fewer than two unique footprints trigger a
  deterministic size-target fallback instead of an unstable fit. (It previously fit
  against a modelled peak RAM; see below.)
- **No modelled throughput or peak RAM (2026-07-27).** The pool carries no estimated
  tok/s and no estimated peak inference RAM. Those were per-model guesses rendered in the
  same shape as measurements, and the gates consumed them as facts — the one checked
  against reality was 20% off. Real numbers live in `config/measured_metrics.json`
  (written by `hardware_eval/measure_model.py`); anything unmeasured reports
  `UNMEASURED` and does not gate. Memory additionally enforces a hard floor: real weight
  bytes must fit in RAM. Ordering and tiering use real on-disk size.

---

## Sources (checked 2026-07-21)
- Qwen3 Technical Report — [arXiv 2505.09388](https://arxiv.org/abs/2505.09388)
- [Qwen3.5-0.8B official card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Qwen3.5-2B official card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3.5-4B official card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3-4B-Instruct-2507 official card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- Unsloth Qwen3.5 fine-tuning guide (FastVisionModel text-only recipe) — unsloth.ai/docs/models/qwen3.5/fine-tune
