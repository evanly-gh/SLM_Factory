# On-device model pool — sourced capability descriptions

Effective date: **2026-07-21**. These offline notes are injected into initial orchestrator
choice, escalation, and downward model-choice prompts.

> **METRIC-COMPARABILITY CAVEAT:** MMLU, MMLU-Pro, and MMLU-Redux are distinct
> evaluations with different datasets and protocols. Their raw scores are not directly
> comparable. Compare only the same named metric under the same mode/protocol. A benchmark
> marked **not reported** is unknown, not zero; do not infer it from another model or metric.
>
> **TIER CAVEAT:** tiers are peak-RAM buckets for a specific deployed quant variant
> (Q4_K_M, Q8_0, or BF16), not intrinsic model-size or capability classes. One base model
> can occupy several tiers.

Official sources checked on the effective date:

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

Measurements injected into prompts retain the exact artifact, mode, protocol identifier, and
source URL. Pool entries use stable selectors such as
`Qwen/Qwen3-1.7B@Q4_K_M`; the quant suffix identifies the deployment artifact, while
capability measurements remain tied to the named source model.

## Qwen/Qwen3-0.6B  (text, ~0.6B)
The smallest text model in the pool. Its Q4_K_M variant is configured in peak-RAM Tier 0;
larger quant variants occupy higher tiers. Qwen3 Technical Report Table 8 values are
base/proxy measurements and are not attached to this post-trained artifact; MMLU and GSM8K
are both **not reported** in the live pool. This model supports Qwen3's
thinking/non-thinking behavior; the pipeline explicitly renders target prompts with thinking
disabled.

## Qwen/Qwen3-1.7B  (text, ~1.7B)
The Q4_K_M, Q8_0, and BF16 variants span configured peak-RAM Tiers 1–3. The Qwen3.5
comparison table reports non-thinking MMLU-Pro **40.2** and MMLU-Redux **64.4** for this
model. The earlier Qwen3 report lists a GSM8K value but does not identify this exact
post-trained artifact/mode, so GSM8K is **not reported** for the pool entry.

## Qwen/Qwen3-4B-Instruct-2507  (text, ~4B)
A text-only, non-thinking instruct model; all configured quant variants fall in peak-RAM
Tier 3. Its official card reports non-thinking MMLU-Pro **69.6** and MMLU-Redux **84.2**.
GSM8K is **not reported** on that card, so no GSM8K value is attached to this pool entry.

## Qwen/Qwen3.5-0.8B  (multimodal, ~0.8B)
A multimodal model fine-tuned text-only in this project through `FastVisionModel` with the
vision layers frozen. Its quant variants span configured peak-RAM Tiers 0–2. The official
card reports non-thinking MMLU-Pro **29.7** and MMLU-Redux **48.5**. GSM8K is **not
reported**.

## Qwen/Qwen3.5-2B  (multimodal, ~2B)
A multimodal model using the base Transformers repository for text-only LoRA. Its configured
variants occupy peak-RAM Tiers 2–3. The official card reports non-thinking MMLU-Pro **55.3**
and MMLU-Redux **69.2**. GSM8K is **not reported**.

## Qwen/Qwen3.5-4B  (multimodal, ~4B)
The largest Qwen3.5 pool entry, fine-tuned text-only with the vision layers frozen; all
configured quant variants fall in peak-RAM Tier 3. The official card table reports
MMLU-Pro **79.1** and MMLU-Redux **88.8**, but that table does not explicitly label these
rows as non-thinking. Their mode is therefore recorded as **not specified by source** and
their protocol is not compared with the small-model non-thinking table. GSM8K is **not
reported**.
