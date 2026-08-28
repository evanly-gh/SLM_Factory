# On-device model pool — sourced capability descriptions

Effective date: **2026-07-21**. These offline notes are injected into initial orchestrator
choice, escalation, and downward model-choice prompts.

> **METRIC-COMPARABILITY CAVEAT:** MMLU, MMLU-Pro, and MMLU-Redux are distinct
> evaluations with different datasets and protocols. Their raw scores are not directly
> comparable. Compare only the same named metric under the same mode/protocol. A benchmark
> marked **not reported** is unknown, not zero; do not infer it from another model or metric.
>
> **TIER CAVEAT:** tiers are ON-DISK WEIGHT-SIZE buckets for a specific deployed quant
> variant (Q4_K_M, Q8_0, or BF16), not intrinsic model-size or capability classes. One base
> model can occupy several tiers. Renumbered **2026-08-24** from 0-3 to **1-5**, with a new
> boundary at 300 MB: tier 1 <300 MB, tier 2 300-750, tier 3 750-1500, tier 4 1500-2500,
> tier 5 >=2500. Any "Tier N" written below refers to the CURRENT numbering.
>
> **TIER IS THE ESCALATION STEP.** `smallest_first` enters the lowest feasible tier and asks
> you to pick the best model WITHIN it; escalation then promotes a whole tier at a time. So a
> choice you make here is a choice among near-equal-cost options, and getting it wrong costs
> iterations rather than the run.

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
The smallest text model in the pool. Its Q4_K_M variant sits in Tier 2 and its
Q8_0/BF16 variants in Tier 3. Qwen3 Technical Report Table 8 values are
base/proxy measurements and are not attached to this post-trained artifact; MMLU and GSM8K
are both **not reported** in the live pool. This model supports Qwen3's
thinking/non-thinking behavior; the pipeline explicitly renders target prompts with thinking
disabled.

## Qwen/Qwen3-1.7B  (text, ~1.7B)
The Q4_K_M, Q8_0, and BF16 variants span Tiers 3–5. The Qwen3.5
comparison table reports non-thinking MMLU-Pro **40.2** and MMLU-Redux **64.4** for this
model. The earlier Qwen3 report lists a GSM8K value but does not identify this exact
post-trained artifact/mode, so GSM8K is **not reported** for the pool entry.

## Qwen/Qwen3-4B-Instruct-2507  (text, ~4B)
A text-only, non-thinking instruct model; all configured quant variants fall in peak-RAM
Tiers 4–5. Its official card reports non-thinking MMLU-Pro **69.6** and MMLU-Redux **84.2**.
GSM8K is **not reported** on that card, so no GSM8K value is attached to this pool entry.

## Qwen/Qwen3.5-0.8B  (multimodal, ~0.8B)
A multimodal model fine-tuned text-only in this project through `FastVisionModel` with the
vision layers frozen. Its quant variants span Tiers 2–4. The official
card reports non-thinking MMLU-Pro **29.7** and MMLU-Redux **48.5**. GSM8K is **not
reported**.

## Qwen/Qwen3.5-2B  (multimodal, ~2B)
A multimodal model using the base Transformers repository for text-only LoRA. Its configured
variants occupy Tiers 3–5. The official card reports non-thinking MMLU-Pro **55.3**
and MMLU-Redux **69.2**. GSM8K is **not reported**.

## Qwen/Qwen3.5-4B  (multimodal, ~4B)
The largest Qwen3.5 pool entry, fine-tuned text-only with the vision layers frozen; all
configured quant variants fall in Tier 5. The official card table reports
MMLU-Pro **79.1** and MMLU-Redux **88.8**, but that table does not explicitly label these
rows as non-thinking. Their mode is therefore recorded as **not specified by source** and
their protocol is not compared with the small-model non-thinking table. GSM8K is **not
reported**.

---

# Sub-billion tier (added 2026-08-24) — non-Qwen

These three are **not** Qwen and were added when the pool's Qwen-only policy was relaxed to
reach below 462 MB, which had been the floor because Qwen ships nothing smaller than 0.6B.

> **A DIFFERENT KIND OF NUMBER.** Every measurement below is **ours**, from slurm
> 38765131 / 38765655: one fixed LoRA fit (r=16, alpha=32, lr=2e-4, 3 epochs) on 3,000 gold
> rows, scored on 300 held-out rows of our own eval sets. They are **task scores on the tasks
> this pipeline runs**, not MMLU-family benchmarks, and they are **not comparable** to the
> MMLU-Pro / MMLU-Redux values on the Qwen entries above. Compare them only with each other.
> Each is quoted at Q4_K_M — the deployed artifact — with the bf16 score in brackets.
>
> **THEY ARE SINGLE FITS.** The noise floor of this pipeline is unmeasured. Treat large gaps
> (0.10 vs 0.45) as real and small ones (0.61 vs 0.69) as undecided.

## HuggingFaceTB/SmolLM2-360M-Instruct  (text, ~0.36B)
**The strongest sub-billion entry and the default choice below Qwen3-0.6B.** All quant
variants sit in Tiers 1–2 (258 MB at Q4_K_M, Tier 1). `LlamaForCausalLM`; ~315M of its 362M parameters
are transformer, only ~47M embedding (49k vocab).

Measured: `ner_bc5cdr` span-F1 **0.7339** [bf16 0.7339] · `xlam_bfcl` ast_arg_match **0.4500**
[bf16 0.4500]. It is the **only** pool entry measured as lossless under Q4_K_M on both tasks.
On BC5CDR it beats the Qwen3.6-35B teacher's five-shot 0.7190 while being ~97x smaller.

Prefer this over `google/gemma-3-270m-it` whenever the task requires composing structured
output, despite Gemma being 17 MB smaller.

## google/gemma-3-270m-it  (text, ~0.27B)
**Task-dependent — strong on extraction, near-floor on function calling.** All quant variants
in Tiers 1–2 (241 MB at Q4_K_M, Tier 1 — the third-smallest entry in the pool). `Gemma3ForCausalLM`, and
unusually shaped: 168M of its 270M parameters are a 262k-token embedding table, leaving only
~102M of transformer.

Measured: `ner_bc5cdr` span-F1 **0.6529** [bf16 0.6926] · `xlam_bfcl` ast_arg_match **0.1000**
[bf16 0.0567]. The large vocabulary appears to help on rare-token extraction (biomedical
entity names) and the thin transformer appears to hurt on compositional tool calls; that
reading is a hypothesis, not an established result.

**Selection caveat:** it is smaller on disk than SmolLM2-360M-Instruct and will therefore be
reached first by size-ordered selection. On a structured-output task that costs ~0.35. If the
task is extraction or classification, it is a legitimate cheapest-first pick.

## HuggingFaceTB/SmolLM2-135M-Instruct  (text, ~0.135B)
**The pool floor: 101 MB at Q4_K_M**, 4.6x below Qwen3-0.6B and the smallest artifact this
project has produced. All three quant variants in Tier 1. `LlamaForCausalLM`; ~107M transformer,
~28M embedding.

Measured: `ner_bc5cdr` span-F1 **0.5476** [bf16 0.6107] · `xlam_bfcl` ast_arg_match **0.1867**
[bf16 0.2300]. Notably it **outperforms gemma-3-270m-it on function calling** despite half the
total size, which is why non-embedding capacity rather than parameter count is the useful way
to read this tier.

**Most quantization-sensitive entry in the pool** (-0.06 span-F1, -0.04 ast_arg_match from
bf16 to Q4_K_M): at 100 MB there is little redundancy left to discard. Prefer Q8_0 (145 MB) if
the storage budget allows.
