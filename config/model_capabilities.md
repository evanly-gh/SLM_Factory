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

> **NO SELF-MEASUREMENT.** Nothing in this file may quote a score this project produced on a
> benchmark this project evaluates. Selection evidence must be PUBLISHED — a model card, its
> paper, or a public leaderboard — because a number we measured on our own eval set is the
> answer to the question the run is about to ask. Removed 2026-08-30; see the sub-billion
> section and `CapabilityMeasurement.published` in `config/android_pool.py`.

Official sources checked on the effective date:

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [SmolLM2 paper (arXiv:2502.02737)](https://arxiv.org/abs/2502.02737)
- [Gemma 3 Technical Report (arXiv:2503.19786)](https://arxiv.org/abs/2503.19786)
- [Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

Measurements injected into prompts retain the exact artifact, mode, protocol identifier, and
source URL. Pool entries use stable selectors such as
`Qwen/Qwen3-1.7B@Q4_K_M`; the quant suffix identifies the deployment artifact, while
capability measurements remain tied to the named source model.

> **COVERAGE (extended 2026-08-30).** Every entry that publishes them now carries the same four
> axes — MMLU-Pro and MMLU-Redux (knowledge), SuperGPQA (hard reasoning) and IFEval (instruction
> following) — so a choice between two Qwen models is made on the same evidence in both cases.
> **IFEval is reported for 8 of the 9 pool entries**, which makes it the one axis that spans the
> whole pool including the non-Qwen sub-billion three.
>
> It is not monotonic in size, and that is the point of publishing it: `Qwen3-1.7B` scores **68.2**
> against `Qwen3.5-2B`'s **61.2** despite being the smaller model. Do not infer instruction
> following from parameter count.
>
> `Qwen/Qwen3-0.6B` is the one entry with NO published benchmark of any kind — see its section.

## Qwen/Qwen3-0.6B  (text, ~0.6B)
The smallest text model in the pool. Its Q4_K_M variant sits in Tier 2 and its
Q8_0/BF16 variants in Tier 3. **Its model card publishes no benchmark table at all**, and the
Qwen3 Technical Report Table 8 values are base/proxy measurements not attached to this
post-trained artifact. So every benchmark is **not reported** for this entry — unknown, not
zero, and not to be ranked below a model that has numbers. This model supports Qwen3's
thinking/non-thinking behavior; the pipeline explicitly renders target prompts with thinking
disabled.

## Qwen/Qwen3-1.7B  (text, ~1.7B)
The Q4_K_M, Q8_0, and BF16 variants span Tiers 3–5. Non-thinking, from the Qwen3.5 comparison
table: MMLU-Pro **40.2** · MMLU-Redux **64.4** · SuperGPQA **21.0** · IFEval **68.2**.

Note the IFEval: it is the **best of any sub-4B entry in the pool**, ahead of the larger
Qwen3.5-2B. The earlier Qwen3 report lists a GSM8K value but does not identify this exact
post-trained artifact/mode, so GSM8K is **not reported** for the pool entry.

## Qwen/Qwen3-4B-Instruct-2507  (text, ~4B)
A text-only, non-thinking instruct model; all configured quant variants fall in peak-RAM
Tiers 4–5. Non-thinking: MMLU-Pro **69.6** · MMLU-Redux **84.2** · SuperGPQA **42.8** ·
IFEval **83.4**. GSM8K is **not reported** on that card, so no GSM8K value is attached.

## Qwen/Qwen3.5-0.8B  (multimodal, ~0.8B)
A multimodal model fine-tuned text-only in this project through `FastVisionModel` with the
vision layers frozen. Its quant variants span Tiers 2–4. Non-thinking: MMLU-Pro **29.7** ·
MMLU-Redux **48.5** · SuperGPQA **16.9** · IFEval **52.1**. GSM8K is **not reported**.

## Qwen/Qwen3.5-2B  (multimodal, ~2B)
A multimodal model using the base Transformers repository for text-only LoRA. Its configured
variants occupy Tiers 3–5. Non-thinking: MMLU-Pro **55.3** · MMLU-Redux **69.2** ·
SuperGPQA **30.4** · IFEval **61.2**. GSM8K is **not reported**.

Its IFEval is **below the smaller Qwen3-1.7B's 68.2** while its knowledge and reasoning scores
are well above. On a format-bound task that trade may not favour the larger model.

## Qwen/Qwen3.5-4B  (multimodal, ~4B)
The largest Qwen3.5 pool entry, fine-tuned text-only with the vision layers frozen; all
configured quant variants fall in Tier 5. Its card table reports MMLU-Pro **79.1** ·
MMLU-Redux **88.8** · SuperGPQA **52.9** · IFEval **89.8** — the strongest entry in the pool on
all four.

**Mode caveat:** that table does not explicitly label these rows as non-thinking, and it
compares against models named `-Thinking`. Their mode is therefore recorded as **not specified
by source** under a separate protocol id, so they are NOT rank-compared with the small-model
non-thinking table above. GSM8K is **not reported**.

---

# Sub-billion tier (added 2026-08-24) — non-Qwen

These three are **not** Qwen and were added when the pool's Qwen-only policy was relaxed to
reach below 462 MB, which had been the floor because Qwen ships nothing smaller than 0.6B.

> **PUBLISHED FIGURES ONLY (changed 2026-08-30).** These three entries used to carry OUR OWN
> scores on `ner_bc5cdr` and `xlam_bfcl` from probe 38765131 / 38765655. Those were removed.
> Measuring a model offline on the exact eval set a run is about to be scored on, then feeding
> the result back as a selection hint, decides the experiment before it starts — on one
> `calendar_json` run the orchestrator justified its choice by quoting our xlam number. What
> remains below is what a practitioner choosing a model would actually have: the model card.
>
> **CROSS-FAMILY COMPARISON IS NOT SAFE HERE.** The SmolLM2 numbers come from the SmolLM2 card
> (lighteval, zero-shot unless a row says otherwise); the Gemma numbers come from the Gemma 3
> card, which does not state its harness. The two SmolLM2 entries ARE comparable with each
> other. `SmolLM2 ARC (average of easy/challenge)` and `Gemma ARC-c` are different subsets and
> must not be ranked against one another.

## HuggingFaceTB/SmolLM2-360M-Instruct  (text, ~0.36B)
**The strongest sub-billion entry on published reasoning and knowledge.** All quant variants
sit in Tiers 1–2 (258 MB at Q4_K_M, Tier 1). `LlamaForCausalLM`; ~315M of its 362M parameters
are transformer, only ~47M embedding (49k vocab). 4T pretraining tokens, SFT + DPO.

Published (SmolLM2 card): MMLU cloze **32.8** · IFEval **41.0** · BBH 3-shot **27.3** ·
ARC avg **43.7** · HellaSwag **52.1** · GSM8K 5-shot **7.43**.

Its card credits SmolLM2 with advances in "instruction following, knowledge, reasoning" over
SmolLM1, and it leads the sub-billion three on every one of those except instruction
following, where Gemma is ahead.

## google/gemma-3-270m-it  (text, ~0.27B)
**Best instruction following of the three, weakest commonsense.** All quant variants in
Tiers 1–2 (241 MB at Q4_K_M, Tier 1 — the third-smallest entry in the pool).
`Gemma3ForCausalLM`, and unusually shaped: 168M of its 270M parameters are a 262k-token
embedding table, leaving only ~102M of transformer. 32K context, 6T pretraining tokens.

Published (Gemma 3 card): IFEval **51.2** · BBH few-shot **26.7** · ARC-c **28.2** ·
HellaSwag **37.7** · WinoGrande **52.3**. **MMLU and GSM8K are not reported** for the 270M
at all — absent, not zero.

Google positions the 270M as a **fine-tuning base rather than a general assistant**, which is
consistent with the shape of these numbers: high IFEval (it follows a format) against low
HellaSwag (it knows little). Its 262k vocabulary is the reason it quantizes unusually flatly.

**Selection caveat:** it is smaller on disk than SmolLM2-360M-Instruct and will therefore be
reached first by size-ordered selection, on a 17 MB difference.

## HuggingFaceTB/SmolLM2-135M-Instruct  (text, ~0.135B)
**The pool floor: 101 MB at Q4_K_M**, 4.6x below Qwen3-0.6B and the smallest artifact this
project has produced. All three quant variants in Tier 1. `LlamaForCausalLM`; ~107M
transformer, ~28M embedding. 2T pretraining tokens, SFT + DPO.

Published (SmolLM2 card, same table as the 360M so the two are directly comparable):
MMLU cloze **29.3** · IFEval **29.9** · BBH 3-shot **28.2** · ARC avg **37.3** ·
HellaSwag **40.9** · GSM8K 5-shot **1.4**.

Uniformly below its 360M sibling except on BBH, where the two are level (28.2 vs 27.3) — and
both sit near the ~25% random baseline of a multiple-choice suite, so neither number should be
read as reasoning ability.

MT-Bench is deliberately omitted for both SmolLM2 entries: the card reports 19.8 here and 3.66
for the 360M, which cannot both be on the 0–10 MT-Bench scale, and guessing which is
mis-scaled would put an invented number in front of the orchestrator.
