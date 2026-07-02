# Android Model Pool: Research Basis and Design Rationale

This document records the research behind `android_pool.py` for future reference. It covers what has actually been run on Android hardware, why the pool is structured the way it is, the justification for each individual model, and SLM Factory-specific considerations.

---

## 1. What Has Been Run on Android and What Happened

### Confirmed deployments (peer-reviewed or primary-source evidence)

All numbers below are **CPU-bound Q4_K_M GGUF via llama.cpp/Ollama on Termux unless noted.** Decode throughput = sustained tok/s after first token. Peak RAM = model weights + KV cache at 2K context + runtime overhead.

| Model | Size | Chipset | Device | Decode tok/s | Peak RAM | Source |
|---|---|---|---|---|---|---|
| Llama 3.2 3B (Q4_K_M) | 2.02 GB | Snapdragon 750G | OnePlus Nord CE 5G | 4–6 tok/s | ~3.4 GB | arXiv 2512.06490 |
| Llama 3.2 3B (Q4_K_M) | 2.02 GB | Snapdragon 8 Gen 3 | Various flagships | 8–12 tok/s | ~3.4 GB | arXiv 2410.03613 |
| Llama 3.2 1B (SpinQuant INT4) | 0.66 GB | Snapdragon 8 Gen 3 | OnePlus 12 | **50.2 tok/s** (CPU, ExecuTorch) | ~900 MB | Meta AI Blog / Arm Blog |
| Llama 3.2 1B (SpinQuant INT4) | 0.66 GB | Snapdragon 8 Gen 3 | Samsung S24+ | 40+ tok/s, **>350 tok/s prefill** | ~900 MB | PyTorch ExecuTorch Blog |
| Llama 2 7B Chat (W4A16) | 4.2 GB | Snapdragon 8 Gen 3 | Samsung Galaxy S24 | **12.85 tok/s** (Hexagon NPU/QNN) | ~6 GB | Qualcomm AI Hub model card |
| Llama 2 7B (Q4_0) | ~4 GB | Snapdragon 888 | Samsung S21 Ultra | 3.2 tok/s (3B proxy) | ~5 GB | Grokipedia community benchmark |
| Qwen 2.5 1.5B (Q4) | ~1 GB | Snapdragon 8 Gen 3 | Samsung S24 Ultra | ~10 tok/s (sustained, throttled) | ~1.3 GB | arXiv 2603.23640 |
| Gemma 3 1B IT (Q4) | 0.81 GB | Snapdragon 8 Gen 3 | Various | ~32 tok/s | ~1.05 GB | Google LiteRT blog / Arm |
| BlueLM-V-3B (2.7B, INT4) | ~1.9 GB | Dimensity 9300 NPU | vivo X100 | **24.4 tok/s** | ~2.5 GB | arXiv 2411.10640 (CVPR 2025) |
| MobileLLM-1B | ~0.7 GB | iPhone 14 (A15) | iPhone 14 | ~13 tok/s | ~1.2 GB | arXiv 2406.10290 (MobileAIBench) |
| Phi-2 2.7B (Q4) | ~1.7 GB | iPhone 14 (A15) | iPhone 14 | ~13 tok/s | ~2 GB | arXiv 2406.10290 |

**Notes on Android-specific behavior:**
- **Thermal throttling is severe**: Samsung S24 Ultra GPU drops from 680 MHz to 231 MHz after ~6 inference iterations (arXiv 2603.23640). Sustained throughput degrades 40–60% after 90 seconds of continuous inference.
- **6 GB phone RAM ceiling**: Llama 2-7B Q4 requires ~3.8 GB RAM (arXiv 2410.03613). On a 6 GB phone with ~1.5 GB OS overhead, only ~1.5 GB is free — making 3B Q4 tight and 7B impossible.
- **mmap on Android**: llama.cpp uses memory-mapped files. Pages are initially not resident in RAM but get faulted in during inference, so peak RAM ≈ model file size. The OS can evict file-backed pages under memory pressure without killing the process; heap-allocated weights cannot be evicted and cause OOM kills.
- **MLC-LLM GPU path fails on Android**: MLC-LLM's OpenCL backend on Adreno GPUs achieves <5% ALU utilization and is slower at prefill than llama.cpp on CPU (arXiv 2410.03613). Thermal throttling also kills GPU frequency quickly. **llama.cpp CPU path is the reliable default.**
- **NPU requires vendor SDK**: Hexagon NPU (Qualcomm) achieves 12.85 tok/s for 7B W4A16 via QNN on SD8 Gen3 — but this requires Qualcomm AI Hub export and Genie runtime, not llama.cpp.

### Accuracy benchmarks under quantization (Q4_K_M vs FP16)

INT4 hurts smaller models disproportionately. Empirical data from arXiv 2505.02214 (Qwen3 family):

| Model | FP16 MMLU | AWQ INT4 MMLU | Drop |
|---|---|---|---|
| Qwen3-0.6B | 52.3% | 43.8% | **−8.5 pp** |
| Qwen3-1.7B | 60.0% | 52.5% | **−7.5 pp** |
| Qwen3-8B | 74.7% | 69.3% | **−5.4 pp** |
| Qwen3-14B | 80.7% | 77.1% | **−3.6 pp** |

Q4_K_M GGUF (weight-only, mixed precision) is significantly better than naive Q4_0:
- Q4_K_M: +0.054 PPL delta vs FP16 (7B reference, llama.cpp official)
- Q4_0: +0.250 PPL delta — **4.7x worse quality at only 8.6% smaller file size**

The scaling law for quantization-induced degradation (arXiv 2411.17691, ACL 2025):
`ΔqLoss = 0.017 × D^0.525 / (N^0.226 × P^5.497)`
Halving model size (N) increases QiD by ~16%. More thoroughly pre-trained models (higher D/N ratio) degrade more under quantization — meaning modern well-trained small models are harder to quantize than their older counterparts at the same size.

**Practical rule**: Q4_K_M is the correct default. Prefer QAT (quantization-aware training) variants for sub-1B models when available. Q5_K_M is recommended when storage is not the binding constraint.

---

## 2. Tier Stratification Justification

### Why 4 tiers instead of 3 or 5

Capability scales as a **power law** (not step-functions) in perplexity/loss across the 0.5–3B range (Kaplan et al. 2020, Chinchilla 2022). However, downstream task accuracy — particularly on multi-step reasoning benchmarks — shows near-step-function jumps at two specific points that justify tier boundaries.

**GSM8K (grade-school math, multi-step arithmetic) is the stratification benchmark** because it is the most sensitive indicator of reasoning capability at small scales: it shows the sharpest capability cliffs while being concrete enough to be reproducible. MMLU is used as a secondary signal for general knowledge breadth.

### The two step-functions

**Step 1: Tier 0 → Tier 1 (sub-0.6B → 0.6B+)**

The largest capability jump in the pool:
- Qwen2.5-0.5B GSM8K: **41.6%**
- Qwen3-0.6B GSM8K: **59.6%** (+18 pp)

At sub-0.6B, models cannot reliably execute multi-step reasoning chains. Passing individual classification labels or named entity spans is feasible; generating coherent multi-sentence reasoning is not. This is the binary threshold between "routing/extraction" and "generation."

Scaling exponents are steeper at small sizes (~0.23 at sub-20M params, decaying to ~0.10 near 3B per arXiv 2603.07365). Every parameter doubling yields larger relative gains at small scale, which is why the 0.5B→0.6B jump is the most cost-effective in the pool.

**Step 2: Tier 1 → Tier 2 (1B → 1.5–1.7B)**

A second meaningful jump:
- Llama 3.2-1B GSM8K: **~44%**
- Qwen3-1.7B GSM8K: **75.4%** (+31 pp)
- MMLU: ~49% → ~63% (+14 pp)

The 1.7B tier reliably executes reasoning chains, follows complex instructions, and generates coherent longer outputs. This is where the model transitions from "works on simple tasks" to "works on most SLM Factory target tasks."

**Step 3: Tier 2 → Tier 3 (1.7B → 3B)**

A real but smaller gap:
- Qwen3-1.7B GSM8K: **75.4%**
- Llama 3.2-3B GSM8K: **77.7%** (+2.3 pp)

The 3B tier is only marginally better than 1.7B for most tasks. It matters only for hard math and complex code, and it costs 8GB+ RAM vs 6GB. Qwen3-1.7B effectively matches Qwen2.5-3B per the official Qwen3 technical report — the tier boundary here is driven by RAM requirements more than capability.

### Tier boundaries and RAM budget

Total RAM budget ≤ 3 GB (project requirement, matching typical mid-range Android constraint):

```
total_RAM ≈ int4_file_mb + 1000–1500 MB (Android OS + app + KV cache at 2K ctx)
```

| Tier | INT4 file range | Peak RAM range | Minimum phone RAM |
|---|---|---|---|
| 0 (Micro) | 295–310 MB | 460–480 MB | 4 GB (any Android) |
| 1 (Small) | 397–806 MB | 580–1,050 MB | 4 GB (any Android) |
| 2 (Mid) | 938–1,190 MB | 1,250–1,430 MB | 6 GB |
| 3 (Large) | 2,020–2,490 MB | 3,400–4,200 MB | 8 GB (some need 12 GB) |

---

## 3. Model-by-Model Justification

### Tier 0 — Micro (sub-0.6B)

**Use for:** Binary classification, keyword extraction, simple NER, routing decisions. Do not use for multi-step generation.

---

#### `Qwen/Qwen2.5-0.5B-Instruct`
- **INT4 size:** ~295 MB | **Peak RAM:** ~460 MB
- **Benchmarks:** GSM8K 41.6%, MMLU 47.5% (Qwen2.5 Technical Report, Table 5, arXiv 2412.15115)
- **Why included:** Strongest sub-0.5B model with published benchmarks. MATH score 19.5% beats Gemma2-2.6B (18.3%) at 5× fewer parameters — demonstrates the data-quality advantage of Qwen's math training.
- **Limitations:** Standard PTQ (not QAT), so quantization degradation is higher than MiniCPM4.

---

#### `openbmb/MiniCPM4-0.5B`
- **INT4 size:** ~310 MB | **Peak RAM:** ~480 MB
- **Benchmarks:** GSM8K ~55% (extrapolated), MMLU ~53% (extrapolated)
- **Why included:** Uses BitCPM4 QAT (quantization-aware training), which substantially reduces INT4 degradation compared to standard PTQ. At sub-0.6B where quantization hurts most, QAT can recover 4–8 MMLU points. Source: arXiv 2506.07900.
- **Special notes:** Sparse attention mechanism (InfLLM v2) enables 128K token context — uniquely useful for long-document classification at this size. Best deployed via MNN-LLM for speed.
- **Limitations:** GSM8K/MMLU scores are extrapolated; no official published benchmark table.

---

### Tier 1 — Small (0.6–1B)

**Use for:** Simple instruction following, summarization, NER, basic generation. GSM8K 53–63%.

---

#### `Qwen/Qwen3-0.6B`
- **INT4 size:** 397 MB | **Peak RAM:** ~580 MB
- **Benchmarks:** GSM8K 59.6%, MMLU 52.8% (Qwen3 Technical Report, Table 8, arXiv 2505.09388)
- **Why included:** Best Tier 1 model with fully published benchmarks. Hybrid thinking/non-thinking mode provides MATH-500 77.6% with thinking enabled — competitive with models twice its size.
- **Special notes:**
  - Hybrid thinking mode (like DeepSeek-R1 approach) — enables chain-of-thought reasoning at 0.6B
  - MNN-LLM gives 8.6x faster prefill than llama.cpp for this model
  - Qwen3-1.7B-Base matches Qwen2.5-3B-Base — the whole Qwen3 family punches above weight class
- **Limitations:** Text-only. 32K context.

---

#### `Qwen/Qwen3.5-0.8B`
- **INT4 size:** ~500 MB (estimated) | **Peak RAM:** ~670 MB
- **Benchmarks:** GSM8K and MMLU not officially published — uses newer MMLU-ProX/MAXIFE/WMT24++ suite. ~54% relative capability score vs 397B flagship.
- **Why included:** New (March 2026) Gated DeltaNet hybrid architecture with native multimodal support (text + images + video) and 262K context window. These capabilities are not available in any other Tier 1 model.
- **Special notes:**
  - **Natively multimodal** — only sub-1B model in the pool that handles images and video
  - **201 languages** — strongest multilingual coverage at this size
  - **262K context** — far larger than any other Tier 1 model (Qwen3-0.6B has 32K)
  - **⚠️ Code generation fragility**: documented 67% → 33% pass@1 collapse when few-shot examples are added (SaladCloud benchmarks). Do not use for code tasks with examples.
  - Thinking mode is OFF by default.
- **Limitations:** No apples-to-apples comparison with other pool models on GSM8K/MMLU. Cannot confirm it beats Qwen3-0.6B on reasoning.

---

#### `meta-llama/Llama-3.2-1B-Instruct`
- **INT4 size:** 658 MB | **Peak RAM:** ~900 MB
- **Benchmarks:** GSM8K ~53.5%, MMLU ~49%, IFEval 53.5% (Meta model card)
- **Why included:** ExecuTorch reference model with SpinQuant + KleidiAI. This achieves **50.2 tok/s decode on SD8 Gen3** (OnePlus 12) — the fastest confirmed 1B decode throughput in the pool. Also has the **largest fine-tuning gains from training** of any 1B model (Distil Labs benchmark), which matters for SLM Factory's core use case.
- **Special notes:**
  - ExecuTorch + SpinQuant + KleidiAI: 50.2 tok/s decode, >350 tok/s prefill on Samsung S24+
  - Most tunable 1B model — largest relative gains from fine-tuning (confirmed by Distil Labs 12-SLM benchmark)
  - Distilled from Llama 3.1-8B and 70B — better aligned than independently trained models
  - 128K context
- **Limitations:** Text-only. Weaker raw benchmarks than Qwen3-0.6B. Best used when ExecuTorch deployment or fine-tuning ROI is the priority.

---

#### `openbmb/MiniCPM5-1B`
- **INT4 size:** 688 MB (confirmed, openbmb/MiniCPM5-1B-GGUF on HuggingFace) | **Peak RAM:** ~920 MB
- **Benchmarks:** MATH-500 **91.6%**, HumanEval+ 78.7%, IFEval **80.4%**, τ²-Bench (agentic) 79.5%, AIME 2025 40.4%, LiveCodeBench v6 33.5%. OpenBMB aggregate score: **42.57** vs Qwen3-0.6B's 26.77 and LFM2.5-1.2B's 35.61.
- **Why included:** Comprehensively beats every other 1B model in the pool on math, code, and agentic tasks. IFEval 80.4% also beats SmolLM2-1.7B (56.7%), making it best-in-class on instruction following at the 1B tier. llama.cpp deployment is officially documented by OpenBMB.
- **Special notes:**
  - 131K context window
  - Hybrid thinking mode (similar to DeepSeek-R1 approach)
  - Best sub-2B model for agentic tool-use tasks (τ²-Bench 79.5%)
  - Deploy via MNN-LLM for fastest mobile prefill
- **Limitations:** GSM8K not separately published; math proxy from MATH-500. MMLU estimated from aggregate score.

---

#### `google/gemma-3-1b-it`
- **INT4 size:** 806 MB | **Peak RAM:** ~1,050 MB
- **Benchmarks:** GSM8K 62.8%, MMLU ~48% (Gemma 3 Technical Report, arXiv 2503.19786)
- **Why included:** Google's official QAT checkpoint — quantization-aware trained by Google, not post-training quantized. This directly reduces INT4 degradation. It is also the native LiteRT/MediaPipe deployment target: any Android app using Google's on-device AI SDK needs this model specifically.
- **Special notes:**
  - Google provides the QAT GGUF at `google_gemma-3-1b-it-qat-GGUF` — higher quality than standard PTQ Q4_K_M
  - Official LiteRT-LM and MediaPipe deployment path (the only Tier 1 model on Google's official on-device stack)
  - IFEval 80.2% — strongest instruction following of any Tier 1 model
  - **Natively multimodal** — supports image+text input (despite the 1B size)
  - 32K context
- **Limitations:** MMLU ~48% is lower than Qwen3-0.6B (52.8%) despite having more parameters — Gemma's training prioritizes instruction following and safety over raw benchmark scores.

---

### Tier 2 — Mid (1–2B)

**Use for:** Most SLM Factory target tasks. Multi-step reasoning, NER, code generation, complex instruction following. GSM8K 70–77%. Fits all 6GB+ Android phones.

---

#### `Qwen/Qwen2.5-1.5B-Instruct`
- **INT4 size:** 938 MB | **Peak RAM:** ~1,250 MB
- **Benchmarks:** GSM8K 73.2%, MMLU 58.4%, HumanEval 61.6% (instruct model card, arXiv 2412.15115)
- **Why included:** Lighter than Qwen3-1.7B while still strong. Best Tier 2 choice when RAM is tighter (saves ~130 MB vs Qwen3-1.7B). Strong NER and classification. HumanEval 61.6% is the highest code score of any Tier 2 model.
- **Special notes:**
  - Best Tier 2 code generation (HumanEval 61.6%)
  - Proven, stable model with extensive community fine-tuning history
  - Smaller and lighter than Qwen3-1.7B — better for 6GB phones with tight headroom
- **Limitations:** Text-only. Superseded on reasoning by Qwen3-1.7B.

---

#### `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- **INT4 size:** 958 MB | **Peak RAM:** ~1,270 MB
- **Benchmarks:** MATH-500 **83.9%**, AIME 2024 28.9%, LiveCodeBench 16.9% (DeepSeek-R1 paper, Table 4, arXiv 2501.12948)
- **Why included:** Uniquely strong on formal mathematical reasoning at this size — MATH-500 83.9% at 1.5B is extraordinary. Only model in the pool specifically suited for competition-math style tasks.
- **Special notes:**
  - Distilled from DeepSeek-R1 671B via SFT on 800K reasoning traces
  - **⚠️ SPECIALIZED REASONING ONLY** — do not use for classification, NER, or general chat. SFT distillation causes catastrophic forgetting of general skills (NLI accuracy: 81% → 16.5% in controlled experiments, arXiv 2507.00432)
  - MATH-500 83.9% rivals models 4–5× its size
  - Generates long reasoning chains — slower effective throughput per task than non-thinking models
  - LiveCodeBench 16.9% confirms weak code generation despite math strength
- **Limitations:** Text-only. MMLU not officially reported (MMLU likely ~58% from base Qwen2.5-Math-1.5B). Not suitable for general SLM Factory tasks — use only when the training target is math/formal reasoning.

---

#### `unsloth/Qwen3.5-2B-GGUF`
- **INT4 size:** ~1,350 MB (Q4_K_M estimated; verify at [huggingface.co/unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)) | **Peak RAM:** ~1,800 MB
- **Benchmarks:** GSM8K and MMLU not officially published — uses newer MMLU-ProX/MAXIFE/WMT24++ suite.
- **Why included:** Same Gated DeltaNet hybrid architecture as Qwen3.5-0.8B but at 2B parameters. Brings multimodal capability (text + images + video) and a **262K context window** to Tier 2 — 8× larger than Qwen3-1.7B's 32K. Unsloth's Dynamic 2.0 GGUF is confirmed working with llama.cpp, Ollama, and compatible tools. Thinking mode disabled by default; enable with `--chat-template-kwargs '{"enable_thinking":true}'`.
- **Special notes:**
  - **Natively multimodal** — text, image, video input
  - **262K context window** — largest in Tier 2; suited for long-document tasks
  - **201 languages** — best multilingual coverage in Tier 2
  - Unsloth Dynamic 2.0 quantization upcasts important layers to 8/16-bit for better quality than standard Q4_K_M
- **Limitations:** No classic GSM8K/MMLU scores available. Cannot directly compare with Qwen3-1.7B on standard benchmarks. Estimated scores are inferred from the architecture and benchmark suite aggregate.

---

#### `Qwen/Qwen3-1.7B`
- **INT4 size:** 1,050 MB | **Peak RAM:** ~1,380 MB
- **Benchmarks:** GSM8K 75.4%, MMLU 62.6% (Qwen3 Technical Report, Table 8, arXiv 2505.09388)
- **Why included:** Best overall Tier 2 model. Qwen3-1.7B-Base matches Qwen2.5-3B-Base on most benchmarks — a 43% parameter efficiency improvement. Hybrid thinking mode enables MATH-500 93.4% with thinking. Best Tier 2 for math, reasoning, code, and multilingual tasks.
- **Special notes:**
  - Matches Qwen2.5-3B on benchmarks — the most significant overperformer in the pool
  - Hybrid thinking/non-thinking mode
  - Strong multilingual (CJK training advantage) and code capabilities
  - MNN-LLM: 8.6x faster CPU prefill than llama.cpp
  - 32K context (128K with extended config)
- **Limitations:** Text-only.

---

#### `google/gemma-3n-e2b-it`
- **INT4 size:** ~1,300 MB | **Peak RAM:** ~2,200 MB
- **Benchmarks:** MMLU 60.1%, HumanEval 66.5%, MBPP 56.6%, Global-MMLU-Lite 59.0%. Beats Gemma3-1B on **9 out of 9** shared benchmarks.
- **Why included:** Substantially stronger than Gemma3-1B-IT despite similar branding — HumanEval 66.5% vs 41.5%, MMLU 60.1% vs ~48%. The MatFormer (Matryoshka Transformer) architecture enables 5B total parameters with only 2.3B active via Per-Layer Embedding caching, giving it near-3B capability at 2B memory cost. Only model in the pool that handles text, images, video, and audio natively.
- **Special notes:**
  - **MatFormer architecture** — nested model design means E4B contains a fully functional E2B sub-model; allows dynamic size selection at runtime
  - **Natively multimodal** — text, image, video, audio input
  - 50–80 tok/s on NPU (Qualcomm/MediaTek partnerships confirmed)
  - Official LiteRT + MediaPipe + Android Studio deployment path
  - **⚠️ Proprietary license** — not Apache 2.0. Check licensing terms before commercial use.
- **Limitations:** GSM8K not directly published for E2B (E4B ~83%). Larger RAM footprint (2.2 GB peak) than other Tier 2 models — effectively sits at the top of Tier 2 / bottom of Tier 3.

---

#### `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **INT4 size:** 1,060 MB | **Peak RAM:** ~1,350 MB
- **Benchmarks:** GSM8K 48.8%, MMLU ~52%, IFEval **56.7%** (SmolLM2 paper, arXiv 2502.02737)
- **Why included:** Best Tier 2 model for instruction following. IFEval 56.7% beats both Llama 3.2-1B (53.5%) and Qwen2.5-1.5B (47.4%). Strong for classification tasks. Trained on 11T tokens with a multi-stage data curriculum emphasizing language quality over math/code.
- **Special notes:**
  - Leads the pool on IFEval — best for structured instruction-following tasks
  - ARC average 60.5% — strong commonsense reasoning
  - Compact training corpus makes it more predictable to fine-tune
  - **⚠️ Math weakness**: GSM8K 48.8% is significantly lower than Qwen3-1.7B (75.4%). Not suitable for math tasks.
  - 8K context
- **Limitations:** Text-only. Weaker on math/code than Qwen peers at same size.

---

### Tier 3 — Large (2–4B)

**Use for:** Hard math, complex code, tasks requiring near-7B capability. Requires 8GB+ RAM phone. GSM8K 77–88%.

---

#### `meta-llama/Llama-3.2-3B-Instruct`
- **INT4 size:** 2,020 MB Q4_K_M (confirmed, hugging-quants HF repo) | **Peak RAM:** ~3,400 MB
- **Benchmarks:** GSM8K 77.7%, MMLU 63.4%, ARC-C 78.6%, IFEval 77.4% (Meta model card)
- **Why included:** ExecuTorch reference model for the 3B class. Best deployment path for production Android apps targeting flagship devices. IFEval 77.4% is the highest in the pool — best instruction-following at 3B+.
- **Special notes:**
  - ExecuTorch + SpinQuant + KleidiAI: best decode latency of any 3B model on Android
  - Distilled from Llama 3.1-8B and 70B
  - 128K context
  - Tool use (BFCL) 67.0% — suitable for agentic tasks
  - Q3_K_M alternative (~1.5 GB) fits tighter storage budgets but incurs meaningful quality loss at 3B scale
- **Device requirement:** 8GB+ RAM phone (peak 3.4 GB with OS overhead).

---

#### `mistralai/Ministral-3B-Instruct`
- **INT4 size:** ~1,900 MB (estimated Q4_K_M) | **Peak RAM:** ~3,200 MB
- **Benchmarks:** MMLU ~65% (beats Llama-3.2-3B's 63.4%), strong chain-of-thought math reasoning (codersera.com / Azure model card)
- **Why included:** Unique differentiator — **256K context window**, the largest in the pool by 2× (others max at 128K). MMLU 65% edges out Llama-3.2-3B. Native function calling and structured JSON output built-in.
- **Special notes:**
  - **256K context** — uniquely suited for long-document classification, RAG, and multi-turn agentic tasks
  - Native function calling + JSON output — no prompt engineering required
  - Edge-optimized architecture (Grouped-Query Attention)
  - Official GGUF from `mistralai/Ministral-3-3B-Instruct-2512-GGUF` on HuggingFace
- **Limitations:** Text-only. GSM8K not specifically published. Slightly lower IFEval than Llama-3.2-3B.
- **Device requirement:** 8GB+ RAM phone.

---

#### `Qwen/Qwen3-4B`
- **INT4 size:** 2,390 MB | **Peak RAM:** ~4,200 MB
- **Benchmarks:** GSM8K ~87%, MMLU ~73% (Qwen3 Technical Report; matches Qwen2.5-72B on many benchmarks)
- **Why included:** Matches Qwen2.5-72B-Instruct on multiple benchmarks — the highest capability-per-parameter ratio in the pool. Officially supported on Qualcomm AI Hub with W4A16 QNN export.
- **Special notes:**
  - Hybrid thinking/non-thinking mode (thinking is on by default at 4B)
  - Matches 72B-class model performance on many tasks
  - Officially on Qualcomm AI Hub (W4A16 Hexagon NPU path available)
  - MoE-equivalent effective capacity from dense architecture
  - 32K context (128K extended)
- **Device requirement:** 12GB+ RAM phones only (4.2 GB peak with OS).

---

#### `microsoft/Phi-4-mini-instruct`
- **INT4 size:** 2,490 MB (confirmed, unsloth HF repo) | **Peak RAM:** ~4,100 MB
- **Benchmarks:** GSM8K 88.6%, MMLU 67.3%, HumanEval 74.4%, ARC-C 83.7% (Microsoft Phi-4-mini model card)
- **Why included:** Best reasoning-per-GB in the pool. HumanEval 74.4% — highest code generation score. ARC-C 83.7% — highest commonsense score. Microsoft's official recommendation for on-device reasoning. ONNX GenAI + LiteRT deployment paths available alongside GGUF.
- **Special notes:**
  - Best HumanEval in the pool (74.4%)
  - ONNX + LiteRT paths available (in addition to GGUF/llama.cpp)
  - 200K vocabulary (vs ~32K for most models) — better tokenization for code and multilingual text
  - **⚠️ Benchmark harness caveat**: Qwen3's LiveCodeBench harness has known bugs inflating coding scores; independently verify Phi-4-mini code benchmarks against EvalPlus if coding is the use case
- **Device requirement:** 8GB+ RAM phones only (4.1 GB peak with OS). Not suitable for 6GB phones.

---

## 4. SLM Factory-Specific Considerations

### Quantization choice for fine-tuned models

SLM Factory fine-tunes in BF16/FP16 (full precision via LoRA on Unsloth/LLaMA-Factory) and the trained weights are what gets evaluated. The INT4 column in the pool is for **inference only** — the pool models are the starting checkpoints, not the training targets.

For the trained adapter to be deployed on Android, the process is:
1. Train LoRA adapter in FP16
2. Merge adapter into base model
3. Quantize to Q4_K_M GGUF for deployment

Important: **always retrain from base** (not from a prior fine-tuned checkpoint) per the Pioneer Agent paper design principle. The pool entries reflect base checkpoint sizes.

### Tunability matters as much as baseline benchmarks

For SLM Factory, the question is not just "which model is strongest?" but "which model gains the most from fine-tuning?" From the Distil Labs 12-model benchmark:

- **Most tunable**: Llama-3.2-1B and Qwen3-0.6B — largest absolute gains from SFT
- **Best post-fine-tuning**: Qwen3-1.7B — best ceiling after training

The implication: start with Llama-3.2-1B or Qwen3-0.6B for tasks that are easy (large training data, clear signal), start with Qwen3-1.7B when the task is hard (small data, noisy signal, complex reasoning).

### The DeepSeek-R1-Distill-1.5B caveat for SLM Factory

This model is not a general-purpose starting checkpoint. Its SFT-only distillation from DeepSeek-R1 causes documented catastrophic forgetting of general skills. If SLM Factory's fine-tuning target is math or formal reasoning, this model is worth trying — but expect failure on any general benchmark during evaluation. The regression gate (`ε=2`) in the iteration policy will correctly catch this.

### The 20 tok/s interactive threshold

The `min_tok_s` field on `HardwareConstraints` (default 0, disabled) enforces the interactive throughput floor from `hardware_metrics.md`:
- **Hard gate: ≥20 tok/s** — a 200-token response completes in 10 seconds (workable)
- **Target: ≥30 tok/s** — responses feel genuinely interactive

On a **Snapdragon 778G** (the design reference chip, CHIP_SCALE_FACTORS = 1.0), Tier 2 and Tier 3 models generally do not hit 20 tok/s. This means SLM Factory deployments on mid-range phones are batch-mode by default unless a Tier 0/1 model is used.

Set `min_tok_s=20.0` when the deployment requires interactive streaming. Leave it at `0.0` for batch fine-tuning eval loops where latency is irrelevant.

### Escalation logic implication

`filter_pool` returns models sorted by tier then size. The escalation node picks the next model up. The `min_tok_s` filter can change which models appear in this list — if a 3B model doesn't hit the throughput floor on the target chip, it won't appear as an escalation candidate, and the system will terminate rather than escalate to an unusable model. This is correct behavior.

### Models that were considered but not included

| Model | Reason excluded |
|---|---|
| Phi-3.5-mini (3.8B) | 2.39 GB INT4 — exceeds practical ceiling for 8GB phones; Phi-4-mini is a better 3.8B |
| Gemma 4 E2B/E4B | Gemma3n-E2B is already in the pool; Gemma 4 E4B (4–5 GB) too large |
| Mistral 7B | 4.4 GB INT4 — well over budget |
| OLMo 2 | No sub-3B model released |
| OpenELM family | "Fell short on MMLU, with a score only slightly better than random chance" — not competitive with Qwen/SmolLM peers |
| TinyLlama 1.1B | Superseded by MiniCPM5-1B, SmolLM2-1.7B, and Llama-3.2-1B on all benchmarks |
| MiniCPM3-4B | 4.0B active params — Qwen2.5-3B beats it on every benchmark per Qwen2.5 Technical Report Table 9 |
| Qwen3.5-2B (original) | Previously excluded due to no GGUF support — now in pool via unsloth/Qwen3.5-2B-GGUF |
| HRM-Text-1B | Research candidate; no llama.cpp or standard runtime support |
| Llama-3.2-3B (Q3_K_M) | Not a separate model — a quantization configuration; noted in Llama-3.2-3B entry instead |

---

## Sources

- Qwen3 Technical Report: [arXiv 2505.09388](https://arxiv.org/abs/2505.09388)
- Qwen2.5 Technical Report: [arXiv 2412.15115](https://arxiv.org/abs/2412.15115)
- Gemma 3 Technical Report: [arXiv 2503.19786](https://arxiv.org/abs/2503.19786)
- SmolLM2 paper: [arXiv 2502.02737](https://arxiv.org/abs/2502.02737)
- DeepSeek-R1 paper: [arXiv 2501.12948](https://arxiv.org/abs/2501.12948)
- MiniCPM4 Technical Report: [arXiv 2506.07900](https://arxiv.org/abs/2506.07900)
- Mobile LLM benchmarking: [arXiv 2410.03613](https://arxiv.org/abs/2410.03613)
- LLM inference at edge (sustained load): [arXiv 2603.23640](https://arxiv.org/abs/2603.23640)
- Scaling laws for quantization: [arXiv 2411.17691](https://arxiv.org/abs/2411.17691)
- Qwen3 quantization empirical study: [arXiv 2505.02214](https://arxiv.org/abs/2505.02214)
- IJCAI 2025 quantization/model size tradeoffs: [arXiv 2409.11055](https://arxiv.org/abs/2409.11055)
- MNN-LLM paper: [arXiv 2506.10443](https://arxiv.org/abs/2506.10443)
- ExecuTorch KleidiAI (Llama 3.2 1B 50.2 tok/s): [PyTorch Blog](https://pytorch.org/blog/unleashing-ai-mobile/)
- Qualcomm AI Hub Llama benchmarks: [aihub.qualcomm.com](https://aihub.qualcomm.com/models/llama_v3_2_3b_instruct)
- Qwen3.5-0.8B: [huggingface.co/Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- SaladCloud Qwen3.5 small benchmarks: [blog.salad.com](https://blog.salad.com/qwen-3-5-small-models-on-saladcloud-benchmarks-cost-and-why-you-dont-need-a-mac-mini/)
- MobileAIBench (iPhone 14 numbers): [arXiv 2406.10290](https://arxiv.org/abs/2406.10290)
- BlueLM-V-3B (Dimensity 9300 24.4 tok/s): [arXiv 2411.10640](https://arxiv.org/abs/2411.10640)
- Distil Labs 12-SLM fine-tuning benchmark: [distillabs.ai](https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/)
