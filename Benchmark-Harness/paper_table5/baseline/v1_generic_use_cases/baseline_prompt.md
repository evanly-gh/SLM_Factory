You are helping evaluate an "SLM Factory": a system that picks an on-device small language model configuration for an app on a specific Android phone. Below are four phones, a pool of 104 configurations (9 models × 3 precisions × 2 runtimes × 2 backends) and 60 scenarios. Each scenario is an app use case on one phone, with a limit on four measured quantities. Your job, for every scenario, is to predict which configurations would meet all four limits on that phone, why the others would not, and which one to deploy.

You have no benchmark measurements from these phones. Use what you know about the phones' chips, the runtimes (llama.cpp and MNN) on Android CPUs and OpenCL GPUs, and the models. Do not search the web for benchmark numbers. You may use code for arithmetic. Commit to an answer for every configuration: this is a test of how well the choice can be made without measuring on the phone.

## How a configuration is measured

Each configuration was benchmarked on each phone with a 256-token prompt followed by a 256-token answer:

- **Peak memory (MB):** the peak memory footprint of the inference process during that run, CPU memory plus GPU allocations (PSS).
- **Prefill (tokens/s):** prompt-processing speed for the 256-token prompt.
- **Decode (tokens/s):** generation speed over the 256-token answer (after the 256-token prompt).
- **Energy (millijoules per token, prefill and decode separately):** whole-phone battery energy net of the phone's idle draw, screen off. A scenario's **energy per request** is prompt tokens × prefill mJ/token + answer tokens × decode mJ/token, in joules.

Tools: llama.cpp `llama-bench` (CPU: no GPU layers; GPU: all layers on OpenCL) and MNN `llm_bench` (CPU or OpenCL). Default thread settings. Builds: Qwen3.5 models need newer builds (llama.cpp 9c2e0e4, MNN a8ec497); all other models use llama.cpp eadc418 and MNN 510ac8f.

**Cannot run.** Some configurations cannot run on a phone at all: the model does not fit in memory, fails to load, crashes, or the GPU backend does not work on that phone's driver. The harness does not even try a configuration whose model file is larger than 55% of the phone's usable RAM (for MNN F16 on CPU, twice the file size, because MNN unpacks F16 weights to 32-bit on CPU). Others can still fail when tried.

## Phones

| Phone | Hardware | Usable RAM | Battery | Pool |
|---|---|--:|--:|---|
| Galaxy S23 | Snapdragon 8 Gen 2 (SM8550): 1× Cortex-X3 3.36 GHz, 2× A715 + 2× A710 2.8 GHz, 3× A510 2.0 GHz; Adreno 740 GPU; 8 GB LPDDR5X; Android 14 | 6.91 GiB | 3,900 mAh (15.0 Wh) | all 104 |
| OnePlus 15 | Snapdragon 8 Elite Gen 5 (SM8850): 2× Oryon prime 4.6 GHz, 6× Oryon performance 3.62 GHz; Adreno 840 GPU; 16 GB LPDDR5X; Android 16 | 14.76 GiB | 7,300 mAh (28.1 Wh) | all 104 |
| OnePlus 8 Pro | Snapdragon 865 (SM8250): 1× Cortex-A77 2.84 GHz, 3× A77 2.42 GHz, 4× A55 1.8 GHz; Adreno 650 GPU; 12 GB LPDDR5; Android 13 | 11.24 GiB | 4,510 mAh (17.4 Wh) | all 104 |
| Redmi 9A | MediaTek Helio G25 (MT6762): 8× Cortex-A53 up to 2.0 GHz; PowerVR GE8320 GPU (no usable OpenCL, so CPU only); 4 GB LPDDR4X; 32-bit Android 11 (armeabi-v7a builds) | 3.72 GiB | 5,000 mAh (19.3 Wh) | CPU configurations only (52) |

## Configuration pool

Parameters and file sizes are from the model cards and files. Precision names differ by runtime: llama.cpp Q4_0 / Q8_0 / F16 (GGUF), MNN Q4 / Q8 (4/8-bit weights) / F16.

| ID | Model | Family | Parameters (B) | Precision | Bits | File size (MB) | Runtime | Backend |
|---|---|---|--:|---|--:|--:|---|---|
| C001 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | Q4_0 | 4 | 92 | llama.cpp | CPU |
| C002 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | Q4_0 | 4 | 92 | llama.cpp | GPU |
| C003 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | Q4 | 4 | 87 | MNN | CPU |
| C004 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | Q4 | 4 | 87 | MNN | GPU |
| C005 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | Q8_0 | 8 | 145 | llama.cpp | CPU |
| C006 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | Q8_0 | 8 | 145 | llama.cpp | GPU |
| C007 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | Q8 | 8 | 154 | MNN | CPU |
| C008 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | Q8 | 8 | 154 | MNN | GPU |
| C009 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | F16 | 16 | 271 | llama.cpp | CPU |
| C010 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | F16 | 16 | 271 | llama.cpp | GPU |
| C011 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | F16 | 16 | 272 | MNN | CPU |
| C012 | SmolLM2-135M | SmolLM2 (HuggingFace) | 0.135 | F16 | 16 | 272 | MNN | GPU |
| C013 | Gemma3-270M | Gemma 3 (Google) | 0.268 | Q4_0 | 4 | 241 | llama.cpp | CPU |
| C014 | Gemma3-270M | Gemma 3 (Google) | 0.268 | Q4_0 | 4 | 241 | llama.cpp | GPU |
| C015 | Gemma3-270M | Gemma 3 (Google) | 0.268 | Q4 | 4 | 176 | MNN | CPU |
| C016 | Gemma3-270M | Gemma 3 (Google) | 0.268 | Q4 | 4 | 176 | MNN | GPU |
| C017 | Gemma3-270M | Gemma 3 (Google) | 0.268 | Q8_0 | 8 | 292 | llama.cpp | CPU |
| C018 | Gemma3-270M | Gemma 3 (Google) | 0.268 | Q8_0 | 8 | 292 | llama.cpp | GPU |
| C019 | Gemma3-270M | Gemma 3 (Google) | 0.268 | Q8 | 8 | 310 | MNN | CPU |
| C020 | Gemma3-270M | Gemma 3 (Google) | 0.268 | Q8 | 8 | 310 | MNN | GPU |
| C021 | Gemma3-270M | Gemma 3 (Google) | 0.268 | F16 | 16 | 543 | llama.cpp | CPU |
| C022 | Gemma3-270M | Gemma 3 (Google) | 0.268 | F16 | 16 | 543 | llama.cpp | GPU |
| C023 | Gemma3-270M | Gemma 3 (Google) | 0.268 | F16 | 16 | 544 | MNN | CPU |
| C024 | Gemma3-270M | Gemma 3 (Google) | 0.268 | F16 | 16 | 544 | MNN | GPU |
| C025 | SmolLM2-360M | SmolLM2 | 0.362 | Q4_0 | 4 | 229 | llama.cpp | CPU |
| C026 | SmolLM2-360M | SmolLM2 | 0.362 | Q4_0 | 4 | 229 | llama.cpp | GPU |
| C027 | SmolLM2-360M | SmolLM2 | 0.362 | Q4 | 4 | 229 | MNN | CPU |
| C028 | SmolLM2-360M | SmolLM2 | 0.362 | Q4 | 4 | 229 | MNN | GPU |
| C029 | SmolLM2-360M | SmolLM2 | 0.362 | Q8_0 | 8 | 386 | llama.cpp | CPU |
| C030 | SmolLM2-360M | SmolLM2 | 0.362 | Q8_0 | 8 | 386 | llama.cpp | GPU |
| C031 | SmolLM2-360M | SmolLM2 | 0.362 | Q8 | 8 | 410 | MNN | CPU |
| C032 | SmolLM2-360M | SmolLM2 | 0.362 | Q8 | 8 | 410 | MNN | GPU |
| C033 | SmolLM2-360M | SmolLM2 | 0.362 | F16 | 16 | 726 | llama.cpp | CPU |
| C034 | SmolLM2-360M | SmolLM2 | 0.362 | F16 | 16 | 726 | llama.cpp | GPU |
| C035 | SmolLM2-360M | SmolLM2 | 0.362 | F16 | 16 | 727 | MNN | CPU |
| C036 | SmolLM2-360M | SmolLM2 | 0.362 | F16 | 16 | 727 | MNN | GPU |
| C037 | Qwen3-0.6B | Qwen3 | 0.596 | Q4_0 | 4 | 469 | llama.cpp | CPU |
| C038 | Qwen3-0.6B | Qwen3 | 0.596 | Q4_0 | 4 | 469 | llama.cpp | GPU |
| C039 | Qwen3-0.6B | Qwen3 | 0.596 | Q4 | 4 | 378 | MNN | CPU |
| C040 | Qwen3-0.6B | Qwen3 | 0.596 | Q4 | 4 | 378 | MNN | GPU |
| C041 | Qwen3-0.6B | Qwen3 | 0.596 | Q8_0 | 8 | 805 | llama.cpp | CPU |
| C042 | Qwen3-0.6B | Qwen3 | 0.596 | Q8_0 | 8 | 805 | llama.cpp | GPU |
| C043 | Qwen3-0.6B | Qwen3 | 0.596 | Q8 | 8 | 676 | MNN | CPU |
| C044 | Qwen3-0.6B | Qwen3 | 0.596 | Q8 | 8 | 676 | MNN | GPU |
| C045 | Qwen3-0.6B | Qwen3 | 0.596 | F16 | 16 | 1,509 | llama.cpp | CPU |
| C046 | Qwen3-0.6B | Qwen3 | 0.596 | F16 | 16 | 1,509 | llama.cpp | GPU |
| C047 | Qwen3-0.6B | Qwen3 | 0.596 | F16 | 16 | 1,198 | MNN | CPU |
| C048 | Qwen3-0.6B | Qwen3 | 0.596 | F16 | 16 | 1,198 | MNN | GPU |
| C049 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | Q4_0 | 4 | 513 | llama.cpp | CPU |
| C050 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | Q4_0 | 4 | 513 | llama.cpp | GPU |
| C051 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | Q4 | 4 | 440 | MNN | CPU |
| C052 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | Q4 | 4 | 440 | MNN | GPU |
| C053 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | Q8_0 | 8 | 834 | llama.cpp | CPU |
| C054 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | Q8_0 | 8 | 834 | llama.cpp | GPU |
| C055 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | Q8 | 8 | 816 | MNN | CPU |
| C056 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | Q8 | 8 | 816 | MNN | GPU |
| C057 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | F16 | 16 | 1,558 | llama.cpp | CPU |
| C058 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | F16 | 16 | 1,558 | llama.cpp | GPU |
| C059 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | F16 | 16 | 1,521 | MNN | CPU |
| C060 | Qwen3.5-0.8B | Qwen3.5 | 0.773 | F16 | 16 | 1,521 | MNN | GPU |
| C061 | Qwen3-1.7B | Qwen3 | 1.72 | Q4_0 | 4 | 1,229 | llama.cpp | CPU |
| C062 | Qwen3-1.7B | Qwen3 | 1.72 | Q4_0 | 4 | 1,229 | llama.cpp | GPU |
| C063 | Qwen3-1.7B | Qwen3 | 1.72 | Q4 | 4 | 1,081 | MNN | CPU |
| C064 | Qwen3-1.7B | Qwen3 | 1.72 | Q4 | 4 | 1,081 | MNN | GPU |
| C065 | Qwen3-1.7B | Qwen3 | 1.72 | Q8_0 | 8 | 2,165 | llama.cpp | CPU |
| C066 | Qwen3-1.7B | Qwen3 | 1.72 | Q8_0 | 8 | 2,165 | llama.cpp | GPU |
| C067 | Qwen3-1.7B | Qwen3 | 1.72 | Q8 | 8 | 1,942 | MNN | CPU |
| C068 | Qwen3-1.7B | Qwen3 | 1.72 | Q8 | 8 | 1,942 | MNN | GPU |
| C069 | Qwen3-1.7B | Qwen3 | 1.72 | F16 | 16 | 4,070 | llama.cpp | CPU |
| C070 | Qwen3-1.7B | Qwen3 | 1.72 | F16 | 16 | 4,070 | llama.cpp | GPU |
| C071 | Qwen3-1.7B | Qwen3 | 1.72 | F16 | 16 | 3,447 | MNN | CPU |
| C072 | Qwen3-1.7B | Qwen3 | 1.72 | F16 | 16 | 3,447 | MNN | GPU |
| C073 | Qwen3.5-2B | Qwen3.5 | 1.9 | Q4_0 | 4 | 1,237 | llama.cpp | CPU |
| C074 | Qwen3.5-2B | Qwen3.5 | 1.9 | Q4_0 | 4 | 1,237 | llama.cpp | GPU |
| C075 | Qwen3.5-2B | Qwen3.5 | 1.9 | Q4 | 4 | 1,076 | MNN | CPU |
| C076 | Qwen3.5-2B | Qwen3.5 | 1.9 | Q4 | 4 | 1,076 | MNN | GPU |
| C077 | Qwen3.5-2B | Qwen3.5 | 1.9 | Q8_0 | 8 | 2,077 | llama.cpp | CPU |
| C078 | Qwen3.5-2B | Qwen3.5 | 1.9 | Q8_0 | 8 | 2,077 | llama.cpp | GPU |
| C079 | Qwen3.5-2B | Qwen3.5 | 1.9 | Q8 | 8 | 2,016 | MNN | CPU |
| C080 | Qwen3.5-2B | Qwen3.5 | 1.9 | Q8 | 8 | 2,016 | MNN | GPU |
| C081 | Qwen3.5-2B | Qwen3.5 | 1.9 | F16 | 16 | 3,897 | llama.cpp | CPU |
| C082 | Qwen3.5-2B | Qwen3.5 | 1.9 | F16 | 16 | 3,897 | llama.cpp | GPU |
| C083 | Qwen3.5-2B | Qwen3.5 | 1.9 | F16 | 16 | 3,780 | MNN | CPU |
| C084 | Qwen3.5-2B | Qwen3.5 | 1.9 | F16 | 16 | 3,780 | MNN | GPU |
| C085 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | Q4_0 | 4 | 2,370 | llama.cpp | CPU |
| C086 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | Q4_0 | 4 | 2,370 | llama.cpp | GPU |
| C087 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | Q4 | 4 | 2,521 | MNN | CPU |
| C088 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | Q4 | 4 | 2,521 | MNN | GPU |
| C089 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | Q8_0 | 8 | 4,280 | llama.cpp | CPU |
| C090 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | Q8_0 | 8 | 4,280 | llama.cpp | GPU |
| C091 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | Q8 | 8 | 4,532 | MNN | CPU |
| C092 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | Q8 | 8 | 4,532 | MNN | GPU |
| C093 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | F16 | 16 | 8,051 | llama.cpp | CPU |
| C094 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | F16 | 16 | 8,051 | llama.cpp | GPU |
| C095 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | F16 | 16 | 8,052 | MNN | CPU |
| C096 | Qwen3-4B | Qwen3 (Instruct-2507) | 4.02 | F16 | 16 | 8,052 | MNN | GPU |
| C097 | Qwen3.5-4B | Qwen3.5 | 4.2 | Q4_0 | 4 | 2,612 | llama.cpp | CPU |
| C098 | Qwen3.5-4B | Qwen3.5 | 4.2 | Q4_0 | 4 | 2,612 | llama.cpp | GPU |
| C099 | Qwen3.5-4B | Qwen3.5 | 4.2 | Q4 | 4 | 2,389 | MNN | CPU |
| C100 | Qwen3.5-4B | Qwen3.5 | 4.2 | Q4 | 4 | 2,389 | MNN | GPU |
| C101 | Qwen3.5-4B | Qwen3.5 | 4.2 | Q8_0 | 8 | 4,611 | llama.cpp | CPU |
| C102 | Qwen3.5-4B | Qwen3.5 | 4.2 | Q8_0 | 8 | 4,611 | llama.cpp | GPU |
| C103 | Qwen3.5-4B | Qwen3.5 | 4.2 | Q8 | 8 | 4,491 | MNN | CPU |
| C104 | Qwen3.5-4B | Qwen3.5 | 4.2 | Q8 | 8 | 4,491 | MNN | GPU |

## The filter and the pick

For each scenario, on that scenario's phone and pool:

1. Mark each configuration that would not run on that phone as **cannot run**.
2. For each configuration that runs, check the four limits: **M** peak memory above the limit, **P** prefill below the limit, **D** decode below the limit, **E** energy per request above the limit. A configuration that misses any limit is **filtered**, with every limit it misses.
3. The rest are **kept**. **Pick** the most capable kept configuration: the most parameters, then the most bits (F16 > 8-bit > 4-bit); if several runtimes or backends tie on model and precision, the one with the fastest decode, then the lowest energy. If nothing is kept, the pick is NONE.

## Scenarios

Limits: peak memory ≤ MB, prefill ≥ tokens/s, decode ≥ tokens/s, energy ≤ joules per request (with that scenario's prompt and answer lengths).

| Scenario | Phone | Use case | Prompt / answer tokens | Peak memory ≤ (MB) | Prefill ≥ (tokens/s) | Decode ≥ (tokens/s) | Energy ≤ (J per request) |
|---|---|---|--:|--:|--:|--:|--:|
| S01 | OnePlus 8 Pro | Live message translation: translate each chat message as it arrives | 64 / 64 | 1,024 | 128 | 10 | 9.4 |
| S02 | OnePlus 15 | Trip planner app: plan a day from a list of places | 400 / 400 | 6,000 | 80 | 5 | 202 |
| S03 | OnePlus 8 Pro | Language tutor: conversational practice with corrections | 300 / 100 | 1,536 | 150 | 6 | 47 |
| S04 | Galaxy S23 | Bedtime story: a story read aloud by text-to-speech | 100 / 500 | 2,048 | 33.3 | 4 | 180 |
| S05 | Redmi 9A | Recipe / how-to helper: step-by-step instructions | 150 / 300 | 2,048 | 30 | 5 | 69.5 |
| S06 | OnePlus 15 | Offline medical reference: answer questions from a clinical reference | 300 / 300 | 6,000 | 60 | 5 | 202 |
| S07 | Galaxy S23 | Language tutor: conversational practice with corrections | 300 / 100 | 1,536 | 150 | 6 | 40.5 |
| S08 | OnePlus 15 | Field-guide app: explain a plant or bird from a description, offline | 200 / 250 | 6,000 | 40 | 5 | 202 |
| S09 | OnePlus 15 | Email drafting app: draft an email from bullet points | 300 / 200 | 6,000 | 100 | 8 | 101 |
| S10 | OnePlus 15 | Document Q&A app: ask questions about an opened PDF | 1,000 / 200 | 6,000 | 100 | 8 | 152 |
| S11 | Redmi 9A | Async FAQ answers: answer a question, delivered like a text message | 128 / 64 | 1,024 | 4.27 | 1 | 104 |
| S12 | OnePlus 15 | Keyboard next-phrase: suggest the next words while typing | 64 / 8 | 300 | 213 | 20 | 3.03 |
| S13 | OnePlus 8 Pro | Bedtime story: a story read aloud by text-to-speech | 100 / 500 | 2,048 | 33.3 | 4 | 209 |
| S14 | OnePlus 15 | Overnight photo captioning: caption the photo library while charging | 200 / 40 | 7,600 | 6.67 | 1 | 67.4 |
| S15 | OnePlus 15 | Meeting-notes cleanup: turn a transcript into notes after a meeting | 1,500 / 300 | 6,000 | 50 | 3 | 674 |
| S16 | Galaxy S23 | Screen-reader descriptions: describe a screen's content aloud | 400 / 60 | 1,024 | 200 | 5 | 16.2 |
| S17 | Galaxy S23 | Health-worker checklist: guide an offline checklist on a budget phone | 256 / 128 | 1,536 | 12.8 | 2 | 81 |
| S18 | OnePlus 15 | Essay feedback app: comments on a pasted essay | 500 / 300 | 6,000 | 50 | 5 | 152 |
| S19 | OnePlus 15 | Offline Q&A (travel): an assistant with no network | 200 / 200 | 6,000 | 20 | 4 | 152 |
| S20 | OnePlus 15 | Statement categorization: categorize a month of bank transactions | 2,500 / 500 | 6,000 | 20.8 | 1 | 1,012 |
| S21 | OnePlus 15 | Interactive chat: a general chat assistant | 256 / 256 | 2,048 | 128 | 10 | 60.7 |
| S22 | OnePlus 15 | Morning digest: a scheduled summary of overnight messages | 2,000 / 300 | 6,000 | 16.7 | 1 | 1,012 |
| S23 | Redmi 9A | Overnight photo captioning: caption the photo library while charging | 200 / 40 | 1,900 | 6.67 | 1 | 46.3 |
| S24 | OnePlus 8 Pro | Voice assistant: spoken answer to a spoken question | 256 / 64 | 1,536 | 256 | 5 | 25.1 |
| S25 | OnePlus 15 | Voice assistant: spoken answer to a spoken question | 256 / 64 | 1,536 | 256 | 5 | 40.5 |
| S26 | OnePlus 15 | Offline translator app: a dedicated translation app for travel | 100 / 100 | 6,000 | 33.3 | 5 | 60.7 |
| S27 | OnePlus 15 | Story-writing app: co-write fiction with the user | 300 / 600 | 6,000 | 60 | 6 | 152 |
| S28 | Galaxy S23 | Coding helper: explain or fix a code snippet | 1,000 / 150 | 3,072 | 333 | 15 | 54 |
| S29 | OnePlus 15 | Overnight inbox sorting: label every email while charging | 300 / 10 | 7,600 | 30 | 1 | 40.5 |
| S30 | Redmi 9A | Document translation: translate a page while the user does something else | 500 / 500 | 1,500 | 16.7 | 2 | 208 |
| S31 | Galaxy S23 | Live message translation: translate each chat message as it arrives | 64 / 64 | 1,024 | 128 | 10 | 8.1 |
| S32 | Galaxy S23 | Journaling prompts: a reflective question after a journal entry | 300 / 150 | 1,536 | 60 | 5 | 108 |
| S33 | OnePlus 8 Pro | Study tutor app: explain homework questions | 300 / 100 | 4,600 | 60 | 6 | 47 |
| S34 | OnePlus 8 Pro | Journaling prompts: a reflective question after a journal entry | 300 / 150 | 1,536 | 60 | 5 | 125 |
| S35 | OnePlus 8 Pro | Field data-entry helper: fill forms from notes on a low-cost phone | 128 / 64 | 1,024 | 12.8 | 2 | 62.6 |
| S36 | OnePlus 15 | SMS auto-reply: draft a reply to a text message | 128 / 32 | 768 | 25.6 | 3 | 40.5 |
| S37 | OnePlus 15 | Smart replies: three reply chips under a received message | 128 / 24 | 512 | 256 | 15 | 10.1 |
| S38 | OnePlus 15 | Notification triage: classify incoming notifications in the background | 128 / 8 | 512 | 25.6 | 2 | 3.37 |
| S39 | Galaxy S23 | Keyboard next-phrase: suggest the next words while typing | 64 / 8 | 300 | 213 | 20 | 1.62 |
| S40 | OnePlus 8 Pro | Smart replies: three reply chips under a received message | 128 / 24 | 512 | 256 | 15 | 6.26 |
| S41 | OnePlus 15 | Game NPC dialogue: a character answers the player | 200 / 40 | 768 | 200 | 12 | 20.2 |
| S42 | Redmi 9A | Field-guide app: explain a plant or bird from a description, offline | 200 / 250 | 1,500 | 40 | 5 | 139 |
| S43 | Galaxy S23 | Document Q&A app: ask questions about an opened PDF | 1,000 / 200 | 2,800 | 100 | 8 | 81 |
| S44 | OnePlus 15 | Voice-memo summary: title and summarize a recorded voice memo | 800 / 80 | 6,000 | 40 | 2 | 101 |
| S45 | OnePlus 15 | Email summary on open: summarize an email when it is opened | 1,000 / 100 | 2,048 | 250 | 8 | 67.4 |
| S46 | OnePlus 15 | Private chat app: a dedicated on-device chat app the user opens | 256 / 256 | 6,000 | 128 | 10 | 60.7 |
| S47 | OnePlus 15 | Weekly journal summary: summarize a week of entries | 3,000 / 400 | 7,600 | 25 | 1 | 1,012 |
| S48 | OnePlus 15 | Language practice app: spoken-style conversation practice | 300 / 150 | 6,000 | 100 | 6 | 101 |
| S49 | OnePlus 15 | Notes Q&A (RAG): answer from retrieved personal notes | 1,500 / 200 | 2,560 | 300 | 10 | 101 |
| S50 | OnePlus 15 | Recipe / how-to helper: step-by-step instructions | 150 / 300 | 2,048 | 30 | 5 | 101 |
| S51 | OnePlus 8 Pro | Notification triage: classify incoming notifications in the background | 128 / 8 | 512 | 25.6 | 2 | 2.09 |
| S52 | OnePlus 15 | Document translation: translate a page while the user does something else | 500 / 500 | 6,000 | 16.7 | 2 | 303 |
| S53 | Redmi 9A | SMS auto-reply: draft a reply to a text message | 128 / 32 | 768 | 25.6 | 3 | 27.8 |
| S54 | Redmi 9A | Offline Q&A (travel): an assistant with no network | 200 / 200 | 1,500 | 20 | 4 | 104 |
| S55 | OnePlus 15 | Study tutor app: explain homework questions | 300 / 100 | 6,000 | 60 | 6 | 75.9 |
| S56 | OnePlus 15 | Fluent premium chat: a chat assistant that streams fast | 512 / 256 | 3,072 | 341 | 20 | 60.7 |
| S57 | OnePlus 8 Pro | Screen-reader descriptions: describe a screen's content aloud | 400 / 60 | 1,024 | 200 | 5 | 18.8 |
| S58 | Galaxy S23 | Game NPC dialogue: a character answers the player | 200 / 40 | 768 | 200 | 12 | 10.8 |
| S59 | Galaxy S23 | Offline translator app: a dedicated translation app for travel | 100 / 100 | 2,800 | 33.3 | 5 | 32.4 |
| S60 | OnePlus 8 Pro | Essay feedback app: comments on a pasted essay | 500 / 300 | 4,600 | 50 | 5 | 94 |

## Answer format

Answer in three messages, so nothing is cut off: S01-S20 now; S21-S40 when I reply "continue"; S41-S60 after the next "continue". Before the first block you may write a short paragraph on your general assumptions (at most 200 words); after that, output only blocks in exactly this format, one per scenario:

```
SCENARIO S01
PICK: C047
KEPT: C001 C002 C005
FILTERED: C003=P C004=MPE C006=D
CANNOT_RUN: C099 C100
END
```

Every configuration in the scenario's pool must appear exactly once across KEPT, FILTERED and CANNOT_RUN (104 IDs, or the 52 CPU IDs on the Redmi 9A). After `=` list every limit it misses, using the letters M, P, D, E. Write `PICK: NONE` if nothing is kept. Use an empty list (e.g. `CANNOT_RUN:`) when a group is empty.
