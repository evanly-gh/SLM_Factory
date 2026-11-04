You are helping evaluate an "SLM Factory": a system that picks an on-device small language model configuration for an app on a specific Android phone. Below are four phones, a pool of 104 configurations (9 models × 3 precisions × 2 runtimes × 2 backends) and 60 scenarios. Each scenario deploys one of ten language tasks in a specific app setting on one phone, with a limit on four measured quantities. Your job, for every scenario, is to predict which configurations would meet all four limits on that phone, why the others would not, and which one to deploy.

You have no benchmark measurements from these phones. Search the web as much as you like: published benchmarks of these chips, runtimes and models, runtime documentation and issue trackers, model cards. You may use code for arithmetic. Commit to an answer for every configuration: this is a test of how well the choice can be made from published information, without measuring on the phone.

## How a configuration is measured

Each configuration was benchmarked on each phone with a 256-token prompt followed by a 256-token answer:

- **Peak memory (MB):** the peak memory footprint of the inference process during that run, CPU memory plus GPU allocations (PSS). A scenario checks the measured peak plus the KV cache (F16 keys and values) of the tokens a request holds beyond those 512 (cached prefix + prefilled + answer); only the intent-routing (~1,060 tokens) and command-parsing (~930) tasks exceed 512.
- **Prefill (tokens/s):** prompt-processing speed for the 256-token prompt.
- **Decode (tokens/s):** generation speed over the 256-token answer (after the 256-token prompt).
- **Energy (millijoules per token, prefill and decode separately):** whole-phone battery energy net of the phone's idle draw, screen off. A scenario's **energy per request** is prefilled tokens × prefill mJ/token + answer tokens × decode mJ/token, in joules.

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

## The ten tasks

Every scenario deploys one of these tasks, each served by a fine-tuned model from the pool. Each task's prompt starts with fixed text (instructions, label lists, function schemas) that the app computes once and keeps in the KV cache, so a request only prefills its new tokens. Text longer than the task's input (an abstract, an essay) is sent sentence by sentence, as several requests.

| Task | What the deployed model does | Request | Cached prefix (tokens) | Prefilled per request (tokens) | Answer (tokens) |
|---|---|---|--:|--:|--:|
| Intent routing | classify a user's request to an assistant into one of a large fixed set of intents (banking, travel, home, small talk, ...) or out-of-scope | the prompt lists every intent name before the request and again after it, so about 500 tokens follow the request and cannot be cached; the answer is one intent name | 540 | 515 | 4 |
| SMS spam check | label a text message as spam or not | one text message; the answer is one label | 67 | 42 | 2 |
| Calendar event from a request | turn a user's scheduling request ("lunch with Sam Friday at 1") into one create-event API call in JSON, with dates resolved against the current date | the API schema and rules are fixed; the request is the current date plus one sentence; the answer is one JSON call | 370 | 40 | 85 |
| Tool calling | turn a user's request into JSON calls to the app's declared functions (name and arguments) | the app's function schemas are fixed (about 300 tokens for three functions); the request is one sentence; the answer is usually one call | 310 | 35 | 35 |
| Drug and disease tagging | find the chemical/drug and disease mentions in one sentence of biomedical text | one sentence; longer text is sent sentence by sentence; the answer is a JSON span list | 40 | 32 | 45 |
| Emotion tagging | tag the emotions a short message expresses, from a fixed list of emotions (one or more) | one short message plus a fixed reply rule after it; longer text is sent sentence by sentence; the answer is one to three labels | 85 | 57 | 5 |
| Grammar correction | rewrite one English sentence with its grammatical errors fixed | one sentence; longer text is sent sentence by sentence; the answer is the corrected sentence | 58 | 24 | 20 |
| Fine-grained entity tagging | tag the named entities in a short query or sentence with about 30 fine-grained types (artist, software, food, medication, ...) | one query or short sentence; longer text is sent sentence by sentence; the answer is a JSON span list | 153 | 22 | 35 |
| Conversation summary | summarize a short conversation between two or more people in one to three sentences | one conversation of about 10 turns; the answer is a short summary, shown as it streams | 30 | 175 | 40 |
| Reminder and weather command parsing | parse a reminder or weather command into a nested intent and slot tree | the intent and slot lists and the grammar are fixed (about 880 tokens); the command is a few words; the answer is a bracketed parse | 876 | 17 | 40 |

## The filter and the pick

For each scenario, on that scenario's phone and pool:

1. Mark each configuration that would not run on that phone as **cannot run**.
2. For each configuration that runs, check the four limits: **M** peak memory above the limit, **P** prefill below the limit, **D** decode below the limit, **E** energy per request above the limit. A configuration that misses any limit is **filtered**, with every limit it misses.
3. The rest are **kept**. **Pick** the most capable kept configuration: the most parameters, then the most bits (F16 > 8-bit > 4-bit); if several runtimes or backends tie on model and precision, the one with the fastest decode, then the lowest energy. If nothing is kept, the pick is NONE.

## Scenarios

Limits: peak memory ≤ MB, prefill ≥ tokens/s, decode ≥ tokens/s, energy ≤ joules per request (with that task's prefilled and answer tokens).

| Scenario | Phone | Task | Deployment | Prefilled / answer tokens | Peak memory ≤ (MB) | Prefill ≥ (tokens/s) | Decode ≥ (tokens/s) | Energy ≤ (J per request) |
|---|---|---|---|--:|--:|--:|--:|--:|
| S01 | OnePlus 8 Pro | Calendar event from a request | Voice 'add to my calendar': create an event from a spoken request | 40 / 85 | 1,536 | 80 | 57 | 104 |
| S02 | OnePlus 15 | Tool calling | Shortcuts app: an on-device shortcuts app that turns a typed request into an action call | 35 / 35 | 6,000 | 35 | 9 | 50.6 |
| S03 | OnePlus 8 Pro | Grammar correction | Email 'fix grammar' button: correct a sentence in an email being written | 24 / 20 | 1,024 | 48 | 8 | 31.3 |
| S04 | OnePlus 15 | Tool calling | Voice-assistant actions: turn a spoken request into a call to one of the assistant's phone actions (send a message, set an alarm, get the weather) | 35 / 35 | 1,536 | 70 | 24 | 50.6 |
| S05 | Redmi 9A | Reminder and weather command parsing | Reminders app quick add: parse a typed reminder in the reminders app | 17 / 40 | 950 | 34 | 16 | 34.7 |
| S06 | OnePlus 15 | Reminder and weather command parsing | Reminders and weather app: an assistant app for reminders and weather that parses requests on the device | 17 / 40 | 6,000 | 17 | 10 | 50.6 |
| S07 | OnePlus 15 | Intent routing | Offline assistant app: a private assistant app that routes typed requests on the device | 515 / 4 | 6,000 | 515 | 1 | 33.7 |
| S08 | Galaxy S23 | Grammar correction | English journal corrections: correct the day's English journal entries of a learner, ready in the morning | 24 / 20 | 3,500 | 0.8 | 1 | 225 |
| S09 | OnePlus 15 | Fine-grained entity tagging | Photo and notes library index: after install, tag the entities in the existing photo captions and notes (2,000 items) | 22 / 35 | 7,600 | 4.07 | 7 | 12.6 |
| S10 | Galaxy S23 | Conversation summary | Overnight call summaries: the built-in dialer summarizes the day's recorded calls overnight | 175 / 40 | 3,500 | 5.83 | 2 | 1,350 |
| S11 | Redmi 9A | Grammar correction | English journal corrections: correct the day's English journal entries of a learner, ready in the morning | 24 / 20 | 1,900 | 0.8 | 1 | 290 |
| S12 | OnePlus 8 Pro | Drug and disease tagging | Paper reader app: highlight drug and disease names in abstracts, sentence by sentence, as the user reads | 32 / 45 | 4,600 | 32 | 12 | 6.26 |
| S13 | Galaxy S23 | SMS spam check | First-run inbox scan: after install, check the whole SMS history (5,000 messages) overnight | 42 / 2 | 3,500 | 19.4 | 1 | 2.7 |
| S14 | OnePlus 15 | Emotion tagging | Keyboard mood insights: an opt-in keyboard feature: tag the messages the user typed that day, for a mood summary in the morning | 57 / 5 | 7,600 | 1.9 | 1 | 253 |
| S15 | OnePlus 15 | SMS spam check | First-run inbox scan: after install, check the whole SMS history (5,000 messages) overnight | 42 / 2 | 7,600 | 19.4 | 1 | 5.06 |
| S16 | Galaxy S23 | Tool calling | Voice-assistant actions: turn a spoken request into a call to one of the assistant's phone actions (send a message, set an alarm, get the weather) | 35 / 35 | 1,536 | 70 | 24 | 27 |
| S17 | OnePlus 15 | Fine-grained entity tagging | Search query understanding: find the people, titles and products in a query typed into a search app | 22 / 35 | 1,024 | 44 | 14 | 10.1 |
| S18 | OnePlus 15 | Calendar event from a request | Planner app: a planning assistant app that turns typed requests into events | 40 / 85 | 6,000 | 40 | 22 | 101 |
| S19 | OnePlus 8 Pro | Conversation summary | Overnight call summaries: the built-in dialer summarizes the day's recorded calls overnight | 175 / 40 | 5,800 | 5.83 | 2 | 1,566 |
| S20 | OnePlus 15 | Grammar correction | Essay proofreading app: proofread an English essay in a writing app, sentence by sentence, corrections filling a review list | 24 / 20 | 6,000 | 24 | 5 | 33.7 |
| S21 | Galaxy S23 | Calendar event from a request | Planner app: a planning assistant app that turns typed requests into events | 40 / 85 | 2,800 | 40 | 22 | 54 |
| S22 | OnePlus 15 | Grammar correction | English-learning feedback: correct a learner's English practice sentence in a language-learning app | 24 / 20 | 6,000 | 24 | 5 | 20.2 |
| S23 | Redmi 9A | Conversation summary | Overnight call summaries: the built-in dialer summarizes the day's recorded calls overnight | 175 / 40 | 1,900 | 5.83 | 2 | 1,737 |
| S24 | OnePlus 8 Pro | Drug and disease tagging | Reference-manager highlight tags: tag the drugs and diseases in a sentence the user highlights in a reference-manager app | 32 / 45 | 1,024 | 64 | 18 | 15.7 |
| S25 | OnePlus 8 Pro | Emotion tagging | Comment tone check: warn before posting a comment that reads as angry, in a social app | 57 / 5 | 1,024 | 114 | 2 | 31.3 |
| S26 | OnePlus 15 | SMS spam check | Scam-check app: paste a suspicious message into a scam-checker app | 42 / 2 | 6,000 | 42 | 1 | 101 |
| S27 | OnePlus 15 | Drug and disease tagging | Paper reader app: highlight drug and disease names in abstracts, sentence by sentence, as the user reads | 32 / 45 | 6,000 | 32 | 12 | 10.1 |
| S28 | Galaxy S23 | Conversation summary | Dialer call summary: the phone's built-in dialer summarizes a short recorded call (from its transcript) | 175 / 40 | 1,024 | 350 | 5 | 90 |
| S29 | OnePlus 15 | Conversation summary | Overnight call summaries: the built-in dialer summarizes the day's recorded calls overnight | 175 / 40 | 7,600 | 5.83 | 2 | 2,529 |
| S30 | Redmi 9A | Fine-grained entity tagging | Long-press smart actions: a system feature: find the entities in text the user long-presses on screen, to offer actions | 22 / 35 | 512 | 22 | 9 | 17.4 |
| S31 | Galaxy S23 | Calendar event from a request | Calendar quick add: type an event in plain language in the calendar app | 40 / 85 | 1,024 | 80 | 34 | 54 |
| S32 | OnePlus 15 | Tool calling | Smart-home app commands: "turn off the kitchen lights" typed in a smart-home app, turned into a device call | 35 / 35 | 1,024 | 70 | 14 | 25.3 |
| S33 | Galaxy S23 | Conversation summary | Meeting-notes app: summarize a short recorded conversation (from its transcript) in a notes app | 175 / 40 | 2,800 | 175 | 5 | 54 |
| S34 | Galaxy S23 | SMS spam check | Scam-check app: paste a suspicious message into a scam-checker app | 42 / 2 | 2,800 | 42 | 1 | 54 |
| S35 | OnePlus 8 Pro | Drug and disease tagging | Paper-alert abstract index: index the abstracts of the day's saved and alerted papers for search, sentence by sentence | 32 / 45 | 5,800 | 2.96 | 5 | 15.7 |
| S36 | OnePlus 15 | Emotion tagging | Keyboard tone check: show the tone of a message before it is sent | 57 / 5 | 300 | 190 | 8 | 10.1 |
| S37 | OnePlus 15 | SMS spam check | Messaging app spam check: the default messaging app checks each incoming text before notifying | 42 / 2 | 512 | 42 | 1 | 10.1 |
| S38 | OnePlus 15 | Grammar correction | English-learner keyboard: a keyboard for English learners that suggests a correction when a sentence is finished | 24 / 20 | 300 | 80 | 29 | 5.06 |
| S39 | Galaxy S23 | Intent routing | Voice-assistant intent router: route each spoken request to the skill that handles it | 515 / 4 | 1,536 | 1,030 | 3 | 27 |
| S40 | OnePlus 15 | Intent routing | Banking app assistant: route a typed question in a bank's assistant chat to the right flow; non-banking intents are answered as out-of-scope | 515 / 4 | 1,024 | 1,030 | 2 | 506 |
| S41 | OnePlus 15 | Conversation summary | Chat catch-up: summarize the last ~10 unread messages of a one-to-one conversation when it is opened | 175 / 40 | 1,024 | 350 | 5 | 50.6 |
| S42 | Redmi 9A | Reminder and weather command parsing | Smartwatch command relay: parse reminder and weather commands relayed from a paired watch | 17 / 40 | 512 | 17 | 10 | 23.2 |
| S43 | OnePlus 15 | Fine-grained entity tagging | Read-later entity tags: tag people, places and products in a saved article, sentence by sentence | 22 / 35 | 6,000 | 22 | 9 | 5.06 |
| S44 | OnePlus 15 | Emotion tagging | Journaling app mood tags: tag the emotions in each sentence of a journal entry after it is saved | 57 / 5 | 6,000 | 57 | 2 | 25.3 |
| S45 | Galaxy S23 | Reminder and weather command parsing | Reminders and weather app: an assistant app for reminders and weather that parses requests on the device | 17 / 40 | 2,800 | 17 | 10 | 27 |
| S46 | Galaxy S23 | Emotion tagging | Keyboard mood insights: an opt-in keyboard feature: tag the messages the user typed that day, for a mood summary in the morning | 57 / 5 | 3,500 | 1.9 | 1 | 135 |
| S47 | OnePlus 15 | Conversation summary | Meeting-notes app: summarize a short recorded conversation (from its transcript) in a notes app | 175 / 40 | 6,000 | 175 | 5 | 101 |
| S48 | OnePlus 8 Pro | Emotion tagging | Keyboard mood insights: an opt-in keyboard feature: tag the messages the user typed that day, for a mood summary in the morning | 57 / 5 | 5,800 | 1.9 | 1 | 157 |
| S49 | OnePlus 8 Pro | Fine-grained entity tagging | Photo and notes library index: after install, tag the entities in the existing photo captions and notes (2,000 items) | 22 / 35 | 5,800 | 4.07 | 7 | 7.83 |
| S50 | Redmi 9A | Emotion tagging | Keyboard mood insights: an opt-in keyboard feature: tag the messages the user typed that day, for a mood summary in the morning | 57 / 5 | 1,900 | 1.9 | 1 | 174 |
| S51 | OnePlus 8 Pro | SMS spam check | Messaging app spam check: the default messaging app checks each incoming text before notifying | 42 / 2 | 512 | 42 | 1 | 6.26 |
| S52 | OnePlus 15 | Drug and disease tagging | Paper-alert abstract index: index the abstracts of the day's saved and alerted papers for search, sentence by sentence | 32 / 45 | 7,600 | 2.96 | 5 | 25.3 |
| S53 | Redmi 9A | Tool calling | Smartwatch actions relay: requests spoken to a paired watch, turned into calls on the phone | 35 / 35 | 512 | 35 | 9 | 17.4 |
| S54 | Redmi 9A | Grammar correction | English-learning feedback: correct a learner's English practice sentence in a language-learning app | 24 / 20 | 1,500 | 24 | 5 | 13.9 |
| S55 | OnePlus 8 Pro | Grammar correction | English journal corrections: correct the day's English journal entries of a learner, ready in the morning | 24 / 20 | 5,800 | 0.8 | 1 | 261 |
| S56 | OnePlus 8 Pro | Reminder and weather command parsing | Reminders and weather app: an assistant app for reminders and weather that parses requests on the device | 17 / 40 | 4,600 | 17 | 10 | 31.3 |
| S57 | OnePlus 8 Pro | Fine-grained entity tagging | Long-press smart actions: a system feature: find the entities in text the user long-presses on screen, to offer actions | 22 / 35 | 512 | 22 | 9 | 15.7 |
| S58 | Galaxy S23 | Emotion tagging | Keyboard tone check: show the tone of a message before it is sent | 57 / 5 | 300 | 190 | 8 | 5.4 |
| S59 | OnePlus 15 | Drug and disease tagging | Reference-manager highlight tags: tag the drugs and diseases in a sentence the user highlights in a reference-manager app | 32 / 45 | 1,024 | 64 | 18 | 25.3 |
| S60 | Galaxy S23 | Tool calling | Shortcuts app: an on-device shortcuts app that turns a typed request into an action call | 35 / 35 | 2,800 | 35 | 9 | 27 |

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
