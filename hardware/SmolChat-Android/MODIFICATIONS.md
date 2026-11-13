# Modifications to SmolChat-Android

This directory is a modified copy of **SmolChat-Android** by Shubham Panchal,
<https://github.com/shubham0204/SmolChat-Android>, licensed under the Apache License 2.0 (see `LICENSE` in this
directory, unchanged from upstream). This notice is provided under section 4(b) of that license: the files listed
below were changed or added for the SLM Factory on-device benchmarks (Mobile AI Factory, arXiv:2611.13509). All other
files are as in upstream.

## Upstream base

- **SmolChat-Android:** commit `b663cd6` ("release: prepare for release v16", 2026-06-21), the last upstream commit
  before the copy was made (2026-07-02). 202 of the 215 files in that commit (excluding `llama.cpp`) are
  byte-identical in the original copy, more than for any other upstream commit; the remainder are the modified files
  listed below.
- **llama.cpp** (`llama.cpp/`, MIT, see `llama.cpp/LICENSE`): upstream SmolChat references it as a git submodule.
  Here it is vendored as plain files, replaced on 2026-07-29 by a checkout of <https://github.com/ggml-org/llama.cpp>
  at commit `3018a11e7` (build b10183) for Qwen3.5 support, then modified in the two files listed below.
- **Third-party build headers** under `smollm/src/main/cpp/vendor/` (Vulkan-Hpp, SPIRV-Headers, OpenCL headers,
  OpenCL ICD loader stub) were added; their sources and pinned commits are in `smollm/src/main/cpp/vendor/README.md`,
  and each keeps its own license (headers in the files, `opencl-icd-loader/LICENSE`).

## Added files

| File | Purpose |
|---|---|
| `app/src/main/java/io/shubham0204/smollmandroid/headless/HeadlessBenchmarkReceiver.kt` | Broadcast receiver for `com.smollmandroid.RUN_PROMPT` (sent over adb): takes a model path, prompt, run id and options, and starts the benchmark service |
| `app/src/main/java/io/shubham0204/smollmandroid/headless/BenchmarkService.kt` | Foreground service that loads a GGUF model, runs the prompt(s) without the UI (cold/warm protocol, trials) and logs the metrics (load time, TTFT, prefill/decode tokens/s, peak RSS, power, energy, thermal) tagged with the run id |
| `app/src/main/java/io/shubham0204/smollmandroid/headless/HeadlessBenchmarkService.kt` | Earlier variant of the headless benchmark service, reusing the chat UI's `SmolLMManager` |
| `app/src/main/java/io/shubham0204/smollmandroid/headless/PowerSampler.kt` | Samples battery current and voltage on a fixed interval with real timestamps and integrates them into energy |
| `app/src/opencl/AndroidManifest.xml` | `opencl` flavor manifest: declares `<uses-native-library libOpenCL.so>` so the app loads the device's vendor OpenCL driver |
| `smollm/src/main/cpp/vendor/**` | Vulkan-Hpp, SPIRV-Headers, OpenCL headers and the OpenCL ICD loader stub needed to build the Vulkan and OpenCL variants (see `vendor/README.md`) |
| `hf-model-hub-api/bin/**` | Build-output copies of `hf-model-hub-api/src/` (identical contents), committed by accident; not used by the build |

## Modified files

| File | Change |
|---|---|
| `.gitignore` | Ignores model files (`*.gguf`, `*.bin`, `*.safetensors`), Python artifacts and output folders |
| `.gitmodules` | Removed: `llama.cpp` is vendored as plain files rather than a submodule |
| `app/build.gradle.kts` | `backend` product flavors (`cpu`, `vulkan`, `opencl`) matching `:smollm`; legacy (extracted) native-library packaging so ggml's dynamic backend loader finds its `.so` files; excludes the link-time `libOpenCL.so` stub from the APK |
| `app/src/main/AndroidManifest.xml` | Registers the headless receiver and the foreground benchmark service; adds foreground-service, notification and wake-lock permissions |
| `app/src/main/java/io/shubham0204/smollmandroid/SmolChatApplication.kt` | Creates the notification channel for the benchmark service |
| `app/src/main/java/io/shubham0204/smollmandroid/llm/SmolLMManager.kt` | Records cold-load time, TTFT, prefill/decode speed, prompt token count, peak RSS (VmHWM) and CPU / skin temperatures; adds `loadIsolatedSingleTurn` (fresh single-turn context per prompt) shared by the headless and chat paths; threads `max_tokens`, early-EOS suppression, GPU layer count and the native library directory to the native layer; foreground promotion during inference |
| `app/src/main/java/io/shubham0204/smollmandroid/llm/ModelsRepository.kt` | Model lookup by id returns null for a missing model instead of failing |
| `app/src/main/java/io/shubham0204/smollmandroid/data/AppDB.kt` | Room schema version 2 with a migration adding per-message metric columns; returns inserted ids; reactive single-chat flow; metric update method |
| `app/src/main/java/io/shubham0204/smollmandroid/data/ChatsDB.kt` | Chat list excludes task chats; adds a reactive single-chat query |
| `app/src/main/java/io/shubham0204/smollmandroid/data/MessagesDB.kt` | Per-message metric columns (TTFT, decode speed, peak RSS, cold load, average current, thermal status) and an update query |
| `app/src/main/java/io/shubham0204/smollmandroid/data/ModelsDB.kt` | `getModel` is nullable; `insertModels` returns the new ids |
| `app/src/main/java/io/shubham0204/smollmandroid/ui/screens/chat/ChatScreenViewModel.kt` | Inference metrics (cold load, TTFT, speeds, power, thermal) for manual chats, persisted per message; reacts to chat rows changed by the headless service; in-app inference benchmark event; chat deletion |
| `app/src/main/java/io/shubham0204/smollmandroid/ui/screens/chat/ChatActivity.kt` | Passes the metrics state to the chat screen |
| `app/src/main/java/io/shubham0204/smollmandroid/ui/screens/chat/BenchmarkModelScreen.kt` | Shows llama.cpp's internal benchmark (pp/tg tokens/s) and a "Measure Inference Metrics" panel |
| `app/src/main/java/io/shubham0204/smollmandroid/ui/screens/chat/ChatListDrawer.kt` | Delete-chat action and layout fixes in the chat list |
| `app/src/main/java/io/shubham0204/smollmandroid/ui/screens/chat/messages/ChatMessageListItem.kt` | Per-message metrics row under each assistant message |
| `app/src/main/java/io/shubham0204/smollmandroid/ui/screens/manage_tasks/ManageTasksActivity.kt` | Handles a task whose model no longer exists |
| `smollm/build.gradle.kts` | `backend` product flavors: `vulkan` and `opencl` (arm64-v8a only) pass the CMake options that build the GPU variants; `cpu` is unchanged |
| `smollm/src/main/cpp/CMakeLists.txt` | Opt-in `SMOLLM_BUILD_VULKAN_VARIANT` / `SMOLLM_BUILD_OPENCL_VARIANT` builds (`libsmollm_vulkan.so`, `libsmollm_opencl.so`) using the vendored headers, with ggml dynamic backend loading |
| `smollm/src/main/cpp/LLMInference.h`, `LLMInference.cpp` | Loads ggml backends from the app's native library directory; `n_gpu_layers`; per-completion `max_tokens` cap; optional early-EOS suppression sampler; repetition stop; prompt token count and llama.cpp perf counters (prompt eval / eval time and counts); llama.cpp and stderr logs forwarded to logcat; backend-device check |
| `smollm/src/main/cpp/smollm.cpp` | JNI: native library dir and GPU layers on load, `max_tokens` / EOS suppression on completion, `getPromptTokenCount`, `getPerfMetrics` |
| `smollm/src/main/java/io/shubham0204/smollm/SmolLM.kt` | Loads the Vulkan or OpenCL native library when the flavor ships it (else the CPU chain); `nGpuLayers` parameter; Kotlin bindings for the new JNI calls |
| `smollm/src/main/java/io/shubham0204/smollm/GGUFReader.kt` | Fails with a clear `IOException` when a GGUF file cannot be opened; `check` instead of `assert` |

## Modified llama.cpp files

| File | Change |
|---|---|
| `llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp` | `ggml_opencl_supports_op` declines Q6_K matrix multiplication on Adreno A6X/A7X GPUs, whose Q6_K GEMV kernel produced garbled output on device, so those ops fall back to the CPU backend |
| `llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Re-initializes the Vulkan-Hpp dispatcher with the created device, and disables buffer device address when the driver reports the feature but `vkGetBufferDeviceAddress` resolves to null (crash seen on an Adreno 650 device) |
