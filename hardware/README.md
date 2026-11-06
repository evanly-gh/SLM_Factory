## SLM Factory Hardware Repository

This repository contains the work that I've done for integrating SLM Inference on various mobile devices across various inference engines (llama.cpp, MNN), gathering metrics, and storing results. This is a work in progress.

## Repository layout

| Path | What it is |
|---|---|
| `Model-Conversion/` | `convert_to_gguf.py` (Hugging Face → BF16 / Q4_K_M / Q8_0 GGUF) and `convert_to_mnn.py` (Hugging Face → MNN 16 / 4 / 8-bit) |
| `SmolChat-Android/` | The SmolChat Android app (llama.cpp runtime) with a headless benchmark receiver, and the vendored `llama.cpp` it is built from (also used for GGUF conversion) |
| `Benchmark-Harness/` | `run_autobench.py` (benchmarks a GGUF in SmolChat), `bench_common.py` (shared readiness gate, page-cache eviction, Perfetto energy), `run_model_sweep.sh` (unattended sweep over models × engines × backends × precisions) |
| `mnn-benchmark-harness/` | `run_mnn_autobench.py` (benchmarks an MNN model in MNN Chat), `compare_engines.py` (result tables and filters) |
| `Power-Monitor/` | Optional Monsoon HVPM power-meter reading, used by both harnesses when a meter is attached |
| `scripts/` | `setup_mnn.sh` / `setup_mnn.ps1`: fetch and build the MNN export toolchain |
| `MNN/`, `.venv`, `.venv_mnn` | Created by setup (git-ignored) |

## Setup

### 1. Prerequisites

- **Python 3.10–3.12** (3.12 recommended). 3.13+ is not supported: llama.cpp's converter pins numpy 1.26.
- **Git** and **CMake ≥ 3.22**.
- **A C++ compiler**: Visual Studio 2022 or its Build Tools with "Desktop development with C++" (Windows), Xcode Command Line Tools (macOS), or `build-essential` (Linux). Needed to build `llama-quantize` and `MNNConvert`.
- **Android Studio**, with these installed from its SDK Manager:
  - Android SDK Platform-Tools (`adb`)
  - NDK `27.2.12479018`
  - CMake `3.22.1`

  The scripts find the SDK through `ANDROID_HOME` / `ANDROID_SDK_ROOT`, or Android Studio's default location (`%LOCALAPPDATA%\Android\Sdk`, `~/Library/Android/sdk`, `~/Android/Sdk`). Set `ANDROID_NDK_HOME` to use a different NDK.
- **A Hugging Face account.** Gated models (e.g. `google/gemma-3-270m-it`) need their licence accepted on the model page and a login: `hf auth login` (after step 2).
- **For the sweep script only:** bash. On Windows, use Git Bash.

### 2. Python environment

From the repo root:

```bash
# macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a Monsoon power meter is attached, install its library together with the main requirements:
`python -m pip install -r requirements.txt -r requirements-monsoon.txt`.

### 3. llama.cpp tools (GGUF conversion)

Nothing to do up front: `convert_to_gguf.py` uses the vendored `SmolChat-Android/llama.cpp`, and builds `llama-quantize` with CMake the first time it is needed. To build it ahead of time:

```bash
cmake -S SmolChat-Android/llama.cpp -B SmolChat-Android/llama.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build SmolChat-Android/llama.cpp/build --config Release --target llama-quantize -j
```

### 4. MNN export toolchain (MNN conversion)

Fetches MNN 3.6.1 (commit `47ccf6c6bb5b`) into `MNN/`, builds `MNNConvert`, and creates `.venv_mnn` with the exporter's dependencies (`requirements-mnn.txt`):

```bash
bash scripts/setup_mnn.sh                                     # macOS / Linux
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_mnn.ps1   # Windows; run from a "Developer PowerShell for VS 2022" if cmake can't find the compiler
```

Re-running skips finished stages; `--force` / `-Force` rebuilds. To use an existing MNN install instead, set `SLM_MNN_ROOT`, `SLM_MNN_LLMEXPORT`, `SLM_MNN_CONVERT_BIN` and `SLM_MNN_PYTHON`.

### 5. Android apps

**SmolChat (llama.cpp).** Build the CPU and OpenCL APKs and install the one you want to benchmark:

```bash
cd SmolChat-Android
./gradlew :app:assembleCpuDebug :app:assembleOpenclDebug      # Windows: gradlew.bat ...
adb install -r app/build/outputs/apk/cpu/debug/app-cpu-debug.apk
```

Gradle needs `ANDROID_HOME` set (or a `SmolChat-Android/local.properties` with `sdk.dir=...`; Android Studio writes it when you open the project).

**MNN Chat (MNN).** `run_mnn_autobench.py` drives a modified MNN Chat build (`com.alibaba.mnnllm.android` with a `.benchmark.headless.BenchmarkHeadlessReceiver`). **Its source is not in this repository yet**, so the MNN half of the benchmarks needs that APK installed from elsewhere.

### 6. Phone

Enable Developer options and USB debugging (or wireless debugging), then check `adb devices` lists the phone. Energy numbers are only valid while the phone is not charging, so use wireless debugging with the cable unplugged for energy runs.

## Usage

Convert a model (outputs land in the `--output` directory):

```bash
cd Model-Conversion
python convert_to_gguf.py --model HuggingFaceTB/SmolLM2-135M-Instruct --output output-smollm2-135m-instruct --quant ALL
python convert_to_mnn.py  --model HuggingFaceTB/SmolLM2-135M-Instruct --output mnn-output-smollm2-135m-instruct --quant ALL
```

Benchmark one configuration:

```bash
cd Benchmark-Harness
python run_autobench.py --model ../Model-Conversion/output-smollm2-135m-instruct/smollm2-135m-instruct-q4_k_m.gguf --n-gpu-layers 0 --energy --output result.json

cd ../mnn-benchmark-harness
adb push ../Model-Conversion/mnn-output-smollm2-135m-instruct/smollm2-135m-instruct-mnn-q4 /data/local/tmp/mnn_models/
python run_mnn_autobench.py --model-path /data/local/tmp/mnn_models/smollm2-135m-instruct-mnn-q4 --backend-type cpu --energy --output result.json
```

Run the whole sweep (converts, installs, pushes and benchmarks every model × engine × backend × precision; resumable):

```bash
bash Benchmark-Harness/run_model_sweep.sh --plan    # show what would run
bash Benchmark-Harness/run_model_sweep.sh
python mnn-benchmark-harness/compare_engines.py --results-dir Benchmark-Harness/sweep_results --list
```

### Environment variables

| Variable | Effect |
|---|---|
| `ANDROID_HOME`, `ANDROID_SDK_ROOT`, `ANDROID_NDK_HOME` | Android SDK / NDK location |
| `SLM_MNN_ROOT`, `SLM_MNN_LLMEXPORT`, `SLM_MNN_CONVERT_BIN`, `SLM_MNN_PYTHON` | Use an MNN toolchain other than `MNN/` + `.venv_mnn` |
| `SLM_MNN_QUANT_BLOCK`, `SLM_MNN_HQQ`, `SLM_MNN_LM_QUANT_BIT` | Override the MNN export recipe (default: 4-bit uses block 32 + HQQ; lm_head at 8-bit) |
| `SLM_QUANT_TIMEOUT_S` | Fixed wall-clock limit for each conversion step |

## Known gaps

- The modified MNN Chat app's source is not in this repository (see step 5).
- `Benchmark-Harness/run_llamacpp_table5.py` and `Benchmark-Harness/paper_table5/` are a separate work in progress and are not covered by this setup.
