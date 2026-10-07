# Table 5 replication (arXiv 2607.05475)

Measures 256-token prefill and 256-token decode throughput (tokens/s) and energy (uJ/token) for
model x quantization x framework (llama.cpp, MNN) x backend (CPU, GPU/OpenCL) on Android phones, following
the paper's protocol, and renders a table laid out like the paper's Table 5. Same commands on Windows and macOS.

## One-time setup (per computer)

- Android Studio with SDK Platform-Tools, **NDK** and **CMake** (SDK Manager > SDK Tools). The scripts find
  them in the default SDK location, or via `ANDROID_HOME` / `ANDROID_NDK_HOME`.
- Python **3.12** environment with `requirements.txt` (3.10-3.12 work; on 3.13 pip has no numpy 1.26 wheel and
  builds it from source, which on Windows with MSYS2 on PATH produces a numpy that crashes):
  ```
  uv venv .venv_pb --python 3.12
  uv pip install --python .venv_pb -r requirements.txt
  ```
  (on networks that inspect TLS add `--system-certs`; without uv: `conda create -p <dir> python=3.12`, then
  `<dir>/python -m venv .venv_pb`). Run every command below with that environment's Python.
- Optional, saves pushing F16 files to the phone for quantization: a host `llama-quantize` built from the pinned
  llama.cpp checkout (`build_binaries.py` fetches it to `%USERPROFILE%\.pb\src` / `third_party/src`), passed as
  `--quantize-bin`. On Windows with MSYS2: configure with `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
  -DCMAKE_EXE_LINKER_FLAGS="-static -static-libgcc -static-libstdc++"` and build the `llama-quantize` target.
- Set `SLM_MNN_PYTHON` to that Python so the MNN exporter uses it and its MNN 3.4.0 converter.

## Per phone

1. **Connect** with USB debugging on; `adb devices` must list it as `device`. For energy on phones without
   chip counters (most unrooted phones), switch to Wi-Fi and unplug:
   `python energy_probe.py wireless --serial <usb serial>`.
2. **Energy method** (once per phone model, ~15 min, unplugged):
   `python energy_probe.py collect --serial <serial>`. Put the recommended method in `devices.json`
   (`energy_method`) for that phone's profile.
3. **Build and push the benchmark binaries** (pinned llama.cpp eadc418, MNN 510ac8f; ~15 min the first time):
   `python build_binaries.py --push --serial <serial>`
4. **Prepare models** (downloads from Hugging Face; Q4 GGUF quantization runs on the phone if this computer has
   no llama-quantize): `python prepare_models.py --serial <serial>`
5. **Run** (resumable; Ctrl+C restores the phone's settings):
   `python run_paper_table5.py --serial <serial>`
6. **Report** (any number of phones): `python report.py results/<dir1> results/<dir2> ...`
   -> `TABLE5.md`, `TABLE5.html`, `TABLE5.csv`

`python tests/fake_phone_test.py` checks the harness end to end against a simulated phone.

## Your own model pool

`run_pool.py` benchmarks every model in a pool file with this protocol, end to end (binaries, conversion, runs,
report), for both build refs the pool needs:
```
python run_pool.py --pool-file ../../../SLM_Factory/config/android_pool.py --serial S --plan     # check the list
python run_pool.py --pool-file ../../../SLM_Factory/config/android_pool.py --serial S --gguf Q4_0 Q8_0 F16 --mnn 4 8 16 --llama-threads 4
```
A pool file is SLM_Factory's `config/android_pool.py` (parsed, not executed: every `"org/name"` string that is a
Hugging Face model), a JSON list/dict of ids, or a text file with one `org/id`, `name=org/id` or `name=org/id@head`
per line (see `pool.py`). Each model runs on the pinned build if the paper's llama.cpp can convert its architecture,
else on `--ref head` (e.g. Qwen3.5). The same `--pool-file` works on `prepare_models.py` and `run_paper_table5.py`
individually; other `run_paper_table5.py` options (`--quants`, `--backends`, `--llama-threads`) pass through.

## Validation, then the model pool

1. **Validate against the paper** with its models at 4 bits (the comparison section of `TABLE5.md` passes a cell
   within 25% of the phone's paper column, or its `proxy_column`):
   ```
   python prepare_models.py --models llama3.2-1b qwen2.5-1.5b llama3.2-3b --gguf Q4_0 Q4_K_M --mnn 4 --serial S
   python run_paper_table5.py --serial S --models llama3.2-1b qwen2.5-1.5b llama3.2-3b
   ```
2. **The pool** at Q4 / Q8 / F16 (F16 rather than BF16: llama.cpp's OpenCL backend and MNN's fp16 path have no
   BF16 kernels). Qwen3.5 postdates the pinned versions, so it is converted and run with `--ref head`
   (build those binaries with `python build_binaries.py --ref head --push --serial S`):
   ```
   python prepare_models.py --pool --gguf Q4_0 Q8_0 F16 --mnn 4 8 16 --serial S
   python run_paper_table5.py --serial S --models gemma3-270m smollm2-135m smollm2-360m qwen3-0.6b qwen3-1.7b qwen3-4b-instruct-2507
   python prepare_models.py --ref head --models qwen3.5-0.8b qwen3.5-2b qwen3.5-4b --gguf Q4_0 Q8_0 F16 --mnn 4 8 16 --serial S
   python run_paper_table5.py --serial S --ref head --models qwen3.5-0.8b qwen3.5-2b qwen3.5-4b
   python report.py results/<validation dir> results/<pool dir> results/<head dir>
   ```
   Q4_0 is the llama.cpp 4-bit type for the pool: at eadc418 the OpenCL backend has no Q4_K matmul, so a Q4_K_M
   GPU row runs most of its matmuls on the CPU (marked `~` in the report). Variants that do not fit in the phone's
   memory (F16 4B on 8 GB) are recorded as failures.

## Protocol and how it maps to the paper

| Paper | Here |
|---|---|
| 256-token prompt, 256 generated tokens, EOS replaced | llama-bench pp256 and tg256 at depth 256; MNN llm_bench `-kv true -p 256 -n 256` with EOS ignored (`PB_IGNORE_EOS`) |
| prefill = prompt / time to first token; decode = tokens / first-to-last time | per-repetition timings from the tools' own timed regions |
| cooled below 28 C | gate before every invocation: battery <= 28 C (per-phone in `devices.json`) and CPU caps at baseline; a phone whose resting temperature is above 28 C (screen on in a warm room) also passes once the battery has stopped cooling (`gate_plateau_s` in `devices.json`, `--gate-plateau`); each run's start temperature is recorded |
| airplane mode, screen off, background off | airplane mode (over wireless adb only if Wi-Fi stays on in it: turn Wi-Fi back on once while in airplane mode and Android remembers it; otherwise mobile data, Bluetooth and location are switched off instead, since dropping Wi-Fi ends Wireless debugging), Do Not Disturb, `am kill-all`; **screen off** (over wireless adb a partial wake lock held by the shell user, `tools/PbWake.java` via `app_process`, stops the phone from suspending mid-run; `--screen on` keeps the screen on at minimum brightness instead) |
| 1 warm-up + >= 3 trials, mean | 1 discarded repetition + `--trials` (default 3); mean +/- std |
| framework defaults, w4 | default precision; threads default too, but `--llama-threads 4` is recommended on phones with little cores (llama-bench's default of one thread per core ran 3-9x slower on the Galaxy S23); GGUF Q4_0 and Q4_K_M (paper says only "w4"); MNN llmexport defaults (block 64, no HQQ) |
| PowerBench: SoC energy from Qualcomm powercap counters | powercap if readable (rooted phones), else Power Stats rails, else battery gauge net of idle (whole phone, needs unplugged; idle measured for `--pre-idle-seconds` right before every invocation, after the cool-down gate) |
| Xiaomi 17 / OnePlus 15 / Xiaomi 15 / Xiaomi 14 columns | phones in the paper are compared with their own column; others with `proxy_column` in `devices.json` (Galaxy S23 -> Xiaomi 14, one SoC generation newer). `report.py` passes a cell within 25% of that value |
| llama.cpp eadc418, MNN 51bac8f (typo; 510ac8f exists) | `build_binaries.py` pins both; `--ref head` builds current upstream for newer model architectures |

Energy from the battery gauge measures the whole phone, so expect it to read higher than the paper's SoC-only
numbers; throughput is unaffected by the energy method.
