# Table 5 replication (arXiv 2607.05475)

Measures 256-token prefill and 256-token decode throughput (tokens/s) and energy (uJ/token) for
model x quantization x framework (llama.cpp, MNN) x backend (CPU, GPU/OpenCL) on Android phones, following
the paper's protocol, and renders a table laid out like the paper's Table 5. Same commands on Windows and macOS.

## One-time setup (per computer)

- Android Studio with SDK Platform-Tools, **NDK** and **CMake** (SDK Manager > SDK Tools). The scripts find
  them in the default SDK location, or via `ANDROID_HOME` / `ANDROID_NDK_HOME`.
- Python 3.10-3.13 environment with `requirements.txt`:
  ```
  uv venv .venv_pb --python 3.12
  uv pip install --python .venv_pb -r requirements.txt
  ```
  (on networks that inspect TLS add `--system-certs`). Run every command below with that environment's Python.
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
| cooled below 28 C | gate before every invocation: battery <= 28 C (per-phone in `devices.json`) and CPU caps at baseline |
| airplane mode, screen off, background off | airplane mode (Wi-Fi kept only for wireless adb), Do Not Disturb, `am kill-all`; screen off on USB, **on at minimum brightness over wireless adb** (a screen-off phone with no USB wake lock suspends every few seconds and freezes the benchmark: energy probe run `SM-S911U_20261004_233333`) |
| 1 warm-up + >= 3 trials, mean | 1 discarded repetition + `--trials` (default 3); mean +/- std |
| framework defaults, w4 | default threads/precision; GGUF Q4_0 and Q4_K_M (paper says only "w4"); MNN llmexport defaults (block 64, no HQQ) |
| PowerBench: SoC energy from Qualcomm powercap counters | powercap if readable (rooted phones), else Power Stats rails, else battery gauge net of idle (whole phone, needs unplugged; idle measured for `--pre-idle-seconds` right before every invocation, after the cool-down gate) |
| Xiaomi 17 / OnePlus 15 / Xiaomi 15 / Xiaomi 14 columns | phones in the paper are compared with their own column; others with `proxy_column` in `devices.json` (Galaxy S23 -> Xiaomi 14, one SoC generation newer). `report.py` passes a cell within 25% of that value |
| llama.cpp eadc418, MNN 51bac8f (typo; 510ac8f exists) | `build_binaries.py` pins both; `--ref head` builds current upstream for newer model architectures |

Energy from the battery gauge measures the whole phone, so expect it to read higher than the paper's SoC-only
numbers; throughput is unaffected by the energy method.
