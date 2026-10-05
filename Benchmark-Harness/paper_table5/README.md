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

## Protocol and how it maps to the paper

| Paper | Here |
|---|---|
| 256-token prompt, 256 generated tokens, EOS replaced | llama-bench pp256 and tg256 at depth 256; MNN llm_bench `-kv true -p 256 -n 256` with EOS ignored (`PB_IGNORE_EOS`) |
| prefill = prompt / time to first token; decode = tokens / first-to-last time | per-repetition timings from the tools' own timed regions |
| cooled below 28 C | gate before every invocation: battery <= 28 C (per-phone in `devices.json`) and CPU caps at baseline |
| airplane mode, screen off, background off | airplane mode (Wi-Fi kept only for wireless adb), screen off, Do Not Disturb, `am kill-all` |
| 1 warm-up + >= 3 trials, mean | 1 discarded repetition + `--trials` (default 3); mean +/- std |
| framework defaults, w4 | default threads/precision; GGUF Q4_0 and Q4_K_M (paper says only "w4"); MNN llmexport defaults (block 64, no HQQ) |
| PowerBench: SoC energy from Qualcomm powercap counters | powercap if readable (rooted phones), else Power Stats rails, else battery gauge net of idle (whole phone, needs unplugged) |
| llama.cpp eadc418, MNN 51bac8f (typo; 510ac8f exists) | `build_binaries.py` pins both; `--ref head` builds current upstream for newer model architectures |

Energy from the battery gauge measures the whole phone, so expect it to read higher than the paper's SoC-only
numbers; throughput is unaffected by the energy method.
