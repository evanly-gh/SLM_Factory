# Fig. 9a/9b on-phone validation

Runs the two phone-in-the-loop models from Fig. 9a/9b of the Mobile AI Factory paper on their real phones: 300
consecutive held-out requests with their real prompts and answer lengths, back to back, under the Section 3
profiling conditions. It reports median and p95 prefill speed, median and p95 decode speed, peak memory and average
energy per request, each compared with the requirement (PASS/FAIL).

| | Fig. 9a: M07 | Fig. 9b: H09 |
|---|---|---|
| Task / deployment | BC5CDR NER, "Paper-alert abstract index" (overnight job) | DialogSum, "Dialer call summary" (feature inside an app) |
| Phone | OnePlus 8 Pro (IN2025, SD865) | Galaxy S23 (SM-S911x, SD8 Gen 2) |
| Model (HF) | `evanlyhf/MobileAIFactory-BC5CDR-NER-Qwen3.5-0.8B-MNN-4bit-OnePlus8Pro` | `evanlyhf/MobileAIFactory-DialogSum-Qwen3-0.6B-MNN-8bit-GalaxyS23` |
| Engine / backend | MNN, CPU, 4 threads | MNN, CPU, 4 threads |
| Peak memory | ≤ 5,800 MB | ≤ 1,024 MB |
| Prefill | ≥ 2.96 tokens/s | ≥ 350 tokens/s |
| Decode | ≥ 5 tokens/s | ≥ 5 tokens/s |
| Energy | ≤ 15.66 J per request (shown as 15.7) | ≤ 90 J per request |
| Latency (secondary) | first token ≤ 10.8 s, answer ≤ 21.6 s | first token ≤ 0.5 s |
| Profile values for this configuration | 550 MB, 167 / 45.9 tokens/s, 7.04 J | 856 MB, 368 / 49.8 tokens/s, 10.4 J |
| Request set (300) prompt tokens | median 72, p95 104 (50-164) | median 219, p95 413 (91-1,272) |
| Reference answer tokens | median 19.5, p95 67 (1-184) | median 26, p95 48 (9-102) |
| Requirement's request shape | 32 prefilled + 45 answer (40 cached) | 175 prefilled + 40 answer (30 cached) |

Limits come from `config/filtered_pools/oneplus8pro_m07.json` and `galaxys23_h09.json`; latency and request shape
from `hardware/docs/SCENARIOS.md`.

## What you need

- Everything from `../README.md` "One-time setup": the `.venv_pb` Python environment (`requirements.txt`, which
  includes `huggingface_hub` and `perfetto`), Android Studio with **NDK** and **CMake**, and a **JDK** (`javac`; the
  harness builds its small on-phone Java helpers for the wake lock, panel-off and battery sampling).
- `adb` (platform-tools). The phone and this computer on the same Wi-Fi network (energy needs wireless adb, unplugged).
- Phones charged above ~80%, in a cool room (the gate waits for the battery to reach 28 C before each batch).
- This folder, plus `requests/m07.jsonl` and `requests/h09.jsonl`. The request files hold dataset text (DialogSum is CC BY-NC-SA), so they are not in git: copy them from the cluster checkout (`hardware/Benchmark-Harness/paper_table5/fig9_validation/requests/`) or rebuild them there with `build_requests.py`.
- About 1.3 GB of disk for the two models (downloaded from Hugging Face on first use into `fig9_validation/models/`,
  and checked against the factory's own SHA-256 fingerprints). To use a copy instead, pass `--model-dir <dir>` (the
  cluster copies are `artifacts/mnn/Qwen_Qwen3.5-0.8B/1d0d381b8a74/model-mnn-q4` and
  `artifacts/mnn/Qwen_Qwen3-0.6B/50602192f8e7/model-mnn-q8`).

## Commands

Run from `hardware/Benchmark-Harness/paper_table5` with the harness's Python (`.venv_pb/bin/python`, or
`.venv_pb\Scripts\python` on Windows; plain `python` below).

**Once per computer** (~15 min): build MNN at the commit the factory exported and scored these models with
(47ccf6c, pymnn 3.6.1), plus the request runner `pb_requests`:
```
python fig9_validation/build_pb_requests.py
```

**OnePlus 8 Pro, Fig. 9a (M07).** Plug in by USB, `adb devices` shows `<USB_SERIAL>`:
```
python fig9_validation/build_pb_requests.py --push-only --serial <USB_SERIAL>
python fig9_validation/run_fig9.py --scenario m07 --serial <USB_SERIAL> --prepare
python energy_probe.py wireless --serial <USB_SERIAL>        # prints <IP>:5555; then UNPLUG the cable
python fig9_validation/run_fig9.py --scenario m07 --serial <IP>:5555
python fig9_validation/run_fig9.py --scenario m07 --serial <IP>:5555 --answer-length eos
```

**Galaxy S23, Fig. 9b (H09).** Same, with `h09`:
```
python fig9_validation/build_pb_requests.py --push-only --serial <USB_SERIAL>
python fig9_validation/run_fig9.py --scenario h09 --serial <USB_SERIAL> --prepare
python energy_probe.py wireless --serial <USB_SERIAL>        # then UNPLUG
python fig9_validation/run_fig9.py --scenario h09 --serial <IP>:5555
python fig9_validation/run_fig9.py --scenario h09 --serial <IP>:5555 --answer-length eos
```

`--prepare` pushes the model (~0.65 GB, do it over USB), the request file and checks the binary; it measures
nothing. The run refuses to start on external power (energy would follow the charger) unless `--allow-powered`
(speed and memory only). Ctrl+C restores the phone's settings.

Optional: `--repeats 3` runs the batch three times, each from its own cool-down gate and idle baseline (session
variance); `--n 100` for a quicker check; `--gap-ms 500` puts a pause between requests.

## Expected duration

| Step | Time |
|---|---|
| Build (once per computer) | ~15 min |
| `--prepare` per phone (download + USB push) | ~5 min |
| One batch: gate (0-30 min, depends on how warm the phone is) + 60 s idle + model load + 3 warm-up + 300 requests + 60 s idle | M07 ~6 min of requests, H09 ~7 min, so ~10-40 min per batch |
| Battery per batch | a few percent |

Request time estimated from the profile's speeds and the request set's token totals (M07: 22,299 prompt + 8,336
answer tokens; H09: 71,727 + 8,497). `--answer-length eos` takes about as long if the model ends its answers near
the reference length; a runaway answer stops at the eval's 512-token cap.

## What it does

- **Conditions (paper Section 3)**, through the harness's own controls (`common.DeviceControls`,
  `bench_common.ReadinessGate`): battery at or below 28 C before each batch (the S23 also passes once it has stopped
  cooling, at most 32.5 C, as in `devices.json`), CPU frequency caps back at baseline, airplane mode with Wi-Fi kept
  for adb (or mobile data/Bluetooth/location off if this phone drops Wi-Fi in airplane mode), Do Not Disturb,
  background apps killed, on battery, screen off (S23: off with a wake lock; OnePlus 8 Pro: awake with only the
  panel off, since it caps its CPU when it sleeps, as in its profile), 4 CPU threads, MNN precision low / memory low
  (the published `config.json` and llm_bench's defaults).
- **Requests**: `requests/<scenario>.jsonl`, the first 300 rows of the run's frozen held-out set
  (`logs/runs/phone_in_the_loop/<run>/artifacts/eval_set.json`, stored order, already shuffled), each prompt
  rendered exactly as the factory scored the MNN model (`_serving_prompt_prefix`: ChatML user turn + empty think
  block, `use_template=false`) and tokenized with the run's tokenizer. 3 warm-up requests come from the end of the
  eval set and are not measured. `pb_requests` gets token ids, so no tokenization is on the timed path.
- **Answer length**: `--answer-length reference` (default) decodes exactly as many tokens as the reference answer
  (the entity JSON for M07, the first human summary for H09), with EOS ignored: the same number of forward passes
  as that answer plus its end-of-turn token. `--answer-length eos` decodes greedily until the model's own
  end-of-turn (cap 512, as in the factory's eval) and keeps the generated text in `requests.csv`, so the outputs can
  be scored later.
- **Per request** (`pb_requests.cpp`, CLOCK_BOOTTIME stamps): prefill tokens/s = prompt tokens / time to first
  token (embedding + prefill forward); decode tokens/s = decode forward passes / time from first token to the end
  (sampling included, like the harness's MNN decode); process Pss, VmRSS, VmHWM, VmSwap after each request.
  `dumpsys meminfo` of the process every 60 s, and battery temperature, CPU caps and thermal status every 30 s.
- **Energy**: the harness's `EnergyRecorder` (battery gauge every 100 ms through the battery service on the OnePlus 8
  Pro, Perfetto's battery counters on the S23, chip counters where readable) for the whole session; idle power for
  60 s before and after each batch.

## Output

`results/fig9_<scenario>_<profile>_<answer-length>_<stamp>/`:
- `summary.md`, `summary.json`: the comparison table, distributions, conditions, per-batch details.
- `requests.csv`: one row per request (tokens, time to first token, prefill/decode tokens/s, memory, battery
  temperature, indicative energy, generated text).
- `session.json`, `raw/` (pb_requests stdout/stderr/outputs), `energy/` (sampler and trace).

`python fig9_validation/run_fig9.py --summarize <results dir>` recomputes the summary without the phone.

How each number is compared:

| Number | Statistic | Against |
|---|---|---|
| Peak memory | max Pss over the run (PSS + GPU, as in the profile); max VmHWM shown as a conservative check | ≤ memory limit |
| Prefill | median; p95 = the slow tail, the speed 95% of requests reach (5th percentile of tokens/s) | ≥ prefill limit |
| Decode | median; p95 slow tail as above | ≥ decode limit |
| Energy | average per request = whole-phone energy over the batch, net of the idle power measured just before it, ÷ requests | ≤ energy limit |
| Latency (secondary) | median and p95 time to first token; p95 whole request | the scenario's latency limits |

`summary.json` also has the 95th percentile of tokens/s (the fast tail), MNN's own internal timings, energy net of
the post-batch idle (sensitivity to idle drift), and the profile's values for this configuration.

## What cannot be measured per request, and how it is handled

- **Energy per request.** One request takes ~0.5-2 s, but the battery gauge reports a moving average that updates
  roughly every 0.1-5 s depending on the phone, so a single request's energy cannot be resolved. The headline is
  therefore the batch total: integrate whole-phone power over the ~6 min of back-to-back requests, subtract the idle
  power measured right before the batch, divide by the 300 requests. `requests.csv` still has a per-request value and
  the number of gauge updates inside each request; treat it as indicative only. The prefill/decode split
  (`phase_split_indicative`, mJ per token) is integrated over all prefill and all decode windows and is blurred by
  the gauge's lag. `--repeats 3` gives three independent batch totals.
- **Energy is whole phone, net of idle** (as in the profile and the requirement), so it includes the SoC, memory and
  the rest of the board while the model runs, not the screen (off). Idle drift over a batch is bounded by comparing
  the pre- and post-batch idle references.
- **Peak memory** is read between requests (Pss) and every 60 s (dumpsys); VmHWM is the kernel's exact high-water
  mark of resident memory and is never lower than the true peak resident set.
- **Sustained heat.** Requests run back to back, so the phone warms up during the batch; only the batch starts from a
  cool phone. The battery temperature and CPU caps are logged through the batch, so throttling is visible.

## Notes

- The model files are the published ones (identical to the factory's validated artifacts). The profile values above
  were measured on MNN's default export of the base model (MNN 510ac8f for Qwen3-0.6B, the harness's head build for
  Qwen3.5) with a synthetic 256 + 256-token workload; this run uses the fine-tuned HQQ (M07) / 8-bit (H09) exports on
  the exporter's runtime with real requests, so differences from the profile are expected and are what is validated.
- Each request is prefilled in full (fresh context, `reuse_kv=false`, as the factory scored). The requirement assumed
  a cached instruction prefix (40 / 30 tokens), so time to first token and energy here are conservative relative to
  the requirement's request shape; prefill tokens/s is comparable.
- If MNN 47ccf6c does not build or load on a phone: `python fig9_validation/build_pb_requests.py --mnn-ref master
  --push --serial <serial>`, then add `--mnn-sha <the commit it prints>` to `run_fig9.py`.
- Rebuilding the request sets (cluster, repo environment, single-threaded):
  `OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv/bin/python hardware/Benchmark-Harness/paper_table5/fig9_validation/build_requests.py`
