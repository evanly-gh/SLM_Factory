# Paper replication on the OnePlus 15 (w4: llama.cpp Q4_0, MNN 4-bit)

Generated from `results/paper_models_oneplus15_pinned/configs/*.json` and `paper_reference.json` (arXiv 2607.05475 Table 9, OnePlus 15 column). Same harness, flags and builds as the model-pool runs: pinned llama.cpp `eadc418` / MNN `510ac8f`, 4 llama.cpp threads, 256-token prompt, 256-token decode, 1 warm-up + 3 recorded repetitions, no priming, wake-kick on, unplugged over wireless adb, energy via Perfetto battery current. Ratio = ours / paper; ⚠ = outside 0.9–1.1×. Energy is whole-phone net of idle (ours) vs SoC-only (paper).

Four cells were rerun after the first pass (three suspected slow-governor runs, one swap-guard failure); the table shows the rerun values and the first-pass values are listed under it and kept in `first_pass/`. The phone was rebooted before the reruns.

| Model | Engine | Backend | Prefill ours | Prefill paper | Ratio | Decode ours | Decode paper | Ratio | Prefill mJ/tok ours | paper | Ratio | Decode mJ/tok ours | paper | Ratio | Note |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| Llama-3.2-1B | llama.cpp | CPU | 216.8 ± 0.1 | 287.6 | 0.75× ⚠ | 49.05 ± 0.03 | 72.2 | 0.68× ⚠ | 9.3 | 16.0 | 0.58× ⚠ | 81.3 | 74.0 | 1.10× | rerun |
| Llama-3.2-1B | llama.cpp | GPU | 1101.1 ± 1.9 | 986.8 | 1.12× ⚠ | 40.12 ± 0.05 | 64.4 | 0.62× ⚠ | 4.2 | 5.4 | 0.78× ⚠ | 105.2 | 110.0 | 0.96× |  |
| Llama-3.2-1B | MNN | CPU | 184.6 ± 59.3 | 184.8 | 1.00× | 51.73 ± 1.00 | 61.6 | 0.84× ⚠ | 12.9 | 15.0 | 0.86× ⚠ | 61.6 | 66.0 | 0.93× |  |
| Llama-3.2-1B | MNN | GPU | 686.1 ± 0.3 | 669.8 | 1.02× | 38.78 ± 0.04 | 21.6 | 1.80× ⚠ | 8.1 | 13.0 | 0.62× ⚠ | 84.9 | 79.0 | 1.08× |  |
| Qwen2.5-1.5B | llama.cpp | CPU | 162.5 ± 0.4 | 299.9 | 0.54× ⚠ | 41.91 ± 0.06 | 55.7 | 0.75× ⚠ | 11.6 | 15.0 | 0.77× ⚠ | 95.0 | 97.0 | 0.98× | rerun |
| Qwen2.5-1.5B | llama.cpp | GPU | 763.7 ± 1.9 | 754.8 | 1.01× | 35.24 ± 0.05 | 50.3 | 0.70× ⚠ | 8.1 | 11.0 | 0.73× ⚠ | 129.5 | 140.0 | 0.92× |  |
| Qwen2.5-1.5B | MNN | CPU | 137.8 ± 45.3 | 228.3 | 0.60× ⚠ | 40.63 ± 0.79 | 47.5 | 0.86× ⚠ | 18.0 | 15.0 | 1.20× ⚠ | 82.7 | 92.0 | 0.90× ⚠ | rerun |
| Qwen2.5-1.5B | MNN | GPU | 474.4 ± 0.5 | 434.1 | 1.09× | 27.95 ± 0.22 | 12.3 | 2.27× ⚠ | 10.8 | 17.0 | 0.64× ⚠ | 114.0 | 99.0 | 1.15× ⚠ |  |
| Llama-3.2-3B | llama.cpp | CPU | 77.8 ± 0.2 | 109.8 | 0.71× ⚠ | 21.54 ± 0.01 | 30.5 | 0.71× ⚠ | 29.4 | 41.0 | 0.72× ⚠ | 211.5 | 210.0 | 1.01× |  |
| Llama-3.2-3B | llama.cpp | GPU | 423.2 ± 0.5 | 357.6 | 1.18× ⚠ | 20.38 ± 0.02 | 25.0 | 0.82× ⚠ | 14.4 | 21.0 | 0.68× ⚠ | 238.3 | 250.0 | 0.95× |  |
| Llama-3.2-3B | MNN | CPU | 66.5 ± 22.0 | 66.1 | 1.01× | 21.60 ± 0.04 | 24.9 | 0.87× ⚠ | 39.3 | 44.0 | 0.89× ⚠ | 148.4 | 180.0 | 0.82× ⚠ |  |
| Llama-3.2-3B | MNN | GPU | 259.1 ± 0.1 | 248.7 | 1.04× | 24.56 ± 0.07 | 12.1 | 2.03× ⚠ | 22.5 | 35.0 | 0.64× ⚠ | 212.0 | 200.0 | 1.06× |  |
| Qwen2.5-7B | llama.cpp | CPU | 36.9 ± 0.1 | 58.6 | 0.63× ⚠ | 10.98 ± 0.00 | 14.3 | 0.77× ⚠ | 62.0 | 79.0 | 0.78× ⚠ | 418.0 | 440.0 | 0.95× |  |
| Qwen2.5-7B | llama.cpp | GPU | 198.1 ± 3.1 | 171.6 | 1.15× ⚠ | 11.29 ± 0.02 | 12.2 | 0.93× | 35.2 | 65.0 | 0.54× ⚠ | 476.7 | 570.0 | 0.84× ⚠ |  |
| Qwen2.5-7B | MNN | CPU | 29.9 ± 10.4 | 30.0 | 1.00× | 10.47 ± 0.10 | 12.2 | 0.86× ⚠ | 109.1 | 100.0 | 1.09× | 343.8 | 390.0 | 0.88× ⚠ | rerun |
| Qwen2.5-7B | MNN | GPU | 97.4 ± 2.8 | 102.1 | 0.95× | 11.54 ± 0.02 | 10.6 | 1.09× | 48.9 | 91.0 | 0.54× ⚠ | 437.7 | 510.0 | 0.86× ⚠ |  |

## First-pass values of the rerun cells

| Cell | First-pass prefill | First-pass decode | Rerun prefill | Rerun decode | Why rerun |
|---|--:|--:|--:|--:|---|
| llama3.2-1b__Q4_0__llama.cpp__cpu | 92.3 | 49.18 | 216.8 | 49.05 | prefill far below normal with normal decode (slow-governor symptom) |
| qwen2.5-1.5b__Q4_0__llama.cpp__cpu | 28.8 | 41.68 | 162.5 | 41.91 | prefill far below normal with normal decode (slow-governor symptom) |
| qwen2.5-1.5b__Q4__mnn__cpu | 111.7 | 41.29 | 137.8 | 40.63 | prefill at 0.49x of the paper, checked for stability |
| qwen2.5-7b__Q4__mnn__cpu | — (failed) | — | 29.9 | 10.47 | swap guard stopped the first pass (1069 MB swapped); phone rebooted before the rerun |

## Summary

- Cells within 0.9–1.1× of the paper (of 16): prefill 8, decode 2, prefill energy 1, decode energy 10.
