# Table 5 replication (arXiv 2607.05475): 256-token prefill and decode

Generated 2026-10-09T00:15:14.

- **OnePlus 15**: sessions oneplus15_pinned_20261008_230103; paper column: oneplus15; energy: whole device (net of idle) via n/a

Throughput in tokens/s (mean +/- std over the recorded trials); energy in uJ/token. `*` = energy recorded but invalid (e.g. phone on USB power with no chip counters); `^` = started before the phone cooled to its gate temperature (gate timed out; see the result JSON); `~` = GPU row whose weight type has no OpenCL matmul kernel in that llama.cpp build (e.g. Q4_K at eadc418), so most matmuls ran on the CPU; FAIL = configuration did not run (see the result JSON).

| Model | Backend | Framework | Quant. | Prefill t/s (OnePlus 15) | Prefill uJ/token (OnePlus 15) | Decode t/s (OnePlus 15) | Decode uJ/token (OnePlus 15) |
|---|---|---|---|--:|--:|--:|--:|
| llama3.2-1b | CPU | llama.cpp | Q4_0 | 394.0 +/- 1.6 | - | 37.7 +/- 30.5 | - |
|  | CPU | llama.cpp | Q4_K_M | 231.9 +/- 0.6 | - | 53.0 +/- 0.1 | - |
|  | CPU | MNN | Q4 | 143.0 +/- 12.8 | - | 54.1 +/- 0.3 | - |
|  | GPU | llama.cpp | Q4_0 | 1067.8 +/- 16.0 | - | 24.2 +/- 0.7 | - |
|  | GPU | llama.cpp | Q4_K_M | 28.5 +/- 0.1~ | - | 23.0 +/- 0.1~ | - |
|  | GPU | MNN | Q4 | 678.7 +/- 7.3 | - | 19.4 +/- 0.1 | - |

## Comparison with the paper (6/12 cells pass)

Pass = within 25% of the paper's value for the same phone (or the closest paper phone for a phone that is not in the paper); the paper's spread across its four phones is shown for context. Energy: the paper reports SoC energy; a phone without chip counters reports whole-device energy net of idle, which reads higher.

| Model | Backend | Framework | Quant. | Phone | Metric | Ours | Paper | Ratio | Paper spread | Pass |
|---|---|---|---|---|---|--:|--:|--:|---|:-:|
| llama3.2-1b | CPU | llama.cpp | Q4_0 | OnePlus 15 | Prefill t/s | 394.0 | 287.6 | 1.37x | 166.3-386.3 | NO |
| llama3.2-1b | CPU | llama.cpp | Q4_0 | OnePlus 15 | Decode t/s | 37.7 | 72.2 | 0.52x | 41.9-76.1 | NO |
| llama3.2-1b | CPU | llama.cpp | Q4_K_M | OnePlus 15 | Prefill t/s | 231.9 | 287.6 | 0.81x | 166.3-386.3 | yes |
| llama3.2-1b | CPU | llama.cpp | Q4_K_M | OnePlus 15 | Decode t/s | 53.0 | 72.2 | 0.73x | 41.9-76.1 | NO |
| llama3.2-1b | CPU | MNN | Q4 | OnePlus 15 | Prefill t/s | 143.0 | 184.8 | 0.77x | 29.8-324.8 | yes |
| llama3.2-1b | CPU | MNN | Q4 | OnePlus 15 | Decode t/s | 54.1 | 61.6 | 0.88x | 3.2-66.6 | yes |
| llama3.2-1b | GPU | llama.cpp | Q4_0 | OnePlus 15 | Prefill t/s | 1067.8 | 986.8 | 1.08x | 450.1-986.8 | yes |
| llama3.2-1b | GPU | llama.cpp | Q4_0 | OnePlus 15 | Decode t/s | 24.2 | 64.4 | 0.38x | 22.5-64.4 | NO |
| llama3.2-1b | GPU | llama.cpp | Q4_K_M | OnePlus 15 | Prefill t/s | 28.5 | 986.8 | 0.03x | 450.1-986.8 | NO |
| llama3.2-1b | GPU | llama.cpp | Q4_K_M | OnePlus 15 | Decode t/s | 23.0 | 64.4 | 0.36x | 22.5-64.4 | NO |
| llama3.2-1b | GPU | MNN | Q4 | OnePlus 15 | Prefill t/s | 678.7 | 669.8 | 1.01x | 371.3-692.1 | yes |
| llama3.2-1b | GPU | MNN | Q4 | OnePlus 15 | Decode t/s | 19.4 | 21.6 | 0.90x | 16.8-60.7 | yes |
