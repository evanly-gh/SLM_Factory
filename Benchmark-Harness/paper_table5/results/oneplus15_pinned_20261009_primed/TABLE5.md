# Table 5 replication (arXiv 2607.05475): 256-token prefill and decode

Generated 2026-10-09T12:52:58.

- **OnePlus 15**: sessions oneplus15_pinned_20261009_primed; paper column: oneplus15; energy: whole device (net of idle) via n/a

Throughput in tokens/s (mean +/- std over the recorded trials); energy in uJ/token. `*` = energy recorded but invalid (e.g. phone on USB power with no chip counters); `^` = started before the phone cooled to its gate temperature (gate timed out; see the result JSON); `~` = GPU row whose weight type has no OpenCL matmul kernel in that llama.cpp build (e.g. Q4_K at eadc418), so most matmuls ran on the CPU; FAIL = configuration did not run (see the result JSON).

| Model | Backend | Framework | Quant. | Prefill t/s (OnePlus 15) | Prefill uJ/token (OnePlus 15) | Decode t/s (OnePlus 15) | Decode uJ/token (OnePlus 15) |
|---|---|---|---|--:|--:|--:|--:|
| llama3.2-1b | CPU | llama.cpp | Q4_0 | 369.9 +/- 4.1 | - | 56.1 +/- 0.6 | - |
|  | CPU | MNN | Q4 | 150.0 +/- 1.2 | - | 53.5 +/- 0.3 | - |
|  | GPU | llama.cpp | Q4_0 | 1073.0 +/- 9.4 | - | 37.6 +/- 0.2 | - |
|  | GPU | MNN | Q4 | 681.8 +/- 4.5 | - | 18.8 +/- 0.4 | - |
| llama3.2-3b | CPU | llama.cpp | Q4_0 | 134.7 +/- 1.6 | - | 23.9 +/- 0.1 | - |
|  | CPU | MNN | Q4 | 67.0 +/- 22.3 | - | 21.5 +/- 0.0 | - |
|  | GPU | llama.cpp | Q4_0 | 413.6 +/- 1.3 | - | 19.3 +/- 0.1 | - |
|  | GPU | MNN | Q4 | 258.3 +/- 0.9 | - | 11.4 +/- 0.0 | - |
| qwen2.5-1.5b | CPU | llama.cpp | Q4_0 | 283.2 +/- 1.7 | - | 45.2 +/- 0.3 | - |
|  | CPU | MNN | Q4 | 111.9 +/- 0.8 | - | 41.3 +/- 0.2 | - |
|  | GPU | llama.cpp | Q4_0 | 747.6 +/- 6.6 | - | 18.5 +/- 0.3 | - |
|  | GPU | MNN | Q4 | 475.8 +/- 2.1 | - | 11.6 +/- 0.2 | - |

## Comparison with the paper (20/24 cells pass)

Pass = within 25% of the paper's value for the same phone (or the closest paper phone for a phone that is not in the paper); the paper's spread across its four phones is shown for context. Energy: the paper reports SoC energy; a phone without chip counters reports whole-device energy net of idle, which reads higher.

| Model | Backend | Framework | Quant. | Phone | Metric | Ours | Paper | Ratio | Paper spread | Pass |
|---|---|---|---|---|---|--:|--:|--:|---|:-:|
| llama3.2-1b | CPU | llama.cpp | Q4_0 | OnePlus 15 | Prefill t/s | 369.9 | 287.6 | 1.29x | 166.3-386.3 | NO |
| llama3.2-1b | CPU | llama.cpp | Q4_0 | OnePlus 15 | Decode t/s | 56.1 | 72.2 | 0.78x | 41.9-76.1 | yes |
| llama3.2-1b | CPU | MNN | Q4 | OnePlus 15 | Prefill t/s | 150.0 | 184.8 | 0.81x | 29.8-324.8 | yes |
| llama3.2-1b | CPU | MNN | Q4 | OnePlus 15 | Decode t/s | 53.5 | 61.6 | 0.87x | 3.2-66.6 | yes |
| llama3.2-1b | GPU | llama.cpp | Q4_0 | OnePlus 15 | Prefill t/s | 1073.0 | 986.8 | 1.09x | 450.1-986.8 | yes |
| llama3.2-1b | GPU | llama.cpp | Q4_0 | OnePlus 15 | Decode t/s | 37.6 | 64.4 | 0.58x | 22.5-64.4 | NO |
| llama3.2-1b | GPU | MNN | Q4 | OnePlus 15 | Prefill t/s | 681.8 | 669.8 | 1.02x | 371.3-692.1 | yes |
| llama3.2-1b | GPU | MNN | Q4 | OnePlus 15 | Decode t/s | 18.8 | 21.6 | 0.87x | 16.8-60.7 | yes |
| llama3.2-3b | CPU | llama.cpp | Q4_0 | OnePlus 15 | Prefill t/s | 134.7 | 109.8 | 1.23x | 58.6-119.7 | yes |
| llama3.2-3b | CPU | llama.cpp | Q4_0 | OnePlus 15 | Decode t/s | 23.9 | 30.5 | 0.78x | 17.1-30.5 | yes |
| llama3.2-3b | CPU | MNN | Q4 | OnePlus 15 | Prefill t/s | 67.0 | 66.1 | 1.01x | 66.1-123.4 | yes |
| llama3.2-3b | CPU | MNN | Q4 | OnePlus 15 | Decode t/s | 21.5 | 24.9 | 0.86x | 12.4-24.9 | yes |
| llama3.2-3b | GPU | llama.cpp | Q4_0 | OnePlus 15 | Prefill t/s | 413.6 | 357.6 | 1.16x | 161.1-357.6 | yes |
| llama3.2-3b | GPU | llama.cpp | Q4_0 | OnePlus 15 | Decode t/s | 19.3 | 25.0 | 0.77x | 11.1-25.2 | yes |
| llama3.2-3b | GPU | MNN | Q4 | OnePlus 15 | Prefill t/s | 258.3 | 248.7 | 1.04x | 145.3-248.7 | yes |
| llama3.2-3b | GPU | MNN | Q4 | OnePlus 15 | Decode t/s | 11.4 | 12.1 | 0.94x | 12.1-25.4 | yes |
| qwen2.5-1.5b | CPU | llama.cpp | Q4_0 | OnePlus 15 | Prefill t/s | 283.2 | 299.9 | 0.94x | 157.2-417.3 | yes |
| qwen2.5-1.5b | CPU | llama.cpp | Q4_0 | OnePlus 15 | Decode t/s | 45.2 | 55.7 | 0.81x | 34.5-55.7 | yes |
| qwen2.5-1.5b | CPU | MNN | Q4 | OnePlus 15 | Prefill t/s | 111.9 | 228.3 | 0.49x | 228.3-349.5 | NO |
| qwen2.5-1.5b | CPU | MNN | Q4 | OnePlus 15 | Decode t/s | 41.3 | 47.5 | 0.87x | 19.0-49.5 | yes |
| qwen2.5-1.5b | GPU | llama.cpp | Q4_0 | OnePlus 15 | Prefill t/s | 747.6 | 754.8 | 0.99x | 365.7-754.8 | yes |
| qwen2.5-1.5b | GPU | llama.cpp | Q4_0 | OnePlus 15 | Decode t/s | 18.5 | 50.3 | 0.37x | 31.8-50.3 | NO |
| qwen2.5-1.5b | GPU | MNN | Q4 | OnePlus 15 | Prefill t/s | 475.8 | 434.1 | 1.10x | 272.5-434.1 | yes |
| qwen2.5-1.5b | GPU | MNN | Q4 | OnePlus 15 | Decode t/s | 11.6 | 12.3 | 0.94x | 12.3-45.8 | yes |
