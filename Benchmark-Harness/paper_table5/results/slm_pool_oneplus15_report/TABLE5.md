# Table 5 replication (arXiv 2607.05475): 256-token prefill and decode

Generated 2026-10-12T22:27:16.

- **OnePlus 15**: sessions slm_pool_oneplus15_pinned; paper column: oneplus15; energy: whole device (net of idle) via perfetto:current

Throughput in tokens/s (mean +/- std over the recorded trials); energy in uJ/token. `*` = energy recorded but invalid (e.g. phone on USB power with no chip counters); `^` = started before the phone cooled to its gate temperature (gate timed out; see the result JSON); `~` = GPU row whose weight type has no OpenCL matmul kernel in that llama.cpp build (e.g. Q4_K at eadc418), so most matmuls ran on the CPU; FAIL = configuration did not run (see the result JSON).

| Model | Backend | Framework | Quant. | Prefill t/s (OnePlus 15) | Prefill uJ/token (OnePlus 15) | Decode t/s (OnePlus 15) | Decode uJ/token (OnePlus 15) |
|---|---|---|---|--:|--:|--:|--:|
| gemma3-270m | CPU | llama.cpp | Q4_0 | 1090.2 +/- 1.2 | 1.8e3 | 185.6 +/- 0.8 | 1.6e4 |
|  | CPU | MNN | Q4 | 1278.4 +/- 302.1 | - | 116.0 +/- 0.2 | - |
|  | GPU | llama.cpp | Q4_0 | 2874.9 +/- 4.4 | 1.3e3 | 73.2 +/- 2.8 | 3.6e4 |
|  | GPU | MNN | Q4 | 3888.8 +/- 4.1 | - | 27.4 +/- 0.2 | - |
| qwen3-0.6b | CPU | llama.cpp | Q4_0 | 309.4 +/- 0.1 | 6.3e3 | 72.0 +/- 0.1 | 4.7e4 |
|  | CPU | MNN | Q4 | 398.1 +/- 93.3 | 5.7e3 | 79.0 +/- 0.6 | 3.7e4 |
|  | GPU | llama.cpp | Q4_0 | 1679.6 +/- 10.0 | 2.3e3 | 61.7 +/- 0.0 | 7.0e4 |
|  | GPU | MNN | Q4 | 1252.2 +/- 10.2 | 3.1e3 | 32.0 +/- 0.1 | 6.3e4 |
| qwen3-1.7b | CPU | llama.cpp | Q4_0 | 144.1 +/- 0.2 | 1.3e4 | 34.9 +/- 0.0 | 1.2e5 |
|  | CPU | MNN | Q4 | 102.8 +/- 0.2 | 1.9e4 | 36.8 +/- 0.1 | 8.6e4 |
|  | GPU | llama.cpp | Q4_0 | 729.0 +/- 5.5 | 8.2e3 | 31.4 +/- 0.0 | 1.5e5 |
|  | GPU | MNN | Q4 | 486.2 +/- 1.7 | 9.6e3 | 25.6 +/- 0.0 | 1.3e5 |
| qwen3-4b-instruct-2507 | CPU | llama.cpp | Q4_0 | 56.4 +/- 0.1 | 4.1e4 | 16.3 +/- 0.3 | 2.7e5 |
|  | CPU | MNN | Q4 | 41.0 +/- 0.1 | 6.0e4 | 16.8 +/- 0.1 | 2.0e5 |
|  | GPU | llama.cpp | Q4_0 | 315.0 +/- 0.1 | 1.9e4 | 16.7 +/- 0.0 | 3.1e5 |
|  | GPU | MNN | Q4 | 191.1 +/- 0.3 | 3.2e4 | 19.7 +/- 0.0 | 2.8e5 |
| smollm2-135m | CPU | llama.cpp | Q4_0 | 1032.0 +/- 1.8 | - | 260.9 +/- 0.2 | - |
|  | CPU | MNN | Q4 | 1027.0 +/- 3.5 | - | 205.6 +/- 0.3 | - |
|  | GPU | llama.cpp | Q4_0 | 3634.0 +/- 11.1 | - | 93.2 +/- 2.9 | - |
|  | GPU | MNN | Q4 | 3666.1 +/- 12.2 | - | 42.2 +/- 0.1 | - |
| smollm2-360m | CPU | llama.cpp | Q4_0 | 456.2 +/- 2.7 | 4.0e3 | 124.5 +/- 0.1 | 2.4e4 |
|  | CPU | MNN | Q4 | 488.3 +/- 132.8 | 3.4e3 | 114.6 +/- 0.4 | 2.3e4 |
|  | GPU | llama.cpp | Q4_0 | 1829.9 +/- 0.5 | 2.5e3 | 85.6 +/- 0.1 | 4.4e4 |
|  | GPU | MNN | Q4 | 1653.0 +/- 3.2 | 2.6e3 | 39.3 +/- 0.2 | 5.4e4 |
