# Table 5 replication (arXiv 2607.05475): 256-token prefill and decode

Generated 2026-10-10T16:59:57.

- **Galaxy S23**: sessions slm_pool_s23_pinned, slm_pool_s23_head; paper column: none (not in the paper); compared with xiaomi14 as the closest paper phone; energy: whole device (net of idle) via perfetto:current

Throughput in tokens/s (mean +/- std over the recorded trials); energy in uJ/token. `*` = energy recorded but invalid (e.g. phone on USB power with no chip counters); `^` = started before the phone cooled to its gate temperature (gate timed out; see the result JSON); `~` = GPU row whose weight type has no OpenCL matmul kernel in that llama.cpp build (e.g. Q4_K at eadc418), so most matmuls ran on the CPU; FAIL = configuration did not run (see the result JSON).

| Model | Backend | Framework | Quant. | Prefill t/s (Galaxy S23) | Prefill uJ/token (Galaxy S23) | Decode t/s (Galaxy S23) | Decode uJ/token (Galaxy S23) |
|---|---|---|---|--:|--:|--:|--:|
| gemma3-270m | CPU | llama.cpp | Q4_0 | 799.2 +/- 0.1 | 8.3e3 | 132.5 +/- 2.8 | 7.0e4 |
|  | CPU | llama.cpp | Q8_0 | 790.2 +/- 0.1 | 9.0e3 | 117.8 +/- 1.4 | 8.1e4 |
|  | CPU | llama.cpp | F16 | 126.7 +/- 0.0 | 4.5e4 | 38.4 +/- 0.7 | 2.0e5 |
|  | CPU | MNN | Q4 | 1510.8 +/- 147.8 | 4.6e3 | 107.7 +/- 2.2 | 8.1e4 |
|  | CPU | MNN | Q8 | 1465.2 +/- 127.9 | 4.8e3 | 82.2 +/- 1.5 | 1.1e5 |
|  | CPU | MNN | F16 | 1082.3 +/- 43.3 | - | 56.2 +/- 1.7 | - |
|  | GPU | llama.cpp | Q4_0 | 360.8 +/- 5.4 | 1.0e4 | 33.3 +/- 1.9 | 9.3e4 |
|  | GPU | llama.cpp | Q8_0 | 26.8 +/- 0.1 | 1.2e5 | 35.6 +/- 0.5 | 1.1e5 |
|  | GPU | llama.cpp | F16 | 24.4 +/- 0.0 | 1.4e5 | 29.3 +/- 0.9 | 1.4e5 |
|  | GPU | MNN | Q4 | 1622.0 +/- 7.5 | 1.9e3 | 9.0 +/- 0.1 | 1.2e5 |
|  | GPU | MNN | Q8 | 1588.5 +/- 6.6 | 2.0e3 | 8.9 +/- 0.1 | 1.4e5 |
|  | GPU | MNN | F16 | FAIL | FAIL | FAIL | FAIL |
| qwen3-0.6b | CPU | llama.cpp | Q4_0 | 232.0 +/- 2.0 | 2.7e4 | 49.8 +/- 1.1 | 1.6e5 |
|  | CPU | MNN | Q4 | 370.8 +/- 17.7 | 1.7e4 | 72.7 +/- 0.8 | 1.3e5 |
|  | CPU | MNN | Q8 | 368.0 +/- 14.0 | - | 49.8 +/- 0.3 | - |
|  | GPU | llama.cpp | Q4_0 | 132.6 +/- 0.2 | 2.7e4 | 14.9 +/- 0.0 | 1.9e5 |
|  | GPU | MNN | Q4 | 558.9 +/- 1.4 | 8.0e3 | 9.7 +/- 1.3 | 1.7e5 |
|  | GPU | MNN | Q8 | 539.7 +/- 1.6 | - | 8.4 +/- 0.1 | - |
| qwen3.5-0.8b | GPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
| qwen3.5-2b | CPU | MNN | Q4 @head | 134.4 +/- 3.5 | 4.7e4 | 34.1 +/- 0.6 | 3.0e5 |
|  | CPU | MNN | Q8 @head | 128.1 +/- 1.6 | 5.0e4 | 20.9 +/- 0.1 | 4.2e5 |
|  | CPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | MNN | Q4 @head | 251.9 +/- 2.6 | 1.5e4 | 28.0 +/- 0.1 | 2.3e5 |
|  | GPU | MNN | Q8 @head | 203.1 +/- 3.9 | 2.3e4 | 19.0 +/- 0.0 | 3.9e5 |
|  | GPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
| qwen3.5-4b | CPU | llama.cpp | Q4_0 @head | 50.8 +/- 1.5 | 1.2e5 | 10.0 +/- 0.2 | 7.6e5 |
|  | CPU | llama.cpp | Q8_0 @head | FAIL | FAIL | FAIL | FAIL |
|  | CPU | llama.cpp | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | CPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | Q4_0 @head | 47.5 +/- 0.3 | 8.5e4 | 6.5 +/- 0.0 | 7.4e5 |
|  | GPU | llama.cpp | Q8_0 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
| smollm2-135m | CPU | llama.cpp | Q4_0 | 793.4 +/- 0.3 | 7.9e3 | 204.5 +/- 1.5 | 4.2e4 |
|  | CPU | llama.cpp | Q8_0 | 786.1 +/- 0.2 | 8.2e3 | 168.6 +/- 1.7 | 5.2e4 |
|  | CPU | llama.cpp | F16 | 119.7 +/- 0.0 | 4.7e4 | 68.5 +/- 1.5 | 1.1e5 |
|  | CPU | MNN | Q4 | 1518.0 +/- 6.4 | 4.7e3 | 224.3 +/- 1.7 | 4.0e4 |
|  | CPU | MNN | Q8 | 1508.5 +/- 73.7 | 5.0e3 | 174.9 +/- 5.0 | 5.1e4 |
|  | CPU | MNN | F16 | 976.4 +/- 159.6 | 6.2e3 | 111.8 +/- 4.8 | 7.6e4 |
|  | GPU | llama.cpp | Q4_0 | 409.7 +/- 1.7 | 8.4e3 | 39.6 +/- 0.5 | 7.1e4 |
|  | GPU | llama.cpp | Q8_0 | 25.8 +/- 0.0 | 1.3e5 | 67.3 +/- 3.1 | 8.7e4 |
|  | GPU | llama.cpp | F16 | 23.3 +/- 0.0 | 1.5e5 | 64.7 +/- 2.3 | 1.1e5 |
|  | GPU | MNN | Q4 | 1725.1 +/- 97.8 | 1.8e3 | 9.1 +/- 0.1 | 1.0e5 |
|  | GPU | MNN | Q8 | 1753.2 +/- 18.2 | 1.9e3 | 9.1 +/- 0.1 | 1.2e5 |
|  | GPU | MNN | F16 | FAIL | FAIL | FAIL | FAIL |
| smollm2-360m | CPU | llama.cpp | Q4_0 | 366.4 +/- 4.1 | 1.8e4 | 95.2 +/- 2.6 | 8.9e4 |
|  | CPU | llama.cpp | Q8_0 | 348.1 +/- 7.2 | 1.9e4 | 75.2 +/- 1.0 | 1.2e5 |
|  | CPU | MNN | Q4 | 554.2 +/- 44.3 | 1.2e4 | 115.2 +/- 1.4 | 7.9e4 |
|  | CPU | MNN | Q8 | 553.0 +/- 37.0 | 1.2e4 | 79.9 +/- 0.9 | 1.1e5 |
|  | GPU | llama.cpp | Q4_0 | 218.6 +/- 0.6 | 1.7e4 | 49.5 +/- 3.1 | 1.1e5 |
|  | GPU | llama.cpp | Q8_0 | 8.9 +/- 0.0 | 3.8e5 | 31.6 +/- 0.4 | 1.5e5 |
|  | GPU | MNN | Q4 | 763.1 +/- 5.8 | 5.2e3 | 9.1 +/- 1.1 | 1.4e5 |
|  | GPU | MNN | Q8 | 734.8 +/- 4.1 | 5.3e3 | 8.6 +/- 0.3 | 1.7e5 |
