# Table 5 replication (arXiv 2607.05475): 256-token prefill and decode

Generated 2026-10-14T01:46:48.

- **Galaxy S23**: sessions slm_pool_s23_pinned, slm_pool_s23_head; paper column: none (not in the paper); compared with xiaomi14 as the closest paper phone; energy: whole device (net of idle) via perfetto:current

Throughput in tokens/s (mean +/- std over the recorded trials); energy in uJ/token. `*` = energy recorded but invalid (e.g. phone on USB power with no chip counters); `^` = started before the phone cooled to its gate temperature (gate timed out; see the result JSON); `~` = GPU row whose weight type has no OpenCL matmul kernel in that llama.cpp build (e.g. Q4_K at eadc418), so most matmuls ran on the CPU; FAIL = configuration did not run (see the result JSON).

| Model | Backend | Framework | Quant. | Prefill t/s (Galaxy S23) | Prefill uJ/token (Galaxy S23) | Decode t/s (Galaxy S23) | Decode uJ/token (Galaxy S23) |
|---|---|---|---|--:|--:|--:|--:|
| gemma3-270m | CPU | llama.cpp | Q4_0 | 799.2 +/- 0.1 | 8.3e3 | 132.5 +/- 2.8 | 7.0e4 |
|  | CPU | llama.cpp | Q8_0 | 790.2 +/- 0.1 | 9.0e3 | 117.8 +/- 1.4 | 8.1e4 |
|  | CPU | llama.cpp | F16 | 126.7 +/- 0.0 | 4.5e4 | 38.4 +/- 0.7 | 2.0e5 |
|  | CPU | MNN | Q4 | 1510.8 +/- 147.8 | 4.6e3 | 107.7 +/- 2.2 | 8.1e4 |
|  | CPU | MNN | Q8 | 1465.2 +/- 127.9 | 4.8e3 | 82.2 +/- 1.5 | 1.1e5 |
|  | CPU | MNN | F16 | 1082.3 +/- 43.3 | 6.0e3 | 56.2 +/- 1.7 | 1.6e5 |
|  | GPU | llama.cpp | Q4_0 | 360.8 +/- 5.4 | 1.0e4 | 33.3 +/- 1.9 | 9.3e4 |
|  | GPU | llama.cpp | Q8_0 | 26.8 +/- 0.1 | 1.2e5 | 35.6 +/- 0.5 | 1.1e5 |
|  | GPU | llama.cpp | F16 | 24.4 +/- 0.0 | 1.4e5 | 29.3 +/- 0.9 | 1.4e5 |
|  | GPU | MNN | Q4 | 1622.0 +/- 7.5 | 1.9e3 | 9.0 +/- 0.1 | 1.2e5 |
|  | GPU | MNN | Q8 | 1588.5 +/- 6.6 | 2.0e3 | 8.9 +/- 0.1 | 1.4e5 |
|  | GPU | MNN | F16 | FAIL | FAIL | FAIL | FAIL |
| qwen3-0.6b | CPU | llama.cpp | Q4_0 | 232.0 +/- 2.0 | 2.7e4 | 49.8 +/- 1.1 | 1.6e5 |
|  | CPU | llama.cpp | Q8_0 | 216.4 +/- 3.0 | 2.9e4 | 47.1 +/- 0.4 | 1.9e5 |
|  | CPU | llama.cpp | F16 | 29.6 +/- 0.0 | 1.8e5 | 16.4 +/- 0.2 | 4.3e5 |
|  | CPU | MNN | Q4 | 370.8 +/- 17.7 | 1.7e4 | 72.7 +/- 0.8 | 1.3e5 |
|  | CPU | MNN | Q8 | 368.0 +/- 14.0 | 1.8e4 | 49.8 +/- 0.3 | 1.8e5 |
|  | CPU | MNN | F16 | 240.3 +/- 17.3 | 2.4e4 | 26.5 +/- 3.3 | 3.4e5 |
|  | GPU | llama.cpp | Q4_0 | 132.6 +/- 0.2 | 2.7e4 | 14.9 +/- 0.0 | 1.9e5 |
|  | GPU | llama.cpp | Q8_0 | 6.3 +/- 0.0 | 5.4e5 | 26.5 +/- 0.6 | 1.9e5 |
|  | GPU | llama.cpp | F16 | 5.3 +/- 0.0 | 6.6e5 | 22.2 +/- 0.3 | 2.7e5 |
|  | GPU | MNN | Q4 | 558.9 +/- 1.4 | 8.0e3 | 9.7 +/- 1.3 | 1.7e5 |
|  | GPU | MNN | Q8 | 539.7 +/- 1.6 | 8.4e3 | 8.4 +/- 0.1 | 2.0e5 |
|  | GPU | MNN | F16 | FAIL | FAIL | FAIL | FAIL |
| qwen3-1.7b | CPU | llama.cpp | Q4_0 | 119.2 +/- 1.1 | 5.3e4 | 24.4 +/- 0.5 | 3.3e5 |
|  | CPU | llama.cpp | Q8_0 | 106.5 +/- 1.1 | 6.3e4 | 20.5 +/- 0.2 | 4.2e5 |
|  | CPU | llama.cpp | F16 | 9.2 +/- 0.1 | 5.5e5 | 5.7 +/- 0.4 | 1.1e6 |
|  | CPU | MNN | Q4 | 137.6 +/- 4.2 | 4.4e4 | 31.3 +/- 0.3 | 3.0e5 |
|  | CPU | MNN | Q8 | 132.4 +/- 2.6 | 4.7e4 | 20.4 +/- 0.1 | 4.0e5 |
|  | CPU | MNN | F16 | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | Q4_0 | 106.8 +/- 0.1 | 3.9e4 | 12.2 +/- 0.0 | 3.5e5 |
|  | GPU | llama.cpp | Q8_0 | 2.0 +/- 0.0^ | 1.5e6^ | 16.3 +/- 0.2^ | 3.7e5^ |
|  | GPU | llama.cpp | F16 | 1.7 +/- 0.0 | 2.2e6 | 11.2 +/- 0.1 | 6.0e5 |
|  | GPU | MNN | Q4 | 219.1 +/- 0.4 | 2.5e4 | 8.5 +/- 0.1 | 3.0e5 |
|  | GPU | MNN | Q8 | 211.1 +/- 0.2 | 2.8e4 | 8.0 +/- 0.1 | 4.4e5 |
|  | GPU | MNN | F16 | FAIL | FAIL | FAIL | FAIL |
| qwen3-4b-instruct-2507 | CPU | llama.cpp | Q4_0 | 44.5 +/- 0.5^ | 1.3e5^ | 10.8 +/- 0.2^ | 6.8e5^ |
|  | CPU | llama.cpp | Q8_0 | FAIL | FAIL | FAIL | FAIL |
|  | CPU | llama.cpp | F16 | FAIL | FAIL | FAIL | FAIL |
|  | CPU | MNN | Q4 | 53.5 +/- 1.9 | 1.1e5 | 14.1 +/- 0.1 | 6.0e5 |
|  | CPU | MNN | Q8 | FAIL | FAIL | FAIL | FAIL |
|  | CPU | MNN | F16 | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | Q4_0 | 40.4 +/- 0.2 | 1.1e5 | 8.7 +/- 0.1 | 6.5e5 |
|  | GPU | llama.cpp | Q8_0 | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | F16 | FAIL | FAIL | FAIL | FAIL |
|  | GPU | MNN | Q4 | 88.4 +/- 0.0 | 6.9e4 | 6.7 +/- 0.0 | 6.2e5 |
|  | GPU | MNN | Q8 | FAIL | FAIL | FAIL | FAIL |
|  | GPU | MNN | F16 | FAIL | FAIL | FAIL | FAIL |
| qwen3.5-0.8b | CPU | llama.cpp | Q4_0 @head | 291.4 +/- 5.8 | - | 43.0 +/- 0.9 | 2.0e5 |
|  | CPU | llama.cpp | Q8_0 @head | 260.2 +/- 2.3 | 2.6e4 | 41.2 +/- 0.3 | 2.2e5 |
|  | CPU | llama.cpp | F16 @head | 26.4 +/- 0.3 | 1.8e5 | 13.7 +/- 0.2 | 5.3e5 |
|  | CPU | MNN | Q4 @head | 273.4 +/- 10.3 | 2.2e4 | 73.6 +/- 0.1 | 1.2e5 |
|  | CPU | MNN | Q8 @head | 273.9 +/- 8.9 | 2.3e4 | 47.0 +/- 0.2 | 2.0e5 |
|  | CPU | MNN | F16 @head | 217.0 +/- 5.6 | 2.8e4 | 20.9 +/- 0.7 | 3.8e5 |
|  | GPU | llama.cpp | Q4_0 @head | 157.2 +/- 1.4 | 2.2e4 | 13.4 +/- 0.1 | 2.5e5 |
|  | GPU | llama.cpp | Q8_0 @head | 162.5 +/- 1.1 | 2.2e4 | 15.2 +/- 0.2 | 2.8e5 |
|  | GPU | llama.cpp | F16 @head | 159.1 +/- 1.7 | 2.7e4 | 11.2 +/- 0.1 | 4.0e5 |
|  | GPU | MNN | Q4 @head | 454.4 +/- 7.5 | 7.5e3 | 46.1 +/- 0.3 | 1.1e5 |
|  | GPU | MNN | Q8 @head | 379.1 +/- 7.0 | 1.0e4 | 35.1 +/- 0.1 | 1.8e5 |
|  | GPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
| qwen3.5-2b | CPU | llama.cpp | Q4_0 @head | 131.9 +/- 1.4 | 4.9e4 | 21.2 +/- 0.4 | 3.7e5 |
|  | CPU | llama.cpp | Q8_0 @head | 112.3 +/- 8.4 | 6.1e4 | 19.4 +/- 0.5 | 4.5e5 |
|  | CPU | llama.cpp | F16 @head | 9.4 +/- 0.1 | 4.9e5 | 5.2 +/- 0.3 | 1.2e6 |
|  | CPU | MNN | Q4 @head | 134.4 +/- 3.5 | 4.7e4 | 34.1 +/- 0.6 | 3.0e5 |
|  | CPU | MNN | Q8 @head | 128.1 +/- 1.6 | 5.0e4 | 20.9 +/- 0.1 | 4.2e5 |
|  | CPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | Q4_0 @head | 108.8 +/- 0.8 | 2.9e4 | 11.3 +/- 0.2 | 3.5e5 |
|  | GPU | llama.cpp | Q8_0 @head | 113.0 +/- 2.3 | 3.5e4 | 11.3 +/- 0.1 | 4.8e5 |
|  | GPU | llama.cpp | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | MNN | Q4 @head | 251.9 +/- 2.6 | 1.5e4 | 28.0 +/- 0.1 | 2.3e5 |
|  | GPU | MNN | Q8 @head | 203.1 +/- 3.9 | 2.3e4 | 19.0 +/- 0.0 | 3.9e5 |
|  | GPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
| qwen3.5-4b | CPU | llama.cpp | Q4_0 @head | 50.8 +/- 1.5 | 1.2e5 | 10.0 +/- 0.2 | 7.6e5 |
|  | CPU | llama.cpp | Q8_0 @head | FAIL | FAIL | FAIL | FAIL |
|  | CPU | llama.cpp | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | CPU | MNN | Q4 @head | 49.9 +/- 1.3 | 1.1e5 | 14.6 +/- 0.4 | 5.8e5 |
|  | CPU | MNN | Q8 @head | FAIL | FAIL | FAIL | FAIL |
|  | CPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | Q4_0 @head | 47.5 +/- 0.3 | 8.5e4 | 6.5 +/- 0.0 | 7.4e5 |
|  | GPU | llama.cpp | Q8_0 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | MNN | Q4 @head | 114.5 +/- 0.2 | 4.0e4 | 15.3 +/- 0.0 | 5.6e5 |
|  | GPU | MNN | Q8 @head | FAIL | FAIL | FAIL | FAIL |
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
|  | CPU | llama.cpp | F16 | 42.1 +/- 0.0 | 1.3e5 | 25.9 +/- 0.6 | 2.9e5 |
|  | CPU | MNN | Q4 | 554.2 +/- 44.3 | 1.2e4 | 115.2 +/- 1.4 | 7.9e4 |
|  | CPU | MNN | Q8 | 553.0 +/- 37.0 | 1.2e4 | 79.9 +/- 0.9 | 1.1e5 |
|  | CPU | MNN | F16 | 405.1 +/- 13.7 | 1.5e4 | 40.8 +/- 1.9 | 2.0e5 |
|  | GPU | llama.cpp | Q4_0 | 218.6 +/- 0.6 | 1.7e4 | 49.5 +/- 3.1 | 1.1e5 |
|  | GPU | llama.cpp | Q8_0 | 8.9 +/- 0.0 | 3.8e5 | 31.6 +/- 0.4 | 1.5e5 |
|  | GPU | llama.cpp | F16 | 7.7 +/- 0.0 | 4.4e5 | 26.0 +/- 0.3 | 1.8e5 |
|  | GPU | MNN | Q4 | 763.1 +/- 5.8 | 5.2e3 | 9.1 +/- 1.1 | 1.4e5 |
|  | GPU | MNN | Q8 | 734.8 +/- 4.1 | 5.3e3 | 8.6 +/- 0.3 | 1.7e5 |
|  | GPU | MNN | F16 | FAIL | FAIL | FAIL | FAIL |
