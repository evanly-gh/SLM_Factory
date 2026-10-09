# Table 5 replication (arXiv 2607.05475): 256-token prefill and decode

Generated 2026-10-09T01:56:40.

- **Galaxy S23**: sessions slm_pool_s23_head; paper column: none (not in the paper); compared with xiaomi14 as the closest paper phone; energy: whole device (net of idle) via perfetto:current

Throughput in tokens/s (mean +/- std over the recorded trials); energy in uJ/token. `*` = energy recorded but invalid (e.g. phone on USB power with no chip counters); `^` = started before the phone cooled to its gate temperature (gate timed out; see the result JSON); `~` = GPU row whose weight type has no OpenCL matmul kernel in that llama.cpp build (e.g. Q4_K at eadc418), so most matmuls ran on the CPU; FAIL = configuration did not run (see the result JSON).

| Model | Backend | Framework | Quant. | Prefill t/s (Galaxy S23) | Prefill uJ/token (Galaxy S23) | Decode t/s (Galaxy S23) | Decode uJ/token (Galaxy S23) |
|---|---|---|---|--:|--:|--:|--:|
| qwen3.5-0.8b | CPU | llama.cpp | Q4_0 @head | 291.4 +/- 5.8 | - | 43.2 +/- 0.9 | - |
|  | CPU | llama.cpp | Q8_0 @head | 261.7 +/- 3.5 | - | 41.3 +/- 0.3 | - |
|  | CPU | llama.cpp | F16 @head | 26.6 +/- 0.3 | - | 13.5 +/- 0.2 | - |
|  | CPU | MNN | Q4 @head | 271.6 +/- 9.3 | - | 74.9 +/- 0.4 | - |
|  | CPU | MNN | Q8 @head | 274.6 +/- 8.6 | - | 47.7 +/- 0.3 | - |
|  | CPU | MNN | F16 @head | 215.9 +/- 5.0 | - | 21.3 +/- 1.9 | - |
|  | GPU | llama.cpp | Q4_0 @head | 155.3 +/- 2.0 | - | 12.9 +/- 0.1 | - |
|  | GPU | llama.cpp | Q8_0 @head | 163.0 +/- 1.8 | - | 14.6 +/- 0.1 | - |
|  | GPU | llama.cpp | F16 @head | 160.7 +/- 1.5 | - | 11.5 +/- 0.0 | - |
|  | GPU | MNN | Q4 @head | 456.9 +/- 5.8 | - | 50.4 +/- 0.1 | - |
|  | GPU | MNN | Q8 @head | 379.2 +/- 5.9 | - | 37.7 +/- 0.1 | - |
|  | GPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
| qwen3.5-2b | CPU | llama.cpp | Q4_0 @head | 133.0 +/- 1.2 | - | 21.0 +/- 0.3 | - |
|  | CPU | llama.cpp | Q8_0 @head | 120.0 +/- 0.9 | - | 19.8 +/- 0.2 | - |
|  | CPU | llama.cpp | F16 @head | 9.3 +/- 0.1 | - | 5.3 +/- 0.3 | - |
|  | CPU | MNN | Q4 @head | 134.4 +/- 3.5 | 4.7e4 | 34.2 +/- 0.6 | 2.9e5 |
|  | CPU | MNN | Q8 @head | 128.1 +/- 1.6 | 5.0e4 | 20.9 +/- 0.1 | 4.2e5 |
|  | CPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | Q4_0 @head | 108.3 +/- 1.2 | - | 11.3 +/- 0.1 | - |
|  | GPU | llama.cpp | Q8_0 @head | 113.0 +/- 0.6 | - | 11.3 +/- 0.0 | - |
|  | GPU | llama.cpp | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | MNN | Q4 @head | 251.9 +/- 2.6 | 1.5e4 | 30.3 +/- 0.1 | 2.1e5 |
|  | GPU | MNN | Q8 @head | 203.1 +/- 3.9 | 2.3e4 | 20.1 +/- 0.0 | 3.7e5 |
|  | GPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
| qwen3.5-4b | CPU | llama.cpp | Q4_0 @head | 50.8 +/- 1.5 | 1.2e5 | 10.0 +/- 0.2 | 7.6e5 |
|  | CPU | llama.cpp | Q8_0 @head | FAIL | FAIL | FAIL | FAIL |
|  | CPU | llama.cpp | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | CPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | Q4_0 @head | 47.5 +/- 0.3 | 8.5e4 | 6.5 +/- 0.0 | 7.4e5 |
|  | GPU | llama.cpp | Q8_0 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | llama.cpp | F16 @head | FAIL | FAIL | FAIL | FAIL |
|  | GPU | MNN | F16 @head | FAIL | FAIL | FAIL | FAIL |
