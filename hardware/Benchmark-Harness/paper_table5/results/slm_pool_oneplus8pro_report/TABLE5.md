# Table 5 replication (arXiv 2607.05475): 256-token prefill and decode

Generated 2026-10-12T21:47:35.

- **OnePlus 8 Pro**: sessions slm_pool_oneplus8pro_pinned; paper column: none (not in the paper); compared with xiaomi14 as the closest paper phone; energy: whole device (net of idle) via n/a

Throughput in tokens/s (mean +/- std over the recorded trials); energy in uJ/token. `*` = energy recorded but invalid (e.g. phone on USB power with no chip counters); `^` = started before the phone cooled to its gate temperature (gate timed out; see the result JSON); `~` = GPU row whose weight type has no OpenCL matmul kernel in that llama.cpp build (e.g. Q4_K at eadc418), so most matmuls ran on the CPU; FAIL = configuration did not run (see the result JSON).

| Model | Backend | Framework | Quant. | Prefill t/s (OnePlus 8 Pro) | Prefill uJ/token (OnePlus 8 Pro) | Decode t/s (OnePlus 8 Pro) | Decode uJ/token (OnePlus 8 Pro) |
|---|---|---|---|--:|--:|--:|--:|
| gemma3-270m | CPU | llama.cpp | Q4_0 | 1012.7 +/- 7.0 | - | 84.0 +/- 0.2 | - |
|  | CPU | MNN | Q4 | 899.1 +/- 1.4 | - | 64.4 +/- 0.3 | - |
|  | GPU | llama.cpp | Q4_0 | 82.3 +/- 0.1 | - | 7.2 +/- 0.1 | - |
|  | GPU | MNN | Q4 | 746.5 +/- 3.7 | - | 8.3 +/- 0.1 | - |
| qwen3-0.6b | CPU | llama.cpp | Q4_0 | 262.0 +/- 0.6^ | - | 47.6 +/- 0.2^ | - |
|  | CPU | MNN | Q4 | 229.5 +/- 0.5 | - | 42.5 +/- 0.8 | - |
|  | GPU | llama.cpp | Q4_0 | 35.2 +/- 0.1 | - | 3.5 +/- 0.0 | - |
|  | GPU | MNN | Q4 | 215.3 +/- 0.3 | - | 9.3 +/- 0.2 | - |
| qwen3-1.7b | CPU | llama.cpp | Q4_0 | 106.2 +/- 0.1 | - | 21.8 +/- 0.0 | - |
|  | CPU | MNN | Q4 | 81.7 +/- 0.1 | - | 19.7 +/- 0.1 | - |
|  | GPU | llama.cpp | Q4_0 | 27.0 +/- 0.0 | - | 2.5 +/- 0.0 | - |
|  | GPU | MNN | Q4 | 77.0 +/- 0.0 | - | 8.8 +/- 0.1 | - |
| qwen3-4b-instruct-2507 | CPU | llama.cpp | Q4_0 | 41.4 +/- 0.2 | - | 9.9 +/- 0.0 | - |
|  | CPU | MNN | Q4 | 32.6 +/- 0.1 | - | 8.9 +/- 0.0 | - |
|  | GPU | llama.cpp | Q4_0 | 11.7 +/- 0.0 | - | 1.7 +/- 0.0 | - |
|  | GPU | MNN | Q4 | 30.9 +/- 0.0 | - | 6.3 +/- 0.0 | - |
| smollm2-135m | CPU | llama.cpp | Q4_0 | 922.3 +/- 0.2 | - | 151.5 +/- 1.1 | - |
|  | CPU | MNN | Q4 | 836.3 +/- 1.1 | - | 142.8 +/- 2.0 | - |
|  | GPU | llama.cpp | Q4_0 | 115.2 +/- 0.2 | - | 12.7 +/- 0.1 | - |
|  | GPU | MNN | Q4 | 747.9 +/- 1.8 | - | 9.8 +/- 0.1 | - |
| smollm2-360m | CPU | llama.cpp | Q4_0 | 377.2 +/- 1.9 | - | 75.5 +/- 0.2 | - |
|  | CPU | MNN | Q4 | 326.8 +/- 0.5 | - | 71.3 +/- 2.7 | - |
|  | GPU | llama.cpp | Q4_0 | 63.2 +/- 0.1 | - | 9.0 +/- 0.0 | - |
|  | GPU | MNN | Q4 | 298.5 +/- 0.7 | - | 9.3 +/- 0.1 | - |
