# SmolChat CPU vs OpenCL — "What is the capital of France?"

- App: SmolChat (~/SLM_Factory-SmolChat, branch llama-pipeline + OpenCL fix), vendored llama.cpp 3018a11e7
- Models: Q4_K_M (verified files), device: OnePlus CPH2749 / SM8850 (Adreno 840)
- Protocol: run_autobench.py --trials 3 (1 discarded warm-up + 3 recorded), output ≤ 256 tokens (app default), CPU = n_gpu_layers 0, OpenCL = n_gpu_layers 99
- Values: mean ± std over recorded trials

| Model | Backend | Runs | Verified | Prefill t/s | Decode t/s | TTFT ms | Energy mJ* | Avg mA* |
|---|---|---|---|--:|--:|--:|--:|--:|
| qwen2.5-1.5b-q4_k_m | cpu | 3/3 ok | CPU | 37.9 ± 0.7 | 25.3 ± 1.3 | 686 ± 12 | 6190.6 ± 3889.1 | 214.7 ± 59.8 |
| qwen2.5-1.5b-q4_k_m | opencl | 3/3 ok | CPU,GPUOpenCL | 193.3 ± 7.7 | 30.4 ± 0.8 | 135 ± 6 | 107.6 ± 0.0 | 8.0 ± 0.0 |
| llama-3.2-1b-q4_k_m | cpu | 3/3 ok | CPU | 53.9 ± 0.6 | 30.1 ± 0.3 | 817 ± 10 | 6442.2 ± 1384.1 | 154.2 ± 34.2 |
| llama-3.2-1b-q4_k_m | opencl | 3/3 ok | CPU,GPUOpenCL | 458.1 ± 31.4 | 41.5 ± 0.1 | 96 ± 6 | 1694.2 ± 639.9 | 85.6 ± 38.0 |
| llama-3.2-3b-q4_k_m | cpu | 3/3 ok | CPU | 18.6 ± 0.2 | 13.1 ± 0.2 | 2366 ± 30 | 9811.0 ± 1756.0 | 127.0 ± 47.4 |
| llama-3.2-3b-q4_k_m | opencl | 3/3 ok | CPU,GPUOpenCL | 217.3 ± 7.1 | 20.8 ± 0.0 | 203 ± 7 | 8245.0 ± 533.2 | 148.2 ± 7.7 |
| qwen2.5-7b-q4_k_m | cpu | 3/3 ok | CPU | 8.2 ± 0.0 | 6.5 ± 0.1 | 3172 ± 18 | 16926.8 ± 1250.4 | 127.2 ± 68.4 |
| qwen2.5-7b-q4_k_m | opencl | 3/3 ok | CPU,GPUOpenCL | 56.0 ± 16.6 | 8.0 ± 2.9 | 488 ± 123 | 8646.3 ± 14344.3 | 85.6 ± 137.7 |

\* Energy/current come from the app's BatteryManager sampler. The phone was USB-powered during the run, so battery current follows the charger's cycle rather than the model's load: these columns are recorded for completeness but are NOT valid inference energy measurements.

Prefill is measured on the chat-templated prompt (a few dozen tokens), so it is dominated by fixed overhead and is not comparable to the paper's 256-token prefill.
