# Hardware Metrics for On-Device LLM Evaluation

This document records the research-backed rationale for the 7 hardware metrics used in the meta-learning loop's hardware evaluation step. These metrics gate model selection and promotion in `android_pool.filter_pool` and `check_hardware_constraints`.

## The Three User Questions

The metric set is organized around three distinct user concerns:

| Question | Metrics |
|---|---|
| **Does it fit on my phone?** | Storage, Memory / RAM |
| **Is it fast enough to be useful?** | Cold Start, TTFT (primary), Throughput (floor check) |
| **Will it drain my battery / overheat?** | Power, Peak Thermal |

## Metric Set

| Metric | Unit | Threshold | How profiled | Profiling cost |
|---|---|---|---|---|
| Storage | MB | ≤ 5% of total device storage | Static INT4 file size | Instant |
| Memory / RAM | MB | ≤ total RAM − 2.5 GB | Peak RSS via `/proc/<pid>/status` during inference | ~10s |
| Cold Start | ms | ≤ 5,000 ms (target ≤ 3,000 ms) | Wall time from process launch to model ready | ~5–40s (one-off) |
| TTFT | ms | ≤ 2,000 ms (target ≤ 1,000 ms) | Wall time to first generated token | ~5s |
| Throughput | tok/s | ≥ 6 tok/s floor (matches human reading speed) | `num_tokens / elapsed_time` over a sustained 60s run | ~60s |
| Power | W | ≤ 5 W sustained | Smolchat power profiling tool | Shared with throughput run |
| Peak Thermal | Status | Below SEVERE during 60s run | `dumpsys thermalservice` thermal headroom level (NONE/LIGHT/MODERATE/SEVERE/CRITICAL) | Shared with throughput run |

Total hardware eval time: ~80 seconds (cold start measured once per model, other metrics measured together).

---

## Why Each Metric — Scaling with Model Size

As a model gets bigger: it uses more storage and RAM, takes longer to load and generate (sublinearly for TTFT, inversely for throughput), and consumes more power leading to higher temperatures.

### Storage (MB)
**Scales linearly.** INT4 bytes are directly proportional to parameter count. Larger models store more weights. This is the first filter: a model that doesn't fit on the device's storage budget is eliminated before any inference runs.

**Threshold: ≤ 5% of total device storage.**
Reasoning: on a 64GB phone, OS + apps typically consume 20–30GB, leaving ~34–44GB free. Limiting the model to 5% of total (~3.2GB) keeps it from crowding user data and is comfortably above the pool's largest models (Phi-4-mini Q4_K_M at ~2490MB; Phi-4-mini Q8_0 at ~4731MB). No citation needed — pure arithmetic from storage spec.

**Derivable from specs alone? Yes.** Total storage is a static hardware field.

---

### Memory / RAM (MB)
**Scales linearly.** Peak RSS includes model weights (loaded fully into DRAM) plus KV-cache, activation buffers, and runtime overhead. Weights dominate for the 0.5B–3B range. A model that fits in storage can still OOM at runtime — this is the second filter.

**Threshold: ≤ total RAM − 2.5 GB.**
Reasoning: Android OS + background system services consume approximately 2–3GB on a typical mid-range device. On a heavily skinned 8GB Samsung device, users have observed ~3GB consumed by the OS alone; on stock Android the overhead is closer to 2GB. A 2.5GB constant reservation is a reasonable median. Anything under 6GB total RAM should be considered high-risk for 3B-class models. Sources: [Android Authority RAM guide](https://www.androidauthority.com/how-much-ram-do-i-need-phone-3086661/), [Samsung OneUI memory overhead reports](https://r1.community.samsung.com/t5/galaxy-s/memory-usage-increase-with-android-16-oneui-8/td-p/35112088).

**Derivable from specs alone? Yes.** Total RAM is a static hardware field; the 2.5GB constant is a conservative empirical estimate.

---

### Cold Start (ms)
**Scales linearly.** Cold start time is dominated by reading model weights from UFS flash storage into DRAM. Larger models have more bytes to load, so load time scales roughly with file size. On a phone with UFS 2.1 (~1,000 MB/s sequential read), a 1.4GB model takes ~1.4s to load; on UFS 3.1 (~2,000 MB/s) the same model takes ~0.7s. Budget phones with slower storage or Kirin-family SoCs can see cold starts exceeding 40 seconds for larger models.

**Threshold: ≤ 5,000 ms hard gate, ≤ 3,000 ms target.**
Reasoning: Research identifies 9–11 second cold starts as "far too slow for any real application." The goal for production-quality mobile apps is ≤ 3 seconds, matching conventional mobile app norms. The 5-second hard gate gives budget/older devices some headroom while excluding truly unusable outliers. Sources: [EdgeFlow: Fast Cold Starts for LLMs on Mobile Devices](https://arxiv.org/html/2604.09083), [Large Language Model Performance Benchmarking on Mobile Platforms](https://arxiv.org/html/2410.03613v1).

**Derivable from specs alone? Partially.** Estimable as `model_file_mb / ufs_read_speed_mbs`, but UFS generation varies by OEM and isn't always published. Measure directly.

---

### TTFT (ms)
**Scales sub-linearly.** TTFT is prefill-dominated (compute-bound phase). NPUs and CPUs process the prompt in parallel, so larger models are slower but not proportionally so — hardware parallelism partially absorbs the cost.

**Threshold: ≤ 2,000 ms hard gate, ≤ 1,000 ms target.**
Reasoning: Classic HCI research places the boundary of "keeps the user's flow of thought intact" at 1 second; above 1.5–2 seconds users begin to disengage. A 2025 controlled experiment explicitly tested 2s, 9s, and 20s delays — the 2s condition was the minimum "tolerable" threshold before subjective experience degraded significantly. The 1,000ms target keeps the experience genuinely interactive. Sources: [The Impact of Response Latency and Task Type on Human-LLM Interaction](https://arxiv.org/html/2604.06183v1), [Time to First Token — Redis](https://redis.io/blog/ttft-meaning/).

**Derivable from specs alone? Weakly.** Estimable from NPU TOPS × prompt length, but runtime and inference-engine choice add ±50% variance. Measure directly.

---

### Throughput (tok/s)
**Scales inversely.** Decode is the autoregressive token-generation phase. For each token, the model must stream all its weights through the DRAM bus. More weights = more data movement per token = lower tok/s. For 1B–3B INT4 models specifically, weights can partially fit in CPU L3 cache, so the bottleneck is a mix of compute and bandwidth — making direct measurement more reliable than spec-derived estimates.

**Role: floor check, not a primary UX gate.** TTFT is the metric users actually feel — it determines whether the response *starts* fast enough. Throughput only matters when it falls below human reading speed, at which point the output visibly lags behind the user's eyes. Above that floor, users cannot perceptually distinguish 10 tok/s from 40 tok/s when streaming. The primary UX gate is TTFT.

**Threshold: ≥ 6 tok/s floor.**
Reasoning: Average English reading speed is ~250 words per minute, which corresponds to approximately 6 tokens/second. Below this floor, streaming output visibly lags behind reading pace — the user is waiting for words to appear mid-sentence. Above it, the bottleneck shifts entirely to TTFT (how long before anything appears), which is gated separately. The previous thresholds of 20–30 tok/s were derived from server-side benchmarking literature that does not account for streaming UX — those numbers assume users watch a blank screen until the full response arrives. Sources: [Streaming, Fast and Slow: Cognitive Load-Aware Streaming — ACM UIST 2025](https://dl.acm.org/doi/10.1145/3746059.3747721), [Key metrics for LLM inference — BentoML](https://bentoml.com/llm/llm-inference-basics/llm-inference-metrics).

**Derivable from specs alone? Weakly, and unreliably for 1–3B range.** Measure directly.

---

### Power (W)
**Scales super-linearly under sustained load.** Larger models push the SoC compute units harder for longer, generating more heat. As junction temperature rises, DVFS reduces clock frequency, which paradoxically reduces throughput while the model continues drawing power — the worst outcome. Profiled via Smolchat during the same 60s throughput run.

**Threshold: ≤ 5 W sustained.**
Reasoning: A phone in active screen-on use draws ~1.5–2W baseline. Adding 4–5W of inference load stays within the ~6–8W thermal envelope most mid-range phones can sustain for several minutes before DVFS onset. Beyond ~5W sustained, the device burns ~1% battery per minute and is likely approaching the thermal throttle boundary simultaneously. This threshold is a practical engineering estimate; exact TDP limits vary by OEM and are not published for inference workloads specifically.

**Derivable from specs alone? No.** Chip TDP figures reflect full-SoC peak load, not single-model inference. Must measure via Smolchat.

---

### Peak Thermal (Status)
**Scales super-linearly.** Larger models sustain higher SoC load, driving junction temperature toward DVFS thresholds faster (~40–43°C skin temperature onset on most Android OEMs). Read via `dumpsys thermalservice` thermal headroom status codes rather than raw thermal zone temps — this is OEM-agnostic and maps directly to Android's DVFS trigger levels.

**Threshold: Below SEVERE during 60s run.**
Reasoning: NONE and LIGHT indicate the device is operating comfortably within its thermal envelope. MODERATE signals that DVFS is beginning to activate — throughput is degrading but the device is not at risk. SEVERE means DVFS has kicked in hard and measured throughput is materially below the actual sustained floor. CRITICAL means inference should stop. Raw thermal zone temperatures are junction temps (5–10°C hotter than skin) and vary by sensor placement across OEMs — `dumpsys thermalservice` abstracts these correctly into a device-agnostic status level. Sources: [Thermal mitigation — Android Open Source Project](https://source.android.com/docs/core/power/thermal-mitigation), [User-specific Skin Temperature-Aware DVFS — IEEE](https://ieeexplore.ieee.org/document/7092573/).

**Derivable from specs alone? No.** Thermal behavior depends on device casing, vapor chamber presence, and ambient conditions — two phones with the same SoC can have completely different profiles. Must measure.

---

## What Was Explicitly Excluded and Why

| Candidate metric | Excluded because |
|---|---|
| Prefill throughput (tok/s on prompt) | Compute-bound and NPU-dominated; TTFT already captures the user-visible effect |
| KV cache context limit | A static architectural cap set by the runtime, not a per-run measurable; encode as a static field if needed |
| Battery % delta | Redundant with power watts over a fixed-duration run; noisier and slower to converge |
| NPU TOPS | Marketing figures are measured under idealized batch workloads at compute-bound utilization; decode is memory-bandwidth-bound, so TOPS is near-zero during the phase that matters |
| LPDDR generation | Useful for hardware database entries, not a profilable runtime metric |

---

## Key Research Findings

- **Thermal throttling is the primary sustained-load constraint on mobile** — devices can lose 40–50% throughput within 2–3 inference iterations as DVFS kicks in. Sustained tok/s (not peak) is the operative metric.
- **Decode phase is memory-bandwidth-bound for large models but partially compute-bound for 1B–3B INT4 models** — weights can fit in L3 cache, making direct measurement more reliable than spec-derived estimates.
- **ADB `current_now` power reading is unreliable on USB-connected devices** — the USB connection itself injects charge current, skewing readings. Smolchat avoids this limitation.
- **Android DVFS onset is ~40–43°C skin temperature** — but thermal zone readings are junction temperature (5–10°C hotter than skin). `dumpsys thermalservice` abstracts this correctly.
- **CPU inference often beats GPU on Android** for 1B–3B INT4 models due to low ALU utilization on Mali/Adreno GPUs. NPU excels at prefill (compute-bound) but not decode.

---

## Does the acceptable threshold for these metrics change depending on the device you're using?

TLDR: No

### "Does it fit?" metrics — already device-relative, and must be
Storage threshold is already a formula (5% × total_storage), so it auto-adjusts. RAM threshold is already total_ram − 2.5GB. These two correctly scale with the device. Nothing to change here.

### "Is it fast enough?" metrics — thresholds are fixed, achievements vary
Cold start, TTFT, and throughput thresholds are user experience constants, not device capabilities. 2,000ms TTFT feels sluggish on a Pixel 9 Pro the same way it does on a Moto G Stylus — the user's perception doesn't change based on what phone they have.
What changes is what the device can achieve. A Snapdragon 660 might only push 8 tok/s on the smallest model, which passes the 6 tok/s floor. The correct response to a device that can't clear the floor — for example, only reaching 4 tok/s on 3B models — isn't lowering the bar: it's that no model in the pool is interactively viable on that device, and the agent should say so clearly rather than pretend streaming at below reading speed is acceptable.
So: fixed thresholds, device-varying measurements. The threshold never moves. The device determines which models can clear it.

### "Battery/heat" metrics — your intuition about efficiency is correct, but the threshold stays fixed
Yes, a Snapdragon 8 Gen 3 (4nm) running the same model at the same tok/s will draw fewer watts than a Snapdragon 778G (6nm) because of better process node efficiency. And yes, an iPhone 16 is dramatically more power-efficient per token than a Moto G Stylus.
But the 5W threshold doesn't need to change per device — because watts is watts for the thermal envelope. The SoC's ability to sustain 5W before DVFS triggers is approximately the same across devices (it's constrained by passive cooling physics, not by how efficiently the chip generates those watts). What actually varies is: at 5W, a more efficient chip generates more tok/s. That's captured automatically by measuring both power and throughput separately — the Snapdragon 8 Gen 3 passes both gates simultaneously; the Snapdragon 660 might pass the power gate but fail throughput.
Peak thermal is similar — SEVERE is SEVERE regardless of device. The status abstraction already normalizes across OEMs.

So: no, the thresholds don't need to be adjusted per cold-start.
The full picture:

Metric	Threshold type	Adjusts per device?
Storage	Device-relative formula	✅ Already does
RAM	Device-relative formula	✅ Already does
Cold Start	UX constant	❌ Fixed
TTFT	UX constant	❌ Fixed
Throughput	UX constant	❌ Fixed — weak devices just fail it
Power	Thermal envelope constant	❌ Fixed — efficiency shows up in throughput
Peak Thermal	Status abstraction	❌ Fixed — status normalizes across OEMs
The only thing that adjusts per device is which models can meet the thresholds — and that's exactly what the hardware DB lookup + filter is for. The loop never lowers the bar; it picks a smaller model or tells the user their device can't run any model acceptably.

---
## Sources

- [LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load](https://arxiv.org/html/2603.23640v1)
- [EnerInfer: Energy-Aware On-Device LLM Inference](https://arxiv.org/html/2606.23001)
- [MobileAIBench: Benchmarking LLMs and LMMs for On-Device Use Cases](https://arxiv.org/html/2406.10290v1)
- [lm-Meter: Unveiling Runtime Inference Latency for On-Device Language Models](https://www.amai-gsu.us/wp-content/uploads/2025/lm-meter.pdf)
- [Measuring Energy use of Android Devices — Scott Logic](https://blog.scottlogic.com/2024/05/01/measuring-android-energy-use.html)
- [Measure device power — Android Open Source Project](https://source.android.com/docs/core/power/device)
- [Thermal mitigation — Android Open Source Project](https://source.android.com/docs/core/power/thermal-mitigation)
- [Fast On-device LLM Inference with NPUs](https://arxiv.org/html/2407.05858v2)
- [Understanding Large Language Models in Your Pockets](https://arxiv.org/html/2410.03613v3)
- [EdgeFlow: Fast Cold Starts for LLMs on Mobile Devices](https://arxiv.org/html/2604.09083)
- [Large Language Model Performance Benchmarking on Mobile Platforms](https://arxiv.org/html/2410.03613v1)
- [The Impact of Response Latency and Task Type on Human-LLM Interaction](https://arxiv.org/html/2604.06183v1)
- [Time to First Token — Redis](https://redis.io/blog/ttft-meaning/)
- [How Many Tokens Per Second Is 'Good' for Local LLMs — ML Journey](https://mljourney.com/how-many-tokens-per-second-is-good-for-local-llms/)
- [A Systematic Evaluation of On-Device LLMs: Quantization, Performance, and Resources](https://arxiv.org/html/2505.15030v5)
- [User-specific Skin Temperature-Aware DVFS — IEEE](https://ieeexplore.ieee.org/document/7092573/)
- [Android Authority: How much RAM do you need in a phone?](https://www.androidauthority.com/how-much-ram-do-i-need-phone-3086661/)
- [Streaming, Fast and Slow: Cognitive Load-Aware Streaming — ACM UIST 2025](https://dl.acm.org/doi/10.1145/3746059.3747721)
- [TTFT vs Tokens Per Second — GMI Cloud](https://www.gmicloud.ai/en/blog/ttft-llm-speed-metrics)
- [Beyond Tokens-per-Second — BentoML](https://www.bentoml.com/blog/beyond-tokens-per-second-how-to-balance-speed-cost-and-quality-in-llm-inference)
