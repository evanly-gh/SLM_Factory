# Energy-method probe: OnePlus IN2025 (SM8250)

- Run: 2026-10-20T02:03:01.611371 -> 2026-10-20T02:09:39.121950; 6 windows of 40 s (first 5.0 s of each dropped); screen panel
- Root: False; powercap usable: False; Power Stats HAL: none
- Battery temperature across windows: 29.5 - 30.4 C
- Largest gap between phone-side samples (s), per workload: {'idle': 1.28, 'cpu1': 2.1, 'cpu_all': 1.53}
- Mean CPU busy fraction per workload (from /proc/stat): {'cpu1': 0.718, 'cpu_all': 0.899, 'idle': 0.656}

Power in mW, mean +/- std over repeats. Net = workload minus idle. Load CV = worst coefficient of variation of the two load workloads (lower = more repeatable). Agreement = net all-core power / median of the battery methods.

| Method | Idle | 1 core | All cores | Net 1 core | Net all cores | Load CV | Updates/s | Agreement | Notes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| bm:charge_counter | -441 +/- 14 | -219 +/- 310 | -436 +/- 11 | 222 +/- 310 | 5 +/- 11 | -2.5% | 0.029 | 0.74x | battery service charge counter (uAh) |
| bm:current_now | -533 +/- 332 | -390 +/- 58 | -525 +/- 56 | 143 +/- 58 | 8 +/- 56 | -10.7% | 0.845 | 1.26x | battery service (BatteryManager) current, unit mA |

**Recommended: none**

- the all-core load did not run as intended (CPU busy per workload: {'cpu1': 0.718, 'cpu_all': 0.899, 'idle': 0.656}); the energy readings cannot be attributed to the workloads, so no method is recommended
- most repeatable battery method: load CV -10.7%, updates 0.845 /s, net all-core power 8 mW (1.26x the median of battery methods)
- battery methods measure the whole phone (not SoC-only like the paper): use idle-subtracted values and expect them to read higher than the paper's uJ/token
