# Energy-method probe: OnePlus IN2025 (SM8250)

- Run: 2026-10-20T02:11:47.191249 -> 2026-10-20T02:18:22.467715; 6 windows of 40 s (first 5.0 s of each dropped); screen panel
- Root: False; powercap usable: False; Power Stats HAL: none
- Battery temperature across windows: 29.1 - 30.7 C
- Largest gap between phone-side samples (s), per workload: {'idle': 0.14, 'cpu1': 0.13, 'cpu_all': 0.13}
- Mean CPU busy fraction per workload (from /proc/stat): {'idle': 0.064, 'cpu1': 0.177, 'cpu_all': 0.995}

Power in mW, mean +/- std over repeats. Net = workload minus idle. Load CV = worst coefficient of variation of the two load workloads (lower = more repeatable). Agreement = net all-core power / median of the battery methods.

| Method | Idle | 1 core | All cores | Net 1 core | Net all cores | Load CV | Updates/s | Agreement | Notes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| bm:charge_counter | 431 +/- 1 | 2117 +/- 12 | 6469 +/- 382 | 1686 +/- 12 | 6038 +/- 382 | 5.9% | 0.143 | 0.99x | battery service charge counter (uAh) |
| bm:current_now **(recommended)** | 389 +/- 34 | 2275 +/- 41 | 6563 +/- 34 | 1886 +/- 41 | 6174 +/- 34 | 1.8% | 0.973 | 1.01x | battery service (BatteryManager) current, unit mA |

**Recommended: bm:current_now**

- most repeatable battery method: load CV 1.8%, updates 0.973 /s, net all-core power 6174 mW (1.01x the median of battery methods)
- runner-up bm:charge_counter: load CV 5.9%, agreement 0.99x
- battery methods measure the whole phone (not SoC-only like the paper): use idle-subtracted values and expect them to read higher than the paper's uJ/token
