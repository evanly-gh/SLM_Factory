# Energy-method probe: OnePlus IN2025 (SM8250)

- Run: 2026-11-03T10:27:12.887619 -> 2026-11-03T10:40:55.415937; 12 windows of 45 s (first 5.0 s of each dropped); screen off
- Root: False; powercap usable: False; Power Stats HAL: none
- Battery temperature across windows: 24.4 - 24.9 C
- Largest gap between phone-side samples (s), per workload: {'idle': 0.14, 'cpu1': 0.14, 'cpu_all': 0.44}
- Mean CPU busy fraction per workload (from /proc/stat): {'idle': 0.061, 'cpu1': 0.183, 'cpu_all': 0.746}

Power in mW, mean +/- std over repeats. Net = workload minus idle. Load CV = worst coefficient of variation of the two load workloads (lower = more repeatable). Agreement = net all-core power / median of the battery methods.

| Method | Idle | 1 core | All cores | Net 1 core | Net all cores | Load CV | Updates/s | Agreement | Notes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| bm:charge_counter | 456 +/- 180 | 1187 +/- 186 | 1896 +/- 462 | 731 +/- 186 | 1440 +/- 462 | 24.4% | 0.075 | 1.04x | battery service charge counter (uAh) |
| bm:current_now **(recommended)** | 376 +/- 89 | 935 +/- 122 | 1713 +/- 124 | 558 +/- 122 | 1336 +/- 124 | 13.0% | 0.959 | 0.96x | battery service (BatteryManager) current, unit mA |

**Recommended: bm:current_now**

- most repeatable battery method: load CV 13.0%, updates 0.959 /s, net all-core power 1336 mW (0.96x the median of battery methods)
- runner-up bm:charge_counter: load CV 24.4%, agreement 1.04x
- battery methods measure the whole phone (not SoC-only like the paper): use idle-subtracted values and expect them to read higher than the paper's uJ/token
