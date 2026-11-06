# Energy-method probe: samsung SM-S911U (SM8550)

- Run: 2026-10-04T23:51:48.188093 -> 2026-10-05T00:05:23.352848; 12 windows of 45 s (first 5.0 s of each dropped); screen on
- Root: False; powercap usable: False; Power Stats HAL: none
- Battery temperature across windows: 24.5 - 34.1 C
- Largest gap between phone-side samples (s), per workload: {'idle': 0.14, 'cpu1': 0.13, 'cpu_all': 0.14}
- Mean CPU busy fraction per workload (from /proc/stat): {'idle': 0.06, 'cpu1': 0.168, 'cpu_all': 0.997}

Power in mW, mean +/- std over repeats. Net = workload minus idle. Load CV = worst coefficient of variation of the two load workloads (lower = more repeatable). Agreement = net all-core power / median of the battery methods.

| Method | Idle | 1 core | All cores | Net 1 core | Net all cores | Load CV | Updates/s | Agreement | Notes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| perfetto:charge | -5068 +/- 3095 | -1548 +/- 1556 | -3905 +/- 3130 | 3520 +/- 1556 | 1163 +/- 3130 | -80.1% | 0.025 | 0.28x | battery HAL charge counter |
| perfetto:current **(recommended)** | 582 +/- 4 | 4549 +/- 103 | 7685 +/- 355 | 3968 +/- 103 | 7103 +/- 355 | 4.6% | 5.652 | 1.72x | battery HAL current, unit mA |

**Recommended: perfetto:current**

- most repeatable battery method: load CV 4.6%, updates 5.652 /s, net all-core power 7103 mW (1.72x the median of battery methods)
- battery methods measure the whole phone (not SoC-only like the paper): use idle-subtracted values and expect them to read higher than the paper's uJ/token
