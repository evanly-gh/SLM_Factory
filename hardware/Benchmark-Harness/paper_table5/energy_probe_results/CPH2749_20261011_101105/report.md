# Energy-method probe: OnePlus CPH2749 (SM8850)

- Run: 2026-10-11T10:11:16.810716 -> 2026-10-11T10:25:00.635828; 12 windows of 45 s (first 5.0 s of each dropped); screen off
- Root: False; powercap usable: False; Power Stats HAL: android.hardware.power.stats.IPowerStats/default
- Battery temperature across windows: 21.6 - 23.0 C
- Largest gap between phone-side samples (s), per workload: {'idle': 0.14, 'cpu1': 0.14, 'cpu_all': 0.24}
- Mean CPU busy fraction per workload (from /proc/stat): {'idle': 0.047, 'cpu1': 0.16, 'cpu_all': 0.9}

Power in mW, mean +/- std over repeats. Net = workload minus idle. Load CV = worst coefficient of variation of the two load workloads (lower = more repeatable). Agreement = net all-core power / median of the battery methods.

| Method | Idle | 1 core | All cores | Net 1 core | Net all cores | Load CV | Updates/s | Agreement | Notes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| perfetto:charge | 200 +/- 400 | 1194 +/- 1379 | 4749 +/- 2484 | 994 +/- 1379 | 4549 +/- 2484 | 115.5% | 0.038 | 1.38x | battery HAL charge counter |
| perfetto:current **(recommended)** | 96 +/- 7 | 606 +/- 523 | 2136 +/- 1214 | 510 +/- 523 | 2040 +/- 1214 | 86.2% | 0.1 | 0.62x | battery HAL current, unit mA |

**Recommended: perfetto:current**

- most repeatable battery method: load CV 86.2%, updates 0.1 /s, net all-core power 2040 mW (0.62x the median of battery methods)
- battery methods measure the whole phone (not SoC-only like the paper): use idle-subtracted values and expect them to read higher than the paper's uJ/token
