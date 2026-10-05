# Energy-method probe: samsung SM-S911U (SM8550)

- Run: 2026-10-04T23:10:40.840346 -> 2026-10-04T23:24:38.642514; 12 windows of 45 s (first 5.0 s of each dropped); screen off
- Root: False; powercap usable: False; Power Stats HAL: none
- Battery temperature across windows: 31.2 - 35.3 C
- Largest gap between phone-side samples (s), per workload: {'idle': 2.61, 'cpu1': 2.72, 'cpu_all': 2.71}
- Mean CPU busy fraction per workload (from /proc/stat): {'idle': 0.593, 'cpu1': 0.585, 'cpu_all': 0.636}

Power in mW, mean +/- std over repeats. Net = workload minus idle. Load CV = worst coefficient of variation of the two load workloads (lower = more repeatable). Agreement = net all-core power / median of the battery methods.

| Method | Idle | 1 core | All cores | Net 1 core | Net all cores | Load CV | Updates/s | Agreement | Notes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| perfetto:charge | 0 +/- 0 | 0 +/- 0 | 10288 +/- 20575 | 0 +/- 0 | 10288 +/- 20575 | 200.0% | 0.0 | 1.83x | battery HAL charge counter |
| perfetto:current | 5458 +/- 520 | 5905 +/- 382 | 6401 +/- 787 | 447 +/- 382 | 943 +/- 787 | 12.3% | 1.18 | 0.17x | battery HAL current, unit mA |

**Recommended: none**

- the all-core load did not run as intended (CPU busy per workload: {'idle': 0.593, 'cpu1': 0.585, 'cpu_all': 0.636}); the energy readings cannot be attributed to the workloads, so no method is recommended
- most repeatable battery method: load CV 12.3%, updates 1.18 /s, net all-core power 943 mW (0.17x the median of battery methods)
- battery methods measure the whole phone (not SoC-only like the paper): use idle-subtracted values and expect them to read higher than the paper's uJ/token
- the phone slept during idle windows (largest sample gap 2.6 s), so the idle baseline is a suspended-phone baseline, lower than the awake idle an inference run sees; rerun with --screen on to compare
