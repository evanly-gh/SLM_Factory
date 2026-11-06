# Energy-method probe: samsung SM-S911U (SM8550)

- Run: 2026-10-04T23:33:44.943205 -> 2026-10-04T23:47:33.172554; 12 windows of 45 s (first 5.0 s of each dropped); screen off
- Root: False; powercap usable: False; Power Stats HAL: none
- Battery temperature across windows: 24.9 - 27.0 C
- Largest gap between phone-side samples (s), per workload: {'idle': 2.36, 'cpu1': 2.36, 'cpu_all': 2.37}
- Mean CPU busy fraction per workload (from /proc/stat): {'idle': 0.122, 'cpu1': 0.162, 'cpu_all': 0.421}

Power in mW, mean +/- std over repeats. Net = workload minus idle. Load CV = worst coefficient of variation of the two load workloads (lower = more repeatable). Agreement = net all-core power / median of the battery methods.

| Method | Idle | 1 core | All cores | Net 1 core | Net all cores | Load CV | Updates/s | Agreement | Notes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| perfetto:charge | -9094 +/- 12280 | 0 +/- 0 | 0 +/- 0 | 9094 +/- 0 | 9094 +/- 0 | n/a | 0.0 | 1.82x | battery HAL charge counter |
| perfetto:current | 684 +/- 322 | 941 +/- 408 | 1601 +/- 1312 | 257 +/- 408 | 916 +/- 1312 | 82.0% | 0.928 | 0.18x | battery HAL current, unit mA |

**Recommended: none**

- the all-core load did not run as intended (CPU busy per workload: {'idle': 0.122, 'cpu1': 0.162, 'cpu_all': 0.421}); the energy readings cannot be attributed to the workloads, so no method is recommended
- most repeatable battery method: load CV 82.0%, updates 0.928 /s, net all-core power 916 mW (0.18x the median of battery methods)
- battery methods measure the whole phone (not SoC-only like the paper): use idle-subtracted values and expect them to read higher than the paper's uJ/token
- the phone slept during idle windows (largest sample gap 2.4 s), so the idle baseline is a suspended-phone baseline, lower than the awake idle an inference run sees; rerun with --screen on to compare
