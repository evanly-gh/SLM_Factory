# Energy-method probe: samsung SM-S911U (SM8550)

- Run: 2026-10-04T18:41:07.340842 -> 2026-10-04T18:54:52.623006; 12 windows of 45 s (first 5.0 s of each dropped); screen off
- Root: False; powercap usable: False; Power Stats HAL: none
- Battery temperature across windows: 31.0 - 32.4 C
- Largest gap between phone-side samples (s), per workload: {'idle': 2.72, 'cpu1': 2.52, 'cpu_all': 2.64}

Power in mW, mean +/- std over repeats. Net = workload minus idle. Load CV = worst coefficient of variation of the two load workloads (lower = more repeatable). Agreement = net all-core power / median of the battery methods.

| Method | Idle | 1 core | All cores | Net 1 core | Net all cores | Load CV | Updates/s | Agreement | Notes |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---|
| perfetto:charge | 0 +/- 0 | -6833 +/- 13665 | 0 +/- 0 | -6833 +/- 13665 | 0 +/- 0 | -200.0% | 0.0 | 0.00x | battery HAL charge counter |
| perfetto:current **(recommended)** | 4185 +/- 2079 | 3684 +/- 1816 | 4849 +/- 934 | -501 +/- 1816 | 665 +/- 934 | 49.3% | 1.049 | 1.00x | battery HAL current, unit mA |

**Recommended: perfetto:current**

- most repeatable battery method: load CV 49.3%, updates 1.049 /s, net all-core power 665 mW (1.00x the median of battery methods)
- battery methods measure the whole phone (not SoC-only like the paper): use idle-subtracted values and expect them to read higher than the paper's uJ/token
- the phone slept during idle windows (largest sample gap 2.7 s), so the idle baseline is a suspended-phone baseline, lower than the awake idle an inference run sees; rerun with --screen on to compare
