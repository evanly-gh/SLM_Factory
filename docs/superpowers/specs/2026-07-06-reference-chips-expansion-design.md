# Design: Expand REFERENCE_CHIPS to Match CHIP_SCALE_FACTORS

**Date:** 2026-07-06  
**Status:** Approved

## Problem

`REFERENCE_CHIPS` in `hardware_research.py` is a hardcoded list of 3 Snapdragon anchors. The LLM must pick the closest match from this list when resolving a device's SoC. With only 3 options, non-Snapdragon devices (Pixel, Samsung, MediaTek) get forced onto the wrong anchor, and mid-range Snapdragon variants (730, 750G, 870, 888) all round to 778G regardless of their actual tier.

`CHIP_SCALE_FACTORS` in `android_pool.py` already defines 14 chips with calibrated throughput multipliers covering all major Android SoC families. `tok_s_for_chip()` uses these to interpolate throughput for any chip — the infrastructure already handles all 14 correctly.

## Solution

Derive `REFERENCE_CHIPS` from `CHIP_SCALE_FACTORS` so the two are always in sync. Adding a chip to the scale table automatically expands the LLM's vocabulary with no further changes.

## Changes

**File:** `agent/nodes/cold_start/hardware_research.py`

1. Import `CHIP_SCALE_FACTORS` alongside the existing `HardwareConstraints` import:
   ```python
   from config.android_pool import CHIP_SCALE_FACTORS, HardwareConstraints
   ```

2. Replace the hardcoded list:
   ```python
   # Before
   REFERENCE_CHIPS = ["snapdragon_660", "snapdragon_778g", "snapdragon_8gen3"]
   # After
   REFERENCE_CHIPS = list(CHIP_SCALE_FACTORS.keys())
   ```

3. Update the validation guard to use the authoritative dict:
   ```python
   # Before
   if ref not in REFERENCE_CHIPS:
   # After
   if ref not in CHIP_SCALE_FACTORS:
   ```

## Scope

No other files change. `tok_s_for_chip()` already handles all 14 chips; model selection and throughput estimation work correctly for all new anchors immediately.

## Result

The LLM prompt expands from 3 options to 14, covering:
- Snapdragon 660 → 730 → 750G → 778G → 870 → 888 → 8 Gen 1 → 8 Gen 2 → 8 Gen 3 → 8 Elite
- Dimensity 9300, 9400
- Exynos 2400, 2500
- Tensor G3, G4
