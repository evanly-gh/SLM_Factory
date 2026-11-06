/*
 * Copyright (C) 2024 Shubham Panchal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.shubham0204.smollmandroid.headless

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.util.Log
import java.io.File
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import kotlin.math.abs

/**
 * One instantaneous current (and, when available, voltage) reading paired with the real
 * wall-clock time it was taken, rather than an index into a nominally-fixed-interval list —
 * scheduleAtFixedRate() targets [PowerSampler.start]'s intervalMs but genuine drift (GC pauses,
 * scheduler contention under the exact CPU/GPU-heavy load this class exists to measure) means
 * consecutive samples are not reliably exactly 100ms apart. Recording the real timestamp per
 * sample lets energy integration use the actual elapsed time between each pair of samples
 * instead of assuming perfect spacing.
 *
 * [voltageMv] is nullable independent of [currentUa] — see [PowerSampler.readVoltageMv]'s kdoc
 * for why a voltage reading can be stale/unavailable on a tick where the current reading
 * succeeds fine.
 */
data class PowerSample(val timestampMs: Long, val currentUa: Long, val voltageMv: Long?)

/**
 * Samples battery current (and voltage) during inference and reports the average current in mA,
 * plus real energy figures via trapezoidal integration.
 *
 * A near-verbatim port of MNN Chat's own PowerSampler
 * (apps/Android/MnnLlmChat/app/src/main/java/com/alibaba/mnnllm/android/benchmark/headless/PowerSampler.kt),
 * which was itself originally ported the other direction — from this project's
 * BenchmarkService.kt's readCurrentUa()/computeAvgCurrentUa() — so this brings SmolChat back in
 * sync with the more complete version that grew from it, rather than reconstructing energy
 * integration from scratch.
 *
 * Three-tier current fallback, unchanged from BenchmarkService.kt's original:
 *   1. [BatteryManager.BATTERY_PROPERTY_CURRENT_NOW] (microamps) via getLongProperty() — the
 *      normal path, works on most devices.
 *   2. If that returns the "unsupported" sentinel (Long.MIN_VALUE), read raw current directly
 *      from common sysfs paths ([CURRENT_SYSFS_PATHS]).
 *   3. If the WHOLE sampling window produced zero valid instantaneous readings from either of
 *      the above, fall back to a charge-counter delta: total charge drained
 *      ([BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER]) over the elapsed duration, converted
 *      to an equivalent average current.
 *
 * Also exposes two independent energy figures alongside the average: [getEnergyMasSampled]
 * (mA·s, trapezoidal integration over real per-sample current-only timestamps — a
 * charge-equivalent quantity, not true energy, since it implicitly assumes constant voltage) and
 * [getEnergyMjSampled] (mJ, the same trapezoidal integration but over current x voltage at each
 * sample — true energy, accounting for real per-sample voltage rather than assuming it
 * constant). A charge-counter-delta energy method and BATTERY_PROPERTY_ENERGY_COUNTER (a direct
 * fuel-gauge energy reading) were both evaluated in the sibling project this was ported from and
 * rejected/confirmed unsupported there — see MNN's PowerSampler.kt kdoc for the real devices
 * that was confirmed on (a OnePlus 8 Pro and a Galaxy S23, the same physical devices used for
 * this project's own testing).
 */
class PowerSampler(context: Context) {

    private val appContext = context.applicationContext
    private val batteryManager =
        appContext.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val batteryChangedFilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
    private val samples = ConcurrentLinkedQueue<PowerSample>()
    private var executor: ScheduledExecutorService? = null
    private var startTimeMs: Long = 0L
    private var stopTimeMs: Long = 0L
    // Used only by getAverageMa()'s own internal fallback (computeAvgCurrentUa()) when
    // instantaneous sampling produced zero valid readings — NOT used for energy: a
    // charge-counter-based energy method was considered and removed in the sibling project this
    // was ported from, since every real benchmark run here is USB-connected for ADB, and
    // charging current contaminates a charge-delta measurement with no fallback to fall back on.
    private var chargeAtStartUah: Long = Long.MIN_VALUE
    private var loggedUnitInterpretation: Boolean = false

    fun start(intervalMs: Long = 100) {
        stop()
        startTimeMs = System.currentTimeMillis()
        stopTimeMs = 0L
        chargeAtStartUah = try {
            batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
        } catch (e: Exception) {
            Long.MIN_VALUE
        }
        val exec = Executors.newSingleThreadScheduledExecutor()
        executor = exec
        exec.scheduleAtFixedRate({
            try {
                val sample = readCurrentUa()
                if (sample > 0L) {
                    // Voltage is sampled at the SAME cadence as current, on the same tick — not
                    // a separate slower poll — so every PowerSample carries whatever voltage
                    // reading was available at that same instant, even though (see
                    // readVoltageMv()'s kdoc) the underlying value itself may not have actually
                    // changed since the last tick.
                    samples.add(PowerSample(System.currentTimeMillis(), sample, readVoltageMv()))
                }
            } catch (e: Exception) {
                Log.w(TAG, "power sample failed: ${e.message}")
            }
        }, 0, intervalMs, TimeUnit.MILLISECONDS)
    }

    fun stop() {
        executor?.shutdownNow()
        executor = null
        // Guarded so a stray extra stop() call (reset() calls stop(), and callers may also
        // stop() directly beforehand) never overwrites a real stop timestamp with a later one —
        // only the FIRST stop() after a start() records it.
        if (stopTimeMs == 0L && startTimeMs != 0L) {
            stopTimeMs = System.currentTimeMillis()
        }
    }

    fun reset() {
        stop()
        samples.clear()
        startTimeMs = 0L
        stopTimeMs = 0L
        chargeAtStartUah = Long.MIN_VALUE
        loggedUnitInterpretation = false
    }

    /**
     * Average current in mA — from instantaneous samples if any were valid, otherwise the
     * charge-counter-delta fallback over the elapsed duration since [start]. Null if neither
     * produced a usable value.
     */
    fun getAverageMa(): Double? {
        val endMs = if (stopTimeMs != 0L) stopTimeMs else System.currentTimeMillis()
        val durationSecs = ((endMs - startTimeMs) / 1000L).toInt()
        val avgUa = computeAvgCurrentUa(samples.map { it.currentUa }, chargeAtStartUah, durationSecs)
        return if (avgUa == Long.MIN_VALUE) null else avgUa / 1000.0
    }

    /**
     * CHARGE-equivalent quantity in mA·s via trapezoidal integration (Σ (i1+i2)/2 · Δt) over the
     * real per-sample timestamps recorded during [start]/[stop] — not an assumed-even-spacing
     * approximation. This is current integrated over time (i.e. charge, mA·s = mC-ish scaling),
     * NOT true energy — it implicitly assumes constant voltage throughout the window. Use
     * [getEnergyMjSampled] for true energy (accounts for real per-sample voltage instead of
     * assuming it constant). Null when fewer than 2 valid samples exist (nothing to integrate
     * between).
     */
    fun getEnergyMasSampled(): Double? {
        val ordered = samples.toList().sortedBy { it.timestampMs }
        if (ordered.size < 2) return null
        var energyUaMs = 0.0
        for (i in 1 until ordered.size) {
            val dtMs = (ordered[i].timestampMs - ordered[i - 1].timestampMs).toDouble()
            if (dtMs <= 0) continue // clock oddity guard; never expected in practice
            val avgUa = (ordered[i].currentUa + ordered[i - 1].currentUa) / 2.0
            energyUaMs += avgUa * dtMs
        }
        // uA * ms -> mA * s is /1000 (uA -> mA) then /1000 (ms -> s) = /1_000_000.
        return energyUaMs / 1_000_000.0
    }

    /**
     * TRUE energy in millijoules via trapezoidal integration over current x voltage at each
     * sample (Σ (i1·v1 + i2·v2)/2 · Δt), using the real per-sample timestamps — Energy =
     * ∫ Power dt = ∫ I·V dt, not the constant-voltage approximation [getEnergyMasSampled] makes.
     * Only pairs where BOTH samples have a voltage reading are integrated (a tick where the
     * concurrent voltage read failed is simply skipped, same real-timestamp-of-the-surviving-pair
     * logic as elsewhere in this class) — see [readVoltageMv]'s kdoc for the real accuracy
     * caveat this still carries even so. Null when fewer than 2 samples have a paired voltage
     * reading.
     *
     * Unit derivation: currentUa (µA) x voltageMv (mV) x dtMs (ms) is in µA·mV·ms =
     * 1e-6 A · 1e-3 V · 1e-3 s = 1e-12 J = 1e-9 mJ, so the raw Σ i·v·dt accumulator is scaled by
     * 1e-9 at the end (not per-term, to avoid needless floating-point precision loss across many
     * small terms).
     */
    fun getEnergyMjSampled(): Double? {
        val ordered = samples.toList()
            .filter { it.voltageMv != null }
            .sortedBy { it.timestampMs }
        if (ordered.size < 2) return null
        var energyRaw = 0.0 // accumulates in uA*mV*ms units — see kdoc above
        for (i in 1 until ordered.size) {
            val dtMs = (ordered[i].timestampMs - ordered[i - 1].timestampMs).toDouble()
            if (dtMs <= 0) continue // clock oddity guard; never expected in practice
            val power1 = ordered[i - 1].currentUa.toDouble() * ordered[i - 1].voltageMv!!.toDouble()
            val power2 = ordered[i].currentUa.toDouble() * ordered[i].voltageMv!!.toDouble()
            energyRaw += (power1 + power2) / 2.0 * dtMs
        }
        return energyRaw * 1e-9
    }

    /**
     * BatteryManager first; on the "unsupported" sentinel, falls back to reading raw current
     * directly from sysfs. Returns microamps (always non-negative — magnitude only, since
     * discharge-current sign convention differs across OEMs), or Long.MIN_VALUE if every source
     * failed.
     */
    private fun readCurrentUa(): Long {
        val apiVal = batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        if (apiVal != Long.MIN_VALUE) {
            return normalizeToUa(apiVal, "BatteryManager.BATTERY_PROPERTY_CURRENT_NOW")
        }
        for (path in CURRENT_SYSFS_PATHS) {
            try {
                val raw = File(path).readText().trim().toLong()
                return normalizeToUa(raw, "sysfs:$path")
            } catch (e: Exception) {
                // try next path
            }
        }
        return Long.MIN_VALUE
    }

    /**
     * Battery voltage in millivolts, or null if unavailable.
     *
     * BatteryManager.getLongProperty() has NO voltage property (only
     * CAPACITY/CHARGE_COUNTER/CURRENT_NOW/CURRENT_AVERAGE/ENERGY_COUNTER/STATUS) — voltage is
     * only exposed via the system's sticky ACTION_BATTERY_CHANGED broadcast's EXTRA_VOLTAGE
     * extra. Passing a null receiver to registerReceiver() with that action just synchronously
     * returns the last-delivered sticky Intent (no actual receiver is registered/leaked, no IPC
     * round-trip beyond reading a cached value) — this is the standard, cheap way to poll
     * current battery state on-demand, safe to call at the same 100ms cadence as [readCurrentUa].
     *
     * REAL ACCURACY CAVEAT: unlike CURRENT_NOW (a genuine instantaneous hardware reading each
     * call), EXTRA_VOLTAGE only changes when the system actually dispatches a fresh
     * ACTION_BATTERY_CHANGED broadcast — which Android rate-limits (typically on a percent-level
     * change or roughly every 30-60s), NOT every 100ms. In practice this means most consecutive
     * samples within one benchmark question will carry the IDENTICAL voltage value even though
     * current is genuinely resampled each tick — true energy is still a meaningful improvement
     * over assuming one constant voltage sitewide, but within a single short question it is
     * closer to "the one voltage reading in effect during this window" than a genuinely
     * continuously-resampled quantity. Voltage sag under real load (which DOES happen,
     * especially under the sustained CPU/GPU draw this class measures) will be under-captured
     * unless a benchmark question happens to span a broadcast update.
     */
    private fun readVoltageMv(): Long? {
        return try {
            val stickyIntent = appContext.registerReceiver(null, batteryChangedFilter)
            val voltage = stickyIntent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1
            if (voltage > 0) voltage.toLong() else null
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Both [BatteryManager.BATTERY_PROPERTY_CURRENT_NOW] and the sysfs current_now paths are
     * documented/conventionally microamps, but at least one real device used for this project
     * (OnePlus 8 Pro, post-Android-13-update) has been confirmed — in the sibling project this
     * class was ported from — to report already-milliamps instead: raw values in the 100-300
     * range under active inference load, vs. the ~100,000-300,000 range genuine microamps would
     * show. Dividing an already-mA value by 1000.0 again (as [getAverageMa] does, assuming
     * microamps) silently produces a ~1000x-undercounted result (e.g. 0.21 mA instead of
     * ~210 mA) with no error.
     *
     * A real microamp reading under active load is never this small, so magnitude alone
     * reliably distinguishes the two: below the threshold, treat the raw value as already-mA
     * and scale up to the microamp-equivalent this class's downstream math (getAverageMa()'s
     * /1000.0, the charge-counter-delta fallback, and both energy integrals) already assumes
     * throughout — normalizing here, once, keeps everything else unit-consistent rather than
     * threading an "already converted" flag through the whole class.
     */
    private fun normalizeToUa(raw: Long, source: String): Long {
        val magnitude = abs(raw)
        val alreadyMa = magnitude < MA_VS_UA_THRESHOLD
        if (!loggedUnitInterpretation) {
            loggedUnitInterpretation = true
            Log.i(
                TAG,
                "POWER_UNIT_INTERPRETATION source=$source rawSample=$raw interpretedAs=" +
                    if (alreadyMa) "milliamps (already-mA workaround applied, scaled x1000)" else "microamps (standard)"
            )
        }
        return if (alreadyMa) magnitude * 1000L else magnitude
    }

    private fun computeAvgCurrentUa(
        samples: List<Long>,
        chargeAtStartUah: Long,
        durationSecs: Int
    ): Long {
        val valid = samples.filter { it > 0L }
        if (valid.isNotEmpty()) {
            return valid.average().toLong()
        }
        if (chargeAtStartUah != Long.MIN_VALUE && durationSecs > 0) {
            val chargeAtEnd = try {
                batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
            } catch (e: Exception) {
                Long.MIN_VALUE
            }
            if (chargeAtEnd != Long.MIN_VALUE) {
                val deltaUah = chargeAtStartUah - chargeAtEnd
                if (deltaUah > 0) {
                    return deltaUah * 3600L / durationSecs
                }
            }
        }
        return Long.MIN_VALUE
    }

    companion object {
        private const val TAG = "PowerSampler"
        // Below this raw magnitude, treat BATTERY_PROPERTY_CURRENT_NOW (or the sysfs
        // current_now fallback) as already-milliamps rather than the documented microamps — see
        // normalizeToUa()'s kdoc for how this was confirmed on a real device.
        private const val MA_VS_UA_THRESHOLD = 10_000L
        private val CURRENT_SYSFS_PATHS = listOf(
            "/sys/class/power_supply/battery/current_now",
            "/sys/class/power_supply/Battery/current_now",
            "/sys/class/power_supply/bms/current_now",
        )
    }
}
