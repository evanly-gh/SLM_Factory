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

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import io.shubham0204.smollm.SmolLM
import io.shubham0204.smollmandroid.R
import io.shubham0204.smollmandroid.data.Chat
import io.shubham0204.smollmandroid.llm.SmolLMManager
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import org.koin.android.ext.android.inject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.math.abs

/**
 * Foreground service that loads a GGUF model and runs a single inference pass, logging all
 * six instrumentation metrics tagged with the caller-supplied run_id. Keeps the process alive
 * for the full duration so it is not killed by the system on Android 8+.
 *
 * Reuses the same [SmolLMManager] singleton used by the chat UI — model load time, TTFT, and
 * peak RSS are measured by SmolLMManager itself. Power/thermal monitoring mirrors the logic in
 * ChatScreenViewModel without duplicating SmolLMManager internals.
 */
class HeadlessBenchmarkService : Service() {

    private val smolLMManager: SmolLMManager by inject()
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    companion object {
        private const val CHANNEL_ID = "headless_benchmark"
        private const val NOTIF_ID = 9001

        private val CURRENT_SYSFS_PATHS = listOf(
            "/sys/class/power_supply/battery/current_now",
            "/sys/class/power_supply/Battery/current_now",
            "/sys/class/power_supply/bms/current_now",
        )
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val modelPath = intent?.getStringExtra("model_path")
        val prompt    = intent?.getStringExtra("prompt")
        val runId     = intent?.getStringExtra("run_id")

        if (modelPath == null || prompt == null || runId == null) {
            Log.d("RUN_ERROR", "run_id=${runId ?: "unknown"} reason=missing_extras")
            stopSelf(startId)
            return START_NOT_STICKY
        }

        // Must call startForeground within 5 s of startForegroundService().
        startForeground(NOTIF_ID, buildNotification(runId))

        serviceScope.launch {
            try {
                runHeadlessInference(modelPath, prompt, runId)
            } finally {
                stopSelf(startId)
            }
        }

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        serviceScope.cancel()
        super.onDestroy()
    }

    // ── Core inference pipeline ────────────────────────────────────────────────

    private suspend fun runHeadlessInference(modelPath: String, prompt: String, runId: String) {
        if (!File(modelPath).exists()) {
            Log.d("RUN_ERROR", "run_id=$runId reason=model_path_not_found path=$modelPath")
            return
        }

        // Only reload the model when the requested path differs from what is currently loaded.
        val needsLoad = smolLMManager.currentModelPath != modelPath
                     || !smolLMManager.isInstanceLoaded.get()
        if (needsLoad) {
            smolLMManager.unload()
            // isTask=true tells SmolLMManager to skip loading conversation history from the DB.
            val dummyChat = Chat(isTask = true)
            try {
                loadModelSuspend(dummyChat, modelPath)
            } catch (e: Exception) {
                Log.d("RUN_ERROR", "run_id=$runId reason=model_load_failed message=${e.message}")
                return
            }
        }

        // ── Power / thermal monitoring (mirrors ChatScreenViewModel logic) ─────
        val currentSamples  = mutableListOf<Long>()
        val thermalThrottled = AtomicBoolean(false)
        val batteryManager  = getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val powerManager    = getSystemService(Context.POWER_SERVICE)   as PowerManager
        val chargeAtStart   = batteryManager.getLongProperty(
            BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER
        )

        val monitorJob: Job = serviceScope.launch(Dispatchers.IO) {
            while (true) {
                val sample = readCurrentUa(batteryManager)
                if (sample != Long.MIN_VALUE) currentSamples.add(sample)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    if (powerManager.currentThermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) {
                        thermalThrottled.set(true)
                    }
                }
                delay(500)
            }
        }

        // ── Inference ─────────────────────────────────────────────────────────
        val response: SmolLMManager.SmolLMResponse
        try {
            response = getResponseSuspend(prompt)
        } catch (e: Exception) {
            monitorJob.cancel()
            Log.d("RUN_ERROR", "run_id=$runId reason=inference_failed message=${e.message}")
            return
        }
        monitorJob.cancel()

        // ── Metric logging ────────────────────────────────────────────────────
        val coldLoadMs  = smolLMManager.getColdLoadTimeMs()
        val avgCurrentUa = computeAvgCurrentUa(
            currentSamples, chargeAtStart, response.generationTimeSecs, batteryManager
        )

        Log.d("TTFT",      "run_id=$runId value=${response.ttftMs}")
        Log.d("TPS",       "run_id=$runId value=${response.generationSpeed}")
        Log.d("MEMORY",    "run_id=$runId value=${response.peakRssKb}")
        Log.d("COLD_LOAD", "run_id=$runId value=${coldLoadMs ?: 0}")
        Log.d("POWER",     "run_id=$runId value=${
            if (avgCurrentUa == Long.MIN_VALUE) "unsupported" else avgCurrentUa
        }")
        Log.d("THERMAL",   "run_id=$runId value=${thermalThrottled.get()}")
        Log.d("RUN_DONE",  "run_id=$runId response=${response.response}")
    }

    // ── Suspend wrappers around SmolLMManager callbacks ───────────────────────

    private suspend fun loadModelSuspend(chat: Chat, modelPath: String) =
        suspendCancellableCoroutine<Unit> { cont ->
            smolLMManager.load(
                chat      = chat,
                modelPath = modelPath,
                params    = SmolLM.InferenceParams(),
                onError   = { e -> if (cont.isActive) cont.resumeWithException(e) },
                onSuccess = {    if (cont.isActive) cont.resume(Unit) },
            )
        }

    private suspend fun getResponseSuspend(prompt: String) =
        suspendCancellableCoroutine<SmolLMManager.SmolLMResponse> { cont ->
            smolLMManager.getResponse(
                query                    = prompt,
                responseTransform        = { it },
                onPartialResponseGenerated = {},
                onSuccess  = { r -> if (cont.isActive) cont.resume(r) },
                onCancelled = {   if (cont.isActive)
                    cont.resumeWithException(CancellationException("Inference cancelled")) },
                onError    = { e -> if (cont.isActive) cont.resumeWithException(e) },
                saveToDb   = false,
            )
        }

    // ── Power helpers (mirrors ChatScreenViewModel) ───────────────────────────

    private fun readCurrentUa(batteryManager: BatteryManager): Long {
        val apiVal = batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        if (apiVal != Long.MIN_VALUE) return abs(apiVal)
        for (path in CURRENT_SYSFS_PATHS) {
            try { return abs(File(path).readText().trim().toLong()) } catch (_: Exception) {}
        }
        return Long.MIN_VALUE
    }

    private fun computeAvgCurrentUa(
        samples: List<Long>,
        chargeAtStartUah: Long,
        durationSecs: Int,
        batteryManager: BatteryManager,
    ): Long {
        val valid = samples.filter { it != Long.MIN_VALUE }
        if (valid.isNotEmpty()) return valid.average().toLong()
        if (chargeAtStartUah != Long.MIN_VALUE && durationSecs > 0) {
            val chargeAtEnd = batteryManager.getLongProperty(
                BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER
            )
            if (chargeAtEnd != Long.MIN_VALUE) {
                val deltaUah = chargeAtStartUah - chargeAtEnd
                if (deltaUah > 0) return deltaUah * 3600L / durationSecs
            }
        }
        return Long.MIN_VALUE
    }

    // ── Notification ──────────────────────────────────────────────────────────

    private fun buildNotification(runId: String) =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("SmolChat Benchmark")
            .setContentText("run_id=$runId")
            .setOngoing(true)
            .setSilent(true)
            .build()

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Headless Benchmark",
                NotificationManager.IMPORTANCE_LOW,
            )
            channel.description = "Shown while a headless inference benchmark is running"
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }
}
