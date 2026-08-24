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

import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import io.shubham0204.smollmandroid.R
import io.shubham0204.smollmandroid.data.AppDB
import io.shubham0204.smollmandroid.llm.ModelsRepository
import io.shubham0204.smollmandroid.llm.SmolLMManager
import io.shubham0204.smollmandroid.llm.readCpuThermalZoneTempC
import io.shubham0204.smollmandroid.llm.readSkinThermalTempC
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.koin.android.ext.android.inject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.Date
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs

/**
 * Foreground service that loads a GGUF model and runs a single inference pass, logging all
 * six instrumentation metrics tagged with the caller-supplied run_id. Keeps the process alive
 * for the full duration so Android cannot kill it mid-inference.
 *
 * Uses [CompletableDeferred] to bridge [SmolLMManager]'s callback-based API into coroutines,
 * which avoids dispatcher mismatch issues that can silently swallow results when callbacks are
 * delivered on Dispatchers.Main.
 */
class BenchmarkService : Service() {

    private val smolLMManager: SmolLMManager by inject()
    private val modelsRepository: ModelsRepository by inject()
    private val appDB: AppDB by inject()
    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    companion object {
        const val CHANNEL_ID = "benchmark_channel"
        private const val NOTIF_ID = 9002

        // Default headless max_tokens budget. Matches the UI's own default exactly — its
        // getResponse() call sites (ChatScreenViewModel.kt) pass no maxTokens, so SmolLM.kt's
        // default of 256 applies there too. Keeping this identical to the UI, together with
        // CONTEXT_SIZE below, means every generation config value (context size, token budget,
        // sampler seed, chat template construction, storeChats/isTask handling) is now the same
        // between the two paths — no headless-specific divergence left to A/B against.
        private const val DEFAULT_MAX_TOKENS = 256

        // Pinned to 2048, exactly matching the UI's default chat.contextSize (AppDB.kt).
        //
        // CORRECTION (see conversation): earlier comments here claimed n_ctx/n_batch changes
        // were a "confirmed root cause" of premature stopping, reasoning that llama.cpp's
        // batch/ubatch splitting alters logits enough to flip sampled tokens "at a fixed sampler
        // seed". That premise was wrong: loadModel() calls llama_sampler_init_dist with
        // LLAMA_DEFAULT_SEED (0xFFFFFFFF), which llama.cpp's own get_rng_seed()
        // (llama-sampler.cpp) treats as "pick a new random seed" via std::random_device/system
        // clock — NOT a fixed value. Every model (re)load, including every single headless call,
        // gets a genuinely different RNG seed. Generation was never reproducible run-to-run.
        //
        // What was actually observed: exactly one run each at 6144 (truncated) and at
        // 3072/2048 (complete) on one prompt. Under a randomized seed, a single run per
        // configuration cannot distinguish "context size causes divergence" from "sampling
        // variance happened to differ" — this is an unconfirmed, non-reproducible single-sample
        // observation, not demonstrated causation. 2048 is kept here only because it's the most
        // conservative choice (bit-for-bit identical to the UI's own default) while this is
        // unresolved, not because larger values were proven unsafe. Don't cite this as a settled
        // root cause, and don't raise this value based on a small number of manual runs — any
        // real conclusion needs a repeated-trial batch (see the suppressEarlyEos experiment run
        // via SUPPRESS_EARLY_EOS_OVERRIDE below) comparing failure *rates*, not anecdotes.
        private const val CONTEXT_SIZE = 2048L

        private val CURRENT_SYSFS_PATHS = listOf(
            "/sys/class/power_supply/battery/current_now",
            "/sys/class/power_supply/Battery/current_now",
            "/sys/class/power_supply/bms/current_now",
        )
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val modelPath = intent?.getStringExtra("model_path")
        val prompt    = intent?.getStringExtra("prompt")
        val runId     = intent?.getStringExtra("run_id")
        // Optional broadcast extra; defaults to 4096 when absent or not a valid int.
        val maxTokens = intent?.getStringExtra("max_tokens")?.toIntOrNull() ?: DEFAULT_MAX_TOKENS
        // DIAGNOSTIC override for the suppressEarlyEos rate-comparison experiment (see
        // conversation): lets a batch of test broadcasts force it off to compare against the
        // true-by-default headless behavior, without touching the actual default.
        val suppressEarlyEos =
            intent?.getStringExtra("suppress_early_eos")?.toBooleanStrictOrNull() ?: true

        if (modelPath == null || prompt == null || runId == null) {
            Log.d("RUN_ERROR", "run_id=${runId ?: "unknown"} reason=missing_extras")
            stopSelf(startId)
            return START_NOT_STICKY
        }

        // startForeground must be called within 5 s of startForegroundService(). If it fails
        // (e.g. missing POST_NOTIFICATIONS on API 33+, or an OEM-specific restriction), log and
        // continue running the inference anyway rather than crashing the whole process.
        // Point 2: explicit success/failure logging so we can confirm from logcat whether this
        // service actually reached PROCESS_STATE_FOREGROUND_SERVICE or silently stayed at a
        // lower process-importance state due to a caught exception here.
        try {
            startForeground(
                NOTIF_ID,
                NotificationCompat.Builder(this, CHANNEL_ID)
                    .setSmallIcon(R.mipmap.ic_launcher)
                    .setContentTitle("Running benchmark: $runId")
                    .setOngoing(true)
                    .setSilent(true)
                    .build(),
            )
            Log.d("BENCHMARK", "run_id=$runId startForeground() succeeded")
        } catch (e: Exception) {
            Log.w(
                "BENCHMARK",
                "run_id=$runId startForeground() THREW — service will run at lower process " +
                "priority than intended: ${e.message}"
            )
        }

        val exceptionHandler = CoroutineExceptionHandler { _, throwable ->
            Log.d("RUN_ERROR", "run_id=$runId reason=unexpected_error message=${throwable.message}")
            Log.e("BenchmarkService", "Unhandled exception", throwable)
            stopSelf(startId)
        }

        serviceScope.launch(exceptionHandler) {
            runBenchmark(modelPath, prompt, runId, maxTokens, suppressEarlyEos)
            stopSelf(startId)
        }

        return START_NOT_STICKY
    }

    override fun onDestroy() {
        serviceScope.cancel()
        super.onDestroy()
    }

    // ── Core pipeline ──────────────────────────────────────────────────────────

    private suspend fun runBenchmark(
        modelPath: String, prompt: String, runId: String, maxTokens: Int, suppressEarlyEos: Boolean,
    ) {
        try {
            runBenchmarkInternal(modelPath, prompt, runId, maxTokens, suppressEarlyEos)
        } catch (e: Exception) {
            Log.e("RUN_ERROR", "run_id=$runId reason=unexpected_error message=${e.message}", e)
        }
    }

    private suspend fun runBenchmarkInternal(
        rawModelPath: String, prompt: String, runId: String, maxTokens: Int, suppressEarlyEos: Boolean,
    ) {
        val modelPath = resolveReadableModelPath(rawModelPath, runId) ?: return

        // Resolve the real target chat up front so the DB writes further down (question, answer,
        // metrics) land in the exact same chat the manual UI would show as "active".
        val targetChat = appDB.getRecentlyUsedChat() ?: appDB.loadDefaultChat()

        // Keep the chat's "selected model" pointer (chat.llmModelId — the same field the manual
        // model-switcher's updateChatLLMParams() writes) in sync with whatever was just
        // benchmarked, so the chat header immediately reflects the most recently actually-used
        // model. Runs on every single headless call, including every question in a
        // run_autobench.py session and every quant level in an agent_quantize.py sweep — it's
        // expected to change rapidly during a sweep rather than settle on a stable summary.
        val modelId = resolveOrRegisterModelId(modelPath)
        if (targetChat.llmModelId != modelId) {
            targetChat.llmModelId = modelId
            appDB.updateChat(targetChat)
        }

        // Every headless call reloads the model before inference — deliberately, not an
        // optimization gap. See SmolLMManager.loadIsolatedSingleTurn() for why this is the only
        // way (without native/JNI changes) to guarantee each question is a genuinely fresh,
        // single-turn interaction, and why targetChat's own settings are still used for an
        // apples-to-apples comparison with how that chat performs manually. The visible chat log
        // and per-message metrics are unaffected — those are written using the real targetChat
        // (not the isTask=true copy loadIsolatedSingleTurn uses internally) further below, exactly
        // as before. This same helper is also used by ChatScreenViewModel for manual chat sends,
        // so both paths define "isolated single turn" identically and stay in sync.
        Log.d("BENCHMARK", "run_id=$runId loading model from $modelPath for an isolated single-turn call")
        val loadDeferred = CompletableDeferred<Result<Unit>>()
        smolLMManager.loadIsolatedSingleTurn(
            chat      = targetChat,
            modelPath = modelPath,
            onError   = { e -> loadDeferred.complete(Result.failure(e)) },
            onSuccess = {    loadDeferred.complete(Result.success(Unit)) },
            contextSizeOverride = CONTEXT_SIZE,
        )
        loadDeferred.await().getOrElse { e ->
            Log.d("RUN_ERROR", "run_id=$runId reason=model_load_failed message=${e.message}")
            return
        }

        // ── Power / thermal monitoring ─────────────────────────────────────────
        val currentSamples  = mutableListOf<Long>()
        // Fix 1: track the actual peak PowerManager status level, not just a crossed-threshold
        // boolean, so we can log its real name (Normal/Light/Moderate/Severe/Critical) below.
        val maxThermalStatus = AtomicInteger(PowerManager.THERMAL_STATUS_NONE)
        // Real hardware temperature (°C), sampled every ~10th tick (roughly once per second),
        // alongside the coarse PowerManager status above — see readCpuThermalZoneTempC() (raw
        // CPU junction sensor) and readSkinThermalTempC() (the sensor OEM throttling policies
        // typically key off instead). Same shared helpers ChatScreenViewModel's manual chat path
        // uses, so both report temperature identically. 0f (the initial value) reads back as
        // "unavailable" if the corresponding sensor is never readable on this device.
        val maxCpuThermalTempC = AtomicReference(0f)
        val maxSkinThermalTempC = AtomicReference(0f)
        val batteryManager   = getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val powerManager     = getSystemService(Context.POWER_SERVICE)   as PowerManager
        val chargeAtStart    = batteryManager.getLongProperty(
            BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER
        )

        val monitorJob: Job = serviceScope.launch {
            var tick = 0
            while (true) {
                val sample = readCurrentUa(batteryManager)
                if (sample > 0L) currentSamples.add(sample)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    val status = powerManager.currentThermalStatus
                    if (status > maxThermalStatus.get()) maxThermalStatus.set(status)
                }
                if (tick % 10 == 0) {
                    readCpuThermalZoneTempC()?.let { temp ->
                        if (temp > maxCpuThermalTempC.get()) maxCpuThermalTempC.set(temp)
                    }
                    readSkinThermalTempC()?.let { temp ->
                        if (temp > maxSkinThermalTempC.get()) maxSkinThermalTempC.set(temp)
                    }
                }
                tick++
                delay(100)
            }
        }

        // ── Inference ──────────────────────────────────────────────────────────
        // Fix 4: write the user prompt into targetChat's history up front, mirroring
        // ChatScreenViewModel.sendUserQuery() which inserts the user message before the response
        // arrives, so it's visible immediately even if inference later fails.
        appDB.addUserMessage(targetChat.id, prompt)

        Log.d("BENCHMARK", "run_id=$runId starting inference")
        // Fix 3: every headless call now reloads before inference (see above), so each one pays
        // the same "first real inference after load" cost measured here (TTFT/TPS) — this can be
        // markedly slower than steady-state chat UI numbers for the same model/phone. This is
        // expected, not a bug in this instrumentation: SmolLM.load()'s cold-load timer only
        // covers llama_model_load_from_file()/llama_init_from_model() (mmap() the GGUF file and
        // allocate the context) — see LLMInference.cpp's loadModel(). The actual llama_decode()
        // call that touches every weight tensor only happens inside completionLoop(), i.e. on the
        // FIRST real prompt. Because the model is loaded with useMmap=true by default, the OS
        // only pages in each weight tensor's memory from storage/page cache on first access,
        // which happens during that first decode — not during the mmap() call itself. So the
        // page-fault cost of "warming up" the mmap'd weights lands entirely on the first
        // inference's timing, not on cold load. A clean fix would require a native-side warmup
        // pass (e.g. a bounded single-token dummy completion run right after loadModel(), calling
        // llama_decode() once outside the timed path) — that needs a new native entry point in
        // LLMInference.cpp/SmolLM.kt and rebuilding all native ABI variants, which is out of scope
        // here. No Kotlin-only fix exists: calling instance.getResponse()/getResponseAsFlow() for
        // a throwaway warmup prompt would itself run through the same unbounded completionLoop()
        // (no token cap, runs until the model emits [EOG]), which is unsafe to fire blindly on
        // every model load. Leaving this documented rather than forcing a risky workaround.
        // Point 1: hold a partial wake lock for the duration of inference. This guarantees the
        // CPU doesn't drop into a suspend/idle state mid-run, matching what implicitly happens
        // when the screen is on and the user is actively looking at the app. Note this does NOT
        // change the process's scheduling class/cgroup (see the finding below) — it only
        // prevents full CPU suspend, which a foreground service already mostly covers, but makes
        // that guarantee explicit and immune to any gap in foreground-service CPU wake handling.
        val wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK, "SmolChat:BenchmarkInference"
        )

        // Point 3, investigated: raising Process.setThreadPriority() on THIS thread (the one
        // calling getResponse()) would NOT affect the thread that actually runs inference.
        // SmolLMManager.getResponse() launches its own independent
        // CoroutineScope(Dispatchers.Default).launch { ... } and returns immediately — the real
        // llama_decode() work happens on whatever pool thread that separate coroutine lands on,
        // not on the calling thread. This is true for BOTH the manual chat path
        // (ChatScreenViewModel calls the exact same getResponse()) and this headless path — they
        // share identical code, so there is no thread-priority difference between them to find or
        // fix here. A real priority boost would have to be applied inside SmolLMManager's own
        // launched coroutine, which would affect both paths identically (not a headless-specific
        // fix), and would still only change scheduling within this process's already-assigned
        // cgroup, not lift the whole process into a higher-priority scheduling class the way a
        // focused foreground Activity gets — see the finding below. Not adding an inert
        // thread-priority call here rather than leaving in code that looks like a fix but isn't.

        // Point 4: log CPU frequency immediately before/after inference so headless vs manual
        // clock speed can be compared directly from logcat in the next test run.
        Log.d("CPU_FREQ", "run_id=$runId before_inference ${readCpuFreqsKhz()}")

        // Wake lock acquisition is best-effort: if it fails for any reason, inference must still
        // proceed rather than hang forever waiting on inferenceDeferred.
        try {
            wakeLock.acquire(2 * 60 * 1000L /* 2 min safety timeout */)
        } catch (e: Exception) {
            Log.w("BENCHMARK", "run_id=$runId could not acquire wake lock: ${e.message}")
        }

        val promptDispatchTimeMs = System.currentTimeMillis()
        val inferenceDeferred = CompletableDeferred<Result<SmolLMManager.SmolLMResponse>>()
        smolLMManager.getResponse(
            query                      = prompt,
            promptDispatchTimeMs       = promptDispatchTimeMs,
            responseTransform          = { it },
            onPartialResponseGenerated = {},
            onSuccess  = { r -> inferenceDeferred.complete(Result.success(r)) },
            onCancelled = {   inferenceDeferred.complete(
                Result.failure(Exception("Inference cancelled"))) },
            onError    = { e -> inferenceDeferred.complete(Result.failure(e)) },
            saveToDb   = false,
            maxTokens  = maxTokens,
            // Headless-only: suppresses this model's EOG token for the first ~40 generated
            // tokens (see LLMInference.h's kEosSuppressTokenWindow) when true (the default —
            // see the suppress_early_eos override above, added only to run a repeated-trial
            // rate comparison; not exposed to normal callers). The manual chat UI's getResponse()
            // calls never pass this, so it stays false/off there — deliberately the one
            // remaining difference from the UI path, unlike contextSize/max_tokens which were
            // unified earlier. Whether this default actually reduces premature-stop *rate* vs.
            // off is still being measured (single before/after runs aren't reproducible evidence
            // — see CONTEXT_SIZE's comment above for why) — don't treat "true" as proven correct.
            suppressEarlyEos = suppressEarlyEos,
        )

        // Await inference before cancelling the power monitor.
        val response = inferenceDeferred.await().getOrElse { e ->
            monitorJob.cancel()
            if (wakeLock.isHeld) wakeLock.release()
            Log.d("RUN_ERROR", "run_id=$runId reason=inference_failed message=${e.message}")
            return
        }
        monitorJob.cancel()

        Log.d("CPU_FREQ", "run_id=$runId after_inference ${readCpuFreqsKhz()}")
        if (wakeLock.isHeld) wakeLock.release()

        // ── Compute metrics ────────────────────────────────────────────────────
        val coldLoadMs   = smolLMManager.getColdLoadTimeMs()
        // Fix 2: average over samples in µA, then convert to mA with one decimal place —
        // matching ChatScreenViewModel's computeAvgCurrentUa()/mA conversion, but with decimal
        // precision instead of truncating integer division.
        val avgCurrentUa = computeAvgCurrentUa(
            currentSamples, chargeAtStart, response.generationTimeSecs, batteryManager
        )
        val avgPowerMa = if (avgCurrentUa == Long.MIN_VALUE) null else avgCurrentUa / 1000.0
        val thermalStatus = thermalStatusName(maxThermalStatus.get())

        // Fix 4 / Issue 1: write the assistant's response into the same chat, attach these same
        // metrics to that message (mirrors ChatScreenViewModel.persistMessageMetrics()), and bump
        // dateUsed exactly like ChatScreenViewModel.sendUserQuery() does, so this chat is
        // correctly considered "recently used" for any subsequent headless run or app open.
        val savedMessageId = appDB.addAssistantMessage(targetChat.id, response.response)
        appDB.updateMessageMetrics(
            messageId = savedMessageId,
            ttftMs = response.ttftMs,
            decodeTps = response.generationSpeed,
            peakRssKb = response.peakRssKb,
            coldLoadTimeMs = coldLoadMs,
            avgCurrentUa = if (avgCurrentUa == Long.MIN_VALUE) null else avgCurrentUa,
            thermalStatus = thermalStatus,
        )
        appDB.updateChat(targetChat.copy(dateUsed = Date()))

        // ── Log metrics ────────────────────────────────────────────────────────
        Log.d("COLD_LOAD", "run_id=$runId value=${coldLoadMs ?: 0}")
        Log.d("TTFT",      "run_id=$runId value=${response.ttftMs}")
        Log.d("TPS",       "run_id=$runId value=${response.generationSpeed}")
        Log.d("MEMORY",    "run_id=$runId value=${response.peakRssKb}")
        Log.d("POWER",     "run_id=$runId value=${
            avgPowerMa?.let { "%.1f".format(it) } ?: "unsupported"}")
        Log.d("THERMAL",   "run_id=$runId value=$thermalStatus")
        // Additive real-temperature readings alongside the coarse THERMAL status above — does
        // not change or replace that line. THERMAL_TEMP_CPU is the raw CPU junction sensor (see
        // readCpuThermalZoneTempC()); THERMAL_TEMP_SKIN is the skin-therm-usr sensor (see
        // readSkinThermalTempC()), which OEM throttling policies more often key off.
        val maxCpuTemp = maxCpuThermalTempC.get()
        val maxSkinTemp = maxSkinThermalTempC.get()
        Log.d("THERMAL_TEMP_CPU", "run_id=$runId value=${
            if (maxCpuTemp > 0f) "%.1f".format(maxCpuTemp) else "unavailable"}")
        Log.d("THERMAL_TEMP_SKIN", "run_id=$runId value=${
            if (maxSkinTemp > 0f) "%.1f".format(maxSkinTemp) else "unavailable"}")
        Log.d("RUN_DONE",  "run_id=$runId response=${response.response}")
    }

    /** Maps a PowerManager.THERMAL_STATUS_* constant to its display name. */
    private fun thermalStatusName(status: Int): String = when (status) {
        PowerManager.THERMAL_STATUS_NONE -> "Normal"
        PowerManager.THERMAL_STATUS_LIGHT -> "Light"
        PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"
        PowerManager.THERMAL_STATUS_SEVERE -> "Severe"
        PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"
        else -> "Unknown"
    }

    /**
     * Resolves a caller-supplied model path (e.g. an `adb push`'d file under
     * /sdcard/Download) to a path this app can actually open.
     *
     * Raw paths on shared external storage are frequently unreadable under scoped storage
     * (Android 10+) unless the file was imported through the app's normal "Import GGUF" flow.
     * That flow (see ImportModelScreen.kt + DownloadModelsViewModel.copyModelFile) opens the
     * file via Storage Access Framework — a content:// Uri from ACTION_OPEN_DOCUMENT, which
     * bypasses scoped storage — and immediately copies its bytes into internal app storage
     * (context.filesDir). Native code then always operates on that internal copy, never the
     * original external path.
     *
     * This mirrors that mechanism for the headless path:
     *   1. If a model with the same file name has already been imported via the normal UI flow,
     *      reuse its registered internal path directly.
     *   2. Otherwise, attempt to copy the raw path's bytes into internal storage ourselves.
     *      This succeeds for any path this app can actually read (e.g. its own
     *      external-files-dir, which requires no permission) and fails cleanly for genuinely
     *      inaccessible paths (e.g. an arbitrary file under /sdcard/Download, which scoped
     *      storage blocks without MANAGE_EXTERNAL_STORAGE — a heavy, Play-Store-restricted
     *      permission not worth requesting just for this benchmarking feature).
     */
    private fun resolveReadableModelPath(rawPath: String, runId: String): String? {
        val fileName = File(rawPath).name

        modelsRepository.getAvailableModelsList().firstOrNull { it.name == fileName }?.let { existing ->
            Log.d(
                "BENCHMARK",
                "run_id=$runId '$fileName' was already imported via the app's normal flow, " +
                "reusing its internal path: ${existing.path}"
            )
            return existing.path
        }

        val cacheDir = File(filesDir, "headless_benchmark_cache").apply { mkdirs() }
        val destFile = File(cacheDir, fileName)
        return try {
            FileInputStream(rawPath).use { input ->
                FileOutputStream(destFile).use { output -> input.copyTo(output) }
            }
            Log.d(
                "BENCHMARK",
                "run_id=$runId copied $rawPath into internal storage at ${destFile.absolutePath}"
            )
            destFile.absolutePath
        } catch (e: Exception) {
            Log.d(
                "RUN_ERROR",
                "run_id=$runId reason=model_path_unreadable path=$rawPath message=${e.message} " +
                "hint=push the file to this app's external files dir instead, e.g. " +
                "'adb push model.gguf /sdcard/Android/data/io.shubham0204.smollmandroid/files/' " +
                "which requires no storage permission under scoped storage"
            )
            null
        }
    }

    /**
     * Resolves [modelPath] to an LLMModel row id, so it can be written to chat.llmModelId — the
     * same field the manual model-switcher (ChatScreenViewModel.updateChatLLMParams) uses,
     * meaning both paths keep the chat header in sync through one shared field rather than two
     * parallel ones.
     *
     * If a row for this exact path already exists (formally imported via the app's normal Import
     * GGUF flow, or already registered by a previous headless run), reuse its id. Otherwise this
     * is a model that only exists as our own headless_benchmark_cache copy — register a minimal
     * row for it (name = file name, no url/chatTemplate/contextSize) so it can be referenced. This
     * also means such a model becomes visible in the manual "Change Model" picker going forward,
     * which is expected: headless and manual read from the same LLMModel table by design.
     */
    private fun resolveOrRegisterModelId(modelPath: String): Long {
        modelsRepository.getAvailableModelsList().firstOrNull { it.path == modelPath }?.let { existing ->
            return existing.id
        }
        return appDB.addModel(
            name = File(modelPath).name,
            url = "",
            path = modelPath,
            contextSize = 0,
            chatTemplate = "",
        )
    }

    /**
     * Reads each online CPU core's current clock speed from
     * /sys/devices/system/cpu/cpuN/cpufreq/scaling_cur_freq (in kHz). Used only for diagnostic
     * logging (point 4) — compares directly against the same read done during a manual chat run
     * to check whether the CPU is actually running at a lower frequency during headless
     * inference, which would indicate governor/cgroup-level throttling rather than an app-code
     * bug.
     */
    private fun readCpuFreqsKhz(): String {
        val freqs = (0 until Runtime.getRuntime().availableProcessors()).map { core ->
            try {
                val khz = File("/sys/devices/system/cpu/cpu$core/cpufreq/scaling_cur_freq")
                    .readText().trim().toLongOrNull()
                "cpu$core=${khz ?: "unreadable"}"
            } catch (_: Exception) {
                "cpu$core=unreadable"
            }
        }
        return freqs.joinToString(" ")
    }

    // ── Power helpers ──────────────────────────────────────────────────────────

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
        val valid = samples.filter { it > 0L }
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
}
