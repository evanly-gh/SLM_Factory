/*
 * Copyright (C) 2025 Shubham Panchal
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

package io.shubham0204.smollmandroid.llm

import android.util.Log
import io.shubham0204.smollm.SmolLM
import io.shubham0204.smollmandroid.data.AppDB
import io.shubham0204.smollmandroid.data.Chat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.koin.core.annotation.Single
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock
import kotlin.time.measureTime

private const val LOGTAG = "[SmolLMManager-Kt]"
private val LOGD: (String) -> Unit = { Log.d(LOGTAG, it) }

private val CPU_THERMAL_ZONE_PATHS = listOf(
    "/sys/class/thermal/thermal_zone0/temp",
    "/sys/class/thermal/thermal_zone1/temp",
    "/sys/class/thermal/thermal_zone2/temp",
)

private const val THERMAL_CLASS_DIR = "/sys/class/thermal"
private const val SKIN_THERMAL_ZONE_TYPE = "skin-therm-usr"

/**
 * Reads the device's raw CPU junction temperature in °C from the first readable thermal_zone
 * sysfs path, trying each in order. These files report millidegree Celsius integers (e.g.
 * "45231" means 45.231°C, hence the /1000 conversion). Returns null if none of the paths are
 * readable (permissions/SELinux vary by device). This is a real, finer-grained temperature
 * reading, complementing (not replacing) PowerManager's coarse THERMAL_STATUS_* level — but note
 * it is NOT necessarily the sensor OEM throttling policies actually key off; see
 * [readSkinThermalTempC] for that.
 *
 * Shared by both BenchmarkService (headless) and ChatScreenViewModel (manual chat) so both paths
 * sample and report hardware temperature identically.
 */
fun readCpuThermalZoneTempC(): Float? {
    for (path in CPU_THERMAL_ZONE_PATHS) {
        try {
            val millidegrees = File(path).readText().trim().toLongOrNull() ?: continue
            return millidegrees / 1000f
        } catch (_: Exception) {
        }
    }
    return null
}

/**
 * Reads the device's skin-temperature sensor in °C — the sensor many OEM throttling policies
 * (e.g. OnePlus/OxygenOS) actually base their THERMAL_STATUS_* decisions on, unlike the raw CPU
 * junction sensor read by [readCpuThermalZoneTempC]. Unlike the CPU junction sensor's fixed
 * thermal_zone0-2 path list, the skin sensor's zone number varies by device and can even shift
 * across reboots, so this never hardcodes a zone number: it scans every
 * /sys/class/thermal/thermal_zoneN/type file present and uses whichever zone's type is exactly
 * "skin-therm-usr". Returns null if no such zone exists, or its type/temp files aren't readable.
 *
 * Shared by both BenchmarkService (headless) and ChatScreenViewModel (manual chat).
 */
fun readSkinThermalTempC(): Float? {
    val zoneDirs = File(THERMAL_CLASS_DIR)
        .listFiles { file -> file.isDirectory && file.name.startsWith("thermal_zone") }
        ?: return null

    for (zoneDir in zoneDirs) {
        val type = try {
            File(zoneDir, "type").readText().trim()
        } catch (_: Exception) {
            continue
        }
        if (type != SKIN_THERMAL_ZONE_TYPE) continue

        return try {
            val millidegrees = File(zoneDir, "temp").readText().trim().toLongOrNull() ?: return null
            millidegrees / 1000f
        } catch (_: Exception) {
            null
        }
    }
    return null
}

@Single
class SmolLMManager(private val appDB: AppDB) {
    private val instance = SmolLM()

    // Use ReentrantLock for thread-safe state management without suspending
    private val stateLock = ReentrantLock()

    @Volatile
    private var responseGenerationJob: Job? = null

    @Volatile
    private var modelInitJob: Job? = null

    @Volatile
    private var chat: Chat? = null

    // Use java.util.concurrent.atomic for better thread safety
    val isInstanceLoaded = AtomicBoolean(false)

    @Volatile
    var isInferenceOn = false
        private set

    @Volatile
    var currentModelPath: String? = null
        private set

    @Volatile
    private var lastColdLoadTimeMs: Long = 0L

    /**
     * Returns the cold load duration recorded during the last [load] call, or null if no model has
     * been loaded yet (or after [unload]).
     */
    fun getColdLoadTimeMs(): Long? = if (lastColdLoadTimeMs > 0L) lastColdLoadTimeMs else null

    data class SmolLMResponse(
        val response: String,
        val generationSpeed: Float,
        val generationTimeSecs: Int,
        val contextLengthUsed: Int,
        val usedJinjaTemplate: Boolean = true,
        val ttftMs: Long = 0L,
        val peakRssKb: Long = 0L,
        // Row id of the assistant message this response was saved as (only set when
        // saveToDb=true and a message was actually inserted), so callers can attach inference
        // metrics to that specific message afterward via AppDB.updateMessageMetrics().
        val savedMessageId: Long? = null,
    )

    fun load(
        chat: Chat,
        modelPath: String,
        params: SmolLM.InferenceParams = SmolLM.InferenceParams(),
        onError: (Exception) -> Unit,
        onSuccess: () -> Unit,
    ) {
        stateLock.withLock {
            // Request cancellation of any existing load immediately, but note this alone does
            // NOT stop it: instance.load() runs a blocking native JNI call inside
            // withContext(Dispatchers.IO), and coroutine cancellation is cooperative — it can't
            // interrupt code that isn't at a suspension point. If a second instance.load() call
            // starts while the first is still physically running, both mutate the same shared
            // native SmolLM instance concurrently, which can corrupt native state (e.g.
            // GGUFReader's native handle, causing a "Use GGUFReader.load() to initialize the
            // reader" crash). The `previousJob` capture + join below ensures the previous load
            // has fully finished before this one touches `instance`.
            val previousJob = modelInitJob
            previousJob?.cancel()

            try {
                this.chat = chat
                modelInitJob = CoroutineScope(Dispatchers.Default).launch {
                    try {
                        previousJob?.join()
                        val loadDuration = measureTime { instance.load(modelPath, params) }
                        lastColdLoadTimeMs = loadDuration.inWholeMilliseconds
                        LOGD("Model loaded | Cold load time: ${lastColdLoadTimeMs}ms")

                        if (chat.systemPrompt.isNotEmpty()) {
                            instance.addSystemPrompt(chat.systemPrompt)
                            LOGD("System prompt added")
                        }

                        if (!chat.isTask) {
                            appDB.getMessagesForModel(chat.id).forEach { message ->
                                if (message.isUserMessage) {
                                    instance.addUserMessage(message.message)
                                    LOGD("User message added: ${message.message}")
                                } else {
                                    instance.addAssistantMessage(message.message)
                                    LOGD("Assistant message added: ${message.message}")
                                }
                            }
                        }

                        withContext(Dispatchers.Main) {
                            isInstanceLoaded.set(true)
                            currentModelPath = modelPath
                            onSuccess()
                        }
                    } catch (e: CancellationException) {
                        LOGD("Model loading cancelled")
                        throw e
                    } catch (e: Exception) {
                        LOGD("Error loading model: ${e.message}")
                        withContext(Dispatchers.Main) {
                            onError(e)
                        }
                    }
                }
            } catch (e: Exception) {
                onError(e)
            }
        }
    }

    /**
     * Loads [modelPath] for a single, context-isolated turn: [chat]'s own inference settings
     * (minP/temperature/contextSize/nThreads/mmap/mlock/chatTemplate/systemPrompt) are applied —
     * for an apples-to-apples comparison with how that chat behaves normally — but its prior DB
     * message history is never replayed into the model. This is achieved by passing
     * chat.copy(isTask = true) to [load] internally: isTask=true skips the DB history replay,
     * and the underlying unload()+load() still clears the native chat state either way (SmolLM's
     * native startCompletion() unconditionally appends each query onto its own internal message
     * list, which is only ever cleared by a fresh instance.load() — there is no cheaper
     * "just clear context" primitive without native/JNI changes).
     *
     * This is the single source of truth for "genuinely fresh, single-turn interaction" — used by
     * both BenchmarkService (headless runs) and ChatScreenViewModel (manual chat sends) so both
     * stay in sync automatically rather than each re-implementing the same isolation logic.
     */
    fun loadIsolatedSingleTurn(
        chat: Chat,
        modelPath: String,
        onError: (Exception) -> Unit,
        onSuccess: () -> Unit,
        // Overrides chat.contextSize for this load only — e.g. the headless benchmark path uses
        // this to run with a larger context window (to fit a raised max_tokens budget) without
        // touching chat.contextSize itself, which the manual chat UI also reads.
        contextSizeOverride: Long? = null,
    ) {
        unload()
        load(
            chat      = chat.copy(isTask = true),
            modelPath = modelPath,
            params    = SmolLM.InferenceParams(
                chat.minP,
                chat.temperature,
                false,
                contextSizeOverride ?: chat.contextSize.toLong(),
                chat.chatTemplate.takeIf { it.isNotBlank() && ("{%" in it || "{{" in it) },
                chat.nThreads,
                chat.useMmap,
                chat.useMlock,
            ),
            onError = onError,
            onSuccess = onSuccess,
        )
    }

    fun unload() {
        stateLock.withLock {
            // Cancel jobs
            responseGenerationJob.safeCancelJobIfActive()

            // Block until any in-flight load has actually stopped before closing the native
            // instance below. Requesting cancellation alone isn't enough — see the comment in
            // load() — so closing the instance while a previous instance.load() native call is
            // still executing would race with it and leave the shared SmolLM instance in a
            // corrupted or stale state.
            modelInitJob?.let { job ->
                if (job.isActive) {
                    job.cancel()
                    runBlocking { job.join() }
                }
            }

            isInstanceLoaded.set(false)
            chat = null
            currentModelPath = null
            lastColdLoadTimeMs = 0L

            // Close synchronously to prevent race with subsequent load()
            try {
                instance.close()
            } catch (e: Exception) {
                LOGD("Error closing instance: ${e.message}")
            }
        }
    }

    fun getResponse(
        query: String,
        promptDispatchTimeMs: Long = 0L,
        responseTransform: (String) -> String,
        onPartialResponseGenerated: (String) -> Unit,
        onSuccess: (SmolLMResponse) -> Unit,
        onCancelled: () -> Unit,
        onError: (Exception) -> Unit,
        saveToDb: Boolean = true,
        maxTokens: Int = 256,
        suppressEarlyEos: Boolean = false,
    ) {
        stateLock.withLock {
            // Check if model is loaded
            if (!isInstanceLoaded.get()) {
                onError(IllegalStateException("Model not loaded"))
                return
            }

            // Cancel any existing response generation
            responseGenerationJob?.cancel()

            responseGenerationJob = CoroutineScope(Dispatchers.Default).launch {
                try {
                    isInferenceOn = true
                    var response = ""
                    var ttftMs = 0L
                    // Use caller-supplied timestamp if provided so TTFT includes coroutine scheduling delay
                    val promptSubmitTime = if (promptDispatchTimeMs > 0L) promptDispatchTimeMs
                                          else System.currentTimeMillis()
                    var firstTokenReceived = false
                    var peakRssKb = 0L

                    val rssPollingJob = launch(Dispatchers.IO) {
                        while (isActive) {
                            val rss = readVmRssKb()
                            if (rss > peakRssKb) peakRssKb = rss
                            delay(100)
                        }
                    }

                    val duration = measureTime {
                        instance.getResponseAsFlow(query, maxTokens, suppressEarlyEos).collect { piece ->
                            if (!firstTokenReceived) {
                                ttftMs = System.currentTimeMillis() - promptSubmitTime
                                firstTokenReceived = true
                            }
                            response += piece
                            withContext(Dispatchers.Main) {
                                onPartialResponseGenerated(response)
                            }
                        }
                    }

                    rssPollingJob.cancel()
                    response = responseTransform(response)

                    // Use llama.cpp's native TPS: it has the accurate token count and already
                    // excludes prompt-processing time from its denominator.
                    val nativeTps = instance.getResponseGenerationSpeed()

                    // Thread-safe access to chat
                    val currentChat = stateLock.withLock { chat }

                    val savedMessageId =
                        if (saveToDb && currentChat != null) {
                            appDB.addAssistantMessage(currentChat.id, response)
                        } else {
                            null
                        }

                    LOGD(
                        "Inference complete | TTFT: ${ttftMs}ms | " +
                        "TPS: $nativeTps | " +
                        "Peak RSS: ${peakRssKb}KB"
                    )

                    withContext(Dispatchers.Main) {
                        isInferenceOn = false
                        onSuccess(
                            SmolLMResponse(
                                response = response,
                                generationSpeed = nativeTps,
                                generationTimeSecs = duration.inWholeSeconds.toInt(),
                                contextLengthUsed = instance.getContextLengthUsed(),
                                usedJinjaTemplate = instance.usedJinjaTemplate,
                                ttftMs = ttftMs,
                                peakRssKb = peakRssKb,
                                savedMessageId = savedMessageId,
                            )
                        )
                    }
                } catch (e: CancellationException) {
                    isInferenceOn = false
                    withContext(Dispatchers.Main) {
                        onCancelled()
                    }
                } catch (e: Exception) {
                    isInferenceOn = false
                    withContext(Dispatchers.Main) {
                        onError(e)
                    }
                }
            }
        }
    }

    private val BENCH_PROMPT_PROCESSING_TOKENS = 512
    private val BENCH_TOKEN_GENERATION_TOKENS = 128
    private val BENCH_SEQUENCE = 1
    private val BENCH_REPETITION = 3

    fun benchmark(onResult: (String) -> Unit) {
        CoroutineScope(Dispatchers.Default).launch {
            val result = instance.benchModel(
                BENCH_PROMPT_PROCESSING_TOKENS,
                BENCH_TOKEN_GENERATION_TOKENS,
                BENCH_SEQUENCE,
                BENCH_REPETITION
            )
            withContext(Dispatchers.Main) {
                onResult(result)
            }
        }
    }

    fun stopResponseGeneration() {
        stateLock.withLock {
            responseGenerationJob.safeCancelJobIfActive()
            isInferenceOn = false
        }
    }

    /**
     * Reads VmRSS (resident set size) from /proc/self/status in kilobytes.
     * Returns 0 if the file cannot be read or the line is missing.
     *
     * /proc/self/status contains a line like: "VmRSS:    627432 kB"
     * VmRSS is the actual RAM pages currently mapped for this process, making
     * it a more accurate measure of true memory pressure than totalPss.
     */
    private fun readVmRssKb(): Long {
        return try {
            File("/proc/self/status").useLines { lines ->
                lines.firstOrNull { it.startsWith("VmRSS:") }
                    ?.split("\\s+".toRegex())
                    ?.getOrNull(1)
                    ?.toLongOrNull()
                    ?: 0L
            }
        } catch (_: Exception) { 0L }
    }

    private fun Job?.safeCancelJobIfActive() {
        this?.cancel()
    }
}