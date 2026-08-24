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

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import io.shubham0204.smollmandroid.ui.screens.chat.ChatActivity

/**
 * Receives `com.smollmandroid.RUN_PROMPT` broadcasts (e.g. from ADB) and delegates to
 * [HeadlessBenchmarkService] which runs inference in a foreground service so the process
 * is not killed mid-run.
 *
 * Required extras:
 *   model_path — absolute path to a .gguf file on device
 *   prompt     — text to send for inference
 *   run_id     — unique string used to correlate log lines
 *
 * Optional extras:
 *   max_tokens — hard cap on generated tokens (defaults to 4096 if absent; see
 *                BenchmarkService.DEFAULT_MAX_TOKENS / CONTEXT_SIZE for the paired context window)
 *
 * Example ADB command:
 *   adb shell am broadcast -a com.smollmandroid.RUN_PROMPT \
 *     -n io.shubham0204.smollmandroid/.headless.HeadlessBenchmarkReceiver \
 *     --es model_path /sdcard/Download/model-q4_k_m.gguf \
 *     --es prompt "What is the capital of France?" \
 *     --es run_id run_001 \
 *     --es max_tokens 4096
 */
class HeadlessBenchmarkReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        Log.d("BROADCAST_RECEIVER", "onReceive called with action: ${intent.action}")

        val modelPath = intent.getStringExtra("model_path")
        val prompt = intent.getStringExtra("prompt")
        val runId = intent.getStringExtra("run_id")
        val maxTokens = intent.getStringExtra("max_tokens")
        // DIAGNOSTIC-only override for a repeated-trial rate comparison; see BenchmarkService.
        val suppressEarlyEos = intent.getStringExtra("suppress_early_eos")

        if (modelPath == null || prompt == null || runId == null) {
            Log.d("RUN_ERROR", "run_id=${runId ?: "unknown"} reason=missing_extras" +
                " (need model_path, prompt, run_id)")
            return
        }

        Log.d("BROADCAST_RECEIVER", "received run_id=$runId")

        // Bring ChatActivity to the foreground so the run is visible as it happens, matching the
        // manual chat flow. No extras are passed — ChatActivity's own ViewModel init resolves the
        // "currently active chat" via ChatScreenViewModel.initializeUIState() ->
        // AppDB.loadDefaultChat() -> getRecentlyUsedChat(), the exact same lookup
        // BenchmarkService already uses to decide which chat to write headless messages into.
        // FLAG_ACTIVITY_REORDER_TO_FRONT brings an already-running instance forward as-is
        // (preserving whatever chat the user is currently viewing) rather than recreating it;
        // FLAG_ACTIVITY_NEW_TASK is required to start an Activity from this non-Activity context.
        //
        // Wrapped: launching a UI Activity from a BroadcastReceiver can throw in restricted
        // states (e.g. pre-first-unlock / Direct Boot "locked" state right after a reboot), and
        // an uncaught exception here would prevent execution from ever reaching the service-start
        // call below — logging RECEIVER_ERROR here rules this in/out as the actual failure point
        // for the "receiver logs fire, but nothing happens afterward, only right after reboot"
        // symptom, rather than assuming the service-start call itself is the culprit.
        val activityIntent = Intent(context, ChatActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT
        }
        try {
            context.startActivity(activityIntent)
        } catch (e: Exception) {
            Log.e(
                "RECEIVER_ERROR",
                "run_id=$runId startActivity(ChatActivity) threw " +
                "${e.javaClass.simpleName}: ${e.message}",
                e
            )
        }

        // Wrapped: startForegroundService()/startService() can throw (SecurityException,
        // IllegalStateException, ForegroundServiceStartNotAllowedException on API 31+) if called
        // from a background-restricted context — plausible right after boot, before first
        // unlock. Logging both the throw case AND a "returned normally" confirmation lets us
        // distinguish two different failure modes: the call itself throwing here (caught below)
        // vs. the call returning normally but the system silently deferring/dropping the actual
        // service start until after unlock (in which case this log line appears but
        // BenchmarkService still never logs anything — pointing at OS-level deferral, not an
        // uncaught exception).
        val serviceIntent = Intent(context, BenchmarkService::class.java).apply {
            putExtra("model_path", modelPath)
            putExtra("prompt", prompt)
            putExtra("run_id", runId)
            // Optional; BenchmarkService defaults to 4096 when absent.
            if (maxTokens != null) putExtra("max_tokens", maxTokens)
            if (suppressEarlyEos != null) putExtra("suppress_early_eos", suppressEarlyEos)
        }
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(serviceIntent)
            } else {
                context.startService(serviceIntent)
            }
            Log.d("BROADCAST_RECEIVER", "run_id=$runId service start call returned normally")
        } catch (e: Exception) {
            Log.e(
                "RECEIVER_ERROR",
                "run_id=$runId BenchmarkService start call threw " +
                "${e.javaClass.simpleName}: ${e.message}",
                e
            )
        }
    }
}
