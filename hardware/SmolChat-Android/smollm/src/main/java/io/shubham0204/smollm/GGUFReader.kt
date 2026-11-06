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

package io.shubham0204.smollm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException

class GGUFReader {
    companion object {
        init {
            System.loadLibrary("ggufreader")
        }
    }

    private var nativeHandle: Long = 0L

    /**
     * Reads the GGUF file header via the native reader. Throws immediately if the native handle
     * comes back invalid (e.g. the file couldn't be opened/parsed — a common cause is scoped
     * storage denying raw native file access to non-media files on shared external storage,
     * such as a `.gguf` file under `/sdcard/Download` without MANAGE_EXTERNAL_STORAGE) instead of
     * silently leaving the reader uninitialized for a later call to fail on.
     */
    suspend fun load(modelPath: String) =
        withContext(Dispatchers.IO) {
            nativeHandle = getGGUFContextNativeHandle(modelPath)
            if (nativeHandle == 0L) {
                throw IOException(
                    "Failed to open/parse GGUF file at '$modelPath'. The file may not exist, may " +
                    "not be readable by this app (e.g. a public external-storage path blocked by " +
                    "scoped storage), or may not be a valid GGUF file."
                )
            }
        }

    fun getContextSize(): Long? {
        // check() always throws regardless of JVM/Android assertion settings, unlike assert()
        // which is silently a no-op unless -ea is enabled — with load() now validating the
        // handle up front, this should be unreachable in practice, but stays as a safety net.
        check(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val contextSize = getContextSize(nativeHandle)
        return if (contextSize == -1L) {
            null
        } else {
            contextSize
        }
    }

    fun getChatTemplate(): String? {
        check(nativeHandle != 0L) { "Use GGUFReader.load() to initialize the reader" }
        val chatTemplate = getChatTemplate(nativeHandle)
        return chatTemplate.ifEmpty { null }
    }

    /** Returns the native handle (pointer to gguf_context created on the native side) */
    private external fun getGGUFContextNativeHandle(modelPath: String): Long

    /** Read the context size (in no. of tokens) from the GGUF file, given the native handle */
    private external fun getContextSize(nativeHandle: Long): Long

    /** Read the chat template from the GGUF file, given the native handle */
    private external fun getChatTemplate(nativeHandle: Long): String
}
