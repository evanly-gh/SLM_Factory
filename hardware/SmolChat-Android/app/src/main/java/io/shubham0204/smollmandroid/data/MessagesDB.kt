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

package io.shubham0204.smollmandroid.data

import android.text.Spanned
import androidx.compose.runtime.Stable
import androidx.core.text.toSpanned
import androidx.room.Dao
import androidx.room.Entity
import androidx.room.Ignore
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "ChatMessage")
@Stable
data class ChatMessage(
    @PrimaryKey(autoGenerate = true) var id: Long = 0,
    var chatId: Long = 0,
    var message: String = "",
    var isUserMessage: Boolean = false,
    // Inference metrics, populated only for assistant messages (null for user messages and for
    // any assistant messages inserted before this column set existed). Mirrors the transient
    // InferenceMetrics computed in ChatScreenViewModel/BenchmarkService, persisted so the metrics
    // panel can be shown per-message, including after reopening the app.
    var ttftMs: Long? = null,
    var decodeTps: Float? = null,
    var peakRssKb: Long? = null,
    var coldLoadTimeMs: Long? = null,
    var avgCurrentUa: Long? = null,
    var thermalStatus: String? = null,
    @Ignore
    var renderedMessage: Spanned = "".toSpanned()
)

@Dao
interface ChatMessageDao {
    @Query("SELECT * FROM ChatMessage WHERE chatId = :chatId")
    fun getMessages(chatId: Long): Flow<List<ChatMessage>>

    @Query("SELECT * FROM ChatMessage WHERE chatId = :chatId")
    suspend fun getMessagesForModel(chatId: Long): List<ChatMessage>

    @Insert
    suspend fun insertMessage(message: ChatMessage): Long

    @Query(
        "UPDATE ChatMessage SET ttftMs = :ttftMs, decodeTps = :decodeTps, " +
        "peakRssKb = :peakRssKb, coldLoadTimeMs = :coldLoadTimeMs, " +
        "avgCurrentUa = :avgCurrentUa, thermalStatus = :thermalStatus WHERE id = :messageId"
    )
    suspend fun updateMessageMetrics(
        messageId: Long,
        ttftMs: Long?,
        decodeTps: Float?,
        peakRssKb: Long?,
        coldLoadTimeMs: Long?,
        avgCurrentUa: Long?,
        thermalStatus: String?,
    )

    @Query("DELETE FROM ChatMessage WHERE chatId = :chatId")
    suspend fun deleteMessages(chatId: Long)

    @Query("DELETE FROM ChatMessage WHERE id = :messageId")
    suspend fun deleteMessage(messageId: Long)
}
