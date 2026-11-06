package io.shubham0204.smollmandroid.data

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.runBlocking
import org.koin.core.annotation.Single
import java.util.Date

/**
 * Adds the inference-metrics columns to ChatMessage (all nullable, so existing rows simply get
 * NULL and require no backfill) so per-message metrics can be persisted and shown again after
 * reopening the app, instead of only living in transient ViewModel state.
 */
val MIGRATION_1_2 = object : Migration(1, 2) {
    override fun migrate(db: SupportSQLiteDatabase) {
        db.execSQL("ALTER TABLE ChatMessage ADD COLUMN ttftMs INTEGER")
        db.execSQL("ALTER TABLE ChatMessage ADD COLUMN decodeTps REAL")
        db.execSQL("ALTER TABLE ChatMessage ADD COLUMN peakRssKb INTEGER")
        db.execSQL("ALTER TABLE ChatMessage ADD COLUMN coldLoadTimeMs INTEGER")
        db.execSQL("ALTER TABLE ChatMessage ADD COLUMN avgCurrentUa INTEGER")
        db.execSQL("ALTER TABLE ChatMessage ADD COLUMN thermalStatus TEXT")
    }
}

@Database(
    entities = [Chat::class, ChatMessage::class, LLMModel::class, Task::class, Folder::class],
    version = 2,
)
@TypeConverters(Converters::class)
abstract class AppRoomDatabase : RoomDatabase() {
    abstract fun chatsDao(): ChatsDao

    abstract fun chatMessagesDao(): ChatMessageDao

    abstract fun llmModelDao(): LLMModelDao

    abstract fun taskDao(): TaskDao

    abstract fun folderDao(): FolderDao
}

@Single
class AppDB(context: Context) {
    private val db =
        Room.databaseBuilder(context, AppRoomDatabase::class.java, "app-database")
            .addMigrations(MIGRATION_1_2)
            .build()

    /** Get all chats from the database sorted by dateUsed in descending order. */
    fun getChats(): Flow<List<Chat>> = db.chatsDao().getChats()

    /**
     * Reactive observer for a single chat by id — emits whenever that row (or any row in the
     * Chat table) changes, regardless of which component wrote it (e.g. a headless benchmark
     * run's AppDB.updateChat() call from BenchmarkService, not just this app's own ViewModels).
     */
    fun getChatFlow(chatId: Long): Flow<Chat?> = db.chatsDao().getChatFlow(chatId)

    fun loadDefaultChat(): Chat {
        val defaultChat =
            if (getChatsCount() == 0L) {
                addChat("Untitled")
                getRecentlyUsedChat()!!
            } else {
                // Given that chatsDB has at least one chat
                // chatsDB.getRecentlyUsedChat() will never return null
                getRecentlyUsedChat()!!
            }
        return defaultChat
    }

    /**
     * Get the most recently used chat from the database. This function might return null, if there
     * are no chats in the database.
     */
    fun getRecentlyUsedChat(): Chat? =
        runBlocking(Dispatchers.IO) { db.chatsDao().getRecentlyUsedChat() }

    /**
     * Adds a new chat to the database initialized with given arguments and returns the new Chat
     * object
     */
    fun addChat(
        chatName: String,
        chatTemplate: String = "",
        systemPrompt: String = "You are a helpful assistant.",
        llmModelId: Long = -1,
        isTask: Boolean = false,
    ): Chat =
        runBlocking(Dispatchers.IO) {
            val newChat =
                Chat(
                    name = chatName,
                    systemPrompt = systemPrompt,
                    dateCreated = Date(),
                    dateUsed = Date(),
                    llmModelId = llmModelId,
                    contextSize = 2048,
                    chatTemplate = chatTemplate,
                    isTask = isTask,
                )
            val newChatId = db.chatsDao().insertChat(newChat)
            newChat.copy(id = newChatId)
        }

    /** Update the chat in the database. */
    fun updateChat(modifiedChat: Chat) =
        runBlocking(Dispatchers.IO) { db.chatsDao().updateChat(modifiedChat) }

    fun deleteChat(chat: Chat) = runBlocking(Dispatchers.IO) { db.chatsDao().deleteChat(chat.id) }

    fun getChatsCount(): Long = runBlocking(Dispatchers.IO) { db.chatsDao().getChatsCount() }

    fun getChatsForFolder(folderId: Long): Flow<List<Chat>> =
        db.chatsDao().getChatsForFolder(folderId)

    // Chat Messages

    fun getMessages(chatId: Long): Flow<List<ChatMessage>> =
        db.chatMessagesDao().getMessages(chatId)

    fun getMessagesForModel(chatId: Long): List<ChatMessage> =
        runBlocking(Dispatchers.IO) { db.chatMessagesDao().getMessagesForModel(chatId) }

    fun addUserMessage(chatId: Long, message: String) =
        runBlocking(Dispatchers.IO) {
            db.chatMessagesDao()
                .insertMessage(
                    ChatMessage(chatId = chatId, message = message, isUserMessage = true)
                )
        }

    /** Inserts the assistant's message and returns its generated row id. */
    fun addAssistantMessage(chatId: Long, message: String): Long =
        runBlocking(Dispatchers.IO) {
            db.chatMessagesDao()
                .insertMessage(
                    ChatMessage(chatId = chatId, message = message, isUserMessage = false)
                )
        }

    /** Attaches inference metrics to an already-inserted assistant message. */
    fun updateMessageMetrics(
        messageId: Long,
        ttftMs: Long?,
        decodeTps: Float?,
        peakRssKb: Long?,
        coldLoadTimeMs: Long?,
        avgCurrentUa: Long?,
        thermalStatus: String?,
    ) = runBlocking(Dispatchers.IO) {
        db.chatMessagesDao().updateMessageMetrics(
            messageId, ttftMs, decodeTps, peakRssKb, coldLoadTimeMs, avgCurrentUa, thermalStatus
        )
    }

    fun deleteMessage(messageId: Long) =
        runBlocking(Dispatchers.IO) { db.chatMessagesDao().deleteMessage(messageId) }

    fun deleteMessages(chatId: Long) =
        runBlocking(Dispatchers.IO) { db.chatMessagesDao().deleteMessages(chatId) }

    // Models

    /** Inserts a new model row and returns its generated id. */
    fun addModel(name: String, url: String, path: String, contextSize: Int, chatTemplate: String): Long =
        runBlocking(Dispatchers.IO) {
            db.llmModelDao()
                .insertModels(
                    LLMModel(
                        name = name,
                        url = url,
                        path = path,
                        contextSize = contextSize,
                        chatTemplate = chatTemplate,
                    )
                )
                .first()
        }

    fun getModel(id: Long): LLMModel? = runBlocking(Dispatchers.IO) { db.llmModelDao().getModel(id) }

    fun getModels(): Flow<List<LLMModel>> =
        runBlocking(Dispatchers.IO) { db.llmModelDao().getAllModels() }

    fun getModelsList(): List<LLMModel> =
        runBlocking(Dispatchers.IO) { db.llmModelDao().getAllModelsList() }

    fun deleteModel(id: Long) = runBlocking(Dispatchers.IO) { db.llmModelDao().deleteModel(id) }

    // Tasks

    fun getTask(taskId: Long): Task = runBlocking(Dispatchers.IO) { db.taskDao().getTask(taskId) }

    fun getTasks(): Flow<List<Task>> = db.taskDao().getTasks()

    fun addTask(name: String, systemPrompt: String, modelId: Long) =
        runBlocking(Dispatchers.IO) {
            db.taskDao()
                .insertTask(Task(name = name, systemPrompt = systemPrompt, modelId = modelId))
        }

    fun deleteTask(taskId: Long) = runBlocking(Dispatchers.IO) { db.taskDao().deleteTask(taskId) }

    fun updateTask(task: Task) = runBlocking(Dispatchers.IO) { db.taskDao().updateTask(task) }

    // Folders

    fun getFolders(): Flow<List<Folder>> = db.folderDao().getFolders()

    fun addFolder(folderName: String) =
        runBlocking(Dispatchers.IO) { db.folderDao().insertFolder(Folder(name = folderName)) }

    fun updateFolder(folder: Folder) =
        runBlocking(Dispatchers.IO) { db.folderDao().updateFolder(folder) }

    /** Deletes the folder from the Folder table only */
    fun deleteFolder(folderId: Long) =
        runBlocking(Dispatchers.IO) {
            db.folderDao().deleteFolder(folderId)
            db.chatsDao().updateFolderIds(folderId, -1L)
        }

    /** Deletes the folder from the Folder table and corresponding chats from the Chat table */
    fun deleteFolderWithChats(folderId: Long) =
        runBlocking(Dispatchers.IO) {
            db.folderDao().deleteFolder(folderId)
            db.chatsDao().deleteChatsInFolder(folderId)
        }
}
