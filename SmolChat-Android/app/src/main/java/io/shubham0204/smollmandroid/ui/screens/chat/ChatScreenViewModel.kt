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

package io.shubham0204.smollmandroid.ui.screens.chat

import android.annotation.SuppressLint
import android.app.ActivityManager
import android.app.ActivityManager.MemoryInfo
import android.content.Context
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.text.Spanned
import android.util.Log
import android.widget.Toast
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import io.shubham0204.smollm.SmolLM
import io.shubham0204.smollmandroid.R
import io.shubham0204.smollmandroid.data.AppDB
import io.shubham0204.smollmandroid.data.Chat
import io.shubham0204.smollmandroid.data.ChatMessage
import io.shubham0204.smollmandroid.data.Folder
import io.shubham0204.smollmandroid.data.LLMModel
import io.shubham0204.smollmandroid.data.SharedPrefStore
import io.shubham0204.smollmandroid.data.SystemPrompt
import io.shubham0204.smollmandroid.data.SystemPromptsStore
import io.shubham0204.smollmandroid.data.Task
import io.shubham0204.smollmandroid.llm.ModelsRepository
import io.shubham0204.smollmandroid.llm.SmolLMManager
import io.shubham0204.smollmandroid.llm.readCpuThermalZoneTempC
import io.shubham0204.smollmandroid.llm.readSkinThermalTempC
import io.shubham0204.smollmandroid.llm.speech2text.AudioTranscriptionService
import io.shubham0204.smollmandroid.ui.components.createAlertDialog
import io.shubham0204.smollmandroid.ui.screens.manage_asr.SETTING_DEF_VALUE_SPEECH2TEXT_CURR_MODEL_NAME
import io.shubham0204.smollmandroid.ui.screens.manage_asr.SETTING_DEF_VALUE_SPEECH2_TEXT_ENABLED
import io.shubham0204.smollmandroid.ui.screens.manage_asr.SETTING_KEY_SPEECH2TEXT_CURR_MODEL_NAME
import io.shubham0204.smollmandroid.ui.screens.manage_asr.SETTING_KEY_SPEECH2TEXT_ENABLED
import io.shubham0204.smollmandroid.ui.screens.manage_asr.availableASRModels
import kotlinx.collections.immutable.ImmutableList
import kotlinx.collections.immutable.toImmutableList
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.koin.android.annotation.KoinViewModel
import java.io.File
import java.util.Collections
import java.util.Date
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.pow

private const val LOGTAG = "[SmolLMAndroid-Kt]"
private const val METRICS_LOGTAG = "[InferenceMetrics]"
private val LOGD: (String) -> Unit = { Log.d(LOGTAG, it) }

private val findThinkHtmlBlockRegex = Regex("<blockquote><i><h6>[\\s\\S]*?</i></h6></blockquote>")
internal fun String.stripThinkingForClipboard() = findThinkHtmlBlockRegex.replace(this, "").trim()

data class InferenceMetrics(
    val ttftMs: Long,
    val decodeTps: Float,
    val peakRssKb: Long,
    val coldLoadTimeMs: Long?,   // null = model was already warm, show nothing
    val avgCurrentUa: Long,      // Long.MIN_VALUE = unsupported hardware
    val thermalThrottled: Boolean,
    val maxCpuThermalTempC: Float?,  // null = no CPU thermal_zone path was readable on this device
    val maxSkinThermalTempC: Float?, // null = skin-therm-usr zone not found/readable
)

sealed class ChatScreenUIEvent {
    sealed class ChatEvents {
        data class UpdateChatModel(val model: LLMModel) : ChatScreenUIEvent()

        data object LoadChatModel : ChatScreenUIEvent()

        data class DeleteModel(val model: LLMModel) : ChatScreenUIEvent()

        data class SendUserQuery(val query: String) : ChatScreenUIEvent()

        data object StopGeneration : ChatScreenUIEvent()

        data class OnTaskSelected(val task: Task) : ChatScreenUIEvent()

        data class OnMessageEdited(
            val chatId: Long,
            val oldMessage: ChatMessage,
            val lastMessage: ChatMessage,
            val newMessageText: String,
        ) : ChatScreenUIEvent()

        data class OnDeleteChat(val chat: Chat) : ChatScreenUIEvent()

        data class OnDeleteChatMessages(val chat: Chat) : ChatScreenUIEvent()

        data object NewChat : ChatScreenUIEvent()

        data class SwitchChat(val chat: Chat) : ChatScreenUIEvent()

        data class UpdateChatSettings(val settings: EditableChatSettings, val existingChat: Chat) :
            ChatScreenUIEvent()

        data class StartBenchmark(val onResult: (String) -> Unit) : ChatScreenUIEvent()

        data class StartInferenceBenchmark(val onComplete: () -> Unit) : ChatScreenUIEvent()

        data class StartAudioTranscription(val onLineComplete: (String) -> Unit) :
            ChatScreenUIEvent()

        data object StopAudioTranscription : ChatScreenUIEvent()
    }

    sealed class SystemPromptEvents {
        data class AddSystemPrompt(val name: String, val body: String) : ChatScreenUIEvent()

        data class DeleteSystemPrompt(val id: Long) : ChatScreenUIEvent()
    }

    sealed class FolderEvents {
        data class UpdateChatFolder(val newFolderId: Long) : ChatScreenUIEvent()

        data class AddFolder(val folderName: String) : ChatScreenUIEvent()

        data class UpdateFolder(val folder: Folder) : ChatScreenUIEvent()

        data class DeleteFolder(val folderId: Long) : ChatScreenUIEvent()

        data class DeleteFolderWithChats(val folderId: Long) : ChatScreenUIEvent()
    }

    sealed class DialogEvents {
        data class ToggleChangeFolderDialog(val visible: Boolean) : ChatScreenUIEvent()

        data class ToggleSelectModelListDialog(val visible: Boolean) : ChatScreenUIEvent()

        data class ToggleMoreOptionsPopup(val visible: Boolean) : ChatScreenUIEvent()

        data class ToggleTaskListBottomList(val visible: Boolean) : ChatScreenUIEvent()

        data object ToggleRAMUsageLabel : ChatScreenUIEvent()

        data class ShowContextLengthUsageDialog(val chat: Chat) : ChatScreenUIEvent()
    }
}

data class AudioTranscriptionUIState(
    val isRecording: Boolean = false,
    val isAvailable: Boolean = false,
)

data class ChatScreenUIState(
    val chat: Chat = Chat(),
    val isGeneratingResponse: Boolean = false,
    val renderedPartialResponse: Spanned? = null,
    val modelLoadingState: ChatScreenViewModel.ModelLoadingState =
        ChatScreenViewModel.ModelLoadingState.NOT_LOADED,
    val responseGenerationsSpeed: Float? = null,
    val responseGenerationTimeSecs: Int? = null,
    val memoryUsage: Pair<Float, Float>? = null,
    val folders: ImmutableList<Folder> = emptyList<Folder>().toImmutableList(),
    val chats: ImmutableList<Chat> = emptyList<Chat>().toImmutableList(),
    val models: ImmutableList<LLMModel> = emptyList<LLMModel>().toImmutableList(),
    val messages: ImmutableList<ChatMessage> = emptyList<ChatMessage>().toImmutableList(),
    val tasks: ImmutableList<Task> = emptyList<Task>().toImmutableList(),
    val systemPrompts: ImmutableList<SystemPrompt> = emptyList<SystemPrompt>().toImmutableList(),
    val benchmarkResult: String? = null,
    val inferenceMetrics: InferenceMetrics? = null,
    val audioTranscriptionUIState: AudioTranscriptionUIState = AudioTranscriptionUIState(),
    val showChangeFolderDialog: Boolean = false,
    val showSelectModelListDialog: Boolean = false,
    val showMoreOptionsPopup: Boolean = false,
    val showTasksBottomSheet: Boolean = false,
)

@KoinViewModel
class ChatScreenViewModel(
    val context: Context,
    val appDB: AppDB,
    val modelsRepository: ModelsRepository,
    val smolLMManager: SmolLMManager,
    val audioTranscriptionService: AudioTranscriptionService,
    val mdRenderer: MDRenderer,
    val sharedPrefStore: SharedPrefStore,
    val systemPromptsStore: SystemPromptsStore
) : ViewModel() {
    enum class ModelLoadingState {
        NOT_LOADED, // model loading not started
        IN_PROGRESS, // model loading in-progress
        SUCCESS, // model loading finished successfully
        FAILURE, // model loading failed
    }

    private val _uiState = MutableStateFlow(initializeUIState())
    val uiState: StateFlow<ChatScreenUIState> = _uiState

    // Used to pre-set a value in the query text-field of the chat screen
    // It is set when a query comes from a 'share-text' intent in ChatActivity
    var questionTextDefaultVal: String? = null

    // regex to replace <think> tags with <blockquote>
    // to render them correctly in Markdown
    private val findThinkTagRegex = Regex("<think>(.*?)</think>", RegexOption.DOT_MATCHES_ALL)
    private var activityManager: ActivityManager
    private var lastRenderTime = 0L

    init {
        setupCollectors()
        loadModel()
        activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    }

    /**
     * Load the model for the current chat. If chat is configured with a LLM (i.e. chat.llModelId !=
     * -1), then load the model. If not, do nothing — new chats have no model assigned by default
     * and no fallback/inheritance is applied; the header renders blank and MessageInput shows its
     * own "select a model" prompt until one is assigned, either manually via the "Change Model"
     * option or by a headless run targeting this chat. If chat.llmModelId points at a model that
     * no longer exists (e.g. deleted), the select-model dialog is still shown below to recover
     * from that dangling reference — that's a different scenario from a freshly created chat.
     */
    fun loadModel(onComplete: (ModelLoadingState) -> Unit = {}) {
        val chat = _uiState.value.chat
        if (chat.llmModelId == -1L) {
            return
        }
        val model = modelsRepository.getModelFromId(chat.llmModelId)
        if (model == null) {
            _uiState.update { it.copy(showSelectModelListDialog = true) }
            return
        }
        _uiState.update { it.copy(modelLoadingState = ModelLoadingState.IN_PROGRESS) }
        smolLMManager.load(
                chat,
                model.path,
                SmolLM.InferenceParams(
                    chat.minP,
                    chat.temperature,
                    !chat.isTask,
                    chat.contextSize.toLong(),
                    chat.chatTemplate.takeIf { it.isNotBlank() && ("{%" in it || "{{" in it) },
                    chat.nThreads,
                    chat.useMmap,
                    chat.useMlock,
                ),
                onError = { e ->
                    _uiState.update { it.copy(modelLoadingState = ModelLoadingState.FAILURE) }
                    onComplete(ModelLoadingState.FAILURE)
                    createAlertDialog(
                        dialogTitle = context.getString(R.string.dialog_err_title),
                        dialogText = context.getString(R.string.dialog_err_text, e.message),
                        dialogPositiveButtonText =
                            context.getString(R.string.dialog_err_change_model),
                        onPositiveButtonClick = {
                            onEvent(
                                ChatScreenUIEvent.DialogEvents.ToggleSelectModelListDialog(
                                    visible = true
                                )
                            )
                        },
                        dialogNegativeButtonText = context.getString(R.string.dialog_err_close),
                        onNegativeButtonClick = {},
                    )
                },
                onSuccess = {
                    _uiState.update {
                        it.copy(
                            modelLoadingState = ModelLoadingState.SUCCESS,
                            memoryUsage =
                                if (it.memoryUsage != null) {
                                    getCurrentMemoryUsage()
                                } else {
                                    null
                                },
                        )
                    }
                    onComplete(ModelLoadingState.SUCCESS)
                },
            )
    }

    /** Clears the resources occupied by the model only if the inference is not in progress. */
    fun unloadModel(): Boolean =
        if (!smolLMManager.isInferenceOn) {
            smolLMManager.unload()
            _uiState.update { it.copy(modelLoadingState = ModelLoadingState.NOT_LOADED) }
            true
        } else {
            false
        }

    @SuppressLint("StringFormatMatches")
    fun onEvent(event: ChatScreenUIEvent) {
        when (event) {
            is ChatScreenUIEvent.DialogEvents.ToggleSelectModelListDialog -> {
                _uiState.update { it.copy(showSelectModelListDialog = event.visible) }
            }

            is ChatScreenUIEvent.DialogEvents.ToggleMoreOptionsPopup -> {
                _uiState.update { it.copy(showMoreOptionsPopup = event.visible) }
            }

            is ChatScreenUIEvent.DialogEvents.ToggleTaskListBottomList -> {
                _uiState.update { it.copy(showTasksBottomSheet = event.visible) }
            }

            is ChatScreenUIEvent.DialogEvents.ToggleChangeFolderDialog -> {
                _uiState.update { it.copy(showChangeFolderDialog = event.visible) }
            }

            ChatScreenUIEvent.DialogEvents.ToggleRAMUsageLabel -> {
                _uiState.update {
                    it.copy(
                        memoryUsage =
                            if (it.memoryUsage != null) {
                                null
                            } else {
                                getCurrentMemoryUsage()
                            }
                    )
                }
            }

            is ChatScreenUIEvent.DialogEvents.ShowContextLengthUsageDialog -> {
                createAlertDialog(
                    dialogTitle = context.getString(R.string.dialog_ctx_usage_title),
                    dialogText =
                        context.getString(
                            R.string.dialog_ctx_usage_text,
                            event.chat.contextSizeConsumed,
                            event.chat.contextSize,
                        ),
                    dialogPositiveButtonText = context.getString(R.string.dialog_ctx_usage_close),
                    onPositiveButtonClick = {},
                    dialogNegativeButtonText = null,
                    onNegativeButtonClick = null,
                )
            }

            is ChatScreenUIEvent.FolderEvents.UpdateChatFolder -> {
                appDB.updateChat(_uiState.value.chat.copy(folderId = event.newFolderId))
            }

            is ChatScreenUIEvent.FolderEvents.AddFolder -> {
                appDB.addFolder(event.folderName)
            }

            is ChatScreenUIEvent.FolderEvents.UpdateFolder -> {
                appDB.updateFolder(event.folder)
            }

            is ChatScreenUIEvent.FolderEvents.DeleteFolder -> {
                appDB.deleteFolder(event.folderId)
            }

            is ChatScreenUIEvent.FolderEvents.DeleteFolderWithChats -> {
                appDB.deleteFolderWithChats(event.folderId)
            }

            is ChatScreenUIEvent.SystemPromptEvents.AddSystemPrompt -> {
                systemPromptsStore.addPrompt(event.name, event.body)
            }

            is ChatScreenUIEvent.SystemPromptEvents.DeleteSystemPrompt -> {
                systemPromptsStore.deletePrompt(event.id)
            }

            is ChatScreenUIEvent.ChatEvents.UpdateChatModel -> {
                updateChatLLMParams(event.model.id, event.model.chatTemplate)
                loadModel()
                onEvent(ChatScreenUIEvent.DialogEvents.ToggleSelectModelListDialog(visible = false))
            }

            is ChatScreenUIEvent.ChatEvents.DeleteModel -> {
                deleteModel(event.model.id)
                Toast.makeText(
                    context,
                    context.getString(R.string.chat_model_deleted, event.model.name),
                    Toast.LENGTH_LONG,
                )
                    .show()
            }

            ChatScreenUIEvent.ChatEvents.LoadChatModel -> {}

            is ChatScreenUIEvent.ChatEvents.SendUserQuery -> {
                sendUserQuery(event.query)
            }

            ChatScreenUIEvent.ChatEvents.StopGeneration -> {
                stopGeneration()
            }

            is ChatScreenUIEvent.ChatEvents.OnTaskSelected -> {
                // Using parameters from the `task`
                // create a `Chat` instance and switch to it
                modelsRepository.getModelFromId(event.task.modelId)?.let { model ->
                    val newTask =
                        appDB.addChat(
                            chatName = event.task.name,
                            chatTemplate = model.chatTemplate,
                            systemPrompt = event.task.systemPrompt,
                            llmModelId = event.task.modelId,
                            isTask = true,
                        )
                    switchChat(newTask)
                    onEvent(
                        ChatScreenUIEvent.DialogEvents.ToggleTaskListBottomList(visible = false)
                    )
                }
            }

            is ChatScreenUIEvent.ChatEvents.OnMessageEdited -> {
                // viewModel.sendUserQuery will add a new message to the chat
                // hence we delete the old message and the corresponding LLM
                // response if there exists one
                // TODO: There should be no need to unload/load the model again
                //       as only the conversation messages have changed.
                //       Currently there's no native function to edit the conversation
                // messages
                //       so unload (remove all messages) and load (add all messages) the
                // model.
                deleteMessage(event.oldMessage.id)
                if (!event.lastMessage.isUserMessage) {
                    deleteMessage(event.lastMessage.id)
                }
                appDB.addUserMessage(event.chatId, event.newMessageText)
                unloadModel()
                loadModel(
                    onComplete = {
                        if (it == ModelLoadingState.SUCCESS) {
                            sendUserQuery(event.newMessageText, addMessageToDB = false)
                        }
                    }
                )
            }

            is ChatScreenUIEvent.ChatEvents.OnDeleteChat -> {
                createAlertDialog(
                    dialogTitle = context.getString(R.string.dialog_title_delete_chat),
                    dialogText =
                        context.getString(R.string.dialog_text_delete_chat, event.chat.name),
                    dialogPositiveButtonText = context.getString(R.string.dialog_pos_delete),
                    dialogNegativeButtonText = context.getString(R.string.dialog_neg_cancel),
                    onPositiveButtonClick = {
                        deleteChat(event.chat)
                        Toast.makeText(
                            context,
                            "Chat '${event.chat.name}' deleted",
                            Toast.LENGTH_LONG,
                        )
                            .show()
                    },
                    onNegativeButtonClick = {},
                )
            }

            is ChatScreenUIEvent.ChatEvents.OnDeleteChatMessages -> {
                createAlertDialog(
                    dialogTitle = context.getString(R.string.chat_options_clear_messages),
                    dialogText = context.getString(R.string.chat_options_clear_messages_text),
                    dialogPositiveButtonText = context.getString(R.string.dialog_pos_clear),
                    dialogNegativeButtonText = context.getString(R.string.dialog_neg_cancel),
                    onPositiveButtonClick = {
                        deleteChatMessages(event.chat)
                        unloadModel()
                        loadModel(
                            onComplete = {
                                if (it == ModelLoadingState.SUCCESS) {
                                    Toast.makeText(
                                        context,
                                        "Chat '${event.chat.name}' cleared",
                                        Toast.LENGTH_LONG,
                                    )
                                        .show()
                                }
                            }
                        )
                    },
                    onNegativeButtonClick = {},
                )
            }

            ChatScreenUIEvent.ChatEvents.NewChat -> {
                val chatCount = appDB.getChatsCount()
                val newChat = appDB.addChat(chatName = "Untitled ${chatCount + 1}")
                switchChat(newChat)
            }

            is ChatScreenUIEvent.ChatEvents.SwitchChat -> {
                switchChat(event.chat)
            }

            is ChatScreenUIEvent.ChatEvents.UpdateChatSettings -> {
                val newChat = event.settings.toChat(event.existingChat)
                _uiState.update { it.copy(chat = newChat) }
                appDB.updateChat(newChat)
                unloadModel()
                loadModel()
            }

            is ChatScreenUIEvent.ChatEvents.StartBenchmark -> {
                smolLMManager.benchmark { result -> event.onResult(result) }
            }

            is ChatScreenUIEvent.ChatEvents.StartInferenceBenchmark -> {
                runInferenceBenchmark(event.onComplete)
            }

            is ChatScreenUIEvent.ChatEvents.StartAudioTranscription -> {
                _uiState.update {
                    it.copy(
                        audioTranscriptionUIState = AudioTranscriptionUIState(
                            true,
                            isAvailable = true
                        )
                    )
                }
                val asrModelName = sharedPrefStore.get(
                    SETTING_KEY_SPEECH2TEXT_CURR_MODEL_NAME,
                    SETTING_DEF_VALUE_SPEECH2TEXT_CURR_MODEL_NAME
                )
                val asrModel = availableASRModels.first {
                    it.name == asrModelName
                }
                val error =
                    audioTranscriptionService.startTranscription(asrModel) { transcription ->
                    _uiState.update {
                        it.copy(
                            audioTranscriptionUIState = AudioTranscriptionUIState(
                                false,
                                isAvailable = true
                            )
                        )
                    }
                    event.onLineComplete(transcription)
                }
                if (error is AudioTranscriptionService.Error.AudioRecordingPermissionNotGranted) {

                }
            }

            is ChatScreenUIEvent.ChatEvents.StopAudioTranscription -> {
                _uiState.update {
                    it.copy(
                        audioTranscriptionUIState = AudioTranscriptionUIState(
                            false,
                            isAvailable = true
                        )
                    )
                }
                audioTranscriptionService.stopTranscription()
            }
        }
    }

    private fun initializeUIState(): ChatScreenUIState {
        val defaultChat = appDB.loadDefaultChat()

        val isSpeech2TextEnabled = sharedPrefStore.get(
            SETTING_KEY_SPEECH2TEXT_ENABLED,
            SETTING_DEF_VALUE_SPEECH2_TEXT_ENABLED
        )
        val audioTranscriptionUIState = AudioTranscriptionUIState(
            isAvailable = isSpeech2TextEnabled
        )

        return ChatScreenUIState(
            chat = defaultChat,
            audioTranscriptionUIState = audioTranscriptionUIState
        )
    }

    private fun setupCollectors() {
        viewModelScope.launch {
            launch {
                appDB.getChats().collect { chats ->
                    _uiState.update { it.copy(chats = chats.toImmutableList()) }
                }
            }
            launch {
                appDB.getFolders().collect { folders ->
                    _uiState.update { it.copy(folders = folders.toImmutableList()) }
                }
            }
            launch {
                appDB.getTasks().collect { tasks ->
                    _uiState.update {
                        it.copy(
                            tasks =
                                tasks
                                    .map { task ->
                                        task.copy(
                                            modelName =
                                                modelsRepository.getModelFromId(task.modelId)?.name ?: ""
                                        )
                                    }
                                    .toImmutableList()
                        )
                    }
                }
            }
            launch {
                appDB.getModels().collect { models ->
                    _uiState.update { it.copy(models = models.toImmutableList()) }
                }
            }
            launch {
                systemPromptsStore.systemPrompts.collect { prompts ->
                    _uiState.update { it.copy(systemPrompts = prompts.toImmutableList()) }
                }
            }
            launch {
                _uiState
                    .map { it.chat }
                    .distinctUntilChanged()
                    .collectLatest { chat ->
                        appDB.getMessages(chat.id).collect { chatMessages ->
                            _uiState.update {
                                it.copy(
                                    messages =
                                        chatMessages
                                            .map { chatMessage ->
                                                chatMessage.renderedMessage =
                                                    mdRenderer.render(chatMessage.message)
                                                chatMessage
                                            }
                                            .toImmutableList()
                                )
                            }
                        }
                    }
            }
            launch {
                // Reactive sync: re-fetch the currently open chat's row whenever it changes in
                // the DB, regardless of which component wrote it — e.g. BenchmarkService calling
                // AppDB.updateChat() from a headless run, not just this ViewModel's own writes.
                // Room's Flow auto-invalidates on any write to the Chat table. Without this, the
                // header only ever reflected llmModelId changes that this exact ViewModel
                // instance itself triggered (switchChat/UpdateChatModel/etc.) — an external write
                // never reached _uiState until the user left and re-entered the screen, which
                // recreated the ViewModel and re-read the row fresh.
                _uiState
                    .map { it.chat.id }
                    .distinctUntilChanged()
                    .flatMapLatest { chatId -> appDB.getChatFlow(chatId) }
                    .filterNotNull()
                    .collect { freshChat ->
                        _uiState.update { uiState ->
                            // Preserve the transient llmModel lookup across the swap to avoid a
                            // one-frame flicker; the collector below re-resolves it if llmModelId
                            // actually changed, since that makes the merged chat object differ.
                            uiState.copy(chat = freshChat.copy(llmModel = uiState.chat.llmModel))
                        }
                    }
            }
            launch {
                _uiState
                    .map { it.chat }
                    .distinctUntilChanged()
                    .collectLatest { chat ->
                        _uiState.update { uiState ->
                            uiState.copy(
                                chat =
                                    uiState.chat.copy(
                                        llmModel =
                                            modelsRepository.getModelFromId(uiState.chat.llmModelId)
                                    )
                            )
                        }
                    }
            }
            launch {
                sharedPrefStore.sharedPrefStoreChanges.collect { prefKey ->
                    if (prefKey == SETTING_KEY_SPEECH2TEXT_ENABLED) {
                        audioTranscriptionService.stopTranscription()
                        val isSpeech2TextEnabled = sharedPrefStore.get(
                            SETTING_KEY_SPEECH2TEXT_ENABLED,
                            SETTING_DEF_VALUE_SPEECH2_TEXT_ENABLED
                        )
                        _uiState.update {
                            it.copy(
                                audioTranscriptionUIState = AudioTranscriptionUIState(
                                    isAvailable = isSpeech2TextEnabled,
                                    isRecording = false
                                )
                            )
                        }
                    } else if (prefKey == SETTING_KEY_SPEECH2TEXT_CURR_MODEL_NAME) {
                        audioTranscriptionService.stopTranscription()
                        _uiState.update {
                            it.copy(
                                audioTranscriptionUIState = AudioTranscriptionUIState(
                                    isAvailable = true,
                                    isRecording = false
                                )
                            )
                        }
                    }
                }
            }
        }
    }

    private fun updateChatLLMParams(modelId: Long, chatTemplate: String) {
        val newChat = _uiState.value.chat.copy(llmModelId = modelId, chatTemplate = chatTemplate)
        _uiState.update { it.copy(chat = newChat) }
        appDB.updateChat(newChat)
    }

    private fun deleteMessage(messageId: Long) {
        appDB.deleteMessage(messageId)
    }

    private fun sendUserQuery(query: String, addMessageToDB: Boolean = true) {
        val chat = uiState.value.chat
        // Update the 'dateUsed' attribute of the current Chat instance
        // when a query is sent by the user
        chat.dateUsed = Date()
        appDB.updateChat(chat)

        if (chat.isTask) {
            // If the chat is a 'task', delete all existing messages
            // to maintain the 'stateless' nature of the task
            appDB.deleteMessages(chat.id)
        }

        if (addMessageToDB) {
            appDB.addUserMessage(chat.id, query)
        }
        _uiState.update { it.copy(isGeneratingResponse = true, renderedPartialResponse = null) }
        lastRenderTime = 0L

        // Mirrors BenchmarkService's headless context isolation exactly (same shared helper —
        // see SmolLMManager.loadIsolatedSingleTurn): reload this chat's model with isTask=true so
        // no prior DB history is replayed and the native chat state starts empty, making this a
        // genuinely single-turn call to the model — while the visible chat log/DB below still
        // gets the full multi-turn message list appended exactly as before.
        val modelPath = smolLMManager.currentModelPath
            ?: modelsRepository.getModelFromId(chat.llmModelId)?.path
        if (modelPath == null) {
            _uiState.update { it.copy(isGeneratingResponse = false) }
            createAlertDialog(
                dialogTitle = "An error occurred",
                dialogText = "No model is loaded for this chat.",
                dialogPositiveButtonText = "Change model",
                onPositiveButtonClick = {},
                dialogNegativeButtonText = "",
                onNegativeButtonClick = {},
            )
            return
        }

        smolLMManager.loadIsolatedSingleTurn(
            chat      = chat,
            modelPath = modelPath,
            onError   = { e ->
                _uiState.update { it.copy(isGeneratingResponse = false) }
                createAlertDialog(
                    dialogTitle = "An error occurred",
                    dialogText =
                        "The app is unable to process the query. The error message is: ${e.message}",
                    dialogPositiveButtonText = "Change model",
                    onPositiveButtonClick = {},
                    dialogNegativeButtonText = "",
                    onNegativeButtonClick = {},
                )
            },
            onSuccess = {
                val currentSamples = Collections.synchronizedList(mutableListOf<Long>())
                val thermalThrottled = AtomicBoolean(false)
                val maxCpuThermalTempC = AtomicReference(0f)
                val maxSkinThermalTempC = AtomicReference(0f)
                val (monitorJob, chargeAtStartUah) = startPowerThermalMonitoring(
                    currentSamples, thermalThrottled, maxCpuThermalTempC, maxSkinThermalTempC
                )
                val promptDispatchTimeMs = System.currentTimeMillis()

                smolLMManager.getResponse(
                    query,
                    promptDispatchTimeMs = promptDispatchTimeMs,
                    responseTransform = {
                        // Replace <think> tags with <blockquote> tags
                        // to get a neat Markdown rendering
                        findThinkTagRegex.replace(it) { matchResult ->
                            "<blockquote><i><h6>${matchResult.groupValues[1].trim()}</i></h6></blockquote>"
                        }
                    },
                    onPartialResponseGenerated = { resp ->
                        val currentTime = System.currentTimeMillis()
                        if (currentTime - lastRenderTime > 100) {
                            _uiState.update { it.copy(renderedPartialResponse = mdRenderer.render(resp)) }
                            lastRenderTime = currentTime
                        }
                    },
                    onSuccess = { response ->
                        monitorJob.cancel()
                        val metrics = buildInferenceMetrics(response, currentSamples, thermalThrottled, chargeAtStartUah, maxCpuThermalTempC, maxSkinThermalTempC)
                        logMetrics(generateManualRunId(response.savedMessageId), metrics, response.response)
                        persistMessageMetrics(response.savedMessageId, metrics)
                        val updatedChat = chat.copy(contextSizeConsumed = response.contextLengthUsed)
                        _uiState.update {
                            it.copy(
                                chat = updatedChat,
                                isGeneratingResponse = false,
                                responseGenerationsSpeed = response.generationSpeed,
                                responseGenerationTimeSecs = response.generationTimeSecs,
                                inferenceMetrics = metrics,
                                memoryUsage =
                                    if (it.memoryUsage != null) {
                                        getCurrentMemoryUsage()
                                    } else {
                                        null
                                    },
                            )
                        }
                        appDB.updateChat(updatedChat)
                        if (!response.usedJinjaTemplate) {
                            Toast.makeText(
                                context,
                                "Model's Jinja chat template not fully supported, using legacy renderer",
                                Toast.LENGTH_LONG,
                            ).show()
                        }
                    },
                    onCancelled = {
                        monitorJob.cancel()
                        // ignore CancellationException, as it was called because
                        // `responseGenerationJob` was cancelled in the `stopGeneration` method
                    },
                    onError = { exception ->
                        monitorJob.cancel()
                        _uiState.update { it.copy(isGeneratingResponse = false) }
                        createAlertDialog(
                            dialogTitle = "An error occurred",
                            dialogText =
                                "The app is unable to process the query. The error message is: ${exception.message}",
                            dialogPositiveButtonText = "Change model",
                            onPositiveButtonClick = {},
                            dialogNegativeButtonText = "",
                            onNegativeButtonClick = {},
                        )
                    },
                )
            },
        )
    }

    private fun runInferenceBenchmark(onComplete: () -> Unit) {
        if (!smolLMManager.isInstanceLoaded.get()) {
            onComplete()
            return
        }
        val benchPrompt = "Explain the process of photosynthesis in detail."
        _uiState.update { it.copy(isGeneratingResponse = true) }

        val currentSamples = Collections.synchronizedList(mutableListOf<Long>())
        val thermalThrottled = AtomicBoolean(false)
        val maxCpuThermalTempC = AtomicReference(0f)
        val maxSkinThermalTempC = AtomicReference(0f)
        val (monitorJob, chargeAtStartUah) = startPowerThermalMonitoring(
            currentSamples, thermalThrottled, maxCpuThermalTempC, maxSkinThermalTempC
        )
        val promptDispatchTimeMs = System.currentTimeMillis()

        smolLMManager.getResponse(
            query = benchPrompt,
            promptDispatchTimeMs = promptDispatchTimeMs,
            responseTransform = { it },
            onPartialResponseGenerated = {},
            saveToDb = false,
            onSuccess = { response ->
                monitorJob.cancel()
                val metrics = buildInferenceMetrics(response, currentSamples, thermalThrottled, chargeAtStartUah, maxCpuThermalTempC, maxSkinThermalTempC)
                logMetrics(generateManualRunId(response.savedMessageId), metrics, response.response)
                _uiState.update { it.copy(isGeneratingResponse = false, inferenceMetrics = metrics) }
                onComplete()
            },
            onCancelled = {
                monitorJob.cancel()
                _uiState.update { it.copy(isGeneratingResponse = false) }
                onComplete()
            },
            onError = {
                monitorJob.cancel()
                _uiState.update { it.copy(isGeneratingResponse = false) }
                onComplete()
            },
        )
    }

    /**
     * Starts polling power current, PowerManager thermal status, and (every ~10th tick, i.e.
     * roughly once per second) real hardware temperature from two thermal_zone sysfs sensors —
     * the coarse PowerManager status is known to lag actual thermal throttling on some devices,
     * so this tracks real numbers alongside it: the raw CPU junction sensor
     * (readCpuThermalZoneTempC) and the skin-therm-usr sensor (readSkinThermalTempC), which OEM
     * throttling policies more often key off. Both max*ThermalTempC refs are left at their
     * initial 0f (read back as "unavailable") if the corresponding sensor is never readable.
     * Returns (monitorJob, chargeCounterAtStartUah) — the charge counter snapshot is used as
     * a last-resort fallback to estimate average current when direct readings are unavailable.
     * chargeCounterAtStartUah == Long.MIN_VALUE means the device doesn't support the counter API.
     */
    private fun startPowerThermalMonitoring(
        currentSamples: MutableList<Long>,
        thermalThrottled: AtomicBoolean,
        maxCpuThermalTempC: AtomicReference<Float>,
        maxSkinThermalTempC: AtomicReference<Float>,
    ): Pair<Job, Long> {
        val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val chargeAtStart = batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
        val job = viewModelScope.launch(Dispatchers.IO) {
            var tick = 0
            while (true) {
                val sample = readCurrentUa(batteryManager)
                if (sample > 0L) currentSamples.add(sample)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    if (powerManager.currentThermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) {
                        thermalThrottled.set(true)
                    }
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
        return Pair(job, chargeAtStart)
    }

    /**
     * Reads instantaneous current draw in µA using a priority fallback chain:
     * 1. BatteryManager.BATTERY_PROPERTY_CURRENT_NOW (standard API)
     * 2. /sys/class/power_supply/battery/current_now (sysfs, common on Qualcomm devices)
     * Returns Long.MIN_VALUE if no method works on this device.
     */
    private fun readCurrentUa(batteryManager: BatteryManager): Long {
        val apiVal = batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        if (apiVal != Long.MIN_VALUE) return abs(apiVal)

        for (path in CURRENT_SYSFS_PATHS) {
            try {
                val raw = File(path).readText().trim().toLong()
                return abs(raw)
            } catch (_: Exception) {}
        }
        return Long.MIN_VALUE
    }

    private fun buildInferenceMetrics(
        response: SmolLMManager.SmolLMResponse,
        currentSamples: List<Long>,
        thermalThrottled: AtomicBoolean,
        chargeAtStartUah: Long,
        maxCpuThermalTempC: AtomicReference<Float>,
        maxSkinThermalTempC: AtomicReference<Float>,
    ): InferenceMetrics {
        val avgCurrentUa = computeAvgCurrentUa(currentSamples, chargeAtStartUah, response.generationTimeSecs)
        val cpuTemp = maxCpuThermalTempC.get()
        val skinTemp = maxSkinThermalTempC.get()
        return InferenceMetrics(
            ttftMs = response.ttftMs,
            decodeTps = response.generationSpeed,
            peakRssKb = response.peakRssKb,
            coldLoadTimeMs = smolLMManager.getColdLoadTimeMs(),
            avgCurrentUa = avgCurrentUa,
            thermalThrottled = thermalThrottled.get(),
            maxCpuThermalTempC = if (cpuTemp > 0f) cpuTemp else null,
            maxSkinThermalTempC = if (skinTemp > 0f) skinTemp else null,
        )
    }

    /**
     * Computes average current in µA using the best available data:
     * 1. Average of direct current samples (from API or sysfs); filters > 0
     * 2. Charge-counter delta divided by elapsed time
     * 3. Long.MIN_VALUE = "not supported on this device"
     */
    private fun computeAvgCurrentUa(
        samples: List<Long>,
        chargeAtStartUah: Long,
        durationSecs: Int,
    ): Long {
        val valid = samples.filter { it > 0L }
        if (valid.isNotEmpty()) return valid.average().toLong()

        // Fallback: charge counter delta → average current
        if (chargeAtStartUah != Long.MIN_VALUE && durationSecs > 0) {
            val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
            val chargeAtEnd = batteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
            if (chargeAtEnd != Long.MIN_VALUE) {
                val deltaUah = chargeAtStartUah - chargeAtEnd
                if (deltaUah > 0) {
                    // µAh ÷ hours = µA  →  µAh × 3600 ÷ seconds
                    return deltaUah * 3600L / durationSecs
                }
            }
        }
        return Long.MIN_VALUE
    }

    private fun logMetrics(runId: String, m: InferenceMetrics, responseText: String) {
        val coldStr = m.coldLoadTimeMs?.let { "${it}ms" } ?: "(warm)"
        val powerStr = if (m.avgCurrentUa == Long.MIN_VALUE) "unsupported hardware" else "${m.avgCurrentUa}µA"
        Log.d(
            METRICS_LOGTAG,
            "TTFT=${m.ttftMs}ms | " +
            "DecodeTPS=${m.decodeTps} tok/s | " +
            "PeakRSS=${m.peakRssKb}KB | " +
            "ColdLoad=$coldStr | " +
            "AvgCurrent=$powerStr | " +
            "ThermalThrottled=${m.thermalThrottled}"
        )

        // Same tags/format as BenchmarkService's headless logging (COLD_LOAD/TTFT/TPS/MEMORY/
        // POWER/THERMAL/RUN_DONE, all keyed by run_id), so logcat output is consistent and
        // parseable the same way regardless of whether a message came from manual chat or a
        // headless broadcast. Values are unchanged from what's already computed above — this is
        // purely an additional, differently-formatted set of log lines.
        val avgPowerMa = if (m.avgCurrentUa == Long.MIN_VALUE) null else m.avgCurrentUa / 1000.0
        val thermalStatus = if (m.thermalThrottled) "Throttled" else "Normal"
        Log.d("COLD_LOAD", "run_id=$runId value=${m.coldLoadTimeMs ?: 0}")
        Log.d("TTFT",      "run_id=$runId value=${m.ttftMs}")
        Log.d("TPS",       "run_id=$runId value=${m.decodeTps}")
        Log.d("MEMORY",    "run_id=$runId value=${m.peakRssKb}")
        Log.d("POWER",     "run_id=$runId value=${avgPowerMa?.let { "%.1f".format(it) } ?: "unsupported"}")
        Log.d("THERMAL",   "run_id=$runId value=$thermalStatus")
        // Additive real-temperature readings alongside the coarse THERMAL status above — does
        // not change or replace that line. THERMAL_TEMP_CPU is the raw CPU junction sensor (see
        // readCpuThermalZoneTempC()); THERMAL_TEMP_SKIN is the skin-therm-usr sensor (see
        // readSkinThermalTempC()), which OEM throttling policies more often key off.
        Log.d("THERMAL_TEMP_CPU", "run_id=$runId value=${
            m.maxCpuThermalTempC?.let { "%.1f".format(it) } ?: "unavailable"}")
        Log.d("THERMAL_TEMP_SKIN", "run_id=$runId value=${
            m.maxSkinThermalTempC?.let { "%.1f".format(it) } ?: "unavailable"}")
        Log.d("RUN_DONE",  "run_id=$runId response=$responseText")
    }

    /**
     * Manual messages aren't triggered by an external caller supplying a run_id (unlike
     * BenchmarkService's headless runs), so one is generated here: the saved message's own id
     * when available (stable, unique per message), falling back to a timestamp for calls that
     * don't persist a message (e.g. the in-app benchmark screen, which uses saveToDb=false).
     */
    private fun generateManualRunId(savedMessageId: Long?): String =
        savedMessageId?.let { "manual_msg_$it" } ?: "manual_${System.currentTimeMillis()}"

    /**
     * Attaches computed metrics to the assistant message they belong to, so the metrics panel
     * can be shown per-message (including after reopening the app), not just for the current
     * session's last response. No-ops when [messageId] is null (e.g. saveToDb=false runs, like
     * the in-app benchmark).
     */
    private fun persistMessageMetrics(messageId: Long?, m: InferenceMetrics) {
        if (messageId == null) return
        appDB.updateMessageMetrics(
            messageId = messageId,
            ttftMs = m.ttftMs,
            decodeTps = m.decodeTps,
            peakRssKb = m.peakRssKb,
            coldLoadTimeMs = m.coldLoadTimeMs,
            avgCurrentUa = if (m.avgCurrentUa == Long.MIN_VALUE) null else m.avgCurrentUa,
            thermalStatus = if (m.thermalThrottled) "Throttled" else "Normal",
        )
    }

    companion object {
        private val CURRENT_SYSFS_PATHS = listOf(
            "/sys/class/power_supply/battery/current_now",
            "/sys/class/power_supply/Battery/current_now",
            "/sys/class/power_supply/bms/current_now",
        )
    }

    private fun stopGeneration() {
        smolLMManager.stopResponseGeneration()
        _uiState.update { it.copy(isGeneratingResponse = false, renderedPartialResponse = null) }
    }

    private fun switchChat(chat: Chat) {
        stopGeneration()
        // Properly close the current model before loading the next one.
        // stopGeneration() sets isInferenceOn = false, so unload is always safe here.
        smolLMManager.unload()
        _uiState.update {
            it.copy(
                chat = chat,
                modelLoadingState = ModelLoadingState.NOT_LOADED,
                inferenceMetrics = null,
            )
        }
        loadModel()
    }

    private fun deleteChat(chat: Chat) {
        stopGeneration()
        appDB.deleteChat(chat)
        appDB.deleteMessages(chat.id)
        switchChat(appDB.loadDefaultChat())
    }

    private fun deleteChatMessages(chat: Chat) {
        stopGeneration()
        appDB.deleteMessages(chat.id)
    }

    private fun deleteModel(modelId: Long) {
        modelsRepository.deleteModel(modelId)
        val newChat = _uiState.value.chat.copy(llmModelId = -1)
        _uiState.update { it.copy(chat = newChat) }
    }

    /**
     * Get the current memory usage of the device. This method returns the memory consumed (in GBs)
     * and the total memory available on the device (in GBs)
     */
    private fun getCurrentMemoryUsage(): Pair<Float, Float> {
        val memoryInfo = MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        val totalMemory = (memoryInfo.totalMem) / 1024.0.pow(3.0)
        val usedMemory = (memoryInfo.availMem) / 1024.0.pow(3.0)
        return Pair(usedMemory.toFloat(), totalMemory.toFloat())
    }
}
