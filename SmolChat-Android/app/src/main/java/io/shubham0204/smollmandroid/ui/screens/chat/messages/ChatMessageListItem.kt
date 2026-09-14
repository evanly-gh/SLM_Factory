package io.shubham0204.smollmandroid.ui.screens.chat.messages

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyItemScope
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.toArgb
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.shubham0204.smollmandroid.R
import io.shubham0204.smollmandroid.data.ChatMessage
import io.shubham0204.smollmandroid.ui.screens.chat.dialogs.createChatMessageOptionsDialog

data class ChatMessageProperties(
    val codeSnippets: List<String>
)

fun buildMessageProperties(message: String): ChatMessageProperties {
    val codeSnippetPattern = Regex("""```(?:[a-zA-Z0-9+#-]*\n)?([\s\S]*?)```""")
    val codeSnippets = codeSnippetPattern.findAll(message).map { it.groupValues[1].trim() }.toList()
    return ChatMessageProperties(codeSnippets)
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun LazyItemScope.MessageListItem(
    message: ChatMessage,
    responseGenerationSpeed: Float?,
    responseGenerationTimeSecs: Int?,
    isUserMessage: Boolean,
    onCopyClicked: () -> Unit,
    onShareClicked: () -> Unit,
    onCodeSnippetCopyClicked: (List<String>) -> Unit,
    onMessageEdited: (String) -> Unit,
    modifier: Modifier = Modifier,
    allowEditing: Boolean,
) {
    var isEditing by rememberSaveable { mutableStateOf(false) }
    if (isUserMessage) {
        UserMessageListItem(
            message,
            isEditing,
            allowEditing,
            onCopyClicked,
            onShareClicked,
            onMessageEdited
        )
    } else {
        LLMMessageListItem(
            modifier,
            message,
            onCopyClicked,
            onShareClicked,
            onCodeSnippetCopyClicked,
            // Fall back to the transient per-session value only when this message has no
            // persisted metrics yet (e.g. it's still streaming, or predates the metrics
            // migration). Once persisted, message.decodeTps/ttftMs etc. are the source of truth
            // and survive app restarts — see AppDB.updateMessageMetrics().
            message.decodeTps ?: responseGenerationSpeed,
            responseGenerationTimeSecs
        )
    }
}

@Composable
private fun LazyItemScope.UserMessageListItem(
    message: ChatMessage,
    isEditing: Boolean,
    allowEditing: Boolean,
    onCopyClicked: () -> Unit,
    onShareClicked: () -> Unit,
    onMessageEdited: (String) -> Unit
) {
    val message1 = message
    var isEditing1 = isEditing
    Row(
        modifier = Modifier
            .fillMaxWidth(),
        horizontalArrangement = Arrangement.End,
    ) {
        Column(horizontalAlignment = Alignment.End) {
            var message by rememberSaveable { mutableStateOf(message1.toString()) }
            if (isEditing1) {
                TextField(
                    value = message,
                    onValueChange = { message = it },
                    modifier =
                        Modifier
                            .padding(8.dp)
                            .background(
                                MaterialTheme.colorScheme.surfaceContainerHighest,
                                RoundedCornerShape(16.dp),
                            )
                            .padding(8.dp)
                            .widthIn(max = 250.dp),
                    colors =
                        TextFieldDefaults.colors(
                            errorContainerColor = Color.Transparent,
                            focusedContainerColor = Color.Transparent,
                            unfocusedContainerColor = Color.Transparent,
                            disabledContainerColor = Color.Transparent,
                        ),
                )
            } else {
                ChatMessageText(
                    modifier =
                        Modifier
                            .padding(4.dp)
                            .background(
                                MaterialTheme.colorScheme.surfaceContainerHighest,
                                RoundedCornerShape(16.dp),
                            )
                            .padding(8.dp)
                            .widthIn(max = 250.dp),
                    textColor = MaterialTheme.colorScheme.onSurface.toArgb(),
                    textSize = 16f,
                    message = message1.renderedMessage,
                    onLongClick = {
                        createChatMessageOptionsDialog(
                            showEditOption = allowEditing,
                            onEditClick = { isEditing1 = true },
                            onCopyClick = { onCopyClicked() },
                            onShareClick = { onShareClicked() },
                            showCopyCodeSnippetOption = false
                        )
                    },
                )
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                if (allowEditing) {
                    Spacer(modifier = Modifier.width(8.dp))
                    if (isEditing1) {
                        Text(
                            text = stringResource(R.string.edit_chat_message_done),
                            modifier =
                                Modifier.clickable {
                                    isEditing1 = false
                                    onMessageEdited(message)
                                },
                            fontSize = 6.sp,
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = stringResource(R.string.dialog_neg_cancel),
                            modifier =
                                Modifier.clickable {
                                    isEditing1 = false
                                    message = message.toString()
                                },
                            fontSize = 6.sp,
                        )
                    }
                }
                Spacer(modifier = Modifier.width(16.dp))
            }
        }
    }
}

@Composable
private fun LazyItemScope.LLMMessageListItem(
    modifier: Modifier,
    message: ChatMessage,
    onCopyClicked: () -> Unit,
    onShareClicked: () -> Unit,
    onCodeSnippetCopyClicked: (List<String>) -> Unit,
    responseGenerationSpeed: Float?,
    responseGenerationTimeSecs: Int?
) {
    Row(
        modifier = modifier
            .fillMaxWidth()
    ) {
        Spacer(modifier = Modifier.width(8.dp))
        Column {
            ChatMessageText(
                // to make pointerInput work in MarkdownText use disableLinkMovementMethod
                // https://github.com/jeziellago/compose-markdown/issues/85#issuecomment-2184040304
                modifier =
                    Modifier
                        .padding(4.dp)
                        .background(MaterialTheme.colorScheme.surface)
                        .padding(4.dp)
                        .fillMaxSize(),
                textColor = MaterialTheme.colorScheme.onBackground.toArgb(),
                textSize = 16f,
                message = message.renderedMessage,
                onLongClick = {
                    val messageProperties = buildMessageProperties(message.message)
                    createChatMessageOptionsDialog(
                        showEditOption = false,
                        onEditClick = {
                            /** Not applicable as showEditOption is set to false * */
                        },
                        onCopyClick = { onCopyClicked() },
                        onShareClick = { onShareClicked() },
                        onCodeSnippetCopyClick = {
                            onCodeSnippetCopyClicked(messageProperties.codeSnippets)
                        },
                        showCopyCodeSnippetOption = messageProperties.codeSnippets.isNotEmpty()
                    )
                },
            )
            Row(verticalAlignment = Alignment.CenterVertically) {
                responseGenerationSpeed?.let {
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(text = "%.2f tokens/s".format(it), fontSize = 8.sp)
                    Spacer(modifier = Modifier.width(6.dp))
                    Box(
                        modifier =
                            Modifier
                                .size(2.dp)
                                .clip(CircleShape)
                                .background(Color.DarkGray)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(text = "$responseGenerationTimeSecs s", fontSize = 8.sp)
                }
            }
            // Persisted per-message metrics (TTFT, peak RSS, cold load, power, thermal) —
            // populated for both manual and headless-benchmark messages. Only rendered once any
            // of these has actually been recorded, so older messages and user messages show
            // nothing extra.
            if (message.ttftMs != null || message.peakRssKb != null || message.coldLoadTimeMs != null ||
                message.avgCurrentUa != null || message.thermalStatus != null
            ) {
                MessageMetricsRow(message)
            }
        }
    }
}

/**
 * Same styled metrics badge row originally shown only as a transient panel for the most recent
 * response (see the removed ChatActivity.InferenceMetricsPanel/InferenceMetricItem). Reused here
 * per-message so every message — historical or newest, manual or headless-triggered — renders
 * identically, reading straight from that message's own persisted metric columns.
 */
@Composable
private fun MessageMetricsRow(message: ChatMessage) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        tonalElevation = 1.dp,
    ) {
        Row(
            modifier = Modifier
                .horizontalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 6.dp),
            horizontalArrangement = Arrangement.spacedBy(20.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            message.ttftMs?.let {
                MetricBadge(label = "TTFT", value = "${it}ms")
            }
            message.decodeTps?.let {
                MetricBadge(label = "TPS", value = "%.1f tok/s".format(it))
            }
            message.peakRssKb?.let {
                MetricBadge(label = "Peak RSS", value = "${it}KB")
            }
            message.coldLoadTimeMs?.let {
                MetricBadge(label = "Cold Load", value = "${it}ms")
            }
            message.avgCurrentUa?.let {
                MetricBadge(label = "Power", value = "%.1fmA".format(it / 1000.0))
            }
            message.thermalStatus?.let {
                MetricBadge(
                    label = "Thermal",
                    value = it,
                    valueColor = if (it == "Throttled" || it == "Moderate" || it == "Severe" ||
                        it == "Critical" || it == "Emergency"
                    ) {
                        MaterialTheme.colorScheme.error
                    } else {
                        MaterialTheme.colorScheme.primary
                    },
                )
            }
        }
    }
}

@Composable
private fun MetricBadge(
    label: String,
    value: String,
    valueColor: Color = MaterialTheme.colorScheme.primary,
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold,
            color = valueColor,
        )
    }
}
