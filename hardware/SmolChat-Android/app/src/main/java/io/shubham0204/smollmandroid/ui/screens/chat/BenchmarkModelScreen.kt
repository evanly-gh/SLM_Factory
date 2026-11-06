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

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import compose.icons.FeatherIcons
import compose.icons.feathericons.ArrowLeft
import io.shubham0204.smollmandroid.ui.components.AppBarTitleText
import io.shubham0204.smollmandroid.ui.theme.SmolLMAndroidTheme

@Preview
@Composable
private fun PreviewBenchmarkModelScreen() {
    BenchmarkModelScreen(
        onBackClicked = {},
        onEvent = {},
        inferenceMetrics = null,
        isGeneratingResponse = false,
    )
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BenchmarkModelScreen(
    onBackClicked: () -> Unit,
    onEvent: (ChatScreenUIEvent) -> Unit,
    inferenceMetrics: InferenceMetrics?,
    isGeneratingResponse: Boolean,
) {
    var benchmarkResult by remember { mutableStateOf<String?>(null) }
    var isBenchmarkRunning by remember { mutableStateOf(false) }
    var isMetricsRunning by remember { mutableStateOf(false) }

    val ppValue = remember(benchmarkResult) {
        benchmarkResult?.let { result ->
            val lines = result.split("\n")
            lines.find { it.contains("| pp ") }?.split("|")?.getOrNull(6)?.trim()
        } ?: "N/A"
    }
    val tgValue = remember(benchmarkResult) {
        benchmarkResult?.let { result ->
            val lines = result.split("\n")
            lines.find { it.contains("| tg ") }?.split("|")?.getOrNull(6)?.trim()
        } ?: "N/A"
    }

    SmolLMAndroidTheme {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { AppBarTitleText("Model Benchmark") },
                    navigationIcon = {
                        IconButton(onClick = onBackClicked) {
                            Icon(FeatherIcons.ArrowLeft, contentDescription = "Back")
                        }
                    },
                )
            }
        ) { paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .background(MaterialTheme.colorScheme.surface)
                    .verticalScroll(rememberScrollState())
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(24.dp),
            ) {
                // ── llama.cpp internal benchmark ──────────────────────────────
                Text(
                    text = "llama.cpp Benchmark",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.fillMaxWidth(),
                )

                Button(
                    onClick = {
                        isBenchmarkRunning = true
                        onEvent(
                            ChatScreenUIEvent.ChatEvents.StartBenchmark(
                                onResult = {
                                    benchmarkResult = it
                                    isBenchmarkRunning = false
                                }
                            )
                        )
                    },
                    enabled = !isBenchmarkRunning && !isMetricsRunning,
                ) {
                    if (isBenchmarkRunning) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(24.dp),
                            color = MaterialTheme.colorScheme.onPrimary,
                            strokeWidth = 2.dp,
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Benchmarking...")
                    } else {
                        Text("Start Benchmark")
                    }
                }

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    ),
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp),
                        horizontalArrangement = Arrangement.SpaceEvenly,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        BenchmarkMetric(modifier = Modifier.weight(1f), label = "PP (tokens/s)", value = ppValue)
                        BenchmarkMetric(modifier = Modifier.weight(1f), label = "TG (tokens/s)", value = tgValue)
                    }
                }

                if (benchmarkResult != null) {
                    Text(
                        text = "Prompt: 512 tokens, Generation: 128 tokens",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                // ── Inference metrics ─────────────────────────────────────────
                Text(
                    text = "Inference Metrics",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.fillMaxWidth(),
                )

                Button(
                    onClick = {
                        isMetricsRunning = true
                        onEvent(
                            ChatScreenUIEvent.ChatEvents.StartInferenceBenchmark(
                                onComplete = { isMetricsRunning = false }
                            )
                        )
                    },
                    enabled = !isBenchmarkRunning && !isMetricsRunning && !isGeneratingResponse,
                ) {
                    if (isMetricsRunning || isGeneratingResponse) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(24.dp),
                            color = MaterialTheme.colorScheme.onPrimary,
                            strokeWidth = 2.dp,
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Running inference...")
                    } else {
                        Text("Measure Inference Metrics")
                    }
                }

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(16.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    ),
                ) {
                    Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceEvenly,
                        ) {
                            BenchmarkMetric(
                                modifier = Modifier.weight(1f),
                                label = "TTFT",
                                value = inferenceMetrics?.let { "${it.ttftMs} ms" } ?: "N/A",
                            )
                            BenchmarkMetric(
                                modifier = Modifier.weight(1f),
                                label = "Decode TPS",
                                value = inferenceMetrics?.let { "%.1f tok/s".format(it.decodeTps) } ?: "N/A",
                            )
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceEvenly,
                        ) {
                            BenchmarkMetric(
                                modifier = Modifier.weight(1f),
                                label = "Peak RSS",
                                value = inferenceMetrics?.let { "${it.peakRssKb} KB" } ?: "N/A",
                            )
                            BenchmarkMetric(
                                modifier = Modifier.weight(1f),
                                label = "Cold Load",
                                value = inferenceMetrics?.coldLoadTimeMs?.let { "$it ms" } ?: "N/A",
                            )
                        }
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceEvenly,
                        ) {
                            BenchmarkMetric(
                                modifier = Modifier.weight(1f),
                                label = "Avg Current",
                                value = inferenceMetrics?.let {
                                    when {
                                        it.avgCurrentUa == Long.MIN_VALUE -> "unsupported hardware"
                                        it.avgCurrentUa > 0 -> "${it.avgCurrentUa / 1000} mA"
                                        else -> "N/A"
                                    }
                                } ?: "N/A",
                            )
                            BenchmarkMetric(
                                modifier = Modifier.weight(1f),
                                label = "Thermal",
                                value = inferenceMetrics?.let {
                                    if (it.thermalThrottled) "Throttled" else "Normal"
                                } ?: "N/A",
                                valueColor = if (inferenceMetrics?.thermalThrottled == true) {
                                    androidx.compose.ui.graphics.Color(0xFFE53935)
                                } else {
                                    null
                                },
                            )
                        }
                    }
                }

                if (inferenceMetrics != null) {
                    Text(
                        text = "Metrics from last inference run",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))
            }
        }
    }
}

@Composable
private fun BenchmarkMetric(
    modifier: Modifier = Modifier,
    label: String,
    value: String,
    valueColor: androidx.compose.ui.graphics.Color? = null,
) {
    Column(modifier = modifier, horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelMedium,
            textAlign = TextAlign.Center,
        )
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = valueColor ?: MaterialTheme.colorScheme.primary,
            textAlign = TextAlign.Center,
        )
    }
}
