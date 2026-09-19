// Importing packages and libraries
package com.example.mobileai

import android.os.Bundle
import android.os.Debug
import android.widget.*
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private val client = OkHttpClient()

    // IP Addr to point to Ollama
     private val BASE_URL = "http://10.0.2.2:11434"

    // IP Addr to point to my Android Phone
//     private val BASE_URL = "http://10.0.0.10:11434"

    // Models to use running on Ollama
    private val models = listOf(
        "qwen2.5:0.5b",
        "tinyllama:latest",
    )

    // Selected model
    private var selectedModel = models[0]


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // Defining variables for metrics
        val metricTTFT = findViewById<TextView>(R.id.metricTTFT)
        val metricColdLoad = findViewById<TextView>(R.id.metricColdLoad)
        val metricTotalDuration = findViewById<TextView>(R.id.metricTotalDuration)
        val metricPower = findViewById<TextView>(R.id.metricPower)
        val metricThermal = findViewById<TextView>(R.id.metricThermal)
        val inputField = findViewById<EditText>(R.id.inputField)
        val sendButton = findViewById<Button>(R.id.sendButton)
        val responseText = findViewById<TextView>(R.id.responseText)
        val metricLatency = findViewById<TextView>(R.id.metricLatency)
        val metricTokens = findViewById<TextView>(R.id.metricTokens)
        val metricTokensPerSec = findViewById<TextView>(R.id.metricTokensPerSec)
        val metricMemory = findViewById<TextView>(R.id.metricMemory)
        val modelSpinner = findViewById<Spinner>(R.id.modelSpinner)

        // Set up the dropdown with the models list
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, models)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        modelSpinner.adapter = adapter

        // Update selectedModel whenever the user picks a different one
        modelSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: android.view.View?, position: Int, id: Long) {
                selectedModel = models[position]
                responseText.text = ""  // clear previous response when switching models
            }
            override fun onNothingSelected(parent: AdapterView<*>) {}
        }

        // What happens when user enters a request and hits send. Convert prompt and send it to
        // Ollama in order to perform inference and get metrics.
        sendButton.setOnClickListener {
            val prompt = inputField.text.toString()
            if (prompt.isNotEmpty()) {
                responseText.text = "Thinking..."
                val memBefore = Debug.getNativeHeapAllocatedSize() / 1024 / 1024
                val startTime = System.currentTimeMillis()

                askOllama(prompt, selectedModel) { reply, evalData ->
                    val latencyMs = System.currentTimeMillis() - startTime
                    val memAfter = Debug.getNativeHeapAllocatedSize() / 1024 / 1024

                    // Already have these
                    val promptTokens = evalData.optInt("prompt_eval_count", -1)
                    val responseTokens = evalData.optInt("eval_count", -1)
                    val evalDurationNs = evalData.optLong("eval_duration", -1L)
                    val tokensPerSec = if (evalDurationNs > 0 && responseTokens > 0)
                        (responseTokens / (evalDurationNs / 1_000_000_000.0)).toInt()
                    else -1

                    // Extract metrics from Ollama response
                    val promptEvalDurationNs = evalData.optLong("prompt_eval_duration", -1L)
                    val loadDurationNs = evalData.optLong("load_duration", -1L)
                    val totalDurationNs = evalData.optLong("total_duration", -1L)

                    // Convert to readable units
                    val ttftMs = if (promptEvalDurationNs > 0) promptEvalDurationNs / 1_000_000 else -1
                    val coldLoadMs = if (loadDurationNs > 0) loadDurationNs / 1_000_000 else -1
                    val totalMs = if (totalDurationNs > 0) totalDurationNs / 1_000_000 else -1

                    runOnUiThread {
                        responseText.text = reply
                        metricLatency.text        = "Latency (round trip): ${latencyMs}ms"
                        metricTTFT.text           = "TTFT: ${if (ttftMs > 0) "${ttftMs}ms" else "n/a"}"
                        metricColdLoad.text       = "Cold load time: ${if (coldLoadMs > 0) "${coldLoadMs}ms" else "n/a"}"
                        metricTokens.text         = "Tokens: ${promptTokens}p + ${responseTokens}r"
                        metricTokensPerSec.text   = "Tokens/sec: ${if (tokensPerSec > 0) tokensPerSec else "n/a"}"
                        metricTotalDuration.text  = "Total duration: ${if (totalMs > 0) "${totalMs}ms" else "n/a"}"
                        metricMemory.text         = "App RAM: ${memBefore}MB → ${memAfter}MB"
                        metricPower.text          = "Power: only available on-device"
                        metricThermal.text        = "Thermal: only available on-device"
                    }
                }
            }
        }
    }

    // Defining askOllama function in order to send a request and get a response back from model
    private fun askOllama(prompt: String, model: String, callback: (String, JSONObject) -> Unit) {
        val json = JSONObject().apply {
            put("model", model)
            put("temperature", 0.3)
            put("messages", JSONArray().apply {
                // System prompt added here
                put(JSONObject().apply {
                    put("role", "system")
                    put("content", "You are a helpful assistant. Only answer what you know confidently. If you are unsure, say so instead of guessing.")
                })
                // User message
                put(JSONObject().apply {
                    put("role", "user")
                    put("content", prompt)
                })
            })
            put("stream", false)
        }

        val body = json.toString().toRequestBody("application/json".toMediaType())
        val request = Request.Builder()
            .url("$BASE_URL/api/chat")
            .post(body)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                callback("Error: ${e.message}", JSONObject())
            }
            override fun onResponse(call: Call, response: Response) {
                val text = response.body?.string() ?: ""
                val fullResponse = JSONObject(text)
                val reply = fullResponse
                    .getJSONObject("message")
                    .getString("content")
                callback(reply, fullResponse)
            }
        })
    }
}