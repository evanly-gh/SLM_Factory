#include "LLMInference.h"
#include <android/log.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <thread>
#include <unistd.h>

#define TAG "[SmolLMAndroid-Cpp]"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// DIAGNOSTIC ONLY: ggml-vulkan.cpp reports some failures (e.g. the pipeline
// name on a compute-pipeline creation failure) only via std::cerr, which by
// default goes nowhere visible on Android -- it isn't forwarded to logcat.
// This redirects the process's stderr into a pipe and relays each line to
// logcat under tag "native-stderr", so those otherwise-invisible diagnostics
// show up. Runs once per process; safe to leave running for the process
// lifetime since it just blocks on read() until stderr is closed.
static void redirectStderrToLogcat() {
    static bool started = false;
    if (started) {
        return;
    }
    started = true;

    int pipeFds[2];
    if (pipe(pipeFds) != 0) {
        LOGe("redirectStderrToLogcat: pipe() failed");
        return;
    }
    dup2(pipeFds[1], STDERR_FILENO);
    close(pipeFds[1]);

    int readFd = pipeFds[0];
    std::thread([readFd]() {
        FILE *stream = fdopen(readFd, "r");
        if (!stream) {
            close(readFd);
            return;
        }
        char *line = nullptr;
        size_t lineCap = 0;
        ssize_t lineLen;
        while ((lineLen = getline(&line, &lineCap, stream)) != -1) {
            if (lineLen > 0 && line[lineLen - 1] == '\n') {
                line[lineLen - 1] = '\0';
            }
            __android_log_print(ANDROID_LOG_ERROR, "native-stderr", "%s", line);
        }
        free(line);
        fclose(stream);
    }).detach();
}

// Forwards llama.cpp's own log lines directly to logcat under tag
// "llama-native", instead of relying solely on the stderr pipe above.
// This is what makes llama_model_load_from_file()'s per-model
// "offloading N repeating layers to GPU" / "offloaded N/M layers to GPU"
// lines (see llama-model.cpp) show up reliably -- those report how many of
// *this specific model's* layers actually got placed on a GPU device, which
// is the real answer to "was the GPU engaged", not just "is a GPU backend
// linked in". Kept permanently (not diagnostic-only): this is the
// authoritative per-run backend confirmation, for every run going forward.
static void llamaLogToLogcat(ggml_log_level level, const char *text, void * /*user_data*/) {
    android_LogPriority priority;
    switch (level) {
        case GGML_LOG_LEVEL_ERROR: priority = ANDROID_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  priority = ANDROID_LOG_WARN;  break;
        case GGML_LOG_LEVEL_DEBUG: priority = ANDROID_LOG_DEBUG; break;
        default:                   priority = ANDROID_LOG_INFO;  break;
    }
    std::string line(text);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    if (!line.empty()) {
        __android_log_print(priority, "llama-native", "%s", line.c_str());
    }
}

// Explicit, permanent confirmation of which ggml backend device(s) are
// actually registered in this process -- logged under tag "BACKEND_CHECK"
// so it's unambiguous in logcat and not easy to miss among the surrounding
// model-load noise. This proves whether a GPU (Vulkan) device is genuinely
// present, independent of and complementary to llama-native's per-model
// layer-offload counts above: this answers "is a GPU device available at
// all", theirs answers "how many of this model's layers went to it".
static void logRegisteredBackendDevices() {
    const size_t deviceCount = ggml_backend_dev_count();
    __android_log_print(ANDROID_LOG_INFO, "BACKEND_CHECK", "%zu ggml backend device(s) registered:", deviceCount);
    for (size_t i = 0; i < deviceCount; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char *typeName;
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:   typeName = "CPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   typeName = "GPU";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  typeName = "IGPU";  break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: typeName = "ACCEL"; break;
            default:                             typeName = "OTHER"; break;
        }
        __android_log_print(ANDROID_LOG_INFO, "BACKEND_CHECK", "  [%zu] %s (%s) -- type=%s", i,
                             ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), typeName);
    }
}

void
LLMInference::loadModel(const char *model_path, float minP, float temperature, bool storeChats, long contextSize,
                        const char *chatTemplate, int nThreads, bool useMmap, bool useMlock,
                        const char *nativeLibraryDir, int nGpuLayers) {
    redirectStderrToLogcat();
    llama_log_set(llamaLogToLogcat, nullptr);

    // Our vendored OpenCL ICD loader (vendor/opencl-icd-loader/libOpenCL.so)
    // discovers vendor drivers by scanning a Linux-style /etc/OpenCL/vendors/
    // *.icd filesystem registry -- which doesn't exist on Android, so
    // clGetPlatformIDs() finds nothing even though a real vendor driver is
    // genuinely present on-device (confirmed via `adb shell ls
    // /vendor/lib64/libOpenCL.so` on a real device; ggml_backend_opencl_reg()
    // itself logged "platform IDs not available" as the reason). The loader
    // supports OCL_ICD_FILENAMES as a direct override -- a colon-separated
    // list of driver .so paths to load, bypassing that registry scan
    // entirely. This is harmless outside the "opencl" flavor (the loader
    // itself, and thus this env var, isn't even present in other flavors'
    // APKs) and harmless on a device without a driver at this exact path
    // (the vendor-add attempt just fails, leaving OpenCL unavailable exactly
    // as before -- no regression risk).
    setenv("OCL_ICD_FILENAMES", "/vendor/lib64/libOpenCL.so", 1);

    LOGi("loading model with"
         "\n\tmodel_path = %s"
         "\n\tminP = %f"
         "\n\ttemperature = %f"
         "\n\tstoreChats = %d"
         "\n\tcontextSize = %li"
         "\n\tchatTemplate = %s"
         "\n\tnThreads = %d"
         "\n\tuseMmap = %d"
         "\n\tuseMlock = %d"
         "\n\tnativeLibraryDir = %s"
         "\n\tnGpuLayers = %d",
         model_path, minP, temperature, storeChats, contextSize, chatTemplate, nThreads, useMmap, useMlock,
         nativeLibraryDir, nGpuLayers);

    // ggml_backend_load_all()'s no-args form only searches the launching
    // process's executable directory and cwd for backend .so files -- on
    // Android neither is where an app's own native libraries live, so
    // dynamic backends (e.g. libggml-vulkan.so) were silently never found.
    // ggml_backend_load_all_from_path(nativeLibraryDir) doesn't fix this
    // either: it discovers backends by fs::directory_iterator-scanning the
    // directory, and /data/app/.../lib/arm64/ is owned by "system" -- Android
    // blocks an app process from *listing* that directory even though it can
    // dlopen() an exact, known file inside it directly (confirmed via a
    // probe dlopen() of the exact path, which succeeded, right after the
    // scan-based loader had already silently found nothing there). So we
    // load each backend by its known exact filename instead, via
    // ggml_backend_load()'s single-file path -- no directory listing
    // involved. This also means CPU now needs an explicit call too: it used
    // to self-register unconditionally regardless of any of this, but that
    // path requires GGML_USE_CPU to be defined on ggml-base, which the
    // "vulkan" flavor's CMakeLists.txt no longer sets now that
    // GGML_BACKEND_DL is on (needed for libggml-vulkan.so's dynamic-load
    // entrypoint to exist at all -- see CMakeLists.txt) -- confirmed via a
    // real on-device regression (BACKEND_CHECK dropped from 1 device to 0),
    // not a guess. libggml-vulkan.so is only present in the "vulkan"
    // flavor's APK, so failing to find it on "cpu" is expected, not an error.
    ggml_backend_load((std::string(nativeLibraryDir) + "/libggml-cpu.so").c_str());
    ggml_backend_reg_t vulkanReg = ggml_backend_load((std::string(nativeLibraryDir) + "/libggml-vulkan.so").c_str());
    if (vulkanReg) {
        LOGi("ggml_backend_load: libggml-vulkan.so loaded successfully");
    } else {
        LOGi("ggml_backend_load: libggml-vulkan.so not loaded (expected outside the \"vulkan\" flavor)");
    }
    // Same as the Vulkan call above, added from day one this time rather
    // than discovered the hard way -- libggml-opencl.so only exists in the
    // "opencl" flavor's APK.
    ggml_backend_reg_t openclReg = ggml_backend_load((std::string(nativeLibraryDir) + "/libggml-opencl.so").c_str());
    if (openclReg) {
        LOGi("ggml_backend_load: libggml-opencl.so loaded successfully");
    } else {
        LOGi("ggml_backend_load: libggml-opencl.so not loaded (expected outside the \"opencl\" flavor)");
    }
    logRegisteredBackendDevices();

    // create an instance of llama_model
    llama_model_params model_params = llama_model_default_params();
    // llama_model_default_params() defaults n_gpu_layers to 0 (CPU-only) -- previously never
    // overridden here regardless of build flavor, so even a genuinely-registered GPU backend
    // (see logRegisteredBackendDevices() above) never had any layers assigned to it by ggml's
    // scheduler. Matches llama-bench/llama-cli's own -ngl flag semantics.
    model_params.n_gpu_layers = nGpuLayers;
    if (useMmap && useMlock) {
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;
    } else if (useMmap) {
        model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    } else if (useMlock) {
        model_params.load_mode = LLAMA_LOAD_MODE_MLOCK;
    } else {
        model_params.load_mode = LLAMA_LOAD_MODE_NONE;
    }
    _model = llama_model_load_from_file(model_path, model_params);
    if (!_model) {
        LOGe("failed to load model from %s", model_path);
        throw std::runtime_error("loadModel() failed");
    }

    // create an instance of llama_context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = contextSize;
    ctx_params.n_batch = contextSize;
    ctx_params.n_threads = nThreads;
    ctx_params.no_perf = true; // disable performance metrics
    _ctx = llama_init_from_model(_model, ctx_params);
    if (!_ctx) {
        LOGe("llama_new_context_with_model() returned null)");
        throw std::runtime_error("llama_new_context_with_model() returned null");
    }

    // create an instance of llama_sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true; // disable performance metrics
    _sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(_sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // second chain, identical except for a leading logit-bias stage that suppresses this
    // model's own EOG token(s) — enumerated dynamically via llama_vocab_is_eog() over the whole
    // vocab, never hardcoded (different models/tokenizers use different EOG token ids). Built
    // once here so completionLoop() can just pick a chain per-token with no per-call rebuild
    // cost. See startCompletion()'s suppressEarlyEos parameter for how/when it's selected.
    const llama_vocab *vocab = llama_model_get_vocab(_model);
    int32_t nVocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_logit_bias> eogBias;
    for (llama_token t = 0; t < nVocab; ++t) {
        if (llama_vocab_is_eog(vocab, t)) {
            eogBias.push_back({t, -100.0f});
        }
    }
    _samplerEosSuppressed = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(_samplerEosSuppressed,
                             llama_sampler_init_logit_bias(nVocab, (int32_t) eogBias.size(), eogBias.data()));
    llama_sampler_chain_add(_samplerEosSuppressed, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(_samplerEosSuppressed, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // TEMPORARY DIAGNOSTIC: confirm the EOG enumeration actually found token(s) to bias — an
    // empty list here would make the logit_bias sampler a no-op ("?logit-bias" empty-sampler
    // fallback in llama_sampler_init_logit_bias when n_logit_bias<=0), silently defeating
    // suppressEarlyEos entirely.
    LOGi("EOS-suppress diag: nVocab=%d eogTokenCount=%zu", nVocab, eogBias.size());
    for (const auto &lb : eogBias) {
        LOGi("EOS-suppress diag: eog token id=%d bias=%.1f", lb.token, lb.bias);
    }

    _formattedMessages = std::vector<char>(llama_n_ctx(_ctx));
    _messages.clear();

    if (chatTemplate == nullptr) {
        _chatTemplate = llama_model_chat_template(_model, nullptr);
    } else {
        _chatTemplate = strdup(chatTemplate);
    }
    this->_storeChats = storeChats;
}

void
LLMInference::addChatMessage(const char *message, const char *role) {
    _messages.push_back({strdup(role), strdup(message)});
}

float
LLMInference::getResponseGenerationTime() const {
    return (float) _responseNumTokens / (_responseGenerationTime / 1e6);
}

int
LLMInference::getPromptTokenCount() const {
    return (int) _promptTokens.size();
}

int
LLMInference::getContextSizeUsed() const {
    return _nCtxUsed;
}

bool
LLMInference::startCompletion(const char *query, int maxTokens, bool suppressEarlyEos) {
    if (!_storeChats) {
        _formattedMessages.clear();
        _formattedMessages = std::vector<char>(llama_n_ctx(_ctx));
    }
    _responseGenerationTime = 0;
    _responseNumTokens = 0;
    _maxTokens = maxTokens;
    _pendingStop = false;
    _suppressEarlyEos = suppressEarlyEos;
    addChatMessage(query, "user");
    // apply the chat-template
    std::vector<common_chat_msg> messages;
    for (const llama_chat_message& message : _messages) {
        common_chat_msg msg;
        msg.role    = message.role;
        msg.content = message.content;
        messages.push_back(msg);
    }
    auto templates = common_chat_templates_init(_model, _chatTemplate ? _chatTemplate : "");
    LOGi("chat template supports enable_thinking: %d", common_chat_templates_support_enable_thinking(templates.get()));

    common_chat_templates_inputs inputs;
    inputs.messages = messages;
    inputs.enable_thinking = false;

    // Try Jinja rendering first with tools defined to prevent "tojson on Undefined" errors.
    // If Jinja fails (e.g. unsupported filters like lstrip), fall back to legacy rendering.
    inputs.use_jinja = true;
    inputs.chat_template_kwargs["tools"] = "[]";

    std::string prompt;
    bool usedJinja = true;
    try {
        prompt = common_chat_templates_apply(templates.get(), inputs).prompt;
    } catch (const std::exception &e) {
        LOGe("Jinja template failed: %s — retrying with legacy renderer", e.what());
        inputs.use_jinja = false;
        inputs.chat_template_kwargs.clear();
        inputs.enable_thinking = false;
        prompt = common_chat_templates_apply(templates.get(), inputs).prompt;
        usedJinja = false;
    }
    _promptTokens = common_tokenize(llama_model_get_vocab(_model), prompt, true, true);

    // create a llama_batch containing a single sequence
    // see llama_batch_init for more details
    _batch = new llama_batch();
    _batch->token = _promptTokens.data();
    _batch->n_tokens = _promptTokens.size();

    return usedJinja;
}

// taken from:
// https://github.com/ggerganov/llama.cpp/blob/master/examples/llama.android/llama/src/main/cpp/llama-android.cpp#L38
bool
LLMInference::_isValidUtf8(const char *response) {
    if (!response) {
        return true;
    }
    const unsigned char *bytes = (const unsigned char *) response;
    int num;
    while (*bytes != 0x00) {
        if ((*bytes & 0x80) == 0x00) {
            // U+0000 to U+007F
            num = 1;
        } else if ((*bytes & 0xE0) == 0xC0) {
            // U+0080 to U+07FF
            num = 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            // U+0800 to U+FFFF
            num = 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            // U+10000 to U+10FFFF
            num = 4;
        } else {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i) {
            if ((*bytes & 0xC0) != 0x80) {
                return false;
            }
            bytes += 1;
        }
    }
    return true;
}

bool
LLMInference::_hasRepetition(const std::string &text) {
    static constexpr int kMinPatternLen = 10;
    static constexpr int kMaxPatternLen = 20;
    static constexpr int kMinRepeats = 5;

    for (int patternLen = kMinPatternLen; patternLen <= kMaxPatternLen; ++patternLen) {
        size_t needed = (size_t) patternLen * kMinRepeats;
        if (text.size() < needed) {
            continue;
        }
        std::string pattern = text.substr(text.size() - patternLen, patternLen);
        bool repeated = true;
        for (int i = 1; i < kMinRepeats; ++i) {
            size_t start = text.size() - (size_t) patternLen * (i + 1);
            if (text.compare(start, patternLen, pattern) != 0) {
                repeated = false;
                break;
            }
        }
        if (repeated) {
            return true;
        }
    }
    return false;
}

std::string
LLMInference::completionLoop() {
    // a previous call already returned the final real piece and flagged a stop (max_tokens or
    // repetition) — finalize now, without decoding another token, so that piece is never dropped
    if (_pendingStop) {
        _pendingStop = false;
        addChatMessage(strdup(_response.data()), "assistant");
        _response.clear();
        return "[EOG]";
    }

    // check if the length of the inputs to the model
    // have exceeded the context size of the model
    uint32_t contextSize = llama_n_ctx(_ctx);
    _nCtxUsed = llama_memory_seq_pos_max(llama_get_memory(_ctx), 0) + 1;
    if (_nCtxUsed + _batch->n_tokens > contextSize) {
        throw std::runtime_error("context size reached");
    }

    auto start = ggml_time_us();
    // run the model
    if (llama_decode(_ctx, *_batch) < 0) {
        throw std::runtime_error("llama_decode() failed");
    }

    // sample a token and check if it is an EOG (end of generation token).
    // For the first kEosSuppressTokenWindow tokens of a completion that requested it, use the
    // EOG-suppressed chain instead — _responseNumTokens is still the pre-increment count here,
    // so this covers tokens 0..kEosSuppressTokenWindow-1.
    llama_sampler *activeSampler =
        (_suppressEarlyEos && _responseNumTokens < kEosSuppressTokenWindow) ? _samplerEosSuppressed : _sampler;

    // TEMPORARY DIAGNOSTIC: one line per token showing exactly what the selection logic decided,
    // so it's visible from logcat whether suppression is actually engaging for early tokens.
    LOGi("EOS-suppress diag: token#=%ld suppressEarlyEos=%d selected=%s",
         _responseNumTokens, _suppressEarlyEos,
         (activeSampler == _samplerEosSuppressed) ? "EOS_SUPPRESSED" : "NORMAL");

    // convert the integer token to its corresponding word-piece
    _currToken = llama_sampler_sample(activeSampler, _ctx, -1);
    if (llama_vocab_is_eog(llama_model_get_vocab(_model), _currToken)) {
        LOGi("EOS-suppress diag: token#=%ld sampled EOG token id=%d despite selected=%s",
             _responseNumTokens, _currToken,
             (activeSampler == _samplerEosSuppressed) ? "EOS_SUPPRESSED" : "NORMAL");
        addChatMessage(strdup(_response.data()), "assistant");
        _response.clear();
        return "[EOG]";
    }
    std::string piece = common_token_to_piece(_ctx, _currToken, true);
    auto end = ggml_time_us();
    _responseGenerationTime += (end - start);
    _responseNumTokens += 1;
    _cacheResponseTokens += piece;

    // re-init the batch with the newly predicted token
    // key, value pairs of all previous tokens have been cached
    // in the KV cache
    _batch->token = &_currToken;
    _batch->n_tokens = 1;

    if (_isValidUtf8(_cacheResponseTokens.c_str())) {
        _response += _cacheResponseTokens;
        std::string valid_utf8_piece = _cacheResponseTokens;
        _cacheResponseTokens.clear();

        // hard cap: force-stop once the max token budget for this completion is spent.
        // Still return this call's real piece — flag the stop so the *next* call finalizes
        // instead, otherwise this piece would be silently dropped from the caller's stream.
        if (_responseNumTokens >= _maxTokens) {
            LOGi("completionLoop: max_tokens (%d) reached after %zu response bytes, stopping",
                 _maxTokens, _response.size());
            _pendingStop = true;
            return valid_utf8_piece;
        }

        // repetition guard: bail out if the tail is a short substring looping 5+ times in a row.
        // Same deferred-stop handling as above so the triggering piece is still returned.
        if (_hasRepetition(_response)) {
            LOGi("completionLoop: repetition detected after %zu response bytes, stopping: \"%s\"",
                 _response.size(), _response.c_str());
            _pendingStop = true;
            return valid_utf8_piece;
        }

        return valid_utf8_piece;
    }

    return "";
}

void
LLMInference::stopCompletion() {
    if (_storeChats) {
        addChatMessage(_response.c_str(), "assistant");
    }
    _response.clear();
}

LLMInference::~LLMInference() {
    // free memory held by the message text in messages
    // (as we had used strdup() to create a malloc'ed copy)
    for (llama_chat_message &message: _messages) {
        free(const_cast<char *>(message.role));
        free(const_cast<char *>(message.content));
    }
    llama_free(_ctx);
    llama_model_free(_model);
    delete _batch;
    llama_sampler_free(_sampler);
    llama_sampler_free(_samplerEosSuppressed);
}

std::string
LLMInference::benchModel(int pp, int tg, int pl, int nr) {
    g_batch     = llama_batch_init(pp, 0, pl);
    auto pp_avg = 0.0;
    auto tg_avg = 0.0;
    auto pp_std = 0.0;
    auto tg_std = 0.0;

    const uint32_t n_ctx = llama_n_ctx(this->_ctx);
    LOGi("n_ctx = %d", n_ctx);

    int i, j;
    int nri;
    for (nri = 0; nri < nr; nri++) {
        LOGi("Benchmark prompt processing (pp = %d)", pp);

        common_batch_clear(g_batch);

        const int n_tokens = pp;
        for (i = 0; i < n_tokens; i++) {
            common_batch_add(g_batch, 1, i, { 0 }, false);
        }

        g_batch.logits[g_batch.n_tokens - 1] = true;
        llama_memory_clear(llama_get_memory(this->_ctx), false);

        const auto t_pp_start = ggml_time_us();
        if (llama_decode(this->_ctx, g_batch) != 0) {
            LOGe("llama_decode() failed during prompt processing");
        }
        const auto t_pp_end = ggml_time_us();

        // bench text generation

        LOGi("Benchmark text generation (tg = %d)", tg);

        llama_memory_clear(llama_get_memory(this->_ctx), false);
        const auto t_tg_start = ggml_time_us();
        for (i = 0; i < tg; i++) {
            common_batch_clear(g_batch);
            for (j = 0; j < pl; j++) {
                common_batch_add(g_batch, 0, i, { j }, true);
            }

            if (llama_decode(this->_ctx, g_batch) != 0) {
                LOGe("llama_decode() failed during text generation");
            }
        }
        const auto t_tg_end = ggml_time_us();

        llama_memory_clear(llama_get_memory(this->_ctx), false);

        const auto t_pp = double(t_pp_end - t_pp_start) / 1000000.0;
        const auto t_tg = double(t_tg_end - t_tg_start) / 1000000.0;

        const auto speed_pp = double(pp) / t_pp;
        const auto speed_tg = double(pl * tg) / t_tg;

        pp_avg += speed_pp;
        tg_avg += speed_tg;

        pp_std += speed_pp * speed_pp;
        tg_std += speed_tg * speed_tg;

        LOGi("pp %f t/s, tg %f t/s", speed_pp, speed_tg);
    }

    llama_batch_free(g_batch);

    pp_avg /= double(nr);
    tg_avg /= double(nr);

    if (nr > 1) {
        pp_std = sqrt(pp_std / double(nr - 1) - pp_avg * pp_avg * double(nr) / double(nr - 1));
        tg_std = sqrt(tg_std / double(nr - 1) - tg_avg * tg_avg * double(nr) / double(nr - 1));
    } else {
        pp_std = 0;
        tg_std = 0;
    }

    char model_desc[128];
    llama_model_desc(this->_model, model_desc, sizeof(model_desc));

    const auto model_size     = double(llama_model_size(this->_model)) / 1024.0 / 1024.0 / 1024.0;
    const auto model_n_params = double(llama_model_n_params(this->_model)) / 1e9;

    std::vector<std::string> backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto*       reg  = ggml_backend_reg_get(i);
        std::string name = ggml_backend_reg_name(reg);
        if (name != "CPU") {
            backends.push_back(ggml_backend_reg_name(reg));
        }
    }
    std::ostringstream str;
    for (size_t i = 0; i < backends.size(); i++) {
        str << backends[i];
        if (i < backends.size() - 1) {
            str << ",";
        }
    }
    const auto backend = str.str();

    std::stringstream result;
    result << std::setprecision(3);
    result << "| model | size | params | backend | test | t/s |\n";
    result << "| --- | --- | --- | --- | --- | --- |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | pp "
           << pp << " | " << pp_avg << " ± " << pp_std << " |\n";
    result << "| " << model_desc << " | " << model_size << "GiB | " << model_n_params << "B | " << backend << " | tg "
           << tg << " | " << tg_avg << " ± " << tg_std << " |\n";
    return result.str();
}
