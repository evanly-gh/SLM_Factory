#include "LLMInference.h"
#include <android/log.h>
#include <cstring>
#include <iomanip>
#include <iostream>

#define TAG "[SmolLMAndroid-Cpp]"
#define LOGi(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGe(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

void
LLMInference::loadModel(const char *model_path, float minP, float temperature, bool storeChats, long contextSize,
                        const char *chatTemplate, int nThreads, bool useMmap, bool useMlock) {
    LOGi("loading model with"
         "\n\tmodel_path = %s"
         "\n\tminP = %f"
         "\n\ttemperature = %f"
         "\n\tstoreChats = %d"
         "\n\tcontextSize = %li"
         "\n\tchatTemplate = %s"
         "\n\tnThreads = %d"
         "\n\tuseMmap = %d"
         "\n\tuseMlock = %d",
         model_path, minP, temperature, storeChats, contextSize, chatTemplate, nThreads, useMmap, useMlock);

    // load dynamic backends
    ggml_backend_load_all();

    // create an instance of llama_model
    llama_model_params model_params = llama_model_default_params();
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
