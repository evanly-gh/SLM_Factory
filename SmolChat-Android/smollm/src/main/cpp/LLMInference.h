#pragma once
#include "chat.h"
#include "common.h"
#include "llama.h"
#include <string>
#include <vector>

class LLMInference {
    // llama.cpp-specific types
    llama_context* _ctx;
    llama_model*   _model;
    llama_sampler* _sampler;
    // Identical chain to _sampler plus a logit-bias stage suppressing this model's own
    // EOG token(s) (read dynamically at load time via llama_vocab_is_eog() — never hardcoded).
    // Used in place of _sampler for the first kEosSuppressTokenWindow tokens of a completion
    // when the caller requested it (see startCompletion()'s suppressEarlyEos parameter).
    llama_sampler* _samplerEosSuppressed;
    llama_token    _currToken;
    llama_batch*   _batch;

    llama_batch g_batch;

    // container to store user/assistant messages in the chat
    std::vector<llama_chat_message> _messages;
    // stores the string generated after applying
    // the chat-template to all messages in `_messages`
    std::vector<char> _formattedMessages;
    // stores the tokens for the last query
    // appended to `_messages`
    std::vector<llama_token> _promptTokens;
    const char*              _chatTemplate;

    // stores the complete response for the given query
    std::string _response;
    std::string _cacheResponseTokens;
    // whether to cache previous messages in `_messages`
    bool _storeChats;

    // response generation metrics
    int64_t _responseGenerationTime = 0;
    long    _responseNumTokens      = 0;

    // hard cap on tokens generated per completion, set by startCompletion()
    int _maxTokens = 256;

    // set when max_tokens/repetition triggers on a call that still returned a real piece;
    // the *next* completionLoop() call finalizes (records history + returns "[EOG]") without
    // decoding another token, so the triggering piece itself is never dropped.
    bool _pendingStop = false;

    // whether this completion should suppress EOG sampling for the first
    // kEosSuppressTokenWindow tokens, set by startCompletion(). Off by default so the manual
    // chat UI (which never passes this) is unaffected — only the headless benchmark path
    // requests it, to avoid premature stops observed on multi-step reasoning prompts.
    bool _suppressEarlyEos = false;
    static constexpr int kEosSuppressTokenWindow = 40;

    // length of context window consumed during the conversation
    int _nCtxUsed = 0;

    bool _isValidUtf8(const char* response);

    // true if the tail of `text` consists of a 10-20 char substring repeated 5+ times in a row
    static bool _hasRepetition(const std::string& text);

  public:
    void loadModel(const char* modelPath, float minP, float temperature, bool storeChats, long contextSize,
                   const char* chatTemplate, int nThreads, bool useMmap, bool useMlock);

    std::string benchModel(int pp, int tg, int pl, int nr);

    void addChatMessage(const char* message, const char* role);

    float getResponseGenerationTime() const;

    int getContextSizeUsed() const;

    // Returns true if Jinja template was used, false if legacy fallback was needed.
    // maxTokens caps the number of tokens completionLoop() will generate before it force-stops
    // (returns "[EOG]"), defaulting to 256 when not provided by the caller.
    // suppressEarlyEos, when true, blocks this model's EOG token(s) from being sampled for the
    // first kEosSuppressTokenWindow tokens (see _samplerEosSuppressed). Defaults to false.
    bool startCompletion(const char* query, int maxTokens = 256, bool suppressEarlyEos = false);

    std::string completionLoop();

    void stopCompletion();

    ~LLMInference();
};