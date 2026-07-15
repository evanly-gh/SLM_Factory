# training/slm_helpers.py
"""
Agent-facing interface for training and inference.
Called by the agent via the bash tool.
Equivalent to the paper's tinker_helpers.py.
"""
import os
import json
from training.lora_trainer import TrainingConfig, TrainingOutput, run_lora_training

# Lazy-loaded inference model cache: weights_ref -> (model, tokenizer).
# Limited to _MAX_CACHED models to avoid OOM on long runs (B47).
_inference_cache: dict = {}
_cache_order: list = []
_MAX_CACHED = 3

# GGUF inference cache: gguf_path -> Llama instance.
_gguf_cache: dict = {}
_gguf_cache_order: list = []


def train(
    dataset_path: str,
    base_model: str,
    nr_epochs: int,
    learning_rate: float,
    batch_size: int,
    lora_rank: int | None,
    output_dir: str = "artifacts",
    task_type: str = "classification",
) -> TrainingOutput:
    """
    Execute the full LoRA training loop.
    Returns TrainingOutput(weights_ref, gguf_path=None).
    Always trains from base_model — never from a prior checkpoint.
    task_type controls the prompt format used during training.
    """
    config = TrainingConfig(
        base_model=base_model,
        nr_epochs=nr_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lora_rank=lora_rank,
        task_type=task_type,
    )
    return run_lora_training(dataset_path, config, output_dir=output_dir, task_type=task_type)


def _is_adapter_only_checkpoint(path: str) -> bool:
    """
    Return True if *path* looks like a LoRA adapter-only directory (i.e. it
    contains adapter_config.json but no full model weights such as
    config.json + pytorch_model*.bin / model*.safetensors).
    """
    if not os.path.isdir(path):
        return False
    has_adapter_cfg = os.path.isfile(os.path.join(path, "adapter_config.json"))
    has_full_weights = any(
        os.path.isfile(os.path.join(path, f))
        for f in os.listdir(path)
        if f.endswith(".bin") or f.endswith(".safetensors")
    ) if os.path.isdir(path) else False
    return has_adapter_cfg and not has_full_weights


def infer(prompt: str, weights_ref: str, base_model: str, max_new_tokens: int = 50) -> str:
    """
    Load a checkpoint and generate text. Loads via Unsloth (not vanilla
    AutoModelForCausalLM): once `unsloth` is imported it globally patches the model
    classes (e.g. Qwen3Attention.apply_qkv), so a vanilla-loaded model crashes at
    generate. Unsloth's loader also transparently handles LoRA-adapter checkpoints.

    For adapter-only checkpoints (directory has adapter_config.json but no full
    weights), `base_model` must be provided so Unsloth can load the base weights
    before merging the adapter.  When `weights_ref` is an adapter-only path and
    `base_model` differs from `weights_ref`, the loader is called with
    `model_name=base_model` followed by `load_adapter`; otherwise `weights_ref`
    is used directly (full merged checkpoint).
    """
    import torch
    # Use (weights_ref, base_model) as the cache key so that the same adapter
    # path loaded on top of different base models never collides in the cache.
    cache_key = (weights_ref, base_model)
    if cache_key not in _inference_cache:
        from unsloth import FastLanguageModel
        from agent.logging_setup import quiet_ml_logging
        quiet_ml_logging()
        adapter_only = _is_adapter_only_checkpoint(weights_ref)
        if adapter_only and (not base_model or base_model == weights_ref):
            raise ValueError(
                f"weights_ref '{weights_ref}' appears to be an adapter-only checkpoint "
                f"(contains adapter_config.json but no full model weights). "
                f"Provide a valid `base_model` path so the base weights can be loaded "
                f"before the adapter is applied."
            )
        # Evict oldest cached model if at capacity
        while len(_inference_cache) >= _MAX_CACHED and _cache_order:
            evict_key = _cache_order.pop(0)
            old = _inference_cache.pop(evict_key, None)
            if old:
                model_evict, tok_evict = old
                del model_evict, tok_evict, old
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        # Multimodal base models ("Causal LM with Vision", e.g. Qwen3.5) must load via
        # FastVisionModel; plain FastLanguageModel routes text through the vision path
        # and crashes (B123/B136). Detected from the pool's `multimodal` flag.
        from training.lora_trainer import text_tokenizer, is_multimodal_model
        if is_multimodal_model(base_model):
            from unsloth import FastVisionModel as _Loader
        else:
            _Loader = FastLanguageModel
        if adapter_only:
            # Load the base model first, then apply the adapter on top.
            model, tokenizer = _Loader.from_pretrained(
                model_name=base_model, max_seq_length=512, load_in_4bit=False,
                trust_remote_code=True,
            )
            model.load_adapter(weights_ref)
        else:
            # Full merged checkpoint: weights_ref contains everything needed.
            model, tokenizer = _Loader.from_pretrained(
                model_name=weights_ref, max_seq_length=512, load_in_4bit=False,
                trust_remote_code=True,
            )
        # Multimodal loads return a processor; use the inner text tokenizer so the chat
        # template / tokenization never routes text through the vision path (B123).
        tokenizer = text_tokenizer(tokenizer)
        _Loader.for_inference(model)
        # We always cap generation with an explicit max_new_tokens at the call site.
        # Many chat models (e.g. Qwen3) also ship a generation_config.max_length (40960),
        # and when BOTH are set transformers logs a "Both max_new_tokens and max_length
        # seem to have been set" warning on EVERY generate() call. Clear the config-level
        # max_length so max_new_tokens is the single, unambiguous length control.
        try:
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.max_length = None
        except Exception:
            pass
        _inference_cache[cache_key] = (model, tokenizer)
        _cache_order.append(cache_key)

    model, tokenizer = _inference_cache[cache_key]
    # Train/serve parity (BUGS B115): training formats every example with the chat
    # template (lora_trainer.format_example → apply_chat_template). Inference must wrap
    # the prompt the same way — as a user turn with the assistant generation prompt
    # appended — or the model sees an out-of-distribution raw string and emits garbage,
    # so eval (classification/NER/generation) scores ~0 even after successful training.
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        prompt_text = prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def infer_batch(
    prompts: list[str],
    weights_ref: str,
    base_model: str,
    max_new_tokens: int = 50,
    max_workers: int = 20,
) -> list[str]:
    """
    Run inference over all prompts. SEQUENTIAL: Unsloth/torch keep process-global state and
    a single GPU serializes the work anyway, so threading the model is unsafe and pointless
    (see BUGS.md B18). `max_workers` is kept for signature compatibility and ignored.
    """
    return [infer(p, weights_ref, base_model, max_new_tokens) for p in prompts]


def infer_batch_gguf(
    prompts: list[str],
    gguf_path: str,
    max_new_tokens: int = 50,
) -> list[str]:
    """
    Run inference over all prompts using a GGUF file via llama-cpp-python.
    Used for quantized model evaluation (Q4_K_M, Q8_0) to get honest on-device
    accuracy scores — the same model format that ships on Android.

    Sequential inference only (same reasoning as infer_batch).
    Caches the Llama instance by gguf_path (max 3 entries).

    Raises:
        ImportError: if llama-cpp-python is not installed.
    """
    try:
        import llama_cpp
    except (ImportError, TypeError):
        # TypeError: Python 3.14+ raises TypeError when sys.modules[key]=None (used in tests)
        raise ImportError(
            "llama-cpp-python is required for GGUF inference. "
            "Install with: pip install llama-cpp-python"
        )

    if gguf_path not in _gguf_cache:
        while len(_gguf_cache) >= _MAX_CACHED and _gguf_cache_order:
            evict_key = _gguf_cache_order.pop(0)
            _gguf_cache.pop(evict_key, None)
        llama = llama_cpp.Llama(
            model_path=gguf_path,
            n_ctx=512,
            n_gpu_layers=0,  # CPU-only: matches Android on-device inference
            verbose=False,
        )
        _gguf_cache[gguf_path] = llama
        _gguf_cache_order.append(gguf_path)

    llama = _gguf_cache[gguf_path]
    results = []
    for prompt in prompts:
        response = llama(prompt, max_tokens=max_new_tokens, echo=False)
        results.append(response["choices"][0]["text"])
    return results
