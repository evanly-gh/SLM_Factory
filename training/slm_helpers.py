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


def infer(prompt: str, weights_ref: str, base_model: str, max_new_tokens: int = 50) -> str:
    """
    Load a checkpoint and generate text. Loads via Unsloth (not vanilla
    AutoModelForCausalLM): once `unsloth` is imported it globally patches the model
    classes (e.g. Qwen3Attention.apply_qkv), so a vanilla-loaded model crashes at
    generate. Unsloth's loader also transparently handles LoRA-adapter checkpoints.
    """
    import torch
    if weights_ref not in _inference_cache:
        from unsloth import FastLanguageModel
        # Evict oldest cached model if at capacity
        while len(_inference_cache) >= _MAX_CACHED and _cache_order:
            evict_key = _cache_order.pop(0)
            old = _inference_cache.pop(evict_key, None)
            if old:
                del old
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=weights_ref, max_seq_length=512, load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        _inference_cache[weights_ref] = (model, tokenizer)
        _cache_order.append(weights_ref)

    model, tokenizer = _inference_cache[weights_ref]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
