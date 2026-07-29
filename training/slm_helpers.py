# training/slm_helpers.py
"""
Agent-facing interface for training and inference.
Called by the agent via the bash tool.
Equivalent to the paper's tinker_helpers.py.
"""
import os
import time
from training.lora_trainer import TrainingConfig, TrainingOutput, run_lora_training

# Lazy-loaded inference model cache:
# (weights_ref, base_model, max_seq_length) -> (model, tokenizer).
# Capped at _MAX_CACHED=1 (B142): evaluation is sequential (one checkpoint at a time),
# and holding several FULL-PRECISION models resident — especially the 4B tier-3 models —
# fills VRAM. When VRAM runs low Unsloth/accelerate silently offload layers to the CPU
# ("Some parameters are on the meta device"), and Unsloth's fast-generate then crashes
# with `ValueError: Invalid target device: None`. Keeping only the current model resident
# (and emptying the CUDA cache on eviction) leaves the whole GPU for it.
_inference_cache: dict = {}
_cache_order: list = []
_MAX_CACHED = 1

# GGUF inference cache: gguf_path -> Llama instance.
_gguf_cache: dict = {}
_gguf_cache_order: list = []

_LONG_OUTPUT_TASKS = {"math_reasoning", "code_generation", "generation", "NER"}
_SHORT_EVAL_BATCH_SIZE = 16
_LONG_EVAL_BATCH_SIZE = 4
_EVAL_BATCH_SIZE_ENV = "SLM_EVAL_BATCH_SIZE"
_DEFAULT_MAX_SEQ_LENGTH = 4096
_MAX_SEQ_LENGTH_ENV = "SLM_MAX_SEQ_LENGTH"


def _is_qwen_model_id(model_id: str | None) -> bool:
    return bool(model_id and "qwen" in model_id.lower())


def _qwen_no_think_prompt(prompt: str, base_model: str) -> str:
    """Model-specific ChatML equivalent of HF non-thinking templates."""
    prefix = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if base_model == "Qwen/Qwen3-4B-Instruct-2507":
        # Non-thinking-only artifact: its HF template emits the plain assistant prefix.
        return prefix
    # Hybrid Qwen3/Qwen3.5 templates disable thinking by pre-filling an empty block.
    return prefix + "<think>\n\n</think>\n\n"


def train(
    dataset_path: str,
    base_model: str,
    nr_epochs: int,
    learning_rate: float,
    batch_size: int | None = None,
    lora_rank: int | None = None,
    output_dir: str = "artifacts",
    task_type: str = "classification",
    *,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.0,
    weight_decay: float = 0.01,
    micro_batch_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    effective_batch_size: int | None = None,
) -> TrainingOutput:
    """
    Execute the full LoRA training loop.
    Returns TrainingOutput(weights_ref, gguf_path=None).
    Always trains from base_model — never from a prior checkpoint.
    task_type controls the prompt format used during training.
    """
    from training.cuda_isolation import isolation_enabled, run_isolated

    # Resolve and strictly validate the legacy batch_size alias before crossing
    # the disposable-worker boundary. The payload contains only canonical fields.
    config = TrainingConfig(
        base_model=base_model,
        nr_epochs=nr_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lora_rank=lora_rank,
        task_type=task_type,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        weight_decay=weight_decay,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        effective_batch_size=effective_batch_size,
    )

    if isolation_enabled():
        payload = {
            "dataset_path": dataset_path,
            "base_model": base_model,
            "nr_epochs": config.nr_epochs,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "weight_decay": config.weight_decay,
            "micro_batch_size": config.micro_batch_size,
            "gradient_accumulation_steps": (
                config.gradient_accumulation_steps
            ),
            "effective_batch_size": config.effective_batch_size,
            "output_dir": output_dir,
            "task_type": task_type,
        }
        # Defense in depth: remove any explicit parent-side inference model before
        # the child takes ownership of the GPU. Worker exit is the hard cleanup boundary.
        clear_inference_cache()
        try:
            return run_isolated("train", payload)
        finally:
            clear_inference_cache()

    return _train_local(dataset_path, config, output_dir, task_type)


def _train_local(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str = "artifacts",
    task_type: str = "classification",
) -> TrainingOutput:
    """In-process implementation used by disposable workers and direct callers."""
    return run_lora_training(dataset_path, config, output_dir=output_dir, task_type=task_type)


def clear_inference_cache() -> None:
    """Evict ALL cached inference models and free their VRAM.

    Call this when the pipeline moves on from a model for good (e.g. on escalation) so a
    stale, no-longer-needed model never sits in VRAM competing with the next (often larger)
    model — the exact condition that pushed the 4B tier-3 model into CPU/meta offload and
    crashed generate (B142). This is the explicit "clear what we moved on from" complement
    to the _MAX_CACHED=1 LRU eviction.
    """
    _inference_cache.clear()
    _cache_order.clear()
    _gguf_cache.clear()
    _gguf_cache_order.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _is_adapter_only_checkpoint(path: str) -> bool:
    """
    Return True if *path* looks like a LoRA adapter-only directory (i.e. it
    contains adapter_config.json but no full model weights such as
    config.json + pytorch_model*.bin / model*.safetensors).
    """
    if not os.path.isdir(path):
        return False
    has_adapter_cfg = os.path.isfile(os.path.join(path, "adapter_config.json"))

    def _is_full_model_weight(filename: str) -> bool:
        return (
            filename == "pytorch_model.bin"
            or (
                filename.startswith("pytorch_model-")
                and filename.endswith(".bin")
            )
            or filename == "model.safetensors"
            or (
                filename.startswith("model-")
                and filename.endswith(".safetensors")
            )
        )

    has_full_weights = any(
        os.path.isfile(os.path.join(path, f))
        for f in os.listdir(path)
        if _is_full_model_weight(f)
    )
    return has_adapter_cfg and not has_full_weights


def _configure_inference_tokenizer(tokenizer):
    """Configure deterministic decoder-only batching on the text tokenizer."""
    tokenizer.padding_side = "left"
    if (
        getattr(tokenizer, "pad_token", None) is None
        or getattr(tokenizer, "pad_token_id", None) is None
    ):
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is None or eos_token_id is None:
            raise RuntimeError(
                "Inference tokenizer has no pad token and no EOS token to use for padding."
            )
        tokenizer.pad_token = eos_token
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token_id = eos_token_id
    return tokenizer


def _inference_max_seq_length() -> int:
    raw_value = os.environ.get(
        _MAX_SEQ_LENGTH_ENV,
        str(_DEFAULT_MAX_SEQ_LENGTH),
    )
    try:
        max_seq_length = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{_MAX_SEQ_LENGTH_ENV} must be a positive integer, got {raw_value!r}."
        ) from exc
    if max_seq_length < 1:
        raise ValueError(
            f"{_MAX_SEQ_LENGTH_ENV} must be a positive integer, got {raw_value!r}."
        )
    return max_seq_length


def _load_inference_model(
    weights_ref: str,
    base_model: str,
    max_seq_length: int,
):
    """Load or retrieve the cached Unsloth model and unwrapped text tokenizer."""
    # Include max_seq_length so a process-level config change cannot reuse a model
    # initialized with a smaller context window.
    cache_key = (weights_ref, base_model, max_seq_length)
    if cache_key not in _inference_cache:
        adapter_only = _is_adapter_only_checkpoint(weights_ref)
        if adapter_only and (not base_model or base_model == weights_ref):
            raise ValueError(
                f"weights_ref '{weights_ref}' appears to be an adapter-only checkpoint "
                f"(contains adapter_config.json but no full model weights). "
                f"Provide a valid `base_model` path so the base weights can be loaded "
                f"before the adapter is applied."
            )
        from training.lora_trainer import (
            _ensure_model_cached,
            is_multimodal_model,
            text_tokenizer,
        )

        # The first zero-shot difficulty probe reaches inference before training.
        # Prefetch both references here so that path receives the same pinned,
        # verified snapshot handling as training. Local checkpoints are no-ops.
        seen_model_refs = set()
        for model_ref in (base_model, weights_ref):
            if model_ref and model_ref not in seen_model_refs:
                _ensure_model_cached(model_ref)
                seen_model_refs.add(model_ref)

        from unsloth import FastLanguageModel
        from agent.logging_setup import quiet_ml_logging

        quiet_ml_logging()
        # Evict oldest cached model if at capacity.
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
        # FastVisionModel. The processor returned by that loader is unwrapped below so
        # text-only inference never enters the image/video processing path.
        if is_multimodal_model(base_model):
            from unsloth import FastVisionModel as _Loader
        else:
            _Loader = FastLanguageModel
        if adapter_only:
            model, tokenizer = _Loader.from_pretrained(
                model_name=base_model,
                max_seq_length=max_seq_length,
                load_in_4bit=False,
                trust_remote_code=True,
            )
            model.load_adapter(weights_ref)
        else:
            model, tokenizer = _Loader.from_pretrained(
                model_name=weights_ref,
                max_seq_length=max_seq_length,
                load_in_4bit=False,
                trust_remote_code=True,
            )
        tokenizer = _configure_inference_tokenizer(text_tokenizer(tokenizer))
        _Loader.for_inference(model)
        # max_new_tokens is the sole length control at every generate call.
        try:
            if getattr(model, "generation_config", None) is not None:
                model.generation_config.max_length = None
        except Exception:
            pass
        _inference_cache[cache_key] = (model, tokenizer)
        _cache_order.append(cache_key)

    model, tokenizer = _inference_cache[cache_key]
    return model, _configure_inference_tokenizer(tokenizer)


def _render_inference_prompt(tokenizer, prompt: str, base_model: str) -> str:
    """Apply the exact explicit non-thinking user/assistant template used in training."""
    if getattr(tokenizer, "chat_template", None):
        from training.lora_trainer import apply_non_thinking_chat_template

        return apply_non_thinking_chat_template(
            tokenizer,
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
    if _is_qwen_model_id(base_model):
        raise RuntimeError(
            f"Qwen tokenizer for {base_model!r} has no chat template; "
            "cannot enforce non-thinking train/inference parity."
        )
    return prompt


def _generate_continuations(
    model,
    tokenizer,
    rendered_prompts: list[str],
    max_new_tokens: int,
    *,
    padding: bool,
    max_seq_length: int,
    prompt_offset: int = 0,
) -> list[str]:
    """Generate and decode each row after its shared padded input width."""
    import torch

    tokenizer_input = rendered_prompts if padding else rendered_prompts[0]
    tokenizer_kwargs = {"return_tensors": "pt", "truncation": False}
    if padding:
        tokenizer_kwargs["padding"] = True
    inputs = tokenizer(tokenizer_input, **tokenizer_kwargs)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        try:
            prompt_lengths = [
                int(value)
                for value in attention_mask.sum(dim=1).tolist()
            ]
        except (AttributeError, TypeError):
            prompt_lengths = [
                sum(int(token) for token in row)
                for row in attention_mask
            ]
    else:
        prompt_lengths = [len(row) for row in inputs["input_ids"]]
    input_budget = max_seq_length - max_new_tokens
    if input_budget < 1:
        raise ValueError(
            f"Requested {max_new_tokens} output tokens leaves no input budget "
            f"inside configured max sequence length {max_seq_length}."
        )
    for row_index, prompt_length in enumerate(prompt_lengths):
        if prompt_length > input_budget:
            prompt_index = prompt_offset + row_index
            raise ValueError(
                f"Rendered prompt index {prompt_index} contains {prompt_length} tokens, "
                f"exceeding input budget {input_budget} after reserving "
                f"{max_new_tokens} output tokens inside configured max sequence "
                f"length {max_seq_length}. "
                "No truncation was applied, so shorter rows were not generated with a "
                f"corrupted batch. Shorten the prompt or increase {_MAX_SEQ_LENGTH_ENV} "
                "and ensure the model supports that context length."
            )
    inputs = inputs.to(model.device)
    input_width = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [
        tokenizer.decode(row[input_width:], skip_special_tokens=True)
        for row in outputs
    ]


def _eval_batch_size(task_type: str) -> int:
    override = os.environ.get(_EVAL_BATCH_SIZE_ENV)
    if override is not None:
        try:
            batch_size = int(override)
        except ValueError as exc:
            raise ValueError(
                f"{_EVAL_BATCH_SIZE_ENV} must be a positive integer, got {override!r}."
            ) from exc
        if batch_size < 1:
            raise ValueError(
                f"{_EVAL_BATCH_SIZE_ENV} must be a positive integer, got {override!r}."
            )
        return batch_size
    return (
        _LONG_EVAL_BATCH_SIZE
        if task_type in _LONG_OUTPUT_TASKS
        else _SHORT_EVAL_BATCH_SIZE
    )


def _is_cuda_oom(exc: BaseException, torch_module) -> bool:
    oom_types = []
    for owner in (torch_module, getattr(torch_module, "cuda", None)):
        oom_type = getattr(owner, "OutOfMemoryError", None)
        if isinstance(oom_type, type):
            oom_types.append(oom_type)
    if oom_types and isinstance(exc, tuple(oom_types)):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _try_generate_continuations(
    model,
    tokenizer,
    rendered_prompts: list[str],
    max_new_tokens: int,
    *,
    max_seq_length: int,
    prompt_offset: int,
    torch_module,
) -> tuple[list[str] | None, dict | None]:
    """Return OOM details without retaining the exception or its tensor traceback."""
    try:
        return (
            _generate_continuations(
                model,
                tokenizer,
                rendered_prompts,
                max_new_tokens,
                padding=True,
                max_seq_length=max_seq_length,
                prompt_offset=prompt_offset,
            ),
            None,
        )
    except Exception as exc:
        if not _is_cuda_oom(exc, torch_module):
            raise
        return None, {
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
        }


def _clear_cuda_oom_cache(torch_module) -> None:
    """Release failed-batch temporaries without evicting the cached model."""
    try:
        import gc

        gc.collect()
    finally:
        try:
            torch_module.cuda.empty_cache()
        except Exception:
            pass


def infer(prompt: str, weights_ref: str, base_model: str, max_new_tokens: int = 50) -> str:
    """
    Load a checkpoint and greedily generate one non-thinking chat continuation.

    The cached loader and prompt renderer are shared with ``infer_batch`` so one-row
    and batched evaluation use the same model, text tokenizer, and chat framing.
    """
    from training.cuda_isolation import isolation_enabled, run_isolated

    if isolation_enabled():
        return run_isolated(
            "infer",
            {
                "prompt": prompt,
                "weights_ref": weights_ref,
                "base_model": base_model,
                "max_new_tokens": max_new_tokens,
            },
        )

    max_seq_length = _inference_max_seq_length()
    model, tokenizer = _load_inference_model(
        weights_ref,
        base_model,
        max_seq_length,
    )
    rendered = _render_inference_prompt(tokenizer, prompt, base_model)
    return _generate_continuations(
        model,
        tokenizer,
        [rendered],
        max_new_tokens,
        padding=False,
        max_seq_length=max_seq_length,
    )[0]


def _infer_batch_local(
    prompts: list[str],
    weights_ref: str,
    base_model: str,
    max_new_tokens: int,
    task_type: str,
) -> list[str]:
    import torch
    from agent.timing import TimingEvent, record_timing_event

    started = time.perf_counter()
    initial_batch_size = None
    max_seq_length = None
    batch_size = None
    attempted_batch_sizes: list[int] = []
    completed_batch_sizes: list[int] = []
    oom_retries = 0
    status = "success"
    metadata = {
        "task_type": task_type,
        "prompt_count": len(prompts),
        "max_new_tokens": max_new_tokens,
        "max_seq_length": max_seq_length,
        "initial_batch_size": initial_batch_size,
        "attempted_batch_sizes": attempted_batch_sizes,
        "completed_batch_sizes": completed_batch_sizes,
        "oom_retries": oom_retries,
    }
    try:
        initial_batch_size = _eval_batch_size(task_type)
        batch_size = initial_batch_size
        max_seq_length = _inference_max_seq_length()
        metadata.update(
            {
                "max_seq_length": max_seq_length,
                "initial_batch_size": initial_batch_size,
            }
        )
        model, tokenizer = _load_inference_model(
            weights_ref,
            base_model,
            max_seq_length,
        )
        results: list[str] = []
        offset = 0
        while offset < len(prompts):
            current_prompts = prompts[offset:offset + batch_size]
            attempted_batch_sizes.append(len(current_prompts))
            rendered_prompts = [
                _render_inference_prompt(tokenizer, prompt, base_model)
                for prompt in current_prompts
            ]
            current_results, oom_details = _try_generate_continuations(
                model,
                tokenizer,
                rendered_prompts,
                max_new_tokens,
                max_seq_length=max_seq_length,
                prompt_offset=offset,
                torch_module=torch,
            )
            if oom_details is not None:
                _clear_cuda_oom_cache(torch)
                if len(current_prompts) == 1:
                    raise RuntimeError(
                        "CUDA OOM during batched inference at batch_size=1 "
                        f"(task_type={task_type}, base_model={base_model!r}, "
                        f"weights_ref={weights_ref!r}, prompt_index={offset}, "
                        f"max_new_tokens={max_new_tokens}). Reduce max_new_tokens "
                        "or select a smaller model. Original error: "
                        f"{oom_details['error_type']}: {oom_details['error']}"
                    ) from None
                batch_size = max(1, len(current_prompts) // 2)
                oom_retries += 1
                metadata["oom_retries"] = oom_retries
                continue
            assert current_results is not None
            results.extend(current_results)
            completed_batch_sizes.append(len(current_prompts))
            offset += len(current_prompts)
        return results
    except BaseException as exc:
        status = "error"
        metadata.update(
            {"error_type": type(exc).__name__, "error": str(exc)[:500]}
        )
        raise
    finally:
        latency_ms = (time.perf_counter() - started) * 1000
        metadata.update(
            {
                "final_batch_size": batch_size,
                "latency_ms": round(latency_ms, 3),
            }
        )
        record_timing_event(
            TimingEvent(
                kind="inference",
                name="infer_batch",
                duration_ms=latency_ms,
                status=status,
                metadata=metadata,
            )
        )


def infer_batch(
    prompts: list[str],
    weights_ref: str,
    base_model: str,
    max_new_tokens: int = 50,
    max_workers: int = 20,
    task_type: str = "classification",
) -> list[str]:
    """
    Greedily generate ordered continuations in padded batches on one cached model.

    ``max_workers`` remains for API compatibility; generation is deliberately
    single-process. The task-aware default batch size is 16 for short classification
    outputs and 4 for long generation/NER/code outputs, overrideable with
    ``SLM_EVAL_BATCH_SIZE``. CUDA OOMs halve the active batch and retry in place.
    """
    if not prompts:
        return []

    from training.cuda_isolation import isolation_enabled, run_isolated

    if isolation_enabled():
        return run_isolated(
            "infer_batch",
            {
                "prompts": prompts,
                "weights_ref": weights_ref,
                "base_model": base_model,
                "max_new_tokens": max_new_tokens,
                "max_workers": max_workers,
                "task_type": task_type,
            },
        )
    return _infer_batch_local(
        prompts,
        weights_ref,
        base_model,
        max_new_tokens,
        task_type,
    )


def infer_batch_gguf(
    prompts: list[str],
    gguf_path: str,
    max_new_tokens: int = 50,
    base_model: str | None = None,
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

    max_seq_length = _inference_max_seq_length()
    cache_key = (gguf_path, max_seq_length)
    if cache_key not in _gguf_cache:
        while len(_gguf_cache) >= _MAX_CACHED and _gguf_cache_order:
            evict_key = _gguf_cache_order.pop(0)
            _gguf_cache.pop(evict_key, None)
        # GPU offload for SPEED without changing measured accuracy (B161): the accuracy of a
        # quantized GGUF is a property of its weights + tokenizer + greedy decoding, NOT the
        # compute device — running the SAME Q4_K_M/Q8_0 file on GPU yields the same predictions
        # as CPU, just ~10-50x faster (the phone runs it on CPU/NPU; only LATENCY differs, and
        # that is measured separately by the on-device backend). This makes the 800-example eval
        # tractable inside the loop. Requires a CUDA-enabled llama-cpp-python build; falls back
        # to CPU automatically if the wheel is CPU-only. Override layer count with SLM_GGUF_GPU_LAYERS
        # (-1 = offload all; 0 = force CPU to exactly mirror device compute).
        import os as _os
        _gpu_layers = int(_os.environ.get("SLM_GGUF_GPU_LAYERS", "-1"))
        try:
            llama = llama_cpp.Llama(
                model_path=gguf_path,
                n_ctx=max_seq_length,
                n_gpu_layers=_gpu_layers,
                verbose=False,
            )
        except Exception:
            # CPU-only wheel (or no CUDA device) → fall back to CPU inference.
            llama = llama_cpp.Llama(
                model_path=gguf_path,
                n_ctx=max_seq_length,
                n_gpu_layers=0,
                verbose=False,
            )
        _gguf_cache[cache_key] = llama
        _gguf_cache_order.append(cache_key)

    llama = _gguf_cache[cache_key]
    results = []
    import inspect

    try:
        chat_parameters = inspect.signature(
            llama.create_chat_completion
        ).parameters
    except (TypeError, ValueError):
        chat_parameters = {}
    supports_mode_kwargs = "chat_template_kwargs" in chat_parameters

    def _validate_gguf_budget(rendered_prompt: str, prompt_index: int) -> None:
        tokenize = getattr(llama, "tokenize", None)
        if not callable(tokenize):
            return
        tokens = tokenize(rendered_prompt.encode("utf-8"), add_bos=True)
        input_budget = max_seq_length - max_new_tokens
        if len(tokens) > input_budget:
            raise ValueError(
                f"Rendered GGUF prompt index {prompt_index} contains {len(tokens)} "
                f"tokens, exceeding input budget {input_budget} after reserving "
                f"{max_new_tokens} output tokens inside configured max sequence "
                f"length {max_seq_length}."
            )

    for prompt in prompts:
        # Newer llama-cpp-python versions forward chat_template_kwargs to the
        # embedded template. Installed 0.3.34 (inspected 2026-07-21) does not,
        # so Qwen uses the exact no-thinking ChatML generation prefix.
        if supports_mode_kwargs:
            _validate_gguf_budget(prompt, len(results))
            resp = llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens, temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
            )
            results.append(resp["choices"][0]["message"]["content"])
        elif _is_qwen_model_id(base_model):
            rendered_prompt = _qwen_no_think_prompt(prompt, base_model)
            _validate_gguf_budget(rendered_prompt, len(results))
            resp = llama(
                rendered_prompt,
                max_tokens=max_new_tokens,
                temperature=0.0,
                echo=False,
                stop=["<|im_end|>"],
            )
            results.append(resp["choices"][0]["text"])
        else:
            raise RuntimeError(
                "Installed llama-cpp-python cannot enforce non-thinking chat-template "
                f"kwargs for {base_model or 'an unspecified model'}; refusing to mix "
                "prompt modes."
            )
    return results
