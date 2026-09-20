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

# Eval batch size and context ceiling are per-TASK and declared on its spec. A CUDA OOM halves
# the active batch and retries in place, so the cost of a task aiming high is one retry, not a
# failed eval.
_EVAL_BATCH_SIZE_ENV = "SLM_EVAL_BATCH_SIZE"
# Only for the two paths that genuinely have no task in hand: the single-prompt helper used by
# ad-hoc probes, and a GGUF load whose caller did not name one. Every task-aware path reads the
# ceiling from the task's spec.
_DEFAULT_INFERENCE_SEQ_LENGTH = 4096
_MAX_SEQ_LENGTH_ENV = "SLM_MAX_SEQ_LENGTH"



def _is_qwen_model_id(model_id: str | None) -> bool:
    return bool(model_id and "qwen" in model_id.lower())


def _qwen_no_think_prompt(prompt: str, base_model: str) -> str:
    """Model-specific ChatML equivalent of HF non-thinking templates.

    These strings must match the SERVING template — the one belonging to the official base model that
    the GGUF is merged from — because that is what governs the deployed artifact. They are correct as
    written; the hazard is on the training side, and it is worth naming here because this function is
    what a reader will suspect first (B290).

    `Qwen/Qwen3-4B-Instruct-2507` is thinking-free, and its official template renders a bare assistant
    prefix. Unsloth's 4-bit mirror, which is what `FastLanguageModel.from_pretrained` actually loads,
    ships a template that inserts `<think>\\n\\n</think>\\n\\n` before the assistant content anyway. So
    training rendered targets containing a think block that this function — correctly, for the served
    model — does not pre-fill, and the fine-tuned model spent its first generated tokens emitting the
    scaffolding straight into the scored output. `_assert_train_serve_prefix_alignment` in
    `training/lora_trainer.py` now fails loudly rather than letting that ship.
    """
    prefix = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if base_model == "Qwen/Qwen3-4B-Instruct-2507":
        # Thinking-free artifact: its official HF template emits the plain assistant prefix.
        return prefix
    # Hybrid Qwen3/Qwen3.5 templates disable thinking by pre-filling an empty block.
    return prefix + "<think>\n\n</think>\n\n"


_serving_tokenizer_cache: dict[str, object | None] = {}


def _serving_tokenizer(base_model: str):
    """The SERVED model's tokenizer, loaded once per model id. `None` if it cannot be loaded.

    Only used to read a chat template, so a failure here is recoverable — the caller falls back to
    a plain completion prompt rather than raising.
    """
    if base_model in _serving_tokenizer_cache:
        return _serving_tokenizer_cache[base_model]
    tokenizer = None
    if base_model:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        except Exception:
            tokenizer = None
    _serving_tokenizer_cache[base_model] = tokenizer
    return tokenizer


def _serving_prompt_prefix(prompt: str, base_model: str) -> str:
    """Render `prompt` the way the DEPLOYED artifact expects to receive it.

    This is the fallback used when the installed llama-cpp-python cannot pass
    `chat_template_kwargs` through `create_chat_completion` — which is every version we have run,
    including 0.3.34, so in practice it is the only path quantized eval takes.

    It used to be `_qwen_no_think_prompt` unconditionally, with a hard refusal for anything that
    was not a Qwen. That was the correct guard for a Qwen-only pool (serving a Gemma a ChatML
    prompt is silent train/serve skew, the B290 failure mode) but it is the wrong general rule:
    the model's own tokenizer already knows the answer. Qwen keeps its hand-written bytes so no
    previously measured score moves; everything else renders from its own template.
    """
    if _is_qwen_model_id(base_model):
        return _qwen_no_think_prompt(prompt, base_model)

    tokenizer = _serving_tokenizer(base_model)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    # A BASE checkpoint with no chat template. Training took the same plain-text branch
    # (`lora_trainer._training_turn`'s fallback), so sending the bare prompt is what keeps the two
    # aligned. Wrapping it in someone else's turn markers is what would break parity.
    return prompt


def _serving_stop_tokens(base_model: str) -> list[str]:
    """Turn-end strings for the raw-completion path, derived from the served model.

    `<|im_end|>` is ChatML and is right for Qwen and for SmolLM2; Gemma ends a turn with
    `<end_of_turn>`. Hardcoding either one truncates nothing on the other family, which shows up
    as the model running to `max_tokens` on every row and the scorer reading trailing garbage.
    """
    if _is_qwen_model_id(base_model):
        return ["<|im_end|>"]

    stops: list[str] = []
    tokenizer = _serving_tokenizer(base_model)
    for candidate in (
        getattr(tokenizer, "eos_token", None),
        *(t for t in ("<|im_end|>", "<end_of_turn>", "<|endoftext|>")),
    ):
        if isinstance(candidate, str) and candidate and candidate not in stops:
            stops.append(candidate)
    return stops or ["<|im_end|>"]


def train(
    dataset_path: str,
    base_model: str,
    nr_epochs: int,
    learning_rate: float,
    batch_size: int | None = None,
    lora_rank: int | None = None,
    output_dir: str = "artifacts",
    task: str = "",
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
    task controls the prompt format used during training.
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
        task=task,
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
            "task": task,
        }
        # Defense in depth: remove any explicit parent-side inference model before
        # the child takes ownership of the GPU. Worker exit is the hard cleanup boundary.
        clear_inference_cache()
        try:
            return run_isolated("train", payload)
        finally:
            clear_inference_cache()

    return _train_local(dataset_path, config, output_dir, task)


def _train_local(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str = "artifacts",
    task: str = "",
) -> TrainingOutput:
    """In-process implementation used by disposable workers and direct callers."""
    return run_lora_training(dataset_path, config, output_dir=output_dir, task=task)


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
    # MNN models are CPU-resident rather than VRAM-resident, so this frees host RAM rather than
    # GPU memory — but the reason is the same one the comment above gives: a model the pipeline has
    # moved on from must not stay loaded while the next, usually larger, one is loaded beside it.
    _mnn_cache.clear()
    _mnn_cache_order.clear()
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


def _assert_adapter_is_live(model, weights_ref: str) -> None:
    """Refuse to score a LoRA adapter that cannot change a single logit.

    LoRA computes ``B @ A``, and B is zero-initialised, so an adapter whose B matrices are ALL
    zero contributes exactly nothing. That failure is invisible from the outside: inference
    runs, produces fluent output, and scores — just the BASE model's score, reported as the
    fine-tuned one. In run 38303490 it cost five iterations and two orchestrator decisions
    before anyone noticed the trajectory was 0.5414 four times in a row, bit for bit.

    Checking is a few microseconds against a load that takes seconds, so it is always on.
    """
    b_params = [
        param for name, param in model.named_parameters() if "lora_B" in name
    ]
    if not b_params:
        raise RuntimeError(
            f"Adapter checkpoint {weights_ref!r} loaded with NO lora_B parameters attached — "
            f"the adapter is not present in the module tree, so inference would silently "
            f"score the base model."
        )
    if not any(param.detach().float().abs().sum().item() > 0 for param in b_params):
        raise RuntimeError(
            f"Adapter checkpoint {weights_ref!r} attached {len(b_params)} lora_B tensors and "
            f"every one is ZERO, which makes the adapter mathematically an identity: eval "
            f"would score the base model and report it as fine-tuned. Either the checkpoint "
            f"never trained, or the loader failed to populate it from disk."
        )


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


def task_max_seq_length(task: str) -> int:
    """Context ceiling for ``task``, from its spec. An explicit SLM_MAX_SEQ_LENGTH always wins.

    Context is allocated, not measured: KV cache and position buffers are sized from this number
    whatever the rows actually contain, so it is a per-task ceiling with real headroom over
    observed lengths rather than a tight fit. A row that would exceed it raises rather than
    truncating. This governs the SMALL MODEL being trained and evaluated; the orchestrator's
    Claude context is a separate budget nothing here affects.
    """
    from tasks import get_task

    override = os.environ.get(_MAX_SEQ_LENGTH_ENV)
    if override is not None:
        return _validated_max_seq_length(override)
    return get_task(task).max_seq_length


def _validated_max_seq_length(raw_value: str) -> int:
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
            # PeftModel.from_pretrained, NOT model.load_adapter (B254). On an Unsloth-patched
            # model, `load_adapter` builds the LoRA modules and marks them active but never
            # populates them from the checkpoint: measured 0 of 186 lora_B tensors non-zero,
            # against 186 of 186 for the same file through PeftModel. Because LoRA initializes
            # B to zeros, that adapter is exactly an identity — eval silently scored the BASE
            # model and reported it as fine-tuned.
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, weights_ref)
            _assert_adapter_is_live(model, weights_ref)
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


def _eval_batch_size(task: str) -> int:
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
    from tasks import get_task

    return get_task(task).eval_batch_size


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

    max_seq_length = _DEFAULT_INFERENCE_SEQ_LENGTH
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
    task: str,
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
        "task": task,
        "prompt_count": len(prompts),
        "max_new_tokens": max_new_tokens,
        "max_seq_length": max_seq_length,
        "initial_batch_size": initial_batch_size,
        "attempted_batch_sizes": attempted_batch_sizes,
        "completed_batch_sizes": completed_batch_sizes,
        "oom_retries": oom_retries,
    }
    try:
        initial_batch_size = _eval_batch_size(task)
        batch_size = initial_batch_size
        max_seq_length = task_max_seq_length(task) if task else _DEFAULT_INFERENCE_SEQ_LENGTH
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
                        f"(task={task}, base_model={base_model!r}, "
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
    task: str = "",
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
                "task": task,
            },
        )
    return _infer_batch_local(
        prompts,
        weights_ref,
        base_model,
        max_new_tokens,
        task,
    )


def _label_score_rows(tokenizer, rendered_prompt: str, labels: list[str]) -> list[tuple[list, int]]:
    """One `(token_ids, n_label_tokens)` pair per label, teacher-forced onto the prompt.

    `add_special_tokens=False` on the label half is not optional: the prompt has already been
    rendered through the chat template, so any BOS the tokenizer would prepend belongs to the
    prompt and adding a second one mid-sequence scores a sequence the model will never see.
    """
    prompt_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
    rows = []
    for label in labels:
        label_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if not label_ids:
            # A label that tokenizes to nothing cannot be scored. Recorded as zero-length so the
            # caller assigns it -inf rather than silently averaging over an empty slice.
            rows.append((list(prompt_ids), 0))
            continue
        rows.append((list(prompt_ids) + list(label_ids), len(label_ids)))
    return rows


def _score_label_batch(model, tokenizer, rows, torch_module) -> list[float]:
    """Mean per-token logprob of each row's label continuation, in one padded forward pass."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0
    width = max(len(ids) for ids, _n in rows)
    input_ids, attention_mask = [], []
    for ids, _n in rows:
        # RIGHT padding, unlike the generation path's left padding. Generation needs every
        # sequence's last real token flush against the output, so it pads left; scoring needs
        # known absolute positions for the label tokens, which right padding gives directly.
        input_ids.append(list(ids) + [pad_id] * (width - len(ids)))
        attention_mask.append([1] * len(ids) + [0] * (width - len(ids)))
    device = getattr(model, "device", None)
    ids_tensor = torch_module.tensor(input_ids)
    mask_tensor = torch_module.tensor(attention_mask)
    if device is not None:
        ids_tensor = ids_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

    with torch_module.no_grad():
        logits = model(input_ids=ids_tensor, attention_mask=mask_tensor).logits
    # Position i's logits predict token i+1, so the logprob of token t at index i comes from
    # logits[i - 1]. Shifting once here keeps that offset in one place.
    logprobs = torch_module.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = ids_tensor[:, 1:]
    gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    out: list[float] = []
    for row_index, (ids, n_label) in enumerate(rows):
        if n_label <= 0:
            out.append(float("-inf"))
            continue
        end = len(ids) - 1            # last index into the shifted arrays
        start = end - n_label         # first label token's shifted index
        total = gathered[row_index, start:end].sum().item()
        # LENGTH-NORMALIZED, i.e. mean logprob per label token. Within one label this is a
        # division by a constant, so it cannot change that label's average precision — AP depends
        # only on the ranking of examples for a fixed label. It is done anyway because the raw
        # sum makes short labels systematically look likelier than long ones, which would make
        # the per-label scores useless to read side by side in the diagnostics.
        out.append(total / n_label)
    return out


def infer_label_scores_batch(
    prompts: list[str],
    labels: list[str],
    weights_ref: str,
    base_model: str,
    task: str = "",
) -> list[dict[str, float]]:
    """Score every candidate label against every prompt. One dict of label -> logprob per prompt.

    WHY THIS EXISTS AT ALL
        The rest of the harness GENERATES text and parses it, which yields a hard decision: this
        label or not. A threshold-free metric needs a RANKING, and generation cannot supply one.
        GoEmotions' published headline is macro average precision — chosen precisely because
        macro-F1 over 28 labels is a thresholding artifact that moves several points between a
        fixed 0.5, a fixed 0.3 and a dev-tuned sweep — so without per-label scores the honest
        metric for that task is not computable.

        This is the only new inference capability the suite needs, and it is deliberately confined
        to the report pass. The agent loop still selects on the 7-way Ekman grouping scored from
        ordinary generation, so nothing in the hot path changes and no other task is affected.

    WHAT IT DOES NOT DO
        It does not reuse the prompt's KV cache across the 28 labels, which is the obvious
        optimization: all labels share the prompt prefix, so a single prefill could serve all of
        them. Skipped on purpose. Hand-rolling cache reuse puts the number this project reports
        behind code whose bugs would shift scores rather than raise, and this runs once per run
        on 5,427 rows. Correctness of a published metric beats the speed of a one-off pass.
    """
    if not prompts or not labels:
        return []

    from training.cuda_isolation import isolation_enabled, run_isolated

    if isolation_enabled():
        return run_isolated(
            "infer_label_scores",
            {
                "prompts": prompts,
                "labels": labels,
                "weights_ref": weights_ref,
                "base_model": base_model,
                "task": task,
            },
        )
    return _infer_label_scores_local(prompts, labels, weights_ref, base_model, task)


def _infer_label_scores_local(
    prompts: list[str],
    labels: list[str],
    weights_ref: str,
    base_model: str,
    task: str,
) -> list[dict[str, float]]:
    import torch

    from agent.timing import TimingEvent, record_timing_event

    started = time.perf_counter()
    status = "success"
    max_seq_length = task_max_seq_length(task) if task else _DEFAULT_INFERENCE_SEQ_LENGTH
    # One prompt's worth of labels is the natural unit of work, but 28 sequences at once can be
    # more than a small GPU wants, so the label dimension is chunked by the task's own eval batch
    # size and halved on OOM exactly as the generation path does.
    chunk = max(1, min(len(labels), _eval_batch_size(task) if task else 16))
    metadata = {
        "task": task,
        "prompt_count": len(prompts),
        "label_count": len(labels),
        "max_seq_length": max_seq_length,
        "initial_label_chunk": chunk,
        "oom_retries": 0,
    }
    try:
        model, tokenizer = _load_inference_model(weights_ref, base_model, max_seq_length)
        results: list[dict[str, float]] = []
        for prompt in prompts:
            rendered = _render_inference_prompt(tokenizer, prompt, base_model)
            rows = _label_score_rows(tokenizer, rendered, labels)
            longest = max(len(ids) for ids, _n in rows)
            if longest > max_seq_length:
                raise ValueError(
                    f"task={task}: a scored prompt plus label is {longest} tokens, over the "
                    f"{max_seq_length}-token context. Raise the task's max_seq_length."
                )
            scored: list[float] = []
            index = 0
            while index < len(rows):
                window = rows[index:index + chunk]
                try:
                    scored.extend(_score_label_batch(model, tokenizer, window, torch))
                except BaseException as exc:  # noqa: BLE001 - re-raised unless it is an OOM
                    if not _is_cuda_oom(exc, torch) or len(window) == 1:
                        raise
                    _clear_cuda_oom_cache(torch)
                    chunk = max(1, len(window) // 2)
                    metadata["oom_retries"] += 1
                    continue
                index += len(window)
            results.append(dict(zip(labels, scored)))
        return results
    except BaseException as exc:
        status = "error"
        metadata.update({"error_type": type(exc).__name__, "error": str(exc)[:500]})
        raise
    finally:
        metadata["final_label_chunk"] = chunk
        metadata["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        record_timing_event(TimingEvent(
            kind="inference",
            name="infer_label_scores",
            duration_ms=metadata["latency_ms"],
            status=status,
            metadata=metadata,
        ))


# Concurrent GGUF scoring: how many requests are in flight at once, and the floor it degrades to.
#
# WHY THIS IS CONCURRENCY AND NOT TENSOR BATCHING — the distinction is load-bearing
#     The bf16 path (`infer_batch`) pads N sequences into one forward pass, which is true batching.
#     llama-cpp-python 0.3.34 cannot do that for GENERATION: its multi-sequence `LlamaBatch` plumbing
#     is wired only into `embed()`, and `create_chat_completion` decodes one sequence per call. The
#     `n_batch`/`n_ubatch` constructor arguments size the PREFILL batch within a single sequence, not
#     the number of sequences.
#
#     Reaching true multi-sequence decode would mean driving `llama_batch_init`/`llama_decode`
#     directly and hand-rolling sampling, stop conditions and detokenization. That code decides every
#     accuracy number this project reports, so a subtle divergence there would silently move results
#     rather than fail. Not a good trade for a speedup.
#
#     So this runs N INDEPENDENT contexts and dispatches prompts across them. Each worker calls
#     exactly the same `create_chat_completion` the sequential path called, with the same greedy
#     settings, so per-row output is unchanged by construction — only the scheduling differs. That is
#     a property tests can pin, which the low-level route's would not be.
#
# THE MEMORY COST, AND WHY THE DEFAULT IS NOT 32
#     Each context carries its own copy of the weights, because `Llama` bundles model and context.
#     A 1.2GB Q4_K_M model plus a 4096-token KV cache is roughly 1.5-2GB per worker, so the task's
#     declared `eval_batch_size` of 32 would ask for 50GB+ and OOM on a 48GB L40S. The concurrency is
#     therefore the task's batch size CLAMPED by `MAX_GGUF_EVAL_CONCURRENCY`, and it halves on OOM the
#     same way the bf16 path halves its batch.
# DEFAULT 1 — the concurrency is OFF unless explicitly asked for, and that is a decision made from
# measurement, not caution.
#
# Enabled at 8 on runs 38734202/38734203 it worked twice and then killed the run: the eval CUDA worker
# died with `exit=-6` and `worker produced no response`. SIGABRT is how llama.cpp reports a failed
# device allocation — `GGML_ASSERT` calls `abort()` rather than raising — so the process is gone before
# any Python handler runs. `_looks_like_oom` and the halving retry below cannot see it, and neither can
# the run: a 7-day job dies on iteration 3 with no diagnosis attached.
#
# And the gain did not justify that. Measured on calendar_json, 535 prompts: 125s sequential against
# 101s at 8-way, about 20%, not the multiple the memory cost implies. Eight contexts submitting to one
# device serialise on it, so thread concurrency buys queueing overlap and little else.
#
# The machinery stays because it is correct and tested, and because the diagnosis above is worth
# keeping next to it. Real eval throughput needs either llama.cpp's multi-sequence decode API — one
# model, one context, `n_seq_max` sequences, no duplicated weights — or routing eval through the vLLM
# server already running idle on the other GPU. Both batch properly; neither duplicates the model.
MAX_GGUF_EVAL_CONCURRENCY = int(os.environ.get("SLM_GGUF_EVAL_CONCURRENCY", "1"))
MIN_GGUF_EVAL_CONCURRENCY = 1


def _looks_like_oom(error: BaseException) -> bool:
    """Whether this failure is an out-of-memory condition worth retrying smaller.

    Matched on the message because llama.cpp surfaces allocation failures as generic exceptions from
    the C library rather than a typed error, so there is nothing else to key on.
    """
    text = f"{type(error).__name__}: {error}".lower()
    return any(
        needle in text for needle in
        ("out of memory", "outofmemory", "cuda error", "failed to allocate",
         "cudamalloc", "not enough memory", "insufficient", "oom")
    )


def _run_gguf_concurrently(
    prompts: list[str],
    *,
    primary,
    score_one,
    gguf_path: str,
    max_seq_length: int,
    task: str,
) -> list[str]:
    """Score every prompt over a pool of GGUF contexts, order-preserving, halving on OOM.

    The already-loaded `primary` context is worker 0, so a concurrency of 1 loads nothing extra and
    behaves exactly like the sequential path it replaced.
    """
    from concurrent.futures import ThreadPoolExecutor

    try:
        requested = _eval_batch_size(task)
    except Exception:  # noqa: BLE001 — an unresolvable task must not break scoring
        # `infer_batch_gguf` accepts an empty task (its signature defaults to ""), and callers outside
        # the agent loop use it that way. Falling back to one context reproduces the pre-2026-08-21
        # sequential behaviour exactly, so an unknown task loses the speedup and nothing else.
        requested = MIN_GGUF_EVAL_CONCURRENCY
    concurrency = max(MIN_GGUF_EVAL_CONCURRENCY, min(requested, MAX_GGUF_EVAL_CONCURRENCY))
    if concurrency < requested:
        print(f"      [gguf-eval] task asks for batch {requested}; running {concurrency} concurrent "
              f"context(s) — each context holds its own copy of the weights, so the ceiling is "
              f"memory, not the task (see MAX_GGUF_EVAL_CONCURRENCY)")

    while True:
        extra: list = []
        try:
            workers = [primary]
            for _ in range(concurrency - 1):
                extra.append(_load_gguf_context(gguf_path, max_seq_length))
            workers.extend(extra)
            if concurrency > 1:
                print(f"      [gguf-eval] scoring {len(prompts)} prompt(s) across "
                      f"{len(workers)} concurrent context(s)")
            results: list[str | None] = [None] * len(prompts)

            def _work(item):
                index, prompt = item
                # Worker assignment is by position so a given prompt always lands on one context and
                # never migrates mid-generation.
                return index, score_one(workers[index % len(workers)], prompt, index)

            with ThreadPoolExecutor(max_workers=len(workers)) as pool:
                for index, text in pool.map(_work, list(enumerate(prompts))):
                    results[index] = text
            return [("" if text is None else text) for text in results]
        except Exception as error:  # noqa: BLE001 — an OOM is retried smaller, anything else raises
            if concurrency > MIN_GGUF_EVAL_CONCURRENCY and _looks_like_oom(error):
                concurrency = max(MIN_GGUF_EVAL_CONCURRENCY, concurrency // 2)
                print(f"      [gguf-eval] out of memory; halving concurrency to {concurrency} "
                      f"and retrying ({type(error).__name__}: {str(error)[:120]})")
                continue
            raise
        finally:
            # Only the contexts THIS attempt created. `primary` is owned by the module-level cache and
            # is reused across iterations; freeing it here would reload the model every eval.
            for context in extra:
                try:
                    context.close()
                except Exception:  # noqa: BLE001 — best-effort teardown
                    pass


def _load_gguf_context(gguf_path: str, max_seq_length: int):
    """One additional GGUF context, offloaded to GPU on the same terms as the primary."""
    import llama_cpp

    gpu_layers = int(os.environ.get("SLM_GGUF_GPU_LAYERS", "-1"))
    try:
        return llama_cpp.Llama(
            model_path=gguf_path, n_ctx=max_seq_length,
            n_gpu_layers=gpu_layers, verbose=False,
        )
    except Exception:
        return llama_cpp.Llama(
            model_path=gguf_path, n_ctx=max_seq_length, n_gpu_layers=0, verbose=False,
        )


def infer_batch_gguf(
    prompts: list[str],
    gguf_path: str,
    max_new_tokens: int = 50,
    base_model: str | None = None,
    task: str = "",
) -> list[str]:
    """
    Run inference over all prompts using a GGUF file via llama-cpp-python.
    Used for quantized model evaluation (Q4_K_M, Q8_0) to get honest on-device
    accuracy scores — the same model format that ships on Android.

    Scored across a pool of independent contexts (see `_run_gguf_concurrently`): llama-cpp-python
    cannot batch-decode multiple sequences, so this is request concurrency rather than tensor
    batching, and per-row output is identical to scoring them one at a time.
    Caches the primary Llama instance by gguf_path (max 3 entries).

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

    max_seq_length = task_max_seq_length(task) if task else _DEFAULT_INFERENCE_SEQ_LENGTH
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

    def _validate_gguf_budget(rendered_prompt: str, prompt_index: int, llama) -> None:
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

    stop_tokens = None if supports_mode_kwargs else _serving_stop_tokens(base_model)

    def _score_one(worker_llama, prompt: str, index: int) -> str:
        """One prompt through one context. Byte-identical to the old sequential body."""
        if supports_mode_kwargs:
            _validate_gguf_budget(prompt, index, worker_llama)
            resp = worker_llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens, temperature=0.0,
                chat_template_kwargs={"enable_thinking": False},
            )
            return resp["choices"][0]["message"]["content"]
        rendered_prompt = _serving_prompt_prefix(prompt, base_model)
        _validate_gguf_budget(rendered_prompt, index, worker_llama)
        resp = worker_llama(
            rendered_prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
            echo=False,
            stop=stop_tokens,
        )
        return resp["choices"][0]["text"]

    return _run_gguf_concurrently(
        prompts,
        primary=llama,
        score_one=_score_one,
        gguf_path=gguf_path,
        max_seq_length=max_seq_length,
        task=task,
    )


# ── MNN: the second on-device runtime ────────────────────────────────────────────────────────────
#
# Everything below is to `infer_batch_gguf` what `training/quantize_mnn.py` is to
# `training/quantize.py`: the same job through the engine that actually runs on an MNN-Chat phone.
# The prompt rendering is deliberately SHARED with the GGUF path (`_serving_prompt_prefix`) rather
# than delegated to MNN's own chat template, for two reasons:
#   1. Train/serve alignment is already asserted against that rendering in
#      `training/lora_trainer.py::_assert_train_serve_prefix_alignment`. A second, subtly different
#      template applied only on the MNN path is precisely the B290 failure — two stray `<think>`
#      tags per prediction — with a new place to hide.
#   2. It is what makes a backend comparison mean anything. If llama.cpp and MNN were each given
#      their own idea of the prompt, a score difference between them could be the quantization,
#      the runtime, or the template, and nothing would say which.
# MNN is therefore configured with `use_template: false` and handed the fully rendered prompt.
_mnn_cache: dict = {}
_mnn_cache_order: list = []

# MNN's own defaults are a chat app's, not an evaluator's: `sampler_type: "mixed"` with
# temperature 0.8 / top_k 40. Scoring under those would make every eval a different experiment.
# Greedy is the MNN spelling of the GGUF path's `temperature=0.0`.
_MNN_EVAL_SAMPLER = "greedy"
# Compute precision. "low" is what the exported config ships, what a phone runs (fp16 accumulate
# where the backend has it), AND — measured — the only precision under which MNN's CUDA backend is
# both correct and fast. See `_MNN_BAD_CUDA_COMBINATIONS`.
_MNN_PRECISION = os.environ.get("SLM_MNN_PRECISION", "low")
# HOW THE WEIGHTS ARE HELD, and on CUDA this is a correctness setting chosen by BIT WIDTH.
#
#   "low"    keeps the quantized weights packed and dequantizes inside the GEMM — MNN's
#            weight-only-quant kernels, the path a phone uses.
#   "normal" dequantizes the weights once into fp16 device memory and runs dense GEMMs. This costs
#            VRAM, NOT accuracy: the values are still the quantized ones, just stored wider, so the
#            arithmetic is the quantized model's arithmetic.
#
# Measured on SmolLM2-360M over clinc150 rows, 15 rows per cell:
#
#     artifact  backend  memory   macro_f1  format_valid  rows/s
#     4-bit     cuda     low        works     works        2.86   <- fastest, and correct
#     4-bit     cuda     normal     works     works        2.79
#     8-bit     cpu      low        0.2667    0.2667       0.14
#     8-bit     cuda     low        0.0000    0.0000       0.36   <- BROKEN: '<|endoftext|>' spam
#     8-bit     cuda     normal     0.2667    0.2667       0.59   <- matches the CPU exactly
#
# So MNN's CUDA int4 weight-only kernel is sound and its int8 one is not. "auto" therefore picks
# `low` for a 4-bit artifact and `normal` for anything else on CUDA, and `low` on the CPU (which
# handles every width). An explicit value is honoured, except for the combination measured to be
# broken, which is refused by name.
_MNN_MEMORY = os.environ.get("SLM_MNN_MEMORY", "auto")

# WHICH DEVICE MNN COMPUTES ON. This is the difference between an eval that takes 6 minutes and one
# that takes 35, measured on the same artifact and rows (SmolLM2-360M @ 4-bit, clinc150):
#
#     backend   prefill        rows/s   exact labels
#     cpu        671 tok/s      0.45      4/5
#     cuda      6867 tok/s      2.86      5/5
#
# Prefill is ~99% of this workload (a CLINC150 prompt is ~1,385 tokens and the answer is one
# label), so the 10x prefill win is the whole story. Accuracy is unchanged, which is the same
# argument the GGUF path makes for offloading to the GPU: the artifact's accuracy is a property of
# its weights, tokenizer and greedy decoding, not of the device that multiplies the matrices.
#
# "auto" (the default) means CUDA when this process can see a GPU, CPU when it cannot — the same
# shape as `SLM_GGUF_GPU_LAYERS=-1` with its CPU fallback, except that here the choice is made ONCE
# and stated, never silently per-load. An explicit "cuda" is a demand: if the backend is missing,
# the load FAILS instead of quietly computing on the CPU (see `_assert_backend_honoured`).
_MNN_BACKEND_TYPE = os.environ.get("SLM_MNN_BACKEND_TYPE", "auto").strip().lower()

# Configurations measured to produce WRONG output on CUDA, refused rather than scored. The
# combination below decodes `'ordinaryritz Hviations:`~ ...'` where every other combination decodes
# `'accept_reservations'`, and it is also no faster than CPU — the int4 weight-only kernel appears
# to have no fp32 path, so the model runs somewhere between the two and produces neither speed nor
# sense. Keyed on (backend, precision, memory).
_MNN_BAD_CUDA_COMBINATIONS = {("cuda", "normal", "low")}

# Read only to REFUSE a value above 1 (see `infer_batch_mnn`). MNN eval is sequential, which is
# parity with the GGUF path rather than a shortfall: `MAX_GGUF_EVAL_CONCURRENCY` defaults to 1 as
# well, and asking MNN for more would be asking it for more than llama.cpp takes.
_MNN_EVAL_CONCURRENCY_ENV = "SLM_MNN_EVAL_CONCURRENCY"

# THREAD COUNT IS A CORRECTNESS SETTING IN MNN, NOT A SPEED SETTING, and it is measured.
#
# Run 40260927 scored `Qwen/Qwen3.5-0.8B@Q4_K_M` at 0.0000 with format_valid 0.0000: every one of
# 1,000 CLINC150 rows decoded to `%+!!!!!!!!!!`, `feier!!!!!!!!!!` or `оте!!!!!!!!!!`. The weights
# were fine. Sweeping the SAME artifact on the SAME prompts, one fresh process per setting:
#
#     threads   1   2   4   8  10  12  13  14  16
#     verdict  ok  ok  ok  ok  ok  ok  ok  BAD  ok
#
# 14 corrupts compute; everything either side of it is correct and agrees token-for-token. The
# logits are finite — no NaN, no inf, plausible magnitudes — with a different argmax, so there is
# nothing to detect in the output except the answer being wrong. It also needs a LONG prompt: the
# same artifact at 14 threads answers a 16-token `Hello` perfectly and only fails at ~1,050 tokens,
# which is why the build's short smoke test passed and 35 minutes of eval was scored anyway.
#
# 8 is the default because it is measured-good, well under any plausible allocation, and close to
# MNN's own exported `thread_num: 4`. An operator's value is NOT trusted on faith: every artifact
# build cross-checks the configured count against `MNN_REFERENCE_THREADS` on a long prompt
# (`quantize_mnn._assert_threaded_compute_is_sound`) and refuses the artifact on a mismatch.
_MNN_THREADS = max(1, int(os.environ.get("SLM_MNN_THREADS", "8")))
# The count the cross-check trusts. Low, so the work split is coarse and every model's tile
# arithmetic is trivial, and measured correct on every configuration tested above.
MNN_REFERENCE_THREADS = int(os.environ.get("SLM_MNN_REFERENCE_THREADS", "4"))
# MNN's engine prints the prompt and the full decoded response for every call through
# `MNN_PRINT`, at C level. Over an 800-row eval that is tens of thousands of lines between the
# log lines a human is actually reading, so it is suppressed at the file-descriptor level unless
# asked for. `SLM_MNN_VERBOSE=1` turns it back on, which is what to do when a load fails and the
# engine's own diagnosis is the thing you need.
_MNN_VERBOSE = os.environ.get("SLM_MNN_VERBOSE", "0") == "1"


class _capture_c_stdout:
    """Run a block with C-level stdout redirected into a file, and return what was written.

    The suppressing sibling of `_suppress_c_stdout`, and it exists for one reason: MNN announces a
    BACKEND FALLBACK only on stdout, as `Can't Find type=2 backend, use 0 instead`, and returns a
    perfectly usable object that computes on the CPU. Nothing in the API reports it. Capturing the
    engine's own words is the only way to tell "running on the GPU" from "asked for the GPU".

    Same file-descriptor swap as the suppressor, for the same reason (`printf` ignores
    `sys.stdout`), including the `fflush(NULL)` before restoring — without it the text is still
    sitting in a C stdio buffer when the file is read.
    """

    def __init__(self):
        self._saved = None
        self._handle = None
        self.text = ""

    def __enter__(self):
        import sys as _sys
        import tempfile

        _sys.stdout.flush()
        _suppress_c_stdout._flush_c_stdio()
        self._handle = tempfile.TemporaryFile(mode="w+b")
        self._saved = os.dup(1)
        os.dup2(self._handle.fileno(), 1)
        return self

    def __exit__(self, *exc_info):
        if self._saved is not None:
            _suppress_c_stdout._flush_c_stdio()
            os.dup2(self._saved, 1)
            os.close(self._saved)
            self._saved = None
        if self._handle is not None:
            try:
                self._handle.seek(0)
                self.text = self._handle.read().decode("utf-8", "replace")
            except OSError:
                self.text = ""
            self._handle.close()
            self._handle = None
        return False


def mnn_backend_type() -> str:
    """The MNN backend this process will compute on: "cuda" or "cpu".

    "auto" asks whether this process can see a GPU at all — a machine without one is not a silent
    fallback, it is a machine without a GPU — while an explicit value is honoured as written so
    that a run pinned to CUDA fails loudly if CUDA is unavailable instead of quietly producing CPU
    numbers 6x slower.

    THE ENVIRONMENT IS READ ON EVERY CALL, not captured at import. `hardware_eval/mnn_backend_
    matrix.py` scores the same artifact on both devices in one process by setting the variable
    between cells, and with an import-time constant its "cpu" column silently ran on CUDA — the
    two columns came back identical to four decimal places, which is what gave it away.
    """
    configured = os.environ.get("SLM_MNN_BACKEND_TYPE", _MNN_BACKEND_TYPE).strip().lower()
    if configured in ("cpu", "cuda"):
        return configured
    if configured != "auto":
        raise ValueError(
            f"SLM_MNN_BACKEND_TYPE={configured!r} is not a supported MNN backend. Valid "
            f"values: 'auto' (CUDA when a GPU is visible, else CPU), 'cuda', 'cpu'."
        )
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 — no torch means no way to ask; CPU is the safe answer
        return "cpu"


# How many models this process has loaded on CUDA. MNN'S CUDA RUNTIME DOES NOT SURVIVE A SECOND
# ONE, and it does not say so: measured over 27 artifacts loaded and freed in one process, the
# first decoded correctly, the second decoded `<|endoftext|><|endoftext|>...`, and every one after
# that returned an EMPTY STRING — scored as 0.0000 with no error anywhere, at a nonsensical 878
# rows/s because generation was doing nothing. The CPU backend has no such limit (the thread
# cross-check loads a second model on purpose).
#
# The pipeline is already safe: `run_eval` and `build_quant_artifact` each run in their own
# disposable CUDA worker, so one process loads one artifact. This counter exists so that anything
# ELSE — `hardware_eval/quant_accuracy_eval.py` sweeping several quants with isolation off, a
# notebook, a future tool — fails loudly instead of recording zeros.
_mnn_cuda_loads = 0

_mnn_tmp_dir: str | None = None


def _artifact_quant_bits(artifact_dir: str) -> int | None:
    """The bit width this MNN artifact was exported at, from the exporter's own record."""
    from training.quantize_mnn import _recorded_quant_bit

    return _recorded_quant_bit(artifact_dir)


def mnn_memory_mode(backend: str, artifact_dir: str) -> str:
    """How MNN should hold the weights: "low" (packed) or "normal" (dequantized into memory).

    See `_MNN_MEMORY` for the measurements. The short version: on CUDA only the int4 weight-only
    kernel is trustworthy, so a 4-bit artifact gets `low` (fastest and correct) and anything wider
    gets `normal`, which bypasses the broken int8 kernel at no cost in accuracy. The CPU handles
    every width with `low`.
    """
    explicit = None if _MNN_MEMORY in ("auto", "") else _MNN_MEMORY
    bits = _artifact_quant_bits(artifact_dir)
    if backend != "cuda":
        return explicit or "low"
    if explicit is None:
        return "low" if bits == 4 else "normal"
    if explicit == "low" and bits is not None and bits != 4:
        raise RuntimeError(
            f"MNN on CUDA with memory=low decodes garbage for a {bits}-bit artifact: measured "
            f"format_valid 0.0000 and '<|endoftext|>' repeated on every row, where the same "
            f"artifact under memory=normal scores exactly what the CPU scores (0.2667 both) and "
            f"4x faster than the CPU. MNN's CUDA int8 weight-only kernel is not sound; its int4 "
            f"one is. Unset SLM_MNN_MEMORY to let the bit width choose (this is what 'auto' does), "
            f"or set SLM_MNN_MEMORY=normal."
        )
    return explicit


def _mnn_tmp_path() -> str:
    """A writable scratch directory for MNN's kernel-tuning cache, created once per process."""
    global _mnn_tmp_dir
    if _mnn_tmp_dir is None:
        import tempfile

        _mnn_tmp_dir = tempfile.mkdtemp(prefix="slm-mnn-cache-")
    return _mnn_tmp_dir


def _assert_backend_honoured(requested: str, engine_output: str, artifact_dir: str) -> None:
    """Refuse a model that MNN loaded onto a different backend than the one asked for.

    THE FAILURE THIS EXISTS FOR. Asked for CUDA with a pymnn whose CUDA backend was not registered,
    MNN printed `Can't Find type=2 backend, use 0 instead` and carried on — on the CPU, at CPU
    speed, and (because it had already configured itself for a GPU and skipped the CPU blockwise-
    quant setup) decoding `'accept<|endoftext|><|endoftext|>...'` instead of `'accept_reservations'`.
    Every one of the first four CUDA measurements taken here was that, and it looked exactly like a
    slow, broken GPU backend rather than an unregistered one.

    It is worth failing hard on: the whole point of the GPU path is speed, so silently getting CPU
    speed defeats it, and the accompanying corruption would be recorded as a model result. The fix
    is a build fix, and the message says which one.
    """
    if "Can't Find type" not in engine_output and "Cant Find type" not in engine_output:
        return
    raise RuntimeError(
        f"MNN could not use the {requested!r} backend and fell back to the CPU while loading "
        f"{artifact_dir}. Its own words: "
        f"{next((line for line in engine_output.splitlines() if 'Find type' in line), '').strip()!r}. "
        f"This is a BUILD problem, not a model problem: the CUDA backend registers through a static "
        f"initializer in libMNN, and a STATIC libMNN.a drops that object because nothing references "
        f"it, so pymnn must be built against a SHARED libMNN.so (see scripts/setup_mnn_env.sh, which "
        f"passes -DMNN_BUILD_SHARED_LIBS=ON and stages libMNN.so + libMNN_Cuda_Main.so). Refusing to "
        f"score, because a fallback here means CPU speed AND corrupted output reported as a model "
        f"result. Set SLM_MNN_BACKEND_TYPE=cpu to evaluate on the CPU deliberately."
    )


def _preload_mnn_shared_libs() -> list[str]:
    """dlopen pymnn's own shared libraries from the venv before the extension asks for them.

    WHY THIS EXISTS RATHER THAN A PATH VARIABLE. pymnn is linked against a SHARED libMNN.so
    (mandatory — a static link drops the CUDA backend registrar, see
    `_assert_backend_honoured`), and those libraries live in `$VIRTUAL_ENV/lib`, which is not a
    system search path. `.venv_gpu/bin/activate` puts it on LD_LIBRARY_PATH, but LD_LIBRARY_PATH is
    read once at process start, so anything invoking `.venv_gpu/bin/python` DIRECTLY — this repo's
    own shell scripts and every bare pytest run — got `ImportError: libMNN.so: cannot open shared
    object file`.

    Loading them by absolute path with RTLD_GLOBAL satisfies the extension's dependency from
    inside the process: the dynamic linker matches an already-loaded object by soname and does not
    search the filesystem again.

    It does NOT remove the one remaining environment dependency: libMNN.so needs a newer
    libstdc++ (GLIBCXX_3.4.30) than the system provides, exactly as llama-cpp-python does, and
    `.venv_gpu/bin/activate` is what puts gcc-12's on LD_LIBRARY_PATH. Returns the reasons any
    library could not be loaded so the caller's ImportError can say THAT instead of the misleading
    "cannot open shared object file".
    """
    import ctypes
    import sys

    failures: list[str] = []
    lib_dir = os.path.join(sys.prefix, "lib")
    for name in ("libMNN_Cuda_Main.so", "libMNN.so"):
        path = os.path.join(lib_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            failures.append(f"{name}: {error}")
    return failures


def _mnn_llm_module():
    """pymnn's LLM API, or an ImportError that says exactly what to run.

    Two distinct failures, distinguished because they have different fixes: pymnn missing entirely,
    and pymnn built WITHOUT `PYMNN_LLM_API` (its `MNN.llm` is then `None`, not absent — the package
    swallows the ImportError). The second is the likely one, because the published wheel is built
    without the LLM bindings; `scripts/setup_mnn_env.sh` builds them.
    """
    try:
        # Suppressed because importing the extension prints MNN's CPU topology probe — a
        # 400-core affinity list on this cluster's login nodes — before any Python runs.
        #
        # `import MNN.llm` rather than `getattr(MNN, "llm")`, which is NOT the same module.
        # `MNN/__init__.py` does `from _mnncengine import *`, and the extension exposes a
        # submodule of its own called `llm`, so the attribute can resolve to the RAW extension
        # module instead of the Python package that wraps it. The two have identically named
        # methods with different argument types — `set_config` takes a dict on the wrapper and a
        # JSON string on the extension — so getting the wrong one surfaces as
        # `SystemError: <method 'set_config' of 'LLM' objects> returned a result with an
        # exception set`, which says nothing about the actual problem. Importing the submodule by
        # path always resolves to the package.
        preload_failures: list[str] = []
        with _suppress_c_stdout(not _MNN_VERBOSE):
            preload_failures = _preload_mnn_shared_libs()
            import MNN  # noqa: F401 - the package must initialise before its submodule
            import MNN.llm as llm_module
    except ImportError as exc:
        # A preload failure is the REAL cause whenever it happened, and it reads nothing like the
        # import error it produces: "libMNN.so: cannot open shared object file" for a file that is
        # plainly there, when what actually failed was its libstdc++ requirement.
        detail = (
            f" The shared libraries are present but did not load: {'; '.join(preload_failures)}."
            f" That is an environment problem, not a build one — `source .venv_gpu/bin/activate`"
            f" puts gcc-12's libstdc++ on LD_LIBRARY_PATH, which libMNN.so needs."
            if preload_failures else ""
        )
        raise ImportError(
            "MNN inference requires pymnn with the LLM API (MNN.llm), which is absent from the "
            "published wheel and from a pymnn built without -DMNN_BUILD_LLM=ON. Build it with "
            f"`bash scripts/setup_mnn_env.sh`. Underlying import error: {exc}.{detail}"
        ) from exc
    if getattr(llm_module, "create", None) is None:
        raise ImportError(
            "pymnn imported but MNN.llm has no `create` entry point, so it cannot load an MNN "
            "LLM artifact. Rebuild it with `bash scripts/setup_mnn_env.sh --force`."
        )
    return llm_module


def _mnn_set_config(llm, config: dict) -> None:
    """Apply a runtime config to a pymnn LLM object, whichever of the two shapes it is.

    pymnn's Python wrapper takes a dict and JSON-encodes it for the extension; the extension's own
    object takes the encoded string. Both are reachable (see `_mnn_llm_module`), and handing either
    one the other's argument fails with a `SystemError` that names neither cause nor fix — so the
    shape is detected rather than guessed, by the wrapper's own handle on the C object.
    """
    if hasattr(llm, "_c_obj"):
        llm.set_config(config)
        return
    import json as _json

    llm.set_config(_json.dumps(config))


def mnn_runtime_versions() -> dict:
    """Which pymnn actually loaded the artifact, for the validation sidecar.

    Read from installed distribution metadata rather than a module attribute: pymnn built from
    source has no `__version__`, and the distribution it was installed as (`mnn`, lowercased by
    setuptools) does carry the version its CMake build stamped in.
    """
    from importlib.metadata import PackageNotFoundError, version

    for name in ("MNN", "mnn"):
        try:
            return {"pymnn": version(name)}
        except PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 — a version string must never break validation
            break
    return {"pymnn": "unknown"}


class _suppress_c_stdout:
    """Silence C-level stdout for the duration of a block, without touching Python's.

    `contextlib.redirect_stdout` cannot do this: MNN prints with `printf`, which writes to file
    descriptor 1 directly and never consults `sys.stdout`. So the descriptor itself is swapped.
    Python-level prints inside the block are still swallowed, which is why every progress message
    here is emitted outside it.

    THE C-LEVEL FLUSH ON EXIT IS THE WHOLE TRICK. Redirecting the descriptor alone suppressed
    nothing: C stdio buffers into a `FILE*`, so MNN's output sat in that buffer until the process
    exited and was then flushed to whatever descriptor 1 pointed at BY THEN — the restored real
    stdout. The result was an eval log with every prompt and response dumped at the end instead of
    interleaved, which is worse than not suppressing at all. `fflush(NULL)` empties the buffers
    into /dev/null while it is still attached.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._saved = None
        self._devnull = None

    @staticmethod
    def _flush_c_stdio() -> None:
        try:
            import ctypes

            ctypes.CDLL(None).fflush(None)
        except Exception:  # noqa: BLE001 — losing the suppression is not worth an exception
            pass

    def __enter__(self):
        if not self._enabled:
            return self
        import sys as _sys

        _sys.stdout.flush()
        self._flush_c_stdio()
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._saved = os.dup(1)
        os.dup2(self._devnull, 1)
        return self

    def __exit__(self, *exc_info):
        if self._saved is not None:
            self._flush_c_stdio()
            os.dup2(self._saved, 1)
            os.close(self._saved)
            self._saved = None
        if self._devnull is not None:
            os.close(self._devnull)
            self._devnull = None
        return False


def load_mnn_llm(
    artifact_dir: str,
    max_seq_length: int | None = None,
    max_new_tokens: int = 512,
    threads: int | None = None,
    backend_type: str | None = None,
):
    """Load an MNN LLM artifact directory and return a configured, loaded pymnn `Llm`.

    The config is applied BEFORE `load()` because two of its entries change what is allocated
    rather than only how it decodes: `max_all_tokens` sizes the KV cache, and MNN's own default of
    2048 is below the 4096-token window this pipeline's tasks are evaluated at. A CLINC150 prompt
    carrying 150 intent names is not a hypothetical overflow.

    `threads` overrides `SLM_MNN_THREADS` for this instance. It exists for the build-time
    cross-check, and there is one thing to know before using it: MNN's thread pool is a
    PROCESS-GLOBAL keyed by CPU mask, created by whoever loads first, and a later load asking for
    MORE threads than the pool has silently gets the pool's count. So a comparison between two
    thread counts in one process is only valid in one order — the higher count first — and an
    in-process sweep that starts low measures the low setting several times over.

    `backend_type` overrides the resolved device for this instance ("cuda" or "cpu"); it exists for
    the same cross-check and for the backend matrix. A load that cannot honour the requested
    backend raises rather than falling back to the CPU.
    """
    from training.quantize_mnn import mnn_config_path, missing_files

    absent = missing_files(artifact_dir)
    if absent:
        raise RuntimeError(
            f"Cannot load MNN artifact {artifact_dir}: missing {absent}"
        )
    context_tokens = max_seq_length or _DEFAULT_INFERENCE_SEQ_LENGTH
    thread_count = _MNN_THREADS if threads is None else max(1, int(threads))
    requested_backend = (backend_type or mnn_backend_type()).strip().lower()
    memory_mode = mnn_memory_mode(requested_backend, artifact_dir)
    # Checked BEFORE the engine is even imported: it is a pure configuration fault, and there is no
    # reason to load a model to discover it.
    combination = (requested_backend, _MNN_PRECISION, memory_mode)
    if combination in _MNN_BAD_CUDA_COMBINATIONS:
        raise RuntimeError(
            f"MNN backend={requested_backend} with precision={_MNN_PRECISION} and "
            f"memory={memory_mode} is measured to decode GARBAGE — the same artifact and prompts "
            f"that give 'accept_reservations' under precision=low returned "
            f"'ordinaryritz Hviations:`~ ...' under this combination, at no speed gain. Use "
            f"SLM_MNN_PRECISION=low (the default, and the fastest on CUDA) or "
            f"SLM_MNN_MEMORY=normal."
        )
    llm_module = _mnn_llm_module()
    if requested_backend == "cuda":
        _note_cuda_load(artifact_dir)

    # Captured rather than suppressed: MNN reports a backend fallback only in this output, and a
    # fallback is fatal here (see `_assert_backend_honoured`).
    with _capture_c_stdout() as engine:
        llm = llm_module.create(mnn_config_path(artifact_dir))
        _mnn_set_config(llm, {
            "backend_type": requested_backend,
            # MNN writes a kernel-tuning cache and, given nowhere to put it, drops
            # `mnn_cachefile.bin` into the CURRENT DIRECTORY — which for the pipeline is the
            # project root. Pointed at a temp dir instead: not the artifact directory, because the
            # cache-hit check hashes every file in there and a cache file written during scoring
            # would invalidate the artifact that produced it.
            "tmp_path": _mnn_tmp_path(),
            "sampler_type": _MNN_EVAL_SAMPLER,
            # The prompt arrives fully rendered by `_serving_prompt_prefix`; see the section note.
            "use_template": False,
            "precision": _MNN_PRECISION,
            "memory": memory_mode,
            "thread_num": thread_count,
            "max_all_tokens": context_tokens,
            "max_new_tokens": max_new_tokens,
            # Each eval row is an independent prompt, so carrying a KV cache between them would be
            # both wrong and slower.
            "reuse_kv": False,
        })
        llm.load()
    if _MNN_VERBOSE and engine.text:
        print(engine.text, end="")
    _assert_backend_honoured(requested_backend, engine.text, artifact_dir)
    return llm


def _note_cuda_load(artifact_dir: str) -> None:
    """Refuse a second CUDA load in one process, because MNN returns empty text after the first.

    See `_mnn_cuda_loads`. Failing here costs nothing in the pipeline — every eval and every
    artifact build already runs in its own disposable worker — and it converts a whole class of
    silent 0.0000 scores into a message that names the cause.
    """
    global _mnn_cuda_loads

    _mnn_cuda_loads += 1
    if _mnn_cuda_loads > 1:
        raise RuntimeError(
            f"This process has already loaded an MNN model on CUDA, and MNN's CUDA runtime does "
            f"not survive a second one: measured over 27 loads in one process, the first decoded "
            f"correctly, the second decoded '<|endoftext|>' repeatedly, and the rest returned "
            f"EMPTY strings — scored as 0.0000 with no error raised. Refusing to load "
            f"{artifact_dir}. Score one artifact per process: the pipeline already does (each "
            f"run_eval and each artifact build runs in its own CUDA worker, SLM_CUDA_ISOLATION=1), "
            f"and an out-of-loop sweep should either set that or use SLM_MNN_BACKEND_TYPE=cpu, "
            f"which has no such limit."
        )


def mnn_generate(llm, rendered_prompt: str, max_new_tokens: int = 512) -> str:
    """One rendered prompt through one loaded MNN model, as a decoded string.

    `reset()` first, always. MNN's `Llm` is a CHAT object: it accumulates history across
    `response()` calls, so without this the second eval row would be answered in the context of the
    first — which does not crash, does not look wrong in the log, and quietly makes every score
    after row 1 a different measurement.
    """
    llm.reset()
    with _suppress_c_stdout(not _MNN_VERBOSE):
        # pymnn's wrapper exposes `response(prompt, stream)` and takes the token cap from the
        # config, which is why `max_new_tokens` is set there rather than passed here.
        _mnn_set_config(llm, {"max_new_tokens": max_new_tokens})
        text = llm.response(rendered_prompt, False)
    return text or ""


def _validate_mnn_budget(llm, rendered_prompt: str, index: int,
                         max_seq_length: int, max_new_tokens: int) -> None:
    """Refuse a prompt that cannot fit the context, instead of letting MNN quietly truncate it.

    The exact counterpart of `infer_batch_gguf`'s `_validate_gguf_budget`, and it matters more here:
    MNN sizes its KV cache from `max_all_tokens` and drops what does not fit, so an over-long prompt
    produces a plausible-looking answer to a QUESTION THE MODEL NEVER SAW rather than an error. The
    llama.cpp path raising on exactly this is what surfaced the real fault the first time this was
    run — a CLINC150 prompt carries all 151 intent names and needs ~1,410 tokens, which does not fit
    the task spec's 1,024 — and the MNN path silently scoring those rows would have hidden it.

    Counted with the ARTIFACT'S OWN tokenizer (`tokenizer_encode`), not a HuggingFace one: the
    exported `tokenizer.mtok` is what the runtime will actually use, so it is the only count that
    describes what happens next.
    """
    encode = getattr(llm, "tokenizer_encode", None)
    if not callable(encode):
        return
    with _suppress_c_stdout(not _MNN_VERBOSE):
        tokens = encode(rendered_prompt)
    if tokens is None:
        return
    budget = max_seq_length - max_new_tokens
    if len(tokens) > budget:
        raise ValueError(
            f"Rendered MNN prompt index {index} contains {len(tokens)} tokens, exceeding input "
            f"budget {budget} after reserving {max_new_tokens} output tokens inside configured "
            f"max sequence length {max_seq_length}."
        )


def mnn_decode_stats(llm) -> dict:
    """Prefill/decode token counts and microsecond timings for the last generation.

    Read from MNN's own context rather than measured around the call, so the numbers recorded in a
    report are the engine's accounting of itself: prompt length, tokens generated, and the two
    phases' costs separately (the split a phone's TTFT-vs-throughput budget is written against).
    """
    try:
        if hasattr(llm, "context"):
            context = llm.context
            data = {
                "prompt_len": context.prompt_len,
                "gen_seq_len": context.gen_seq_len,
                "prefill_us": context.prefill_us,
                "decode_us": context.decode_us,
            }
        else:
            raw = llm.get_context()
            data = {key: raw.get(key) for key in
                    ("prompt_len", "gen_seq_len", "prefill_us", "decode_us")}
        return data
    except Exception:  # noqa: BLE001 — telemetry must never break an eval
        return {}


def infer_batch_mnn(
    prompts: list[str],
    mnn_dir: str,
    max_new_tokens: int = 50,
    base_model: str | None = None,
    task: str = "",
) -> list[str]:
    """Run inference over all prompts against an MNN artifact via pymnn's LLM API.

    The MNN counterpart of `infer_batch_gguf`, with the same contract: greedy decoding, one
    independent prompt per row, output in input order. Caches the loaded model by
    (artifact, context length) exactly as the GGUF path does, because a reload costs seconds and
    every iteration scores hundreds of rows.

    SEQUENTIAL, WHICH IS PARITY WITH THE GGUF PATH RATHER THAN A LIMITATION OF THIS ONE.
    `MAX_GGUF_EVAL_CONCURRENCY` defaults to 1, so llama.cpp scores one row at a time too — and for
    a measured reason recorded there: at 8-way it bought ~20% for 8x the memory and killed a run
    with a SIGABRT the retry logic could not see. The same arithmetic is worse here, because an MNN
    `Llm` owns one session and one KV cache, so a second worker means a second full copy of the
    weights. `SLM_MNN_THREADS` (within one instance) and the GPU backend are the cheaper axes, and
    the GPU is where the 6x came from.

    Raises:
        ImportError: if pymnn with the LLM API is not installed.
    """
    requested = int(os.environ.get(_MNN_EVAL_CONCURRENCY_ENV, "1"))
    if requested > 1:
        # Refused rather than ignored. A variable that silently does nothing is worse than one that
        # is not supported, and the GGUF path's own default is 1 — so raising this would ask MNN for
        # MORE concurrency than llama.cpp takes, not the same.
        raise ValueError(
            f"{_MNN_EVAL_CONCURRENCY_ENV}={requested} is not implemented: MNN eval is sequential, "
            f"matching the GGUF path, whose MAX_GGUF_EVAL_CONCURRENCY also defaults to 1. Each MNN "
            f"worker would need its own copy of the weights, and the GPU backend "
            f"(SLM_MNN_BACKEND_TYPE) is where the throughput is."
        )
    max_seq_length = task_max_seq_length(task) if task else _DEFAULT_INFERENCE_SEQ_LENGTH
    backend = mnn_backend_type()
    # The backend is part of the key: a cached CPU model must never answer a request that asked for
    # the GPU, which is the same reason the GGUF cache keys on its context length.
    cache_key = (os.path.abspath(mnn_dir), max_seq_length, backend)
    if cache_key not in _mnn_cache:
        while len(_mnn_cache) >= _MAX_CACHED and _mnn_cache_order:
            _mnn_cache.pop(_mnn_cache_order.pop(0), None)
        started = time.time()
        _mnn_cache[cache_key] = load_mnn_llm(
            mnn_dir, max_seq_length=max_seq_length, max_new_tokens=max_new_tokens,
            backend_type=backend,
        )
        _mnn_cache_order.append(cache_key)
        print(
            f"      [mnn-eval] loaded {os.path.basename(mnn_dir.rstrip(os.sep))} in "
            f"{time.time() - started:.1f}s (backend={backend} threads={_MNN_THREADS} "
            f"precision={_MNN_PRECISION} memory={mnn_memory_mode(backend, mnn_dir)} "
            f"ctx={max_seq_length})"
        )

    llm = _mnn_cache[cache_key]
    outputs: list[str] = []
    started = time.time()
    for index, prompt in enumerate(prompts):
        rendered = _serving_prompt_prefix(prompt, base_model or "")
        # OUTSIDE the try below, deliberately: a prompt that does not fit the context is a
        # configuration fault that applies to every row, not one unscoreable row, so it must end
        # the eval rather than be absorbed into 800 empty predictions and a score of zero.
        _validate_mnn_budget(llm, rendered, index, max_seq_length, max_new_tokens)
        try:
            outputs.append(mnn_generate(llm, rendered, max_new_tokens=max_new_tokens))
        except Exception as error:  # noqa: BLE001 — one unscoreable row must not lose the eval
            print(
                f"      [mnn-eval] row {index} failed to decode "
                f"({type(error).__name__}: {str(error)[:160]}); scoring it as empty"
            )
            outputs.append("")
        if prompts and (index + 1) % 100 == 0:
            elapsed = time.time() - started
            print(
                f"      [mnn-eval] {index + 1}/{len(prompts)} rows in {elapsed:.0f}s "
                f"({(index + 1) / max(elapsed, 1e-9):.2f} rows/s)"
            )
    stats = mnn_decode_stats(llm)
    if stats:
        print(
            f"      [mnn-eval] {len(prompts)} rows in {time.time() - started:.0f}s; last row: "
            f"prompt={stats.get('prompt_len')} tok, generated={stats.get('gen_seq_len')} tok, "
            f"prefill={stats.get('prefill_us', 0) / 1000:.0f}ms, "
            f"decode={stats.get('decode_us', 0) / 1000:.0f}ms"
        )
    return outputs
