# training/slm_helpers.py
"""
Agent-facing interface for training and inference.
Called by the agent via the bash tool.
Equivalent to the paper's tinker_helpers.py.
"""
import os
import json
from concurrent.futures import ThreadPoolExecutor
from training.lora_trainer import TrainingConfig, run_lora_training

# Lazy-loaded inference model cache: weights_ref -> (model, tokenizer)
_inference_cache: dict = {}

def train(
    dataset_path: str,
    base_model: str,
    nr_epochs: int,
    learning_rate: float,
    batch_size: int,
    lora_rank: int | None,
    output_dir: str = "artifacts",
    task_type: str = "classification",
) -> str:
    """
    Execute the full LoRA training loop.
    Returns weights_ref string for immediate inference.
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
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=weights_ref, max_seq_length=512, load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)   # 2x faster, sets the patched fast path
        _inference_cache[weights_ref] = (model, tokenizer)

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
