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
    """Load checkpoint and generate text. No deployment step required."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    key = weights_ref
    if key not in _inference_cache:
        tokenizer = AutoTokenizer.from_pretrained(weights_ref)
        model = AutoModelForCausalLM.from_pretrained(
            weights_ref,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        _inference_cache[key] = (model, tokenizer)

    model, tokenizer = _inference_cache[key]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Return only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def infer_batch(
    prompts: list[str],
    weights_ref: str,
    base_model: str,
    max_new_tokens: int = 50,
    max_workers: int = 20,
) -> list[str]:
    """Parallel inference via ThreadPoolExecutor with max_workers=20."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(infer, p, weights_ref, base_model, max_new_tokens)
            for p in prompts
        ]
        return [f.result() for f in futures]
