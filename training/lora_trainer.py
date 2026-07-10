# training/lora_trainer.py
import collections
import os, json
from dataclasses import dataclass

TrainingOutput = collections.namedtuple("TrainingOutput", ["weights_ref", "gguf_path"])

VALID_LORA_RANKS = {4, 8, 16, 32, 64}

@dataclass
class TrainingConfig:
    base_model: str
    nr_epochs: int
    learning_rate: float
    batch_size: int
    lora_rank: int | None  # None = full fine-tune
    task_type: str = "classification"

    def __post_init__(self):
        if self.lora_rank is not None and self.lora_rank not in VALID_LORA_RANKS:
            raise ValueError(f"lora_rank must be one of {VALID_LORA_RANKS}, got {self.lora_rank}")
        if not (0 < self.learning_rate < 1):
            raise ValueError(f"learning_rate must be in (0,1), got {self.learning_rate}")

def _run_unsloth_training(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str,
    task_type: str = "classification",
) -> str:
    """
    Run LoRA fine-tuning via Unsloth.
    Returns the path to the saved checkpoint directory.
    task_type controls the prompt format:
      - "classification": label classification prompt
      - "NER": entity extraction prompt
      - "generation": text-to-answer prompt
    """
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import torch

    max_seq_length = 512
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=config.lora_rank is not None,
        # Many pool models (MiniCPM4/5, Qwen3.5, Gemma3n) ship custom modeling code and
        # will not load without this — transformers raises and asks for it explicitly.
        trust_remote_code=True,
    )

    if config.lora_rank is not None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            lora_alpha=config.lora_rank * 2,
            lora_dropout=0.0,
            # Unsloth expects a LIST of module names, not the string "all-linear"
            # (a string gets iterated character-by-character by PEFT).
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )

    # Load dataset
    with open(dataset_path, encoding="utf-8") as f:
        raw = [json.loads(line) for line in f if line.strip()]

    # Format training examples using the model's chat template when available.
    # This achieves train/serve parity (paper Appendix D.1) and structurally
    # separates the prompt from the completion so loss masking works correctly.
    has_chat_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )

    if task_type == "classification":
        from eval.scorers.classification import CLASSIFY_PROMPT
        def format_example(ex):
            user_msg = CLASSIFY_PROMPT.format(text=ex.get("text", ""))
            assistant_msg = ex.get("label", "")
            if has_chat_template:
                messages = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
                return {"text": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )}
            return {"text": f"{user_msg}\n\nAnswer: {assistant_msg}"}
    elif task_type == "NER":
        def format_example(ex):
            entities_str = json.dumps(ex.get("entities", []))
            user_msg = (
                "Extract named entities from the text. "
                "Reply with a JSON list of objects with \"text\" and \"type\" keys.\n\n"
                f"Text: {ex['text']}"
            )
            assistant_msg = entities_str
            if has_chat_template:
                messages = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
                return {"text": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )}
            return {"text": f"{user_msg}\n\nEntities: {assistant_msg}"}
    elif task_type in ("generation", "math_reasoning", "code_generation"):
        def format_example(ex):
            user_msg = ex.get("text", ex.get("prompt", ""))
            raw_answer = ex.get("answer", ex.get("response", ex.get("label", "")))
            cot = ex.get("cot_reasoning", "")
            assistant_msg = f"<reasoning>\n{cot}\n</reasoning>\n\n{raw_answer}" if cot else raw_answer
            if has_chat_template:
                messages = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
                return {"text": tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )}
            return {"text": f"{user_msg}\n\nAnswer: {assistant_msg}"}
    else:
        def format_example(ex):
            return {"text": ex.get("text", "")}

    from datasets import Dataset
    dataset = Dataset.from_list([format_example(e) for e in raw])

    from trl import SFTConfig
    _bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    # trl >= ~0.20 moved the SFT-specific knobs into SFTConfig: dataset_text_field and
    # max_length live there (max_seq_length was renamed to max_length), and the tokenizer
    # is passed as processing_class. The previous code passed max_seq_length/tokenizer as
    # bare SFTTrainer kwargs, which the current trl silently ignores — so NO truncation was
    # applied. Long (math CoT / NER abstract) sequences then reached Unsloth's fused
    # cross-entropy with mismatched logits vs label lengths and crashed (see BUGS B109).
    # Build an SFTConfig with an explicit max_length so sequences are truncated to 512.
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.nr_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        fp16=not _bf16 and torch.cuda.is_available(),
        bf16=_bf16,
        logging_steps=10,
        # "no" avoids mid-training checkpointing, which torch.save's the trainer args and
        # fails under Unsloth's patched SFTConfig (pickle identity mismatch). We persist the
        # final model explicitly via save_pretrained below.
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=max_seq_length,
    )
    try:
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer,
            train_dataset=dataset, args=sft_config,
        )
    except TypeError:
        # Older trl still accepts tokenizer + dataset_text_field on the trainer itself.
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, train_dataset=dataset,
            dataset_text_field="text", args=sft_config,
        )
    trainer.train()

    checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    return checkpoint_path

def merge_for_quantization(checkpoint_path: str, output_dir: str) -> str:
    """
    Merge LoRA adapters into the base model weights and save as a full HF checkpoint.
    Required before GGUF quantization — GGUF cannot be produced from adapter-only checkpoints.
    Returns path to the merged HF checkpoint directory.
    """
    from unsloth import FastLanguageModel
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=512,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    return merged_dir


def run_lora_training(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str = "artifacts",
    task_type: str | None = None,
) -> TrainingOutput:
    """
    Train model with LoRA (or full fine-tune if lora_rank is None).
    Returns TrainingOutput(weights_ref, gguf_path=None).
    weights_ref is a path string usable by infer() and infer_batch().
    gguf_path is always None here — set by the quantize step in evaluate_node.
    Always trains from the base model, never from a prior checkpoint.
    task_type defaults to config.task_type if not provided.
    """
    os.makedirs(output_dir, exist_ok=True)
    effective_task_type = task_type if task_type is not None else config.task_type
    checkpoint_path = _run_unsloth_training(dataset_path, config, output_dir, task_type=effective_task_type)
    return TrainingOutput(weights_ref=checkpoint_path, gguf_path=None)
