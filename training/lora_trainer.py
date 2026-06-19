# training/lora_trainer.py
import os, json
from dataclasses import dataclass

VALID_LORA_RANKS = {4, 8, 16, 32, 64}

@dataclass
class TrainingConfig:
    base_model: str
    nr_epochs: int
    learning_rate: float
    batch_size: int
    lora_rank: int | None  # None = full fine-tune

    def __post_init__(self):
        if self.lora_rank is not None and self.lora_rank not in VALID_LORA_RANKS:
            raise ValueError(f"lora_rank must be one of {VALID_LORA_RANKS}, got {self.lora_rank}")
        if not (0 < self.learning_rate < 1):
            raise ValueError(f"learning_rate must be in (0,1), got {self.learning_rate}")

def _run_unsloth_training(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str,
) -> str:
    """
    Run LoRA fine-tuning via Unsloth.
    Returns the path to the saved checkpoint directory.
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
    )

    if config.lora_rank is not None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            lora_alpha=config.lora_rank * 2,
            lora_dropout=0.0,
            target_modules="all-linear",
            bias="none",
        )

    # Load dataset
    with open(dataset_path) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    PROMPT_TEMPLATE = (
        "Classify this SMS message as spam or ham.\n\nMessage: {text}\n\nLabel: {label}"
    )

    def format_example(ex):
        return {"text": PROMPT_TEMPLATE.format(**ex)}

    from datasets import Dataset
    dataset = Dataset.from_list([format_example(e) for e in raw])

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.nr_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=args,
    )
    trainer.train()

    checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    return checkpoint_path

def run_lora_training(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str = "artifacts",
) -> str:
    """
    Train model with LoRA (or full fine-tune if lora_rank is None).
    Returns weights_ref — a path string usable by infer() and infer_batch().
    Always trains from the base model, never from a prior checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = _run_unsloth_training(dataset_path, config, output_dir)
    return checkpoint_path
