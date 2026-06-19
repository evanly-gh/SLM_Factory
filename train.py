# Importing necessary libraries
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Defining model, token length, and dir to save lora adapter
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 512
ADAPTER_DIR = "./lora_adapter"

# This function loads the tokenizer, model, prepares it for LoRA fine-tuning on the MBPP dataset. 
# The MBPP dataset is converted to a format desired for causal language modeling for training.
# The trained LoRA adapter is then saved to disk.
def main():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False
    )

    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    print("\nNumber of Trainable Parameters:")
    model.print_trainable_parameters()

    mbpp_ds = load_dataset("mbpp", split="train")
    def format_example(example):
        return {
            "text": f"""
                Write a Python function.
                Problem:
                {example['text']}
                Return only Python code.
            """
        }
    mbpp_ds = mbpp_ds.map(format_example)
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )
    mbpp_ds = mbpp_ds.map(tokenize)
    mbpp_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Creating a data collator to handle the batching and padding of the input data.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir="./out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        fp16=False,
        bf16=False,
        report_to="none",
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=mbpp_ds,
        data_collator=data_collator 
    )
    trainer.train()

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\nSaved LoRA adapter to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()