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

    # Loading tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    # Configuring LoRA parameters for fine-tuning
    # We only care about q_proj and v_proj layers for LoRA fine-tuning, 
    # q_proj determines which queries to pay attention to, while v_proj decides which information
    # should get passed along.
    # lora_alpha controls the scaling of the LoRA updates, and lora_dropout adds regularization to prevent overfitting during training.
    # r is the rank of the low-rank decomposition, which determines how many parameters are added to the model for fine-tuning.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False
    )

    # Applying LoRA configuration to the model
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False

    # Checking the number of trainable parameters in the model after applying LoRA
    print("\nNumber of Trainable Parameters:")
    model.print_trainable_parameters()

    # Loading the MBPP dataset and formatting it for training. 
    # Each example is converted into a prompt format suitable for causal language modeling.
    # Will convert each example into a single string and tokenize it for fine-tuning.
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
    # This is so that the model receives properly formatted batches during training, with padding applied to the input sequences as needed,
    # so that everything is the same length within a batch.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Defining training arguments for fine-tuning the model on the MBPP dataset on a local machine.
    # Then, we'll train the model using the Trainer API from Hugging Face, which will handle the training loop, optimization, and logging.
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

    # After training, we save the fine-tuned LoRA adapter to disk. 
    # This will allow us to load the adapter later for evaluation or inference without needing to retrain the model.
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\nSaved LoRA adapter to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()