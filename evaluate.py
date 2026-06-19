# Importing necessary libraries for evaluation
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from human_eval.data import read_problems
from human_eval.execution import check_correctness
import os

# Disable parallelism warnings from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Defining model name and directory to load lora adapter
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ADAPTER_DIR = "./lora_adapter"

# Extract generated python code from the model's output.
def clean_completion(text: str):
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1]
    if "def" in text:
        text = text[text.index("def"):]
    return text.strip()

# Load the base model and the fine-tuned LoRA adapter, and prepare them for evaluation. 
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer

# Generates complete Python function code for a given prompt using the loaded model and tokenizer.
def generate(model, tokenizer, prompt):

    input_text = f"""
        You are a Python coding assistant.

        Return only a complete Python function.

        Rules:
        - No explanation
        - No markdown
        - Must start with def

        Problem:
        {prompt}
    """
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return clean_completion(decoded)

# Main function to load the model, read problems, generate completions, and evaluate correctness.
def main():
    model, tokenizer = load_model()
    problems = read_problems()
    total = 0
    passed = 0

    for task_id, task in tqdm(problems.items()):
        try:
            completion = generate(model, tokenizer, task["prompt"])
            result = check_correctness(
                task,
                completion,
                timeout=3.0
            )
            passed += int(result["passed"])
            total += 1
        except Exception:
            total += 1

    print(f"HumanEval pass@1: {passed / total:.4f}")

if __name__ == "__main__":
    main()