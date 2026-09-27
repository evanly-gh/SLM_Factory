#!/usr/bin/env python3
"""
merge_lora_adapter.py — Merges a LoRA adapter into its base model, producing
a single, complete HuggingFace checkpoint.

Pipeline:
  1. Read adapter_config.json to determine base_model_name_or_path
  2. Load the base model (bf16) via transformers
  3. Load the LoRA adapter on top via peft, then merge_and_unload()
  4. Save the merged model, plus the adapter folder's own tokenizer/chat
     template files (not the base model's), to --output

Usage:
    python3 merge_lora_adapter.py --adapter lora_adapter/ --output merged_model/

Requires: transformers, peft, torch
    pip install transformers peft --break-system-packages
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# convert_hf_to_gguf.py/convert_to_gguf.py's own precedent for this exact
# failure mode: loading torch (and transitively numpy) can end up with two
# copies of libomp linked in, which aborts with "OMP: Error #15:
# Initializing libomp.dylib, but found libomp.dylib already initialized."
# Set before importing torch so the workaround actually takes effect.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Tokenizer/template files that reflect the fine-tuned model's actual
# tokenizer and chat template - copied from the ADAPTER folder, not
# regenerated from the base model, since fine-tuning may have changed added
# tokens, special tokens, or the chat template itself.
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
]


def check_dependencies():
    """Verify transformers/peft/torch are importable before doing any real
    work, with a clear install command if not - failing fast here beats a
    confusing traceback partway through loading a multi-GB model."""
    missing = []
    for module in ("torch", "transformers", "peft"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        print(f"[ERROR] Missing required package(s): {', '.join(missing)}")
        print("Install with: pip install transformers peft torch --break-system-packages")
        sys.exit(1)


def read_base_model_name(adapter_dir: Path) -> str:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        print(f"[ERROR] adapter_config.json not found in {adapter_dir}")
        sys.exit(1)

    try:
        with open(config_path) as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] Failed to read/parse {config_path}: {exc}")
        sys.exit(1)

    base_model = config.get("base_model_name_or_path")
    if not base_model:
        print(f"[ERROR] 'base_model_name_or_path' not found in {config_path}")
        sys.exit(1)

    return base_model


def dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 2)


def copy_tokenizer_files(adapter_dir: Path, output_dir: Path):
    copied = []
    for name in TOKENIZER_FILES:
        src = adapter_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)
            copied.append(name)
    return copied


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge a LoRA adapter into its base model, producing a complete HF checkpoint."
    )
    p.add_argument(
        "--adapter", required=True,
        help="Path to the adapter folder (adapter_model.safetensors, adapter_config.json, tokenizer files, chat_template.jinja).",
    )
    p.add_argument("--output", required=True, help="Output directory for the merged model.")
    return p.parse_args()


def main():
    args = parse_args()

    adapter_dir = Path(args.adapter).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not adapter_dir.is_dir():
        print(f"[ERROR] Adapter path is not a directory: {adapter_dir}")
        sys.exit(1)

    check_dependencies()

    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base_model_name = read_base_model_name(adapter_dir)
    print(f"[INFO] Adapter: {adapter_dir}")
    print(f"[INFO] Base model (from adapter_config.json): {base_model_name}")
    print(f"[INFO] Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    print(f"\n[LOAD] Loading base model '{base_model_name}' (bf16) …")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
    )
    print("[LOAD] ✓ Base model loaded")

    print(f"\n[LOAD] Loading LoRA adapter from {adapter_dir} …")
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    print("[LOAD] ✓ Adapter loaded")

    print("\n[MERGE] Merging adapter weights into base model (merge_and_unload) …")
    merged_model = merged_model.merge_and_unload()
    print("[MERGE] ✓ Merge complete")

    print(f"\n[SAVE] Saving merged model to {output_dir} …")
    merged_model.save_pretrained(str(output_dir))
    print("[SAVE] ✓ Model weights saved")

    print(f"\n[SAVE] Copying tokenizer/chat-template files from adapter folder …")
    copied = copy_tokenizer_files(adapter_dir, output_dir)
    if copied:
        print(f"[SAVE] ✓ Copied: {', '.join(copied)}")
    else:
        print("[SAVE] [WARN] No tokenizer files found in the adapter folder to copy.")

    elapsed = time.time() - t_start
    total_size_mb = dir_size_mb(output_dir)

    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Base model:    {base_model_name}")
    print(f"Adapter:       {adapter_dir}")
    print(f"Output:        {output_dir}")
    print(f"Total size:    {total_size_mb:.0f} MB")
    print(f"Elapsed:       {elapsed:.1f}s")
    print("=" * 60)
    print("\n[DONE] Merged model is a complete, standalone HuggingFace checkpoint.")


if __name__ == "__main__":
    main()
