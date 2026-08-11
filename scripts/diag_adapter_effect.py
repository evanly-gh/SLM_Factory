"""Verify a LoRA adapter actually changes inference at eval time (B254).

Generates from the base model and from the production eval loader with a trained adapter.
Identical outputs mean the adapter is inert and every "fine-tuned" score for that tier is
really the base model's score.

Run under .venv_gpu on a GPU node. Diagnostic only; not part of the pipeline.
"""
import os
import sys

os.environ.setdefault("ANTHROPIC_API_KEY", "diag")
os.environ.setdefault("EXA_API_KEY", "diag")

BASE = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
ADAPTER = sys.argv[2]

PROMPTS = [
    "Summarize the following conversation in one to three sentences. Write only the summary "
    "— do not continue the conversation or reply to it.\n\n"
    "#Person1#: Hey, are you coming to the team lunch tomorrow?\n"
    "#Person2#: I wish I could, but I have a dentist appointment at noon.\n"
    "#Person1#: That's too bad. I'll save you a slice of cake.\n",
    "Summarize the following conversation in one to three sentences. Write only the summary "
    "— do not continue the conversation or reply to it.\n\n"
    "#Person1#: Did you finish the quarterly report?\n"
    "#Person2#: Almost. I still need the numbers from marketing.\n"
    "#Person1#: I'll ping them now so you can wrap it up today.\n",
]


def lora_b_report(model, label):
    nonzero, zero, peak = 0, 0, 0.0
    for name, param in model.named_parameters():
        if "lora_B" not in name:
            continue
        norm = param.detach().float().norm().item()
        peak = max(peak, norm)
        if norm > 0:
            nonzero += 1
        else:
            zero += 1
    print(f"  [{label}] lora_B: nonzero={nonzero} zero={zero} max_norm={peak:.4f}")
    return nonzero


def run(weights_ref, label):
    from training.slm_helpers import clear_inference_cache, infer_batch

    clear_inference_cache()
    outs = infer_batch(PROMPTS, weights_ref, BASE, max_new_tokens=64, task_type="generation")
    print(f"\n===== {label} =====")
    for i, out in enumerate(outs):
        print(f"  [{i}] {out.strip()[:200]!r}")
    return outs


def main():
    from training.slm_helpers import _load_inference_model, clear_inference_cache

    clear_inference_cache()
    model, _ = _load_inference_model(ADAPTER, BASE, 4096)
    print("\n--- adapter weights resident in the loaded model ---")
    nonzero = lora_b_report(model, "production eval loader")

    base_out = run(BASE, "BASE (no adapter)")
    tuned_out = run(ADAPTER, "BASE + trained adapter")

    identical = base_out == tuned_out
    print("\n" + "=" * 72)
    print(f"  lora_B tensors populated : {nonzero}")
    print(f"  outputs identical to base: {identical}")
    if identical or nonzero == 0:
        print("VERDICT: ADAPTER IS INERT — eval would score the base model")
        print("=" * 72)
        return 1
    print("VERDICT: adapter is APPLIED — fine-tuning affects inference")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
