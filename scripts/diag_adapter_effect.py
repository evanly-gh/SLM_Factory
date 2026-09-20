"""Verify a LoRA adapter actually changes inference at eval time (B254).

Generates from the base model and from the production eval loader with a trained adapter.
Identical outputs mean the adapter is inert and every "fine-tuned" score for that tier is
really the base model's score.

The probe runs against `dialogsum`, whose adapter the original defect was found on. Naming the
task is not optional any more: batch size and the context ceiling are read from the task's spec,
so `infer_batch` with no task raises rather than silently taking a generic default.

Run under .venv_gpu on a GPU node. Diagnostic only; not part of the pipeline.
"""
import os
import sys

os.environ.setdefault("ANTHROPIC_API_KEY", "diag")
os.environ.setdefault("EXA_API_KEY", "diag")

BASE = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-0.8B"
ADAPTER = sys.argv[2]

TASK = "dialogsum"

DIALOGUES = [
    "#Person1#: Hey, are you coming to the team lunch tomorrow?\n"
    "#Person2#: I wish I could, but I have a dentist appointment at noon.\n"
    "#Person1#: That's too bad. I'll save you a slice of cake.\n",
    "#Person1#: Did you finish the quarterly report?\n"
    "#Person2#: Almost. I still need the numbers from marketing.\n"
    "#Person1#: I'll ping them now so you can wrap it up today.\n",
]


def build_prompts():
    """The prompts the eval harness would send for these dialogues.

    Built through the task's own prompt builder rather than pasted in, because a diagnostic that
    asks "does the adapter change the output" is worthless if it asks a question the adapter was
    never trained on — the summarization instruction is carried on the rows and resolved by the
    scorer, and a hand-copied prefix is exactly what drifts (B250).
    """
    from data.eval_set import EvalSet
    from data.loaders.dialogsum import SUMMARIZATION_INSTRUCTION
    from tasks import get_task

    # `references` alongside `answer`: the reworked task grades against a reference LIST, and its
    # `require_fields` step checks for it. Empty here because this diagnostic only asks whether
    # the adapter changes the OUTPUT — nothing is scored.
    rows = [
        {"text": dialogue, "answer": "", "references": [],
         "_instruction": SUMMARIZATION_INSTRUCTION}
        for dialogue in DIALOGUES
    ]
    return get_task(TASK).build_prompts(EvalSet(all=rows, task=TASK))


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


def run(prompts, weights_ref, label):
    from training.slm_helpers import clear_inference_cache, infer_batch

    clear_inference_cache()
    outs = infer_batch(prompts, weights_ref, BASE, max_new_tokens=64, task=TASK)
    print(f"\n===== {label} =====")
    for i, out in enumerate(outs):
        print(f"  [{i}] {out.strip()[:200]!r}")
    return outs


def main():
    from training.slm_helpers import (
        _load_inference_model,
        clear_inference_cache,
        task_max_seq_length,
    )

    prompts = build_prompts()
    clear_inference_cache()
    model, _ = _load_inference_model(ADAPTER, BASE, task_max_seq_length(TASK))
    print("\n--- adapter weights resident in the loaded model ---")
    nonzero = lora_b_report(model, "production eval loader")

    base_out = run(prompts, BASE, "BASE (no adapter)")
    tuned_out = run(prompts, ADAPTER, "BASE + trained adapter")

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
