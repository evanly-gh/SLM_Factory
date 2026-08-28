"""Does the vLLM eval backend produce the SAME predictions as the in-process one, and how much faster?

WHY THIS EXISTS
    `eval/student_server.py` argues for prompt and sampling parity from construction: it renders with
    the same function, decodes with the same rule, and samples greedily like `do_sample=False`. That
    argument is necessary but not sufficient. Two engines can agree on all of that and still diverge —
    different kernels, a different rounding order in attention, a stop condition that fires one token
    earlier. On a task scored by exact JSON match, one token is the difference between 1.0 and 0.0.

    So the backend does not get turned on because it is faster. It gets turned on when this script
    shows it scores the same, on real rows, with the real weights. Run it once per (task, model) pair
    you intend to use it for, and read the SCORE row before the timing row.

USAGE
    python scripts/compare_eval_backends.py --task toolbench --base-model HuggingFaceTB/SmolLM2-135M \
        --weights-ref logs/runs/<run>/artifacts/adapter --rows 40

    Requires a GPU. Starts one vLLM engine (~1-2 min) and runs the in-process path once.

READING THE OUTPUT
    exact agreement    fraction of rows where both backends emitted byte-identical text. Expect high
                       but NOT 1.00 — greedy decode is not bit-reproducible across engines.
    score delta        the number that matters. The two backends must produce the same METRIC to
                       within noise; if they do not, the speedup is a measurement change and the
                       backend must stay off for this task.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--weights-ref", default="",
                        help="LoRA adapter directory. Omit to compare the BASE model, which still "
                             "tests engine parity and needs no trained artifact.")
    parser.add_argument("--rows", type=int, default=40)
    args = parser.parse_args()

    from data.eval_set import EvalSet
    from eval.harness import eval_output_token_reserve
    from tasks import get_task
    from training.slm_helpers import infer_batch, task_max_seq_length

    spec = get_task(args.task)
    _train, eval_rows = spec.load(max_train=8, max_test=args.rows)
    eval_set = EvalSet(all=list(eval_rows)[:args.rows], task=args.task)
    prompts = spec.build_prompts(eval_set)
    max_new_tokens = eval_output_token_reserve(spec.name)
    weights_ref = args.weights_ref or args.base_model
    print(f"task={args.task} rows={len(prompts)} max_new_tokens={max_new_tokens} "
          f"base={args.base_model} weights={weights_ref}")

    def scored(outputs: list[str]) -> tuple[float, float]:
        predictions = spec.extract_predictions(outputs, eval_set)
        result = spec.score(eval_set, predictions)
        return float(result["f1"]), float(result.get("format_valid", 1.0))

    print("\n--- in-process (HuggingFace padded batches) ---")
    started = time.perf_counter()
    in_process = infer_batch(prompts, weights_ref, args.base_model,
                             max_new_tokens=max_new_tokens, task=spec.name)
    in_process_s = time.perf_counter() - started
    in_process_score, in_process_format = scored(in_process)

    print("\n--- served (vLLM continuous batching) ---")
    from eval.student_server import infer_batch_served

    started = time.perf_counter()
    served = infer_batch_served(prompts, weights_ref, args.base_model,
                                max_new_tokens=max_new_tokens,
                                max_model_len=task_max_seq_length(spec.name))
    served_s = time.perf_counter() - started
    served_score, served_format = scored(served)

    identical = sum(1 for a, b in zip(in_process, served) if a == b)
    print("\n" + "=" * 72)
    print(f"exact agreement    {identical}/{len(prompts)} rows byte-identical "
          f"({identical / max(1, len(prompts)):.1%})")
    print(f"score              in-process {in_process_score:.4f}   served {served_score:.4f}   "
          f"delta {served_score - in_process_score:+.4f}   <-- READ THIS FIRST")
    print(f"format_valid       in-process {in_process_format:.4f}   served {served_format:.4f}")
    print(f"wall clock         in-process {in_process_s:.0f}s   served {served_s:.0f}s   "
          f"speedup {in_process_s / max(served_s, 1e-9):.2f}x "
          f"(served time INCLUDES engine startup, so the per-row gain is larger)")

    # A divergence in the metric is the disqualifying result, so say so rather than leaving it to be
    # inferred from two numbers printed next to each other.
    if abs(served_score - in_process_score) > 0.01:
        print("\nVERDICT: do NOT enable the served backend for this task. The two backends "
              "disagree on the metric by more than 0.01, so switching would change the "
              "measurement, not just its cost.")
        for index, (a, b) in enumerate(zip(in_process, served)):
            if a != b:
                print(f"\n  first divergence, row {index}:\n    in-process: {a[:300]!r}\n"
                      f"    served:     {b[:300]!r}")
                break
        return 1
    print("\nVERDICT: metric agrees within 0.01. Safe to set SLM_EVAL_BACKEND=vllm for this "
          "(task, model) pair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
