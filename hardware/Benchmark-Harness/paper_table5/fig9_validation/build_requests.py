#!/usr/bin/env python3
"""
Build the Fig. 9a/9b validation request sets from each run's frozen held-out set.

Runs on the cluster / any checkout of SLM_Factory with the run directories and the repo's Python environment
(it imports the factory's own task code, so the prompts are byte-identical to what the scorer sent):

  OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false .venv/bin/python \
      hardware/Benchmark-Harness/paper_table5/fig9_validation/build_requests.py

For each scenario it writes requests/<scenario>.jsonl and requests/<scenario>.meta.json. The phone runner
(run_fig9.py) needs only those two files, not the factory code.

Per request:
  * the prompt exactly as the factory scored the MNN model: the task's build_prompts() on the eval row, rendered by
    training.slm_helpers._serving_prompt_prefix (ChatML user turn + assistant turn with an empty think block), sent
    with use_template=false; tokenized with the run's own tokenizer (final_checkpoint/tokenizer.json), so the ids are
    the ones MNN would get from tokenizer.mtok.
  * the reference answer as the model was trained to emit it (the task's build_training_turn target: the entity JSON
    for BC5CDR, references[0] for DialogSum), and its token count without the end-of-turn token.
  * which rows: the first N rows of the frozen eval set in its stored (already shuffled) order, i.e. N consecutive
    held-out requests; W warm-up rows come from the END of the eval set so they never overlap the measured ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]  # fig9_validation -> paper_table5 -> Benchmark-Harness -> hardware -> repo
RUNS = REPO / "logs" / "runs" / "phone_in_the_loop"

SCENARIOS = {
    "m07": {"run": "slm-ner-bc5cdr-m07-op8pro-l40s-42809109", "base_model": "Qwen/Qwen3.5-0.8B",
            "weights": "Qwen_Qwen3.5-0.8B__Q4_K_M/iter15-d2"},
    "h09": {"run": "slm-dialogsum-h09-s23-l40s-43019548", "base_model": "Qwen/Qwen3-0.6B",
            "weights": "Qwen_Qwen3-0.6B__Q8_0/iter14-d6"},
}


def pct(xs, q):
    """Linear-interpolated percentile (numpy's default), q in [0, 100]."""
    s = sorted(xs)
    if not s:
        return None
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def dist(xs):
    return {"n": len(xs), "min": min(xs), "median": pct(xs, 50), "mean": round(statistics.mean(xs), 1),
            "p95": pct(xs, 95), "max": max(xs), "sum": sum(xs)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build(name: str, n: int, warmup: int, out_dir: Path):
    from tokenizers import Tokenizer

    from data.eval_set import EvalSet
    from eval.harness import eval_output_token_reserve
    from tasks import get_task
    from training.slm_helpers import _serving_prompt_prefix

    sc = SCENARIOS[name]
    run_dir = RUNS / sc["run"]
    eval_path = run_dir / "artifacts" / "eval_set.json"
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    rows = payload.get("all") if payload.get("all") is not None else payload.get("examples")
    eval_set = EvalSet.from_serialized({**payload, "all": rows})  # as scripts/export_mnn_winners.py loads it
    task = get_task(eval_set.task)
    prompts = task.build_prompts(eval_set)
    ctx = SimpleNamespace(labels=(), instruction="")
    tok_path = run_dir / "artifacts" / "training" / sc["weights"] / "final_checkpoint" / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    eos_text = "<|im_end|>"
    eos_id = tok.token_to_id(eos_text)
    cap = eval_output_token_reserve(eval_set.task)
    total = len(eval_set.all)
    if n + warmup > total:
        sys.exit(f"{name}: {n} + {warmup} rows requested, eval set has {total}")

    order = [(i, False) for i in range(n)] + [(total - 1 - j, True) for j in range(warmup)]
    out = []
    for i, warm in order:
        row = eval_set.all[i]
        rendered = _serving_prompt_prefix(prompts[i], sc["base_model"])
        train_prompt, target, _marker = task.build_training_turn(row, ctx)
        if train_prompt != prompts[i]:  # train/serve prompt parity: the target belongs to this prompt
            sys.exit(f"{name} row {i}: training prompt differs from the scorer's prompt")
        ids = tok.encode(rendered, add_special_tokens=False).ids
        ans = tok.encode(target, add_special_tokens=False).ids
        if ids.count(tok.token_to_id("<|im_start|>")) != 2:
            sys.exit(f"{name} row {i}: special tokens not parsed as single ids")
        out.append({
            "seq": len(out), "eval_index": i, "warmup": warm,
            "prompt_tokens": len(ids), "answer_tokens": len(ans),
            # forced-length decode: answer_tokens forward passes, the same work as the real answer + end-of-turn
            "max_new_tokens_reference": max(1, len(ans)),
            "max_new_tokens_eos": cap,  # the scorer's cap when the model decodes to its own end-of-turn
            "prompt_ids": ids, "rendered_prompt": rendered, "reference_answer": target,
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{name}.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meas = [r for r in out if not r["warmup"]]
    meta = {
        "scenario": name, "run": sc["run"], "task": eval_set.task, "base_model": sc["base_model"],
        "eval_set": {"path": f"logs/runs/phone_in_the_loop/{sc['run']}/artifacts/eval_set.json",
                     "sha256": sha256(eval_path), "rows": total},
        "tokenizer": {"path": str(tok_path.relative_to(REPO)), "sha256": sha256(tok_path),
                      "end_of_turn": eos_text, "end_of_turn_id": eos_id},
        "selection": f"eval rows 0..{n - 1} in stored order (measured); rows {total - warmup}..{total - 1} (warm-up)",
        "prompt_format": "training.slm_helpers._serving_prompt_prefix(task.build_prompts(row)), use_template=false",
        "answer": "task.build_training_turn target; answer_tokens excludes the end-of-turn token",
        "eval_max_new_tokens": cap,
        "measured": {"prompt_tokens": dist([r["prompt_tokens"] for r in meas]),
                     "answer_tokens": dist([r["answer_tokens"] for r in meas])},
        "warmup_rows": warmup,
        "jsonl_sha256": sha256(jsonl),
    }
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    p, a = meta["measured"]["prompt_tokens"], meta["measured"]["answer_tokens"]
    print(f"{name}: {len(meas)} requests (+{warmup} warm-up) -> {jsonl}\n"
          f"   prompt tokens median {p['median']} p95 {p['p95']:.0f} (min {p['min']}, max {p['max']}); "
          f"answer tokens median {a['median']} p95 {a['p95']:.0f} (min {a['min']}, max {a['max']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    ap.add_argument("--n", type=int, default=300, help="measured requests per scenario")
    ap.add_argument("--warmup", type=int, default=3, help="warm-up requests (not measured)")
    ap.add_argument("--out", default=str(HERE / "requests"))
    a = ap.parse_args()
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    for name in a.scenarios:
        build(name, a.n, a.warmup, Path(a.out))


if __name__ == "__main__":
    main()
