#!/usr/bin/env python3
"""
Benchmark a model pool (e.g. SLM_Factory's config/android_pool.py) end to end with the Table 5 protocol:

  python run_pool.py --pool-file ../SLM_Factory/config/android_pool.py --serial S
  python run_pool.py --pool-file pool.txt --serial S --gguf Q4_0 Q8_0 F16 --mnn 4 8 16 --llama-threads 4
  python run_pool.py --pool-file pool.txt --serial S --plan        # show what would run

For each build ref the pool needs (pinned = the paper's llama.cpp/MNN; head = current upstream, for architectures
newer than the pinned versions, e.g. Qwen3.5): build and push the binaries if missing, convert the models
(prepare_models.py --pool-file), run them (run_paper_table5.py --pool-file, resumable), then write one report
(report.py) over all result directories. Options not listed below are passed to run_paper_table5.py
(e.g. --llama-threads 4, --quants Q4_0 Q4, --backends cpu).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import RESULTS_DIR, THIRD_PARTY  # noqa: E402
from pool import load_pool  # noqa: E402


def step(cmd: list[str]):
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"step failed ({r.returncode}): {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-file", required=True)
    ap.add_argument("--serial", required=True)
    ap.add_argument("--gguf", nargs="*", default=["Q4_0"], help="GGUF quants (default Q4_0: the paper's w4 that "
                    "llama.cpp's OpenCL backend runs natively)")
    ap.add_argument("--mnn", nargs="*", default=["4"], help="MNN quant_bit levels")
    ap.add_argument("--mnn-recipe", choices=["paper-default", "pool"], default="paper-default")
    ap.add_argument("--quantize-bin", help="host llama-quantize (else prepare_models.py's lookup / the phone)")
    ap.add_argument("--results-prefix", help="results/<prefix>_<ref> (default: the pool file's name)")
    ap.add_argument("--plan", action="store_true", help="print the pool and the per-ref steps, then exit")
    args, run_args = ap.parse_known_args()

    pool = load_pool(args.pool_file)
    refs = [r for r in ("pinned", "head") if any(ref == r for _, ref in pool.values())]
    prefix = args.results_prefix or re.sub(r"[^A-Za-z0-9_-]+", "_", Path(args.pool_file).stem)
    print(f"pool {args.pool_file}: {len(pool)} models")
    for name, (hf_id, ref) in pool.items():
        print(f"  {name:28s} {hf_id:45s} {ref}")
    if args.plan:
        for ref in refs:
            print(f"\n{ref}: binaries {THIRD_PARTY / 'out' / ref}, results {RESULTS_DIR / f'{prefix}_{ref}'}")
        return

    dirs = []
    for ref in refs:
        if not (THIRD_PARTY / "out" / ref / "mnn" / "llm_bench").is_file():
            step(["build_binaries.py", "--ref", ref, "--push", "--serial", args.serial])
        else:
            step(["build_binaries.py", "--ref", ref, "--push-only", "--serial", args.serial])
        prep = ["prepare_models.py", "--pool-file", args.pool_file, "--ref", ref, "--gguf", *args.gguf,
                "--mnn", *args.mnn, "--mnn-recipe", args.mnn_recipe, "--serial", args.serial]
        if args.quantize_bin:
            prep += ["--quantize-bin", args.quantize_bin]
        step(prep)
        out = RESULTS_DIR / f"{prefix}_{ref}"
        step(["run_paper_table5.py", "--serial", args.serial, "--ref", ref, "--pool-file", args.pool_file,
              "--results-dir", str(out), *run_args])
        dirs.append(str(out))
    step(["report.py", *dirs, "--out", str(RESULTS_DIR / f"{prefix}_report")])


if __name__ == "__main__":
    main()
