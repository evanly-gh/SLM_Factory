#!/usr/bin/env python3
"""One-off reclamation of checkpoint-only artifacts from retained runs.

Deletes three classes of dead weight (see
docs/superpowers/specs/2026-07-25-checkpoint-artifact-retention-design.md):

  1. HF Trainer `checkpoint-*` resume state  — dead once final_checkpoint exists
  2. GGUF dirs outside the new-best keep-set — write-once-read-once eval scratch
  3. langgraph.sqlite for terminated runs    — resume DB for runs that already ended

NEVER touches `final_checkpoint/`, datasets, logs, or JSON.

The GGUF keep-set is derived from the run logs, not from path globs: rows without a
`✗` in a run summary's per-tier iteration table set a new best, and each resolves to
`sha1(f"{weights_ref}|{quant}")[:12]` — the exact expression agent/nodes/evaluate.py
uses. The script aborts before deleting anything unless both known winner hashes are
in the derived keep-set, so a log-format drift fails loudly instead of silently
deleting a model we meant to keep.

Usage:
    python scripts/reclaim_checkpoint_artifacts.py            # dry run
    python scripts/reclaim_checkpoint_artifacts.py --apply    # delete
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Runs whose artifacts we keep. Anything not listed here was already removed.
RUNS = ["slm-ner-l40s-37531245", "slm-math-l40s-37576194", "20260721_020113_37387566"]

# Terminated runs whose LangGraph resume DB is dead weight.
SQLITE_RUNS = ["slm-ner-l40s-37531245", "slm-math-l40s-37576194"]

# The models the pipeline chose. Deletion is refused unless every adapter exists and
# every quantized winner is in the keep-set. `quant`/`safe` are None for the emotion
# run, which ran bf16 and produced no GGUF at all.
WINNERS = {
    "NER  iter46-d4": (
        f"{REPO}/logs/runs/slm-ner-l40s-37531245/artifacts/training"
        f"/Qwen_Qwen3.5-4B__Q4_K_M/iter46-d4/final_checkpoint",
        "Q4_K_M",
        "Qwen_Qwen3.5-4B",
    ),
    "MATH iter6-d2": (
        f"{REPO}/logs/runs/slm-math-l40s-37576194/artifacts/training"
        f"/Qwen_Qwen3.5-4B__Q4_K_M/iter6-d2/final_checkpoint",
        "Q4_K_M",
        "Qwen_Qwen3.5-4B",
    ),
    "EMO  iter53": (
        f"{REPO}/logs/runs/20260721_020113_37387566/artifacts/iter53/final_checkpoint",
        None,
        None,
    ),
}

TIER_RE = re.compile(r"──\s*Tier\s+(\d+):\s*(\S+)\s*\[([^\]]+)\]\s*\((\d+) iterations\)")
ROW_RE = re.compile(r"^\[[\d:]+\]\s+(\d+)\s+(\d\.\d+)(\s+✗)?\s")


def gguf_key(weights_ref: str, quant: str) -> str:
    """Mirror agent/nodes/evaluate.py:_build_or_reuse_gguf exactly."""
    return hashlib.sha1(f"{weights_ref}|{quant}".encode()).hexdigest()[:12]


def derive_keep_set() -> tuple[set[str], list[str]]:
    """Return ({"<model_safe>/<hash>"}, human-readable rows) for every new-best."""
    keep: set[str] = set()
    rows: list[str] = []
    for run in RUNS:
        log = f"{REPO}/logs/runs/{run}/run.log"
        if not os.path.isfile(log):
            continue
        tier = None
        with open(log, errors="replace") as fh:
            for line in fh:
                m = TIER_RE.search(line)
                if m:
                    tier = (m.group(2), m.group(3).strip())
                    continue
                if tier is None:
                    continue
                m = ROW_RE.match(line.strip())
                if not m or m.group(3):  # no match, or `✗` = not a new best
                    continue
                it, score = int(m.group(1)), float(m.group(2))
                model, quant = tier
                safe = model.replace("/", "_")
                cands = glob.glob(
                    f"{REPO}/logs/runs/{run}/artifacts/training/{safe}__{quant}/iter{it}-d*"
                )
                if len(cands) != 1:
                    rows.append(f"  !! {run} {model}[{quant}] iter{it}: {len(cands)} dirs matched")
                    continue
                h = gguf_key(os.path.join(cands[0], "final_checkpoint"), quant)
                found = os.path.isdir(f"{REPO}/artifacts/gguf/{safe}/{h}")
                if found:
                    keep.add(f"{safe}/{h}")
                rows.append(
                    f"  {run[:22]:22} {model:16} {quant:7} iter{it:<3} "
                    f"{score:.4f}  {h}  {'KEEP' if found else 'no-gguf'}"
                )
    return keep, rows


def dir_size(path: str) -> int:
    return sum(
        os.path.getsize(os.path.join(p, f))
        for p, _, fs in os.walk(path)
        for f in fs
        if os.path.exists(os.path.join(p, f))
    )


def gb(n: int) -> str:
    return f"{n / 2**30:.1f} GB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    keep, rows = derive_keep_set()
    print("=== new-best iterations (GGUF keep-set derivation) ===")
    print("\n".join(rows))

    print("\n=== winner safety check ===")
    ok = True
    for name, (wref, quant, safe) in WINNERS.items():
        adapter = os.path.isdir(wref)
        if quant is None:  # bf16 run, no GGUF to protect
            ok &= adapter
            print(f"  {name}: adapter_exists={adapter} (bf16, no gguf)")
            continue
        key = f"{safe}/{gguf_key(wref, quant)}"
        present = key in keep
        ok &= present and adapter
        print(f"  {name}: {key} in keep-set={present} adapter_exists={adapter}")
    if not ok:
        print("\nABORT: a winner is missing from the keep-set or its adapter is gone.")
        return 1

    # --- class 1: trainer checkpoint-* ---
    ckpts = []
    for run in RUNS:
        for dirpath, dirnames, _ in os.walk(f"{REPO}/logs/runs/{run}"):
            for d in list(dirnames):
                if d.startswith("checkpoint-"):
                    ckpts.append(os.path.join(dirpath, d))
                    dirnames.remove(d)  # don't descend into what we're deleting
    assert not any(p.endswith("final_checkpoint") for p in ckpts), "refusing: matched an adapter"

    # --- class 2: GGUF outside keep-set ---
    ggufs = [
        d
        for d in glob.glob(f"{REPO}/artifacts/gguf/*/*")
        if os.path.isdir(d)
        and f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}" not in keep
    ]

    # --- class 3: langgraph.sqlite for terminated runs ---
    sqlites = [
        p
        for p in (f"{REPO}/logs/runs/{r}/langgraph.sqlite" for r in SQLITE_RUNS)
        if os.path.isfile(p)
    ]

    c_sz = sum(dir_size(p) for p in ckpts)
    g_sz = sum(dir_size(p) for p in ggufs)
    s_sz = sum(os.path.getsize(p) for p in sqlites)

    print("\n=== deletion plan ===")
    print(f"  trainer checkpoint-*  : {len(ckpts):4} dirs   {gb(c_sz)}")
    print(f"  gguf (non new-best)   : {len(ggufs):4} dirs   {gb(g_sz)}")
    print(f"  langgraph.sqlite      : {len(sqlites):4} files  {gb(s_sz)}")
    print(f"  TOTAL                 :             {gb(c_sz + g_sz + s_sz)}")
    print(f"\n  retained gguf: {len(keep)} dirs")

    if not args.apply:
        print("\nDRY RUN — nothing deleted. Re-run with --apply.")
        return 0

    print("\n=== deleting ===")
    for p in ckpts:
        shutil.rmtree(p, ignore_errors=True)
    print(f"  removed {len(ckpts)} checkpoint-* dirs")
    for p in ggufs:
        shutil.rmtree(p, ignore_errors=True)
    print(f"  removed {len(ggufs)} gguf dirs")
    for p in sqlites:
        os.remove(p)
    print(f"  removed {len(sqlites)} sqlite files")

    # post-conditions
    for name, (wref, quant, safe) in WINNERS.items():
        assert os.path.isdir(wref), f"POST-CHECK FAILED: adapter gone for {name}"
        if quant is None:
            print(f"  verified {name}: adapter intact (bf16)")
            continue
        g = f"{REPO}/artifacts/gguf/{safe}/{gguf_key(wref, quant)}"
        assert os.path.isdir(g), f"POST-CHECK FAILED: gguf gone for {name}"
        print(f"  verified {name}: adapter + gguf intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
