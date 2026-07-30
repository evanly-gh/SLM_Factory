#!/bin/bash
#SBATCH --job-name=b113-cold
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/gpu_setup/b113-cold-%j.out
#
# B113 reproduction test — does Qwen/Qwen3.5-2B download cleanly on a COMPUTE NODE?
#
# WHY THIS EXISTS
#   The pool-measurement sweep (job 37905779) built 3.5-2B successfully and I wrongly called
#   that a reproduction test. It was not: the shared cache at .hf-cache already held a complete
#   snapshot (blobs timestamped 2026-07-10 11:32-11:33, zero broken symlinks, zero .incomplete
#   blobs), so resolve_hf_snapshot found it locally and NEVER CONTACTED THE HUB. The download
#   path — the thing that failed on 2026-07-10 — was not executed at all.
#
#   This job forces the download by pointing HF_HOME at a fresh scratch directory. The shared
#   cache is never touched, read, or modified.
#
# WHAT IT TESTS, in order
#   1. Cold download of Qwen/Qwen3.5-2B into an empty HF_HOME  <- the actual B113 condition
#   2. Snapshot completeness (no .incomplete blobs, no broken symlinks)
#   3. A real load through the same path training uses
#   4. Two control models for comparison: Qwen3-4B-Instruct-2507 (larger, 3 shards) and
#      Qwen3-1.7B (2 shards). If ALL three stall, it is the node's network, not the model.
#      If only 3.5-2B stalls, it is model-specific after all.
#
#   Control models matter because the 2026-07-30 cache inventory FALSIFIED the original
#   "high file count" theory: 3.5-2B has ONE weight shard, while 4B-Instruct has three at
#   1.8x the total size and has never stalled.
set -uo pipefail   # NOT -e: a stall is the result we are measuring, not a reason to abort

PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
mkdir -p logs/gpu_setup

source .venv_gpu/bin/activate

NV_LIBS=$(find "$PROJ/.venv_gpu/lib/python3.11/site-packages/nvidia" \
    -maxdepth 3 -name 'lib' -type d 2>/dev/null | tr '\n' ':' || true)
export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:-}"

# THE POINT OF THIS JOB: a fresh, empty cache. The shared cache is left untouched.
COLD_HOME="/tmp/b113-cold-${SLURM_JOB_ID:-manual}"
mkdir -p "$COLD_HOME"
export HF_HOME="$COLD_HOME"
export HF_HUB_DISABLE_XET=1     # same setting that did NOT help on 2026-07-10
export ANTHROPIC_API_KEY=b113-no-call
export EXA_API_KEY=b113-no-call

echo "=== node:        $(hostname)"
echo "=== cold HF_HOME: $HF_HOME  (shared cache NOT used)"
echo "=== disk free:"; df -h /tmp | tail -1
echo "=== HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"
echo

python - <<'PY'
import os, sys, time, glob, traceback
sys.path.insert(0, "/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory")

# 3.5-2B first (the suspect), then two controls. If all three fail it is the node.
MODELS = [
    ("Qwen/Qwen3.5-2B",              "SUSPECT — stalled twice on 2026-07-10"),
    ("Qwen/Qwen3-1.7B",              "control — 2 shards, 3.9 G, never stalled"),
    ("Qwen/Qwen3-4B-Instruct-2507",  "control — 3 shards, 7.6 G, never stalled"),
]

def inspect_cache(model_id: str) -> dict:
    """Completeness checks that mirror what a DownloadStallError actually reports."""
    root = os.path.join(
        os.environ["HF_HOME"], "hub", "models--" + model_id.replace("/", "--")
    )
    if not os.path.isdir(root):
        return {"cached": False}
    snaps = glob.glob(os.path.join(root, "snapshots", "*"))
    links = [
        p for s in snaps
        for p in glob.glob(os.path.join(s, "*"))
        if os.path.islink(p)
    ]
    return {
        "cached": True,
        "files": len(links),
        "broken_symlinks": sum(1 for p in links if not os.path.exists(p)),
        "incomplete_blobs": len(glob.glob(os.path.join(root, "blobs", "*.incomplete"))),
        "weight_shards": len([
            p for s in snaps for p in glob.glob(os.path.join(s, "*.safetensors"))
        ]),
    }

results = []
for model_id, note in MODELS:
    print("#" * 72)
    print(f"### {model_id}  [{note}]")
    print("#" * 72)
    record = {"model": model_id, "note": note}
    started = time.time()
    try:
        from training.quantize import resolve_hf_snapshot
        # Same entry point evaluate.py / measure_pool.py use for a remote base.
        snapshot = resolve_hf_snapshot(model_id)
        record["download_s"] = round(time.time() - started, 1)
        record["snapshot"] = snapshot
        record["outcome"] = "downloaded"
        print(f"  downloaded in {record['download_s']}s -> {snapshot}")
    except Exception as exc:
        record["download_s"] = round(time.time() - started, 1)
        record["outcome"] = "FAILED"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  !! FAILED after {record['download_s']}s: {record['error']}")
        traceback.print_exc()
        results.append(record)
        continue

    record["cache"] = inspect_cache(model_id)
    print(f"  cache: {record['cache']}")

    # A snapshot that resolves but is incomplete is exactly the B113 signature.
    if record["cache"].get("broken_symlinks") or record["cache"].get("incomplete_blobs"):
        record["outcome"] = "INCOMPLETE_SNAPSHOT"
        print("  !! snapshot resolved but is INCOMPLETE — this is the B113 signature")
        results.append(record)
        continue

    # Prove it actually loads, not just that files exist.
    try:
        load_started = time.time()
        from transformers import AutoConfig, AutoTokenizer
        AutoConfig.from_pretrained(snapshot, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=True)
        record["load_s"] = round(time.time() - load_started, 1)
        record["tokenizer_vocab"] = len(tok)
        record["outcome"] = "downloaded+loaded"
        print(f"  config+tokenizer loaded in {record['load_s']}s "
              f"(vocab={record['tokenizer_vocab']})")
    except Exception as exc:
        record["outcome"] = "LOAD_FAILED"
        record["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  !! load failed: {record['error']}")
        traceback.print_exc()
    results.append(record)

print()
print("#" * 72)
print("### B113 VERDICT")
print("#" * 72)
for record in results:
    print(f"  {record['model']:32s} {record['outcome']:22s} "
          f"download={record.get('download_s')}s load={record.get('load_s')}s")

suspect = next(r for r in results if r["model"] == "Qwen/Qwen3.5-2B")
controls = [r for r in results if r["model"] != "Qwen/Qwen3.5-2B"]
ok = lambda r: r["outcome"] == "downloaded+loaded"

print()
if ok(suspect) and all(ok(c) for c in controls):
    print("  => B113 did NOT reproduce on a genuinely cold cache. Qwen3.5-2B downloads and")
    print("     loads cleanly on this node. Consistent with a transient 2026-07-10 network")
    print("     fault that self-resolved. NOT proof it can never recur.")
elif not ok(suspect) and all(ok(c) for c in controls):
    print("  => B113 REPRODUCED and is MODEL-SPECIFIC: the suspect failed while both larger")
    print("     controls succeeded on the same node in the same job. Apply fix (c) — a bounded")
    print("     download-with-retry gated on a snapshot completeness check.")
elif not ok(suspect) and not all(ok(c) for c in controls):
    print("  => NODE/NETWORK issue, not model-specific: controls failed too. Re-run elsewhere")
    print("     before drawing any conclusion about Qwen3.5-2B.")
else:
    print("  => Mixed result; read the per-model records above.")
PY

echo
echo "=== cleaning up the cold cache ($COLD_HOME)"
du -sh "$COLD_HOME" 2>/dev/null || true
rm -rf "$COLD_HOME"
echo "=== done. The shared .hf-cache was never read or modified by this job."
