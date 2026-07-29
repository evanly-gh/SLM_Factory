#!/bin/bash
#SBATCH --job-name=slm-cuda-soak
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-cuda-soak-%j.out
set -eo pipefail

PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
source .venv_gpu/bin/activate
export SLM_CUDA_ISOLATION=1
export PYTHONUNBUFFERED=1

python - <<'PY'
import os
import subprocess

from training.cuda_isolation import run_isolated


def gpu_used_mib() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={visible}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(output.strip().splitlines()[0])


baseline = gpu_used_mib()
peak_delta = 0
for cycle in range(1, 101):
    result = run_isolated("cuda_probe", {"mib": 4096})
    child_peak = result["stats"]["peak_allocated_mib"]
    if child_peak < 4096:
        raise RuntimeError(
            f"probe did not exercise the requested allocation: peak={child_peak} MiB"
        )
    used = gpu_used_mib()
    delta = used - baseline
    peak_delta = max(peak_delta, delta)
    if cycle == 1 or cycle % 10 == 0:
        print(
            f"[soak] cycle={cycle}/100 child_pid={result['pid']} "
            f"child_peak_mib={child_peak} "
            f"post_exit_used_mib={used} delta_mib={delta}",
            flush=True,
        )
    if delta > 256:
        raise RuntimeError(
            f"post-exit GPU memory grew by {delta} MiB at cycle {cycle}; "
            "worker isolation is not returning to baseline"
        )

print(f"[soak] PASS cycles=100 baseline_mib={baseline} peak_delta_mib={peak_delta}")
PY
