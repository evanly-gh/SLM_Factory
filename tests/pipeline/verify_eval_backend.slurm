#!/bin/bash
#SBATCH --job-name=slm-verify-eval-backend
#SBATCH --account=gpu-l40s-intelligentsystems
#SBATCH --partition=gpu-l40s
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/logs/slurm/slm-verify-eval-backend-%j.out
#
# Does `SLM_EVAL_BACKEND=vllm` score the same as the in-process path, and how much faster?
#
# One GPU, because the point is a like-for-like comparison of two engines on the SAME card — the
# whole eval fits on one, and giving it two would only hide which card the served engine landed on.
#
# Read the VERDICT line at the end of each task block. Speed is the reason to want this; agreement on
# the metric is the reason it is allowed. See eval/student_server.py.

set -uo pipefail
cd /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory

echo "=== node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Which flags this vLLM build actually accepts. Checked on a GPU node because vLLM 0.25 cannot even
# construct its argument parser without a device, so this cannot be verified from a login node.
echo
echo "=== vLLM flag support ==="
.venv_vllm/bin/python -m vllm.entrypoints.openai.api_server --help 2>/dev/null \
    | grep -oE -- "--(disable-log-requests|max-lora-rank|enable-lora|lora-modules|served-model-name|max-model-len|gpu-memory-utilization)" \
    | sort -u

source .venv_gpu/bin/activate
export SLM_STUDENT_GPU_IDS=0
export SLM_STUDENT_GPU_UTILIZATION=0.60
export SLM_LOG_DIR=logs/slurm
export HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache

# Short outputs first: the cheapest possible parity check. If two engines disagree here they will
# disagree everywhere, and this costs two minutes rather than twenty.
for spec in "sms_spam:HuggingFaceTB/SmolLM2-135M:60" "calendar_json:HuggingFaceTB/SmolLM2-135M:40"; do
    task="${spec%%:*}"; rest="${spec#*:}"; model="${rest%%:*}"; rows="${rest##*:}"
    echo
    echo "############ $task ($model, $rows rows) ############"
    python scripts/compare_eval_backends.py --task "$task" --base-model "$model" --rows "$rows"
    echo "############ $task exit=$? ############"
done
