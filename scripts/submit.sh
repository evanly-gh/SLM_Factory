#!/bin/bash
# Submit a Slurm launcher with this site's settings (site.env).
#
#   scripts/submit.sh tests/pipeline/run_clinc150_l40s.slurm
#   scripts/submit.sh -A other-account -p other-partition tests/pipeline/run_x.slurm
#
# Loads site.env (SBATCH_ACCOUNT / SBATCH_PARTITION / SBATCH_QOS, caches, scratch), then runs
# sbatch from the repo root: the launchers' `--output=logs/slurm/...` paths and their
# `$SLURM_SUBMIT_DIR` lookup of the repo both depend on that. Options given here override
# site.env, because sbatch ranks command-line options above SBATCH_* variables.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/site_env.sh"
cd "$SLM_ROOT"
mkdir -p logs/slurm
exec sbatch "$@"
