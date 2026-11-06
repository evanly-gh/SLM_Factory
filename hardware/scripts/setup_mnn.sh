#!/usr/bin/env bash
# Set up the MNN export toolchain that Model-Conversion/convert_to_mnn.py uses (macOS / Linux):
#   1. MNN source at the pinned commit, in MNN/ at the repo root (git-ignored)
#   2. MNN/build/MNNConvert, the graph converter llmexport.py is handed via --mnnconvert
#   3. .venv_mnn at the repo root, llmexport.py's own Python environment (requirements-mnn.txt)
#
#   bash scripts/setup_mnn.sh           # skips stages that are already done
#   bash scripts/setup_mnn.sh --force   # rebuilds MNNConvert and recreates .venv_mnn
#
# Needs git, cmake >= 3.22, a C++17 compiler, and Python 3.10-3.12 (override with PYTHON=...).
set -euo pipefail

MNN_COMMIT=47ccf6c6bb5b6d357cd1f9b4370cbecb6188fd34   # MNN 3.6.1, the version the reference pipeline used
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MNN_ROOT="$REPO_ROOT/MNN"
VENV="$REPO_ROOT/.venv_mnn"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

pick_python() {
  if [[ -n "${PYTHON:-}" ]]; then echo "$PYTHON"; return; fi
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then echo "$candidate"; return; fi
  done
  echo "ERROR: no python3 found; set PYTHON=/path/to/python3.12" >&2
  exit 1
}

# 1. MNN source at the pinned commit
if [[ ! -d "$MNN_ROOT/.git" ]]; then
  echo "[setup] fetching MNN @ ${MNN_COMMIT:0:12} into $MNN_ROOT"
  git init -q "$MNN_ROOT"
  git -C "$MNN_ROOT" remote add origin https://github.com/alibaba/MNN.git
fi
if [[ "$(git -C "$MNN_ROOT" rev-parse HEAD 2>/dev/null || true)" != "$MNN_COMMIT" ]]; then
  git -C "$MNN_ROOT" fetch --depth 1 origin "$MNN_COMMIT"
  git -C "$MNN_ROOT" checkout -q --detach FETCH_HEAD
  FORCE=1
fi
echo "=== MNN @ $(git -C "$MNN_ROOT" rev-parse --short=12 HEAD) ==="

# 2. MNNConvert
if [[ "$FORCE" == 1 ]]; then rm -rf "$MNN_ROOT/build"; fi
if [[ ! -x "$MNN_ROOT/build/MNNConvert" ]]; then
  echo "[setup] building MNNConvert (-j$JOBS)"
  cmake -S "$MNN_ROOT" -B "$MNN_ROOT/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_BUILD_CONVERTER=ON \
    -DMNN_BUILD_LLM=ON \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
  cmake --build "$MNN_ROOT/build" --config Release --target MNNConvert -j "$JOBS"
fi
test -x "$MNN_ROOT/build/MNNConvert" || { echo "ERROR: MNNConvert did not build" >&2; exit 1; }

# 3. .venv_mnn
if [[ "$FORCE" == 1 ]]; then rm -rf "$VENV"; fi
if ! "$VENV/bin/python" -c "import torch, onnx, onnxslim, transformers" >/dev/null 2>&1; then
  echo "[setup] creating $VENV"
  rm -rf "$VENV"
  "$(pick_python)" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r "$REPO_ROOT/requirements-mnn.txt"
fi
"$VENV/bin/python" -c "import torch, transformers, onnx; print(f'llmexport env: torch {torch.__version__}, transformers {transformers.__version__}, onnx {onnx.__version__}')"

cat <<EOF

MNN export toolchain ready.
  MNN         $MNN_ROOT
  MNNConvert  $MNN_ROOT/build/MNNConvert
  exporter    $VENV/bin/python $MNN_ROOT/transformers/llm/export/llmexport.py
EOF
