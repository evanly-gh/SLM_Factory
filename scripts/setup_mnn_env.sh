#!/bin/bash
# Build the MNN quantization backend: the MNNConvert/llm_demo binaries, the exporter's venv, and
# pymnn with the LLM API. This is to `SLM_QUANT_BACKEND=mnn` what the local llama.cpp checkout
# plus `scripts/build_llamacpp_cuda.sh` are to the default GGUF path.
#
# Idempotent and safe to re-run: each of the four stages is skipped when its product is already
# present and working. Pass --force to rebuild everything from scratch.
#
#   bash scripts/setup_mnn_env.sh
#   bash scripts/setup_mnn_env.sh --force
#
# RUNS ON A LOGIN NODE, and needs the CUDA TOOLKIT but not a GPU — nvcc compiles the sm_89 kernels
# without a device present. The exporter traces the model on CPU, which is why `.venv_mnn` gets a
# CPU torch wheel: buying a CUDA one would reserve GPU memory the training stack wants, the same
# reasoning `scripts/setup_metric_envs.sh` gives for `.venv_metrics`.
#
# WHY CUDA AT ALL, when the phone runs MNN on its CPU: eval speed, and it is the same argument the
# GGUF path makes for `n_gpu_layers=-1`. A quantized artifact's accuracy is a property of its
# weights, tokenizer and greedy decoding, not of the device that multiplies the matrices. Measured
# on SmolLM2-360M @ 4-bit over clinc150 rows: prefill 671 tok/s on CPU against 6,867 on an L40S,
# and since a CLINC150 prompt is ~1,385 tokens against a one-label answer, that is ~6x end to end.
#
# WHAT EACH STAGE PRODUCES, AND WHY IT IS SEPARATE
#   1. MNN/build/MNNConvert   — the graph converter `llmexport.py` must be handed via
#                               `--mnnconvert`. Without it llmexport falls back to the pymnn
#                               bindings, which the reference harness recorded crashing with a
#                               bus error, so this binary is mandatory rather than an optimisation.
#      MNN/build/llm_demo     — a standalone CLI that loads an exported artifact. Not on the eval
#                               path (pymnn is), but it is the one tool that can answer "is the
#                               artifact broken or is my Python wrong?" in isolation.
#   2. .venv_mnn              — the exporter's own interpreter: torch + onnx + onnxslim +
#                               transformers. Kept OUT of .venv_gpu because a resolver that moves
#                               numpy or transformers underneath Unsloth does not fail at install
#                               time, it fails twenty minutes into a training run.
#   3. pymnn_build            — a SECOND MNN build, SHARED and with CUDA, which is what pymnn's
#                               setup.py links the extension against.
#
#      TWO NON-OBVIOUS FLAGS, BOTH LOAD-BEARING:
#      * -DMNN_BUILD_SHARED_LIBS=ON. MNN's pip_package/build_deps.py builds STATIC on Linux, and
#        with a static libMNN.a the CUDA backend DOES NOT REGISTER: it registers through a
#        file-scope initializer in `source/backend/cuda/Register.cpp`, which is compiled into
#        libMNN as an OBJECT library, and a static archive contributes only the objects needed to
#        resolve referenced symbols — nothing references that one, so the linker drops it. The
#        symptom is not an error. MNN prints `Can't Find type=2 backend, use 0 instead`, runs on
#        the CPU at CPU speed, and — having already configured itself for a GPU — decodes
#        `'accept<|endoftext|><|endoftext|>...'` instead of `'accept_reservations'`. Four
#        measurements were taken that way before the cause was found; `slm_helpers.
#        _assert_backend_honoured` now fails the load rather than letting it happen again.
#      * -DCUDA_ARCHS=8.9 with -DMNN_CUDA_NATIVE_ARCH=ON. Emits ONLY sm_89 (Ada: L40S and L40, the
#        cards this project's launchers pin with `--gres=gpu:l40s`). Without it MNN compiles its
#        whole gencode list, which is many times the build time for arches nothing here runs.
#   4. MNN in .venv_gpu       — the extension, installed into the PIPELINE's venv, because that is
#                               the interpreter the eval CUDA worker runs in. Built with the LLM
#                               API (setup.py adds -DPYMNN_LLM_API when it sees MNN_BUILD_LLM=ON
#                               in the build's CMakeCache), without which `MNN.llm` imports as
#                               None and no artifact can be loaded. Because the link is now
#                               dynamic, `libMNN.so` and `libMNN_Cuda_Main.so` are staged into
#                               `.venv_gpu/lib` and that directory is added to the venv's
#                               LD_LIBRARY_PATH — the same mechanism already used to give
#                               llama-cpp-python its gcc-12 libstdc++.
#
# NOTE: no `-u` (nounset) — the Lmod bash init references $LD_LIBRARY_PATH, which is unbound in a
# fresh SLURM shell and would abort the script before any build. Pre-init it and use -eo.
set -eo pipefail
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
PROJ=/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory
cd "$PROJ"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/.pip-cache}"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# Sibling of the project directory, matching where the llama.cpp checkout this pipeline already
# depends on lives. `training/quantize_mnn.py::default_mnn_root` derives the same path, so the two
# agree without either hardcoding the other's answer.
MNN_ROOT="${SLM_MNN_ROOT:-$(dirname "$PROJ")/MNN}"
BUILD_JOBS="${SLM_MNN_BUILD_JOBS:-32}"

# IMPORTANT: do NOT pipe `module load` — `module` is a shell function, and piping runs it in a
# SUBSHELL so its PATH changes are lost. Load cmake BEFORE gcc: the cmake module depends on
# gcc/11.2.0 and loading it second silently downgrades the compiler ("reloaded with a version
# change"), which is how you end up compiling the extension against a libstdc++ that .venv_gpu's
# LD_LIBRARY_PATH does not point at.
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cmake/3.25.1
module load gcc/12.3.0
CUDA_MOD=$(module avail cuda 2>&1 | grep -oE "cuda/12\.8[0-9.]*" | sort -V | tail -1)
module load "${CUDA_MOD:-cuda/12.8.1}"
export CUDACXX="$(command -v nvcc)"
export CUDA_HOME="$(dirname "$(dirname "$CUDACXX")")"
if [ -z "$CUDACXX" ]; then
  echo "ERROR: nvcc not found after loading ${CUDA_MOD:-cuda/12.8.1}; the CUDA backend cannot be built" >&2
  exit 2
fi
echo "=== toolchain: $(gcc --version | head -1), $(cmake --version | head -1), nvcc=$CUDACXX ==="
# Ada only (L40S/L40), which is what every GPU launcher in this repo pins.
MNN_CUDA_ARCH="${SLM_MNN_CUDA_ARCH:-8.9}"

# ── 1. MNN source + the converter and llm_demo binaries ─────────────────────────────────────────
if [ ! -d "$MNN_ROOT/.git" ]; then
  echo "[setup] cloning MNN → $MNN_ROOT"
  git clone --depth 1 https://github.com/alibaba/MNN.git "$MNN_ROOT"
fi
MNN_COMMIT=$(git -C "$MNN_ROOT" rev-parse --short HEAD)
MNN_VERSION=$(sed -n 's/^#define MNN_VERSION_\(MAJOR\|MINOR\|PATCH\) *//p' \
  "$MNN_ROOT/include/MNN/MNNDefine.h" | paste -sd. -)
echo "=== MNN $MNN_VERSION @ $MNN_COMMIT ($MNN_ROOT) ==="

if [ "$FORCE" = "1" ]; then
  rm -rf "$MNN_ROOT/build" "$MNN_ROOT/pymnn_build"
fi
if [ ! -x "$MNN_ROOT/build/MNNConvert" ]; then
  echo "[setup] building MNNConvert + llm_demo (-j$BUILD_JOBS)"
  mkdir -p "$MNN_ROOT/build"
  # MNN_BUILD_LLM pulls in the transformer LLM engine (and implies MNN_LOW_MEMORY +
  # MNN_SUPPORT_TRANSFORMER_FUSE, which are what make a 4-bit model decode without dequantizing
  # the whole weight tensor first). MNN_AVX512 compiles the AVX-512 kernels; MNN dispatches on
  # cpuid at runtime, so the binary still runs on nodes without them.
  ( cd "$MNN_ROOT/build" && cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DMNN_BUILD_CONVERTER=ON \
      -DMNN_BUILD_TOOLS=ON \
      -DMNN_BUILD_LLM=ON \
      -DMNN_LOW_MEMORY=ON \
      -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
      -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
      -DMNN_BUILD_SHARED_LIBS=ON \
      -DMNN_SEP_BUILD=OFF \
      -DMNN_AVX512=ON \
    && make -j"$BUILD_JOBS" )
fi
test -x "$MNN_ROOT/build/MNNConvert" || { echo "ERROR: MNNConvert did not build" >&2; exit 1; }
echo "=== MNNConvert: $("$MNN_ROOT/build/MNNConvert" --version 2>&1 | tail -1) ==="

# ── 2. The exporter's venv ──────────────────────────────────────────────────────────────────────
if [ "$FORCE" = "1" ]; then
  rm -rf "$PROJ/.venv_mnn"
fi
if ! .venv_mnn/bin/python -c 'import torch, onnx, onnxslim, transformers' >/dev/null 2>&1; then
  echo "[setup] creating .venv_mnn (llmexport dependencies)"
  rm -rf .venv_mnn
  uv venv --python 3.11 .venv_mnn
  uv pip install --python .venv_mnn/bin/python torch --torch-backend=cpu
  # MNN/transformers/llm/export/requirements.txt, minus the `MNN` pip wheel: the wheel exists to
  # provide a pymnn fallback converter, and we deliberately pass a locally built --mnnconvert
  # instead (the wheel's bindings are the bus-error path). numpy is pinned below 3 because
  # onnxruntime's published wheels are not built against it yet.
  uv pip install --python .venv_mnn/bin/python \
    transformers peft onnx onnxslim onnxruntime sentencepiece "numpy<3" tqdm yaspin Pillow \
    requests datasets
fi
.venv_mnn/bin/python - <<'PY'
import onnx, onnxslim, torch, transformers
print(f"llmexport env: torch {torch.__version__} transformers {transformers.__version__} "
      f"onnx {onnx.__version__} onnxslim {getattr(onnxslim, '__version__', 'n/a')}")
PY

# ── 3+4. pymnn with the LLM API and CUDA, installed into the pipeline venv ───────────────────────
# NOT `build_deps.py llm cuda`: that helper hardcodes -DMNN_BUILD_SHARED_LIBS=OFF on Linux, and a
# static libMNN.a silently drops the CUDA backend registrar (see the header). The cmake invocation
# below is build_deps' own Linux command with SHARED libraries and a pinned CUDA arch.
_pymnn_has_cuda() {
  # Both halves matter: the LLM API must be present AND the CUDA backend must actually register.
  # The second is invisible to Python, so it is read off the extension's dynamic dependencies.
  .venv_gpu/bin/python -c 'import MNN; raise SystemExit(0 if MNN.llm is not None else 1)' \
    >/dev/null 2>&1 &&
  ldd .venv_gpu/lib/python3.11/site-packages/_mnncengine*.so 2>/dev/null | grep -q libMNN_Cuda_Main
}
if [ "$FORCE" = "1" ] || ! _pymnn_has_cuda; then
  echo "[setup] building pymnn (shared + CUDA sm_$MNN_CUDA_ARCH + LLM API) into .venv_gpu"
  rm -rf "$MNN_ROOT/pymnn_build"
  mkdir -p "$MNN_ROOT/pymnn_build"
  ( cd "$MNN_ROOT/pymnn_build" && cmake \
      -DMNN_LOW_MEMORY=ON -DMNN_BUILD_LLM=ON -DMNN_BUILD_LLM_OMNI=ON \
      -DMNN_USE_THREAD_POOL=ON -DMNN_OPENMP=OFF \
      -DMNN_CUDA=ON -DMNN_CUDA_NATIVE_ARCH=ON -DCUDA_ARCHS="$MNN_CUDA_ARCH" \
      -DMNN_BUILD_CONVERTER=on -DMNN_BUILD_TRAIN=ON -DCMAKE_BUILD_TYPE=Release \
      -DMNN_BUILD_SHARED_LIBS=ON -DMNN_AAPL_FMWK=OFF -DMNN_SEP_BUILD=OFF \
      -DMNN_BUILD_OPENCV=ON -DMNN_IMGCODECS=ON -DMNN_BUILD_AUDIO=ON -DMNN_AVX512=ON \
      "$MNN_ROOT" \
    && make MNN MNNTrain MNNConvertDeps -j"$BUILD_JOBS" )

  # setuptools will NOT recompile an extension whose build tree looks current, and it will not
  # overwrite an installed .so it believes it already installed. Both happened here: two "CUDA
  # installs" in a row reported success while leaving the CPU-only extension in place, which is
  # how the first four GPU measurements came to be taken on the CPU.
  rm -rf "$MNN_ROOT/pymnn/pip_package/build" "$MNN_ROOT/pymnn/pip_package"/*.egg-info
  rm -f .venv_gpu/lib/python3.11/site-packages/_mnncengine*.so
  ( cd "$MNN_ROOT/pymnn/pip_package" \
    && "$PROJ/.venv_gpu/bin/python" setup.py install --version "$MNN_VERSION" --deps 'llm cuda' )

  # The extension is dynamically linked now, so its libraries have to be findable. Staged into the
  # venv and put on LD_LIBRARY_PATH by `activate`, alongside the gcc-12 libstdc++ entry that
  # llama-cpp-python already depends on.
  mkdir -p .venv_gpu/lib
  cp "$MNN_ROOT/pymnn_build/libMNN.so" .venv_gpu/lib/
  cp "$MNN_ROOT/pymnn_build/source/backend/cuda/libMNN_Cuda_Main.so" .venv_gpu/lib/
  [ -f "$MNN_ROOT/pymnn_build/tools/converter/libMNNConvertDeps.so" ] &&
    cp "$MNN_ROOT/pymnn_build/tools/converter/libMNNConvertDeps.so" .venv_gpu/lib/
  if ! grep -q 'SLM_Factory pymnn shared libs' .venv_gpu/bin/activate; then
    printf '%s\n' \
      'export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"  # SLM_Factory pymnn shared libs (libMNN.so + libMNN_Cuda_Main.so; the CUDA backend registrar only survives a SHARED link)' \
      >> .venv_gpu/bin/activate
    echo "[setup] added \$VIRTUAL_ENV/lib to .venv_gpu/bin/activate's LD_LIBRARY_PATH"
  fi
fi

echo "=== verify: pymnn LLM API + CUDA backend in .venv_gpu ==="
# shellcheck disable=SC1091
source .venv_gpu/bin/activate
.venv_gpu/bin/python - <<'PY'
import MNN
assert MNN.llm is not None, (
    "pymnn imported but MNN.llm is None — it was built WITHOUT -DPYMNN_LLM_API, so it cannot "
    "load an MNN LLM artifact. Re-run this script with --force."
)
print(f"pymnn with LLM API: MNN.llm = {MNN.llm}")
print("create/load entry point:", MNN.llm.create)
PY
if ldd .venv_gpu/lib/python3.11/site-packages/_mnncengine*.so 2>/dev/null | grep -q "libMNN_Cuda_Main.*=> /"; then
  echo "CUDA backend linked and resolvable."
else
  echo "ERROR: the extension does not resolve libMNN_Cuda_Main.so, so MNN will fall back to the CPU" >&2
  ldd .venv_gpu/lib/python3.11/site-packages/_mnncengine*.so 2>/dev/null | grep -i "mnn\|not found" >&2
  exit 1
fi
echo "NOTE: whether CUDA actually REGISTERS can only be checked on a GPU node. Run:"
echo "  srun -A gpu-l40s-intelligentsystems -p gpu-l40s --gres=gpu:l40s:1 --time=00:20:00 \\"
echo "    bash -c 'source .venv_gpu/bin/activate && python hardware_eval/mnn_backend_matrix.py \\"
echo "             --models HuggingFaceTB/SmolLM2-360M-Instruct --precisions Q4_K_M --rows 25 --discard'"

cat <<EOF

MNN backend ready.
  MNN_ROOT      $MNN_ROOT  (MNN $MNN_VERSION @ $MNN_COMMIT)
  MNNConvert    $MNN_ROOT/build/MNNConvert
  llm_demo      $MNN_ROOT/build/llm_demo
  exporter      $PROJ/.venv_mnn/bin/python $MNN_ROOT/transformers/llm/export/llmexport.py
  runtime       pymnn (MNN.llm) in $PROJ/.venv_gpu, CUDA sm_$MNN_CUDA_ARCH + CPU
  device        SLM_MNN_BACKEND_TYPE=auto (CUDA when a GPU is visible, else CPU)

Use it with:
  python tests/pipeline/run.py --quant-backend mnn "<task description>"
  SLM_QUANT_BACKEND=mnn python hardware_eval/quant_accuracy_eval.py ...
  python hardware_eval/mnn_backend_matrix.py --rows 100 --discard   # every tier, every precision
EOF
