# Sourced (never executed) by every Slurm launcher and setup script.
#
# Resolves SLM_ROOT, loads site.env, and fills in the defaults documented in site.env.example.
# Precedence: a variable already in the environment > site.env > the default here.
# config/site.py does the same for Python entry points; keep the two in step.

if [ -z "${SLM_ROOT:-}" ]; then
  SLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export SLM_ROOT

if [ -f "$SLM_ROOT/site.env" ]; then
  while IFS= read -r _slm_line || [ -n "$_slm_line" ]; do
    case "$_slm_line" in ''|'#'*) continue ;; esac
    _slm_key="${_slm_line%%=*}"
    _slm_val="${_slm_line#*=}"
    _slm_key="${_slm_key//[[:space:]]/}"
    _slm_val="${_slm_val%\"}"; _slm_val="${_slm_val#\"}"
    _slm_val="${_slm_val%\'}"; _slm_val="${_slm_val#\'}"
    [ -n "$_slm_key" ] || continue
    if [ -z "${!_slm_key+x}" ]; then
      export "$_slm_key=$_slm_val"
    fi
  done < "$SLM_ROOT/site.env"
  unset _slm_line _slm_key _slm_val
fi

export SLM_CACHE_DIR="${SLM_CACHE_DIR:-$SLM_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SLM_CACHE_DIR/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SLM_CACHE_DIR/pip}"
export HF_HOME="${HF_HOME:-$SLM_CACHE_DIR/huggingface}"
export SLM_TMP_DIR="${SLM_TMP_DIR:-$SLM_ROOT/.tmp}"
export LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$(dirname "$SLM_ROOT")/llama.cpp}"
