#!/bin/bash
# Build the two SEPARATE venvs the report metrics need, kept out of .venv_gpu on purpose.
#
# WHY NOT JUST PIP INSTALL THEM
#   `.venv_gpu` is the training stack: unsloth, transformers 5.5.0, torch, numpy 2.4.6. Both of
#   these metric packages want to renegotiate parts of that.
#
#     errant 3.0.2  requires spacy<4, which drags in thinc/pydantic/numpy constraints of its own.
#     bert-score    0.3.13, last released well before transformers 5.x, and it imports
#                   AutoModel/AutoTokenizer directly.
#
#   A resolver that downgrades numpy or pydantic underneath Unsloth does not fail here; it fails
#   twenty minutes into a training run, which is the same shape as the nvcc failure that killed
#   run 39361189 after the teacher measurement had already been paid for. The .venv_vllm precedent
#   is exactly this: an environment that cannot coexist in one interpreter talks over a boundary
#   instead. ERRANT's boundary is text files, BERTScore's is a subprocess with JSON on stdout.
#
# BOTH ARE CPU-ONLY AND OFF THE HOT PATH. Neither is imported by the agent loop: ERRANT scores
# gec_bea19 and BERTScore is a dialogsum REPORT metric, so both run at most once per eval pass and
# BERTScore only from scripts/report_eval.py.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/mmfs1/gscratch/intelligentsystems/evanly/.uv-cache}"

cd /mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory

# THE SPACY MODEL IS PINNED TO AN EXACT WHEEL, not `python -m spacy download en_core_web_sm`.
# ERRANT's edit extraction aligns source against hypothesis using en_core_web_sm's POS tags and
# lemmas, and its 25 error-type labels are assigned by rules over those tags. A model version bump
# therefore moves F0.5 with no change to the model being evaluated — a silent, unattributable
# score shift across runs. `spacy download` resolves "whatever is current", which is precisely the
# thing that must not happen. The version is recorded in the score output as well.
SPACY_MODEL_VERSION="3.8.0"
SPACY_MODEL_WHEEL="https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-${SPACY_MODEL_VERSION}/en_core_web_sm-${SPACY_MODEL_VERSION}-py3-none-any.whl"

# --- ERRANT: the GEC scorer -------------------------------------------------------------
if ! .venv_errant/bin/python -c 'import errant' >/dev/null 2>&1; then
  echo "[setup] creating .venv_errant"
  rm -rf .venv_errant
  uv venv --python 3.11 .venv_errant
  # No --torch-backend and no torch at all: ERRANT is pure CPU string alignment.
  uv pip install --python .venv_errant/bin/python "errant==3.0.2" "spacy<4"
  uv pip install --python .venv_errant/bin/python "$SPACY_MODEL_WHEEL"
fi
.venv_errant/bin/python - <<'PY'
import errant, spacy, en_core_web_sm
# Load through the same entry point the CLI uses, so a broken model fails at setup rather than
# during an eval pass where it would look like a scoring bug.
annotator = errant.load("en")
orig = annotator.parse("I has went to the store yesterday")
cor = annotator.parse("I went to the store yesterday")
# `annotate(orig, cor)` — ERRANT 3.x aligns internally. In 2.x this was
# `annotate(align(orig, cor))`, which is a TypeError here rather than a wrong answer, but the
# scorer talks to the CLI and not to this API, so the pin in the install above is what matters.
edits = annotator.annotate(orig, cor)
print(f"errant {errant.__version__} spacy {spacy.__version__} "
      f"en_core_web_sm {en_core_web_sm.__version__}")
print(f"  smoke: {len(edits)} edit(s), types {[e.type for e in edits]}")
assert edits, "errant found no edits in a sentence that plainly has one"
PY
for tool in errant_parallel errant_compare; do
  test -x ".venv_errant/bin/$tool" || { echo "ERROR: .venv_errant/bin/$tool missing" >&2; exit 1; }
done
echo "ERRANT env ready: .venv_errant  (export ERRANT_VENV=\$PWD/.venv_errant)"

# --- BERTScore: the dialogsum secondary report metric -----------------------------------
# Its own venv rather than .venv_errant: spacy<4 and an old transformers pin have no reason to
# share a resolver, and a conflict between two things we do not control would be untraceable.
if ! .venv_metrics/bin/python -c 'import bert_score' >/dev/null 2>&1; then
  echo "[setup] creating .venv_metrics"
  rm -rf .venv_metrics
  uv venv --python 3.11 .venv_metrics
  # CPU torch deliberately. BERTScore runs once per report pass over 500 DialogSum summaries;
  # buying a CUDA wheel for that would reserve GPU memory the training stack wants.
  uv pip install --python .venv_metrics/bin/python torch --torch-backend=cpu
  uv pip install --python .venv_metrics/bin/python "bert-score>=0.3.13"
fi
.venv_metrics/bin/python - <<'PY'
import bert_score, transformers
print(f"bert-score {bert_score.__version__} transformers {transformers.__version__}")
PY
echo "BERTScore env ready: .venv_metrics"
