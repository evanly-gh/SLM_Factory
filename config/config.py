# config.py
"""
Central configuration for the SLM Factory pipeline.

MODEL SELECTION — single source of truth
----------------------------------------
Every LLM call in the pipeline resolves its model from the constants below.
Change `ORCHESTRATOR_MODEL` here and it propagates to the planner, hardware
research, iterate/escalate decisions, taxonomy clustering, sub-agent delegation,
hard-negative synthesis, NER annotation, and the Claude CoT teacher fallback.

An env var overrides each constant so you can switch models per-run without a
code edit, e.g.:
    SLM_ORCHESTRATOR_MODEL=claude-opus-4-1 python tests/pipeline/run.py "..."

Roles
-----
- ORCHESTRATOR_MODEL : the "brain" — all planning/decision/generation calls.
- JUDGE_MODEL        : LLM-as-judge for `generation` eval scoring. Cheaper is
                       fine here; it only rates 0..1. Defaults to a Haiku tier.
- TEACHER_MODEL_*    : CoT annotation teachers (paper §2.3/§2.5). Domain-routed
                       in data/curriculum.py:get_teacher_client().

Model options (Anthropic, as of 2026-07)
----------------------------------------
    claude-opus-4-8       most capable; slowest/most expensive — hardest reasoning
    claude-sonnet-5       strong general default (newer than sonnet-4-6)
    claude-sonnet-4-6     current pipeline default; good cost/quality balance
    claude-haiku-4-5      fast/cheap; good for judging and high-volume calls
    claude-fable-5        specialized; see model card before use
Full dated IDs (e.g. claude-haiku-4-5-20251001) also work. Prefer the undated
alias so you always get the latest snapshot of a tier.

Teacher options (non-Anthropic, optional — require their API keys)
-----------------------------------------------------------------
    deepseek-reasoner (DeepSeek-R1) : math/science CoT — needs DEEPSEEK_API_KEY
    gpt-4.1                          : code/QA CoT      — needs OPENAI_API_KEY
If the relevant key is unset, curriculum.py falls back to TEACHER_MODEL_CLAUDE.
"""
import os

# --- API keys ---
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
EXA_API_KEY = os.environ["EXA_API_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# --- Orchestrator: the single model that drives every planning/decision call ---
# Override per-run with SLM_ORCHESTRATOR_MODEL without editing this file.
ORCHESTRATOR_MODEL = os.environ.get("SLM_ORCHESTRATOR_MODEL", "claude-sonnet-4-6")

# --- LLM-as-judge for `generation` eval scoring (eval/scorers/generation.py) ---
# Cheaper tier is fine; it only emits a 0..1 score. Override with SLM_JUDGE_MODEL.
JUDGE_MODEL = os.environ.get("SLM_JUDGE_MODEL", "claude-haiku-4-5")

# --- Teacher models for CoT annotation (paper §2.3, §2.5) ---
# DeepSeek-R1 for math/science reasoning; GPT-4.1 for code/QA; Claude otherwise.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TEACHER_MODEL_DEEPSEEK = os.environ.get("SLM_TEACHER_MODEL_DEEPSEEK", "deepseek-reasoner")
TEACHER_MODEL_GPT = os.environ.get("SLM_TEACHER_MODEL_GPT", "gpt-4.1")
# Claude teacher fallback tracks the orchestrator model by default so a single
# ORCHESTRATOR_MODEL change moves the Claude CoT teacher with it.
TEACHER_MODEL_CLAUDE = os.environ.get("SLM_TEACHER_MODEL_CLAUDE", ORCHESTRATOR_MODEL)

# --- Model selection strategy ---
# Which approach to use for initial model selection from the feasible set.
# Override per-run with SLM_MODEL_SELECTION_STRATEGY.
#   "smallest_first"      — start smallest, escalate on failure (no probing)
#   "largest_first"       — probe largest for feasibility, then start smallest
#   "interpolation"       — 3-probe scaling curve, pick closest to RAM target
#   "orchestrator_choice" — LLM picks based on task context (no probing)
MODEL_SELECTION_STRATEGY = os.environ.get(
    "SLM_MODEL_SELECTION_STRATEGY", "smallest_first"
)

# --- Target dataset size per task type (single source of truth for curate + eval_setup) ---
# Total examples the curriculum aims for (gold = 65%, hard = 35%). Grounded in the paper's
# §4.3 quality-over-quantity guidance (classification/NER 100–200; generation 500–3,000) and
# its findings that 173 > 348 (HumanEval) and 500 selected > 2,000 random (SAMSum) — i.e.
# these are UPPER bounds on *curated-quality* data, not counts to fill with noise. Adjusted
# 2026-07-15 down from a flat 1000 for the generation-family (B155):
#   NER 300→200, math_reasoning 1000→700, code_generation 1000→300, generation 1000→600.
DATASET_SIZE_BY_TYPE = {
    "classification":             150,   # paper: 100–200
    "multi_label_classification": 300,   # label co-occurrence needs coverage
    "NER":                        200,   # paper: 100–200 (entity diversity handled by controls)
    "structured_extraction":      400,   # schema field coverage + negatives
    "math_reasoning":             700,   # verified verbose CoT quality dominates raw count
    "code_generation":            300,   # paper: 173 curated > 348 on HumanEval
    "multilingual":               400,   # language-pair coverage
    "generation":                 600,   # paper: 500 agent-selected > 2,000 random
}

MAX_TURNS_MAIN = 1500

DEFAULT_STOP_THRESHOLD = 0.96

ARTIFACTS_DIR = "artifacts"

# --- Fixed hardware-gate constants ---
# RAM, storage, AND reference chip are resolved per-device by hardware_research (they
# vary with the phone). The gates below are fixed system/UX constants, NOT device-specific,
# so they live here rather than being guessed by the LLM each run.
#   HW_LATENCY_TTFT_MS — interactive time-to-first-token ceiling.
#   HW_POWER_WATTS     — sustained-inference power budget.
#   HW_MIN_TOK_S       — decode-throughput UX floor (0 disables; 6 ≈ reading speed).
#   HW_FALLBACK_CHIP   — throughput-scaling anchor used ONLY when the device chip
#                        cannot be resolved from the user's description.
HW_LATENCY_TTFT_MS = 2000
HW_POWER_WATTS = 5.0
HW_MIN_TOK_S = 0.0
HW_FALLBACK_CHIP = "snapdragon_778g"

# --- On-device hardware evaluation (hardware_eval/on_device_eval.py) ---
# HW_ONDEVICE_BACKEND — how metrics are gathered:
#     "theoretical" (default) : ModelSpec estimates, no hardware. Safe on a GPU node.
#     "llama_cpp"             : local llama-cli timing on the built GGUF (no phone).
#     "adb_llama"             : llama-cli on a connected device via ADB.
#     "smolchat"              : broadcast to the SmolChat app + logcat scrape (richest).
# HW_GATING_ENABLED — when True, latency/power/memory become HARD gates in iterate_node
#     (a converged model that violates hardware is not accepted as terminal).
# HW_VERIFY_ON_DEVICE — when True, run.py runs a real on-device measurement pass after
#     convergence and writes hardware_eval.json. Requires a non-theoretical backend
#     and a connected device; on failure it logs and continues (never crashes the run).
HW_ONDEVICE_BACKEND = os.environ.get("SLM_HW_BACKEND", "theoretical")
HW_GATING_ENABLED = os.environ.get("SLM_HW_GATING", "0") == "1"
HW_VERIFY_ON_DEVICE = os.environ.get("SLM_HW_VERIFY_ON_DEVICE", "0") == "1"

# QUANT_ACCURACY_EVAL — score the ACTUAL quantized GGUF (Q4_K_M / Q8_0) for honest
# per-quant ACCURACY, WITHOUT any on-device (latency/power) measurement. When True,
# evaluate_node merges the LoRA adapter, quantizes to the variant's GGUF via llama.cpp,
# and scores it on-CPU through llama-cpp-python — so Q4_K_M and Q8_0 of a model finally
# produce DIFFERENT accuracy numbers (in theoretical mode they were identical, B156).
# Requires: (1) llama.cpp `convert_hf_to_gguf` + `llama-quantize` on PATH, and
# (2) `pip install llama-cpp-python`. Independent of SLM_HW_BACKEND / on-device eval.
QUANT_ACCURACY_EVAL = os.environ.get("SLM_QUANT_EVAL", "0") == "1"
# Nominal Li-ion voltage for mA→W power conversion when live voltage is unreadable.
HW_BATTERY_VOLTAGE_V = float(os.environ.get("SLM_HW_BATTERY_VOLTAGE_V", "3.85"))
