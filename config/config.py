# config.py
"""
Central configuration for the SLM Factory pipeline.

MODEL SELECTION — single source of truth
----------------------------------------
Every LLM call in the pipeline resolves its model from the constants below.
Change `ORCHESTRATOR_MODEL` here and it propagates to the planner, hardware
research, iterate/escalate decisions, taxonomy clustering, sub-agent delegation,
and other orchestrator decisions. CoT generation is deliberately independent.

An env var overrides each constant so you can switch models per-run without a
code edit, e.g.:
    SLM_ORCHESTRATOR_MODEL=claude-opus-4-1 python tests/pipeline/run.py "..."

Roles
-----
- ORCHESTRATOR_MODEL : the "brain" — all planning/decision/generation calls.
- JUDGE_MODEL        : required local Qwen3.6 LLM-as-judge for `generation`
                       scoring, served by JUDGE_ENDPOINT with no cloud fallback.
- TEACHER_MODEL_*    : DeepSeek/OpenAI CoT fallback models (paper §2.3/§2.5).
                       Local Qwen3.6 is always primary.

Model options (Anthropic, as of 2026-07)
----------------------------------------
    claude-opus-4-8       most capable; slowest/most expensive — hardest reasoning
    claude-sonnet-5       strong general default (newer than sonnet-4-6)
    claude-sonnet-4-6     current pipeline default; good cost/quality balance
    claude-haiku-4-5      fast/cheap; good for high-volume orchestrator calls
    claude-fable-5        specialized; see model card before use
Full dated IDs (e.g. claude-haiku-4-5-20251001) also work. Prefer the undated
alias so you always get the latest snapshot of a tier.

CoT backends
------------
    Qwen3.6-35B (local vLLM)        : primary for every generation-family task
    deepseek-v4-flash (thinking)     : math/science CoT — needs DEEPSEEK_API_KEY
    gpt-4.1                          : code/QA CoT      — needs OPENAI_API_KEY
Cloud order is task-aware; if both keys are absent or fail, CoT is skipped.
Claude/the orchestrator is never a CoT fallback.
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

# --- Cloud fallback models for CoT annotation (paper §2.3, §2.5) ---
# Qwen3.6 is primary. DeepSeek V4 Flash/OpenAI are ordered fallbacks; Claude is never used for CoT.
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TEACHER_MODEL_DEEPSEEK = os.environ.get("SLM_TEACHER_MODEL_DEEPSEEK", "deepseek-v4-flash")
TEACHER_MODEL_GPT = os.environ.get("SLM_TEACHER_MODEL_GPT", "gpt-4.1")
# Retained only for the legacy hard-negative compatibility path in curriculum.py.
# It is not reachable from CoT generation.
TEACHER_MODEL_CLAUDE = os.environ.get("SLM_TEACHER_MODEL_CLAUDE", ORCHESTRATOR_MODEL)

# --- Cheap mode (SLM_CHEAP=1, or `python run.py --cheap`) ---
# For iterating on the pipeline without burning a big Claude bill on runs that may crash.
# It keeps the agent INTELLIGENT — the orchestrator still makes real intervention decisions
# in iterate_node through one bounded tool-free JSON call (plus at most one reask).
# Separately reserved Exa calls can still drive dataset acquisition, but never from the
# intervention decision itself. Cheap mode slashes Claude spend two ways:
#   1. Forces the cheapest Anthropic model tier (Haiku) for the orchestrator and
#      legacy Claude teacher path. The required judge remains local Qwen3.6.
#   2. Skips the two bulk pure-Claude GENERATION passes in curate_node (read via SLM_CHEAP):
#        - hard-negative synthesis (~1 LLM call per negative, per rebuild, per tier), and
#        - CoT teacher annotation.
#      Training then runs on gold-only data (still real, Exa-acquired).
# iterate_node's decision loop is deliberately NOT disabled — cheap ≠ dumb; we want the
# real EXPAND reasoning, just on Haiku. Nodes read os.environ["SLM_CHEAP"] directly so
# checking the flag needs no API-key-bearing config import.
CHEAP_MODE = os.environ.get("SLM_CHEAP", "0") == "1"
if CHEAP_MODE:
    ORCHESTRATOR_MODEL = "claude-haiku-4-5"
    TEACHER_MODEL_CLAUDE = "claude-haiku-4-5"

# --- Orchestrator context window: Sonnet 1M-token context (beta) ---
# Sonnet's 1M-token context is gated behind an Anthropic beta header. Enable per-run with
# SLM_ORCHESTRATOR_1M=1 so the orchestrator can hold a long trajectory + tool outputs without
# truncation on big runs. The beta id is env-configurable (SLM_ANTHROPIC_BETAS) so it can
# track Anthropic's header id without a code edit. Applied ONLY to the ORCHESTRATOR model's
# Anthropic clients (planner / iterate / escalate / model-selection / acquire) — not the
# local judge or synthesis server. Disabled in cheap mode (Haiku lacks the 1M window).
ORCHESTRATOR_1M = os.environ.get("SLM_ORCHESTRATOR_1M", "0") == "1"
ANTHROPIC_BETAS = [b.strip() for b in os.environ.get(
    "SLM_ANTHROPIC_BETAS", "context-1m-2025-08-07").split(",") if b.strip()]


def orchestrator_client_kwargs() -> dict:
    """Extra kwargs for the ORCHESTRATOR's Anthropic client to turn on the 1M context beta.

    Returns a ``default_headers`` dict when SLM_ORCHESTRATOR_1M=1, else empty (default context).
    Works for both the raw ``anthropic.Anthropic(**kw)`` client and langchain's
    ``ChatAnthropic(**kw)`` — both forward ``default_headers`` to the underlying HTTP client.
    """
    if ORCHESTRATOR_1M and ANTHROPIC_BETAS and not CHEAP_MODE:
        return {"default_headers": {"anthropic-beta": ",".join(ANTHROPIC_BETAS)}}
    return {}

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

# --- Data-size targets: floors, ceiling, and orchestrator override ---
# The orchestrator (task_planner) chooses the curriculum + eval example targets per run,
# biased UP for obscure/less-popular benchmarks (less pretraining exposure ⇒ more data
# needed to instill, per the fine-tuning sample-size research). Whatever it picks is clamped
# to [floor, ceiling]:
#   - CURRICULUM_SIZE_FLOOR: never train on fewer than this (small on-device models need
#     more data than 8B models — selection→instillation regime shift). Set to 3000 as the
#     per-task floor: curricula are synth-filled up to this size when real data + synthesis
#     fall short, so every task trains on ≥3000 rows regardless of DATASET_SIZE_BY_TYPE.
#   - EVAL_SET_SIZE: held-out eval floor. Larger eval sets give statistically reliable
#     metrics — F1 CIs under-cover below n≈100; per-class macro-F1 needs ≥30–50/class or a
#     3-example rare class swings it wildly (the main source of the earlier score oscillation).
#   - DATA_SIZE_CEILING: hard cap so an over-eager target can't blow the wall clock.
# DATASET_SIZE_BY_TYPE (above) is now only the FALLBACK when the planner gives no number.
CURRICULUM_SIZE_FLOOR = int(os.environ.get("SLM_CURRICULUM_FLOOR", "5000"))
EVAL_SET_SIZE = int(os.environ.get("SLM_EVAL_SET_SIZE", "800"))
DATA_SIZE_CEILING = int(os.environ.get("SLM_DATA_CEILING", "25000"))

# --- Local hard-negative / balancing synthesis model (contamination-safe, no Claude) ---
# Served by a local vLLM OpenAI-compatible endpoint (see scripts/serve_synth.slurm). The
# pipeline hits SYNTH_ENDPOINT for hard-negative + rare-class synthesis instead of the
# orchestrator API, so synthetic data is fully local and its provenance is a model we own.
# If the endpoint is unreachable, curate logs a warning and proceeds gold-only (never Claude).
SYNTH_ENDPOINT = os.environ.get("SLM_SYNTH_ENDPOINT", "")            # e.g. http://g3107:8000/v1
SYNTH_MODEL = os.environ.get("SLM_SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B")
SYNTH_API_KEY = os.environ.get("SLM_SYNTH_API_KEY", "EMPTY")

# --- Required local LLM-as-judge for open `generation` eval scoring ---
# By default the judge shares the local synthesis vLLM server. Overrides must name
# the exact model ID returned by JUDGE_ENDPOINT/models; there is no cloud fallback.
# Remote hosts (including private cluster nodes) require an explicit safety opt-in.
# The cache defaults under the stable run's artifacts directory.
JUDGE_ENDPOINT = os.environ.get("SLM_JUDGE_ENDPOINT", SYNTH_ENDPOINT)
JUDGE_MODEL = os.environ.get("SLM_JUDGE_MODEL", "Qwen/Qwen3.6-35B-A3B")
JUDGE_API_KEY = os.environ.get("SLM_JUDGE_API_KEY", SYNTH_API_KEY)
JUDGE_ALLOW_REMOTE = os.environ.get("SLM_JUDGE_ALLOW_REMOTE", "0") == "1"
JUDGE_CACHE_PATH = os.environ.get("SLM_JUDGE_CACHE_PATH", "")
JUDGE_CONCURRENCY = max(1, int(os.environ.get("SLM_JUDGE_CONCURRENCY", "16")))
JUDGE_REQUEST_TIMEOUT_S = max(
    0.1, float(os.environ.get("SLM_JUDGE_REQUEST_TIMEOUT_S", "120"))
)

# --- Local dataset fallback dir (offline copies for when Exa/LLM discovery fails) ---
# scripts/download_datasets.py writes {name}/{train,test}.jsonl here. web_acquire checks
# this BEFORE the agentic Exa path so a known task always has clean data to fall back to.
LOCAL_DATASET_DIR = os.environ.get(
    "SLM_LOCAL_DATASET_DIR",
    "/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/data/local",
)

# --- Aggregate wall-clock guard (graceful terminal state before SLURM SIGKILL) ---
# iterate_node forces a clean terminate once elapsed run time exceeds this, so a long run
# (high stall cap + downward re-exploration + big eval) always writes its full summary/DAG
# instead of being hard-killed at the --time limit. 0 disables. Non-requeue runs default
# to 14h for a 16h allocation; checkpoint/requeue weeklong scripts explicitly set 0 and
# let Slurm USR1 drive safe segment rollover.
MAX_WALLCLOCK_S = int(os.environ.get("SLM_MAX_WALLCLOCK_S", str(14 * 3600)))

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
# DEFAULT ON (B160): the deployed artifact is the quantized GGUF, so its measured accuracy
# is the number that actually matters — theoretical bf16-as-a-proxy is misleading. The GGUF
# is cached per exact weights (evaluate._build_or_reuse_gguf), so identical weights are
# never requantized; a new build per iteration is inherent because each iteration retrains.
# If the llama.cpp toolchain is missing/broken or the GGUF fails a real model-load
# validation, evaluation fails hard rather than recording BF16 or 0.0 under a quantized
# label. Set SLM_QUANT_EVAL=0 explicitly to use the old theoretical behavior. Requires:
# (1) llama.cpp `convert_hf_to_gguf` + `llama-quantize` on PATH, and (2)
# `llama-cpp-python`. Independent of SLM_HW_BACKEND / on-device eval.
QUANT_ACCURACY_EVAL = os.environ.get("SLM_QUANT_EVAL", "1") == "1"
# Nominal Li-ion voltage for mA→W power conversion when live voltage is unreadable.
HW_BATTERY_VOLTAGE_V = float(os.environ.get("SLM_HW_BATTERY_VOLTAGE_V", "3.85"))
