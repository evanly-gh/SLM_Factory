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
- CoT teacher        : the local Qwen3.6 synth model, and ONLY that model — there is
                       no cloud CoT fallback (paper §2.3/§2.5).

Model options (Anthropic, as of 2026-08)
----------------------------------------
    claude-opus-4-8       most capable; slowest/most expensive — hardest reasoning
    claude-sonnet-5       CURRENT PIPELINE DEFAULT — $2/$10 per MTok introductory through
                          2026-08-31, then $3/$15 (i.e. same as sonnet-4-6 afterwards).
                          1M context at standard pricing, 128k max output.
    claude-sonnet-4-6     previous default; $3/$15, functionally superseded by sonnet-5
    claude-haiku-4-5      fast/cheap; good for high-volume orchestrator calls
    claude-fable-5        specialized; see model card before use
From the 4.6 generation onward the dateless id IS the pinned snapshot (not a moving
alias), so `claude-sonnet-5` always refers to one fixed model.

NOTE on the 1M context window: since 2026-03-13 the full 1M window is generally
available on 4.6-and-later at STANDARD pricing — there is no long-context surcharge and
the `context-1m-2025-08-07` beta header is ignored. SLM_ORCHESTRATOR_1M is therefore
vestigial for these models; it is kept only for older ids.

CoT backend
-----------
    Qwen3.6-35B (local vLLM) : the sole CoT teacher for every generation-family task.
If the local synth endpoint is unavailable, CoT annotation is skipped (non-fatal). There is
no cloud CoT fallback, and Claude/the orchestrator is never a CoT teacher.
"""
import os

# --- API keys ---
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
EXA_API_KEY = os.environ["EXA_API_KEY"]

# --- Orchestrator: the single model that drives every planning/decision call ---
# Override per-run with SLM_ORCHESTRATOR_MODEL without editing this file.
ORCHESTRATOR_MODEL = os.environ.get("SLM_ORCHESTRATOR_MODEL", "claude-sonnet-5")

# --- Legacy Claude teacher tier: no live consumer ---
# NOT reachable from CoT generation or curriculum synthesis, both of which are Qwen3.6-only.
# Retained solely because `agent/checkpoint.py` snapshots this name in the run config, so
# deleting it would change the checkpoint schema.
TEACHER_MODEL_CLAUDE = os.environ.get("SLM_TEACHER_MODEL_CLAUDE", ORCHESTRATOR_MODEL)

# --- Cheap mode (SLM_CHEAP=1, or `python run.py --cheap`) ---
# For iterating on the pipeline without burning a big Claude bill on runs that may crash.
# It keeps the agent INTELLIGENT — the orchestrator still makes real intervention decisions
# in iterate_node through one bounded tool-free JSON call (plus at most one reask).
# Separately reserved Exa calls can still drive dataset acquisition, but never from the
# intervention decision itself. Cheap mode slashes Claude spend two ways:
#   1. Forces the cheapest Anthropic model tier (Haiku) for the orchestrator and
#      legacy Claude teacher path. The required judge remains local Qwen3.6.
#   2. Skips the two bulk GENERATION passes in curate_node (read via SLM_CHEAP):
#        - curriculum synthesis (~1 call per generated row, per rebuild, per tier), and
#        - CoT annotation.
#      Training then runs on real acquired data only, with no synthetic top-up.
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
#   "single_model"        — NAIVE BASELINE: orchestrator picks one model and the run stays on it.
#                           No escalation on failure, no downward regression on success.
MODEL_SELECTION_STRATEGY = os.environ.get(
    "SLM_MODEL_SELECTION_STRATEGY", "smallest_first"
)

# Strategies that pin the run to ONE model for its whole life. This is the ablation control for
# "what does the model ladder actually buy?": everything else in the loop (hyperparameter search,
# data rebuilds, rollback, the accuracy goal) still runs, but the model never changes. It is
# deliberately close to the naive human workflow — pick a model, pick data, tune, train — so the
# ladder's contribution is the difference between this and the other strategies.
_SINGLE_MODEL_STRATEGIES = frozenset({"single_model"})


def model_ladder_enabled(strategy: str | None = None) -> bool:
    """False when the run is pinned to one model, so escalation and regression are both off.

    Read through this rather than comparing strategy strings at each gate — there are three of
    them (stagnation escalation, eval-cap escalation, post-convergence downward probe) and a
    missed one silently reintroduces the ladder.
    """
    name = strategy if strategy is not None else os.environ.get(
        "SLM_MODEL_SELECTION_STRATEGY", MODEL_SELECTION_STRATEGY
    )
    return str(name) not in _SINGLE_MODEL_STRATEGIES

# --- Data-size targets: floor, ceiling, and per-tier sizing ---
# The per-task `DATASET_SIZE_BY_TYPE` table was REMOVED on 2026-08-05. Every value in it
# (classification 150, NER 200, generation 600, …) sat far below CURRICULUM_SIZE_FLOOR, so it was
# clamped away on every code path — it only created the impression that per-task sizes were being
# honoured. Sizing is now computed per model tier by `agent/data_sizing.py` from two measured
# signals: task novelty (1 − zero-shot baseline) and model capacity (inverse parameter count),
# then clamped to [CURRICULUM_SIZE_FLOOR, DATA_SIZE_CEILING].
#   - CURRICULUM_SIZE_FLOOR: never train on fewer than this (small on-device models need
#     more data than 8B models — selection→instillation regime shift). Curricula are
#     synth-filled toward the target, though quality control may land the final dataset below it.
#   - EVAL_SET_SIZE: held-out eval floor. Larger eval sets give statistically reliable
#     metrics — F1 CIs under-cover below n≈100; per-class macro-F1 needs ≥30–50/class or a
#     3-example rare class swings it wildly (the main source of the earlier score oscillation).
#   - DATA_SIZE_CEILING: hard cap so an over-eager target can't blow the wall clock.
CURRICULUM_SIZE_FLOOR = int(os.environ.get("SLM_CURRICULUM_FLOOR", "5000"))
EVAL_SET_SIZE = int(os.environ.get("SLM_EVAL_SET_SIZE", "800"))
DATA_SIZE_CEILING = int(os.environ.get("SLM_DATA_CEILING", "25000"))

# --- Local curriculum-synthesis model (contamination-safe, no Claude) ---
# Served by a vLLM OpenAI-compatible endpoint co-located with the run on its own GPU, launched
# by tests/pipeline/_l40s_task_body.sh and exported as SLM_SYNTH_ENDPOINT. The
# pipeline hits SYNTH_ENDPOINT for training-example synthesis, label verification, and CoT
# annotation instead of the orchestrator API, so synthetic data is fully local and its
# provenance is a model we own.
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
#   HW_FALLBACK_CHIP   — the reference chip NAME used ONLY when the device chip cannot be
#                        resolved from the user's description. It is a tier label for
#                        constraint bookkeeping; it carries no performance estimate.
HW_LATENCY_TTFT_MS = 2000
HW_POWER_WATTS = 5.0
HW_MIN_TOK_S = 0.0
HW_FALLBACK_CHIP = "snapdragon_778g"

# --- On-device hardware evaluation (hardware_eval/on_device_eval.py) ---
# HW_ONDEVICE_BACKEND — how metrics are gathered:
#     "unmeasured" (default)  : all-None metrics, NO hardware and NO estimates. Downstream
#                               gating renders these as "UNMEASURED" and declines to gate,
#                               rather than eliminating a candidate on a fabricated number.
#     "llama_cpp"             : local llama-cli timing on the built GGUF (no phone).
#     "adb_llama"             : llama-cli on a connected device via ADB.
#     "smolchat"              : broadcast to the SmolChat app + logcat scrape (richest).
#     ("theoretical" is accepted as a legacy alias for "unmeasured".)
# HW_GATING_ENABLED — when True, latency/power/memory become HARD gates in iterate_node
#     (a converged model that violates hardware is not accepted as terminal).
# HW_VERIFY_ON_DEVICE — when True, run.py runs a real on-device measurement pass after
#     convergence and writes hardware_eval.json. Requires a real (measuring) backend and a
#     connected device; on failure it logs and continues (never crashes the run).
HW_ONDEVICE_BACKEND = os.environ.get("SLM_HW_BACKEND", "unmeasured")
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
