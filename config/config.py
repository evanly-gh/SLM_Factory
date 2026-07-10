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
