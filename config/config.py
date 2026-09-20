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
- JUDGE_MODEL        : required LLM-as-judge for `generation` scoring, served by
                       JUDGE_ENDPOINT. Local Qwen3.6 by default; the DeepSeek model
                       under SLM_SYNTH_API_MODE=1. Never Claude.
- CoT teacher        : the synth model, and ONLY that model — the local Qwen3.6, or
                       DeepSeek under API mode. Claude is never a CoT teacher
                       (paper §2.3/§2.5).

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

Teacher backend
---------------
    SLM_SYNTH_API_MODE=1 (or `python tests/pipeline/run.py --synth-api`) replaces the local
vLLM teacher with the DeepSeek API for synthesis, CoT, the fitness gate, the accuracy-goal
baseline AND the eval judge, so no vLLM server is launched at all. See SYNTH_API_MODE below
for the two consequences that are deliberate: synthetic-row provenance changes, and judged
eval scores stop being comparable to locally-judged runs.
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


# --- Orchestrator prompt caching ---
# The `iterate` decision is 96% of this project's Anthropic spend: on run 39294409 it was 39 calls,
# 379,829 INPUT tokens against 95,148 output, $1.71 of a $1.78 bill. Every one of those calls resent
# the same ~2,800-token system prompt, and every one reported `cache=0`.
#
# WHAT IS CACHED, AND WHY ONLY THAT
#     The breakpoint goes on the SYSTEM message and nothing else. Anthropic caches the prefix up to
#     and including the marked block, and a cache entry is written ONLY at a breakpoint — so marking
#     a block that changes every request means paying a write every time and never getting a read.
#     The iterate user message carries the trajectory, the tried-config list and the test report,
#     all of which change every turn. The system prompt is the longest genuinely stable prefix, so
#     it is the whole of what can be cached here.
#
# WHY A ONE-HOUR TTL
#     The default is five minutes, and consecutive iterate calls are separated by a full train +
#     evaluate cycle — about 14 minutes apart on run 39294409, longer on bigger models. At five
#     minutes essentially every lookup would miss, and a miss is worse than not caching because a
#     write costs more than a plain input token. One hour covers the gap with room to spare, and a
#     read also refreshes the TTL.
#
# THE TRADE, at Sonnet 5 rates ($2/MTok input, $4 1h-write, $0.20 read): a hit costs a tenth of an
# uncached token and a miss costs double. So this pays as long as most iterations land within the
# hour, and the worst case — every single call missing — is roughly twice the system-prompt
# component of the bill, which is cents. Set SLM_ORCHESTRATOR_CACHE=0 to turn it off.
ORCHESTRATOR_CACHE = os.environ.get("SLM_ORCHESTRATOR_CACHE", "1") == "1"
ORCHESTRATOR_CACHE_TTL = os.environ.get("SLM_ORCHESTRATOR_CACHE_TTL", "1h")

# Sonnet 5 and Sonnet 4.x will not cache a prefix below this, and say nothing when they decline:
# both `cache_creation_input_tokens` and `cache_read_input_tokens` come back 0. Checked here so a
# prompt that shrinks below the floor shows up as a deliberate skip rather than as silence.
ORCHESTRATOR_CACHE_MIN_TOKENS = int(
    os.environ.get("SLM_ORCHESTRATOR_CACHE_MIN_TOKENS", "1024")
)


def cacheable_system_content(text: str) -> str | list[dict]:
    """The system prompt as a cache-marked content block, or unchanged text when caching is off.

    Returns the plain string when caching is disabled or the prompt is too short to be cacheable,
    so callers can pass the result straight to `SystemMessage(content=...)` either way.

    The length test is a character estimate rather than a real tokenization: it only has to decide
    whether we are near a 1,024-token floor, the API silently ignores a request to cache something
    shorter, and importing a tokenizer here would pull a model download into config import.
    """
    if not ORCHESTRATOR_CACHE or CHEAP_MODE:
        return text
    if len(text) / 4.0 < ORCHESTRATOR_CACHE_MIN_TOKENS:
        return text
    return [{
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral", "ttl": ORCHESTRATOR_CACHE_TTL},
    }]

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
# Curriculum and eval sizes are per-task caps on `TaskSpec` (`initial_train_cap`, `select_cap`), not
# config constants. Two earlier attempts lived here and both were fictions:
#   * `DATASET_SIZE_BY_TYPE` (classification 150, NER 200, generation 600, ...) sat entirely below
#     the floor, so every value was clamped away on every path and it only created the impression
#     that per-task sizes were honoured. Removed 2026-08-05.
#   * `CURRICULUM_SIZE_FLOOR`/`DATA_SIZE_CEILING` then bounded a per-tier target computed from task
#     novelty and model capacity. Removed 2026-08-19 — the curriculum has no target: it starts at
#     whatever the loader supplied and grows by rebuild. The only thing that ever read the computed
#     figure was the `x 0.65` split behind the mystery 3,250-row curriculum.

# --- Teacher backend: local vLLM (default) or a hosted API (SLM_SYNTH_API_MODE=1) ---
# The pipeline's teacher does five things: curriculum synthesis, generated-row verification, CoT
# annotation, the teacher-fitness gate, and the accuracy-goal baseline. By DEFAULT all five run on
# a co-located vLLM server we own, which is what makes synthetic data contamination-safe and free.
#
# API MODE routes every one of them — plus the eval judge — to a hosted OpenAI-compatible endpoint
# instead, so no vLLM server is launched and no GPU is reserved for the teacher. It exists for runs
# on allocations too small to host a 35B alongside training, and for iterating on the pipeline
# without waiting ~40 minutes for a server to come up.
#
# Two consequences are deliberate and are NOT silently absorbed:
#   * Rows generated this way are labelled `synth:deepseek`, not `synth:vllm`. Their provenance is a
#     model we do not own and whose training data we cannot inspect, so a results run that used API
#     mode must say so.
#   * The eval judge changes model, and the judge's score cache is keyed by model, so `dialogsum` and
#     `toolbench` numbers from an API-mode run are NOT comparable to numbers judged by local Qwen3.6.
SYNTH_API_MODE = os.environ.get("SLM_SYNTH_API_MODE", "0") == "1"
SYNTH_API_PROVIDER = "deepseek"
# `deepseek-chat` and `deepseek-reasoner` were retired 2026-07-24; the live models are
# `deepseek-v4-flash` (cheap, the default) and `deepseek-v4-pro` (~3x the price).
SYNTH_API_BASE_URL = os.environ.get("SLM_SYNTH_API_BASE_URL", "https://api.deepseek.com")
SYNTH_API_MODEL = os.environ.get("SLM_SYNTH_API_MODEL", "deepseek-v4-flash")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if SYNTH_API_MODE and not DEEPSEEK_API_KEY:
    raise RuntimeError(
        "SLM_SYNTH_API_MODE=1 requires DEEPSEEK_API_KEY. Add it to .env (the runner loads that "
        "file with override=True) or unset SLM_SYNTH_API_MODE to use the local vLLM teacher."
    )

# --- Local curriculum-synthesis model (contamination-safe, no Claude) ---
# Served by a vLLM OpenAI-compatible endpoint co-located with the run on its own GPU, launched
# by tests/pipeline/_l40s_task_body.sh and exported as SLM_SYNTH_ENDPOINT. The
# pipeline hits SYNTH_ENDPOINT for training-example synthesis, label verification, and CoT
# annotation instead of the orchestrator API, so synthetic data is fully local and its
# provenance is a model we own.
# If the endpoint is unreachable, curate logs a warning and proceeds gold-only (never Claude).
SYNTH_ENDPOINT = os.environ.get("SLM_SYNTH_ENDPOINT", "")            # e.g. http://g3107:8000/v1
SYNTH_MODEL = os.environ.get("SLM_SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B")
# The context the teacher is SERVED at, which is a property of the vLLM launch and NOT of the model:
# Qwen3.6-35B-A3B declares max_position_embeddings=262144, and `_l40s_task_body.sh` serves it at 8192
# because KV cache scales linearly with this number and the server has one L40S.
#
# It is read here as well as passed to `vllm serve` so that callers which have to fit a prompt into it
# can ask instead of assuming. The body exports the same variable it launches with, so the two cannot
# drift — which they did: `agent/teacher_fitness` bounded its demonstration block by the TASK's
# max_seq_length, i.e. the student's window, and on toolbench that produced 199 HTTP 400s while on
# routerbench it would have wrongly disabled demonstrations that fit the teacher perfectly well.
SYNTH_MAX_MODEL_LEN = int(os.environ.get("SLM_SYNTH_MAX_MODEL_LEN", "8192"))
SYNTH_API_KEY = os.environ.get("SLM_SYNTH_API_KEY", "EMPTY")

# API mode redirects the three teacher constants at their source rather than making every consumer
# branch. `data/synth_client`, `agent/teacher_fitness`, `eval/endpoint_eval` and `data/curriculum`
# all resolve the teacher from these names, so one substitution here moves all of them at once.
# The context bound is the API model's, not a served-at figure we chose: DeepSeek V4 serves 1M, so
# the demonstration-fitting guards in `agent/teacher_fitness` stop discarding shots that fit fine.
if SYNTH_API_MODE:
    SYNTH_ENDPOINT = SYNTH_API_BASE_URL
    SYNTH_MODEL = SYNTH_API_MODEL
    SYNTH_API_KEY = DEEPSEEK_API_KEY
    SYNTH_MAX_MODEL_LEN = int(os.environ.get("SLM_SYNTH_MAX_MODEL_LEN", "131072"))

# --- Required LLM-as-judge for open `generation` eval scoring ---
# By default the judge shares the local synthesis vLLM server. Overrides must name
# the exact model ID returned by JUDGE_ENDPOINT/models; there is no cloud fallback.
# Remote hosts (including private cluster nodes) require an explicit safety opt-in, EXCEPT in API
# mode, where the hosted endpoint is the deliberate configuration rather than an accident.
# The cache defaults under the stable run's artifacts directory.
JUDGE_ENDPOINT = os.environ.get("SLM_JUDGE_ENDPOINT", SYNTH_ENDPOINT)
JUDGE_MODEL = os.environ.get(
    "SLM_JUDGE_MODEL", SYNTH_API_MODEL if SYNTH_API_MODE else "Qwen/Qwen3.6-35B-A3B"
)
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
# is cached per exact weights (evaluate._build_or_reuse_quant_artifact), so identical weights are
# never requantized; a new build per iteration is inherent because each iteration retrains.
# If the llama.cpp toolchain is missing/broken or the GGUF fails a real model-load
# validation, evaluation fails hard rather than recording BF16 or 0.0 under a quantized
# label. Set SLM_QUANT_EVAL=0 explicitly to use the old theoretical behavior. Requires:
# (1) llama.cpp `convert_hf_to_gguf` + `llama-quantize` on PATH, and (2)
# `llama-cpp-python`. Independent of SLM_HW_BACKEND / on-device eval.
QUANT_ACCURACY_EVAL = os.environ.get("SLM_QUANT_EVAL", "1") == "1"

# QUANT_BACKEND — WHICH on-device runtime the variant's quantized artifact is built for, and
# therefore which toolchain quantizes it and which engine scores it:
#   "llama_cpp" (DEFAULT) : convert_hf_to_gguf + llama-quantize → a single `model-<method>.gguf`,
#                           scored through llama-cpp-python. Every number this project has
#                           published so far came from this path; it stays the default so no
#                           existing launcher changes behaviour.
#   "mnn"                 : MNN's `llmexport.py` (+ a locally built MNNConvert) → an MNN model
#                           DIRECTORY (llm.mnn + llm.mnn.weight + tokenizer + config), scored
#                           in-process through pymnn's LLM API. This is the runtime MNN-Chat
#                           and the Qwen-family mobile stack actually use, and the reference
#                           harness this was ported from drives it the same way.
#
# ONE BACKEND PER RUN, deliberately. The quant variant a run is pinned to (`@Q4_K_M` / `@Q8_0`)
# describes the artifact that ships, and a run that built both would have to say which of the two
# scores `best_score` refers to. The selector names are kept for both backends and mapped to MNN's
# `--quant_bit` (Q4_K_M → 4 bits, Q8_0 → 8 bits, `SLM_MNN_QUANT_BLOCK`-wide blocks), so the
# backend is the only thing that changes between two otherwise identical runs — which is what
# makes a llama.cpp-vs-MNN accuracy comparison meaningful.
QUANT_BACKENDS = ("llama_cpp", "mnn")
QUANT_BACKEND = os.environ.get("SLM_QUANT_BACKEND", "llama_cpp").strip().lower()
if QUANT_BACKEND not in QUANT_BACKENDS:
    raise RuntimeError(
        f"SLM_QUANT_BACKEND={QUANT_BACKEND!r} is not a known quantization backend. "
        f"Valid values: {list(QUANT_BACKENDS)}."
    )

# Nominal Li-ion voltage for mA→W power conversion when live voltage is unreadable.
HW_BATTERY_VOLTAGE_V = float(os.environ.get("SLM_HW_BATTERY_VOLTAGE_V", "3.85"))
