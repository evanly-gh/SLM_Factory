# SLM Factory

**An agentic pipeline that autonomously fine-tunes a small language model for any user task, sized to run on the phone the user actually owns.**

Give it a plain-English task description and a target device ("fine-tune a model for biomedical NER, deployed on my Galaxy S24 Ultra with 12 GB RAM"). SLM Factory determines the task type, researches the device's constraints, selects a feasible model + quantization from an on-device pool, acquires and curates training data, and then runs a closed-loop LoRA fine-tuning search — diagnosing failures, rebuilding data, tuning hyperparameters, rolling back regressions, and escalating to larger models only when necessary — until it converges on the smallest model that meets an accuracy goal within its hardware budget.

---

## Overview & Motivation

Fine-tuning a small model is not primarily a *training* problem — it is a *surrounding-decisions* problem: task determination, data acquisition and curation, failure diagnosis, regression avoidance, model/quantization selection under hardware constraints, and iteration control. Training itself is the easy part; everything before and after it is hard.

SLM Factory is an independent, open-backend implementation of the **Pioneer Agent** architecture ("Continual Improvement of Small Language Models in Production", arXiv:2604.09791) adapted to a concrete, under-served setting: **on-device deployment of quantized Qwen3 / Qwen3.5 models to consumer Android phones.** The original paper's system runs on a proprietary hosted training service (Tinker SDK) and a managed sandbox; SLM Factory re-implements the closed loop on a standard open-source stack (Unsloth, PEFT, Transformers, llama.cpp) running on local/SLURM GPUs, and adds a first-class **hardware-and-quantization constraint model** the paper never had. `docs/PAPER.md` contains a full paper analysis plus an independent 200+-paper literature critique of every headline claim.

The orchestrator is a frontier LLM (Claude); the *product* it builds is a tiny, quantized, LoRA-fine-tuned model plus a deployable adapter.

---

## Key Features

- **Autonomous task determination** — a planner LLM parses a free-text task into one of five task types (classification, NER, math reasoning, code generation, generation), a benchmark, a label space, an eval metric, a stop threshold, and data-sizing targets.
- **Hardware-aware model selection** — device specs are resolved from a local phone-spec database (Kaggle) with an Exa web-search fallback; a pool of 6 Qwen base models × 3 quantizations (18 variants) is filtered on *measured, non-modelled* constraints (weight bytes must fit RAM and storage), then searched smallest-first with escalation.
- **π = (D, H, S) joint search** — each training attempt is a tuple of Dataset, Hyperparameters, and Strategy, searched jointly as a durable DAG rather than a fixed hyperparameter sweep.
- **Failure-driven data curation** — a contamination-firewalled "test-data agent" reports only aggregate difficulty/confusion signals, which drive a three-strategy data rebuild (`resample` / `acquire` / `synthesize`) with task-adaptive, verified synthetic generation.
- **Rollback-first iteration** — any score regression reverts to the best prior checkpoint instead of compensating; training always restarts from the base model for clean causal attribution.
- **Quantization-honest evaluation, on either on-device runtime** — quantized variants are scored on a real quantized build, never credited a full-precision score; a failed measurement is recorded as `n/a`, never a fabricated `0.0`. `--quant-backend` selects which runtime the artifact is built for and scored through: `llama_cpp` (default; GGUF via llama.cpp) or `mnn` (MNN model directory via MNN's `llmexport.py`, scored with pymnn on the GPU). One backend per run, and it is part of the resume fingerprint.
- **Durable long-running execution** — SQLite-authoritative LangGraph checkpointing, atomic artifact writes, a locked paid-API spend ledger, and SLURM signal-driven requeue that rolls a job over across multi-day segments with continuous progress.
- **Stretch goals** — once a goal is cleared, the orchestrator may ratchet the accuracy target upward under a hard ceiling; a missed stretch goal never turns a successful run into a reported failure.

---

## Architecture / How It Works

The pipeline is a **LangGraph state machine** (`agent/graph.py`) over a single `AgentState` TypedDict (`agent/state.py`). Every edge is conditional and passes through a global guard that can divert to `END` on wall-clock or step-budget exhaustion. There are two modes — **cold-start** (train a fresh model for a task) and **production** (continually improve a deployed model from judged traces); they share the same core loop.

### Pre-graph: hardware research & filtering
Runs in the driver before the graph builds (`agent/nodes/cold_start/hardware_research.py`, `hardware_filter.py`):
1. Resolve the device (local `data/devices.csv` fuzzy match → Exa fallback → LLM extracts only device-specific values: usable RAM, storage budget, reference chip).
2. Filter the model pool on quantities that are *known, not modelled* — the on-disk weight file must fit storage, and weight bytes must fit RAM. Throughput is never estimated; unmeasured candidates are never gated on a guess. Real measurements (`config/measured_metrics.json`) additionally gate when present.

### Cold-start entry stages
1. **`task_analysis`** — plan the task (type, labels, benchmark, threshold, data sizes); sort feasible models largest→smallest.
2. **`eval_setup`** — build a held-out eval set *before any training*, pin a closed label space from it, run four independent contamination firewalls, and difficulty-label every eval row (easy/medium/hard) by the smallest-vs-largest base-model capability gap.
3. **`model_selection`** — pick the starting variant. Four strategies: `smallest_first` (default, no probe), `largest_first` (feasibility probe), `interpolation` (fit F1 vs. log weight size across 3 probes), `orchestrator_choice` (one LLM call optimizing for resource efficiency).

### The shared loop
```
curate → train → evaluate → iterate ↺   (with rollback, escalate, downward_probe branches)
```
- **`curate`** — build one dataset artifact. No-op unless the last intervention was a data rebuild (keeps score movements causally attributable). Executes one of three strategies, synth-fills to a per-tier target, applies four quality controls (label balancing, length-outlier removal anchored on trusted rows, entity capping, near-duplicate removal), and records honest per-origin provenance.
- **`train`** — one LoRA configuration, always from the base model. Five tunable hyperparameters (rank, alpha ratio, weight decay, learning rate, epochs); batch shape and dropout are deliberately trainer-derived / retired. Completion-only loss, best-validation checkpointing, atomic write-or-nothing.
- **`evaluate`** — score on the frozen eval set. The zero-shot baseline competes as a candidate (a fine-tune that loses to zero-shot is discarded). Builds/reuses cache-keyed quantized artifacts for scoring — a GGUF file or an MNN model directory, per `--quant-backend` — each load-validated in its real runtime before it is scored, and reaped if it did not set a new best.
- **`iterate`** — the decision node. A strictly-ordered ladder: budget/wall-clock checks → threshold handling (with stretch-goal raising and downward probing) → forced escalation on stagnation (measured over an append-only eval history) → one tool-free LLM decision carrying a structured **run memory** (what worked, what failed since the last improvement, surgical spend per confusion pair). Chooses one of two mutually-exclusive interventions: `data_rebuild` or `hyperparameter`.
- **`rollback`** — pop the regressing score, prune the DAG node, restore the best non-pruned state, and force `iterate` to choose a *different* action.
- **`escalate`** — step to the nearest higher non-empty size tier (a different model or quant variant), reset per-model search state, carry `lifetime_best_score` across tiers.
- **`downward_probe`** — after convergence, search *downward* for the smallest variant that still clears the goal, to minimize on-device footprint.

### Production mode
Replaces the three cold-start entry stages with `trace_ingest` → `live_confirm` → `parent_awareness` (ingest judged production traces, re-confirm reproducible failures, build a complementary curriculum + replay buffer), then runs the identical loop. (Note: production mode is architected but not yet runnable end-to-end — see Status.)

`docs/PIPELINE.md` is the authoritative, code-verified specification of every node, guard, and state field.

---

## Tech Stack

- **Language:** Python ≥ 3.11
- **Orchestration:** LangGraph state machine with a SQLite checkpointer (`langgraph`, `langgraph-checkpoint-sqlite`)
- **Orchestrator LLM ("the brain"):** Claude (default `claude-sonnet-5`) via `anthropic` / `langchain-anthropic`, using the Sonnet 1M-token context beta for long trajectories; drives all planning, model-selection, and iteration-decision calls
- **Training:** Unsloth + PEFT + Transformers + PyTorch (4-bit base load, LoRA / text-only multimodal `FastVisionModel`)
- **Quantization & on-device eval:** llama.cpp (`llama-cpp-python`, CUDA-offloaded) for GGUF build + inference, or MNN (`llmexport.py` + `MNNConvert` + pymnn with its CUDA backend) for MNN builds — see `scripts/setup_mnn_env.sh`; real peak-RSS measurement via `getrusage`
- **Local teacher / judge / synthesis:** a self-hosted vLLM endpoint serving **Qwen3.6-35B-A3B** — the sole CoT teacher, LLM-as-judge, and curriculum-synthesis model (chosen for zero cloud cost, reproducibility, and contamination safety; Claude is never a teacher)
- **Data:** HuggingFace `datasets`, `pandas`, Exa (`exa-py`) for dataset/spec discovery, Kaggle for the device DB
- **Model pool:** official Qwen3 (text) and Qwen3.5 (multimodal, tuned text-only) — 0.6B to 4B, each in `Q4_K_M` / `Q8_0` / `bf16`
- **Infra:** SLURM (L40S / RTX 6000 Ada nodes) with signal-driven checkpoint/requeue; `uv` for dependency management; `pytest` (200+ tests)

---

## Results / Capabilities

The repository includes a curated benchmark suite (`agent/nodes/cold_start/eval_setup.py::NAMED_BENCHMARK_TASK_TYPES`) organized by *what fine-tuning is expected to buy*:

| Category | Tasks | Metric |
|---|---|---|
| **in-distribution** (FT buys format discipline) | `dialogsum_samsum` | LLM-judge 0–1 |
| **format-bound** (knows content, not the contract) | `xlam_bfcl`, `calendar_json` (function-call), `ner_bc5cdr` | AST arg-match / span-F1 |
| **out-of-distribution** (label not in surface form) | `clinc150`, `routerbench`, `proactive_listening` | macro / minority-class F1 |

Representative development runs committed to `logs/` (SLURM, L40S):

- **Biomedical NER (BC5CDR)** — escalated Qwen3.5-2B Q4 → Qwen3.5-4B Q4; best span-F1 **0.8628** (baseline ~0.025), 142 train→eval iterations, 2 model tiers, ~$13.43 Claude cost.
- **GSM8K math reasoning** — Qwen3.5-2B Q4 zero-shot 0.666; Qwen3.5-4B Q4 fine-tuned to **0.8263**, ~$3.20 Claude cost.

Key honesty findings surfaced by the system and documented in `docs/Evan's Notes/`: `clinc150` behaves as an in-distribution control (label *is* the utterance meaning, teacher scores 0.89, FT adds +0.003); `dialogsum_samsum` showed +0.0000 over 15 iterations, where the honest output of the loop is "ship the base model." These are treated as legitimate results, not failures.

---

## Installation / Setup

Requires a CUDA GPU for training and GGUF inference (the login-node `libstdc++` lacks `GLIBCXX_3.4.29`; use a compute node / the GPU venv).

```bash
# Dependencies (uv recommended)
uv sync                      # or: pip install -r requirements.txt

# GPU + llama.cpp CUDA build
bash scripts/setup_gpu_env.sh
bash scripts/build_llamacpp_cuda.sh

# Credentials — copy and fill in
cp .env.example .env         # ANTHROPIC_API_KEY, EXA_API_KEY, KAGGLE creds, endpoints

# (optional) refresh the device spec DB
python scripts/refresh_device_db.py
```

The local Qwen3.6 vLLM synthesis/judge endpoint must be reachable before a run starts (the driver blocks on a preflight check); set `SLM_REQUIRE_SYNTH=0` to opt out.

## Usage / Example

```bash
python tests/pipeline/run.py \
  "Fine tune a small model for biomedical named entity recognition on PubMed abstracts, \
   extracting Chemical and Disease entity spans (BC5CDR style), targeting deployment on my \
   Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
```

Common overrides (all `SLM_`-prefixed env vars): `SLM_ORCHESTRATOR_MODEL`, `SLM_MODEL_SELECTION_STRATEGY`, `SLM_BENCHMARK_TASK`, `SLM_CURRICULUM_SIZE`, `SLM_STOP_THRESHOLD`, `SLM_CHEAP=1` (cheap orchestrator for pipeline iteration), `SLM_FORCE_MODEL`. SLURM launchers for every benchmark task live in `tests/pipeline/*.slurm`.

---

## Project Structure

```
agent/                      Orchestration: LangGraph graph, state, guards
  graph.py                  Graph topology, global guards, routing table
  state.py                  AgentState TypedDict (the durable contract)
  nodes/                    Loop nodes: curate, train, evaluate, iterate,
                            rollback, escalate, downward_probe, test_agent
  nodes/cold_start/         task_analysis, eval_setup, hardware_*, model_selection/
  nodes/production/         trace_ingest, live_confirm, parent_awareness
  data_rebuild.py           Three-strategy data-rebuild plans
  run_memory.py             Structured decision memory built from the DAG
  checkpoint.py, state_codec.py   SQLite-authoritative durability
  cost.py                   Locked, cross-process API spend ledger
config/                     Model pool (android_pool.py), capabilities,
                            measured_metrics.json, global constants
data/                       Loaders (BC5CDR, CLINC150, xLAM/BFCL, ...), curriculum,
                            quality controls, label space, synth client, provenance
training/                   LoRA trainer, CUDA worker isolation, quantize, hparams
eval/                       Harness, scorers (classification/NER/function_call/diff/
                            code_execution/generation), LLM-judge client
hardware_eval/              On-device measurement, quantize, quant-accuracy eval (either backend)
scripts/                    GPU/vLLM setup, dataset download, SLURM supervision
tests/                      200+ pytest tests + tests/pipeline/ SLURM launchers
docs/                       PIPELINE.md (spec), PAPER.md (analysis+critique),
                            DATA_CURATION_AND_CAPS.md, model_pool.md, Evan's Notes/
```

---

## Status / Roadmap

**Working:** cold-start mode end-to-end (task analysis → hardware filtering → model selection → curate/train/evaluate/iterate loop with rollback, escalation, and downward probing), durable SQLite checkpoint/requeue, quantization-honest evaluation on both llama.cpp/GGUF and MNN, six-plus curated benchmark loaders, 200+ passing tests.

**Known gaps** (tracked in `docs/PIPELINE.md §12` and `docs/BUGS.md`):
- Production mode is architected but not runnable end-to-end (the production graph never builds an eval set, which `curate` requires).
- Synthesis has not yet run at scale in a completed benchmark run (endpoint availability); several curation strategies steer *which* strategy runs but do not yet reweight individual rows.
- The `delegate_task` / sub-agent tooling is defined but not yet wired into either graph.

**Roadmap** (from `docs/PAPER.md` §16, informed by 2024–2026 literature): CURLoRA to resist sequential-task collapse, a calibration-aware (KL/ECE) regression gate, formalized on-policy replay, and — Phase 2+ — GRPO as an SFT replacement for on-distribution continual learning.

---

*Orchestrator: Claude. Product: a phone-sized, quantized, LoRA-fine-tuned Qwen. See `docs/PIPELINE.md` for the authoritative, code-verified specification.*
