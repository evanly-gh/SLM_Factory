# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

SLM Factory is an agentic fine-tuning loop for small language models (SLMs) targeting Android on-device deployment. It replicates the Pioneer Agent architecture (arXiv:2604.09791) with the model search space constrained to Android-feasible sizes (≤~1.5GB at INT4 quantization). The long-term goal is hardware-in-the-loop optimization: the fine-tuning agent is aware of Qualcomm Snapdragon/MediaTek NPU constraints and optimizes accuracy within them.

## Architecture Overview

The system has three tiers:

**Orchestrator (LangGraph + Claude Sonnet)** — the main agent, runs as a LangGraph state machine. Drives the full fine-tune → eval → diagnose → curate → retrain loop. Calls Claude API for failure diagnosis and synthetic data generation. Manages a context window budget across hundreds of turns using a Context Manager module.

**Tool Layer** — the agent's interface to the outside world:
- `bash` — executes training/eval scripts in the compute environment
- `read_file` / `edit_file` — reads datasets, configs, eval results
- `query_traces` — SQL-style queries on inference logs (production mode)
- `delegate_task` — spawns parallel sub-agents for concurrent work

**Training/Eval Backend** — LoRA fine-tuning scripts (via Unsloth or LLaMA-Factory), dataset formatters, and a held-out eval harness. Runs on the GPU cluster (H200/A100/L40). Training takes 10–30 min per run.

## Key Design Constraints

**Android model pool** — the agent may only select models from this bounded set:
- Tier 1 (≤800MB INT4): Qwen3-0.6B, Qwen3.5-0.8B, MiniCPM5-1B, MiniCPM4-0.5B, Gemma3-1B, Llama3.2-1B
- Tier 2 (800MB–1.5GB INT4): SmolLM2-1.7B, Qwen3-1.7B, Qwen3.5-2B, DeepSeek-R1-Distill-1.5B, Gemma3n-E2B
- Tier 3 / generous upper bound: Llama3.2-3B (Q3_K_M ~1.4GB), Ministral-3B

The agent starts at the smallest model and escalates only if accuracy targets are unmet.

**Regression gate** — no checkpoint is promoted unless it passes BOTH: (1) improvement on the failure eval set AND (2) preservation of accuracy on previously-correct examples above a threshold. Rollback to the prior checkpoint if the gate fails.

**HRM-Text (sapientinc/HRM-Text-1B)** — in the model pool as a research candidate. Requires custom handling: `token_type_ids` must be set, FlashAttention must be disabled at inference, standard llama.cpp/Ollama will not load it. Defer on-device export to Phase 2+.

## Operational Modes (from Pioneer Agent paper)

**Cold-Start Mode** (Phase 1): Input is a task description + public dataset. Agent downloads data, builds held-out eval set first, curates training data, trains, iterates. Up to 1,500 LangGraph turns.

**Production Mode** (Phase 2+): Input is a deployed model + judged inference failures. Agent runs an 8-stage pipeline: trace ingestion → taxonomy construction → live confirmation → parent model awareness → curriculum synthesis → retraining → regression gate → rollback. Up to 500 LangGraph turns.

## Pioneer Agent Paper — Key Implementation Details

Paper: "Pioneer Agent: Continual Improvement of Small Language Models in Production" (arXiv:2604.09791, Fastino Labs, April 2026)

**Context Manager**: Custom module that compacts older turns while preserving key decisions, eval results, and dataset lineage. Critical for sustaining 500–1500 turn runs without losing track.

**Data organization**: Each training attempt is a node in a DAG (Monte Carlo Graph Search). Lineage is tracked so the agent can attribute accuracy changes to specific interventions (data composition, hyperparameters, learning strategy). Never treat training attempts as isolated trials.

**AdaptFT-Bench structure**: Three deployment stages with increasing poison rates (15% → 40%), each with ~500 inference logs. Held-out eval set = union of all stage test splits. Every checkpoint is measured on the same mixed clean+noisy slice. Naive baseline retrains once per stage without filtering — the agent must do better by detecting and excluding poisoned/corrupted labels.

**Curriculum synthesis components**: corrected examples (from fixable failures) + hard negatives + contrastive pairs + replay data (to prevent forgetting).

**Diagnostic logic**: Low accuracy → data problem. Mid-range → optimization problem. High accuracy with specific failures → surgical intervention (hard negatives, threshold tuning).

## Phase 1 Scope (Weeks 1-4)

Task: Intent classification on CLINC150 (150-class benchmark, use 30-class subset for fast iteration).

- Weeks 1-2: Single fine-tune-and-eval pass. Prove the pipeline connects end-to-end.
- Weeks 3-4: Add the iteration loop. Agent inspects failures, generates synthetic data via Claude API, retrains. 3-4 rounds. No hardware involved yet.

Success: Accuracy improves across ≥3 consecutive rounds, full run <8 hours, <$50.

## Commands

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # add ANTHROPIC_API_KEY, EXA_API_KEY
```

### Run Phase 1 loop (SMS Spam classification by default)
```bash
python run.py
```

### Run on a different task
```bash
python -c "from run import run_cold_start; run_cold_start('fine-tune on ARC-Challenge reasoning')"
```

### Run tests
```bash
pytest tests/ -v
pytest tests/test_metrics.py tests/test_harness.py -v   # just eval tests
pytest tests/test_nodes.py -v                            # just node logic
pytest tests/test_graph.py -v                            # just graph wiring
```

### Smoke test (no GPU required)
```bash
python -c "from tests.test_graph import test_graph_smoke_terminates; test_graph_smoke_terminates()"
```

## Docs Structure

- `docs/superpowers/specs/` — design documents and specs
- `docs/progress/` — session-by-session progress log (see `progress-log.md`)
