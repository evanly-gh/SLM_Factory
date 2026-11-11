# SLM Factory

**An agentic pipeline that fine-tunes a small language model for a task, sized to run on the phone the user actually owns.**

This is the code behind the paper *Mobile AI Factory: Building Specialized Language Models for Your Mobile Device*.
Given a task and a target phone, an orchestrator LLM (Claude) plans the work and searches over model size,
quantization, training data and hyperparameters. It fine-tunes candidates with LoRA, scores each one on a
real quantized build (llama.cpp GGUF or MNN), and returns the smallest model that meets the accuracy goal
within the phone's measured limits.

---

## How it works

The pipeline is a **LangGraph state machine** (`agent/graph.py`) over one `AgentState` (`agent/state.py`),
checkpointed to SQLite so a multi-day Slurm job can be requeued and resumed. Every edge passes through a
global guard that can stop the run on wall-clock or step budget.

**Before the graph:** `tests/pipeline/run.py` resolves the target device from the task text
(`agent/nodes/cold_start/hardware_research.py`, backed by `data/devices.csv`).

**Cold-start stages:**
- `task_analysis` resolves the task from the registry (`SLM_BENCHMARK_TASK`). It filters the model pool
  (`config/android_pool.py`: SmolLM2 135M/360M, Gemma 3 270M, Qwen3 0.6B/1.7B/4B, Qwen3.5 0.8B/2B/4B, each
  at Q4_K_M and Q8_0) on the device's memory and storage, and parks the accuracy goal until it can be
  measured.
- `eval_setup` builds the frozen held-out eval set, pins a closed label space, labels each row's
  difficulty, and measures the teacher on that set to set the goal.
- `model_selection` picks the first variant: default `smallest_first`; also `largest_first`, `interpolation`,
  `orchestrator_choice`, `single_model` and `size_sweep`; set with `SLM_MODEL_SELECTION_STRATEGY`.

**Phone-measured pool:** when `config/filtered_pool.json` exists, or `SLM_FILTERED_POOL` points at a
contract such as `config/filtered_pools/*.json`, only the variants measured to fit the phone are allowed.
Each one is built for the runtime (llama.cpp or MNN) that won on the phone (`config/filtered_pool.py`).
The contracts are produced by `hardware/Benchmark-Harness/paper_table5/filter_pool.py`.

**The loop:**
```
curate → train → evaluate → iterate ↺   (with rollback, escalate and downward_probe branches)
```
- **curate**: grow the curriculum by one dataset version (`agent/data_rebuild.py`). The curriculum is
  cumulative, and a rebuild only adds rows: either `mine_new_real` (more real rows from sourced or
  newly found datasets) or `surgical_synthesis` (teacher-generated rows aimed at the model's failure
  categories). New rows pass task-specific quality controls and verifiers.
- **train**: one LoRA run, always from the base model, with five tunable hyperparameters and a
  completion-only loss (`training/lora_trainer.py`, isolated in a CUDA worker).
- **evaluate**: score the candidate on the frozen eval set from a real quantized artifact. The zero-shot
  base model competes as a candidate. Artifacts are cached under `artifacts/{gguf,mnn}/<model>/<hash>/`.
- **iterate**: the orchestrator's decision. It can stop, raise or lower the goal (a stretch goal), change
  data or hyperparameters, or escalate to the next size tier when progress stalls.
- **rollback / escalate / downward_probe**: undo a regression; move up a tier, keeping the best score;
  after convergence, look for a smaller variant that still meets the goal.

The accuracy goal is the local teacher's five-shot score on the eval set, floored at 0.80 and capped at
0.99 (`agent/threshold.py`). If five demonstrations don't fit the context, zero-shot is used, as for
CLINC150. `SLM_STOP_THRESHOLD` pins the goal manually. The teacher
(`Qwen3.6-35B-A3B`, served locally by vLLM) is also the synthesis model. The orchestrator defaults to
`claude-sonnet-5`.

## Tasks

The task registry is `tasks/__init__.py`, with one module per task:

| Category | Tasks |
|---|---|
| in-distribution | `dialogsum`, `sms_spam`, `gec_bea19`, `goemotions` |
| format-bound | `xlam_bfcl`, `calendar_json`, `ner_bc5cdr`, `topv2`, `multiconer` |
| out-of-distribution | `clinc150` |

An eleventh task, the `address_bench` conversational address gate, lives on the `address-bench` branch.

## Results (main runs, 2× L40S each)

Best student and its fine-tuning gain on the ten tasks, from the paper's main table:

| Task | Metric | Best student | Model | Gain over first fine-tune |
|---|---|---|---|---|
| clinc150 | macro F1 | 0.937 | Qwen3-4B-Instruct Q8_0 | +0.036 |
| sms_spam | minority F1 | 0.961 | Qwen3-1.7B Q4_K_M | +0.254 |
| calendar_json | AST arg match | 0.811 | Qwen3.5-0.8B Q4_K_M | +0.277 |
| xlam_bfcl | AST arg match | 0.852 | Qwen3.5-4B Q4_K_M | +0.018 |
| ner_bc5cdr | span F1 | 0.830 | Qwen3.5-0.8B Q4_K_M | +0.125 |
| goemotions | Ekman macro F1 | 0.598 | Qwen3-4B-Instruct Q8_0 | +0.000 |
| gec_bea19 | ERRANT F0.5 | 0.565 | Qwen3-4B-Instruct Q8_0 | +0.029 |
| multiconer | micro F1 | 0.653 | Qwen3-4B-Instruct Q8_0 | +0.050 |
| dialogsum | ROUGE-L | 0.482 | Qwen3-4B-Instruct Q8_0 | +0.000 |
| topv2 | exact match | 0.599 | Qwen3-4B-Instruct Q8_0 | +0.065 |

Run logs and artifacts are not in git. On the cluster they are in `logs/` (see `logs/README.md`,
which maps every kept run to the table or figure it backs) and `artifacts/`.

## Setup

Training and quantized evaluation need a CUDA GPU. Run setup inside a Slurm job, not on a login node.

```bash
uv sync                               # .venv: analysis/plotting env (pyproject.toml, uv.lock)
bash scripts/setup_gpu_env.sh         # .venv_gpu: training + evaluation stack
bash scripts/build_llamacpp_cuda.sh   # llama.cpp with CUDA, for GGUF builds and inference
bash scripts/setup_vllm_env.sh        # .venv_vllm: the local teacher server
bash scripts/setup_metric_envs.sh     # .venv_errant (GEC) and .venv_metrics (BERTScore)
bash scripts/setup_mnn_env.sh         # .venv_mnn: optional MNN backend (SLM_QUANT_BACKEND=mnn or run.py --quant-backend mnn)
cp .env.example .env                  # ANTHROPIC_API_KEY, endpoints, SLM_* settings
python scripts/download_datasets.py   # rebuild the bc5cdr and dialogsum bundles in data/local/
                                      # (other tasks' loaders fetch their data on first use)
```

## Running

Each task has a Slurm launcher. It starts the vLLM teacher on one GPU and the pipeline on the other:

```bash
sbatch tests/pipeline/run_ner_bc5cdr_l40s.slurm           # main run
sbatch tests/pipeline/run_ner_bc5cdr_ablation_singlemodel_l40s.slurm   # one-shot model pick
sbatch tests/pipeline/run_ner_bc5cdr_ablation_scarce_l40s.slurm        # 100 real rows
sbatch tests/pipeline/run_ner_bc5cdr_m07_oneplus8pro_l40s.slurm        # phone-in-the-loop pool
```

All task launchers share `tests/pipeline/_l40s_task_body.sh`, which calls `tests/pipeline/run.py`. Settings are
`SLM_`-prefixed environment variables (`SLM_BENCHMARK_TASK`, `SLM_MODEL_SELECTION_STRATEGY`,
`SLM_FILTERED_POOL`, `SLM_STOP_THRESHOLD`, ...); see `.env.example` and `config/config.py`.

**Analysis.**
- `scripts/analyze_runs.py` and `scripts/plot_run_analysis.py` compute search cost, exploration and escalation
  metrics from the run directories.
- `scripts/rescore_unseen.py`, via `tests/pipeline/rescore_unseen_l40s.slurm`, rescores the final models on
  rows the search never saw.
- `scripts/probe_api_*.py` produce the frontier-model baselines.
- These write into the local-only `docs/SLMF_Results/`. `plot_run_analysis.py` needs `analyze_runs.py`
  to have run first, and `scripts/plot_results.py` needs the local `docs/SLMF_Results/RESULTS.md`.

## Repository layout

```
agent/            LangGraph graph, state, checkpointing, cost ledger; nodes/ (curate, train, evaluate,
                  iterate, rollback, escalate, downward_probe, test_agent) and nodes/cold_start/
tasks/            Task registry: one TaskSpec per task (prompt, metric, loader, QC steps)
data/             Loaders, curriculum, quality controls, synthesis client and verifiers, label space
training/         LoRA trainer, CUDA worker isolation, GGUF / MNN quantization
eval/             Evaluation harness, per-task scorers, judge client, student server
config/           Model pool, model capabilities, measured metrics, filtered pools, global config
hardware_eval/    On-device measurement and quantized-accuracy tools used by the pipeline
hardware/         Phone-side benchmarking (harnesses, model conversion, power monitor, SmolChat-Android);
                  see hardware/README.md
scripts/          Environment setup, dataset bundling, analysis, baselines, run maintenance
tests/            Unit tests; tests/pipeline/ holds run.py and the Slurm launchers
```

Project notes, results write-ups and design documents are kept locally under `docs/` (gitignored).
