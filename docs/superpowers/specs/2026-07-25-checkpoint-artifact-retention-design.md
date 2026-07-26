# Checkpoint & Artifact Retention — Design

**Date:** 2026-07-25
**Status:** Approved, pending implementation

## Problem

The pipeline writes a full copy of the model to disk on every iteration and never
reclaims it. Across the three runs we keep, this is 395 GB, of which ~305 GB exists
only to serve checkpointing and one-shot evaluation.

Three distinct leaks, measured:

| Class | Size | Why it exists | Why it is dead weight |
|---|---|---|---|
| `artifacts/gguf/<model>/<hash>/` | 219 GB | Quantized model built so `evaluate_node` can score honestly on-device | Cache key is `(weights_ref, quant)` and `weights_ref` is unique per iteration, so the cache can never hit. Measured hit rate: 0/138 (NER), 2/66 (math). Every 2.6 GB file is written, read once, and kept forever. |
| `artifacts/training/<sel>/iter<N>-d<V>/checkpoint-*/` | 93 GB | HF Trainer mid-training resume state: `optimizer.pt`, a duplicate adapter, a duplicate 20 MB `tokenizer.json` | Dead the moment the iteration's `final_checkpoint` is saved. `training/lora_trainer.py:870` already deletes these, but only on the fallback-retry path — successful iterations never clean up. |
| `logs/runs/<run>/langgraph.sqlite` | 19 GB | LangGraph checkpointer, for resuming an interrupted run | Useless once a run has terminated. |

Separately, `logs/slurm/` mixes pipeline run logs with GPU infrastructure logs
(vLLM synth server, llama.cpp CUDA build), which makes the directory hard to read.

## Goals

1. Stop copying the model every iteration; bound steady-state disk per run.
2. Reclaim the existing ~305 GB without losing any model the pipeline would have chosen.
3. Split GPU setup logs into their own directory, and make the writers target it.

## Non-goals

- Reducing `langgraph.sqlite` growth *during* a run. Only post-run deletion is in scope.
- Deduplicating the 20 MB `tokenizer.json` copied into each `final_checkpoint`.
- Changing the eval, quantization, or training algorithms in any way.

## Retention policy

**Adapters (`final_checkpoint/`): keep all.** These are the trained models and the
trajectory record — 63 GB across 272 iterations. Not touched.

**GGUFs: keep every new-best.** An iteration whose score set a new best for its tier
keeps its GGUF; every other iteration's GGUF is reaped. This retains each
score-improving model quantized and immediately usable, and the run's final winner is
a new-best by construction.

**Trainer `checkpoint-*`: keep none** once the iteration's `final_checkpoint` exists.

**`langgraph.sqlite`: keep only for live/interrupted runs.**

## The models the pipeline chose

Confirmed with the user before any deletion. These must survive:

| Run | Model | Iteration | F1 | Adapter | GGUF |
|---|---|---|---|---|---|
| `slm-ner-l40s-37531245` | Qwen/Qwen3.5-4B @ Q4_K_M | 46 (dataset v4) | 0.8628 | `artifacts/training/Qwen_Qwen3.5-4B__Q4_K_M/iter46-d4/final_checkpoint` | `artifacts/gguf/Qwen_Qwen3.5-4B/485ca7619192/` |
| `slm-math-l40s-37576194` | Qwen/Qwen3.5-4B @ Q4_K_M | 6 (dataset v2) | 0.8263 ✓converged | `artifacts/training/Qwen_Qwen3.5-4B__Q4_K_M/iter6-d2/final_checkpoint` | `artifacts/gguf/Qwen_Qwen3.5-4B/c9d6e57ebb9c/` |
| `20260721_020113_37387566` | Qwen/Qwen3-1.7B bf16 | 53 | 0.7562 | `artifacts/iter53/final_checkpoint` | none (predates quantized eval) |

Evidence used: `scores.json.best_score`, the run summary's per-tier iteration table
(rows without `✗` set a new best), and the run's own
`origin=... score=... weights=...` declaration.

**Known trap:** the math run's *last* `[rollback] Restored to:` line points at iter2
(0.8163), which is a mid-run rollback superseded by iter6 (0.8263). Selecting the best
model by "last rollback" picks the wrong checkpoint for this run. Always corroborate
against `scores.json` and the `origin=` line.

## Design

### 1. Purge trainer checkpoints on success — `training/lora_trainer.py`

Extract the existing inline loop at `lora_trainer.py:870-872` into a module-level helper:

```python
def _purge_trainer_checkpoints(output_dir: str) -> None:
    """Remove HF Trainer `checkpoint-*` resume state. Never touches final_checkpoint."""
```

Call it from two places:
- the existing fallback-retry site (behaviour unchanged), and
- immediately **after** `model.save_pretrained(checkpoint_path)` /
  `tokenizer.save_pretrained(checkpoint_path)` at `lora_trainer.py:910-912`.

Safe because `load_best_model_at_end=True` has already loaded the best-val weights into
the model being saved, so `checkpoint-*` holds nothing that `final_checkpoint` lacks.
Ordering matters: purge only after a successful save, so a crash mid-training still
leaves resume state on disk.

### 2. Reap non-improving GGUFs — `agent/nodes/evaluate.py`

`_build_or_reuse_gguf` is unchanged — it still builds, validates, and writes the
sidecar. New behaviour lives after scoring in `evaluate_node`:

```python
def _reap_gguf(state, gguf_path: str, *, is_new_best: bool) -> None:
    """Retain the GGUF iff this iteration set a new best for its tier; else delete it."""
```

- On a new best: record `gguf_path` in `state["retained_gguf_paths"]` (a list, so it
  survives checkpoint/resume through the existing state codec) and leave it on disk.
- Otherwise: delete the GGUF directory and its `.validation.json` sidecar via
  `invalidate_gguf_cache`, so no stale sidecar can produce a false cache hit.
- Never reap a path present in `state["retained_gguf_paths"]`.

A rollback or downward-probe that needs a reaped GGUF rebuilds it through the existing
path. Correctness is unaffected; the cost is one re-quantization, which the measured
0–3% hit rate says is rare.

### 3. GPU setup logs — new `logs/gpu_setup/`

`logs/slurm/` keeps pipeline run logs only. GPU infrastructure logs move to
`logs/gpu_setup/`:

- `tests/pipeline/_l40s_task_body.sh:192` — `SYNTH_LOG` path changes to
  `$PROJ/logs/gpu_setup/synth-l40s-<profile>-<jobid>.out`, with `mkdir -p` before the
  server launch (the redirect at line 202 fails if the directory is absent).
- `scripts/serve_synth.slurm:9` — `#SBATCH --output` →
  `logs/gpu_setup/slm-synth-server-%j.out`.
- `scripts/build_llamacpp_cuda.sh` — toolchain build log written to the same directory.
- Existing kept synth logs (`synth-l40s-37387566.out`,
  `synth-l40s-auto-2gpu-37531245.out`, `synth-l40s-compat-4gpu-shared-37576194.out`)
  are moved, not deleted.
- `logs/README.md` documents the split.

`logs/` is gitignored (zero tracked files), so the directory cannot be committed into
existence with a `.gitkeep`. Slurm does **not** create the `--output` directory and the
job fails outright if it is missing, so creation must happen before submission:

- `scripts/setup_gpu_env.sh` gains `mkdir -p logs/slurm logs/gpu_setup`, making a fresh
  checkout submit-ready after the standard setup step.
- `_l40s_task_body.sh` additionally does `mkdir -p "$(dirname "$SYNTH_LOG")"` before the
  server launch, since that redirect happens at run time rather than submit time.

### 4. Retrospective cleanup script

A one-off script under `scripts/`, not wired into the pipeline. It derives its keep-set
from the run logs rather than from path globs:

1. Parse each run summary's per-tier sections
   (`── Tier N: <model> [<quant>]  (M iterations) ──`); rows lacking `✗` are new bests.
2. Resolve each to its on-disk `iter<N>-d*` directory (the summary omits the dataset
   version), then to `final_checkpoint`.
3. Recompute the content hash exactly as `evaluate.py` does —
   `sha1(f"{weights_ref}|{quant}")[:12]` — and keep `artifacts/gguf/<model_safe>/<hash>/`.
4. **Abort before deleting anything** unless both winner hashes from the table above are
   in the keep-set.
5. Delete: all `checkpoint-*` dirs, every GGUF dir outside the keep-set, and
   `langgraph.sqlite` for the two terminated runs that have one —
   `slm-ner-l40s-37531245` (9.8 GB) and `slm-math-l40s-37576194` (9.3 GB). The emotion
   run has no sqlite file.

Expected: keep 11 GGUF dirs (25.8 GB), delete 84 (193.1 GB), plus 93 GB of
`checkpoint-*` and 19 GB of sqlite. Total ~305 GB; kept runs go 395 GB → ~90 GB.

## Testing

- `_purge_trainer_checkpoints` removes `checkpoint-*` and leaves `final_checkpoint`
  intact, including when the two are siblings and when no checkpoints exist.
- Purge is not invoked when the save raises, so resume state survives a mid-training crash.
- `_reap_gguf` retains on `is_new_best=True`, deletes dir + sidecar otherwise, and never
  deletes a path in `state["retained_gguf_paths"]`.
- `retained_gguf_paths` round-trips through the state codec.
- The keep-set parser, run against the two committed run logs as fixtures, yields the
  two winner hashes.
- `tests/pipeline/test_task_slurm_scripts.py` extended to assert every `--output` and
  `SYNTH_LOG` points at the expected directory.

## Risks

- **Reaping a GGUF that is then needed.** Mitigated: rebuild is automatic, and the
  measured hit rate makes it rare. Worst case is wasted minutes, not a wrong result.
- **Keep-set parser drifting from the log format.** Mitigated by the winner-hash
  assertion, which turns a silent mis-parse into a hard abort.
- **Hash formula drifting from `evaluate.py`.** The retro script must import or mirror
  the exact expression; a divergence would delete live artifacts. The winner check
  catches the common case.
