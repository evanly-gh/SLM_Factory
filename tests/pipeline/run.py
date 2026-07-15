#!/usr/bin/env python
"""
SLM Factory — cold-start runner.

Runs the full agentic fine-tuning loop for a given task description.
The loop terminates when f(π) >= stop_threshold (calibrated by the planner
against published SOTA at the target model size) or the 1500-turn cold-start
budget is exhausted. No artificial step caps.

Usage:
    python run.py "fine tune a model for SMS spam detection on my Pixel 8 (8GB RAM)"
    python run.py --model "Qwen/Qwen3-1.7B" "your task description"

The task description should mention:
  - The target task (what the model should do)
  - The target device (name, RAM, storage) so hardware constraints can be resolved

Environment variables (all optional):
    SLM_FORCE_MODEL   override model selection (e.g. "Qwen/Qwen3-0.6B")
"""
import argparse
import datetime
import json
import os
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

# Quiet the ML-stack log flood BEFORE transformers/unsloth/datasets are imported anywhere
# (transformers reads TRANSFORMERS_VERBOSITY at import time). Keeps training-loss lines.
from agent.logging_setup import set_ml_env
set_ml_env()

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJ, ".env"))

# --------------------------------------------------------------------------
# Run directory + tee logger (set up before any imports that might print)
# --------------------------------------------------------------------------
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Disambiguate concurrent runs: several jobs launched together start Python within
# the same second, and a bare %H%M%S timestamp makes them share one run dir and
# clobber each other's run.log/scores.json/artifacts (all opened with "w"). Append
# the SLURM job id (or PID off-cluster) so every run gets its own directory.
_RUN_UID = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
TS = f"{TS}_{_RUN_UID}"
RUN_DIR = os.path.join(PROJ, "logs", "runs", TS)
os.makedirs(os.path.join(RUN_DIR, "artifacts"), exist_ok=False)
_LOGF = open(os.path.join(RUN_DIR, "run.log"), "w", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, d):
        for s in self.streams:
            if not getattr(s, "closed", False):
                s.write(d)
                s.flush()

    def flush(self):
        for s in self.streams:
            if not getattr(s, "closed", False):
                s.flush()

    def isatty(self):
        return False

    def fileno(self):
        return sys.__stdout__.fileno() if sys.__stdout__ is not None else -1

    def __getattr__(self, name):
        return getattr(sys.__stdout__, name) if sys.__stdout__ is not None else None


sys.stdout = _Tee(sys.__stdout__, _LOGF)


def log(msg=""):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# --------------------------------------------------------------------------
# Cost-tracking wrappers — wrap before any node imports so every API call
# that happens inside nodes is metered automatically.
# --------------------------------------------------------------------------
from agent.cost import LEDGER
import anthropic as _anthropic_mod

_RealAnthropic = _anthropic_mod.Anthropic


class _TrackedMessages:
    def __init__(self, real):
        self._real = real

    def create(self, *a, **k):
        resp = self._real.create(*a, **k)
        u = getattr(resp, "usage", None)
        if u is not None:
            LEDGER.record_anthropic(
                getattr(u, "input_tokens", 0),
                getattr(u, "output_tokens", 0),
            )
        return resp


class _TrackedAnthropic:
    def __init__(self, *a, **k):
        self._c = _RealAnthropic(*a, **k)
        self.messages = _TrackedMessages(self._c.messages)


_anthropic_mod.Anthropic = _TrackedAnthropic

import exa_py as _exa_mod

_RealExa = _exa_mod.Exa


class _TrackedExa:
    def __init__(self, *a, **k):
        self._e = _RealExa(*a, **k)

    def search_and_contents(self, *a, **k):
        LEDGER.record_exa(1)
        return self._e.search_and_contents(*a, **k)

    def search(self, *a, **k):
        LEDGER.record_exa(1)
        return self._e.search(*a, **k)


_exa_mod.Exa = _TrackedExa

# --------------------------------------------------------------------------
# Redirect artifact paths and curation log to the per-run directory.
# Must happen before any node module caches ARTIFACTS_DIR at import time.
# --------------------------------------------------------------------------
import agent.nodes.curate as _curate_mod
import agent.nodes.train as _train_mod
import agent.nodes.evaluate as _evaluate_mod
import agent.nodes.cold_start.eval_setup as _eval_setup_mod
from data.curation_log import CurationLog

ART = os.path.join(RUN_DIR, "artifacts")
_curate_mod.ARTIFACTS_DIR = ART
_train_mod.ARTIFACTS_DIR = ART
_eval_setup_mod.ARTIFACTS_DIR = ART

CUR_LOG_PATH = os.path.join(RUN_DIR, "data-curation.md")
_evaluate_mod.CurationLog = lambda *a, **k: CurationLog(CUR_LOG_PATH)

# --------------------------------------------------------------------------
# Args
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="SLM Factory cold-start runner",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("description", nargs="?", default="", help="Natural-language task description")
parser.add_argument("--model", default="", help="Force a specific HuggingFace model ID")
args = parser.parse_args()

description = args.description.strip()
if not description:
    parser.error("A task description is required.")

# SLM_FORCE_MODEL env takes precedence over --model flag (keeps slurm scripts simple)
force_model = os.environ.get("SLM_FORCE_MODEL", "") or args.model

# --------------------------------------------------------------------------
# Hardware research
# --------------------------------------------------------------------------
import config.config as config
from agent.nodes.cold_start.hardware_research import research_device

log(f"=== SLM Factory cold-start  (run {TS}) ===")
log(f"task: {description}")
log(f"turn budget: {config.MAX_TURNS_MAIN}")
log(f"orchestrator model: {config.ORCHESTRATOR_MODEL}  |  judge model: {config.JUDGE_MODEL}")
log("")
log("  ▶ hardware_research")

_hw_t0 = time.time()
_hw_cost_before = LEDGER.total_cost
HW, hw_info = research_device(description, log=log)
_hw_dt = time.time() - _hw_t0
_hw_dcost = LEDGER.total_cost - _hw_cost_before

log(f"      constraints: storage={HW.storage_mb}MB  memory={HW.memory_mb}MB  "
    f"latency_ttft={HW.latency_ttft_ms}ms  power={HW.power_watts}W  "
    f"chip={HW.target_chip}  [{_hw_dt:.1f}s  +${_hw_dcost:.4f}]")

with open(os.path.join(RUN_DIR, "device_research.json"), "w") as f:
    json.dump(hw_info, f, indent=2)
log("")

# --------------------------------------------------------------------------
# Initial state
# --------------------------------------------------------------------------
initial_state = {
    "description": description,
    "target_metric": "F1",
    "hardware_constraints": HW,
    "task_type": "",
    "autonomous": True,
    "task_plan": None,
    "selected_model": None,
    "feasible_models": [],
    "stop_threshold": config.DEFAULT_STOP_THRESHOLD,
    "initial_stop_threshold": config.DEFAULT_STOP_THRESHOLD,
    "train_examples": [],
    "eval_set": None,
    "data_source": None,
    "current_dataset_path": None,
    "dataset_version": 0,
    "best_weights_ref": None,
    "best_score": 0.0,
    "lifetime_best_score": 0.0,
    "iteration": 0,
    "scores": [],
    "dag": [],
    "consecutive_no_improvement": 0,
    "downward_probe_done": False,
    "last_eval": None,
    "last_curation": None,
    "last_intervention": "data_rebuild",
    "last_hypothesis": "",
    "llm_iterate_decision": None,
    "next_action": "train",
    "model_baselines": [],
    "quantize_enabled": False,
    "hw_gating_enabled": config.HW_GATING_ENABLED,
    "mode": "cold_start",
    "deployed_model_ref": None,
    "traces": None,
    "failure_taxonomy": None,
    "regression_set": None,
    "replay_buffer": None,
    "turn_budget": config.MAX_TURNS_MAIN,
    "_pending_weights_refs": None,
    "_pending_training_outputs": None,
    "_pending_configs": None,
}

if force_model:
    os.environ["SLM_FORCE_MODEL"] = force_model
    log(f"  model override: {force_model}")

# --------------------------------------------------------------------------
# Compatibility shim (BUGS B111)
# --------------------------------------------------------------------------
# Several HF model repos ship custom modeling code (loaded via trust_remote_code)
# that still imports `is_torch_fx_available` from transformers.utils.import_utils —
# e.g. openbmb/MiniCPM4-0.5B's modeling_minicpm.py. transformers 5.5.0 removed that
# symbol, so the import crashes model load (ARC/MiniCPM) and the scaling-curve probe.
# Re-add it: torch.fx has existed since torch 1.8, so availability tracks torch itself.
import importlib.util as _ilu
import transformers.utils.import_utils as _tf_iu
if not hasattr(_tf_iu, "is_torch_fx_available"):
    def _is_torch_fx_available():
        return _ilu.find_spec("torch") is not None and _ilu.find_spec("torch.fx") is not None
    _tf_iu.is_torch_fx_available = _is_torch_fx_available
    # Some remote code imports it from the transformers top-level namespace too.
    import transformers as _tf
    if not hasattr(_tf, "is_torch_fx_available"):
        _tf.is_torch_fx_available = _is_torch_fx_available

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
from agent.graph import build_graph

last_state = initial_state
t_start = time.time()

try:
    graph = build_graph(mode="cold_start")

    # Export the static graph structure as Mermaid
    try:
        mermaid_text = graph.get_graph().draw_mermaid()
        mermaid_path = os.path.join(RUN_DIR, "graph.mermaid")
        with open(mermaid_path, "w") as f:
            f.write(mermaid_text)
        log(f"  Graph structure saved: {mermaid_path}")
    except Exception as e:
        log(f"  Could not export Mermaid graph: {e}")

    for delta in graph.stream(
        initial_state,
        stream_mode="updates",
        config={"recursion_limit": config.MAX_TURNS_MAIN},
    ):
        node_name = list(delta.keys())[0]
        last_state = list(delta.values())[0]
except KeyboardInterrupt:
    log("\n  interrupted by user")
except Exception as exc:
    import traceback
    log(f"\n  !! exception in {type(exc).__name__}: {exc}")
    log(traceback.format_exc())

elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Post-convergence on-device hardware verification (opt-in via SLM_HW_VERIFY_ON_DEVICE=1)
# --------------------------------------------------------------------------
# Measures the FINAL selected model on real hardware, writes hardware_eval.json,
# re-checks the four constraints against MEASURED values, and records the deployed
# model reference. Fully guarded: any failure logs and continues.
hardware_eval_report = None
if config.HW_VERIFY_ON_DEVICE:
    log("")
    log("  ▶ on-device hardware verification")
    try:
        final_model = last_state.get("selected_model")
        best_ref = last_state.get("best_weights_ref")
        if final_model is None or not best_ref:
            log("      [hw-verify] no converged model/weights — skipping")
        elif final_model.quant is None:
            log("      [hw-verify] final model is BF16 (no GGUF path) — skipping "
                "SmolChat/GGUF verification; set a quantized variant to enable")
        else:
            from training.lora_trainer import merge_for_quantization
            from training.quantize import quantize_from_model_spec
            from hardware_eval.on_device_eval import run_on_device_eval, result_to_dict
            from config.android_pool import check_hardware_constraints, all_constraints_pass

            mid_safe = final_model.model_id.replace("/", "_")
            merged = merge_for_quantization(
                best_ref, os.path.join(ART, "merged", mid_safe, "final_verify"))
            gguf = quantize_from_model_spec(
                merged, os.path.join(ART, "gguf", mid_safe, "final_verify"), final_model.quant)

            hw_result = run_on_device_eval(
                final_model, HW, gguf_path=gguf,
                backend=config.HW_ONDEVICE_BACKEND, log=log)
            hw_check = check_hardware_constraints(
                final_model, HW, measured=hw_result.to_measured())
            passed = all_constraints_pass(hw_check)

            hardware_eval_report = {
                "backend": config.HW_ONDEVICE_BACKEND,
                "model_id": final_model.model_id,
                "quant": final_model.quant,
                "gguf_path": gguf,
                "measured_success": hw_result.success,
                "measured_error": hw_result.error,
                "result": result_to_dict(hw_result),
                "constraint_check": hw_check,
                "all_constraints_pass": passed,
            }
            with open(os.path.join(RUN_DIR, "hardware_eval.json"), "w") as f:
                json.dump(hardware_eval_report, f, indent=2, default=str)

            if hw_result.success:
                log(f"      [hw-verify] {hw_result.eval_method}: "
                    f"TTFT={hw_result.ttft_ms}ms  tok/s={hw_result.tok_per_s}  "
                    f"peakRSS={hw_result.peak_memory_mb}MB  power={hw_result.avg_watts}W  "
                    f"→ constraints {'PASS' if passed else 'FAIL'}")
                if passed:
                    last_state["deployed_model_ref"] = best_ref
            else:
                log(f"      [hw-verify] measurement failed: {hw_result.error}")
    except Exception as exc:
        import traceback
        log(f"      [hw-verify] skipped due to error: {exc}")
        log(traceback.format_exc())

# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
cost = LEDGER.snapshot()

with open(os.path.join(RUN_DIR, "dag.json"), "w") as f:
    json.dump(last_state.get("dag", []), f, indent=2)

with open(os.path.join(RUN_DIR, "scores.json"), "w") as f:
    json.dump({
        "scores": last_state.get("scores", []),
        "best_score": last_state.get("best_score", 0.0),
        "stop_threshold": last_state.get("stop_threshold", config.DEFAULT_STOP_THRESHOLD),
        "iterations": last_state.get("iteration", 0),
        "converged": last_state.get("best_score", 0.0) >= last_state.get("stop_threshold", 1.0),
    }, f, indent=2)

with open(os.path.join(RUN_DIR, "cost.json"), "w") as f:
    json.dump(cost, f, indent=2)

if last_state.get("task_plan"):
    with open(os.path.join(RUN_DIR, "task_plan.json"), "w") as f:
        json.dump(last_state["task_plan"], f, indent=2)

# --------------------------------------------------------------------------
# DAG summary (what was tried and how well it worked)
# --------------------------------------------------------------------------
dag = last_state.get("dag", [])
dag_lines = ["# DAG Summary — Training Attempts", ""]
dag_lines.append(f"{'Iter':>4}  {'Model':<30}  {'Score':>7}  {'Pruned':>6}  {'Config':<25}  {'Intervention'}")
dag_lines.append("-" * 100)
for node in dag:
    pruned_str = "✗" if node.get("pruned") else ""
    dag_lines.append(
        f"{node.get('iteration', '?'):>4}  "
        f"{node.get('model_id', '?'):<30}  "
        f"{node.get('score', 0):.4f}  "
        f"{pruned_str:>6}  "
        f"{node.get('best_config', '?'):<25}  "
        f"{node.get('intervention', '?')}"
    )
dag_lines.append("")
dag_text = "\n".join(dag_lines)
with open(os.path.join(RUN_DIR, "dag_summary.txt"), "w") as f:
    f.write(dag_text)

# --------------------------------------------------------------------------
# Baseline vs fine-tuned summary
# --------------------------------------------------------------------------
baselines = last_state.get("model_baselines", [])
# Update the final model's best_finetuned_f1 with the current best score
if baselines:
    final_model_id = last_state.get("selected_model")
    if final_model_id:
        final_id = final_model_id.model_id
        for entry in baselines:
            if entry["model_id"] == final_id:
                entry["best_finetuned_f1"] = max(
                    entry.get("best_finetuned_f1", 0.0),
                    last_state.get("best_score", 0.0),
                )

with open(os.path.join(RUN_DIR, "baselines.json"), "w") as f:
    json.dump(baselines, f, indent=2)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
m = last_state.get("selected_model")
best = last_state.get("best_score", 0.0)
threshold = last_state.get("stop_threshold", config.DEFAULT_STOP_THRESHOLD)
converged = best >= threshold

log("")
log(f"{'='*70}")
log(f"  RUN COMPLETE — {elapsed:.1f}s")
log(f"{'='*70}")
log(f"  task_type : {last_state.get('task_type')}")
log(f"  model     : {m.model_id if m else None}")
log(f"  iterations: {last_state.get('iteration', 0)}")
log(f"  best F1   : {best:.4f}  (threshold {threshold:.4f})  "
    f"{'✓ converged' if converged else '✗ budget exhausted'}")
log(f"  trajectory: {[f'{x:.3f}' for x in last_state.get('scores', [])]}")

# Baseline improvement table
if baselines:
    log("")
    log(f"  Model Improvement Report:")
    log(f"  {'Model':<35} {'Baseline':>9} {'Best FT':>9} {'Δ':>9}")
    log(f"  {'-'*65}")
    for entry in baselines:
        bl = entry.get("baseline_f1", 0.0)
        ft = entry.get("best_finetuned_f1", 0.0)
        delta = ft - bl
        log(f"  {entry['model_id']:<35} {bl:>8.4f}  {ft:>8.4f}  {delta:>+8.4f}")

# DAG summary
if dag:
    log("")
    log(f"  DAG Traversal:")
    for line in dag_lines[2:]:  # skip header and blank
        log(f"  {line}")

log("")
log(
    f"  cost: Claude {cost['anthropic_calls']} calls "
    f"({cost['input_tokens']}→{cost['output_tokens']} tok) ${cost['anthropic_cost_usd']:.4f}  |  "
    f"Exa {cost['exa_calls']} searches ${cost['exa_cost_usd']:.4f}  |  "
    f"total ${cost['total_cost_usd']:.4f}"
)
log(f"  logs: {RUN_DIR}")

sys.stdout = sys.__stdout__
_LOGF.close()
