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

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJ, ".env"))

# --------------------------------------------------------------------------
# Run directory + tee logger (set up before any imports that might print)
# --------------------------------------------------------------------------
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(PROJ, "logs", "runs", TS)
os.makedirs(os.path.join(RUN_DIR, "artifacts"), exist_ok=True)
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
    "stop_threshold": config.DEFAULT_STOP_THRESHOLD,
    "initial_stop_threshold": config.DEFAULT_STOP_THRESHOLD,
    "train_examples": [],
    "eval_set": None,
    "current_dataset_path": None,
    "dataset_version": 0,
    "best_weights_ref": None,
    "best_score": 0.0,
    "iteration": 0,
    "scores": [],
    "dag": [],
    "consecutive_no_improvement": 0,
    "last_eval": None,
    "last_curation": None,
    "last_intervention": "",
    "last_hypothesis": "",
    "llm_iterate_decision": None,
    "next_action": "train",
    "quantize_enabled": False,
    "hw_gating_enabled": False,
    "mode": "cold_start",
    "deployed_model_ref": None,
    "traces": None,
    "failure_taxonomy": None,
    "regression_set": None,
    "replay_buffer": None,
    "turn_budget": config.MAX_TURNS_MAIN,
    "messages": [],
    "_pending_weights_refs": None,
    "_pending_configs": None,
}

if force_model:
    os.environ["SLM_FORCE_MODEL"] = force_model
    log(f"  model override: {force_model}")

# --------------------------------------------------------------------------
# Node logging helper
# --------------------------------------------------------------------------
def _log_node_update(node_name: str, state: dict):
    """Print a one-line structured summary for each completed node."""
    cost_str = f"  running cost ${LEDGER.total_cost:.4f}"

    if node_name == "task_analysis":
        m = state.get("selected_model")
        if m:
            log(f"  ✓ task_analysis  type={state.get('task_type')}  "
                f"model={m.model_id} ({m.int4_size_mb}MB)  "
                f"threshold={state.get('stop_threshold'):.3f}{cost_str}")

    elif node_name == "eval_setup":
        es = state.get("eval_set")
        if es:
            log(f"  ✓ eval_setup  train={len(state.get('train_examples', []))}  "
                f"eval pos={len(es.pos)}/neg={len(es.neg)}/boundary={len(es.boundary)}"
                f"{cost_str}")

    elif node_name == "curate":
        log(f"  ✓ curate  v{state.get('dataset_version')}  "
            f"{os.path.basename(state.get('current_dataset_path') or '')}"
            f"{cost_str}")

    elif node_name == "train":
        log(f"  ✓ train  iteration={state.get('iteration')}{cost_str}")

    elif node_name == "evaluate":
        ev = state.get("last_eval")
        if ev:
            log(f"  ✓ evaluate  F1={ev.f1:.4f}  best={state.get('best_score', 0):.4f}  "
                f"failures={len(ev.failures)}  "
                f"scores={[f'{x:.3f}' for x in state.get('scores', [])]}"
                f"{cost_str}")

    elif node_name == "iterate":
        log(f"  ✓ iterate  next={state.get('next_action')}  "
            f"intervention={state.get('last_intervention')}"
            f"{cost_str}")

    elif node_name == "rollback":
        log(f"  ✓ rollback  restored best={state.get('best_score', 0):.4f}{cost_str}")

    elif node_name == "escalate":
        m = state.get("selected_model")
        action = state.get("next_action")
        if action == "terminate":
            log(f"  ✓ escalate  no further models available — terminating{cost_str}")
        elif m:
            log(f"  ✓ escalate  → {m.model_id}{cost_str}")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
from agent.graph import build_graph

last_state = initial_state
t_start = time.time()

try:
    graph = build_graph(mode="cold_start")
    for delta in graph.stream(
        initial_state,
        stream_mode="updates",
        config={"recursion_limit": config.MAX_TURNS_MAIN},
    ):
        node_name = list(delta.keys())[0]
        last_state = list(delta.values())[0]
        _log_node_update(node_name, last_state)
except KeyboardInterrupt:
    log("\n  interrupted by user")
except Exception as exc:
    import traceback
    log(f"\n  !! exception in {type(exc).__name__}: {exc}")
    log(traceback.format_exc())

elapsed = time.time() - t_start

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
# Summary
# --------------------------------------------------------------------------
m = last_state.get("selected_model")
best = last_state.get("best_score", 0.0)
threshold = last_state.get("stop_threshold", config.DEFAULT_STOP_THRESHOLD)
converged = best >= threshold

log("")
log(f"=== done in {elapsed:.1f}s ===")
log(f"task_type : {last_state.get('task_type')}")
log(f"model     : {m.model_id if m else None}")
log(f"iterations: {last_state.get('iteration', 0)}")
log(f"best F1   : {best:.4f}  (threshold {threshold:.4f})  "
    f"{'✓ converged' if converged else '✗ budget exhausted'}")
log(f"trajectory: {[f'{x:.3f}' for x in last_state.get('scores', [])]}")
log(
    f"cost      : Claude {cost['anthropic_calls']} calls "
    f"({cost['input_tokens']}→{cost['output_tokens']} tok) ${cost['anthropic_cost_usd']:.4f}  |  "
    f"Exa {cost['exa_calls']} searches ${cost['exa_cost_usd']:.4f}  |  "
    f"total ${cost['total_cost_usd']:.4f}"
)
log(f"logs      : {RUN_DIR}")

sys.stdout = sys.__stdout__
_LOGF.close()
