#!/usr/bin/env python
"""
Autonomous cold-start runner (Pioneer Agent, arXiv:2604.09791v1).

Given ANY natural-language task description, the orchestrator (Claude Sonnet):
  1. classifies the task + builds a data-acquisition plan          (task_analysis  -> task_planner)
  2. autonomously acquires a labeled dataset from the web via Exa  (eval_setup     -> web_acquire)
  3. builds the held-out eval set + curriculum (hard negatives)    (eval_setup / curate, real Sonnet)
  4. iterates train -> evaluate -> diagnose -> curate              (the LangGraph loop)

Proof-of-concept settings:
  - NO step cap. The loop runs until the held-out accuracy rises 3 times, then stops
    (a generous safety cap prevents a true runaway).
  - GPU train + model inference are STUBBED (no GPU); eval F1 follows a simulated
    rising trajectory so the orchestration loop is exercised end to end.
  - Every Claude + Exa call is metered and the USD cost is written to the logs.

Usage:  python run_autonomous.py "your task description here"
"""
import os, sys, json, time, datetime, functools
from unittest.mock import patch

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ); os.chdir(PROJ)
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJ, ".env"))

# ---------------- run dir + tee'd logger -------------------------------
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(PROJ, "logs", "runs", TS)
os.makedirs(os.path.join(RUN_DIR, "artifacts"), exist_ok=True)
_LOGF = open(os.path.join(RUN_DIR, "run.log"), "w", buffering=1)

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams:
            if not getattr(s, "closed", False):
                s.write(d); s.flush()
    def flush(self):
        for s in self.streams:
            if not getattr(s, "closed", False):
                s.flush()
    # Be a well-behaved stream: libraries (e.g. transformers' loading report) probe
    # isatty()/fileno()/encoding on sys.stdout.
    def isatty(self): return False
    def fileno(self): return sys.__stdout__.fileno()
    def __getattr__(self, name): return getattr(sys.__stdout__, name)
sys.stdout = _Tee(sys.__stdout__, _LOGF)

def log(msg=""):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---------------- cost-tracking wrappers (Claude + Exa) ----------------
from agent.cost import LEDGER
import anthropic
_RealAnthropic = anthropic.Anthropic
class _CostMsgs:
    def __init__(self, real): self._real = real
    def create(self, *a, **k):
        resp = self._real.create(*a, **k)
        u = getattr(resp, "usage", None)
        if u is not None:
            LEDGER.record_anthropic(getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
        return resp
class _CostAnthropic:
    def __init__(self, *a, **k):
        self._c = _RealAnthropic(*a, **k)
        self.messages = _CostMsgs(self._c.messages)
anthropic.Anthropic = _CostAnthropic

import exa_py
_RealExa = exa_py.Exa
class _CostExa:
    def __init__(self, *a, **k): self._e = _RealExa(*a, **k)
    def search_and_contents(self, *a, **k):
        LEDGER.record_exa(1); return self._e.search_and_contents(*a, **k)
    def search(self, *a, **k):
        LEDGER.record_exa(1); return self._e.search(*a, **k)
exa_py.Exa = _CostExa

# ---------------- redirect artifacts + curation log --------------------
import agent.nodes.curate as curate_mod
import agent.nodes.train as train_mod
import agent.nodes.evaluate as evaluate_mod
import agent.nodes.eval_setup as eval_setup_mod
from data.curation_log import CurationLog
ART = os.path.join(RUN_DIR, "artifacts")
curate_mod.ARTIFACTS_DIR = ART
train_mod.ARTIFACTS_DIR = ART
eval_setup_mod.ARTIFACTS_DIR = ART
CUR_LOG = os.path.join(RUN_DIR, "data-curation.md")
evaluate_mod.CurationLog = lambda *a, **k: CurationLog(CUR_LOG)

# ---------------- GPU stubs (simulated rising accuracy) ----------------
SCHEDULE = [0.62, 0.74, 0.70, 0.85, 0.93, 0.97]   # one regression -> rollback, else rising
_golds = {"labels": []}
_calls = {"n": 0}

def stub_slm_train(dataset_path, base_model, nr_epochs, learning_rate,
                   batch_size, lora_rank, output_dir="artifacts", **kw):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "final_checkpoint")

def stub_infer_batch(prompts, weights_ref, base_model, max_workers=20, **kw):
    idx = _calls["n"] // 2            # 2 configs scored per evaluate round
    _calls["n"] += 1
    target = SCHEDULE[min(idx, len(SCHEDULE) - 1)]
    g = _golds["labels"]
    labels = sorted(set(g))
    other = {l: next((x for x in labels if x != l), l) for l in labels}
    n_correct = int(round(target * len(g)))
    return [lab if i < n_correct else other[lab] for i, lab in enumerate(g)]

# ---------------- node instrumentation (with per-node cost) ------------
import agent.graph as G
NODE_FNS = ["task_analysis_node", "eval_setup_node", "train_node", "evaluate_node",
            "iterate_node", "curate_node", "rollback_node", "escalate_node"]

def wrap(name, fn):
    @functools.wraps(fn)
    def inner(state):
        if name == "evaluate_node" and state.get("eval_set"):
            _golds["labels"] = [e["label"] for e in state["eval_set"].all]
        before = LEDGER.total_cost
        t = time.time()
        log(f"  ▶ {name}")
        out = fn(state)
        dt, dcost = time.time() - t, LEDGER.total_cost - before
        _report(name, out, dt)
        if dcost > 1e-9:
            log(f"      [cost] this step +${dcost:.4f}   (running total ${LEDGER.total_cost:.4f})")
        return out
    return inner

def _report(name, s, dt):
    if name == "task_analysis_node":
        m = s["selected_model"]
        log(f"      model={m.model_id} ({m.int4_size_mb}MB, {m.peak_memory_mb}MB RAM)  "
            f"stop_threshold={s['stop_threshold']}  [{dt:.1f}s]")
    elif name == "eval_setup_node":
        es = s["eval_set"]
        log(f"      acquired {len(s['train_examples'])} train; eval_set "
            f"pos={len(es.pos)}/neg={len(es.neg)}/boundary={len(es.boundary)}  [{dt:.1f}s]")
    elif name == "curate_node":
        log(f"      curriculum v{s['dataset_version']} -> {os.path.basename(s['current_dataset_path'])}  [{dt:.1f}s]")
    elif name == "train_node":
        kind = "real GPU" if REAL else "STUB"
        log(f"      iteration={s['iteration']} (2 configs, {kind} train)  [{dt:.1f}s]")
    elif name == "evaluate_node":
        ev = s["last_eval"]
        tag = "" if REAL else "[SIMULATED] "
        log(f"      {tag}F1={ev.f1:.3f}  best={s['best_score']:.3f}  "
            f"scores={[f'{x:.3f}' for x in s['scores']]}  [{dt:.1f}s]")

# ---------------- build initial state ----------------------------------
from android_pool import HardwareConstraints
from agent.graph import build_graph

_args = [a for a in sys.argv[1:] if a != "--real"]
REAL = "--real" in sys.argv          # --real uses the GPU (no train/infer stubs)
DESC = (_args[0] if _args else
        "Fine tune a small model to classify a research paper by general area of study, "
        "that should run efficiently on my moto g stylus 5g - 2023 with 6 Gb of ram and 256 gb of ROM")

log(f"=== Autonomous cold-start  (run {TS}) ===")
log(f"task: {DESC}")
log(f"mode: {'REAL GPU train/infer' if REAL else 'SIMULATED train/infer (no GPU)'}")
log("")

# --- autonomous hardware research (replaces hardcoded constraints) -----
log("  ▶ hardware_research")
from agent.hardware_research import research_device
_hw_before = LEDGER.total_cost
HW, _hw_info = research_device(DESC, log=log)
log(f"      [cost] this step +${LEDGER.total_cost - _hw_before:.4f}   (running total ${LEDGER.total_cost:.4f})")
with open(os.path.join(RUN_DIR, "device_research.json"), "w") as f:
    json.dump(_hw_info, f, indent=2)
log("")

initial_state = {
    "description": DESC, "target_metric": "F1", "hardware_constraints": HW,
    "task_type": "", "autonomous": True, "task_plan": None,
    "selected_model": None, "stop_threshold": 0.96,
    "train_examples": [], "eval_set": None, "current_dataset_path": None,
    "dataset_version": 0, "best_weights_ref": None, "best_score": 0.0,
    "iteration": 0, "scores": [], "dag": [], "consecutive_no_improvement": 0,
    "last_eval": None, "last_curation": None, "last_intervention": "", "next_action": "train",
    "messages": [], "_pending_weights_refs": None, "_pending_configs": None,
}

TARGET_INCREASES = int(os.environ.get("SLM_TARGET_INCREASES", "3"))
SAFETY_STEPS = int(os.environ.get("SLM_MAX_STEPS", "40"))   # lower for bounded GPU smoke runs
patches = [patch.object(G, fn, wrap(fn, getattr(G, fn))) for fn in NODE_FNS]
last_state = initial_state
increases = 0
prev_best = -1.0
t_start = time.time()

import contextlib
_stub_ctx = contextlib.ExitStack()
if not REAL:
    # SIMULATED mode: stub GPU train + model inference (rising accuracy schedule).
    _stub_ctx.enter_context(patch("agent.nodes.train.slm_train", side_effect=stub_slm_train))
    _stub_ctx.enter_context(patch("eval.harness.infer_batch", side_effect=stub_infer_batch))

with _stub_ctx:
    for p in patches: p.start()
    try:
        graph = build_graph()
        n = 0
        for delta in graph.stream(initial_state, stream_mode="updates",
                                   config={"recursion_limit": 200}):
            node = list(delta.keys())[0]
            last_state = list(delta.values())[0]
            n += 1
            if node == "evaluate":
                b = last_state.get("best_score", 0.0)
                if b > prev_best + 1e-9:
                    increases += 1
                    prev_best = b
                    log(f"      ✅ accuracy increase #{increases}  (best={b:.3f})")
                    if increases >= TARGET_INCREASES:
                        log(f"\n  🎯 accuracy rose {TARGET_INCREASES} times — proof of concept met. Stopping.")
                        break
            if n >= SAFETY_STEPS:
                log(f"\n  ⛔ safety cap ({SAFETY_STEPS} steps) reached — stopping.")
                break
    except Exception as e:
        import traceback
        log(f"\n  !! stopped on exception: {type(e).__name__}: {e}")
        log(traceback.format_exc())
    finally:
        for p in patches: p.stop()

elapsed = time.time() - t_start
# ---------------- dump artifacts + cost --------------------------------
cost = LEDGER.snapshot()
with open(os.path.join(RUN_DIR, "dag.json"), "w") as f:
    json.dump(last_state.get("dag", []), f, indent=2)
with open(os.path.join(RUN_DIR, "scores.json"), "w") as f:
    json.dump({"scores": last_state.get("scores", []),
               "best_score": last_state.get("best_score", 0.0),
               "iterations": last_state.get("iteration", 0),
               "accuracy_increases": increases}, f, indent=2)
with open(os.path.join(RUN_DIR, "cost.json"), "w") as f:
    json.dump(cost, f, indent=2)
if last_state.get("task_plan"):
    with open(os.path.join(RUN_DIR, "task_plan.json"), "w") as f:
        json.dump(last_state["task_plan"], f, indent=2)

log("")
log(f"=== done in {elapsed:.1f}s ===")
log(f"task_type={last_state.get('task_type')}  model={last_state['selected_model'].model_id if last_state.get('selected_model') else None}")
log(f"accuracy increases: {increases}   score trajectory: {[f'{x:.3f}' for x in last_state.get('scores', [])]}")
log(f"API COST: Claude {cost['anthropic_calls']} calls "
    f"({cost['input_tokens']} in / {cost['output_tokens']} out tok) = ${cost['anthropic_cost_usd']:.4f}  |  "
    f"Exa {cost['exa_calls']} searches = ${cost['exa_cost_usd']:.4f}  |  TOTAL ${cost['total_cost_usd']:.4f}")
log(f"logs: {RUN_DIR}")
sys.stdout = sys.__stdout__   # restore before closing the log file
_LOGF.close()
