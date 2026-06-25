#!/usr/bin/env python
"""
Test case: autonomous cold-start run for the user prompt

  "Fine tune a small model to classify a research paper by general area of study,
   that should run efficiently on my moto g stylus 5g - 2023 with 6 Gb of ram and
   256 gb of ROM"

Pioneer Agent cold-start (arXiv:2604.09791v1). REAL steps 1-3:
  1 task_analysis  : classify task, select smallest Android-pool model under device limits
  2 eval_setup     : autonomously acquire paper abstracts via Exa, build held-out eval set
  3 curate         : build curriculum + hard negatives via Claude Sonnet (real API)
STUBBED steps 4-5 (no GPU): train / evaluate use placeholders.

Hard cap: 5 LangGraph super-steps, so a crash or loop cannot keep spending credits.
Everything is logged under logs/runs/<timestamp>/.
"""
import os, sys, json, time, datetime, functools
from unittest.mock import patch

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJ, ".env"))

# ---------------- run directory + logger -------------------------------
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(PROJ, "logs", "runs", TS)
os.makedirs(os.path.join(RUN_DIR, "artifacts"), exist_ok=True)
_LOGF = open(os.path.join(RUN_DIR, "run.log"), "w", buffering=1)

def log(msg=""):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    _LOGF.write(line + "\n")

# ---------------- redirect artifacts + curation log into RUN_DIR -------
import agent.nodes.curate as curate_mod
import agent.nodes.train as train_mod
import agent.nodes.evaluate as evaluate_mod
from data.curation_log import CurationLog
curate_mod.ARTIFACTS_DIR = os.path.join(RUN_DIR, "artifacts")
train_mod.ARTIFACTS_DIR = os.path.join(RUN_DIR, "artifacts")
CUR_LOG = os.path.join(RUN_DIR, "data-curation.md")
evaluate_mod.CurationLog = lambda *a, **k: CurationLog(CUR_LOG)

# ---------------- GPU stubs (steps 4-5) --------------------------------
_golds = {"labels": []}

def stub_slm_train(dataset_path, base_model, nr_epochs, learning_rate,
                   batch_size, lora_rank, output_dir="artifacts", **kw):
    os.makedirs(output_dir, exist_ok=True)
    log(f"      [STUB train] base={base_model} epochs={nr_epochs} lr={learning_rate} "
        f"lora_rank={lora_rank} -> would run Unsloth on a GPU here")
    return os.path.join(output_dir, "final_checkpoint")

def stub_infer_batch(prompts, weights_ref, base_model, max_workers=20, **kw):
    # SIMULATED predictions (no trained model): echo gold for half the eval set so
    # the real scorer/harness executes end-to-end. NOT a real accuracy number.
    g = _golds["labels"]
    labels = sorted(set(g))
    other = {l: next((x for x in labels if x != l), l) for l in labels}
    return [lab if i % 2 == 0 else other[lab] for i, lab in enumerate(g)]

# ---------------- node instrumentation ---------------------------------
import agent.graph as G
NODE_FNS = ["task_analysis_node", "eval_setup_node", "train_node", "evaluate_node",
            "iterate_node", "curate_node", "rollback_node", "escalate_node"]
_t0 = {}

def wrap(name, fn):
    @functools.wraps(fn)
    def inner(state):
        if name == "evaluate_node" and state.get("eval_set"):
            _golds["labels"] = [e["label"] for e in state["eval_set"].all]
        start = time.time()
        log(f"  ▶ {name} ...")
        out = fn(state)
        _report(name, out, time.time() - start)
        return out
    return inner

def _report(name, s, dt):
    if name == "task_analysis_node":
        m = s["selected_model"]
        log(f"      selected model: {m.model_id}  ({m.int4_size_mb}MB INT4, tier{m.tier}, "
            f"~{m.peak_memory_mb}MB RAM)  stop_threshold={s['stop_threshold']}  [{dt:.1f}s]")
    elif name == "eval_setup_node":
        es = s["eval_set"]
        areas = sorted(set(e["label"] for e in s["train_examples"]))
        log(f"      acquired {len(s['train_examples'])} train + eval_set "
            f"pos={len(es.pos)}/neg={len(es.neg)}/boundary={len(es.boundary)}  "
            f"areas={areas}  [{dt:.1f}s]")
    elif name == "curate_node":
        log(f"      curriculum -> {s['current_dataset_path']} (v{s['dataset_version']})  [{dt:.1f}s]")
    elif name == "train_node":
        log(f"      iteration={s['iteration']}  configs={list(s.get('_pending_configs',{}).keys())}  [{dt:.1f}s]")
    elif name == "evaluate_node":
        ev = s["last_eval"]
        log(f"      [SIMULATED] F1={ev.f1:.3f}  failures={len(ev.failures)}  "
            f"scores={[f'{x:.3f}' for x in s['scores']]}  [{dt:.1f}s]")

# ---------------- build initial state ----------------------------------
from android_pool import HardwareConstraints
from agent.graph import build_graph
from run import _infer_task_type

DESC = ("Fine tune a small model to classify a research paper by general area of study, "
        "that should run efficiently on my moto g stylus 5g - 2023 with 6 Gb of ram and 256 gb of ROM")

# Moto G Stylus 5G (2023): Snapdragon 6 Gen 1, 6GB RAM, 256GB ROM.
# 256GB storage is ample; ~3.5GB RAM usable after OS; chip ~= snapdragon_778g proxy.
HW = HardwareConstraints(storage_mb=4000, memory_mb=3500, latency_ttft_ms=2000,
                         power_watts=5.0, target_chip="snapdragon_778g")

task_type = _infer_task_type(DESC)
log(f"=== Research-paper cold-start test  (run {TS}) ===")
log(f"prompt: {DESC}")
log(f"autonomous task_type classification -> {task_type}")
log(f"device limits: storage={HW.storage_mb}MB memory={HW.memory_mb}MB chip={HW.target_chip}")
log("")

initial_state = {
    "description": DESC, "target_metric": "F1", "hardware_constraints": HW,
    "task_type": task_type, "selected_model": None, "stop_threshold": 0.96,
    "train_examples": [], "eval_set": None, "current_dataset_path": None,
    "dataset_version": 0, "best_weights_ref": None, "best_score": 0.0,
    "iteration": 0, "scores": [], "dag": [], "consecutive_no_improvement": 0,
    "last_eval": None, "last_intervention": "", "next_action": "train",
    "messages": [], "_pending_weights_refs": None, "_pending_configs": None,
}

MAX_STEPS = 5
patches = [patch.object(G, fn, wrap(fn, getattr(G, fn))) for fn in NODE_FNS]
last_state = initial_state
t_start = time.time()

with patch("agent.nodes.train.slm_train", side_effect=stub_slm_train), \
     patch("eval.harness.infer_batch", side_effect=stub_infer_batch):
    for p in patches: p.start()
    try:
        graph = build_graph()
        n = 0
        for delta in graph.stream(initial_state, stream_mode="updates",
                                   config={"recursion_limit": 50}):
            node = list(delta.keys())[0]
            last_state = list(delta.values())[0]
            n += 1
            if n >= MAX_STEPS:
                log(f"\n  ⛔ hit {MAX_STEPS}-step cap after '{node}' — stopping (credit guard).")
                break
    except Exception as e:
        log(f"\n  !! run stopped on exception: {type(e).__name__}: {e}")
    finally:
        for p in patches: p.stop()

elapsed = time.time() - t_start
# ---------------- dump structured artifacts ----------------------------
with open(os.path.join(RUN_DIR, "dag.json"), "w") as f:
    json.dump(last_state.get("dag", []), f, indent=2)
with open(os.path.join(RUN_DIR, "scores.json"), "w") as f:
    json.dump({"scores": last_state.get("scores", []),
               "best_score": last_state.get("best_score", 0.0),
               "iteration": last_state.get("iteration", 0)}, f, indent=2)

log("")
log(f"=== done in {elapsed:.1f}s ===")
log(f"steps executed: {n}/{MAX_STEPS}")
log(f"model selected: {last_state['selected_model'].model_id if last_state.get('selected_model') else None}")
log(f"logs + artifacts: {RUN_DIR}")
for fn in ["run.log", "data-curation.md", "dag.json", "scores.json"]:
    p = os.path.join(RUN_DIR, fn)
    log(f"   {fn:20} {'%d bytes' % os.path.getsize(p) if os.path.exists(p) else '(not written)'}")
_LOGF.close()
