"""Revive the terminal size-sweep thread so it can continue through variants 19-27.

WHY THIS IS NEEDED. Run 40827150 did not crash and did not exhaust its model pool: `_must_terminate`
fired at `_graph_steps >= MAX_TURNS_MAIN` (1500), so `_route_after_evaluate` returned "terminate"
and the graph routed to END after 18 of 27 variants. The thread is therefore TERMINAL
(`next_nodes == ()`), and LangGraph resumes an existing thread by passing `None`, which on a
finished thread is a no-op. Raising the budget alone does not restart it.

WHAT THIS DOES. With SLM_MAX_TURNS_MAIN raised, `_must_terminate` is False again, so re-anchoring
the thread at its last real node (`evaluate`) makes LangGraph recompute that node's conditional
edge and schedule the successor it would have taken had the budget not run out. Nothing in the
run's history is rewritten: `update_state` APPENDS a checkpoint.

IT ALSO RE-STAMPS THE TWO FINGERPRINT FILES, deliberately. `MAX_TURNS_MAIN` is part of
`runtime_config_snapshot`, so both `run-manifest.json` and `checkpoint.json` carry a
`config_fingerprint` computed under 1500 and a resume under 6000 is refused by design. That guard
exists so "budget exhausted" cannot be made unfalsifiable after the fact — so the change is
recorded in `budget_raise_history` on the manifest rather than quietly applied.
"""
import argparse, json, os, shutil, sys
from datetime import datetime, timezone

RUN = "logs/runs/slm-ner-bc5cdr-ablation-sizesweep-l40s-40827150"
os.environ.setdefault("ANTHROPIC_API_KEY", "revive")
os.environ.setdefault("EXA_API_KEY", "revive")
os.environ.setdefault("SLM_BENCHMARK_TASK", "ner_bc5cdr")
os.environ.setdefault("SLM_MODEL_SELECTION_STRATEGY", "size_sweep")
os.environ.setdefault("SLM_STOP_THRESHOLD", "0.99")
os.environ.setdefault("SLM_VERIFY_SYNTH", "0")
os.environ.setdefault("SLM_TEACHER_SYNTH_BYPASS", "1")
os.environ.setdefault("SLM_MAX_WALLCLOCK_S", "0")
os.environ.pop("SLM_SYNTH_API_MODE", None)

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true", help="without this, changes nothing")
ap.add_argument("--budget", default=os.environ.get("SLM_MAX_TURNS_MAIN", "6000"))
args = ap.parse_args()
os.environ["SLM_MAX_TURNS_MAIN"] = str(args.budget)

import config.config as config
from agent.checkpoint import (checkpoint_compatibility, inspect_sqlite_state,
                              runtime_config_snapshot, sqlite_checkpointer)
from agent.graph import build_graph, graph_run_config

MANIFEST, CKPT = f"{RUN}/run-manifest.json", f"{RUN}/checkpoint.json"
man = json.load(open(MANIFEST)); ck = json.load(open(CKPT))

# The new fingerprint is derived from the STORED effective config with only MAX_TURNS_MAIN
# changed, NOT from this process's environment. This script does not run under the launcher, so
# recomputing live would fold in a dozen unrelated env differences (CUDA isolation, the curation
# log path, the MNN knobs) and silently re-stamp the run as something it is not. Deriving keeps
# the delta provably one key wide; the live check in run.py, which DOES run under the launcher,
# is what verifies the derived value against the real environment at resume time.
from agent.checkpoint import _canonical_hash

stored_eff = dict(man["effective_config"])
new_eff = {**stored_eff, "MAX_TURNS_MAIN": config.MAX_TURNS_MAIN}
new_compat = dict(man["compatibility"])
new_compat["config_fingerprint"] = _canonical_hash(new_eff)

print(f"MAX_TURNS_MAIN      : {stored_eff.get('MAX_TURNS_MAIN')} -> {config.MAX_TURNS_MAIN}")
print(f"config_fingerprint  : {man['compatibility']['config_fingerprint'][:16]} -> {new_compat['config_fingerprint'][:16]}")
delta = [k for k in set(stored_eff) | set(new_eff) if stored_eff.get(k) != new_eff.get(k)]
print(f"effective config delta: {delta}")
assert delta == ["MAX_TURNS_MAIN"], f"refusing: delta is wider than the budget: {delta}"

thread = ck["thread_id"]
with sqlite_checkpointer(f"{RUN}/langgraph.sqlite") as saver:
    graph = build_graph(mode="cold_start", checkpointer=saver)
    cfg = graph_run_config(thread, recursion_limit=10)
    before = inspect_sqlite_state(graph, cfg)
    print(f"\nbefore: step={before.step} last={before.last_node} next={before.next_nodes} terminal={before.terminal}")
    if not before.terminal:
        sys.exit("thread is already non-terminal; nothing to revive")
    if not args.apply:
        print("\nDRY RUN — re-run with --apply to revive"); sys.exit(0)

    for path in (MANIFEST, CKPT):
        shutil.copy2(path, path + ".pre-revive.bak")
    print(f"backed up manifest + checkpoint (.pre-revive.bak)")

    graph.update_state(cfg, {}, as_node="evaluate")
    after = inspect_sqlite_state(graph, cfg)
    print(f"after : step={after.step} last={after.last_node} next={after.next_nodes} terminal={after.terminal}")
    if after.terminal:
        sys.exit("revival did not schedule a successor node — thread still terminal")

    man["compatibility"] = new_compat
    man["effective_config"] = new_eff
    man.setdefault("budget_raise_history", []).append({
        "at": datetime.now(timezone.utc).isoformat(),
        "from": 1500, "to": config.MAX_TURNS_MAIN,
        "reason": "run routed to END at the step cap after 18 of 27 variants; "
                  "continuing the sweep through the remaining 9",
        "revived_as_node": "evaluate",
        "steps_at_revival": before.step,
    })
    json.dump(man, open(MANIFEST, "w"), indent=2)
    ck["compatibility"] = new_compat
    ck["progress"]["status"] = "running"
    ck["progress"]["next_nodes"] = list(after.next_nodes)
    json.dump(ck, open(CKPT, "w"))
    print("re-stamped run-manifest.json and checkpoint.json")
