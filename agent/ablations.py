# agent/ablations.py
"""Operator-selected ablation switches. Every one of them is OFF unless its env var is set,
and with none set this module changes nothing about how a run behaves.

These exist to answer questions the normal pipeline cannot be asked, because the normal pipeline
has already made the choice being questioned:

  SLM_ABLATION_RESET_DATA_ON_ESCALATION
      Does REUSING the curriculum across model tiers actually buy anything? `escalate_node`
      deliberately carries the dataset forward — a new tier's first iteration retrains the
      curriculum built by every tier below it. That is the right default (it answers "what is a
      bigger model worth on the data we already have?"), but it also means the top tier trains on
      data that cost real teacher tokens to produce at every tier beneath it. If resetting to the
      seed each time costs nothing in final accuracy, the synthesis spend at the lower tiers was
      wasted; if it costs a lot, data reuse is the cheapest intervention in the suite. Setting this
      flag rewinds the curriculum to `dataset_v1.jsonl` on every promotion.

  SLM_SYNTH_DISALLOW
      Does synthetic data help at all? `synthesis_allowed` can only be forced ON (by
      SLM_TEACHER_SYNTH_BYPASS); nothing could force it off, because the fitness gate is supposed
      to be the only thing that decides. This flag forces it off regardless of what the teacher
      measured, leaving mining and hyperparameters as the run's whole action space.

WHY THE FLAGS ARE READ PER CALL rather than captured in module constants at import: the constant
form (`agent.teacher_fitness.BYPASS`) cannot be exercised by a test without reloading the module,
and an ablation whose behaviour is hard to test is an ablation whose results are hard to trust.
The reads are once per escalation and a few times per iteration, so there is nothing to save.
Both flags are also recorded in the resume fingerprint (`agent.checkpoint`), so a requeue cannot
silently continue a run under different settings than it started with.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from agent.checkpoint import atomic_write_jsonl

# Same relative directory `curate_node` and `eval_setup` write to; the run's CWD is its run dir.
ARTIFACTS_DIR = "artifacts"
SEED_TRAIN_EXAMPLES_FILE = "seed_train_examples.jsonl"

# Named in `data_source_usage` and the curriculum-growth ledger so a reset shows up as an event in
# the provenance trail rather than as an unexplained drop in the curriculum size.
RESET_STRATEGY = "ablation_reset_to_seed"

_TRUE = {"1", "true", "yes", "on"}


class AblationStateError(RuntimeError):
    """An ablation was requested but the state it needs is missing.

    Raised rather than warned. A reset ablation that cannot find its seed snapshot would carry the
    curriculum forward instead — i.e. silently run the BASELINE under an ablation job name, which
    is the one outcome that makes the experiment worse than not running it.
    """


def _flag(name: str) -> bool:
    return str(os.environ.get(name, "0")).strip().lower() in _TRUE


def reset_data_on_escalation() -> bool:
    """Rewind the curriculum to the seed on every tier promotion (ablation 1)."""
    return _flag("SLM_ABLATION_RESET_DATA_ON_ESCALATION")


def synthesis_disallowed() -> bool:
    """Refuse synthetic data for the whole run, whatever the teacher measured (ablation 3)."""
    return _flag("SLM_SYNTH_DISALLOW")


def mining_disallowed() -> bool:
    """Refuse `mine_new_real` for the whole run, however much corpus is left (ablation 4)."""
    return _flag("SLM_ABLATION_DISALLOW_MINING")


def train_cap_override() -> int | None:
    """Replace `TaskSpec.initial_train_cap` with a fixed row count, or None to leave it alone.

    ABLATION 4 NEEDS BOTH THIS AND `mining_disallowed`, and neither existed. The question is whether
    synthetic data earns its keep when real data is SCARCE — the suite's runs all had thousands of
    real rows, and synthesis contributed between +0.008 and +0.050 on four of the five 09-2026
    tasks. Starving a task to 100 real rows and holding everything else fixed is the way to find
    out whether abundance was the reason.

    Capping alone is not enough: `mine_new_real` reads deeper into the SAME corpus, so a run capped
    at 100 rows would simply mine its way back to thousands and measure nothing. The two flags are
    separate rather than one combined switch because they answer to different parts of the loop —
    the cap is a loader concern and the refusal is an intervention concern — and because an arm
    that caps without disabling mining is a legitimate third condition someone may want later.

    Returns None rather than a sentinel int so a caller can distinguish "not set" from "set to 0",
    and rejects a non-positive or unparseable value loudly: a typo that silently trained on the
    full corpus would produce a baseline wearing an ablation's job name, which is the exact failure
    this module's docstring warns about.
    """
    raw = str(os.environ.get("SLM_ABLATION_TRAIN_CAP", "")).strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"SLM_ABLATION_TRAIN_CAP={raw!r} is not an integer; refusing to fall back to the "
            f"task's own cap, because that would run the baseline under an ablation's name"
        ) from error
    if value <= 0:
        raise ValueError(f"SLM_ABLATION_TRAIN_CAP={value} must be positive")
    return value


def active_ablations() -> list[str]:
    """Human-readable list of what is switched on, for the run header and the final report."""
    active = []
    if reset_data_on_escalation():
        active.append(
            "SLM_ABLATION_RESET_DATA_ON_ESCALATION=1 — the curriculum is rewound to the seed "
            "(dataset_v1) on every tier promotion, so no tier inherits another tier's mined or "
            "synthesized rows"
        )
    if synthesis_disallowed():
        active.append(
            "SLM_SYNTH_DISALLOW=1 — surgical_synthesis is refused for the whole run regardless of "
            "the teacher's fitness score; only mine_new_real and hyperparameter interventions run"
        )
    if mining_disallowed():
        active.append(
            "SLM_ABLATION_DISALLOW_MINING=1 — mine_new_real is refused for the whole run, so the "
            "curriculum cannot grow with real rows however much of the corpus is left unread"
        )
    cap = train_cap_override()
    if cap is not None:
        active.append(
            f"SLM_ABLATION_TRAIN_CAP={cap} — the initial gold load is capped at {cap} row(s) "
            f"instead of the task's own initial_train_cap"
        )
    return active


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def capture_seed_snapshot(state, curriculum_path: str, *, log=print) -> None:
    """Record what "the seed" means, once, at the initial_gold build.

    Called from `curate_node` immediately after the first `dataset_v1.jsonl` is written. Two
    different things have to be kept, because they are not the same rows:

      * The CURRICULUM (`dataset_v1.jsonl`) — gold rows after dedup and the eval firewall. This is
        what gets trained on, and what a reset restores. Already on disk; only its path is stored.
      * The TRAIN POOL (`state["train_examples"]`) — the loader's full gold set, which is larger
        (BC5CDR: 5,000 pool rows behind a 2,603-row curriculum). Mining MERGES into this list, so
        by tier 3 it is no longer gold-only and cannot be reconstructed from state. It is the
        anchor pool few-shot synthesis demonstrations are drawn from, so restoring it matters for
        an ablation whose whole claim is that no tier inherits another tier's rows.

    `source_progress` and `last_curation` are copied rather than recomputed. Both are small, and
    re-deriving them would mean re-deciding at reset time what curate already worked out at cold
    start — what `asked_for` and `exhausted` were, and how the seed curriculum breaks down by
    provenance, source and label. Keeping the seed's own composition record is also what lets a
    reset leave an HONEST `last_curation` behind instead of a null one, which the orchestrator
    prompt and the curation log would both render as a zero-row curriculum.

    No-ops when the flag is off (nothing will ever read the snapshot) and when a snapshot already
    exists, so a requeue resuming past the first curate does not overwrite the seed with whatever
    the curriculum has since become.
    """
    if not reset_data_on_escalation():
        return
    if state.get("seed_dataset_path"):
        return

    gold_pool = list(state.get("train_examples") or [])
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    train_path = os.path.join(ARTIFACTS_DIR, SEED_TRAIN_EXAMPLES_FILE)
    atomic_write_jsonl(train_path, gold_pool)

    state["seed_dataset_path"] = curriculum_path
    state["seed_train_examples_path"] = train_path
    state["seed_source_progress"] = deepcopy(state.get("source_progress") or {})
    state["seed_last_curation"] = deepcopy(state.get("last_curation"))
    log(
        f"  [ablation] seed snapshot captured: curriculum={curriculum_path}, "
        f"train pool={train_path} ({len(gold_pool)} gold row(s)). Every tier promotion will "
        f"rewind to exactly this."
    )


def reset_curriculum_to_seed(state, *, log=print) -> dict[str, Any]:
    """Rewind the curriculum, the train pool and all data bookkeeping to the seed.

    THE VERSION COUNTER STILL GOES UP. The seed rows are re-published as a NEW
    `dataset_v{N+1}.jsonl` rather than by pointing `current_dataset_path` back at
    `dataset_v1.jsonl`. Rewinding the counter instead would make the next rebuild write
    `dataset_v2.jsonl` a second time, overwriting tier 1's artifact and destroying the only
    on-disk record of what tier 1 trained on — and the reset itself would leave no trace at all.
    This way each reset IS a dataset version, and the artifacts read as the run's actual history.

    Everything that survives a normal escalation and would leak one tier's data work into the next
    is reverted here:

      train_examples             mined rows were merged into the gold pool
      source_progress            how deep each source has been read
      surgical_category_history  which failure categories the teacher has already targeted
      failed_discovery_rounds    web-research attempts already spent
      mining_retired_reason      a route closed by run health on evidence that no longer applies
      run health empty counters  consecutive-empty tallies for routes that now have work to do
      last_curation              the composition of a curriculum that no longer exists

    `run_health["history"]` and `load_failures` are deliberately KEPT: the first is the run's audit
    trail and clearing it would erase the record of the resets themselves, and the second tracks an
    environment fault that a curriculum rewind does not fix.
    """
    seed_dataset = state.get("seed_dataset_path")
    seed_train = state.get("seed_train_examples_path")
    if not seed_dataset or not os.path.exists(str(seed_dataset)):
        raise AblationStateError(
            "SLM_ABLATION_RESET_DATA_ON_ESCALATION=1 but no seed curriculum snapshot exists "
            f"(seed_dataset_path={seed_dataset!r}). The snapshot is taken by curate_node at the "
            "initial_gold build, so this means the flag was set on a resume of a run that started "
            "without it. Refusing to carry the dataset forward, because that would run the "
            "baseline under an ablation's name."
        )
    if not seed_train or not os.path.exists(str(seed_train)):
        raise AblationStateError(
            "the seed curriculum snapshot exists but the seed train pool does not "
            f"(seed_train_examples_path={seed_train!r}); the two are written together, so one "
            "without the other means the artifacts directory has been altered mid-run."
        )

    previous_path = state.get("current_dataset_path")
    previous_version = int(state.get("dataset_version", 0) or 0)
    previous_curriculum_rows = int(
        (state.get("last_curation") or {}).get("total_examples", 0) or 0
    )

    seed_rows = _read_jsonl(str(seed_dataset))
    next_version = previous_version + 1
    for row in seed_rows:
        row["_dataset_version"] = next_version
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, f"dataset_v{next_version}.jsonl")
    atomic_write_jsonl(path, seed_rows)

    state["dataset_version"] = next_version
    state["current_dataset_path"] = path
    state["train_examples"] = _read_jsonl(str(seed_train))
    state["source_progress"] = deepcopy(state.get("seed_source_progress") or {})
    state["surgical_category_history"] = {}
    state["failed_discovery_rounds"] = 0
    state["data_rebuild_plan"] = None
    state["data_rebuild_plan_identity"] = None

    # The seed's OWN composition record, with the three fields that describe the transition rather
    # than the data overwritten. Clearing this to None instead would be read as a zero-row
    # curriculum by both the orchestrator prompt (`iterate` prints
    # "curriculum=<previous_rows>-><total_examples>") and the curation log, for every iteration
    # until the tier's first rebuild rewrote it.
    composition = deepcopy(state.get("seed_last_curation")) or {}
    if composition:
        composition["strategy"] = RESET_STRATEGY
        composition["previous_rows"] = previous_curriculum_rows
        composition["rows_added"] = 0
        composition["novel_rows"] = 0
        composition["source_progress"] = deepcopy(state["source_progress"])
        composition["failed_discovery_rounds"] = 0
        composition["mining_report"] = {}
        composition["data_rebuild_plan"] = None
        composition["target_categories"] = []
    state["last_curation"] = composition or None

    from agent.run_health import MINING_RETIRED_KEY, RunHealth

    # Assigned, not popped: these are LangGraph state channels, and a deletion does not always
    # propagate the way an assignment does. Every reader tests truthiness.
    retired = state.get(MINING_RETIRED_KEY)
    state[MINING_RETIRED_KEY] = None

    health = RunHealth.from_state(state)
    health.empty_rebuilds = 0
    health.empty_mining = 0
    health.empty_synthesis = 0
    health.verify_wipeouts = 0
    health.mining_shutouts = 0
    health.to_state(state)

    state["data_source_usage"] = list(state.get("data_source_usage") or []) + [{
        "iteration": int(state.get("iteration", 0) or 0),
        "dataset_version": f"v{next_version}",
        "strategy": RESET_STRATEGY,
        # Empty on purpose. The seed rows' sources are already accounted for by the initial_gold
        # entry, and re-listing them here would add their row counts to the run's per-source totals
        # once per reset — reporting BC5CDR as having contributed four times the rows it holds.
        "sources": [],
    }]

    summary = {
        "previous_path": previous_path,
        "previous_version": previous_version,
        "previous_rows": previous_curriculum_rows,
        "path": path,
        "version": next_version,
        "rows": len(seed_rows),
        "train_pool_rows": len(state["train_examples"]),
        "mining_unretired": bool(retired),
    }
    log(
        f"  Dataset RESET to seed (ablation): v{previous_version} → v{next_version}, "
        f"{previous_curriculum_rows} → {len(seed_rows)} row(s) "
        f"({summary['train_pool_rows']} gold pool row(s))"
    )
    log(f"  [ablation]   was: {previous_path}")
    log(f"  [ablation]   now: {path} (seed re-published, {seed_dataset} left untouched)")
    log(
        "  [ablation] also reverted: train pool (mined rows dropped), source_progress, "
        "surgical category history, failed discovery rounds and the run-health empty counters. "
        "This tier starts from the same curriculum tier 1 started from."
    )
    if retired:
        log(f"  [ablation] mine_new_real UN-RETIRED (was: {retired}) — the seed's sources have "
            f"unread rows again, so the evidence that closed the route no longer applies.")
    return summary
