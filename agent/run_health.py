"""Stop a run that is burning GPU hours without learning anything.

WHY THIS EXISTS
    xlam run 38566712 ran for 7h42m and $1.34 while every one of its eight data rebuilds added zero
    rows. Nothing stopped it, because every individual symptom looked survivable: a `0 novel` line,
    a quality-control drop, a mining round that found nothing. Each of those IS survivable once. The
    failure was that they repeated, and no component was watching across iterations.

    This module watches. It records what each iteration actually did to the curriculum and raises
    `RunHealthError` when a pattern appears that means the run cannot make progress. The point is to
    fail in minutes rather than hours, and to fail with the diagnosis attached — the checks are
    written so the message names the mechanism, not just the symptom.

WHAT IT DELIBERATELY DOES NOT DO
    It does not stop a run for one bad iteration. Every threshold below requires either a repeat or a
    magnitude that cannot be explained by ordinary variance, because a guard that fires on noise gets
    switched off, and a switched-off guard is worse than none.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from agent.state import SKIPPED_NO_ROWS


class RunHealthError(RuntimeError):
    """The run cannot make progress and should stop now rather than spend more GPU time."""


# --- thresholds -------------------------------------------------------------------------------
# An empty rebuild is treated differently depending on WHICH route came back empty, because the two
# routes fail for different reasons and only one of them is fatal.
#
# MINING returning nothing is an answer, not a malfunction: it means the sources this run knows about
# have no more rows carrying this task's labels. Re-asking cannot change that, so after two empties
# `mine_new_real` is RETIRED for the rest of the run and the run CONTINUES on the routes that remain.
# Killing the run here is what run 38735780 and run 38832588 did, and in both cases synthesis and
# hyperparameter tuning were still perfectly able to make progress.
MAX_EMPTY_MINING_ROUNDS = int(os.environ.get("SLM_MAX_EMPTY_MINING", "2"))

# SYNTHESIS returning nothing is a malfunction: the teacher is up, a category was targeted, and still
# no row survived. That is a contract disagreement between generator and verifier which will repeat
# identically next turn, and it is the one shape worth stopping for. Three rather than two so that a
# single bad batch plus one unlucky retry does not end a run.
MAX_CONSECUTIVE_EMPTY_SYNTHESIS = int(os.environ.get("SLM_MAX_EMPTY_SYNTHESIS", "3"))

# Any OTHER rebuild strategy — there are none today, but a future one would otherwise be unguarded.
MAX_CONSECUTIVE_EMPTY_REBUILDS = int(os.environ.get("SLM_MAX_EMPTY_REBUILDS", "4"))

# The curriculum SHRINKING. This is the specific thing worth alarming on, and it is deliberately not
# "quality control removed a lot of rows": QC removing 400 of 404 because a mining round returned 400
# near-duplicates is a mining-novelty problem, already covered by the shutout and empty-rebuild
# counters. QC eating rows that were ALREADY in the curriculum is different — those rows trained a
# model successfully on a previous iteration, so a step now rejecting them is rejecting the task's own
# data, and the curriculum is supposed to be cumulative.
#
# Expressed both ways because either alone has a blind spot: a fraction misses a large drop from a
# large curriculum, and an absolute count misses a catastrophic drop from a small one.
CURRICULUM_SHRINK_ROW_ALARM = int(os.environ.get("SLM_QC_DROP_ROW_ALARM", "1000"))
CURRICULUM_SHRINK_FRACTION_ALARM = float(os.environ.get("SLM_QC_DROP_FRACTION_ALARM", "0.25"))

# Consecutive iterations in which every generated row was rejected by verification. Once is a bad
# batch; twice means the verifier and the generator disagree systematically, and the teacher budget
# is being spent to produce nothing.
MAX_CONSECUTIVE_TOTAL_VERIFY_REJECTIONS = int(
    os.environ.get("SLM_MAX_VERIFY_WIPEOUTS", "2")
)

# Mining attempts that found candidate rows but accepted none of them. Distinct from "no candidates
# found", which is an honest empty search: this is the shape where a filter rejects everything, which
# is what the pre-2026-08-19 whole-source overlap check did on every single candidate.
MAX_MINING_ATTEMPTS_WITHOUT_ACCEPTANCE = int(
    os.environ.get("SLM_MAX_MINING_SHUTOUTS", "3")
)

# Loader failures. A loader is deterministic against a cached corpus, so a repeat failure is an
# environment or data problem that will not resolve by retrying for another six hours.
MAX_LOAD_FAILURES = int(os.environ.get("SLM_MAX_LOAD_FAILURES", "2"))

# State key set when mining is retired. Read by `data_rebuild.mining_available_for_state`, so the
# retirement is honored by the PLANNER rather than only reported here: once it is set, `mine_new_real`
# is no longer an option the orchestrator can pick, and when synthesis is also closed
# `data_rebuild_available` goes false and `iterate` tunes hyperparameters every turn instead.
MINING_RETIRED_KEY = "mining_retired_reason"


@dataclass
class IterationRecord:
    """What one curate pass did to the curriculum."""

    iteration: int
    strategy: str
    rows_added: int
    curriculum_before: int
    curriculum_after: int
    qc_removed: int
    firewall_removed: int
    # Rows quality control actually LOOKED at: the previous curriculum plus whatever this rebuild
    # added. Recorded and reported, but deliberately NOT the alarm's denominator — a mining round that
    # returns 400 near-duplicates and loses all 400 to dedup is 99% of what QC examined and nothing is
    # wrong with QC. What the alarm measures is the curriculum SHRINKING; see
    # CURRICULUM_SHRINK_ROW_ALARM.
    qc_examined: int = 0
    candidates_seen: int = 0
    verify_attempted: int = 0
    verify_kept: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class RunHealth:
    """Cross-iteration health state. Lives on the run state so it survives a resume."""

    empty_rebuilds: int = 0
    # Not reset by a later success, unlike the others: these two count how many times a ROUTE came
    # back empty, and the second empty retires the route permanently. A cumulative count is the right
    # shape for a permanent decision.
    empty_mining: int = 0
    empty_synthesis: int = 0
    verify_wipeouts: int = 0
    mining_shutouts: int = 0
    load_failures: int = 0
    history: list[dict] = field(default_factory=list)

    @classmethod
    def from_state(cls, state) -> "RunHealth":
        raw = state.get("run_health")
        if not isinstance(raw, dict):
            return cls()
        return cls(
            empty_rebuilds=int(raw.get("empty_rebuilds", 0) or 0),
            empty_mining=int(raw.get("empty_mining", 0) or 0),
            empty_synthesis=int(raw.get("empty_synthesis", 0) or 0),
            verify_wipeouts=int(raw.get("verify_wipeouts", 0) or 0),
            mining_shutouts=int(raw.get("mining_shutouts", 0) or 0),
            load_failures=int(raw.get("load_failures", 0) or 0),
            history=list(raw.get("history") or []),
        )

    def to_state(self, state) -> None:
        # Assigned as a NEW dict, never mutated in place, so it persists to the LangGraph channel
        # (the same trap that froze `scores` at three entries and made stagnation never fire, B122).
        state["run_health"] = {
            "empty_rebuilds": self.empty_rebuilds,
            "empty_mining": self.empty_mining,
            "empty_synthesis": self.empty_synthesis,
            "verify_wipeouts": self.verify_wipeouts,
            "mining_shutouts": self.mining_shutouts,
            "load_failures": self.load_failures,
            # Bounded: the history is for the final report, not an audit log, and an unbounded list
            # crosses the checkpoint boundary on every write.
            "history": self.history[-40:],
        }


def record_load_failure(state, *, what: str, error: BaseException, log=print) -> None:
    """Note a data-load failure and stop the run on a repeat."""
    health = RunHealth.from_state(state)
    health.load_failures += 1
    health.to_state(state)
    log(f"  ✗ LOAD FAILURE ({health.load_failures}/{MAX_LOAD_FAILURES}) loading {what}: "
        f"{type(error).__name__}: {error}")
    if health.load_failures >= MAX_LOAD_FAILURES:
        raise RunHealthError(
            f"data loading failed {health.load_failures} times (most recently {what}: "
            f"{type(error).__name__}: {error}). A loader reads a cached corpus deterministically, so "
            "a repeat failure is an environment or data problem that will not resolve by retrying. "
            "Stopping rather than spending more GPU time."
        )


def _retire_mining(state, *, reason: str, log=print) -> None:
    """Close the `mine_new_real` route for the rest of the run, without stopping the run.

    Idempotent, and it writes to the run state rather than only logging, because the decision has to
    be visible to the PLANNER: `data_rebuild.mining_available_for_state` reads this key, so a retired
    route stops being offered to the orchestrator instead of being offered and then refused.
    """
    if state.get(MINING_RETIRED_KEY):
        return
    state[MINING_RETIRED_KEY] = reason
    log(f"  ⊘ mine_new_real is RETIRED for the rest of this run: {reason}. The run CONTINUES — "
        "remaining turns choose surgical synthesis while it is open, and hyperparameter tuning "
        "once it is not. Mining finding nothing is an answer about the available corpora, not a "
        "malfunction, so it is not a reason to stop.")


def observe_iteration(state, record: IterationRecord, *, log=print) -> None:
    """Record one curate pass and raise if the run can no longer make progress.

    Called at the end of `curate_node`, after the artifact is written, so the diagnosis it prints
    sits directly under the numbers it is about.
    """
    health = RunHealth.from_state(state)
    # The SELECTOR is recorded because `iteration` alone does not identify a turn: escalate_node
    # resets it to 0 for each new tier, so tier 1 iteration 3 and tier 3 iteration 3 are two
    # different turns with the same key. `_dag_rows` joins the ledger to the DAG on iteration, and
    # without this the join silently attributed one tier's curriculum numbers to another's scores
    # as soon as the report started covering every tier. Entries written before 2026-08-30 have no
    # selector; the join falls back for those rather than guessing.
    _selected = state.get("selected_model")
    health.history.append({
        "iteration": record.iteration,
        "selector": getattr(_selected, "selector", None),
        "strategy": record.strategy,
        "rows_added": record.rows_added,
        "curriculum": [record.curriculum_before, record.curriculum_after],
        "qc_removed": record.qc_removed,
        "firewall_removed": record.firewall_removed,
        "verify": [record.verify_kept, record.verify_attempted],
    })
    problems: list[str] = []

    # --- 1. a rebuild that added nothing ---------------------------------------------------
    is_rebuild = record.strategy not in ("initial_gold", "")
    # An empty rebuild that has just closed off its own route is not a wasted cycle in the sense this
    # counter is about. The counter exists to catch the loop REPEATING a futile action; if
    # `data_rebuild` can no longer be chosen at all — every source exhausted, web research spent, and
    # the teacher short of the synthesis gate — then `iterate` routes to a hyperparameter step next
    # turn and the mistake is not repeatable. Counting it anyway killed run 38734724 two turns before
    # the loop would have corrected itself.
    try:
        from agent.data_rebuild import data_rebuild_available

        can_still_rebuild = data_rebuild_available(state)
    except Exception:  # noqa: BLE001 — a health check must not be the thing that breaks a run
        can_still_rebuild = True
    empty = is_rebuild and record.rows_added <= 0
    if empty and not can_still_rebuild:
        log("  data_rebuild is now exhausted by both routes, so this empty rebuild closed off its "
            "own option rather than wasting a repeatable one — the next turn will tune "
            "hyperparameters instead. Not counted against the wasted-iteration budget.")
    elif empty and record.strategy == "mine_new_real":
        # Never fatal. Mining finding nothing means there is nothing left to find; the run keeps
        # going on the routes that remain, and this route is closed so it cannot be picked again.
        health.empty_mining += 1
        log(f"  ✗ MINING FOUND NOTHING ({health.empty_mining}/{MAX_EMPTY_MINING_ROUNDS}): "
            f"mine_new_real added 0 rows. {_diagnose_empty(record)}")
        if health.empty_mining >= MAX_EMPTY_MINING_ROUNDS:
            _retire_mining(
                state,
                reason=(f"mine_new_real added 0 rows on {health.empty_mining} rounds "
                        f"({_diagnose_empty(record)})"),
                log=log,
            )
    elif empty and record.strategy == "surgical_synthesis":
        health.empty_synthesis += 1
        log(f"  ✗ SYNTHESIS PRODUCED NOTHING ({health.empty_synthesis}/"
            f"{MAX_CONSECUTIVE_EMPTY_SYNTHESIS}): surgical_synthesis added 0 rows.")
        if health.empty_synthesis >= MAX_CONSECUTIVE_EMPTY_SYNTHESIS:
            problems.append(
                f"surgical_synthesis added ZERO rows on {health.empty_synthesis} consecutive "
                "attempts. Unlike an empty mining round, this is not a shortage of material — the "
                "teacher was reachable and a target category was chosen, and still nothing "
                "survived, so the generator and the verifier disagree about the task's contract and "
                "will disagree identically next turn. " + _diagnose_empty(record)
            )
    elif empty:
        health.empty_rebuilds += 1
        log(f"  ✗ WASTED ITERATION ({health.empty_rebuilds}/"
            f"{MAX_CONSECUTIVE_EMPTY_REBUILDS}): data_rebuild/{record.strategy} added 0 rows, so "
            f"this iteration trains on a curriculum identical to the previous one.")
        if health.empty_rebuilds >= MAX_CONSECUTIVE_EMPTY_REBUILDS:
            problems.append(
                f"{health.empty_rebuilds} consecutive data rebuilds added ZERO rows. The last "
                f"strategy was {record.strategy!r}. Every such iteration trains on a curriculum "
                f"identical to the previous one, so the loop cannot learn anything from them — "
                f"which is exactly how run 38566712 spent 7h42m on eight empty rebuilds. "
                + _diagnose_empty(record)
            )
    elif is_rebuild:
        health.empty_rebuilds = 0
        if record.strategy == "surgical_synthesis":
            health.empty_synthesis = 0

    # --- 2. the curriculum shrinking ---------------------------------------------------------
    shrank = record.curriculum_before - record.curriculum_after
    if record.curriculum_before and shrank > 0:
        share = shrank / record.curriculum_before
        if shrank >= CURRICULUM_SHRINK_ROW_ALARM or share >= CURRICULUM_SHRINK_FRACTION_ALARM:
            problems.append(
                f"the curriculum SHRANK by {shrank} row(s) in one iteration "
                f"({share:.0%} of {record.curriculum_before}), from quality control removing "
                f"{record.qc_removed} and the eval firewall removing {record.firewall_removed}. "
                "The curriculum is supposed to be cumulative, and these rows already trained a model "
                "on a previous iteration — so a step now rejecting them is rejecting the task's own "
                "data. The usual cause is a quality-control step whose key does not match the task's "
                "row schema, or a label-space filter running against the wrong vocabulary. Check the "
                "[qc] and [firewall] lines above: each names what it dropped and why."
            )

    # --- 3. verification rejecting everything ---------------------------------------------
    if record.verify_attempted and record.verify_kept == 0:
        health.verify_wipeouts += 1
        log(f"  ✗ VERIFICATION WIPEOUT ({health.verify_wipeouts}/"
            f"{MAX_CONSECUTIVE_TOTAL_VERIFY_REJECTIONS}): all "
            f"{record.verify_attempted} generated row(s) were rejected.")
        if health.verify_wipeouts >= MAX_CONSECUTIVE_TOTAL_VERIFY_REJECTIONS:
            problems.append(
                f"verification rejected EVERY generated row on {health.verify_wipeouts} consecutive "
                f"iterations ({record.verify_attempted} rows in the last one). The generator and the "
                "verifier disagree systematically, so the teacher budget is producing nothing. Check "
                "the [verify] reasons above: if they are all the same reason, the generator prompt "
                "is violating a contract the verifier enforces."
            )
    elif record.verify_attempted:
        health.verify_wipeouts = 0

    # --- 4. mining finding candidates but accepting none ----------------------------------
    if record.strategy == "mine_new_real":
        if record.candidates_seen and record.rows_added <= 0:
            health.mining_shutouts += 1
            log(f"  ✗ MINING SHUTOUT ({health.mining_shutouts}/"
                f"{MAX_MINING_ATTEMPTS_WITHOUT_ACCEPTANCE}): "
                f"{record.candidates_seen} candidate row(s) seen, none accepted.")
            if health.mining_shutouts >= MAX_MINING_ATTEMPTS_WITHOUT_ACCEPTANCE:
                # Retires rather than stops. This is still worth its own diagnosis — a filter
                # rejecting every candidate is a different fault from an empty search, and it is the
                # shape the pre-2026-08-19 whole-source overlap check produced (B293) — but the
                # remedy is the same, and it is not a reason to end a run that can still synthesize
                # or tune.
                _retire_mining(
                    state,
                    reason=(f"mining saw candidate rows on {health.mining_shutouts} attempts and "
                            "accepted NONE of them, which is a filter rejecting everything rather "
                            "than an empty search (B293). Check the [mine]/[acquire] lines for "
                            "which gate is refusing them"),
                    log=log,
                )
        elif record.rows_added > 0:
            health.mining_shutouts = 0

    health.to_state(state)
    if problems:
        raise RunHealthError(
            "the run cannot make progress and is stopping early:\n  - "
            + "\n  - ".join(problems)
        )


def _diagnose_empty(record: IterationRecord) -> str:
    """Name the most likely reason THIS rebuild produced nothing, from what it recorded."""
    if record.strategy == "mine_new_real":
        if record.candidates_seen:
            return (
                f"Mining saw {record.candidates_seen} candidate row(s) and kept none, so the "
                "blockage is a filter, not a shortage: every candidate was either already in the "
                "curriculum or was refused by the label/eval gates."
            )
        return (
            "Mining saw no candidate rows at all: every known source is exhausted and web research "
            "found nothing, so there is no more real data to add for this task."
        )
    if record.strategy == "surgical_synthesis":
        if record.verify_attempted and record.verify_kept == 0:
            return (
                f"Synthesis generated {record.verify_attempted} row(s) and verification rejected "
                "all of them, so the generator is producing rows the verifier considers wrong."
            )
        if not record.verify_attempted:
            return (
                "Synthesis generated nothing at all, so the failure is upstream of verification: "
                "either no failure category was eligible to target, no anchor rows were available, "
                "or the teacher endpoint did not respond."
            )
        return "Synthesis produced rows but every one duplicated a row already in the curriculum."
    return "No sub-strategy was recorded for this rebuild."


def _tier_dags(state) -> list[tuple[str | None, list]]:
    """Every tier's DAG in order, not just the one still in `state["dag"]`.

    WHY THIS IS NOT `state["dag"]`
        `escalate_node` CLEARS the DAG on every promotion and stashes the finished tier in
        `escalation_history`. So the live DAG holds only the last model, and a report reading it
        described a three-tier run as though tier 3 were the whole run — nineteen tier-1
        iterations and six tier-2 iterations simply absent from the attribution table.

        `build_run_progression` is the existing answer to exactly this: the "DAG Traversal (all N
        models)" section has always printed every tier because it reads that instead. This routes
        the attribution table through the same source so the two sections cannot disagree.
    """
    try:
        from agent.pipeline_status import build_run_progression

        progression = build_run_progression(state, [])
    except Exception:  # noqa: BLE001 — reporting must never take a run down
        progression = []
    if not progression:
        model = state.get("selected_model")
        return [(getattr(model, "selector", None), list(state.get("dag") or []))]
    return [
        (entry.get("selector"), list(entry.get("dag") or []))
        for entry in progression
    ]


def _dag_rows(state) -> list[dict]:
    """Per-iteration facts joined across the DAG and the curriculum ledger, for EVERY tier.

    The two records answer different halves of the same question and neither is sufficient alone: the
    DAG knows the SCORE each attempt produced and whether it was kept, while `run_health.history`
    knows what the attempt did to the DATA. Joining them on iteration is what turns "the curriculum
    grew by 982 rows" into "the curriculum grew by 982 rows and that was worth +0.0190".

    The join is scoped by SELECTOR as well as iteration, because iteration numbers restart at each
    escalation and would otherwise collide across tiers.
    """
    health = RunHealth.from_state(state)
    # Ledger entries indexed by (selector, iteration) where the selector was recorded, and by
    # iteration alone for entries written before that field existed. The fallback is only consulted
    # for single-tier runs: on a multi-tier run an untagged entry cannot be attributed to a tier, and
    # a blank cell is honest where a borrowed number is not.
    by_key: dict[tuple[str | None, int], dict] = {}
    legacy_by_iteration: dict[int, dict] = {}
    for entry in health.history:
        iteration = int(entry.get("iteration", -1))
        selector = entry.get("selector")
        if selector:
            by_key[(str(selector), iteration)] = entry
        else:
            legacy_by_iteration[iteration] = entry

    tier_dags = _tier_dags(state)
    allow_legacy = len(tier_dags) <= 1

    rows = []
    for selector, dag in tier_dags:
        rows.extend(_rows_for_dag(dag, selector, by_key, legacy_by_iteration, allow_legacy))
    return rows


def _rows_for_dag(
    dag,
    selector: str | None,
    by_key: dict,
    legacy_by_iteration: dict,
    allow_legacy: bool,
) -> list[dict]:
    """The attribution rows for ONE tier's DAG."""
    rows = []
    for node in dag:
        if not isinstance(node, dict):
            continue
        iteration = int(node.get("iteration", 0) or 0)
        ledger = by_key.get((str(selector), iteration)) if selector else None
        if ledger is None and allow_legacy:
            ledger = legacy_by_iteration.get(iteration)
        ledger = ledger or {}
        pi_d = ((node.get("pi") or {}).get("D") or {})
        plan = pi_d.get("plan") or {}
        # Cross-check the join rather than trusting it. The two records are written by different
        # nodes at different points in the turn, so if either side's notion of "iteration" shifts
        # again the mismatch shows up as a blank row instead of another iteration's numbers
        # attributed to this strategy.
        recorded = str(ledger.get("strategy") or "")
        if ledger and recorded and node.get("intervention") == "data_rebuild":
            planned = str(plan.get("strategy") or "")
            if planned and recorded != planned:
                ledger = {}
        # The strategy that produced this score. A hyperparameter step carries no plan, and the
        # ledger's strategy is whatever the LAST curate wrote — which for a hyperparameter turn is
        # the previous rebuild's, so it must not be read as this iteration's cause.
        if node.get("intervention") == "data_rebuild":
            strategy = str(plan.get("strategy") or ledger.get("strategy") or "data_rebuild")
        else:
            strategy = "hyperparameter"
        if iteration <= 1 and not plan:
            strategy = "initial_gold"
        rows.append({
            "iteration": iteration,
            "strategy": strategy,
            "score": node.get("score"),
            # Train+evaluate were skipped because the rebuild added no rows, so this attempt has no
            # score by construction rather than by failure. Carried through so the table can say
            # "skipped" instead of drawing a blank that reads like a missing measurement.
            "skipped": node.get("status") == SKIPPED_NO_ROWS,
            "pruned": bool(node.get("pruned")),
            # Fall back to the tier's own selector: a stashed DAG node does not always carry one,
            # and a blank model name would collapse two tiers into a single unnamed heading.
            "model": str(node.get("model") or node.get("selector") or selector or ""),
            "tier": node.get("tier"),
            "rows_added": ledger.get("rows_added"),
            "curriculum": ledger.get("curriculum"),
            "qc_removed": ledger.get("qc_removed"),
            "verify": ledger.get("verify"),
            "total": (pi_d.get("composition") or {}).get("total_examples"),
        })
    return rows



def _outcome_label(row: dict) -> str:
    """What happened to this attempt, in the table's rightmost column.

    A skipped iteration is neither kept nor rolled back — it was never trained, so there is no
    checkpoint to keep or discard. Saying "rolled back" would claim an experiment ran and lost.
    """
    if row.get("skipped"):
        return "⏭ skipped (added no rows)"
    return "✗ rolled back" if row["pruned"] else "✓ kept"


def _attribute(rows: list[dict]) -> tuple[list[dict], dict]:
    """Annotate each attempt with the score change it caused, and total those by strategy.

    The delta is measured against the last KEPT score, not the previous iteration's score, because
    that is the checkpoint the attempt actually started from — a pruned attempt is rolled back, so the
    next attempt builds on the kept one and comparing to the discarded score would misattribute the
    difference between two rejected experiments.

    Only KEPT steps contribute to the total, and they sum exactly to the final score minus the
    starting one, which is what makes the table auditable: a strategy's number is the accuracy the run
    actually retained from it, not the accuracy it transiently reached.
    """
    contributions: dict[str, dict] = {}
    baseline = None
    for row in rows:
        score = row.get("score")
        strategy = row["strategy"]
        slot = contributions.setdefault(strategy, {"gain": 0.0, "kept": 0, "attempts": 0})
        slot["attempts"] += 1
        if score is None:
            row["delta"] = None
            continue
        score = float(score)
        if baseline is None:
            # The first scored iteration is the starting point, not a gain over anything.
            # `start` is recorded on the slot for provenance — which strategy happened to run
            # first — and NOT added to `gain`, so the strategy's own contribution stays separable
            # from the accuracy the run began with. See the renderer.
            row["delta"] = None
            baseline = score
            slot["kept"] += 1
            slot["start"] = score
            slot["start_iteration"] = row.get("iteration")
            continue
        row["delta"] = score - baseline
        if not row["pruned"]:
            slot["gain"] += score - baseline
            slot["kept"] += 1
            baseline = score
    return rows, contributions


def _wasted_rebuild_lines(state) -> list[str]:
    """How many rebuilds added nothing — the question the whole ledger exists to answer.

    Counted from the data ledger rather than the DAG, because a rebuild that added no rows is a wasted
    cycle whether or not its score happened to survive: the model retrained on an identical curriculum.
    """
    health = RunHealth.from_state(state)
    rebuilds = [entry for entry in health.history
                if entry.get("strategy") not in ("initial_gold", "", None)]
    if not rebuilds:
        return []
    empty = sum(1 for entry in rebuilds if int(entry.get("rows_added", 0) or 0) <= 0)
    line = f"    {empty} of {len(rebuilds)} rebuild(s) added no rows"
    if empty:
        line += "  ← every one of these was a wasted train+eval cycle"
    return [line]


def format_health_summary(state) -> list[str]:
    """Per-iteration curriculum growth AND the score each strategy actually bought.

    Printed whether or not the run finished, because a cancelled run still answers the questions this
    table exists for: did the interventions add data, and did the data buy accuracy. Grouped by model
    tier, since a run that escalates is really several experiments and pooling their deltas would
    average across models with different ceilings.
    """
    rows, contributions = _attribute(_dag_rows(state))
    if not rows:
        # No DAG (a run that died in cold start) — fall back to the data-only ledger.
        return _format_growth_only(state)

    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(f"{row['model']}", []).append(row)

    lines = ["  Curriculum growth and score attribution per iteration:"]
    for model, group in by_model.items():
        tier = next((r["tier"] for r in group if r["tier"] is not None), None)
        heading = f"    ── {model or 'model unrecorded'}"
        if tier is not None:
            heading = f"    ── Tier {tier}: {model}"
        lines.append(f"{heading}  ({len(group)} iteration(s)) ──")
        lines.append(
            f"    {'Iter':>4}  {'Strategy':<20} {'Added':>7} {'Curriculum':>16} {'QC':>5} "
            f"{'Verified':>9} {'Score':>7} {'Δ':>8}  Kept"
        )
        for row in group:
            curriculum = row.get("curriculum")
            span = (f"{curriculum[0]} → {curriculum[1]}"
                    if isinstance(curriculum, list) and len(curriculum) == 2 else
                    (str(row.get("total")) if row.get("total") else "—"))
            added = row.get("rows_added")
            kept_v, attempted_v = (row.get("verify") or [0, 0])[:2]
            score = row.get("score")
            delta = row.get("delta")
            lines.append(
                f"    {row['iteration']:>4}  {row['strategy']:<20} "
                f"{('—' if added is None else f'{added:+d}'):>7} {span:>16} "
                f"{('—' if row.get('qc_removed') is None else row['qc_removed']):>5} "
                f"{(f'{kept_v}/{attempted_v}' if attempted_v else '—'):>9} "
                f"{('—' if score is None else f'{float(score):.4f}'):>7} "
                f"{('—' if delta is None else f'{delta:+.4f}'):>8}  "
                f"{_outcome_label(row)}"
            )

    lines.extend(_wasted_rebuild_lines(state))

    final = next((float(r["score"]) for r in reversed(rows)
                  if r.get("score") is not None and not r["pruned"]), None)
    start = next((c.get("start") for c in contributions.values() if "start" in c), None)
    lines.append("")
    lines.append("  What each strategy actually bought (kept steps only — pruned attempts were "
                 "rolled back and contributed nothing):")
    # THE STARTING POINT IS ITS OWN ROW, not folded into whichever strategy happened to run first.
    #
    # It used to be the same row: a strategy that both opened the run AND won later iterations
    # printed only `0.8320  starting point (3/5 kept)`, and its actual gain — the number the table
    # exists to report — was computed and then never shown. There was no way to read how much
    # `mine_new_real` had bought, only that the run began at 0.8320 and that mining was involved.
    start_slot = next((s for s in contributions.values() if "start" in s), None)
    if start_slot is not None:
        opener = next(name for name, s in contributions.items() if "start" in s)
        where = start_slot.get("start_iteration")
        lines.append(
            f"    {'STARTING POINT':<22} {start_slot['start']:>8.4f}   "
            f"first measured score"
            + (f" (iteration {where}, {opener})" if where is not None else f" ({opener})")
            + " — a baseline, not a gain"
        )
    ordered = sorted(contributions.items(), key=lambda kv: -kv[1]["gain"])
    for strategy, slot in ordered:
        # Every strategy now reports its GAIN, including the one that opened the run. Its kept count
        # still includes the opening iteration, so that is said plainly rather than left to imply
        # the gain was spread over more steps than it was.
        note = ""
        if "start" in slot:
            note = (
                ", 1 of which set the starting point above and so is worth 0 here"
                if slot["kept"] > 1 else
                " — the starting point only, no further gain"
            )
        lines.append(f"    {strategy:<22} {slot['gain']:>+8.4f}   "
                     f"{slot['kept']}/{slot['attempts']} attempt(s) kept{note}")
    if final is not None and start is not None:
        gained = sum(s["gain"] for s in contributions.values())
        lines.append(f"    {'':<22} {'':>8}   ")
        lines.append(f"    {'FINAL':<22} {final:>8.4f}   "
                     f"= {start:.4f} start {gained:+.4f} from interventions")
    return lines


def _format_growth_only(state) -> list[str]:
    """The data-only ledger, for a run with no DAG to join against."""
    health = RunHealth.from_state(state)
    if not health.history:
        return []
    lines = [
        "  Curriculum growth per iteration:  (no DAG recorded, so no score attribution)",
        f"    {'Iter':>4}  {'Strategy':<20} {'Added':>7} {'Curriculum':>18} {'QC':>7} "
        f"{'Firewall':>9}  {'Verified'}",
    ]
    for entry in health.history:
        before, after = (entry.get("curriculum") or [0, 0])[:2]
        kept, attempted = (entry.get("verify") or [0, 0])[:2]
        lines.append(
            f"    {entry.get('iteration', '?'):>4}  {str(entry.get('strategy', '?')):<20} "
            f"{entry.get('rows_added', 0):>+7} {f'{before} → {after}':>18} "
            f"{entry.get('qc_removed', 0):>7} {entry.get('firewall_removed', 0):>9}  "
            f"{f'{kept}/{attempted}' if attempted else '—'}"
        )
    lines.extend(_wasted_rebuild_lines(state))
    return lines
