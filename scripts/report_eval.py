"""The number you publish. Full held-out split, report metric, N seeds, run once.

WHY THIS IS A SCRIPT AND NOT A NODE
    The agent loop evaluates on every iteration, and that eval exists to RANK CHECKPOINTS. Ranking
    is a paired comparison — the same fixed rows scored against successive checkpoints — so most of
    its sampling error is common-mode and cancels, which is what makes `TaskSpec.select_cap`'s
    1,000 rows defensible there. It is not defensible in a paper: at n=1,000 and 80% accuracy the
    95% CI half-width is about +/-2.5 points, wide enough to swallow most of the effects this
    project is trying to measure.

    The fix is not to raise the in-loop cap. That multiplies the cost by the iteration count and
    still leaves the loop selecting on the wrong metric for two tasks. The fix is to separate the
    two jobs, which is what `report_load` / `report_score` / `report_metric_name` do, and to run the
    reporting half exactly once — from here, after a run has finished and chosen its checkpoint.

    Being outside the graph is also why seeds live here. A seed sweep is N independent training
    runs, which is not a shape the loop has; it is a shape a script has.

SELECTION AND REPORTING METRICS DIFFER ON PURPOSE
    For three tasks the honest headline is unusable as a per-iteration signal, so the two are not
    the same number:

        multiconer   micro-F1 selects, macro-F1 publishes. A <=1,000-row draw can hold zero
                     examples of a class that is 0.18% of entities, so macro-F1 is undefined or
                     wildly noisy as a ranking signal while remaining the right headline.
        goemotions   Ekman-7 macro-F1 selects, threshold-free macro AUPRC publishes. Macro-F1 over
                     28 labels moves several points with nothing but the threshold.
        dialogsum    ROUGE-L selects, ROUGE-1/2/L + BERTScore publishes.

    Every task states both on its spec, so this script needs no per-task table of its own. That is
    the point: the registry is the single place a task is defined, and the five side registries it
    replaced were each added after a bug caused by their being out of sync.

USAGE
    # One checkpoint, full split, report metric
    python scripts/report_eval.py --task gec_bea19 \
        --checkpoint logs/runs/slm-gec.../artifacts/adapter_v3 \
        --base-model Qwen/Qwen3-1.7B --out logs/reports/gec.json

    # Paired seed comparison against a baseline. IDENTICAL seeds on both arms.
    python scripts/report_eval.py --task topv2 --seeds 5 \
        --checkpoint ... --baseline-checkpoint ... --out logs/reports/topv2.json

    # The zero-shot anchor: no adapter, just the base model
    python scripts/report_eval.py --task goemotions --checkpoint base --base-model Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The doc's guidance, encoded so the default is the defensible one. TOPv2 trains on 493 and 176
# rows, which is squarely where seed variance bites; the 12k-43k tasks are far more stable. A
# 3-seed probe first is still the right move: if the spread is 0.3 and the gain is 5 points, stop
# seeding, and if the spread is 2 and the gain is 1.5 there is no result yet.
SEEDS_BY_TASK = {"topv2": 5}
DEFAULT_SEEDS = 3

# Fixed and published, so two people running this script compare the same arms. Passing --seeds 3
# takes the first three.
SEED_VALUES = (0, 1, 2, 3, 4, 5, 6, 7)


def _log(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)


def _rows_fingerprint(rows: list[dict]) -> str:
    """A content hash of the exact rows scored.

    Published alongside the score because a report metric is only comparable against another
    report metric computed on the same rows. `multiconer`'s headline is a stratified 20k slice of a
    249,980-row test split; without this, two runs could report "macro-F1 on the 20k slice" for two
    different 20k slices and nothing would say so.
    """
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str).encode())
    return digest.hexdigest()[:16]


def load_report_rows(task: str, log=_log) -> list[dict]:
    """The FULL report split for a task, uncapped.

    Two paths, because for most tasks the report split is the same split the loop evaluates on,
    just not truncated to `select_cap`. Only a task whose report split is a genuinely DIFFERENT
    split declares `report_load`.
    """
    from tasks import get_task

    spec = get_task(task)
    if spec.report_load is not None:
        rows = spec.report_load(log=log)
        log(f"  report split: {len(rows)} row(s) from {task}'s own report_load")
        return list(rows)

    # `max_test` is deliberately enormous rather than `spec.select_cap`: the loader returns as many
    # rows as it has, up to what it is asked for, and the whole purpose of this script is to ask
    # for all of them. Train rows are discarded — nothing here trains.
    _train, rows = spec.load(max_train=1, max_test=10**9, log=log)
    log(f"  report split: {len(rows)} row(s) (full eval split from {task}'s loader)")
    return list(rows)


def _attach_label_scores(eval_set, task: str, weights_ref: str, base_model: str, log=_log) -> None:
    """Attach per-label rankings to each row, for a report metric that needs them.

    ONLY GoEmotions needs this, and only for its headline. Threshold-free macro AUPRC measures
    whether the model RANKS the correct labels highly, and generation returns a hard decision with
    no ranking in it — so without this pass the honest metric for that task is not computable at
    all. The selection metric (Ekman-7 macro-F1) comes from ordinary generation and is unaffected.

    Keyed off the report metric name rather than the task name so the report script needs no
    per-task table: the registry is the single place a task is defined.
    """
    from tasks import get_task

    spec = get_task(task)
    if spec.report_metric_name != "macro_auprc":
        return
    from data.loaders.goemotions import EMOTIONS
    from eval.scorers.multilabel_emotion import build_prompt
    from training.slm_helpers import infer_label_scores_batch

    prompts = [build_prompt(row.get("text", "")) for row in eval_set.all]
    log(f"  scoring {len(EMOTIONS)} labels against {len(prompts)} row(s) for macro AUPRC")
    scores = infer_label_scores_batch(prompts, list(EMOTIONS), weights_ref, base_model, task=task)
    for row, row_scores in zip(eval_set.all, scores):
        row["label_scores"] = row_scores


def _score_once(rows: list[dict], task: str, weights_ref: str, base_model: str,
                gguf_path: str | None = None) -> dict:
    """One report-metric pass over `rows`, through the pipeline's own harness."""
    from data.eval_set import build_eval_set
    from eval.harness import run_eval

    # `target=len(rows)` because these rows ARE the report split. `build_eval_set` would otherwise
    # subsample to its default of 100 — the same off-by-a-cap that made raising the eval size a
    # no-op in the loop (B288).
    eval_set = build_eval_set(rows, task=task, target=len(rows))
    _attach_label_scores(eval_set, task, weights_ref, base_model)
    result = run_eval(
        eval_set,
        weights_ref,
        base_model,
        quant="Q4_K_M" if gguf_path else None,
        # A GGUF by construction — this script is handed one on the command line (`--gguf`) — so
        # the backend is named here rather than read from the environment.
        quant_artifact=gguf_path,
        quant_backend="llama_cpp",
        # The whole reason this script exists.
        scoring="report",
    )
    return {
        "score": round(float(result.f1), 6),
        "metric": result.metric,
        "format_valid": round(float(getattr(result, "format_valid", 1.0)), 6),
        "n": len(eval_set.all),
        # The diagnostics each report scorer hangs off `per_class` — per-error-type F0.5 for GEC,
        # the four per-domain EMs for TOPv2, frequency-banded F1 for GoEmotions. These are the
        # numbers the writeup decomposes, so they are kept in full rather than summarized.
        "diagnostics": result.per_class,
    }


def _summarize(scores: list[float]) -> dict:
    """Mean, spread, and the spread's own honesty check.

    `std` is the population std for a single seed (0.0) rather than a crash: one seed has no
    spread, and reporting that as an error would be wrong — it is a fact about the experiment.
    """
    if not scores:
        return {"mean": None, "std": None, "min": None, "max": None, "n_seeds": 0}
    return {
        "mean": round(statistics.fmean(scores), 6),
        "std": round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0,
        "min": round(min(scores), 6),
        "max": round(max(scores), 6),
        "n_seeds": len(scores),
    }


def _paired_delta(method: list[float], baseline: list[float]) -> dict | None:
    """The per-seed difference, which is the whole reason to fix the seeds.

    Comparing mean(method) against mean(baseline) throws away the pairing and leaves a difference
    whose error bar is inflated by seed variance that affects BOTH arms identically. Differencing
    within a seed removes that shared component, which tightens the comparison for free — the
    cheapest precision available on this project.

    Reported as a plain summary rather than a p-value: with 3-5 seeds a bootstrap over seeds is
    fitting a distribution to five points, and `wins` plus the paired spread says the same thing
    without dressing it up.
    """
    if not method or len(method) != len(baseline):
        return None
    deltas = [m - b for m, b in zip(method, baseline)]
    return {
        "per_seed": [round(d, 6) for d in deltas],
        "mean": round(statistics.fmean(deltas), 6),
        "std": round(statistics.pstdev(deltas), 6) if len(deltas) > 1 else 0.0,
        "wins": sum(1 for d in deltas if d > 0),
        "of": len(deltas),
    }


def _checkpoint_for_seed(template: str, seed: int) -> str:
    """Resolve a per-seed checkpoint path.

    `{seed}` in the path is substituted; a path without it is used unchanged. The unchanged case is
    the honest one for a single already-trained checkpoint, and it is recorded in the output as
    `seed_varied: false` so nobody reads N identical numbers as N seeds of evidence.
    """
    return template.format(seed=seed) if "{seed}" in template else template


def _run_arm(name: str, template: str, rows: list[dict], task: str, base_model: str,
             seeds: tuple[int, ...], gguf: str | None) -> dict:
    varied = "{seed}" in template
    per_seed: list[dict] = []
    for seed in seeds:
        checkpoint = _checkpoint_for_seed(template, seed)
        _log(f"  [{name}] seed={seed} checkpoint={checkpoint}")
        cell = _score_once(rows, task, checkpoint, base_model, gguf_path=gguf)
        cell["seed"] = seed
        cell["checkpoint"] = checkpoint
        per_seed.append(cell)
        _log(f"  [{name}] seed={seed} {cell['metric']}={cell['score']:.4f} "
             f"(format_valid={cell['format_valid']:.3f}, n={cell['n']})")
        if not varied:
            # Re-evaluating one fixed checkpoint N times is deterministic greedy decoding over
            # identical rows, so it would produce N copies of one number and a fake std of 0.0.
            _log(f"  [{name}] checkpoint path has no {{seed}} placeholder; "
                 f"one checkpoint means one measurement, not {len(seeds)}")
            break
    return {
        "seed_varied": varied,
        "per_seed": per_seed,
        "summary": _summarize([cell["score"] for cell in per_seed]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, help="registry task key")
    parser.add_argument(
        "--checkpoint", required=True,
        help="adapter path, or 'base' for the zero-shot anchor. May contain {seed}.")
    parser.add_argument("--base-model", required=True, help="base model id the adapter fits")
    parser.add_argument(
        "--baseline-checkpoint", default=None,
        help="second arm for a paired comparison, scored on the SAME seeds and rows")
    parser.add_argument(
        "--seeds", type=int, default=None,
        help="how many of the fixed seed values to use; default is per-task (topv2 5, else 3)")
    parser.add_argument("--gguf", default=None, help="score this GGUF instead of the bf16 adapter")
    parser.add_argument("--out", default=None, help="write the report JSON here")
    args = parser.parse_args(argv)

    from tasks import get_task

    spec = get_task(args.task)
    n_seeds = args.seeds if args.seeds is not None else SEEDS_BY_TASK.get(args.task, DEFAULT_SEEDS)
    if n_seeds < 1 or n_seeds > len(SEED_VALUES):
        parser.error(f"--seeds must be between 1 and {len(SEED_VALUES)}")
    seeds = SEED_VALUES[:n_seeds]

    _log(f"report eval — task={args.task} ({spec.title})")
    _log(f"  selection metric (in-loop, NOT published): {spec.metric_name} "
         f"at select_cap={spec.select_cap}")
    _log(f"  report metric (published): {spec.report_metric_name}")
    _log(f"  seeds: {list(seeds)}")

    rows = load_report_rows(args.task)
    if not rows:
        _log("  ERROR: the report split is empty; refusing to report a score over no rows")
        return 1
    fingerprint = _rows_fingerprint(rows)
    _log(f"  rows fingerprint: {fingerprint}")

    report: dict = {
        "task": args.task,
        "title": spec.title,
        "base_model": args.base_model,
        "select_metric": spec.metric_name,
        "select_cap": spec.select_cap,
        "report_metric": spec.report_metric_name,
        "n_rows": len(rows),
        "rows_fingerprint": fingerprint,
        "seeds": list(seeds),
        "quant": "Q4_K_M" if args.gguf else None,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    report["method"] = _run_arm(
        "method", args.checkpoint, rows, args.task, args.base_model, seeds, args.gguf)
    if args.baseline_checkpoint:
        report["baseline"] = _run_arm(
            "baseline", args.baseline_checkpoint, rows, args.task, args.base_model, seeds,
            args.gguf)
        report["paired_delta"] = _paired_delta(
            [cell["score"] for cell in report["method"]["per_seed"]],
            [cell["score"] for cell in report["baseline"]["per_seed"]],
        )

    summary = report["method"]["summary"]
    _log(f"  {spec.report_metric_name}: mean={summary['mean']} std={summary['std']} "
         f"over {summary['n_seeds']} seed(s), n={len(rows)}")
    delta = report.get("paired_delta")
    if delta:
        _log(f"  paired delta: mean={delta['mean']} std={delta['std']} "
             f"({delta['wins']}/{delta['of']} seeds favour the method)")
        if delta["std"] and abs(delta["mean"]) < delta["std"]:
            _log("  NOTE: the paired delta is smaller than its own spread across seeds. "
                 "That is not a result yet — add seeds or accept there is no effect.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
        _log(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
