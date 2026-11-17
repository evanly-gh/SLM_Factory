"""Rescore a finished run's best model on held-out rows the run never selected on.

WHY. In a run, every checkpoint is scored on the same frozen eval set (`artifacts/eval_set.json`,
up to 1,000 rows), and the run reports the best of those scores. That eval set is therefore also
the selection set: a tier's best is the maximum of up to 30 evaluations on it, which reads slightly
high. This script answers the reviewer's question "what does the chosen model score on rows the
search never saw?" without retraining anything.

WHAT IT DOES, for one run:
  1. Picks the run's best model from run.log: every `weights:` line is followed by its
     `[evaluate][<model> [<quant>]] Score:` line (checked); the highest score wins, which is the
     number the paper's Table 3 reports.
  2. Loads the task's full eval split (the same loader `scripts/report_eval.py` uses) and removes
     every row whose normalized text is in the run's selection set OR in any training dataset the
     run wrote (`artifacts/dataset_v*.jsonl`) — mining reads further into a task's own sources, so
     the second exclusion is not optional. The normalizer is the contamination firewall's own.
  3. Builds (or reuses from the cache) the exact quantized artifact the loop scored, through the
     evaluate node's own builder, and scores with the task's SELECTION metric — the metric in
     Table 3, not `report_eval.py`'s report metric.
  4. Scores the run's own selection rows again first ("replay"): greedy decoding of the same
     artifact should reproduce the logged score, which checks that nothing in this path differs
     from the loop's.
  5. Scores up to --max-rows unseen rows, drawn with a fixed seed, and writes one JSON report.

Usage (GPU job; see tests/pipeline/rescore_unseen_l40s.slurm):
  python scripts/rescore_unseen.py --task sms_spam \
      --run-dir logs/runs/slm-sms-spam-calibrated-l40s-42806318 --out logs/reports/unseen/x.json
  python scripts/rescore_unseen.py ... --dry-run     # parse only: best model, no data, no GPU
  python scripts/rescore_unseen.py ... --first-ft    # score the chosen variant's FIRST fine-tune instead
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WEIGHTS_RE = re.compile(r"^\[evaluate\]\[(?P<label>.+?)\]\s+weights: (?P<ref>\S+)")
SCORE_RE = re.compile(
    r"^\[evaluate\]\[(?P<model>.+?) \[(?P<quant>[\w.]+)\]\] Score: (?P<metric>\w+)=(?P<score>[\d.]+)")
FIRST_FT_RE = re.compile(
    r"^\[evaluate\]\[(?P<model>.+?) \[(?P<quant>[\w.]+)\]\]\s+First fine-tuned score = (?P<score>[\d.]+)")
# The per-config result printed right after each `weights:` line ("  → span_f1=0.7051  format_valid=…").
CONFIG_RESULT_RE = re.compile(
    r"^\[evaluate\]\[(?P<model>.+?) \[(?P<quant>[\w.]+)\]\]\s+→ (?P<metric>\w+)=(?P<score>[\d.]+)")


def log(*parts) -> None:
    print("[rescore_unseen]", *parts, flush=True)


def best_model(run_dir: str) -> dict:
    """Highest-scoring evaluation in run.log, with the weights it scored."""
    pending, evals = None, []
    with open(os.path.join(run_dir, "run.log"), errors="replace") as fh:
        for line in fh:
            w = WEIGHTS_RE.match(line)
            if w:
                pending = w.group("ref")
                continue
            s = SCORE_RE.match(line)
            if s:
                if pending is None:
                    raise SystemExit(f"Score line without a preceding weights line: {line.strip()}")
                evals.append({"model_id": s.group("model"), "quant": s.group("quant"),
                              "metric": s.group("metric"), "score": float(s.group("score")),
                              "weights_ref": pending})
                pending = None
    if not evals:
        raise SystemExit(f"no evaluations found in {run_dir}/run.log")
    best = max(evals, key=lambda e: e["score"])
    best["n_evals"] = len(evals)
    return best


def first_ft_model(run_dir: str, best: dict) -> dict:
    """The FIRST fine-tune of the variant the run chose (the best model's model and quant): the
    highest-scoring config of that variant's iteration 1, i.e. the checkpoint behind the run's
    `First fine-tuned score` line and Table 3's "First FT" column. Every config result (`weights:`
    then `→ metric=`) logged since the previous variant's First-FT line belongs to iteration 1."""
    pending, configs, start = None, [], 0
    with open(os.path.join(run_dir, "run.log"), errors="replace") as fh:
        for line in fh:
            w = WEIGHTS_RE.match(line)
            if w:
                pending = w.group("ref")
                continue
            c = CONFIG_RESULT_RE.match(line)
            if c and pending is not None:
                configs.append({"model_id": c.group("model"), "quant": c.group("quant"),
                                "metric": c.group("metric"), "score": float(c.group("score")),
                                "weights_ref": pending})
                pending = None
                continue
            f = FIRST_FT_RE.match(line)
            if not f:
                continue
            if (f.group("model"), f.group("quant")) != (best["model_id"], best["quant"]):
                start = len(configs)
                continue
            logged = float(f.group("score"))
            cands = [e for e in configs[start:]
                     if (e["model_id"], e["quant"]) == (best["model_id"], best["quant"])]
            if not cands:
                raise SystemExit("no iteration-1 config results before the First fine-tuned score line")
            pick = dict(max(cands, key=lambda e: e["score"]))
            if abs(pick["score"] - logged) > 5e-5:
                raise SystemExit(f"iteration-1 best {pick['score']} != logged First FT {logged}")
            pick.update({"first_ft_logged": logged, "iteration1_configs": len(cands)})
            return pick
    raise SystemExit(f"no 'First fine-tuned score' line for {best['model_id']} [{best['quant']}]")


def row_keys(rows) -> set[str]:
    from agent.nodes.curate import _normalized_texts
    return _normalized_texts([r for r in rows if isinstance(r, dict)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-replay", action="store_true", help="skip rescoring the selection rows")
    ap.add_argument("--dry-run", action="store_true", help="only identify the best model")
    ap.add_argument("--allow-disjoint-split", action="store_true",
                    help="the loader's uncapped eval split is a DIFFERENT split from the one the run "
                         "selected on (multiconer: dev vs a 20k test slice); skip the match check")
    ap.add_argument("--count-only", action="store_true",
                    help="identify the model and count unseen rows from each source; no GPU")
    ap.add_argument("--control-base", action="store_true",
                    help="also score the UNTRAINED base model (same quant) on the selection rows and "
                         "the unseen sample: nothing ever selected on it, so its gap between the two "
                         "row sets measures row difficulty, not selection bias")
    ap.add_argument("--score-unused-train", action="store_true",
                    help="also score unused training-split rows. Off by default: they are what the "
                         "run's own sampling left behind (e.g. a balanced draw leaves almost no spam), "
                         "so they are biased by construction")
    ap.add_argument("--min-rows", type=int, default=200,
                    help="score a source only if it has at least this many unseen rows")
    ap.add_argument("--chunk-rows", type=int, default=0,
                    help="instead of one sample, split the unseen rows into disjoint sets of this many "
                         "rows (use the selection-set size) and score each: their spread is the "
                         "sampling noise of a selection-sized score, so the selection score can be "
                         "placed against it. With --control-base the base model scores every chunk too")
    ap.add_argument("--max-chunks", type=int, default=5)
    ap.add_argument("--external-rows", default=None,
                    help="JSONL of rows from ANOTHER dataset, each with a `set` index: score each set "
                         "(factory, and base with --control-base) instead of the task's own unseen "
                         "rows. Rows are given as-is; dedup against the task's data is the caller's job")
    ap.add_argument("--first-ft", action="store_true",
                    help="score the run's FIRST fine-tune of its chosen variant (Table 3 'First FT') "
                         "instead of its best model. Row selection is unchanged, so with the same "
                         "--seed/--chunk-rows/--max-chunks/--external-rows the rows are the same as "
                         "the best model's report")
    ap.add_argument("--reuse-from", default=None,
                    help="read-only fallback cache: a directory laid out like `artifacts/` (e.g. the "
                         "cleanup trash's artifacts/) where a validated quantized artifact for the "
                         "scored weights is used in place if `artifacts/` has none, instead of rebuilding")
    args = ap.parse_args(argv)

    run_best = best_model(args.run_dir)
    best = first_ft_model(args.run_dir, run_best) if args.first_ft else run_best
    quant = None if best["quant"].lower() in ("bf16", "none") else best["quant"]
    log(f"task={args.task} run={os.path.basename(args.run_dir)}")
    log(f"best of {run_best['n_evals']} evals: {run_best['model_id']} [{run_best['quant']}] "
        f"{run_best['metric']}={run_best['score']:.4f}")
    if args.first_ft:
        log(f"--first-ft: scoring the first fine-tune of that variant "
            f"({best['iteration1_configs']} iteration-1 config(s)): "
            f"{best['metric']}={best['score']:.4f} (logged First FT {best['first_ft_logged']:.4f})")
    log(f"weights: {best['weights_ref']}  (exists: {os.path.exists(best['weights_ref'])})")
    if args.dry_run:
        return 0

    from agent.nodes.curate import _row_text
    from data.eval_set import build_eval_set
    from data.loaders.dataset_integrity import normalize_text
    from eval.harness import run_eval
    from scripts.report_eval import load_report_rows
    from tasks import get_task

    get_task(args.task)  # fail fast on a bad key
    selection = json.load(open(os.path.join(args.run_dir, "artifacts", "eval_set.json")))["examples"]
    train_rows = []
    for path in sorted(glob.glob(os.path.join(args.run_dir, "artifacts", "dataset_v*.jsonl"))):
        with open(path) as fh:
            train_rows.extend(json.loads(line) for line in fh if line.strip())
    sel_keys, train_keys = row_keys(selection), row_keys(train_rows)

    report_rows = load_report_rows(args.task, log=log)
    keyed = [(normalize_text(_row_text(r)), r) for r in report_rows if isinstance(r, dict)]
    found = sum(1 for k, _ in keyed if k in sel_keys)
    log(f"full eval split: {len(keyed)} rows; selection rows found in it: {found}/{len(sel_keys)}")
    if args.allow_disjoint_split:
        log("--allow-disjoint-split: the eval split here is a separate split from the selection set")
    elif sel_keys and found < 0.9 * len(sel_keys):
        raise SystemExit("selection rows do not match the loader's rows; the text key is wrong "
                         "for this task, so 'unseen' cannot be established — stopping")
    seen_dupe, unseen = set(), []
    for k, r in keyed:
        if not k or k in sel_keys or k in train_keys or k in seen_dupe:
            continue
        seen_dupe.add(k)
        unseen.append(r)
    log(f"unseen eval-split rows (not in selection set, not in any training dataset): {len(unseen)}")

    # Second source: rows of the task's TRAINING split that the run never loaded into any training
    # dataset. New to the model and from the same dataset; reported separately, never pooled with
    # the eval-split rows, because an official train split can differ in distribution from test.
    spec = get_task(args.task)
    train_split, _ = spec.load(max_train=10**9, max_test=1, log=log)
    unused_train, seen2 = [], set()
    for r in train_split:
        if not isinstance(r, dict):
            continue
        k = normalize_text(_row_text(r))
        if not k or k in sel_keys or k in train_keys or k in seen_dupe or k in seen2:
            continue
        seen2.add(k)
        unused_train.append(r)
    log(f"unused training-split rows (never in any training dataset or the selection set): "
        f"{len(unused_train)} of {len(train_split)}")
    report = {"task": args.task, "run_dir": args.run_dir, "best": best,
              **({"scored_model": "first_ft", "run_best": run_best} if args.first_ft else {}),
              "full_split_rows": len(keyed), "selection_rows": len(sel_keys),
              "training_rows_excluded_from": len(train_keys), "unseen_available": len(unseen),
              "train_split_rows": len(train_split), "unused_train_available": len(unused_train),
              "eval_split_disjoint_from_selection": bool(args.allow_disjoint_split)}
    if args.count_only:
        print(json.dumps({k: v for k, v in report.items() if k != "best"}))
        return 0

    artifact, backend = None, None
    if quant:
        from agent.nodes.evaluate import _build_quant_artifact_for_eval
        from training.quant_backend import backend_for_model
        backend = backend_for_model(best["model_id"], quant)
        if args.reuse_from:
            import hashlib
            from training.quant_backend import artifact_name, artifacts_subdir, validated_cache_hit
            rel = os.path.join(artifacts_subdir(backend), best["model_id"].replace("/", "_"),
                               hashlib.sha1(f"{best['weights_ref']}|{quant}".encode()).hexdigest()[:12],
                               artifact_name(quant, backend))
            if not validated_cache_hit(os.path.join("artifacts", rel), backend):
                fallback = os.path.join(args.reuse_from, rel)
                if validated_cache_hit(fallback, backend):
                    artifact = fallback
                    log(f"--reuse-from: validated artifact found, used in place: {fallback}")
                else:
                    log(f"--reuse-from: no validated artifact at {fallback}; building")
        if artifact is None:
            artifact = _build_quant_artifact_for_eval(best["weights_ref"], best["model_id"], quant,
                                                      f"{best['model_id']} [{quant}]")
        log(f"artifact: {artifact} (backend {backend})")

    # Keep only unseen rows whose rendered prompt is no longer than the longest selection-set
    # prompt: every one of those fit the loop's context, so this applies the loop's own implicit
    # bound instead of truncating or crashing (a BFCL test prompt runs to 5,188 tokens).
    spec_prompts = get_task(args.task)
    def _prompt_lengths(rows):
        es = build_eval_set(rows, task=args.task, target=len(rows))
        return [len(p) for p in spec_prompts.build_prompts(es)], es.all
    sel_len, _ = _prompt_lengths(selection)
    limit = max(sel_len)
    if unseen:
        lens, ordered = _prompt_lengths(unseen)
        kept = [r for r, n in zip(ordered, lens) if n <= limit]
        log(f"length filter (prompt <= {limit} chars, the longest selection prompt): "
            f"kept {len(kept)} of {len(ordered)} unseen rows")
        report["unseen_after_length_filter"] = len(kept)
        unseen = kept

    def score(rows, name, weights=None, art=None):
        t0 = time.time()
        es = build_eval_set(rows, task=args.task, target=len(rows))
        res = run_eval(es, weights or best["weights_ref"], best["model_id"], quant=quant,
                       quant_artifact=art if weights else artifact, quant_backend=backend,
                       scoring="select")
        out = {"score": round(float(res.f1), 6), "metric": res.metric, "n": len(es.all),
               "format_valid": round(float(getattr(res, "format_valid", 1.0)), 6),
               "seconds": round(time.time() - t0, 1)}
        log(f"{name}: {out}")
        return out

    if not args.no_replay:
        report["replay_selection"] = score(selection, "replay on selection rows")
    rng = random.Random(args.seed)

    if args.chunk_rows or args.external_rows:
        import statistics
        if args.external_rows:
            ext = [json.loads(line) for line in open(args.external_rows) if line.strip()]
            by_set = {}
            for r in ext:
                by_set.setdefault(r["set"], []).append({k: v for k, v in r.items()
                                                        if k not in ("set", "source")})
            chunks = [by_set[k] for k in sorted(by_set)]
            size = len(chunks[0])
            report["external_rows"] = args.external_rows
            log(f"external rows: {len(chunks)} set(s) from {args.external_rows}")
        else:
            pool = list(unseen)
            rng.shuffle(pool)
            size = args.chunk_rows
            if len(pool) < 2 * size:  # too few rows for two full chunks: halves (noise scaled below)
                size = len(pool) // 2
            chunks = [pool[i * size:(i + 1) * size]
                      for i in range(min(args.max_chunks, len(pool) // size))]
        base_art = None
        if args.control_base and quant:
            base_art = _build_quant_artifact_for_eval(best["model_id"], best["model_id"], quant,
                                                      f"{best['model_id']} [{quant}] base")
        rows_out = []
        for i, ch in enumerate(chunks):
            entry = {"factory": score(ch, f"chunk {i} ({len(ch)} rows)")}
            if args.control_base:
                entry["base"] = score(ch, f"UNTRAINED base, chunk {i}", weights=best["model_id"],
                                      art=base_art)
            rows_out.append(entry)
        f_scores = [c["factory"]["score"] for c in rows_out]
        summary = {"chunk_rows": size, "n_chunks": len(chunks), "factory_scores": f_scores,
                   "factory_mean": round(statistics.mean(f_scores), 6),
                   "factory_sd": round(statistics.stdev(f_scores), 6) if len(f_scores) > 1 else None,
                   "selection_score": best["score"]}
        if args.control_base:
            b_scores = [c["base"]["score"] for c in rows_out]
            summary.update({"base_scores": b_scores, "base_mean": round(statistics.mean(b_scores), 6),
                            "base_sd": round(statistics.stdev(b_scores), 6) if len(b_scores) > 1 else None})
            report["base_selection"] = score(selection, "UNTRAINED base on selection rows",
                                             weights=best["model_id"], art=base_art)
        report["chunks"] = rows_out
        report["chunk_summary"] = summary
        log(f"chunk summary: {summary}")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        log(f"wrote {args.out}")
        return 0

    pools = [("unseen", unseen)] + ([("unused_train", unused_train)] if args.score_unused_train else [])
    samples = {}
    for name, pool in pools:
        if len(pool) < args.min_rows:
            report[name] = None
            log(f"{name}: only {len(pool)} rows (< {args.min_rows}); not scored")
            continue
        sample = pool if len(pool) <= args.max_rows else rng.sample(pool, args.max_rows)
        samples[name] = sample
        report[name] = score(sample, f"{name} sample (seed {args.seed})")

    if args.control_base and samples:
        base_art = None
        if quant:
            base_art = _build_quant_artifact_for_eval(best["model_id"], best["model_id"], quant,
                                                      f"{best['model_id']} [{quant}] base")
        report["base_selection"] = score(selection, "UNTRAINED base on selection rows",
                                         weights=best["model_id"], art=base_art)
        # Every scored pool gets the base model on the SAME rows, so fine-tuned vs base is paired
        # even where the pool itself is not distribution-matched (unused training-split rows).
        for name, sample in samples.items():
            report[f"base_{name}"] = score(sample, f"UNTRAINED base on {name} sample",
                                           weights=best["model_id"], art=base_art)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
