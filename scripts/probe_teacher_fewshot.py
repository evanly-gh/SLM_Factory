"""Measure the Qwen3.6 teacher zero-shot vs few-shot on a task's own eval set.

WHY
    The accuracy goal, and every argument about whether the teacher is fit to generate synthetic
    data, currently rests on ONE number: its ZERO-SHOT score
    (`eval/endpoint_eval.py::measure_endpoint_baseline`). On BC5CDR that number is 0.0999, which
    reads as "the teacher cannot do biomedical NER".

    That conflates two very different claims. Exact-span NER is dominated by output-format
    compliance, so a zero-shot score mostly measures whether the model guessed our JSON contract
    and our span conventions — not whether it can find entities. Few-shot prompting supplies the
    convention explicitly. If the gap is large, the teacher's competence was never the problem and
    the "is the teacher good enough to generate data" question needs a different input than the
    zero-shot eval score.

WHAT IT DOES
    Scores the same frozen eval set three ways through the same scorer the pipeline uses, so the
    numbers are directly comparable to the run logs:
      * 0-shot  — byte-identical to what `measure_endpoint_baseline` does today
      * k-shot  — the same prompt with k demonstrations prepended, drawn from TRAIN only
    Demonstrations come from the task's train split, never from the eval set, so this cannot leak.

    Prompts, extraction, scoring and the token reserve all come from the task's own `TaskSpec`, so
    the probe measures the task the pipeline actually runs rather than a channel-shaped
    approximation of it.

USAGE (needs a reachable vLLM endpoint; see scripts/probe_teacher_fewshot.slurm)
    python scripts/probe_teacher_fewshot.py --task ner_bc5cdr --shots 0 1 3 5 --n 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _training_context(rows: list[dict], spec):
    """The dataset-level context the task's turn builder needs (label vocabulary, instruction).

    Resolved exactly as `training.lora_trainer` resolves it, so a demonstration shows the model
    the same prompt/answer pair fine-tuning would have shown it.
    """
    from eval.scorers.generation import resolve_generation_instruction
    from tasks._builders import TrainingContext

    labels = tuple(sorted({
        str(row.get("label", "")) for row in rows if row.get("label")
    })) if spec.closed_label_space else ()
    return TrainingContext(labels=labels, instruction=resolve_generation_instruction(rows))


def _corrupt_gold(row: dict, rng, pool: list[dict]) -> dict:
    """A copy of `row` whose ANSWER is wrong but whose FORMAT is perfect.

    This is the Min et al. (arXiv:2202.12837) manipulation, adapted to exact-span extraction. They
    showed that for classification, "randomly replacing labels in the demonstrations barely hurts
    performance", because demonstrations work by specifying the label space, the input distribution
    and the output format rather than by teaching the input→label mapping.

    Applying it here separates two explanations of the 0.1131 → 0.7190 few-shot gain:

      * FORMAT — the demonstrations taught the JSON shape, the `Chemical`/`Disease` vocabulary and
        the absence of a markdown fence. If so, corrupting WHICH spans are labelled should barely
        matter, and the score should stay high.
      * KNOWLEDGE — the demonstrations taught the teacher what a biomedical entity looks like. If
        so, corrupting the spans should collapse the score back toward zero-shot.

    The corruption keeps everything about the format identical: still a JSON list, still
    `{"text": ..., "type": ...}`, still only the types this row already used, still the same number
    of spans. Only the span TEXT is replaced — with entity strings borrowed from OTHER rows, so they
    are real terms that simply do not occur in this row's sentence. The demonstration is therefore
    perfectly formatted and factually wrong, which is exactly the contrast we want.

    Only span-labelled rows can be corrupted this way, so the trigger is the row carrying an
    `entities` list rather than the task's identity: an equivalent manipulation for a summary or a
    function call would be a different experiment, not this one, and silently corrupting them some
    other way would report a number this probe's verdict text cannot interpret.
    """
    entities = row.get("entities")
    if not isinstance(entities, list) or not entities:
        return row
    donors = [
        entity
        for other in rng.sample(pool, min(24, len(pool)))
        if other is not row
        for entity in (other.get("entities") or [])
    ]
    if not donors:
        return row
    swapped = []
    for entity in entities:
        donor = rng.choice(donors)
        swapped.append({"text": donor.get("text"), "type": entity.get("type")})
    return {**row, "entities": swapped}


def _demo_block(demos: list[dict], spec, ctx) -> str:
    """k demonstrations in the same prompt/answer shape the model is about to be asked for.

    Built from the task's `build_training_turn`, which imports its prompt from the eval scorer and
    renders the gold answer the way the model is trained to emit it. Writing the demonstration
    format out by hand here would make the probe measure a prompt nothing else in the pipeline
    sends.
    """
    parts = []
    for row in demos:
        prompt, target, _marker = spec.build_training_turn(row, ctx)
        parts.append(f"{prompt}\n{target}")
    return "\n\n".join(parts)


def main() -> int:
    import tasks

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ner_bc5cdr", choices=tasks.task_names())
    ap.add_argument("--shots", type=int, nargs="+", default=[0, 1, 3, 5])
    ap.add_argument("--corrupt-shots", type=int, nargs="*", default=[],
                    help="also run these k with CORRUPTED demonstration answers (perfect format, "
                         "wrong spans) — the Min et al. ablation separating format from knowledge; "
                         "implemented for span-labelled tasks only")
    ap.add_argument("--n", type=int, default=200, help="eval rows to score per condition")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from data.eval_set import EvalSet, build_eval_set
    from data.synth_client import get_generate_fn, is_available
    from eval.endpoint_eval import _generate_all
    from eval.harness import eval_output_token_reserve

    spec = tasks.get_task(args.task)

    if not is_available(log=print):
        print("FATAL: no reachable synth endpoint (set SLM_SYNTH_ENDPOINT).")
        return 3
    generate_fn = get_generate_fn(log=print)

    # Load the task through its own loader, so the eval set is the one the pipeline would score.
    train, test = spec.load(max_train=2000, max_test=max(args.n, 200), log=print)
    eval_set = build_eval_set(test, task=spec.name, target=args.n)
    rows = list(eval_set.all)[: args.n]
    eval_set = EvalSet(all=rows, task=spec.name)

    base_prompts = spec.build_prompts(eval_set)
    max_tokens = eval_output_token_reserve(spec.name)
    metric = spec.metric_name
    demo_ctx = _training_context(train, spec)
    rng = random.Random(args.seed)

    print(f"\n=== teacher few-shot probe: {spec.name} ({spec.title}, metric={metric}) ===")
    print(f"    model  : {os.environ.get('SLM_SYNTH_MODEL', 'Qwen/Qwen3.6-35B-A3B')}")
    print(f"    rows   : {len(rows)} eval, {len(train)} train available for demos")
    print(f"    tokens : max_new_tokens={max_tokens}\n")

    corrupt_shots = list(args.corrupt_shots)
    if corrupt_shots and not any(isinstance(r.get("entities"), list) for r in train):
        # Saying so beats running the conditions and printing a FORMAT-vs-KNOWLEDGE verdict about
        # demonstrations that were never actually corrupted.
        print("  note: --corrupt-shots ignored — the corruption is defined for span-labelled rows "
              f"and {spec.name} rows carry no `entities`")
        corrupt_shots = []

    conditions = [(k, False) for k in args.shots] + [(k, True) for k in corrupt_shots]
    results = []
    for k, corrupt in conditions:
        if k > 0 and not train:
            print(f"  skip k={k}: no train rows for demonstrations")
            continue
        if corrupt and k == 0:
            continue  # nothing to corrupt with zero demonstrations
        prompts = base_prompts
        if k > 0:
            # Fresh demonstrations per eval row, drawn from TRAIN only — never the eval set, so a
            # demonstration can never be the row being scored.
            prompts = []
            for base in base_prompts:
                demos = rng.sample(train, min(k, len(train)))
                if corrupt:
                    demos = [_corrupt_gold(d, rng, train) for d in demos]
                prompts.append(
                    "Here are worked examples showing the exact output format expected:\n\n"
                    + _demo_block(demos, spec, demo_ctx)
                    + "\n\nNow do the same for this one.\n\n"
                    + base
                )
        t0 = time.perf_counter()
        raw = _generate_all(generate_fn, prompts, max_tokens, 16)
        preds = spec.extract_predictions(raw, eval_set)
        scored = spec.score(eval_set, preds)
        elapsed = time.perf_counter() - t0
        # An empty prediction is the format-failure signature for every task here: the extractors
        # return an empty container or the __EXTRACTION_FAILED__ sentinel when they cannot read the
        # output at all.
        empty = sum(1 for p in preds if not p or p == "__EXTRACTION_FAILED__")
        tag = f"{k}-shot" + (" CORRUPTED" if corrupt else "")
        results.append({
            "shots": k, "corrupted_demos": corrupt, "condition": tag,
            "score": scored["f1"], "metric": scored.get("metric", metric),
            "empty_or_failed": empty, "n": len(rows), "seconds": round(elapsed, 1),
        })
        print(f"  {tag}:  {scored.get('metric', metric)}={scored['f1']:.4f}   "
              f"empty/failed={empty}/{len(rows)}   ({elapsed:.0f}s)")
        if raw:
            print(f"        sample raw output: {str(raw[0])[:160]!r}")
        if corrupt and prompts:
            # Show that the corrupted demonstration really is well-formed but wrong.
            head = prompts[0][:prompts[0].rfind("Now do the same")] if "Now do the same" in prompts[0] else prompts[0][:600]
            print(f"        corrupted demo tail: {head.strip()[-200:]!r}")

    if results:
        zero = next((r for r in results if r["shots"] == 0 and not r["corrupted_demos"]), None)
        best = max((r for r in results if not r["corrupted_demos"]), key=lambda r: r["score"],
                   default=None)
        # The verdict this probe exists to produce.
        for corrupt_run in [r for r in results if r["corrupted_demos"]]:
            clean = next((r for r in results
                          if r["shots"] == corrupt_run["shots"] and not r["corrupted_demos"]), None)
            if clean and zero:
                span = clean["score"] - zero["score"]
                kept = (corrupt_run["score"] - zero["score"]) / span if span else 0.0
                print(f"\n=== FORMAT vs KNOWLEDGE at k={corrupt_run['shots']} ===")
                print(f"  zero-shot                : {zero['score']:.4f}")
                print(f"  {corrupt_run['shots']}-shot, correct demos : {clean['score']:.4f}")
                print(f"  {corrupt_run['shots']}-shot, CORRUPTED     : {corrupt_run['score']:.4f}")
                print(f"  -> corrupted demos retain {kept:.0%} of the few-shot gain")
                if kept >= 0.6:
                    print("  VERDICT: mostly FORMAT. The demonstrations taught the output contract, "
                          "not the task, so the teacher's zero-shot score is not a valid gate on "
                          "its synthesis fitness.")
                elif kept <= 0.3:
                    print("  VERDICT: mostly KNOWLEDGE. The demonstrations carried task information, "
                          "so the low zero-shot score is real and should gate synthesis.")
                else:
                    print("  VERDICT: MIXED — both format and knowledge contribute materially.")
        print("\n=== summary ===")
        for r in results:
            delta = f"  (Δ vs 0-shot {r['score'] - zero['score']:+.4f})" if zero else ""
            print(f"  {r['shots']}-shot: {r['score']:.4f}{delta}")
        if zero and best["shots"] != 0:
            print(f"\n  BEST is {best['shots']}-shot at {best['score']:.4f}, "
                  f"{best['score'] - zero['score']:+.4f} over zero-shot "
                  f"({best['score'] / zero['score']:.1f}x)" if zero["score"] > 0 else
                  f"\n  BEST is {best['shots']}-shot at {best['score']:.4f}, "
                  f"up from {zero['score']:.4f} zero-shot")

    out = args.out or f"logs/probes/teacher_fewshot_{spec.name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"task": spec.name, "metric": metric,
                   "model": os.environ.get("SLM_SYNTH_MODEL", "Qwen/Qwen3.6-35B-A3B"),
                   "results": results}, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
