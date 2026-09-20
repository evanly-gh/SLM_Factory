"""Per-task preflight: does the data load, does the eval harness work, does the trainer accept it?

WHY THIS EXISTS
    Two runs have been lost to defects that a check like this would have caught in seconds:

      * `xlam_bfcl` and `calendar_json` loaded their data cleanly, measured a baseline, and then died
        at the FIRST training step on `completion-only SFT does not support task_type='function_call'`
        — a bare `raise` on the one code path both had to traverse. 68 and 42 minutes of GPU time.
      * `calendar_json` passed gold-vs-gold self-consistency and was still unfair: 82% of its gold
        answers encoded a year convention no model could infer. Self-consistency cannot catch that,
        because the gold agrees with itself by construction.

    So this checks four things per task, in the order they would fail in a real run, and NONE of them
    needs a GPU or a network model:

      1. LOAD      — the loader returns train and eval rows carrying the task's `required_fields`.
      2. EVAL      — gold predictions score 1.0 through the task's real scorer, and a DEGENERATE
                     prediction (always the majority class / empty answer) scores near 0. The second
                     half matters: a metric that rewards collapse cannot detect it.
      3. TRAIN     — `_training_turn` produces a (prompt, target) pair, and that prompt is
                     byte-identical to the one the eval harness would send for the same row.
      4. HYGIENE   — no train/eval text overlap, curriculum above the viability floor, eval-set size
                     against target, label distribution.

    Every one of those reads the TASK'S OWN SPEC (`tasks.get_task(name)`) rather than branching on an
    abstract channel. That is the point of the registry, and it is also what makes this script honest:
    it exercises the exact callables the pipeline will, so a preflight pass cannot mean "the channel I
    guessed for this task works".

USAGE
    python scripts/preflight_tasks.py                # every registered task
    python scripts/preflight_tasks.py --task ner_bc5cdr proactive_listening
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Predictions that must NOT score well. A task whose metric rewards these cannot detect collapse.
_DEGENERATE = "degenerate (majority class / empty answer)"


def _norm(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _training_context(rows: list[dict], spec):
    """The dataset-level context a training turn needs, resolved the way the trainer resolves it.

    Mirrors `training.lora_trainer._build_completion_only_rows` exactly — same label vocabulary,
    same instruction, both taken over the rows being turned into turns. Resolving it differently
    here would make the TRAIN parity check below compare two things the pipeline never compares.
    """
    from eval.scorers.generation import resolve_generation_instruction
    from tasks._builders import TrainingContext

    labels = tuple(sorted({
        str(row.get("label", "")) for row in rows if row.get("label")
    })) if spec.closed_label_space else ()
    return TrainingContext(labels=labels, instruction=resolve_generation_instruction(rows))


def _degenerate_raw(eval_set, spec) -> tuple[list[str], str]:
    """The laziest possible RAW model output for this task, and a human description of it.

    Raw text rather than ready-made predictions, so it goes through the task's own
    `extract_predictions` — the same path a real collapsed model's output takes. A task that
    collapses does not hand the scorer a tidy `[]`; it emits nothing, or the one word it always
    emits, and the extractor decides what that becomes.
    """
    rows = eval_set.all
    if spec.closed_label_space:
        majority = collections.Counter(row.get("label") for row in rows).most_common(1)[0][0]
        return [str(majority)] * len(rows), f"always {majority!r}"
    return ["" for _ in rows], "always empty"


# Tasks whose gold cannot score 1.0 against their own scorer, with the reason and the measured
# ceiling. Every other task must reach 1.0, which the 0.999 default below keeps strict.
#
# THIS TABLE IS DANGEROUS AND IS DELIBERATELY SHORT. A floor below 1.0 is precisely how a real
# scorer/gold mismatch hides. `calendar_json` passed self-consistency while 82% of its gold
# encoded a year convention no model could infer — self-consistency cannot catch that — and a
# lowered floor would conceal the reverse case too. An entry earns its place only by naming a
# STRUCTURAL reason the gap cannot be closed, never a bug someone intends to fix.
GOLD_SELF_CONSISTENCY_FLOOR = {
    # ERRANT compares EDIT LISTS, not strings. The reference edits are the annotator's own
    # segmentation, taken from the shipped M2 file; the hypothesis edits must be DERIVED from the
    # corrected sentence by `errant_parallel`'s alignment rules. Where a human merged two adjacent
    # corrections into one edit and the rules split them, the lists differ even though the
    # sentences are identical — measured at F0.5 0.8934 on 120 dev sentences, with 23 false
    # positives and 22 false negatives against 213 gold edits.
    #
    # Closing the gap would mean regenerating the REFERENCE side the same way, which does score
    # 1.0 and is the wrong trade: it replaces the corpus's gold annotation with our scorer's
    # opinion of it, and changes the number every published F0.5 is compared against. So the
    # ceiling is accepted and recorded instead. A system at 0.75 is at ~84% of achievable here,
    # not 75%.
    "gec_bea19": 0.85,
}


def check_task(name: str, *, n_eval: int | None, verbose: bool) -> dict:
    from data.eval_set import EvalSet, build_eval_set
    from tasks import get_task

    spec = get_task(name)
    # The task's own cap, not a script-wide constant: BC5CDR ships 5,865 held-out rows and
    # RouterBench 7,267, and each task picked its own point on the variance/iteration-cost
    # trade-off. `--n-eval` overrides for a one-off.
    n_eval = int(n_eval) if n_eval else spec.select_cap
    result: dict = {"task": name, "family": spec.family, "metric": spec.metric_name,
                    "ok": True, "notes": []}

    def fail(stage: str, detail: str) -> dict:
        result["ok"] = False
        result["notes"].append(f"{stage}: {detail}")
        return result

    # --- 1. LOAD ---------------------------------------------------------------
    train, test = spec.load(
        max_train=3250, max_test=n_eval,
        log=(print if verbose else (lambda *_a, **_k: None)),
    )
    result["n_train"] = len(train)
    result["n_test"] = len(test)
    if not train or not test:
        return fail("LOAD", f"train={len(train)} test={len(test)} — need both")

    missing = [f for f in spec.required_fields if not all(f in r for r in train[:50])]
    if missing:
        return fail("LOAD", f"train rows missing required field(s) {missing}")

    # --- 2. EVAL ---------------------------------------------------------------
    eval_set = build_eval_set(test, task=spec.name, target=n_eval)

    # Gold-vs-gold, reconstructed from the TRAINING TARGET rather than from a hand-written guess at
    # what a perfect answer looks like. The training target is by definition the string the model is
    # taught to emit, so feeding it through the eval extractor and scorer asks the one question that
    # actually matters: does what we teach score 1.0 on what we grade? A hand-written reconstruction
    # can agree with the scorer while the trainer teaches something else entirely.
    eval_ctx = _training_context(eval_set.all, spec)
    if spec.needs_judge:
        # Scored by an LLM judge, which cannot run without a live endpoint. Skip the scoring half
        # rather than reporting a task defect — LOAD, TRAIN and HYGIENE still run, and they are
        # where the run-killing defects have actually been.
        result["gold_score"] = "judge"
        result["degenerate_score"] = "judge"
        result["degenerate_how"] = "n/a"
        result["notes"].append(
            "EVAL: SKIPPED — this task is scored by an LLM judge, which needs a live endpoint. "
            "Run with ANTHROPIC_API_KEY set, or on a node with the local judge, to check it."
        )
    else:
        try:
            gold_raw = [spec.build_training_turn(row, eval_ctx)[1] for row in eval_set.all]
        except Exception as exc:  # noqa: BLE001
            return fail("EVAL", f"could not render a gold answer for every eval row "
                                f"({type(exc).__name__}: {exc})")
        gold = spec.extract_predictions(gold_raw, eval_set)
        gold_score = spec.score(eval_set, gold)["f1"]
        result["gold_score"] = round(gold_score, 4)
        floor = GOLD_SELF_CONSISTENCY_FLOOR.get(name, 0.999)
        if gold_score < floor:
            fail("EVAL", f"gold-vs-gold scored {gold_score:.4f}, expected at least {floor} — the "
                         f"eval set and the scorer disagree about what a correct answer looks like")
        elif floor < 0.999:
            result["notes"].append(
                f"EVAL: gold-vs-gold is {gold_score:.4f}, not 1.0, and that is EXPECTED for this "
                f"task — see GOLD_SELF_CONSISTENCY_FLOOR. Read scores against that ceiling."
            )

        degenerate_raw, how = _degenerate_raw(eval_set, spec)
        degen_score = spec.score(eval_set, spec.extract_predictions(degenerate_raw, eval_set))["f1"]
        result["degenerate_score"] = round(degen_score, 4)
        result["degenerate_how"] = how
        if degen_score > 0.35:
            fail("EVAL", f"{_DEGENERATE} ({how}) scored {degen_score:.4f} — the metric rewards "
                         f"collapse, so a collapsed model would look acceptable")

    # --- 3. TRAIN --------------------------------------------------------------
    try:
        from training.lora_trainer import _training_turn

        train_ctx = _training_context(train, spec)
        prompt, target, _marker = _training_turn(train[0], spec.name, train_ctx)
        if not str(prompt).strip():
            fail("TRAIN", "_training_turn produced an empty prompt")
        if not str(target).strip():
            fail("TRAIN", "_training_turn produced an empty target — the model would be taught "
                          "to emit nothing")
        result["train_prompt_chars"] = len(prompt)
        result["train_target"] = str(target)[:80]

        # Train/serve parity. Every task's training builder imports its prompt from the eval scorer,
        # so the two are supposed to be incapable of drifting — this proves it for real rows rather
        # than trusting the import.
        #
        # The eval prompt is built over the TRAINING rows on purpose. Both sides resolve their
        # dataset-level context (the enumerated label vocabulary, the generation instruction)
        # independently from whatever rows they are given, so comparing against a one-row eval set
        # would report skew for every classification task simply because one row has one label.
        # Whether the two VOCABULARIES agree is a separate question, checked below.
        eval_prompt = spec.build_prompts(EvalSet(all=list(train), task=spec.name))[0]
        if spec.closed_label_space:
            train_vocab = {str(r.get("label")) for r in train if r.get("label") is not None}
            eval_vocab = {str(r.get("label")) for r in eval_set.all if r.get("label") is not None}
            if train_vocab != eval_vocab:
                only_train = sorted(train_vocab - eval_vocab)[:6]
                only_eval = sorted(eval_vocab - train_vocab)[:6]
                fail("TRAIN", f"label VOCABULARY differs between train and eval, so the enumerated "
                              f"label list in the prompt differs too — train/serve skew. "
                              f"train-only={only_train} eval-only={only_eval}")
            elif prompt != eval_prompt:
                fail("TRAIN", "training prompt is NOT byte-identical to the eval prompt despite a "
                              "matching label vocabulary")
            else:
                result["notes"].append(
                    f"TRAIN: prompt byte-identical to eval ✓ (label vocabulary matches, "
                    f"{len(train_vocab)} classes)"
                )
        elif prompt != eval_prompt:
            fail("TRAIN", "training prompt is NOT byte-identical to the eval prompt — "
                          "train/serve skew (this is how the BC5CDR prompt drift survived a "
                          "44.8-hour run)")
        else:
            result["notes"].append("TRAIN: prompt byte-identical to eval ✓")
    except Exception as exc:  # noqa: BLE001
        fail("TRAIN", f"{type(exc).__name__}: {exc}")

    # --- 4. HYGIENE ------------------------------------------------------------
    overlap = len({_norm(r.get("text", "")) for r in train}
                  & {_norm(r.get("text", "")) for r in eval_set.all})
    result["train_eval_overlap"] = overlap
    if overlap:
        # A WARNING, not a failure: curate's eval firewall removes these rows before training, so
        # the run is not contaminated. It is still worth surfacing — an official split that is not
        # actually disjoint means the loader is shipping rows the firewall then silently deletes, and
        # in DialogSum's case the same dialogue carried CONTRADICTORY gold in the two splits (B278).
        result["notes"].append(
            f"HYGIENE: {overlap} train row(s) appear verbatim in the eval set at the LOADER level. "
            f"curate's eval firewall removes them before training, so this does not contaminate the "
            f"run — but the loader should not be shipping them."
        )

    from agent.nodes.curate import MIN_CURRICULUM_ROWS

    if len(train) < MIN_CURRICULUM_ROWS:
        fail("HYGIENE", f"only {len(train)} train rows, below the {MIN_CURRICULUM_ROWS}-row "
                        f"curriculum floor — curate would refuse this task")
    if len(eval_set.all) < n_eval:
        result["notes"].append(
            f"HYGIENE: eval set is short — {len(eval_set.all)}/{n_eval} rows "
            f"({len(eval_set.all) / n_eval:.0%} of target). Accepted; scores carry more variance."
        )
    if spec.closed_label_space:
        dist = collections.Counter(r.get("label") for r in eval_set.all)
        result["eval_label_dist"] = dict(dist.most_common(6))
        if len(dist) < 2:
            fail("HYGIENE", "eval set has only one class — nothing to discriminate")
    return result


def main() -> int:
    import tasks

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="*", choices=tasks.task_names(), default=tasks.task_names())
    # Defaults to each task's own `select_cap`, so the preflight sees the same eval set the pipeline
    # would build. A single number here would misreport every task whose cap is not that number.
    ap.add_argument("--n-eval", type=int, default=None,
                    help="eval rows per condition; default is each task's own TaskSpec.select_cap")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache")
    results = []
    for name in args.task:
        print(f"\n{'=' * 78}\n### {name}\n{'=' * 78}")
        try:
            res = check_task(name, n_eval=args.n_eval, verbose=args.verbose)
        except Exception as exc:  # noqa: BLE001
            res = {"task": name, "ok": False, "notes": [f"CRASHED: {type(exc).__name__}: {exc}"]}
            if args.verbose:
                traceback.print_exc()
        results.append(res)
        status = "PASS" if res["ok"] else "FAIL"
        print(f"  [{status}] train={res.get('n_train','?')} eval={res.get('n_test','?')}  "
              f"gold={res.get('gold_score','?')}  degenerate={res.get('degenerate_score','?')} "
              f"({res.get('degenerate_how','?')})")
        if "eval_label_dist" in res:
            print(f"         eval labels: {res['eval_label_dist']}")
        if "train_target" in res:
            print(f"         train target sample: {res['train_target']!r}")
        for note in res["notes"]:
            print(f"         - {note}")

    print(f"\n{'=' * 78}\n### SUMMARY\n{'=' * 78}")
    width = max(len(r["task"]) for r in results)
    for r in results:
        print(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['task']:<{width}}  "
              f"train={str(r.get('n_train','?')):>5}  eval={str(r.get('n_test','?')):>4}  "
              f"{str(r.get('metric','?')):>12}  gold={str(r.get('gold_score','?')):>6}  "
              f"degen={str(r.get('degenerate_score','?')):>6}")
    failed = [r["task"] for r in results if not r["ok"]]
    print(f"\n  {len(results) - len(failed)}/{len(results)} passed"
          + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
