# task registry rebuild

*2026-08-19 — implementation + answers*

Follows [08-17b-xlam-single-debug.md](08-17b-xlam-single-debug.md). That note diagnosed why the xlam
run's reports described a different run than the one that happened. This one replaces the machinery
that made it possible.

**Headline: the `task_type` channel abstraction is gone.** Eight concrete benchmarks used to share
five abstract channels, so behaviour was decided by `if task_type == ...` chains — 47 of them — and a
task got whatever the chain's `else` branch happened to do. That is now a per-task registry in
`tasks/` where every field is required, so "we never considered this for that task" is an import-time
error instead of a silent runtime fallthrough. Along the way: quality control turned out to be a
**no-op for four of the eight tasks**, the curriculum is now cumulative, `data_rebuild` has exactly
two sub-strategies, and mining can finally reach the ~57,000 xLAM rows that were sitting unreachable
in the local cache.

Suite: **1,317 passing**, rebuilt from scratch. Docs: new [interventions.md](../interventions.md).

---

## 1. Your questions, answered

### What are xLAM and BFCL? (Q4)

Two separate corpora that make one task. **xLAM** is Salesforce's `xlam-function-calling-60k` —
60,000 examples of *(user request + available tool schemas → correct JSON function call)*. It is the
**training** source. **BFCL** is the Berkeley Function-Calling Leaderboard (`gorilla-llm/...`), the
standard public **evaluation** suite for tool calling. So `xlam_bfcl` means: train on xLAM, score on
BFCL. They were built independently, which is why train/eval contamination on this task is genuinely
low — the 15 rows the firewall removes are coincidental overlaps, not a leak.

### The 3,250 split (Q4)

Hardcoded, and now gone. It was `curriculum_size_target × 0.65` — 65% of a 5,000-row "target" that
the curriculum was then never allowed to reach, because the only thing that could have closed the gap
was re-drawing rows it already had. Every task now declares `initial_train_cap=5000` and
`eval_cap=1000`, and the loader returns as many as the source has up to those numbers. No fraction,
no split, no target.

### Gold fill re-drawing from the same split (Q4)

There was no point, and this was the deeper version of the same defect. Because the curriculum was
*rebuilt from scratch* to a target size every iteration, something had to refill it from the train
pool. With nothing else changed it re-selected the identical ~3,235 rows and honestly reported
`0 novel` on eight consecutive rebuilds.

**The curriculum is now cumulative.** Cold start loads gold rows; every rebuild ADDS; rows leave only
via quality control or the eval firewall. `_dedupe_into` is the single place it grows.

### Which tasks are judged on format? (Q6)

Your guess — only calendar and xlam — is too narrow. **Seven of eight** have a real parse step, and
`format_valid` is now reported for all of them, in every per-iteration log line and in the final
report:

| task | what "format valid" means | needs parsing? |
|---|---|---|
| xlam_bfcl, calendar_json | output parses as a JSON list of `{name, arguments}` | yes |
| ner_bc5cdr | output parses as a JSON array of `{text, type}` | yes |
| clinc150, routerbench, proactive_listening | an in-vocabulary label was extractable | yes |
| **gsm8k** | a final numeric answer was extractable | **yes** |
| dialogsum | non-empty output; no contract to satisfy | no |

**gsm8k absolutely needs parsing.** `_final_answer` regexes the number out of the working. A model
that reasons correctly and never states a parseable number is a *format* failure — the fix is the
prompt or the answer marker, not more data. Reporting only content made those indistinguishable,
which is exactly how B290 hid for two runs: every prediction carried two stray `<think>` tags, so
content collapsed while format told the real story.

While adding this I found NER had a related defect: `extract_predictions` returned `[]` both for
"parsed, no entities" and "did not parse at all", so unparseable prose was indistinguishable from a
correct empty prediction. It now returns `None` for a parse failure.

### The overlap confusion (Q9) — you were right, the check was incoherent

That `(80 rows)` was **not** overlap with our eval set. It compared the *discovered dataset's own*
train slice against its *own* test slice, and since `_materialize_from_mapping` sliced both from the
front of the same underlying split, the test rows were a subset of the train rows **by
construction** — so it always "found" overlap exactly equal to `max_test` and rejected every
candidate. Two different things were conflated:

1. mined rows vs our existing curriculum → that is **deduplication**, and it is what `_dedupe_into`
   does per row;
2. mined rows vs our held-out eval set → that is the **eval firewall**, also per row.

The source's internal split structure is irrelevant, because mining consumes only the train side. The
check is gone. So are two other whole-source rejections: a dataset is no longer refused for having
columns we do not need, and no longer refused for a single-class slice. Labels are filtered per row
too — a source whose mapping got most rows right keeps those rows, and is refused only when
*nothing* survives.

### What budget? (Q9, second)

`MAX_PAID_ACQUIRE_ROUNDS_PER_RUN = 9` plus a durable per-plan ledger. It bounded **paid Exa calls
during dataset discovery** — searching the hub for a corpus we do not have. You are right that it had
no business on the re-read path: re-reading a dataset we already sourced costs nothing. The ledger is
deleted. What is metered now is failure, not spend: after **2** consecutive discovery rounds that
contribute zero novel rows, mining is retired for the run. The `str(e)[:80]` truncation and the
`candidates[:6]` shortlist are also gone — every candidate Exa returns is probed, and errors are
logged whole.

---

## 2. Why mining and synthesis were both adding nothing (Q8)

Four independent causes, all fixed.

**Synthesis produced nothing** because `function_call` appeared in neither branch of
`synthesize_examples`' dispatch and fell through to a bare `return []`. Six rebuilds announced
250–500 rows against a healthy teacher endpoint and produced zero, silently. That also meant the
exact verifiers written for that path had never executed in production. (B291, fixed in the previous
note; the dispatch table it lived in no longer exists.)

**Mining produced nothing** for three separate reasons, each sufficient on its own:

1. **No way back to its own corpus.** The initial load took the first 3,250 of xLAM's 60,000 rows and
   `train_examples` was never re-sliced. `acquire` went straight to paid discovery — which then found
   *mirrors of xLAM* and rejected them.
2. **The canonical sources were dropped unprobed.** Exa returned
   `Salesforce/xlam-function-calling-60k` and `gorilla-llm/Berkeley-Function-Calling-Leaderboard` at
   positions 7 and 8; the loop probed `candidates[:6]`.
3. **The self-inflicted overlap rejection** above killed every mirror that did load.

**Now:** rung 1 of the ladder re-reads the task's own datasets for rows the curriculum has not taken.
Every loader takes a head slice, so asking for a larger slice returns a superset and the tail is
novel by construction — no provider call, no LLM mapping, no schema risk. `state["source_progress"]`
records what we consumed per source, and a source is marked exhausted only when the loader returns
*fewer* rows than asked for.

**And when a rebuild adds zero rows it is now an ERROR**, not a line among twenty:

```
✗ ERROR: data_rebuild/mine_new_real added 0 new rows. The curriculum is unchanged at 3235 row(s),
  so training this iteration would repeat the previous one exactly. See the [mine]/[synth] lines
  above for which stage produced nothing.
```

---

## 3. The two sub-strategies (Q10)

`docs/interventions.md` is the full reference. In brief, `data_rebuild` is now exactly two things:

**`mine_new_real`** — a ladder:
1. re-read datasets already sourced that are not exhausted;
2. only if all are exhausted, web research for a dataset never used, with the orchestrator judging
   fit and mapping columns;
3. after 2 consecutive fruitless discovery rounds, **retired** for the run — the orchestrator is told
   so in its prompt, and a plan still asking for it is rewritten to surgical synthesis.

Both exhaustion events are logged loudly, because "mining added nothing" and "mining had nothing left
to add" are different facts that the run used to report identically.

**`surgical_synthesis`** — generate rows aimed at the failure **categories** costing the most points.
Budget per category is proportional to its failure count; a category already targeted whose count did
not fall is skipped as exhausted. The categories come from each task's own scorer, so they name
something measured — `wrong_arguments`, `entities_hallucinated`, `no_numeric_answer`, a real class
confusion — rather than the constant `gold_verifier -> incorrect` that every open-ended task used to
report and that the orchestrator wrote pages of reasoning about.

Gone: `resample`, the universal gold fill, untargeted balanced synthesis, `target_rows`,
`resample_fraction`, `new_real_rows`, `synth_rows`, `max_acquire_rounds`, `difficulty_buckets`. A plan
that sets any of them is **rejected**, not silently trimmed — a plan written against the wrong
contract means the orchestrator believes it asked for something it did not.

### Verification, and the task brief

You asked for programmatic checks on the format-bound tasks and teacher cross-verification elsewhere,
with the verifier prompt carrying the task description. Both done.

| task | exact verifier | what it proves for free |
|---|---|---|
| xlam_bfcl | `verify_function_call_row` | parses; calls a *declared* tool; arguments exist in its schema; required ones present |
| calendar_json | `verify_calendar_row` | the above, plus datetimes parse, `end` after `start`, unstated duration is 60 min, event resolves near the request's own reference instant |
| **ner_bc5cdr** | `verify_ner_row` *(new)* | every span appears **verbatim** in the row's own text, no duplicates — this catches the dominant teacher error, a plausible entity that was never written down |
| the other five | none | only the teacher's judgement; logged as such rather than implied |

NER span synthesis looked unverifiable and is not, which is why bc5cdr went from "gold-only by
design" to having a verifier. What the check *cannot* catch is a **missed** entity, so the teacher
pass still runs and that limit is stated in the code rather than left implicit.

**The task brief** (`agent/task_brief.py`) is new and is what you asked for. At cold start, after real
data is loaded, the orchestrator is shown real rows and writes: what the benchmark is, the exact
output contract, and the likely failure modes. It is logged in full and every synthesis and
verification prompt is built from it. It replaced a one-line table keyed by task *type*, under which
xlam and calendar were described identically — omitting every convention that makes a calendar row
correct, so a verifier judging against it was judging its own guess (B269, calendar synthesis at
0.2176).

Worked examples shown to the teacher are always **real rows**, never orchestrator-invented: a wrong
example is worse than none.

---

## 4. Eight bugs I introduced, and the scan that now catches them

The refactor landed with eight real bugs. Four broke every run. All are fixed, but the honest lesson
is in *why they were invisible*: every one was a runtime failure that `import` cannot see — a lazy
import inside a function, an undefined name on one branch, a keyword argument the callee had dropped.

| # | Site | Effect |
|---|---|---|
| 1 | `checkpoint.py` imported a deleted constant | `runtime_config_snapshot()` raised at module scope in the runner — **every run died before the graph was built** |
| 2 | two encoders read removed `EvalSet` fields | **no checkpoint could be written, no run could resume** |
| 3 | `curate.py` used `hashlib` after its import was removed | the **eval firewall raised on the first row it blocked** — the safety mechanism killing the run at the moment it caught a leak |
| 4 | `web_acquire.py` read `_spec` after the assignment moved | rung 2 of the mining ladder raised on entry |
| 5 | same for `_closed_label_space` | the B259 label guard could not run |
| 6 | `annotate_cot(task=...)` after the parameter was dropped | `TypeError` on every gsm8k rebuild once the CoT teacher was reachable |
| 7 | `fallback_data_rebuild_plan(score=, mining_available=)` | `TypeError` on the orchestrator-failure route — the path that exists so a bad reply cannot stop the run |
| 8 | `accept(discovered[0], ...)` unwrapped one level too far | rung 2 could never accept a row; every discovery round reported `no_novelty`, retiring mining after two rounds while a usable dataset went unused |

`scripts/check_unresolved_names.py` now scans for exactly these shapes — unbound name loads and
keyword-argument mismatches against same-file callees — across all 95 production modules. It is
self-tested against all five shapes and currently reports zero. Worth running before any refactor
that moves code between functions; a module-level import smoke test would have caught none of them.

---

## 5. Quality control was a no-op for half the suite

Found while checking your diagnosis, and it is the strongest evidence for it. `apply_quality_controls`
was one `if task_type == ...` chain ending in `else: return dataset`:

| task | before | why |
|---|---|---|
| xlam_bfcl, calendar_json | **no QC at all** | fell into the `else`, returned untouched |
| gsm8k, dialogsum | **no effective QC** | entered their branch but filtered length and duplicates on a `"prompt"` key their rows do not carry — a 100,000-character row survived, and nothing was logged |
| clinc150, routerbench, proactive_listening, ner_bc5cdr | worked | — |

Four of eight, silently. Nothing distinguished "this task chose not to deduplicate" from "this task
fell through a branch nobody updated".

Now each task lists its steps explicitly in `TaskSpec.quality_controls` (`data/quality_controls.py`).
An empty tuple is a legal, visible choice; falling through is impossible because there is no branch.
A step that cannot find the field it was told to filter on says so loudly instead of passing. xlam and
calendar also gained `valid_json_answer`, which matters most there — a gold answer that does not parse
trains the model to emit something the scorer marks wrong no matter what it predicts.

---

## 6. Other open issues

Plainly, in rough priority order.

1. **`calendar_json`'s mining source is a placeholder.** I declared `TOPv2/reminder` as its corpus,
   but TOPv2 is a GitHub tarball, not a hub dataset, and I have not verified the loader can be asked
   for a larger slice. Rung 1 may be a no-op for calendar until that is checked.
2. **`allow_paid_discovery=True` is now set for all eight tasks.** For `routerbench` and
   `proactive_listening` the label is *derived* (from whether a small model answered correctly; from
   an interruption judgement), so no other corpus carries it natively. Discovery there can only find
   data whose labels an LLM must invent. The per-row closed-label filter should catch that, but it is
   worth deciding whether those two should simply be `False`.
3. **No live run has exercised any of this.** Everything above is verified by 1,317 tests and static
   analysis. The suite cannot cover real inference, real training, the live judge, or an actual Exa
   discovery round. Bugs 1–8 are a fair warning about the gap between "tests pass" and "a run works",
   and the next thing worth doing is a short single-task run rather than more code.
4. ~~The `data_sizing` module is vestigial.~~ **Resolved.** It turned out to be worse than
   vestigial — `resize_curriculum_for_tier` had no caller at all, so the novelty × capacity formula
   never ran. Deleted, along with `CURRICULUM_SIZE_FLOOR`, `EVAL_SET_SIZE`, `DATA_SIZE_CEILING`,
   `curriculum_size_target`, `eval_size_target` and `data/acquisition_budget.py` (also zero
   importers). The only thing that ever read the computed target was the `× 0.65` split behind the
   3,250. The autonomous initial-acquisition path went with them: every run now names a registry
   task, and a task with no loader has no scorer, no training prompt and no synthesis path either.
5. **The difficulty buckets still do not drive anything.** They are computed once at cold start, feed
   reporting and the orchestrator prompt, and `plan["difficulty_buckets"]` was removed because
   nothing ever read it back to sample with. Per-difficulty *targeting* is now expressed through
   failure categories instead, which is more direct — but the easy/medium/hard split is still
   advisory only, and small-right/big-wrong rows are still bucketed as `hard` (B298).
6. **The autonomous discovery path is now nearly unreachable** and carries real weight —
   `web_acquire.py` is still the largest module in the repo. With every task curated and mining
   preferring known sources, most of it only runs on rung 2. Worth deciding whether the paper needs
   it before it decays further.
7. ~~`docs/PAPER.md` and `docs/PIPELINE.md` describe the old three-strategy design.~~ **Resolved** —
   PIPELINE.md, DATA_CURATION_AND_CAPS.md, PROMPTS.md, PAPER.md and model_pool.md were all swept, and
   B299–B305 were added to BUGS.md. `interventions.md` remains the authority for the loop; the others
   cross-reference it rather than restating it.
8. **A ninth latent bug class, now guarded.** Every one of the eight bugs above was invisible to
   `import`. `scripts/check_unresolved_names.py` is a static scan for exactly those shapes and is
   worth running before any refactor that moves code between functions — it is self-tested against
   all five shapes and currently reports 0 across 93 modules. It is not wired into CI; there is no CI.
