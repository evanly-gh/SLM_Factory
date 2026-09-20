# Calendar converged at 0.8748, and four loop defects it exposed on the way

**Date:** 2026-08-30
**Run:** `slm-calendar-json-l40s-39294409` — COMPLETED, 9h02m, `ast_arg_match=0.8748` vs a 0.8600 goal
**Companions:** `08-23-calendar-variance-sms-spam-small-models.md` (§1 is the direct predecessor to §3 here),
`08-21-run-38708719-review.md`

The run succeeded, and its own attribution table makes the case for synthesis better than any
argument could:

```
surgical_synthesis       0.8168   starting point (1/1 kept)
hyperparameter          +0.0579   1/5 attempt(s) kept
FINAL                    0.8748
```

Tiers 1 and 2 spent **nineteen** hyperparameter attempts and never cleared 0.714. Tier 3's single
`surgical_synthesis` rebuild landed 0.8168 in one iteration. The teacher gate had `surgical_synthesis`
switched off for the first two tiers (teacher 0.6318 < the 0.80 gate); it only ran at all because
mining exhausted every source and the plan validator rewrote `mine_new_real` into it.

What follows is everything you asked about, in order, plus four defects worth fixing.

---

## 1. How the orchestrator already knows SmolLM2's xlam score

This line appears at model-selection time, before anything has been trained:

```
xlam_bfcl ast_arg_match (fine-tuned, ours): 45.0 [artifact=HuggingFaceTB/SmolLM2-360M-Instruct;
mode=LoRA r=16 a=32 lr=2e-4 ep=3 on 3000 gold rows;
protocol=slm-factory-probe-38765131/38765655-lora-fixed-recipe-300-eval-rows]
```

**It is not a prediction and not a published benchmark. It is our own measurement, from a probe job,
pasted into a text file that gets injected into the selection prompt.**

The chain:

| Step | Where |
|---|---|
| Probe measures 3 models × 2 tasks, one fixed LoRA recipe | `scripts/probe_small_models.py`, jobs **38765131** (zero-shot + fine-tuned) and **38765655** (quantized) |
| Numbers written by hand into a capability doc | `config/model_capabilities.md:95-96` |
| Doc loaded and cached | `config/model_capabilities.py:load_capability_doc` |
| Only the candidate models' sections spliced into the prompt | `config/model_capabilities.py:capability_sections` |

The doc entry it is quoting:

```95:97:config/model_capabilities.md
Measured: `ner_bc5cdr` span-F1 **0.7339** [bf16 0.7339] · `xlam_bfcl` ast_arg_match **0.4500**
[bf16 0.4500]. It is the **only** pool entry measured as lossless under Q4_K_M on both tasks.
On BC5CDR it beats the Qwen3.6-35B teacher's five-shot 0.7190 while being ~97x smaller.
```

Three things follow from that being a hand-maintained file:

1. **The provenance string is doing real work.** `measured by us on our own eval sets (NOT a published
   benchmark, not comparable to the rows above)` sits in the same line as MMLU-style figures
   specifically so the orchestrator does not average them together. That is B161 and it is holding.
2. **It is n=1 per cell on 300 eval rows.** `08-23` §4.3 is explicit that this is a screen, not a
   result. The orchestrator is nonetheless reasoning with it as a point estimate — on this run it
   wrote *"SmolLM2-360M-Instruct shows by far the best xlam_bfcl ast_arg_match (45.0)"* and picked
   accordingly. The pick happens to be right, but the confidence is not earned.
3. **It goes stale silently.** Nothing recomputes it and nothing checks it against run results. If a
   loader or scorer changes, the doc keeps asserting the old number.

**Worth noting the number is being used cross-task.** On *calendar_json* the orchestrator justified
its choice with the *xlam* and *NER* figures, because those are the only two tasks the probe covered.
That is a reasonable proxy — both are format-bound structured output — but it is a proxy, and it is
not labelled as one anywhere in the prompt.

---

## 2. The 14 rejected datasets — I checked all of them myself

Two rounds of web discovery probed 14 candidates and kept nothing. I queried the HuggingFace API and
`datasets-server` for every one, read sample rows, and judged them against what a `calendar_json` row
actually requires: an English natural-language scheduling request **plus a stated reference instant**,
answered by a single `calendar.events.insert` call whose `start`/`end` are absolute ISO-8601.

**Verdict: 12 of 14 rejections are substantively correct. Two are worth revisiting, one of which is a
real bug in the probe.**

| Dataset | Logged reason | What it actually is | Verdict |
|---|---|---|---|
| `Ehsanrs2/Forex_Factory_Calendar` | unsuitable | Forex economic calendar, CSV time-series | **Correct** |
| `huggingXG/forex_calendar` | split error | Same, per-year CSVs | **Correct** |
| `NEAR-AI/clawbench` | unsuitable | Agentic terminal tasks (BibTeX graphs, file IO). "calendar" match was spurious | **Correct** |
| `kzfastino/TaskBenchCalendarEvents` | unsuitable | A *table of events* (title, start, attendees). No NL request exists to map to | **Correct** |
| `vidhikatkoria/DA_SGD_Calendar` | unsuitable | SGD Calendar as **dialogue-act / response generation** (`context`, `response`, `act`). No API call | **Correct** |
| `yananchen/natural_plan__calendar_scheduling` | unsuitable | Natural Plan: constraint solving ("find a slot all attendees are free") | **Correct** |
| `clembench-playpen/natural-plan-calendar` | unsuitable | Same benchmark | **Correct** |
| `microsoft/ba-calendar` | unsuitable | BA-Calendar, constraint scheduling | **Correct** |
| `nvidia/Nemotron-RL-agent-calendar_scheduling` | unsuitable | Constraint scheduling with an agent loop | **Correct** |
| `asu-kim/conversation-calendar` | unsuitable | Raw chat transcripts as `.txt`, no labels | **Correct** |
| `NotoriousH2/calendar-agent-benchmark` | split error | **Genuinely a calendar tool-calling set** — but **Korean**, and its action space is search/delete/confirm, not a single insert | **Correct, narrowly** |
| `ConvLab/sgd` | split error | The real SGD. Script-based, dead under `datasets` 4.x | **Correct** (infrastructure, not judgement) |
| `nvidia/Nemotron-RL-Instruction-Following-Calendar-v2` | **"repo does not exist or is gated"** | **Public and ungated.** Content is constraint scheduling, so unsuitable anyway | **Right answer, wrong reason — see §2.1** |
| `WillHeld/top_v2` | unsuitable / unmappable | **This is the dataset the task already trains on** | **Right answer, wrong prose — see §2.2** |

**Prior art:** `docs/BUGS.md` B324 records the same audit on run 38735780 and reached the same
place — 13 of 14 correct, with the note that `orchestrator judged it unsuitable / unmappable` is
"too terse to audit ... confirming that required loading each dataset by hand." That is now twice.
The terse message is the thing to fix.

### 2.1 The Nemotron probe reported a false reason

The log says:

```
[acquire] nvidia/Nemotron-RL-Instruction-Following-Calendar-v2: SKIPPED — repo does not exist or is gated
```

It exists, it is `gated: False`, it has 369 downloads and ships `train.jsonl` / `validation.jsonl`. I
loaded its first row without credentials.

The dataset is still unsuitable — it is constraint scheduling (`{'event_id': 0, 'duration': 50,
'constraint': 'after 10am', 'min_time': '10:00'}`) with an agent that prints a JSON calendar state,
not a `calendar.events.insert` call. So the outcome is right.

But the message is wrong, and a wrong message here is expensive: "does not exist or is gated" reads as
*nothing we can do*, when the real cause is something in our loading path (most likely no `default`
config, or a features mismatch). Every future run will re-probe it, re-fail it, and re-log the same
misleading line. **The acquire step should report the actual exception rather than collapsing every
load failure into "does not exist or is gated".**

### 2.2 TOPv2 was rejected as "unmappable" — it is the training set

```
[acquire] WillHeld/top_v2: orchestrator judged it unsuitable / unmappable
```

`calendar_json` trains on TOPv2. The loader is `data/loaders/calendar_json.py`, and it hard-filters to
one domain:

```480:485:data/loaders/calendar_json.py
    """Load and convert the TOPv2 `reminder` domain. Parquet-native, no loading script."""
    ...
    reminder = raw.filter(lambda r: r["domain"] == "reminder")
    return convert_topv2_rows(reminder)[:max_train]
```

So the orchestrator was shown its own training source and called it unmappable. That is a prompt
problem — the discovery step does not tell the orchestrator which datasets are already in use, so it
cannot recognise one — but the practical question is whether any rows were left. Measured from the
`datasets-server` filter API:

```
alarm     20,430 rows    ← never touched
reminder  17,840 rows    ← the only domain the loader reads
event      9,170 rows    ← IN:GET_EVENT (querying), not creation
```

**`reminder` was reported exhausted at 3,900 rows against 17,840 available**, which is its own
question (the cap is `initial_train_cap`, and "exhausted" means *exhausted of rows not already in the
curriculum* — worth confirming that accounting is right).

The `alarm` domain is 20,430 rows the run never considered, and my first read of it was **wrong** — I
assumed it was structurally compatible because it carries the same `[SL:DATE_TIME ...]` slot. It is
not. I sampled 100 alarm rows through the filter API:

```
intents:  CREATE_ALARM 53 | DELETE_ALARM 19 | GET_ALARM 16 | SILENCE_ALARM 5 | SNOOZE_ALARM 4
CREATE_ALARM rows: 53   with SL:DATE_TIME: 49   with SL:TODO: 0
```

**Zero of 53 carry `SL:TODO`,** and `SL:TODO` is precisely the slot `convert_topv2_rows` turns into
the event `summary`:

```460:471:data/loaders/calendar_json.py
        todo = _topv2_slot(parse, "TODO")
        when_text = _topv2_slot(parse, "DATE_TIME")
        if not todo or not when_text:
            continue
        ...
        summary = re.sub(r"\s+([',.])", r"\1", todo).strip()
```

The reason is semantic, not incidental: **a reminder has a subject and an alarm does not.** "Set an
alarm for 1 pm" names no event. A handful carry `SL:ALARM_NAME` ("add basketball game alarm tuesday at
3:00 pm"), but that was 1 of 16 CREATE rows sampled, and some of those use `SL:DATE_TIME_RECURRING`,
which the converter deliberately skips (recurrence needs an RRULE).

So the realistic yield is 20,430 → ~53% CREATE → minus recurring → minus those with no name at all,
leaving perhaps a few hundred rows whose titles are of the form "basketball game alarm". Feeding the
rest would mean inventing a constant `summary`, which on a metric scored by exact argument match would
teach the model a degenerate title and cost more than it adds.

`event` is wrong for a different reason: it is `IN:GET_EVENT` ("what's happening at Liberty Science
Center"), a retrieval intent with no creation and nothing to resolve.

**Corrected summary: the orchestrator's rejection of `top_v2` reached the right answer, and for a
better reason than it gave.** The description "unmappable" is still wrong as prose — it is the
training set — but there genuinely are no easy rows left in the other domains. The real open question
is narrower and is about `reminder` itself. Sampling 100 of its 17,840 rows:

```
intents: CREATE_REMINDER 81 | DELETE_REMINDER 6 | GET_REMINDER 6 | UPDATE_* 6
CREATE_REMINDER with both SL:TODO and SL:DATE_TIME, non-recurring: 51/100  → est. ~9,100
```

The run consumed 3,900 and called the source exhausted. The gap between ~9,100 and 3,900 is not
necessarily waste: `resolve_datetime` deliberately refuses relative-to-event offsets ("15 minutes
before"), vague spans ("this week") and timezone-qualified times, and each refusal drops a row. That
could easily account for the difference. **I have not run the loader to find out**, so this is a
question, not a finding — but it is the one place where more real calendar rows might actually exist.

---

## 3. Why the score moves when nothing changed

You saw two rebuilds add zero rows and the score move anyway. This is the same finding as `08-23` §1,
reproducing.

**The mechanism is not smooth noise.** `ast_arg_match` scores each row 1 or 0 on exact equality of the
whole argument dict, and every calendar row hinges on the same few conventions. Adopt the wrong shape
for `end` and *all 535 rows* fail at once; adopt the right one and the score jumps to whatever the
remaining errors allow. There is no middle. That is why the tier-1 column in this run reads

```
0.5981  0.6860  0.6841  0.5607  0.7140  0.6056  0.5121  0.6654  0.6561  0.5794 ...
```

as clusters rather than a distribution around a mean.

**Three sources of run-to-run variation, none of them controlled:**

1. **Training is not seeded.** `_build_sft_config` passes no `seed` and no `data_seed`, and nothing sets
   the torch determinism flags. The only seeded randomness in the path is the validation-split shuffle
   (`training/lora_trainer.py:876`, `_rnd.Random(1234)`).
2. **Checkpoint selection optimises the wrong quantity.** Early stopping keeps the best checkpoint by
   `eval_loss` (`training/lora_trainer.py:935-937`), which is token-level cross-entropy on a held-out
   slice of *training* rows. Writing `"end": "2026-..."` instead of `"end": {"dateTime": ...}` is three
   tokens of cross-entropy and the entire row on the metric. `08-23` §1.2 has the clean demonstration:
   a fit that trained twice as long to a *lower* loss scored **0.0019** against another's **0.4598**.
3. **Every iteration re-quantizes.** The score is on a freshly built Q4_K_M GGUF, not the bf16
   checkpoint. `08-23` §1.4 could not rule this out and it is still open.

**What this means for reading the tier-1 table:** the spread on a *fixed* configuration is at least
±0.46 on this task. Iterations 3 (0.6841) and 18 (0.6785) differ by 0.006. Those two are not
distinguishable, and the rollback machinery is nonetheless making keep/discard decisions on
differences that size.

**This is the strongest argument for your empty-rebuild change in §4.** A rebuild that adds zero rows
currently produces a *new score* for an *unchanged experiment*, and that score enters the trajectory,
the rollback decision, and the attribution table as though it meant something. Iteration 5 of run
38708719 was credited `mine_new_real +0.4561` for a rebuild whose own log says it added nothing.

---

## 4. Restart at `iterate` when a rebuild adds no rows — the change you asked for

**Yes, this makes sense, and it fixes more than the wasted GPU time.**

Today, when `mine_new_real` or `surgical_synthesis` adds nothing, curate logs:

```
✗ ERROR: data_rebuild/mine_new_real added 0 new rows. The curriculum is unchanged at 3900 row(s),
  so training this iteration would repeat the previous one exactly.
```

...and then trains and evaluates anyway. That costs a train+eval cycle **and** injects a meaningless
score into the trajectory, which §3 shows is not harmless.

It happened **twice in this run** (log lines 1926 and 2795), and the end-of-run report counted it:

```
2 of 7 rebuild(s) added no rows  ← every one of these was a wasted train+eval cycle
```

The report already knows. It just cannot act on it.

**My understanding of what you want:** when a data rebuild adds zero rows, skip train and evaluate,
return to `iterate`, increment the iteration counter, and let the orchestrator choose again — with the
failed attempt recorded so it does not repeat it.

Questions before I build it, in §10.

---

## 5. The attribution table only shows the last tier

You are right, and the cause is one line of data plumbing.

**Builder:** `format_health_summary` → `_dag_rows` in `agent/run_health.py:469-514` and `361-410`.
**Printer:** `tests/pipeline/run.py:1517-1524`.

`_dag_rows` reads **only the live DAG**:

```361:368:agent/run_health.py
def _dag_rows(state) -> list[dict]:
    health = RunHealth.from_state(state)
    by_iteration = {int(entry.get("iteration", -1)): entry for entry in health.history}
    rows = []
    for node in (state.get("dag") or []):
```

and `escalate_node` **clears** `state["dag"]` on every promotion, stashing the old one:

```308:341:agent/nodes/escalate.py
    history.append({... "dag": list(state.get("dag") or []), })
    state["escalation_history"] = history
    ...
    state["dag"] = []
    state["iteration"] = 0
```

So at end of run `state["dag"]` holds tier 3 only. Tiers 1 and 2 are in
`escalation_history[*]["dag"]`, which `_dag_rows` never reads.

**The fix already exists in the same report.** The "DAG Traversal (all N models)" section prints every
tier correctly because it uses `build_run_progression` (`agent/pipeline_status.py:52-114`), which
merges `escalation_history` with the live DAG. Pointing `_dag_rows` at the same source is the change.

**One trap.** `escalate_node` also resets `state["iteration"] = 0`, and the curriculum ledger keys on
iteration number. Merging tiers naively would collide iteration 1 of tier 1 with iteration 1 of tier 3
in `by_iteration` and mis-attribute the curriculum columns. The join needs to be scoped per tier, or
`observe_iteration` needs to record the tier alongside the iteration.

---

## 6. `surgical_synthesis` on the first iteration of a new tier

You are right that this should be a plain retrain, and it is worse than a mislabel: **the synthesis
actually runs.**

The escalation path routes to `curate`, not `train`:

```335:361:agent/nodes/escalate.py
    state["last_intervention"] = "data_rebuild"
    state["data_rebuild_plan"] = None
    state["llm_iterate_decision"] = None
    ...
    state["next_action"] = "curate"
```

`curate` only skips when the intervention is *not* `data_rebuild`:

```1045:1048:agent/nodes/curate.py
    intervention = state.get("last_intervention", "data_rebuild")
    if intervention != "data_rebuild":
        _log(model_id, f"SKIP: intervention={intervention} — dataset held fixed")
        return state
```

With no plan in state, it builds one from the deterministic fallback, which picks synthesis whenever
mining is unavailable (`agent/data_rebuild.py:327`), and then executes it
(`agent/nodes/curate.py:1153-1157`). That is why tier 3 iteration 1 reads
`data_rebuild/surgical_synthesis` and shows `+173` rows added.

**There is no "retrain on the unchanged curriculum" step anywhere after an escalation.**

This also confounds the one measurement an escalation exists to produce. Tier 3 iteration 1 changed
*two* variables at once — new model **and** +173 synthetic rows — so its 0.8168 cannot be attributed to
either. A plain retrain first would give a clean read on what the bigger model is worth on the
curriculum you already have.

**The fix is small:** have `escalate_node` set `last_intervention` to something curate skips (or add an
explicit `"none"`) and route `next_action = "train"`.

**A wrinkle worth your call:** on this run the *first* tier-3 iteration is also the one that produced
the run's entire gain. Making escalation a plain retrain does not lose that — synthesis would simply
happen at iteration 2, chosen deliberately by the orchestrator, with a clean baseline to compare
against.

---

## 7. The 400 at log line 9823 — diagnosed, and now fixed

```
[synth] new-correct (5-shot): 0/25 kept (57 attempts)
[generate:failed] 57 generation(s) never produced a usable row and were NEVER VERIFIED
  x57  endpoint error
      BadRequestError: 400 - This model's maximum context length is 8192 tokens. However, you
      requested 6285 output tokens and your prompt contains at least 1908 input tokens, for a
      total of at least 8193 tokens.
```

**This is not a verifier rejection and not truncation. Every one of the 57 requests was refused by the
server before generating a token.** The log says so explicitly, which is `_note_generation_failure`
doing its job — without it this would read as "the verifier rejected everything", which is exactly how
runs 38832586 / 38985393 / 39041380 were misdiagnosed.

**Root cause.** `output_budget()` estimated the prompt at 4 chars/token and kept a fixed 64-token
margin:

```
room = served_context() - int(len(prompt) / CHARS_PER_TOKEN) - _CONTEXT_MARGIN_TOKENS
     = 8192 - 1843 - 64 = 6285      real prompt: 1908      1908 + 6285 = 8193
```

Estimate short by **65 tokens** against a **64-token** margin. Over by exactly one.

**The same defect, same 65-token gap, at three different context sizes** (all from xlam runs
39311800 and 39321471):

| Served context | Requested output | Real prompt | Total | Estimate error |
|---|---|---|---|---|
| 8192 | 6285 | 1908 | 8193 | −65 |
| 8192 | 4736 | 3457 | 8193 | −65 |
| 16384 | 13515 | 2870 | 16385 | −65 |
| 16384 | 12419 | 3966 | 16385 | −65 |

That table is the whole diagnosis: **the shortfall is proportional to prompt length, not constant, so
a fixed margin can never absorb it — and raising the served context reproduces it exactly.** Dense
tool-schema and calendar JSON tokenizes near 3.9 chars/token, so the 4.0 divisor is ~2% optimistic.

**Fixed today** in `config/token_budget.py` by giving prompt estimation its own conservative divisor
(`_PROMPT_CHARS_PER_TOKEN = 3.5`, ~14% headroom) rather than sharing the 4.0 used to size outputs.
Verified against the recorded failures:

```
calendar 8192 : prompt 1908 + budget 6002 = 7910   (was 8193)  OK
xlam     8192 : prompt 3457 + budget 4276 = 7733   (was 8193)  OK
xlam    16384 : prompt 3966 + budget 11901 = 15867 (was 16385) OK
xlam    16384 : prompt 4234 + budget 11603 = 15837 (was 16385) OK
```

Regression test at `tests/config/test_served_context_clamp.py` reproduces both recorded 400s and
asserts prompt + budget fits for contexts 4096→32768. 21 passed; 41 passed across the related suites.

**Status by run:** calendar 39294409 hit this once and finished anyway. xlam 39311800 lost two whole
batches to it (0/25 and 4/126). xlam was restarted twice — once to widen the context, once to pick up
this fix, since a running process holds the old bytecode.

---

## 8. Why the learning rate and weight decay barely move

Nineteen tier-1 iterations, and `lr=1e-04` appears in 15 of them, `wd=0.05` in 13. Here is why.

### What the orchestrator is told

Exactly five knobs are tunable (`agent/nodes/iterate.py:747-763`):

```747:756:agent/nodes/iterate.py
- EXACTLY FIVE hyperparameters are tunable: lora_rank, alpha_ratio, weight_decay,
  learning_rate, nr_epochs. Emitting any other key is rejected.
- lora_rank must be one of [4, 8, 16, 32, 64]
- alpha_ratio must be one of [1, 2, 4]; the LoRA update is scaled by alpha/rank, so
  the ratio is the meaningful quantity and alpha is derived as rank * alpha_ratio.
- weight_decay must be one of [0.0, 0.01, 0.05, 0.1] — this is the regularizer to
  reach for when the model overfits a small curriculum.
- learning_rate is bounded to [1e-5, 5e-4] and nr_epochs to [1, 8]
```

### The bounds, and how they are enforced

From `training/hparams.py:15-52`, enforced in `normalize_hyperparams` and re-checked in
`TrainingConfig.__post_init__`:

| Knob | Allowed | Mechanism |
|---|---|---|
| `lora_rank` | `{4, 8, 16, 32, 64}` | snap to nearest |
| `alpha_ratio` | `{1, 2, 4}` → `alpha = rank × ratio` | snap |
| `weight_decay` | `{0.0, 0.01, 0.05, 0.1}` | **snap** |
| `learning_rate` | `[1e-5, 5e-4]` **continuous** | **clamp** |
| `nr_epochs` | `[1, 8]` | clamp |
| `lora_dropout` | fixed `0.0` | not tunable |
| batch shape | trainer-derived, `micro × accum ≤ 64` | not tunable |

So `weight_decay` has only **four legal values** and the model reaches for the middle one. `lr` is
continuous but the prompt gives no ladder, so the model anchors on round numbers.

### Why the search stalls

**There is no instruction to vary anything in particular.** The only constraint is:

```761:763:agent/nodes/iterate.py
- Never propose an exact (dataset, hyperparameter) repeat from the tried list,
  including a pruned/rolled-back trial. The same complete hyperparameter config is
  allowed after a data rebuild because the dataset identity changed.
```

Uniqueness is on the **full 9-field identity** (`HYPERPARAMETER_IDENTITY_FIELDS`, `hparams.py:54-64`)
plus dataset version. So changing rank 32→64 while holding `lr=1e-04, wd=0.05` is a legal new config,
and the model can walk rank and epochs indefinitely without ever touching the two knobs that most
affect optimisation. Nothing tells it to.

Enforcement is in `train._build_config`, not at decision time — a repeat is silently substituted with
`_next_untried_config` (`agent/nodes/train.py:300-314`), so the orchestrator never learns it repeated.

**On iterations 7 and 16 having the identical label** `r=32 a=64 wd=0.05 lr=1e-04 ep=5`: the displayed
label is only five fields (`train.py:192-206`), while identity is nine fields **plus the dataset**.
Between those iterations the curriculum changed (3900 → 4255 rows), and the prompt explicitly permits
the same hyperparameters after a data rebuild. So both ran legally. Given §3, re-running the same
config on a changed dataset is arguably the most informative thing in the whole table — it is the only
near-repeat measurement the run contains, and the two scores differ by **0.084**.

### The lever that exists but is not used for steering

`deterministic_neighbor_configs()` (`hparams.py:314-425`) contains a proper LR ladder
`(1e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4)`. It is used only to substitute a config when the train node
rejects a repeat — never to steer the orchestrator. Feeding that ladder into the prompt, or telling
the model which knobs it has left untouched, is the cheap fix.

---

## 9. What changed in code today

| # | Change | Where |
|---|---|---|
| 1 | Prompt-token estimation given its own conservative divisor; fixes the proportional under-count behind every recorded 400 | `config/token_budget.py` |
| 2 | Regression test reproducing both recorded 400s across four context sizes | `tests/config/test_served_context_clamp.py` |
| 3 | `SLM_TEACHER_SYNTH_BYPASS=1` written into the launchers instead of relying on the submitting shell | `tests/pipeline/run_{calendar_json,ner_bc5cdr,xlam_bfcl}_{l40s,cse}.slurm`, `run_ner_bc5cdr_ckpt.slurm` |
| 4 | xlam teacher served at 16384 with `max-num-seqs` halved, so 5-shot fitness fits | `tests/pipeline/run_xlam_bfcl_{l40s,cse}.slurm` |
| 5 | clinc150 manifest/checkpoint repaired (added `SYNTH_API_MODE`, recomputed `config_fingerprint`) so run 39040699 could resume | `logs/runs/slm-clinc150-l40s-39040699/` (backups kept) |

Nothing in §4, §5, §6 or §2.1 is implemented — those are proposals awaiting your answers.

---

## 10. Questions

**On the empty-rebuild restart (§4):**

1. **Does the failed attempt consume the iteration number?** You said "it should tick the iteration
   count". Confirming: iteration 4 attempts `mine_new_real`, adds 0 rows, no train/eval happens, and
   the next attempt is iteration 5 — so the ledger shows iteration 4 as a data-only attempt with no
   score. Correct?
2. **What does the DAG record for that iteration?** A node with no score would break the trajectory
   arrays that rollback and stagnation detection read. Cleanest is a node marked
   `status="no_rows_added"` that is excluded from score-based logic. Or do you want no DAG node at all,
   with only the curriculum ledger recording the attempt?
3. **What stops an infinite loop?** If mining is retired and synthesis is gated off, every subsequent
   `data_rebuild` adds zero rows and we would bounce between curate and iterate forever without
   burning wall-clock — which is worse than the current waste, because nothing advances. I would cap
   consecutive empty rebuilds (2 or 3) and then force either a hyperparameter intervention or an
   escalation. What cap do you want?
4. **Does stagnation detection count these?** The escalation trigger reads a window of recent scores.
   A no-score iteration either extends the window or is skipped; skipping seems right, but it means a
   run can spend several iterations without moving toward escalation.

**On the other three:**

5. **§6 escalation** — plain retrain on the unchanged curriculum as iteration 1 of every new tier.
   Confirm you want this even though it costs one extra train+eval per escalation (the benefit is a
   clean read on what the model change alone bought).
6. **§2.2 TOPv2 `reminder` exhaustion** — withdrawn as originally posed; the `alarm` domain has no
   `SL:TODO` and is not mappable. The live question is why `reminder` reported exhausted at 3,900 of
   17,840 rows. Worth me digging into `initial_train_cap` and the novelty check?
7. **§3 noise floor** — this is the third run where attribution is being read off differences smaller
   than the demonstrated spread. `08-23` §1.7 proposed the experiment that separates training variance
   from quantization instability and it has still not been run. It is one short job and it decides how
   much of the last five runs' attribution survives.
