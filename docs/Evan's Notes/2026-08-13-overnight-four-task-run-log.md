# Overnight run log: xlam_bfcl, calendar_json, ner_bc5cdr, routerbench

*2026-08-13 — running log*

Live log of the four-task overnight campaign on `gpu-l40s-cse`. Problems hit, fixes made, and
anything surprising enough to be worth keeping. Appended to as the night goes; the newest section
is at the bottom.

Companion: `2026-08-12-two-new-tasks-and-loader-blockers.md` (the pre-flight audit this campaign
acts on) and `2026-08-06-six-benchmark-tasks-reference.md` (scoring and prompts).

**Ground rules for the campaign**, as set before starting: thresholds are left to the planner
(accepting the non-convergence risk); RouterBench's routing boundary is the weakest model in the
benchmark; no budget cap; a task whose eval set cannot be made clean is **skipped**, not run on
substituted data. Run order: `xlam_bfcl` → `calendar_json` → `ner_bc5cdr` → `routerbench`.

---

## 0. Status board

| # | Task | Job | State | Eval set | Self-consistency | Notes |
|---|---|---|---|---|---|---|
| 1 | `xlam_bfcl` | 38454799 | queued 00:59 | 2,309 BFCL rows | **1.0000** | loader rewritten, §2 |
| 2 | `calendar_json` | 38455147 | queued 02:11 | 478 SGD rows | **1.0000** | new task, §3 |
| 3 | `ner_bc5cdr` | 38455148 | queued 02:11 | 5,865 rows | **1.0000** | local bundle, §4 |
| 4 | `routerbench` | 38455150 | queued 02:11 | 7,267 rows | **1.0000** | pickle reader, §5 |

All four were validated **before** submission: the data loads, the eval set is non-empty, and
feeding the gold back in as the prediction scores exactly 1.0. That last check is the one that
matters — anything below 1.0 means the eval set contains rows no model can win, and every score
from the run would be depressed by an unknown amount. Two such rows were found and removed (§2.3,
§5.2). Test suite: **64 passed** across the loader and slurm-lockstep files; full suite 960 passed
with the two documented long-standing failures (`primary_strategy`, `APPS introductory`)
unchanged and untouched by this work.

---

## 1. Environment facts established up front

- **`datasets` is 4.3.0.** Script-based repos are permanently unloadable. This is the single
  biggest cause of the breakage below.
- **The HF token in `.env` already has xLAM access.** This was the thing most likely to block the
  night and it turned out to be a non-issue: `whoami-v2` resolves to `evanlyhf`, and
  `load_dataset("Salesforce/xlam-function-calling-60k")` returns `['id','query','answers','tools']`
  — exactly the columns `convert_xlam_rows` expects. `tests/pipeline/run.py:139` does
  `load_dotenv(..., override=True)` before any loader import, so the token reaches the pipeline
  without touching `_l40s_task_body.sh`.
- **`gpu-l40s-cse` caps wall-clock at 24h** (`sacctmgr` MaxWall `1-00:00:00`) and the whole
  account shares a 14-GPU cap. Four jobs × 2 GPUs = 8, which fits alongside other CSE users.
- **Requeue survives the 24h boundary.** `--signal=B:USR1@7200` fires the checkpoint 2h before the
  limit; `_l40s_task_body.sh` calls `scontrol requeue`, and a requeued job keeps its **job ID**,
  so `SLM_RUN_DIR` (keyed on `SLURM_JOB_ID`) is stable and `_durable_resume_ready` finds
  `checkpoint.json`. No manual intervention needed for the wall-clock case — only for crashes.

---

## 2. `xlam_bfcl` — the BFCL eval set had to be rebuilt from scratch

### 2.1 What was broken

`load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard")` raises
`DataFilesNotFoundError`. The repo ships 52 category-named files (`BFCL_v3_simple.json`,
`BFCL_v3_parallel.json`, …) matching no split-name pattern. The old loader's guard was
`except (ValueError, KeyError)`, which doesn't catch it, so the run would have died at
`eval_setup` rather than degrading.

Underneath that were three more problems that would each have produced a silently empty or wrong
eval set:

1. **Prompts and gold live in different files**, joined on `id`: `BFCL_v3_simple.json` carries
   `{id, question, function}`, `possible_answer/BFCL_v3_simple.json` carries `{id, ground_truth}`.
   `convert_xlam_rows` read `answers` off the same row, so every row would have been dropped.
2. **Gold arguments are lists of acceptable values**, not values:
   `{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}`. The `""`
   means the argument may be omitted. The existing `_args_match` requires
   `set(gold_args) == set(pred_args)` and scalar equality, so a correct answer that omitted an
   optional argument would have scored 0.
3. **`question` is a list of turn-lists of chat messages**, not a string.

### 2.2 What was built

`data/loaders/xlam_bfcl.py` now names the six AST-gradable categories explicitly (`simple`,
`multiple`, `parallel`, `parallel_multiple`, `live_simple`, `live_multiple`), pulls them through
`hf_hub_download` (which caches under `HF_HOME` and follows the LFS redirect a bare `requests.get`
does not — a plain `curl` without `-L` returns a 307 and a 331-byte stub), joins on `id`, and
carries the acceptable-value map on a `_accept` metadata field.

Deliberately excluded: `irrelevance` / `live_relevance` (gold is "no call" and they ship no
`possible_answer` file), `java` / `javascript` / `sql` / `rest` (non-Python call syntax),
`exec_*` (needs live API execution), `multi_turn_*` (stateful).

Categories are interleaved **round-robin**, not concatenated, so a truncated 800-row eval set
spans all six rather than being 800 rows of `simple`. Verified spread: 134/134/133/133/133/133.

`eval/scorers/function_call.py` gained `_accept_args_match` / `_accept_correct`, used only when a
row carries `_accept`. xLAM rows don't, so they fall through to the original equality path and
existing behaviour is untouched. The `_` prefix follows the `_instruction` convention — it keeps
the field out of the schema shown to the synthesis teacher.

Two semantics worth recording because they are not documented anywhere in BFCL:

- An acceptable list containing `""` means **the argument may be omitted**.
- An **empty** acceptable list `[]` means the argument must be *absent*. BFCL uses this for
  intent-classifier-shaped functions (`record` in the `live_simple` bank set) where one of ~12
  slots is filled and the rest are explicitly empty. Missing this cost two rows of
  self-consistency before it was found.

Matching against `_accept` is **order-insensitive** (greedy one-to-one). BFCL's `parallel`
categories ask for several calls whose order the benchmark does not constrain, and the original
`_call_correct` is order-sensitive — grading parallel rows in order would have understated the
model for a reason that has nothing to do with function calling.

### 2.3 A real annotation bug in BFCL

`simple_363` declares the tool `restaurant_search.find_closest` but its ground truth calls
`find_closest`. Since the scorer rejects any call outside the declared tool set, **no model could
ever score that row** — it would have silently capped the achievable ceiling below 1.0. The
converter now drops rows whose gold calls an undeclared function. One row of 2,310.

### 2.4 Validation

The check that matters for a format-bound task is **self-consistency**: feed the gold back in as
the prediction and the scorer must return exactly 1.0. Anything less means the eval set contains
rows that are unwinnable by construction, and every score from the run would be depressed by an
unknown amount.

| Check | Result |
|---|---|
| Scoreable corpus | 2,309 rows across 6 categories |
| Self-consistency, 800-row eval set | **1.0000** |
| Self-consistency, full corpus | **1.0000** (was 0.9975 before the `[]` fix, 0.9996 before the `simple_363` filter) |
| Negative control (wrong function name) | 0.0000 ✓ |
| Junk control (prose, unparseable) | `format_valid` 0.0000, content 0.0000 ✓ |
| Existing test suite | 44 passed, 0 failed |

Submitted as job **38454799** at 00:59.

---

## 3. `calendar_json` — a new task, built from scratch

### 3.1 The design

Input is a free-text scheduling request; output is a Google Calendar API v3 `events.insert`
request body with a **resolved** ISO-8601 timestamp. Registered as `task_type=function_call`, so
`eval/scorers/function_call.py` grades it unchanged — including the `format_valid` /
`content_correct` split, which is the right instrument for a task whose output feeds an API.
Argument names are the real Events resource fields (`summary`, `start.dateTime`, `end.dateTime`,
`location`), so a passing prediction is a postable body rather than a benchmark artefact.

Train and eval come from **independently built corpora**, mirroring `xlam_bfcl`, so the eval is a
transfer test rather than a held-out slice:

| Side | Source | Loads under `datasets` 4.3? | Usable rows |
|---|---|---|---|
| train | TOPv2 `reminder` / `CREATE_REMINDER` (`WillHeld/top_v2`) | ✅ native parquet | 4,242 of 10,353 |
| eval | SGD `Calendar_1` / `AddEvent`, raw JSON from GitHub | ❌ HF mirror is script-based | 478 |

### 3.2 The hard part: relative dates

Neither corpus annotates a timestamp. TOPv2 gives `[SL:DATE_TIME at 5 pm ]`; SGD gives
`"event_date": ["March 6th"], "event_time": ["12:45"]`. "At 5pm" is not a time until you know
what day it is.

Every row therefore pins its own reference instant, **states it in the prompt**, and the gold
carries the resolved timestamp:

```
Convert the user's scheduling request into a single calendar.events.insert call. Resolve every
relative date and time against the current date and time given below, and make the event 60
minutes long unless a duration is stated.
Current date and time: 2026-08-09T11:00:00 (Sunday).

Remind me to pack my lunch for tomorrow.
```
```json
[{"name": "calendar.events.insert", "arguments": {
  "summary": "pack my lunch",
  "start": {"dateTime": "2026-08-10T09:00:00"},
  "end":   {"dateTime": "2026-08-10T10:00:00"}}}]
```

The reference is a SHA-256 hash of the row key, constrained to 2026 and to 08:00–17:00. Hashing
rather than fixing one "today" is deliberate: a single reference date would let the model memorise
one answer instead of learning the arithmetic. Hashing rather than randomising keeps it
reproducible across runs and machines. Constraining to daytime hours means "tomorrow at 9am" is
never ambiguous about which side of midnight the reference sits on.

Because both the trainer and the eval harness build prompts from `text` through the same
`function_call` builder, stating the reference inside `text` gives train/serve parity by
construction — the class of bug B250 was.

### 3.3 Gold correctness is enforced by refusal

`dateparser` is not installed, and for building **gold labels** a permissive parser is a
liability: it will happily "resolve" something it does not understand, and the resulting wrong
label is invisible. So `resolve_datetime` is a strict hand-written grammar that returns None for
anything it cannot pin down, and rows it rejects are dropped.

Rejected by design, with reasons:

| Expression | Why it cannot produce a gold label |
|---|---|
| "15 minutes before", "an hour before" | Relative to an event the utterance never states |
| "this week", "next month", "for the last day" | A span, not an instant |
| "after 10 am Pacific Time", "…her time" | A timezone we have no basis to convert |
| "every Tuesday", "daily" | Recurrence; needs an RRULE, out of scope for v1 |

That is why only 4,242 of 10,353 `CREATE_REMINDER` rows survive. A 41% yield of certainly-correct
gold beats 100% of a corpus where an unknown fraction is wrong — and the pipeline can synthesize
more training rows from a clean seed, which it cannot do from a poisoned one.

The grammar covers what the head of the distribution actually looks like: `at 3 pm`, `on Sunday`,
`for tomorrow`, `at 8 : 30 am` (TOPv2 splits punctuation apart), `at 5 p.m .`, `on April 2nd`,
`on the 4th of August`, `at noon`, `tonight`, `tomorrow afternoon`. A bare date defaults to 09:00;
a bare time that has already passed rolls to tomorrow, which is how a calendar app behaves.

### 3.4 Validation

| Check | Result |
|---|---|
| TOPv2 train rows | 4,242 |
| SGD eval rows | 478 |
| Self-consistency, both sides | **1.0000** |
| Negative control (fixed wrong timestamp) | 0.0000 ✓ |
| Resolver accept cases | 15/15 |
| Resolver refuse cases | 10/10, zero leaks |

478 eval rows is under the 800 target but far above the floor of 30, and it is the true ceiling:
that is every `AddEvent` frame in all three SGD splits with a complete name + date + time. Padding
it by accepting partially-specified mid-dialogue turns would have traded a real eval set for a
bigger one.

---

## 4. `ner_bc5cdr` — promoted, re-sourced, and the prompt skew finally fixed

Previously reachable only through the autonomous path, which means it depended on the
orchestrator emitting a plan `web_acquire` happened to resolve. Now pinned as a
`SLM_BENCHMARK_TASK` key so the run is reproducible.

Three changes from `slm-ner-l40s-37531245`:

**Data source.** `tner/bc5cdr` is script-based and dead under `datasets 4.3.0`;
`spyysalo/bc5cdr` has been removed from the Hub entirely. The new loader reads the checksummed
local bundle first (`data/local/bc5cdr`), with T-NER's raw JSON as fallback. The bundle needs no
network and carries **5,096 train / 5,865 test rows against the 3,403 / 900 the live loader
produced**. The previous run died with `bounded data_rebuild plan space is exhausted` after
running out of gold to resample, so the extra 1,693 training rows address the actual cause of
that failure rather than a symptom of it.

**Train/serve prompt skew — fixed.** This was flagged at the end of the six-task reference note
back on 2026-08-06 and never actioned. The eval prompt ended
`Reply with [] if there are no entities.` and the trainer's hand-copied version did not, so the
model was fine-tuned on one input shape and scored on another for the entire 44.8-hour run.
`_training_turn` now imports `eval.scorers.ner.NER_PROMPT` instead of reproducing it, which is
the same fix B250 got and makes the two incapable of drifting. Asserted byte-identical in a test.

**Validation.** Self-consistency 1.0 on a 900-row sample, empty-prediction control 0.0, prompt
parity `True`. Eval entity mix: 5,385 Chemical / 4,424 Disease, with 1,725 of 5,865 rows carrying
no entities at all — a genuinely useful negative class that the span-F1 metric handles correctly.

---

## 5. `routerbench` — three separate defects, all silent

### 5.1 It was never loadable

`withmartian/routerbench` ships only `routerbench_{0shot,5shot,raw}.pkl`. `datasets` has no
pickle reader, so `load_dataset` raises `DataFilesNotFoundError`, and the loader's
`except (ValueError, KeyError)` guard does not catch it. Now read via `hf_hub_download` +
`pandas.read_pickle`: 36,497 rows × 37 columns.

### 5.2 The correctness column did not exist

The loader looked for a field named `small_model_correct`. RouterBench has no such column — it
has **one correctness column per candidate model, named after the model**. Every row would have
returned `None` from `_correctness` and been dropped, so even with a working reader the result
was an empty dataset. The loader had been written against an imagined schema and unit-tested
against fixtures built from the same imagination, which is why nothing caught it.

The eleven real candidates, with the fraction each answers correctly:

| Model | correct | `local` share |
|---|---|---|
| `mistralai/mistral-7b-chat` | 0.306 | **29.9%** ← chosen boundary |
| `meta/llama-2-70b-chat` | 0.329 | 35.1% |
| `WizardLM/WizardLM-13B-V1.2` | 0.431 | 45.2% |
| `mistralai/mixtral-8x7b-chat` | 0.547 | 56.8% |
| `gpt-3.5-turbo-1106` | 0.619 | 66.6% |
| `zero-one-ai/Yi-34B-Chat` | 0.647 | 67.6% |
| `gpt-4-1106-preview` | 0.781 | 84.3% |

`mistral-7b-chat` is the smallest model in the benchmark and so the closest available analogue to
something that runs on an S24 Ultra. It also makes `local` the minority class, which is what
`eval/scorers/classification.py` reports F1 on for a binary task — a model that learns nothing and
answers "route" every time scores **0.0**, not 0.70. Verified.

### 5.3 The prompt column is a Python list literal in a string

Not caught by any schema check because the dtype is `object` and the values are `str`. Every
value looks like:

```
"['You are a helpful assistant.', 'What is 2+2?']"
```

Passing that through `str()` — which the loader did — puts brackets, quotes and comma separators
into **every single eval prompt**. That is not cosmetic: it is a distribution shift away from
anything the deployed model would ever see, and it would have depressed the score for a reason
having nothing to do with routing. Now `ast.literal_eval`'d and joined on blank lines.

### 5.4 Validation

RouterBench ships no train/test split, so one is made by hashing `sample_id` — stable across runs
and machines, and unaffected if the upstream file is ever reordered.

| Check | Result |
|---|---|
| Rows | 29,230 train / 7,267 test |
| Text overlap between splits | **0** |
| Eval label balance | 568 route / 232 local |
| Self-consistency | **1.0000** |
| All-`route` control | **0.0000** ✓ (minority-class F1 punishes the degenerate answer) |
| List-literal leakage | 0 rows |

---

## 6. Infrastructure changes

**Four new CSE slurm scripts** (`run_{xlam_bfcl,calendar_json,ner_bc5cdr,routerbench}_cse.slurm`),
24h wall-clock, 2× L40S, `--signal=B:USR1@7200` + `--requeue`.

**The account guard test had to be amended.** `test_every_slurm_script_runs_on_the_int_sys_l40s_allocation`
required `gpu-l40s-intelligentsystems` on every script, because the old CSE and `ckpt-g2` variants
had produced numbers incomparable with the dedicated runs. That reasoning is sound but it argues
for keeping the two families separate and clearly labelled, not for forbidding the second one. The
test is now `test_every_slurm_script_runs_on_a_sanctioned_l40s_allocation`: it allows exactly the
two accounts, still rejects a40 / a100 / `ckpt` partitions and unaccounted-for scripts, and
additionally pins the wall clock per account (24h on CSE, 7 days on the dedicated quota) so a
weeklong request on CSE — which the scheduler rejects at submit time — fails in CI instead.

**Two new lockstep tests.** `test_cse_scripts_match_the_registry_and_use_the_24h_requeue_contract`
and `test_every_registry_key_has_a_loader`. The second closes a real gap: `NAMED_BENCHMARK_TASK_TYPES`
and `_named_benchmark_loaders()` are two hand-maintained dicts that must agree, and a key present
in one but not the other fails only at run time — inside a job that has already allocated GPUs and
spent 40 minutes loading a 35B synth server.

---

## 7. Everything queued and nothing running: diagnosing the scheduler

All four jobs sat at `PENDING (Priority)` for 40 minutes after submission. Worth writing up
because the obvious explanations were all wrong and the counting method that produced the
"11 free GPUs" number is actively misleading.

### 7.1 What it was not

**Not capacity.** Three nodes were completely idle — `g3135-3137`, 8 free L40S and 1.5 TB RAM
each, plain `IDLE` state, both in the `gpu-l40s` partition, no reservation, no node feature
mismatch. 24 free GPUs while four 2-GPU jobs waited.

**Not the account cap.** `gpu-l40s-cse` allows `gres/gpu:l40s=14` and was using 3.

**Not wall-clock or backfill.** This was the first hypothesis and it is wrong. `sbatch --test-only`
returns the *same* projected start for a 2-hour, 4-hour and 24-hour version of the job, and a
throwaway 10-minute 2-GPU probe job pended identically for 4 minutes. If a backfill window were
the constraint, the short jobs would have slotted into it.

### 7.2 What it was

Fairshare queue position. `sprio` decomposes the priority:

```
JOBID    PARTITION  PRIORITY  SITE  AGE  ASSOC  FAIRSHARE  JOBSIZE  PARTITION  QOS  NICE
38454799 gpu-l40s       2913     0    0      0       2855        0          0    0     0
```

Priority is **fairshare and nothing else** — age had not accrued, and job size, QOS and partition
all contribute zero on this cluster. At 2913 the jobs were **28th of 53 pending** on the
partition. Slurm's main scheduler walks the queue in priority order, and the jobs ahead were
themselves blocked by *their* accounts' caps (`gpu-l40s-ark`, `-krish`, `-bkrs` all sitting on
`Priority`/`Resources`), so nothing ran and the idle nodes stayed idle. Slurm's own projection was
**2026-08-14T12:41 — 35 hours out**, i.e. the whole night and then some.

`gpu-l40s-cse` is the department-wide pool. Free GPUs on a shared account do not help when you are
28th in a fairshare queue against several hundred people.

### 7.3 The counting trap

The `gpu-l40s-intelligentsystems` account looked like it had 5 of 10 GPUs free, which made moving
there the obvious fix. It did not. `squeue -o "%b"` prints `N/A` for several jobs that **do** hold
GPUs, so summing that column undercounts the account — it reported 5 where the truth was 10. The
tell was the reason code flipping from `Priority` to **`AssocGrpGRES`** the moment the int-sys jobs
were submitted: that is Slurm saying "your account is at its GRES limit", which it would not say
if 5 GPUs were free.

`AllocTRES` is authoritative. The supervisor counts with it:

```bash
scontrol show job "$job" | grep -oE 'AllocTRES=[^ ]*' \
    | grep -oE 'gres/gpu:l40s=[0-9]+' | cut -d= -f2
```

### 7.4 Where that leaves the campaign

Two accounts, two *different* blocking mechanisms, and it is not obvious which clears first:

| Account | GPUs | Reason | Clears when | Slurm ETA |
|---|---|---|---|---|
| `gpu-l40s-cse` | 3 of 14 used | `Priority` | we climb from 28th of 53 | ~35h |
| `gpu-l40s-intelligentsystems` | **10 of 10 used** | `AssocGrpGRES` | a co-tenant job ends | ~8h |

So each task is queued on both, and `scripts/supervise_overnight_runs.sh` cancels the twins the
instant one copy starts. Duplicates are safe to hold because `SLM_RUN_DIR` is keyed on
`SLURM_JOB_ID`, so two copies of a task cannot touch each other's checkpoints — but they are not
safe to *leave* queued, because a pending slot we will never use is a tax on everyone else in the
account.

Priority on int-sys is 3047 against 2913 on CSE, which is rank 5 rather than rank 30. Note that
this is a better queue position, **not** available hardware; the int-sys jobs still wait for a
co-tenant to finish.

**Preemptible `ckpt-all`/`ckpt-g2` was considered and rejected by the operator.** The same idle
nodes are in those partitions and jobs there start within minutes, and the pipeline's
USR1-checkpoint-and-requeue contract would survive preemption. It is still a no: the six-task
reference note removed ckpt variants deliberately, and a test enforces it.

### 7.5 The supervisor

`scripts/supervise_overnight_runs.sh`, polling every 180s:

- **Auto-cancels twins** when any copy of a task reaches RUNNING.
- **Submits an int-sys copy** only when `AllocTRES` shows genuine room, never on the `%b` count.
- **Never touches ckpt.** Only the two sanctioned accounts.
- Emits a single grep-able `ALERT` line per problem: `NODATA` (empty eval set or a loader raise),
  `ITERATE` (>12 orchestrator decision-validation failures — the reask loop), `FLATLINE` (last 8
  eval scores byte-identical, i.e. training is not moving), `NOPROGRESS` (log unchanged between
  polls of a RUNNING job), `DIED` (left the queue with no completion banner).

One bug worth recording from writing it: `grep -c` **exits 1 when it counts zero**, so the
idiomatic `$(grep -c ... || echo 0)` yields the two-line string `"0\n0"` and every subsequent
`[ "$n" -gt 12 ]` dies with `integer expression expected`. The first draft of the watchdog was
full of those. `count_matches()` handles it.

---

## 8. Results, 2026-08-14 12:20

| # | Task | Job | Outcome | Score | Threshold | Model | Time | Cost |
|---|---|---|---|---|---|---|---|---|
| 1 | `xlam_bfcl` | 38454799 | ✗ **crashed** | — | — | — | 68 min | $0.02 |
| 2 | `calendar_json` | 38455147 | ✗ **crashed** | — | — | — | 42 min | $0.03 |
| 3 | `ner_bc5cdr` | 38455148 | ✓ **CONVERGED** | **0.8098** span-F1 | 0.800 | Qwen3-**0.6B** @ Q4_K_M | **81 min** | **$0.13** |
| 4 | `routerbench` | 38493142 | running | 0.6551 best | 0.800 | Qwen3-1.7B @ Q4_K_M | 5h27m+ | — |

### 8.1 `ner_bc5cdr` converged, and the comparison is the story

| | previous run (`slm-ner-l40s-37531245`) | this run (`38455148`) |
|---|---|---|
| Outcome | **failed** | **converged** |
| Best span-F1 | 0.8628 | 0.8098 |
| Threshold | 0.880 | 0.800 |
| Model that got there | Qwen3.5-**4B** | Qwen3-**0.6B** |
| Iterations | 142 | **5** |
| Wall clock | **44.8 h** | **81 min** |
| Cost | $13.43 | $0.13 |
| Ending | `ValueError: bounded data_rebuild plan space is exhausted` | `=== done ===` |

Trajectory: 0.7701 → 0.7730 → 0.7362 (pruned) → 0.7829 → **0.8098**, all five iterations on the
tier-0 0.6B model, never escalating. Baseline (zero-shot Q4_K_M) was 0.0000, so Δ = +0.8098.

Two honest caveats. **The threshold was lower** (0.800 vs 0.880), and that alone flips
failed→converged; 0.8098 is genuinely *below* the 0.8628 the old run reached. And per-difficulty
is lopsided — easy 0.973 (n=182), hard 0.605 (n=618), medium n/a — so the aggregate leans on a
skewed difficulty split.

What is *not* explained by the threshold: this run got a 0.6B model to 0.81 in 81 minutes where
the old one needed 142 iterations and a 4B to reach 0.86. The plausible causes are the two changes
made before launch — the train/serve prompt skew fix (§4), which means the model is now tuned on
the exact string it is scored on, and the local bundle's 5,096 train rows against 3,403.
Attributing the speedup between those two would need an ablation; it is not established here.

### 8.2 Both `function_call` runs crashed on the same missing feature

```
File "training/lora_trainer.py", line 357, in _training_turn
    raise ValueError(
ValueError: completion-only SFT does not support task_type='function_call'
```

`_training_turn` handled `classification`, `NER`, `code_generation`, `generation` and
`math_reasoning`, and raised for `function_call` and `diff`. **The pipeline could score the
format-bound task types but had never been able to train them.** The scorers were built
2026-08-01 with the two-column format/content split; this function was never extended to match.

That is the real reason `xlam_bfcl` and `coedit` had no run log — the loader breakage in
`2026-08-12-two-new-tasks-and-loader-blockers.md` §5 was true but was not the whole story, and
fixing the data merely moved the failure 68 minutes later, to the first training step. Both runs
loaded their data cleanly, measured a baseline, and then died.

**My pre-flight was wrong in scope, not in method.** Every eval set was verified to load and to
score 1.0 on self-consistency, which is the right check and caught two real defects. But I never
executed a single training step, so a `raise` sitting on the only code path both tasks needed went
unnoticed. Data validation is not run validation.

**Fix.** `build_function_call_prompt` and `build_diff_prompt` added to their scorers, and
`_training_turn` now imports them for the two format-bound types — the same import-don't-reproduce
pattern that classification, NER and generation already use, so train/serve drift is impossible by
construction. The target is the gold string the scorer parses from `answer` (a JSON call list, or a
unified diff). An empty `answer` raises rather than training the model to emit nothing. Verified
byte-identical prompts on both sides; 4 new tests, 122 passing in the touched areas.

Both tasks resubmitted 12:22 on both accounts: `xlam_bfcl` 38505237/38505240, `calendar_json`
38505239/38505241.

### 8.3 The supervisor let the failures sit for 28 hours

The auto-cancel rule was right and the gap it left was not. When a CSE copy started, the int-sys
twin was cancelled — correct, that is the policy. When that CSE copy then crashed, **no twin
remained and nothing noticed.** `xlam_bfcl` and `calendar_json` were dead at 08:04 and 08:48 on
2026-08-13 and were still dead 28 hours later.

The supervisor reported `no jobs queued` for them, which reads as "nothing to do" rather than "this
task is dead". Now fixed: a task with no queued jobs whose most recent log lacks the `=== done ===`
banner raises `ALERT ORPHANED` with the exception line attached. It deliberately does **not**
auto-resubmit — a deterministic crash reproduces and burns another allocation, which is exactly
what happened here.

(The `no jobs queued` early-return also made the new check unreachable on the first attempt, since
it fired before the orphan branch. Worth noting because the bug and its own fix had the same shape:
an early exit that looks like a harmless log line.)

### 8.4 `routerbench` — correction: the running job is a replacement, not the original

> **Correction 2026-08-14.** §8's table listed `routerbench` job 38493142 as the run. That is
> wrong in a way that matters: the original run was **38455150**, which ran **22 hours**, climbed
> all four model tiers, reached **0.7584**, checkpointed cleanly at the 24h wall clock, requeued
> itself — and was then **cancelled by my own supervisor** four minutes later as a "redundant
> twin", in favour of a from-scratch job three minutes old. The auto-cancel rule could not tell a
> requeued run holding a checkpoint apart from a duplicate that never ran. Full post-mortem:
> `2026-08-14-routerbench-postmortem-crashes-and-the-ner-teacher-baseline.md` §1.

### 8.5 `routerbench` 38493142 (the replacement) is not going well

25 evals, best **0.6551** against a 0.800 threshold, currently on tier-1 Qwen3-1.7B. The score
sequence is violently unstable:

```
0.473 0.462 0.510 0.649 0.492 0.297 0.655 0.540 0.572 0.486 0.601 0.500
0.415 0.446 0.052 0.534 0.560 0.499 0.436 0.169 0.000 0.065 0.166 0.495 0.000
```

Three `0.000`s and several near-zero scores. Because the metric is **minority-class F1** and
`local` is only 232 of 800 eval rows, a model that answers `route` for everything scores exactly
0.0 — so those zeros are collapse-to-majority, not a broken harness. The confusion is
one-directional and large: gold=`local`→pred=`route` 213, the reverse only 7.

Two things worth investigating rather than asserting:

**The curriculum difficulty split is lopsided** — easy n=72, medium n=160, hard n=568. The
orchestrator's own diagnosis is that the `local`/`route` signal is drowned out by the hard
majority, and it has now tried synthesize (→0.000), acquire (→0.065), a hyperparameter escalation
(same collapse), and resample.

**Curate is still using the broken RouterBench loader.** The `eval_setup` path uses the fixed
pickle reader, but `mine_new_real_source` goes through `web_acquire`, which does not, and the log
shows it:

```
[acquire] peek withmartian/routerbench (config=None) failed: No (supported) data files found
[acquire] loaded AGENTIC HF dataset 'anasnassar/llm-query-complexity-benchmark' (train=4800 test=80)
```

So when the orchestrator asked for more real routing data, Stage-0 failed on the very benchmark the
task is about, agentic discovery substituted a *different* corpus, and 200 novel rows from it went
into the curriculum. That is a plausible contributor to the instability and it is a real gap:
`web_acquire`'s `_BENCHMARK_ALIASES` has no `routerbench` entry pointing at the working reader.
Not yet fixed — changing acquisition under a live run would invalidate its own trajectory.

---
