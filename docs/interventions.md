# Interventions — what happens on every `iterate` call

Reference for the decision the orchestrator makes each turn and everything that follows from it.
Code is the authority; this file is kept in step with `agent/nodes/iterate.py`,
`agent/data_rebuild.py` and `agent/nodes/curate.py`.

---

## 1. The loop

```
eval_setup ──► train ──► evaluate ──► [rollback?] ──► iterate ──► train | curate | escalate | END
                 ▲                                                   │        │
                 └───────────────────────────────────────────────────┘        │
                            (hyperparameter: dataset held fixed)              │
                 ▲                                                            │
                 └────────────────────────────────────────────────────────────┘
                            (data_rebuild: curriculum grows, then retrain)
```

`evaluate` produces two numbers for every iteration — a **content** score (the task's metric) and a
**format** score (what fraction of predictions the scorer could read at all) — plus a per-difficulty
breakdown and a list of failure **categories** from the task's own taxonomy. `iterate` sees only
those aggregates, never raw eval text.

## 2. The decision

`iterate` asks the orchestrator for JSON with one `intervention`:

| intervention | means | routes to |
|---|---|---|
| `hyperparameter` | the data is fine, the fit is wrong | `train` (curriculum untouched) |
| `data_rebuild` | the data is the limit | `curate`, then `train` |

Deterministic routes that pre-empt the orchestrator entirely:

| condition | route |
|---|---|
| score ≥ `stop_threshold` | stretch-goal check, then `downward_probe` or `END` |
| no gain > 2% over the last 15 evals | `escalate` (or `END` under `single_model`) |
| 30 evals on one model | `escalate` |
| turn/wall-clock budget spent | `END` |
| score regressed vs best | `rollback` first, then `iterate` again |

If the orchestrator call fails or returns invalid JSON, `apply_iteration_policy` picks by score
band (`<0.80` → data, `≥0.80` → hyperparameter) and `fallback_data_rebuild_plan` fills in the plan.

## 3. `hyperparameter`

Five fields are tunable, and only five (`iterate._TUNABLE_HYPERPARAMS`):

`lora_rank` · `alpha_ratio` · `weight_decay` · `learning_rate` · `nr_epochs`

`lora_dropout` is fixed at 0.0. Batch shape (`micro_batch_size`, `gradient_accumulation_steps`,
`effective_batch_size`) is derived by the trainer from the device and is rejected if proposed. An
exact repeat of a tried `(dataset, hyperparameters)` identity is rejected and replaced with a
deterministic untried neighbour; pruned and rolled-back trials count as tried.

The curriculum is not touched, so the comparison is clean.

## 4. `data_rebuild`

**The curriculum is cumulative.** Cold start loads gold rows; every rebuild ADDS to what is there;
rows leave only via quality control or the eval firewall. There is no target size and nothing
re-draws from a pool it has already drawn from.

Exactly **two** sub-strategies exist. A plan names one.

### 4.1 `mine_new_real` — add real rows

A ladder, stopping as soon as it has rows:

1. **Re-read known sources.** For each dataset in `TaskSpec.mining_sources` that
   `state["source_progress"]` does not mark exhausted, ask its loader for a larger slice than we
   have consumed. Every loader takes a head slice, so a bigger ask returns a superset and the tail
   is novel by construction. Free — no provider call, no LLM mapping, no schema risk.
   A source is marked **exhausted** only when the loader returns *fewer* rows than asked for; that
   is the only reliable evidence a head slice has hit the end of the split.
2. **Web research**, only once every source is exhausted. Exa finds candidate HuggingFace datasets,
   the orchestrator judges fit and maps columns onto our schema, and rows are taken per-row.
3. **Retirement.** After `MAX_FAILED_DISCOVERY_ROUNDS` (2) consecutive discovery rounds that
   contribute zero novel rows, `mine_new_real` is retired for the rest of the run: the orchestrator
   is told so in its prompt, and a plan that still asks for it is rewritten to
   `surgical_synthesis`.

Each of those transitions is logged loudly, because "mining added nothing" and "mining had nothing
left to add" are different facts.

What a candidate dataset is **not** rejected for (all relaxed 2026-08-19):

- extra columns we do not need — the mapping takes what it needs and ignores the rest;
- rows duplicating the curriculum — those are dropped per-row by `curate._dedupe_into`;
- rows overlapping the held-out eval set — those are dropped per-row by the eval firewall;
- its own internal train/test split structure — irrelevant, since mining consumes only the train
  side. The check that compared a discovered dataset's train slice to its own test slice was a
  tautology: both were sliced from the front of the same split, so it always "found" overlap equal
  to `max_test` and rejected every candidate.

A source is useless only when *nothing* survives per-row filtering.

### 4.2 `surgical_synthesis` — add generated rows

Targets the failure **categories** that are costing the most points. Categories come from the task's
own scorer (`TaskSpec.failure_category`), so they name something measured:

| task | example categories |
|---|---|
| `xlam_bfcl`, `calendar_json` | `unparseable_output`, `undeclared_function`, `wrong_function`, `wrong_call_count`, `wrong_arguments` |
| `ner_bc5cdr` | `unparseable_output`, `no_entities_predicted`, `entities_hallucinated`, `wrong_entity_type`, `wrong_span_boundaries` |
| `clinc150`, `routerbench`, `proactive_listening` | the real (gold, predicted) class confusions |
| `gsm8k` | `empty_output`, `no_numeric_answer`, `wrong_value` |
| `dialogsum` | `empty_output`, `unrelated_output`, `partially_correct` |

Budget per category is proportional to its failure count, bounded to
`[SURGICAL_MIN_ROWS_PER_CATEGORY, SURGICAL_MAX_ROWS_PER_CATEGORY]`. A category already targeted whose
count did not fall is **exhausted** and skipped, so the run stops paying for something that is not
responding.

Two row shapes, derived from the task rather than chosen:

- **closed label space** (`clinc150`, `routerbench`, `proactive_listening`) → a new *input* for an
  existing class. The row inherits a real anchor's label, so the target cannot be wrong; only the
  phrasing can, and that is what the teacher pass checks.
- **open-ended target** (everything else) → a whole new *(input, answer)* pair. The teacher invents
  both halves, so the row is gated twice: by an **exact programmatic verifier** where one exists,
  then by the teacher's own answer check.

| task | exact verifier | what it proves |
|---|---|---|
| `xlam_bfcl` | `verify_function_call_row` | parses; calls a declared tool; arguments exist in its schema; required arguments present |
| `calendar_json` | `verify_calendar_row` | all of the above, plus datetimes parse, `end` follows `start`, an unstated duration is 60 minutes, the event resolves near the request's own reference instant |
| `ner_bc5cdr` | `verify_ner_row` | every span appears verbatim in the row's own text, no duplicates |
| the other five | none | only the teacher's judgement is available; logged as such |

### 4.3 Every teacher prompt is built from the task brief

At cold start, after real data is loaded, the orchestrator is shown real rows and writes a
**task brief** (`agent/task_brief.py`): what the benchmark is, the exact output contract, and the
likely failure modes. It is logged in full and stored on the run state, and every synthesis and
verification prompt is built from it.

This replaced a one-line description keyed by task *type*, under which `xlam_bfcl` and
`calendar_json` were described identically — omitting every convention that makes a calendar row
correct, so a verifier judging against it was judging its own guess.

Worked examples shown to the teacher are always **real rows** sampled from the task's own training
split, never orchestrator-invented: a wrong example is worse than none.

### 4.4 After either sub-strategy

1. New rows are deduplicated into the curriculum (`_dedupe_into`) — only novel normalized texts.
2. Eval firewall, per row.
3. Chain-of-thought annotation, if the task asks for it (`TaskSpec.cot_annotation`).
4. Quality control — the steps *this task declared*, in order (`TaskSpec.quality_controls`).
5. Eval firewall again, then the viability floor.
6. Artifact written to `artifacts/dataset_v{N}.jsonl`; composition recorded in `last_curation`.

**If the rebuild added zero rows, that is logged as an ERROR.** An intervention was chosen, a plan
was built, and the mechanism it named could not do the thing it exists to do — so training this
iteration would repeat the previous one exactly.

## 5. Threshold movement

The orchestrator may **lower** the goal down to `initial_stop_threshold` when the failures look like
a model-capacity limit, and it is asked whether to **raise** it once a goal is cleared. Both are
audited (`threshold_raises`, `threshold_lowers`) and the value in force is stamped on each DAG node,
so the post-run accuracy chart draws the goal as a step line rather than one flat final value.

## 6. What is recorded per iteration

On the DAG node: content score, format score, metric name, the threshold in force, the config label
(mutable hyperparameters only), the intervention with its sub-strategy and row counts, the
orchestrator's full hypothesis, and the curriculum composition (`pi.D.composition`).

In the final report: per-tier baseline → first fine-tune → best, and a per-iteration table with
content and format side by side.
