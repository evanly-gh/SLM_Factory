# xlam single debug

*2026-08-17 — audit + implementation*

Reference run: **`slm-xlam-single3-l40s-38566712`** — `single_model` on
`Qwen/Qwen3-4B-Instruct-2507 [Q4_K_M]`, 16 iterations, 7h42m, $1.34, best `ast_arg_match`
**0.8530** against an untrained baseline of **0.8010** and a goal of 0.9053. Terminated on
stagnation, did not converge.
Log: `logs/slurm/slm-xlam-single3-l40s-38566712.out`.

Companions: [08-17-task-status.md](08-17-task-status.md),
[08-16b-train-serve-skew.md](08-16b-train-serve-skew.md),
[08-18-exact-verifiers-fewshot.md](08-18-exact-verifiers-fewshot.md).

**Headline: the run's reports were describing a different run.** Six of its eight data rebuilds
announced 250–500 rows of synthesis and produced **zero**, silently, because `function_call` was
missing from the synthesis dispatch table — which also means the exact verifiers built for that
exact path had never executed in production. Both mining rounds rejected every candidate for a
train/test overlap the loader itself had manufactured, and the two canonical source repositories
were discovered by Exa and then dropped, unprobed, by a shortlist cap. The orchestrator read all of
this as "data thinness ruled out" and spent the remaining thirteen iterations on hyperparameters.

Five bugs fixed (B291–B295, B296), two design gaps opened (B297, B298), 29 regression tests added
in `tests/test_xlam_single_debug_findings.py`. Everything below was re-read against the code.

---

## 0. Note naming convention (Q1)

`docs/Evan's Notes/` is now `MM-DD-two-or-three-words.md` — day and month only, no year, short
title. All 24 existing notes were renamed and every cross-reference in the repo was updated
(19 files). Same-day notes get a letter suffix, as `08-05b`/`08-15b` already did.

| was | is |
|---|---|
| `2026-07-26.md` | `07-26-ner-math-review.md` |
| `2026-07-28-followups.md` | `07-28-followup-fixes.md` |
| `2026-07-28-task-suite-benchmark-research.md` | `07-28b-benchmark-research.md` |
| `2026-07-31-six-task-benchmark-selection.md` | `07-31-benchmark-selection.md` |
| `2026-08-01-firewall-filtering-rebuild-and-escalation.md` | `08-01-pipeline-deep-dive.md` |
| `2026-08-02-eval-firewall-test-agent-synth-prompt.md` | `08-02-firewall-test-agent.md` |
| `2026-08-04-clinc150-run-review-questions-and-log-changes.md` | `08-04-clinc150-review.md` |
| `2026-08-05-orchestrator-synthesis-escalation-answers.md` | `08-05-orchestrator-synthesis-escalation.md` |
| `2026-08-05b-hardware-sizing-synthesis-and-frozen-diagnosis.md` | `08-05b-hardware-sizing-diagnosis.md` |
| `2026-08-05c-hard-negative-removal-escalation-and-rollback-context.md` | `08-05c-hard-negative-removal.md` |
| `2026-08-05d-hypothesis-length-output-budget-and-extraction-failures.md` | `08-05d-extraction-failures.md` |
| `2026-08-06-six-benchmark-tasks-reference.md` | `08-06-benchmark-tasks-reference.md` |
| `2026-08-10-dialogsum-run-postmortem-and-open-issues.md` | `08-10-dialogsum-postmortem.md` |
| `2026-08-11-throughput-and-sizing-fixes.md` | `08-11-throughput-sizing-fixes.md` |
| `2026-08-12-two-new-tasks-and-loader-blockers.md` | `08-12-new-tasks-loaders.md` |
| `2026-08-13-overnight-four-task-run-log.md` | `08-13-overnight-run-log.md` |
| `2026-08-14-routerbench-postmortem-crashes-and-the-ner-teacher-baseline.md` | `08-14-routerbench-postmortem.md` |
| `2026-08-15-routerbench-contamination-qc-audit-and-stretch-goals.md` | `08-15-contamination-qc-audit.md` |
| `2026-08-15b-label-space-lockdown-synthesis-quality-and-open-decisions.md` | `08-15b-label-space-lockdown.md` |
| `2026-08-16-extraction-collapse-tier-confounds-and-synthetic-data-verdict.md` | `08-16-extraction-collapse-verdict.md` |
| `2026-08-16-train-serve-skew-single-model-run-and-pipeline-launch.md` | `08-16b-train-serve-skew.md` |
| `2026-08-17-task-status.md` | `08-17-task-status.md` |
| `2026-08-18-exact-verifiers-fewshot-synthesis-and-whole-eval-splits.md` | `08-18-exact-verifiers-fewshot.md` |
| `2026-08-19-corrupt-label-ablation-mining-flow-and-calendar-gold.md` | `08-19-corrupt-label-ablation.md` |

---

## 1. How easy / medium / hard are decided (Q2)

Your description is right in shape and wrong in two details.

The code is `agent/nodes/test_agent.py`. It takes the **smallest and largest unique base model IDs
from `state["feasible_models"]`** — the hardware-filtered pool, so on this cluster usually
Qwen3-0.6B and Qwen3-4B-Instruct-2507 — and runs the whole eval set through **both at BF16**, once,
at cold start:

```62:67:agent/nodes/test_agent.py
            smallest, largest = _unique_base_endpoints(feasible_models)
            if smallest is None or largest is None:
                raise ValueError("no unique base-model endpoints")
            cf = correctness_fn or _zeroshot_correctness(eval_set, task_type, log)
            small_ok = cf(smallest.model_id)
            large_ok = cf(largest.model_id)
```

Then:

```69:77:agent/nodes/test_agent.py
            for t in texts:
                small_passed = small_ok.get(t, False)
                large_passed = large_ok.get(t, False)
                if small_passed and large_passed:
                    easy.append(t)
                elif large_passed:
                    medium.append(t)
                else:
                    hard.append(t)
```

**Correction 1 — precision.** Both probes run at **BF16**, not "4B at max precision vs 0.6B at
lowest". Quant siblings are deduped away before ranking, because a quantisation level is a
deployment variant rather than a distinct capability endpoint. `_zeroshot_correctness` calls
`run_eval(eval_set, model_id, model_id, ...)` with no quant argument, which loads the base artifact.

**Correction 2 — the fourth case.** Small right + big wrong falls through the `else` into **hard**,
not medium. Nothing about such an item is hard, and it inflates the apparent difficulty of the
bucket the orchestrator weights most heavily. Filed as **B298** (⚪ minor: it is a real logical
flaw, but the population is small and it never made the run take a wrong turn).

**What the buckets do and do not drive.** Bucketing runs **once**, at cold start
(`agent/nodes/cold_start/eval_setup.py:487-521`), and is frozen with the eval set. Every later
evaluation slices the *current* model's pass/fail by those frozen buckets. They feed reporting and
the orchestrator prompt. They do **not** select training rows: `plan["difficulty_buckets"]` is
normalised and stored but nothing reads it back to sample with, and no training row is ever tagged
with an eval difficulty (`row.get("_difficulty")` in `curate.py` always reports `"unassigned"`).
So when the orchestrator says it "weights buckets inversely to accuracy", that weight is advisory
and has no mechanical effect.

On this run: easy=479, medium=300, hard=221 of 1000 BFCL rows.

---

## 2. Graphics (Q3) — fixed

Four changes in `agent/run_graphics.py`, plus the state plumbing they needed.

### 2.1 The threshold moves, and the chart said it did not

`iterate_node` can **lower** the goal (when the failures look like a capacity limit, down to
`initial_stop_threshold`) and **raise** it (as a stretch goal once one is cleared). The chart drew
`state["stop_threshold"]` — the *final* value — as one `axhline`. On a run that lowered its goal
that line sits *below* iterations the loop had actually judged as failures.

Worse, only raises were recorded anywhere. Lowers just overwrote the field, so the bar a run was
actually held to was unrecoverable afterwards.

Three fixes:

- `evaluate_node` now stamps the in-force threshold on each DAG node. It runs before `iterate_node`,
  so the value is the one this score was judged against.
- `iterate_node` appends to a new `state["threshold_lowers"]` audit list, mirroring
  `threshold_raises`; both are persisted to `scores.json`.
- The chart draws a **step** line (`where="post"`) and annotates each change with ▲/▼ and the new
  value. Records predating the field carry the value forward, so the line is never discontinuous.

### 2.2 Composition is now explicit per iteration

Every band is labelled with its row count inside the bar, the curriculum total sits above each
stack, and the rows a rebuild *added* are called out separately (`+90 syn`, `+140 mined`) — because
a 90-row synthetic band on a 3,235-row curriculum is under 3% of the bar height and has nowhere to
put a number, which is exactly why this run's charts looked like nothing was ever added.

### 2.3 Gold is divided by source dataset

`last_curation` gained `gold_by_source`, a provenance × source cross-tab. The existing
`source_composition` counts *every* row per source and so cannot answer "how much of the **gold**
came from which corpus" once a rebuild has mixed in mined or synthetic rows. When gold comes from
more than one dataset the band is drawn as one sub-band per source, largest first, with a white
divider between them and each labelled. A single-corpus run gets one solid band — no spurious
dividers. (For this run gold is 100% `xLAM-60k / BFCL`, so one band is correct; see §5 for why
per-row source tagging is coarser than it should be.)

### 2.4 x-axis ticks every iteration

`MaxNLocator(integer=True)` fixed fractional ticks but still thinned to a round stride, so a
16-iteration tier was labelled 2, 4, 6, … Now `MultipleLocator(1)`; past ~24 iterations labels
shrink and rotate rather than being dropped.

---

## 3. The trajectory table (Q4) — fixed

Before:

```
   Iter    Score  Pruned  Config                     Intervention
      4  0.8260       ✗  LoRA r=32 a=64 drop=0 wd=0.01 lr=1e-04 ep=5 mb=8 ga=1 eb=8 [carry-fwd best]  data_rebuild
```

Three problems, all fixed:

**`[carry-fwd best]` is redundant.** Holding the best config is the *definition* of a
non-hyperparameter intervention, and the intervention is named in the next column. Removed from
`agent/nodes/train.py`.

**Four of the nine fields can never change.** `_TUNABLE_HYPERPARAMS` in `agent/nodes/iterate.py` is
exactly `{lora_rank, alpha_ratio, weight_decay, learning_rate, nr_epochs}`, and the deterministic
neighbour search in `training/hparams.py` enumerates the same five. `lora_dropout` is pinned at 0.0
and `micro_batch_size`/`gradient_accumulation_steps`/`effective_batch_size` are trainer-derived
batch shape the orchestrator is explicitly forbidden to set. `_label_for` now emits only the five:

```192:205:agent/nodes/train.py
def _label_for(cfg: dict) -> str:
    """Name a config by the fields a hyperparameter intervention can actually change.
    ...
    """
    return (
        f"r={cfg['lora_rank']} a={cfg['lora_alpha']} "
        f"wd={cfg['weight_decay']:g} lr={cfg['learning_rate']:.0e} "
        f"ep={cfg['nr_epochs']}"
    )
```

**`data_rebuild` did not say what it rebuilt.** New `format_intervention_detail` in
`agent/pipeline_status.py` reads the node's own `pi.D.composition` and names the sub-strategy, the
rows it produced, and for mining the source:

```
data_rebuild/synthesize: +240 synthetic row(s) (240 novel)
data_rebuild/synthesize: 0 synthetic rows kept
data_rebuild/mine-new-real: +300 mined row(s) from https://huggingface.co/datasets/Foo/bar
data_rebuild/mine-new-real: 0 new rows (no_novelty, 8 source(s) rejected)
```

After:

```
   Iter    Score  Pruned  Config                              Intervention
      4  0.8260       ✗  r=32 a=64 wd=0.01 lr=1e-04 ep=5      data_rebuild/synthesize: 0 synthetic rows kept
```

That second line is the one that would have made this whole investigation unnecessary.

---

## 4. How many rows the loop takes, and can it take more (Q5)

**It takes a capped slice, not the whole dataset.** For a curated benchmark
(`SLM_BENCHMARK_TASK=xlam_bfcl`):

```111:124:agent/nodes/cold_start/eval_setup.py
    max_train = int(int(state.get("curriculum_size_target") or 1000) * 0.65)
```

`curriculum_size_target` starts at `CURRICULUM_SIZE_FLOOR = 5000`, so `max_train = 3250`, and
`load_xlam_bfcl` turns that straight into an HF split slice `train[:3250]`. Eval is capped
separately at `_EVAL_SIZE_CAP = 1000` and round-robins across the six AST-gradable BFCL categories.
The eval firewall then removed 15 train rows that normalised-matched a BFCL eval row, leaving the
**3,235** you see on every iteration.

**Where `target_rows=5000` comes from.** `agent/data_sizing.py`:
`raw = floor × (0.5 + novelty) × size_factor`, `novelty = 1 − zero_shot_baseline`,
`size_factor = clamp(1e9 / n_params, 0.5, 2.0)`. For the 4B model with no baseline yet:
`5000 × (0.5 + 0.500) × 0.50 = 2500`, then clamped up to the 5000 floor. So the target is the
**floor winning over a smaller computed value** — it is not evidence that 5000 rows were wanted.

**Can it take more from the same local dataset later? No — and that is a design gap.**
`state["train_examples"]` is set once in `eval_setup` and never re-sliced. The remaining ~57,000
xLAM rows are sitting in the local HF cache, addressable by a one-line change to the split
expression, and completely unreachable by the loop. What runs instead:

- **Gold fill from the train pool** re-draws from those same 3,235 rows, so from iteration 2 onward
  it is 3,235 rows and **0 novel**. This is not a bug — it is how every curriculum is assembled —
  but it cannot add material.
- **`acquire`** goes to `mine_additional_real_rows`, which tries local bundles, then
  `load_benchmark_dataset`, then paid Exa discovery. `_BENCHMARK_ALIASES` in
  `data/loaders/web_acquire.py:66-87` has **no `xlam`/`bfcl` entry** (nor `calendar_json`), so the
  canonical loader is never reached and mining goes straight to paid discovery. Filed as **B297**.
  The `routerbench` entry two lines above carries a comment describing this exact failure being
  fixed for that task; xlam and calendar were never added.

**If the dataset were not local:** it would be fetched from the hub on demand. xLAM goes through
`datasets.load_dataset` and BFCL through `hf_hub_download`, both caching under `HF_HOME`
(`/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache` on this cluster). There is no offline mode
and `HF_HUB_OFFLINE` is not referenced anywhere; a cache miss is a network fetch. xLAM is a **gated**
repo, so it needs the `HF_TOKEN` that `tests/pipeline/run.py` loads from `.env` before any loader
import — without it, `eval_setup` fails at load time.

---

## 5. What Exa was used for (Q6)

8 Exa searches, $0.056, all `stage=acquire_dataset_discovery` — 4 in each of the two `acquire`
rounds (iterations 4 and 15). Each is paired with a Claude `acquire_schema_mapping` call (8 of
those, part of the $1.28 Anthropic total).

Exa's only job here is **finding candidate HuggingFace dataset repositories**:

```990:996:data/loaders/web_acquire.py
        r = tracked_exa_call(
            exa.search_and_contents,
            f"HuggingFace dataset for {query} site:huggingface.co/datasets",
            stage="acquire_dataset_discovery",
            model="search-and-contents",
            num_results=n, type="auto", text={"max_characters": 400},
        )
```

Repo IDs are regexed out of the result URLs and text; Claude is then asked to map each candidate's
columns onto our row schema. Exa's page *content* is not used as training data on this path — the
web-scraping ladder that does that (`_exa_round`, `stage=acquire_exa`) is a last resort that never
fired here.

**So why call it at all when the data is already local?** Because mining's contract is *novel* rows,
and it does not know that "novel" and "a different repo" are different things. It asks for local
data first, gets the 3,235 rows it already has, counts zero novel, and falls through to paid
discovery — where it pays to rediscover mirrors of the same corpus. The right fix is B297: give the
task a benchmark alias so the canonical loader can serve unseen rows from the cache, and paid
discovery is never reached.

---

## 6. `input` / `gold` / `raw` / `parsed`, and how output is parsed (Q7)

These four are **log labels**, not stored fields. `eval/harness.py` prints a sample of predictions
after every eval:

```181:187:eval/harness.py
        print(f"        input : {_clip(rows[i].get('text'))}")
        print(f"        gold  : {_clip(gold, 60)}")
        print(f"        raw   : {_clip(raw_outputs[i])}")
        print(f"        parsed: {_clip(predictions[i], 60)}{flag}")
```

- **input** — the eval row's `text`: the user request.
- **gold** — the correct answer (`answer`), a JSON call list for this task.
- **raw** — the exact string the model emitted, before any processing.
- **parsed** — what the scorer managed to extract from `raw`, which is what gets compared.

The pipeline is: `build_prompts` → `infer` → `extract_predictions` → `score`. For function calling,
extraction is `_parse_calls`: try `json.loads` on the whole string; if that fails, regex out the
first `[...]` or `{...}` and parse that (so fences and prose are tolerated); wrap a lone object in a
list; then require every element to be a dict with a `name` and a dict-valued
`arguments`/`args`. Anything else returns `None`:

```62:75:eval/scorers/function_call.py
def _parse_calls(raw: object) -> list[dict] | None:
    """Parse a model/gold string into a list of {name, arguments} calls, or None if it does
    not parse as the expected shape. A lone object is accepted and wrapped in a list."""
    if isinstance(raw, (list, dict)):
        parsed = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            # Tolerate fences / prose: grab the first JSON array or object.
            match = re.search(r"\[.*\]", text, re.DOTALL) or re.search(r"\{.*\}", text, re.DOTALL)
```

`None` is an **extraction failure**: `format_valid = 0.0`, content score `0.0`. A parsed-but-wrong
answer is `format_valid = 1.0`, content `0.0`. Both count equally against `ast_arg_match`; the
`format_valid` mean, reported alongside in `per_class`, is what separates them. This distinction is
what caught B290 — every prediction began with two stray `<think>` tags, so JSON that survived the
prefix scored and JSON that did not failed extraction, producing the 0.0000–0.6120 spread.

---

## 7. Merge logging (Q8) — fixed

A 3-shard merge emitted 16 lines: two tqdm bars replayed at every step, one
`Copied model-0000N.safetensors` per shard, the local-snapshot and hub-cache banners, and a final
`Merge process complete`. None of it is actionable — the merge either produces a verified snapshot or
raises.

`_MergeProgressLine` in `training/lora_trainer.py` now stands in for stdout/stderr for the duration
of the merge, recognises the phase markers, and rewrites **one** carriage-returned line that advances
in place and is terminated once with a newline. In a log file that is literally a single line:

```
      [merge] Qwen/Qwen3-4B-Instruct-2507: merged to 16-bit in 81s (3 shard(s)) → artifacts/merged/.../merged
```

On failure it ends the line before the traceback, so the two do not interleave. The "pinned merge
base" note is collected and printed after the line closes rather than being swallowed by it.

Separately, the sentence *"Synth-fill was removed on 2026-08-16 — padding to a heuristic target with
generated rows is not worth the label risk"* is gone from both places it appeared. The rationale
lives in the code comment where it belongs; the log now just states the shortfall.

---

## 8. The `Plan yield` line (Q9)

```
Plan yield: {'status': 'no_novelty', 'previous_rows': 3235, 'final_rows': 3235,
             'novel_rows': 0, 'novel_fraction': 0.0};
composition=[{'strategy': 'resample', 'kind': 'resample-fill', 'rows': 3235, 'novel_rows': 0}]
```

Built at `agent/nodes/curate.py:1097-1138`. Field by field:

| field | meaning |
|---|---|
| `previous_rows` | rows in the curriculum before this rebuild |
| `final_rows` | rows after it |
| `novel_rows` | normalised texts in the new curriculum that were **not** in the old one |
| `novel_fraction` | `novel_rows / len(final_texts)` |
| `status` | `"novel"` if `novel_rows > 0`, else `"no_novelty"` |
| `composition` | per-producer breakdown, keyed by the `_strategy_origin` each producer stamps on its rows |

**Yes — this rebuild added nothing.** It rewrote the curriculum from exactly the same 3,235 texts.
And your instinct is right that this makes the iteration close to pointless: the curriculum is
byte-for-byte equivalent in content, so training on it is a re-run of the previous iteration with a
different row order.

**What it triggers downstream: nothing.** `no_novelty` does not force a different intervention next
turn, does not skip training, and does not count as a wasted iteration. It is passed to the
orchestrator as advisory context and persisted in `last_curation`. Given how many iterations this
run burned on empty rebuilds, that is arguably too permissive — but I have not changed routing here,
because the honest fix is to make the rebuilds *work* (B291, B293, B297) rather than to add a rule
that hides them.

One thing to watch when reading these: `novel_fraction` is **1.0 on iteration 1**, because
`previous_rows` is 0 and everything is novel by definition. The orchestrator misread exactly that as
evidence that mining had delivered — see §12.

---

## 9. What happened with Exa and the schema checker (Q10)

The candidate list, both rounds:

```
['edbuildingstuff/bfcl-ft-data', 'lockon/xlam-function-calling-60k',
 'tuandunghcmut/BFCL_evaluation_notebooks', 'bitsydarel/Berkeley-Function-Calling-Leaderboard',
 'BitAgent/bfcl_shuffle_full', 'minpeter/xlam-function-calling-60k-parsed',
 'Salesforce/xlam-function-calling-60k', 'gorilla-llm/Berkeley-Function-Calling-Leaderboard']
```

### 9.1 The peek failures were real, and the error text was misleading

Four candidates genuinely could not be opened — a repo with no loadable data files, a repo that does
not exist, a repo with unsupported file types, and one whose JSON shards disagree on a column type
(`Column(/function/[]/parameters/properties/year/default) changed from number to string in row 189`).
All four are the remote repository's problem, not ours. Skipping them is correct.

What was wrong was the reporting. `str(e)[:80]` cut the message mid-path, so
`Couldn't find any data file at /mmfs1/gscratch/intelligentsystems/evanly/SLM_Fac` looked like a
local directory being passed as a hub ID. **It never was** — `hf_id` is always the repo string, and
that path is HuggingFace's own cache directory, where it correctly looked and found nothing.
`_peek_failure_reason` now classifies instead of truncating, one line per candidate:

```
[acquire] edbuildingstuff/bfcl-ft-data: SKIPPED — no loadable data files in the repo
[acquire] BitAgent/bfcl_shuffle_full: SKIPPED — malformed data files (inconsistent JSON schema across shards)
```

### 9.2 The overlap rejections were WRONG, and self-inflicted (B293 — fixed)

```
agentic lockon/xlam-function-calling-60k dataset REJECTED by schema/integrity/overlap validation
  (normalized train/test text overlap (80 rows))
```

**80 is exactly `max_test`.** Every single test row overlapped — which is the signature of a
tautology, not of contamination. Here is the mechanism.

`_mapped_split_names` resolves the test split, and a repo with only a `train` split falls all the way
through to `tr`:

```1110:1114:data/loaders/web_acquire.py
def _mapped_split_names(splits, mapping):
    tr = mapping.get("train_split") or ("train" if "train" in splits else splits[0])
    te = mapping.get("test_split") or ("test" if "test" in splits else
                                       ("validation" if "validation" in splits else tr))
    return tr, te
```

`_materialize_from_mapping` then loaded `train[:max_train]` **and** `train[:80]` — both from the
front. So test ⊂ train by construction, and `_validate_discovered_splits` rejected the source for an
overlap this function had just manufactured:

```1238:1242:data/loaders/web_acquire.py
    overlap = normalized_text_overlap(train, test)
    if overlap:
        raise ValueError(
            f"{source}: normalized train/test text overlap ({len(overlap)} rows)"
        )
```

Note the check has **zero tolerance** — any overlap at all rejects the whole source — whereas the
Stage-0 benchmark path *strips* train-side overlap first and only fails if any remains. Agentic
discovery got the strict check without the cleanup step.

Most instruction-tuning corpora on the Hub ship a single split, so this was a **guaranteed rejection
for the common case** — including for `Salesforce/xlam-function-calling-60k` itself, which is
train-only. The fix takes a disjoint window when the mapper collapses test onto train
(`train[:300]` and `train[300:380]`), which makes the integrity check measure real contamination
again.

To be clear about what the rejection was *not*: it was never about overlap with our held-out BFCL
eval set. That firewall is separate, runs in `eval_setup` and again on every mined row, and reported
`0 row(s) removed` throughout this run.

### 9.3 The canonical sources were found and then thrown away (B294 — fixed)

`Salesforce/xlam-function-calling-60k` and `gorilla-llm/Berkeley-Function-Calling-Leaderboard` — the
two authoritative sources for this task — are at **positions 7 and 8**. The loop probes
`candidates[:6]`. They were discovered, logged, and never touched, while six broken community
mirrors consumed the entire budget. The log printed `candidates[:8]`, actively implying all eight had
been considered.

Cause: per-query hit lists were **concatenated**, so the first query's entire result set outranked
every later query's best hit, and the canonical repos came from the third query. Now round-robin
interleaved, so each query's top hit lands near the front. The log also states both numbers
honestly: `Exa found 8 candidate HF dataset(s); probing 6`.

### 9.4 So how does "mine new real rows" work?

`strategy: "acquire"` → `mine_additional_real_rows`:

1. Build `seen` = normalised texts of every row already in the pool ∪ every eval row.
2. **Local bundles** (`data/local/`). If they yield novel rows, stop — paid budget untouched.
3. **`load_benchmark_dataset`** via `_BENCHMARK_ALIASES`. This is the step that should serve xlam and
   does not (B297).
4. **Paid Exa rounds**, metered by a durable ledger (`data/acquisition_budget.py`, 9 rounds per run):
   Exa finds repos → peek 2 rows → Claude maps columns → `load_dataset` and convert → schema /
   integrity / overlap validation → per-row dedupe against `seen`.
5. Survivors pass the eval firewall, merge permanently into `state["train_examples"]`, and are tagged
   `_provenance: "mined_real"`, `_strategy_origin: "mine_new_real_source"`.

Every gate that rejects logs a reason. On this run it reached step 4 twice and every candidate died
at validation or peek.

---

## 10. "I thought we got rid of resample and fill synthesis" (Q11)

Two different things were conflated, and the naming was actively responsible.

**What was actually removed (2026-08-16): `synth-fill`.** That padded every curriculum up to
`target_rows` with generated rows. It is gone; `_synth_fill_to_target` does not exist, and
`tests/nodes/test_curate_synth_fill.py` asserts it stays gone.

**What is still live, and correctly so:**

- **`resample-fill`** — drawing gold rows from the existing train pool. What was removed was
  `resample` as an *orchestrator-selectable strategy*. The mechanism is the universal filler that
  supplies every curriculum's gold rows; without it there would be no gold data at all.
- **`fill-synth`** — balanced synthesis across the label space under the `synthesize` strategy. A
  different thing from `synth-fill`, one character apart.

So the log was reporting live mechanisms under names that read like dead ones. Rather than delete
mentions of machinery that is running, I renamed the kinds to say what they do:

| `_strategy_origin` | was | now |
|---|---|---|
| `resample` | `resample-fill` | `train-pool-gold` |
| `synthesize` | `fill-synth` | `balanced-synth` |
| `synth_fill` | `synth-fill` | *(removed — no longer a named kind)* |

The banner is now `▶ BALANCED SYNTHESIS`, and `test_no_two_kind_names_are_confusable` fails if two
kinds ever differ only by word order again. The row tags are unchanged, so historical datasets and
resumed checkpoints still read correctly.

Two stale claims that were **lying to the orchestrator** are also fixed: the iterate system prompt
told it *"the curriculum is synth-filled up to the system-computed target when real data falls
short, so that target is the size you are actually training on"* — false since 2026-08-16, and
precisely backwards for a run training on 3,235 rows against a 5,000 target. Same claim in
`config/config.py`.

---

## 11. Does surgical synth work by itself, and why 0 rows? (Q12)

Two separate answers, and the second is the run's biggest bug.

### 11.1 "surgical already produced 0" was a category error, not a failure

Surgical synthesis is **classification-only**:

```435:436:agent/nodes/curate.py
    if task_type == "classification":
        surgical_rows = _surgical_synthesize(
```

`xlam_bfcl` is `function_call`, so `_surgical_synthesize` was **never called**. The log said
"surgical already produced 0", which reads as a failure of something that never ran, and the banner
advertised `kind=surgical-synth + fill-synth` for a task that can only reach one of them. Both now
tell the truth:

```
▶ BALANCED SYNTHESIS: 450 row(s) balanced across the label space
  (surgical-synth does not apply to function_call)
```

On classification it can run standalone: it takes up to 20% of the synthesis budget, so if it fills
that, balanced synthesis is skipped; if it returns nothing, balanced synthesis gets the full budget.
It needs confusion pairs from `test_report`, skips pairs whose prediction is
`__EXTRACTION_FAILED__` (not a class), and skips pairs it has already targeted without improvement.

### 11.2 The dataset shows no synthetic data because synthesis produced none, ever (B291 — fixed)

**This is the serious one.** `synthesize_examples` dispatches on task type and ended in a bare
`return []`:

```879:903:data/curriculum.py
    if task_type in ("classification", "NER"):
        ...
    if task_type in _GENERATION_FAMILY:
        ...
    return []
```

`_GENERATION_FAMILY` was `{math_reasoning, code_generation, generation, multilingual,
structured_extraction}`. **`function_call` is in neither branch.** Every synthesis call on this task
returned an empty list, silently, with no log line naming a cause.

The evidence, six times over:

| iter | announced | `DATA REBUILD kinds` |
|---|---|---|
| 1 | `FILL SYNTHESIS: 500 row(s)` | `resample-fill=3235 (3235 novel)` |
| 3 | `FILL SYNTHESIS: 400 row(s)` | `resample-fill=3235 (0 novel)` |
| 7 | `FILL SYNTHESIS: 350 row(s)` | `resample-fill=3235 (0 novel)` |
| 9 | `FILL SYNTHESIS: 450 row(s)` | `resample-fill=3235 (0 novel)` |
| 11 | `FILL SYNTHESIS: 300 row(s)` | `resample-fill=3235 (0 novel)` |
| 14 | `FILL SYNTHESIS: 250 row(s)` | `resample-fill=3235 (0 novel)` |

2,250 rows announced, 0 produced, teacher endpoint reachable and logged as such on the line above
each one. `Provenance: {'train_anchor': 3235}` on every single dataset report.

**And it means the exact verifiers had never run in production.** `verify_function_call_row` and
`verify_calendar_row` in `data/synth_verifiers.py` — the whole point of the
[08-18 note](08-18-exact-verifiers-fewshot.md) — are wired up through `curate._verifier_for`, which
correctly returns a verifier for `function_call`. It was passed to a function that returned before
using it. `_synthesize_new_correct` was clearly written *for* these tasks: it pins `tools` from the
anchor, a field only a function-calling row has.

Fixed: `function_call` and `diff` added to `_GENERATION_FAMILY`, and the fallthrough now says so
loudly rather than returning silently. `test_every_registered_task_type_can_synthesize` asserts that
every task type in `TASK_METRIC_NAMES` has a synthesis path, so this cannot recur for a new task.
Verified end to end against a stub teacher: generate → exact programmatic verify → teacher answer
verify → keep, with `tools` correctly pinned from the anchor and undeclared-function rows dropped.

`calendar_json` is also `function_call` and was equally affected — its synthesis has never worked
either.

---

## 12. Does mining work? (Q13)

**Yes, in general.** It has produced rows on other tasks:

- `slm-ner-bc5cdr-cse-38569606` — `mine-new-real=213 row(s) (213 novel)` from local bundles.
- `slm-routerbench-cse-38569605` — `mine-new-real=200 row(s) (200 novel)` from the benchmark loader.
- `slm-routerbench-l40s-38493142` — `[mine] paid-round-1: candidates=4800 novel=300` from paid Exa
  discovery.

**No, not for xlam**, and it could not have. It used mining twice and both failed, for the stack of
reasons in §9: no benchmark alias (B297), so it went straight to paid discovery; the canonical
sources dropped by the shortlist cap (B294); and every mirror that did load rejected by a
self-inflicted overlap (B293). Three independent causes, of which two are now fixed and the third
(B297) is the one that would make it work reliably.

### The consequence that cost the run

This is why the bugs matter more than they look. At iteration 4 the orchestrator wrote:

> *"Since prior yield showed the pool was fully novel (3235/3235) yet still failed, the fix is to
> bring in genuinely new REAL rows (acquire)"*

and by iteration 8, having tried both:

> *"Both data strategies (synthesize anchored rows, acquire genuinely novel real rows at 3235/3235
> novelty) also failed to move this confusion, **ruling out a simple data-thinness explanation**"*

It ruled out data thinness on the evidence of two interventions that **added zero rows**, and it
read the iteration-1 `novel_fraction: 1.0` — which is 1.0 only because `previous_rows` was 0 — as
proof that mining had delivered 3,235 novel real rows. Having "ruled out" data, it spent the
remaining thirteen iterations on hyperparameters, plateaued at 0.8530, and terminated on stagnation.
Its reasoning was sound. Its inputs were wrong.

---

## 13. Other findings (Q14)

### 13.1 The orchestrator was reasoning about a constant (B296 — fixed)

For every non-classification, non-NER task, `build_test_report` labelled all failures with a fixed
string:

```200:208:agent/nodes/test_agent.py
        else:
            # Open-ended targets/predictions may contain the raw held-out answer.
            # Report only an aggregate verifier category for these tasks.
            gold = str(
                failure.get("error_type")
                or failure.get("judge_category")
                or "gold_verifier"
            )[:64]
            predicted = "incorrect"
```

Nothing set `error_type` for function calling, so the sole "confusion pair" was always
`gold_verifier → incorrect` with count = the failure count the orchestrator already had. It then
wrote paragraphs about it: *"the dominant confusion gold_verifier->incorrect (147) essentially
unchanged since iter2"* — a sentence about a constant, repeated across a dozen iterations, and used
as evidence for hypotheses.

The scorer already computes everything needed to say something real. `failure_category` in
`eval/scorers/function_call.py` now splits failures into four categories that call for four
different interventions:

| category | means | points at |
|---|---|---|
| `unparseable_output` | output is not JSON calls at all | format / chat-template problem |
| `undeclared_function` | called a tool the row does not declare | prompt or unwinnable row |
| `wrong_function` | right shape, wrong tool | tool-selection capability |
| `wrong_call_count` | right tools, wrong number of calls | parallel-call handling |
| `wrong_arguments` | right tool, wrong argument values | argument extraction |

`build_test_report` already reads `error_type`, so this flows straight through with no change there.
On this run, the ~147 failures would have separated into these buckets — which would have told the
orchestrator whether to reach for data or for the template.

### 13.2 The "synthesis unavailable" reason was hardcoded and wrong

When synthesis produced nothing, `allocation_fallbacks` recorded
`"synthesis produced no rows (endpoint unavailable or cheap mode)"` — a cause, asserted without
checking, that was false every time in this run (the endpoint was healthy and logged one line
above). It sent me looking at vLLM first. Now it points at the `[synth]` lines instead of naming a
cause, and a rebuild that generates nothing says so explicitly:

```
⚠ SYNTHESIS PRODUCED 0 ROWS — this rebuild adds nothing the previous curriculum did not
  already have. See the [synth] lines above for the cause.
```

### 13.3 HuggingFace loader noise (Q12, second part) — fixed

Lines 3626–3648 of the log are `datasets` internals: two `Downloading data: 100%|...| 39/39` bars
and, per attempt, `Failed to load JSON from file '...BFCL_v3_irrelevance.json' with error
<class 'pyarrow.lib.ArrowInvalid'>` plus three `Generating train split: 0 examples` lines — repeated
because `_peek_hf_dataset` retries with a second config, and again on the second acquire round.

`DATASETS_VERBOSITY=error` cannot suppress it: those messages *are* logged at ERROR, and the
progress bars are raw tqdm writes rather than HF progress-bar API calls. So the interception has to
be at the file-descriptor level. `quiet_output()` in `agent/logging_setup.py` does that, and the
discovery worker now runs inside it — safe because the worker accumulates its own verdict lines in a
list that the parent replays, so nothing of value travels on those descriptors. Per candidate the
log is now one line, as in §9.1.

### 13.4 Things I checked and that are fine

- **`Decision failed validation ... — asking the orchestrator to correct itself (1 reask)`** at
  iteration 8: the re-ask worked. Correct behaviour.
- **`provider=local calls=1018 failures=11`**: an 11/1018 failure rate on synthesis/judge calls,
  all retried. Not a defect.
- **Eval firewall**: `0 row(s) removed` on every rebuild after the initial 15. Correct — no new rows
  were ever introduced to contaminate.
- **The merge itself**: succeeded every time, pinned to the official base, and the B290 fix held —
  the fine-tuned model beat the 0.8010 baseline from iteration 1 onward with no stray tags.
- **Rollback**: fired correctly on every regression; the best checkpoint was never lost.
- **Curriculum under-target**: 3,235 vs 5,000 is expected and intended post-synth-fill. Not a bug,
  though it is a consequence of B297.

---

## 14. Status of everything changed

| # | Finding | Status |
|---|---|---|
| B291 | `synthesize` a silent no-op on `function_call`/`diff`; exact verifiers never ran | 🟢 fixed |
| B293 | Discovery rejected single-split repos for an overlap it manufactured | 🟢 fixed |
| B294 | Canonical sources discovered then dropped by the shortlist cap | 🟢 fixed |
| B295 | Accuracy chart drew one flat threshold; lowers never recorded | 🟢 fixed |
| B296 | Open-ended failures reported as a constant confusion pair | 🟢 fixed |
| B297 | No benchmark alias for `xlam_bfcl`/`calendar_json`; local corpus unreachable | 🔴 open |
| B298 | Small-right/big-wrong eval rows bucketed as `hard` | ⚪ minor |

Also changed, not bugs: note naming, merge log collapsed to one line, HF loader noise silenced,
rebuild kinds renamed, trajectory table Config/Intervention columns, composition chart, x-axis ticks,
two stale prompt/config claims corrected.

Tests: `tests/test_xlam_single_debug_findings.py` (29). Full suite 1,192 passed, 2 skipped, 1
pre-existing unrelated failure (`test_capability_prompts.py::test_code_planner_and_model_choice_prompts_target_apps_introductory`
— `agent/task_planner.py` is unmodified from HEAD).

---

## 15. What I would do next

**B297 first, and re-run.** Everything else in this run was downstream of having no way to add real
data. A benchmark alias for `xlam_bfcl` that serves rows `3251+` from the already-cached corpus
turns `acquire` from a guaranteed no-op into the strongest lever available, and costs no Exa or
Claude calls at all. `calendar_json` needs the same.

**Then re-run with synthesis actually working.** B291 means the synthetic-data verdict in
[08-16-extraction-collapse-verdict.md](08-16-extraction-collapse-verdict.md) has never been tested on
a format-bound task — every such measurement was of a curriculum containing zero synthetic rows. The
exact verifiers are the strongest quality gate in the project and they have never been exercised
against real generations.

**Consider making an empty rebuild cost something.** A `no_novelty` rebuild currently trains a full
iteration on a content-identical curriculum. Now that the reports say so plainly, the orchestrator
may route around it on its own — worth measuring before adding a rule.
