# Two new tasks, the next four runs, and three loaders that are already broken

*2026-08-12 — reference note*

Answers four questions in order: where the data actually comes from (preloaded, or discovered by
an agent at runtime), what the eval harness is and where it gets the eval set, what the BC5CDR
NER task already did and how it was wired, and which benchmark to use for a new
calendar/scheduling NL→JSON task. It ends with the thing that matters most for scheduling work:
**three of the four runs you want to launch next cannot load their data today**, verified live
against `datasets 4.3.0` on the cluster, not inferred from the code.

Companion docs: `08-06-benchmark-tasks-reference.md` (the six as of Aug 6 — still
accurate on scoring and prompts, now incomplete on the task list),
`07-31-benchmark-selection.md` (why those six), and
`08-11-throughput-sizing-fixes.md` (the most recent implementation state).

---

## 0. The short version

**Preloaded, not discovered — on the curated path.** The six benchmark tasks resolve through a
hardcoded registry to hardcoded HuggingFace IDs in dedicated loader modules. No LLM, no Exa, no
search. The agentic discovery-and-curation ladder exists and is real, but it only runs on the
**autonomous** path — which is what your BC5CDR NER run used, because
`tests/pipeline/run_ner_l40s.slurm` sets `TASK=...` and never sets `SLM_BENCHMARK_TASK`. The two
paths are mutually exclusive and chosen by one `elif` in `eval_setup.py`.

**The eval harness is custom.** Not lm-eval-harness, not HELM. It is `eval/harness.py` plus one
scorer module per task type under `eval/scorers/`. The eval set is **never** a holdout of the
training curriculum — it is built once from the loader's official `test` split, frozen to disk,
and firewalled out of training data by normalized-text match.

**Three loaders are dead.** Verified by actually calling `load_dataset` on the cluster:

| Target | Result |
|---|---|
| `Salesforce/xlam-function-calling-60k` | `DatasetNotFoundError` — **gated**, needs an access request |
| `gorilla-llm/Berkeley-Function-Calling-Leaderboard` | `DataFilesNotFoundError` — no resolvable data files |
| `withmartian/routerbench` | `DataFilesNotFoundError` — ships only `.pkl`, which `datasets` cannot read |
| `tner/bc5cdr` | `RuntimeError: Dataset scripts are no longer supported` — **but has a working fallback** |

That kills `xlam_bfcl` and `routerbench` outright and would have killed the NER re-run too, except
someone already wrote a script-free JSON fallback for BC5CDR (`web_acquire.py:153-161`) and it
still resolves. Section 5 has the evidence and the fixes.

**For the calendar task, use TOPv2 `reminder` + SGD `Calendar_1`, scored as `function_call`.**
Section 4. The single best academic reference point is SMCalFlow, but its output language is
Lispress and its test set is withheld, so it is a citation and a difficulty calibration rather
than something to train on directly.

---

## 1. Where the data comes from

### 1.1 Two paths, one `elif`

Everything routes through `eval_setup_node`. The branch order is load-bearing:

```306:320:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/agent/nodes/cold_start/eval_setup.py
        from data.loaders.web_acquire import acquire_dataset
        train_examples, test_examples = acquire_dataset(
            plan, description=state.get("description", ""),
            target_examples=max(_gold_target, 120),
            benchmark_max_train=_bench_train, benchmark_max_test=_bench_test,
            meta=acquire_meta,
        )
    elif os.environ.get("SLM_BENCHMARK_TASK"):
        # Curated non-autonomous path: load one of the six benchmark loaders by env key. This
        # feeds the same build_eval_set + overlap-firewall path the classification branch uses.
        train_examples, test_examples = _load_named_benchmark(
            os.environ["SLM_BENCHMARK_TASK"], state, acquire_meta)
    elif task_type == "classification":
        from data.loaders.sms_spam import download_sms_spam
        train_examples, test_examples = download_sms_spam()
```

There are four branches in total, in priority order:

| Branch | Trigger | Behaviour |
|---|---|---|
| Frozen bundle | `SLM_SHARED_DATASET_DIR` set | checksum-verified `train.jsonl` / `test.jsonl`, no network |
| **Autonomous** | orchestrator produced a `task_plan` | `web_acquire.acquire_dataset` — the agentic ladder |
| **Curated** | `SLM_BENCHMARK_TASK=<key>` | hardcoded loader, live HF pull |
| Fallback | `task_type == classification`, nothing else set | UCI SMS-spam over plain HTTP |

The reason the curated branch sits *below* the autonomous one is that a `task_plan` wins. That is
why `tests/pipeline/run.py` explicitly forces `task_plan=None` when `SLM_BENCHMARK_TASK` is set —
otherwise the env var would be silently ignored and you would get a discovery run you did not ask
for.

### 1.2 The curated path is fully hardcoded

```31:38:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/agent/nodes/cold_start/eval_setup.py
NAMED_BENCHMARK_TASK_TYPES: dict[str, tuple[str, str]] = {
    "clinc150": ("classification", "CLINC150 (clinc_oos/plus)"),
    "dialogsum_samsum": ("generation", "DialogSum + SAMSum"),
    "xlam_bfcl": ("function_call", "xLAM-60k / BFCL"),
    "coedit": ("diff", "CoEdIT (grammarly/coedit)"),
    "routerbench": ("classification", "RouterBench"),
    "medqa": ("classification", "MedQA-USMLE-4-options"),
}
```

Each key maps to a loader module whose HF ID is a module-level constant — `XLAM_ID`,
`BFCL_ID`, `HF_ID`, and so on. Nothing about the choice of dataset is dynamic. The only runtime
decision is how many rows to pull, which comes from the curriculum sizing computed upstream.

### 1.3 The autonomous path is a five-rung ladder, and only rungs 3–5 are agentic

`data/loaders/web_acquire.py::acquire_dataset` tries, in order:

1. **Local offline bundle** — `data/local/{name}/train.jsonl`, checksummed. No network.
2. **Deterministic Stage-0 benchmark** — a normalized alias map (`_BENCHMARK_ALIASES`,
   `web_acquire.py:66-81`) turns the planner's free-text benchmark name into a catalog key
   (`gsm8k`, `bc5cdr`, `conll`, `apps`, `mbpp`, `samsum`, `sms_spam`, `fpb`, `arc`) with pinned
   HF IDs and fallbacks. Still fully hardcoded.
3. **Agentic HF discovery** — Exa searches the Hub, candidates are peeked, and **Claude** maps
   the winner's columns onto the row schema (`_llm_map_dataset`, `web_acquire.py:953-994`).
4. **Bounded Exa web scrape** — at most three diversified rounds.
5. **Orchestrator gold synthesis** — Claude writes seed examples, but only if the pool is below
   the viability floor.

So "agentic data discovery" is a **fallback**, not the default. A named benchmark like BC5CDR
short-circuits at rung 2 and never touches Exa or Claude for acquisition. Your NER run resolved
`"Biomedical NER (BC5CDR)"` through `_COMPOSITE_BENCHMARK_MARKERS` (`web_acquire.py:86-93`) to the
`bc5cdr` key, and pulled real `tner/bc5cdr` rows.

Curriculum synthesis is separate from acquisition and uses a **local** teacher, not Claude:
`data/synth_client.py` talks to a local vLLM server running Qwen3.6-35B-A3B, and the module
comment is explicit that it must never silently fall back to the Claude API.

### 1.4 What is actually on disk right now

`data/local/` holds ~295 MB of frozen offline bundles: `apps` (269M), `samsum` (11M), `gsm8k`
(5.3M), `go_emotions` (5.2M), **`bc5cdr` (2.7M)**, `emotion` (2.3M), `mbpp` (514K). Each has
`train.jsonl`, `test.jsonl`, `manifest.json`, `checksums.sha256`.

The BC5CDR bundle is already in the exact post-conversion row schema, entity spans and all:

```json
{"text": "Naloxone reverses the antihypertensive effect of clonidine .", "entities": [{"text": "Naloxone", "type": "Chemical"}, {"text": "clonidine", "type": "Chemical"}]}
```

5,096 train rows and 5,865 test rows, pinned to `tner/bc5cdr` revision
`f68cdc7db924369241e7868656f583072acd4e90`. This is the safest source for the NER re-run.

The shared HF cache is `HF_HOME=/mmfs1/gscratch/intelligentsystems/evanly/.hf-cache`, set in
`tests/pipeline/_l40s_task_body.sh:27`. It is 184 GB across 30 dataset repos, mostly model
weights. **None of the six curated benchmarks are materialized under `data/local/`** — they all
download live, which is precisely why the breakage in section 5 matters.

---

## 2. The eval harness

### 2.1 The call chain

`agent/nodes/evaluate.py::evaluate_node` → `eval/harness.py::run_eval` → `_run_eval_local`, which
dispatches on `task_type` to one scorer module and then runs four steps:
`scorer.build_prompts()` → `infer_batch` (Unsloth) or `infer_batch_gguf` (llama.cpp) →
`scorer.extract_predictions()` → `scorer.score()`.

Two properties are worth internalizing because they explain most surprising numbers:

- **Quantized-honest.** The scored inference runs against the merged-and-quantized GGUF, not the
  bf16 LoRA-adapted model. The number you see is what the phone would get.
- **Baseline parity.** The Qwen reference baseline goes through
  `eval/endpoint_eval.py::measure_endpoint_baseline` and dispatches to the **same** scorers, so
  baseline and candidate are directly comparable. This is what makes the goal threshold
  meaningful.

### 2.2 One scalar, seven metric names

```58:68:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/eval/harness.py
TASK_METRIC_NAMES = {
    "classification": "macro_f1",
    "NER": "span_f1",
    "math_reasoning": "exact_match",
    "code_generation": "execution_pass@1",
    "generation": "judge_mean_0_1",
    # Format-bound: the comparison scalar is content-correctness; format_valid rides
    # alongside in per_class (see eval/scorers/function_call.py, eval/scorers/diff.py).
    "function_call": "ast_arg_match",
    "diff": "apply_match",
}
```

Whatever the metric is, the comparison scalar always lands in `EvalResult.f1` so every gate,
threshold, rollback rule and plot works unchanged across task types. `EvalResult.metric` carries
the honest name. When a log line says `F1=0.8628` on an NER run, that is span-F1.

For the two **format-bound** task types (`function_call`, `diff`) the scorer reports two numbers:
`format_valid` — did the output parse as the required structure at all — and `content_correct` —
were the fields right. `content_correct` is the scalar; `format_valid` rides in `per_class`
(`eval/scorers/function_call.py:158-179`). That split is the whole point of a format-bound task:
it tells you "the model cannot emit JSON" apart from "the model emits JSON with wrong arguments,"
which are different bugs with different fixes.

### 2.3 Where the eval set comes from

Not from the curriculum. `eval_setup_node` calls `data/eval_set.py::build_eval_set` on the
loader's **official `test` rows** (or `validation` for CoEdIT and MedQA), once, at cold start, and
freezes the result.

- Multi-class classification gets label-coverage round-robin sampling so rare labels survive.
- Everything else gets a shuffled top-N.
- Target defaults to 800 (`config.EVAL_SET_SIZE`), floor 30, seed 42.

The frozen set is written to `logs/runs/<run_id>/artifacts/eval_set.json`, and
`curate._exclude_eval_rows` removes any training row whose normalized text matches an eval row —
that is the firewall. In the NER run it removed 24 training rows.

For `xlam_bfcl` the separation is stronger than a split: it trains on xLAM-60k and evaluates on
BFCL, two independently constructed corpora, so the eval is a genuine transfer test.

---

## 3. The NER task: what already ran

### 3.1 How it is wired

`tests/pipeline/run_ner_l40s.slurm` is an **autonomous** run. It sets no `SLM_BENCHMARK_TASK`; it
sets a natural-language task string and lets the orchestrator plan:

```24:24:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/tests/pipeline/run_ner_l40s.slurm
export TASK="Fine tune a small model for biomedical named entity recognition on PubMed abstracts, extracting Chemical and Disease entity spans (BC5CDR dataset style), targeting deployment on my Samsung Galaxy S24 Ultra with 12GB RAM and 256GB storage"
```

The planner emitted `task_type=NER`, benchmark `BC5CDR`, labels `[Chemical, Disease]`, and
`stop_threshold=0.88`. `web_acquire` resolved that to the `bc5cdr` catalog key and called
`_load_ner_benchmark`, which converts token+BIO-tag rows into `{text, entities}` spans. Scoring is
`eval/scorers/ner.py::score` → `eval/metrics.py::entity_f1`, an **exact multiset match on
`(surface_text, entity_type)` pairs**. There is no partial credit for an overlapping span and no
credit for the right span with the wrong type.

Two L40S under the `auto-2gpu` profile: the Qwen3.6-35B synth/judge server on GPU 0, the pipeline
on GPU 1.

### 3.2 The run: `slm-ner-l40s-37531245`

| Field | Value |
|---|---|
| Data | `tner/bc5cdr`, train 3,403 / test 900 after conversion; 24 rows dropped by the eval firewall |
| Eval set | 900 frozen — 360 positive, 360 negative, 180 boundary |
| Models | Qwen3.5-2B (tier 2) then Qwen3.5-4B (tier 3), both merged to **Q4_K_M** GGUF for scoring |
| Best config | LoRA r=64, α=256, dropout 0.05, wd 0.05, lr 1e-4, 5 epochs, micro-batch 4 × grad-accum 8 = effective 32 |
| Tier 3 zero-shot baseline | span-F1 **0.0254** |
| Tier 2 best | **0.8476** |
| Tier 3 best | **0.8628** (iteration 46) — easy 0.887, medium 0.723, hard 0.630 |
| Threshold | **0.88** |
| Outcome | **failed to converge**; crashed at `curate` with `ValueError: bounded data_rebuild plan space is exhausted` |
| Cost / time | $13.43 (Claude $13.41 + Exa $0.02), 231 orchestrator calls, 142 train→eval iterations, **44.8 h** |

Train throughput was 20–34 samples/s, so the wall-clock was not dominated by training. It was
dominated by 142 rounds of merge → quantize → CPU-side GGUF inference over 900 rows, plus
orchestrator latency.

The near-zero tier-3 baseline (0.0254) against a 0.8628 fine-tuned score is the headline result
and it is a real one: a 4B instruct model zero-shot cannot emit the required span JSON at all,
and fine-tuning takes it to 0.86. The gap is the product.

### 3.3 Why it stalled at 0.8628, and what to change

Three things, in order of expected effect.

**The threshold was set above what the data supports.** Human inter-annotator agreement on
BC5CDR chemical/disease spans is in the low 0.90s, and published span-F1 for dedicated encoder
models (BioBERT-class, ~110M params) sits around 0.87–0.90 under exact-match protocols that do
not always agree with ours. Demanding 0.88 from a 4B decoder scored under exact
`(text, type)` multiset match, on a 900-row set that is deliberately 20% boundary cases, is close
to the ceiling. This is the same failure mode as the DialogSum run
(`08-10-dialogsum-postmortem.md` §1) and it is still the open policy
item. `config/benchmark_baselines.md` currently carries BC5CDR as `span_f1 | n/a`, which is why
the planner fell back to recall and guessed high.

**The gold pool ran out.** Curriculum target was 4,500 rows; the available gold pool after
firewalling capped out around 3,379, and the run actually trained on ~1,277–1,321. Every
`data_rebuild` strategy the orchestrator could propose had already been tried, which is literally
what the terminating exception says. The local `data/local/bc5cdr` bundle has 5,096 train rows —
more than the 3,403 the live loader produced — so pointing at the bundle is free extra data.

**There is known train/serve prompt skew.** Flagged at the end of the six-task reference and
still unfixed: `eval/scorers/ner.py`'s prompt includes a "Reply with `[]` if there are no
entities" clause that `training/lora_trainer.py::_training_turn` omits. B250 was exactly this
class of bug on DialogSum and cost real points. Fix it before re-running.

One more thing worth knowing before you re-run: the first GGUF load failed on the tier-2 baseline
eval and the baseline was recorded as **0.0**, while the actual iteration-1 zero-shot on the
merged model measured 0.8107. Any "improvement over baseline" figure computed from tier 2 in that
log is meaningless.

---

## 4. The new task: calendar request → structured event JSON

### 4.1 What the task is

Input is a free-text scheduling request — *"Schedule a meeting at 5pm tomorrow"*, *"lunch with
Sarah next Tuesday at the usual place, 90 minutes"*. Output is a single JSON object the host
application can hand to the Google Calendar API without further parsing. Success is binary and
mechanical, which makes it a good format-bound task and a genuinely useful on-device one: this is
the canonical thing you want a 1B model doing locally instead of shipping your calendar to a
server.

### 4.2 The benchmark survey

I checked availability and loadability of every candidate rather than trusting dataset cards.
Ranked by fitness for this specific task:

| Dataset | What it is | Size | Loads on `datasets` 4.3? | Verdict |
|---|---|---|---|---|
| [**SMCalFlow**](https://microsoft.github.io/task_oriented_dialogue_as_dataflow_synthesis/) | The canonical calendar semantic-parsing benchmark. Wizard-of-Oz dialogues about events/weather/places/people, each turn annotated with an executable dataflow program. | 41,517 dialogues / 155k turns | ✗ HF mirror `iohadrubin/smcalflow` is script-based; use the [GitHub tarballs](https://github.com/microsoft/task_oriented_dialogue_as_dataflow_synthesis/tree/master/datasets) | **Reference + calibration.** Output is Lispress, test set is withheld. |
| [**TOPv2**](https://huggingface.co/datasets/WillHeld/top_v2) `reminder` domain | Hierarchical semantic parses of reminder/scheduling utterances. `CREATE_REMINDER` with `TODO`, `DATE_TIME`, `RECURRING_DATE_TIME`, `PERSON_REMINDED`, `FREQUENCY`, `ATTENDEE`. | **17,840** reminder rows in train (124,597 total across 8 domains) | ✓ **verified working**, native parquet | **Primary training source.** |
| [**SGD** `Calendar_1`](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | Google's Schema-Guided Dialogue. The `Calendar_1` service defines `AddEvent` with required slots `event_name`, `event_date`, `event_time`, `event_location` — literally a calendar-insert schema. | ~1,600 calendar dialogues | ✗ HF mirror script-based; ✓ **raw JSON straight from GitHub** | **Primary eval source.** |
| [**MASSIVE**](https://huggingface.co/datasets/AmazonScience/massive) | Amazon's 51-language SLU set; `calendar_set` / `calendar_query` / `calendar_remove` intents with slot annotations. | 1M+ utterances, ~19k per language | ✗ script-based (`massive.py`) | Optional augmentation; needs the GitHub release for slots. |
| [`microsoft/ba-calendar`](https://huggingface.co/datasets/microsoft/ba-calendar) | Constraint-satisfaction scheduling: find a slot satisfying availability/buffer/priority constraints. | ~2k | ✓ plain jsonl | **Wrong task** — planning, not extraction. Named here so nobody picks it by keyword. |
| [`yananchen/natural_plan__calendar_scheduling`](https://huggingface.co/datasets/yananchen/natural_plan__calendar_scheduling) | Google's Natural Plan meeting-scheduling split. Same story. | ~1k | ✓ | **Wrong task.** |
| [Calendar-Event-Entity-Extraction](https://github.com/muskaanwalia098/Calendar-Event-Entity-Extraction) / [SmolLM2-360M-Text-2-JSON](https://github.com/pramodkoujalagi/smollm2-360m-instruct-text-2-json) | Community SLM fine-tunes on exactly this task, 8-field schema (`action`, `date`, `time`, `attendees`, `location`, `duration`, `recurrence`, `notes`). Reports 0.983 field-F1 and 1.00 JSON parse rate after tuning, from 0.29 / 0.86 base. | ~792 examples | ✓ jsonl in repo | **Not a benchmark** — too small and self-generated. But it is the closest published prior art and a useful sanity target. |
| [LLMStructBench](https://arxiv.org/html/2602.14743v1) | General NL→JSON extraction across 5 schemas, 995 manually verified samples, 22 models. | 995 | — | Good **metric design** reference; no calendar scenario. |

Two things fall out of this. First, there is no single clean "NL → Google Calendar JSON"
benchmark with a leaderboard — SMCalFlow is the closest and it predates the JSON-output era.
Second, the 792-example community set reporting near-perfect field-F1 after fine-tuning a 360M
model tells you the *easy* version of this task is nearly solved, so the task has to be built
hard enough to be worth measuring.

### 4.3 Recommendation

Register `calendar_json` as **`task_type=function_call`** and reuse the existing scorer verbatim.

The reasoning is that the `function_call` scorer already does exactly what this task needs and
nothing it does not. It parses the output as a JSON call list, checks the function name against
an allowed set, checks that every gold-required argument is present, compares values with light
type coercion so `3 == "3"`, and reports `format_valid` separately from `content_correct`. Model
a calendar insert as a single-tool call to `calendar.events.insert` and the whole thing works with
no new scoring code:

```json
{
  "text": "Schedule a meeting at 5pm tomorrow",
  "answer": "[{\"arguments\": {\"end\": {\"dateTime\": \"2026-08-13T18:00:00\"}, \"start\": {\"dateTime\": \"2026-08-13T17:00:00\"}, \"summary\": \"meeting\"}, \"name\": \"calendar.events.insert\"}]",
  "tools": [{"name": "calendar.events.insert", "parameters": {"summary": "string", "start": "object", "end": "object", "location": "string", "attendees": "array", "recurrence": "array", "description": "string"}}],
  "label": "function_call"
}
```

The argument names are the real Google Calendar API v3 Events resource fields, so a passing model
produces a request body the application can post unmodified. That is the difference between a
benchmark score and a shipped feature.

**Data plan.** Train on converted TOPv2 `reminder` rows (17,840 available — a real corpus, not a
few hundred synthetic ones), evaluate on converted SGD `Calendar_1` `AddEvent` frames. That
mirrors the `xlam_bfcl` design: two independently constructed sources, so the eval is transfer
rather than a holdout, and a model that memorized TOP bracket conventions gets no credit.

**The relative-date problem is the interesting part, and it must be handled explicitly.**
"tomorrow" is only resolvable against a reference timestamp. Both source corpora annotate the
*surface string* (`[SL:DATE_TIME at 5 pm ]`, `"event_date": ["6th of this month", "March 6th"]`),
not a resolved ISO datetime. Two options:

- **Pin a reference time in the prompt** — inject `Current date and time: 2026-08-12T23:40:00-07:00`
  as a preamble and require resolved ISO-8601 in the output. Harder, far more useful, and it makes
  the task genuinely non-trivial for a 1B model. The gold conversion resolves the surface string
  once, deterministically, at dataset-build time.
- **Emit the surface string** — trivially easier, and it just moves the hard part into the host
  application, which defeats the purpose.

Take the first. It is the only version of this task that produces something deployable, and it
gives the run somewhere to actually improve, which the NER run ran out of.

### 4.4 What has to be built

Nothing in the scorer or the harness. Four small pieces:

1. `data/loaders/calendar_json.py` — TOPv2 bracket-parse → JSON converter, SGD frame → JSON
   converter, relative-date resolution against a pinned reference timestamp, emitting
   `{text, answer, tools, label}`.
2. Registry entry `"calendar_json": ("function_call", "Calendar NL→JSON (TOPv2 reminder / SGD Calendar_1)")`
   in `NAMED_BENCHMARK_TASK_TYPES`.
3. `tests/pipeline/run_calendar_json_l40s.slurm` — `tests/pipeline/test_task_slurm_scripts.py`
   fails the build if a registry key has no matching Slurm script, so this is not optional.
4. Unit tests on the pure converters against in-memory samples, matching how every other loader
   in `data/loaders/` is tested.

And, before the run: pin a `stop_threshold` from a measured baseline rather than recall. Sections
3.3 and 5 both exist because that step keeps getting skipped.

---

## 5. The loader blockers, verified live

### 5.1 The evidence

Run on the cluster in the project venv, `HF_HOME` pointed at the shared cache:

```
datasets 4.3.0
OK   clinc150       ['text', 'intent']
OK   coedit         ['_id', 'task', 'src', 'tgt']
OK   medqa          ['question', 'answer', 'options', 'meta_info', 'answer_idx', 'metamap_phrases']
OK   topv2          ['domain', 'utterance', 'semantic_parse']
FAIL bc5cdr_tner    RuntimeError: Dataset scripts are no longer supported, but found bc5cdr.py
FAIL bc5cdr_spyy    DatasetNotFoundError: Dataset 'spyysalo/bc5cdr' doesn't exist on the Hub
FAIL massive        RuntimeError: Dataset scripts are no longer supported, but found massive.py
FAIL smcalflow_hf   RuntimeError: Dataset scripts are no longer supported, but found smcalflow.py
FAIL routerbench    DataFilesNotFoundError: No (supported) data files found in withmartian/routerbench
FAIL bfcl           DataFilesNotFoundError: No (supported) data files found in gorilla-llm/Berkeley-Function-Calling-Leaderboard
FAIL xlam           DatasetNotFoundError: 'Salesforce/xlam-function-calling-60k' is a gated dataset
```

Only `clinc150` and `dialogsum_samsum` have ever produced a run log. There is no
`slm-xlam-bfcl-*`, `slm-coedit-*`, `slm-routerbench-*`, or `slm-medqa-*` anywhere in
`logs/slurm/`. Those four loaders are unit-tested against in-memory samples — which is why
`convert_xlam_rows` and `convert_routerbench_rows` are pure functions and pass — but the live
`load_dataset` calls beneath them have never been executed. The tests were never going to catch
this.

### 5.2 `xlam_bfcl` — both sources broken, two different reasons

**xLAM is gated.** `Salesforce/xlam-function-calling-60k` is `gated: auto`, meaning a
click-through license. Fix: accept the terms on the dataset page under the account whose token is
in the environment, and make sure `HF_TOKEN` is actually exported into the Slurm environment.
`_l40s_task_body.sh` does not currently export one.

**BFCL has no resolvable data files.** The repo holds 52 files named `BFCL_v3_simple.json`,
`BFCL_v3_parallel.json`, `BFCL_v3_multi_turn_base.json`, and so on. None match the split-name
patterns `datasets` uses to auto-build a config, so `load_dataset(BFCL_ID, split="train[:N]")` at
`data/loaders/xlam_bfcl.py:85-87` cannot resolve anything — and the `except (ValueError, KeyError)`
around it does not catch `DataFilesNotFoundError`, so the run dies rather than falling back.

Beyond loading, the schema is also wrong. BFCL v3 splits the prompt and the gold answer across
*separate* files (`BFCL_v3_simple.json` and a matching `possible_answer/` entry), and gold answers
are lists of acceptable values rather than a single call. `convert_xlam_rows` expects
`answers` on the same row, so even a successful load would silently drop every row and produce an
empty eval set.

Fix: name the file explicitly and join against the answer file —
`load_dataset("json", data_files={"test": "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/BFCL_v3_simple.json"})` — and add a BFCL-specific converter.
Widen the `except` to `Exception` while you are there. Same pattern as the BC5CDR fallback that
already works.

### 5.3 `routerbench` — the format is unreadable

`withmartian/routerbench` ships `routerbench_0shot.pkl`, `routerbench_5shot.pkl`,
`routerbench_raw.pkl` and nothing else. `datasets` has no pickle reader, so
`data/loaders/routerbench.py:77` fails on any split. The `except (ValueError, KeyError)` fallback
at line 83 does not catch it either.

There is a second, quieter problem underneath. `_correctness` looks for a field literally named
`small_model_correct` (`routerbench.py:22`); RouterBench actually stores one column per candidate
model, named after the model. Even with a working reader every row would return `None` and get
dropped, producing an empty dataset. Whoever wrote the loader wrote it against an imagined schema.

Fix: read the pickle directly with `pandas.read_pickle` via `huggingface_hub.hf_hub_download`,
then pick a real column as the routing boundary and pass it as `small_model_key`. That is a
schema decision, not a mechanical fix — you have to choose which model defines "small enough to
keep on device", and the honest choice is the one closest to what you would actually deploy.

### 5.4 `bc5cdr` — already survived this

The script-based load fails, but someone anticipated it:

```146:162:/mmfs1/gscratch/intelligentsystems/evanly/SLM_Factory/data/loaders/web_acquire.py
        "bc5cdr": [
            {"id": "tner/bc5cdr", "config": None, "tokens": "tokens",
             "tags": "tags", "splits": {"train": "train", "test": "test"}},
            {"id": "spyysalo/bc5cdr", "config": None, "tokens": "tokens",
             "tags": "ner_tags", "splits": {"train": "train", "test": "test"}},
            # datasets>=4 refuses script-based repos. Read T-NER's public official JSON
            # files directly as a script-free compatibility path.
            {"id": "tner/bc5cdr", "config": None, "loader": "json",
             "tokens": "tokens", "tags": "tags",
             "splits": {"train": "train", "test": "test"},
             "data_files": {
                 "train": ("https://huggingface.co/datasets/tner/bc5cdr/"
                           "resolve/main/dataset/train.json"),
                 "test": ("https://huggingface.co/datasets/tner/bc5cdr/"
                          "resolve/main/dataset/test.json"),
             }},
        ],
```

Candidates one and two now fail; candidate three works — verified, returns `['tags', 'tokens']`.
So the NER re-run will load, just after two failed attempts and a slow cold fetch.

Better: set `SLM_LOCAL_DATASET_DIR` and use the frozen `data/local/bc5cdr` bundle. It is rung 1 of
the ladder, needs no network, is checksummed, and carries **5,096 train rows against the 3,403 the
live loader produced** — which directly addresses the exhausted-gold-pool crash from §3.3.

---

## 6. The suite is now eight

| Key | `task_type` | Category | Source | Status |
|---|---|---|---|---|
| `clinc150` | classification | in-distribution control | `clinc/clinc_oos` (plus) | ✓ run, loads |
| `dialogsum_samsum` | generation | in-distribution control | `knkarthick/{dialogsum,samsum}` | ✓ run, loads |
| `xlam_bfcl` | function_call | **format-bound** | xLAM-60k → BFCL | ✗ **gated + unresolvable** |
| `coedit` | diff | format-bound | `grammarly/coedit` | ✓ loads, never run |
| `routerbench` | classification | data-scarce (router) | `withmartian/routerbench` | ✗ **pickle-only + wrong schema** |
| `medqa` | classification | data-scarce (medical QA) | `GBaker/MedQA-USMLE-4-options` | ✓ loads, never run |
| **`ner_bc5cdr`** | **NER** | **span extraction** | `tner/bc5cdr` / local bundle | ✓ run (0.8628 / 0.88), autonomous path today |
| **`calendar_json`** | **function_call** | **format-bound** | TOPv2 `reminder` → SGD `Calendar_1` | **to build** |

BC5CDR NER is listed here as a suite member because you are re-running it and because
`08-06-benchmark-tasks-reference.md` §1 explicitly excludes it. It runs through the
autonomous path today. Promoting it to a `SLM_BENCHMARK_TASK` key would make it reproducible
without depending on the orchestrator producing the same plan twice — worth doing, and cheap,
since the loader and scorer already exist.

Your two format-bound tasks are therefore `xlam_bfcl` and `calendar_json`, which is a good pairing:
same `task_type`, same scorer, same `format_valid` / `content_correct` split, but one is
open-domain function calling over arbitrary tool signatures and the other is a single fixed schema
with hard temporal reasoning inside it.

---

## 7. Suggested order

1. **`calendar_json` loader + Slurm + tests.** Nothing blocks it, TOPv2 is verified working, and
   it is the only one of the four with no external dependency to unblock.
2. **`ner_bc5cdr` re-run.** Point at `data/local/bc5cdr`, fix the train/serve prompt skew from
   §3.3, and set `stop_threshold` from a measured baseline rather than 0.88. The extra 1,693 gold
   rows in the bundle should also prevent the plan-space-exhausted crash.
3. **`routerbench`.** Needs a pickle reader and a schema decision about which model defines the
   routing boundary. Half a day.
4. **`xlam_bfcl`.** Needs the gate accepted, `HF_TOKEN` plumbed into Slurm, an explicit BFCL file
   list, and a BFCL-specific converter that joins prompts to `possible_answer` entries. The most
   work of the four.

---

## 8. Corrections to existing docs

- `08-06-benchmark-tasks-reference.md` — the registry is unchanged in code, but the
  claim that all six "pull live from HuggingFace at load time" is now only true for four of them.
  Update banner added.
- `07-31-benchmark-selection.md` — the selection rationale stands; the availability
  assumption behind two of the six does not. Update banner added.
- `config/benchmark_baselines.md` — BC5CDR still carries `span_f1 | n/a`, which is why the NER
  planner guessed 0.88. Measured-anchor row added, plus a placeholder for `calendar_json`.
- The stale-docs note at the end of the six-task reference flags the NER train/serve prompt skew.
  It is still unfixed and it now matters, because NER is being re-run.
