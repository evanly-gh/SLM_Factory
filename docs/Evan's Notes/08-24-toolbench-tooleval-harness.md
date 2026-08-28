# ToolBench is in as task #10, with a real ToolEval judge — and the paper it comes from does not add up

**Date:** 2026-08-24
**Companions:** `08-19b-task-registry-rebuild.md`, `08-18-exact-verifiers-fewshot.md`
**Paper:** [arXiv:2512.15943](https://arxiv.org/abs/2512.15943) — *Small Language Models for
Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning* (Jhandi,
Kazi, Subramanian, Sendas — AWS; AAAI 2026 Workshop on Agentic AI Benchmarks)

**Headline.** `toolbench` is registered, loads live, and is the suite's first task whose metric is a
genuine LLM-judge pass rate: ToolEval's own `check_answer_status` rubric, run by the Qwen3.6 teacher,
3 rounds per query, majority-voted, aggregated over the six G1/G2/G3 subsets by the paper's
query-count-weighted rule. The gold-replay ceiling is exactly **1.0000**, so the harness itself does
not cap the score. Full suite is 1,572 passing.

**Two things you should read before trusting any number this task produces.** §2 is why I do not
believe the paper's results — you asked me to spell out the ToolLLaMA-DFS discrepancy, and it turned
out to be one of six independent problems. §5 is the one deviation in my implementation that changes
what the metric means: there is no API server, so the model plans a whole path with no observations
fed back, which means pass rate partly rewards writing a *plausible* final answer. I added a
`undeclared_api_rate` column specifically to measure that, and it is the number I would watch first.

---

## 1. What the paper actually does

Worth pinning down because "ToolBench" is two different benchmarks and the paper never links a repo.

It is the **ToolLLM / OpenBMB ToolBench** ([arXiv:2307.16789](https://arxiv.org/abs/2307.16789),
`github.com/OpenBMB/ToolBench`): ~16,000 real RapidAPI endpoints, and solution paths where a model
interleaves a free-text `Thought` with an `Action` / `Action Input` call and terminates by calling a
pseudo-function `Finish`. It is **not** SambaNova's identically-named benchmark
([arXiv:2305.16504](https://arxiv.org/abs/2305.16504)), which is eight hand-built execution tasks
scored by success rate. Four independent confirmations in the text: "16,000 real-world APIs from
RapidAPI Hub", the G1/G2/G3 six-subset structure, ToolEval as the harness, and the
Thought/Action/Action Input format.

Their training set is stated as **187,542 examples**. That is exactly the row count of ToolBench's
own preprocessed `toolllama_G123_dfs_train.json`, so that file is what they used, essentially as-is —
despite the paper describing a bespoke transformation. Their preprocessing, in full: "System prompts,
user queries, and assistant responses were concatenated with appropriate delimiters." The scripts
"were generated using Amazon Q." No filtering, no dedup, no loss masking to assistant spans is
described.

Their claim: `facebook/opt-350m`, one epoch of TRL `SFTTrainer`, **77.55% ToolEval pass rate**,
against ChatGPT-CoT 26.00%, ToolLLaMA-DFS 30.18%, ToolLLaMA-CoT 16.27%, Claude-CoT 2.73%.

---

## 2. Why I do not believe the paper's numbers

You asked specifically about the ToolLLaMA-DFS line. Here it is, and then the five other things I
found while trying to reproduce the setup.

### 2.1 The baselines match no published source, in either direction

The paper reports **ToolLLaMA-DFS at 30.18%**. ToolBench's own README reports **ToolLLaMA-DFSDT at
66.7%** average pass rate. That is not a small gap — it is less than half the official number for
what should be the same model on the same benchmark with the same search strategy.

The full comparison:

| Model | Paper | Official ToolBench README | StableToolBench (SoPR) |
|---|---|---|---|
| ToolLLaMA-DFS(DT) | 30.18 | 66.7 | 54.6 |
| ChatGPT | 26.00 (CoT) | 64.8 (DFSDT) / 40.2 (ReACT) | — |
| ToolLLaMA-CoT/ReACT | 16.27 | 29.0 (ReACT) | — |
| Claude | 2.73 (CoT) | 6.8 (Claude-2 ReACT) | — |
| GPT-4-DFSDT | — | 71.1 | 60.6 (GPT-4-Turbo DFS) |
| **Their 350M SLM** | **77.55** | — | — |

Two readings, and both are bad. If the baselines were **copied** from the source papers, they were
copied wrong. If they were **re-evaluated**, no re-evaluation procedure is documented that would
explain a 36-point drop on ToolLLaMA — and the paper would need to explain why its re-run of every
baseline landed far below every published value while its own model landed far above.

The consequence for the headline: 77.55% would be, by a wide margin, the **highest ToolBench pass
rate ever published** — beating GPT-4 with DFSDT by six points, with a 350M-parameter model doing
greedy single-pass CoT. That is the claim, and the baselines are what it is measured against.

### 2.2 The Claude number is a format penalty, not a capability measurement

Claude-CoT at **2.73%** is the tell. A frontier model does not solve 2.7% of tool-use tasks. What
2.7% means is that the grader wanted a specific `Thought:` / `Action:` / `Action Input:` structure
and a general chat model does not emit it, so nearly every path failed to parse before anything was
judged. The paper's own limitations section half-concedes this, listing "the tight coupling between
training data and evaluation metrics." So the 51-point gap in the title is substantially a gap
between a model fine-tuned on the output format and models that were never given it.

This is measurable in my implementation rather than arguable: `format_valid` is reported separately
from the pass rate, so a model failing on shape is visibly distinguishable from one failing on
content. That is the B290 lesson applied here.

### 2.3 The context length is impossible as stated

The paper reports a max sequence length of **8192** on `facebook/opt-350m`. That checkpoint's
`max_position_embeddings` is **2048**. The claim cannot hold without position-embedding surgery,
which is never mentioned.

It also matters materially, not just pedantically. I measured real ToolBench prompts with a Qwen2.5
tokenizer: median **1,248** tokens, p95 **2,838**, p99 **3,973**, max **9,155**. A 2,048 context
truncates the prompt on roughly a third of rows — and the part that gets cut is the tail of the API
schema list, i.e. the names the model is supposed to choose between.

### 2.4 The test set size matches neither ToolBench's release nor anything else

The paper's Table 2 uses 200 queries for each of five subsets and 100 for G3_instruction, totalling
1,100. I checked what ToolBench actually ships:

| Source | G1_inst | G1_cat | G1_tool | G2_inst | G2_cat | G3_inst | Total |
|---|---|---|---|---|---|---|---|
| Paper's Table 2 | 200 | 200 | 200 | 200 | 200 | 100 | 1,100 |
| ToolBench `test_query_ids` | 100 | 100 | 100 | 100 | 100 | 100 | **600** |
| StableToolBench solvable | 163 | 153 | 158 | 106 | 124 | 61 | **765** |

The official released test-query-id files are 100 per subset. The paper's per-subset numbers are all
multiples of 0.5 for the G1/G2 sets and of 1.0 for G3, which is internally consistent with 200s and
a 100 — so it is not a typo in the table, they really evaluated on something of that size. It just
is not the released test set.

### 2.5 It cites StableToolBench and then does not use it

StableToolBench is cited in the abstract and the reference list. The methodology never uses it. The
paper also states its evaluator "assessed solution paths **without requiring live API execution**" —
which leaves genuinely unclear what tool observations the model saw during rollout, given RapidAPI's
2023 endpoints have substantially decayed. That decay is the entire reason StableToolBench exists.

### 2.6 The reference list is unreliable

Relevant because I spent time chasing citations that go nowhere:

- Ref [3] attributes ToolLLM to `arXiv:2304.08354`, which is *Tool Learning with Foundation Models*.
  Ref [15] gives the correct `2307.16789` for the same paper, so ToolLLM is double-cited with one
  wrong ID.
- Ref [4] attributes StableToolBench to "Z. Zhang, T. Yu, Q. Sun, …, `arXiv:2406.11939`". The real
  paper is Guo et al., `arXiv:2403.07714`.
- Ref [6] gives API-Blend as `arXiv:2406.01614` with a fabricated author list; the real one is Basu
  et al. (IBM), `arXiv:2402.15491`.
- The body cites "Zhou et al. (2023) on parameter-efficient fine-tuning methods like LoRA" as [25],
  which is LIMA — a data-quality paper with nothing to do with LoRA.

Several of the wrong author lists look like the ToolLLM author list permuted.

### 2.7 What I did with this

I implemented the **methodology** faithfully — the dataset, the output format, the judge rubric, the
majority vote, the six-subset weighted aggregation — and treated the **numbers** as unreplicated. So
this task gives us a real ToolEval harness we can measure our own models on. It does not give us
77.55% to chase, and I would not put that figure in a comparison table without the caveats above.

---

## 3. The data

### 3.1 Train: a 2 GB file read ~51 MB at a time

`Yhyu13/ToolBench_toolllama_G123_dfs` / `toolllama_G123_dfs_train.json`. Ungated and reachable; the
canonical `ToolBench/ToolBench` repo 404s for authenticated and unauthenticated reads alike. Its
train split is **187,542** rows — the paper's number exactly.

It is a single **2,002,419,658-byte** pretty-printed JSON array. `json.load` costs the whole 2 GB and
several GB resident to produce rows of which a cold start takes 5,000. It is served with byte-range
support (verified HTTP 206), so the loader reads a growing **prefix** and decodes objects out of it
incrementally — measured at 390 complete objects per 4 MiB, i.e. ~12,000 bytes per row.

The prefix is cached on disk and **extended**, not refetched. That is what makes rung 1 of the mining
ladder cheap here: `_reread_known_sources` calls `load` again with a larger `max_train` every
rebuild, so re-reading 6,000 rows after 5,000 costs the 11 MB difference rather than 2 GB. Cache
location honours `SLM_TOOLBENCH_PREFIX_CACHE`, else lands under `HF_HOME`.

### 3.2 One complete path per row

The `_dfs` file is **already exploded by step**: an `id` of `"Step 9: <query>"` means the trajectory
truncated after 9 assistant turns, and the same query appears at several depths. Measured on the eval
split: 762 rows over **300 distinct queries**. Consecutive depths share nearly all their text.

The loader keeps only rows whose **final assistant turn is `Finish` with
`return_type == "give_answer"`** — the complete, successful paths. Measured drop profile on a
6,144-row prefix:

```
kept 1509 complete paths (24.6%)
  path does not end in Finish ........ 3811   (incomplete step prefixes)
  path gave up rather than answering .. 736   (DFSDT explored and abandoned)
  Finish input is not valid JSON ......  46
  gold calls an undeclared API ........  41
  give_answer with empty final_answer ..  1
```

Three things fall out of that one rule, and this is why it is the rule rather than a filter chain:

1. **The target is a whole path**, which is what the eval asks the model to produce. Train and serve
   agree about the unit of work.
2. **Near-duplicate step prefixes disappear structurally**, without a similarity filter — which
   matters because a trigram filter *cannot* be used here. Rows are ~8,000 characters of shared API
   schema and step N vs N+1 differ by one action, so any threshold low enough to catch them also
   removes genuinely distinct trajectories. This is the `calendar_json` dedup argument, but stronger.
   `toolbench` therefore joins `dialogsum` and `calendar_json` in `test_three_tasks_deliberately_keep_near_duplicates`.
3. **Abandoned paths are excluded**, since a curriculum built from them teaches giving up.

The `gold calls an undeclared API` drop deserves a name: it is the ToolBench analogue of BFCL's
`simple_363`, which `data/loaders/xlam_bfcl.py` already drops for the same reason. The gold calls a
function the prompt never offered, the scorer rejects any undeclared call, so the row is unwinnable
by construction and would silently cap the ceiling below 1.0. It is **2.6% of otherwise-complete
train paths** and **4.3% (33/762) of the eval-split step rows**.

### 3.3 Eval: the six ToolEval subsets, solvability-filtered

Reproducing the six-subset structure needs each query's `api_list`. ToolBench distributes that only
inside 1.7 GB of `instruction/G*_query.json` — its `test_query_ids` files carry ids and nothing else.

StableToolBench republishes the same six subsets with `api_list` **inline**, one small file each, and
additionally filters them for **solvability** by majority vote of three frontier models. I used
those: **765 queries** (163/153/158/106/124/61).

The solvability filter is not a convenience — it is load-bearing for this metric. Pass rate asks "did
the model solve it". A query nobody can solve caps the achievable score below 1.0 and reports a
benchmark defect as a model failure. Since 765 < `eval_cap` 1000, nothing is truncated and the subset
proportions are exact. The aggregation weights by *actual* subset size, so it stays correct for
whichever set it is handed.

### 3.4 Rebuilding ToolBench's prompt byte-for-byte

This was the fiddly part and the place a silent disaster was most likely. Train rows arrive with
ToolBench's system prompt already baked in; eval rows ship as a raw `api_list` and no prompt, so it
has to be rebuilt. If the rebuild drifts, the model is fine-tuned on one input shape and scored on
another — B250/B290 — and here it would be **invisible**, because both prompts are ~5,000 characters
of API schema and nobody diffs that in a log.

So `data/loaders/toolbench_prompt.py` reproduces upstream exactly, from three files of
`OpenBMB/ToolBench`: `standardize` / `change_name` / `process_system_message` from `toolbench/utils.py`,
`FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION` from `Prompts/ReAct_prompts.py`, and
`api_json_to_openai_json` + the `Finish` dict + the numbered tool listing from
`Downstream_tasks/rapidapi.py`. Quirks reproduced rather than tidied:

- The API block is `str(functions)` — a **Python repr with single quotes**, not JSON. This is why the
  loader reads schemas back out with `ast.literal_eval` (works on 762/762 rows). Emitting JSON would
  have been tidier and would have put every eval row off-distribution.
- API names truncate to the **last** 64 characters (`[-64:]`), because the `_for_<tool>` suffix is
  what disambiguates same-named APIs.
- `example_value` is included when `len(str(default)) != 0`, so a default of `0` **does** produce one
  and `""` does not. A truthiness test would silently drop it for every numeric parameter.
- Reserved parameter names get an `is_` prefix — this is where xLAM's much-queried `is_id` comes
  from, and "correcting" it to `id` produces a call ToolBench's own schema rejects.

`test_a_real_api_entry_renders_exactly_as_upstream_rendered_it` pins the output against a function
dict **captured verbatim from a real training row**, and
`test_toolbench_eval_and_train_prompts_have_the_same_shape` asserts a train row and an eval row for
the same query produce identical `text`. Both pass.

One field the test files omit and ToolBench's prompt includes: each tool's prose description. I take
those from `stabletoolbench/ToolEnv2404` (11.7 MB, 11,792 descriptions). Verified against a training
row — `greyhound_racing_uk` resolves to exactly the string that row's system prompt contains. Without
it, every eval prompt would say `None` where every train prompt has real prose, in the largest block
of the prompt.

---

## 4. The eval harness

### 4.1 The metric, exactly as the paper defines it

1. The model produces a solution path for a query.
2. A judge assesses it against the query. Repeated N times, verdict is a **majority vote**. The paper
   uses ≥4 rounds with ChatGPT; per your call this uses **3** with the run's own Qwen3.6.
3. Each subset reports `passes / queries`.
4. The headline is `total passes / total queries` — a query-count **weighted** mean.

Point 4 is not a detail. I verified the rule against the paper's own tables:
`(78.5+74.0+79.0+80.5+74.5)×200 + 80.0×100 = 85,300`, over 1,100 = **77.545% → 77.55%**, which is
what they report. The unweighted mean of the six percentages is **77.75%**. So a careless
implementation produces a plausible-looking 0.2-point error, and
`test_the_paper_headline_number_is_reproduced_by_this_aggregation` pins the arithmetic.

The rubric is ToolEval's `check_answer_status` from
`toolbench/tooleval/evaluators/tooleval_gpt-3.5-turbo_default/template.txt`, its four rules verbatim.
Upstream delivers them as an OpenAI function-calling description and reads `answer_status` out of a
tool call; a local vLLM server is asked for the same JSON object directly. That is the only change.

### 4.2 Majority voting needs a non-zero temperature

At temperature 0, three assessments of the same path are the same assessment, the vote is a no-op,
and the judge cache correctly collapses them into one request. The ToolEval rubric therefore runs at
**0.7**, and the round index is part of the cache key so the rounds cache separately. This is the one
place I deliberately traded reproducibility for the metric's actual definition, and it is a rubric
property rather than a global, so `dialogsum` still judges at temperature 0.

`Solved` requires a **strict** majority. `Unsure` is a verdict of its own, not half a pass —
ToolEval's own `eval_pass_rate` counts only `Solved` — and `judge_unsure_rate` is reported so it is
visible when the metric is measuring the judge's uncertainty rather than the model.

### 4.3 What computation decides before the judge is asked

Three of ToolEval's rules are exact, so they run first and cost nothing:

- a path that never calls `Finish` produced no answer → Unsolved;
- `Finish` with `give_up_and_restart` is ToolBench's explicit "I could not do this" → Unsolved;
- a path over the API-call budget (the paper's 10 iterations) → Unsolved.

So every row the judge sees has already produced a syntactically complete, in-budget path claiming an
answer. Judge spend goes only to the question needing judgement, and a failure category exists for
every row without a judge call: `unparseable_path`, `no_finish_call`, `gave_up`, `malformed_finish`,
`empty_final_answer`, `budget_exceeded`, `undeclared_api`, `judged_unsolved`, `judge_unsure`. Eight
distinct interventions instead of B296's one constant.

### 4.4 One judge client, two rubrics

Rather than a second judge module, `LocalJudgeClient` now takes a `JudgeRubric` value — system
prompt, payload→message, reply→float, and sampling params. Everything below the prompt (endpoint
validation, the Qwen3.6 identity preflight, the process-safe disk cache, the bounded sliding
concurrency window, the fail-fast error surface) is rubric-independent and is the part that is hard to
get right.

The compatibility risk was the cache. `NUMERIC_RUBRIC` keeps the original constants, and I verified
the resulting keys and user messages are **byte-identical** to what the pre-refactor code produced,
so every score already on disk stays valid. `JudgeRubric.fingerprint()` hashes the system prompt as
well as the name, so rewording a rubric without bumping its name cannot serve stale scores — there is
a new test for that, which the old design could not have had.

---

## 5. The deviation that changes what the number means

**Read this before quoting a pass rate.**

Upstream ToolEval scores an **interactive** rollout: it calls RapidAPI, feeds each observation back,
and lets DFSDT backtrack. This pipeline has no API server, RapidAPI's 2023 endpoints have largely
decayed, and the paper itself says its evaluator needed no live API execution. So here the model
emits a complete path in **one generation** and is judged on it.

The consequence: **with no observations, the model cannot know what any API returned, so pass rate
partly measures writing a plausible final answer.** ToolEval's second prompt,
`parse_answer_status`, exists precisely to catch this — it cross-checks the final answer against the
tool nodes' messages — and it is unreachable without execution. Only `check_answer_status` is used.

The honest reading of a number from this scorer is "produced a well-formed path whose stated answer a
judge found responsive to the query", not "solved the task".

This is also, I think, the most likely explanation for the paper's headline being what it is. A model
fine-tuned to emit fluent ToolBench-shaped paths with confident final answers, graded by a judge that
sees only the query and the answer, should score very well — and a frontier model that declines to
invent an answer it cannot verify should score badly. That is roughly the pattern their table shows.

So I added a column for it: **`undeclared_api_rate`**, computed over *all* rows including passing
ones. A path calling an API the prompt never declared cannot have learned anything from it, so its
final answer was invented, and `check_answer_status` has no way to notice. It is the direct measure
of how much of a pass rate could be fluent fabrication. It is **not** subtracted from the score,
because upstream does not subtract it either — but if it is high, the pass rate is not what it looks
like. Watch it first.

**Upgrade path if we want the real thing:** a StableToolBench-style virtual API server, where the
teacher simulates API responses (they use GPT-4; Qwen3.6 is already sitting there), plus a rollout
loop. That would make `parse_answer_status` reachable and the metric properly grounded. It is a
separate subsystem — roughly 10 teacher calls per query per eval on top of what we already pay — so I
did not build it speculatively.

---

## 6. What changed

### New files

| File | What |
|---|---|
| `data/loaders/toolbench_prompt.py` | ToolBench's system-prompt construction, reproduced exactly, with upstream quoted alongside |
| `data/loaders/toolbench.py` | Range-streamed train prefix, complete-path rule, six-subset eval set, tool-description index |
| `eval/scorers/toolbench.py` | Path parsing, exact pre-rules, ToolEval rubric, 3-round majority vote, weighted aggregation |
| `tasks/toolbench.py` | The `TaskSpec` |
| `tests/pipeline/run_toolbench_l40s.slurm` | Weeklong launcher, dedicated quota |
| `tests/pipeline/run_toolbench_cse.slurm` | 24h launcher, CSE quota |
| `tests/test_toolbench_prompt.py` | 27 tests pinning the prompt reconstruction |
| `tests/eval/test_scorer_toolbench.py` | 41 tests pinning the metric |

### Modified

| File | What |
|---|---|
| `tasks/__init__.py` | Registered; nine tasks → ten |
| `eval/judge_client.py` | `JudgeRubric`; `score_payloads`; cache keys provably unchanged for `dialogsum` |
| `tasks/_builders.py` | `toolbench_turn` — whole path as the target |
| `data/quality_controls.py` | `complete_toolbench_path()` — this task's analogue of `valid_json_answer` |
| `data/synth_verifiers.py` | `verify_toolbench_row` — schema + declared-API + termination + budget |
| `tests/test_task_registry.py` | `EXPECTED_TASKS`; `targets["toolbench"] = "query"`; judged set is now two |
| `tests/pipeline/test_task_slurm_scripts.py` | Both launcher maps |
| `tests/eval/test_task_scoring_contract.py` | Fixture + a ToolEval judge stub |
| `tests/data/test_task_context_block.py` | Fixture (10 tasks) |
| `tests/data/test_quality_controls_per_task.py` | Row builder + path-check tests + the dedup exemption |
| `tests/data/test_synth_verifiers.py` | Verifier tests; format-bound set is now four |
| `tests/test_benchmark_loaders.py` | Converter tests + the truncated-JSON streaming tests |
| `tests/eval/test_local_judge.py` | Cache-invalidation test moved to the rubric; two new rubric tests |

### Choices worth flagging

- **`required_fields = ("text", "query")`, not `answer`.** ToolEval pass rate is reference-free and
  the test queries ship with no gold path, so requiring `answer` would reject every eval row. `query`
  is the field the scorer actually grades against, and it is on both splits. Train rows carry
  `answer` and `toolbench_turn` raises without it.
- **`needs_judge = True`** — the suite's second judged task and first judged structured-output one. A
  judge outage now stops the run instead of scoring every query unsolved, which would look exactly
  like a model that cannot use tools.
- **`judge_overlap = False`.** The overlap warmer in `eval/harness.py` pre-populates the cache with
  the *generation* scorer's `(text, gold, prediction)` triples. ToolEval judges
  `(query, answer, round)` under a different rubric, so the warmer would miss every entry and judging
  would run twice. Overlapping this task needs a rubric-aware warmer, which does not exist yet — this
  is the obvious next optimisation, since judging is the bottleneck.
- **`allow_paid_discovery = False`.** No other corpus carries ToolBench's API namespace, and with
  ~180,000 rows in reserve there is nothing discovery could add that rung 1 cannot.

---

## 7. Measured, live

From an end-to-end load on the login node (2026-08-24):

```
load (400 train / full eval)             6.3 s
raw -> complete paths                    24.6% keep rate
eval set                                 765 rows
  G1_instruction 163  G1_category 153  G1_tool 158
  G2_instruction 106  G2_category 124  G3_instruction 61
frozen eval set (eval_cap 1000)          765, subset proportions exact
train/serve prompt parity                50/50 identical
quality control                          399/400 kept (1 length outlier)
synth verifier on real gold              395/400 accepted
GOLD REPLAY  pass_rate                   1.0000     <- the harness does not cap the score
             format_valid                1.0000
             judged_rows                 1.0000
             undeclared_api_rate         0.0000
PROSE BASELINE (ignores the format)      0.0000, format_valid 0.0000,
                                         all 20 rows -> unparseable_path
```

Cold-start prefix reads, cache cleared between them — this is the number that justifies the
streaming read:

```
ask   400 complete paths -> read  21.8 MB   (1.09% of the file)    6.4 s
ask 5,000 complete paths -> read 272.7 MB  (13.62% of the file)   13.7 s
```

So a cold start at `initial_train_cap=5000` reads ~273 MB instead of 2 GB, and each mining re-read
extends by the difference rather than starting over.

Token budgets, measured on the real splits with a Qwen2.5 tokenizer:

```
eval prompt    median 1,248   p95 2,838   p99 3,973   max 9,155
train target   median   652   p95 1,127   p99 1,410   max 2,445
```

`max_seq_length=8192` with `max_new_tokens=1536` fits **99.6%** of eval prompts and **99.3%** of
train targets. A 1024 reserve would truncate **8%** of *targets*, which is worse than truncating a
prompt — a clipped target teaches the model to stop mid-path, and the eval scores exactly that as
`no_finish_call`. 16384 buys the last 0.4% of prompts (3 rows of 765) for double the KV cache on
every row, which is not worth it.

### Two bugs I hit and fixed during the smoke

**The mining contract.** My first loader asked for `max_train × 3` raw rows and returned whatever
survived — so `load(max_train=400)` returned **274**. A short return is exactly how
`_reread_known_sources` detects that a split is exhausted, so mining would have marked ToolBench
exhausted on its **first** call and dropped straight to paid discovery. That is B303's shape, on a
corpus with 180,000 rows in reserve. `load_complete_paths` now targets the **kept** count and derives
the next prefix size from the keep rate the corpus just exhibited. Verified: 100 → 100, 400 → 400,
1200 → 1200, and cold at 5,000 → 6,191 before the cap trims it.

**Token reserve.** Set from the eval-file measurements before I had built whole-path targets, so it
was sized for a single turn (median 135 tokens) rather than a whole path (median 652). Caught by
re-measuring the real loaded rows.

### One known false positive

`verify_toolbench_row` rejects **4 of 274 (1.5%)** real gold paths, every one for a missing
*required* argument — e.g. `body_fat_percentage_for_fitness_calculator` called without
`hip`/`neck`/`waist`. That is looseness in ToolBench, not a bug here: RapidAPI's schema declares
those required and the recorded trace omitted them. I kept the check, because the verifier gates
*generated* rows where the asymmetry favours strictness, and because `verify_function_call_row`
applies the same rule to xlam. Recorded so a future reader can tell a known 1.5% from a regression.

---

## 8. Running it

```bash
sbatch tests/pipeline/run_toolbench_l40s.slurm   # 7 days, dedicated quota
sbatch tests/pipeline/run_toolbench_cse.slurm    # 24 h, CSE quota, requeues
```

Two live network dependencies beyond the Hub: `raw.githubusercontent.com` for StableToolBench's
solvable queries, and the Hub for the tool-environment archive. Both are hit at `eval_setup` and the
run refuses to start on an empty held-out set rather than proceeding.

**Budget expectation.** This is the suite's most expensive eval by a wide margin: 765 generations of
up to 1,536 tokens from prompts averaging ~1,250, then ~2,295 judge calls (765 × 3 rounds). The judge
cache is keyed by `(query, answer, round)` and lives in `$SLM_RUN_DIR/artifacts`, so it survives a
requeue and an unchanged prediction is never re-judged. Consider `SLM_EVAL_JUDGE_OVERLAP_CHUNK` once
a rubric-aware warmer exists.

---

## 9. Unrelated: one heavy test is red on your working tree

`tests/training/test_train_eval_prefix_alignment.py::test_non_qwen_models_are_skipped` fails. It is
not mine. Your uncommitted change to `training/lora_trainer.py` (comment dated 2026-08-24)
deliberately removed the Qwen-only gate on the prefix-alignment check — "EVERY model is checked, not
just Qwen. This was Qwen-gated while the pool was Qwen-only" — and that test asserts the old
behaviour, that a non-Qwen model is skipped. The test is now stale rather than the code being wrong,
but it is your call which side to change, so I left both alone.

Everything else is green: **1,572 passing** (`SLM_TEST_HEAVY=1`: 1,731 passing, that one failure).

---

## 10. What I would do next

1. **Run it once and look at `undeclared_api_rate` and `format_valid` before the pass rate.** If
   fabrication is high, the pass rate is measuring fluency and §5 is the story, not the score.
2. **A rubric-aware judge overlap warmer.** Judging is the bottleneck and `judge_overlap` is off
   purely because the existing warmer is generation-specific. This is cheap and buys real wall time.
3. **Decide whether we want the StableToolBench simulator.** It is the difference between a grounded
   pass rate and a plausibility score. It is also the largest single piece of work on the horizon for
   this task, and worth deciding deliberately rather than drifting into.
4. **Do not put 77.55% in a comparison table.** If ToolBench appears in a writeup, our number belongs
   next to the *official* ToolBench and StableToolBench baselines (§2.1), with the one-pass caveat
   stated. The paper's figures are not a usable reference point.
