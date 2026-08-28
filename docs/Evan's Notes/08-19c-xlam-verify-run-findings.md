# xlam_bfcl single-model verification runs — what broke, and what the data actually looks like

**Date:** 2026-08-19
**Runs:** 38656655 (died 27 min), 38658213 (stopped at 5 iterations), 38661753 (stopped at ~4 rebuilds)
**Purpose of the runs:** prove the two data interventions actually add rows before a full-length run
pays to find out they do not. All three were stopped deliberately. Between them they found five
production bugs, four of which were invisible to a 1,377-test CPU suite because they sit on GPU-only
paths.

---

## 1. What the confusion labels mean

The pairs in the report read `gold='X' predicted='incorrect'`. For a task with a closed label space
(CLINC150, RouterBench) `gold`/`predicted` really are two classes and the pair names a genuine
confusion. `xlam_bfcl` has no label space — every row's label is the constant `function_call` — so the
left side carries the **failure category** instead and the right side is the filler `incorrect`. The
categories come from `eval/scorers/function_call.py::failure_category`, evaluated in this order (first
match wins), against the row's own `tools` schema:

| category | test | what it means | what fixes it |
|---|---|---|---|
| `unparseable_output` | `pred is None` | the model's text did not parse as a JSON array of calls at all — a stray prose preamble, a markdown fence, a truncated object | a format/template problem. More examples of the exact output shape; check the chat template and the `max_new_tokens` reserve |
| `empty_call_list` | `pred == []` | valid JSON, but no calls — the model declined | usually a prompt problem, occasionally an unanswerable row |
| `undeclared_function` | a called name is not in this row's `tools` | the model invented a tool, or reached for one from a different row | tool-selection grounding; also worth checking the row is winnable |
| `wrong_call_count` | right names, wrong number of calls | the request implied N invocations and the model emitted M — the classic miss is a multi-target request ("beta access **and** games") that needs two separate calls | examples of multi-call requests specifically |
| `wrong_function` | the called names differ from gold | picked the wrong tool from a correct list | tool-selection problem: descriptions were read but discriminated wrongly |
| `wrong_arguments` | names and count correct, arguments differ | **the residual bucket.** The right tool, called the right number of times, with at least one argument wrong: a missing required key, an invented key, a wrong type, a schema default left in place instead of a value pulled from the text | an extraction problem — the hardest of the six and the one that dominated every run |

Run 38661753's final distribution, out of 206 failures on 1,000 eval rows:

```
wrong_arguments      164   ← 80% of all failures
wrong_call_count      18
unparseable_output    16
wrong_function         7
undeclared_function    1
```

That shape is worth reading carefully. `format_valid` was 0.984, and only 16 rows failed to parse, so
the model has the output contract. It picks the right tool essentially always (7 wrong out of 1,000).
Nearly everything it gets wrong is **argument extraction** — which is exactly the category the
orchestrator kept targeting, and correctly so.

---

## 2. What a row actually contains

There are four content fields, not one. The concern that only `answer` exists is understandable if
you open the file in a viewer that collapses long values, because `tools` is by far the largest field
and pushes the rest off screen.

```
text     "Where can I find live giveaways for beta access and games?"      ← the user prompt
tools    [{"name": "live_giveaways_by_type", "description": "...",         ← the callable tools,
           "parameters": {"type": {"type": "str", "default": "game", …}}}]    with full JSON schemas
answer   "[{\"arguments\": {\"type\": \"beta\"}, \"name\": \"live_giveaways_by_type\"},
           {\"arguments\": {\"type\": \"game\"}, \"name\": \"live_giveaways_by_type\"}]"
label    "function_call"                                                   ← constant, not a class
```

Note `answer` is a **JSON string**, not a JSON array — it is serialized once more inside the row. That
is why `qc.valid_json_answer()` exists as a quality-control step: a gold answer that does not parse
trains the model to emit something the scorer marks wrong no matter what it predicts.

The `_`-prefixed fields are bookkeeping, not task content, and are stripped before anything reaches
the model: `_provenance` (`train_anchor` / `synthetic` / `mined`), `_source`, `_dataset_version`,
`_strategy_origin`, `_target_category`.

The prompt the model actually sees is assembled by `build_prompts` from `text` + `tools`; it never
sees `answer`.

---

## 3. The prompts the teacher gets

### Generating a new row

`data/curriculum.py::_synthesize_new_correct`. Five real rows, shown in full, then the instruction:

```
TASK: <orchestrator-authored summary of the benchmark>
OUTPUT CONTRACT: <the exact output contract, derived from real rows>
COMMON MISTAKES TO AVOID: wrong_arguments; unparseable_output

Here are 5 real examples of this task, in the exact output format required:

{"answer": "[{\"arguments\": {…}, \"name\": \"get_all_bodies_positions\"}]", "label": "function_call",
 "text": "For a star party in Cape Town on April 10th…", "tools": [ …full schemas… ]}

… four more …

Generate ONE new, correct example in EXACTLY this JSON schema (same keys, same value types):
<the anchor row, rendered as JSON>.
The model being trained is currently failing on: wrong_arguments. Favour examples that exercise
exactly that difficulty.

It must be a genuinely new, diverse, and CORRECT instance — not a copy or a paraphrase of the
reference, and never a wrong answer. Obey the output contract above exactly; a well-formed answer
that breaks a stated convention is graded wrong. Return only the JSON object, no preamble or fences.
```

The demonstrations are **shown, not described**, which is the whole point: the teacher scores 0.1131
span-F1 on BC5CDR zero-shot and 0.7190 with five demonstrations. A generator asked for output in a
contract it has only been told about is being tested on guessing the contract.

The TASK / OUTPUT CONTRACT / COMMON MISTAKES lines are the orchestrator-authored brief
(`agent/task_brief.py`), written at cold start from real rows. For xlam it independently derived
conventions the old hardcoded one-liner never mentioned — including that a multi-target request needs
separate call objects, and that arguments must be filled from the user text rather than left as
schema defaults.

### Verifying a generated row

Two stages, and this is where a bug was hiding.

**Stage 1 — programmatic** (`data/synth_verifiers.py::verify_function_call_row`). Free, exact, no
teacher call: parse the generated `answer`, check every call's name against the row's own `tools`, and
check every argument key against that tool's declared parameters. A row that fails here is dropped
with no LLM involved.

**Stage 2 — teacher** (`data/curriculum.py::verify_generated_answers`):

```
You are checking one training example for the task of <task description>.

Real, confirmed examples of this task:
Request: …
Correct answer: …
  (five of these)

User input / question:
<the generated text>

Context provided WITH this example (the answer must be consistent with exactly this, and
anything named here is valid by definition):
{"tools": [ …the row's full tool schemas… ]}          ← ADDED 2026-08-19, see §4.2

Proposed answer:
<the generated answer>

Does the proposed answer correctly and directly satisfy the user's request, in the context of
<task description>? Answer strictly as JSON: {"valid": true|false, "reason": "<max 15 words>"}.
Answer false if the answer is wrong, incomplete, in the wrong format for this task, or does not
address what was actually asked. Judge ONLY against the request and the context above — if a name
or field appears in the context, it is valid by definition, and you must not reject the answer for
using it or claim it was not provided.
```

Both stages fail **open**: an unparseable verdict or an endpoint error KEEPS the row. A verifier must
never be able to empty a dataset.

---

## 4. The bugs

Five bugs, each written up the same way: what was observed, the exact mechanism, why it survived a
1,377-test suite, the fix and what was rejected on the way to it, and the test that stops it coming
back. All five sat between two working components, which is why they needed a GPU to surface.

A quick map, because they interact:

| # | bug | what it silently did | blast radius |
|---|---|---|---|
| 4.1 | B313 | teacher baseline read 0.0000 | the accuracy goal for the whole run |
| 4.2 | B314 | verifier rejected valid rows for invented reasons | ~24% of every synthesis batch |
| 4.3 | B312 | `data_rebuild` never executed | 5 iterations of a 5-iteration run |
| 4.4 | B311 | probes scored 0.0; prompts lost their hints | model selection, and one run's life |
| 4.5 | B315 | mining delivered 4x, discoveries forgotten | curriculum size, corpus lifetime |

---

### 4.1 B313 — the teacher baseline read 0.0000 because the task name was passed as the generator

**Observed.** A thousand identical lines, then a baseline of zero, then an accuracy goal derived from
it:

```
endpoint baseline generation failed: 'str' object is not callable      (x1000)
  [baseline] reference ast_arg_match=0.0000 format_valid=0.0000
  [threshold] Qwen baseline 0.0000 → goal 0.8000 (floor 0.80) — FLOOR WON: the teacher
             scored below 0.80, so the goal is the floor, not the teacher's 0.0000
```

**Mechanism.** One line, in `agent/nodes/cold_start/eval_setup.py`:

```python
baseline = measure_endpoint_baseline(eval_set, task, log=print)   # before
baseline = measure_endpoint_baseline(eval_set, log=print)         # after
```

The signature is `measure_endpoint_baseline(eval_set, generate_fn=None, *, max_workers, log)`. The
task is read off the `EvalSet` — deliberately, so the two can no longer be passed inconsistently — so
there is no task parameter, and `generate_fn` is the second **positional** slot. The string
`"xlam_bfcl"` went in there. Then, for each of 1,000 rows:

```python
generate_fn(prompt, temperature=0.0, max_tokens=max_tokens)
# "xlam_bfcl"(prompt, ...) → TypeError: 'str' object is not callable
```

Every one was caught by the deliberate per-row handler, which exists so that one truncated response
cannot abort a 1,000-row baseline, and turned into `""`. A thousand empty strings score 0.0. The
scorer was working correctly and the threshold logic was working correctly; both were reasoning about
a measurement that had not happened.

**Why it survived.** Three reinforcing reasons, and the third is the interesting one.

1. It is on a GPU-only path — the baseline needs a live vLLM endpoint, so no CPU test reaches it.
2. `generate_fn` has a default of `None`, so the arity is legal. Passing two positional arguments to a
   function that accepts two positional arguments is not a static error, and the parameter carries no
   type annotation to contradict.
3. **The failure was designed to be survivable.** The `except` clause around each call is correct
   policy, and it converted a total harness failure into a plausible-looking number. That is the real
   lesson here: an error handler scoped to "one bad row" was silently applied to "every row", and the
   difference between those two situations is the entire meaning of the result.

The same refactor dropped `task` from `run_eval`, which is why several call sites made this mistake at
once (§4.4).

**What made it expensive.** The teacher's real ability was already in the same log, from a different
code path, minutes earlier: the new fitness gate measured **0.8250** five-shot on this exact eval set.
So the run had two numbers for one quantity — 0.0000 and 0.8250 — and used the wrong one to set the
target it then spent hours chasing. Worse, had the gate consulted the existing baseline instead of
measuring for itself, it would have refused synthetic data on this task for a reason that was pure
harness noise.

**Fix — three parts, because fixing only the call site leaves the trap loaded.**

*One:* the call site drops the argument.

*Two:* refuse a non-callable `generate_fn` before making a single call. This catches the whole family
rather than the one instance, which matters precisely because the arity is legal and the static scan
cannot see it:

```python
if not callable(generate_fn):
    raise TypeError(
        f"measure_endpoint_baseline needs a callable generate_fn, got "
        f"{type(generate_fn).__name__} ({generate_fn!r}). The task is read from the eval set, so "
        f"there is no task parameter to pass positionally."
    )
```

*Three:* count the failures and refuse to report a score when too many fail. This is the general
guard, and it would have caught B313 even without the `callable` check — along with an endpoint that
dies halfway, a model that refuses every prompt, or any future variant of "the harness broke":

```python
MAX_GENERATION_FAILURE_RATE = 0.25   # not 0: a few refusals or truncations are normal

if prompts and len(failures) / len(prompts) > MAX_GENERATION_FAILURE_RATE:
    raise BaselineGenerationError(
        f"generation failed on {len(failures)} of {len(prompts)} eval row(s) "
        f"({len(failures) / len(prompts):.0%}), so the resulting score would describe the "
        f"harness rather than the model. Most common errors: ..."
    )
```

The threshold is 25% rather than 0% on purpose. Zero would make the guard fire on the normal case the
per-row handler exists for, and a guard that fires on noise gets switched off. Well under half,
because a teacher that cannot answer most of the eval set is not being measured either.

**Rejected:** adding a type annotation to `generate_fn` and relying on a type checker. The repo has no
type-checking gate in CI, so the annotation would document the contract without enforcing it, and the
next person to make this mistake would still get 0.0000 rather than a stack trace.

**Pinned by** three tests in `tests/test_qwen_baseline_goal.py`:
`test_the_task_name_in_the_generate_fn_slot_is_rejected_immediately` (the literal bug),
`test_a_baseline_whose_every_row_failed_is_refused_rather_than_scored_zero` (the general guard), and
`test_measure_endpoint_baseline_survives_a_failing_row` (the tolerance that must *not* regress — its
stub now fails on exactly one row of twenty, because on the old single-row fixture "one bad row" was
100% and the new guard correctly fired on it).

---

### 4.2 B314 — the verifier was asked to check calls against a tools list it was never shown

**Observed.** Rejection reasons that referred to something absent from the prompt:

```
REJECTED 'Calculate the distance between point (1,2,3) and (4,5,6).'
  — teacher: Tool name 'calculate_distance' is not present in the provided tools list.
REJECTED 'Retrieve the current price of Bitcoin in USD…'
  — teacher: Tool names and arguments are invented and not from the provided tools list.
REJECTED 'Get the player stats for player ID 237…'
  — teacher: Invented argument key 'is_id' not present in tool schemas.
```

**79 of 330 rows (24%) on the first synthesis round; 21 of 108 (19%) on the second.**

**Mechanism.** The stage-2 prompt was assembled from four things: the task description, five reference
(request, answer) pairs, the generated request, and the generated answer. The row's `tools` field —
which *defines* what a correct call is for that row — was never rendered. So the teacher was asked
"does this function call satisfy this request?" without being shown the functions.

It answered anyway. That is the part worth sitting with: the question was unanswerable as posed, and
the model did not say so. It reconstructed a plausible tools list from the five reference rows and the
request, then judged against that. Every reason above is internally coherent and about a prompt that
does not exist.

**Why every one of those rejections was provably wrong.** Not "probably wrong" — provably. Stage 1 is
`data/synth_verifiers.py::verify_function_call_row`, which parses the generated answer and checks every
call's name against the row's own `tools`, and every argument key against that tool's declared
parameters. It runs *first*, and a row that fails it is dropped without any teacher call. So a row
reaching stage 2 has already been proven to use only tools from its own list, with only declared
argument keys. A row cannot both pass stage 1 and use a tool absent from its tools list. Every stage-2
rejection citing that discarded a valid row.

The `is_id` rejection is the same error in subtler dress: `is_id` is a real xLAM convention, visible in
the schema the teacher was not shown, and it was overruled by the model's prior about what an ID field
"should" be called. That is B267/B269 again — on RouterBench the same mechanism rejected a grade-school
math problem from the `local` class because "the utterance is a math problem, not a local query", when
`local` means "route to the local model".

**Why it survived.** The prompt was *correct for the tasks it was written against*. For GSM8K or
DialogSum, request plus answer really is the whole of the row, and there is no per-row context to
omit. The bug appears only for tasks whose correctness is defined by per-row context — `xlam_bfcl`'s
`tools`, `calendar_json`'s reference instant — and it presents as a plausible rejection rate rather
than an error. 24% rejected reads like a working filter.

**Fix.** Render the row's own fields into the prompt, and tell the model they are authoritative:

```python
_NON_CONTEXT_FIELDS = frozenset({"text", "answer", "response", "label", "prompt"})

context = {
    key: value for key, value in row.items()
    if not key.startswith("_") and key not in _NON_CONTEXT_FIELDS
    and value not in (None, "", [], {})
}
```

rendered as:

```
Context provided WITH this example (the answer must be consistent with exactly this, and
anything named here is valid by definition):
{"tools": [ …the row's full tool schemas… ]}
```

and closing with `Judge ONLY against the request and the context above — if a name or field appears in
the context, it is valid by definition, and you must not reject the answer for using it or claim it was
not provided.`

Two details that are doing real work:

*Built by exclusion, not by enumeration.* The obvious design is a `context_fields` tuple on each
`TaskSpec` naming which fields to show. Rejected: silent omission is the exact failure mode being
fixed, and a per-task list reintroduces it the moment a task gains a field and nobody updates the
list. Excluding the question and the answer means anything else on the row is shown automatically, and
the failure mode inverts to showing something harmless.

*The "valid by definition" clause.* Adding the schema alone would fix most of these, but the model had
already demonstrated it will overrule the row from its own prior. The instruction removes the ambiguity
about which authority wins. It is scoped to the context block specifically, not a general instruction
to be lenient — the model must still reject a call that misuses a tool that *is* in the list.

Applied to `verify_generated_labels` on the same grounds.

**Consequence for a conclusion this run produced.** The orchestrator wrote, correctly reasoning from
what it could see:

> confirming teacher-generated argument rows carry label/format noise the model overfits to rather
> than genuine signal

It reached that from watching two synthesis rounds make the dominant confusion worse (147 → 167) while
a mining round improved it. But those rounds ran through a filter discarding a quarter of its input for
invented reasons, and — worse for the conclusion — the *survivors* were selected by a verifier
reasoning without the schemas. The selection was close to arbitrary with respect to the thing being
judged. The conclusion may well hold; it has not yet been tested under a working filter.

---

### 4.3 B312 — the plan validator rejected its own output, so `data_rebuild` never ran

**Observed.** Five iterations, five orchestrator decisions asking for `data_rebuild`, and a curriculum
that never moved:

```
LLM call failed (ValueError("data_rebuild has unknown field(s) ['hypothesis', 'task']; allowed:
  ['pattern_hint', 'rows', 'schema_version', 'strategy', 'target_categories']"));
  using test-agent suggestion: hyperparameter
  → TRAIN (hyperparameter intervention, dataset held fixed)
```

Score 0.789 → 0.793 → 0.728 → 0.769 → 0.690.

**Mechanism.** `normalize_data_rebuild_plan` *returns* a dict containing `hypothesis` and `task`:

```python
return {
    "schema_version": ..., "strategy": ..., "rows": ...,
    "target_categories": ..., "pattern_hint": ...,
    "hypothesis": ...,   # ← in the output
    "task": task,        # ← in the output
}
```

but `_PLAN_FIELDS`, the set it validates input against, listed only the first five. The function could
not accept its own output. Two callers feed exactly that back:

- `curate_node` re-normalizes `state["data_rebuild_plan"]`, which `iterate` normalized a turn earlier.
  This is not redundant — mining availability is re-derived at execution time against the
  decontaminated pool — so the second call is load-bearing, and it could never succeed.
- the orchestrator, shown a schema and a worked example, nested `hypothesis` inside the plan object
  instead of leaving it at the top level. A formatting slip, and an entirely predictable one.

**Why it survived, and why it was so hard to see from the log.** The validator is strict on purpose:
an unknown field usually means the orchestrator invented a knob, and silently ignoring it would let it
believe it had asked for something. But `iterate` wraps the decision in a broad `except` and falls back
to the test-agent suggestion, which was `hyperparameter`. So the destroyed intent was replaced with a
*valid, plausible alternative*, and the log line reads `using test-agent suggestion: hyperparameter` —
which is exactly what it would say if the orchestrator had legitimately wanted hyperparameters. The
run looked like five deliberate tuning decisions. Nothing in the report distinguished "chose
hyperparameters" from "asked for data and was overruled by a schema bug".

None of the 1,377 tests caught it because each tested one direction: the validator accepts a plan the
orchestrator sends, and the fallback produces a plan the validator accepts. Nothing composed them.

**Fix — three parts.**

*One:* accept the two fields, and ignore their values in favour of the authoritative arguments —
except that a plan-carried `hypothesis` survives when no argument is given, so re-normalizing cannot
blank a hypothesis the orchestrator wrote:

```python
_PLAN_FIELDS = frozenset({
    "schema_version", "strategy", "rows", "target_categories", "pattern_hint",
    "hypothesis", "task",     # accepted so the function can accept its own output
})
```

A genuinely unknown field still raises — the strictness that motivated the check is intact.

*Two:* say it out loud when a data-rebuild decision is downgraded. The bug's cost was concealment, not
the ValueError:

```
✗ ORCHESTRATOR PLAN REJECTED: it asked for a data_rebuild and the plan was refused (...).
  This iteration will NOT rebuild data.
```

*Three:* count them, and stop at two. A rejection this systematic is a schema disagreement between the
prompt and the validator, not a bad sample — if the plan shape is wrong once it is wrong every time —
so `run_health.record_rejected_data_plan` raises on the repeat rather than letting the run spend hours
proving it.

**Rejected:** relaxing the validator to ignore all unknown fields. That trades a loud failure for a
silent one — the orchestrator would then be able to ask for a knob that does not exist and read the
resulting no-op as evidence. Accepting exactly the two fields the function itself emits keeps the
strictness where it earns its keep.

**Also fixed alongside**, found by review rather than by the run: `curate_node` re-normalized the plan
passing `mining_available` but not `synthesis_allowed`, so at the execution gate that flag defaulted to
`True`. Since mining availability is re-derived there and can flip to `False`, the validator would
rewrite `mine_new_real` into `surgical_synthesis` **behind a teacher that had failed its fitness gate**
— spending the teacher budget on precisely the output the gate exists to refuse. Both flags are now
passed, which also makes `DataInterventionUnavailable` reachable at that gate; `curate` catches it,
logs that the curriculum is unchanged, and lets the empty-rebuild counter handle the repeat, because
killing a run over one bad turn is worse than losing one turn.

**Pinned by** `test_normalizing_an_already_normalized_plan_returns_the_same_plan` (parametrized over
both strategies — this is *the* test that would have caught it),
`test_the_shape_the_orchestrator_actually_sent_is_accepted`,
`test_a_genuinely_unknown_field_still_raises_after_the_widening`, and three tests in
`tests/test_run_health.py` covering the counter and its threshold.

---

### 4.4 B311 — six call sites still spoke the deleted channel vocabulary

**Observed.** Run 38656655 died 27 minutes in — past the loader, the task brief and the fitness gate,
i.e. at the most expensive possible moment:

```
TypeError: run_eval() got an unexpected keyword argument 'task'
ImportError: cannot import name 'TASK_METRIC_NAMES' from 'eval.harness'
```

**Mechanism.** The 2026-08-18 refactor removed the abstract `task_type` channel and made `run_eval`
read the task from its `EvalSet`. Six callers still passed `task=` or `task_type=`. Two were worse than
a crash:

- `interpolation._probe_model` called `slm_train(..., task_type=state["task_type"])`. Wrong keyword
  **and** a state field that no longer exists — and it sat inside a `try/except` that reports a failed
  probe as `f1=0.0`. So every interpolation probe scored zero, the scaling curve was fitted through
  three zeros, and the log said `⚠ probe FAILED for <model> ('task_type') → f1=0.0` in a format that
  reads like a model limitation rather than a `KeyError`.
- `orchestrator_choice` read `state.get("task_type", "classification")` and passed it to
  `_benchmark_hint`, which looks its argument up in the task registry. `"classification"` is not a
  task, so the lookup missed on **every run since the refactor** and the model-selection prompt
  silently received the generic fallback hint instead of the family-specific one. No error, no log
  line, just a quietly worse prompt.

And `tests/pipeline/run.py` imported `TASK_METRIC_NAMES`, a constant deleted with the channels — 26
lines above the final report, so the crash also took the entire report with it (§6).

**Why it survived.** Every instance is on a GPU-only path, and the one that was not —
`test_quantized_probe_scores_exact_deployment_artifact` — passed because its state fixture still
supplied `task_type` and no `task`. The fixture provided a key production state does not have, so the
test was exercising a world that no longer existed. That is the most useful thing in this section: a
stale fixture does not merely fail to catch a bug, it actively certifies the broken code.

**Fix.** All six call sites corrected, and the fixture given a real registry task name. But the durable
fix is in `scripts/check_unresolved_names.py`, which now checks **across module boundaries**:

1. every `from <project module> import <name>` names something that module actually defines;
2. every keyword argument passed to a function **imported from another project module** is one that
   function accepts.

Previously it only checked names within a file and signatures of same-file functions. The index is
built by parsing, never importing — importing `training.lora_trainer` to ask what it exports would load
torch, and a lint that needs a GPU is a lint nobody runs. Lazy imports inside functions are tracked at
any depth, because they are the norm in this codebase (they keep torch off CPU-only paths) and three of
the four faults were behind one.

Run against the tree it immediately found two more instances nobody had noticed:
`interpolation.py:138` and `downward_probe.py:116`. Both were live GPU paths.

**The limit, stated plainly because it matters:** this scan cannot catch B313. A string in the
`generate_fn` slot is a *positional* argument of legal arity, and the parameter has no type
annotation, so nothing about the call is statically wrong. Checks 3 and 4 also skip anything whose
definition they cannot see plainly — a re-export, a conditional definition, a decorated function, a
`**kwargs` signature. A clean run is not proof of correctness, only the absence of the mistakes it can
prove. That is exactly why B313 needed runtime guards instead.

---

### 4.5 B315 — mining delivered four times what was asked, and forgot every source it discovered

**Observed.** The orchestrator's `rows` field had no effect:

```
DATA REBUILD: mine_new_real — asking for 600 new row(s)
  [mine] re-read Salesforce/xlam-function-calling-60k: asked 7400, got 7400 row(s)
  CURRICULUM: 5288 → 7627 row(s) (+2399 added this rebuild)

DATA REBUILD: mine_new_real — asking for 800 new row(s)
  CURRICULUM: 7627 → 10701 row(s) (+3161 added this rebuild)
```

The curriculum more than doubled in three rebuilds. The orchestrator noticed and adapted — it raised
its request from 600 to 800 while receiving thousands — which is a good sign about the orchestrator and
a bad sign about the knob.

**Mechanism, part one: the over-fetch.** Rung 1 re-reads the task's own corpus. Every loader takes a
head slice, so asking for a larger slice returns a superset and everything past `consumed` is novel by
construction. The code asked for four times the request:

```python
ask = consumed + max(want, 1) * 4      # before
```

and then returned **the entire slice**, not the tail:

```python
rows.extend(more_train)                # before: all 7,400 rows, including the 5,288 already held
```

so downstream deduplication had to discover the overlap. The 4x was hedging against deduplication
losses that do not exist on this path — novel by construction means there is nothing to lose — and
because nothing trimmed the surplus, the hedge became the delivery.

**The fix that does not work, and was tried first.** The obvious repair is to keep the over-fetch and
truncate the result to `want`. This is wrong, and subtly so: `consumed` records the slice depth
**read**, not the rows **kept**. Truncating advances the pointer past rows that were never used, so
they are not deferred to the next rebuild — they are permanently skipped. I wrote that version,
including a log line claiming "the remainder stays available for the next rebuild", which was false.
The correct repair is to ask for the amount intended to be kept, so the pointer and the rows agree:

```python
take = min(max(want, 1), remaining_budget[0])   # after
ask = consumed + take
novel = more_train[consumed:]                   # only the tail past the high-water mark
rows.extend(novel)
```

**Mechanism, part two: discoveries were forgotten.** Rung 2 pays Exa to find a dataset the run has
never used. `_discover_new_source` returned its rows and registered the dataset **nowhere**. So a
corpus found by paid web research was drained of whatever it happened to return in one pass and then
did not exist as far as the run was concerned: a later rebuild could not read more of it without paying
to rediscover it, and rung 1 — the free rung — would never visit it. That makes discovery worth doing
at most once, when it should make the run permanently richer.

This never fired in these runs (rung 1 always had rows, which is correct behaviour) so it was found by
reading rather than by observation.

**Fix.**

*A per-rebuild ceiling*, applied across all sources together rather than per source, or three sources
would jointly deliver three times the cap:

```python
MAX_MINED_ROWS_PER_REBUILD = int(os.environ.get("SLM_MAX_MINED_ROWS_PER_REBUILD", "1000"))
```

*Discovered sources registered*, with `consumed` set to what was actually **taken** rather than what
the provider returned — the same distinction that made truncation wrong above:

```python
record.update({
    "consumed": int(record.get("consumed", 0) or 0) + taken,
    "url": ..., "discovered": True,
})
progress[dataset_id] = record
```

so the next `mine_new_real` re-reads it from rung 1 at the right offset, for free. A discovery that
names no dataset id logs that it cannot be recorded rather than writing a blank key, which rung 1 would
otherwise visit forever without finding rows.

**Why cap at all, when real gold rows are the best data available?** Two reasons, and the second is
the load-bearing one.

Growth has to stay legible. A rebuild that adds a few hundred rows is an experiment whose effect can
be read off the next eval. One that adds three thousand changes the curriculum size, the training time
and the class balance simultaneously, and the score movement afterwards cannot be attributed to any of
them — which defeats the purpose of choosing an intervention.

And a finite corpus has to last. xLAM holds ~60,000 rows. At 3,000 a rebuild it is exhausted in twenty
iterations, after which the ladder retires rung 1 and falls through to synthesis with most of the
corpus never read — i.e. the run starts generating data while real data sits unused, which is the
worst available trade.

**Verified against the live loader**, not just in stubs: three consecutive rebuilds asking 600 each
returned exactly 600 novel rows and advanced the position 5,000 → 5,600 → 6,200 → 6,800; a rebuild
asking 5,000 returned 1,000.

**Pinned by** six tests in `tests/nodes/test_mining_ladder.py`:
`..._returns_only_the_novel_tail`, `..._never_exceeds_the_per_rebuild_ceiling`,
`..._resumes_where_this_one_stopped`, `..._is_recorded_so_a_later_rebuild_can_read_more_of_it`,
`..._defers_the_remainder_rather_than_dropping_it`, and
`..._names_no_dataset_says_so_instead_of_recording_a_blank_key`.

---

### 4.6 What these five have in common

Worth naming, because it suggests where to look next.

**Every one sat between two correct components.** No single function was wrong in isolation. The
per-row error handler was right; the scorer was right; the threshold logic was right — and B313 lived
in the seam. The plan validator was right to be strict; the fallback was right to substitute a valid
alternative — B312 lived in the seam. The verification prompt was right for five of eight tasks.

**Four of five presented as a plausible number rather than an error.** A baseline of 0.0000, a 24%
rejection rate, five hyperparameter decisions, a probe scoring 0.0. Only B311's `TypeError` announced
itself, and that one crashed 27 minutes in. A system that reports a number for everything can report a
number for a broken measurement, and the number is what gets read.

**The suite was green throughout at 1,377 tests.** Not because the tests were bad, but because they
tested components and these were composition failures — and in one case (§4.4) because a fixture
carrying a deleted field certified the broken code. The two durable mitigations that came out of this
are therefore both about composition: the cross-module scan for static seams, and `agent/run_health.py`
for behavioural ones, watching across iterations for the patterns a single iteration cannot show.

## 5. What the runs did verify

Not everything was broken. Against the checklist these runs existed to test:

| claim | verdict |
|---|---|
| the xlam loader works | ✅ 5,000 train / 1,000 eval, exactly the spec caps, ~10s |
| the initial curriculum is a few thousand rows | ✅ 5,000 gold → 4,954 after QC |
| quality control is not over-eager | ✅ 0.9% removed (45 length outliers, 1 near-duplicate) |
| the orchestrator authors a usable task brief | ✅ derived the multi-call and no-schema-defaults conventions from real rows unaided |
| the teacher fitness gate works | ✅ measured 0.8250 five-shot, format_valid 0.9950, cleared the 0.80 gate — and would have refused synthesis on a task where it could not |
| synthesis adds rows | ✅ +334 in one rebuild, with per-row verification |
| mining rung 1 adds rows | ✅ +2,399 and +3,161 (over-delivered — §4.5 — but real, novel gold) |
| mining reads deeper on each pass | ✅ slice advanced 5,000 → 7,400 → 10,600 |
| format and content are reported separately | ✅ `ast_arg_match=0.7940 format_valid=0.9840` every iteration |
| the failure taxonomy is real | ✅ five distinct categories with counts, not one constant |
| the score improves from data | ✅ 0.777 → 0.794, and the orchestrator correctly attributed it: real rows moved the dominant confusion down while two synthesis rounds made it worse |

That last row is the most interesting result in the notes, and it is the orchestrator's own reading:

> The last mine_new_real rebuild (2399 fresh gold rows) was the first intervention to actually reduce
> this dominant confusion (167→164) after two surgical_synthesis rounds made it strictly worse
> (147→167), confirming teacher-generated argument rows carry label/format noise the model overfits to
> rather than genuine signal.

Given §4.2 — that 24% of generated rows were being discarded for false reasons, and the survivors were
selected by a verifier reasoning without the tool schemas — the synthesis rounds were running with a
badly compromised filter. That conclusion deserves a re-test now that the verifier can see what it is
judging.

---

## 6. Also landed

- **The final report always prints.** It is now a function invoked through an idempotent
  `atexit`-registered hook, registered *above* the artifact-writing and provenance-aggregation block
  rather than below it — that ordering mattered: B311's `ImportError` sat 26 lines above the report and
  took all of it. There is a `_emit_minimal_report` fallback for when the full report is unavailable or
  itself raises. Only SIGKILL is unrecoverable.
- **`label_performance.png`**, written for every run and included in `summary.png`: correct vs failed
  eval rows per label (closed label space) or per failure category (everything else), worst first,
  with counts. Counts rather than rates, deliberately — a class at 50% on four rows and one at 50% on
  four hundred are the same rate and completely different problems, and choosing a target is precisely
  about telling them apart.
- **The curriculum growth ledger** in the report: rows added per iteration, QC and firewall removals,
  verification keep rate, and a count of rebuilds that added nothing.
- **`agent/run_health.py`**, which watches across iterations and stops a run on: two consecutive
  rebuilds adding no rows; the curriculum shrinking by ≥1,000 rows or ≥25%; two consecutive total
  verification wipeouts; three mining attempts that saw candidates and accepted none; two load
  failures; two orchestrator data-plans refused. Every threshold needs a repeat or a magnitude
  variance cannot explain, because a guard that fires on noise gets switched off.

## 7. Next

The three fixed bugs all sat between a working component and another working component, and the run
had to reach a GPU to find them. Before the next full-length run:

1. Re-measure the accuracy goal now that the baseline is real. It will not be 0.80 — the teacher scores
   ~0.825 five-shot on this task — so the goal the last run converged against was too easy.
2. Re-run synthesis with the fixed verifier and compare the keep rate against 76%/81%. The
   orchestrator's conclusion that synthetic argument rows are actively harmful was formed under a
   filter that was rejecting valid rows for invented reasons.
3. Watch that rebuilds now land near their requested size, and that a discovered source is re-read at
   the right offset on the rebuild after it is found.
