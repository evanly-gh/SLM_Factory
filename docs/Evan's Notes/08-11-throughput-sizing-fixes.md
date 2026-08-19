# Five fixes: output cap, no-shrink curriculum, `<think>` leak, judge overlap, right-sized context

*2026-08-11 — implementation notes*

Follows the run post-mortem (`08-10-dialogsum-postmortem.md`). Five of
the open items are now implemented. The threshold floor (the single biggest cause of
non-convergence) is deliberately **still open** — it is a policy decision, not a bug.

Test suite: **960 passing**, plus the two long-standing failures documented in the post-mortem
(`APPS introductory` and `primary_strategy`), neither touched by this work.

---

## 1. Orchestrator output cap: 4096 → 20000

`agent/nodes/iterate.py`. One line, plus a comment explaining why the old number was wrong for a
non-obvious reason.

```python
_ITERATE_MAX_TOKENS = int(os.environ.get("SLM_ITERATE_MAX_TOKENS", "20000"))
```

**Why the old value failed.** The decision JSON is ~520 tokens. But `max_tokens` bounds
**thinking plus answer**, and a measured replay of a real iterate prompt spent **1,923 tokens
thinking** before writing 520 of JSON. Telling the model to be brief cannot help: it neither
sees nor budgets its own thinking. 38 of 69 calls in run 38303490 were truncated mid-object.

**Why this costs nothing.** Output is billed on tokens *actually generated*, never on the cap. A
call that produces 2,443 tokens costs the same at a 20,000 cap as at 4,096. It should be
strictly cheaper overall, because ~$2.20 of the run's $4.87 went to reask calls that a larger cap
would have avoided.

**Why not remove the cap.** `max_tokens` is a required parameter on the Anthropic Messages API —
omitting it is a 400. The ceiling for `claude-sonnet-5` is **128,000** (verified: 100,000
accepted, 200,000 returns `max_tokens: 200000 > 128000`). 20,000 leaves 6× headroom over the
worst observed call while staying far below the point where the SDK wants streaming.

Thinking is left ON. It is worth its cost on this call, unlike the pick-from-a-list calls in
escalate/model-selection.

---

## 2. The curriculum never shrinks (B257)

**What made it drop rows.** Two independent places, both fixed.

`agent/data_sizing.py::resize_curriculum_for_tier` recomputes the target per tier from that
tier's own baseline and parameter count:

```
target = clamp(5000 × (0.5 + novelty) × size_factor, 5000, 25000)
```

A *bigger* model legitimately computes a *smaller* number — Qwen3-4B asked for 1961 (floored to
5000) right after Qwen3.5-0.8B had asked for 5754. Now the target ratchets:

```python
ratcheted = target
if previous and previous > target:
    ratcheted = int(previous)
```

`agent/nodes/curate.py` was the second leak. `allocate()` stops once `len(selected_rows) >=
target_rows`, and the orchestrator can also write `target_rows` straight into a rebuild plan,
bypassing the sizing function entirely. So the effective target is now floored at the size of
the dataset already on disk:

```python
if previous_rows and len(previous_rows) > target_rows:
    target_rows = len(previous_rows)
    plan["target_rows"] = target_rows
```

**Net effect:** rows leave the curriculum only through quality control or the eval firewall,
never to satisfy a lower cap. `SLM_CURRICULUM_SIZE` still pins the size explicitly when you want
that. In the affected run this preserves the 754 rows that were discarded at the tier-2→3
boundary.

---

## 3. The `<think>` leak — yes, it was a real problem

**In plain terms:** the small model sometimes wraps its answer in reasoning tags, like

```
<think> </think> Shelly is volunteering at the food shelter.
```

That is a **correct summary** with markup stuck to the front. The pipeline is supposed to peel
those tags off before showing the answer to the judge, because the judge is asked "how good is
this summary?" and will mark down anything that looks like malformed output.

The peeling code only recognised `<reasoning>...</reasoning>`, which is the tag **our own
training targets** use. It did not recognise `<think>`, which is the tag **Qwen chat templates**
actually emit. So every Qwen model's tags sailed straight through to the judge.

Fixed in `eval/scorers/generation.py` by matching both tag families, and by matching the closing
tag independently of the opening one, because a confused model emits mismatched pairs like
`<think> </tool_call>`:

```python
_REASONING_TAGS = r"(?:reasoning|think)"
_REASONING_BLOCK_RE = re.compile(
    rf"\s*<\s*{_REASONING_TAGS}\s*>.*?<\s*/\s*(?:{_REASONING_TAGS}|tool_call)\s*>\s*", ...
)
```

Verified on the actual strings from the run:

| Raw model output | Now judged as |
|---|---|
| `<think> </tool_call> Daina needs to put on her makeup…` | `Daina needs to put on her makeup…` |
| `<think> </think> Shelly is volunteering at the food shelter.…` | `Shelly is volunteering at the food shelter.…` |
| `<reasoning>step one</reasoning>\n\nThe answer is 42.` | `The answer is 42.` |
| `<think> </tool_call>` (no answer at all) | unchanged — deliberately |

That last row is intentional: a model that emitted *only* reasoning has no answer, and handing
the judge an empty string would score 0 and hide the real failure.

**Important scope note.** This fixes the *scoring* half. It does **not** explain why the tier-3
fine-tune emitted those tags in the first place. Two hypotheses were tested and eliminated:

- *Prompt-template mismatch* — **ruled out.** Training and GGUF-eval renders are byte-identical
  for `Qwen3-4B-Instruct-2507` (both end `...<|im_start|>assistant\n`) and for `Qwen3-0.6B`
  (both end `<think>\n\n</think>\n\n`).
- *Training-target contamination* — **ruled out.** For `generation` the assistant target is the
  bare answer; the `<reasoning>` wrapper only appears when `cot_reasoning` exists, and CoT is
  disabled for this task.

Remaining candidates are GGUF special-token conversion of the merged 4B, or LoRA degenerating
that model. The decisive test is to generate from the base 4B GGUF and the merged 4B GGUF on
identical prompts and diff the raw token streams (~10 min on one GPU). **Still open.**

---

## 4. Judging now overlaps generation (B258)

**The problem.** Generation runs on the pipeline GPU, judging on the synthesis GPU, but they ran
strictly one after the other: all 800 predictions produced, *then* all 800 judged. Measured
overlap across the two devices was **0.0%**, while the judge alone accounted for **22% of wall
time**.

**How it is implemented.** `eval/harness.py::_infer_overlapping_judge`. Generation is chunked
(default 100 rows, `SLM_EVAL_JUDGE_OVERLAP_CHUNK`; `0` disables). After each chunk finishes on
the pipeline GPU, a single background thread asks the judge to score that chunk while the *next*
chunk generates:

```python
with ThreadPoolExecutor(max_workers=1) as pool:
    pending = []
    for start in range(0, len(prompts), chunk_size):
        chunk_raw = infer(prompts[start:start + chunk_size])
        raw_outputs.extend(chunk_raw)
        pending.append(pool.submit(warm, start, chunk_raw))
    for future in pending:
        future.result()
```

**Why this is safe.** It does not change how scoring works. The background call is
`LocalJudgeClient.score_many`, whose only side effect is populating the client's in-memory cache
and the on-disk `local-judge-cache.jsonl`. The scorer's own `score()` call afterwards is
completely unchanged and simply finds those rows already cached, so **ordering, scores and
failure records are bit-identical to the sequential path**. That is also why warm failures are
swallowed — anything genuinely broken resurfaces in `score()`, which raises there.

The warm builds its triples exactly the way `generation.score` does
(`text`, `answer`/`label`, `split_reasoning(raw)[1]`); a divergence would miss the cache rather
than corrupt anything, but it would silently undo the speedup, so a test pins it.

Two design choices worth recording:

- **One warm worker, not many.** The point is to overlap judging with the *next generation
  chunk*, not to run several judge batches concurrently against a single vLLM server — the judge
  client already parallelises within a batch.
- **Chunking is free here.** Inside the eval worker `isolation_enabled()` is False
  (`SLM_CUDA_WORKER=1`), so chunked calls reuse the cached model instead of re-spawning a
  subprocess per chunk. Verified before implementing.

Only `task_type == "generation"` takes this path; every other task scores without a judge.

---

## 5. Eval batch sizes raised

`training/slm_helpers.py`:

| | before | after |
|---|---|---|
| short-output tasks (classification) | 16 | **32** |
| long-output tasks (generation, NER, math, code) | 4 | **16** |

The old numbers were chosen when `max_seq_length` was assumed to be the real sequence length.
Measured rows are p50=228 / p99=601 tokens, so the per-sequence KV footprint is a fraction of
what those numbers assumed. A CUDA OOM already halves the active batch and retries in place, so
the downside of aiming high is one retry, not a failed eval. `SLM_EVAL_BATCH_SIZE` still
overrides.

---

## 6. Context right-sized, per task

**What the oversize actually consumed.** Context is *allocated*, not measured — KV cache and
position buffers are sized from the configured number whatever the rows contain. At 4096 against
a measured max of 1206 tokens, ~70% of that allocation was never touched, and
`truncated=0/5754` confirms the cap never bound. That memory is exactly what a larger eval batch
needs, so the two changes compound.

**This does not touch the orchestrator.** `SLM_MAX_SEQ_LENGTH` governs the *small model* being
trained and evaluated. Claude's context is a separate budget that nothing here affects — the
orchestrator keeps every token of context it has today.

Replaced the single global 4096 with a per-task table (`training/slm_helpers.py`):

| task_type | ceiling | rationale |
|---|---|---|
| classification | 1024 | labels are a few tokens; prompts are short |
| generation, math_reasoning, NER, function_call, diff | 2048 | ~3.4× the observed max of 601 |
| code_generation | 4096 | APPS prompts plus a 1024-token completion genuinely need it |
| unknown / unset | 4096 | falls back to the safe maximum rather than a guess |

These are ceilings with real headroom, not tight fits: an over-length row **raises** rather than
truncating, so they are deliberately generous.

Training and eval read the same table — `training/lora_trainer.py::_configured_max_seq_length`
now delegates to `training.slm_helpers.task_max_seq_length` — so a row can never fit one side
and be truncated on the other. An explicit `SLM_MAX_SEQ_LENGTH` still overrides everything.

---

## 7. Also fixed since the post-mortem

- **B256 — system prompt logged 69 times.** `_iterate_prompt_logged` was set but not declared on
  `AgentState`; LangGraph drops undeclared keys on the state merge, so the guard reset every
  turn. Declared it, and added a test that scans `iterate.py` for every private key it writes
  and fails if any is missing from the schema.
- **B255 — per-tier graphics.** `logs/graphics/<run_id>/` now holds a combined view plus one
  `tier<N>_<selector>/` subdirectory per model, each with its own `hypotheses.md` and four PNGs.

---

## Still open

1. **Threshold floor** — the goal (0.80) is set above the score the reference Qwen3.6-35B itself
   achieved (0.7327), because the floor overrides the measured target. Until this is decided, no
   on-device model can pass this task and every other measurement is hard to interpret. Policy
   decision required.
2. **Origin of the `<think>` tags** (§3) — scoring is fixed; the emission is not explained.
3. **GPU overlap beyond eval** — curate still serializes against train. Prefetching the next
   rebuild during training is the remaining large wall-clock win.
4. **86% of GGUF builds discarded** — quantize only after eval confirms an improvement.
