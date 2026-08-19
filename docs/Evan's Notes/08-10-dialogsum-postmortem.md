# DialogSum run 38303490: post-mortem, root causes, and open issues

*2026-08-10 — handoff*

Covers the completed `slm-dialogsum-samsum-l40s-38303490` run (Aug 9, 12h27m, COMPLETED),
what went wrong, what was fixed, and what is still open. Companion: the six-benchmark
reference note (`08-06-benchmark-tasks-reference.md`).

> **Update 2026-08-11.** Sections 3, 4, 6 (context/batch) and the `<think>` *scoring* half of
> §5 are now implemented — see `08-11-throughput-sizing-fixes.md`. §1 (threshold floor),
> the *origin* of the `<think>` tags, and curate/train GPU overlap remain open.

**Headline: the run did not converge, and the single biggest reason is that the goal was set
above the score the reference model itself achieved.** Everything else compounded that.

---

## 0. Run summary

| Tier | Model | Quant | Baseline | Best fine-tuned | Δ |
|---|---|---|---|---|---|
| 0 | Qwen/Qwen3-0.6B | Q4_K_M | 0.4487 | 0.5706 | +0.1219 |
| 1 | Qwen/Qwen3-0.6B | Q8_0 | 0.4497 | 0.5850 | +0.1353 |
| 2 | Qwen/Qwen3.5-0.8B | bf16 | 0.5414 | 0.5965 | +0.0551 |
| 3 | Qwen/Qwen3-4B-Instruct-2507 | Q8_0 | 0.7157 | 0.7157 | **+0.0000** |

Goal: 0.80. Metric: `judge_mean_0_1` (local Qwen3.6-35B judge). Cost: $4.87 Claude + $0.06 Exa.
55 of 73 iterations were rolled back.

---

## 1. The goal was unreachable by construction

The reference Qwen3.6-35B scored **0.7327** on this eval set. The threshold is computed as

```python
value = min(THRESHOLD_CEILING, max(measured_value, floor_value))   # floor = 0.80
```

`agent/threshold.py::threshold_from_endpoint_baseline`. Because the measured reference score
was below the floor, the floor won and the goal became **0.80 — 0.0673 above what the 35B
teacher scored on the identical rows with the identical judge.**

The docstring says "good enough means matches the reference," and that is the right idea; the
floor silently inverts it whenever the reference lands under 0.80. The tier-3 base model's
*untrained* 0.7157 was already within 0.017 of the 35B. The run then spent 15 iterations trying
to beat a 35B model with a 4B one, on a task where the 35B itself could not reach the bar.

**Status: OPEN.** The fix is a policy decision — either drop the floor for judge-scored
generation, or make the floor `min(floor, measured)` so a weak reference cannot set an
impossible goal. Nothing else in the run matters until this is settled.

---

## 2. Confirmed bugs found and FIXED

### 2.1 B253 — orchestrator answer read from the wrong content block

`resp.content[0]` assumed the answer is the first block. With extended thinking the response is
`[ThinkingBlock, TextBlock]`; the answer is at index 1, and a ThinkingBlock has no `.text`.
Escalate/model-selection guarded with `isinstance(..., TextBlock) else ""`, so the selector
parsed as `''` and fell back to the smallest candidate — twice — at full API cost and with no
error in the log.

Fixed by `agent/llm_text.py::response_text` (scans every text block) and raising the budgets.
Seven call sites shared the assumption, including `hardware_research`, which is fatal-on-failure.
A test fails if any file reintroduces `content[0]`.

### 2.2 B254 — LoRA adapters were inert on every bf16 tier

`model.load_adapter(path)` on an Unsloth-patched model built the LoRA modules and marked them
active but never populated them from disk. Measured on real weights:

| Loader | `lora_B` populated | max norm |
|---|---|---|
| `model.load_adapter(path)` | **0 of 186** | 0.0000 |
| `PeftModel.from_pretrained(base, path)` | 186 of 186 | 0.3323 |

LoRA computes `B @ A` with `B` zero-initialised, so the adapter was exactly the identity and
eval scored the BASE model while reporting it as fine-tuned. Tier 2's five bit-identical
0.5414 scores (with identical `failures=392/800`, equal to its own zero-shot baseline) are
entirely this bug.

Quantized tiers were unaffected — the GGUF path calls `merge_for_quantization`, which merges
properly. Only bf16 tiers took the broken path, which is why it only surfaced deep into a run.

Fixed by switching to `PeftModel.from_pretrained`, verified on the real tier-5 checkpoint
(outputs changed from generic prose to the DialogSum `#Person1#` gold style). Added
`_assert_adapter_is_live`, which aborts the eval if every `lora_B` is zero.

### 2.3 B255 — graphics covered only the final tier

`escalate_node` stashes each finished tier into `escalation_history` without a `"kind"` field,
and `_iteration_records` skips anything where `kind != "model_trajectory"`. All earlier tiers
were silently dropped. The end-of-run text report does not filter on `kind`, which is why it
listed four tiers while the graphics showed one.

Fixed in `build_run_progression` (tag at source) plus per-tier subdirectories in
`generate_run_graphics`. Output is now:

```
logs/graphics/<run_id>/
  accuracy.png, difficulty.png, dataset_composition.png, summary.png, hypotheses.md   <- all tiers
  tier0_Qwen_Qwen3-0.6B__Q4_K_M/          ... baseline 0.4487
  tier1_Qwen_Qwen3-0.6B__Q8_0/            ... baseline 0.4497
  tier2_Qwen_Qwen3.5-0.8B__bf16/          ... baseline 0.5414
  tier3_Qwen_Qwen3-4B-Instruct-2507__Q8_0/... baseline 0.7157
```

### 2.4 B256 — system prompt logged on every turn

`_llm_iterate` sets `state["_iterate_prompt_logged"] = True`, but the key was not declared on
`AgentState`. LangGraph merges a node's returned state against that TypedDict schema and drops
undeclared keys, so the flag read back False every call and the full system prompt was logged
**69 times** instead of once. Every other surviving private key (`_graph_steps`,
`_pending_weights_refs`, `_pending_configs`, …) is declared; this one was not.

Fixed by declaring it. A test now asserts that every private key `iterate.py` writes is on the
schema, so the whole class of bug is guarded.

---

## 3. Orchestrator output-cap failures (21 in one run)

Two symptoms, one cause.

- **20×** `response was cut off after 4096 output tokens (stop_reason=max_tokens)` — the JSON
  was truncated mid-object and the decision discarded. The reask usually recovered.
- **1×** `no parseable JSON object ... "[{'signature': 'EtJYCokB...'"`. That base64 decodes to
  a payload containing `claude-sonnet-5` and `thinking`: it is a **thinking-block signature**.
  The whole budget went to thinking, no text block was produced, and `_coerce_to_text` fell
  through to `str(raw)` — a Python repr of the transport envelope.

Measured across the run: **38 of 69 iterate calls hit exactly 4096.** Replaying a real iterate
prompt against the live API:

```
stop_reason = end_turn   output_tokens = 2443
thinking_tokens = 1923   blocks = ['thinking', 'text']   text = 1296 chars (~520 tokens)
```

**The decision JSON is ~520 tokens. Thinking consumed 1,923 — 79% of the budget.**

Why prompting does not fix it: the prompt tells the model its response must fit in the budget,
and it complies — its *answer* is short. But `max_tokens` bounds thinking **plus** answer, and
the model cannot see or budget its own thinking. On a hard decision thinking alone exceeds
3,500 and the answer is cut off.

`max_tokens` is a **required** parameter on the Anthropic Messages API; it cannot be omitted,
only raised. Ceiling for `claude-sonnet-5` is **128,000** output tokens (verified: 100,000 is
accepted, 200,000 returns `max_tokens: 200000 > 128000`).

**Status: OPEN.** Raising `_ITERATE_MAX_TOKENS` from 4096 to ~16k–32k costs nothing extra
unless tokens are actually generated (output billing is on real usage, not the cap) and
preserves thinking, which is worth keeping for this call. Note the SDK requires streaming for
requests that may exceed 10 minutes, which is only a concern at very large caps.

---

## 4. Curriculum size is refreshed at fixed capacity, not grown

The target is recomputed from a fixed formula on entry to each tier, never from the previous
dataset size (`agent/data_sizing.py::compute_curriculum_target`):

```
target = clamp(5000 × (0.5 + novelty) × size_factor, 5000, 25000)      novelty = 1 − zero_shot_baseline
```

For tier 3: `5000 × (0.5 + 0.284) × 0.50 = 1961`, clamped **up** to the 5000 floor. The target
then sat at exactly 5000 for eight consecutive rebuilds.

QC is not the culprit: **6 rows removed across 30+ rebuilds**, and synthesis survival is ~100%
(`450/450 kept`, `500/500 kept`). For `generation`, QC only filters length outliers.

The real mechanism is that each rebuild reconstructs the dataset from scratch and `allocate()`
stops at `target_rows` (`agent/nodes/curate.py:873`). Prior synthetic rows are discarded and the
remainder is resampled from the real pool, so the synthetic fraction *fell* from 46% (v1) to 5%
(v12). Only the mined-real pool persists.

Consequence: when tier 3's target dropped to 5000 while the previous dataset held 5754 rows,
**754 perfectly good rows were dropped to meet a lower cap.**

**Status: OPEN.** Desired behavior (per Evan): the target is a floor, never a reason to shrink.
Minimal change is `target_rows = max(computed_target, len(previous_dataset))`, so the curriculum
ratchets upward and only QC ever removes rows.

---

## 5. The `<think>` tag leak (tier 3 collapse)

Tier 3 fell 0.7157 → 0.1923 after fine-tuning. Eval samples:

```
raw : <think> </tool_call> Daina needs to put on her makeup and it will take her about an hour.
raw : <think> </tool_call>
raw : <think> </think> Shelly is volunteering at the food shelter. Jody does some charity work every year.
```

The third is a **correct summary** carrying a `<think> </think>` prefix.

**Confirmed:** `split_reasoning` only strips `<reasoning>...</reasoning>`:

```python
_REASONING_BLOCK_RE = re.compile(r"\s*<\s*reasoning\s*>.*?<\s*/\s*reasoning\s*>\s*", ...)
```

Tested directly — `<think> </think> Shelly is volunteering...` passes through **unstripped**, so
the judge grades the markup along with the summary. B251 fixed this for `<reasoning>` and missed
`<think>`, which is the tag Qwen actually emits.

**Two hypotheses were tested and ELIMINATED:**

1. *Prompt-template mismatch* — **ruled out.** Rendering both paths for
   `Qwen/Qwen3-4B-Instruct-2507` gives byte-identical strings:
   training/HF `...<|im_start|>assistant\n` and GGUF-eval `...<|im_start|>assistant\n`.
   (Qwen3-0.6B likewise matches, both with `<think>\n\n</think>\n\n`.)
2. *Training-target contamination* — **ruled out.** For `generation` the assistant target is the
   bare answer (`lora_trainer.py:335-339`); the `<reasoning>` wrapper only appears when
   `cot_reasoning` is present, and CoT is disabled for this task.

**Status: OPEN, origin unknown.** Remaining candidates: GGUF special-token conversion of the
merged 4B, or LoRA genuinely degenerating this model. The decisive experiment is to generate
from the base 4B GGUF and the merged 4B GGUF on identical prompts and diff the raw token
streams. Regardless of origin, stripping `<think>` in `split_reasoning` is correct and cheap.

---

## 6. Resource findings

- **The two GPUs never overlap.** Measured GPU0 ∩ GPU1 busy overlap: **0.0%**. vLLM idles for
  stretches up to 27.7 minutes while training runs on GPU 1, then training idles while curate
  runs on GPU 0. You are paying for two L40S and getting roughly one at a time.
- **Context is 4–18× oversized.** `SLM_MAX_SEQ_LENGTH=4096` against measured
  `min=62 p50=228 p95=458 p99=601 max=1206`, `truncated=0/5754`. The cap never binds. This is
  the *training/eval* sequence length for the small model — unrelated to the orchestrator's
  Claude context, which is a separate budget.
- **86% of GGUF builds are discarded** — 52 built, 45 reaped, ~1.25h wasted. Quantizing only
  after eval confirms an improvement would recover nearly all of it.
- **55 of 73 iterations rolled back** (~75%), roughly 18.7h of train+eval discarded.
- Time split: train 43.7%, judge 22.0%, curate 16.8%, eval 14.3%, GGUF 3.6%.

---

## 7. Training data is NOT the problem

Audit of `dataset_v12.jsonl` (4,999 rows: 2,500 anchor / 2,250 mined / 249 synthetic):

- 99.9% are genuine dialogue→summary pairs from DialogSum and SAMSum. Not web-scraped junk.
- Zero duplicate answers, ~0.1% degenerate rows (answer longer than text).
- Mined DialogSum carries inherited label noise — one confirmed role-swap where the gold summary
  reverses which speaker is the salesman and which is the reporter — and skews toward
  `#PersonN#`-style summaries (88% vs 44.5% in anchors).
- Synthetic rows are only 5% of the set and are on-task.

Worth cleaning, but not an 8-point effect. Data did not cause this plateau.

---

## 8. Priority queue

1. **Threshold floor** (§1) — nothing else matters until the goal is reachable.
2. **Strip `<think>` in `split_reasoning`** (§5) — cheap, and it is currently penalizing correct
   summaries.
3. **Curriculum ratchet** (§4) — stop discarding rows to meet a lower cap.
4. **Raise `_ITERATE_MAX_TOKENS`** (§3) — kills 38 wasted calls per run, keeps thinking.
5. **GPU overlap** (§6) — the largest wall-clock win available.
6. **Investigate the `<think>` origin** (§5) — needs the base-vs-merged GGUF diff.
