# Data Curation, Failure Analysis & Token Caps

> **Redesign 2026-07-31.** The data-curation subsystem was reworked (see
> [spec](superpowers/specs/2026-07-31-data-curation-redesign-design.md)). The six rebuild
> strategies collapsed to **three** — `resample`, `acquire`, `synthesize` — chosen freely by
> the orchestrator with no task/score gating. Curation is now **non-deterministic**
> (entropy-seeded sampling; the plan-identity dedup, untried-plan rotation, and
> `DataRebuildPlanSpaceExhausted` were removed). Synthesis is **ungated** and **task-adaptive**
> and **synth-fills** every curriculum up to the target size.
>
> **UPDATE 2026-08-05 — escalation policy.** ONE mechanism: escalate after
> `STAGNATION_WINDOW` (15) evals without a gain greater than `STAGNATION_MIN_DELTA` (2%),
> measured over an append-only eval history so rollback cannot reset it, plus an
> unconditional ceiling at `MAX_EVALS_BEFORE_ESCALATION` (30) evals. `MAX_STALL_EVALS` was
> deleted. Curriculum floor is 5000, ceiling 25000.
>
> **UPDATE 2026-08-05 — dataset sizing is now per-tier and deterministic.** `DATASET_SIZE_BY_TYPE`
> was deleted (every value sat below the floor and was clamped away). `agent/data_sizing.py`
> computes the target from two measured signals — task novelty (`1 − zero_shot_baseline`) and model
> capacity (inverse parameter count) — clamped to [5000, 25000], recomputed on entry to every tier:
>
> ```
> target = clamp(5000 × (0.5 + novelty) × clamp(1e9/n_params, 0.5, 2.0), 5000, 25000)
> ```
>
> A bigger model therefore gets a SMALLER target. Override with `SLM_CURRICULUM_SIZE`.
>
> **UPDATE 2026-08-05 — `synthesize` splits into fill and surgical.** `surgical_synthesize` spends
> `SLM_SURGICAL_SYNTH_SHARE` (default 20%) of the plan's `synth_rows` on the top
> `SLM_SURGICAL_MAX_PAIRS` (5) confusion pairs, budgeted **proportionally to each pair's confusion
> count** and clamped to 10–100 rows per pair. Anchors come from the **gold** class (the one the
> model should have predicted). A pair that was targeted before and whose count did not fall is
> logged `EXHAUSTED` and skipped. The remainder is `fill` — balanced across the whole label space.
>
> Sections below marked *(pre-redesign)* describe the old model and are retained for history.

A walkthrough of three tightly-related parts of the pipeline:

1. How the `data_rebuild` intervention works, including synthetic data generation.
2. How the system analyzes failures, and what that analysis is used for.
3. Every token cap in the repo and the rationale behind each.

---

## 1. The `data_rebuild` intervention & synthetic data generation

### The intervention model

The pipeline is a LangGraph loop (`agent/graph.py`):

```
task_analysis → eval_setup → model_selection → curate → train → evaluate → iterate ↺
```

Each iteration the orchestrator LLM picks **exactly one** of two mutually-exclusive
interventions (`agent/nodes/iterate.py:510-513`):

- **`data_rebuild`** → changes the training data → routes to `curate`
- **`hyperparameter`** → changes LoRA settings → routes to `train`

Only one thing changes per iteration so score movements stay causally attributable.
`curate_node` is a **no-op unless `last_intervention == "data_rebuild"`**
(`agent/nodes/curate.py:503-506`) — otherwise the dataset is frozen.

### The declarative "data_rebuild plan"

Curation is driven by a bounded, JSON-only plan (validated in `agent/data_rebuild.py`)
that declares a `primary_strategy` + up to two support strategies. The six strategies
(`agent/data_rebuild.py:24-31`):

| Strategy | What it does |
|---|---|
| `resample_existing` | reshuffle current pool |
| `preserve_elite_resample` | resample top-quality rows from a prior elite version |
| `mine_new_real_source` | acquire **new real** rows from HF/Exa |
| `source_diversification` | round-robin across sources |
| `difficulty_weighted_sampling` | sample by easy/medium/hard quotas |
| `targeted_synth_positive` | **LLM synthetic data generation** |

The plan validator (`normalize_data_rebuild_plan`, `agent/data_rebuild.py:313-517`)
clamps every numeric field (`target_rows` [16,2000], `synth_rows` [0,200], etc.), and
`ensure_untried_data_rebuild_plan` rotates the plan to an untried variant — raising
`DataRebuildPlanSpaceExhausted` to **terminate cleanly** (added after a 44.8-hour NER
run crashed on exhaustion, `agent/nodes/curate.py:547-560`).

### Where synthetic data is actually generated

Entered via `_synthesize_positive_rows` (plan `synthesize` strategy) or
`_synth_fill_to_target` (top-up), both in `agent/nodes/curate.py`, then
`data/curriculum.py::synthesize_examples`.

**The backend is a LOCAL vLLM endpoint serving Qwen3.6-35B-A3B, *not* Claude**
(`data/synth_client.py:1-13`) — chosen for zero Claude cost, reproducibility, and
contamination safety. It runs in non-thinking mode (`enable_thinking=False`,
`top_p=0.80`, `top_k=20`).

**Gold-only generation (2026-08-05).** For every task family, synthesis produces NEW CORRECT
in-distribution examples. A generated row **inherits its label from a real anchor row** rather
than being assigned one by the model, so an out-of-vocabulary or mismatched label is impossible
by construction:

- **classification / NER** — `_synthesize_new_gold`: anchors are drawn **round-robin across the
  label space** (so rare classes get equal attention), and the model is asked for one new
  utterance of the *same* class as its anchor.
- **math/code/generation** — `_synthesize_new_correct`: new correct instances in the anchor's
  JSON schema, kept only if a supplied `verify_fn` accepts them. Wrong-answer SFT harms these
  families. They also get **chain-of-thought annotation** (`annotate_cot`).

**Teacher label verification** (`verify_generated_labels`, on by default, `SLM_VERIFY_SYNTH=0`
to disable). Every generated classification row is shown back to the teacher model, which
answers `{"valid": bool, "reason": str}`. Rejected rows are dropped and the teacher's reason is
logged. Verification failures (unparseable reply, endpoint error) KEEP the row — the verifier
can never empty a dataset.

### The "honest attribution" invariant

A key design rule: a plan declaring synthesis that produces **zero** synthetic rows is a
"strategy attribution lie." So if `SLM_REQUIRE_SYNTH != "0"` (default) and the local
endpoint isn't reachable, the run **stops** with `SynthesisUnavailableError` rather than
silently falling back to gold-only (`agent/nodes/curate.py:385-403`). When a strategy
legitimately no-ops, it's rewritten to `base_fill` and logged in `allocation_fallbacks`.

### Real-data acquisition ladder

For `mine_new_real_source` and initial setup, `data/loaders/web_acquire.py` walks:
local bundle → deterministic benchmark loaders (GSM8K, CoNLL, BC5CDR…) → agentic HF
discovery via Exa + orchestrator column-mapping → bounded Exa web scrapes →
**last-resort gold synthesis via the Claude orchestrator** (distinct from the local curriculum
synthesis above).

Everything runs through an **eval firewall** (`_exclude_eval_rows`, applied at mining,
synthesis, and final write) to prevent contamination, and output is atomically written to
`artifacts/dataset_v{N}.jsonl`.

---

## 2. Failure analysis: the "test-data agent"

### What it produces

Failure analysis lives in `agent/nodes/test_agent.py`. It's a **contamination
firewall**: it reports only aggregate numbers, never raw failing examples. It produces
three things:

1. **Difficulty bucketing** (easy/medium/hard) by base-model capability gradient — easy =
   both smallest & largest base models pass; hard = neither
   (`agent/nodes/test_agent.py:47-88`).
2. **A confusion table** — top-8 `(gold, predicted)` pairs per task type in a `Counter`
   (`agent/nodes/test_agent.py:180-216`).
3. **A rule-based diagnosis → intervention** (`agent/nodes/test_agent.py:130-170`): easy
   bucket < 0.6 → `data_rebuild`; easy solid but medium/hard weak → `hyperparameter`;
   below goal with no single failing bucket → `data_rebuild`.

### What the failure analysis is used for

- **Iteration decisions** (`agent/nodes/iterate.py:778-801`): per-difficulty accuracy +
  diagnosis + confusion pairs feed the orchestrator prompt, driving the next intervention
  and even letting the LLM lower `stop_threshold` if a failure reflects genuine capacity
  limits.
- **Reporting** (`data/curation_log.py:82-102`): top confusion pairs are written to the
  curation log under an "Aggregate confusion counts" heading.
- **Targeted curriculum synthesis** (`agent/nodes/curate.py::_surgical_synthesize`): the top
  confusion pairs set the budget for extra in-class gold rows, so synthesis effort follows the
  evidence about which decision boundaries the model is actually getting wrong.

---

## 3. Every token cap in the repo, and why

No token caps live in YAML/JSON — they're Python constants, function defaults,
env-var-with-default, or the `_EVAL_OUTPUT_TOKEN_SETTINGS` table.

### Response/generation caps that affect the shipped SLM

**Eval output reserve — the central task-aware cap** (`eval/harness.py:8-14`):

```python
_EVAL_OUTPUT_TOKEN_SETTINGS = {
    "classification": ("SLM_EVAL_MAX_NEW_TOKENS_CLASSIFICATION", 50),
    "NER":            ("SLM_EVAL_MAX_NEW_TOKENS_NER", 512),
    "math_reasoning": ("SLM_EVAL_MAX_NEW_TOKENS_MATH", 512),
    "generation":     ("SLM_EVAL_MAX_NEW_TOKENS_GENERATION", 512),
    "code_generation":("SLM_EVAL_MAX_NEW_TOKENS_APPS", 1024),
}
```

**Why these values:** they match the shape of a correct answer. Classification emits a
single label → 50 tokens is plenty and prevents rambling into wrong territory. NER/math/
generation need room for entity lists / reasoning chains / summaries → 512 (raised from
256 in B147). Code needs the most → 1024. `eval_output_token_reserve()`
(`eval/harness.py:17-49`) validates the reserve is positive and `< max_seq_length`,
guaranteeing prompt budget remains. Defaults are duplicated in
`agent/checkpoint.py:150-154` so resumed runs reproduce identical caps.

**Max sequence length — the global context cap** (default 4096, `SLM_MAX_SEQ_LENGTH`):
set in `training/slm_helpers.py:31-32`, used for model load
(`training/slm_helpers.py:286`), GGUF `n_ctx` (`training/slm_helpers.py:705`), and
training `SFTConfig.max_length` (`training/lora_trainer.py:855`). **Why:** input budget is
always `max_seq_length - max_new_tokens`, and the code **never truncates** — over-length
rows raise (`training/lora_trainer.py:154-158`) rather than silently dropping content.
Notably `training/slm_helpers.py:303` sets `generation_config.max_length = None` to
disable Qwen's built-in 40960 cap so `max_new_tokens` is the *sole* length knob (fix
B128). 4096 was raised from 512 (B204/B188) to fit APPS code prompts.

**Inference primitive defaults** — `max_new_tokens=50` on
`infer`/`infer_batch`/`infer_batch_gguf` (`training/slm_helpers.py:475,622,663`).
**Why:** a safe minimal default for the common classification case; eval always overrides
it via the table above.

### Synthetic-data generation caps

- **Local synth default 200 tokens** (`data/synth_client.py:177,188`) — generated classification
  rows are short texts; 200 keeps them tight and fast under continuous batching.
- **CoT authoring 512 tokens** (`data/curriculum.py:147`) — reasoning chains need room. This is
  the only CoT path: the local Qwen3.6 synth endpoint. There is no cloud CoT fallback; if it is
  unreachable the example is left CoT-less.

### Judge and orchestrator caps

- **LLM-judge: `max_tokens=8`** (`eval/judge_client.py:489-490`) — the judge outputs only
  a score token, so 8 forbids explanation drift and cuts cost.
- **Orchestrator (Claude) per-call caps** — hardcoded per decision, sized to the JSON each
  stage returns: task analysis 1024 (`agent/task_planner.py:263`), iterate 1536
  (`agent/nodes/iterate.py:49`), escalate 256, downward_probe 120, model_selection 256,
  schema-mapping 400, **seed synthesis 4096** (`data/loaders/web_acquire.py:1779`).
  **Why varied:** each is tuned to the expected output size — a yes/no probe gets 120,
  seed-data generation gets 4096. The orchestrator's *input* side is separately governed
  by the Sonnet 1M-token beta (`config/config.py:90-104`) so long trajectories aren't
  truncated.
- **Trajectory compaction at 8000 tokens** (`agent/context_manager.py:118-120`) —
  proactively compacts agent history before it bloats orchestrator calls.

### Overarching justification

Three consistent principles explain every cap:

1. **Right-size output to the task** — a label gets 50, code gets 1024, a judge score gets
   8. Tight caps prevent runaway generation, cut cost/latency, and stop models wandering
   into wrong answers.
2. **Never silently truncate** — input budget = context − output reserve, and over-length
   rows raise rather than lose data. Caps are enforced as hard invariants, not lossy
   conveniences.
3. **Reproducibility** — defaults are duplicated into checkpoint snapshots and SLURM
   exports so a resumed or re-run job reproduces identical caps.
