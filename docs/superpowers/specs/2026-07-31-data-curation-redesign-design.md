# Data Curation Redesign — Design

**Date:** 2026-07-31
**Status:** Approved for implementation (user directed: proceed to implementation)

## Goal

Rework the data-curation subsystem so that:

1. Curation is **non-deterministic** — genuine run-to-run variety, no seeded reproducibility.
2. Synthetic generation is **ungated** — available for every task type and every score band.
3. Interventions collapse from six strategies to **exactly three**, freely chosen by the
   orchestrator from the trajectory and its estimate of what is failing.
4. The **initial curriculum** (and every curate pass) is topped up with synthetic data to
   hit the target size when real data + hard negatives fall short.
5. Config: per-task dataset floor **3000**; escalate after **20** non-improving turns.

Non-goals: changing the training loop, eval harness, model selection, or the acquisition
ladder's real-source loaders.

---

## 1. The three strategies

Replace `DATA_REBUILD_STRATEGIES` (six) with three, single-choice (no primary/support
composition, no eligibility gating by task type or score):

| Strategy | Meaning | Replaces |
|---|---|---|
| `resample` | Reshuffle / re-draw rows from the existing pool (real entropy). | `resample_existing`, `source_diversification`, `difficulty_weighted_sampling` |
| `acquire` | Add new rows from the same or a new provenance (real-source mining). | `mine_new_real_source` |
| `synthesize` | Generate new synthetic rows, task-adaptive (see §2). | `targeted_synth_positive` |

- The orchestrator chooses **one** strategy per `data_rebuild` intervention.
- `preserve_elite_resample` is **removed** (it depended on `_quality_score`, which is never
  populated — see [DATA_CURATION_AND_CAPS.md] follow-up). Elite preservation is dropped.
- No `support_strategies`, no `TARGETED_SYNTH_TASK_TYPES` gate, no `SAMPLING_STRATEGIES`
  single-sampling rule, no score-band gate on synthesis.

### Orchestrator decision

`iterate.py` decision prompt is rewritten to present the three strategies and instruct the
model to pick based on the trajectory + per-difficulty accuracy + confusion pairs (the
existing `test_report` signal). No restriction on which it may pick, at any score.

**Fallback (LLM parse failure / cheap-mode parse miss):** a lightweight *non-deterministic*
heuristic picks a strategy weighted by the failure signal (e.g. failing easy bucket →
`acquire`; weak medium/hard → `synthesize`; otherwise random among the three). This
replaces `_fallback_strategy_from_signal`'s deterministic chooser. It never rotates for
"untried" — see §3.

---

## 2. Task-adaptive synthesis

`synthesize` behaves differently by task family, always producing rows **in the same format
as the real data**:

- **classification / NER:** contrastive hard negatives (current `synthesize_hard_negatives`
  behavior), ungated.
- **math_reasoning / code_generation / generation (and other generation-family):** generate
  **new correct in-distribution examples** — never wrong-answer contrastive pairs. Where a
  verifier exists (math answers, code tests) the synthetic example is verified before it is
  kept; CoT is annotated as today. Unverifiable generation examples are kept after the
  standard quality controls (dedup, length, balance).

This preserves the empirical finding that wrong-answer SFT harms generation-family tasks
while still letting synthesis fill those curricula.

### Ungating

- Remove the classification/NER task gate.
- Remove the `score >= 0.95` synthesis gate.
- Remove the "stop the run if synth endpoint unavailable" hard requirement
  (`SLM_REQUIRE_SYNTH` fail-closed). Instead: if `synthesize` is chosen and the endpoint is
  genuinely unreachable, **degrade to `acquire`** (or `resample` if acquisition is
  exhausted) and record an honest `allocation_fallbacks` entry. No crash, no silent
  misattribution.
- `SLM_CHEAP` still skips synthesis (cost control) — unchanged.

---

## 3. Non-determinism

- **Sampling/synthesis entropy:** replace the derived `seed` (identity hash + version +
  query_variant) and all `random.Random(seed)` samplers in `curate.py` with entropy-seeded
  RNGs (`random.Random()` / `os.urandom`). Reshuffles and synthesis anchor draws genuinely
  vary run to run.
- **Remove the dedup/exhaustion machinery** in `data_rebuild.py`:
  - `data_rebuild_plan_identity` — remove its dedup role (may retain a hash purely for log
    provenance, but nothing gates on it).
  - `tried_data_rebuild_plans`, `ensure_untried_data_rebuild_plan`,
    `DataRebuildPlanSpaceExhausted` — **removed**. Curate no longer rotates to an "untried"
    plan and no longer terminates on plan-space exhaustion.
  - The `query_variant` field (only existed to vary the seed) — **removed**.
- **Backstop:** escalation-on-no-improvement (§5) becomes the sole mechanism that ends a
  stuck run. The wall-clock guard remains.
- **Consequence (accepted):** exact checkpoint-resume reproducibility is dropped. A resumed
  segment will not reproduce identical datasets; it continues with fresh entropy.

---

## 4. Synth-fill to target

Add a general top-up step at the end of curate's assembly (after the chosen strategy's rows
+ any hard negatives are allocated, before final quality controls / eval firewall):

```
if len(dataset) < target_rows and synthesis available and not SLM_CHEAP:
    dataset += task_adaptive_synth(target_rows - len(dataset))   # verified/format-matched
```

Because the first curate pass builds the initial curriculum (via the fallback plan), this
covers the user's "initial curriculum" case *and* every later rebuild. Synth-fill respects
the same task-adaptive rules and quality controls as §2. If synthesis is unavailable, the
dataset is left at whatever real+resampled size was reached (logged), never crashing.

---

## 5. Config changes

In `config/config.py`:

- `CURRICULUM_SIZE_FLOOR`: `1000 → 3000`. This is the per-task floor the planner target is
  clamped up to, so every task trains on ≥3000 rows. `DATASET_SIZE_BY_TYPE` remains as the
  no-planner fallback but is effectively clamped up to 3000; values left as-is.
- `DATA_SIZE_CEILING`: unchanged at 10000 (≥ 3000, no conflict).

In `agent/data_rebuild.py`:

- `target_rows` clamp upper bound `2000 → DATA_SIZE_CEILING` (10000) so a rebuild can reach
  the 3000 floor. Default stays sourced from `curriculum_size_target`.

In `agent/nodes/iterate.py`:

- `MAX_STALL_EVALS`: `30 → 20` (escalate after 20 consecutive non-improving evals). Note:
  this knob was 30, not 50, as of this change; `STAGNATION_WINDOW` is already 20 and is left
  aligned at 20. (User referred to it as "50"; the operative no-improvement knob is
  `MAX_STALL_EVALS`.)

Env overrides (`SLM_CURRICULUM_FLOOR`, `SLM_MAX_STALL_EVALS`, etc.) are preserved; only
defaults change.

---

## 6. Affected files

- `agent/data_rebuild.py` — collapse strategies to 3; remove dedup/rotation/exhaustion,
  `query_variant`, elite, support-strategy validation; simplify `normalize_data_rebuild_plan`;
  rewrite the fallback chooser to a non-deterministic heuristic; raise `target_rows` clamp.
- `agent/nodes/curate.py` — entropy seeds; execute the 3 strategies; task-adaptive synthesis;
  synth-fill-to-target; drop elite/difficulty/round-robin paths and the exhaustion
  terminate-clean branch.
- `data/curriculum.py` — add task-adaptive "new correct example" synthesis for
  generation-family alongside the existing hard-negative path; expose a synth-fill entry.
- `agent/nodes/iterate.py` — rewrite decision prompt/schema to the 3 ungated strategies;
  `MAX_STALL_EVALS = 20`.
- `config/config.py` — `CURRICULUM_SIZE_FLOOR = 3000`.
- `agent/nodes/cold_start/eval_setup.py` — ensure initial acquisition target reflects the
  3000 floor (already clamped via `curriculum_size_target`; verify).
- Tests under `tests/` referencing removed symbols (`DataRebuildPlanSpaceExhausted`,
  `ensure_untried_data_rebuild_plan`, elite, `query_variant`, six-strategy validation) —
  updated/removed to match.
- Docs (`docs/PIPELINE.md`, `docs/DATA_CURATION_AND_CAPS.md`, `docs/PROMPTS.md`) — updated to
  describe the 3-strategy, non-deterministic, ungated design.

## 7. Testing

- Unit: strategy normalization accepts exactly the 3 strategies and rejects the old ones;
  synth-fill tops up to target; task-adaptive synthesis returns correct-format rows and
  verified correctness for math/code; endpoint-down degrades to `acquire`/`resample` with a
  logged fallback (no raise).
- Non-determinism: two curate passes on identical state produce **different** datasets.
- Config: floor clamps a small planner target up to 3000; `MAX_STALL_EVALS` escalation fires
  at 20.
- Regression: removed symbols are gone; no import references remain.
