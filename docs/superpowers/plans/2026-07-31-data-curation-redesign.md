# Data Curation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make data curation non-deterministic and ungated, collapse the six rebuild strategies to three freely-chosen options, task-adaptively synth-fill curricula to a 3000-row floor, and escalate after 20 non-improving turns.

**Architecture:** `agent/data_rebuild.py` defines the three-strategy plan and a non-deterministic heuristic fallback (no dedup/rotation/exhaustion). `agent/nodes/curate.py` executes the chosen strategy with entropy-seeded sampling and tops up to target with task-adaptive synthesis. `data/curriculum.py` provides both hard-negative (classification/NER) and new-correct-example (generation-family) synthesis. `agent/nodes/iterate.py` presents the three ungated strategies to the orchestrator and escalates after 20 stalled evals. `config/config.py` sets the 3000 floor.

**Tech Stack:** Python 3.11, LangGraph, pytest, local vLLM (Qwen3.6-35B) synth endpoint.

## Global Constraints

- Three strategies only: `resample`, `acquire`, `synthesize`. No primary/support composition, no task/score gating.
- Non-deterministic: entropy-seeded sampling; no `data_rebuild_plan_identity` dedup, no `ensure_untried_data_rebuild_plan`, no `DataRebuildPlanSpaceExhausted`, no `query_variant`.
- Task-adaptive synthesis: hard-negatives for classification/NER; new *correct* verified/CoT examples for math/code/generation. Never wrong-answer SFT on generation-family.
- Endpoint-down or `SLM_CHEAP=1`: synthesis degrades to `acquire`/`resample` with a logged `allocation_fallbacks` entry — never crash.
- `CURRICULUM_SIZE_FLOOR = 3000`; `target_rows` clamp upper bound = `DATA_SIZE_CEILING` (10000); `MAX_STALL_EVALS = 20`.
- TDD, DRY, YAGNI, frequent commits. Run tests from repo root with `python -m pytest`.

---

### Task 1: Config — 3000 floor and 20-turn escalation

**Files:**
- Modify: `config/config.py:147` (`CURRICULUM_SIZE_FLOOR`)
- Modify: `agent/nodes/iterate.py:677` (`MAX_STALL_EVALS`)
- Test: `tests/config/test_curation_config.py` (create)

**Interfaces:**
- Produces: `config.config.CURRICULUM_SIZE_FLOOR == 3000` (default), `iterate.MAX_STALL_EVALS == 20` (default).

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_curation_config.py
import importlib, os

def test_curriculum_floor_default_is_3000(monkeypatch):
    monkeypatch.delenv("SLM_CURRICULUM_FLOOR", raising=False)
    import config.config as cfg
    importlib.reload(cfg)
    assert cfg.CURRICULUM_SIZE_FLOOR == 3000

def test_max_stall_evals_default_is_20(monkeypatch):
    monkeypatch.delenv("SLM_MAX_STALL_EVALS", raising=False)
    import agent.nodes.iterate as it
    importlib.reload(it)
    assert it.MAX_STALL_EVALS == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/config/test_curation_config.py -v`
Expected: FAIL (values are 1000 and 30).

- [ ] **Step 3: Implement the config changes**

In `config/config.py` change the floor default:

```python
CURRICULUM_SIZE_FLOOR = int(os.environ.get("SLM_CURRICULUM_FLOOR", "3000"))
```

In `agent/nodes/iterate.py` change the stall default:

```python
MAX_STALL_EVALS = int(_os.environ.get("SLM_MAX_STALL_EVALS", "20"))
```

Also update the comment above `MAX_STALL_EVALS` to say "20 evals in a row" and update the `CURRICULUM_SIZE_FLOOR` comment to note the 3000 per-task floor.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/config/test_curation_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/config.py agent/nodes/iterate.py tests/config/test_curation_config.py
git commit -m "config: 3000 curriculum floor, escalate after 20 stalled evals"
```

---

### Task 2: Three-strategy plan model (data_rebuild.py)

**Files:**
- Modify: `agent/data_rebuild.py` (whole strategy model)
- Test: `tests/nodes/test_data_rebuild_plan.py` (create; replaces assertions in `tests/nodes/test_data_rebuild_curate.py` covering removed symbols)

**Interfaces:**
- Produces:
  - `DATA_REBUILD_STRATEGIES == ("resample", "acquire", "synthesize")`
  - `normalize_data_rebuild_plan(raw, *, task_type, hypothesis, target_rows=3000, default_dataset_version=0, remaining_acquire_rounds=..., forbidden_eval_texts=()) -> dict` with keys: `schema_version, strategy, target_rows, resample_fraction, new_real_rows, synth_rows, max_acquire_rounds, difficulty_buckets, confusion_pairs, pattern_hint`. No `primary_strategy`, `support_strategies`, `query_variant`, `preserve_elite_fraction`, `elite`.
  - `fallback_data_rebuild_plan(state, *, hypothesis, score=None) -> dict` — non-deterministic strategy pick.
  - Removed (must not exist): `data_rebuild_plan_identity`, `tried_data_rebuild_plans`, `ensure_untried_data_rebuild_plan`, `DataRebuildPlanSpaceExhausted`, `eligible_data_rebuild_strategies`, `resolve_elite_source_path`, `require_resolvable_elite_source`, `TARGETED_SYNTH_TASK_TYPES`, `SAMPLING_STRATEGIES`, `MAX_SUPPORT_STRATEGIES`, `QUERY_VARIANTS`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing tests**

```python
# tests/nodes/test_data_rebuild_plan.py
import pytest
from agent import data_rebuild as dr

def test_only_three_strategies():
    assert dr.DATA_REBUILD_STRATEGIES == ("resample", "acquire", "synthesize")

def test_normalize_accepts_synthesize_for_math():
    plan = dr.normalize_data_rebuild_plan(
        {"strategy": "synthesize", "synth_rows": 50},
        task_type="math_reasoning", hypothesis="weak on hard bucket",
    )
    assert plan["strategy"] == "synthesize"
    assert plan["synth_rows"] == 50
    assert "primary_strategy" not in plan and "query_variant" not in plan

def test_normalize_rejects_removed_strategy():
    with pytest.raises(ValueError):
        dr.normalize_data_rebuild_plan(
            {"strategy": "preserve_elite_resample"},
            task_type="classification", hypothesis="x",
        )

def test_target_rows_clamps_to_ceiling_not_2000():
    plan = dr.normalize_data_rebuild_plan(
        {"strategy": "resample", "target_rows": 5000},
        task_type="classification", hypothesis="x",
    )
    assert plan["target_rows"] == 5000

def test_removed_symbols_absent():
    for name in ("DataRebuildPlanSpaceExhausted", "ensure_untried_data_rebuild_plan",
                 "data_rebuild_plan_identity", "tried_data_rebuild_plans",
                 "eligible_data_rebuild_strategies"):
        assert not hasattr(dr, name), name

def test_fallback_is_nondeterministic_or_signal_driven():
    state = {"task_type": "classification", "scores": [0.4],
             "test_report": {"by_difficulty": {"easy": {"accuracy": 0.3}}}}
    plan = dr.fallback_data_rebuild_plan(state, hypothesis="failing easy")
    assert plan["strategy"] in dr.DATA_REBUILD_STRATEGIES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/nodes/test_data_rebuild_plan.py -v`
Expected: FAIL (old six-strategy model, removed symbols still present).

- [ ] **Step 3: Rewrite `agent/data_rebuild.py`**

Replace the strategy constants and validation. Key edits:

```python
DATA_REBUILD_SCHEMA_VERSION = 2
DATA_REBUILD_STRATEGIES = ("resample", "acquire", "synthesize")
MAX_CONFUSION_PAIRS = 8
MAX_PAID_ACQUIRE_ROUNDS_PER_PLAN = 3
MAX_PAID_ACQUIRE_ROUNDS_PER_RUN = 9

_PLAN_FIELDS = frozenset({
    "schema_version", "strategy", "target_rows", "resample_fraction",
    "new_real_rows", "synth_rows", "max_acquire_rounds",
    "difficulty_buckets", "confusion_pairs", "pattern_hint",
})
```

Delete: `TARGETED_SYNTH_TASK_TYPES`, `MAX_SUPPORT_STRATEGIES`, `QUERY_VARIANTS`, `SAMPLING_STRATEGIES`, `_ELITE_PROVENANCE`, `_elite_reference`, `eligible_data_rebuild_strategies`, `resolve_elite_source_path`, `require_resolvable_elite_source`, `data_rebuild_plan_identity`, `tried_data_rebuild_plans`, `ensure_untried_data_rebuild_plan`, `DataRebuildPlanSpaceExhausted`, `_best_dataset_version` (keep if still used by fallback for dataset_version default; inline otherwise).

Rewrite `normalize_data_rebuild_plan` to validate a single `strategy` field against `DATA_REBUILD_STRATEGIES` (no task/score gating), keep the numeric clamps for `target_rows` (lower 16, **upper `DATA_SIZE_CEILING`**, step 8), `resample_fraction`, `new_real_rows`, `synth_rows`, `max_acquire_rounds`, plus `_normalized_difficulty`, `_confusion_pairs`, and the `pattern_hint` eval-text firewall (unchanged). Enforce a positive `synth_rows` when `strategy == "synthesize"` and positive `new_real_rows` when `strategy == "acquire"`:

```python
from config.config import DATA_SIZE_CEILING
# ... inside normalize:
strategy = raw.get("strategy")
if strategy not in DATA_REBUILD_STRATEGIES:
    raise ValueError(
        f"data_rebuild.strategy {strategy!r} must be one of "
        + ", ".join(DATA_REBUILD_STRATEGIES)
    )
# target_rows clamp: lower=16, upper=DATA_SIZE_CEILING, step=8
# ... build plan dict with the single "strategy" key ...
if strategy == "synthesize" and plan["synth_rows"] <= 0:
    plan["synth_rows"] = min(plan["target_rows"], 50)
if strategy == "acquire" and plan["new_real_rows"] <= 0:
    plan["new_real_rows"] = min(plan["target_rows"], 40)
```

Rewrite `fallback_data_rebuild_plan` + its helper to a **non-deterministic, signal-driven** chooser (no `tried` set, no rotation). Use `random` seeded from entropy:

```python
import random

def _fallback_strategy_from_signal(state, *, task_type, score=None):
    report = state.get("test_report") or {}
    buckets = report.get("by_difficulty") or {}
    def acc(name):
        v = (buckets.get(name) or {}).get("accuracy")
        return float(v) if v is not None else None
    easy, medium, hard = acc("easy"), acc("medium"), acc("hard")
    confusion = report.get("confusion_pairs") or []
    # Weighted, non-deterministic choice biased by the failure signal.
    weights = {"resample": 1.0, "acquire": 1.0, "synthesize": 1.0}
    if easy is not None and easy < 0.6:
        weights["acquire"] += 2.0            # bad labels/format => new real data
    if any(v is not None and v < 0.6 for v in (medium, hard)):
        weights["synthesize"] += 2.0         # boundary weakness => targeted synth
    if confusion:
        weights["synthesize"] += 1.0
    choices, w = zip(*weights.items())
    return random.choices(choices, weights=w, k=1)[0]
```

`fallback_data_rebuild_plan` builds the `raw` dict with the single `strategy`, `target_rows` from `state["curriculum_size_target"]` (default 3000), difficulty buckets weighted by deficit (keep existing logic), `synth_rows`/`new_real_rows` set to match the chosen strategy, then calls `normalize_data_rebuild_plan`. Remove all `query_variant`/`elite` fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/nodes/test_data_rebuild_plan.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/data_rebuild.py tests/nodes/test_data_rebuild_plan.py
git commit -m "data_rebuild: collapse to 3 ungated strategies, drop determinism/dedup/elite"
```

---

### Task 3: Task-adaptive synthesis in curriculum.py

**Files:**
- Modify: `data/curriculum.py` (add generation-family synthesis + a unified entry)
- Test: `tests/data/test_task_adaptive_synth.py` (create)

**Interfaces:**
- Produces:
  - `synthesize_examples(examples, *, task_type, n, generate_fn, verify_fn=None, fallback_teachers=(), log=None) -> list[dict]` — unified entry. For classification/NER delegates to existing `synthesize_hard_negatives`. For generation-family generates new correct examples in the same schema, verified via `verify_fn` when provided.
- Consumes: `data.synth_client.get_generate_fn` (from curate).

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_task_adaptive_synth.py
from data import curriculum

def _fake_gen(prompt, temperature=0.7, max_tokens=200):
    # Return a plausible math problem+answer JSON the parser expects.
    return '{"text": "What is 2+2?", "answer": "4", "cot_reasoning": "2+2=4"}'

def test_generation_family_produces_new_correct_rows():
    seed = [{"text": "What is 1+1?", "answer": "2"}]
    out = curriculum.synthesize_examples(
        seed, task_type="math_reasoning", n=1, generate_fn=_fake_gen,
        verify_fn=lambda row: row.get("answer") == "4",
    )
    assert len(out) == 1
    assert out[0]["answer"] == "4"          # correct, not a wrong-answer negative
    assert "_source" in out[0] and out[0]["_source"].startswith("synth:")

def test_classification_delegates_to_hard_negatives():
    seed = [{"text": "win a free prize now", "label": "spam"}]
    out = curriculum.synthesize_examples(
        seed, task_type="classification", n=1,
        generate_fn=lambda p, **k: "call me about the meeting",
    )
    assert out and out[0].get("label") is not None

def test_verify_fn_filters_incorrect_rows():
    seed = [{"text": "q", "answer": "2"}]
    out = curriculum.synthesize_examples(
        seed, task_type="math_reasoning", n=3, generate_fn=_fake_gen,
        verify_fn=lambda row: False,       # reject everything
    )
    assert out == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/data/test_task_adaptive_synth.py -v`
Expected: FAIL (`synthesize_examples` undefined).

- [ ] **Step 3: Implement `synthesize_examples` and generation-family synthesis**

Add to `data/curriculum.py`:

```python
_GENERATION_FAMILY = frozenset({
    "math_reasoning", "code_generation", "generation",
    "multilingual", "structured_extraction",
})

def _synthesize_new_correct(examples, *, task_type, n, generate_fn,
                            verify_fn=None, log=None):
    """Generate NEW correct in-distribution examples (never wrong-answer pairs)."""
    import json, random
    out = []
    anchors = list(examples)
    random.shuffle(anchors)
    attempts = 0
    max_attempts = n * 4
    while len(out) < n and attempts < max_attempts and anchors:
        attempts += 1
        anchor = anchors[attempts % len(anchors)]
        prompt = _new_example_prompt(anchor, task_type)   # task-aware, "same format"
        try:
            raw = generate_fn(prompt, temperature=0.7, max_tokens=512)
            row = json.loads(raw)
        except Exception:
            continue
        if not isinstance(row, dict) or not row.get("text"):
            continue
        if verify_fn is not None and not verify_fn(row):
            continue
        row["_source"] = f"synth:{task_type}"
        row["_provenance"] = "synthetic_positive"
        out.append(row)
    if log:
        log(f"  new-correct synthesis: {len(out)}/{n} kept ({attempts} attempts)")
    return out

def _new_example_prompt(anchor, task_type):
    import json
    return (
        f"Generate ONE new, correct {task_type} example in EXACTLY this JSON schema: "
        f"{json.dumps({k: anchor.get(k) for k in anchor if not k.startswith('_')})}. "
        "It must be a genuinely new, diverse, and correct instance — not a copy. "
        "Return only the JSON object."
    )

def synthesize_examples(examples, *, task_type, n, generate_fn,
                        verify_fn=None, fallback_teachers=(), log=None):
    if n <= 0 or not examples:
        return []
    if task_type in ("classification", "NER"):
        return synthesize_hard_negatives(
            examples, task_type=task_type, generate_fn=generate_fn,
            anthropic_client=None, log=log,
        )[: 2 * n]
    if task_type in _GENERATION_FAMILY:
        return _synthesize_new_correct(
            examples, task_type=task_type, n=n, generate_fn=generate_fn,
            verify_fn=verify_fn, log=log,
        )
    return []
```

If `synthesize_hard_negatives`' signature differs, adapt the call to its actual keyword names (check `data/curriculum.py:419`). Reuse existing verifiers where available (math answer match, code test execution) by passing `verify_fn` from curate.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/data/test_task_adaptive_synth.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add data/curriculum.py tests/data/test_task_adaptive_synth.py
git commit -m "curriculum: task-adaptive synthesis (new-correct examples for generation-family)"
```

---

### Task 4: Curate executes 3 strategies with entropy seeds

**Files:**
- Modify: `agent/nodes/curate.py:496-812` (strategy execution, seeds)
- Test: `tests/nodes/test_curate_strategies.py` (create)

**Interfaces:**
- Consumes: `data_rebuild.normalize_data_rebuild_plan`/`fallback_data_rebuild_plan` (Task 2), `curriculum.synthesize_examples` (Task 3).
- Produces: `curate_node(state)` that reads `plan["strategy"]`, seeds samplers from entropy, and writes `artifacts/dataset_v{N}.jsonl`.

- [ ] **Step 1: Write the failing test**

```python
# tests/nodes/test_curate_strategies.py
import agent.nodes.curate as curate

def test_seed_is_entropy_based_not_derived(monkeypatch):
    # Two calls to the internal seed source differ.
    s1 = curate._entropy_seed()
    s2 = curate._entropy_seed()
    assert s1 != s2

def test_resample_produces_dataset(tmp_path, monkeypatch):
    # Minimal state exercising the resample path; asserts a dataset file is written.
    # (Fill with the project's existing curate test fixture pattern from
    #  tests/nodes/test_data_rebuild_curate.py.)
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/nodes/test_curate_strategies.py::test_seed_is_entropy_based_not_derived -v`
Expected: FAIL (`_entropy_seed` undefined).

- [ ] **Step 3: Rewrite curate strategy execution**

- Add `def _entropy_seed() -> int: return int.from_bytes(os.urandom(8), "big")`.
- Replace the derived `seed = int(identity[:8],16) + ...` block (`curate.py:569-573`) with `seed = _entropy_seed()` and give each sampler its own `_entropy_seed()` (or `random.Random()` with no arg).
- Replace `strategies = [plan["primary_strategy"], *plan["support_strategies"]]` with `strategy = plan["strategy"]`.
- Map execution:
  - `acquire` → existing `mine_additional_real_rows` block (guarded by `if strategy == "acquire"`).
  - `synthesize` → `_synthesize_positive_rows` rewired to call `curriculum.synthesize_examples` (task-adaptive), guarded by `if strategy == "synthesize"`.
  - `resample` → the `_balanced_sample` fill path.
- Delete `_select_elite_rows`, `_difficulty_sample`, `_round_robin_sample` usage and the elite/difficulty/source-diversification branches. Remove `require_resolvable_elite_source` import and the `DataRebuildPlanSpaceExhausted` try/except around plan resolution (`curate.py:526-560`) — on missing plan just call `fallback_data_rebuild_plan`.
- Keep `_exclude_eval_rows` firewalls, `apply_quality_controls`, atomic write, provenance logging.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/nodes/test_curate_strategies.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/nodes/curate.py tests/nodes/test_curate_strategies.py
git commit -m "curate: execute 3 strategies with entropy-seeded sampling, drop elite/difficulty paths"
```

---

### Task 5: Synth-fill to target + graceful degradation

**Files:**
- Modify: `agent/nodes/curate.py` (post-assembly top-up, before final QC)
- Test: `tests/nodes/test_curate_synth_fill.py` (create)

**Interfaces:**
- Consumes: `curriculum.synthesize_examples`, `synth_client.is_available`/`get_generate_fn`.
- Produces: curate assembles `>= target_rows` when synthesis is available; logs an `allocation_fallbacks` entry and degrades when not.

- [ ] **Step 1: Write the failing tests**

```python
# tests/nodes/test_curate_synth_fill.py
import agent.nodes.curate as curate

def test_synth_fill_tops_up_to_target(monkeypatch):
    rows = [{"text": f"r{i}", "label": "a"} for i in range(10)]
    filled = curate._synth_fill_to_target(
        rows, target_rows=20, task_type="classification",
        generate_fn=lambda p, **k: "synthetic text",
        state={}, model_id="m",
    )
    assert len(filled) >= 20

def test_synth_fill_degrades_when_unavailable(monkeypatch):
    rows = [{"text": "r", "label": "a"}]
    fallbacks = []
    out = curate._synth_fill_to_target(
        rows, target_rows=50, task_type="classification",
        generate_fn=None, state={}, model_id="m", fallbacks=fallbacks,
    )
    assert out == rows                       # unchanged, no crash
    assert fallbacks and fallbacks[0]["from"] == "synthesize"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/nodes/test_curate_synth_fill.py -v`
Expected: FAIL (`_synth_fill_to_target` undefined).

- [ ] **Step 3: Implement synth-fill and degradation**

```python
def _synth_fill_to_target(dataset, *, target_rows, task_type, generate_fn,
                          state, model_id, fallbacks=None):
    """Top up the dataset with task-adaptive synthesis up to target_rows.

    Covers the initial curriculum (first curate pass) and every rebuild. If
    synthesis is unavailable (no endpoint / cheap mode), leave the dataset as-is
    and record an honest fallback instead of crashing.
    """
    deficit = target_rows - len(dataset)
    if deficit <= 0:
        return dataset
    if os.environ.get("SLM_CHEAP") == "1" or generate_fn is None:
        if fallbacks is not None:
            fallbacks.append({
                "policy": "synth_unavailable_degrade",
                "reason": "synthesis endpoint unavailable or cheap mode",
                "from": "synthesize", "to": "base_fill",
                "unfilled_rows": deficit,
            })
        _log(model_id, f"  Synth-fill unavailable; leaving {len(dataset)} rows "
                       f"({deficit} short of {target_rows})")
        return dataset
    from data.curriculum import synthesize_examples
    extra = synthesize_examples(
        dataset, task_type=task_type, n=deficit, generate_fn=generate_fn,
        verify_fn=_verifier_for(task_type, state), log=lambda m: _log(model_id, m),
    )
    return dataset + extra
```

Call `_synth_fill_to_target` in `curate_node` after the strategy assembly and CoT annotation, before `apply_quality_controls`/final firewall. Add `_verifier_for(task_type, state)` returning a math/code verifier where one exists, else `None`. When `strategy == "synthesize"` and the endpoint is down, the existing `_synthesize_positive_rows` no-op already records a fallback; synth-fill's degradation handles the top-up path.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/nodes/test_curate_synth_fill.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/nodes/curate.py tests/nodes/test_curate_synth_fill.py
git commit -m "curate: synth-fill curricula to target with graceful degradation"
```

---

### Task 6: Iterate decision prompt/schema — 3 ungated strategies

**Files:**
- Modify: `agent/nodes/iterate.py:497-623` (decision prompt), `:225-290` (plan validation path)
- Test: `tests/nodes/test_iterate_decision.py` (create or extend)

**Interfaces:**
- Consumes: `data_rebuild.normalize_data_rebuild_plan` (Task 2).
- Produces: the decision prompt presents `resample`/`acquire`/`synthesize` with no gating and instructs selection from the trajectory + `test_report`; validation accepts the single-`strategy` schema.

- [ ] **Step 1: Write the failing test**

```python
# tests/nodes/test_iterate_decision.py
import agent.nodes.iterate as it

def test_decision_prompt_lists_three_strategies():
    prompt = it._build_intervention_prompt_for_test()  # small helper exposing the text
    for s in ("resample", "acquire", "synthesize"):
        assert s in prompt
    for gone in ("preserve_elite_resample", "difficulty_weighted_sampling",
                 "source_diversification", "query_variant"):
        assert gone not in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/nodes/test_iterate_decision.py -v`
Expected: FAIL (old strategy names present / helper missing).

- [ ] **Step 3: Rewrite the decision prompt and validation**

- Replace the six-strategy JSON schema block (`iterate.py:562-577`) and surrounding prose with the three-strategy schema: `{"intervention": "data_rebuild"|"hyperparameter", "data_rebuild": {"strategy": "resample"|"acquire"|"synthesize", "target_rows": int, "synth_rows": int, "new_real_rows": int, "pattern_hint": str, "difficulty_buckets": {...}}}`.
- Instruction text: "Choose the strategy from the trajectory and your estimate of what is failing. There is no restriction — any strategy may be chosen at any score." Remove score-band gating language that implied eligibility.
- Update the validation path (`:225-290`) to call the new `normalize_data_rebuild_plan` (single `strategy`) and drop references to `support_strategies`/`ensure_untried_data_rebuild_plan`.
- Add the tiny `_build_intervention_prompt_for_test()` helper (or refactor the prompt into a named function the test can call) to make the prompt testable.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/nodes/test_iterate_decision.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/nodes/iterate.py tests/nodes/test_iterate_decision.py
git commit -m "iterate: present 3 ungated strategies, single-strategy plan schema"
```

---

### Task 7: Purge removed symbols + update docs

**Files:**
- Modify: `tests/nodes/test_data_rebuild_curate.py` (remove elite/exhaustion/query_variant/six-strategy tests), any other referencing removed symbols
- Modify: `agent/checkpoint.py` (drop `query_variant`/elite from any env/plan snapshot if present)
- Modify: `docs/PIPELINE.md`, `docs/DATA_CURATION_AND_CAPS.md`, `docs/PROMPTS.md`
- Test: full suite

**Interfaces:**
- Produces: green test suite; docs describe the 3-strategy, non-deterministic, ungated design.

- [ ] **Step 1: Find all references to removed symbols**

Run (expect matches to fix):
```bash
python -m pytest -q 2>&1 | tail -40
grep -rn "primary_strategy\|support_strategies\|query_variant\|preserve_elite\|DataRebuildPlanSpaceExhausted\|ensure_untried\|difficulty_weighted_sampling\|source_diversification\|targeted_synth_positive\|resample_existing" --include=*.py .
```

- [ ] **Step 2: Update/remove stale tests and references**

Delete or rewrite each test asserting old behavior (e.g. `test_data_rebuild_curate.py` elite-selection test at `:290-330`, any exhaustion/rotation tests). Remove removed-symbol imports across `agent/` and `tests/`.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS (no import errors, no references to removed symbols).

- [ ] **Step 4: Update docs**

- `docs/DATA_CURATION_AND_CAPS.md`: replace the six-strategy table and determinism section with the 3-strategy, entropy-seeded, ungated model; note the 3000 floor and 20-turn escalation; note `_quality_score`/elite removal.
- `docs/PIPELINE.md`: update the strategy list, `CURRICULUM_SIZE_FLOOR` (3000), and stagnation/stall knobs.
- `docs/PROMPTS.md`: update the intervention decision schema.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "cleanup: purge removed curation symbols, update docs for 3-strategy redesign"
```

---

## Self-Review

**Spec coverage:**
- §1 three strategies → Task 2 (model), Task 4 (execution), Task 6 (orchestrator).
- §2 task-adaptive synthesis → Task 3; ungating → Task 2 (no gates) + Task 6 (prompt).
- §3 non-determinism → Task 2 (remove dedup/exhaustion) + Task 4 (entropy seeds).
- §4 synth-fill → Task 5.
- §5 config → Task 1; `target_rows` clamp → Task 2.
- §6 affected files → Tasks 2–7. §7 testing → each task's tests + Task 7 suite.

**Placeholder scan:** Task 4 Step 1 second test and Task 5 verifier reference the existing curate fixture pattern; implementer must copy the fixture from `tests/nodes/test_data_rebuild_curate.py`. All code steps show real code. No "TBD"/"handle edge cases".

**Type consistency:** `strategy` (str) key used consistently across Tasks 2/4/6; `synthesize_examples(examples, *, task_type, n, generate_fn, verify_fn, fallback_teachers, log)` signature identical in Tasks 3/5; `_synth_fill_to_target` and `_entropy_seed` names consistent in Tasks 4/5.
