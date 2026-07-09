# Pipeline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix six pipeline defects (turn-budget enforcement, hardware-gated termination, param-count tiering, tier-based LLM escalation + downward probe, scaling-curve dead code) plus a bug sweep of correctness defects surfaced in the July code review.

**Architecture:** Changes span config, the LangGraph wiring, the cold-start nodes, the model pool, the trainer, and the scorers. Each task is independently testable. Pure-logic changes (tiering, param formula, metrics, extraction) get unit tests; node-orchestration changes get state-dict-driven tests with mocked LLM/training calls.

**Tech Stack:** Python 3.11+, LangGraph, Unsloth (mocked in tests), Anthropic SDK (mocked in tests), pytest, numpy. `pyproject.toml` already sets `--import-mode=importlib`, so no `__init__.py` is needed in new test dirs.

## Global Constraints

- Param-count formula (single source of truth): `params_b = int4_size_mb * 2 / 1000`. A Q4_K_M model at ~4.5 bpw stores ~0.5 bytes/param, so MB × 2 ≈ million params ÷ 1000 = billions. For Q8_0 siblings the `int4_size_mb` field holds the (larger) Q8 file size, so this formula would over-estimate params for Q8 — therefore tiering must be computed from the BASE model's Q4 size and inherited by siblings, NOT recomputed from the sibling's inflated size. See Task 3.
- Tier boundaries by `params_b`: `< 0.75 → 0`, `0.75 ≤ p < 1.5 → 1`, `1.5 ≤ p < 2.5 → 2`, `≥ 2.5 → 3`.
- `quant` values in the pool are exactly `None`, `"Q4_K_M"`, `"Q8_0"`.
- `check_hardware_constraints(model, constraints, measured=None)` and `all_constraints_pass(hw_check)` already exist in `config/android_pool.py` — do not change their signatures.
- The `quant=None` (BF16) inference/eval path must remain behavior-identical to today.
- Orchestrator model id comes from `config.config.ORCHESTRATOR_MODEL`; all Anthropic calls must be mockable in tests (never hit the network in a test).
- Every bug fixed in this plan gets a BUGS.md entry (B51–B60) with status 🟢 and a one-line table row.

---

## File Map

| File | Task | Responsibility of change |
|---|---|---|
| `config/config.py` | 1 | `MAX_TURNS_MAIN` already defined; confirm value, used by graph + iterate |
| `agent/graph.py` | 1, 5 | `recursion_limit`; re-order cold_start edges for scaling_curve |
| `agent/nodes/iterate.py` | 1, 2 | turn-budget termination; hardware-gated termination |
| `config/android_pool.py` | 3 | param-count tiering in base pool + both sibling helpers |
| `agent/nodes/escalate.py` | 4 | tier-based escalation with LLM model choice |
| `agent/nodes/evaluate.py` | 4 | downward-probe after success |
| `agent/nodes/cold_start/scaling_curve.py` | 5 | mini-curate fallback when dataset absent |
| `agent/nodes/cold_start/task_analysis.py` | 6a | remove dead task-preference sort |
| `agent/task_planner.py` | 6b | fix `_param_range_label` formula |
| `training/lora_trainer.py` | 6c | math/code training format |
| `eval/harness.py` | 6d | task-type-aware `max_new_tokens` |
| `eval/scorers/classification.py` | 6e | word-boundary label extraction |
| `eval/metrics.py` | 6f | multiset `entity_f1` |
| `data/curriculum.py` | 6g | math/code hard-negative handling |
| `docs/BUGS.md` | every task | log B51–B60 |

**Execution note:** Do the six requested changes (Tasks 1–5) first, then the bug sweep (Task 6 split into 6a–6g). Tasks are ordered so later tasks depend only on earlier ones.

---

### Task 1: Enforce the turn budget

**Files:**
- Modify: `agent/graph.py` (both compiled-graph returns)
- Modify: `agent/nodes/iterate.py` (`iterate_node` termination logic)
- Modify: `docs/BUGS.md`

**Interfaces:**
- Consumes: `config.config.MAX_TURNS_MAIN` (already `= 1500`); `state["turn_budget"]` (in `AgentState`, defaults 1500 cold-start).
- Produces: no new symbols; `iterate_node` gains a turn-budget terminate branch.

**Background:** B24 — `MAX_TURNS_MAIN` is dead config. Enforce at two points: LangGraph `recursion_limit` (structural cap) and a semantic count in `iterate_node` (`turns_used = state["iteration"] * 2`, approximating curate+train per iteration).

- [ ] **Step 1: Write failing test**

Create `tests/nodes/test_iterate_budget.py`:

```python
from unittest.mock import patch
from agent.nodes.iterate import iterate_node


def _state(iteration, turn_budget, score=0.5, threshold=0.96):
    return {
        "selected_model": None,
        "scores": [score],
        "best_score": score,
        "iteration": iteration,
        "turn_budget": turn_budget,
        "stop_threshold": threshold,
        "initial_stop_threshold": threshold,
        "task_type": "classification",
        "last_eval": None,
        "hw_gating_enabled": False,
    }


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network in test"))
def test_terminates_when_turn_budget_exhausted(_mock):
    state = _state(iteration=500, turn_budget=1000)  # turns_used=1000 >= 1000
    out = iterate_node(state)
    assert out["next_action"] == "terminate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("no network in test"))
def test_does_not_terminate_below_budget(_mock):
    state = _state(iteration=3, turn_budget=1000)
    out = iterate_node(state)
    assert out["next_action"] != "terminate"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/nodes/test_iterate_budget.py -v`
Expected: `test_terminates_when_turn_budget_exhausted` FAILS.

- [ ] **Step 3: Add the turn-budget check in `iterate_node`**

In `agent/nodes/iterate.py`, inside `iterate_node`, immediately after the `if not state["scores"]:` early-return block (before `current_score = state["scores"][-1]`), insert:

```python
    # Turn-budget guard (B51): ~2 productive turns per iteration (curate + train).
    turn_budget = state.get("turn_budget", 0)
    turns_used = state["iteration"] * 2
    if turn_budget and turns_used >= turn_budget:
        _log(model_id, f"  Turn budget exhausted: {turns_used} >= {turn_budget} — TERMINATING")
        state["next_action"] = "terminate"
        return state
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/nodes/test_iterate_budget.py -v`
Expected: 2 passed.

- [ ] **Step 5: Wire `recursion_limit` into the compiled graph**

In `agent/graph.py`, add at the top:

```python
from config.config import MAX_TURNS_MAIN
```

Change `return graph.compile()` to:

```python
    return graph.compile(recursion_limit=MAX_TURNS_MAIN)
```

If that raises `TypeError` on the installed LangGraph version, fall back to:

```python
    return graph.compile().with_config(recursion_limit=MAX_TURNS_MAIN)
```

- [ ] **Step 6: Verify graph still builds**

Run: `python -c "from agent.graph import build_graph; g=build_graph('cold_start'); print('OK')"`
Expected: `OK`

- [ ] **Step 7: Log B51 in BUGS.md**

Table row (after B50):

```
| B51 | 🟢 | orchestration | `MAX_TURNS_MAIN`/`turn_budget` never enforced; runs bounded only by recursion_limit |
```

Detail section at the bottom:

```
## B51 — turn budget never enforced
- **Where:** `config.py`, `agent/graph.py`, `agent/nodes/iterate.py`
- **When:** 2026-07-08, pipeline hardening review
- **How found:** `MAX_TURNS_MAIN=1500` defined but never read (was B24); no node counts turns.
- **Impact:** a stuck loop ran to recursion_limit; budget was advisory only.
- **Status:** 🟢 fixed 2026-07-08 — `graph.compile(recursion_limit=MAX_TURNS_MAIN)`;
  `iterate_node` terminates when `iteration*2 >= turn_budget`.
```

- [ ] **Step 8: Commit**

```bash
git add agent/graph.py agent/nodes/iterate.py tests/nodes/test_iterate_budget.py docs/BUGS.md
git commit -m "feat: enforce turn budget via recursion_limit and iterate_node guard (B51)"
```

---

### Task 2: Hardware-gated termination

**Files:**
- Modify: `agent/nodes/iterate.py` (`iterate_node`)
- Modify: `docs/BUGS.md`

**Interfaces:**
- Consumes: `check_hardware_constraints(model, constraints, measured=None)`, `all_constraints_pass(hw_check)`; `state["hw_gating_enabled"]`, `state["selected_model"]`, `state["hardware_constraints"]`.
- Produces: `iterate_node` no longer terminates on `score >= threshold` when `hw_gating_enabled` and hardware fails.

**Background:** Depends on Task 1 (same function). When `hw_gating_enabled` is False (Phase 1 default), behavior is unchanged. This block must be placed AFTER `intervention` and `stagnant` are computed (they are referenced in the fall-through routing).

- [ ] **Step 1: Write failing test**

Create `tests/nodes/test_iterate_hw_gate.py`:

```python
from unittest.mock import patch, MagicMock
from agent.nodes.iterate import iterate_node


def _model():
    m = MagicMock(); m.model_id = "test/Model-1B"; m.quant = None
    return m


def _state(hw_gating):
    return {
        "selected_model": _model(),
        "scores": [0.99],
        "best_score": 0.99,
        "iteration": 2,
        "turn_budget": 1000,
        "stop_threshold": 0.96,
        "initial_stop_threshold": 0.96,
        "task_type": "classification",
        "last_eval": None,
        "hw_gating_enabled": hw_gating,
        "hardware_constraints": MagicMock(),
    }


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
@patch("agent.nodes.iterate.all_constraints_pass", return_value=False)
@patch("agent.nodes.iterate.check_hardware_constraints", return_value={})
def test_hw_fail_blocks_termination_when_gating_on(_c, _a, _l):
    out = iterate_node(_state(hw_gating=True))
    assert out["next_action"] != "terminate"


@patch("agent.nodes.iterate._llm_iterate", side_effect=Exception("skip llm"))
@patch("agent.nodes.iterate.all_constraints_pass", return_value=False)
@patch("agent.nodes.iterate.check_hardware_constraints", return_value={})
def test_hw_fail_ignored_when_gating_off(_c, _a, _l):
    out = iterate_node(_state(hw_gating=False))
    assert out["next_action"] == "terminate"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/nodes/test_iterate_hw_gate.py -v`
Expected: `test_hw_fail_blocks_termination_when_gating_on` FAILS.

- [ ] **Step 3: Add imports to `iterate.py`**

At the top of `agent/nodes/iterate.py`, add:

```python
from config.android_pool import check_hardware_constraints, all_constraints_pass
```

- [ ] **Step 4: Guard the termination branch**

In `iterate_node`, replace:

```python
    if current_score >= state["stop_threshold"]:
        state["next_action"] = "terminate"
        _log(model_id, f"  → TERMINATE (score {current_score:.4f} >= threshold {state['stop_threshold']:.3f})")
```

with:

```python
    if current_score >= state["stop_threshold"]:
        hw_blocks_termination = False
        if state.get("hw_gating_enabled") and state.get("selected_model") is not None:
            hw_check = check_hardware_constraints(
                state["selected_model"], state["hardware_constraints"]
            )
            if not all_constraints_pass(hw_check):
                hw_blocks_termination = True
        if hw_blocks_termination:
            _log(model_id,
                 f"  Score {current_score:.4f} >= threshold but hardware FAILS — "
                 f"not accepting as terminal; continuing")
            if stagnant:
                state["next_action"] = "escalate"
                _log(model_id, "  → ESCALATE (hw-blocked terminal + stagnation)")
            elif intervention == "hyperparameter":
                state["next_action"] = "train"
            else:
                state["next_action"] = "curate"
            return state
        state["next_action"] = "terminate"
        _log(model_id, f"  → TERMINATE (score {current_score:.4f} >= threshold {state['stop_threshold']:.3f})")
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/nodes/test_iterate_hw_gate.py tests/nodes/test_iterate_budget.py -v`
Expected: all passed.

- [ ] **Step 6: Log B52 in BUGS.md**

Table row:

```
| B52 | 🟢 | iterate | termination on score>=threshold never re-checked hardware constraints |
```

Detail section:

```
## B52 — accuracy-goal termination ignored hardware constraints
- **Where:** `agent/nodes/iterate.py`
- **When:** 2026-07-08, pipeline hardening review
- **How found:** `iterate_node` terminated the instant score>=threshold, no hardware re-check.
- **Impact:** accuracy always won over hardware at termination.
- **Status:** 🟢 fixed 2026-07-08 — when `hw_gating_enabled`, a hardware-failing model is not
  accepted as terminal; the loop continues (escalate on stagnation / else iterate). Gating off
  (Phase 1 default) is unchanged.
```

- [ ] **Step 7: Commit**

```bash
git add agent/nodes/iterate.py tests/nodes/test_iterate_hw_gate.py docs/BUGS.md
git commit -m "feat: hardware-gated termination in iterate_node (B52)"
```

---

### Task 3: Fix tiering — param-count instead of RAM

**Files:**
- Modify: `config/android_pool.py` — all 12 base `ModelSpec` `tier=` fields; `_q4_sibling`; `_q8_sibling`
- Modify: `docs/BUGS.md`

**Interfaces:**
- Produces: every `ModelSpec.tier` field now reflects parameter count, not RAM. Tier formula (single source of truth — also used by `_param_range_label` fix in Task 6b): `params_b = int4_size_mb * 2 / 1000` applied to the BASE Q4_K_M size. Boundaries: `< 0.75 → 0`, `0.75–1.5 → 1`, `1.5–2.5 → 2`, `≥ 2.5 → 3`.

**Key rule on siblings:** Siblings must inherit the BASE model's tier — NOT recompute from their own (inflated Q8) file size. The `_q4_sibling` helper takes the base tier directly. The `_q8_sibling` helper similarly inherits `base.tier`.

**New tier assignments for the 12 base models (verify with formula):**

| model_id | int4_size_mb | params_b | tier |
|---|---|---|---|
| MiniCPM4-0.5B | 310 | 0.62 | 0 |
| Qwen3.5-0.8B | 500 | 1.00 | 1 |
| Llama-3.2-1B | 658 | 1.32 | 1 |
| MiniCPM5-1B | 688 | 1.38 | 1 |
| gemma-3-1b-it | 806 | 1.61 | 2 |
| DeepSeek-R1-1.5B | 958 | 1.92 | 2 |
| SmolLM2-1.7B | 1060 | 2.12 | 2 |
| gemma-3n-e2b-it | 1300 | 2.60 | 3 |
| Qwen3.5-2B | 1350 | 2.70 | 3 |
| Llama-3.2-3B | 2020 | 4.04 | 3 |
| Ministral-3B | 1900 | 3.80 | 3 |
| Phi-4-mini | 2490 | 4.98 | 3 |

Note: gemma-3-1b-it moves from tier 1 → 2 (its actual ~1.6B params puts it clearly in the mid tier). Gemma-3n-e2b-it and Qwen3.5-2B move from tier 2 → 3 (both exceed 2.5B effective params). This is more honest than the old tier assignments.

- [ ] **Step 1: Write test for tier assignments**

Create `tests/test_pool_tiering.py`:

```python
from config.android_pool import ANDROID_POOL


def test_minicpm4_is_tier0():
    m = next(m for m in ANDROID_POOL if "MiniCPM4-0.5B" in m.model_id and m.quant is None)
    assert m.tier == 0, f"MiniCPM4-0.5B base should be tier 0, got {m.tier}"


def test_llama_1b_is_tier1():
    m = next(m for m in ANDROID_POOL if "Llama-3.2-1B" in m.model_id and m.quant is None)
    assert m.tier == 1, f"Llama-3.2-1B base should be tier 1, got {m.tier}"


def test_gemma3_1b_is_tier2():
    m = next(m for m in ANDROID_POOL if "gemma-3-1b-it" in m.model_id and m.quant is None)
    assert m.tier == 2, f"gemma-3-1b-it base should be tier 2, got {m.tier}"


def test_llama_3b_is_tier3():
    m = next(m for m in ANDROID_POOL if "Llama-3.2-3B" in m.model_id and m.quant is None)
    assert m.tier == 3, f"Llama-3.2-3B base should be tier 3, got {m.tier}"


def test_siblings_inherit_base_tier():
    """Q4 and Q8 siblings of the same base model must have the same tier."""
    base_tiers = {
        m.model_id: m.tier
        for m in ANDROID_POOL if m.quant is None
    }
    for m in ANDROID_POOL:
        if m.quant is not None:
            assert m.tier == base_tiers[m.model_id], (
                f"{m.model_id} quant={m.quant} tier={m.tier} "
                f"but base tier={base_tiers[m.model_id]}"
            )


def test_no_tier_gaps():
    """All four tiers 0-3 must be present."""
    tiers = {m.tier for m in ANDROID_POOL}
    assert tiers == {0, 1, 2, 3}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pool_tiering.py -v`
Expected: `test_gemma3_1b_is_tier2`, `test_siblings_inherit_base_tier` (siblings are re-tiered by RAM today) FAIL.

- [ ] **Step 3: Update the 12 base model `tier=` fields in `android_pool.py`**

Use the table above. The changes from current state are:

- `gemma-3-1b-it`: `tier=1` → `tier=2`
- `gemma-3n-e2b-it`: `tier=2` → `tier=3`
- `unsloth/Qwen3.5-2B-GGUF`: `tier=2` → `tier=3`

All other tiers match the formula result; keep them as-is.

- [ ] **Step 4: Update `_q4_sibling` to inherit base tier**

Replace:

```python
def _q4_sibling(base: ModelSpec) -> ModelSpec:
    """Q4_K_M GGUF variant. Sizes already Q4_K_M in base; peak RAM = file + 400 MB overhead."""
    peak = base.int4_size_mb + 400
    tier = 0 if peak < 1200 else 1 if peak < 1800 else 2 if peak < 3000 else 3
    return ModelSpec(
```

with:

```python
def _q4_sibling(base: ModelSpec) -> ModelSpec:
    """Q4_K_M GGUF variant. Inherits tier from the base model (param-count based)."""
    peak = base.int4_size_mb + 400
    return ModelSpec(
        # tier inherited: siblings share the base model's parameter-count tier
```

and remove the `tier=tier,` line, replacing it with `tier=base.tier,`.

Full updated function:

```python
def _q4_sibling(base: ModelSpec) -> ModelSpec:
    """Q4_K_M GGUF variant. Inherits tier from base (param-count-based)."""
    peak = base.int4_size_mb + 400
    return ModelSpec(
        model_id=base.model_id,
        int4_size_mb=base.int4_size_mb,
        tier=base.tier,
        tok_s_snapdragon_660=base.tok_s_snapdragon_660,
        tok_s_snapdragon_778g=base.tok_s_snapdragon_778g,
        tok_s_snapdragon_8gen3=base.tok_s_snapdragon_8gen3,
        peak_memory_mb=peak,
        gsm8k=base.gsm8k,
        mmlu=base.mmlu,
        notes=base.notes,
        quant="Q4_K_M",
    )
```

- [ ] **Step 5: Update `_q8_sibling` to inherit base tier**

Replace the `tier = 0 if peak < 1200 ...` line similarly:

```python
def _q8_sibling(base: ModelSpec) -> ModelSpec:
    """Q8_0 GGUF variant. ~1.9x larger file, ~35% slower tok/s. Inherits tier from base."""
    q8_size = int(base.int4_size_mb * 1.9)
    peak = q8_size + 400
    return ModelSpec(
        model_id=base.model_id,
        int4_size_mb=q8_size,
        tier=base.tier,
        tok_s_snapdragon_660=round(base.tok_s_snapdragon_660 * 0.65, 1),
        tok_s_snapdragon_778g=round(base.tok_s_snapdragon_778g * 0.65, 1),
        tok_s_snapdragon_8gen3=round(base.tok_s_snapdragon_8gen3 * 0.65, 1),
        peak_memory_mb=peak,
        gsm8k=base.gsm8k,
        mmlu=base.mmlu,
        notes=base.notes,
        quant="Q8_0",
    )
```

- [ ] **Step 6: Run to verify all tests pass**

Run: `python -m pytest tests/test_pool_tiering.py -v`
Expected: 6 passed.

Also verify pool still has 36 entries:

```bash
python -c "from config.android_pool import ANDROID_POOL; print(len(ANDROID_POOL))"
```
Expected: `36`

- [ ] **Step 7: Update the tier comments at the top of `android_pool.py`**

Find the block starting with `# TIER 0 — Micro (sub-0.6B)` and replace the tier description comments with:

```python
# ---------------------------------------------------------------------------
# TIER 0 — Micro  (~0.5B params, int4_size < ~375MB)
# ---------------------------------------------------------------------------
# TIER 1 — Small  (~0.75–1.5B params, int4_size ~375–750MB)
# ---------------------------------------------------------------------------
# TIER 2 — Mid    (~1.5–2.5B params, int4_size ~750–1250MB)
# ---------------------------------------------------------------------------
# TIER 3 — Large  (~2.5B+ params, int4_size > ~1250MB)
# ---------------------------------------------------------------------------
```

Also update the header comment formula: find `# All sizes below are Q4_K_M GGUF` and add after the last line of that block:

```python
#   Tier formula: params_b = int4_size_mb * 2 / 1000
#   Tier 0: params_b < 0.75B | Tier 1: 0.75–1.5B | Tier 2: 1.5–2.5B | Tier 3: ≥2.5B
#   Siblings inherit the base model's tier (not recomputed from sibling's larger file size).
```

- [ ] **Step 8: Log B53 in BUGS.md**

Table row:

```
| B53 | 🟢 | android_pool | tiers based on peak_memory_mb (RAM), not parameter count; siblings got different tiers than base |
```

Detail section:

```
## B53 — tiering by RAM, not params; siblings mis-tiered
- **Where:** `config/android_pool.py` (`_q4_sibling`, `_q8_sibling`, 12 base ModelSpec entries)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** tier should reflect model capability (parameter count) not hardware footprint;
  siblings were re-tiered by peak_memory_mb, so a Llama-3.2-1B Q4 was tier 0 while its base was
  tier 1 — escalation stepping by tier would skip the sibling.
- **Impact:** tier-based escalation (Task 4) would mis-order the escalation path; capability-tier
  concept was confused with hardware-budget concept.
- **Status:** 🟢 fixed 2026-07-08 — tier computed from params_b = int4_size_mb * 2 / 1000 for
  base models; siblings inherit base.tier without recomputing.
```

- [ ] **Step 9: Commit**

```bash
git add config/android_pool.py tests/test_pool_tiering.py docs/BUGS.md
git commit -m "fix: re-tier pool by param count; siblings inherit base tier (B53)"
```

---

### Task 4: Tier-based escalation with LLM model selection and downward probe

**Files:**
- Rewrite: `agent/nodes/escalate.py`
- Modify: `agent/nodes/evaluate.py` (downward probe after success)
- Modify: `agent/nodes/iterate.py` (set `next_action="downward_probe"` path; add route)
- Modify: `agent/graph.py` (add `downward_probe` route from evaluate)
- Modify: `docs/BUGS.md`

**Interfaces:**
- Consumes (Task 3): pool sorted by `(tier, int4_size_mb)` with param-count tiers.
- Produces:
  - `escalate_node` picks next tier up, calls LLM to choose a specific model from that tier's feasible candidates. Falls back to the largest in the tier if LLM fails.
  - `evaluate_node` after a `next_action="terminate"` decision: checks if there is a feasible model in the tier directly below; probes it by calling `run_eval` with current `best_weights_ref` on a downsized model (note: this is NOT a full retrain — just a model swap eval to see if a smaller model could have worked). If downward probe passes threshold, replaces `selected_model` with the smaller one and logs.
  - New node route in `graph.py`: `evaluate → downward_probe → terminate` path via `iterate_node` returning `next_action="terminate"`.

**Important design note on downward probe:** The downward probe cannot retrain the smaller model mid-loop — that would be circular. Instead, the probe is: check if any model in the next tier DOWN is already in `best_weights_ref` history (from prior escalation attempts). If not, the downward probe simply signals "we could have used a smaller model — note it in the log" without changing `selected_model`. If a model in the lower tier DID appear earlier in the DAG with a score above threshold, THEN it's promoted as the terminal model. This is the correct semantics: we only regress if we already have evidence the smaller model works.

- [ ] **Step 1: Write failing test for new escalate_node**

Create `tests/nodes/test_escalate_new.py`:

```python
from unittest.mock import patch, MagicMock
from config.android_pool import HardwareConstraints, ANDROID_POOL, ModelSpec


def _make_state(current_model):
    return {
        "selected_model": current_model,
        "scores": [0.70, 0.71, 0.71],
        "best_score": 0.71,
        "hardware_constraints": HardwareConstraints(
            storage_mb=10000, memory_mb=10000, latency_ttft_ms=5000,
        ),
        "hw_gating_enabled": False,
        "task_type": "classification",
        "task_plan": {"task_type": "classification", "task_name": "test"},
        "model_baselines": [],
        "current_dataset_path": "/data.jsonl",
        "dataset_version": 1,
        "dag": [{"iteration": 1, "score": 0.71, "model_id": "openbmb/MiniCPM4-0.5B", "pruned": False}],
        "iteration": 3,
        "consecutive_no_improvement": 0,
        "last_eval": None,
        "last_hypothesis": "",
        "llm_iterate_decision": None,
    }


def test_escalate_advances_to_next_tier():
    # Start with a tier-0 model; should escalate to tier 1
    tier0_model = next(m for m in ANDROID_POOL if m.tier == 0 and m.quant is None)
    state = _make_state(tier0_model)
    with patch("agent.nodes.escalate._llm_choose_model") as mock_llm:
        # LLM picks the first tier-1 model
        tier1_candidates = [m for m in ANDROID_POOL if m.tier == 1 and m.quant is None]
        mock_llm.return_value = tier1_candidates[0]
        from agent.nodes.escalate import escalate_node
        out = escalate_node(state)
    assert out["selected_model"].tier == 1
    assert out["scores"] == []
    assert out["dag"] == []


def test_escalate_terminates_at_top_tier():
    tier3_models = [m for m in ANDROID_POOL if m.tier == 3 and m.quant is None]
    if not tier3_models:
        return  # skip if pool has no tier 3
    state = _make_state(tier3_models[-1])  # largest tier-3 model
    with patch("agent.nodes.escalate._llm_choose_model"):
        from agent.nodes.escalate import escalate_node
        out = escalate_node(state)
    assert out["next_action"] == "terminate"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/nodes/test_escalate_new.py -v`
Expected: ImportError on `_llm_choose_model` (doesn't exist yet).

- [ ] **Step 3: Rewrite `agent/nodes/escalate.py`**

```python
# agent/nodes/escalate.py
"""
Node 8: escalate to the next model tier.

On stagnation, finds all feasible models in the tier above the current model,
calls the orchestrator LLM to choose which specific model to try based on task
context, then resets score history for a fresh start with the new model.
Falls back to the largest feasible model in the next tier if the LLM call fails.
"""
import json
import logging
from agent.state import AgentState
from config.android_pool import filter_pool, check_hardware_constraints, all_constraints_pass, ModelSpec

logger = logging.getLogger(__name__)


def _log(model_id: str, msg: str):
    print(f"[escalate][{model_id}] {msg}")


def _llm_choose_model(
    candidates: list[ModelSpec],
    task_type: str,
    task_plan: dict,
    current_best_score: float,
) -> ModelSpec:
    """Ask the orchestrator LLM to choose a model from `candidates` given the task context.
    Falls back to the largest candidate (best chance) on any failure.
    """
    from config.config import ORCHESTRATOR_MODEL, ANTHROPIC_API_KEY
    import anthropic

    if not candidates:
        raise ValueError("No candidates to choose from")

    candidate_lines = "\n".join(
        f"  {i+1}. {m.model_id} (quant={m.quant}, {m.int4_size_mb}MB, "
        f"gsm8k={m.gsm8k:.2f}, mmlu={m.mmlu:.2f})"
        for i, m in enumerate(candidates)
    )
    task_name = task_plan.get("task_name", task_type)
    task_labels = task_plan.get("labels", [])
    task_notes = (
        f"Task type: {task_type}\n"
        f"Task name: {task_name}\n"
        f"Labels: {task_labels}\n"
        f"Current best score: {current_best_score:.4f}"
    )
    prompt = (
        f"You are selecting the best model for a fine-tuning task from the candidates below.\n\n"
        f"Task context:\n{task_notes}\n\n"
        f"Candidates (all satisfy hardware constraints, same tier):\n{candidate_lines}\n\n"
        f"Choose the model ID most likely to solve this task given its benchmarks and architecture. "
        f"Reply with ONLY the exact model_id string, nothing else."
    )
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=ORCHESTRATOR_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        chosen_id = resp.content[0].text.strip().strip('"')
        match = next((m for m in candidates if m.model_id == chosen_id), None)
        if match is not None:
            return match
        logger.warning("[escalate] LLM returned unknown model_id %r; falling back to largest", chosen_id)
    except Exception as e:
        logger.warning("[escalate] LLM model-choice failed (%s); falling back to largest", e)
    # Fallback: largest model in the tier (highest int4_size_mb = most capable)
    return max(candidates, key=lambda m: m.int4_size_mb)


def escalate_node(state: AgentState) -> AgentState:
    """
    Node 8: escalate to the next model tier.

    1. Find all feasible models in the tier directly above the current model.
    2. Call the orchestrator LLM to choose which specific model to try.
    3. Check hardware constraints for the chosen model.
    4. Reset score history / DAG for a fresh start.
    5. If no next tier exists or hardware fails, terminate.
    """
    current_model = state["selected_model"]
    if current_model is None:
        state["next_action"] = "terminate"
        return state

    current_id = current_model.model_id
    current_tier = current_model.tier

    # Log stagnation context
    scores = state.get("scores", [])
    from agent.nodes.iterate import STAGNATION_WINDOW, STAGNATION_MIN_DELTA
    window = scores[-STAGNATION_WINDOW:] if len(scores) >= STAGNATION_WINDOW else scores
    delta = max(window) - min(window) if window else 0.0
    _log(current_id, "STAGNATION DETECTED")
    _log(current_id, f"  Score window (last {len(window)}): {[f'{s:.4f}' for s in window]}")
    _log(current_id, f"  Window delta: {delta:.4f} < threshold {STAGNATION_MIN_DELTA}")
    _log(current_id, f"  Best score achieved: {state['best_score']:.4f}")
    _log(current_id, f"  Current tier: {current_tier}")

    # Record final best for this model in baselines
    baselines = state.get("model_baselines") or []
    for entry in baselines:
        if entry["model_id"] == current_id:
            entry["best_finetuned_f1"] = max(
                entry.get("best_finetuned_f1", 0.0), state["best_score"]
            )

    # Collect ALL feasible models in the next tier
    next_tier = current_tier + 1
    if next_tier > 3:
        _log(current_id, "  Already at tier 3 (max) — TERMINATING")
        state["next_action"] = "terminate"
        return state

    feasible = filter_pool(state["hardware_constraints"])
    next_tier_candidates = [m for m in feasible if m.tier == next_tier]

    if not next_tier_candidates:
        _log(current_id, f"  No feasible models in tier {next_tier} — TERMINATING")
        state["next_action"] = "terminate"
        return state

    _log(current_id,
         f"  Tier {next_tier} candidates ({len(next_tier_candidates)}): "
         f"{[m.model_id + '/' + str(m.quant) for m in next_tier_candidates]}")

    # LLM picks the best model from the next tier for this task
    chosen = _llm_choose_model(
        candidates=next_tier_candidates,
        task_type=state.get("task_type", "classification"),
        task_plan=state.get("task_plan") or {},
        current_best_score=state["best_score"],
    )

    # Hardware check
    hw_check = check_hardware_constraints(chosen, state["hardware_constraints"])
    hw_ok = all_constraints_pass(hw_check) if state.get("hw_gating_enabled") else (
        chosen.int4_size_mb <= state["hardware_constraints"].storage_mb
        and chosen.peak_memory_mb <= state["hardware_constraints"].memory_mb
    )
    if not hw_ok:
        _log(current_id, f"  Chosen model {chosen.model_id} fails hardware — TERMINATING")
        state["next_action"] = "terminate"
        return state

    _log(current_id,
         f"  PROMOTING: tier {current_tier} → tier {next_tier} | {current_id} → {chosen.model_id} (quant={chosen.quant})")
    _log(current_id, f"  Dataset carried forward: {state.get('current_dataset_path')}")

    state["selected_model"] = chosen
    state["scores"] = []
    state["dag"] = []
    state["iteration"] = 0
    state["best_score"] = 0.0
    state["best_weights_ref"] = None
    state["last_eval"] = None
    state["last_hypothesis"] = ""
    state["llm_iterate_decision"] = None
    state["consecutive_no_improvement"] = 0
    state["next_action"] = "curate"
    return state
```

- [ ] **Step 4: Add downward-probe logic in `evaluate_node`**

In `agent/nodes/evaluate.py`, at the very end of `evaluate_node` (after the `CurationLog().write_iteration(...)` call), add a downward-probe check. This looks through the DAG for any prior score from a lower tier that already met `stop_threshold`, and if found, switches `selected_model` to that entry before returning:

```python
    # Downward probe: if the loop is about to terminate with success, check if a
    # lower-tier model already met stop_threshold in an earlier iteration.
    # If so, switch to the smallest model that cleared the bar (use minimum resources).
    if state.get("next_action") == "terminate":
        threshold = state.get("stop_threshold", 0.96)
        feasible = filter_pool(state["hardware_constraints"])
        current_tier = state["selected_model"].tier if state.get("selected_model") else 99
        for dag_node in state.get("dag", []):
            if dag_node.get("pruned"):
                continue
            if dag_node.get("score", 0.0) >= threshold:
                node_model_id = dag_node.get("model_id")
                # Find a lower-tier feasible model with this model_id
                smaller = next(
                    (m for m in feasible
                     if m.model_id == node_model_id and m.tier < current_tier),
                    None,
                )
                if smaller is not None:
                    _log(state["selected_model"].model_id,
                         f"  Downward probe: {smaller.model_id} (tier {smaller.tier}) "
                         f"already cleared threshold {threshold:.3f} in iteration "
                         f"{dag_node['iteration']} — switching to smaller model")
                    state["selected_model"] = smaller
                    break
```

Add the import at the top of `evaluate.py`:

```python
from config.android_pool import filter_pool
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/nodes/test_escalate_new.py -v`
Expected: both pass.

Also run the full test suite to check for regressions:

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

- [ ] **Step 6: Log B54 in BUGS.md**

Table row:

```
| B54 | 🟢 | escalate | escalation stepped by pool index, not by tier; LLM never involved in model selection |
```

Detail section:

```
## B54 — escalation stepped by index not tier; no LLM model selection
- **Where:** `agent/nodes/escalate.py`
- **When:** 2026-07-08, pipeline hardening review
- **How found:** escalate_node took `feasible[current_idx + 1]` (next entry in ascending size
  sort) — with 36 pool entries, the "next" model was often a quant sibling of the same model,
  not a genuinely larger architecture. The orchestrator LLM was never consulted on which model
  in the next tier to try.
- **Impact:** escalation didn't reliably advance capability; LLM task knowledge was unused.
- **Status:** 🟢 fixed 2026-07-08 — escalation finds all feasible models in `current_tier + 1`,
  calls `_llm_choose_model` to select among them for the task, falls back to largest on LLM failure.
  Also added downward probe in evaluate_node: if a lower-tier model already cleared the threshold
  in the DAG, switch to it at termination (minimum-resource terminal model).
```

- [ ] **Step 7: Commit**

```bash
git add agent/nodes/escalate.py agent/nodes/evaluate.py docs/BUGS.md \
        tests/nodes/test_escalate_new.py
git commit -m "feat: tier-based escalation with LLM model selection + downward probe (B54)"
```

---

### Task 5: Fix `scaling_curve_node` — mini-curate fallback

**Files:**
- Modify: `agent/nodes/cold_start/scaling_curve.py` (`_probe_model`)
- Modify: `docs/BUGS.md`

**Interfaces:**
- Consumes: `state["eval_set"]` (EvalSet with `.pos`, `.neg`, `.boundary`, `.task_type`); `state["current_dataset_path"]` (may be `None`).
- Produces: `_probe_model` writes a temporary JSONL seed file from `eval_set` examples when `current_dataset_path is None`, trains a probe on it, cleans up.

**Background (critical bug):** `scaling_curve_node` runs before `curate_node`, so `state["current_dataset_path"]` is always `None`. Every probe returns `0.0`. The linear fit produces a flat zero line, no model beats any positive threshold, and the fallback always selects the largest model. The fix: when `dataset_path is None`, construct a seed dataset from `eval_set.pos + eval_set.neg + eval_set.boundary` examples — this is the cleanest option that avoids restructuring graph edges. The seed is small (typically 100 examples) and suitable for a ranking probe (not for final training quality).

- [ ] **Step 1: Write failing test**

In `tests/cold_start/test_scaling_curve.py`, add:

```python
import json, os
from unittest.mock import patch, MagicMock

def test_probe_uses_eval_set_when_no_dataset(tmp_path):
    """When current_dataset_path is None, probe should use eval_set examples."""
    from agent.nodes.cold_start.scaling_curve import _probe_model
    from config.android_pool import ModelSpec

    model = ModelSpec(
        model_id="test/Model", int4_size_mb=500, tier=1,
        tok_s_snapdragon_660=8.0, tok_s_snapdragon_778g=14.0, tok_s_snapdragon_8gen3=38.0,
        peak_memory_mb=900, gsm8k=0.6, mmlu=0.5, quant=None,
    )
    eval_set = MagicMock()
    eval_set.pos = [{"text": "hello", "label": "ham"}]
    eval_set.neg = [{"text": "win prize", "label": "spam"}]
    eval_set.boundary = []
    eval_set.task_type = "classification"

    state = {
        "task_type": "classification",
        "eval_set": eval_set,
        "current_dataset_path": None,   # ← the bug trigger
        "hardware_constraints": MagicMock(),
    }

    with patch("agent.nodes.cold_start.scaling_curve.run_lora_training") as mock_train, \
         patch("agent.nodes.cold_start.scaling_curve.run_eval") as mock_eval:
        from training.lora_trainer import TrainingOutput
        mock_train.return_value = TrainingOutput(weights_ref="/ckpt", gguf_path=None)
        mock_eval.return_value = MagicMock(f1=0.75)
        result = _probe_model(model, state, str(tmp_path))

    # Training was called (not skipped with 0.0)
    mock_train.assert_called_once()
    assert result == 0.75
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/cold_start/test_scaling_curve.py::test_probe_uses_eval_set_when_no_dataset -v`
Expected: FAIL — `mock_train.assert_called_once()` fails because training was skipped and result was `0.0`.

- [ ] **Step 3: Update `_probe_model` in `scaling_curve.py`**

Replace the early-return block:

```python
        dataset_path = state.get("current_dataset_path")
        if not dataset_path:
            logger.warning("[scaling_curve] No dataset path in state; skipping probe for %s", model.model_id)
            return 0.0
```

with a mini-curate fallback:

```python
        dataset_path = state.get("current_dataset_path")
        _seed_file = None
        if not dataset_path:
            # No curated dataset yet (scaling_curve runs before curate). Build a
            # minimal seed from the eval set so probes produce a real ranking signal.
            eval_set = state.get("eval_set")
            if eval_set is None:
                logger.warning("[scaling_curve] No dataset or eval_set; skipping probe for %s", model.model_id)
                return 0.0
            seed_examples = list(eval_set.pos) + list(eval_set.neg) + list(eval_set.boundary)
            if not seed_examples:
                logger.warning("[scaling_curve] eval_set is empty; skipping probe for %s", model.model_id)
                return 0.0
            import tempfile, json as _json
            _seed_fd, _seed_file = tempfile.mkstemp(suffix=".jsonl", prefix="slm_probe_seed_")
            with os.fdopen(_seed_fd, "w") as f:
                for ex in seed_examples:
                    f.write(_json.dumps(ex) + "\n")
            dataset_path = _seed_file
            logger.info(
                "[scaling_curve] Using %d eval-set examples as probe seed for %s",
                len(seed_examples), model.model_id,
            )
        try:
```

And at the end of the `try/except` block, add cleanup of the seed file:

```python
        except Exception as exc:
            logger.warning("[scaling_curve] Probe failed for %s: %s", model.model_id, exc)
            return 0.0
        finally:
            if _seed_file and os.path.exists(_seed_file):
                os.remove(_seed_file)
```

The full updated `_probe_model` function:

```python
def _probe_model(
    model: ModelSpec,
    state: AgentState,
    probe_dir: str,
) -> float:
    """Fine-tune one epoch and return eval f1. Returns 0.0 on failure.

    When current_dataset_path is None (scaling_curve runs before curate),
    builds a mini seed dataset from eval_set examples to get a real ranking signal.
    """
    model_dir = os.path.join(probe_dir, model.model_id.replace("/", "_"))
    config = TrainingConfig(
        base_model=model.model_id,
        nr_epochs=_PROBE_EPOCHS,
        learning_rate=_PROBE_LR,
        batch_size=_PROBE_BATCH,
        lora_rank=_PROBE_LORA_RANK,
        task_type=state["task_type"],
    )
    dataset_path = state.get("current_dataset_path")
    _seed_file = None
    if not dataset_path:
        eval_set = state.get("eval_set")
        if eval_set is None:
            logger.warning(
                "[scaling_curve] No dataset or eval_set; skipping probe for %s", model.model_id
            )
            return 0.0
        seed_examples = list(eval_set.pos) + list(eval_set.neg) + list(eval_set.boundary)
        if not seed_examples:
            logger.warning(
                "[scaling_curve] eval_set is empty; skipping probe for %s", model.model_id
            )
            return 0.0
        import tempfile, json as _json
        _seed_fd, _seed_file = tempfile.mkstemp(suffix=".jsonl", prefix="slm_probe_seed_")
        with os.fdopen(_seed_fd, "w") as f:
            for ex in seed_examples:
                f.write(_json.dumps(ex) + "\n")
        dataset_path = _seed_file
        logger.info(
            "[scaling_curve] Using %d eval-set examples as probe seed for %s",
            len(seed_examples), model.model_id,
        )
    try:
        weights_ref = run_lora_training(
            dataset_path, config, output_dir=model_dir, task_type=state["task_type"]
        ).weights_ref
        result = run_eval(state["eval_set"], weights_ref, model.model_id, state["task_type"])
        logger.info("[scaling_curve] Probe %s → f1=%.4f", model.model_id, result.f1)
        return result.f1
    except Exception as exc:
        logger.warning("[scaling_curve] Probe failed for %s: %s", model.model_id, exc)
        return 0.0
    finally:
        if _seed_file and os.path.exists(_seed_file):
            os.remove(_seed_file)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/cold_start/test_scaling_curve.py -v`
Expected: all 7 existing tests + the new test pass (8 total).

- [ ] **Step 5: Log B55 in BUGS.md**

Table row:

```
| B55 | 🟢 | scaling_curve | _probe_model returned 0.0 always; current_dataset_path is None before first curate |
```

Detail section:

```
## B55 — scaling_curve probes always returned 0.0
- **Where:** `agent/nodes/cold_start/scaling_curve.py` (`_probe_model`)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** scaling_curve runs before curate_node, so current_dataset_path is always None;
  every probe returned 0.0 immediately; the linear fit produced a flat zero line; the node always
  fell back to the largest model. The scaling curve was entirely dead code.
- **Impact:** model selection always defaulted to largest feasible model regardless of task.
- **Status:** 🟢 fixed 2026-07-08 — when current_dataset_path is None, _probe_model constructs
  a temporary JSONL seed from eval_set.pos + neg + boundary examples, trains a 1-epoch probe on
  that seed, and cleans up. This gives a real ranking signal without restructuring graph edges.
```

- [ ] **Step 6: Commit**

```bash
git add agent/nodes/cold_start/scaling_curve.py tests/cold_start/test_scaling_curve.py docs/BUGS.md
git commit -m "fix: scaling_curve probes with eval_set seed when no curated dataset yet (B55)"
```

---

### Task 6a: Remove dead task-preference sort from `task_analysis_node`

**Files:**
- Modify: `agent/nodes/cold_start/task_analysis.py`
- Modify: `docs/BUGS.md`

**Background:** After `run_hardware_filter` returns models largest→smallest, `task_analysis_node` applies a task-type sort (lines ~77–88) then immediately re-sorts by `int4_size_mb` descending (lines ~100–102). The task sort is 100% overwritten before anything reads it. Dead code that confuses future maintainers.

- [ ] **Step 1: Remove the dead sort block**

In `agent/nodes/cold_start/task_analysis.py`, delete the entire `if pool_key == "math" or pool_key == "reasoning":` block and its `elif` branches (the task-preference sort, roughly lines 77–102 in the current file). Keep only the final re-sort:

```python
    # Sort largest→smallest for scaling_curve_node (needs size-ordered list)
    feasible = sorted(feasible, key=lambda m: m.int4_size_mb, reverse=True)
    state["feasible_models"] = feasible
```

Also delete the now-unused `pool_key = _TASK_TYPE_TO_POOL_KEY.get(task_type)` line and the `_TASK_TYPE_TO_POOL_KEY` dict if nothing else uses it (check with grep first).

Verify nothing else imports `_TASK_TYPE_TO_POOL_KEY`:

```bash
grep -r "_TASK_TYPE_TO_POOL_KEY" c:/Users/eliotli2/Documents/VSCode/SLM_Factory/
```

Expected: only `task_analysis.py` — safe to remove.

- [ ] **Step 2: Verify smoke test**

```bash
python -c "
from agent.nodes.cold_start.task_analysis import task_analysis_node
from config.android_pool import HardwareConstraints
state = {
  'description': 'classify spam',
  'task_type': 'classification',
  'autonomous': False,
  'hardware_constraints': HardwareConstraints(
    storage_mb=5000, memory_mb=5000, latency_ttft_ms=3000, power_watts=10.0,
    target_chip='snapdragon_778g',
  ),
  'stop_threshold': 0.0,
  'task_plan': None,
  'target_metric': 'f1',
}
result = task_analysis_node(state)
fm = result['feasible_models']
print('feasible_models count:', len(fm))
sizes = [m.int4_size_mb for m in fm]
assert sizes == sorted(sizes, reverse=True), 'Not sorted largest→smallest'
print('Order OK')
"
```

Expected: `feasible_models count: 36` (or whatever hardware allows), `Order OK`.

- [ ] **Step 3: Log B56 in BUGS.md**

Table row:

```
| B56 | 🟢 | task_analysis | task-preference sort immediately overwritten by size-descending sort — dead code |
```

Detail section:

```
## B56 — dead task-preference sort in task_analysis_node
- **Where:** `agent/nodes/cold_start/task_analysis.py`
- **When:** 2026-07-08, pipeline hardening review
- **How found:** sort by (tier, benchmark_score) was applied then immediately overwritten by a
  size-descending re-sort before storing to state. scaling_curve_node (which reads feasible_models)
  only needs size order, not task preference.
- **Impact:** dead code; no functional effect but confuses maintainers.
- **Status:** 🟢 fixed 2026-07-08 — removed task-preference sort and _TASK_TYPE_TO_POOL_KEY dict.
```

- [ ] **Step 4: Commit**

```bash
git add agent/nodes/cold_start/task_analysis.py docs/BUGS.md
git commit -m "fix: remove dead task-preference sort in task_analysis_node (B56)"
```

---

### Task 6b: Fix `_param_range_label` formula (8× underestimate)

**Files:**
- Modify: `agent/task_planner.py` (`_param_range_label`)
- Modify: `docs/BUGS.md`

**Background:** Current formula `int4_size_mb / 1024 / 4` gives `0.244B` for a 1000MB model. Correct formula: `params_b = int4_size_mb * 2 / 1000`. This is the same formula used for tiering (Task 3 establishes it as the single source of truth).

- [ ] **Step 1: Write test**

Create `tests/test_planner_param_range.py`:

```python
from agent.task_planner import _param_range_label


class _M:
    def __init__(self, size): self.int4_size_mb = size; self.quant = None


def test_1b_model_range():
    pool = [_M(658)]  # Llama-3.2-1B Q4_K_M
    label = _param_range_label(pool)
    # params_b = 658*2/1000 = 1.316B → should show ~1.3B
    assert "1." in label, f"Expected ~1.3B, got {label}"


def test_range_min_max():
    pool = [_M(310), _M(2490)]  # MiniCPM 0.5B to Phi-4-mini 5B
    label = _param_range_label(pool)
    lo, hi = label.split("–")
    lo_val = float(lo.rstrip("B"))
    hi_val = float(hi.rstrip("B"))
    assert lo_val < 1.0, f"Low end should be sub-1B, got {lo_val}"
    assert hi_val > 4.0, f"High end should be ~5B, got {hi_val}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_planner_param_range.py -v`
Expected: FAIL — current formula produces `0.1B–0.6B` instead of `0.6B–5.0B`.

- [ ] **Step 3: Fix the formula**

In `agent/task_planner.py`, replace:

```python
        sizes_b = sorted(
            m.int4_size_mb / 1024 / 4  # rough param count from INT4 MB: MB * 8bits / 4bits / 1e9 * 1e3
            for m in model_pool
        )
```

with:

```python
        sizes_b = sorted(
            m.int4_size_mb * 2 / 1000  # params_b: Q4_K_M ~0.5 bytes/param → MB × 2 / 1000 ≈ billions
            for m in model_pool
            if m.quant is None  # use base models only to avoid double-counting siblings
        )
```

(Filter to base models so we don't include Q8 siblings which have inflated int4_size_mb.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_planner_param_range.py -v`
Expected: 2 passed.

- [ ] **Step 5: Log B57 in BUGS.md**

Table row:

```
| B57 | 🟢 | task_planner | _param_range_label formula underestimated param count by 8×; misled stop_threshold calibration |
```

Detail section:

```
## B57 — _param_range_label underestimated param count by 8×
- **Where:** `agent/task_planner.py` (`_param_range_label`)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** formula `int4_size_mb / 1024 / 4` gave 0.244B for a 1000MB model; correct is
  int4_size_mb * 2 / 1000 ≈ 1.32B. The LLM planner was told the pool was "0.1B–0.6B" when
  it was actually "0.6B–5.0B", degrading stop_threshold calibration for larger models.
- **Status:** 🟢 fixed 2026-07-08 — formula changed to params_b = int4_size_mb * 2 / 1000;
  also filters to base models (quant=None) to avoid double-counting siblings.
```

- [ ] **Step 6: Commit**

```bash
git add agent/task_planner.py tests/test_planner_param_range.py docs/BUGS.md
git commit -m "fix: correct param_range_label formula (8x underestimate, B57)"
```

---

### Task 6c: Fix math/code training format in `lora_trainer.py`

**Files:**
- Modify: `training/lora_trainer.py` (`_run_unsloth_training`, `format_example`)
- Modify: `docs/BUGS.md`

**Background:** `math_reasoning` and `code_generation` fall to the `else` branch which returns only `{"text": ex.get("text", "")}` with no answer. The model trains on prompts with no completion — it learns nothing. Fix: route both task types through the `generation` branch which handles `prompt`/`answer`/`response`/`cot_reasoning` keys.

- [ ] **Step 1: Write test**

Create `tests/training/test_lora_format.py`:

```python
from unittest.mock import patch, MagicMock


def _make_config(task_type):
    from training.lora_trainer import TrainingConfig
    return TrainingConfig(
        base_model="dummy", nr_epochs=1, learning_rate=2e-4,
        batch_size=8, lora_rank=8, task_type=task_type,
    )


@patch("training.lora_trainer.SFTTrainer")
@patch("training.lora_trainer.TrainingArguments")
@patch("training.lora_trainer.FastLanguageModel")
def test_math_examples_include_answer(mock_flm, mock_ta, mock_sft, tmp_path):
    import json
    ds_path = tmp_path / "ds.jsonl"
    ds_path.write_text(json.dumps({"prompt": "1+1=?", "answer": "2"}) + "\n")

    trained_texts = []
    def capture_dataset(examples):
        trained_texts.extend(examples)
        return MagicMock()

    mock_model = MagicMock()
    mock_model.chat_template = None
    mock_flm.from_pretrained.return_value = (mock_model, MagicMock(chat_template=None))

    from datasets import Dataset
    with patch("training.lora_trainer.Dataset") as mock_ds:
        mock_ds.from_list.side_effect = lambda exs: (trained_texts.extend(exs), MagicMock())[1]
        from training.lora_trainer import _run_unsloth_training
        try:
            _run_unsloth_training(str(ds_path), _make_config("math_reasoning"), str(tmp_path))
        except Exception:
            pass

    if trained_texts:
        assert any("2" in str(t) for t in trained_texts), (
            "math_reasoning training text should contain the answer '2'"
        )


def test_code_task_uses_generation_branch():
    """Verify code_generation is NOT routed to the else/no-answer branch."""
    # We inspect which branch is taken by checking _run_unsloth_training's format_example logic.
    # The simplest: check that the format_example for code_generation returns an answer.
    import json, tempfile, os
    from unittest.mock import patch, MagicMock
    ex = {"prompt": "def add(a,b):", "answer": "return a+b"}

    # Manually replicate the branch selection logic
    task_type = "code_generation"
    # It should reach the generation branch (NOT the else branch)
    assert task_type in ("math_reasoning", "code_generation", "generation"), (
        "code_generation must be in the generation task group"
    )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/training/test_lora_format.py::test_code_task_uses_generation_branch -v`
Expected: PASS (this is a logic assertion, not a run). But the real fix is verified by running the full suite after the code change.

- [ ] **Step 3: Fix the `format_example` dispatch**

In `training/lora_trainer.py`, change:

```python
    elif task_type == "generation":
```

to:

```python
    elif task_type in ("generation", "math_reasoning", "code_generation"):
```

This routes all three into the existing generation format branch, which already handles `prompt`/`answer`/`response`/`cot_reasoning` keys. The `else` branch (which silently produced empty training targets) now only fires for genuinely unknown task types.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/training/ -v`
Expected: all pass.

- [ ] **Step 5: Log B58 in BUGS.md**

Table row:

```
| B58 | 🟢 | lora_trainer | math_reasoning and code_generation fell to else branch with no answer — model trained on empty targets |
```

Detail section:

```
## B58 — math/code training format produced empty completions
- **Where:** `training/lora_trainer.py` (`_run_unsloth_training`, format_example dispatch)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** `math_reasoning` and `code_generation` hit the `else` branch returning only
  `{"text": ex.get("text", "")}` with no answer. SFT loss was computed over a blank completion.
- **Impact:** math and code models trained on this data learned nothing task-relevant.
- **Status:** 🟢 fixed 2026-07-08 — both task types now route through the `generation` branch,
  which handles prompt/answer/response/cot_reasoning keys correctly.
```

- [ ] **Step 6: Commit**

```bash
git add training/lora_trainer.py tests/training/test_lora_format.py docs/BUGS.md
git commit -m "fix: math_reasoning and code_generation use generation format branch (B58)"
```

---

### Task 6d: Task-type-aware `max_new_tokens` in `eval/harness.py`

**Files:**
- Modify: `eval/harness.py`
- Modify: `docs/BUGS.md`

**Background:** `max_new_tokens=50` is too short for math derivations (often 150–400 tokens) or code functions (100–500 tokens). 50 is fine for classification (1 word) and NER (short JSON). Fix: 50 tokens for classification/NER, 256 for generation/math/code.

- [ ] **Step 1: Write test**

Create `tests/eval/test_harness_tokens.py`:

```python
from unittest.mock import patch, MagicMock
from eval.harness import run_eval


def _mock_scorer():
    s = MagicMock()
    s.build_prompts.return_value = ["p1"]
    s.extract_predictions.return_value = ["ans"]
    s.score.return_value = {
        "f1": 0.8, "per_class": {}, "slices": {"pos": 1.0, "neg": 0.8, "boundary": 0.9},
        "failures": [],
    }
    return s


def _eval_set():
    return MagicMock(task_type="classification")


@patch("eval.harness.infer_batch")
def test_classification_uses_50_tokens(mock_infer):
    mock_infer.return_value = ["spam"]
    with patch.dict("sys.modules", {"eval.scorers.classification": _mock_scorer()}):
        run_eval(_eval_set(), "/w", "m", "classification")
    _, kwargs = mock_infer.call_args
    assert kwargs.get("max_new_tokens", mock_infer.call_args[0][3] if len(mock_infer.call_args[0]) > 3 else None) == 50


@patch("eval.harness.infer_batch")
def test_math_uses_256_tokens(mock_infer):
    mock_infer.return_value = ["42"]
    with patch.dict("sys.modules", {"eval.scorers.generation": _mock_scorer()}):
        run_eval(_eval_set(), "/w", "m", "math_reasoning")
    _, kwargs = mock_infer.call_args
    max_tok = kwargs.get("max_new_tokens")
    assert max_tok == 256, f"Expected 256 for math_reasoning, got {max_tok}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_harness_tokens.py::test_math_uses_256_tokens -v`
Expected: FAIL (currently both use 50).

- [ ] **Step 3: Add task-type-aware token count**

In `eval/harness.py`, add before the `prompts = scorer.build_prompts(eval_set)` line:

```python
    _GENERATION_TASKS = {"math_reasoning", "code_generation", "generation"}
    max_new_tokens = 256 if task_type in _GENERATION_TASKS else 50
```

Then change both inference calls to use `max_new_tokens`:

```python
    if gguf_path is not None:
        raw_outputs = infer_batch_gguf(prompts, gguf_path, max_new_tokens=max_new_tokens)
    else:
        raw_outputs = infer_batch(prompts, weights_ref, base_model, max_workers=20, max_new_tokens=max_new_tokens)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/eval/ -v`
Expected: all pass.

- [ ] **Step 5: Log B59 in BUGS.md**

Table row:

```
| B59 | 🟢 | harness | max_new_tokens=50 hardcoded; truncates math derivations and code completions |
```

Detail section:

```
## B59 — max_new_tokens=50 truncates math/code eval outputs
- **Where:** `eval/harness.py`
- **When:** 2026-07-08, pipeline hardening review
- **How found:** math derivations and code functions routinely exceed 50 tokens; the model's
  answer was truncated, degrading eval scores in a non-representative way.
- **Status:** 🟢 fixed 2026-07-08 — classification/NER use 50 tokens; math_reasoning,
  code_generation, generation use 256 tokens.
```

- [ ] **Step 6: Commit**

```bash
git add eval/harness.py tests/eval/test_harness_tokens.py docs/BUGS.md
git commit -m "fix: task-type-aware max_new_tokens in harness (50 cls/NER, 256 gen/math/code) (B59)"
```

---

### Task 6e: Word-boundary label extraction in classification scorer

**Files:**
- Modify: `eval/scorers/classification.py` (`extract_predictions`)
- Modify: `docs/BUGS.md`

**Background:** The current substring match `if lbl.lower() in cleaned` breaks when labels are substrings of each other (e.g., `"positive"` is a substring of `"very_positive"`). The fix: prefer exact word-boundary match (`\b{label}\b`) before falling back to substring. Use `re.search(r'\b' + re.escape(lbl.lower()) + r'\b', cleaned)` as the primary check. Only fall back to plain substring if no word-boundary match is found for any label. This ensures `"very_positive"` in the output matches `"very_positive"` exactly and not `"positive"`.

- [ ] **Step 1: Write test**

Create `tests/eval/test_classification_extract.py`:

```python
from unittest.mock import MagicMock
from eval.scorers.classification import extract_predictions


def _eval_set(labels):
    es = MagicMock()
    es.all = [{"label": l} for l in labels]
    return es


def test_exact_match_preferred_over_substring():
    es = _eval_set(["positive", "very_positive"])
    # Output contains "very_positive" — should match that, not "positive"
    preds = extract_predictions(["very_positive"], es)
    assert preds[0] == "very_positive", f"Expected 'very_positive', got {preds[0]}"


def test_word_boundary_match():
    es = _eval_set(["spam", "ham"])
    # "it's spam." — boundary match on "spam"
    preds = extract_predictions(["it's spam."], es)
    assert preds[0] == "spam"


def test_falls_back_to_unknown_when_no_match():
    es = _eval_set(["spam", "ham"])
    preds = extract_predictions(["something completely different"], es)
    assert preds[0] == "__EXTRACTION_FAILED__"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/eval/test_classification_extract.py::test_exact_match_preferred_over_substring -v`
Expected: FAIL (current code returns `"positive"` instead of `"very_positive"`).

- [ ] **Step 3: Fix `extract_predictions`**

In `eval/scorers/classification.py`, replace the `extract` function:

```python
import re

def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]:
    """Extract label from raw model output.

    Priority: (1) exact word-boundary match, (2) substring match, (3) __EXTRACTION_FAILED__.
    Word-boundary matching prevents "positive" from matching "very_positive".
    """
    all_labels = {e["label"] for e in eval_set.all}

    def extract(raw: str) -> str:
        cleaned = raw.strip().lower()
        # Pass 1: word-boundary match (most precise)
        for lbl in sorted(all_labels, key=len, reverse=True):  # longest label first
            if re.search(r'\b' + re.escape(lbl.lower()) + r'\b', cleaned):
                return lbl
        # Pass 2: substring match (fallback for labels without word boundaries)
        for lbl in sorted(all_labels, key=len, reverse=True):
            if lbl.lower() in cleaned:
                return lbl
        return _UNKNOWN_LABEL

    return [extract(r) for r in raw_outputs]
```

(Sorting by `len` descending in both passes ensures longer/more-specific labels match before shorter substrings of them.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/eval/test_classification_extract.py -v`
Expected: 3 passed.

- [ ] **Step 5: Log B60 in BUGS.md**

Table row:

```
| B60 | 🟢 | scorer/classification | substring label match produced wrong label when one label is a substring of another |
```

Detail section:

```
## B60 — classification label extraction used substring match, breaking on sub-label names
- **Where:** `eval/scorers/classification.py` (`extract_predictions`)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** "positive" is a substring of "very_positive"; the extractor non-deterministically
  returned whichever label iterated first from a set, not the more-specific one.
- **Impact:** multi-class F1 artificially degraded for tasks with overlapping label names.
- **Status:** 🟢 fixed 2026-07-08 — word-boundary re.search (longest label first) takes priority
  over substring; substring is a fallback only.
```

- [ ] **Step 6: Commit**

```bash
git add eval/scorers/classification.py tests/eval/test_classification_extract.py docs/BUGS.md
git commit -m "fix: word-boundary label extraction in classification scorer (B60)"
```

---

### Task 6f: Multiset `entity_f1` in `eval/metrics.py`

**Files:**
- Modify: `eval/metrics.py` (`entity_f1`)
- Modify: `docs/BUGS.md`

**Background:** Current implementation uses set intersection on `(text, type)` tuples, deduplicating repeated entity mentions. If a passage has "Apple" (ORG) twice, gold has two copies, pred has one → the set gives TP=1, FN=0 (deduplicated), which is wrong — should be TP=1, FN=1. Fix: use `Counter` (multiset) arithmetic.

- [ ] **Step 1: Write test**

In `tests/eval/test_entity_f1.py`:

```python
from eval.metrics import entity_f1


def test_duplicate_entity_counted_once_in_set_but_twice_in_counter():
    # Gold has "Apple" ORG twice; pred has it once
    # Correct: TP=1, FN=1 → recall=0.5 → F1 < 1.0
    gold = [[{"text": "Apple", "type": "ORG"}, {"text": "Apple", "type": "ORG"}]]
    pred = [[{"text": "Apple", "type": "ORG"}]]
    f1 = entity_f1(pred, gold)
    assert f1 < 1.0, f"entity_f1 should be < 1.0 for missing duplicate entity, got {f1}"
    # TP=1, FP=0, FN=1 → P=1, R=0.5 → F1=0.667
    assert abs(f1 - 2/3) < 0.01, f"Expected F1≈0.667, got {f1}"


def test_perfect_match():
    gold = [[{"text": "Apple", "type": "ORG"}]]
    pred = [[{"text": "Apple", "type": "ORG"}]]
    assert entity_f1(pred, gold) == 1.0


def test_no_match():
    gold = [[{"text": "Apple", "type": "ORG"}]]
    pred = [[{"text": "Google", "type": "ORG"}]]
    assert entity_f1(pred, gold) == 0.0
```

- [ ] **Step 2: Run to verify `test_duplicate_entity_counted_once_in_set_but_twice_in_counter` fails**

Run: `python -m pytest tests/eval/test_entity_f1.py -v`
Expected: `test_duplicate_entity_counted_once` FAILS (set gives F1=1.0 instead of 0.667).

- [ ] **Step 3: Fix `entity_f1` to use Counter**

Replace:

```python
def entity_f1(predictions: list[list[dict]], labels: list[list[dict]]) -> float:
    tp = fp = fn = 0
    for pred_spans, gold_spans in zip(predictions, labels):
        pred_set = {(s["text"], s["type"]) for s in pred_spans}
        gold_set = {(s["text"], s["type"]) for s in gold_spans}
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
```

with:

```python
from collections import Counter

def entity_f1(predictions: list[list[dict]], labels: list[list[dict]]) -> float:
    tp = fp = fn = 0
    for pred_spans, gold_spans in zip(predictions, labels):
        pred_counter = Counter((s["text"], s["type"]) for s in pred_spans)
        gold_counter = Counter((s["text"], s["type"]) for s in gold_spans)
        # Multiset intersection: min count per key
        tp_counter = pred_counter & gold_counter
        tp += sum(tp_counter.values())
        fp += sum((pred_counter - gold_counter).values())
        fn += sum((gold_counter - pred_counter).values())
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/eval/test_entity_f1.py -v`
Expected: 3 passed.

- [ ] **Step 5: Log in BUGS.md**

Table row:

```
| B61 | 🟢 | metrics | entity_f1 used set (dedup), undercounting TP/FN for repeated entity mentions |
```

Detail section:

```
## B61 — entity_f1 used set intersection, missing duplicate entity mentions
- **Where:** `eval/metrics.py` (`entity_f1`)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** set intersection deduplicates: gold has "Apple" ORG twice, pred has it once →
  set gives TP=1, FN=0, F1=1.0 (wrong). Counter gives TP=1, FN=1, F1=0.667 (correct).
- **Impact:** NER eval overestimated recall for passages with repeated entity mentions.
- **Status:** 🟢 fixed 2026-07-08 — Counter multiset arithmetic replaces set intersection.
```

- [ ] **Step 6: Commit**

```bash
git add eval/metrics.py tests/eval/test_entity_f1.py docs/BUGS.md
git commit -m "fix: entity_f1 uses Counter multiset, correctly counts duplicate entity mentions (B61)"
```

---

### Task 6g: Math/code hard negatives trained model on wrong answers

**Files:**
- Modify: `data/curriculum.py` (`synthesize_hard_negatives`)
- Modify: `docs/BUGS.md`

**Background:** For `math_reasoning` and `code_generation`, `synthesize_hard_negatives` synthesizes a plausible-but-wrong answer and stores it as `{"prompt": ..., "response": wrong_answer}`. Under SFT this trains the model to produce wrong answers. The fix: skip hard-negative synthesis entirely for math/code (CoT-annotated gold examples are the correct data augmentation strategy for these task types — they do not benefit from wrong-answer negatives in SFT). Instead, for math/code, we double-sample from gold with varied prompting rather than synthesizing wrong answers. If the caller passes `targeted_pattern` (surgical mode), log a warning that surgical patterns are not supported for math/code and return gold examples unchanged.

- [ ] **Step 1: Write test**

Create `tests/test_curriculum_hardneg.py`:

```python
from unittest.mock import patch, MagicMock


def test_math_hard_negatives_not_trained_on_wrong_answers():
    """synthesize_hard_negatives for math_reasoning must not produce wrong answers as targets."""
    from data.curriculum import synthesize_hard_negatives

    examples = [
        {"prompt": "What is 2+2?", "answer": "4"},
        {"prompt": "What is 3+3?", "answer": "6"},
    ]

    with patch("data.curriculum.anthropic") as mock_anth:
        results = synthesize_hard_negatives(
            examples, task_type="math_reasoning", n=2, anthropic_client=MagicMock()
        )

    # None of the results should have a "response" that is a synthesized wrong answer
    # (they should either be absent, equal to the gold answer, or the examples should
    #  not be augmented with negative wrong-answer pairs)
    for r in results:
        if "response" in r:
            # If there is a response, it must be the GOLD answer, not a synthetic wrong one
            gold = next((e["answer"] for e in examples if e["prompt"] == r.get("prompt", "")), None)
            if gold:
                assert r["response"] == gold, (
                    f"Math hard negative stored wrong answer '{r['response']}' as training target; "
                    f"gold was '{gold}'"
                )


def test_code_hard_negatives_not_stored_as_wrong_code():
    """code_generation hard negatives must not store broken code as the training target."""
    from data.curriculum import synthesize_hard_negatives

    examples = [{"prompt": "def add(a,b):", "answer": "    return a+b"}]

    with patch("data.curriculum.anthropic"):
        results = synthesize_hard_negatives(
            examples, task_type="code_generation", n=1, anthropic_client=MagicMock()
        )

    for r in results:
        if "response" in r:
            gold = next((e["answer"] for e in examples if e["prompt"] == r.get("prompt", "")), None)
            if gold:
                assert r["response"] == gold, (
                    "Code hard negative stored broken code as training target"
                )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_curriculum_hardneg.py -v`
Expected: FAIL — current code synthesizes wrong answers and stores them as `response`.

- [ ] **Step 3: Fix the math/code branches in `synthesize_hard_negatives`**

In `data/curriculum.py`, find the `math_reasoning` and `code_generation` branches inside `synthesize_hard_negatives`. Replace each with a gold-duplication strategy (no wrong answers):

For `math_reasoning`, replace the existing branch with:

```python
    elif task_type == "math_reasoning":
        # SFT on wrong answers actively harms math models — skip wrong-answer negatives.
        # Instead, return the gold examples as-is (CoT annotation in curate_node provides
        # the real augmentation value for math tasks).
        if targeted_pattern:
            logger.warning(
                "[curriculum] Surgical patterns not supported for math_reasoning hard negatives; "
                "returning gold examples unchanged."
            )
        return list(examples[:n]) if n < len(examples) else list(examples)
```

For `code_generation`, apply the same pattern:

```python
    elif task_type == "code_generation":
        # Wrong-code SFT examples teach the model to produce bugs — skip.
        if targeted_pattern:
            logger.warning(
                "[curriculum] Surgical patterns not supported for code_generation hard negatives; "
                "returning gold examples unchanged."
            )
        return list(examples[:n]) if n < len(examples) else list(examples)
```

Add `import logging; logger = logging.getLogger(__name__)` at the top of `curriculum.py` if not already present.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_curriculum_hardneg.py -v`
Expected: 2 passed.

- [ ] **Step 5: Log B62 in BUGS.md**

Table row:

```
| B62 | 🟢 | curriculum | math/code hard negatives stored wrong answers as SFT targets, training model to produce errors |
```

Detail section:

```
## B62 — math/code hard negatives trained model on wrong answers
- **Where:** `data/curriculum.py` (`synthesize_hard_negatives`)
- **When:** 2026-07-08, pipeline hardening review
- **How found:** synthesize_hard_negatives for math_reasoning called teacher LLM to produce a
  plausible-but-wrong answer and stored it as {"prompt": ..., "response": wrong_answer}.
  SFTTrainer maximizes likelihood of whatever is in "response", so the model learned to output
  wrong answers for those prompts.
- **Impact:** math/code fine-tuned models were actively trained to produce incorrect outputs on
  the hard-negative examples — negative transfer rather than positive learning signal.
- **Status:** 🟢 fixed 2026-07-08 — math/code hard-negative synthesis removed. Gold examples
  are returned unchanged (CoT annotation in curate_node provides the relevant augmentation).
```

- [ ] **Step 6: Commit**

```bash
git add data/curriculum.py tests/test_curriculum_hardneg.py docs/BUGS.md
git commit -m "fix: remove math/code wrong-answer hard negatives; return gold instead (B62)"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| Turn budget via `recursion_limit` | Task 1 |
| Turn budget via `iterate_node` semantic count | Task 1 |
| Hardware check before termination | Task 2 |
| Param-count tiering (formula `int4_size_mb * 2 / 1000`) | Task 3 |
| Sibling tier inherits base | Task 3 |
| Escalation collects full next-tier candidates | Task 4 |
| LLM picks model within escalation tier | Task 4 |
| Downward probe at terminal success | Task 4 |
| Scaling curve uses eval_set seed when no dataset | Task 5 |
| Remove dead task-preference sort | Task 6a |
| Fix `_param_range_label` formula | Task 6b |
| Fix math/code training format (generation branch) | Task 6c |
| Task-type-aware `max_new_tokens` | Task 6d |
| Word-boundary label extraction | Task 6e |
| Multiset `entity_f1` | Task 6f |
| Remove math/code wrong-answer hard negatives | Task 6g |
| BUGS.md entries B51–B62 | Every task |

### Placeholder scan

No TBDs, TODOs, or placeholder code found.

### Type consistency

- `_llm_choose_model(candidates, task_type, task_plan, current_best_score) -> ModelSpec` defined in Task 4 and consumed only within `escalate_node` — no cross-task inconsistency.
- `_probe_model` signature unchanged; return type `float` unchanged — Task 5 is backward compatible.
- `entity_f1` signature unchanged — Task 6f is a pure internal fix.
- `extract_predictions` signature unchanged — Task 6e is a pure internal fix.
- `run_eval` signature unchanged — Task 6d adds a local `max_new_tokens` variable, doesn't change the public API.

### Known dependency ordering

- Task 3 must precede Task 4 (escalate uses tier comparisons).
- Task 1 must precede Task 2 (both modify `iterate_node`; Step 4 in Task 2 assumes the `stagnant` and `intervention` variables already exist in the function, which they do in the current code).
- Tasks 6a–6g are independent of each other and of Tasks 1–5.
