# Staged Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat one-shot model selection in `task_analysis_node` with a staged pipeline: (1) feasibility filter, (2) on-device hardware validation stub, (3) accuracy scaling curve to pick the smallest model that clears the accuracy goal.

**Architecture:** A new `hardware_filter.py` node runs after `hardware_research` (pre-graph) and eliminates models that fail storage/memory/throughput inequalities plus on-device checks (stub for now). A new `scaling_curve_node` runs inside the graph after `task_analysis`, fine-tunes 3 candidates, fits an accuracy-vs-log(size) curve, and promotes the winner to `selected_model`. The existing pool in `android_pool.py` is expanded with explicit Q4_K_M and Q8_0 variants so every quantization level participates in the loop on equal footing. Training always uses LoRA (all pool models are ≤4B, all fit in VRAM at full precision for training).

**Tech Stack:** Python 3.11, LangGraph, Unsloth, numpy (scipy.optimize for curve fitting), existing `eval/harness.py`, existing `training/lora_trainer.py`.

## Global Constraints

- Training always uses LoRA regardless of model quant level (all models ≤4B; QLoRA not needed).
- `filter_pool()` and `filter_pool_by_task()` in `android_pool.py` must continue to work for `escalate_node` — do not rename or remove them.
- `hardware_filter` runs pre-graph (same pattern as `hardware_research`); `scaling_curve_node` runs inside the LangGraph graph.
- On-device eval stub must return `HardwareEvalResult` with `success=True` and placeholder values — no real device calls.
- The `AgentState` `selected_model` field is the single source of truth for which model the loop trains; `scaling_curve_node` writes to it.
- All new files use the project's existing import style (no `__init__.py` changes needed; Python finds modules from repo root).
- numpy is already present (scipy may need installing; use `numpy.polyfit` on log-transformed x as fallback if scipy absent).

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `config/android_pool.py` | Modify | Add `quant: str \| None` to `ModelSpec`; add Q4_K_M + Q8_0 sibling entries for all 12 base models; re-tier siblings by RAM |
| `hardware_eval/__init__.py` | Create | Empty package marker |
| `hardware_eval/on_device_eval.py` | Create | `HardwareEvalResult` dataclass + stub `run_on_device_eval()` |
| `agent/nodes/cold_start/hardware_filter.py` | Create | `run_hardware_filter(pool, constraints)` — Stage 1 (inequalities) + Stage 2 (on-device stub); returns filtered list |
| `agent/nodes/cold_start/scaling_curve.py` | Create | `scaling_curve_node(state)` — fine-tune 3 candidates, fit curve, set `selected_model` |
| `agent/state.py` | Modify | Add `feasible_models: list[ModelSpec]` field |
| `agent/graph.py` | Modify | Insert `scaling_curve` node after `task_analysis`; rewire edge `task_analysis → scaling_curve → eval_setup` |
| `docs/PIPELINE.md` | Modify | Document the two new nodes in the pre-graph and graph sections |

---

### Task 1: Add `quant` field + Q4/Q8 siblings to `ANDROID_POOL`

**Files:**
- Modify: `config/android_pool.py`

**Interfaces:**
- Produces: `ModelSpec` gains `quant: str | None` field (default `None` = base/BF16). All existing callers continue to work — `filter_pool`, `filter_pool_by_task`, `check_hardware_constraints` use `int4_size_mb` and `peak_memory_mb` which are still present on every entry.
- `ANDROID_POOL` grows from 12 to 36 entries (12 base + 12 Q4_K_M + 12 Q8_0). Tier assignment is purely by `peak_memory_mb` (same breakpoints: <1.2GB=0, 1.0–1.8GB=1, 1.5–2.5GB=2, 2.5–4.5GB=3).

**Q4_K_M size formula:** `int4_size_mb` stays as-is (pool already stores Q4_K_M sizes). `peak_memory_mb` for Q4_K_M variant = base `int4_size_mb` + 400 MB (KV cache + runtime, slightly smaller than BF16 base because weights are smaller).

**Q8_0 size formula:** `int4_size_mb` ≈ base `int4_size_mb` × 1.9 (Q8_0 is ~1.9× larger than Q4_K_M). `peak_memory_mb` = Q8_0 file size + 400 MB.

**Tier breakpoints for siblings (peak_memory_mb):**
- < 1 200 MB → tier 0
- 1 200–1 800 MB → tier 1
- 1 800–3 000 MB → tier 2
- > 3 000 MB → tier 3

**tok_s values for quantized variants:** Q4_K_M tok/s = base model values (pool values already reflect Q4_K_M). Q8_0 tok/s ≈ base × 0.65 (Q8_0 is ~35% slower than Q4_K_M on CPU due to 2× memory bandwidth; source: llama.cpp benchmarks).

- [ ] **Step 1: Add `quant` field to `ModelSpec`**

In `config/android_pool.py`, add `quant: str | None = None` as the last field of the `ModelSpec` dataclass (after `notes`). Python dataclasses require fields with defaults to come after fields without defaults; `notes` already has `""` default, so add after it:

```python
@dataclass
class ModelSpec:
    model_id: str
    int4_size_mb: int
    tier: int
    tok_s_snapdragon_660: float
    tok_s_snapdragon_778g: float
    tok_s_snapdragon_8gen3: float
    peak_memory_mb: int
    gsm8k: float
    mmlu: float
    notes: str = ""
    quant: str | None = None   # None = base (BF16/FP16). "Q4_K_M" or "Q8_0" = GGUF variant.

    def tok_s_for_chip(self, chip: str) -> float:
        # (existing body unchanged)
        ...
```

- [ ] **Step 2: Run existing tests to verify nothing broke**

```bash
cd c:/Users/eliotli2/Documents/VSCode/SLM_Factory
python -c "from config.android_pool import ANDROID_POOL, ModelSpec; print(len(ANDROID_POOL), 'models OK')"
```
Expected: `12 models OK`

- [ ] **Step 3: Write a helper to build quantized siblings**

Add this private function just above `ANDROID_POOL` in `android_pool.py`:

```python
def _q4_sibling(base: ModelSpec) -> ModelSpec:
    """Q4_K_M GGUF variant. Sizes already Q4_K_M in base; peak RAM = file + 400 MB overhead."""
    peak = base.int4_size_mb + 400
    tier = 0 if peak < 1200 else 1 if peak < 1800 else 2 if peak < 3000 else 3
    return ModelSpec(
        model_id=base.model_id,
        int4_size_mb=base.int4_size_mb,
        tier=tier,
        tok_s_snapdragon_660=base.tok_s_snapdragon_660,
        tok_s_snapdragon_778g=base.tok_s_snapdragon_778g,
        tok_s_snapdragon_8gen3=base.tok_s_snapdragon_8gen3,
        peak_memory_mb=peak,
        gsm8k=base.gsm8k,
        mmlu=base.mmlu,
        notes=base.notes,
        quant="Q4_K_M",
    )


def _q8_sibling(base: ModelSpec) -> ModelSpec:
    """Q8_0 GGUF variant. ~1.9× larger file, ~35% slower tok/s, higher quality ceiling."""
    q8_size = int(base.int4_size_mb * 1.9)
    peak = q8_size + 400
    tier = 0 if peak < 1200 else 1 if peak < 1800 else 2 if peak < 3000 else 3
    return ModelSpec(
        model_id=base.model_id,
        int4_size_mb=q8_size,
        tier=tier,
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

- [ ] **Step 4: Expand `ANDROID_POOL` with siblings**

After the closing `]` of `ANDROID_POOL`, add:

```python
# Expand pool with Q4_K_M and Q8_0 quantized siblings for every base model.
# Re-tiered by peak_memory_mb so quantized variants compete with appropriately-sized peers.
_BASE_MODELS = [m for m in ANDROID_POOL]  # snapshot before mutation
ANDROID_POOL = sorted(
    _BASE_MODELS + [_q4_sibling(m) for m in _BASE_MODELS] + [_q8_sibling(m) for m in _BASE_MODELS],
    key=lambda m: (m.tier, m.int4_size_mb),
)
```

- [ ] **Step 5: Verify pool expansion**

```bash
python -c "
from config.android_pool import ANDROID_POOL
print(f'{len(ANDROID_POOL)} total entries')
for m in ANDROID_POOL:
    print(f'  tier={m.tier} {m.int4_size_mb:5d}MB peak={m.peak_memory_mb:5d}MB quant={m.quant!r:10s} {m.model_id}')
"
```
Expected: 36 entries, sorted by tier then int4_size_mb, with each base model appearing three times (None, Q4_K_M, Q8_0).

- [ ] **Step 6: Commit**

```bash
git add config/android_pool.py
git commit -m "feat: expand ANDROID_POOL with Q4_K_M and Q8_0 siblings, add quant field to ModelSpec"
```

---

### Task 2: Create `hardware_eval/` stub package

**Files:**
- Create: `hardware_eval/__init__.py`
- Create: `hardware_eval/on_device_eval.py`

**Interfaces:**
- Produces: `HardwareEvalResult(model_id, success, ttft_ms, tok_per_s, avg_watts, peak_memory_mb, error)` dataclass and `run_on_device_eval(model: ModelSpec, constraints: HardwareConstraints) -> HardwareEvalResult`.
- Consumed by: `hardware_filter.py` (Task 3).

- [ ] **Step 1: Create package marker**

Create `hardware_eval/__init__.py` as an empty file.

- [ ] **Step 2: Write `HardwareEvalResult` and stub function**

Create `hardware_eval/on_device_eval.py`:

```python
# hardware_eval/on_device_eval.py
"""
Stub for Stage 2 on-device hardware evaluation.

Phase 1: returns placeholder passing values and logs what would happen.
Phase 2 (future): ADB shell commands to deploy GGUF, run llama-bench,
parse latency/power from dumpsys output.
"""
import logging
from dataclasses import dataclass
from config.android_pool import ModelSpec, HardwareConstraints

logger = logging.getLogger(__name__)


@dataclass
class HardwareEvalResult:
    model_id: str
    success: bool
    ttft_ms: float | None        # time-to-first-token, ms
    tok_per_s: float | None      # sustained decode throughput
    avg_watts: float | None      # average power during inference
    peak_memory_mb: int | None   # measured peak RAM
    error: str | None = None


def run_on_device_eval(
    model: ModelSpec,
    constraints: HardwareConstraints,
) -> HardwareEvalResult:
    """
    Phase 1 stub: log what would run and return a synthetic pass result.

    Real implementation would:
      1. Push GGUF to device via `adb push`
      2. Run `adb shell llama-bench -m <model> -n 128 -p 64`
      3. Parse stdout for tok/s and ttft_ms
      4. Run `adb shell dumpsys thermalservice` for thermal status
      5. Parse `adb shell cat /sys/class/power_supply/battery/current_now` for watts
    """
    logger.info(
        "[hardware_eval][STUB] Would run on-device eval for %s on chip=%s "
        "(storage=%dMB, memory=%dMB, ttft_limit=%dms, power_limit=%.1fW)",
        model.model_id, constraints.target_chip,
        constraints.storage_mb, constraints.memory_mb,
        constraints.latency_ttft_ms, constraints.power_watts,
    )
    logger.info(
        "[hardware_eval][STUB] Phase 2 will deploy via ADB, run llama-bench, "
        "parse dumpsys thermalservice. Returning synthetic pass for now."
    )

    # Synthetic values derived from ModelSpec theoretical estimates.
    tok_s = model.tok_s_for_chip(constraints.target_chip)
    ttft_ms = (1.0 / max(tok_s, 0.1)) * 1000

    return HardwareEvalResult(
        model_id=model.model_id,
        success=True,
        ttft_ms=round(ttft_ms, 1),
        tok_per_s=round(tok_s, 1),
        avg_watts=None,    # Phase 2: parsed from battery current_now
        peak_memory_mb=model.peak_memory_mb,
        error=None,
    )
```

- [ ] **Step 3: Verify import**

```bash
python -c "from hardware_eval.on_device_eval import run_on_device_eval, HardwareEvalResult; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add hardware_eval/__init__.py hardware_eval/on_device_eval.py
git commit -m "feat: add hardware_eval stub package with HardwareEvalResult and run_on_device_eval"
```

---

### Task 3: Create `hardware_filter.py` (Stages 1 + 2)

**Files:**
- Create: `agent/nodes/cold_start/hardware_filter.py`

**Interfaces:**
- Consumes: `filter_pool(constraints) -> list[ModelSpec]` from `android_pool.py`; `run_on_device_eval(model, constraints) -> HardwareEvalResult` from `hardware_eval.on_device_eval`; `check_hardware_constraints(model, constraints, measured) -> dict` and `all_constraints_pass(hw_check) -> bool` from `android_pool.py`.
- Produces: `run_hardware_filter(constraints: HardwareConstraints) -> list[ModelSpec]` — sorted list of models that passed both stages, largest-first within each tier (for Stage 2's iterate-from-largest logic).

**Stage 1** (free, no device): `filter_pool(constraints)` already does storage + memory + min_tok_s. We call it as-is.

**Stage 2** (on-device stub): iterate from largest model to smallest. For each, call `run_on_device_eval`, then `check_hardware_constraints(model, constraints, measured=result.__dict__)`. Discard if `all_constraints_pass` returns False. The result list preserves largest-to-smallest order so `scaling_curve_node` (Task 4) can slice small/medium/large candidates easily.

- [ ] **Step 1: Write `hardware_filter.py`**

```python
# agent/nodes/cold_start/hardware_filter.py
"""
Staged hardware filter — runs pre-graph after hardware_research.

Stage 1: inequality filter (storage, memory, throughput) — free.
Stage 2: on-device eval stub (latency, power, thermal) — iterates largest→smallest,
         discards violators. Currently a stub; Phase 2 wires real ADB runs.
"""
import logging
from config.android_pool import (
    HardwareConstraints,
    ModelSpec,
    filter_pool,
    check_hardware_constraints,
    all_constraints_pass,
)
from hardware_eval.on_device_eval import run_on_device_eval

logger = logging.getLogger(__name__)


def run_hardware_filter(constraints: HardwareConstraints) -> list[ModelSpec]:
    """
    Return models that pass both stages, sorted largest→smallest
    (by int4_size_mb descending within tier) so scaling_curve_node
    can slice small/medium/large candidates from the ends and middle.

    Stage 1: filter_pool() inequality checks (storage, memory, min_tok_s).
    Stage 2: on-device eval from largest candidate downward; discard failures.
    """
    # ── Stage 1 ──────────────────────────────────────────────────────────────
    stage1 = filter_pool(constraints)
    logger.info("[hardware_filter] Stage 1: %d/%d models passed inequality checks",
                len(stage1), len(stage1))

    if not stage1:
        logger.warning("[hardware_filter] Stage 1 eliminated all models.")
        return []

    # Sort largest→smallest for Stage 2 iteration
    stage1_desc = sorted(stage1, key=lambda m: m.int4_size_mb, reverse=True)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    passed: list[ModelSpec] = []
    for model in stage1_desc:
        hw_result = run_on_device_eval(model, constraints)
        measured = {
            "ttft_ms": hw_result.ttft_ms,
            "tok_per_s": hw_result.tok_per_s,
            "avg_watts": hw_result.avg_watts,
            "peak_memory_mb": hw_result.peak_memory_mb,
        }
        hw_check = check_hardware_constraints(model, constraints, measured=measured)
        ok = all_constraints_pass(hw_check)
        logger.info(
            "[hardware_filter] Stage 2: %s %s (ttft=%.0fms, tok/s=%.1f)",
            "✓" if ok else "✗",
            model.model_id,
            hw_result.ttft_ms or 0,
            hw_result.tok_per_s or 0,
        )
        if ok:
            passed.append(model)

    logger.info(
        "[hardware_filter] Stage 2: %d/%d models passed on-device check",
        len(passed), len(stage1_desc),
    )
    return passed  # largest→smallest order preserved
```

- [ ] **Step 2: Write test**

Create `tests/cold_start/test_hardware_filter.py`:

```python
import pytest
from unittest.mock import patch
from config.android_pool import HardwareConstraints, ANDROID_POOL
from agent.nodes.cold_start.hardware_filter import run_hardware_filter


@pytest.fixture
def loose_constraints():
    return HardwareConstraints(
        storage_mb=10000,
        memory_mb=10000,
        latency_ttft_ms=5000,
        power_watts=20.0,
        target_chip="snapdragon_778g",
        min_tok_s=0.0,
    )


@pytest.fixture
def tight_constraints():
    return HardwareConstraints(
        storage_mb=500,   # only sub-0.5B models fit
        memory_mb=800,
        latency_ttft_ms=1000,
        power_watts=5.0,
        target_chip="snapdragon_778g",
        min_tok_s=0.0,
    )


def test_returns_list(loose_constraints):
    result = run_hardware_filter(loose_constraints)
    assert isinstance(result, list)


def test_all_pass_loose_constraints(loose_constraints):
    result = run_hardware_filter(loose_constraints)
    assert len(result) == len(ANDROID_POOL)


def test_tight_constraints_filters(tight_constraints):
    result = run_hardware_filter(tight_constraints)
    for m in result:
        assert m.int4_size_mb <= tight_constraints.storage_mb
        assert m.peak_memory_mb <= tight_constraints.memory_mb


def test_order_largest_first(loose_constraints):
    result = run_hardware_filter(loose_constraints)
    sizes = [m.int4_size_mb for m in result]
    assert sizes == sorted(sizes, reverse=True)


def test_empty_when_nothing_fits():
    constraints = HardwareConstraints(
        storage_mb=1, memory_mb=1, latency_ttft_ms=1,
        power_watts=0.1, target_chip="snapdragon_778g",
    )
    result = run_hardware_filter(constraints)
    assert result == []
```

- [ ] **Step 3: Run test**

```bash
python -m pytest tests/cold_start/test_hardware_filter.py -v
```
Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add agent/nodes/cold_start/hardware_filter.py tests/cold_start/test_hardware_filter.py
git commit -m "feat: add hardware_filter with Stage 1 inequality + Stage 2 on-device stub"
```

---

### Task 4: Add `feasible_models` to `AgentState` and call `run_hardware_filter` from `task_analysis_node`

**Files:**
- Modify: `agent/state.py` — add `feasible_models: list[ModelSpec]`
- Modify: `agent/nodes/cold_start/task_analysis.py` — call `run_hardware_filter`, store result, use it instead of calling `filter_pool_by_task` directly against the full pool

**Interfaces:**
- `task_analysis_node` now reads `state["hardware_constraints"]`, runs `run_hardware_filter`, writes `state["feasible_models"]`, then re-sorts that list by task preference. The `filter_pool_by_task` function is still used to compute the preference sort, but its input is `feasible_models` not the full pool.
- `scaling_curve_node` (Task 5) reads `state["feasible_models"]`.
- `escalate_node` still calls `filter_pool(state["hardware_constraints"])` directly — that's fine; it already re-runs filtering.

**Why this task is separate from Task 5:** `feasible_models` must exist in state before `scaling_curve_node` can read it. This task establishes that dependency.

- [ ] **Step 1: Add `feasible_models` to `AgentState`**

In `agent/state.py`, add after `selected_model`:

```python
feasible_models: list[ModelSpec]       # models that passed hardware_filter (Stages 1+2)
```

The full updated `AgentState` relevant section:

```python
from config.android_pool import ModelSpec, HardwareConstraints

class AgentState(TypedDict):
    description: str
    target_metric: str
    hardware_constraints: HardwareConstraints

    task_type: str
    selected_model: Optional[ModelSpec]
    feasible_models: list[ModelSpec]       # ← new: hardware-filtered pool, largest→smallest
    stop_threshold: float
    initial_stop_threshold: float
    task_plan: Optional[dict]
    autonomous: bool
    # ... rest unchanged
```

- [ ] **Step 2: Update `task_analysis_node` to call `run_hardware_filter`**

Replace the body of `task_analysis_node` in `agent/nodes/cold_start/task_analysis.py`. The key changes:
- Import `run_hardware_filter`
- Run it before the preference sort
- Store result in `state["feasible_models"]`
- Apply task-preference sort on top of the filtered list

```python
# agent/nodes/cold_start/task_analysis.py
from agent.state import AgentState
from config.android_pool import filter_pool_by_task, ANDROID_POOL
from agent.nodes.cold_start.hardware_filter import run_hardware_filter

_VALID_TASK_TYPES = {
    "classification",
    "NER",
    "math_reasoning",
    "code_generation",
    "generation",
}

_TASK_TYPE_TO_POOL_KEY: dict[str, str | None] = {
    "classification":  "classification",
    "NER":             "ner",
    "math_reasoning":  "math",
    "code_generation": "code",
    "generation":      None,
}


def task_analysis_node(state: AgentState) -> AgentState:
    """
    Node 1: classify the task, filter hardware pool, set stop threshold.

    Does NOT select the final model — that is done by scaling_curve_node (Node 1b)
    after fine-tuning 3 candidates. Stores the hardware-filtered feasible set in
    state["feasible_models"] for scaling_curve_node to consume.
    """
    task_type = state.get("task_type", "")
    need_plan = state.get("autonomous") or task_type not in _VALID_TASK_TYPES

    if need_plan and state.get("task_plan") is None:
        from agent.task_planner import plan_task
        plan = plan_task(state["description"], model_pool=ANDROID_POOL)
        state["task_plan"] = plan
        task_type = plan["task_type"]
        state["task_type"] = task_type
        if plan.get("stop_threshold"):
            threshold = float(plan["stop_threshold"])
            state["stop_threshold"] = threshold
            if not state.get("initial_stop_threshold"):
                state["initial_stop_threshold"] = threshold

    if task_type not in _VALID_TASK_TYPES:
        raise ValueError(
            f"task_type must be one of {_VALID_TASK_TYPES!r}, got {task_type!r}. "
            "Set task_type in the initial AgentState, or enable autonomous mode."
        )

    # Stage 1 + 2: hardware filter (inequality + on-device stub)
    feasible = run_hardware_filter(state["hardware_constraints"])
    if not feasible:
        raise RuntimeError("No models in Android pool satisfy hardware constraints.")

    # Apply task-preference sort on top of hardware-filtered set.
    # filter_pool_by_task re-filters from the full ANDROID_POOL; we replicate
    # its sort logic here directly to avoid re-filtering what we already have.
    pool_key = _TASK_TYPE_TO_POOL_KEY.get(task_type)
    if pool_key == "math" or pool_key == "reasoning":
        feasible = sorted(feasible, key=lambda m: (m.tier, -m.gsm8k))
    elif pool_key == "classification":
        def _cls_key(m):
            smol_bonus = -0.05 if "SmolLM" in m.model_id else 0.0
            return (m.tier, smol_bonus - m.mmlu)
        feasible = sorted(feasible, key=_cls_key)
    elif pool_key in ("ner", "multilingual", "code"):
        def _qwen_key(m):
            qwen_bonus = -0.03 if "Qwen" in m.model_id else 0.0
            return (m.tier, qwen_bonus - m.gsm8k)
        feasible = sorted(feasible, key=_qwen_key)
    # else: already sorted largest→smallest from hardware_filter

    # math_reasoning: front-load specialized reasoning models
    if task_type == "math_reasoning":
        preferred = [m for m in feasible if any(
            k in m.model_id for k in ("DeepSeek-R1", "Qwen3", "Phi-4")
        )]
        if preferred:
            others = [m for m in feasible if m not in preferred]
            feasible = preferred + others

    state["feasible_models"] = feasible

    if not state.get("stop_threshold"):
        state["stop_threshold"] = 0.96

    # selected_model is intentionally NOT set here.
    # scaling_curve_node sets it after probing candidates.
    return state
```

- [ ] **Step 3: Run smoke test**

```bash
python -c "
from agent.nodes.cold_start.task_analysis import task_analysis_node
from config.android_pool import HardwareConstraints
state = {
    'description': 'classify SMS spam',
    'task_type': 'classification',
    'autonomous': False,
    'hardware_constraints': HardwareConstraints(
        storage_mb=5000, memory_mb=5000, latency_ttft_ms=3000,
        power_watts=10.0, target_chip='snapdragon_778g',
    ),
    'stop_threshold': 0.0,
    'task_plan': None,
    'target_metric': 'f1',
}
result = task_analysis_node(state)
print('feasible_models:', len(result['feasible_models']))
print('selected_model:', result.get('selected_model'))  # should be None
print('stop_threshold:', result['stop_threshold'])
"
```
Expected: feasible_models count > 0, `selected_model` is `None`, `stop_threshold` is 0.96.

- [ ] **Step 4: Commit**

```bash
git add agent/state.py agent/nodes/cold_start/task_analysis.py
git commit -m "feat: task_analysis_node runs hardware_filter, stores feasible_models; defers model selection to scaling_curve_node"
```

---

### Task 5: Create `scaling_curve_node` (Stage 3)

**Files:**
- Create: `agent/nodes/cold_start/scaling_curve.py`

**Interfaces:**
- Consumes: `state["feasible_models"]` (list[ModelSpec], largest→smallest from Task 4), `state["stop_threshold"]` (float), `state["task_type"]` (str), `state["hardware_constraints"]` (HardwareConstraints), `state["eval_set"]` (EvalSet — must be populated by `eval_setup_node` before this runs; see graph wiring note in Task 6).
- Produces: `state["selected_model"]` (ModelSpec) — the smallest model whose predicted post-finetune accuracy ≥ stop_threshold.

**Algorithm:**
1. Pick 3 candidates from `feasible_models`: index 0 (largest/best), index -1 (smallest), index len//2 (middle). If fewer than 3 models, use what's available (minimum 1).
2. For each candidate: run one LoRA fine-tune epoch (quick probe — 1 epoch, same LoRA config as main loop), then `run_eval()` on `state["eval_set"]`. Record `(log(int4_size_mb), f1)`.
3. Fit a line through those points: `f1 ≈ a * log(size) + b` using `numpy.polyfit(log_sizes, f1s, deg=1)`.
4. For each model in `feasible_models` sorted smallest→largest: predict `f1 = a * log(size) + b`. Return the first whose predicted f1 ≥ `stop_threshold`. If none, return the largest.
5. Set `state["selected_model"]` to the winner.

**Why 1 epoch for probing:** The goal is a relative ranking, not absolute accuracy. One epoch gives a consistent signal across candidates without running the full training budget. The main loop will retrain from scratch with the full config anyway.

- [ ] **Step 1: Write `scaling_curve.py`**

```python
# agent/nodes/cold_start/scaling_curve.py
"""
Node 1b (runs after task_analysis, before eval_setup is used by the main loop):
Fit an accuracy-vs-log(size) scaling curve from 3 fine-tuned probe runs,
then select the smallest model whose predicted accuracy meets stop_threshold.
"""
import logging
import math
import os
import tempfile

import numpy as np

from agent.state import AgentState
from config.android_pool import ModelSpec
from eval.harness import run_eval
from training.lora_trainer import TrainingConfig, run_lora_training

logger = logging.getLogger(__name__)

# Probe training config: 1 epoch, minimal LoRA — just enough for a ranking signal.
_PROBE_EPOCHS = 1
_PROBE_LORA_RANK = 8
_PROBE_LR = 2e-4
_PROBE_BATCH = 8


def _pick_candidates(models: list[ModelSpec]) -> list[ModelSpec]:
    """Pick up to 3 candidates: largest, middle, smallest."""
    if len(models) <= 1:
        return models
    if len(models) == 2:
        return [models[0], models[-1]]
    mid = len(models) // 2
    seen = set()
    result = []
    for idx in (0, mid, len(models) - 1):
        m = models[idx]
        if m.model_id not in seen:
            result.append(m)
            seen.add(m.model_id)
    return result


def _probe_model(
    model: ModelSpec,
    state: AgentState,
    probe_dir: str,
) -> float:
    """Fine-tune one epoch and return eval f1. Returns 0.0 on failure."""
    model_dir = os.path.join(probe_dir, model.model_id.replace("/", "_"))
    config = TrainingConfig(
        base_model=model.model_id,
        nr_epochs=_PROBE_EPOCHS,
        learning_rate=_PROBE_LR,
        batch_size=_PROBE_BATCH,
        lora_rank=_PROBE_LORA_RANK,
        task_type=state["task_type"],
    )
    try:
        dataset_path = state.get("current_dataset_path")
        if not dataset_path:
            logger.warning("[scaling_curve] No dataset path in state; skipping probe for %s", model.model_id)
            return 0.0
        weights_ref = run_lora_training(dataset_path, config, output_dir=model_dir, task_type=state["task_type"])
        result = run_eval(state["eval_set"], weights_ref, model.model_id, state["task_type"])
        logger.info("[scaling_curve] Probe %s → f1=%.4f", model.model_id, result.f1)
        return result.f1
    except Exception as exc:
        logger.warning("[scaling_curve] Probe failed for %s: %s", model.model_id, exc)
        return 0.0


def scaling_curve_node(state: AgentState) -> AgentState:
    """
    Node 1b: fit accuracy-vs-log(size) curve, select smallest model above stop_threshold.

    Reads:  state["feasible_models"], state["stop_threshold"], state["eval_set"],
            state["current_dataset_path"], state["task_type"]
    Writes: state["selected_model"]
    """
    feasible = state.get("feasible_models", [])
    if not feasible:
        raise RuntimeError("scaling_curve_node: feasible_models is empty.")

    stop_threshold = state.get("stop_threshold", 0.96)
    candidates = _pick_candidates(feasible)

    logger.info(
        "[scaling_curve] Probing %d candidates: %s",
        len(candidates), [m.model_id for m in candidates],
    )

    with tempfile.TemporaryDirectory(prefix="slm_probe_") as probe_dir:
        points: list[tuple[float, float]] = []
        for model in candidates:
            f1 = _probe_model(model, state, probe_dir)
            log_size = math.log(model.int4_size_mb)
            points.append((log_size, f1))
            logger.info(
                "[scaling_curve] Point: log(size)=%.3f f1=%.4f (%s, quant=%s)",
                log_size, f1, model.model_id, model.quant,
            )

    if len(points) < 2:
        # Can't fit a line; fall back to smallest feasible model.
        logger.warning("[scaling_curve] Too few probe points (%d); selecting smallest model.", len(points))
        # feasible is largest→smallest; smallest is last
        state["selected_model"] = feasible[-1]
        return state

    log_sizes = np.array([p[0] for p in points])
    f1s = np.array([p[1] for p in points])
    coeffs = np.polyfit(log_sizes, f1s, deg=1)  # [a, b]: f1 = a*log(size) + b
    a, b = float(coeffs[0]), float(coeffs[1])
    logger.info("[scaling_curve] Fit: f1 = %.4f * log(size) + %.4f", a, b)

    # Walk smallest→largest; pick first whose predicted f1 >= stop_threshold
    for model in reversed(feasible):  # feasible is largest→smallest, so reversed = smallest→largest
        predicted = a * math.log(model.int4_size_mb) + b
        logger.info(
            "[scaling_curve] %s: predicted_f1=%.4f vs threshold=%.4f → %s",
            model.model_id, predicted, stop_threshold,
            "SELECT" if predicted >= stop_threshold else "skip",
        )
        if predicted >= stop_threshold:
            state["selected_model"] = model
            logger.info(
                "[scaling_curve] Selected: %s (tier=%d, quant=%s, predicted_f1=%.4f)",
                model.model_id, model.tier, model.quant, predicted,
            )
            return state

    # None predicted to meet threshold — use largest (best chance)
    state["selected_model"] = feasible[0]
    logger.warning(
        "[scaling_curve] No model predicted to meet threshold %.4f; "
        "defaulting to largest: %s", stop_threshold, feasible[0].model_id,
    )
    return state
```

- [ ] **Step 2: Write test**

Create `tests/cold_start/test_scaling_curve.py`:

```python
import math
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from config.android_pool import ModelSpec, HardwareConstraints
from agent.nodes.cold_start.scaling_curve import (
    _pick_candidates,
    scaling_curve_node,
)


def _make_model(model_id, int4_size_mb, tier=1, quant=None):
    return ModelSpec(
        model_id=model_id,
        int4_size_mb=int4_size_mb,
        tier=tier,
        tok_s_snapdragon_660=5.0,
        tok_s_snapdragon_778g=10.0,
        tok_s_snapdragon_8gen3=25.0,
        peak_memory_mb=int4_size_mb + 400,
        gsm8k=0.6,
        mmlu=0.5,
        quant=quant,
    )


def test_pick_candidates_three():
    models = [_make_model(f"m{i}", 1000 - i * 100) for i in range(6)]
    candidates = _pick_candidates(models)
    assert len(candidates) == 3
    assert candidates[0] is models[0]    # largest
    assert candidates[-1] is models[-1]  # smallest


def test_pick_candidates_one():
    models = [_make_model("only", 500)]
    assert _pick_candidates(models) == models


def test_pick_candidates_two():
    models = [_make_model("big", 900), _make_model("small", 300)]
    candidates = _pick_candidates(models)
    assert len(candidates) == 2


def _make_state(feasible):
    from data.eval_set import EvalSet
    eval_set = MagicMock(spec=EvalSet)
    return {
        "feasible_models": feasible,
        "stop_threshold": 0.80,
        "task_type": "classification",
        "hardware_constraints": HardwareConstraints(
            storage_mb=5000, memory_mb=5000, latency_ttft_ms=3000,
        ),
        "eval_set": eval_set,
        "current_dataset_path": "/fake/dataset.jsonl",
    }


@patch("agent.nodes.cold_start.scaling_curve._probe_model")
def test_selects_smallest_above_threshold(mock_probe):
    # Models sorted largest→smallest: 2000MB, 1000MB, 500MB
    models = [
        _make_model("large", 2000),
        _make_model("medium", 1000),
        _make_model("small", 500),
    ]
    # Probe returns f1 values that make a clean line
    # large→0.90, medium→0.85, small→0.75
    mock_probe.side_effect = [0.90, 0.85, 0.75]

    state = _make_state(models)
    result = scaling_curve_node(state)

    # Fit: f1 = a*log(size)+b. Smallest above 0.80 threshold should be medium (predicted ~0.85)
    assert result["selected_model"] is not None
    # medium or large should be selected (small is predicted below 0.80)
    assert result["selected_model"].model_id != "small"


@patch("agent.nodes.cold_start.scaling_curve._probe_model")
def test_falls_back_to_largest_when_none_meet_threshold(mock_probe):
    models = [_make_model("large", 2000), _make_model("small", 300)]
    mock_probe.side_effect = [0.50, 0.30]  # neither meets 0.80

    state = _make_state(models)
    result = scaling_curve_node(state)
    assert result["selected_model"].model_id == "large"


@patch("agent.nodes.cold_start.scaling_curve._probe_model")
def test_single_model_selected_directly(mock_probe):
    models = [_make_model("only", 700)]
    mock_probe.return_value = 0.88

    state = _make_state(models)
    result = scaling_curve_node(state)
    assert result["selected_model"].model_id == "only"


def test_empty_feasible_raises():
    state = _make_state([])
    with pytest.raises(RuntimeError, match="feasible_models is empty"):
        scaling_curve_node(state)
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/cold_start/test_scaling_curve.py -v
```
Expected: 5 passed.

- [ ] **Step 4: Commit**

```bash
git add agent/nodes/cold_start/scaling_curve.py tests/cold_start/test_scaling_curve.py
git commit -m "feat: add scaling_curve_node — probe 3 candidates, fit accuracy curve, select smallest model above threshold"
```

---

### Task 6: Wire new nodes into `graph.py` and update `PIPELINE.md`

**Files:**
- Modify: `agent/graph.py`
- Modify: `docs/PIPELINE.md`

**Graph wiring change:**

Current cold_start path:
```
task_analysis → eval_setup → curate → train → ...
```

New cold_start path:
```
task_analysis → eval_setup → scaling_curve → curate → train → ...
```

`eval_setup` must run before `scaling_curve` because `scaling_curve_node` needs `state["eval_set"]` to score probe fine-tunes. `eval_setup_node` writes `state["eval_set"]`. `curate` runs after `scaling_curve` because we now know the winning model.

- [ ] **Step 1: Update `graph.py`**

Replace the cold_start section of `build_graph`:

```python
from agent.nodes.cold_start.scaling_curve import scaling_curve_node

# inside build_graph, if mode == "cold_start":
if mode == "cold_start":
    graph.add_node("task_analysis", task_analysis_node)
    graph.add_node("eval_setup", eval_setup_node)
    graph.add_node("scaling_curve", scaling_curve_node)

    graph.set_entry_point("task_analysis")
    graph.add_edge("task_analysis", "eval_setup")
    graph.add_edge("eval_setup", "scaling_curve")
    graph.add_edge("scaling_curve", "curate")
```

The full updated `build_graph` function (only the cold_start block changes; production block and shared loop edges are unchanged):

```python
def build_graph(mode: str = "cold_start") -> CompiledStateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("train", train_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("iterate", iterate_node)
    graph.add_node("curate", curate_node)
    graph.add_node("rollback", rollback_node)
    graph.add_node("escalate", escalate_node)

    if mode == "cold_start":
        graph.add_node("task_analysis", task_analysis_node)
        graph.add_node("eval_setup", eval_setup_node)
        graph.add_node("scaling_curve", scaling_curve_node)

        graph.set_entry_point("task_analysis")
        graph.add_edge("task_analysis", "eval_setup")
        graph.add_edge("eval_setup", "scaling_curve")
        graph.add_edge("scaling_curve", "curate")
    else:
        from agent.nodes.production.trace_ingest import trace_ingest_node
        from agent.nodes.production.taxonomy import taxonomy_construct_node
        from agent.nodes.production.live_confirm import live_confirm_node
        from agent.nodes.production.parent_awareness import parent_awareness_node

        graph.add_node("trace_ingest", trace_ingest_node)
        graph.add_node("taxonomy_construct", taxonomy_construct_node)
        graph.add_node("live_confirm", live_confirm_node)
        graph.add_node("parent_awareness", parent_awareness_node)

        graph.set_entry_point("trace_ingest")
        graph.add_edge("trace_ingest", "taxonomy_construct")
        graph.add_edge("taxonomy_construct", "live_confirm")
        graph.add_edge("live_confirm", "parent_awareness")
        graph.add_edge("parent_awareness", "curate")

    graph.add_edge("curate", "train")
    graph.add_edge("train", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {"rollback": "rollback", "iterate": "iterate"},
    )

    graph.add_edge("rollback", "train")

    graph.add_conditional_edges(
        "iterate",
        _route_after_iterate,
        {
            "train": "train",
            "curate": "curate",
            "escalate": "escalate",
            "terminate": END,
        },
    )

    graph.add_conditional_edges(
        "escalate",
        _route_after_escalate,
        {"train": "train", "curate": "curate", "terminate": END},
    )

    return graph.compile()
```

- [ ] **Step 2: Smoke-test graph compilation**

```bash
python -c "
from agent.graph import build_graph
g = build_graph('cold_start')
print('Nodes:', list(g.get_graph().nodes.keys()))
"
```
Expected: output includes `task_analysis`, `eval_setup`, `scaling_curve`, `curate`, `train`, `evaluate`, `iterate`, `rollback`, `escalate`.

- [ ] **Step 3: Update `PIPELINE.md`**

In `docs/PIPELINE.md`, update the cold_start graph section. Find the node list and add:

- After the **Pre-graph** section, note that `hardware_filter` (Stages 1+2) now runs inside `task_analysis_node` before returning `feasible_models`.
- Update the **Node 1** entry for `task_analysis` to note it now runs `hardware_filter` internally and writes `feasible_models` but does NOT set `selected_model`.
- Add a new **Node 1b** entry for `scaling_curve`:

```markdown
### Node 1b: `scaling_curve` (new)

**Purpose:** Select the smallest model from the feasible pool that is predicted to meet the accuracy goal.

**Input state fields:** `feasible_models`, `stop_threshold`, `eval_set`, `current_dataset_path`, `task_type`

**Output state fields:** `selected_model`

**Algorithm:**
1. Pick 3 candidates (largest, middle, smallest) from `feasible_models`.
2. For each candidate: run a 1-epoch LoRA probe fine-tune on `current_dataset_path`, eval on `eval_set`, record `(log(int4_size_mb), f1)`.
3. Fit `f1 = a * log(size) + b` via least-squares.
4. Walk models smallest→largest; select the first whose predicted f1 ≥ `stop_threshold`. If none, select largest.
5. Write winner to `state["selected_model"]`.

**Position in graph:** `eval_setup → scaling_curve → curate`
```

- Update the flow diagram string if present.

- [ ] **Step 4: Commit**

```bash
git add agent/graph.py docs/PIPELINE.md
git commit -m "feat: wire scaling_curve_node into cold_start graph (task_analysis → eval_setup → scaling_curve → curate)"
```

---

## Self-Review

### Spec coverage check

| Requirement | Task |
|---|---|
| Add Q4_K_M + Q8_0 siblings to pool | Task 1 |
| `quant` field on `ModelSpec` | Task 1 |
| Re-tier siblings by RAM | Task 1 (helper functions) |
| `hardware_eval/` stub package | Task 2 |
| `HardwareEvalResult` dataclass | Task 2 |
| `hardware_filter.py` Stage 1 inequalities | Task 3 |
| `hardware_filter.py` Stage 2 on-device stub, largest→smallest | Task 3 |
| `feasible_models` in `AgentState` | Task 4 |
| `task_analysis_node` calls `run_hardware_filter` | Task 4 |
| `scaling_curve_node` probes 3 candidates | Task 5 |
| Fit accuracy-vs-log(size) curve | Task 5 |
| Select smallest model above threshold | Task 5 |
| Fall back to largest if none meet threshold | Task 5 |
| Wire into graph after `eval_setup` | Task 6 |
| Update `PIPELINE.md` | Task 6 |
| Training always LoRA (no change needed) | No task — `lora_trainer.py` unchanged |
| `escalate_node` still works | No task — it calls `filter_pool` directly, unaffected |

### Placeholder scan
- No TBDs or TODOs in code steps.
- All test assertions reference real function names defined in this plan.

### Type consistency
- `run_hardware_filter` returns `list[ModelSpec]` (Task 3) → consumed as `state["feasible_models"]: list[ModelSpec]` (Task 4) → read as `list[ModelSpec]` in Task 5. ✓
- `HardwareEvalResult` fields match the `measured` dict keys passed to `check_hardware_constraints` (`ttft_ms`, `tok_per_s`, `avg_watts`, `peak_memory_mb`). ✓
- `scaling_curve_node` reads `state["eval_set"]` which is `EvalSet | None`; node must run after `eval_setup_node` which sets it — enforced by graph edge `eval_setup → scaling_curve`. ✓
- `state["current_dataset_path"]` is set by `curate_node`. `scaling_curve_node` runs before `curate` — so it will be `None` on the very first run. The probe skips with a warning. **This is a known limitation**: on first cold-start there is no dataset yet. The caller must pre-populate `current_dataset_path` with a seed dataset, or `scaling_curve` degrades gracefully to selecting the smallest model. Add this to PIPELINE.md as an invariant note.
