# Qwen3.6-first CoT Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local Qwen3.6 the primary CoT author, use task-aware DeepSeek/OpenAI fallback chains, and eliminate Claude/orchestrator from CoT generation.

**Architecture:** `data.curriculum` will expose an ordered fallback builder containing only OpenAI-compatible specialist clients and make `annotate_cot` execute a per-example backend chain: Qwen first, then task-aware fallbacks. `agent.nodes.curate` will only assemble that chain and log it; it will never construct or receive an Anthropic CoT client.

**Tech Stack:** Python 3.11, Anthropic-independent OpenAI-compatible clients, local vLLM OpenAI endpoint, pytest, `ThreadPoolExecutor`.

## Global Constraints

- Qwen3.6 is always the primary CoT backend.
- Math/science fallback order: DeepSeek, then OpenAI.
- Code/QA/general-generation fallback order: OpenAI, then DeepSeek.
- Claude/orchestrator must never be called by the CoT path.
- Missing keys, exceptions, and empty output advance to the next backend.
- If every backend fails, preserve the original example without CoT.
- Preserve existing non-empty gold `cot_reasoning`.
- Cheap mode continues to skip CoT annotation.
- Do not create a git commit unless the user explicitly requests one.

---

### Task 1: Specialist-only fallback builder

**Files:**
- Modify: `data/curriculum.py:1-72`
- Test: `tests/test_curriculum_hardneg.py`

**Interfaces:**
- Produces: `get_cot_fallbacks(task_type: str, benchmark: str | None = None) -> list[tuple[object, str]]`
- Each tuple is `(openai_compatible_client, model_name)`.
- Never imports `anthropic`, `ANTHROPIC_API_KEY`, `TEACHER_MODEL_CLAUDE`, or `ORCHESTRATOR_MODEL`.

- [ ] **Step 1: Write failing routing tests**

Add tests using monkeypatched config values and a stubbed `openai.OpenAI` constructor:

```python
def test_math_cot_fallback_order_is_deepseek_then_openai(monkeypatch):
    import config.config as config
    from data import curriculum

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "deep-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "open-key")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: kwargs)
    fallbacks = curriculum.get_cot_fallbacks("math_reasoning", "gsm8k")
    assert [model for _, model in fallbacks] == [
        config.TEACHER_MODEL_DEEPSEEK,
        config.TEACHER_MODEL_GPT,
    ]


def test_general_cot_fallback_order_is_openai_then_deepseek(monkeypatch):
    import config.config as config
    from data import curriculum

    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "deep-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "open-key")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: kwargs)
    fallbacks = curriculum.get_cot_fallbacks("generation", "samsum")
    assert [model for _, model in fallbacks] == [
        config.TEACHER_MODEL_GPT,
        config.TEACHER_MODEL_DEEPSEEK,
    ]
```

Add an AST/source assertion that the function body contains no Anthropic construction:

```python
def test_cot_fallback_builder_has_no_anthropic_backend():
    import inspect
    from data import curriculum

    source = inspect.getsource(curriculum.get_cot_fallbacks)
    assert "anthropic" not in source.lower()
    assert "ORCHESTRATOR_MODEL" not in source
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
source .venv_gpu/bin/activate
python -m pytest tests/test_curriculum_hardneg.py -k "fallback_order or no_anthropic" -q
```

Expected: FAIL because `get_cot_fallbacks` does not exist.

- [ ] **Step 3: Implement the specialist-only builder**

Replace `get_teacher_client` with:

```python
def get_cot_fallbacks(task_type: str, benchmark: str | None = None) -> list[tuple[object, str]]:
    from config.config import (
        DEEPSEEK_API_KEY,
        DEEPSEEK_BASE_URL,
        TEACHER_MODEL_DEEPSEEK,
        OPENAI_API_KEY,
        TEACHER_MODEL_GPT,
    )
    from openai import OpenAI

    deepseek = (
        (OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL), TEACHER_MODEL_DEEPSEEK)
        if DEEPSEEK_API_KEY else None
    )
    openai = (
        (OpenAI(api_key=OPENAI_API_KEY), TEACHER_MODEL_GPT)
        if OPENAI_API_KEY else None
    )
    math_like = task_type == "math_reasoning" or (
        task_type == "generation"
        and (benchmark or "").lower().replace(" ", "_") in _MATH_SCIENCE_BENCHMARKS
    )
    ordered = [deepseek, openai] if math_like else [openai, deepseek]
    return [backend for backend in ordered if backend is not None]
```

Remove the module-level `TEACHER_MODEL_CLAUDE` import and update comments/docstrings to describe Qwen-primary specialist fallbacks.

- [ ] **Step 4: Run routing tests and verify GREEN**

Run the command from Step 2.

Expected: all selected tests PASS.

---

### Task 2: Per-example Qwen → fallback execution

**Files:**
- Modify: `data/curriculum.py:74-172`
- Test: `tests/test_curriculum_hardneg.py`

**Interfaces:**
- Modify: `annotate_cot(examples, task_type="generation", generate_fn=None, fallback_teachers=None, log=print) -> list[dict]`
- `generate_fn(prompt, temperature, max_tokens) -> str` is the local Qwen primary.
- `fallback_teachers` is an ordered `list[tuple[openai_compatible_client, model_name]]`.

- [ ] **Step 1: Write failing backend-chain tests**

Add focused tests:

```python
def test_qwen_success_prevents_cloud_fallback():
    fallback = MagicMock()
    out = annotate_cot(
        [{"prompt": "p", "response": "r"}],
        generate_fn=lambda *_: "local reasoning",
        fallback_teachers=[(fallback, "gpt-4.1")],
    )
    assert out[0]["cot_reasoning"] == "local reasoning"
    fallback.chat.completions.create.assert_not_called()


def test_empty_qwen_output_uses_first_fallback():
    fallback = MagicMock()
    fallback.chat.completions.create.return_value.choices[0].message.content = "cloud reasoning"
    out = annotate_cot(
        [{"prompt": "p", "response": "r"}],
        generate_fn=lambda *_: "",
        fallback_teachers=[(fallback, "deepseek-reasoner")],
    )
    assert out[0]["cot_reasoning"] == "cloud reasoning"


def test_failed_first_fallback_advances_to_second():
    first, second = MagicMock(), MagicMock()
    first.chat.completions.create.side_effect = RuntimeError("down")
    second.chat.completions.create.return_value.choices[0].message.content = "second reasoning"
    out = annotate_cot(
        [{"prompt": "p", "response": "r"}],
        generate_fn=lambda *_: (_ for _ in ()).throw(RuntimeError("local down")),
        fallback_teachers=[(first, "deepseek-reasoner"), (second, "gpt-4.1")],
    )
    assert out[0]["cot_reasoning"] == "second reasoning"
```

Keep the existing tests for all-backends-failed and gold-CoT preservation.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tests/test_curriculum_hardneg.py -k "qwen_success or empty_qwen or advances_to_second" -q
```

Expected: FAIL because `fallback_teachers` is not accepted and Qwen failures do not advance.

- [ ] **Step 3: Implement the minimal backend chain**

Inside `_annotate_one`, build an ordered attempt sequence:

```python
attempts = []
if generate_fn is not None:
    attempts.append(("Qwen3.6", lambda: generate_fn(cot_prompt, 0.3, 512)))
for client, model in fallback_teachers or []:
    attempts.append((
        model,
        lambda client=client, model=model: client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": cot_prompt}],
            max_tokens=500,
        ).choices[0].message.content,
    ))

for backend, call in attempts:
    try:
        cot = (call() or "").strip()
        if cot:
            return {**ex, "cot_reasoning": cot}, backend
    except Exception:
        continue
return ex, None
```

Map these tuple results through the existing thread pool, preserve input order, aggregate backend counts after joining, and emit one log line:

```python
log(f"      [cot] annotated={sum(counts.values())}/{len(need_idx)} by_backend={dict(counts)}")
```

Delete the Anthropic `teacher_client.messages.create` branch. Preserve the original example if no attempt succeeds.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2.

Expected: all selected tests PASS.

---

### Task 3: Curate integration and regression verification

**Files:**
- Modify: `agent/nodes/curate.py:1-12,191-219`
- Modify: `config/config.py:15-24,56-64` documentation only
- Test: `tests/test_curriculum_hardneg.py`

**Interfaces:**
- Consumes: `get_cot_fallbacks(...)` and the revised `annotate_cot(...)`.
- Produces: one Qwen-primary annotation call with task-aware fallback order.

- [ ] **Step 1: Write a failing source-level regression test**

```python
def test_curate_cot_path_has_no_orchestrator_teacher():
    import inspect
    from agent.nodes import curate

    source = inspect.getsource(curate.curate_node)
    assert "get_teacher_client" not in source
    assert "teacher_client" not in source
    assert "get_cot_fallbacks" in source
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest tests/test_curriculum_hardneg.py::test_curate_cot_path_has_no_orchestrator_teacher -q
```

Expected: FAIL because curate still calls `get_teacher_client`.

- [ ] **Step 3: Rewire curate**

Replace the current specialist-first branching with:

```python
fallbacks = get_cot_fallbacks(task_type, benchmark)
_cot_gen = (
    get_generate_fn(log=lambda m: _log(model_id, m))
    if is_available(log=lambda m: _log(model_id, m))
    else None
)
from config.config import SYNTH_MODEL
fallback_names = [model for _, model in fallbacks]
_log(
    model_id,
    f"  CoT annotation: primary=LOCAL {SYNTH_MODEL}; "
    f"fallbacks={fallback_names or ['none']} (Claude/orchestrator disabled)",
)
gold = annotate_cot(
    gold,
    task_type=task_type,
    generate_fn=_cot_gen,
    fallback_teachers=fallbacks,
    log=lambda m: _log(model_id, m),
)
```

Update imports from `get_teacher_client` to `get_cot_fallbacks`. Update config comments so they no longer claim Claude is the CoT fallback.

- [ ] **Step 4: Run focused and regression suites**

Run:

```bash
python -m pytest tests/test_curriculum_hardneg.py -q
python -m pytest tests/nodes/ tests/cold_start/ tests/test_curriculum_hardneg.py -q
```

Expected: focused suite PASS; regression suite PASS with no warnings/errors introduced by this change.

- [ ] **Step 5: Check edited-file diagnostics**

Check lints for:

```text
data/curriculum.py
agent/nodes/curate.py
config/config.py
tests/test_curriculum_hardneg.py
```

Expected: no new diagnostics.
