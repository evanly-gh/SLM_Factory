# Honest Quantized Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `selected_model.quant` is not None, merge LoRA adapters into the BF16 checkpoint, quantize to GGUF, and score using llama-cpp-python — so every eval score in the loop reflects the exact model that will ship on-device.

**Architecture:** `run_lora_training` gains an optional `merge_and_save` step that calls `model.save_pretrained_merged()` (Unsloth) to produce a full-precision merged checkpoint. `training/quantize.py`'s existing `quantize_checkpoint()` converts that to GGUF. A new `infer_batch_gguf()` in `slm_helpers.py` runs prompts through `llama-cpp-python`. `run_eval()` in `eval/harness.py` accepts a `quant` parameter and routes to the GGUF path when set. `evaluate_node` passes `state["selected_model"].quant` into `run_eval`. Base model (`quant=None`) path is completely unchanged.

**Tech Stack:** Unsloth (`model.save_pretrained_merged`), llama-cpp-python (`llama_cpp.Llama`), existing `training/quantize.py` (`quantize_checkpoint`), LangGraph state machine.

## Global Constraints

- `quant=None` path must be byte-for-byte identical to current behavior — no regressions.
- `quant` values in the pool are exactly `"Q4_K_M"` and `"Q8_0"` — map to llama.cpp method strings `"q4_k_m"` and `"q8_0"` (lowercase).
- Merged checkpoint is written to a subdirectory of the existing `output_dir` (`<output_dir>/merged/`); GGUF is written to `<output_dir>/gguf/`.
- `infer_batch_gguf` signature mirrors `infer_batch`: `(prompts: list[str], gguf_path: str, max_new_tokens: int = 50) -> list[str]`.
- `run_eval` new signature: `(eval_set, weights_ref, base_model, task_type, quant=None, gguf_path=None)`. When `quant` is not None, `gguf_path` must be provided.
- Scorers (`eval/scorers/`) are NOT modified — they receive predictions as `list[str]` regardless of inference path.
- `evaluate_node` baseline measurement (zero-shot, `iteration == 1`) always uses the BF16 path (`quant=None`) since there is no fine-tuned GGUF yet at baseline time.
- llama-cpp-python must be importable; if not installed, `infer_batch_gguf` raises `ImportError` with a clear message.
- `run_lora_training` returns a `TrainingOutput` named tuple `(weights_ref: str, gguf_path: str | None)` instead of a plain `str`. All callers updated.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `training/lora_trainer.py` | Modify | Add `merge_for_quantization(checkpoint_path, output_dir) -> str` and `TrainingOutput` namedtuple; `run_lora_training` returns `TrainingOutput` |
| `training/slm_helpers.py` | Modify | Add `infer_batch_gguf(prompts, gguf_path, max_new_tokens) -> list[str]`; update `train()` to return `TrainingOutput` |
| `training/quantize.py` | Modify | Add `quantize_from_model_spec(checkpoint_path, output_dir, quant) -> str` convenience wrapper that maps `quant` string to llama.cpp method |
| `eval/harness.py` | Modify | `run_eval` accepts `quant=None, gguf_path=None`; routes inference to `infer_batch_gguf` when `gguf_path` is set |
| `agent/nodes/evaluate.py` | Modify | After training produces a `TrainingOutput`, quantize if `selected_model.quant` is set, then pass `gguf_path` to `run_eval` |
| `agent/nodes/train.py` | Modify | Update to handle `TrainingOutput` namedtuple instead of plain `str` |

---

### Task 1: Add `merge_for_quantization` to `lora_trainer.py` and introduce `TrainingOutput`

**Files:**
- Modify: `training/lora_trainer.py`
- Test: `tests/training/test_lora_trainer.py`

**Interfaces:**
- Produces:
  - `TrainingOutput = collections.namedtuple("TrainingOutput", ["weights_ref", "gguf_path"])` — `gguf_path` is always `None` from this file (set later by quantize step). Returned by `run_lora_training`.
  - `merge_for_quantization(checkpoint_path: str, output_dir: str) -> str` — merges LoRA adapters into base weights and saves a full HF-format directory. Returns path to merged directory.

**Why a namedtuple:** all existing callers that do `weights_ref = run_lora_training(...)` and pass `weights_ref` to `infer_batch` will break visibly (AttributeError) rather than silently. This is intentional — it forces all callers to be updated.

- [ ] **Step 1: Write failing test for `TrainingOutput`**

Create `tests/training/test_lora_trainer.py` (create `tests/training/__init__.py` too if missing):

```python
# tests/training/test_lora_trainer.py
from training.lora_trainer import TrainingOutput, merge_for_quantization
import os, tempfile, pytest

def test_training_output_is_namedtuple():
    out = TrainingOutput(weights_ref="/some/path", gguf_path=None)
    assert out.weights_ref == "/some/path"
    assert out.gguf_path is None

def test_training_output_fields():
    out = TrainingOutput(weights_ref="/a", gguf_path="/b")
    assert out[0] == "/a"
    assert out[1] == "/b"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd c:/Users/eliotli2/Documents/VSCode/SLM_Factory
python -m pytest tests/training/test_lora_trainer.py -v
```
Expected: FAIL — `ImportError: cannot import name 'TrainingOutput'`

- [ ] **Step 3: Add `TrainingOutput` and `merge_for_quantization` to `lora_trainer.py`**

Add at the top of `training/lora_trainer.py` after existing imports:

```python
import collections

TrainingOutput = collections.namedtuple("TrainingOutput", ["weights_ref", "gguf_path"])
```

Add `merge_for_quantization` function before `run_lora_training`:

```python
def merge_for_quantization(checkpoint_path: str, output_dir: str) -> str:
    """
    Merge LoRA adapters into the base model weights and save as a full HF checkpoint.
    Required before GGUF quantization — GGUF cannot be produced from adapter-only checkpoints.
    Returns path to the merged HF checkpoint directory.
    """
    from unsloth import FastLanguageModel
    merged_dir = os.path.join(output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=512,
        load_in_4bit=False,
    )
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    return merged_dir
```

Update `run_lora_training` to return `TrainingOutput`:

```python
def run_lora_training(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str = "artifacts",
    task_type: str | None = None,
) -> TrainingOutput:
    """
    Train model with LoRA (or full fine-tune if lora_rank is None).
    Returns TrainingOutput(weights_ref, gguf_path=None).
    weights_ref is a path string usable by infer() and infer_batch().
    gguf_path is always None here — set by the quantize step in evaluate_node.
    Always trains from the base model, never from a prior checkpoint.
    task_type defaults to config.task_type if not provided.
    """
    os.makedirs(output_dir, exist_ok=True)
    effective_task_type = task_type if task_type is not None else config.task_type
    checkpoint_path = _run_unsloth_training(dataset_path, config, output_dir, task_type=effective_task_type)
    return TrainingOutput(weights_ref=checkpoint_path, gguf_path=None)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/training/test_lora_trainer.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add training/lora_trainer.py tests/training/test_lora_trainer.py tests/training/__init__.py
git commit -m "feat: add TrainingOutput namedtuple and merge_for_quantization to lora_trainer"
```

---

### Task 2: Add `quantize_from_model_spec` to `quantize.py`

**Files:**
- Modify: `training/quantize.py`
- Test: `tests/training/test_quantize.py`

**Interfaces:**
- Consumes: `quantize_checkpoint(checkpoint_path, output_dir, method) -> QuantizationResult` — already exists in `training/quantize.py`
- Produces: `quantize_from_model_spec(checkpoint_path: str, output_dir: str, quant: str) -> str` — maps `quant` string to llama.cpp method, calls `quantize_checkpoint`, returns GGUF path on success, raises `RuntimeError` on failure.

**quant → method mapping:**
- `"Q4_K_M"` → `"q4_k_m"`
- `"Q8_0"` → `"q8_0"`
- anything else → `ValueError`

- [ ] **Step 1: Write failing test**

Create `tests/training/test_quantize.py`:

```python
# tests/training/test_quantize.py
import pytest
from unittest.mock import patch, MagicMock
from training.quantize import quantize_from_model_spec, QuantizationResult


def _mock_result(success=True, gguf_path="/out/model-q4_k_m.gguf", error=None):
    return QuantizationResult(
        gguf_path=gguf_path,
        original_size_mb=1000.0,
        quantized_size_mb=500.0,
        compression_ratio=2.0,
        method="q4_k_m",
        success=success,
        error=error,
    )


@patch("training.quantize.quantize_checkpoint")
def test_q4_k_m_maps_to_correct_method(mock_qc):
    mock_qc.return_value = _mock_result(gguf_path="/out/model-q4_k_m.gguf")
    result = quantize_from_model_spec("/checkpoint", "/out", "Q4_K_M")
    mock_qc.assert_called_once_with("/checkpoint", "/out", "q4_k_m")
    assert result == "/out/model-q4_k_m.gguf"


@patch("training.quantize.quantize_checkpoint")
def test_q8_0_maps_to_correct_method(mock_qc):
    mock_qc.return_value = _mock_result(gguf_path="/out/model-q8_0.gguf", method="q8_0")
    result = quantize_from_model_spec("/checkpoint", "/out", "Q8_0")
    mock_qc.assert_called_once_with("/checkpoint", "/out", "q8_0")
    assert result == "/out/model-q8_0.gguf"


def test_unknown_quant_raises_value_error():
    with pytest.raises(ValueError, match="Unknown quant"):
        quantize_from_model_spec("/checkpoint", "/out", "INT8")


@patch("training.quantize.quantize_checkpoint")
def test_failed_quantization_raises_runtime_error(mock_qc):
    mock_qc.return_value = _mock_result(success=False, gguf_path=None, error="llama-quantize not found")
    with pytest.raises(RuntimeError, match="Quantization failed"):
        quantize_from_model_spec("/checkpoint", "/out", "Q4_K_M")
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/training/test_quantize.py -v
```
Expected: FAIL — `ImportError: cannot import name 'quantize_from_model_spec'`

- [ ] **Step 3: Add `quantize_from_model_spec` to `quantize.py`**

Add at the bottom of `training/quantize.py`:

```python
_QUANT_METHOD_MAP: dict[str, str] = {
    "Q4_K_M": "q4_k_m",
    "Q8_0": "q8_0",
}


def quantize_from_model_spec(checkpoint_path: str, output_dir: str, quant: str) -> str:
    """
    Quantize a merged HF checkpoint to GGUF using the quant string from ModelSpec.

    Args:
        checkpoint_path: Path to a merged full-precision HF checkpoint directory.
        output_dir: Directory to write the GGUF file into.
        quant: ModelSpec.quant value — "Q4_K_M" or "Q8_0".

    Returns:
        Absolute path to the produced GGUF file.

    Raises:
        ValueError: if quant is not a recognized value.
        RuntimeError: if llama.cpp tools are not installed or quantization fails.
    """
    if quant not in _QUANT_METHOD_MAP:
        raise ValueError(
            f"Unknown quant {quant!r}. Valid values: {list(_QUANT_METHOD_MAP)}"
        )
    method = _QUANT_METHOD_MAP[quant]
    result = quantize_checkpoint(checkpoint_path, output_dir, method=method)
    if not result.success or result.gguf_path is None:
        raise RuntimeError(
            f"Quantization failed for {checkpoint_path!r} → {quant} ({method}): "
            f"{result.error or 'unknown error'}. "
            f"Ensure llama.cpp tools (llama-quantize, convert_hf_to_gguf) are installed."
        )
    return result.gguf_path
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/training/test_quantize.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add training/quantize.py tests/training/test_quantize.py
git commit -m "feat: add quantize_from_model_spec with quant string → llama.cpp method mapping"
```

---

### Task 3: Add `infer_batch_gguf` to `slm_helpers.py` and update `train()`

**Files:**
- Modify: `training/slm_helpers.py`
- Test: `tests/training/test_slm_helpers.py`

**Interfaces:**
- Produces: `infer_batch_gguf(prompts: list[str], gguf_path: str, max_new_tokens: int = 50) -> list[str]`
- Modifies: `train()` return type from `str` to `TrainingOutput` (now returns `run_lora_training(...)` which is already `TrainingOutput` from Task 1)

**`infer_batch_gguf` implementation:** Uses `llama_cpp.Llama` to load the GGUF, runs each prompt sequentially (same pattern as `infer_batch` — no threading). The `Llama` instance is cached by `gguf_path` in a module-level dict `_gguf_cache` (same eviction policy as `_inference_cache`: max 3 entries).

```python
_gguf_cache: dict = {}
_gguf_cache_order: list = []
```

Llama constructor options: `n_ctx=512`, `n_gpu_layers=0` (CPU only — on-device target), `verbose=False`.

Inference: `llama(prompt, max_tokens=max_new_tokens, echo=False)["choices"][0]["text"]`

- [ ] **Step 1: Write failing test**

Create `tests/training/test_slm_helpers.py`:

```python
# tests/training/test_slm_helpers.py
import pytest
from unittest.mock import patch, MagicMock
from training.slm_helpers import infer_batch_gguf
from training.lora_trainer import TrainingOutput


def test_infer_batch_gguf_raises_import_error_when_llama_cpp_missing():
    with patch.dict("sys.modules", {"llama_cpp": None}):
        with pytest.raises(ImportError, match="llama-cpp-python"):
            infer_batch_gguf(["hello"], "/fake/model.gguf")


def test_infer_batch_gguf_returns_list_of_strings():
    mock_llama_instance = MagicMock()
    mock_llama_instance.return_value = {"choices": [{"text": "spam"}]}
    mock_llama_cls = MagicMock(return_value=mock_llama_instance)

    with patch("training.slm_helpers._gguf_cache", {}), \
         patch("training.slm_helpers._gguf_cache_order", []):
        with patch.dict("sys.modules", {"llama_cpp": MagicMock(Llama=mock_llama_cls)}):
            # Clear module-level cache to force load
            import training.slm_helpers as sh
            sh._gguf_cache.clear()
            sh._gguf_cache_order.clear()
            result = infer_batch_gguf(["hello", "world"], "/fake/model.gguf")

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(s, str) for s in result)


def test_train_returns_training_output():
    from training.slm_helpers import train
    with patch("training.slm_helpers.run_lora_training") as mock_train:
        mock_train.return_value = TrainingOutput(weights_ref="/ckpt", gguf_path=None)
        result = train("/data.jsonl", "model-id", 1, 2e-4, 8, 8)
    assert isinstance(result, TrainingOutput)
    assert result.weights_ref == "/ckpt"
    assert result.gguf_path is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/training/test_slm_helpers.py -v
```
Expected: FAIL — `ImportError: cannot import name 'infer_batch_gguf'`

- [ ] **Step 3: Add `infer_batch_gguf` and update `train()` in `slm_helpers.py`**

Replace the entire `training/slm_helpers.py` with:

```python
# training/slm_helpers.py
"""
Agent-facing interface for training and inference.
Called by the agent via the bash tool.
Equivalent to the paper's tinker_helpers.py.
"""
import os
import json
from training.lora_trainer import TrainingConfig, TrainingOutput, run_lora_training

# Lazy-loaded inference model cache: weights_ref -> (model, tokenizer).
# Limited to _MAX_CACHED models to avoid OOM on long runs (B47).
_inference_cache: dict = {}
_cache_order: list = []
_MAX_CACHED = 3

# GGUF inference cache: gguf_path -> Llama instance.
_gguf_cache: dict = {}
_gguf_cache_order: list = []


def train(
    dataset_path: str,
    base_model: str,
    nr_epochs: int,
    learning_rate: float,
    batch_size: int,
    lora_rank: int | None,
    output_dir: str = "artifacts",
    task_type: str = "classification",
) -> TrainingOutput:
    """
    Execute the full LoRA training loop.
    Returns TrainingOutput(weights_ref, gguf_path=None).
    Always trains from base_model — never from a prior checkpoint.
    task_type controls the prompt format used during training.
    """
    config = TrainingConfig(
        base_model=base_model,
        nr_epochs=nr_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lora_rank=lora_rank,
        task_type=task_type,
    )
    return run_lora_training(dataset_path, config, output_dir=output_dir, task_type=task_type)


def infer(prompt: str, weights_ref: str, base_model: str, max_new_tokens: int = 50) -> str:
    """
    Load a checkpoint and generate text. Loads via Unsloth (not vanilla
    AutoModelForCausalLM): once `unsloth` is imported it globally patches the model
    classes (e.g. Qwen3Attention.apply_qkv), so a vanilla-loaded model crashes at
    generate. Unsloth's loader also transparently handles LoRA-adapter checkpoints.
    """
    import torch
    if weights_ref not in _inference_cache:
        from unsloth import FastLanguageModel
        # Evict oldest cached model if at capacity
        while len(_inference_cache) >= _MAX_CACHED and _cache_order:
            evict_key = _cache_order.pop(0)
            old = _inference_cache.pop(evict_key, None)
            if old:
                del old
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=weights_ref, max_seq_length=512, load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)
        _inference_cache[weights_ref] = (model, tokenizer)
        _cache_order.append(weights_ref)

    model, tokenizer = _inference_cache[weights_ref]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def infer_batch(
    prompts: list[str],
    weights_ref: str,
    base_model: str,
    max_new_tokens: int = 50,
    max_workers: int = 20,
) -> list[str]:
    """
    Run inference over all prompts. SEQUENTIAL: Unsloth/torch keep process-global state and
    a single GPU serializes the work anyway, so threading the model is unsafe and pointless
    (see BUGS.md B18). `max_workers` is kept for signature compatibility and ignored.
    """
    return [infer(p, weights_ref, base_model, max_new_tokens) for p in prompts]


def infer_batch_gguf(
    prompts: list[str],
    gguf_path: str,
    max_new_tokens: int = 50,
) -> list[str]:
    """
    Run inference over all prompts using a GGUF file via llama-cpp-python.
    Used for quantized model evaluation (Q4_K_M, Q8_0) to get honest on-device
    accuracy scores — the same model format that ships on Android.

    Sequential inference only (same reasoning as infer_batch).
    Caches the Llama instance by gguf_path (max 3 entries).

    Raises:
        ImportError: if llama-cpp-python is not installed.
    """
    try:
        import llama_cpp
    except (ImportError, TypeError):
        raise ImportError(
            "llama-cpp-python is required for GGUF inference. "
            "Install with: pip install llama-cpp-python"
        )

    if gguf_path not in _gguf_cache:
        while len(_gguf_cache) >= _MAX_CACHED and _gguf_cache_order:
            evict_key = _gguf_cache_order.pop(0)
            _gguf_cache.pop(evict_key, None)
        llama = llama_cpp.Llama(
            model_path=gguf_path,
            n_ctx=512,
            n_gpu_layers=0,  # CPU-only: matches Android on-device inference
            verbose=False,
        )
        _gguf_cache[gguf_path] = llama
        _gguf_cache_order.append(gguf_path)

    llama = _gguf_cache[gguf_path]
    results = []
    for prompt in prompts:
        response = llama(prompt, max_tokens=max_new_tokens, echo=False)
        results.append(response["choices"][0]["text"])
    return results
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/training/test_slm_helpers.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add training/slm_helpers.py tests/training/test_slm_helpers.py
git commit -m "feat: add infer_batch_gguf with llama-cpp-python, update train() to return TrainingOutput"
```

---

### Task 4: Update `run_eval` in `eval/harness.py` to accept GGUF path

**Files:**
- Modify: `eval/harness.py`
- Test: `tests/eval/test_harness.py`

**Interfaces:**
- Consumes: `infer_batch_gguf(prompts, gguf_path, max_new_tokens) -> list[str]` from Task 3
- Produces: updated `run_eval(eval_set, weights_ref, base_model, task_type, quant=None, gguf_path=None) -> EvalResult`. When `gguf_path` is not None, uses `infer_batch_gguf` instead of `infer_batch`. `weights_ref` and `base_model` are unused in the GGUF path but kept for interface stability.

- [ ] **Step 1: Write failing test**

Create `tests/eval/test_harness.py` (create `tests/eval/__init__.py` too):

```python
# tests/eval/test_harness.py
import pytest
from unittest.mock import patch, MagicMock
from eval.harness import run_eval
from data.eval_set import EvalSet


def _make_eval_set():
    es = MagicMock(spec=EvalSet)
    es.task_type = "classification"
    es.multi_label = False
    return es


@patch("eval.harness.infer_batch")
def test_run_eval_uses_infer_batch_when_no_gguf(mock_infer):
    mock_infer.return_value = ["spam", "ham"]
    scorer_mock = MagicMock()
    scorer_mock.build_prompts.return_value = ["p1", "p2"]
    scorer_mock.extract_predictions.return_value = ["spam", "ham"]
    scorer_mock.score.return_value = {
        "f1": 0.9, "per_class": {}, "slices": {"pos": 1.0, "neg": 0.8, "boundary": 0.9}, "failures": []
    }
    with patch("eval.harness.classification", scorer_mock):
        result = run_eval(_make_eval_set(), "/weights", "model-id", "classification")
    mock_infer.assert_called_once()
    assert result.f1 == 0.9


@patch("eval.harness.infer_batch_gguf")
def test_run_eval_uses_infer_batch_gguf_when_gguf_path_set(mock_gguf):
    mock_gguf.return_value = ["spam", "ham"]
    scorer_mock = MagicMock()
    scorer_mock.build_prompts.return_value = ["p1", "p2"]
    scorer_mock.extract_predictions.return_value = ["spam", "ham"]
    scorer_mock.score.return_value = {
        "f1": 0.85, "per_class": {}, "slices": {"pos": 0.9, "neg": 0.8, "boundary": 0.85}, "failures": []
    }
    with patch("eval.harness.classification", scorer_mock):
        result = run_eval(
            _make_eval_set(), "/weights", "model-id", "classification",
            quant="Q4_K_M", gguf_path="/model.gguf"
        )
    mock_gguf.assert_called_once_with(["p1", "p2"], "/model.gguf", 50)
    assert result.f1 == 0.85


def test_run_eval_raises_on_unknown_task_type():
    with pytest.raises(ValueError, match="Unknown task_type"):
        run_eval(_make_eval_set(), "/w", "m", "unknown_task")
```

- [ ] **Step 2: Run to verify it fails**

```bash
python -m pytest tests/eval/test_harness.py -v
```
Expected: FAIL — `TypeError` or assertion failure (current `run_eval` has no `quant` param).

- [ ] **Step 3: Update `eval/harness.py`**

```python
# eval/harness.py
from dataclasses import dataclass
from data.eval_set import EvalSet
from training.slm_helpers import infer_batch, infer_batch_gguf


@dataclass
class EvalResult:
    f1: float
    per_class: dict
    pos_score: float
    neg_score: float
    boundary_score: float
    failures: list[dict]


def run_eval(
    eval_set: EvalSet,
    weights_ref: str,
    base_model: str,
    task_type: str,
    quant: str | None = None,
    gguf_path: str | None = None,
) -> EvalResult:
    """
    Run inference on E and compute task-type-appropriate metrics.

    When gguf_path is provided (quantized model path), inference uses
    llama-cpp-python (infer_batch_gguf) to get honest on-device accuracy.
    When gguf_path is None (base/BF16 model), uses Unsloth (infer_batch).

    Dispatches to eval/scorers/{task_type}.py for prompting, extraction, scoring.
    Scorers are inference-backend agnostic — they receive list[str] predictions.
    """
    if task_type == "classification":
        from eval.scorers import classification as scorer
    elif task_type == "NER":
        from eval.scorers import ner as scorer
    elif task_type in ("math_reasoning", "code_generation", "generation"):
        from eval.scorers import generation as scorer
    else:
        raise ValueError(
            f"Unknown task_type: {task_type!r}. "
            f"Must be one of: classification, NER, math_reasoning, code_generation, generation."
        )

    prompts = scorer.build_prompts(eval_set)

    if gguf_path is not None:
        raw_outputs = infer_batch_gguf(prompts, gguf_path, max_new_tokens=50)
    else:
        raw_outputs = infer_batch(prompts, weights_ref, base_model, max_workers=20)

    predictions = scorer.extract_predictions(raw_outputs, eval_set)
    result = scorer.score(eval_set, predictions)

    return EvalResult(
        f1=result["f1"],
        per_class=result["per_class"],
        pos_score=result["slices"]["pos"],
        neg_score=result["slices"]["neg"],
        boundary_score=result["slices"]["boundary"],
        failures=result["failures"],
    )
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/eval/test_harness.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/harness.py tests/eval/test_harness.py tests/eval/__init__.py
git commit -m "feat: run_eval accepts gguf_path, routes to infer_batch_gguf for quantized models"
```

---

### Task 5: Update `train_node` and `evaluate_node` to drive the quantize→eval flow

**Files:**
- Modify: `agent/nodes/train.py`
- Modify: `agent/nodes/evaluate.py`
- Test: `tests/nodes/test_evaluate_node.py`

**Interfaces:**
- Consumes from Task 1: `TrainingOutput(weights_ref, gguf_path)` returned by `run_lora_training` / `train()`
- Consumes from Task 2: `merge_for_quantization(checkpoint_path, output_dir) -> str`; `quantize_from_model_spec(checkpoint_path, output_dir, quant) -> str`
- Consumes from Task 4: `run_eval(..., quant=None, gguf_path=None)`

**`train_node` change:** `train_node` currently stores `weights_ref` strings into `state["_pending_weights_refs"]`. It must now store `TrainingOutput` objects (or just the `.weights_ref` string — see below). The cleanest approach: store the full `TrainingOutput` in a new state field `_pending_training_outputs: dict[str, TrainingOutput]`, keep `_pending_weights_refs` populated from `.weights_ref` for backward-compat with DAG logging in `evaluate_node`.

**`evaluate_node` change:** After training produces weights, if `state["selected_model"].quant` is not None:
1. Call `merge_for_quantization(weights_ref, output_dir)` to get `merged_path`
2. Call `quantize_from_model_spec(merged_path, gguf_output_dir, quant)` to get `gguf_path`
3. Pass `quant` and `gguf_path` into `run_eval`

Baseline measurement (iteration == 1) always uses `quant=None, gguf_path=None` — no GGUF for base model zero-shot.

Output dirs: `merged_path = "artifacts/merged/{model_id_safe}/{label}"`, `gguf_output_dir = "artifacts/gguf/{model_id_safe}/{label}"` where `model_id_safe = model_id.replace("/", "_")`.

First read `agent/nodes/train.py` to understand its current structure before writing the task steps.

- [ ] **Step 1: Read `agent/nodes/train.py`**

```bash
cat c:/Users/eliotli2/Documents/VSCode/SLM_Factory/agent/nodes/train.py
```

Verify it stores results into `state["_pending_weights_refs"]`. The update is: where it currently does `state["_pending_weights_refs"][label] = run_lora_training(...)`, it now does:
```python
output = run_lora_training(...)   # returns TrainingOutput
state["_pending_weights_refs"][label] = output.weights_ref
state["_pending_training_outputs"][label] = output
```
And `state["_pending_training_outputs"]` must be initialized to `{}` at the start of `train_node` if not present.

- [ ] **Step 2: Add `_pending_training_outputs` to `AgentState`**

In `agent/state.py`, add after `_pending_weights_refs`:
```python
_pending_training_outputs: Optional[dict]  # label -> TrainingOutput
```

- [ ] **Step 3: Update `train_node` in `agent/nodes/train.py`**

Find every line that assigns to `state["_pending_weights_refs"][label]` and update to also populate `_pending_training_outputs`. Initialize `state["_pending_training_outputs"] = {}` alongside `state["_pending_weights_refs"] = {}` at the top of the node.

The exact change depends on `train.py`'s current structure — read it first (Step 1), then apply the minimal change to store `TrainingOutput` objects.

- [ ] **Step 4: Write failing test for `evaluate_node` quantized path**

Create `tests/nodes/test_evaluate_node.py` (create `tests/nodes/__init__.py` too):

```python
# tests/nodes/test_evaluate_node.py
import pytest
from unittest.mock import patch, MagicMock, call
from config.android_pool import ModelSpec, HardwareConstraints
from training.lora_trainer import TrainingOutput
from eval.harness import EvalResult


def _make_model(quant=None):
    return ModelSpec(
        model_id="test/Model-1B",
        int4_size_mb=700,
        tier=1,
        tok_s_snapdragon_660=8.0,
        tok_s_snapdragon_778g=14.0,
        tok_s_snapdragon_8gen3=38.0,
        peak_memory_mb=1100,
        gsm8k=0.6,
        mmlu=0.5,
        quant=quant,
    )


def _make_state(quant=None):
    from data.eval_set import EvalSet
    return {
        "task_type": "classification",
        "selected_model": _make_model(quant=quant),
        "eval_set": MagicMock(spec=EvalSet),
        "iteration": 2,
        "best_score": 0.0,
        "scores": [],
        "dag": [],
        "dataset_version": 1,
        "current_dataset_path": "/data.jsonl",
        "hardware_constraints": HardwareConstraints(
            storage_mb=5000, memory_mb=5000, latency_ttft_ms=3000,
        ),
        "model_baselines": [],
        "last_curation": {},
        "last_hypothesis": "",
        "_pending_weights_refs": {"main": "/ckpt"},
        "_pending_training_outputs": {"main": TrainingOutput(weights_ref="/ckpt", gguf_path=None)},
        "_pending_configs": {"main": {"label": "main", "lora_rank": 8, "learning_rate": 2e-4, "nr_epochs": 3, "batch_size": 8}},
        "consecutive_no_improvement": 0,
    }


def _mock_eval_result(f1=0.85):
    return EvalResult(f1=f1, per_class={}, pos_score=0.9, neg_score=0.8, boundary_score=0.85, failures=[])


@patch("agent.nodes.evaluate.run_eval")
@patch("agent.nodes.evaluate.CurationLog")
@patch("agent.nodes.evaluate.theoretical_hardware_profile", return_value={"int4_size_mb": 700, "tier": 1})
@patch("agent.nodes.evaluate.check_hardware_constraints", return_value={"storage": {"pass": True}, "memory": {"pass": True}, "latency": {"pass": True}, "power": {"pass": True}})
def test_evaluate_node_uses_bf16_path_when_quant_none(mock_hw, mock_profile, mock_log, mock_eval):
    mock_eval.return_value = _mock_eval_result()
    mock_log.return_value.write_iteration = MagicMock()
    from agent.nodes.evaluate import evaluate_node
    state = _make_state(quant=None)
    evaluate_node(state)
    # run_eval called without gguf_path
    mock_eval.assert_called_once()
    call_kwargs = mock_eval.call_args[1]
    assert call_kwargs.get("gguf_path") is None
    assert call_kwargs.get("quant") is None


@patch("agent.nodes.evaluate.quantize_from_model_spec", return_value="/gguf/model.gguf")
@patch("agent.nodes.evaluate.merge_for_quantization", return_value="/merged/checkpoint")
@patch("agent.nodes.evaluate.run_eval")
@patch("agent.nodes.evaluate.CurationLog")
@patch("agent.nodes.evaluate.theoretical_hardware_profile", return_value={"int4_size_mb": 700, "tier": 1})
@patch("agent.nodes.evaluate.check_hardware_constraints", return_value={"storage": {"pass": True}, "memory": {"pass": True}, "latency": {"pass": True}, "power": {"pass": True}})
def test_evaluate_node_quantizes_and_uses_gguf_path_when_quant_set(
    mock_hw, mock_profile, mock_log, mock_eval, mock_merge, mock_quantize
):
    mock_eval.return_value = _mock_eval_result()
    mock_log.return_value.write_iteration = MagicMock()
    from agent.nodes.evaluate import evaluate_node
    state = _make_state(quant="Q4_K_M")
    evaluate_node(state)
    # merge was called
    mock_merge.assert_called_once_with("/ckpt", pytest.approx(str, abs=0))
    # quantize was called with correct quant string
    mock_quantize.assert_called_once()
    assert mock_quantize.call_args[0][2] == "Q4_K_M"
    # run_eval called with gguf_path
    mock_eval.assert_called_once()
    call_kwargs = mock_eval.call_args[1]
    assert call_kwargs.get("gguf_path") == "/gguf/model.gguf"
    assert call_kwargs.get("quant") == "Q4_K_M"
```

- [ ] **Step 5: Run to verify it fails**

```bash
python -m pytest tests/nodes/test_evaluate_node.py -v
```
Expected: FAIL — imports succeed but assertions fail (no quantize logic in evaluate_node yet).

- [ ] **Step 6: Update `evaluate_node` in `agent/nodes/evaluate.py`**

Add imports at the top:
```python
from training.lora_trainer import merge_for_quantization
from training.quantize import quantize_from_model_spec
```

In the `# --- Score all trained configs ---` loop, replace:
```python
result = run_eval(eval_set, weights_ref, model_id, task_type=task_type)
```
with:
```python
quant = state["selected_model"].quant
gguf_path = None
if quant is not None:
    model_id_safe = model_id.replace("/", "_")
    merged_path = merge_for_quantization(
        weights_ref,
        os.path.join("artifacts", "merged", model_id_safe, label),
    )
    gguf_path = quantize_from_model_spec(
        merged_path,
        os.path.join("artifacts", "gguf", model_id_safe, label),
        quant,
    )
result = run_eval(eval_set, weights_ref, model_id, task_type=task_type, quant=quant, gguf_path=gguf_path)
```

Also add `import os` at the top of `evaluate.py` if not already present.

- [ ] **Step 7: Run all tests**

```bash
python -m pytest tests/nodes/test_evaluate_node.py tests/training/ tests/eval/ -v
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add agent/nodes/evaluate.py agent/nodes/train.py agent/state.py \
        tests/nodes/test_evaluate_node.py tests/nodes/__init__.py
git commit -m "feat: evaluate_node merges + quantizes checkpoint before eval when model.quant is set"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|---|---|
| Merge LoRA adapters before quantization | Task 1 (`merge_for_quantization`) |
| `quant` → llama.cpp method mapping | Task 2 (`quantize_from_model_spec`) |
| GGUF inference via llama-cpp-python | Task 3 (`infer_batch_gguf`) |
| `run_eval` routes to GGUF path when `gguf_path` set | Task 4 |
| `evaluate_node` drives merge→quantize→eval for quantized models | Task 5 |
| `train_node` updated for `TrainingOutput` | Task 5 |
| `quant=None` path completely unchanged | Task 4 (default params), Task 5 (baseline always BF16) |
| llama-cpp-python raises `ImportError` with clear message | Task 3 |
| `TrainingOutput` namedtuple returned by `run_lora_training` | Task 1 |

### Placeholder scan
None found.

### Type consistency
- `run_lora_training` returns `TrainingOutput` (Task 1) → `train()` returns `TrainingOutput` (Task 3) → `train_node` stores `.weights_ref` into `_pending_weights_refs` and full `TrainingOutput` into `_pending_training_outputs` (Task 5) → `evaluate_node` reads `.weights_ref` from `_pending_weights_refs` (unchanged) and calls `merge_for_quantization(weights_ref, ...)` (Task 5). ✓
- `quantize_from_model_spec` returns `str` (gguf_path) (Task 2) → `run_eval(..., gguf_path=str)` (Task 4) → `infer_batch_gguf(prompts, gguf_path: str)` (Task 3). ✓
- `infer_batch_gguf` signature: `(prompts: list[str], gguf_path: str, max_new_tokens: int = 50) -> list[str]`. Called in `run_eval` as `infer_batch_gguf(prompts, gguf_path, max_new_tokens=50)`. ✓

### Known gap
`train_node` structure was not read before writing Task 5 Step 3 — the implementer must read `agent/nodes/train.py` first (Step 1) to apply the minimal change correctly. This is called out explicitly in Step 1 and Step 3.
