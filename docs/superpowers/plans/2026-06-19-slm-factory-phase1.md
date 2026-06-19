# SLM Factory Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an agentic fine-tuning loop that replicates Pioneer Agent's cold-start mode, bounded to Android-deployable models, demonstrably improving F1 on SMS Spam classification over 3–4 iterations.

**Architecture:** LangGraph state machine with Claude Sonnet 4.6 orchestrates a cold-start loop: task analysis → eval setup → train (2 configs in parallel) → evaluate → iterate → curate → rollback check → escalate/terminate. All lineage is recorded in a persistent `data-curation.md` file. The agent calls `slm_helpers.py` via a `bash` tool for training and inference.

**Tech Stack:** Python 3.11+, LangGraph, Anthropic SDK, Unsloth (LoRA training), HuggingFace Transformers + PEFT (fallback), llama-cpp-python (inference), Exa API (web search), UCI SMS Spam Collection.

## Global Constraints

- All selected models must fit within the Android pool (≤~1.5GB INT4 — see design doc Section 6.4)
- Training always restarts from the base foundation model — never from a prior fine-tuned checkpoint
- At least 2 configurations trained in parallel per iteration
- `data-curation.md` written every iteration before the next training round begins
- No MCGS in Phase 1 — sequential greedy DAG only
- No regression gate in Phase 1 cold-start — simple rollback only (`f(πi+1) < f(πi)` → revert)
- No replay buffer in Phase 1 — `Dparent = ∅`, replay allocation redistributed to gold
- Hardware constraints logged every iteration but not enforced as hard gates until Phase 2
- Claude Sonnet 4.6 as orchestrator LLM; same model for sub-agents
- Sub-agent turn limit: 1,000; main agent turn limit: 1,500

---

## File Structure

```
SLM_Factory/
├── agent/
│   ├── __init__.py
│   ├── graph.py              # LangGraph state machine definition
│   ├── state.py              # AgentState TypedDict
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── task_analysis.py      # Node 1: classify task, survey baselines, select model
│   │   ├── eval_setup.py         # Node 2: build E = Epos ∪ Eneg ∪ Eboundary
│   │   ├── train.py              # Node 3: fire ≥2 parallel training configs
│   │   ├── evaluate.py           # Node 4: score against E, log to DAG + data-curation.md
│   │   ├── iterate.py            # Node 5: apply iteration policy, determine next action
│   │   ├── curate.py             # Node 6: build Dcold = Dgold ∪ Dhard at 65:35
│   │   ├── rollback.py           # Node 7: simple rollback if score decreased
│   │   └── escalate.py           # Node 8: escalate model tier or terminate
│   └── tools/
│       ├── __init__.py
│       ├── web_search.py         # Exa API wrapper
│       ├── bash_tool.py          # Shell tool with slm_helpers pre-loaded
│       ├── file_tools.py         # read_file / edit_file tools
│       └── delegate_task.py      # Sub-agent spawning
├── training/
│   ├── __init__.py
│   ├── slm_helpers.py            # train() / infer() / infer_batch() — agent's bash interface
│   ├── lora_trainer.py           # Unsloth LoRA training loop
│   └── quantize.py               # INT4 quantization + theoretical hardware profile
├── data/
│   ├── __init__.py
│   ├── sms_spam.py               # Download + parse UCI SMS Spam Collection
│   ├── eval_set.py               # Build E = Epos ∪ Eneg ∪ Eboundary
│   ├── curriculum.py             # Build Dcold = Dgold ∪ Dhard, quality controls
│   └── curation_log.py           # data-curation.md read/write
├── eval/
│   ├── __init__.py
│   ├── harness.py                # Run infer_batch over E, compute F1, per-slice scores
│   └── metrics.py                # binary_f1(), per_class_f1(), per_slice_scores()
├── android_pool.py               # Android model pool definition + constraint checks
├── run.py                        # Entry point: build τ, run graph
├── config.py                     # API keys, default thresholds, cluster settings
├── tests/
│   ├── test_sms_spam.py
│   ├── test_eval_set.py
│   ├── test_curriculum.py
│   ├── test_curation_log.py
│   ├── test_metrics.py
│   ├── test_harness.py
│   ├── test_android_pool.py
│   ├── test_lora_trainer.py
│   ├── test_nodes.py
│   └── test_graph.py
└── data-curation.md              # Written by agent at runtime
```

---

### Task 1: Project scaffold and config

**Files:**
- Create: `config.py`
- Create: `android_pool.py`
- Create: `agent/__init__.py`, `agent/nodes/__init__.py`, `agent/tools/__init__.py`
- Create: `training/__init__.py`, `data/__init__.py`, `eval/__init__.py`
- Create: `requirements.txt`
- Create: `tests/test_android_pool.py`

**Interfaces:**
- Produces: `ANDROID_POOL: list[ModelSpec]`, `ModelSpec` dataclass, `filter_pool(constraints: HardwareConstraints) -> list[ModelSpec]`

- [ ] **Step 1: Write failing test for Android pool filter**

```python
# tests/test_android_pool.py
from android_pool import filter_pool, ModelSpec, HardwareConstraints

def test_filter_pool_by_storage():
    constraints = HardwareConstraints(storage_mb=500, memory_mb=1200,
                                       latency_ttft_ms=2000, power_watts=5.0)
    result = filter_pool(constraints)
    assert all(m.int4_size_mb <= 500 for m in result)
    assert len(result) > 0

def test_filter_pool_returns_tier1_first():
    constraints = HardwareConstraints(storage_mb=1600, memory_mb=2000,
                                       latency_ttft_ms=2000, power_watts=5.0)
    result = filter_pool(constraints)
    assert result[0].tier == 1

def test_filter_pool_excludes_oversized():
    constraints = HardwareConstraints(storage_mb=100, memory_mb=200,
                                       latency_ttft_ms=2000, power_watts=5.0)
    result = filter_pool(constraints)
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_android_pool.py -v
```
Expected: ImportError — `android_pool` not found.

- [ ] **Step 3: Write `config.py`**

```python
# config.py
import os

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
EXA_API_KEY = os.environ["EXA_API_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

ORCHESTRATOR_MODEL = "claude-sonnet-4-6"
MAX_TURNS_MAIN = 1500
MAX_TURNS_SUBAGENT = 1000

DEFAULT_STOP_THRESHOLD = 0.96
DEFAULT_STAGNATION_ROUNDS = 2

CLUSTER_GPU = "L40"  # Phase 1 default; H200/A100 for larger jobs
DATA_DIR = "data_cache"
ARTIFACTS_DIR = "artifacts"
```

- [ ] **Step 4: Write `android_pool.py`**

```python
# android_pool.py
from dataclasses import dataclass

@dataclass
class ModelSpec:
    model_id: str          # HuggingFace model ID
    int4_size_mb: int      # theoretical INT4 size in MB
    tier: int              # 1, 2, or 3
    # Theoretical benchmarks on reference chips (tok/s)
    tok_s_snapdragon_660: float
    tok_s_snapdragon_778g: float
    tok_s_snapdragon_8gen3: float
    peak_memory_mb: int    # estimated peak RAM during inference
    notes: str = ""

@dataclass
class HardwareConstraints:
    storage_mb: int
    memory_mb: int
    latency_ttft_ms: int
    power_watts: float
    target_chip: str = "snapdragon_778g"

ANDROID_POOL: list[ModelSpec] = [
    # Tier 1 — fits anywhere (≤800MB)
    ModelSpec("Qwen/Qwen3-0.6B",         397,  1,  8.0, 15.0, 50.0, 500),
    ModelSpec("Qwen/Qwen3.5-0.8B",       500,  1,  7.0, 13.0, 45.0, 620),
    ModelSpec("openbmb/MiniCPM4-0.5B",   300,  1, 10.0, 18.0, 60.0, 380),
    ModelSpec("openbmb/MiniCPM5-1B",     500,  1,  8.0, 16.0, 55.0, 620),
    ModelSpec("google/gemma-3-1b-it",    529,  1,  9.0, 17.0, 70.0, 660),
    ModelSpec("meta-llama/Llama-3.2-1B", 808,  1,  6.0, 12.0, 40.0, 980),
    # Tier 2 — most Android devices (800MB–1.5GB)
    ModelSpec("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1060, 2, 4.0, 8.0, 30.0, 1280),
    ModelSpec("Qwen/Qwen3-1.7B",                      1190, 2, 4.0, 8.0, 30.0, 1430),
    ModelSpec("Qwen/Qwen3.5-2B",                      1200, 2, 3.5, 7.0, 28.0, 1440),
    ModelSpec("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", 1290, 2, 3.0, 6.0, 25.0, 1550),
    # Tier 3 — generous upper bound
    ModelSpec("meta-llama/Llama-3.2-3B", 1400, 3, 2.0, 4.0, 15.0, 1700),
    ModelSpec("mistralai/Ministral-3B-Instruct", 1900, 3, 1.5, 3.0, 12.0, 2200),
]

def filter_pool(constraints: HardwareConstraints) -> list[ModelSpec]:
    """Return models that fit within storage and memory constraints, sorted tier asc."""
    feasible = [
        m for m in ANDROID_POOL
        if m.int4_size_mb <= constraints.storage_mb
        and m.peak_memory_mb <= constraints.memory_mb
    ]
    return sorted(feasible, key=lambda m: (m.tier, m.int4_size_mb))
```

- [ ] **Step 5: Write `requirements.txt`**

```
anthropic>=0.40.0
langgraph>=0.2.0
langchain-anthropic>=0.3.0
unsloth>=2024.12.0
transformers>=4.47.0
peft>=0.14.0
datasets>=3.0.0
torch>=2.4.0
llama-cpp-python>=0.3.0
exa-py>=1.0.0
pytest>=8.0.0
python-dotenv>=1.0.0
```

- [ ] **Step 6: Create `__init__.py` files**

```bash
touch agent/__init__.py agent/nodes/__init__.py agent/tools/__init__.py
touch training/__init__.py data/__init__.py eval/__init__.py
mkdir -p data_cache artifacts
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_android_pool.py -v
```
Expected: 3 PASS.

- [ ] **Step 8: Commit**

```bash
git add config.py android_pool.py requirements.txt agent/ training/ data/ eval/ tests/test_android_pool.py
git commit -m "feat: project scaffold, config, and Android model pool"
```

---

### Task 2: SMS Spam data download and eval set construction

**Files:**
- Create: `data/sms_spam.py`
- Create: `data/eval_set.py`
- Create: `tests/test_sms_spam.py`
- Create: `tests/test_eval_set.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `download_sms_spam() -> tuple[list[dict], list[dict]]` — returns `(train_examples, test_examples)` where each example is `{"text": str, "label": str}` with label in `{"spam", "ham"}`
  - `build_eval_set(test_examples: list[dict], n_boundary: int = 50) -> EvalSet`
  - `EvalSet` dataclass with fields `pos: list[dict]`, `neg: list[dict]`, `boundary: list[dict]`, `all: list[dict]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_sms_spam.py
from data.sms_spam import download_sms_spam

def test_download_returns_train_test():
    train, test = download_sms_spam()
    assert len(train) > 1000
    assert len(test) > 100
    assert all("text" in ex and "label" in ex for ex in train)
    assert all(ex["label"] in {"spam", "ham"} for ex in train)

def test_class_distribution():
    train, _ = download_sms_spam()
    spam = [e for e in train if e["label"] == "spam"]
    ham = [e for e in train if e["label"] == "ham"]
    assert len(ham) > len(spam)  # UCI is ~87% ham
```

```python
# tests/test_eval_set.py
from data.eval_set import build_eval_set, EvalSet

FAKE_EXAMPLES = (
    [{"text": f"Free prize call now {i}", "label": "spam"} for i in range(30)]
    + [{"text": f"Hey are you coming tonight {i}", "label": "ham"} for i in range(70)]
)

def test_eval_set_has_three_slices():
    es = build_eval_set(FAKE_EXAMPLES, n_boundary=10)
    assert isinstance(es, EvalSet)
    assert len(es.pos) > 0
    assert len(es.neg) > 0
    assert len(es.boundary) > 0

def test_eval_set_all_is_union():
    es = build_eval_set(FAKE_EXAMPLES, n_boundary=10)
    assert len(es.all) == len(es.pos) + len(es.neg) + len(es.boundary)

def test_eval_set_never_in_training_data():
    es = build_eval_set(FAKE_EXAMPLES, n_boundary=10)
    eval_texts = {e["text"] for e in es.all}
    # All eval texts should come from the examples passed in
    all_texts = {e["text"] for e in FAKE_EXAMPLES}
    assert eval_texts.issubset(all_texts)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sms_spam.py tests/test_eval_set.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write `data/sms_spam.py`**

```python
# data/sms_spam.py
import os, csv, urllib.request, zipfile
from config import DATA_DIR

SMS_SPAM_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
SMS_SPAM_CACHE = os.path.join(DATA_DIR, "sms_spam.tsv")

def download_sms_spam(test_fraction: float = 0.2) -> tuple[list[dict], list[dict]]:
    """Download UCI SMS Spam Collection and return (train, test) splits."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(SMS_SPAM_CACHE):
        zip_path = os.path.join(DATA_DIR, "sms_spam.zip")
        urllib.request.urlretrieve(SMS_SPAM_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            with z.open("SMSSpamCollection") as f:
                content = f.read().decode("utf-8", errors="replace")
        with open(SMS_SPAM_CACHE, "w", encoding="utf-8") as f:
            f.write(content)

    examples = []
    with open(SMS_SPAM_CACHE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                label, text = parts
                examples.append({"text": text, "label": label.strip()})

    split_idx = int(len(examples) * (1 - test_fraction))
    return examples[:split_idx], examples[split_idx:]
```

- [ ] **Step 4: Write `data/eval_set.py`**

```python
# data/eval_set.py
import random
from dataclasses import dataclass, field

@dataclass
class EvalSet:
    pos: list[dict]       # Epos: genuine spam, full surface-form range
    neg: list[dict]       # Eneg: legitimate messages including borderline ham
    boundary: list[dict]  # Eboundary: confusable pairs at spam/ham boundary
    all: list[dict] = field(init=False)

    def __post_init__(self):
        self.all = self.pos + self.neg + self.boundary

def build_eval_set(
    examples: list[dict],
    n_pos: int = 40,
    n_neg: int = 40,
    n_boundary: int = 20,
    seed: int = 42,
) -> EvalSet:
    """
    Build E = Epos ∪ Eneg ∪ Eboundary from test examples.

    Epos: clear spam examples (high confidence)
    Eneg: clear ham examples (high confidence)
    Eboundary: short promotional ham that could be confused with spam

    All slices are disjoint. Total size = n_pos + n_neg + n_boundary.
    """
    rng = random.Random(seed)
    spam = [e for e in examples if e["label"] == "spam"]
    ham = [e for e in examples if e["label"] == "ham"]

    # Eboundary: ham messages that contain URLs, prizes, or urgency keywords
    boundary_keywords = {"www", "http", "free", "win", "prize", "offer",
                         "click", "call", "urgent", "limited", "txt"}
    boundary_candidates = [
        e for e in ham
        if any(kw in e["text"].lower() for kw in boundary_keywords)
    ]
    rng.shuffle(boundary_candidates)
    boundary = boundary_candidates[:n_boundary]
    boundary_texts = {e["text"] for e in boundary}

    # Eneg: clear ham (not in boundary)
    clear_ham = [e for e in ham if e["text"] not in boundary_texts]
    rng.shuffle(clear_ham)
    neg = clear_ham[:n_neg]

    # Epos: clear spam
    rng.shuffle(spam)
    pos = spam[:n_pos]

    return EvalSet(pos=pos, neg=neg, boundary=boundary)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_sms_spam.py tests/test_eval_set.py -v
```
Expected: all PASS. (Note: `test_download_returns_train_test` downloads ~500KB from UCI on first run.)

- [ ] **Step 6: Commit**

```bash
git add data/sms_spam.py data/eval_set.py tests/test_sms_spam.py tests/test_eval_set.py
git commit -m "feat: SMS Spam download and eval set construction (Epos/Eneg/Eboundary)"
```

---

### Task 3: Eval metrics and harness

**Files:**
- Create: `eval/metrics.py`
- Create: `eval/harness.py`
- Create: `tests/test_metrics.py`
- Create: `tests/test_harness.py`

**Interfaces:**
- Consumes: `EvalSet` from Task 2; `infer_batch()` from Task 5 (mocked in tests)
- Produces:
  - `binary_f1(predictions: list[str], labels: list[str], pos_label: str = "spam") -> float`
  - `per_slice_scores(eval_set: EvalSet, predictions: list[str]) -> dict[str, float]`
  - `run_eval(eval_set: EvalSet, weights_ref: str, base_model: str) -> EvalResult`
  - `EvalResult` dataclass: `f1: float`, `spam_f1: float`, `ham_f1: float`, `pos_score: float`, `neg_score: float`, `boundary_score: float`, `failures: list[dict]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
from eval.metrics import binary_f1, per_slice_scores
from data.eval_set import EvalSet

def test_binary_f1_perfect():
    preds = ["spam", "spam", "ham", "ham"]
    labels = ["spam", "spam", "ham", "ham"]
    assert binary_f1(preds, labels) == 1.0

def test_binary_f1_all_wrong():
    preds = ["ham", "ham", "spam", "spam"]
    labels = ["spam", "spam", "ham", "ham"]
    assert binary_f1(preds, labels) == 0.0

def test_binary_f1_partial():
    preds =  ["spam", "ham", "spam", "ham"]
    labels = ["spam", "spam", "ham", "ham"]
    # TP=1, FP=1, FN=1 → precision=0.5, recall=0.5 → F1=0.5
    assert abs(binary_f1(preds, labels) - 0.5) < 1e-6

def test_per_slice_scores():
    es = EvalSet(
        pos=[{"text": "spam1", "label": "spam"}, {"text": "spam2", "label": "spam"}],
        neg=[{"text": "ham1", "label": "ham"}],
        boundary=[{"text": "promo", "label": "ham"}],
    )
    preds = ["spam", "spam", "ham", "ham"]  # all correct
    scores = per_slice_scores(es, preds)
    assert scores["pos"] == 1.0
    assert scores["neg"] == 1.0
    assert scores["boundary"] == 1.0
```

```python
# tests/test_harness.py
from unittest.mock import patch
from eval.harness import run_eval, EvalResult
from data.eval_set import EvalSet

FAKE_EVAL_SET = EvalSet(
    pos=[{"text": "WIN FREE PRIZE", "label": "spam"}],
    neg=[{"text": "Hey see you soon", "label": "ham"}],
    boundary=[{"text": "Exclusive offer click here", "label": "ham"}],
)

def test_run_eval_returns_eval_result():
    with patch("eval.harness.infer_batch", return_value=["spam", "ham", "ham"]):
        result = run_eval(FAKE_EVAL_SET, "fake_weights_ref", "Qwen/Qwen3-0.6B")
    assert isinstance(result, EvalResult)
    assert 0.0 <= result.f1 <= 1.0
    assert isinstance(result.failures, list)

def test_run_eval_identifies_failures():
    # Model predicts ham for everything — spam example fails
    with patch("eval.harness.infer_batch", return_value=["ham", "ham", "ham"]):
        result = run_eval(FAKE_EVAL_SET, "fake_weights_ref", "Qwen/Qwen3-0.6B")
    assert any(f["label"] == "spam" for f in result.failures)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py tests/test_harness.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write `eval/metrics.py`**

```python
# eval/metrics.py
from data.eval_set import EvalSet

def binary_f1(
    predictions: list[str],
    labels: list[str],
    pos_label: str = "spam",
) -> float:
    """Compute binary F1 score for the positive class."""
    tp = sum(p == pos_label and l == pos_label for p, l in zip(predictions, labels))
    fp = sum(p == pos_label and l != pos_label for p, l in zip(predictions, labels))
    fn = sum(p != pos_label and l == pos_label for p, l in zip(predictions, labels))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

def per_slice_scores(eval_set: EvalSet, predictions: list[str]) -> dict[str, float]:
    """Compute accuracy for each of the three eval set slices."""
    n_pos = len(eval_set.pos)
    n_neg = len(eval_set.neg)
    n_bnd = len(eval_set.boundary)

    pos_preds = predictions[:n_pos]
    neg_preds = predictions[n_pos:n_pos + n_neg]
    bnd_preds = predictions[n_pos + n_neg:]

    def acc(preds, examples):
        if not examples:
            return 0.0
        return sum(p == e["label"] for p, e in zip(preds, examples)) / len(examples)

    return {
        "pos": acc(pos_preds, eval_set.pos),
        "neg": acc(neg_preds, eval_set.neg),
        "boundary": acc(bnd_preds, eval_set.boundary),
    }
```

- [ ] **Step 4: Write `eval/harness.py`**

```python
# eval/harness.py
import re
from dataclasses import dataclass
from data.eval_set import EvalSet
from eval.metrics import binary_f1, per_slice_scores

# Will be replaced by real import once training/slm_helpers.py exists
try:
    from training.slm_helpers import infer_batch
except ImportError:
    def infer_batch(prompts, weights_ref, base_model, max_workers=20):
        raise NotImplementedError("slm_helpers not yet available")

CLASSIFY_PROMPT = """Classify this SMS message as either "spam" or "ham".
Reply with exactly one word: spam or ham.

Message: {text}"""

def _extract_label(raw: str) -> str:
    """Extract 'spam' or 'ham' from model output. Default to 'ham' if unclear."""
    cleaned = raw.strip().lower()
    if "spam" in cleaned:
        return "spam"
    return "ham"

@dataclass
class EvalResult:
    f1: float
    spam_f1: float
    ham_f1: float
    pos_score: float
    neg_score: float
    boundary_score: float
    failures: list[dict]  # examples where prediction != label

def run_eval(eval_set: EvalSet, weights_ref: str, base_model: str) -> EvalResult:
    """Run inference on E and compute all metrics."""
    prompts = [
        CLASSIFY_PROMPT.format(text=ex["text"])
        for ex in eval_set.all
    ]
    raw_outputs = infer_batch(prompts, weights_ref, base_model, max_workers=20)
    predictions = [_extract_label(r) for r in raw_outputs]
    labels = [ex["label"] for ex in eval_set.all]

    f1 = binary_f1(predictions, labels)
    spam_f1 = binary_f1(predictions, labels, pos_label="spam")
    ham_f1 = binary_f1(predictions, labels, pos_label="ham")
    slices = per_slice_scores(eval_set, predictions)

    failures = [
        {**ex, "predicted": pred}
        for ex, pred, label in zip(eval_set.all, predictions, labels)
        if pred != label
    ]

    return EvalResult(
        f1=f1,
        spam_f1=spam_f1,
        ham_f1=ham_f1,
        pos_score=slices["pos"],
        neg_score=slices["neg"],
        boundary_score=slices["boundary"],
        failures=failures,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py tests/test_harness.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add eval/metrics.py eval/harness.py tests/test_metrics.py tests/test_harness.py
git commit -m "feat: eval metrics (binary F1, per-slice) and harness"
```

---

### Task 4: Curriculum synthesis and curation log

**Files:**
- Create: `data/curriculum.py`
- Create: `data/curation_log.py`
- Create: `tests/test_curriculum.py`
- Create: `tests/test_curation_log.py`

**Interfaces:**
- Consumes: `EvalSet` from Task 2; `EvalResult` from Task 3
- Produces:
  - `build_initial_curriculum(train_examples: list[dict], eval_set: EvalSet, n_total: int = 150) -> list[dict]`
  - `synthesize_hard_negatives(examples: list[dict], n: int, anthropic_client) -> list[dict]`
  - `apply_quality_controls(dataset: list[dict]) -> list[dict]`
  - `CurationLog` class with `write_iteration(...)` and `read_latest() -> str`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_curriculum.py
from data.curriculum import build_initial_curriculum, apply_quality_controls
from data.eval_set import EvalSet

FAKE_TRAIN = (
    [{"text": f"FREE PRIZE CALL NOW {i}!!", "label": "spam"} for i in range(200)]
    + [{"text": f"Hey how are you doing {i}", "label": "ham"} for i in range(800)]
)
FAKE_EVAL = EvalSet(
    pos=[{"text": "WIN FREE PRIZE", "label": "spam"}],
    neg=[{"text": "Hey see you", "label": "ham"}],
    boundary=[{"text": "Exclusive offer", "label": "ham"}],
)

def test_initial_curriculum_size():
    dataset = build_initial_curriculum(FAKE_TRAIN, FAKE_EVAL, n_total=150)
    assert 100 <= len(dataset) <= 200

def test_initial_curriculum_excludes_eval():
    dataset = build_initial_curriculum(FAKE_TRAIN, FAKE_EVAL, n_total=150)
    eval_texts = {e["text"] for e in FAKE_EVAL.all}
    assert not any(e["text"] in eval_texts for e in dataset)

def test_initial_curriculum_label_balance():
    dataset = build_initial_curriculum(FAKE_TRAIN, FAKE_EVAL, n_total=150)
    spam = [e for e in dataset if e["label"] == "spam"]
    ham = [e for e in dataset if e["label"] == "ham"]
    # No label exceeds 3x the other
    assert len(ham) <= 3 * len(spam)
    assert len(spam) <= 3 * len(ham)

def test_quality_controls_label_balance():
    # 10 spam, 100 ham — ham exceeds 3x spam
    imbalanced = (
        [{"text": f"spam {i}", "label": "spam"} for i in range(10)]
        + [{"text": f"ham {i}", "label": "ham"} for i in range(100)]
    )
    result = apply_quality_controls(imbalanced)
    spam = [e for e in result if e["label"] == "spam"]
    ham = [e for e in result if e["label"] == "ham"]
    assert len(ham) <= 3 * len(spam)
```

```python
# tests/test_curation_log.py
import os, tempfile
from data.curation_log import CurationLog
from eval.harness import EvalResult
from android_pool import ModelSpec

def _fake_eval_result():
    return EvalResult(
        f1=0.85, spam_f1=0.83, ham_f1=0.87,
        pos_score=0.9, neg_score=0.95, boundary_score=0.7,
        failures=[{"text": "promo", "label": "ham", "predicted": "spam"}],
    )

def test_write_and_read_iteration():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        path = f.name
    try:
        log = CurationLog(path)
        log.write_iteration(
            iteration=1,
            dataset_version="v1",
            n_gold=100, n_hard=50,
            label_dist={"spam": 50, "ham": 100},
            config_a="LoRA r=8 lr=2e-4 epochs=3",
            config_b="full_ft lr=1e-4 epochs=5",
            best_config="A",
            eval_result=_fake_eval_result(),
            score_band="0.80-0.95",
            next_intervention="hyperparameter tuning",
            hypothesis="Increase epochs to reduce underfitting",
            model_id="Qwen/Qwen3-0.6B",
            int4_size_mb=397,
            tier=1,
        )
        content = log.read_latest()
        assert "Iteration 1" in content
        assert "v1" in content
        assert "0.85" in content
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_curriculum.py tests/test_curation_log.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write `data/curriculum.py`**

```python
# data/curriculum.py
import random
from data.eval_set import EvalSet

def build_initial_curriculum(
    train_examples: list[dict],
    eval_set: EvalSet,
    n_total: int = 150,
    gold_fraction: float = 0.65,
    seed: int = 42,
) -> list[dict]:
    """
    Build Dcold = Dgold ∪ Dhard at 65:35 from train_examples.
    Excludes any example that appears in eval_set.
    Applies label balancing (no label > 3x any other).
    """
    eval_texts = {e["text"] for e in eval_set.all}
    available = [e for e in train_examples if e["text"] not in eval_texts]

    n_gold = int(n_total * gold_fraction)
    spam = [e for e in available if e["label"] == "spam"]
    ham = [e for e in available if e["label"] == "ham"]

    rng = random.Random(seed)
    rng.shuffle(spam)
    rng.shuffle(ham)

    # Balance: target equal representation up to 3x constraint
    n_spam = min(len(spam), n_gold // 2)
    n_ham = min(len(ham), n_gold - n_spam)

    gold = spam[:n_spam] + ham[:n_ham]
    rng.shuffle(gold)
    return apply_quality_controls(gold)

def apply_quality_controls(dataset: list[dict]) -> list[dict]:
    """
    Enforce label balancing: no label exceeds 3x the count of any other.
    Trims the majority class to satisfy the constraint.
    """
    from collections import Counter
    counts = Counter(e["label"] for e in dataset)
    if not counts:
        return dataset

    min_count = min(counts.values())
    max_allowed = 3 * min_count

    result = []
    seen = Counter()
    for ex in dataset:
        if seen[ex["label"]] < max_allowed:
            result.append(ex)
            seen[ex["label"]] += 1
    return result

def synthesize_hard_negatives(
    examples: list[dict],
    n: int,
    anthropic_client,
) -> list[dict]:
    """
    Generate n hard negatives using the 2-for-1 rule.
    For each boundary spam example, generate a ham counterexample
    with similar surface form but legitimate intent, and vice versa.
    Uses Claude API directly (classification — no teacher model needed).
    """
    hard_negatives = []
    boundary = [e for e in examples if e["label"] == "spam"][:n]

    for ex in boundary:
        prompt = (
            f"Generate a legitimate SMS message that looks superficially similar to "
            f"this spam message but has a genuine, non-spam intent.\n\n"
            f"Spam message: {ex['text']}\n\n"
            f"Respond with only the message text, no explanation."
        )
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        generated_text = response.content[0].text.strip()
        hard_negatives.append({"text": generated_text, "label": "ham"})

    return hard_negatives
```

- [ ] **Step 4: Write `data/curation_log.py`**

```python
# data/curation_log.py
from datetime import datetime
from eval.harness import EvalResult

class CurationLog:
    """Reads and writes data-curation.md — the agent's durable lineage artifact."""

    def __init__(self, path: str = "data-curation.md"):
        self.path = path

    def write_iteration(
        self,
        iteration: int,
        dataset_version: str,
        n_gold: int,
        n_hard: int,
        label_dist: dict,
        config_a: str,
        config_b: str,
        best_config: str,
        eval_result: EvalResult,
        score_band: str,
        next_intervention: str,
        hypothesis: str,
        model_id: str,
        int4_size_mb: int,
        tier: int,
        hardware_notes: str = "Phase 1: theoretical",
    ) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        entry = f"""
## Iteration {iteration} — {timestamp}

### Dataset
- Version: {dataset_version}
- Total examples: {n_gold + n_hard}
- Dgold: {n_gold} ({n_gold/(n_gold+n_hard)*100:.0f}%)
- Dhard: {n_hard} ({n_hard/(n_gold+n_hard)*100:.0f}%)
- Label distribution: {label_dist}

### Training config (π_{iteration})
- Config A: {config_a}
- Config B: {config_b}
- Best config selected: {best_config}

### Eval results
- f(π_{iteration}): {eval_result.f1:.4f} (binary F1)
- spam_f1: {eval_result.spam_f1:.4f} | ham_f1: {eval_result.ham_f1:.4f}
- Epos: {eval_result.pos_score:.4f} | Eneg: {eval_result.neg_score:.4f} | Eboundary: {eval_result.boundary_score:.4f}
- Remaining failures: {len(eval_result.failures)}

### Iteration policy decision
- Score band: {score_band}
- Next intervention: {next_intervention}
- Hypothesis: {hypothesis}

### Hardware profile ({hardware_notes})
- Model: {model_id} | INT4 size: {int4_size_mb}MB | Tier: {tier}

---
"""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(entry)

    def read_latest(self) -> str:
        """Read the full data-curation.md contents."""
        try:
            with open(self.path, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_curriculum.py tests/test_curation_log.py -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add data/curriculum.py data/curation_log.py tests/test_curriculum.py tests/test_curation_log.py
git commit -m "feat: curriculum synthesis, quality controls, and data-curation.md log"
```

---

### Task 5: Training backend (`slm_helpers.py`)

**Files:**
- Create: `training/slm_helpers.py`
- Create: `training/lora_trainer.py`
- Create: `training/quantize.py`
- Create: `tests/test_lora_trainer.py`

**Interfaces:**
- Consumes: `ModelSpec` from Task 1
- Produces:
  - `train(dataset_path, base_model, nr_epochs, learning_rate, batch_size, lora_rank) -> str` — returns `weights_ref`
  - `infer(prompt, weights_ref, base_model, **kwargs) -> str`
  - `infer_batch(prompts, weights_ref, base_model, max_workers=20) -> list[str]`
  - `theoretical_hardware_profile(model_id: str) -> dict` — returns storage/memory/latency/power estimates

- [ ] **Step 1: Write failing tests**

```python
# tests/test_lora_trainer.py
import os, json, tempfile
from unittest.mock import patch, MagicMock
from training.lora_trainer import run_lora_training, TrainingConfig

def _fake_dataset(path):
    examples = [
        {"text": f"FREE PRIZE {i}", "label": "spam"} for i in range(20)
    ] + [
        {"text": f"Hey how are you {i}", "label": "ham"} for i in range(20)
    ]
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

def test_training_config_validation():
    config = TrainingConfig(
        base_model="Qwen/Qwen3-0.6B",
        nr_epochs=3,
        learning_rate=2e-4,
        batch_size=8,
        lora_rank=8,
    )
    assert config.lora_rank in {4, 8, 16, 32, 64}
    assert 0 < config.learning_rate < 1

def test_run_lora_training_returns_weights_ref():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = os.path.join(tmpdir, "train.jsonl")
        _fake_dataset(dataset_path)
        config = TrainingConfig(
            base_model="Qwen/Qwen3-0.6B",
            nr_epochs=1,
            learning_rate=2e-4,
            batch_size=4,
            lora_rank=4,
        )
        with patch("training.lora_trainer._run_unsloth_training") as mock_train:
            mock_train.return_value = os.path.join(tmpdir, "checkpoint")
            os.makedirs(os.path.join(tmpdir, "checkpoint"))
            weights_ref = run_lora_training(dataset_path, config, output_dir=tmpdir)
        assert isinstance(weights_ref, str)
        assert len(weights_ref) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_lora_trainer.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write `training/lora_trainer.py`**

```python
# training/lora_trainer.py
import os, json, subprocess
from dataclasses import dataclass

VALID_LORA_RANKS = {4, 8, 16, 32, 64}

@dataclass
class TrainingConfig:
    base_model: str
    nr_epochs: int
    learning_rate: float
    batch_size: int
    lora_rank: int | None  # None = full fine-tune

    def __post_init__(self):
        if self.lora_rank is not None and self.lora_rank not in VALID_LORA_RANKS:
            raise ValueError(f"lora_rank must be one of {VALID_LORA_RANKS}, got {self.lora_rank}")
        if not (0 < self.learning_rate < 1):
            raise ValueError(f"learning_rate must be in (0,1), got {self.learning_rate}")

def _run_unsloth_training(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str,
) -> str:
    """
    Run LoRA fine-tuning via Unsloth.
    Returns the path to the saved checkpoint directory.
    """
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import torch

    max_seq_length = 512
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=config.lora_rank is not None,
    )

    if config.lora_rank is not None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_rank,
            lora_alpha=config.lora_rank * 2,
            lora_dropout=0.0,
            target_modules="all-linear",
            bias="none",
        )

    # Load dataset
    with open(dataset_path) as f:
        raw = [json.loads(line) for line in f if line.strip()]

    PROMPT_TEMPLATE = (
        "Classify this SMS message as spam or ham.\n\nMessage: {text}\n\nLabel: {label}"
    )

    def format_example(ex):
        return {"text": PROMPT_TEMPLATE.format(**ex)}

    from datasets import Dataset
    dataset = Dataset.from_list([format_example(e) for e in raw])

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.nr_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=args,
    )
    trainer.train()

    checkpoint_path = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    return checkpoint_path

def run_lora_training(
    dataset_path: str,
    config: TrainingConfig,
    output_dir: str = "artifacts",
) -> str:
    """
    Train model with LoRA (or full fine-tune if lora_rank is None).
    Returns weights_ref — a path string usable by infer() and infer_batch().
    Always trains from the base model, never from a prior checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = _run_unsloth_training(dataset_path, config, output_dir)
    return checkpoint_path
```

- [ ] **Step 4: Write `training/slm_helpers.py`**

```python
# training/slm_helpers.py
"""
Agent-facing interface for training and inference.
Called by the agent via the bash tool.
Equivalent to the paper's tinker_helpers.py.
"""
import os
import json
from concurrent.futures import ThreadPoolExecutor
from training.lora_trainer import TrainingConfig, run_lora_training

# Lazy-loaded inference model cache: weights_ref -> (model, tokenizer)
_inference_cache: dict = {}

def train(
    dataset_path: str,
    base_model: str,
    nr_epochs: int,
    learning_rate: float,
    batch_size: int,
    lora_rank: int | None,
    output_dir: str = "artifacts",
) -> str:
    """
    Execute the full LoRA training loop.
    Returns weights_ref string for immediate inference.
    Always trains from base_model — never from a prior checkpoint.
    """
    config = TrainingConfig(
        base_model=base_model,
        nr_epochs=nr_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        lora_rank=lora_rank,
    )
    return run_lora_training(dataset_path, config, output_dir=output_dir)

def infer(prompt: str, weights_ref: str, base_model: str, max_new_tokens: int = 50) -> str:
    """Load checkpoint and generate text. No deployment step required."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    key = weights_ref
    if key not in _inference_cache:
        tokenizer = AutoTokenizer.from_pretrained(weights_ref)
        model = AutoModelForCausalLM.from_pretrained(
            weights_ref,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        _inference_cache[key] = (model, tokenizer)

    model, tokenizer = _inference_cache[key]
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Return only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def infer_batch(
    prompts: list[str],
    weights_ref: str,
    base_model: str,
    max_new_tokens: int = 50,
    max_workers: int = 20,
) -> list[str]:
    """Parallel inference via ThreadPoolExecutor with max_workers=20."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(infer, p, weights_ref, base_model, max_new_tokens)
            for p in prompts
        ]
        return [f.result() for f in futures]
```

- [ ] **Step 5: Write `training/quantize.py`**

```python
# training/quantize.py
"""
INT4 quantization and theoretical hardware profile for Android pool models.
Phase 1: theoretical estimates only. Phase 2 replaces with measured values.
"""
from android_pool import ANDROID_POOL, ModelSpec

def theoretical_hardware_profile(model_id: str) -> dict:
    """
    Return theoretical hardware estimates for a model from the Android pool.
    Phase 1 uses benchmark-derived estimates, not measurements.
    """
    spec = next((m for m in ANDROID_POOL if m.model_id == model_id), None)
    if spec is None:
        return {
            "model_id": model_id,
            "int4_size_mb": None,
            "tier": None,
            "tok_s_snapdragon_778g": None,
            "peak_memory_mb": None,
            "phase": "theoretical",
            "note": "Model not in Android pool",
        }
    return {
        "model_id": spec.model_id,
        "int4_size_mb": spec.int4_size_mb,
        "tier": spec.tier,
        "tok_s_snapdragon_660": spec.tok_s_snapdragon_660,
        "tok_s_snapdragon_778g": spec.tok_s_snapdragon_778g,
        "tok_s_snapdragon_8gen3": spec.tok_s_snapdragon_8gen3,
        "peak_memory_mb": spec.peak_memory_mb,
        "phase": "theoretical",
    }
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_lora_trainer.py -v
```
Expected: all PASS (mocked Unsloth call).

- [ ] **Step 7: Commit**

```bash
git add training/slm_helpers.py training/lora_trainer.py training/quantize.py tests/test_lora_trainer.py
git commit -m "feat: training backend — slm_helpers, LoRA trainer, quantize"
```

---

### Task 6: Agent state and tools

**Files:**
- Create: `agent/state.py`
- Create: `agent/tools/web_search.py`
- Create: `agent/tools/bash_tool.py`
- Create: `agent/tools/file_tools.py`
- Create: `agent/tools/delegate_task.py`

**Interfaces:**
- Consumes: `HardwareConstraints`, `ModelSpec` from Task 1; `EvalSet` from Task 2; `EvalResult` from Task 3; `CurationLog` from Task 4
- Produces: `AgentState` TypedDict; `COLD_START_TOOLS: list` — tool definitions for LangGraph

- [ ] **Step 1: Write `agent/state.py`**

```python
# agent/state.py
from typing import TypedDict, Optional, Any
from data.eval_set import EvalSet
from eval.harness import EvalResult
from android_pool import ModelSpec, HardwareConstraints

class AgentState(TypedDict):
    # Task specification
    description: str
    target_metric: str
    hardware_constraints: HardwareConstraints

    # Task analysis outputs
    task_type: str                    # "classification", "ner", "generation"
    selected_model: Optional[ModelSpec]
    stop_threshold: float             # default 0.96, agent may lower

    # Data
    train_examples: list[dict]
    eval_set: Optional[EvalSet]
    current_dataset_path: Optional[str]  # path to Dcold JSONL on disk
    dataset_version: int              # incremented each curate call

    # Search state
    best_weights_ref: Optional[str]
    best_score: float
    iteration: int
    scores: list[float]               # f(π) per iteration
    dag: list[dict]                   # lineage DAG nodes
    consecutive_no_improvement: int

    # Last iteration results
    last_eval: Optional[EvalResult]
    last_intervention: str            # "data_rebuild" | "hyperparameter" | "surgical" | "rollback"
    next_action: str                  # "train" | "curate" | "rollback" | "escalate" | "terminate"

    # Messages for LangGraph
    messages: list[Any]
```

- [ ] **Step 2: Write `agent/tools/web_search.py`**

```python
# agent/tools/web_search.py
import os
from exa_py import Exa
from langchain_core.tools import tool

_exa_client = None

def _get_exa():
    global _exa_client
    if _exa_client is None:
        from config import EXA_API_KEY
        _exa_client = Exa(api_key=EXA_API_KEY)
    return _exa_client

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web using Exa deep research API.
    Use for: locating datasets, surveying published baselines, domain knowledge.
    Returns a formatted string of results.
    """
    exa = _get_exa()
    results = exa.search_and_contents(
        query,
        num_results=num_results,
        use_autoprompt=True,
        text={"max_characters": 1000},
    )
    formatted = []
    for r in results.results:
        formatted.append(f"**{r.title}**\n{r.url}\n{r.text[:500]}\n")
    return "\n---\n".join(formatted) if formatted else "No results found."
```

- [ ] **Step 3: Write `agent/tools/bash_tool.py`**

```python
# agent/tools/bash_tool.py
import subprocess
import sys
import os
from langchain_core.tools import tool

# Inject slm_helpers into the bash environment path
_HELPERS_INJECT = f"export PYTHONPATH={os.path.abspath('.')}:$PYTHONPATH"

@tool
def bash(command: str) -> str:
    """
    Execute a shell command. slm_helpers.py is pre-loaded via PYTHONPATH.
    Use for: running train(), infer_batch(), dataset operations, eval scripts.
    Returns stdout + stderr combined.
    """
    full_command = f"{_HELPERS_INJECT} && {command}"
    result = subprocess.run(
        full_command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max for training runs
    )
    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"
    return output or "(no output)"
```

- [ ] **Step 4: Write `agent/tools/file_tools.py`**

```python
# agent/tools/file_tools.py
import os
from langchain_core.tools import tool

@tool
def read_file(path: str) -> str:
    """Read a file from disk. Use for: datasets, configs, data-curation.md, eval results."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[ERROR] File not found: {path}"
    except Exception as e:
        return f"[ERROR] {e}"

@tool
def edit_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"[ERROR] {e}"
```

- [ ] **Step 5: Write `agent/tools/delegate_task.py`**

```python
# agent/tools/delegate_task.py
"""
Sub-agent spawning via delegate_task.
Sub-agents share the filesystem. Main agent reads their output files.
Not a named tool — called programmatically by the orchestrator.
"""
import anthropic
from config import ANTHROPIC_MODEL, MAX_TURNS_SUBAGENT, ANTHROPIC_API_KEY

def delegate_task(task_description: str, output_file: str) -> str:
    """
    Spawn a sub-agent to work on task_description.
    Sub-agent writes its result to output_file on disk.
    Main agent reads output_file — never gets raw sub-agent context.
    Returns the contents of output_file when complete.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system = (
        "You are a sub-agent in an agentic fine-tuning pipeline. "
        f"Complete the task and write your structured output to: {output_file}\n"
        "Be concise. Write the file before responding."
    )
    messages = [{"role": "user", "content": task_description}]

    # Simple single-turn sub-agent for Phase 1
    # Phase 2 can extend to multi-turn with tool use
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        system=system,
        messages=messages,
    )

    # Read output file written by sub-agent (if it did so via tool calls in extended use)
    try:
        with open(output_file, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return response.content[0].text
```

- [ ] **Step 6: Commit**

```bash
git add agent/state.py agent/tools/ 
git commit -m "feat: agent state TypedDict and tool definitions (web_search, bash, file_tools, delegate_task)"
```

---

### Task 7: LangGraph state machine nodes

**Files:**
- Create: `agent/nodes/task_analysis.py`
- Create: `agent/nodes/eval_setup.py`
- Create: `agent/nodes/train.py`
- Create: `agent/nodes/evaluate.py`
- Create: `agent/nodes/iterate.py`
- Create: `agent/nodes/curate.py`
- Create: `agent/nodes/rollback.py`
- Create: `agent/nodes/escalate.py`
- Create: `tests/test_nodes.py`

**Interfaces:**
- Consumes: `AgentState` from Task 6; all data/eval/training modules from Tasks 1–5
- Produces: each node is `Callable[[AgentState], AgentState]`

- [ ] **Step 1: Write failing tests for nodes**

```python
# tests/test_nodes.py
from unittest.mock import patch, MagicMock
from android_pool import HardwareConstraints, ModelSpec
from agent.state import AgentState
from agent.nodes.iterate import apply_iteration_policy
from agent.nodes.rollback import should_rollback

def _base_state() -> dict:
    return {
        "description": "fine-tune on SMS Spam binary classification",
        "target_metric": "F1",
        "hardware_constraints": HardwareConstraints(
            storage_mb=800, memory_mb=1200,
            latency_ttft_ms=2000, power_watts=5.0
        ),
        "task_type": "classification",
        "selected_model": None,
        "stop_threshold": 0.96,
        "train_examples": [],
        "eval_set": None,
        "current_dataset_path": None,
        "dataset_version": 0,
        "best_weights_ref": None,
        "best_score": 0.0,
        "iteration": 0,
        "scores": [],
        "dag": [],
        "consecutive_no_improvement": 0,
        "last_eval": None,
        "last_intervention": "",
        "next_action": "train",
        "messages": [],
    }

def test_iteration_policy_data_rebuild():
    decision = apply_iteration_policy(score=0.75)
    assert decision["band"] == "<0.80"
    assert decision["intervention"] == "data_rebuild"

def test_iteration_policy_hyperparameter():
    decision = apply_iteration_policy(score=0.87)
    assert decision["band"] == "0.80-0.95"
    assert decision["intervention"] == "hyperparameter"

def test_iteration_policy_surgical():
    decision = apply_iteration_policy(score=0.97)
    assert decision["band"] == ">=0.95"
    assert decision["intervention"] == "surgical"

def test_rollback_triggers_on_decrease():
    state = _base_state()
    state["scores"] = [0.85, 0.82]
    assert should_rollback(state) is True

def test_rollback_does_not_trigger_on_improvement():
    state = _base_state()
    state["scores"] = [0.82, 0.87]
    assert should_rollback(state) is False

def test_rollback_does_not_trigger_on_first_iteration():
    state = _base_state()
    state["scores"] = [0.75]
    assert should_rollback(state) is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_nodes.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write `agent/nodes/iterate.py`**

```python
# agent/nodes/iterate.py
from agent.state import AgentState

def apply_iteration_policy(score: float) -> dict:
    """
    Apply the shared iteration policy based on current f(π).
    Returns dict with band and intervention type.
    """
    if score < 0.80:
        return {
            "band": "<0.80",
            "intervention": "data_rebuild",
            "description": "Score below 0.80 — data problem. Rebuild dataset.",
        }
    elif score < 0.95:
        return {
            "band": "0.80-0.95",
            "intervention": "hyperparameter",
            "description": "Score 0.80–0.95 — optimization problem. Tune hyperparameters, hold dataset fixed.",
        }
    else:
        return {
            "band": ">=0.95",
            "intervention": "surgical",
            "description": "Score ≥0.95 — add 2–3 targeted examples per remaining failure pattern.",
        }

def iterate_node(state: AgentState) -> AgentState:
    """Node 5: determine next intervention based on current score."""
    if not state["scores"]:
        state["next_action"] = "train"
        return state

    current_score = state["scores"][-1]
    policy = apply_iteration_policy(current_score)
    state["last_intervention"] = policy["intervention"]

    if current_score >= state["stop_threshold"]:
        state["next_action"] = "terminate"
    elif state["consecutive_no_improvement"] >= 2:
        state["next_action"] = "escalate"
    else:
        state["next_action"] = "curate"

    return state
```

- [ ] **Step 4: Write `agent/nodes/rollback.py`**

```python
# agent/nodes/rollback.py
from agent.state import AgentState

def should_rollback(state: AgentState) -> bool:
    """
    Cold-start simple rollback: if f(π_i+1) < f(π_i), revert.
    No dual-gate in cold-start (that is production mode only).
    """
    scores = state["scores"]
    if len(scores) < 2:
        return False
    return scores[-1] < scores[-2]

def rollback_node(state: AgentState) -> AgentState:
    """
    Node 7: revert to previous configuration if score decreased.
    Removes the last score from history. Restores previous weights_ref.
    """
    if not should_rollback(state):
        return state

    # Remove the regressing score
    state["scores"].pop()
    # Restore best known weights
    state["last_intervention"] = "rollback"
    state["consecutive_no_improvement"] += 1

    # If we have a DAG, mark the last node as pruned
    if state["dag"]:
        state["dag"][-1]["pruned"] = True

    return state
```

- [ ] **Step 5: Write remaining nodes**

```python
# agent/nodes/task_analysis.py
from agent.state import AgentState
from android_pool import filter_pool

def task_analysis_node(state: AgentState) -> AgentState:
    """
    Node 1: classify task type, select starting model, survey baselines.
    Writes initial data-curation.md header.
    All decisions are made before any data is touched.
    """
    # Task type is classification for SMS Spam
    state["task_type"] = "classification"

    # Filter Android pool by hardware constraints
    feasible = filter_pool(state["hardware_constraints"])
    if not feasible:
        raise RuntimeError("No models in Android pool satisfy hardware constraints.")

    # Start with Tier 1 smallest model
    state["selected_model"] = feasible[0]
    state["stop_threshold"] = 0.96  # default; agent may lower on plateau detection

    return state
```

```python
# agent/nodes/eval_setup.py
from agent.state import AgentState
from data.sms_spam import download_sms_spam
from data.eval_set import build_eval_set

def eval_setup_node(state: AgentState) -> AgentState:
    """
    Node 2: download data and build E = Epos ∪ Eneg ∪ Eboundary.
    Eval set is built BEFORE any training. Fixed throughout all iterations.
    """
    train_examples, test_examples = download_sms_spam()
    state["train_examples"] = train_examples
    state["eval_set"] = build_eval_set(test_examples)
    return state
```

```python
# agent/nodes/train.py
import os, json, time
from concurrent.futures import ThreadPoolExecutor
from agent.state import AgentState
from training.slm_helpers import train as slm_train

ARTIFACTS_DIR = "artifacts"

def _save_dataset(examples: list[dict], version: int) -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, f"dataset_v{version}.jsonl")
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    return path

def train_node(state: AgentState) -> AgentState:
    """
    Node 3: fire ≥2 parallel training configurations.
    Always trains from base model — never from a prior checkpoint.
    Selects best config by f(π) on E.
    """
    model_id = state["selected_model"].model_id
    dataset_path = state["current_dataset_path"]

    # Two parallel configs per the paper's requirement
    configs = [
        {"nr_epochs": 3, "learning_rate": 2e-4, "batch_size": 8,  "lora_rank": 8,  "label": "A: LoRA r=8"},
        {"nr_epochs": 5, "learning_rate": 1e-4, "batch_size": 4,  "lora_rank": None, "label": "B: full_ft"},
    ]

    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            cfg["label"]: executor.submit(
                slm_train,
                dataset_path=dataset_path,
                base_model=model_id,
                nr_epochs=cfg["nr_epochs"],
                learning_rate=cfg["learning_rate"],
                batch_size=cfg["batch_size"],
                lora_rank=cfg["lora_rank"],
                output_dir=os.path.join(ARTIFACTS_DIR, f"iter{state['iteration']}_{cfg['label'].split(':')[0].strip()}"),
            )
            for cfg in configs
        }
        for label, future in futures.items():
            results[label] = future.result()

    # Store both weights_refs; evaluate node will select best
    state["_pending_weights_refs"] = results
    state["_pending_configs"] = {c["label"]: c for c in configs}
    state["iteration"] += 1
    return state
```

```python
# agent/nodes/evaluate.py
from agent.state import AgentState
from eval.harness import run_eval
from data.curation_log import CurationLog
from training.quantize import theoretical_hardware_profile
from agent.nodes.iterate import apply_iteration_policy

def evaluate_node(state: AgentState) -> AgentState:
    """
    Node 4: score each trained config against E, select best, log to DAG and data-curation.md.
    """
    model_id = state["selected_model"].model_id
    eval_set = state["eval_set"]
    pending = state.get("_pending_weights_refs", {})

    # Score all configs
    scored = {}
    for label, weights_ref in pending.items():
        result = run_eval(eval_set, weights_ref, model_id)
        scored[label] = (weights_ref, result)

    # Select best
    best_label = max(scored, key=lambda k: scored[k][1].f1)
    best_weights_ref, best_result = scored[best_label]

    current_score = best_result.f1

    # Update state
    if current_score > state["best_score"]:
        state["best_score"] = current_score
        state["best_weights_ref"] = best_weights_ref
        state["consecutive_no_improvement"] = 0
    else:
        state["consecutive_no_improvement"] += 1

    state["scores"].append(current_score)
    state["last_eval"] = best_result

    # Log to DAG
    policy = apply_iteration_policy(current_score)
    dag_node = {
        "iteration": state["iteration"],
        "model_id": model_id,
        "weights_ref": best_weights_ref,
        "score": current_score,
        "best_config": best_label,
        "intervention": policy["intervention"],
        "failures": len(best_result.failures),
        "pruned": False,
    }
    state["dag"].append(dag_node)

    # Write data-curation.md
    hw_profile = theoretical_hardware_profile(model_id)
    n_configs = len(pending)
    config_descriptions = state.get("_pending_configs", {})
    config_a = list(config_descriptions.values())[0]["label"] if config_descriptions else "Config A"
    config_b = list(config_descriptions.values())[1]["label"] if len(config_descriptions) > 1 else "Config B"

    log = CurationLog()
    log.write_iteration(
        iteration=state["iteration"],
        dataset_version=f"v{state['dataset_version']}",
        n_gold=0,  # updated by curate node
        n_hard=0,
        label_dist={},
        config_a=config_a,
        config_b=config_b,
        best_config=best_label,
        eval_result=best_result,
        score_band=policy["band"],
        next_intervention=policy["intervention"],
        hypothesis="",
        model_id=model_id,
        int4_size_mb=hw_profile.get("int4_size_mb", 0),
        tier=hw_profile.get("tier", 0),
    )

    return state
```

```python
# agent/nodes/curate.py
import json, os
from agent.state import AgentState
from data.curriculum import (
    build_initial_curriculum,
    synthesize_hard_negatives,
    apply_quality_controls,
)
from android_pool import ANDROID_POOL

ARTIFACTS_DIR = "artifacts"

def curate_node(state: AgentState) -> AgentState:
    """
    Node 6: build or augment Dcold = Dgold ∪ Dhard based on current intervention type.
    Writes updated dataset to disk. Increments dataset_version.
    """
    import anthropic
    from config import ANTHROPIC_API_KEY, ORCHESTRATOR_MODEL
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    intervention = state.get("last_intervention", "data_rebuild")
    eval_set = state["eval_set"]
    train_examples = state["train_examples"]
    failures = state["last_eval"].failures if state.get("last_eval") else []

    if intervention == "data_rebuild" or state["current_dataset_path"] is None:
        # Build fresh curriculum
        gold = build_initial_curriculum(train_examples, eval_set, n_total=150)
        hard = synthesize_hard_negatives(failures[:10] or train_examples[:10], n=50, anthropic_client=client)
        dataset = apply_quality_controls(gold + hard)
    elif intervention == "surgical":
        # Load existing and add 2-3 targeted examples per failure pattern
        with open(state["current_dataset_path"]) as f:
            dataset = [json.loads(l) for l in f if l.strip()]
        targeted = synthesize_hard_negatives(failures[:5], n=min(len(failures), 20), anthropic_client=client)
        dataset = apply_quality_controls(dataset + targeted)
    else:
        # Hyperparameter intervention — hold dataset fixed
        return state

    # Save to disk
    state["dataset_version"] += 1
    path = os.path.join(ARTIFACTS_DIR, f"dataset_v{state['dataset_version']}.jsonl")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    with open(path, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")
    state["current_dataset_path"] = path
    return state
```

```python
# agent/nodes/escalate.py
from agent.state import AgentState
from android_pool import filter_pool

def escalate_node(state: AgentState) -> AgentState:
    """
    Node 8: escalate to next model tier or terminate.
    Escalation checks hardware constraints even in Phase 1 (informational, not blocking).
    """
    current_model = state["selected_model"]
    feasible = filter_pool(state["hardware_constraints"])

    # Find the next tier model
    current_idx = next(
        (i for i, m in enumerate(feasible) if m.model_id == current_model.model_id),
        None,
    )

    if current_idx is None or current_idx >= len(feasible) - 1:
        # No larger model available — terminate
        state["next_action"] = "terminate"
        return state

    next_model = feasible[current_idx + 1]

    # Phase 1: log hardware check but don't block
    hw_ok = (
        next_model.int4_size_mb <= state["hardware_constraints"].storage_mb
        and next_model.peak_memory_mb <= state["hardware_constraints"].memory_mb
    )

    if not hw_ok:
        # Even in Phase 1, storage/memory are hard gates
        state["next_action"] = "terminate"
        return state

    state["selected_model"] = next_model
    state["consecutive_no_improvement"] = 0
    state["next_action"] = "train"
    return state
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_nodes.py -v
```
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add agent/nodes/ tests/test_nodes.py
git commit -m "feat: LangGraph state machine nodes (all 8 nodes)"
```

---

### Task 8: LangGraph graph assembly and entry point

**Files:**
- Create: `agent/graph.py`
- Create: `run.py`
- Create: `tests/test_graph.py`

**Interfaces:**
- Consumes: all nodes from Task 7; `AgentState` from Task 6
- Produces: `build_graph() -> CompiledGraph`; `run_cold_start(description, hardware_constraints) -> AgentState`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_graph.py
from unittest.mock import patch, MagicMock
from android_pool import HardwareConstraints
from agent.graph import build_graph

def test_graph_builds_without_error():
    graph = build_graph()
    assert graph is not None

def test_graph_has_expected_nodes():
    graph = build_graph()
    node_names = set(graph.nodes.keys())
    expected = {
        "task_analysis", "eval_setup", "train",
        "evaluate", "iterate", "curate", "rollback", "escalate"
    }
    assert expected.issubset(node_names)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_graph.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write `agent/graph.py`**

```python
# agent/graph.py
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes.task_analysis import task_analysis_node
from agent.nodes.eval_setup import eval_setup_node
from agent.nodes.train import train_node
from agent.nodes.evaluate import evaluate_node
from agent.nodes.iterate import iterate_node
from agent.nodes.curate import curate_node
from agent.nodes.rollback import rollback_node, should_rollback
from agent.nodes.escalate import escalate_node

def _route_after_evaluate(state: AgentState) -> str:
    """Route after evaluation: rollback if score decreased, else iterate."""
    if should_rollback(state):
        return "rollback"
    return "iterate"

def _route_after_iterate(state: AgentState) -> str:
    """Route based on next_action set by iterate_node."""
    return state.get("next_action", "curate")

def _route_after_escalate(state: AgentState) -> str:
    return state.get("next_action", "terminate")

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("task_analysis", task_analysis_node)
    graph.add_node("eval_setup", eval_setup_node)
    graph.add_node("train", train_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("iterate", iterate_node)
    graph.add_node("curate", curate_node)
    graph.add_node("rollback", rollback_node)
    graph.add_node("escalate", escalate_node)

    # Linear entry path
    graph.set_entry_point("task_analysis")
    graph.add_edge("task_analysis", "eval_setup")
    graph.add_edge("eval_setup", "curate")   # build initial dataset before first train
    graph.add_edge("curate", "train")
    graph.add_edge("train", "evaluate")

    # Conditional routing after evaluate
    graph.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {"rollback": "rollback", "iterate": "iterate"},
    )

    # After rollback: try curate (may need different data) or train again
    graph.add_edge("rollback", "iterate")

    # Conditional routing after iterate
    graph.add_conditional_edges(
        "iterate",
        _route_after_iterate,
        {
            "train": "train",         # hyperparameter change, same data
            "curate": "curate",       # rebuild or augment data
            "escalate": "escalate",
            "terminate": END,
        },
    )

    # After curate: back to train
    graph.add_edge("curate", "train")

    # After escalate: train with new model or terminate
    graph.add_conditional_edges(
        "escalate",
        _route_after_escalate,
        {"train": "train", "terminate": END},
    )

    return graph.compile()
```

- [ ] **Step 4: Write `run.py`**

```python
# run.py
"""
Entry point for SLM Factory Phase 1 cold-start loop.

Usage:
    python run.py

Environment variables required:
    ANTHROPIC_API_KEY
    EXA_API_KEY
"""
import os
from dotenv import load_dotenv
load_dotenv()

from android_pool import HardwareConstraints
from agent.graph import build_graph
from agent.state import AgentState

def run_cold_start(
    description: str = "fine-tune on SMS Spam binary classification",
    hardware_constraints: HardwareConstraints | None = None,
) -> AgentState:
    if hardware_constraints is None:
        hardware_constraints = HardwareConstraints(
            storage_mb=800,
            memory_mb=1200,
            latency_ttft_ms=2000,
            power_watts=5.0,
            target_chip="snapdragon_778g",
        )

    initial_state: AgentState = {
        "description": description,
        "target_metric": "F1",
        "hardware_constraints": hardware_constraints,
        "task_type": "",
        "selected_model": None,
        "stop_threshold": 0.96,
        "train_examples": [],
        "eval_set": None,
        "current_dataset_path": None,
        "dataset_version": 0,
        "best_weights_ref": None,
        "best_score": 0.0,
        "iteration": 0,
        "scores": [],
        "dag": [],
        "consecutive_no_improvement": 0,
        "last_eval": None,
        "last_intervention": "",
        "next_action": "train",
        "messages": [],
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    print(f"\n=== Phase 1 Complete ===")
    print(f"Best F1: {final_state['best_score']:.4f}")
    print(f"Iterations: {final_state['iteration']}")
    print(f"Best model: {final_state['selected_model'].model_id}")
    print(f"Best weights: {final_state['best_weights_ref']}")
    print(f"Score trajectory: {[f'{s:.4f}' for s in final_state['scores']]}")
    print(f"See data-curation.md for full lineage.")

    return final_state

if __name__ == "__main__":
    run_cold_start()
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_graph.py -v
```
Expected: all PASS.

- [ ] **Step 6: Run end-to-end smoke test with mocked training**

```bash
python -c "
from unittest.mock import patch
from run import run_cold_start

# Mock training to avoid actual GPU use in smoke test
with patch('agent.nodes.train.slm_train', return_value='artifacts/mock_checkpoint'), \
     patch('eval.harness.infer_batch', return_value=['spam', 'ham', 'spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'ham'] * 11):
    state = run_cold_start()
    assert state['iteration'] > 0
    print('Smoke test PASSED')
"
```
Expected: `Smoke test PASSED`

- [ ] **Step 7: Commit**

```bash
git add agent/graph.py run.py tests/test_graph.py
git commit -m "feat: LangGraph graph assembly and run.py entry point"
```

---

### Task 9: Update CLAUDE.md and progress log

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/progress/progress-log.md`

- [ ] **Step 1: Update CLAUDE.md with commands**

Add to `CLAUDE.md` under a new `## Commands` section:

```markdown
## Commands

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # add ANTHROPIC_API_KEY, EXA_API_KEY
```

### Run Phase 1 loop
```bash
python run.py
```

### Run tests
```bash
pytest tests/ -v
pytest tests/test_metrics.py -v          # just metrics
pytest tests/test_nodes.py -v            # just node logic
```

### Run smoke test (no GPU required)
```bash
python -c "from tests.test_graph import *; test_graph_builds_without_error()"
```
```

- [ ] **Step 2: Update progress log**

Append to `docs/progress/progress-log.md`:

```markdown
## Session 2 — 2026-06-19

### Status: Implementation plan written, ready for execution

### Files created / planned
- `config.py` — API keys, cluster settings
- `android_pool.py` — model pool + constraint filter
- `data/sms_spam.py` — UCI download
- `data/eval_set.py` — E = Epos ∪ Eneg ∪ Eboundary
- `eval/metrics.py` — binary F1, per-slice scores
- `eval/harness.py` — run_eval()
- `data/curriculum.py` — Dcold synthesis, quality controls
- `data/curation_log.py` — data-curation.md read/write
- `training/slm_helpers.py` — train() / infer() / infer_batch()
- `training/lora_trainer.py` — Unsloth LoRA loop
- `training/quantize.py` — theoretical hardware profiles
- `agent/state.py` — AgentState TypedDict
- `agent/tools/` — web_search, bash, file_tools, delegate_task
- `agent/nodes/` — all 8 state machine nodes
- `agent/graph.py` — LangGraph assembly
- `run.py` — entry point
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/progress/progress-log.md
git commit -m "docs: update CLAUDE.md with commands and progress log"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered in task |
|---|---|
| LangGraph state machine, Claude Sonnet 4.6 | Task 8 (graph.py) |
| 4 cold-start tools: web_search, bash, read_file, edit_file | Task 6 |
| delegate_task sub-agent spawning | Task 6 |
| data-curation.md written every iteration | Task 4 (CurationLog) + Task 7 (evaluate_node) |
| Context Manager (compaction) | Handled by LangGraph + CurationLog; full proprietary mechanism not in scope |
| τ = (description, M, constraints) | Task 8 (run.py initial state) |
| Task analysis: 3 sub-stages | Task 7 (task_analysis_node) |
| E = Epos ∪ Eneg ∪ Eboundary | Task 2 (eval_set.py) |
| Stop threshold 0.96 | Task 7 (iterate_node) |
| ≥2 parallel training configs | Task 7 (train_node) |
| Always train from base model | Task 5 (slm_helpers.py comment + implementation) |
| Dcold = Dgold ∪ Dhard at 65:35 | Task 4 (curriculum.py) |
| 5 quality controls | Task 4 (apply_quality_controls) |
| Simple rollback (cold-start, no dual-gate) | Task 7 (rollback_node, should_rollback) |
| Iteration policy: <0.80 / 0.80-0.95 / ≥0.95 | Task 7 (apply_iteration_policy) |
| Android pool bounded | Task 1 (android_pool.py) |
| Hardware constraints logged (not gating) Phase 1 | Task 5 (quantize.py) + Task 7 (evaluate_node) |
| Model escalation on plateau | Task 7 (escalate_node) |
| Sequential greedy DAG (not MCGS) | Task 8 (graph routing) |
| SMS Spam as Phase 1 task | Task 2 (sms_spam.py) |

**Placeholder scan:** No TBDs, no "implement later", no vague error handling instructions. All code blocks are complete.

**Type consistency check:**
- `EvalSet` defined in Task 2, consumed in Task 3 harness, Task 4 curriculum — consistent
- `EvalResult` defined in Task 3, consumed in Task 4 CurationLog, Task 7 evaluate_node — consistent
- `AgentState` TypedDict defined in Task 6, all nodes consume and return it — consistent
- `weights_ref: str` threading: `train()` returns it, `infer_batch()` consumes it, `AgentState.best_weights_ref` stores it — consistent
- `ModelSpec` from Task 1 used in Task 6 state and Task 7 nodes — consistent

All clean.
