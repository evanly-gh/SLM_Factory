# Qwen-3.6 Accuracy Goal + Format Verifiers — Design

**Date:** 2026-08-01
**Status:** approved design, pending spec review
**Companion doc:** [2026-07-31-six-task-benchmark-selection.md](../../Evan's%20Notes/2026-07-31-six-task-benchmark-selection.md)
(progress log for the build lives there)

---

## 1. Goal

Three coupled changes, driven by the user's request:

1. **Accuracy goal = the separately-hosted Qwen 3.6 base performance.** Replace the
   registry/deferred `stop_threshold` calibration with one anchored to Qwen 3.6's *measured
   zero-shot score on this run's own frozen eval set*, floored at **0.8** and capped at the
   existing `THRESHOLD_CEILING` (0.99). The goal a run must beat is therefore
   `min(0.99, max(measured_qwen, 0.8))`.
2. **Two new task types + verifiers** — `function_call` (BFCL-style AST arg-match) and `diff`
   (`git apply --check` + applied-result match), added as lightweight scorer modules, not a
   full `TaskContract`/verifier registry.
3. **Six dataset loaders + eval-set slice mapping** so all six benchmark tasks
   (CLINC150, DialogSum/SAMSum, xlam/BFCL, CoEdIT, RouterBench, MedQA) can build an eval set.

The binding rule (unchanged): **one comparison scalar** (`EvalResult.f1`) drives
`best_score` / stagnation / rollback. For the two new verifiers, `f1 = content_correct`; the
`format_valid` number rides alongside in `per_class` so `EvalResult` does not change.

### What is verifiable on this dev box vs. the cluster

- **Fully unit-testable here (Windows dev box):** the two scorers, the threshold math, the
  task-type registration wiring, the endpoint-baseline measurement logic (via a mocked
  `generate_fn`), and each loader's row-shaping on a small in-memory sample.
- **Cluster-only (Qwen endpoint unreachable here — `SLM_SYNTH_ENDPOINT` unset):** the *live*
  Qwen 3.6 number and live HF dataset pulls. The code path is exercised here with mocks; the
  real measurement runs where the vLLM endpoint is reachable.

---

## 2. Component 1 — `function_call` and `diff` task types + verifiers

### 2.1 Registration touch-points (both new types)

Adding a task type requires edits in exactly these places (verified this session):

| File | Change |
|---|---|
| `data/eval_set.py` | add `function_call`, `diff` to `TASK_TYPES`; both map to the **generation** split family (`_GENERATION_FAMILY`) |
| `agent/nodes/cold_start/task_analysis.py` | add both to `_VALID_TASK_TYPES` |
| `eval/harness.py` | add `_EVAL_OUTPUT_TOKEN_SETTINGS` entries (`function_call`→256, `diff`→512); add `TASK_METRIC_NAMES` entries (`function_call`→`ast_arg_match`, `diff`→`apply_match`); add dispatch branches in `_run_eval_local` |
| `data/loaders/dataset_integrity.py` | add `TASK_REQUIRED_FIELDS`: `function_call`→`("text","answer")`, `diff`→`("text","answer")` |
| `agent/task_planner.py` | add both to `_VALID`; document both in `_PLANNER_PROMPT` so autonomous runs can select them |

`function_call` and `diff` rows both use the `{text, answer}` schema (the existing generation
schema), so `dataset_integrity.validate_rows` already validates them with no new branch — only
the `TASK_REQUIRED_FIELDS` entries are added. `answer` holds the gold call-JSON / gold unified
diff respectively.

### 2.2 `eval/scorers/function_call.py`

BFCL-style AST argument match. Gold `answer` is a JSON string encoding a list of calls
`[{"name": str, "arguments": {..}}]`; the model is prompted (in `build_prompts`) to emit the
same JSON.

- `build_prompts(eval_set)` — instruct the model to output ONLY the call JSON for the given
  query + available-function signatures (signatures carried on the row, e.g. `tools` field).
- `extract_predictions(raw, eval_set)` — pull the first JSON array/object out of each raw
  string (tolerate fences/prose), same tolerance style as `generation._extract_code`.
- `score(eval_set, predictions)` — per row compute:
  - `format_valid` = prediction parses as JSON of the expected call shape (1.0/0.0).
  - `content_correct` = **all** of: (1) function name ∈ the row's allowed set,
    (2) name matches gold, (3) all gold-required args present, (4) values equal gold with
    light type coercion (str/num). 1.0 only if every predicted call matches gold; else 0.0.
  - Returns the standard scorer dict: `f1 = mean(content_correct)`,
    `metric = "ast_arg_match"`, `per_class = {"ast_arg_match": <f1>, "format_valid": <mean>}`,
    `slices = per_slice_scores(...)`, `failures = [...]`.

### 2.3 `eval/scorers/diff.py`

Prose-edit as a unified diff. Gold `answer` is the unified diff of `src`→`tgt`; the row carries
`src` (original text) and `tgt` (edited text). `git apply --check` decides format validity.

- `build_prompts(eval_set)` — instruct the model to output ONLY a unified diff that edits the
  given source per the instruction.
- `extract_predictions` — strip fences, keep the diff body.
- `score`:
  - `format_valid` = writing `src` to a temp file and running `git apply --check <diff>` in a
    temp dir exits 0 (git is on PATH here; if git is missing, `format_valid=0` and a diagnostic
    is recorded — never crash the run).
  - `content_correct` = after `git apply`, the resulting file text equals `tgt` (exact match
    after normalizing trailing whitespace/newline). 1.0/0.0.
  - Standard dict: `f1 = mean(content_correct)`, `metric = "apply_match"`,
    `per_class = {"apply_match": <f1>, "format_valid": <mean>}`.
  - All git invocations are `subprocess.run` with a short timeout in an isolated temp dir; no
    network, no repo mutation.

### 2.4 Harness dispatch

In `_run_eval_local`, extend the dispatch:

```python
elif task_type == "function_call":
    from eval.scorers import function_call as scorer
elif task_type == "diff":
    from eval.scorers import diff as scorer
```

Both scorers honor the same `build_prompts / extract_predictions / score` interface, so the
rest of `_run_eval_local` (token reserve, infer, `EvalResult(...)`) is unchanged.

---

## 3. Component 2 — Qwen-3.6-baseline accuracy goal

### 3.1 Why task_analysis parks and eval_setup measures

`task_analysis_node` runs **before** `eval_setup_node` builds the frozen `E`. Qwen cannot be
*measured* before `E` exists. So we reuse the existing deferred-calibration pattern:

- **task_analysis** parks the goal as *pending* with a new source tag, exactly like the
  existing `pending_measured_anchor` path (sets `stop_threshold = UNREACHABLE_PENDING_THRESHOLD`
  so nothing converges early).
- **eval_setup** performs the actual Qwen measurement right after `E` is built, then writes the
  immutable `initial_stop_threshold` once.

### 3.2 `agent/threshold.py` — new function

```python
def threshold_from_endpoint_baseline(measured: float, floor: float = 0.8) -> tuple[float, str]:
    value = min(THRESHOLD_CEILING, max(float(measured), floor))
    reason = f"Qwen-3.6 baseline {measured:.4f} floored at {floor:.2f}, capped at {THRESHOLD_CEILING}"
    return round(value, 4), reason
```

### 3.3 `agent/nodes/cold_start/task_analysis.py`

In `_calibrate_stop_threshold`, add a branch **above** the registry lookup, gated on a new
env flag `SLM_GOAL_FROM_QWEN` (default **on** per the user's request — the Qwen goal is now the
intended mechanism; `SLM_STOP_THRESHOLD` still overrides everything):

```python
if os.environ.get("SLM_GOAL_FROM_QWEN", "1") != "0":
    state["stop_threshold"] = UNREACHABLE_PENDING_THRESHOLD
    state["initial_stop_threshold"] = UNREACHABLE_PENDING_THRESHOLD
    state["threshold_calibration"] = {
        "source": "pending_qwen_baseline",
        "threshold": None, "headroom": None, "floor": 0.8,
        "reason": "goal = Qwen-3.6 base score on E, floored 0.8; measured in eval_setup",
        "pending": True,
    }
    return
```

### 3.4 `eval/endpoint_eval.py` (new)

```python
def measure_endpoint_baseline(eval_set, task_type, generate_fn=None, log=print) -> EvalResult:
    """Score Qwen 3.6 zero-shot on E using the task's scorer. generate_fn defaults to
    data.synth_client.get_generate_fn(); returns None-signal handling to the caller."""
```

- Resolves `generate_fn` from `data.synth_client.get_generate_fn()` when not injected.
- Reuses the **same scorer module** for `task_type` (`build_prompts` → generate over the
  endpoint → `extract_predictions` → `score`) so Qwen is scored by the identical metric the SLM
  will be judged on.
- Returns an `EvalResult` (so `.f1` is the comparable scalar).

### 3.5 eval_setup wiring

After `E` is built and persisted, if `threshold_calibration.source == "pending_qwen_baseline"`:

```python
from data.synth_client import get_generate_fn
gen = get_generate_fn(log=print)
if gen is None:
    # endpoint unreachable (this dev box): degrade honestly — park at floor, log it.
    threshold, reason = threshold_from_endpoint_baseline(0.0, floor=0.8)  # → 0.8
    provenance = {"measured_qwen": None, "endpoint": "unreachable", ...}
else:
    baseline = measure_endpoint_baseline(eval_set, task_type, generate_fn=gen)
    threshold, reason = threshold_from_endpoint_baseline(baseline.f1, floor=0.8)
    provenance = {"measured_qwen": baseline.f1, ...}
state["stop_threshold"] = threshold
state["initial_stop_threshold"] = threshold
state["threshold_calibration"] = {"source": "qwen_baseline", "threshold": threshold,
                                   "pending": False, "reason": reason, **provenance}
```

The unreachable-endpoint fallback floors the goal at 0.8 (never crashes, never silently uses a
bogus high goal). This matches the redesign's "degrade gracefully" stance for synthesis.

---

## 4. Component 3 — six dataset loaders + slice mapping

Six deterministic loaders under `data/loaders/`, each returning `(train, test)` lists of
schema-correct rows (pattern: `data/loaders/sms_spam.py`). HF pulls run on the cluster; here
each loader is unit-tested on a small in-memory/fixture sample for row shape.

| Loader (new file) | Dataset | task_type | Row schema |
|---|---|---|---|
| `clinc150.py` | CLINC150 | `classification` | `{text,label}` (label=intent; OOS→neg) |
| `dialogsum_samsum.py` | DialogSum + SAMSum | `generation` | `{text,answer}` (text=dialogue, answer=summary) |
| `xlam_bfcl.py` | xlam-60k (train) / BFCL (test) | `function_call` | `{text,answer,tools}` (answer=gold call JSON) |
| `coedit.py` | CoEdIT | `diff` | `{text,answer,src,tgt}` (answer=difflib unified diff of src→tgt) |
| `routerbench.py` | RouterBench | `classification` | `{text,label}` (label=`local_correct` derived) |
| `medqa.py` | MedQA-USMLE-4-opt | `classification` | `{text,label}` (MCQ; label=A/B/C/D) |

### 4.1 eval_setup non-autonomous loader branches

The non-autonomous branch of `eval_setup_node` currently handles only `classification`
(sms_spam) and raises `NotImplementedError` otherwise. Add a small task→loader dispatch there
(selectable via an env var, e.g. `SLM_BENCHMARK_TASK`, defaulting to sms_spam for
back-compat) so a chosen benchmark loads its `(train, test)` and flows through the existing
`build_eval_set` + overlap-firewall path. Autonomous runs keep using `web_acquire`.

### 4.2 CoEdIT gold diff

CoEdIT ships `src`/`tgt` full-text pairs, not diffs. The loader computes the gold unified diff
with Python `difflib.unified_diff(src, tgt)` and stores it in `answer` (free, exact). This is
the same gold the `diff` scorer applies with `git apply`.

---

## 5. Tests (all runnable on this box)

| Test file (new) | Covers |
|---|---|
| `tests/test_scorer_function_call.py` | AST match: exact call → 1.0; wrong name → 0.0 content but format_valid=1; missing arg → 0.0; malformed JSON → format_valid=0 |
| `tests/test_scorer_diff.py` | `git apply --check` passes for a valid diff (content=1.0); malformed diff → format_valid=0; correct-format-wrong-result → format_valid=1, content=0 |
| `tests/test_qwen_baseline_goal.py` | `threshold_from_endpoint_baseline` floor/cap; `measure_endpoint_baseline` with a mocked `generate_fn` returns a scored `EvalResult`; unreachable endpoint → goal parked at 0.8; `pending_qwen_baseline` set in task_analysis |
| `tests/test_new_task_type_registration.py` | `function_call`/`diff` accepted by `EvalSet`, `_VALID_TASK_TYPES`, `TASK_METRIC_NAMES`, `eval_output_token_reserve`, `required_fields_for_task`, `task_planner._VALID` |
| `tests/test_benchmark_loaders.py` | each loader shapes a small fixture into the correct schema; CoEdIT produces a `git apply`-able diff |

---

## 6. Files created / changed (to be logged in the six-task doc)

**New:** `eval/scorers/function_call.py`, `eval/scorers/diff.py`, `eval/endpoint_eval.py`,
`data/loaders/clinc150.py`, `data/loaders/dialogsum_samsum.py`, `data/loaders/xlam_bfcl.py`,
`data/loaders/coedit.py`, `data/loaders/routerbench.py`, `data/loaders/medqa.py`, plus the five
test files.

**Changed:** `data/eval_set.py`, `agent/nodes/cold_start/task_analysis.py`, `agent/threshold.py`,
`agent/nodes/cold_start/eval_setup.py`, `eval/harness.py`,
`data/loaders/dataset_integrity.py`, `agent/task_planner.py`, and the six-task benchmark doc
(progress log).

---

## 7. Build order

1. **A** — register `function_call`/`diff` task types + both verifiers + their tests.
2. **B** — Qwen-baseline goal: `threshold_from_endpoint_baseline`, `endpoint_eval.py`,
   task_analysis park branch, eval_setup measurement + tests.
3. **C** — six loaders + eval_setup dispatch + tests.

Each step is independently testable and committed separately.
