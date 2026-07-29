# CUDA Worker Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound pipeline VRAM across 1,500 graph steps by running every model-owning operation in a disposable process.

**Architecture:** A parent-side IPC runner streams child logs and exchanges trusted local pickle messages with `training.cuda_worker`. Training and evaluation wrappers delegate when `SLM_CUDA_ISOLATION=1`; the worker marks itself with `SLM_CUDA_WORKER=1` to execute locally without recursion. The pipeline runner enables isolation and reports failures with a nonzero exit.

**Tech Stack:** Python 3.11, `subprocess`, `pickle`, `tempfile`, PyTorch CUDA, pytest, SLURM.

## Global Constraints

- No tensor/model/trainer/optimizer crosses a process boundary.
- Train, eval/probes, and GGUF build each use fresh worker processes.
- Worker stdout/stderr streams into the parent logger.
- Worker errors preserve remote traceback and fail the parent operation.
- Direct library unit tests remain in-process unless isolation is enabled.
- Pipeline execution enables isolation by default.
- Graph exceptions produce a nonzero process exit and are never labeled budget exhaustion.
- Explicit inference-cache cleanup remains as defense in depth.
- Do not create a git commit unless explicitly requested.

---

### Task 1: Parent IPC runner and disposable worker

**Files:**
- Create: `training/cuda_isolation.py`
- Create: `training/cuda_worker.py`
- Create: `tests/training/test_cuda_isolation.py`

**Interfaces:**
- `isolation_enabled() -> bool`
- `run_isolated(operation: str, payload: dict) -> object`
- Worker operations: `ping`, `cuda_probe`, `train`, `eval`, `build_gguf`
- `CudaWorkerError(RuntimeError)` carries operation, exit code, and remote traceback.

- [ ] Write failing tests proving disabled/default behavior, worker-marker recursion prevention, real `ping` subprocess execution with a different PID, streamed log output, and unsupported-operation traceback propagation.
- [ ] Run `python -m pytest tests/training/test_cuda_isolation.py -q`; verify RED because modules do not exist.
- [ ] Implement `cuda_isolation.py`: write request pickle in a temporary directory, launch `sys.executable -m training.cuda_worker`, stream merged stdout/stderr line-by-line through `print`, load response, and raise `CudaWorkerError` on any failure.
- [ ] Implement `cuda_worker.py`: set `SLM_CUDA_WORKER=1` before operation imports; dispatch `ping`, `cuda_probe`, `train`, `eval`, and `build_gguf`; log PID/GPU and peak/exit CUDA allocations; atomically write `{ok, result}` or `{ok, error_type, error, traceback}`.
- [ ] Re-run focused tests; verify GREEN.

The parent runner must use this shape:

```python
def isolation_enabled() -> bool:
    return (
        os.environ.get("SLM_CUDA_ISOLATION", "0") == "1"
        and os.environ.get("SLM_CUDA_WORKER") != "1"
    )


def run_isolated(operation: str, payload: dict):
    with tempfile.TemporaryDirectory(prefix="slm-cuda-") as tmp:
        request_path = os.path.join(tmp, "request.pkl")
        response_path = os.path.join(tmp, "response.pkl")
        with open(request_path, "wb") as f:
            pickle.dump({"operation": operation, "payload": payload}, f)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(
            [sys.executable, "-m", "training.cuda_worker", request_path, response_path],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout or ():
            print(line, end="")
        return_code = proc.wait()
        response = _read_response(response_path)
        if return_code != 0 or not response.get("ok"):
            raise CudaWorkerError(operation, return_code, response)
        return response["result"]
```

The worker dispatch must call existing public APIs after setting the marker:

```python
if operation == "train":
    from training.slm_helpers import train
    result = train(**payload)
elif operation == "eval":
    from eval.harness import run_eval
    result = run_eval(**payload)
elif operation == "build_gguf":
    from agent.nodes.evaluate import _build_or_reuse_gguf
    result = _build_or_reuse_gguf(**payload)
```

---

### Task 2: Isolate training and every evaluation caller

**Files:**
- Modify: `training/slm_helpers.py:25-52`
- Modify: `eval/harness.py:17-74`
- Modify: `tests/training/test_slm_helpers.py` or create focused wrapper tests
- Modify: `tests/eval/test_harness.py`

**Interfaces:**
- Preserve `train(...) -> TrainingOutput`.
- Preserve `run_eval(...) -> EvalResult`.
- Add private local implementations so worker-marker execution cannot recurse.

- [ ] Write failing tests with `SLM_CUDA_ISOLATION=1` and mocked `run_isolated`, asserting `train` sends only primitive/path payload and returns the worker result.
- [ ] Write failing eval test asserting `run_eval` sends the picklable `EvalSet` and arguments to operation `eval`.
- [ ] Verify RED.
- [ ] Refactor `train` to clear explicit inference caches, delegate when enabled, otherwise call `_train_local`.
- [ ] Refactor `run_eval` to delegate when enabled, otherwise call `_run_eval_local`; clear parent inference cache before and after delegation.
- [ ] Verify focused tests GREEN and existing harness tests unchanged when isolation is disabled.

Because all probes call `run_eval`, this automatically covers baseline evaluation,
difficulty labeling, interpolation probes, downward probes, and quant-accuracy evaluation.

---

### Task 3: Isolate GGUF merge/quantization

**Files:**
- Modify: `agent/nodes/evaluate.py:100-113`
- Test: `tests/nodes/test_evaluate_node.py`

- [ ] Add a failing test with isolation enabled asserting quantized evaluation delegates operation `build_gguf` instead of running the merge in the parent.
- [ ] Verify RED.
- [ ] Add a small `_build_gguf_for_eval` wrapper that chooses `run_isolated("build_gguf", payload)` or `_build_or_reuse_gguf` in-process.
- [ ] Verify focused evaluate tests GREEN.

---

### Task 4: Accurate pipeline status and nonzero failures

**Files:**
- Create: `agent/pipeline_status.py`
- Modify: `tests/pipeline/run.py:30-40,385-415,553-583,end`
- Create: `tests/test_pipeline_status.py`

**Interfaces:**
- `run_heading(error: BaseException | None) -> str`
- `outcome_text(converged: bool, error: BaseException | None, model_label: str) -> str`
- `process_exit_code(error: BaseException | None) -> int`

- [ ] Write failing tests proving exceptions produce `RUN FAILED`, describe the exception rather than budget exhaustion, and return exit code 1.
- [ ] Verify RED.
- [ ] Implement the status helpers.
- [ ] In `run.py`, set `SLM_CUDA_ISOLATION=1` before ML imports, retain `pipeline_error` in the graph exception handler, use status helpers in the report, finish artifact writing, then `raise SystemExit(1)` when `pipeline_error` is set.
- [ ] Verify focused status tests and runner syntax.

---

### Task 5: Regression and L40S soak validation

**Files:**
- Create: `tests/pipeline/run_cuda_isolation_soak.slurm`

- [ ] Create a one-L40S job that calls `run_isolated("cuda_probe", {"mib": 4096})` 100 times, records `nvidia-smi memory.used` after each worker exit, and fails if post-exit memory grows by more than 256 MiB above the first-cycle baseline.
- [ ] Run unit/regression suites:

```bash
python -m pytest tests/training/ tests/eval/ tests/nodes/ tests/cold_start/ \
  tests/test_curriculum_hardneg.py tests/test_pipeline_status.py -q
```

- [ ] Check edited-file diagnostics.
- [ ] Submit the soak test to `gpu-l40s-intelligentsystems`, monitor it through completion, and verify all 100 cycles return to a stable baseline.
