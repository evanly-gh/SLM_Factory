# CUDA worker isolation

## Goal

Keep VRAM bounded for the full 1,500-step pipeline by ensuring every GPU-heavy operation
runs in a disposable process. No model, optimizer, trainer, adapter, compiled graph, or CUDA
tensor crosses a graph-node boundary.

## Root cause

The current LangGraph pipeline executes 64+ train/evaluate cycles in one Python process.
Unsloth, Accelerate, PEFT, Trainer, and inference caches can retain references after local
variables are deleted. `torch.cuda.empty_cache()` only releases unused allocator blocks; it
cannot release live tensors. The failed run held 43.71 GiB in live PyTorch allocations and
only 113 MiB in unused reserved memory, so allocator cleanup could not prevent the OOM.

## Architecture

The parent LangGraph process owns only orchestration state, paths, examples, and metrics.
It delegates these operations to fresh external Python workers:

- LoRA training
- Evaluation, including difficulty and interpolation probes
- LoRA merge and GGUF quantization

Each worker receives a trusted local pickle request, performs one operation, writes a pickle
response, and exits. Exiting destroys the worker CUDA context and every hidden library
reference. The parent receives only `TrainingOutput`, `EvalResult`, or a GGUF path.

`tests/pipeline/run.py` enables isolation by default. Unit tests and direct library callers
remain in-process unless `SLM_CUDA_ISOLATION=1` is set. Workers set
`SLM_CUDA_WORKER=1` to prevent recursive delegation.

## Logging and errors

Worker stdout/stderr is streamed through the parent logger so run logs retain training
progress. Workers log operation, PID, logical GPU, and peak/exit allocated/reserved VRAM.
Worker exceptions include their traceback in the response and are re-raised by the parent.

The pipeline runner records its final report in `finally`, then exits nonzero after a graph
exception. It must never label an OOM or other exception as “budget exhausted.”

## Defense in depth

Before training and after evaluation, explicit inference caches are cleared. This lowers
normal peak usage but is not relied upon for correctness; process exit is the hard boundary.

## Compatibility

- Local vLLM synthesis remains a separate process on its assigned GPUs.
- Worker processes inherit `CUDA_VISIBLE_DEVICES`, API keys, model caches, and run paths.
- Existing direct unit tests stay in-process.
- Return types and node state schemas do not change.
- No tensor or model object is serialized.

## Verification

1. Unit-test request/response success, exception propagation, log streaming, and recursion
   prevention with lightweight fake operations.
2. Verify training/evaluation wrappers delegate only when isolation is enabled.
3. Verify pipeline exceptions produce a nonzero exit after artifact/report writing.
4. Run existing node, evaluation, training, and cold-start suites.
5. Run an L40S soak test that repeatedly creates GPU allocations in isolated workers and
   confirms parent/idle GPU memory returns to a stable baseline across many cycles.
