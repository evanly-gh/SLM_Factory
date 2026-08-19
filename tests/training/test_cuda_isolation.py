import os
from unittest.mock import patch

import pytest


def test_isolation_enabled_only_for_parent(monkeypatch):
    from training.cuda_isolation import isolation_enabled

    monkeypatch.delenv("SLM_CUDA_ISOLATION", raising=False)
    monkeypatch.delenv("SLM_CUDA_WORKER", raising=False)
    assert isolation_enabled() is False

    monkeypatch.setenv("SLM_CUDA_ISOLATION", "1")
    assert isolation_enabled() is True

    monkeypatch.setenv("SLM_CUDA_WORKER", "1")
    assert isolation_enabled() is False


def test_ping_runs_in_fresh_process_and_streams_logs(capsys):
    from training.cuda_isolation import run_isolated

    result = run_isolated("ping", {"value": "ok"})

    assert result["value"] == "ok"
    assert result["pid"] != os.getpid()
    assert "[cuda-worker] start op=ping" in capsys.readouterr().out


def test_unsupported_operation_propagates_remote_traceback():
    from training.cuda_isolation import CudaWorkerError, run_isolated

    with pytest.raises(CudaWorkerError) as exc:
        run_isolated("not-an-operation", {})

    message = str(exc.value)
    assert "not-an-operation" in message
    assert "ValueError" in message
    assert "Unsupported CUDA worker operation" in message
    assert exc.value.remote_error_type == "ValueError"


def test_parent_interruption_terminates_worker(monkeypatch):
    from training.cuda_isolation import run_isolated

    class ExplodingOutput:
        def __iter__(self):
            raise KeyboardInterrupt()

        def close(self):
            pass

    class FakeProcess:
        def __init__(self):
            self.stdout = ExplodingOutput()
            self.terminated = False
            self.killed = False

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return None if not self.terminated else 0

    process = FakeProcess()
    monkeypatch.setattr("training.cuda_isolation.subprocess.Popen", lambda *a, **k: process)

    with pytest.raises(KeyboardInterrupt):
        run_isolated("ping", {})

    assert process.terminated is True


def test_merge_and_quantize_delegates_when_isolation_enabled(monkeypatch):
    from training.cuda_isolation import merge_and_quantize

    monkeypatch.setenv("SLM_CUDA_ISOLATION", "1")
    monkeypatch.delenv("SLM_CUDA_WORKER", raising=False)
    with patch("training.cuda_isolation.run_isolated", return_value="/out/model.gguf") as worker:
        result = merge_and_quantize("/ckpt", "/merged", "/gguf", "Q4_K_M")

    assert result == "/out/model.gguf"
    worker.assert_called_once_with(
        "merge_quantize",
        {
            "checkpoint_path": "/ckpt",
            "merged_output_dir": "/merged",
            "gguf_output_dir": "/gguf",
            "quant": "Q4_K_M",
        },
    )


def test_cuda_worker_dispatches_infer_batch_as_one_operation():
    from training.cuda_worker import _dispatch

    payload = {
        "prompts": ["p1", "p2"],
        "weights_ref": "/weights",
        "base_model": "model-id",
        "max_new_tokens": 77,
        "max_workers": 3,
        "task": "ner_bc5cdr",
    }
    with patch(
        "training.slm_helpers.infer_batch",
        return_value=["first", "second"],
    ) as infer_batch:
        result = _dispatch("infer_batch", payload)

    assert result == ["first", "second"]
    infer_batch.assert_called_once_with(**payload)
