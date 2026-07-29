import inspect
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")


def test_interpolation_probe_uses_isolation_aware_train_wrapper():
    from agent.nodes.cold_start.model_selection import interpolation

    source = inspect.getsource(interpolation._probe_model)
    assert "run_lora_training" not in source
    assert "slm_train" in source
    assert "_build_gguf_for_eval" in source


def test_downward_probe_uses_isolation_aware_gpu_wrappers():
    from agent.nodes import downward_probe

    source = inspect.getsource(downward_probe._train_and_eval)
    assert "run_lora_training" not in source
    assert "merge_for_quantization" not in source
    assert "slm_train" in source
    assert "_build_gguf_for_eval" in source


def test_pipeline_final_verification_uses_isolated_merge_quantize():
    from pathlib import Path

    runner = Path(__file__).parents[1] / "pipeline" / "run.py"
    source = runner.read_text()
    assert "from training.lora_trainer import merge_for_quantization" not in source
    assert "from training.quantize import quantize_from_model_spec" not in source
    assert "from training.cuda_isolation import merge_and_quantize" in source
