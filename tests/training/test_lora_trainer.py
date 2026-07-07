# tests/training/test_lora_trainer.py
from training.lora_trainer import TrainingOutput, merge_for_quantization

def test_training_output_is_namedtuple():
    out = TrainingOutput(weights_ref="/some/path", gguf_path=None)
    assert out.weights_ref == "/some/path"
    assert out.gguf_path is None

def test_training_output_fields():
    out = TrainingOutput(weights_ref="/a", gguf_path="/b")
    assert out[0] == "/a"
    assert out[1] == "/b"
