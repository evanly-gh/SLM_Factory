from agent.task_planner import _param_range_label


class _M:
    def __init__(self, size): self.int4_size_mb = size; self.quant = None


def test_1b_model_range():
    pool = [_M(658)]  # Llama-3.2-1B Q4_K_M
    label = _param_range_label(pool)
    # params_b = 658*2/1000 = 1.316B → should show ~1.3B
    assert "1." in label, f"Expected ~1.3B, got {label}"


def test_range_min_max():
    pool = [_M(310), _M(2490)]  # MiniCPM 0.5B to Phi-4-mini 5B
    label = _param_range_label(pool)
    lo, hi = label.split("–")
    lo_val = float(lo.rstrip("B"))
    hi_val = float(hi.rstrip("B"))
    assert lo_val < 1.0, f"Low end should be sub-1B, got {lo_val}"
    assert hi_val > 4.0, f"High end should be ~5B, got {hi_val}"
