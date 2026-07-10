from agent.task_planner import _param_range_label


class _M:
    # Q4_K_M variants: size_mb is the 4-bit on-disk size (~0.55 GB/1B params).
    def __init__(self, size, quant="Q4_K_M"):
        self.size_mb = size
        self.quant = quant


def test_1b_model_range():
    pool = [_M(658)]  # Llama-3.2-1B Q4_K_M → 658/1000/0.55 ≈ 1.2B
    label = _param_range_label(pool)
    assert "1." in label, f"Expected ~1.2B, got {label}"


def test_range_min_max():
    pool = [_M(310), _M(2490)]  # MiniCPM 0.5B (Q4 310MB) to Phi-4-mini (Q4 2490MB)
    label = _param_range_label(pool)
    lo, hi = label.split("–")
    lo_val = float(lo.rstrip("B"))
    hi_val = float(hi.rstrip("B"))
    # 310/1000/0.55 ≈ 0.56B ; 2490/1000/0.55 ≈ 4.5B
    assert lo_val < 1.0, f"Low end should be sub-1B, got {lo_val}"
    assert hi_val > 4.0, f"High end should be ~4.5B, got {hi_val}"
