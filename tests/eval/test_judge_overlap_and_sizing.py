"""B257/B258: curriculum never shrinks, and judging overlaps generation.

Run 38303490 dropped 754 already-curated rows when a bigger model's sizing formula returned a
smaller number, and spent 22% of wall time judging AFTER all generation had finished, with a
measured 0.0% overlap between the two GPUs.
"""
import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")


# --- curriculum ratchet -------------------------------------------------------

def test_curriculum_target_never_shrinks_between_tiers(monkeypatch):
    from agent import data_sizing

    monkeypatch.delenv("SLM_CURRICULUM_SIZE", raising=False)
    monkeypatch.setattr(
        data_sizing, "compute_curriculum_target", lambda **_: (5000, "formula said 5000")
    )
    state = {"curriculum_size_target": 5754, "selected_model": None}

    target = data_sizing.resize_curriculum_for_tier(state, log=lambda *_: None)

    assert target == 5754
    assert state["curriculum_size_target"] == 5754


def test_curriculum_target_still_grows_when_the_formula_asks_for_more(monkeypatch):
    from agent import data_sizing

    monkeypatch.delenv("SLM_CURRICULUM_SIZE", raising=False)
    monkeypatch.setattr(
        data_sizing, "compute_curriculum_target", lambda **_: (6838, "formula said 6838")
    )
    state = {"curriculum_size_target": 5754, "selected_model": None}

    assert data_sizing.resize_curriculum_for_tier(state, log=lambda *_: None) == 6838


def test_explicit_pin_still_wins_over_the_ratchet(monkeypatch):
    from agent import data_sizing

    monkeypatch.setenv("SLM_CURRICULUM_SIZE", "1200")
    state = {"curriculum_size_target": 5754, "selected_model": None}

    assert data_sizing.resize_curriculum_for_tier(state, log=lambda *_: None) == 1200


# --- per-task context ceilings ------------------------------------------------

def test_context_ceiling_is_task_aware_and_leaves_headroom(monkeypatch):
    from training.slm_helpers import task_max_seq_length

    monkeypatch.delenv("SLM_MAX_SEQ_LENGTH", raising=False)

    assert task_max_seq_length("classification") == 1024
    assert task_max_seq_length("generation") == 2048
    # APPS prompts plus a 1024-token completion genuinely need the full window.
    assert task_max_seq_length("code_generation") == 4096
    # Unknown/None falls back to the safe maximum rather than a tight guess.
    assert task_max_seq_length(None) == 4096

    # Every ceiling must still clear the task's own output reserve.
    from eval.harness import eval_output_token_reserve

    for task in ("classification", "generation", "math_reasoning", "NER", "code_generation"):
        assert eval_output_token_reserve(task) < task_max_seq_length(task), task


def test_explicit_max_seq_length_overrides_every_task(monkeypatch):
    from training.slm_helpers import task_max_seq_length

    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "3000")
    assert task_max_seq_length("classification") == 3000
    assert task_max_seq_length("code_generation") == 3000


def test_training_and_eval_agree_on_context(monkeypatch):
    """A row that fits training but not eval would be scored on a truncated prompt."""
    from training.lora_trainer import _configured_max_seq_length
    from training.slm_helpers import task_max_seq_length

    monkeypatch.delenv("SLM_MAX_SEQ_LENGTH", raising=False)
    for task in ("classification", "generation", "code_generation"):
        assert _configured_max_seq_length(task) == task_max_seq_length(task), task


# --- eval batch size ----------------------------------------------------------

def test_eval_batches_are_larger_than_the_pre_measurement_defaults(monkeypatch):
    from training.slm_helpers import _eval_batch_size

    monkeypatch.delenv("SLM_EVAL_BATCH_SIZE", raising=False)
    assert _eval_batch_size("generation") >= 16
    assert _eval_batch_size("classification") >= 32
    monkeypatch.setenv("SLM_EVAL_BATCH_SIZE", "3")
    assert _eval_batch_size("generation") == 3


# --- judge / generation overlap ----------------------------------------------

def test_judge_is_warmed_per_chunk_while_generation_continues(monkeypatch):
    """Each finished chunk must be handed to the judge before the next chunk is generated."""
    from data.eval_set import EvalSet
    from eval import harness

    rows = [{"text": f"dialogue {i}", "answer": f"summary {i}"} for i in range(10)]
    eval_set = EvalSet(all=rows, task_type="generation")
    monkeypatch.setenv("SLM_EVAL_JUDGE_OVERLAP_CHUNK", "4")

    order: list[str] = []
    warmed: list[tuple] = []

    class FakeJudge:
        def score_many(self, triples):
            triples = list(triples)
            order.append(f"judge:{len(triples)}")
            warmed.extend(triples)
            return [1.0] * len(triples)

    monkeypatch.setattr(harness, "_judge_overlap_chunk", lambda: 4)
    import eval.judge_client as judge_client

    monkeypatch.setattr(judge_client.LocalJudgeClient, "from_config", classmethod(lambda cls: FakeJudge()))

    def fake_infer(chunk):
        order.append(f"gen:{len(chunk)}")
        return [f"<think> </think> out {p}" for p in chunk]

    raw = harness._infer_overlapping_judge(
        [f"prompt {i}" for i in range(10)], eval_set, fake_infer
    )

    assert raw == [f"<think> </think> out prompt {i}" for i in range(10)]
    # Chunked into 4/4/2, and every row reached the judge exactly once.
    assert [o for o in order if o.startswith("gen")] == ["gen:4", "gen:4", "gen:2"]
    assert len(warmed) == 10
    # The judge receives the STRIPPED prediction, matching what generation.score would build.
    assert warmed[0] == ("dialogue 0", "summary 0", "out prompt 0")


def test_overlap_warm_failure_never_breaks_the_eval(monkeypatch):
    """score() is the authority; a failed warm must be invisible."""
    from data.eval_set import EvalSet
    from eval import harness
    import eval.judge_client as judge_client

    rows = [{"text": "d", "answer": "s"}]
    eval_set = EvalSet(all=rows, task_type="generation")

    class BrokenJudge:
        def score_many(self, triples):
            raise RuntimeError("judge down")

    monkeypatch.setattr(harness, "_judge_overlap_chunk", lambda: 4)
    monkeypatch.setattr(
        judge_client.LocalJudgeClient, "from_config", classmethod(lambda cls: BrokenJudge())
    )

    raw = harness._infer_overlapping_judge(["p"], eval_set, lambda chunk: ["answer"])
    assert raw == ["answer"]


def test_overlap_can_be_disabled(monkeypatch):
    from eval import harness

    monkeypatch.setenv("SLM_EVAL_JUDGE_OVERLAP_CHUNK", "0")
    assert harness._judge_overlap_chunk() == 0
