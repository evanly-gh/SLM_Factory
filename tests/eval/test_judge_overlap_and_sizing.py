"""B258: per-task context ceilings, eval batching, and judging that overlaps generation.

Run 38303490 spent 22% of its wall time judging AFTER all generation had finished, with a measured
0.0% overlap between the two GPUs.

This file used to open with B257, the curriculum-size ratchet: a per-tier sizing formula could
return a smaller number for a bigger model and drop 754 already-curated rows, so the target was
made monotonic. Both the formula and the target are gone (2026-08-19) — the curriculum is
cumulative and only `_dedupe_into` grows it, which makes shrinking structurally impossible rather
than something a ratchet has to prevent. The surviving invariant is asserted end to end in
`tests/nodes/test_cumulative_curriculum.py`.
"""
import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("EXA_API_KEY", "test-key")


# --- per-task context ceilings ------------------------------------------------

def test_context_ceiling_is_task_aware_and_leaves_headroom(monkeypatch):
    from tasks import TASKS
    from training.slm_helpers import task_max_seq_length

    monkeypatch.delenv("SLM_MAX_SEQ_LENGTH", raising=False)

    # A one-word answer needs no room to reason; a summary or a chain of thought does.
    assert task_max_seq_length("clinc150") == 1024
    assert task_max_seq_length("dialogsum") == 2048

    # Every ceiling must still clear the task's own output reserve. The spec enforces this at
    # import time, which is why an unknown task can no longer receive a generous 4096 default.
    from eval.harness import eval_output_token_reserve

    for task in sorted(TASKS):
        assert eval_output_token_reserve(task) < task_max_seq_length(task), task


def test_explicit_max_seq_length_overrides_every_task(monkeypatch):
    from training.slm_helpers import task_max_seq_length

    monkeypatch.setenv("SLM_MAX_SEQ_LENGTH", "3000")
    assert task_max_seq_length("clinc150") == 3000
    assert task_max_seq_length("gsm8k") == 3000


def test_training_and_eval_agree_on_context(monkeypatch):
    """A row that fits training but not eval would be scored on a truncated prompt."""
    from tasks import TASKS
    from training.lora_trainer import _configured_max_seq_length
    from training.slm_helpers import task_max_seq_length

    monkeypatch.delenv("SLM_MAX_SEQ_LENGTH", raising=False)
    for task in sorted(TASKS):
        assert _configured_max_seq_length(task) == task_max_seq_length(task), task


# --- eval batch size ----------------------------------------------------------

def test_eval_batches_are_larger_than_the_pre_measurement_defaults(monkeypatch):
    from training.slm_helpers import _eval_batch_size

    monkeypatch.delenv("SLM_EVAL_BATCH_SIZE", raising=False)
    assert _eval_batch_size("dialogsum") >= 16
    assert _eval_batch_size("clinc150") >= 32
    monkeypatch.setenv("SLM_EVAL_BATCH_SIZE", "3")
    assert _eval_batch_size("dialogsum") == 3


# --- judge / generation overlap ----------------------------------------------

def test_judge_is_warmed_per_chunk_while_generation_continues(monkeypatch):
    """Each finished chunk must be handed to the judge before the next chunk is generated."""
    from data.eval_set import EvalSet
    from eval import harness

    rows = [{"text": f"dialogue {i}", "answer": f"summary {i}"} for i in range(10)]
    eval_set = EvalSet(all=rows, task="dialogsum")
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
    eval_set = EvalSet(all=rows, task="dialogsum")

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
