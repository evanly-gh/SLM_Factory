# tests/test_generation_prompt_parity.py
"""
The generation family had two prompt defects, both visible in slm-dialogsum-samsum-cse-38186375.

1. The eval prompt was the hardcoded "Answer the following question:" for EVERY generation task.
   A dialogue transcript asks no question and the prompt never said "summarize", so the model
   continued the conversation: against the gold summary "Shelly is volunteering at a food
   shelter..." it replied "Shelly: How about you? Any volunteer work? Tracy: Nah. Not into that."

2. Training and eval built their inputs INDEPENDENTLY — training passed the bare text with no
   instruction at all — so the model was fine-tuned on one input shape and scored on another.

Fixed by one shared builder both sides call, with the instruction carried on the rows. See B250.
"""
from data.eval_set import EvalSet
from data.loaders.dialogsum import (
    SUMMARIZATION_INSTRUCTION,
    convert_dialogsum_rows,
)
from eval.scorers.generation import (
    DEFAULT_GENERATION_INSTRUCTION,
    build_generation_prompt,
    resolve_generation_instruction,
)
# `dialogsum` moved off the judge and onto multi-reference ROUGE on 2026-09-06, so its prompt now
# lives in `eval.scorers.summarization`. The B250 lessons this file guards are unchanged and still
# apply: ONE builder both sides call, and the instruction carried on the rows. What moved is which
# module owns them.
from eval.scorers.summarization import build_prompts, resolve_instruction
from tasks._builders import TrainingContext
from training.lora_trainer import _training_turn

_RAW = [{"dialogue": "Amanda: I baked cookies.\nJerry: Sure!", "summary": "Amanda baked cookies."}]


def _ctx(rows):
    """The dataset-level context a training turn needs, resolved the way the trainer resolves it.

    `dialogsum` has no closed label space, so there is no class vocabulary to pass — the trainer
    reads that off the spec rather than being told.
    """
    return TrainingContext(labels=(), instruction=resolve_instruction(rows))


class TestTrainEvalParity:
    """The defect that unit tests would previously have missed: two independent formats."""

    def test_trainer_and_eval_produce_the_identical_prompt(self):
        rows = convert_dialogsum_rows(_RAW)
        eval_prompt = build_prompts(EvalSet(all=rows, task="dialogsum"))[0]
        train_prompt, _target, _marker = _training_turn(rows[0], "dialogsum", _ctx(rows))
        assert train_prompt == eval_prompt

    def test_training_target_is_the_summary_not_the_prompt(self):
        rows = convert_dialogsum_rows(_RAW)
        _prompt, target, _marker = _training_turn(rows[0], "dialogsum", _ctx(rows))
        assert target == "Amanda baked cookies."

    def test_synthetic_rows_without_an_instruction_share_the_dataset_instruction(self):
        """
        Synthetic rows are built fresh and carry no `_instruction`. Resolving per ROW would give
        them a different prompt from the real rows in the same training set; resolving once per
        DATASET keeps the set consistent.
        """
        rows = convert_dialogsum_rows(_RAW) + [
            {"text": "A: hi\nB: hello", "answer": "They greet.", "references": ["They greet."]}
        ]
        ctx = _ctx(rows)
        prompts = [_training_turn(row, "dialogsum", ctx)[0] for row in rows]
        assert all(p.startswith(SUMMARIZATION_INSTRUCTION) for p in prompts)


class TestTheInstructionActuallyDescribesTheTask:
    def test_dialogsum_rows_carry_a_summarization_instruction(self):
        rows = convert_dialogsum_rows(_RAW)
        assert rows[0]["_instruction"] == SUMMARIZATION_INSTRUCTION

    def test_the_prompt_says_summarize_not_answer(self):
        rows = convert_dialogsum_rows(_RAW)
        prompt = build_prompts(EvalSet(all=rows, task="dialogsum"))[0]
        assert "Summarize" in prompt
        assert "Answer the following question" not in prompt

    def test_the_prompt_forbids_continuing_the_conversation(self):
        """Directly targets the observed failure mode."""
        assert "do not continue the conversation" in SUMMARIZATION_INSTRUCTION

    def test_the_dialogue_is_still_in_the_prompt(self):
        rows = convert_dialogsum_rows(_RAW)
        prompt = build_prompts(EvalSet(all=rows, task="dialogsum"))[0]
        assert "Amanda: I baked cookies." in prompt


class TestFallbackForTasksThatStateNothing:
    def test_rows_without_an_instruction_get_the_family_default(self):
        assert resolve_generation_instruction(
            [{"text": "What is 2+2?", "answer": "4"}]
        ) == DEFAULT_GENERATION_INSTRUCTION

    def test_default_preserves_the_previous_wording(self):
        """Math/QA behaviour must not change; only datasets that opt in are affected."""
        assert build_generation_prompt("What is 2+2?", DEFAULT_GENERATION_INSTRUCTION) == (
            "Answer the following question:\n\nWhat is 2+2?"
        )

    def test_empty_and_malformed_rows_do_not_crash(self):
        assert resolve_generation_instruction([]) == DEFAULT_GENERATION_INSTRUCTION
        assert resolve_generation_instruction(
            [None, "junk", {}]
        ) == DEFAULT_GENERATION_INSTRUCTION


class TestInstructionIsMetadata:
    def test_field_is_underscore_prefixed_so_synthesis_cannot_reword_it(self):
        """
        `_new_example_prompt` builds the teacher's JSON schema from non-underscore keys, so an
        underscore-prefixed instruction is never shown to the teacher and cannot be regenerated.
        """
        from agent.task_brief import brief_context_block
        from data.curriculum import _new_example_prompt
        from tasks import get_task

        row = convert_dialogsum_rows(_RAW)[0]
        # The description now comes from the orchestrator's task brief rather than a table keyed by
        # task type; what the row must not leak is unchanged.
        description = brief_context_block(None, get_task("dialogsum"))
        assert "_instruction" not in _new_example_prompt(row, description)
