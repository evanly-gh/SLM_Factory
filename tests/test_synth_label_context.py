"""The teacher is told what a label MEANS, not just its name (B267).

The verifier prompt showed the teacher only the label string, so it read the label as an ordinary
English word. On RouterBench — where `local` means "a small on-device model can answer this
correctly" — it rejected 70% of generated rows with verdicts that are correct answers to the wrong
question:

    REJECTED [local] 'What is 15 percent of 200?' — the utterance is a math question, not
                                                    related to local services or location
    REJECTED [cloud] 'The thick fog rolled in...'  — the utterance describes fog, not clouds

The teacher was not being too strict. The prompt was asking it to judge the topic instead of the
class. These tests pin that both the generator and the verifier receive the definitions.
"""
from data.curriculum import (
    _label_context_block,
    _synthesize_new_gold,
    verify_generated_labels,
)
from data.label_space import label_definitions_for

ROUTER_DEFS = label_definitions_for("routerbench")


def test_context_block_states_the_definition_and_warns_off_the_wording():
    block = _label_context_block("local", ROUTER_DEFS, ["local", "route"])
    assert "everyday meaning of the label word" in block
    assert "SMALL on-device model" in block
    # Both classes are listed, so the teacher can see the actual decision boundary.
    assert "'local'" in block and "'route'" in block


def test_context_block_is_empty_without_definitions():
    """Intent-style tasks need none — CLINC150's label IS a description of the utterance."""
    assert _label_context_block("accept_reservations", None, None) == ""
    assert _label_context_block("accept_reservations", {}, None) == ""


def test_verifier_prompt_carries_the_definitions():
    seen: list[str] = []

    def fake_generate(prompt, temperature, max_tokens):
        seen.append(prompt)
        return '{"valid": true, "reason": "ok"}'

    rows = [{"text": "What is 15 percent of 200?", "label": "local"}]
    kept = verify_generated_labels(
        rows,
        generate_fn=fake_generate,
        label_definitions=ROUTER_DEFS,
        all_labels=["local", "route"],
    )
    assert kept == rows
    assert seen, "verifier made no call"
    prompt = seen[0]
    assert "SMALL on-device model" in prompt
    assert "judge only whether the" in prompt  # the topic-vs-class instruction
    assert "everyday meaning of the label word" in prompt


def test_verifier_without_definitions_keeps_the_old_prompt_shape():
    seen: list[str] = []

    def fake_generate(prompt, temperature, max_tokens):
        seen.append(prompt)
        return '{"valid": true, "reason": "ok"}'

    verify_generated_labels(
        [{"text": "book me a table", "label": "accept_reservations"}],
        generate_fn=fake_generate,
    )
    assert "What the labels mean" not in seen[0]
    assert "accept_reservations" in seen[0]


def test_generator_prompt_carries_the_definitions():
    seen: list[str] = []

    def fake_generate(prompt, temperature, max_tokens):
        seen.append(prompt)
        return "Compute the determinant of a 4x4 matrix."

    rows = _synthesize_new_gold(
        [{"text": "Solve for x: 3x + 7 = 22", "label": "route"}],
        task_type="classification",
        n=1,
        generate_fn=fake_generate,
        label_definitions=ROUTER_DEFS,
    )
    assert len(rows) == 1 and rows[0]["label"] == "route"
    assert "escalated to a LARGER cloud model" in seen[0]


# --------------------------------------------------------------------------
# NER synthesis is a deliberate no-op, and now says so
# --------------------------------------------------------------------------

def test_ner_synthesis_produces_nothing_and_explains_why():
    """NER rows have no `label` to anchor in-class generation, and this generator cannot emit
    entity spans — so it must stay a no-op rather than fabricate gold. It had been silently
    returning 0 rows on every call (`requested 5693 -> kept 0`) with no explanation."""
    logs: list[str] = []
    calls: list[str] = []

    def fake_generate(prompt, temperature, max_tokens):
        calls.append(prompt)
        return "should never be called"

    rows = _synthesize_new_gold(
        [{"text": "Famotidine - associated delirium .",
          "entities": [{"text": "Famotidine", "type": "Chemical"}]}],
        task_type="NER",
        n=500,
        generate_fn=fake_generate,
        log=logs.append,
    )
    assert rows == []
    assert calls == [], "no teacher call may be made for a path that cannot produce a row"
    joined = " ".join(logs)
    assert "SKIPPED" in joined and "NER" in joined
    assert "gold-only" in joined
