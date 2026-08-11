# tests/test_revalidation_roundtrip.py
"""
`iterate_node` re-validates the decision that `_llm_iterate` already validated ("defense in
depth", allow_internal=True). That second pass must accept the first pass's OWN output.

It did not. Validation stored the NORMALIZED hyperparams — which include the trainer-derived
lora_alpha, lora_dropout, micro_batch_size, gradient_accumulation_steps, effective_batch_size
and batch_size — back into the decision, and the retired-key check then rejected them on the
second pass. Every `hyperparameter` decision died with an error naming six fields the
orchestrator never proposed, and since that pass sits outside the reask path, no
self-correction ran — the "validation failures but ZERO reasks" puzzle from B223.

Fixed at the source: the decision now carries ONLY the five tunable knobs. See B244.
"""
import pytest

from agent.nodes.iterate import _validate_decision_json
from training.hparams import normalize_hyperparams

DERIVED_FIELDS = (
    "lora_alpha", "lora_dropout", "micro_batch_size",
    "gradient_accumulation_steps", "effective_batch_size", "batch_size",
)


def _llm_decision() -> dict:
    """What the orchestrator actually sends: only the five tunable knobs."""
    return {
        "intervention": "hyperparameter",
        "hypothesis": "hard bucket lags; raise capacity",
        "hyperparams": {
            "lora_rank": 32,
            "alpha_ratio": 4,
            "weight_decay": 0.01,
            "learning_rate": 2e-4,
            "nr_epochs": 3,
        },
    }


class TestDecisionCarriesOnlyWhatWasDecided:
    def test_no_derived_fields_are_added(self):
        once = _validate_decision_json(_llm_decision(), state={})
        for field in DERIVED_FIELDS:
            assert field not in once["hyperparams"], f"{field} should not be in the decision"

    def test_the_five_tunable_knobs_survive(self):
        once = _validate_decision_json(_llm_decision(), state={})
        assert set(once["hyperparams"]) == {
            "lora_rank", "alpha_ratio", "weight_decay", "learning_rate", "nr_epochs",
        }

    def test_alpha_ratio_is_preserved_not_dropped(self):
        """Normalization drops alpha_ratio, so the decision log used to print None."""
        once = _validate_decision_json(_llm_decision(), state={})
        assert once["hyperparams"]["alpha_ratio"] == 4

    def test_alpha_ratio_records_the_SNAPPED_value(self):
        """3 is equidistant from 2 and 4; _snap tie-breaks low, so 2 was actually applied."""
        decision = _llm_decision()
        decision["hyperparams"]["alpha_ratio"] = 3
        once = _validate_decision_json(decision, state={})
        assert once["hyperparams"]["alpha_ratio"] == 2


class TestValidationIsIdempotent:
    def test_second_pass_accepts_the_first_pass_output(self):
        once = _validate_decision_json(_llm_decision(), state={})
        twice = _validate_decision_json(once, state={}, allow_internal=True)
        assert twice["hyperparams"] == once["hyperparams"]

    def test_third_pass_also_survives(self):
        d = _validate_decision_json(_llm_decision(), state={})
        for _ in range(2):
            d = _validate_decision_json(d, state={}, allow_internal=True)
        assert d["hyperparams"]["lora_rank"] == 32

    @pytest.mark.parametrize("rank,ratio,expected_alpha", [
        (16, 1, 16), (16, 2, 32), (16, 4, 64),
        (32, 1, 32), (32, 2, 64), (32, 4, 128),
    ])
    def test_alpha_is_reconstructed_exactly_by_the_trainer(self, rank, ratio, expected_alpha):
        """
        The decision keeps the ratio; the trainer re-derives absolute alpha. This must round-trip
        exactly, including through re-validation — otherwise alpha silently reverts to the
        default ratio of 2, which is wrong for 4 of these 6 cases.
        """
        decision = _llm_decision()
        decision["hyperparams"].update({"lora_rank": rank, "alpha_ratio": ratio})
        once = _validate_decision_json(decision, state={})
        twice = _validate_decision_json(once, state={}, allow_internal=True)
        assert twice["hyperparams"]["alpha_ratio"] == ratio

        trainer_config, _ = normalize_hyperparams(twice["hyperparams"])
        assert trainer_config["lora_alpha"] == expected_alpha


class TestRetiredKeysStillRejectedFromTheModel:
    @pytest.mark.parametrize("field", DERIVED_FIELDS)
    def test_model_proposing_a_retired_knob_is_still_an_error(self, field):
        decision = _llm_decision()
        decision["hyperparams"][field] = 8
        with pytest.raises(ValueError, match="no longer tunable"):
            _validate_decision_json(decision, state={})

    def test_unknown_field_is_still_rejected(self):
        decision = _llm_decision()
        decision["hyperparams"]["not_a_real_knob"] = 1
        with pytest.raises(ValueError, match="unsupported hyperparameter field"):
            _validate_decision_json(decision, state={})
