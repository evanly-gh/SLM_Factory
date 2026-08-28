# tests/training/test_micro_batch_ceiling.py
"""
`VALID_MICRO_BATCH_SIZES` was `(1, 2, 4, 8)` with a DEFAULT of 8, so the orchestrator had exactly
one setting above the default on this axis: none. It is now `(1, 2, 4, 8, 16, 32)`.

Widening a bounded search space is a two-sided change and each side has a way of going wrong
quietly, which is what this file is for:

  * the new values must reach the trainer — `TrainingConfig` validates against the same tuple, and
    a normalization step that snapped 16 back down to 8 would look like the orchestrator ignoring
    its own decision;
  * everything that was previously guaranteed BY the narrow tuple must still hold. Chiefly
    `MAX_EFFECTIVE_BATCH_SIZE`, which the old ceiling could not reach with any legal accumulation
    and now can be blown past by `32 x 4`; and `DEFAULT_MICRO_BATCH_SIZE`, because moving the
    default with the ceiling would change the memory profile of every run that never asked for it —
    on a trainer whose docstring records that it has NO out-of-memory recovery.

Pure validation and normalization: no model, no CUDA, no dataset.
"""
import importlib
from itertools import islice

import pytest


def _hparams():
    """Resolve at call time — `tests/config` reloads modules, so a module-scope alias can go stale."""
    return importlib.import_module("training.hparams")


def _config(**overrides):
    """A minimal valid `TrainingConfig`; only the batch-shape fields matter here."""
    from training.lora_trainer import TrainingConfig

    return TrainingConfig(
        **{
            "base_model": "Qwen/Qwen3-1.7B",
            "nr_epochs": 1,
            "learning_rate": 2e-4,
            **overrides,
        }
    )


class TestTheGridItself:
    def test_the_declared_grid_is_the_powers_of_two_up_to_thirty_two(self):
        """One place records the axis; the trainer and the normalizer both read this tuple."""
        assert _hparams().VALID_MICRO_BATCH_SIZES == (1, 2, 4, 8, 16, 32)

    def test_the_grid_stops_where_the_effective_batch_ceiling_would_close_the_axis(self):
        """
        64 is excluded on purpose: with `MAX_EFFECTIVE_BATCH_SIZE` at 64 it would pin accumulation
        to 1, removing a whole axis from the search space to add one point to this one.
        """
        hparams = _hparams()
        largest = max(hparams.VALID_MICRO_BATCH_SIZES)
        usable_accumulations = [
            steps
            for steps in hparams.VALID_GRADIENT_ACCUMULATION_STEPS
            if largest * steps <= hparams.MAX_EFFECTIVE_BATCH_SIZE
        ]
        assert usable_accumulations == [1, 2]


class TestTheNewValuesReachTheTrainer:
    @pytest.mark.parametrize("micro_batch", [1, 2, 4, 8, 16, 32])
    def test_every_declared_size_is_accepted(self, micro_batch):
        """A value the orchestrator may choose must survive runtime validation unchanged."""
        config = _config(micro_batch_size=micro_batch)
        assert config.micro_batch_size == micro_batch
        assert config.batch_size == micro_batch  # the legacy alias is kept in step

    @pytest.mark.parametrize("micro_batch", [3, 24, 64])
    def test_off_grid_sizes_are_still_rejected(self, micro_batch):
        """
        Runtime validation stays strict — it is the last place a bad value can be caught before an
        hour of GPU time. 24 sits between two legal values and 64 is one doubling past the top, so
        both are the kind of near-miss a widened grid invites.
        """
        with pytest.raises(ValueError, match="micro_batch_size must be one of"):
            _config(micro_batch_size=micro_batch)

    @pytest.mark.parametrize("micro_batch", [16, 32])
    def test_the_legacy_batch_size_alias_accepts_the_new_values_too(self, micro_batch):
        """Checkpoint and DAG replay of older configs goes through `batch_size`, not the new name."""
        config = _config(batch_size=micro_batch)
        assert config.micro_batch_size == micro_batch
        assert config.effective_batch_size == micro_batch


class TestTheEffectiveBatchCeilingStillBinds:
    def test_thirty_two_by_four_is_refused(self):
        """
        128 > 64. The old ceiling of 8 could not exceed `MAX_EFFECTIVE_BATCH_SIZE` with any legal
        accumulation, so this guard was unreachable until the grid moved; now it is the only thing
        standing between the orchestrator and an effective batch twice the declared maximum.
        """
        with pytest.raises(ValueError, match="derived effective_batch_size=128 exceeds 64"):
            _config(micro_batch_size=32, gradient_accumulation_steps=4)

    def test_thirty_two_by_two_is_the_largest_legal_pair(self):
        """The ceiling is a bound, not a ban on the top of the grid."""
        config = _config(micro_batch_size=32, gradient_accumulation_steps=2)
        assert config.effective_batch_size == 64

    @pytest.mark.parametrize(
        ("micro_batch", "steps"),
        [(16, 8), (32, 4), (32, 8)],
    )
    def test_the_normalizer_refuses_the_same_pairs_the_trainer_does(self, micro_batch, steps):
        """
        Both boundaries must agree. If normalization let `32 x 4` through, the orchestrator would
        get a config that then died inside `TrainingConfig` — a validated decision failing at the
        point of use is the failure shape that hides reasks.
        """
        with pytest.raises(ValueError, match="exceeds bounded maximum 64"):
            _hparams().normalize_hyperparams(
                {"micro_batch_size": micro_batch, "gradient_accumulation_steps": steps}
            )

    @pytest.mark.parametrize(
        ("micro_batch", "steps", "effective"),
        [(16, 2, 32), (16, 4, 64), (32, 1, 32), (32, 2, 64)],
    )
    def test_legal_pairs_derive_the_effective_batch_the_trainer_derives(
        self, micro_batch, steps, effective
    ):
        """`effective_batch_size` is derived in two places; they must not disagree."""
        normalized, _ = _hparams().normalize_hyperparams(
            {"micro_batch_size": micro_batch, "gradient_accumulation_steps": steps}
        )
        assert normalized["effective_batch_size"] == effective
        config = _config(
            micro_batch_size=micro_batch, gradient_accumulation_steps=steps
        )
        assert config.effective_batch_size == effective


class TestTheDefaultDidNotMove:
    def test_the_declared_default_is_still_eight(self):
        """
        Raising a ceiling must not raise the default. `training/lora_trainer.py` has no OOM
        recovery, so a default of 16 or 32 would change the memory profile of every run that never
        asked for a larger batch — including long-sequence tasks — and an OOM there kills the run
        rather than backing off.
        """
        hparams = _hparams()
        assert hparams.DEFAULT_MICRO_BATCH_SIZE == 8
        assert hparams.DEFAULT_MICRO_BATCH_SIZE != max(hparams.VALID_MICRO_BATCH_SIZES)

    def test_a_config_that_names_no_batch_size_still_gets_eight(self):
        """The path every run without an explicit hyperparameter decision takes."""
        config = _config()
        assert config.micro_batch_size == 8
        assert config.gradient_accumulation_steps == 1
        assert config.effective_batch_size == 8

    def test_normalizing_an_empty_decision_still_yields_eight(self):
        """Same default from the orchestrator boundary, which is a separate implementation."""
        normalized, _ = _hparams().normalize_hyperparams({})
        assert normalized["micro_batch_size"] == 8
        assert normalized["batch_size"] == 8
        assert normalized["effective_batch_size"] == 8

    def test_the_default_is_reported_as_unchanged_rather_than_snapped(self):
        """A default that produced a "snapped" note would read as the orchestrator being corrected."""
        _, rationale = _hparams().normalize_hyperparams({})
        assert "snapped" not in rationale
        assert "micro batch=8" in rationale


class TestNormalizationOverTheWiderGrid:
    @pytest.mark.parametrize("micro_batch", [1, 2, 4, 8, 16, 32])
    def test_a_declared_grid_value_passes_through_untouched(self, micro_batch):
        """Snapping a legal value would silently overrule a decision the orchestrator made."""
        normalized, rationale = _hparams().normalize_hyperparams(
            {"micro_batch_size": micro_batch}
        )
        assert normalized["micro_batch_size"] == micro_batch
        assert "snapped" not in rationale
        assert f"micro batch={micro_batch}" in rationale

    @pytest.mark.parametrize(
        ("proposed", "expected"),
        [
            (12, 8),  # equidistant from 8 and 16; `_snap` tie-breaks low
            (13, 16),
            (16, 16),
            (24, 16),  # equidistant from 16 and 32
            (25, 32),
            (40, 32),
            (1024, 32),  # clamped by the top of the grid, not accepted
        ],
    )
    def test_an_off_grid_proposal_snaps_to_the_widened_grid(self, proposed, expected):
        """
        The orchestrator boundary is permissive: it snaps rather than rejects. The values that
        matter are the ones whose destination MOVED — 24 and 40 used to land on 8, so a run that
        asked for a large batch quietly got the default one.
        """
        normalized, _ = _hparams().normalize_hyperparams({"micro_batch_size": proposed})
        assert normalized["micro_batch_size"] == expected

    def test_the_new_values_are_part_of_the_hyperparameter_identity(self):
        """
        Identity drives dedup and checkpoint replay. If 16 and 32 collapsed onto the same identity
        as 8, the loop would treat a batch-shape change as an already-tried config and skip it.
        """
        identity = _hparams().hyperparameter_identity
        identities = {
            micro_batch: identity({"micro_batch_size": micro_batch})
            for micro_batch in (8, 16, 32)
        }
        assert len(set(identities.values())) == 3

    def test_identity_is_stable_across_the_alias_for_a_new_value(self):
        """`batch_size=16` and `micro_batch_size=16` are the same run and must not dedup apart."""
        identity = _hparams().hyperparameter_identity
        assert identity({"batch_size": 16}) == identity({"micro_batch_size": 16})

    def test_normalization_is_idempotent_for_a_new_value(self):
        """Re-validation re-normalizes its own output; a second pass must be a no-op."""
        normalize = _hparams().normalize_hyperparams
        once, _ = normalize({"micro_batch_size": 32, "gradient_accumulation_steps": 2})
        twice, _ = normalize(once)
        assert twice == once

    def test_a_conflicting_alias_pair_is_still_caught_at_the_new_resolution(self):
        """
        `batch_size=16` with `micro_batch_size=32` used to be invisible: both snapped to the old
        ceiling of 8 and agreed. The wider grid keeps them distinct, so the conflict is now
        reported instead of being resolved to a value neither caller asked for.
        """
        with pytest.raises(ValueError, match="batch_size and micro_batch_size conflict"):
            _hparams().normalize_hyperparams(
                {"micro_batch_size": 32, "batch_size": 16}
            )

    def test_a_claimed_effective_batch_must_still_match_the_derived_one(self):
        """The derived field is a cross-check, and 16 and 32 give it new ways to be wrong."""
        with pytest.raises(ValueError, match="conflicts with the derived value"):
            _hparams().normalize_hyperparams(
                {
                    "micro_batch_size": 16,
                    "gradient_accumulation_steps": 2,
                    "effective_batch_size": 16,
                }
            )


class TestTheSearchSpaceShapeIsUnchanged:
    def test_batch_shape_is_still_absent_from_the_neighbor_enumeration(self):
        """
        `deterministic_neighbor_configs` deliberately does not step batch shape: splitting a batch
        to fit VRAM does not change what the model learns, so iterations on that axis measured
        run-to-run noise. A wider grid must not have reopened it — six values would otherwise
        multiply the neighbor space with candidates that cannot move the score.
        """
        hparams = _hparams()
        base = {"micro_batch_size": 8, "gradient_accumulation_steps": 2}
        neighbors = list(
            islice(hparams.deterministic_neighbor_configs(base), 400)
        )

        assert neighbors  # the enumeration still produces alternatives on the real axes
        assert {neighbor["micro_batch_size"] for neighbor in neighbors} == {8}
        assert {neighbor["gradient_accumulation_steps"] for neighbor in neighbors} == {2}
        assert {neighbor["effective_batch_size"] for neighbor in neighbors} == {16}

    def test_a_base_at_the_new_top_of_the_grid_is_carried_into_every_neighbor(self):
        """A neighbor that reset the batch shape would compare configs that differ on two axes."""
        neighbors = list(
            islice(
                _hparams().deterministic_neighbor_configs({"micro_batch_size": 32}),
                200,
            )
        )

        assert neighbors
        assert {neighbor["micro_batch_size"] for neighbor in neighbors} == {32}


# --------------------------------------------------------------------------
# Vocabulary-aware micro-batch fitting (the gemma-3-270m-it OOM)
# --------------------------------------------------------------------------


def _fit_config(micro=8, accum=1, model="google/gemma-3-270m-it"):
    from types import SimpleNamespace

    return SimpleNamespace(
        base_model=model,
        micro_batch_size=micro,
        batch_size=micro,
        gradient_accumulation_steps=accum,
        effective_batch_size=micro * accum,
    )


def _fit_tokenizer(vocab):
    from types import SimpleNamespace

    return SimpleNamespace(vocab_size=vocab)


def test_a_wide_vocabulary_model_on_long_sequences_gets_a_smaller_micro_batch(capsys):
    """The measured OOM, as arithmetic (toolbench job 38820306).

    `google/gemma-3-270m-it` died on a 44 GiB L40S while training a 270M-parameter model. The
    allocation was the fp32 cross-entropy logits tensor, and it is exactly predictable:

        micro_batch 8 x 6,006 padded tokens x 262,144 vocab x 4 bytes = 46.9 GiB

    168M of that model's 270M parameters are a 262,144-token embedding table, so its output
    projection is 5.33x wider than SmolLM2's 49,152. Nothing had caught it because the OOM needs
    BOTH a wide vocabulary and long sequences, and every other pool model has a small vocabulary
    while every other task has far shorter rows.
    """
    from training.lora_trainer import _fit_micro_batch_to_logits

    config = _fit_config()
    _fit_micro_batch_to_logits(config, _fit_tokenizer(262_144), {"max": 6006})

    assert config.micro_batch_size < 8
    # The reduction must not silently change the optimization.
    assert config.effective_batch_size == 8
    assert config.micro_batch_size * config.gradient_accumulation_steps == 8
    out = capsys.readouterr().out
    assert "WIDE-VOCABULARY" in out
    assert "262,144 vocab" in out, "the log must name the cause, not just the symptom"
    assert "effective batch size unchanged" in out


def test_a_narrow_vocabulary_model_is_left_alone(capsys):
    """The guard must be narrow: SmolLM2 trains fine on the same rows and must not be slowed down.

    Same 6,006-token sequences, same micro-batch, 49,152 vocab -> 8.8 GiB, under the budget.
    """
    from training.lora_trainer import _fit_micro_batch_to_logits

    config = _fit_config(model="HuggingFaceTB/SmolLM2-360M-Instruct")
    _fit_micro_batch_to_logits(config, _fit_tokenizer(49_152), {"max": 6006})

    assert config.micro_batch_size == 8
    assert config.gradient_accumulation_steps == 1
    assert capsys.readouterr().out == ""


def test_short_sequences_are_left_alone_even_with_a_wide_vocabulary():
    """The other half of the conjunction. gemma on a short-row task must keep its micro-batch."""
    from training.lora_trainer import _fit_micro_batch_to_logits

    config = _fit_config()
    _fit_micro_batch_to_logits(config, _fit_tokenizer(262_144), {"max": 512})
    assert config.micro_batch_size == 8


def test_the_fitted_micro_batch_is_always_a_legal_value():
    """`TrainingConfig` validates against VALID_MICRO_BATCH_SIZES, so an off-tuple value would
    raise later, on the next config built from these hyperparameters."""
    from training.lora_trainer import _fit_micro_batch_to_logits

    valid = _hparams().VALID_MICRO_BATCH_SIZES
    accum_valid = _hparams().VALID_GRADIENT_ACCUMULATION_STEPS
    for vocab in (49_152, 151_936, 262_144):
        for seq in (512, 2048, 6006, 8192):
            for micro in valid:
                config = _fit_config(micro=micro)
                _fit_micro_batch_to_logits(config, _fit_tokenizer(vocab), {"max": seq})
                assert config.micro_batch_size in valid, (vocab, seq, micro)
                assert config.gradient_accumulation_steps in accum_valid, (vocab, seq, micro)


def test_a_missing_vocab_or_length_leaves_the_config_untouched():
    """An unknown budget must not silently halve the batch — absent information is not evidence."""
    from types import SimpleNamespace

    from training.lora_trainer import _fit_micro_batch_to_logits

    for tokenizer, summary in (
        (SimpleNamespace(), {"max": 6006}),
        (_fit_tokenizer(262_144), {}),
        (_fit_tokenizer(262_144), None),
    ):
        config = _fit_config()
        _fit_micro_batch_to_logits(config, tokenizer, summary)
        assert config.micro_batch_size == 8
