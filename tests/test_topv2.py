"""TOPv2: the SPIS reconstruction, the domain balance, and the extractive verifier.

Network-free by construction — every test drives the pure converters and samplers with rows built
here. The live parquet pull is exercised by `scripts/preflight_tasks.py`.
"""
from __future__ import annotations

import pytest

from data.eval_set import build_eval_set
from data.loaders.topv2 import (
    INTENTS,
    SLOTS,
    SOURCE_DOMAINS,
    TARGET_DOMAINS,
    _interleave_by_domain,
    convert_topv2_rows,
    parse_labels,
    resolve_spis,
    spis_sample,
)
from data.synth_verifiers import verify_semantic_parse_row
from eval.scorers import semantic_parse as scorer


def _row(utterance: str, parse: str, domain: str = "reminder") -> dict:
    return {"utterance": utterance, "semantic_parse": parse, "domain": domain}


# --------------------------------------------------------------------------
# SPIS sampling
# --------------------------------------------------------------------------


def test_parse_labels_counts_an_intent_or_slot_once_per_utterance():
    """SPIS is samples per intent and slot, so a slot used twice in one parse is ONE sample of it.

    Counting mentions instead would let a single utterance satisfy a label's whole quota, which
    defeats the purpose: the point of SPIS is coverage across examples, not occurrences.
    """
    parse = "[IN:CREATE_REMINDER [SL:DATE_TIME at 8 am ] and [SL:DATE_TIME 6 pm ] ]"
    assert parse_labels(parse) == {"IN:CREATE_REMINDER", "SL:DATE_TIME"}


def test_spis_sampling_gives_every_label_its_quota():
    """The property that makes SPIS worth implementing rather than taking N at random."""
    parses = (
        ["[IN:COMMON [SL:A x ] ]"] * 100
        + ["[IN:COMMON [SL:B x ] ]"] * 100
        + ["[IN:RARE [SL:C x ] ]"] * 5
    )
    chosen = spis_sample(parses, k=3)
    counts: dict[str, int] = {}
    for index in chosen:
        for label in parse_labels(parses[index]):
            counts[label] = counts.get(label, 0) + 1
    for label in ("IN:COMMON", "SL:A", "SL:B", "IN:RARE", "SL:C"):
        assert counts[label] >= 3, (label, counts)


def test_a_label_that_cannot_reach_the_quota_does_not_consume_the_corpus():
    """`min(k, available)` is what keeps a low-resource split low-resource.

    Reminder's train split has labels occurring exactly once (`SL:JOB`, `IN:GET_BIRTHDAY`).
    Without the clamp, the greedy loop keeps every row it sees while hunting for a 25th occurrence
    that does not exist, and the 25-SPIS split silently becomes the full 17,840-row split.
    """
    parses = ["[IN:COMMON [SL:A x ] ]"] * 500 + ["[IN:ONLY_ONCE [SL:Z x ] ]"]
    chosen = spis_sample(parses, k=25)
    assert len(chosen) < 60, f"clamp failed: kept {len(chosen)} of 501"
    # And the once-only label is still present, which is the whole reason it is clamped not skipped.
    assert any("IN:ONLY_ONCE" in parse_labels(parses[i]) for i in chosen)


def test_spis_sampling_is_deterministic_and_returns_corpus_order():
    """Two runs at one seed must produce byte-identical splits, or nothing downstream reproduces."""
    parses = [f"[IN:I{i % 7} [SL:S{i % 11} w{i} ] ]" for i in range(400)]
    first = spis_sample(parses, k=5)
    assert first == spis_sample(parses, k=5)
    assert first == sorted(first), "indices must be ascending, not shuffle order"
    assert first != spis_sample(parses, k=5, seed=999)


def test_spis_rejects_a_nonsense_quota():
    with pytest.raises(ValueError, match="must be positive"):
        spis_sample(["[IN:X a ]"], k=0)


def test_only_the_two_published_operating_points_are_selectable(monkeypatch):
    """An arbitrary k produces a split with no published counterpart and no baseline to cite."""
    assert resolve_spis(25) == 25
    assert resolve_spis(500) == 500
    monkeypatch.setenv("SLM_TOPV2_SPIS", "500")
    assert resolve_spis() == 500
    with pytest.raises(ValueError, match="must be one of"):
        resolve_spis(100)
    with pytest.raises(ValueError, match="must be one of"):
        resolve_spis("banana")


# --------------------------------------------------------------------------
# Row shaping and domain balance
# --------------------------------------------------------------------------


def test_the_parse_string_is_carried_verbatim():
    """Exact match is measured against this string, so any normalization here redefines the metric."""
    parse = "[IN:CREATE_REMINDER remind [SL:PERSON_REMINDED me ] ]"
    rows = convert_topv2_rows([_row("remind me", parse)])
    assert rows[0]["answer"] == parse
    assert rows[0]["text"] == "remind me"
    assert rows[0]["domain"] == "reminder"


def test_domain_filtering_separates_the_targets_from_the_source_domains():
    records = [
        _row("a", "[IN:X a ]", "reminder"),
        _row("b", "[IN:X b ]", "weather"),
        _row("c", "[IN:X c ]", "alarm"),
    ]
    assert {r["domain"] for r in convert_topv2_rows(records, TARGET_DOMAINS)} == {
        "reminder", "weather"}
    assert {r["domain"] for r in convert_topv2_rows(records, SOURCE_DOMAINS)} == {"alarm"}


def test_interleaving_keeps_every_prefix_domain_balanced():
    """THE BUG THIS EXISTS FOR.

    The mirror's test parquet is ordered by domain, and the loader's `max_test` truncation happens
    before `build_eval_set` can shuffle. So `rows[:1000]` was 1,000 reminder rows and zero weather
    rows — the headline is the MEAN of the two domains, so the in-loop eval was measuring one
    domain and reporting it as the average, on the one task whose point is the contrast.
    """
    rows = (
        [_row(f"r{i}", "[IN:X a ]", "reminder") for i in range(50)]
        + [_row(f"w{i}", "[IN:X a ]", "weather") for i in range(50)]
    )
    interleaved = _interleave_by_domain(convert_topv2_rows(rows))
    for cut in (2, 7, 40, 99, 100):
        prefix = interleaved[:cut]
        reminder = sum(1 for r in prefix if r["domain"] == "reminder")
        assert abs(reminder - (len(prefix) - reminder)) <= 1, (cut, reminder)


def test_interleaving_an_uneven_pool_keeps_the_leftovers():
    """5,767 reminder against 5,682 weather is uneven; no row may be dropped."""
    rows = (
        [_row(f"r{i}", "[IN:X a ]", "reminder") for i in range(5)]
        + [_row(f"w{i}", "[IN:X a ]", "weather") for i in range(2)]
    )
    assert len(_interleave_by_domain(convert_topv2_rows(rows))) == 7


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _eval_set(rows: list[dict]):
    converted = convert_topv2_rows(rows)
    return build_eval_set(converted, task="topv2", target=len(converted))


def _raw_for(eval_set, by_text: dict[str, str]) -> list[str]:
    """Model outputs in `eval_set.all` order, looked up by utterance.

    `topv2` uses `eval_sampling="shuffled"`, so a hand-written list positioned to match the INPUT
    rows silently misaligns with the eval set. Keying on the utterance makes these tests say what
    they mean regardless of the shuffle.
    """
    return [by_text[row["text"]] for row in eval_set.all]


def test_the_headline_is_the_mean_of_domains_not_the_pooled_rate():
    """An unbalanced draw must not move the score.

    Three reminder rows all wrong and one weather row right: pooled EM is 0.25, but the per-domain
    mean is (0.0 + 1.0) / 2 = 0.5. The mean is the honest number because the sampling ratio is an
    accident of the draw, not a property of the model.
    """
    rows = [
        _row("a", "[IN:A a ]", "reminder"),
        _row("b", "[IN:B b ]", "reminder"),
        _row("c", "[IN:C c ]", "reminder"),
        _row("d", "[IN:D d ]", "weather"),
    ]
    eval_set = _eval_set(rows)
    raw = _raw_for(eval_set, {
        "a": "[IN:WRONG a ]", "b": "[IN:WRONG b ]", "c": "[IN:WRONG c ]", "d": "[IN:D d ]",
    })
    result = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))

    assert result["f1"] == pytest.approx(0.5)
    assert result["per_class"]["em_reminder"] == pytest.approx(0.0)
    assert result["per_class"]["em_weather"] == pytest.approx(1.0)
    assert result["per_class"]["em_pooled"] == pytest.approx(0.25)


def test_a_tree_wrapped_in_prose_is_still_a_parse():
    """A model that prefaces its answer has a formatting habit, not a parsing failure.

    Discarding it would report a content error for something the prompt can fix, which is exactly
    the ambiguity `format_valid` exists to remove.
    """
    eval_set = _eval_set([_row("remind me", "[IN:CREATE_REMINDER remind me ]")])
    raw = ["Sure! Here is the parse:\n```\n[IN:CREATE_REMINDER remind me ]\n```\nHope that helps."]
    result = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))
    assert result["f1"] == pytest.approx(1.0)
    assert result["format_valid"] == pytest.approx(1.0)


def test_whitespace_differences_are_not_parsing_errors():
    eval_set = _eval_set([_row("remind me", "[IN:CREATE_REMINDER remind me ]")])
    raw = ["[IN:CREATE_REMINDER    remind   me   ]"]
    result = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))
    assert result["f1"] == pytest.approx(1.0)


def test_an_unbalanced_tree_is_a_format_failure_not_a_wrong_answer():
    """There is no parse to score, so it must not be reported as a content error."""
    eval_set = _eval_set([_row("remind me", "[IN:CREATE_REMINDER remind me ]")])
    result = scorer.score(eval_set, scorer.extract_predictions(["[IN:CREATE_REMINDER remind me"], eval_set))
    assert result["format_valid"] == pytest.approx(0.0)
    assert result["failures"][0]["error_type"] == "unparseable_output"


def test_the_failure_taxonomy_separates_intent_slot_and_span_errors():
    """Each category wants a different fix, which is the only reason to have a taxonomy.

    An intent error is a classification problem over a small label set; a span error is a copying
    problem. Reporting both as "wrong parse" is the B296 shape.
    """
    gold = "[IN:CREATE_REMINDER remind [SL:PERSON_REMINDED me ] ]"
    eval_set = _eval_set([_row("remind me", gold)] * 3)
    raw = [
        "[IN:CREATE_ALARM remind [SL:PERSON_REMINDED me ] ]",   # wrong intent
        "[IN:CREATE_REMINDER remind [SL:TODO me ] ]",           # wrong slot
        "[IN:CREATE_REMINDER remind [SL:PERSON_REMINDED you ] ]",  # wrong span text
    ]
    result = scorer.score(eval_set, scorer.extract_predictions(raw, eval_set))
    # A set, because the three rows are identical and the eval set is shuffled: which of them
    # receives which output is arbitrary, and only the categories produced are the claim here.
    assert {f["error_type"] for f in result["failures"]} == {
        "wrong_intent", "wrong_slot_set", "wrong_span_text",
    }


def test_an_empty_eval_set_scores_zero_without_dividing_by_zero():
    empty = build_eval_set([], task="topv2", target=0)
    result = scorer.score(empty, [])
    assert result["f1"] == 0.0
    assert result["format_valid"] == 0.0


# --------------------------------------------------------------------------
# The extractive verifier
# --------------------------------------------------------------------------


def test_the_verifier_accepts_a_tokenized_gold_parse_against_an_untokenized_utterance():
    """The comparison must ignore whitespace or every gold row with punctuation is rejected.

    Real gold: the utterance is "Remind Anita, Madi" and the parse's leaves are
    "Remind Anita , Madi". A word-by-word match rejects it, which is how the first version of this
    verifier failed 1,000 out of 1,000 real rows.
    """
    ok, reason = verify_semantic_parse_row({
        "text": "Delete reminders I had for all of this weekend.",
        "answer": (
            "[IN:DELETE_REMINDER Delete reminders [SL:PERSON_REMINDED I ] had "
            "[SL:DATE_TIME for all of this weekend ] . ]"
        ),
    })
    assert ok, reason


@pytest.mark.parametrize(
    "answer,expected",
    [
        # Paraphrase, omission and invention are the three ways a fluent generated parse lies
        # about its input, and one exact comparison catches all of them.
        ("[IN:CREATE_ALARM set [SL:DATE_TIME an alarm ] ]", "does not reproduce the command"),
        ("[IN:CREATE_ALARM wake me ]", "does not reproduce the command"),
        ("[IN:CREATE_ALARM wake me up at six tomorrow ]", "does not reproduce the command"),
        ("[IN:CREATE_ALARM wake me up at six", "brackets do not balance"),
        ("[IN:A wake me ] [IN:B up at six ]", "more than one tree"),
        ("[SL:DATE_TIME wake me up at six ]", "does not open on an intent"),
        # A real intent, so the malformed SLOT label is what this case actually exercises. With a
        # placeholder root the membership check fires first and the shape check is never reached.
        ("[IN:CREATE_ALARM wake me up at [Slot:time six ] ]", "not IN:NAME or SL:NAME"),
        # Shape and membership are different questions: `SL:WAKE_TIME` is a well-formed label and
        # not a TOPv2 slot, and exact-match scoring makes it unlearnable either way.
        ("[IN:CREATE_ALARM wake me up at [SL:WAKE_TIME six ] ]", "closed vocabulary"),
        ("", "empty or not a string"),
    ],
)
def test_the_verifier_rejects_each_way_a_generated_parse_can_be_wrong(answer, expected):
    ok, reason = verify_semantic_parse_row({"text": "wake me up at six", "answer": answer})
    assert not ok
    assert expected in reason


def test_the_verifier_needs_an_utterance_to_check_against():
    ok, reason = verify_semantic_parse_row({"text": "  ", "answer": "[IN:X a ]"})
    assert not ok
    assert "no utterance" in reason


# --------------------------------------------------------------------------
# The enumerated label vocabulary
# --------------------------------------------------------------------------


def test_the_prompt_enumerates_every_intent_and_slot_name():
    """THE BUG THIS EXISTS FOR, measured on run 39719567.

    Exact match compares the WHOLE parse string, label names included, and a model cannot guess
    `IN:GET_REMINDER_DATE_TIME` or `IN:UNSUPPORTED_WEATHER`. The target test split uses 53 distinct
    labels and a 5-shot demonstration block can show at most ~15, so a few-shot model had to invent
    ~38 names it was then scored against character by character. The local Qwen teacher measured
    exact_match=0.0030 — three correct out of a thousand — which measures vocabulary telepathy, not
    parsing.

    This repo already paid for the identical mistake on BC5CDR, where an unenumerated two-type
    vocabulary put the teacher at 0.1011 against a real 0.6140. `multiconer` enumerates its 33
    types for the same reason.

    A fine-tuned student learns the vocabulary from its training rows either way — what this fixes
    is the ZERO-SHOT BASELINE and the TEACHER measurement. Since the suite reports SFT deltas, a
    delta measured against a vocabulary-telepathy baseline is not a delta.
    """
    eval_set = _eval_set([_row("remind me", "[IN:CREATE_REMINDER remind me ]")])
    prompt = scorer.build_prompts(eval_set)[0]
    for name in INTENTS:
        assert name in prompt, f"intent {name} missing from the prompt"
    for name in SLOTS:
        assert name in prompt, f"slot {name} missing from the prompt"
    assert "remind me" in prompt


def test_the_prompt_shows_a_worked_example_not_a_copyable_placeholder():
    """The prompt used to show `Format: [IN:INTENT_NAME words [SL:SLOT_NAME words ] ]`, which is
    itself a syntactically valid answer — so the base model copied it verbatim and the extractor
    accepted it, because `INTENT_NAME` matches the label pattern and the brackets balance.

    Observed on run 39719567: asked to parse "Remind me to hook up the DVD player in the bedroom
    next week", the model's entire answer was `[IN:INTENT_NAME words [SL:SLOT_NAME words ] ]`.
    """
    eval_set = _eval_set([_row("remind me", "[IN:CREATE_REMINDER remind me ]")])
    prompt = scorer.build_prompts(eval_set)[0]

    assert "INTENT_NAME" not in prompt
    assert "SL:SLOT_NAME" not in prompt
    # A real worked example instead, which cannot be copied into a passing answer because its
    # spans reproduce a different command.
    assert "[IN:CREATE_ALARM Set alarm [SL:DATE_TIME_RECURRING for 6 am every day ] ]" in prompt


def test_a_label_outside_the_vocabulary_is_its_own_failure_category():
    """A label-space error wants a different fix from picking the wrong name off the right list,
    and it is also how a placeholder echo now presents."""
    eval_set = _eval_set([_row("remind me", "[IN:CREATE_REMINDER remind me ]")])
    result = scorer.score(
        eval_set,
        scorer.extract_predictions(["[IN:INTENT_NAME words [SL:SLOT_NAME words ] ]"], eval_set),
    )
    assert result["failures"][0]["error_type"] == "label_outside_vocabulary"


def test_the_pinned_vocabulary_still_matches_the_corpus():
    """The vocabulary is hardcoded so the prompt needs no data access, which means it can drift
    from the mirror. Skipped without the parquet cached, because this is the one test here that
    needs the real corpus."""
    try:
        from data.loaders.topv2 import (
            SOURCE_DOMAINS as src,
            TARGET_DOMAINS as tgt,
            _read_split,
            convert_topv2_rows,
        )
        rows = []
        for split in ("train", "eval", "test"):
            rows += convert_topv2_rows(_read_split(split), domain_filter=tuple(src) + tuple(tgt))
    except Exception as exc:  # noqa: BLE001 - no cached corpus in a bare environment
        pytest.skip(f"TOPv2 corpus unavailable: {type(exc).__name__}")

    used = set()
    for row in rows:
        used |= parse_labels(row["answer"])
    declared = {f"IN:{n}" for n in INTENTS} | {f"SL:{n}" for n in SLOTS}
    assert used - declared == set(), "the corpus uses labels the prompt does not enumerate"


def test_removing_test_contamination_costs_the_spis_split_no_label_coverage():
    """The eval-leakage fix must not quietly shrink the low-resource protocol's coverage.

    `load_topv2` drops adaptation rows that appear verbatim in the eval split — 4 of 678 as of
    2026-09-09, and all 4 were adaptation rows rather than source-domain ones, so the protocol's
    own scarce set is what gives way. That is the correct trade (training on eval rows is
    contamination, and the eval split is never the thing that yields), but SPIS's entire promise is
    per-label coverage, and losing the sole carrier of a rare label would break it.

    Measured: 61 labels covered before the removal and 61 after. That holds for a structural
    reason rather than by luck — a row colliding VERBATIM across splits is a short formulaic
    command like "will it rain today?", whose labels are the most abundant in the corpus, while a
    rare label lives on a long specific utterance that has no duplicate. This test pins the
    property, not the reason.

    `test_spis_sampling_gives_every_label_its_quota` cannot catch this: it runs `spis_sample` over
    synthetic parses and never sees the loaded, deduplicated output.
    """
    from collections import Counter

    import data.loaders.topv2 as loader
    from data.loaders.dataset_integrity import remove_normalized_train_overlap

    records = loader._read_split("train")
    pre_dedupe = []
    for domain in loader.TARGET_DOMAINS:
        rows = loader.convert_topv2_rows(records, domain_filter=(domain,))
        for index in loader.spis_sample([r["answer"] for r in rows], loader.resolve_spis(None)):
            pre_dedupe.append(rows[index])

    eval_rows = loader._interleave_by_domain(
        loader.convert_topv2_rows(loader._read_split("test"),
                                  domain_filter=loader.TARGET_DOMAINS)
    )[:1000]

    surviving, dropped = remove_normalized_train_overlap(list(pre_dedupe), eval_rows)

    def covered(rows):
        counts: Counter = Counter()
        for row in rows:
            for label in loader.parse_labels(str(row["answer"])):
                counts[label] += 1
        return counts

    before, after = covered(pre_dedupe), covered(surviving)
    lost = sorted(set(before) - set(after))
    assert not lost, (
        f"de-contaminating the adaptation split dropped the last row carrying {lost}; SPIS "
        f"guarantees per-label coverage, so this breaks the low-resource protocol"
    )
    # Removal must stay incidental to the protocol, not a material share of it.
    assert dropped < 0.02 * len(pre_dedupe), (
        f"{dropped} of {len(pre_dedupe)} adaptation rows collide with the eval split; that is too "
        f"much of the protocol to be losing to contamination"
    )
