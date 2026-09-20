"""TOPv2 — an assistant command in, a nested intent/slot parse tree out.

THE LOW-RESOURCE PROTOCOL IS THE TASK
    TOPv2 ships 124,597 training utterances across eight domains, and training on all of them is
    not the experiment. The published protocol trains on the SIX source domains and then adapts to
    one of two targets with a tiny sample: `reminder` or `weather` at 25 or 500 SPIS (samples per
    intent and slot). That is where the interesting numbers live — meta-learning matching a
    normally-trained model at 10x less data, RINE gaining +13.0 EM on reminder at 25 SPIS — and it
    is why this loader puts the adaptation rows FIRST, ahead of the source-domain rows. The
    caller's `max_train` cap then truncates source data, never the few hundred target rows the
    whole protocol is about.

    The two targets are deliberately unalike and both are reported. Measured on this mirror's
    train split: `reminder` has 18 intents, 30 slots and 21.5% of parses containing more than one
    intent; `weather` has 7 intents, 11 slots and 0.1%. So weather measures flat slot filling and
    will saturate, while reminder measures compositional structure and will not. An unlabelled
    average over the two would hide which one moved.

THE SPIS SPLITS ARE RECONSTRUCTED, NOT THE RELEASED FILES — READ THIS BEFORE CITING A NUMBER
    The official low-resource splits are files distributed with the gated TOPv2 release. They are
    NOT in the `WillHeld/top_v2` mirror, which carries only the three full splits, and no mirror of
    them exists on the Hub (checked 2026-09-06). So `spis_sample` below reimplements the sampling
    rule from its definition, deterministically and from a published seed.

    It agrees closely with the released sizes, which is the evidence that the rule and the data are
    both right:

        split                 reconstructed    published
        reminder 25 train              501          493
        reminder 25 valid              342          337
        weather  25 train              177          176
        weather  25 valid              150          147
        reminder 500 train           4,835        4,788
        weather  500 train           2,422        2,372

    The 25-SPIS numbers land within 1.6%. The 500-SPIS VALID splits do not (1,875 against 2,526 on
    reminder, 1,275 against 2,667 on weather): the validation pool is small enough that the
    per-label quota saturates against availability rather than against k, so whatever the official
    release did there is not what this rule does. The 500-SPIS valid split is therefore not used.

    Consequence, stated plainly: our EM is comparable BETWEEN OUR OWN RUNS and is not directly
    comparable to published RINE or shift-reduce numbers. That was already true for a second
    reason — EM depends on the parse serialization, and ours is the mirror's `semantic_parse`
    string verbatim — so the honest reading of this task is a controlled comparison against our
    own baseline, not a leaderboard entry.
"""
from __future__ import annotations

import os
import random
import re
from collections import Counter
from data.loaders.dataset_integrity import remove_normalized_train_overlap

TOPV2_ID = "WillHeld/top_v2"

# The mirror ships plain parquet with no configs, so the paths are pinned. `load_dataset` on a
# repo with no loader script and no configs is a coin flip about which files it globs; naming them
# means a new file appearing in the repo cannot silently change the split.
PARQUET_FILES = {
    "train": "data/train-00000-of-00001-4f5cf905029cbf9d.parquet",
    "eval": "data/eval-00000-of-00001-3ffa52405fac46ab.parquet",
    "test": "data/test-00000-of-00001-deac2888ce8ad39d.parquet",
}

# Train on these six, adapt to the two targets. This is the protocol, not a convenience split.
SOURCE_DOMAINS = ("alarm", "event", "messaging", "music", "navigation", "timer")
TARGET_DOMAINS = ("reminder", "weather")

# Published so a reader can regenerate the exact splits. Changing it changes the training set,
# which is why it is a module constant and not a parameter with a default.
SPIS_SEED = 20260906

# Which low-resource operating point to adapt at. 25 is the headline; 500 is the high-resource
# anchor and a SEPARATE RUN, not a second metric from the same one — the two differ in training
# set size, so no single trained model can report both.
SPIS_ENV = "SLM_TOPV2_SPIS"
DEFAULT_SPIS = 25
ALLOWED_SPIS = (25, 500)

# THE FULL LABEL VOCABULARY, enumerated in the prompt. Derived once from every split of all eight
# domains and pinned here; `tests/test_topv2.py` asserts it still matches the corpus.
#
# WHY THE PROMPT NAMES THEM, WHICH COSTS ~666 TOKENS ON A TEN-WORD INPUT
#     Exact match compares the WHOLE parse string, label names included, and a model cannot guess
#     `IN:GET_REMINDER_DATE_TIME` or `IN:UNSUPPORTED_WEATHER`. Measured on the target test split:
#     53 distinct labels appear, and a 5-shot demonstration block can show at most ~15 of them, so
#     a few-shot model must invent ~38 names it is then scored against character by character.
#
#     Run 39719567 is what that costs: the local Qwen teacher measured exact_match=0.0030 — three
#     correct out of a thousand — which is not a measurement of parsing ability. This repo already
#     paid for the identical mistake on BC5CDR, where an unenumerated two-type vocabulary put the
#     teacher at 0.1011 against a real 0.6140 because it returned the right spans under its own
#     label names. `multiconer` enumerates its 33 types for exactly this reason.
#
#     A FINE-TUNED student learns the vocabulary from its training rows either way, so this does
#     not change its ceiling. What it fixes is the ZERO-SHOT BASELINE and the TEACHER measurement
#     — and since this whole suite reports SFT deltas, a delta measured against a
#     vocabulary-telepathy baseline is not a delta.
#
#     All eight domains, not just the two scored ones: training includes source-domain rows, and a
#     prompt listing only the target labels would be training the model to disobey its own label
#     list on most of its data.
INTENTS = (
    "ADD_TIME_TIMER", "ADD_TO_PLAYLIST_MUSIC", "CANCEL_MESSAGE", "CREATE_ALARM",
    "CREATE_PLAYLIST_MUSIC", "CREATE_REMINDER", "CREATE_TIMER", "DELETE_ALARM",
    "DELETE_REMINDER", "DELETE_TIMER", "DISLIKE_MUSIC", "GET_ALARM",
    "GET_BIRTHDAY", "GET_CONTACT", "GET_DIRECTIONS", "GET_DISTANCE",
    "GET_ESTIMATED_ARRIVAL", "GET_ESTIMATED_DEPARTURE", "GET_ESTIMATED_DURATION", "GET_EVENT",
    "GET_EVENT_ATTENDEE", "GET_EVENT_ATTENDEE_AMOUNT", "GET_EVENT_ORGANIZER", "GET_INFO_CONTACT",
    "GET_INFO_ROAD_CONDITION", "GET_INFO_ROUTE", "GET_INFO_TRAFFIC", "GET_LOCATION",
    "GET_LOCATION_HOME", "GET_LOCATION_HOMETOWN", "GET_LOCATION_SCHOOL", "GET_LOCATION_WORK",
    "GET_MESSAGE", "GET_RECURRING_DATE_TIME", "GET_REMINDER", "GET_REMINDER_AMOUNT",
    "GET_REMINDER_DATE_TIME", "GET_REMINDER_LOCATION", "GET_SUNRISE", "GET_SUNSET",
    "GET_TIME", "GET_TIMER", "GET_TODO", "GET_WEATHER",
    "HELP_REMINDER", "IGNORE_MESSAGE", "LIKE_MUSIC", "LOOP_MUSIC",
    "NEGATION", "PAUSE_MUSIC", "PAUSE_TIMER", "PLAY_MUSIC",
    "PREVIOUS_TRACK_MUSIC", "REACT_MESSAGE", "REMOVE_FROM_PLAYLIST_MUSIC", "REPLAY_MUSIC",
    "REPLY_MESSAGE", "RESTART_TIMER", "RESUME_TIMER", "SELECT_ITEM",
    "SEND_MESSAGE", "SEND_TEXT_MESSAGE", "SET_DEFAULT_PROVIDER_MUSIC", "SILENCE_ALARM",
    "SKIP_TRACK_MUSIC", "SNOOZE_ALARM", "START_SHUFFLE_MUSIC", "STOP_MUSIC",
    "SUBTRACT_TIME_TIMER", "UNSUPPORTED_ALARM", "UNSUPPORTED_EVENT", "UNSUPPORTED_MESSAGING",
    "UNSUPPORTED_MUSIC", "UNSUPPORTED_NAVIGATION", "UNSUPPORTED_TIMER", "UNSUPPORTED_WEATHER",
    "UPDATE_ALARM", "UPDATE_DIRECTIONS", "UPDATE_REMINDER", "UPDATE_REMINDER_DATE_TIME",
    "UPDATE_REMINDER_TODO", "UPDATE_TIMER",
)
SLOTS = (
    "AGE", "ALARM_NAME", "AMOUNT", "ATTENDEE",
    "ATTENDEE_ADDED", "ATTENDEE_EVENT", "ATTENDEE_REMOVED", "ATTRIBUTE_EVENT",
    "BIRTHDAY", "CATEGORY_EVENT", "CATEGORY_LOCATION", "CONTACT",
    "CONTACT_RELATED", "CONTENT_EMOJI", "CONTENT_EXACT", "DATE_TIME",
    "DATE_TIME_ARRIVAL", "DATE_TIME_BIRTHDAY", "DATE_TIME_DEPARTURE", "DATE_TIME_NEW",
    "DATE_TIME_RECURRING", "DESTINATION", "DURATION", "FREQUENCY",
    "GROUP", "JOB", "LOCATION", "LOCATION_CURRENT",
    "LOCATION_HOME", "LOCATION_MODIFIER", "LOCATION_USER", "LOCATION_WORK",
    "MEASUREMENT_UNIT", "METHOD_RETRIEVAL_REMINDER", "METHOD_TIMER", "METHOD_TRAVEL",
    "MUSIC_ALBUM_TITLE", "MUSIC_ARTIST_NAME", "MUSIC_GENRE", "MUSIC_PLAYLIST_TITLE",
    "MUSIC_PROVIDER_NAME", "MUSIC_RADIO_ID", "MUSIC_TRACK_TITLE", "MUSIC_TYPE",
    "MUTUAL_EMPLOYER", "MUTUAL_LOCATION", "MUTUAL_SCHOOL", "NAME_APP",
    "NAME_EVENT", "OBSTRUCTION_AVOID", "ORDINAL", "ORGANIZER_EVENT",
    "PATH", "PATH_AVOID", "PERIOD", "PERSON_REMINDED",
    "PERSON_REMINDED_ADDED", "PERSON_REMINDED_REMOVED", "POINT_ON_MAP", "RECIPIENT",
    "RECURRING_DATE_TIME", "RECURRING_DATE_TIME_NEW", "RESOURCE", "ROAD_CONDITION",
    "ROAD_CONDITION_AVOID", "SEARCH_RADIUS", "SENDER", "SOURCE",
    "TAG_MESSAGE", "TIMER_NAME", "TIME_ZONE", "TODO",
    "TODO_NEW", "TYPE_CONTACT", "TYPE_CONTENT", "TYPE_INFO",
    "TYPE_REACTION", "TYPE_RELATION", "UNIT_DISTANCE", "WAYPOINT",
    "WAYPOINT_ADDED", "WAYPOINT_AVOID", "WEATHER_ATTRIBUTE", "WEATHER_TEMPERATURE_UNIT",
)

_LABEL_RE = re.compile(r"\[(IN:[A-Z0-9_]+|SL:[A-Z0-9_]+)")


def parse_labels(semantic_parse: str) -> set[str]:
    """The intent and slot labels a parse uses, e.g. `{"IN:CREATE_REMINDER", "SL:TODO"}`.

    The unit SPIS counts. Deliberately a SET rather than a multiset: SPIS is "samples per intent
    and slot", so one utterance mentioning `SL:DATE_TIME` twice is one sample of it, not two.
    """
    return set(_LABEL_RE.findall(str(semantic_parse or "")))


def spis_sample(parses: list[str], k: int, seed: int = SPIS_SEED) -> list[int]:
    """Indices of a subset in which every intent and slot appears at least `k` times.

    Greedy over a seeded shuffle: walk the rows in random order and keep one whenever it still
    contributes to an unmet quota. This is the point of SPIS rather than "take N at random" — a
    random 500 rows of `reminder` would contain zero examples of labels that occur a handful of
    times in 17,840, and the model would then be scored on a test split that does contain them.

    The quota is `min(k, available)`, because some labels genuinely cannot reach k: `SL:JOB` and
    `IN:GET_BIRTHDAY` each occur exactly once in reminder's train split. Without the clamp the
    loop would consume every row hunting for a 25th occurrence that does not exist, silently
    turning the low-resource split into the full one.

    Returns ASCENDING indices, not shuffle order, so the caller's row order is the corpus's own
    and two runs with the same seed produce byte-identical files.
    """
    if k < 1:
        raise ValueError(f"SPIS k must be positive, got {k}")
    label_sets = [parse_labels(parse) for parse in parses]
    available: Counter = Counter()
    for labels in label_sets:
        available.update(labels)
    quota = {label: min(k, count) for label, count in available.items()}

    order = list(range(len(parses)))
    random.Random(seed).shuffle(order)
    counts: Counter = Counter()
    chosen: list[int] = []
    for index in order:
        labels = label_sets[index]
        if any(counts[label] < quota[label] for label in labels):
            chosen.append(index)
            counts.update(labels)
    return sorted(chosen)


def resolve_spis(raw: object = None) -> int:
    """The SPIS operating point, from the argument or `SLM_TOPV2_SPIS`.

    Restricted to the two published points rather than accepting any integer. An arbitrary k would
    produce a split with no published counterpart and no baseline to compare against, which is a
    run that cannot be written up.
    """
    value = raw if raw is not None else os.environ.get(SPIS_ENV) or DEFAULT_SPIS
    try:
        spis = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{SPIS_ENV} must be one of {ALLOWED_SPIS}, got {value!r}") from exc
    if spis not in ALLOWED_SPIS:
        raise ValueError(f"{SPIS_ENV} must be one of {ALLOWED_SPIS}, got {spis}")
    return spis


INSTRUCTION = (
    "Parse the command into a nested intent and slot tree."
)


def convert_topv2_rows(dataset, domain_filter=None) -> list[dict]:
    """Mirror rows (`domain`, `utterance`, `semantic_parse`) into task rows.

    `answer` carries the parse string VERBATIM. Re-serializing it — normalizing bracket spacing,
    sorting slots, anything — would change what exact match means and make our numbers
    incomparable even with our own earlier runs. `domain` is kept because the scorer reports the
    two target domains separately.
    """
    wanted = set(domain_filter) if domain_filter else None
    out: list[dict] = []
    for example in dataset:
        domain = str(example.get("domain") or "").strip()
        if wanted is not None and domain not in wanted:
            continue
        utterance = str(example.get("utterance") or "").strip()
        parse = str(example.get("semantic_parse") or "").strip()
        if not utterance or not parse:
            continue
        out.append({
            "text": utterance,
            "answer": parse,
            "domain": domain,
            "_instruction": INSTRUCTION,
        })
    return out


def _interleave_by_domain(rows: list[dict]) -> list[dict]:
    """Round-robin the rows across domains so ANY prefix is domain-balanced.

    THE BUG THIS EXISTS FOR. The mirror's test parquet is ordered by domain, so `rows[:1000]`
    returned 1,000 reminder rows and zero weather rows — and the caller's cap is applied before
    `build_eval_set` ever gets to shuffle. The headline is the mean of the two domains' exact
    match, so the in-loop eval was silently measuring one domain and calling it the average, on a
    task whose entire point is the contrast between them.

    Round-robin rather than a seeded shuffle because it is balanced at every truncation point, not
    just in expectation: a 47-row eval gets 24 and 23, and no seed can make that go wrong.
    """
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row.get("domain", "unknown"), []).append(row)
    out: list[dict] = []
    for index in range(max((len(bucket) for bucket in buckets.values()), default=0)):
        for domain in sorted(buckets):
            if index < len(buckets[domain]):
                out.append(buckets[domain][index])
    return out


def _read_split(split: str):
    """One split of the mirror as a list of dicts, from the pinned parquet path."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TOPV2_ID, PARQUET_FILES[split], repo_type="dataset")
    import pandas as pd

    return pd.read_parquet(path).to_dict("records")


def load_topv2(
    max_train: int = 5000,
    max_test: int = 1000,
    log=print,
    spis: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return `(train, eval)` for the low-resource adaptation protocol.

    Train is the SPIS adaptation sample for both target domains FIRST, then source-domain rows to
    fill the remaining budget. The ordering is the mechanism: `initial_train_cap` is 5,000 and the
    six source domains hold 83,703 rows, so a naive concatenation would let the cap discard the
    669 target rows the protocol is built on.

    Eval is the two target test splits pooled — 5,767 reminder plus 5,682 weather — with `domain`
    on every row so the scorer can report them separately and average.
    """
    spis_k = resolve_spis(spis)

    train_records = _read_split("train")
    adaptation: list[dict] = []
    for domain in TARGET_DOMAINS:
        rows = convert_topv2_rows(train_records, domain_filter=(domain,))
        chosen = spis_sample([row["answer"] for row in rows], spis_k)
        picked = [rows[i] for i in chosen]
        for row in picked:
            row["_provenance"] = f"topv2_{domain}_{spis_k}spis"
        adaptation.extend(picked)
        log(f"      [topv2] {domain} {spis_k}-SPIS adaptation: {len(picked)} of {len(rows)} rows "
            f"(seed {SPIS_SEED}; reconstructed, not the released file)")

    source = convert_topv2_rows(train_records, domain_filter=SOURCE_DOMAINS)
    room = max(0, int(max_train) - len(adaptation))
    train = adaptation + source[:room]
    log(f"      [topv2] train: {len(adaptation)} adaptation + {min(room, len(source))} source-domain "
        f"row(s) from {sorted(SOURCE_DOMAINS)} = {len(train)}")
    if room == 0 and len(adaptation) > int(max_train):
        # Never silently drop adaptation rows: the whole protocol is those few hundred rows.
        log(f"      [topv2] WARNING: max_train={max_train} is below the {len(adaptation)} "
            f"adaptation rows; keeping them all and taking no source data")

    test = _interleave_by_domain(
        convert_topv2_rows(_read_split("test"), domain_filter=TARGET_DOMAINS)
    )
    test = test[:max_test]
    by_domain = Counter(row["domain"] for row in test)
    log(f"      [topv2] eval: {len(test)} target-domain test row(s) {dict(by_domain)}")

    # TOPv2's train and test splits share a few commands verbatim — short formulaic ones like
    # "set a reminder". `curate`'s eval firewall would strip them before training anyway, so the
    # run was never contaminated, but the reported curriculum size would then silently shrink and
    # this loader would be shipping known leakage. `multiconer` and `gec_bea19` already dedupe at
    # load; doing it here removes the last place in this suite that relies on a downstream net.
    # TEST IS KEPT INTACT: the eval split is never the thing that gives way.
    train, leaked = remove_normalized_train_overlap(train, test)
    if leaked:
        log(f"      [topv2] dropped {leaked} train row(s) whose command also appears in test")
    return train, test
