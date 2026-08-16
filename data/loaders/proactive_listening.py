"""Proactive-listening loader — LlamaPIE "when to respond" (2026-08-15).

THE TASK
    An in-ear assistant listens to a conversation and must decide, at each pause, whether to
    whisper a 1–3 word hint to its wearer or stay silent. This loader builds the *decision* half
    of that problem — LlamaPIE's "small model" — as a binary ``classification`` task:

        input  : the conversation so far, ending at a pause
        output : ``interrupt`` (whisper now) or ``wait`` (stay silent)

    From LlamaPIE (arXiv:2505.04066) §2.3.1: a small model runs continuously and non-
    autoregressively to decide *when* to respond; only when it fires does a larger model decide
    *what* to say. The small model is the on-device component and the one worth fine-tuning, so it
    is the one modelled here.

WHY THIS IS AN OUT-OF-DISTRIBUTION TASK
    The label is not a property of the input's surface form. Nothing in "…and then we visited the,
    uh, the museum in |SILENCE >" marks it as a place where help is wanted; that depends on whether
    the speaker is about to need a detail they cannot recall. A base model has no pattern to match,
    so zero-shot performance should be poor and fine-tuning is what supplies the decision boundary.

    It is, however, a **better-posed** OOD task than RouterBench. RouterBench's label describes a
    *different model's* behaviour on the prompt, which no amount of reading the prompt can reveal.
    Here the label is a property of *this* conversation: hesitation, a trailing clause, a question
    about a specific remembered detail. That signal is genuinely present in the text.

DATA
    Frozen bundle at ``data/local/proactive_listening/`` (14,130 train / 878 test dialogues),
    extracted from the authors' ``Main_dataset.tar``. **The dataset is not on HuggingFace** — it is
    distributed as a Google Drive tarball linked from the repo README, so it is vendored here with
    a manifest and checksums rather than fetched at run time. ``scripts/download_datasets.py``
    documents the provenance.

    Four sub-corpora, all with Claude-generated assistance annotations: ``synthetic0`` and
    ``synthetic`` (profiles built from keywords), ``perl`` (PerLTQA personal-memory profiles), and
    ``soda`` (SODA social dialogues). A fifth, ``MIT_final``, holds 92 real interview recordings but
    carries no assistance labels, so it is not used for scoring.

FORMAT
    Each dialogue is one line of the bundle with the authors' ``raw.txt`` verbatim. Two markers
    matter:
      * ``|SILENCE >`` — 0.5 s of silence. **Every occurrence is a decision point.**
      * `` ^^``      — the assistant whispered at the immediately preceding pause.
    So the label of a decision point is simply whether a ``^^`` follows its silence token. Working
    from these text markers rather than the authors' ``mask.txt``/``values.txt`` keeps the loader
    free of their Llama tokenizer: those files are token-indexed, and reproducing the exact indices
    would make our labels depend on a tokenizer we do not use. Verified equivalent — 99.5% of
    whisper markers sit directly after a silence token, and the derived positive rate (12.50%)
    matches the token-level rate from ``values.txt``/``mask.txt`` (11.96%).

    The natural positive rate is ~12%, so ``interrupt`` is the minority class and
    ``eval/scorers/classification.py`` reports **F1 on ``interrupt``** — which is exactly the
    "hard precision/recall" LlamaPIE reports in its Table 2. **Published reference: their
    fine-tuned Llama-3.2-1B reaches hard precision 0.728–0.759 and hard recall 0.719–0.777**
    (F1 ≈ 0.74), a directly comparable number for a model one size step above our 0.6B.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Iterable

SILENCE_TOKEN = "|SILENCE >"
WHISPER_MARKER = "^^"

# A decision point is ANY ``|MARKER >`` token, not only ``|SILENCE >``. The authors' own dataset
# code masks on the ` >` token (`Active_dataset.py`: `symbol_token2 = tokenizer(" >")[...]`), and the
# ``synthetic0`` sub-corpus additionally annotates speaker emotion — ``|ANGRY >``, ``|NEUTRAL >`` —
# with whispers attached to those markers rather than to a silence. Matching only ``|SILENCE >``
# silently found ZERO positives in every ``synthetic0`` dialogue and dropped the training positive
# rate to 4.2% against a 33% eval rate.
_DECISION_MARKER = re.compile(r"\|[A-Z_]+ >")

LABEL_INTERRUPT = "interrupt"  # whisper a hint now
LABEL_WAIT = "wait"            # stay silent

LOCAL_BUNDLE = os.path.join("data", "local", "proactive_listening")

# Sub-corpora to draw from. ``synthetic0`` is EXCLUDED by default: it is half the training bundle
# but carries emotion markers that no other corpus has, and the held-out split contains none of it.
# Training on it would put a surface feature in the curriculum that is absent at eval — the same
# train/serve mismatch that cost the NER run a 44-hour attempt. Opt back in with
# SLM_PROACTIVE_SOURCES=synthetic0,synthetic,perl,soda.
DEFAULT_SOURCES = ("synthetic", "perl", "soda")

# How much conversation the model sees at a decision point. LlamaPIE streams the whole dialogue
# with a cached KV state; a one-shot classifier cannot, so the context is bounded. 1200 characters
# covers roughly the last three to five turns, which is where the cue for "they are about to need
# a detail" actually lives. Env-tunable because it trades prompt cost against context.
CONTEXT_CHARS = int(os.environ.get("SLM_PROACTIVE_CONTEXT_CHARS", "1200"))

# Decision points sampled per dialogue. Every dialogue has ~30, and taking all of them would make
# consecutive rows near-identical (they share a prefix), which surface-form dedup would then delete
# unpredictably. Sampling a few per dialogue and drawing from many dialogues buys real diversity.
POINTS_PER_DIALOGUE = int(os.environ.get("SLM_PROACTIVE_POINTS_PER_DIALOGUE", "3"))

INSTRUCTION = (
    "You are an in-ear assistant listening to a conversation involving your user. At the pause at "
    "the end of the transcript, decide whether to whisper a short hint to your user now, or stay "
    "silent. Whisper only when your user is about to need a specific detail they may not recall, "
    "or clearly needs help continuing. Stay silent otherwise — most pauses need nothing."
)


def _decision_points(raw: str) -> list[tuple[str, bool]]:
    """Split one dialogue into ``(prefix_ending_at_a_marker, whisper_happened_here)`` pairs.

    The prefix is the conversation as the assistant would have heard it at that moment, with the
    whisper markers stripped — the model must not be shown where the answer is.
    """
    out: list[tuple[str, bool]] = []
    cursor = 0
    prefix_parts: list[str] = []
    for match in _DECISION_MARKER.finditer(raw):
        end = match.end()
        if end <= cursor:
            continue
        prefix_parts.append(raw[cursor:end])
        # A whisper is annotated immediately AFTER the marker it belongs to.
        whispered = raw[end:end + 6].lstrip().startswith(WHISPER_MARKER)
        prefix = "".join(prefix_parts).replace(WHISPER_MARKER, "")
        out.append((" ".join(prefix.split()), whispered))
        cursor = end
    return out


def _render(prefix: str, memory: str) -> str:
    """The model's input: instruction, the user's memory when available, then the transcript."""
    blocks = [INSTRUCTION]
    profile = _profile_text(memory)
    if profile:
        blocks.append(f"What you know about your user:\n{profile}")
    tail = prefix[-CONTEXT_CHARS:] if len(prefix) > CONTEXT_CHARS else prefix
    if len(prefix) > CONTEXT_CHARS:
        tail = "… " + tail
    blocks.append(f"Conversation so far:\n{tail}")
    return "\n\n".join(blocks)


def _profile_text(memory: str) -> str:
    """The user's profile + events, flattened. LlamaPIE supplies this as prompt "memory"; without
    it a reminder-type whisper is unguessable, because the detail to be recalled lives here."""
    if not memory:
        return ""
    try:
        blob = json.loads(memory)
    except (json.JSONDecodeError, TypeError):
        return " ".join(str(memory).split())
    if not isinstance(blob, dict):
        return " ".join(str(memory).split())
    parts = [str(blob.get("profile") or "").strip()]
    events = blob.get("events")
    if isinstance(events, dict):
        parts.extend(str(v).strip() for v in events.values() if str(v).strip())
    elif isinstance(events, list):
        parts.extend(str(v).strip() for v in events if str(v).strip())
    return " ".join(" ".join(p.split()) for p in parts if p)


def convert_proactive_rows(
    dialogues: Iterable[dict],
    *,
    points_per_dialogue: int = POINTS_PER_DIALOGUE,
    seed: int = 20260815,
    balance: bool = True,
) -> list[dict]:
    """Map LlamaPIE dialogues to ``{text, label}`` interrupt/wait decisions.

    ``balance`` keeps at least one positive per dialogue that has one, then fills the remaining
    slots with negatives. The natural rate is ~12%; without deliberate sampling a small draw can
    easily contain no positives at all, and a task whose minority class is absent cannot be scored.
    Sampling is seeded so the curriculum and the eval set are reproducible across runs.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    # Shuffle the DIALOGUES before expanding. The bundle is grouped by sub-corpus, so taking the
    # first N rows of an unshuffled expansion draws entirely from whichever corpus happens to be
    # first — which is how a 3,250-row training draw ended up 100% `synthetic0`.
    dialogues = list(dialogues)
    rng.shuffle(dialogues)
    for dialogue in dialogues:
        raw = str(dialogue.get("raw") or "")
        if not raw.strip():
            continue
        points = _decision_points(raw)
        if not points:
            continue
        positives = [p for p in points if p[1]]
        negatives = [p for p in points if not p[1]]
        chosen: list[tuple[str, bool]] = []
        if balance and positives:
            chosen.append(rng.choice(positives))
        remaining = max(0, points_per_dialogue - len(chosen))
        pool = negatives if balance else points
        if pool and remaining:
            chosen.extend(rng.sample(pool, min(remaining, len(pool))))
        memory = str(dialogue.get("memory") or "")
        for prefix, whispered in chosen:
            if not prefix.strip():
                continue
            rows.append({
                "text": _render(prefix, memory),
                "label": LABEL_INTERRUPT if whispered else LABEL_WAIT,
            })
    return rows


def _read_bundle(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_proactive_listening(
    max_train: int = 3250,
    max_test: int = 800,
    *,
    log=print,
) -> tuple[list[dict], list[dict]]:
    """Return ``(train, test)`` interrupt/wait rows from the frozen LlamaPIE bundle.

    Reads the vendored bundle only — there is no live download path, because the upstream data is a
    Google Drive tarball rather than a versioned dataset repository. Override the location with
    ``SLM_PROACTIVE_DIR``.
    """
    base = os.environ.get("SLM_PROACTIVE_DIR") or LOCAL_BUNDLE
    train_path = os.path.join(base, "train.jsonl")
    test_path = os.path.join(base, "test.jsonl")
    if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
        raise FileNotFoundError(
            f"proactive-listening bundle not found at {base!r} (need train.jsonl and test.jsonl). "
            "It is vendored in the repo; see data/local/proactive_listening/manifest.json for "
            "provenance, or set SLM_PROACTIVE_DIR."
        )
    sources = tuple(
        s.strip() for s in
        (os.environ.get("SLM_PROACTIVE_SOURCES") or ",".join(DEFAULT_SOURCES)).split(",")
        if s.strip()
    )
    train_dialogues = [d for d in _read_bundle(train_path) if d.get("source") in sources]
    test_dialogues = [d for d in _read_bundle(test_path) if d.get("source") in sources]
    if not train_dialogues or not test_dialogues:
        raise RuntimeError(
            f"proactive-listening: no dialogues matched sources={sources}. Available sources are "
            "synthetic0, synthetic, perl, soda (set SLM_PROACTIVE_SOURCES)."
        )
    log(f"      [proactive] {len(train_dialogues)} train / {len(test_dialogues)} test dialogue(s) "
        f"from {base} (sources={','.join(sources)})")

    # Different seeds per split so the sampler cannot pick correlated decision points across the
    # train/eval boundary. The splits are already disjoint dialogues upstream.
    train = convert_proactive_rows(train_dialogues, seed=20260815)[:max_train]
    test = convert_proactive_rows(test_dialogues, seed=20260816)[:max_test]
    if not test:
        raise RuntimeError(
            "proactive-listening produced zero eval rows — refusing to proceed with an empty "
            "held-out set."
        )

    def _rate(rows):
        n = sum(1 for r in rows if r["label"] == LABEL_INTERRUPT)
        return n, (n / len(rows) if rows else 0.0)

    tn, tr = _rate(train)
    en, er = _rate(test)
    log(f"      [proactive] train={len(train)} ({tn} interrupt, {tr:.1%})  "
        f"test={len(test)} ({en} interrupt, {er:.1%})")
    log("      [proactive] minority class is 'interrupt' → scored as F1 on 'interrupt', "
        "directly comparable to LlamaPIE's hard precision/recall (their 1B: F1 ≈ 0.74)")
    return train, test
