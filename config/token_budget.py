"""One global output ceiling for every teacher call, replacing per-call-site constants.

WHY THIS EXISTS
    Nine call sites each carried their own hand-picked output limit: 512 for a synthesized row, 512
    for a chain of thought, 200 for a new utterance, 160 for one verification verdict, 120 for
    another, 200 as the client default. None was derived from anything, and nothing checked any of
    them against the data it had to carry.

    The 512 was the expensive one. A toolbench row serialises to 3,913 characters at the very
    smallest, so ZERO of 4,995 rows could fit; every generation was cut off mid-JSON and dropped by a
    bare `except`. Three runs (38832586, 38985393, 39041380) reported "0 rows kept" from 1,019 / 133 /
    425 attempts and all three were diagnosed as the VERIFIER rejecting rows.

    The others fail more quietly and in a worse direction. The two verification verdicts are
    `{valid, reason}` with a free-text reason; when the reason overruns, the JSON does not parse, and
    the fail-open contract KEEPS the row. So overflow does not make the verifier strict, it makes it
    incapable of rejecting anything.

THE RULE
    Ask for as much as the endpoint can physically return, and never less because someone guessed. The
    only real bound is the server's own: a request for more than `max_model_len - prompt` is an HTTP
    400, which is not truncation but an outright failure. That bound is arithmetic, so it is computed
    here rather than approximated by a constant at each call site.

    Over-asking costs nothing. Generation stops at the EOS token, so a budget larger than the reply
    changes neither the output nor the latency — it only removes a ceiling that could have been hit.

WHAT IS DELIBERATELY NOT HERE
    `TaskSpec.max_new_tokens`, the STUDENT's eval reserve. That number cannot be globally raised,
    because the reserve and the prompt share one window: the input budget is
    `max_seq_length - max_new_tokens`, and `eval_output_token_reserve` raises when a reserve leaves no
    room for a prompt. Raising it does not uncap the model, it starts REFUSING rows at load time —
    and it changes every task's measured score. That is a per-task measurement decision, not a
    plumbing one, so it stays in the registry where it is visible.
"""
from __future__ import annotations

import os

# The global ceiling. Generous rather than tuned: it exists to stop a runaway generation from
# consuming a whole context window, not to shape any particular reply. Nothing in the suite has a
# legitimate reply longer than this, so hitting it means something has gone wrong.
MAX_OUTPUT_TOKENS = int(os.environ.get("SLM_MAX_OUTPUT_TOKENS", "16384"))

# Characters per token for dense JSON and English prose, used to size what a reply must CONTAIN.
# Over-asking for output is free (generation stops at EOS), so 4.0 is fine here.
CHARS_PER_TOKEN = 4.0

# Characters per token used to estimate what the PROMPT has already spent. Deliberately SMALLER than
# CHARS_PER_TOKEN, which makes the estimate larger and the resulting output budget smaller — the
# safe direction, because this number is subtracted from a hard server limit.
#
# WHY IT IS NOT 4.0 — this is the "over by exactly one token" bug
#     xlam run 39311800 died on 400s that read:
#         requested 4736 output tokens, prompt contains at least 3457, total at least 8193 (max 8192)
#     4.0 estimated that prompt at 3392 against a real 3457: 65 tokens short, against a 64-token
#     margin. Raising the served context to 16384 did NOT fix it, because the shortfall is
#     PROPORTIONAL to prompt length, not constant — run 39321471 then produced
#         requested 13515, prompt at least 2870, total at least 16385 (max 16384)
#     the identical 65-token gap at double the context. A fixed margin cannot absorb an error that
#     grows with the prompt; the estimate itself has to be conservative.
#
#     Dense tool-schema JSON is punctuation-heavy and tokenizes near 3.9 chars/token, so 4.0 is
#     optimistic by ~2%. 3.5 leaves ~14% headroom, which covers that with room to spare and costs
#     only a slightly smaller output budget on very long prompts.
#
# WHY IT IS NO LONGER 3.5 — the SAME bug, on a corpus that tokenizes far worse (2026-09-09)
#     `gec_bea19` synthesis failed 97 of 112 generation attempts with, verbatim:
#         requested 15796 output tokens, prompt contains at least 589 input tokens,
#         for a total of at least 16385 tokens (max 16384)
#     Over by exactly one token again, from a 65-token shortfall against the 64-token margin —
#     numerically identical to the xlam case above, which is what makes it worth spelling out:
#     lowering 4.0 to 3.5 did not fix the class of bug, it moved the threshold.
#
#     W&I+LOCNESS is distributed WORD-TOKENIZED. Every punctuation mark is its own whitespace-
#     separated token and contractions are split (`do n't`, `It 's`), so a two-character " ." costs
#     a whole token. Measured over real 5-shot generation prompts:
#
#         task         chars  real tok  chars/token   est@3.5   short by
#         gec_bea19     2302       766         3.01       657       +109
#         multiconer    1445       406         3.56       412         -6
#         dialogsum     4298      1171         3.67      1228        -57
#         topv2         1600       420         3.81       457        -37
#         goemotions    1289       337         3.82       368        -31
#
#     Only GEC falls below 3.5, and it does so by enough to blow through a fixed margin — 109
#     tokens against 64. The other four are over-estimated, which is the safe direction.
#
#     2.5, not 3.0. Three would sit exactly at the worst observed ratio, and a margin of 1.00x is
#     what this comment has now been written twice about: an estimate that is merely adequate for
#     today's prompts fails the next corpus. 2.5 over-estimates GEC's prompt by ~20%.
#
#     IT COSTS ALMOST NOTHING. On GEC's 2,302-character prompt the estimate rises from 657 to 921
#     tokens, so the output budget falls by 264 out of roughly 15,700 — and real replies are ~100
#     tokens, because generation stops at EOS. The budget was never the binding constraint on a
#     reply's length; it only has to not exceed what the server will accept.
_PROMPT_CHARS_PER_TOKEN = 2.5

# Reserved for the chat template, role markers and the server's own accounting. This covers the
# FIXED overhead only; proportional tokenizer drift is handled by _PROMPT_CHARS_PER_TOKEN above.
_CONTEXT_MARGIN_TOKENS = 64

# Never return less than this, even when the prompt has nearly filled the context. A budget below
# this is not a working request, and returning it would trade a loud 400 for a silent truncation.
_FLOOR_TOKENS = 256


# The two defaults `config.config` uses for the teacher's context, mirrored here. LOCAL is what
# `_l40s_task_body.sh` launches vLLM with; API is what DeepSeek serves.
_LOCAL_DEFAULT_CONTEXT = 8192
_API_DEFAULT_CONTEXT = 131072


def served_context() -> int:
    """The context the teacher is actually served at — local vLLM or the hosted API.

    Read from the environment rather than `from config.config import SYNTH_MAX_MODEL_LEN`, which
    imports a module that raises KeyError on any unset API key. That import, wrapped in a bare
    `except: pass`, is how the first version of this clamp came to be silently inactive whenever an
    unrelated credential was missing.

    THE DEFAULT IS PER-MODE, because `config.config` has two and matching only one of them is how
    the two came to disagree. This returned a flat 8192 while API mode's config default is 131072,
    and the xlam launcher had just stopped exporting the variable in API mode — deliberately, so its
    local 16384 cap could not throttle DeepSeek. The result was a budget computed against an 8,192
    window the teacher did not have: on run 39361648, `output_budget` allowed roughly 4,600 output
    tokens where 16,384 were available, and `deepseek-v4-flash` spends most of a budget reasoning
    before it emits content, so replies were cut off a few hundred characters in — 19 of one
    350-attempt batch, reported as `unparseable JSON from the generator`.

    An explicit `SLM_SYNTH_MAX_MODEL_LEN` still wins in both modes; it is the operator override that
    the local profiles use to trade context against KV cache.
    """
    explicit = os.environ.get("SLM_SYNTH_MAX_MODEL_LEN")
    if explicit:
        return int(explicit)
    if os.environ.get("SLM_SYNTH_API_MODE", "0") == "1":
        return _API_DEFAULT_CONTEXT
    return _LOCAL_DEFAULT_CONTEXT


def output_budget(prompt: str = "", *, needed_chars: int = 0) -> int:
    """As many output tokens as the endpoint can return for this prompt.

    `needed_chars` is an optional measurement of what the reply must contain — the serialized length
    of a row the teacher is being asked to reproduce, say. It raises the floor when it is known,
    which is what lets a caller assert "this reply cannot be shorter than X" without inventing a
    constant. It never lowers the budget.
    """
    budget = MAX_OUTPUT_TOKENS
    if needed_chars > 0:
        # 1.5x, because a genuinely new instance is allowed to be somewhat longer than the row it
        # was modelled on.
        budget = max(budget, int(needed_chars / CHARS_PER_TOKEN * 1.5) + 128)
    room = served_context() - int(len(prompt) / _PROMPT_CHARS_PER_TOKEN) - _CONTEXT_MARGIN_TOKENS
    return max(_FLOOR_TOKENS, min(budget, room))


def prompt_char_budget(fraction: float = 0.25) -> int:
    """How many characters of the served context one part of a prompt may spend.

    For the places that clip an INPUT rather than an output: the rows shown to the task-brief author,
    the reference examples shown to the verifier. Those carried constants too — 1,200 and 400
    characters — which on a task whose rows are ~10,000 characters meant the brief author saw about
    an eighth of one example and wrote the task description every later prompt depends on from that
    fragment.

    A fraction of the real context rather than a constant, so a task with large rows automatically
    gets a share proportional to what the server can hold.
    """
    return max(400, int(served_context() * CHARS_PER_TOKEN * fraction))
