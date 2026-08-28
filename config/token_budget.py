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

# Characters per token for dense JSON and English prose. Conservative on purpose: this figure is used
# to estimate how much of the context a PROMPT has already spent, and over-estimating the prompt
# leaves a smaller output budget, which is the safe direction to be wrong in.
CHARS_PER_TOKEN = 4.0

# Reserved for the chat template, role markers and the server's own accounting.
_CONTEXT_MARGIN_TOKENS = 64

# Never return less than this, even when the prompt has nearly filled the context. A budget below
# this is not a working request, and returning it would trade a loud 400 for a silent truncation.
_FLOOR_TOKENS = 256


def served_context() -> int:
    """The context the teacher's vLLM server was actually launched with.

    Read from the environment rather than `from config.config import SYNTH_MAX_MODEL_LEN`, which
    imports a module that raises KeyError on any unset API key. That import, wrapped in a bare
    `except: pass`, is how the first version of this clamp came to be silently inactive whenever an
    unrelated credential was missing. The default matches `config.py` so the two cannot disagree.
    """
    return int(os.environ.get("SLM_SYNTH_MAX_MODEL_LEN", "8192"))


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
    room = served_context() - int(len(prompt) / CHARS_PER_TOKEN) - _CONTEXT_MARGIN_TOKENS
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
