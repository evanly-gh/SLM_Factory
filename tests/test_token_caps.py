"""Every token cap in the pipeline must be triaged, and the risky ones must stay named.

WHY THIS FILE EXISTS
    `data/curriculum.py` generated synthetic rows with `max_tokens=512`, hardcoded, for every task in
    the registry. A toolbench row serialises to 3,913 characters at the smallest, so ZERO of 4,995
    rows could fit; every generation was cut off mid-JSON and dropped by a bare
    `except Exception: return None`. Three runs reported "0 rows kept" from 1,019 / 133 / 425
    attempts and all three were diagnosed as the VERIFIER rejecting rows, because the message that
    reported it computes attempted-minus-kept and blames verification.

    The constant was not hidden. Nothing was watching it, and nothing connected it to the size of the
    data it had to carry. That is what these tests do.
"""
from __future__ import annotations

import importlib.util
import os

SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "audit_token_caps.py"
)


def _audit():
    """Load the audit script as a module.

    Registered in `sys.modules` BEFORE executing it: `@dataclass` resolves its annotations through
    `sys.modules[cls.__module__]`, which is None for a module loaded from a path and never inserted,
    and the decorator dies with `'NoneType' object has no attribute '__dict__'`.
    """
    import sys

    if "audit_token_caps" in sys.modules:
        return sys.modules["audit_token_caps"]
    spec = importlib.util.spec_from_file_location("audit_token_caps", SPEC_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_token_caps"] = module
    spec.loader.exec_module(module)
    return module


def test_no_token_cap_is_untriaged():
    """A new cap must be added to the inventory WITH what happens when the data exceeds it.

    Deliberately not "no new caps": caps are necessary. What is not acceptable is a cap whose
    overflow behaviour nobody has written down, because that is exactly the state the 512 was in.
    """
    import fnmatch

    audit = _audit()
    inventoried = [cap.where.split(":")[0] for cap in audit.INVENTORY]
    untriaged = [
        where for where, _text in audit.scan()
        if not any(fnmatch.fnmatch(where.split(":")[0], p) for p in inventoried)
    ]
    assert not untriaged, (
        "these token caps are not in scripts/audit_token_caps.py:INVENTORY. Add each one with the "
        "answer to 'what happens when the data is bigger than this':\n  " + "\n  ".join(untriaged)
    )


def test_every_inventory_entry_says_what_overflow_does():
    """An entry that only records the number repeats the original mistake in a nicer font."""
    audit = _audit()
    thin = [cap.where for cap in audit.INVENTORY if len(cap.on_overflow.strip()) < 40]
    assert not thin, f"these entries do not explain overflow behaviour: {thin}"


def test_the_synthesized_row_budget_covers_the_row_it_must_produce():
    """The specific regression, guarded by behaviour rather than by grepping for `512`.

    Note what is NOT asserted: that the budget differs between a small row and a large one. It used
    to, when each call site sized itself; now every teacher call asks for as much as the served
    context can return, so a small row and a large row both get the ceiling. That is the point of
    having one global budget — the property worth protecting is only ever "big enough".
    """
    import json

    from data.curriculum import _row_output_budget

    large = {"text": "You are AutoGPT. " + ("api_name description parameters " * 800),
             "answer": "Thought: go\nAction: x\nAction Input: {}",
             "tools": [{"name": f"api_{i}", "parameters": {"p": {}}} for i in range(16)]}
    needed = len(json.dumps(large)) / 4.0
    budget = _row_output_budget(large)
    assert budget >= needed, (
        f"budget {budget} < the ~{needed:.0f} tokens the row needs, so every generation would "
        "truncate mid-JSON and be dropped — this is exactly the 512 bug"
    )
    assert budget > 512, "512 is the value that lost 100% of toolbench generations"


def test_a_row_larger_than_the_global_ceiling_still_gets_room_for_itself(monkeypatch):
    """The ceiling is a runaway guard, not a limit on legitimate replies.

    A row that genuinely needs more than `SLM_MAX_OUTPUT_TOKENS` must still be granted what it needs
    — up to what the context can physically return — or the global ceiling just reintroduces the bug
    at a higher number.
    """
    import config.token_budget as budget_module
    from config.token_budget import output_budget

    monkeypatch.setattr(budget_module, "MAX_OUTPUT_TOKENS", 1024)
    monkeypatch.setenv("SLM_SYNTH_MAX_MODEL_LEN", "32768")
    assert output_budget(needed_chars=40_000) > 1024


def test_no_generation_call_site_carries_its_own_numeric_cap():
    """Every teacher call must route through `config.token_budget`, not a literal.

    Nine call sites each had their own hand-picked number and none was derived from anything. This
    catches a tenth being added, which is the mechanism by which the first nine accumulated.
    """
    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    literal_cap = re.compile(r"generate_fn\([^)]*,\s*[\d.]+\s*,\s*\d+\s*\)")
    offenders = []
    for relative in ("data/curriculum.py", "data/synth_client.py", "agent/teacher_fitness.py",
                     "eval/endpoint_eval.py", "agent/task_brief.py"):
        for number, line in enumerate(
            open(os.path.join(root, relative), encoding="utf-8").read().splitlines(), start=1
        ):
            if line.strip().startswith("#"):
                continue
            if literal_cap.search(line):
                offenders.append(f"{relative}:{number}  {line.strip()[:90]}")
    assert not offenders, (
        "these generation calls pass a hardcoded token cap; use "
        "config.token_budget.output_budget(prompt) instead:\n  " + "\n  ".join(offenders)
    )


def test_no_cap_can_silently_lose_data():
    """ZERO caps may be graded LOSSY.

    This started at six and was pinned as a tripwire; all six were replaced by
    `config.token_budget` on 2026-08-28, so the bar is now absolute rather than a ratchet. A new
    LOSSY entry has to fail this test and be argued for, which is the correct amount of friction for
    "this cap can silently throw data away".
    """
    audit = _audit()
    lossy = [f"{cap.where} — {cap.on_overflow[:80]}" for cap in audit.INVENTORY
             if cap.grade == audit.LOSSY]
    assert not lossy, (
        "these caps can silently lose or degrade data. Route the call through "
        "config.token_budget.output_budget / prompt_char_budget, or make the overflow LOUD:\n  "
        + "\n  ".join(lossy)
    )
