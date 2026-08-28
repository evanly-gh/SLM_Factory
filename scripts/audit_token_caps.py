"""Every token cap and truncation in the pipeline, and whether anything checks it against the data.

WHY THIS EXISTS
    `data/curriculum.py` generated synthetic rows with `max_tokens=512`, hardcoded, for every task in
    the registry. A toolbench row serialises to 3,913 characters at the very smallest, so ZERO of
    4,995 rows could fit. Every generation was cut off mid-JSON and dropped by a bare
    `except Exception: return None`. Three runs — 38832586, 38985393, 39041380 — reported "0 rows
    kept" from 1,019 / 133 / 425 attempts, and all three were read as the VERIFIER rejecting
    everything, because the counter that produced that message computes attempted-minus-kept and
    attributes the gap to verification. Roughly 40 hours of GPU time went into three different wrong
    diagnoses of one constant.

    A cap is dangerous in proportion to how quietly it fails. The ones worth fearing share three
    properties: the value is a CONSTANT, the thing being capped VARIES BY TASK, and overflowing it
    produces something that looks like a different failure. That is the pattern this file exists to
    make visible.

WHAT IT DOES
    Holds a hand-written INVENTORY of every cap, each with the question "what happens when the data
    exceeds this", then scans the source tree and FAILS on any cap that is not in the inventory. The
    scan cannot judge intent, so it does not try; it just refuses to let a new cap go untriaged. Same
    contract as the slurm-script accounting test.

USAGE
    python scripts/audit_token_caps.py            # inventory + scan for untriaged caps
    python scripts/audit_token_caps.py --risky    # only the ones that can silently lose data
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Risk grades. The distinction that matters is not the size of the cap, it is what overflowing it
# looks like from the outside.
SAFE = "SAFE"          # derived from the data or the spec, so it cannot be outgrown
BOUNDED = "BOUNDED"    # constant, but the thing capped is provably smaller — with the measurement
LOSSY = "LOSSY"        # overflow silently discards or corrupts data. These are the bugs.
COSMETIC = "COSMETIC"  # truncates a log line or an error string; affects reading, not results


@dataclass(frozen=True)
class Cap:
    where: str          # file:line
    what: str           # what it bounds
    value: str          # the cap itself
    grade: str
    on_overflow: str    # what actually happens when the data is bigger


INVENTORY: tuple[Cap, ...] = (
    # --- the single global ceiling every teacher call now routes through -------------------
    Cap("config/token_budget.py:MAX_OUTPUT_TOKENS", "output tokens for ANY teacher call",
        "SLM_MAX_OUTPUT_TOKENS (16384), clamped to max_model_len minus the prompt", SAFE,
        "a reply longer than the ceiling would be cut, but nothing in the suite has a legitimate "
        "reply that long, so hitting it means something has gone wrong rather than that the number "
        "is too small. `needed_chars` raises the ceiling for a caller that can MEASURE what its "
        "reply must contain, so the ceiling cannot re-create the 512 bug at a higher number. The "
        "real bound is arithmetic: asking for more than the server can return is an HTTP 400, not a "
        "truncation, so it is computed rather than guessed."),
    # --- generation, all now derived (were six separate hand-picked constants) --------------
    Cap("data/curriculum.py:_row_output_budget", "one synthesized row",
        "global ceiling, floored by the anchor's measured JSON length", SAFE,
        "cannot be outgrown: the floor is measured from the anchor the row must reproduce. WAS a "
        "hardcoded 512, which no toolbench row could fit — 0 of 4,995 — losing three runs to a "
        "misdiagnosis (see module docstring)."),
    Cap("data/curriculum.py:78", "teacher's chain-of-thought annotation", "global ceiling", SAFE,
        "was 512, which cut long math and code chains mid-sentence and attached the PARTIAL "
        "reasoning to the row as training data with nothing checking it terminated."),
    Cap("data/curriculum.py:689", "teacher's answer-verification verdict", "global ceiling", SAFE,
        "was 160. Verification is FAIL-OPEN, so a verdict truncated mid-reason did not make the "
        "verifier strict — it made it structurally unable to reject anything."),
    Cap("data/curriculum.py:965", "teacher's label-verification verdict", "global ceiling", SAFE,
        "was 120: the same fail-open truncation, 40 tokens tighter."),
    Cap("data/curriculum.py:1106", "one new synthesized utterance", "global ceiling", SAFE,
        "was 200 (~800 chars). This path serves the CLOSED-LABEL tasks only (clinc150, sms_spam, "
        "routerbench, proactive_listening), whose utterances are one sentence, so 200 was ample — "
        "ner_bc5cdr does NOT route here despite the old comment saying so. Still replaced, because "
        "'ample for the tasks that happen to use it today' is exactly the reasoning that produced "
        "the 512."),
    Cap("data/synth_client.py:215", "default when a caller omits max_tokens", "global ceiling", SAFE,
        "was 200, so a new call site that forgot the argument silently got 200. That default is "
        "where the habit of guessing this number came from."),
    # --- prompt-side clips, now a share of the real context --------------------------------
    Cap("agent/task_brief.py:99", "each example row shown to the brief author",
        "prompt_char_budget(0.20)", SAFE,
        "was a flat 1,200 chars. A toolbench row is ~10,000, so the author saw roughly an eighth of "
        "one example and wrote the task description every later synthesis and verification prompt "
        "depends on from that fragment. Silent, because a brief written from a fragment still reads "
        "fluently."),
    Cap("data/curriculum.py:653", "reference examples in the verification prompt",
        "prompt_char_budget(0.10) each", SAFE,
        "was a flat 400 chars. These pairs teach the verifier the task's conventions, and on a "
        "long-row task the convention being demonstrated fell off the end, leaving the verifier "
        "judging its own guess (B269)."),
    # --- evaluation: the student's reserve, which CANNOT be globally raised ------------------
    Cap("tasks/*.py:max_new_tokens", "student's eval output reserve, per task",
        "50 (proactive) - 1536 (toolbench)", BOUNDED,
        "deliberately NOT routed through the global ceiling. The reserve and the prompt share one "
        "window — input budget is `max_seq_length - max_new_tokens`, and eval_output_token_reserve "
        "RAISES when a reserve leaves no prompt room — so raising it does not uncap the model, it "
        "starts refusing rows at load time and changes every task's measured score. A per-task "
        "measurement decision, kept in the registry where it is visible."),
    Cap("eval/harness.py:eval_output_token_reserve", "reads the per-task reserve",
        "TaskSpec.max_new_tokens", SAFE,
        "validates the reserve against the task's own context and raises rather than silently "
        "clamping, which is what makes the coupling above safe to rely on."),
    Cap("eval/scorers/toolbench.py:_judge_verdict_budget", "one ToolEval judge verdict",
        "global ceiling", SAFE,
        "was 192. This is the highest-volume call in a toolbench run (7,209 measured in one run) and "
        "a cut-off verdict was counted judge_unsure, quietly moving the metric. Generation stops at "
        "EOS, so the larger ceiling costs nothing."),
    Cap("eval/judge_client.py:207", "numeric judge rubric output", "JudgeRubric.max_tokens (8)",
        BOUNDED,
        "declared on the rubric rather than at the call site, so it is a property of the rubric's "
        "contract: it asks for a bare number. A model that ignored that and wrote prose would "
        "truncate, fail to parse as a float, and be reported as an unusable judgement rather than "
        "silently averaged in as a zero."),
    Cap("agent/teacher_fitness.py:301", "teacher's answers when measuring fitness",
        "eval_output_token_reserve(spec)", SAFE,
        "cannot be outgrown: the same per-task reserve the student is judged under, which is what "
        "makes the two numbers comparable at all."),
    Cap("eval/endpoint_eval.py:110", "reference model's baseline answers",
        "eval_output_token_reserve(spec)", SAFE,
        "as above. A row the reserve cannot hold is one the student could not have answered either, "
        "so the two fail identically rather than the baseline being flattered."),
    Cap("training/quantize.py:155", "smoke-test generation after quantizing", "16 tokens", SAFE,
        "checks only that the quantized model emits anything at all; the content is never read."),
    # --- orchestrator and loaders ----------------------------------------------------------
    Cap("agent/nodes/iterate.py:_ITERATE_MAX_TOKENS", "orchestrator decision JSON", "env-tunable",
        BOUNDED, "the schema is small and fixed, and a truncation fails plan validation loudly and "
        "falls back to a deterministic plan rather than proceeding on half a decision."),
    Cap("data/loaders/toolbench.py:127", "prompt and target length", "per-task char budget", SAFE,
        "REFUSES an oversized row instead of truncating it, because cutting a ToolBench prompt "
        "removes the tail of the API list — the names the model is meant to call. Both consumers "
        "raise loudly rather than score a truncated prompt. This is the pattern the rest should "
        "follow: refuse, do not truncate, whenever the tail carries meaning."),
    Cap("data/loaders/web_acquire.py:1872", "window sent for NER auto-annotation",
        "env SLM_NER_ANNOTATION_WINDOW_CHARS", BOUNDED,
        "entities outside the window are simply not annotated, so a row comes out incomplete rather "
        "than wrong, and the substring verifier still passes it — which is why it is worth watching."),
    Cap("data/loaders/proactive_listening.py:92", "dialogue context kept per row", "1200 chars",
        SAFE, "keeps the TAIL, which is the part the label depends on, and records on the row that "
        "it did so."),
    # --- cosmetic --------------------------------------------------------------------------
    Cap("data/curriculum.py:603", "rendered row in a log line", "explicit limit + marker", COSMETIC,
        "appends '… [truncated, N chars total]' so the reader is told the line was shortened."),
    Cap("scripts/diag_adapter_effect.py:72", "output length in the adapter-effect diagnostic",
        "64 tokens", COSMETIC,
        "a hand-run diagnostic asking only whether the adapter changed the output at all; it scores "
        "nothing and feeds no decision."),
    Cap("*:str(exc)[:120..500]", "exception text in logs and telemetry", "120-500 chars", COSMETIC,
        "a long message loses its tail in the log while the exception object itself is untouched, so "
        "control flow cannot change. The cost is purely diagnostic, and it has bitten once: "
        "web_acquire cut a Hub error at 80 characters mid-path and hid which file was missing, which "
        "is why these are 120+ today and why the limit belongs on the log line, never on the raise."),
)

# Patterns that constitute a cap for scanning purposes.
SCAN = (
    re.compile(r"max_tokens\s*=\s*(\d+)"),
    re.compile(r"max_new_tokens\s*=\s*(\d+)"),
    re.compile(r"generate_fn\([^)]*,\s*[\d.]+\s*,\s*(\d+)\s*\)"),
)
# `unsloth_compiled_cache` is machine-generated vendored trainer code we neither wrote nor call
# directly; its caps belong to upstream and triaging them here would be noise that trains the
# reader to ignore this report.
SKIP_DIRS = ("tests", "logs", ".venv", ".venv_gpu", ".venv_vllm", "docs", "__pycache__",
             "unsloth_compiled_cache")
SKIP_FILES = ("audit_token_caps.py",)  # this file quotes caps in prose


def scan() -> list[tuple[str, str]]:
    """Every numeric token cap in the source tree, as (file:line, text)."""
    found: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if not name.endswith(".py") or name in SKIP_FILES:
                continue
            path = os.path.join(root, name)
            relative = os.path.relpath(path, PROJECT_ROOT)
            try:
                lines = open(path, encoding="utf-8").read().splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, start=1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"'):
                    continue
                for pattern in SCAN:
                    if pattern.search(line):
                        found.append((f"{relative}:{number}", stripped[:110]))
                        break
    return found


def main() -> int:
    risky_only = "--risky" in sys.argv
    order = {LOSSY: 0, BOUNDED: 1, SAFE: 2, COSMETIC: 3}
    caps = sorted(INVENTORY, key=lambda c: (order[c.grade], c.where))
    if risky_only:
        caps = [c for c in caps if c.grade == LOSSY]

    for grade in (LOSSY, BOUNDED, SAFE, COSMETIC):
        group = [c for c in caps if c.grade == grade]
        if not group:
            continue
        print("=" * 100)
        print(f"{grade}  ({len(group)})")
        print("=" * 100)
        for cap in group:
            print(f"\n  {cap.where}")
            print(f"    bounds : {cap.what}")
            print(f"    cap    : {cap.value}")
            print(f"    if data exceeds it: {cap.on_overflow}")
    print()

    # The scan does not judge; it refuses to let a new cap go untriaged.
    import fnmatch

    # Glob-aware: one entry may cover a family, e.g. `tasks/*.py:max_new_tokens` covers the
    # per-task reserve declared in every task module. Matching on exact filenames reported all ten
    # of those as untriaged while the inventory plainly described them.
    inventoried = [c.where.split(":")[0] for c in INVENTORY]
    untriaged = [
        (where, text) for where, text in scan()
        if not any(fnmatch.fnmatch(where.split(":")[0], pattern) for pattern in inventoried)
    ]
    print("=" * 100)
    if untriaged:
        print(f"UNTRIAGED CAPS ({len(untriaged)}) — add each to INVENTORY with what happens on "
              f"overflow:")
        for where, text in untriaged:
            print(f"  {where:52} {text}")
        return 1
    print(f"all token caps are triaged ({len(INVENTORY)} entries, "
          f"{sum(1 for c in INVENTORY if c.grade == LOSSY)} graded LOSSY)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
