"""Unified-diff verifier (2026-08-01): `git apply --check` + applied-result match.

A judge-free scorer for the ``diff`` task type: the model edits a source text and must express
the change as a UNIFIED DIFF. Each eval row carries ``{text, answer, src, tgt}`` where:

- ``text``   — the edit instruction (may embed the source).
- ``src``    — the original text the diff applies to.
- ``tgt``    — the expected edited text.
- ``answer`` — the GOLD unified diff of ``src`` → ``tgt`` (computed with difflib at load time).

Two numbers per row (two-column format-vs-content split):

- ``format_valid``   — the predicted diff cleanly passes ``git apply --check`` against ``src``.
- ``content_correct``— applying the predicted diff to ``src`` yields ``tgt`` (exact match after
                       trailing-whitespace/newline normalization). This is the ``f1`` scalar.

``git`` runs in an isolated temp dir with a short timeout — no network, no repo mutation. If
``git`` is unavailable the row scores 0 with a diagnostic rather than crashing the run.
"""
import os
import shutil
import subprocess
import tempfile

from data.eval_set import EvalSet

DIFF_PROMPT = (
    "Edit the source text as instructed and reply with ONLY a unified diff (the output of "
    "`diff -u`) that applies the change. Do not include prose or Markdown fences.\n\n"
    "Instruction: {text}\n\nSource:\n{src}"
)

_GIT_TIMEOUT_S = 10.0


def build_prompts(eval_set: EvalSet) -> list[str]:
    return [
        DIFF_PROMPT.format(text=ex.get("text", ""), src=ex.get("src", ""))
        for ex in eval_set.all
    ]


_FENCE_LANGS = ("diff", "patch", "")


def _strip_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop the opening fence line and a trailing fence line if present.
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def extract_predictions(raw_outputs: list[str], eval_set: EvalSet) -> list[str]:
    return [_strip_fences(raw) for raw in raw_outputs]


def _git_available() -> bool:
    return shutil.which("git") is not None


def _normalize(text: str) -> str:
    """Normalize trailing whitespace/newlines for a content-equality check."""
    return "\n".join(line.rstrip() for line in str(text or "").splitlines()).rstrip("\n")


def _apply_diff(src: str, diff_text: str) -> tuple[bool, str | None, str]:
    """Apply a unified diff to ``src`` in an isolated temp dir.

    Returns (format_valid, applied_text_or_None, diagnostic). format_valid is True iff
    ``git apply --check`` passes; applied_text is the file content after ``git apply`` (None if
    the check failed or git is unavailable).
    """
    if not diff_text.strip():
        return False, None, "empty diff"
    if not _git_available():
        return False, None, "git not on PATH"

    workdir = tempfile.mkdtemp(prefix="diffscore_")
    try:
        # git apply needs a repo to resolve paths against.
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        subprocess.run(["git", "init", "-q"], cwd=workdir, env=env,
                       capture_output=True, timeout=_GIT_TIMEOUT_S)

        # The gold diff (from difflib) uses 'a'/'b' path prefixes over a file named for the
        # row; we write both the plain target name and let -p1 strip the a/b prefixes.
        target = _diff_target_path(diff_text)
        file_path = os.path.join(workdir, target)
        os.makedirs(os.path.dirname(file_path) or workdir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(src if src.endswith("\n") else src + "\n")

        diff_bytes = (diff_text if diff_text.endswith("\n") else diff_text + "\n").encode("utf-8")
        check = subprocess.run(
            ["git", "apply", "--check", "--recount", "-"],
            cwd=workdir, env=env, input=diff_bytes,
            capture_output=True, timeout=_GIT_TIMEOUT_S,
        )
        if check.returncode != 0:
            return False, None, (check.stderr.decode("utf-8", "replace")[:200] or "apply --check failed")

        applied = subprocess.run(
            ["git", "apply", "--recount", "-"],
            cwd=workdir, env=env, input=diff_bytes,
            capture_output=True, timeout=_GIT_TIMEOUT_S,
        )
        if applied.returncode != 0:
            return True, None, (applied.stderr.decode("utf-8", "replace")[:200] or "apply failed")
        with open(file_path, encoding="utf-8") as handle:
            return True, handle.read(), ""
    except subprocess.TimeoutExpired:
        return False, None, "git apply timed out"
    except Exception as error:  # noqa: BLE001 - never crash the eval on a bad diff
        return False, None, f"git apply error: {str(error)[:160]}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _diff_target_path(diff_text: str) -> str:
    """Best-effort file path the diff edits, from its +++ header; default 'file.txt'."""
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip().split("\t")[0]
            if path.startswith("b/"):
                path = path[2:]
            if path and path != "/dev/null":
                return path
    return "file.txt"


def score(eval_set: EvalSet, predictions: list[str]) -> dict:
    content_scores: list[float] = []
    format_scores: list[float] = []
    failures: list[dict] = []
    diagnostics: list[dict] = []
    for ex, pred in zip(eval_set.all, predictions):
        src = ex.get("src", "")
        tgt = ex.get("tgt", "")
        format_valid, applied, diagnostic = _apply_diff(src, pred)
        content = 1.0 if (applied is not None and _normalize(applied) == _normalize(tgt)) else 0.0
        format_scores.append(1.0 if format_valid else 0.0)
        content_scores.append(content)
        diagnostics.append({"format_valid": format_valid, "diagnostic": diagnostic})
        if content < 1.0:
            failures.append({**ex, "predicted": pred, "format_valid": format_valid,
                             "diagnostic": diagnostic})

    n = len(content_scores)
    f1 = sum(content_scores) / n if n else 0.0
    format_valid_mean = sum(format_scores) / n if n else 0.0

    return {
        "f1": f1,
        "metric": "apply_match",
        "per_class": {"apply_match": f1, "format_valid": format_valid_mean},
        "failures": failures,
        "execution_diagnostics": diagnostics,
    }
