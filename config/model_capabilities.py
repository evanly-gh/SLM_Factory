"""Loader for the offline model-capability descriptions doc (B161).

The doc (model_capabilities.md) gives sourced per-model capability notes plus the
MMLU/MMLU-Pro/MMLU-Redux comparability contract, injected into orchestrator prompts (initial
selection, escalation, downward re-exploration) so the LLM does not collapse distinct metrics
or interpret a missing value as zero.
"""
import os

_DOC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_capabilities.md")
_CACHE: str | None = None


def load_capability_doc() -> str:
    """Return the full capability doc text (cached). Empty string if missing."""
    global _CACHE
    if _CACHE is None:
        try:
            with open(_DOC_PATH, encoding="utf-8") as f:
                _CACHE = f.read()
        except FileNotFoundError:
            _CACHE = ""
    return _CACHE


def capability_sections(model_ids) -> str:
    """Return only the doc sections for the given model_ids (plus the shared caveat header),
    so a prompt shows capability notes for just the candidates under consideration."""
    doc = load_capability_doc()
    if not doc:
        return ""
    lines = doc.splitlines()
    # Keep the header/caveat (everything before the first "## ") always.
    header, sections, cur, cur_id = [], [], [], None
    in_sections = False
    for ln in lines:
        if ln.startswith("## "):
            in_sections = True
            if cur and cur_id is not None:
                sections.append((cur_id, "\n".join(cur)))
            cur = [ln]
            cur_id = ln[3:].split("(")[0].strip()
        elif in_sections:
            cur.append(ln)
        else:
            header.append(ln)
    if cur and cur_id is not None:
        sections.append((cur_id, "\n".join(cur)))

    wanted = {str(m) for m in model_ids}
    picked = [txt for sid, txt in sections if sid in wanted]
    if not picked:
        return doc  # fall back to the whole doc if we couldn't match ids
    return "\n".join(header).strip() + "\n\n" + "\n\n".join(picked)
