"""
Model pool files: the list of Hugging Face models to benchmark, e.g. SLM_Factory's config/android_pool.py.

Accepted formats (by extension):
  .py    SLM_Factory's android_pool.py or any Python file: every string literal that looks like a Hugging Face repo
         id ("org/name") is a model, in order of appearance. The file is parsed, never executed.
  .json  a list of ids; {"name": "org/id", ...}; {"name": {"hf_id": "org/id", "ref": "head"}, ...};
         or any of these under a top-level "models" key.
  other  text, one model per line: "org/id", "name=org/id" or "name=org/id@head"; '#' starts a comment.

Each model gets a short name (the built-in name if the id is in common.PAPER_MODELS / POOL_MODELS, else the
lower-cased repo name) and a build ref: "pinned" if the paper's llama.cpp (eadc418) can convert its architecture
(config.json "architectures" registered in that commit's convert_hf_to_gguf.py), else "head". An explicit "@head" /
"@pinned" or a JSON "ref" overrides the check.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from common import PAPER_MODELS, POOL_MODELS, REFS, THIRD_PARTY

HF_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
KNOWN = {hf: (name, "pinned") for name, hf in PAPER_MODELS.items()}
KNOWN.update({hf: (name, ref) for name, (hf, ref) in POOL_MODELS.items()})


def _looks_like_id(s: str) -> bool:
    return bool(HF_ID.match(s)) and not s.startswith((".", "/")) and not re.search(r"\.(py|json|txt|gguf|mnn)$", s)


def _ids_from_python(text: str) -> list[str]:
    nodes = [n for n in ast.walk(ast.parse(text))
             if isinstance(n, ast.Constant) and isinstance(n.value, str) and _looks_like_id(n.value)]
    nodes.sort(key=lambda n: (n.lineno, n.col_offset))  # ast.walk is breadth-first; keep the file's order
    return list(dict.fromkeys(n.value for n in nodes))


def _entries(path: Path) -> list[tuple[str | None, str, str | None]]:
    """[(name or None, hf_id, ref or None)] in file order."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return [(None, i, None) for i in _ids_from_python(text)]
    if path.suffix == ".json":
        data = json.loads(text)
        data = data.get("models", data) if isinstance(data, dict) else data
        if isinstance(data, list):
            return [(None, d, None) if isinstance(d, str) else (d.get("name"), d["hf_id"], d.get("ref")) for d in data]
        return [(n, v, None) if isinstance(v, str) else (n, v["hf_id"], v.get("ref")) for n, v in data.items()]
    out = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, rest = line.rpartition("=")
        hf, _, ref = rest.partition("@")
        out.append((name.strip() or None, hf.strip(), ref.strip() or None))
    return out


def default_name(hf_id: str) -> str:
    return hf_id.split("/")[-1].lower().replace("_", "-")


_registered: set | None = None


def pinned_architectures() -> set[str]:
    """Architectures the pinned llama.cpp converter registers (needs build_binaries.py's checkout)."""
    global _registered
    if _registered is None:
        src = THIRD_PARTY / "src" / f"llama.cpp-{REFS['pinned']['llama.cpp'][:12]}" / "convert_hf_to_gguf.py"
        if not src.is_file():
            raise SystemExit(f"{src} not found: run build_binaries.py (or --fetch-only) first, or give each pool "
                             f"entry an explicit @pinned / @head")
        _registered = set(re.findall(r'"([A-Za-z0-9_]+(?:ForCausalLM|ForConditionalGeneration|Model|LMHeadModel))"',
                                     src.read_text(encoding="utf-8")))
    return _registered


def detect_ref(hf_id: str) -> str:
    from huggingface_hub import hf_hub_download  # downloads config.json only (uses the cached HF login if gated)
    cfg = json.loads(Path(hf_hub_download(hf_id, "config.json")).read_text(encoding="utf-8"))
    archs = cfg.get("architectures") or []
    return "pinned" if archs and all(a in pinned_architectures() for a in archs) else "head"


def load_pool(path: str | Path) -> dict[str, tuple[str, str]]:
    """{name: (hf_id, ref)} in file order."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"pool file not found: {path}")
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    pool: dict[str, tuple[str, str]] = {}
    for name, hf_id, ref in _entries(path):
        if not _looks_like_id(hf_id):
            raise SystemExit(f"{path}: {hf_id!r} is not a Hugging Face repo id (org/name)")
        known = KNOWN.get(hf_id)
        name = name or (known[0] if known else default_name(hf_id))
        if not ref and not known:
            try:
                ref = detect_ref(hf_id)
            except GatedRepoError:
                raise SystemExit(f"{hf_id} is gated: accept its licence on huggingface.co and log in "
                                 f"(`hf auth login` or HF_TOKEN)")
            except RepositoryNotFoundError:
                if path.suffix != ".py":
                    raise SystemExit(f"{path}: {hf_id} not found on Hugging Face")
                print(f"[pool] skipping {hf_id!r} from {path.name}: not a Hugging Face model (a path?)")
                continue
        ref = ref or known[1]
        if ref not in REFS:
            raise SystemExit(f"{path}: {hf_id}: ref must be one of {list(REFS)}, got {ref!r}")
        pool[name] = (hf_id, ref)
    if not pool:
        raise SystemExit(f"{path}: no Hugging Face model ids found")
    return pool
