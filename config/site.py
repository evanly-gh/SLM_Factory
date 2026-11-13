"""Site configuration (site.env) for Python entry points.

The Python twin of scripts/site_env.sh: loads `site.env` from the repo root and fills in the
same defaults, with the same precedence (environment > site.env > default). Call
`load_site_env()` before importing anything that reads HF_HOME at import time.
See site.env.example for the keys.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def load_site_env() -> None:
    """Export site.env and the documented defaults into os.environ (never overriding)."""
    os.environ.setdefault("SLM_ROOT", str(ROOT))
    for key, value in _read(ROOT / "site.env").items():
        os.environ.setdefault(key, value)
    cache = os.environ.setdefault("SLM_CACHE_DIR", str(ROOT / ".cache"))
    os.environ.setdefault("UV_CACHE_DIR", os.path.join(cache, "uv"))
    os.environ.setdefault("PIP_CACHE_DIR", os.path.join(cache, "pip"))
    os.environ.setdefault("HF_HOME", os.path.join(cache, "huggingface"))
    os.environ.setdefault("SLM_TMP_DIR", str(ROOT / ".tmp"))
    os.environ.setdefault("LLAMA_CPP_DIR", str(ROOT.parent / "llama.cpp"))


def tmp_dir() -> str:
    """SLM_TMP_DIR after loading site.env."""
    load_site_env()
    return os.environ["SLM_TMP_DIR"]
