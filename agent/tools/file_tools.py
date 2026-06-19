# agent/tools/file_tools.py
import os
from langchain_core.tools import tool

@tool
def read_file(path: str) -> str:
    """Read a file from disk. Use for: datasets, configs, data-curation.md, eval results."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[ERROR] File not found: {path}"
    except Exception as e:
        return f"[ERROR] {e}"

@tool
def edit_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"[ERROR] {e}"
