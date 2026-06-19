# agent/tools/bash_tool.py
import subprocess
import sys
import os
from langchain_core.tools import tool

# Inject slm_helpers into the bash environment path
_HELPERS_INJECT = f"export PYTHONPATH={os.path.abspath('.')}:$PYTHONPATH"

@tool
def bash(command: str) -> str:
    """
    Execute a shell command. slm_helpers.py is pre-loaded via PYTHONPATH.
    Use for: running train(), infer_batch(), dataset operations, eval scripts.
    Returns stdout + stderr combined.
    """
    full_command = f"{_HELPERS_INJECT} && {command}"
    result = subprocess.run(
        full_command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max for training runs
    )
    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"
    return output or "(no output)"
