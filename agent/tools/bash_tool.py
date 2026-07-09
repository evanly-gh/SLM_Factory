# agent/tools/bash_tool.py
import subprocess
import sys
import os
from langchain_core.tools import tool

@tool
def bash(command: str) -> str:
    """
    Execute a shell command. slm_helpers.py is pre-loaded via PYTHONPATH.
    Use for: running train(), infer_batch(), dataset operations, eval scripts.
    Returns stdout + stderr combined.
    """
    # Build PYTHONPATH at call time (not import time) so CWD is correct for
    # every invocation regardless of where the module was first imported from.
    env = {
        **os.environ,
        "PYTHONPATH": f"{os.path.abspath('.')}:{os.environ.get('PYTHONPATH', '')}",
    }
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max for training runs
        env=env,
    )
    output = result.stdout
    if result.stderr:
        output += f"\n[stderr]\n{result.stderr}"
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"
    return output or "(no output)"
