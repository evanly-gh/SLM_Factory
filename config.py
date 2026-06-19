# config.py
import os

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
EXA_API_KEY = os.environ["EXA_API_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

ORCHESTRATOR_MODEL = "claude-sonnet-4-6"
MAX_TURNS_MAIN = 1500
MAX_TURNS_SUBAGENT = 1000

DEFAULT_STOP_THRESHOLD = 0.96
DEFAULT_STAGNATION_ROUNDS = 2

CLUSTER_GPU = "L40"  # Phase 1 default; H200/A100 for larger jobs
DATA_DIR = "data_cache"
ARTIFACTS_DIR = "artifacts"
