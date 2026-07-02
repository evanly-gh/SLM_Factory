# config.py
import os

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
EXA_API_KEY = os.environ["EXA_API_KEY"]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

ORCHESTRATOR_MODEL = "claude-sonnet-4-6"

# Teacher models for CoT annotation (paper §2.3, §2.5)
# DeepSeek-R1 for math/science reasoning; GPT-4.1 for code/QA
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
TEACHER_MODEL_DEEPSEEK = "deepseek-reasoner"
TEACHER_MODEL_GPT = "gpt-4.1"
TEACHER_MODEL_CLAUDE = "claude-sonnet-4-6"
MAX_TURNS_MAIN = 1500

DEFAULT_STOP_THRESHOLD = 0.96

ARTIFACTS_DIR = "artifacts"
