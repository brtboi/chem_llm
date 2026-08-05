import os
from dotenv import load_dotenv
from pathlib import Path

from . import REPO_ROOT

load_dotenv()

# --- Model ---
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_HOME = os.environ.get("HF_HOME", "/pscratch/sd/b/brenthu/huggingface")
print("HF_HOME: ", HF_HOME)

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# --- Generation settings ---
MAX_NEW_TOKENS = 5000
TEMPERATURE = 0.0
DO_SAMPLE = False

MAX_AGENT_STEPS = 24

# --- Tool settings ---
READ_MAX_CHARS = 10000

MP_API_KEY = os.environ.get("MP_API_KEY")

LOG_FILE = REPO_ROOT / "logs" / "log.jsonl"

# --- Working directory ---
# The directory the agent operates in (contains ./calculations, ./template,
# and the example scripts it reads/writes). Anchored to REPO_ROOT rather
# than cwd, so it resolves the same regardless of where a script/notebook
# was launched from. MATAGENT_WORK_DIR may still override it with either a
# path relative to REPO_ROOT or an absolute path -- an absolute value on
# the right of `/` replaces the left side entirely, per pathlib semantics,
# so no separate branch is needed for the two cases.
WORK_DIR = (REPO_ROOT / os.environ.get("MATAGENT_WORK_DIR", "sandbox/test10")).resolve()