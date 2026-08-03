import os
from dotenv import load_dotenv
from pathlib import Path

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

CRYSTALLM_DIR = Path("tools/CrystaLLM").resolve()
CRYSTALLM_PYTHON = CRYSTALLM_DIR / ".venv" / "bin" / "python"
CRYSTALLM_MODEL_DIR = CRYSTALLM_DIR / "crystallm_v1_large"

MP_API_KEY = os.environ.get("MP_API_KEY")

LOG_FILE = Path("logs/log.jsonl").resolve()

# --- Working directory ---
# The directory the agent operates in (contains ./calculations, ./template,
# and the example scripts it reads/writes). Override with MATAGENT_WORK_DIR.
WORK_DIR = Path(os.environ.get("MATAGENT_WORK_DIR", "test10")).resolve()
PROJECT_ROOT = "/nfs/roberts/project/pi_vsb4/byh2/chem_llm"