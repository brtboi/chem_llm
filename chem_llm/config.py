import os
from dotenv import load_dotenv
from pathlib import Path

from . import REPO_ROOT

load_dotenv()

# --- Model ---
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_HOME = os.environ.get("HF_HOME", "/pscratch/sd/b/brenthu/huggingface")
print("HF_HOME: ", HF_HOME)

# MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_NAME = "Qwen/Qwen3.8-27B"

# --- Claude API backend (chem_llm/claude_backend.py) ---
# An alternative to the local Qwen model: same prompt, same JSON tool-call
# protocol, same agent loop -- only the text generation differs.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
# Non-streaming default; leaves room for adaptive thinking, which is on by
# default on Opus 5 and is billed/emitted separately from the text we parse.
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", 16000))
# low | medium | high | xhigh | max. Each agent step is a small, well-specified
# decision, so the default sits below the API default of high.
CLAUDE_EFFORT = os.environ.get("CLAUDE_EFFORT", "medium")

# --- Generation settings ---
MAX_NEW_TOKENS = 5000
TEMPERATURE = 0.0
DO_SAMPLE = False
MAX_HISTORY = 32

MAX_AGENT_STEPS = 44

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
WORK_DIR = (REPO_ROOT / os.environ.get("MATAGENT_WORK_DIR", "sandbox/test22")).resolve()

EXAMPLE_DIR = (REPO_ROOT / os.environ.get("EXAMPLE_DIR", "example")).resolve()

# Reference scripts + real, validated DFT output for a single (unperturbed
# cubic) CsPbBr3 structure -- the load_path for the DeePseudopot tool-use
# notebook (deepseudopot_agent.ipynb). See deepseudopot_example/README.md.
DPP_EXAMPLE_DIR = (REPO_ROOT / os.environ.get("DPP_EXAMPLE_DIR", "deepseudopot_example")).resolve()

# Reference scripts (qe_bands_to_ref.py, setup_nn_inputs.py, nn_template/)
# for adapting this pipeline's own QE DFT output into a DeePseudopot input
# bundle -- lives under EXAMPLE_DIR so it travels with the main example/
# load, but is only copied into a run's working directory when a caller
# opts in (see run_agent's load_deepseudopot flag), since most runs never
# touch DeePseudopot at all.
DPP_EXAMPLE_SUBDIR = EXAMPLE_DIR / "deepseudopot"