"""Rutile TiO2 from-scratch benchmark -- local Qwen model.

Paired with run_test_claude_api_pipeline.py, which runs the identical task
through the Claude API. Everything shared (task text, tool set, step budget,
doc index) lives in tio2_task.py so the two runs differ only in the model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tio2_task import DOC_INDEX_DIR, MAX_STEPS, TASK, hide_deepseudopot

# Must be set before any chem_llm import: config reads MATAGENT_WORK_DIR and
# retrieval.config reads DOC_INDEX_DIR at import time.
os.environ["MATAGENT_WORK_DIR"] = "sandbox/test27"
os.environ["DOC_INDEX_DIR"] = DOC_INDEX_DIR

import torch
import json
from chem_llm import config
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HOME"] = config.HF_HOME
from transformers import AutoModelForCausalLM, AutoTokenizer

hide_deepseudopot()

from chem_llm.agent_core import run_agent

if not config.HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set.")

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16,
    token=config.HF_TOKEN,
)

os.makedirs(config.WORK_DIR, exist_ok=True)
os.chdir(config.WORK_DIR)
print(f"Working directory: {os.getcwd()}")

final_state = run_agent(
    TASK, model, tokenizer, verbose=True,
    clear_dir=True, load_path=None,  # no example directory
    max_steps=MAX_STEPS,
    model_name=config.MODEL_NAME,
)

print("\nFINAL STATE:\n", json.dumps(final_state.to_dict(), indent=2))
