#!/usr/bin/env python3
import json
import os

import config

# HF_HOME must be set before transformers is imported so it picks up the cache dir.
os.environ["HF_HOME"] = config.HF_HOME

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from agent_core import run_agent

# TASK = """
#     Your task is to create a version of the existing workflow in the ../example/ directory for a simple silicon crystal while preserving overall functionality
#     Please open and creafully read ../example/generate_structures.py and ../example/setup_jobs.py.
#     Directories ./calculations and ./template has already been made for you.
#     Please use the following pseudopotential file: ./template/Si.upf.
#     Write in a note the function of each constant, function, randomization, including whether it is specific for CsPbBr3 or generic,
#     determine which parts must be rewritten for the new target compound listed above.
#     Then, write to a new file a python script that follows the same workflow structure but for the new target compound.
#     Please review to make sure it works and then run generate_structures.py, setup_jobs.py
#     In the final step, please provide a concise summary of the changes made, any assumptions made, and any remaining uncertainties requiring domain expertise
# """

TASK = """
    Your task is to use the generate cif file to generate a cif file for CsPbBr3. Then, read the cif file and confirm that it makes sense scientifically.
"""


def load_model():
    if not config.HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. Run `export HF_TOKEN=hf_xxx` before launching main.py."
        )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        token=config.HF_TOKEN,
    )
    return model, tokenizer


def main():
    os.chdir(config.WORK_DIR)
    print(f"Working directory: {os.getcwd()}")

    model, tokenizer = load_model()
    final_state = run_agent(TASK, model, tokenizer, verbose=False)

    print("\nFINAL STATE:\n", json.dumps(final_state.to_dict(), indent=2))


if __name__ == "__main__":
    main()