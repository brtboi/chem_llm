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

# TASK = """
#     Your task is to use the generate cif file to generate a cif file for Cs4Pb4Br12 with space group Pnma. Then, read the cif file and confirm that it makes sense scientifically.
# """

# TASK = """
#     Your task is to create a version of the existing workflow in the ../example/ directory for a CsPbCl3 while preserving overall functionality
#     Please open and creafully read ../example/setup_jobs.py.
#     Directories ./calculations and ./template has already been made for you.
#     Write in a note the function of each constant, function, randomization, including whether it is specific for CsPbBr3 or generic,
#     determine which parts must be rewritten for the new target compound listed above.
#     Then, write to a new file a python script that follows the same workflow structure but for the new target compound.
#     Please use the generate cif tool to generate a cif file for CsPbCl3 at ./structures/structure_000.cif and NOT a python script.
#     When writing setup_jobs.py, please use brent.hu@yale.edu as the email and m4735 as the iris slurm account.
#     Please review to make sure it works and then run setup_jobs.py
#     In the final step, please provide a concise summary of the changes made, any assumptions made, and any remaining uncertainties requiring domain expertise.
# """

# TASK = """
#     Your task is to generate a .cif file for CsPbBr3. Do NOT use the generate cif file tool. Simply write to a new file.
#     The crystal should be of the space group Pnma with corner sharing octohedra and a unit cell of Cs4Pb4Br12.
#     Call the file CsPbBr3_model.cif.
#     After writing, please review the file to make sure all numbers look scientifically correct, and write a note with any assumptions made.
# """




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