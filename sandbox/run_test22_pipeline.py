import torch
import json
import os
from chem_llm import config
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HOME"] = config.HF_HOME
from transformers import AutoModelForCausalLM, AutoTokenizer

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

COMPOUND = 'TiO2'
GROUP = 'rutile'
RELATIVISTIC = 'scalar'

TASK = f"""
Write three new Python scripts based off of the pre-existing generate_structures.py, setup_jobs.py, and plot_bands.py.
that generates perturbed {COMPOUND} crystal structures using pymatgen and input files for quantum espresso.
The existing pipeline interpolates between cubic and orthorhombic CsPbBr3.

Required workflow:
1. Read the existing example/generate_structures.py and note down how each constant and function works as well as anything you don't need for {COMPOUND} (for example, interpolating between phase transitions)
2. Use the generate_cif tool to obtain the base {GROUP} {COMPOUND} crystal structure.
3. In the python script, use pymatgen to generate 50 additional structures for 51 total structures.
4. The perturbations should be chemically reasonable random displacements of each atom at room temperature.
5. Then read the existing example/setup_jobs.py, noting down what it does and any changes you must make to work with the new workflow.
6. Use get_pseudopotential to get necessary pseudopotentials. Please use norm conserving, {RELATIVISTIC} relativisitc, pbesol, stringent, UPF pseudopotentials.
7. Before writing any code for steup_jobs, you MUST write a note where you go through every option in the ESPRESSO input file and note down what the option means and what the value should be for this new workflow.
8. Use the search_docs tool for any options you are unsure about.
9. For each generated submit.sh script, specify account m4735 and email brent.hu@yale.edu
10. Use the run_espresso_workflow tool to test the espresso input file for directory calculations/000
11. Fix all errors before moving on.
10. Read example/plot_bands.py, noting down anything that needs to be changed.
11. Write new plot_bands.py file and run it.

Requirements:
- Organize the output files in a clear directory structure.
- Comment the code where nontrivial crystallographic operations are performed.

Before calling done:
- Execute generate_structures.py, setup_jobs.py, and plot_bands.py.
- Verify it completes without errors.
- Read at least one intermediate CIF and one quantum espresso input file to confirm it appears reasonable.

In the done summary, include:
- The Materials Project IDs used for the endpoint structures.
- An outline of how the new python workflow works.
- Any scientific assumptions made and a quick justification.
- Any details that might require further expert domain knowledge.
"""

os.makedirs(config.WORK_DIR, exist_ok=True)
os.chdir(config.WORK_DIR)
print(f"Working directory: {os.getcwd()}")

final_state = run_agent(TASK, model, tokenizer, verbose=True, clear_dir=True, load_path=config.EXAMPLE_DIR)

print("\nFINAL STATE:\n", json.dumps(final_state.to_dict(), indent=2))
