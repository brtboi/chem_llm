import os

# Must be set before chem_llm.config is imported (it reads this env var at
# import time to compute WORK_DIR). This is the master-branch run (pymatgen,
# restructured INVARIANTS/DECISION RULES/WORKFLOW prompt per
# new_prompt_framework.md), kept in its own sandbox/test24 directory.
os.environ["MATAGENT_WORK_DIR"] = "sandbox/test24"

import torch
import json
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

# Kept identical to main.ipynb's TASK cell -- see that notebook for the
# source of truth; this is the headless mirror used for unattended runs.
TASK = f"""
Write three new Python scripts based off of the pre-existing generate_structures.py, setup_jobs.py, and plot_bands.py, that generate perturbed {COMPOUND} crystal structures using pymatgen and input files for Quantum ESPRESSO. The existing pipeline interpolates between cubic and orthorhombic CsPbBr3; this new workflow is a single {GROUP} {COMPOUND} phase (no phase interpolation).

Apply the INVARIANTS and DECISION RULES from the system prompt throughout this task -- they are not repeated here.

WORKFLOW (checkpoint after each step: state in 1-2 sentences what you verified before moving to the next step; do not batch verification to the end):

STEP 0: Read example/generate_structures.py, example/setup_jobs.py, and example/plot_bands.py in full before writing anything. Note per file what each constant/function does, and what is CsPbBr3-specific (e.g. the cubic/orthorhombic phase interpolation) that must be removed/replaced for {COMPOUND}.

STEP 1: Obtain the base {GROUP} {COMPOUND} structure via generate_cif. Record the Materials Project ID used.

STEP 2: Generate 50 additional perturbed structures with pymatgen (51 total). The perturbations should be chemically reasonable random displacements of each atom at room temperature. Apply the nat/ntyp INVARIANT check immediately per structure.

STEP 3: Get pseudopotentials via get_pseudopotential -- norm conserving, {RELATIVISTIC} relativistic, pbesol, stringent, UPF -- applying the pseudopotential relativistic-treatment DECISION RULE.

STEP 4: Before writing any setup_jobs code, write a note walking through every option in the Quantum ESPRESSO input file and what its value should be for this workflow. Use search_docs for anything you are unsure about.

STEP 5: Generate the QE input files and submit.sh (account m4735, email brent.hu@yale.edu). Apply the nat/ntyp and k-path INVARIANT checks per file, and call validate_qe_input on every input file.

STEP 6: Test ONE representative input file with run_espresso_workflow (calculations/000). Fix all errors before scaling to the rest; only generate/submit the remaining cases after a clean single-case run.

STEP 7: Read example/plot_bands.py, note what needs to change (apply the k-path source DECISION RULE), then write and run the new version.

Requirements:
- Organize the output files in a clear directory structure.
- Comment the code where nontrivial crystallographic operations are performed.

DONE CRITERIA:
- Execute generate_structures.py, setup_jobs.py, and plot_bands.py end-to-end without errors.
- Read at least one intermediate CIF and one Quantum ESPRESSO input file; confirm each INVARIANT holds for that specific case (not just "looks reasonable").
- The done summary must include: the Materials Project ID(s) used, an outline of how the new workflow works, any scientific assumptions made and a quick justification, and anything that might need further expert review.
"""

os.makedirs(config.WORK_DIR, exist_ok=True)
os.chdir(config.WORK_DIR)
print(f"Working directory: {os.getcwd()}")

final_state = run_agent(TASK, model, tokenizer, verbose=True, clear_dir=True, load_path=config.EXAMPLE_DIR)

print("\nFINAL STATE:\n", json.dumps(final_state.to_dict(), indent=2))
