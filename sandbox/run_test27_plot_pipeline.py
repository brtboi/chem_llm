import os

# Continuation of sandbox/run_test27_pipeline.py, which was killed when its
# SLURM allocation ended at step 44/90 -- mid-STEP 7 (plot_bands.py). STEP 1-6
# already succeeded and are on disk: the agent's own generate_structures.py /
# setup_jobs.py, 51 structures, 51 calculations/ dirs, and a complete, real
# QE run for calculations/000 (SCF + bands + bands.x, JOB DONE). This run
# finishes the last step only; clear_dir=False so none of that is destroyed.
os.environ["MATAGENT_WORK_DIR"] = "sandbox/test27"
os.environ["DOC_INDEX_DIR"] = "retrieval_index/ase"

import torch
import json
from chem_llm import config
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HOME"] = config.HF_HOME
from transformers import AutoModelForCausalLM, AutoTokenizer

import chem_llm.tools as tools_module
tools_module.TOOLS[:] = [t for t in tools_module.TOOLS if t["name"] != "train_deepseudopot"]
del tools_module.TOOL_DISPATCH["train_deepseudopot"]

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

TASK = """
A rutile TiO2 workflow already exists in this working directory and is complete except for its last step. Do NOT redo any earlier step: generate_structures.py and setup_jobs.py are written and were already run (structures/structure_000.cif ... structure_050.cif and calculations/000 ... calculations/050 exist), and the Quantum ESPRESSO run for calculations/000 already finished successfully (SCF + bands + bands.x; pw.out, bands_pw.out, bands_post.out and 000.bands.dat.gnu are all on disk). Do NOT call generate_cif, get_pseudopotential or run_espresso_workflow, and do not run DFT for any other structure.

Your only job is to finish plot_bands.py so it works, then run it.

plot_bands.py exists but currently FAILS when run. Run it first to see the actual traceback. Do not assume you know the cause: before trusting any parsing code in it, read the real files it parses (read_file on calculations/000/000.bands.dat.gnu -- look carefully at how many columns each line actually has and how the blocks are separated -- and calculations/000/band_labels.dat and calculations/000/pw.out) and make the parsing match what is really there. The number of k-points and the number of bands both have to come out right; sanity-check them against 'number of k points' in calculations/000/bands_pw.out and 'nbnd' in calculations/000/bands.in.

The finished script must, for calculations/000:
- read the band energies from 000.bands.dat.gnu (its first column is cumulative k-path distance, NOT a point index),
- read the occupied-band count from pw.out rather than assuming it from per-element valences (INVARIANT 2), and align energies so the valence band maximum sits at 0 eV,
- place the high-symmetry tick marks from band_labels.dat at x positions in the SAME units as the plotted band data (INVARIANT 4) -- assert this in code before plotting,
- save calculations/000/bands_000.pdf, and print the VBM, CBM and band gap in eV.

Apply the INVARIANTS and DECISION RULES from the system prompt. Use ASE and search_docs as needed; do not import pymatgen.

DONE CRITERIA:
- plot_bands.py runs with empty stderr and writes calculations/000/bands_000.pdf.
- The INVARIANT 4 x-axis-unit assertion is in the script and passes.
- The done summary states the VBM, CBM and band gap you obtained, what was actually wrong with the original plot_bands.py, and how you fixed it.
"""

os.makedirs(config.WORK_DIR, exist_ok=True)
os.chdir(config.WORK_DIR)
print(f"Working directory: {os.getcwd()}")

final_state = run_agent(
    TASK, model, tokenizer, verbose=True,
    clear_dir=False, load_path=None,
    max_steps=30,
)

print("\nFINAL STATE:\n", json.dumps(final_state.to_dict(), indent=2))
