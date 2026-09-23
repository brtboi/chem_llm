import os

# Must be set before any chem_llm import: config reads MATAGENT_WORK_DIR and
# retrieval.config reads DOC_INDEX_DIR at import time. retrieval_index/ase
# is the ASE + QE (pw.x, bands.x) index, kept separate from master's
# pymatgen index at retrieval_index/.
#
# test27: ase-migration branch, rutile TiO2, NO example directory -- the
# agent writes every script from scratch using ASE and search_docs.
os.environ["MATAGENT_WORK_DIR"] = "sandbox/test27"
os.environ["DOC_INDEX_DIR"] = "retrieval_index/ase"

import torch
import json
from chem_llm import config
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HOME"] = config.HF_HOME
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hide train_deepseudopot (out of scope for this run). Must happen before
# chem_llm.agent_core is imported, since SYSTEM_PROMPT_TEMPLATE serializes
# TOOLS once at import time.
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

COMPOUND = "TiO2"
GROUP = "rutile"
RELATIVISTIC = "scalar"

TASK = f"""
Build, from scratch, a three-script Python workflow for {GROUP} {COMPOUND} that generates perturbed crystal structures and Quantum ESPRESSO (QE) band-structure inputs, runs one representative DFT calculation, and plots its band structure. There is NO example or template code available to you -- do not look for any; write every script yourself.

Use ASE (Atomic Simulation Environment) as the interface layer for everything: read/write structures with ase.io, represent structures as ase.Atoms, write pw.x input files with ase.io.espresso.write_espresso_in (not hand-formatted strings), and read QE output with ase.io where possible. Do not import pymatgen in any script you write. Before writing code that depends on an ASE function signature or a QE / bands.x input variable, call search_docs (sources 'ase' and/or 'quantum_espresso') to confirm it.

Apply the INVARIANTS and DECISION RULES from the system prompt throughout -- they are not repeated here.

WORKFLOW (checkpoint after each step: state in 1-2 sentences what you verified before moving to the next step; do not batch verification to the end):

STEP 1: Fetch the base {GROUP} {COMPOUND} structure with generate_cif (space group P4_2/mnm, #136) into structures/. Record the Materials Project ID and the space group generate_cif reports.

STEP 2: Write and run generate_structures.py: read the base CIF with ase.io.read, then write 51 structures, structures/structure_000.cif through structure_050.cif, where 000 is the unperturbed base and 001-050 are perturbed copies made with chemically reasonable random atomic displacements at room temperature plus a small random lattice strain. Put the chosen displacement magnitude and its justification in a code comment. Assert the nat/ntyp INVARIANT per structure.

STEP 3: Fetch pseudopotentials for Ti and O with get_pseudopotential -- norm conserving, {RELATIVISTIC} relativistic, pbesol, stringent, UPF -- into template/. Note the recommended ecut the tool returns.

STEP 4: Before writing setup_jobs.py, write a note deciding every pw.x setting for the SCF and bands calculations (calculation, prefix, outdir, pseudo_dir, ecutwfc/ecutrho from the pseudopotential hints, occupations, conv_thr, SCF k-point mesh, nbnd for bands) and the bands.x post-processing namelist (&BANDS: prefix, outdir, filband). Use search_docs for anything you are unsure about.

STEP 5: Write and run setup_jobs.py. For each structure NNN, create calculations/NNN/ containing: the pseudopotential files (pw.x runs inside calculations/NNN, so pseudo_dir must resolve from there); pw.in (SCF) and bands.in (non-self-consistent bands along a k-path derived per the k-path INVARIANT and DECISION RULE), both written with write_espresso_in; bands_post.in (bands.x input with filband = 'NNN.bands.dat'); band_labels.dat (each high-symmetry label and its index in the bands k-point list, for plotting); and submit.sh (a SLURM script running pw.x SCF, pw.x bands, then bands.x; account m4735, email brent.hu@yale.edu). Use exactly the filenames pw.in, bands.in, bands_post.in -- run_espresso_workflow expects them. Call validate_qe_input on calculations/000/pw.in and calculations/000/bands.in.

STEP 6: Run run_espresso_workflow on calculations/000 ONLY. Fix all errors before moving on. After the SCF, compare the 'number of electrons' in calculations/000/pw.out against nbnd per the occupied-band INVARIANT; if nbnd is too small, fix setup_jobs.py, regenerate, and rerun calculations/000. Do not run DFT for any other structure.

STEP 7: Write and run plot_bands.py: read calculations/000/000.bands.dat.gnu (bands.x output; the first column is cumulative k-path distance, not a point index) and band_labels.dat, align energies to the VBM using the occupied-band count read from pw.out, place the high-symmetry ticks in the same x units as the band data, save calculations/000/bands_000.pdf, and print the band gap.

Requirements:
- Organize the output files in a clear directory structure.
- Comment the code where nontrivial crystallographic operations are performed.

DONE CRITERIA:
- generate_structures.py, setup_jobs.py, and plot_bands.py all run end-to-end with empty stderr, and none of them import pymatgen.
- run_espresso_workflow returned success: true for calculations/000.
- Read at least one generated CIF and calculations/000/pw.in; confirm each INVARIANT holds for that specific case (not just "looks reasonable").
- The done summary must include: the Materials Project ID, an outline of the workflow, which ASE functions you used for structure and QE I/O, the band gap you obtained, any scientific assumptions with a quick justification, and anything that needs expert review.
"""

os.makedirs(config.WORK_DIR, exist_ok=True)
os.chdir(config.WORK_DIR)
print(f"Working directory: {os.getcwd()}")

final_state = run_agent(
    TASK, model, tokenizer, verbose=True,
    clear_dir=True, load_path=None,  # no example directory
    max_steps=90,  # from-scratch runs need more steps than example-adapting ones (44)
)

print("\nFINAL STATE:\n", json.dumps(final_state.to_dict(), indent=2))
