import os

# Must be set before chem_llm.config is imported (it reads this env var at
# import time to compute WORK_DIR). This is the "pure ASE" experiment: the
# ase-migration branch already replaced pymatgen with ASE for structure
# generation, but test23 showed the agent still ran Quantum ESPRESSO through
# the custom run_espresso_workflow tool (a subprocess wrapper), never
# touching ASE's own Espresso calculator interface. This run removes that
# tool entirely to see whether/how the agent can drive QE through ASE
# instead. Kept in its own sandbox/test25 directory.
os.environ["MATAGENT_WORK_DIR"] = "sandbox/test25"

import torch
import json
from chem_llm import config
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HOME"] = config.HF_HOME
from transformers import AutoModelForCausalLM, AutoTokenizer

# Remove run_espresso_workflow from the tool list/dispatch table BEFORE
# chem_llm.agent_core is imported -- agent_core.SYSTEM_PROMPT_TEMPLATE
# serializes chem_llm.tools.TOOLS into the system prompt once, at import
# time, so the tool must already be gone from that list by then for the
# model to never see it as an available tool. Deleting from TOOL_DISPATCH
# too means even a stray/hallucinated call to it fails cleanly ("Unknown
# tool") instead of silently succeeding.
import chem_llm.tools as tools_module
tools_module.TOOLS[:] = [t for t in tools_module.TOOLS if t["name"] != "run_espresso_workflow"]
del tools_module.TOOL_DISPATCH["run_espresso_workflow"]

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
Write three new Python scripts based off of the pre-existing generate_structures.py, setup_jobs.py, and plot_bands.py, that generate perturbed {COMPOUND} crystal structures using ASE (Atomic Simulation Environment) and input files for Quantum ESPRESSO. The existing pipeline interpolates between cubic and orthorhombic CsPbBr3; this new workflow is a single {GROUP} {COMPOUND} phase (no phase interpolation).

Apply the INVARIANTS and DECISION RULES from the system prompt throughout this task -- they are not repeated here.

THIS RUN HAS AN EXTRA CONSTRAINT beyond the usual workflow: use ASE for EVERYTHING possible, not just structure generation. Concretely:
- Do not import or use pymatgen anywhere in the scripts you write (generate_cif already returns/writes a CIF; read it back with ase.io.read).
- The run_espresso_workflow tool has been deliberately REMOVED for this run -- it does not appear in your tool list, and calling it will fail. You must run the actual Quantum ESPRESSO calculations yourself, preferring ASE's own calculator interface (ase.calculators.espresso.Espresso / ase.calculators.espresso.EspressoProfile) over a bare subprocess call, wherever ASE actually has native support for the step. Use search_docs on calculators/espresso.html and calculators/calculators.html before writing this code.
- Two environment facts you will need (these are facts about this cluster, not something to derive from ASE docs): (1) the pw.x and bands.x binaries are only on PATH after `module load espresso/7.5-libxc-7.0.0-cpu` runs in the SAME shell invocation that then execs the binary -- Lmod modules are not active by default in a plain subprocess, so whatever command/profile you configure (for ASE's calculator, or for a raw subprocess call) must account for this, e.g. via a small wrapper shell script that loads the module then execs the binary with whatever arguments it was given. (2) this run happens directly on an already-allocated compute node with no srun/mpirun -- run pw.x/bands.x as a plain (non-MPI) process, and explicitly set OMP_NUM_THREADS=1 in the environment your command/profile runs in (the node has 128 cores; without pinning this, a serial pw.x/bands.x call oversubscribes to ~128 OpenMP threads for a tiny matrix and becomes extremely slow).
- ASE's Espresso calculator is designed around SCF-style calculations (get_potential_energy(), etc.) and has no equivalent for Quantum ESPRESSO's separate bands.x post-processing step. For that step ONLY, it is acceptable to invoke bands.x directly via a subprocess call inside your own script (this is not the removed tool -- it's your own code, following the same module-load/threading requirements above); prefer ASE's calculator for the SCF and bands pw.x steps.
- If, after genuinely investigating via search_docs and small test scripts, some specific step truly has no reasonable ASE-native path (state exactly why in a note, referencing what you found), it is acceptable to fall back to a direct subprocess call for that one step -- but do not skip trying the ASE-native route first, and do not fall back merely because the ASE-native route requires more code.

WORKFLOW (checkpoint after each step: state in 1-2 sentences what you verified before moving to the next step; do not batch verification to the end):

STEP 0: Read example/generate_structures.py, example/setup_jobs.py, and example/plot_bands.py in full before writing anything. Note per file what each constant/function does, and what is CsPbBr3-specific (e.g. the cubic/orthorhombic phase interpolation) that must be removed/replaced for {COMPOUND}.

STEP 1: Obtain the base {GROUP} {COMPOUND} structure via generate_cif. Record the Materials Project ID used.

STEP 2: Generate 50 additional perturbed structures with ASE (51 total). The perturbations should be chemically reasonable random displacements of each atom at room temperature. Apply the nat/ntyp INVARIANT check immediately per structure.

STEP 3: Get pseudopotentials via get_pseudopotential -- norm conserving, {RELATIVISTIC} relativistic, pbesol, stringent, UPF -- applying the pseudopotential relativistic-treatment DECISION RULE.

STEP 4: Before writing any setup_jobs code, write a note walking through every option in the Quantum ESPRESSO input file and what its value should be for this workflow, AND a separate note walking through how you plan to actually RUN pw.x/bands.x given the constraints above (ASE calculator configuration or documented fallback). Use search_docs for anything you are unsure about.

STEP 5: Generate the QE input files and submit.sh (account m4735, email brent.hu@yale.edu) -- via ASE's Espresso calculator/IO functions, not hand-written string formatting, unless you have a specific documented reason ASE cannot express something QE needs. Apply the nat/ntyp and k-path INVARIANT checks per file, and call validate_qe_input on every input file.

STEP 6: Test ONE representative case (calculations/000) by actually running SCF + bands + bands-post yourself (no run_espresso_workflow tool). Fix all errors before scaling to the rest; only generate/run the remaining cases after a clean single-case run.

STEP 7: Read example/plot_bands.py, note what needs to change (apply the k-path source and bands plot construction DECISION RULES), then write and run the new version.

Requirements:
- Organize the output files in a clear directory structure.
- Comment the code where nontrivial crystallographic operations are performed, and where you had to work around ASE not having a native QE feature.

DONE CRITERIA:
- Execute generate_structures.py, setup_jobs.py, and plot_bands.py end-to-end without errors.
- Read at least one intermediate CIF and one Quantum ESPRESSO input file; confirm each INVARIANT holds for that specific case (not just "looks reasonable").
- The done summary must include: the Materials Project ID(s) used, an outline of how the new workflow works, EXACTLY which steps used ASE's native QE interface vs. a documented fallback and why, any other scientific assumptions made and a quick justification, and anything that might need further expert review.
"""

os.makedirs(config.WORK_DIR, exist_ok=True)
os.chdir(config.WORK_DIR)
print(f"Working directory: {os.getcwd()}")

final_state = run_agent(
    TASK, model, tokenizer, verbose=True, clear_dir=True, load_path=config.EXAMPLE_DIR,
    max_steps=60,  # higher than the default 44: this run has genuinely more
                   # plumbing to work out (no run_espresso_workflow shortcut)
)

print("\nFINAL STATE:\n", json.dumps(final_state.to_dict(), indent=2))
