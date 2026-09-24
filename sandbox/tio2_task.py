"""Shared setup for the rutile TiO2 from-scratch benchmark.

Imported by both run_test27_pipeline.py (local Qwen) and
run_test_claude_api_pipeline.py (Claude API) so the two runs differ ONLY in
which model generates the tool calls: identical task text, identical tool
set, identical step budget, identical doc index, identical working-directory
setup (nothing preloaded -- no example/ directory, no template scripts).
"""

COMPOUND = "TiO2"
GROUP = "rutile"
RELATIVISTIC = "scalar"

# Same doc index for both runs: ASE + Quantum ESPRESSO (pw.x and bands.x).
DOC_INDEX_DIR = "retrieval_index/ase"

MAX_STEPS = 90

TASK = f"""
Build, from scratch, a three-script Python workflow for {GROUP} {COMPOUND} that generates perturbed crystal structures and Quantum ESPRESSO (QE) band-structure inputs, runs one representative DFT calculation, and plots its band structure. There is NO example or template code available to you -- do not look for any; write every script yourself.

Use ASE (Atomic Simulation Environment) as the interface layer for everything: read/write structures with ase.io, represent structures as ase.Atoms, write pw.x input files with ase.io.espresso.write_espresso_in (not hand-formatted strings), and read QE output with ase.io where possible. Do not import pymatgen in any script you write. Before writing code that depends on an ASE function signature or a QE / bands.x input variable, call search_docs (sources 'ase' and/or 'quantum_espresso') to confirm it.

Apply the INVARIANTS and DECISION RULES from the system prompt throughout -- they are not repeated here.

WORKFLOW (checkpoint after each step: state in 1-2 sentences what you verified before moving to the next step; do not batch verification to the end):

STEP 1: Fetch the base {GROUP} {COMPOUND} structure with generate_cif (space group P4_2/mnm, #136) into structures/. Record the Materials Project ID and the space group generate_cif reports. Then reduce it to its primitive cell per the unit-cell-choice DECISION RULE before anything downstream uses it, and note the atom count before and after.

STEP 2: Write and run generate_structures.py: read the base CIF with ase.io.read, then write 51 structures, structures/structure_000.cif through structure_050.cif, where 000 is the unperturbed base and 001-050 are perturbed copies made with chemically reasonable random atomic displacements at room temperature plus a small random lattice strain. Put the chosen displacement magnitude and its justification in a code comment. Assert the nat/ntyp INVARIANT per structure.

STEP 3: Fetch pseudopotentials for Ti and O with get_pseudopotential -- norm conserving, {RELATIVISTIC} relativistic, pbesol, stringent, UPF -- into template/. Record the recommended cutoff it returns AND the unit that comes with it; you will need it in STEP 4, converted per INVARIANT 5.

STEP 4: Before writing setup_jobs.py, write a note deciding every pw.x setting for the SCF and bands calculations (calculation, prefix, outdir, pseudo_dir, ecutwfc/ecutrho, occupations, conv_thr, SCF k-point mesh, nbnd for bands) and the bands.x post-processing namelist (&BANDS: prefix, outdir, filband). For every option carrying a unit, write down the unit and any conversion you applied (INVARIANT 5). Use search_docs for anything you are unsure about.

STEP 5: Write and run setup_jobs.py. For each structure NNN, create calculations/NNN/ containing: the pseudopotential files (pw.x runs inside calculations/NNN, so pseudo_dir must resolve from there); pw.in (SCF) and bands.in (non-self-consistent bands along a k-path derived per the k-path INVARIANT and DECISION RULE), both written with write_espresso_in; bands_post.in (bands.x input with filband = 'NNN.bands.dat'); and band_labels.dat (each high-symmetry label and its index in the bands k-point list, for plotting). Do not write a submit.sh or any other SLURM script -- the DFT is run for you by the run_espresso_workflow tool in STEP 6, not through the queue. Use exactly the filenames pw.in, bands.in, bands_post.in -- run_espresso_workflow expects them. Call validate_qe_input on calculations/000/pw.in and calculations/000/bands.in.

STEP 6: Run run_espresso_workflow on calculations/000 ONLY. Fix all errors before moving on. After the SCF, compare the 'number of electrons' in calculations/000/pw.out against nbnd per the occupied-band INVARIANT; if nbnd is too small, fix setup_jobs.py, regenerate, and rerun calculations/000. Do not run DFT for any other structure.

STEP 7: Write and run plot_bands.py: read calculations/000/000.bands.dat.gnu (bands.x output; the first column is cumulative k-path distance, not a point index) and band_labels.dat, align energies to the VBM using the occupied-band count read from pw.out, place the high-symmetry ticks in the same x units as the band data and assert it, handle any path discontinuities so no line is drawn across a jump and the tick shows both sides ('Z|X'), restrict the y-axis to about +/-10 eV around the VBM, save calculations/000/bands_000.pdf, and print the VBM, CBM and band gap.

Requirements:
- Organize the output files in a clear directory structure.
- Comment the code where nontrivial crystallographic operations are performed.

DONE CRITERIA:
- generate_structures.py, setup_jobs.py, and plot_bands.py all run end-to-end with empty stderr, and none of them import pymatgen.
- run_espresso_workflow returned success: true for calculations/000.
- Read at least one generated CIF and calculations/000/pw.in; confirm each INVARIANT holds for that specific case.
- The done summary must include: the Materials Project ID, the atom count before and after the primitive-cell reduction, an outline of the workflow, which ASE functions you used for structure and QE I/O, the band gap you obtained, any scientific assumptions with a quick justification, and anything that needs expert review.
"""


def hide_deepseudopot():
    """Drop train_deepseudopot from the tool list both runs see.

    Must run before chem_llm.agent_core is imported, since
    SYSTEM_PROMPT_TEMPLATE serializes TOOLS once at import time.
    """
    import chem_llm.tools as tools_module

    tools_module.TOOLS[:] = [t for t in tools_module.TOOLS if t["name"] != "train_deepseudopot"]
    tools_module.TOOL_DISPATCH.pop("train_deepseudopot", None)
