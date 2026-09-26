"""The task prompt the agent is given, parameterised by compound.

Kept separate from the system prompt in agent_core: the system prompt holds
the invariants and decision rules that hold for every run, this holds the
per-job workflow. Only `compound` and `spacegroup` are required; everything
else has a default that suits a band-structure run.
"""
from __future__ import annotations

BAND_STRUCTURE_TASK = """
Build, from scratch, a three-script Python workflow for {spacegroup_name}{compound} that generates perturbed crystal structures and Quantum ESPRESSO (QE) band-structure inputs, runs one representative DFT calculation, and plots its band structure. There is NO example or template code available to you -- do not look for any; write every script yourself.

Use ASE (Atomic Simulation Environment) as the interface layer for everything: read/write structures with ase.io, represent structures as ase.Atoms, write pw.x input files with ase.io.espresso.write_espresso_in (not hand-formatted strings), and read QE output with ase.io where possible. Do not import pymatgen in any script you write. Before writing code that depends on an ASE function signature or a QE / bands.x input variable, call search_docs (sources 'ase' and/or 'quantum_espresso') to confirm it.

Apply the INVARIANTS and DECISION RULES from the system prompt throughout -- they are not repeated here.

WORKFLOW (checkpoint after each step: state in 1-2 sentences what you verified before moving to the next step; do not batch verification to the end):

STEP 1: Fetch the base {compound} structure with generate_cif (space group {spacegroup}{spacegroup_number_clause}) into structures/. Record the Materials Project ID and the space group generate_cif reports. Then reduce it to its primitive cell per the unit-cell-choice DECISION RULE before anything downstream uses it, and note the atom count before and after.

STEP 2: Write and run generate_structures.py: read the base CIF with ase.io.read, then write {n_structures} structures, structures/structure_000.cif through structure_{last_index:03d}.cif, where 000 is the unperturbed base and the rest are perturbed copies made with chemically reasonable random atomic displacements at room temperature plus a small random lattice strain. Put the chosen displacement magnitude and its justification in a code comment. Assert the nat/ntyp INVARIANT per structure.

STEP 3: Fetch pseudopotentials for every element in {compound} with get_pseudopotential -- norm conserving, {relativistic} relativistic, {functional}, stringent, UPF -- into template/. Record the recommended cutoff it returns AND the unit that comes with it; you will need it in STEP 4, converted per INVARIANT 5.

STEP 4: Before writing setup_jobs.py, write a note deciding every pw.x setting for the SCF and bands calculations (calculation, prefix, outdir, pseudo_dir, ecutwfc/ecutrho, occupations, conv_thr, SCF k-point mesh, nbnd for bands) and the bands.x post-processing namelist (&BANDS: prefix, outdir, filband). For every option carrying a unit, write down the unit and any conversion you applied (INVARIANT 5). Use search_docs for anything you are unsure about.

STEP 5: Write and run setup_jobs.py. For each structure NNN, create calculations/NNN/ containing: the pseudopotential files (pw.x runs inside calculations/NNN, so pseudo_dir must resolve from there); pw.in (SCF) and bands.in (non-self-consistent bands along a k-path derived per the k-path INVARIANT and DECISION RULE), both written with write_espresso_in; bands_post.in (bands.x input with filband = 'NNN.bands.dat'); and band_labels.dat (each high-symmetry label and its index in the bands k-point list, for plotting). Do not write a submit.sh or any other SLURM script -- the DFT is run for you by the run_espresso_workflow tool in STEP 6, not through the queue. Use exactly the filenames pw.in, bands.in, bands_post.in -- run_espresso_workflow expects them. Call validate_qe_input on calculations/000/pw.in and calculations/000/bands.in.

STEP 6: Run run_espresso_workflow on calculations/000 ONLY. Fix all errors before moving on. After the SCF, compare the 'number of electrons' in calculations/000/pw.out against nbnd per the occupied-band INVARIANT; if nbnd is too small, fix setup_jobs.py, regenerate, and rerun calculations/000. Do not run DFT for any other structure.

STEP 7: Write and run plot_bands.py for calculations/000: read the bands.x output (check which file it actually produced -- the .gnu file is one option, the primary filband file another -- and parse whichever is real and non-empty) together with band_labels.dat, align energies to the VBM using the occupied-band count read from pw.out, put the high-symmetry ticks in the same x units as the band data and assert it, label a tick with both sides ('Z|X') ONLY where the path is genuinely discontinuous (the previous segment's end differs from the next segment's start) and draw no line across such a jump, restrict the y-axis to about +/-10 eV around the VBM, save calculations/000/bands_000.pdf, and print the VBM, CBM and band gap.

Requirements:
- Organize the output files in a clear directory structure.
- Comment the code where nontrivial crystallographic operations are performed.

DONE CRITERIA:
- generate_structures.py, setup_jobs.py, and plot_bands.py all run end-to-end with empty stderr, and none of them import pymatgen.
- run_espresso_workflow returned success: true for calculations/000.
- Read at least one generated CIF and calculations/000/pw.in; confirm each INVARIANT holds for that specific case.
- The done summary must include: the Materials Project ID, the atom count before and after the primitive-cell reduction, an outline of the workflow, which ASE functions you used for structure and QE I/O, the band gap you obtained, any scientific assumptions with a quick justification, and anything that needs expert review.
"""


def build_task(
    compound: str,
    spacegroup: str,
    spacegroup_number: int | None = None,
    spacegroup_name: str = "",
    n_structures: int = 51,
    relativistic: str = "scalar",
    functional: str = "pbesol",
    template: str = BAND_STRUCTURE_TASK,
) -> str:
    """Render the workflow prompt for one compound.

    `spacegroup` is the Hermann-Mauguin symbol in Materials Project format
    (underscores for subscripts, dashes for inversion bars: 'P4_2/mnm',
    'Fd-3m'). `spacegroup_name` is an optional human label that reads
    naturally in front of the formula, e.g. 'rutile ' or 'cubic '.
    """
    if spacegroup_name and not spacegroup_name.endswith(" "):
        spacegroup_name += " "
    number_clause = f", #{spacegroup_number}" if spacegroup_number else ""
    return template.format(
        compound=compound,
        spacegroup=spacegroup,
        spacegroup_number_clause=number_clause,
        spacegroup_name=spacegroup_name,
        n_structures=n_structures,
        last_index=n_structures - 1,
        relativistic=relativistic,
        functional=functional,
    )
