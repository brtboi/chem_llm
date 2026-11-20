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

STEP 1: Fetch the base {compound} structure with generate_cif ({spacegroup_clause}) into structures/. Record the Materials Project ID and the space group generate_cif reports. Then reduce it to its primitive cell per the unit-cell-choice DECISION RULE before anything downstream uses it -- `kp_atoms(standard_kpath(atoms))` from chem_llm.dft.kpath IS that primitive cell (and the one the k-path refers to), so use it rather than calling seekpath yourself -- and note the atom count before and after.

STEP 2: Write and run generate_structures.py: read the base CIF with ase.io.read, then write {n_structures} structures, structures/structure_000.cif through structure_{last_index:03d}.cif, where 000 is the unperturbed base and the rest are perturbed copies: random atomic displacements representative of thermal motion, plus a small random lattice strain. Choose the temperature yourself: one at which THIS phase is actually stable (some phases exist only well above room temperature -- say which, and why). Give each species its own amplitude (lighter or more weakly bound atoms move more). Be explicit about units: crystallographic displacement parameters U are mean-square displacements in A^2, and the per-axis RMS displacement is sqrt(U) in A -- write which one each number is. Put the temperature, every species' amplitude and the justification in code comments. Assert the nat/ntyp INVARIANT per structure.

STEP 3: Fetch pseudopotentials for every element in {compound} with get_pseudopotential -- norm conserving, {functional}, stringent, UPF -- into template/. get_pseudopotential has no defaults: decide the relativity (scalar or fully relativistic) yourself from the pseudopotential DECISION RULE, and choose hint_level, writing both decisions and their reasons in a note before the first call. From each result record file.z_valence, file.relativistic, and the recommended cutoff AND the unit that comes with it; you will need them in STEP 4, converted per INVARIANT 5.

STEP 4: Before writing setup_jobs.py, write a note deciding every pw.x setting for the SCF and bands calculations (calculation, prefix, outdir, pseudo_dir, ecutwfc/ecutrho, occupations, conv_thr, SCF k-point mesh, nbnd for bands) and the bands.x post-processing namelist (&BANDS: prefix, outdir, filband). For every option carrying a unit, write down the unit and any conversion you applied (INVARIANT 5). Use search_docs for anything you are unsure about.

STEP 5: First call get_kpath on structures/structure_000.cif to see the standard path. Then write and run setup_jobs.py. For each structure NNN, create calculations/NNN/ containing: the pseudopotential files (pw.x runs inside calculations/NNN, so pseudo_dir must resolve from there); kpath.json from chem_llm.dft.kpath (kp = standard_kpath(atoms); kp_save(kp, ...)); pw.in (SCF) and bands.in (non-self-consistent bands), both written with write_espresso_in for the cell kp_atoms(kp), bands.in with kpts=kp_kpts(kp) -- see INVARIANT 3; and bands_post.in (bands.x input with filband = 'NNN.bands.dat' and lsym = .false. -- the symmetry analysis lsym enables only writes a band-representation file nothing here uses, and with spin-orbit it alone has taken 20 minutes). Do not write a labels file: kpath.json holds the labels and ticks. Do not write a submit.sh or any other SLURM script -- the DFT is run for you by the run_espresso_workflow tool in STEP 6, not through the queue. Use exactly the filenames pw.in, bands.in, bands_post.in -- run_espresso_workflow expects them. Call validate_qe_input on calculations/000/pw.in and calculations/000/bands.in.

STEP 6: Run run_espresso_workflow on calculations/000 ONLY. Fix all errors before moving on. After the SCF, compare the 'number of electrons' in calculations/000/pw.out against nbnd per the occupied-band INVARIANT; if nbnd is too small, fix setup_jobs.py, regenerate, and rerun calculations/000. Do not run DFT for any other structure.

STEP 7: Write and run plot_bands.py for calculations/000. Get the x-axis, energies, ticks and line segments from chem_llm.dft.kpath.plot_axis('calculations/000/kpath.json', 'calculations/000/000.bands.dat') -- it checks that the bands were computed on kpath.json's k-points. Align energies to the VBM -- band_edges(E, n_occupied) with n_occupied from chem_llm.dft.qe_output.read_pw_out (INVARIANT 2) -- or to the Fermi energy read the same way for a metal or semimetal. Plot each band as one line per segment (never across a discontinuity), draw a vertical line at every tick, label the ticks from plot_axis's ticks (they already read 'A|L' at discontinuities), set the x-limits to the full path, restrict the y-axis to about +/-10 eV, save calculations/000/bands_000.pdf, and print the VBM, CBM and band gap (or the Fermi energy).

Requirements:
- Organize the output files in a clear directory structure.
- Comment the code where nontrivial crystallographic operations are performed.

DONE CRITERIA:
- generate_structures.py, setup_jobs.py, and plot_bands.py all run end-to-end with returncode 0, and none of them import pymatgen.
- run_espresso_workflow returned success: true for calculations/000.
- Read at least one generated CIF and calculations/000/pw.in; confirm each INVARIANT holds for that specific case.
- The done summary must include: the Materials Project ID, the atom count before and after the primitive-cell reduction, an outline of the workflow, which ASE functions you used for structure and QE I/O, the band gap you obtained, any scientific assumptions with a quick justification, and anything that needs expert review.
"""


def build_task(
    compound: str,
    spacegroup: str | None = None,
    spacegroup_number: int | None = None,
    spacegroup_name: str = "",
    n_structures: int = 51,
    functional: str = "pbesol",
    template: str = BAND_STRUCTURE_TASK,
    workflow: dict | None = None,
) -> str:
    """Render the workflow prompt for one compound.

    `spacegroup` is the Hermann-Mauguin symbol in Materials Project format
    (underscores for subscripts, dashes for inversion bars: 'P4_2/mnm',
    'Fd-3m'). `spacegroup_name` is an optional human label that reads
    naturally in front of the formula, e.g. 'rutile ' or 'cubic '.
    `workflow` is Settings.workflow.resolved(): it switches in the k-mesh
    convergence step, Materials Project's k-path and pseudopotential match,
    and a fixed spin-orbit choice (apply_workflow). None = the agent decides.
    """
    if spacegroup_name and not spacegroup_name.endswith(" "):
        spacegroup_name += " "
    number_clause = f", #{spacegroup_number}" if spacegroup_number else ""
    if spacegroup:
        clause = f"space group {spacegroup}{number_clause}"
    elif spacegroup_number:
        clause = f"space group #{spacegroup_number}"
    else:
        name = spacegroup_name.strip()
        clause = (f"no space group was given: the phase is {name}. First call generate_cif with only "
                  f"composition and output_path -- it then writes nothing and lists Materials Project's "
                  f"structures for {compound} by space group. The list carries no phase names, and its order (by PBE "
                  f"energy) does not say which entry is {name}: the most stable entry is often not the common phase. "
                  f"Before choosing, write in a note which space group the {name} structure has and how you know, then "
                  f"the entry with that space group; if none matches, or you are not sure, say so in the note and in the "
                  f"done summary. Then call generate_cif again with its space group symbol and number")
    text = template.format(
        compound=compound,
        spacegroup=spacegroup,
        spacegroup_clause=clause,
        spacegroup_number_clause=number_clause,
        spacegroup_name=spacegroup_name,
        n_structures=n_structures,
        last_index=n_structures - 1,
        functional=functional,
    )
    return apply_workflow(text, workflow) if workflow and template is BAND_STRUCTURE_TASK else text


def _swap(text: str, old: str, new: str) -> str:
    if old not in text:  # the template changed: fail loudly rather than silently drop a rule
        raise ValueError(f"task template no longer contains {old[:60]!r}")
    return text.replace(old, new, 1)


def apply_workflow(text: str, wf: dict) -> str:
    """The band-structure task with the workflow choices written in."""
    mp_path = wf.get("kpath") == "materials_project"
    if mp_path:
        text = _swap(
            text,
            "Then reduce it to its primitive cell per the unit-cell-choice DECISION RULE before anything downstream "
            "uses it -- `kp_atoms(standard_kpath(atoms))` from chem_llm.dft.kpath IS that primitive cell (and the one "
            "the k-path refers to), so use it rather than calling seekpath yourself -- and note the atom count before "
            "and after.",
            "This run compares its bands with Materials Project's point by point, so generate_cif writes the primitive "
            "cell MP computed its band structure on (its structure_note says so): use that cell as is -- do not reduce, "
            "re-standardise or rotate it. Call get_kpath on the CIF with output_path structures/kpath.json: it writes "
            "MP's k-path for that exact cell. Note the atom count (before and after are the same here).")
        text = _swap(
            text,
            "STEP 5: First call get_kpath on structures/structure_000.cif to see the standard path. Then write and run "
            "setup_jobs.py.",
            "STEP 5: The k-path is Materials Project's, in structures/kpath.json (get_kpath, STEP 1). Write and run "
            "setup_jobs.py.")
        text = _swap(
            text,
            "kpath.json from chem_llm.dft.kpath (kp = standard_kpath(atoms); kp_save(kp, ...));",
            "kpath.json from chem_llm.dft.kpath -- for structure 000, kp = load_kpath('structures/kpath.json') "
            "(MP's path, which belongs to the unperturbed cell only), for the perturbed ones kp = "
            "standard_kpath(atoms) -- then kp_save(kp, ...);")
    if wf.get("pseudopotentials") == "match_mp":
        text = _swap(
            text,
            _step3_fetch(text),
            "STEP 3: Fetch pseudopotentials for every element with get_pseudopotential into template/, as close to "
            "Materials Project's calculation as Quantum ESPRESSO allows -- PAW (kind 'paw'), generator 'pbe', "
            "stringent, UPF; norm-conserving 'nc' only for an element with no PAW file, said in a note. MP ran VASP: "
            "generate_cif returned `materials_project_calculation` with its POTCARs, what each puts in valence, the "
            "valence electrons per cell, any Hubbard U, and its cutoff. QE cannot read POTCARs, so match them: compare "
            "each file.z_valence with the POTCAR's valence (semicore states), and the cell's total with "
            "valence_electrons_per_cell, and write in a note where they differ and by how much. If hubbard_U_eV is not "
            "empty, put the same U values on the same elements' d (or f) states in pw.in and bands.in. PAW needs "
            "ecutrho about 8 x ecutwfc, not the 4 x of norm-conserving.")
    soc = wf.get("spin_orbit", "agent")
    if soc in ("on", "off"):
        why = ("this run is compared with Materials Project, whose band structures have no spin-orbit"
               if soc == "off" and wf.get("mode") == "match_mp" else "the run's configuration fixes it")
        rule = ("OFF: scalar-relativistic ('sr') pseudopotentials, and no lspinorb / noncolin"
                if soc == "off" else
                "ON: fully-relativistic ('fr') pseudopotentials, with lspinorb = .true. and noncolin = .true.")
        text = _swap(text, "decide the relativity (scalar or fully relativistic) yourself from the pseudopotential "
                           "DECISION RULE,",
                     f"the relativity is NOT yours to decide here -- spin-orbit is {rule}, whatever the relativistic "
                     f"DECISION RULE would say ({why}) --")
    km = wf.get("kmesh") or {}
    if km.get("converge"):
        text = _swap(
            text,
            "STEP 6: Run run_espresso_workflow on calculations/000 ONLY.",
            "STEP 6: First converge the SCF k-mesh: call converge_kmesh on calculations/000. It runs pw.x SCFs on "
            "finer and finer meshes until the total energy per atom stops changing (threshold "
            f"{km.get('threshold_meV_per_atom')} meV/atom over {km.get('confirmations')} further refinements), then "
            "rewrites the K_POINTS of calculations/000/pw.in to the converged mesh. Put the `spacing` it returns into "
            "setup_jobs.py (kpts = scf_kmesh(prim, spacing=<that value>)), rerun setup_jobs.py so every structure "
            "gets that mesh, and read calculations/000/pw.in to confirm it has the returned mesh. Then run "
            "run_espresso_workflow on calculations/000 ONLY.")
        text = _swap(text, "- run_espresso_workflow returned success: true for calculations/000.",
                     "- converge_kmesh returned success: true, and calculations/000/pw.in uses the mesh it returned.\n"
                     "- run_espresso_workflow returned success: true for calculations/000.")
    return text


def _step3_fetch(text: str) -> str:
    """The template's STEP 3 opening sentence, as rendered (compound, functional)."""
    import re

    m = re.search(r"STEP 3: Fetch pseudopotentials for every element in \S+ with get_pseudopotential -- norm "
                  r"conserving, \S+, stringent, UPF -- into template/\.", text)
    return m.group(0) if m else "STEP 3: (template changed)"
