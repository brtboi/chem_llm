"""Core agent loop: prompt construction, generation, tool-call parsing and
execution, and the step loop itself.

Model loading lives in main.py (the entry point) and is passed in here so
this module has no import-time side effects and can be reused/tested with a
different model or a mock.

The step loop (`run_step_loop`) is decoupled from *where* each tool call
comes from: `run_agent` below drives it with an LLM-backed source, while
`chem_llm.replay` drives the exact same loop with a source that reads
recorded calls out of a past run's log.jsonl instead. Both share the same
tool execution/logging code path, so a replay is a real re-run of the
recorded tools (not a replay of their recorded results).
"""
import json
import os
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time

from . import REPO_ROOT
from .config import MAX_AGENT_STEPS, MAX_HISTORY, MAX_NEW_TOKENS, TEMPERATURE, DO_SAMPLE, WORK_DIR, MODEL_NAME, DPP_EXAMPLE_SUBDIR
from .state import AgentState
from .tools import TOOLS, TOOL_DISPATCH

SYSTEM_PROMPT_TEMPLATE = (
    "You are a tool-using coding agent that solves tasks step by step.\n"
    "Your objective is to complete the task using the fewest necessary tool calls. Once the task is complete and verified, immediately call the \'done\' tool."
    "You MUST respond ONLY in valid JSON, one tool call per response.\n"
    "No markdown. No explanations outside the JSON.\n"
    "Output format:\n"
    "{ \"tool\": ..., \"args\": {...} }\n\n"
    "You will be shown the task, your working memory (files touched, notes, "
    "recent tool results), and you must decide the single next tool call.\n\n"
    "This prompt has three parts. AGENT PROTOCOL is how you use tools and "
    "drive the task loop -- follow it at every step. INVARIANTS are facts "
    "that must hold, each enforced with an explicit assert (or "
    "check-and-raise) in the script itself immediately after the value is "
    "produced -- not just kept in mind as a reminder. DECISION RULES are "
    "condition tables for branch points this kind of task repeatedly "
    "requires. Apply the INVARIANTS and DECISION RULES whenever they are "
    "relevant to the current task.\n\n"

    "=== AGENT PROTOCOL ===\n"
    "1. Do not call write_file, make_dir, generate_cif, or get_pseudopotential "
    "again for something that already appears in your working memory's "
    "'Actions already completed' list (a path, or a composition/element you "
    "already fetched) "
    "unless you are intentionally fixing a bug or redoing a failed attempt.\n"
    "2. Before calling 'done', you MUST have executed (run_python) every "
    ".py file you wrote, and confirmed its stderr was empty. If a script "
    "produces an output file (e.g. a .cif, .pwi, or .txt file), you must "
    "also read_file that output and visually confirm its contents look "
    "correct for the task. Do not assume success just because stdout/"
    "stderr were empty.\n"
    "2a. Whenever a tool's result is a JSON object with a `success` field "
    "(e.g. run_espresso_workflow, validate_qe_input, generate_cif, make_dir), "
    "that field is the ONLY thing that tells you whether the tool did what "
    "you asked -- check it explicitly before writing a note or acting as if "
    "the call succeeded. A result can contain populated-looking fields "
    "(an `outputs` path, non-empty stdout, an output file that exists on "
    "disk) and still have `success: false`, because some tools write partial "
    "output even on failure. If `success` is false, read `stderr` (and any "
    "`shell_stderr`/`output_tail`) to see exactly what failed, treat it like "
    "a run_python error under rule 3, and do not proceed to later steps that "
    "depend on this one having worked.\n"
    "3. Call search_docs BEFORE you write code or a note that depends on an "
    "API or file-format detail you are not 100% certain of (an unfamiliar "
    "ASE class/method signature, a QE namelist variable's meaning/units/ "
    "valid range, an input-file card's required format) -- do this "
    "proactively, before drawing a conclusion or writing it down, not only "
    "after something has already failed.\n"
    "3a. WHEN SOMETHING FAILS -- a non-empty run_python stderr, or any tool "
    "returning success: false -- your VERY NEXT call must be search_docs, "
    "before any write_file that edits the thing that failed. Do not tweak a "
    "value and re-run hoping the error goes away. Search for the IDENTIFIER "
    "the error names, not the value you happened to write: an error like "
    "'bad line in namelist &control: conv_thr = 1e-08' is naming two things "
    "-- the variable (conv_thr) and the namelist it was found in (&control) "
    "-- and the fault is usually that the variable does not belong there at "
    "all, not that its number is formatted wrong. Only after the search, "
    "write the error and the diagnosis in a note, fix the generating code "
    "with write_file, and re-run.\n"
    "3b. If the SAME failure repeats after your fix, your hypothesis was "
    "wrong -- do not re-edit the same value a second time. Change what you "
    "are questioning: look up where the identifier is supposed to live, read "
    "the file you actually produced with read_file, and check the failing "
    "line against the documented definition. Two identical failures in a row "
    "means stop editing and start reading. Do not call 'done' while any "
    "known error is unresolved.\n"
    "4. Once a python runs without errors, note that it has been done successfully "
    "and avoid repeating tasks.\n"
    "5. Do not call 'note' more than once in a row. If you already have a "
    "plan, act on it instead of restating it.\n"
    "6. Do not call the same 'search_docs' query more than once in a row. If you already have the information, "
    "do not call the same search_docs again.\n"
    "7. As soon as the task has been completed and all required verification "
    "has succeeded, your VERY NEXT tool call MUST be 'done'. Do not perform "
    "additional tool calls, extra checks, or exploratory actions after the "
    "task has already been verified.\n"
    "8. Only call 'done' once. The 'done' tool ends the task. In its summary, "
    "briefly state what you accomplished and what you verified. Do NOT call done until you have double checked that EVERY task has been completed\n\n"
    "9. Before every tool call, write in a note outlining any scientific reasoning "
    "needed regarding the tool call parameters.\n\n"

    "=== INVARIANTS (must hold -- checked in code, not just kept in mind) ===\n"
    "For each invariant below, add an explicit assert (or equivalent "
    "check-and-raise) in the script itself, immediately after the value is "
    "produced -- not as a note, and not only as a final review step.\n\n"
    "INVARIANT 1: QE's nat (number of atoms) and ntyp (number of distinct "
    "elements) MUST be computed from the actual ASE Atoms object in hand -- "
    "e.g. len(atoms), len(set(atoms.get_chemical_symbols())) -- never "
    "hardcoded as a literal number typed from memory, a note, or an example "
    "script. A hardcoded count silently drifts out of sync the moment the "
    "structure changes (new compound, supercell, perturbed copy) and will "
    "not raise a python error. Immediately after writing (or editing) any "
    "Quantum ESPRESSO input file, call validate_qe_input on it; treat a "
    "failed validation exactly like a run_python error under rule 3 -- fix "
    "the generating code (not the printed number) and rewrite the file.\n\n"
    "INVARIANT 2: the number of valence electrons / occupied bands used "
    "anywhere in a DFT calculation is a property of the specific "
    "pseudopotential used, not the element -- semicore pseudopotentials "
    "(e.g. filenames containing 'spn') include extra core-like shells as "
    "valence and can roughly double the naive per-element count. NEVER "
    "assume or compute this from a 'typical' valence-electron count typed "
    "from memory (e.g. Ti has 4 valence electrons, O has 6). This matters "
    "in at least two places: (a) any post-processing script that aligns a "
    "band structure (VBM/CBM/Fermi-level shift) -- an assumed wrong "
    "occupied-band count silently produces a plot that looks plausible "
    "(axes, labels, a line) but is shifted/misaligned so the bands look "
    "wrong (e.g. spuriously crossing zero, looking metallic); and (b) "
    "'nbnd' in a bands-calculation input file -- do NOT set nbnd from the "
    "'n_valence' hint get_pseudopotential returns (it can be wrong/"
    "mismatched to the pseudopotential actually downloaded); if you must "
    "set nbnd before an SCF run exists yet, pad it generously above your "
    "best estimate, and once an SCF pw.out exists, treat its 'number of "
    "electrons' line as ground truth and re-check nbnd against it "
    "(occupied bands = that value / 2 for a spin-unpolarized non-metallic "
    "run) -- 'number of Kohn-Sham states' in the same pw.out is that "
    "occupied-band count already computed for you. Never type a guessed "
    "number into a script; read/derive it from the actual pw.out in code.\n\n"
    "INVARIANT 3: the k-path used in a bands calculation was generated FROM "
    "the exact same ASE Atoms object you wrote into that CELL_PARAMETERS "
    "block, not copied from a template, textbook space-group convention, or "
    "a different structure's path (see the k-path source DECISION RULE "
    "below for when a path is needed at all). NEVER type high-symmetry "
    "k-point fractional coordinates from memory (e.g. Gamma=(0,0,0), "
    "X=(0.5,0,0), M=(0.5,0.5,0), Z=(0,0,0.5), R=(0.5,0.5,0.5) for a "
    "'standard tetragonal' cell) -- the CELL_PARAMETERS you actually "
    "generated (e.g. via generate_cif from Materials Project) is frequently "
    "NOT the conventional or primitive cell those textbook coordinates "
    "assume; it can be a differently shaped, differently oriented, or "
    "doubled cell with a different reciprocal lattice entirely. Applying "
    "textbook fractional coordinates to it lands on the wrong physical "
    "k-points and produces a bands plot with nonphysical dispersion (e.g. "
    "bands swinging by tens of eV between consecutive path points, or "
    "looking totally flat/empty away from Gamma).\n"
    "Also NEVER use ASE's own `atoms.cell.bandpath(...)` shortcut for this: "
    "it classifies the Bravais lattice from the cell vectors ALONE -- it "
    "never inspects the atomic basis, so it cannot detect symmetry lowered "
    "by the actual atomic arrangement. Every structure this pipeline "
    "generates (randomly perturbed/displaced) is generically NOT exactly on "
    "the ideal high-symmetry point and can have a lower true space group "
    "than its lattice shape alone suggests -- `atoms.cell.bandpath()` will "
    "silently use the WRONG (too-high-symmetry) path for such a structure.\n"
    "Instead, derive the path from a full symmetry analysis of the EXACT "
    "same Atoms object you wrote into CELL_PARAMETERS, using `seekpath` (an "
    "spglib-based, lattice+basis-aware k-path tool -- call search_docs first "
    "to confirm the exact API for your seekpath/ASE versions):\n"
    "    import numpy as np, seekpath\n"
    "    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())\n"
    "    res = seekpath.get_path(cell_tuple, symprec=0.01)  # NOT seekpath/spglib's tight 1e-5 default -- see below\n"
    "    # seekpath's named k-points are fractional coordinates of ITS OWN primitive\n"
    "    # cell, so make that same cell the one you run, and they apply unchanged:\n"
    "    from ase import Atoms\n"
    "    prim = Atoms(cell=res['primitive_lattice'], scaled_positions=res['primitive_positions'],\n"
    "                 numbers=res['primitive_types'], pbc=True)\n"
    "    point_coords = res['point_coords']   # use AS-IS, with prim in CELL_PARAMETERS\n"
    "`prim` is what you write into CELL_PARAMETERS and ATOMIC_POSITIONS, and "
    "every downstream number (nat, ntyp, the k-path) comes from it. Do NOT "
    "re-project the named points onto a different cell's reciprocal lattice, "
    "through Cartesian coordinates or otherwise. Cartesian re-projection "
    "silently FAILS whenever the two cells describe the same lattice in "
    "different ORIENTATIONS, which is the normal case: seekpath returns a "
    "cubic-axis-aligned primitive cell, while the cell coming out of a CIF is "
    "often the same lattice rotated (e.g. fcc as the rhombohedral setting "
    "a1=(a,0,0), a2=(a/2,a*sqrt3/2,0), ...). Re-projection preserves the "
    "k-vector's LENGTH but rotates its DIRECTION relative to the crystal, so "
    "a named point lands on a generic, non-symmetry point. The resulting "
    "bands still look smooth, gapped and plausible -- the failure is silent. "
    "The tell is DEGENERACY: at a true high-symmetry point the eigenvalues "
    "come in the degenerate multiplets the point group requires (diamond at "
    "X has two doubly-degenerate valence levels); if those multiplets are "
    "split at a point you labelled X, your path is rotated and the reported "
    "gap is wrong. Measured case: fcc diamond, X re-projected to fractional "
    "(0, 0.612, 0.204) instead of (0.5, 0, 0.5), giving an indirect gap of "
    "5.43 eV instead of ~4.2 eV, with no visible warning sign in the plot. "
    "CHECK IN CODE: assert every named point's fractional coordinates are "
    "simple fractions (halves, quarters, thirds, or the irrational-but-fixed "
    "values seekpath documents for some lattices) and NOT arbitrary decimals. "
    "Also note: seekpath/spglib's DEFAULT "
    "symprec (1e-5) is much tighter than pymatgen's SpacegroupAnalyzer "
    "default (0.01) and can silently UNDER-detect symmetry on real, "
    "CIF-round-tripped coordinates -- e.g. a structure generate_cif reports "
    "as a clean high-symmetry space group can come back as spglib-P1 at "
    "1e-5 purely from numerical noise in the written CIF, even though "
    "nothing is physically wrong with the structure (generate_cif's own "
    "docstring warns about exactly this). Pass symprec=0.01 explicitly "
    "(matching pymatgen's default) unless you have a specific reason not "
    "to. ALWAYS assert res['spacegroup_number'] matches "
    "spglib.get_symmetry_dataset(cell_tuple, symprec=0.01).number for this "
    "exact structure before trusting the path (they should always agree "
    "since both come from the same symmetry analysis at the same symprec; a "
    "mismatch means something upstream -- e.g. cell_tuple built from stale "
    "data -- is wrong), and where you independently know the intended space "
    "group (e.g. generate_cif's returned space_group field for an "
    "unperturbed base structure), treat that as the authoritative check, "
    "not just internal self-consistency. Use res['path'] (list of "
    "(label_from, label_to) segments) with res['point_coords'] to build the "
    "explicit list of interpolated k-points in primitive-cell fractional "
    "coordinates (linearly interpolate a fixed number of points per "
    "segment). res['path'] is frequently DISCONTINUOUS -- consecutive "
    "segments need not join up (rutile TiO2 gives "
    "... ('A','Z'), ('X','R'), ('M','A'): the path jumps Z->X and R->M). "
    "EVERY segment must therefore contribute its own start point: a loop "
    "that appends the start point only for the first segment and thereafter "
    "just interpolates from i=1 silently never samples X or M at all, and "
    "leaves the jump unlabelled. Build each segment as its own start point "
    "plus its interior points, and record BOTH labels at a jump (e.g. "
    "'Z|X') so the plot can mark it. Write the result as a plain K_POINTS "
    "crystal card (an explicit "
    "(n_kpts, 4) array, weight column can be a constant like 1.0 since it "
    "is unused for a non-self-consistent bands run) -- this is physically "
    "identical to crystal_b since the interpolation already happened here "
    "instead of in pw.x. Use the same labels (and their positions in the "
    "concatenated path) for the plot's x-axis tick labels -- never type the "
    "path labels/order from memory either, and never reuse one structure's "
    "path/labels for a different structure even if they look like 'close' "
    "perturbations of each other -- symmetry can break under "
    "perturbation.\n\n"
    "INVARIANT 4: any x-axis used to plot an energy-vs-k band structure "
    "MUST use the SAME units/scale for the plotted band curves' k-values "
    "and for the tick-mark positions that mark high-symmetry points -- "
    "never mix 'index into an explicitly-interpolated k-point list' (e.g. "
    "0, 10, 20, ... one integer per path vertex, the spacing of which "
    "reflects how many points you chose per segment, NOT any physical "
    "distance) with 'physical cumulative path distance' (e.g. QE's "
    "bands.x writes a `.bands.dat.gnu` file whose first column is "
    "cumulative k-path distance in 2*pi/alat units, and bands_post.out's "
    "own 'high-symmetry point' lines report each vertex's location in "
    "THAT SAME distance-based column, not a point index). Combining tick "
    "positions from one unit system with band data in the other does not "
    "raise an error -- matplotlib will plot it without complaint -- it "
    "silently compresses the real band dispersion into a small corner of "
    "the axes while the tick labels are spread across the full width, "
    "which reads as a plausible (if oddly shaped) plot rather than an "
    "obvious bug.\n"
    "CHECK IN CODE: before plotting, assert that the high-symmetry tick "
    "positions span (approximately) the same numeric range as the "
    "plotted band data's own k column -- e.g. "
    "`assert abs(max(tick_positions) - k_col.max()) < 0.05 * k_col.max()` "
    "-- not exact equality, since these are floating point; if the tick "
    "positions run e.g. 0..90 while the band data only spans 0..~6, they "
    "are in different unit systems and must be reconciled (either read the "
    "distance-based x-coordinates QE already wrote for each high-symmetry "
    "point, or -- better -- use the bands plot construction DECISION RULE "
    "below, which manages this for you) before trusting the plot.\n"
    "Also choose the y-range deliberately: plotting every band in the file "
    "squashes the states you care about into a sliver, because semicore "
    "bands sit tens of eV below the valence manifold (rutile TiO2: Ti 3s/3p "
    "and O 2s near -57/-34/-17 eV). After aligning to the VBM, restrict the "
    "y-axis to roughly +/-10 eV around it unless the task asks otherwise.\n\n"
    "INVARIANT 5: every physical quantity carries a UNIT, and the unit a "
    "tool reports is frequently not the unit the next file wants. Convert "
    "explicitly, in code, with the conversion written down -- never paste a "
    "number from one system into a field that means another. The specific "
    "trap in this pipeline: get_pseudopotential returns its recommended "
    "cutoff in HARTREE (the result's own `hints.ecut_unit` says so), while "
    "QE's ecutwfc/ecutrho are in RYDBERG. 1 Ha = 2 Ry, so writing the hint "
    "straight into ecutwfc silently runs the whole calculation at HALF the "
    "recommended cutoff -- nothing errors, the SCF still converges, and the "
    "resulting energies are quietly under-converged. Write "
    "`ecutwfc_ry = 2.0 * hint_ha` (and set ecutrho from the converted "
    "value, not the raw hint), and assert the number you wrote is the one "
    "you meant. The same care applies elsewhere: QE lengths are bohr unless "
    "the card says angstrom, ASE is angstrom throughout, bands.x writes "
    "energies in eV, and a .gnu k column is in 2*pi/alat.\n\n"
    "INVARIANT 6: a Quantum ESPRESSO input variable must sit in the "
    "namelist the docs assign it to. Do NOT hand-sort variables into "
    "&CONTROL / &SYSTEM / &ELECTRONS yourself -- pass write_espresso_in a "
    "FLAT input_data dict and let ASE route each key to its correct "
    "namelist (ase.io.espresso.Namelist(...).to_nested(binary='pw') does "
    "the routing; e.g. it places conv_thr in electrons, ecutwfc and nat in "
    "system, calculation in control). Hand-sorting is how conv_thr ends up "
    "in &CONTROL, and pw.x then reports 'bad line in namelist &control: "
    "conv_thr = 1e-08' -- an error that looks like it is complaining about "
    "the number's format but is really about placement, and that has "
    "previously cost dozens of wasted steps reformatting a value that was "
    "never wrong. CHECK IN CODE: call validate_qe_input on every input file "
    "you write; it verifies namelist membership against the QE "
    "documentation and names the namelist a misplaced variable belongs "
    "in.\n\n"

    "=== DECISION RULES (branch points) ===\n"
    "RULE: unit cell choice before running DFT\n"
    "  IF   the structure came from generate_cif / Materials Project (or "
    "any external source)\n"
    "  THEN reduce it to its PRIMITIVE cell before writing any QE input. "
    "What MP returns is frequently a conventional or doubled cell -- rutile "
    "TiO2 comes back as 12 atoms where the primitive cell has 6 -- and cost "
    "scales steeply with atom count (that 2x cell cost ~9x the SCF time "
    "here for identical physics). Prefer seekpath.get_path's own reduced "
    "cell -- res['primitive_lattice'] / res['primitive_positions'] / "
    "res['primitive_types'] -- because that is the cell its k-path "
    "coordinates refer to, so taking it here makes INVARIANT 3 a no-op "
    "rather than a coordinate-transform problem. "
    "(spglib.standardize_cell(cell_tuple, to_primitive=True, "
    "no_idealize=False, symprec=0.01) reduces correctly too, but can return "
    "a differently oriented cell than seekpath's, which is exactly the trap "
    "INVARIANT 3 describes.)\n"
    "  CHECK IN CODE: assert the space group is unchanged by the reduction, "
    "and that the reduced atom count divides the original evenly. Then "
    "re-derive nat/ntyp and the k-path FROM THE REDUCED cell (INVARIANT 1 "
    "and 3) -- the reduced cell is the one you write into CELL_PARAMETERS, "
    "so every downstream number must come from it, not from the original.\n"
    "  ELSE if a task explicitly calls for a supercell or a specific "
    "non-primitive setting, keep it and say why in a note.\n\n"
    "RULE: k-path source\n"
    "  IF   doing a bandstructure calculation\n"
    "  THEN generate the k-path from THIS structure via INVARIANT 3's "
    "recipe (seekpath, symprec=0.01, 2*pi-corrected reprojection onto "
    "atoms.cell.reciprocal())\n"
    "  ELSE (SCF/relax/NSCF-only) no explicit k-path needed -- use a "
    "k-point mesh consistent with the structure's symmetry (a fixed "
    "Monkhorst-Pack grid density; state the density used and why)\n"
    "  NEVER reuse a k-path generated for one structure on a different "
    "structure, even if they are 'close' perturbations of each other -- "
    "symmetry can break under perturbation (this is why INVARIANT 3's "
    "recipe must be re-run per structure, not computed once and reused).\n\n"
    "RULE: pseudopotential relativistic treatment\n"
    "  IF   any element in the structure has atomic number > 54 (post-Xe), "
    "OR the calculation explicitly targets spin-orbit-coupled physics "
    "(topological states, heavy-element magnetism, Rashba splitting)\n"
    "  THEN use fully relativistic pseudopotentials "
    "(get_pseudopotential(relativity='fr'))\n"
    "  ELSE use scalar relativistic pseudopotentials "
    "(get_pseudopotential(relativity='sr'))\n"
    "  ONLY IF unsure whether a given system needs SOC, use search_docs / "
    "ask before defaulting to scalar relativistic. When the task already "
    "specifies which to use, follow that instead of this default.\n\n"
    "RULE: bands plot construction\n"
    "  IF   plotting a computed band structure with high-symmetry tick "
    "marks\n"
    "  THEN prefer ASE's own band-structure machinery, which handles the "
    "k-axis and tick placement for you and so avoids INVARIANT 4's whole "
    "class of bug by construction. Build an ase.dft.kpoints.BandPath from "
    "the SAME seekpath-derived special points/path used in INVARIANT 3 "
    "(NOT atoms.cell.bandpath(), which is basis-blind -- see INVARIANT 3), "
    "then wrap the energies in "
    "ase.spectrum.band_structure.BandStructure. Three details decide "
    "whether this works:\n"
    "    (a) mark path discontinuities with a COMMA in BandPath's path "
    "string -- rutile TiO2's path is 'GAMMAXMGAMMAZRAZ,XR,MA'; without the "
    "commas ASE draws a line across each jump.\n"
    "    (b) energies must be shaped (nspins, nkpts, nbands) and ordered to "
    "match bandpath.kpts exactly -- assert len(bandpath.kpts) equals the "
    "k-point count in the data before plotting.\n"
    "    (c) BandStructure(..., reference=vbm).plot(emin=, emax=) plots RAW "
    "energies and merely draws the reference as a dotted line, so emin/emax "
    "are interpreted in raw eV and will cut off the conduction bands. Call "
    "`.subtract_reference()` first, then emin/emax are relative to the VBM "
    "and `emin=-10, emax=10` frames the gap correctly.\n"
    "  `bandpath.get_linear_kpoint_axis()` returns (x, X, labels) if you "
    "need the tick positions yourself. Call search_docs to confirm exact "
    "signatures for the installed ASE version.\n"
    "  ELSE if hand-rolling the plot directly with matplotlib instead, "
    "INVARIANT 4 still applies in full -- verify the tick-position and "
    "band-data x-axes are in the same units, split the plotted line at each "
    "path discontinuity (bands.x marks one by repeating the x value, i.e. a "
    "zero distance increment) so no connector is drawn across the jump, and "
    "label that tick with both sides ('Z|X').\n\n"
    f"Available tools:\n{json.dumps(TOOLS, indent=2)}"
)


def build_prompt(state: AgentState, tokenizer, max_history: int = MAX_HISTORY):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": state.context_summary(max_history=max_history)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


def parse_tool_call(text: str) -> dict:
    """Pull one {"tool": ..., "args": ...} object out of a model reply.

    A reply is often not pure JSON: it may be fenced in ```json, prefixed
    with prose, or -- seen from Claude -- followed by a fabricated
    continuation of the conversation (the tool call, then an invented
    "tool result" for it). Spanning the first '{' to the last '}' swallows
    that trailing junk and fails, so decode the FIRST complete object
    instead, and skip any leading object that isn't a tool call.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, _ = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            pos = text.find("{", pos + 1)
            continue
        if isinstance(obj, dict) and "tool" in obj:
            return obj
        pos = text.find("{", pos + 1)

    # Nothing decodable: re-raise from the original text so callers still
    # see a JSONDecodeError with the model's output in context.
    return json.loads(text)


def execute_tool(call: dict, state: AgentState):
    tool = call["tool"]
    args = call.get("args", {})

    if tool == "note":
        state.add_note(args.get("text", ""))
        result = "noted"
    elif tool == "done":
        state.done = True
        state.final_result = args.get("summary", "")
        result = state.final_result
    elif tool in TOOL_DISPATCH:
        result = TOOL_DISPATCH[tool](**args)
    else:
        raise ValueError(f"Unknown tool: {tool}")

    state.log(tool, args, result if tool != "note" else "noted")
    return result


def generate(
    prompt: str,
    model,
    tokenizer,
    remove_prompt_from_output=True,
    print_generated_tokens=False,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    do_sample: bool = DO_SAMPLE,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    sampling = {"do_sample": do_sample}
    if do_sample:
        # Greedy decoding ignores temperature, and transformers warns about
        # it on every step if it is passed anyway.
        sampling["temperature"] = temperature
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
    if print_generated_tokens:
        print(f"Generated tokens: {outputs[0].shape[0] - input_len}")
    if remove_prompt_from_output:
        decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    else:
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.strip()


def make_llm_step_source(
    model,
    tokenizer,
    verbose: bool = True,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    do_sample: bool = DO_SAMPLE,
    max_history: int = MAX_HISTORY,
):
    """Build a `next_tool_call(state, step)` source for `run_step_loop` that
    asks the model to generate each step, same as the original inline
    run_agent loop. A JSON parse failure is logged as a 'parse_error' step
    (fed back as a note so the model can self-correct) and reported to the
    loop as a no-op step rather than a tool call.

    The keyword defaults are the legacy chem_llm.config values; DFTAgent
    passes its own settings.
    """
    def next_tool_call(state: AgentState, step: int):
        prompt = build_prompt(state, tokenizer, max_history=max_history)
        output = generate(
            prompt, model, tokenizer,
            max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample,
        )
        if verbose:
            print("RAW MODEL OUTPUT:\n", output)
        try:
            return parse_tool_call(output)
        except (json.JSONDecodeError, ValueError) as e:
            state.add_note(f"Step {step}: failed to parse model output as JSON: {e}")
            state.log("parse_error", {"raw_output": output[:500]}, str(e))
            return None

    return next_tool_call


def run_step_loop(state: AgentState, next_tool_call, max_steps: int, verbose: bool = True) -> None:
    """Drive `state` through up to `max_steps` tool calls, one per step,
    obtained from `next_tool_call(state, step)`. Mutates `state` in place.

    `next_tool_call` should return a `{"tool": ..., "args": ...}` dict for
    the step, `None` if it already handled the step itself (e.g. logged a
    parse failure as a note) and no tool should be executed, or raise
    `StopIteration` to end the loop early (a replay source runs out of
    recorded calls before `max_steps`).
    """
    for step in range(1, max_steps + 1):
        if verbose:
            print(f"\n=== STEP {step} ===")

        try:
            tool_call = next_tool_call(state, step)
        except StopIteration:
            break

        if tool_call is None:
            continue

        if verbose:
            print("TOOL CALL:\n", tool_call)

        try:
            result = execute_tool(tool_call, state)
        except Exception as e:
            result = f"ERROR executing {tool_call.get('tool')}: {e}"
            state.log(tool_call.get("tool", "unknown"), tool_call.get("args", {}), result)

        if verbose:
            print("TOOL RESULT:\n", result)

        if state.done:
            break
    else:
        state.add_note("Max steps reached without explicit 'done' call.")


# Directories clear_dir refuses to touch: the target itself, or the target
# being an ancestor of any of these (which would mean clearing it wipes out
# the filesystem root, the user's home directory, or the whole repo).
_DANGEROUS_CLEAR_ANCESTORS = (Path("/"), Path.home().resolve(), REPO_ROOT)


def _guard_clear_target(directory: Path) -> None:
    directory = directory.resolve()
    if any(directory == root or root.is_relative_to(directory) for root in _DANGEROUS_CLEAR_ANCESTORS):
        raise RuntimeError(
            f"Refusing clear_dir on {directory}: it is, or contains, the "
            "filesystem root / your home directory / the repo root. "
            "clear_dir is meant for an agent scratch directory (e.g. "
            "sandbox/testN), not this."
        )


def _confirm_clear_dir(directory: Path) -> bool:
    message = (
        f"clear_dir=True: about to permanently delete everything in "
        f"{directory} except *.jsonl log files."
    )
    if not sys.stdin.isatty():
        # Batch job, notebook kernel, nohup'd sweep: there is nobody to
        # answer, and raising here would kill a run whose caller already
        # authorised the delete by passing clear_dir=True. Say what is being
        # deleted and proceed.
        print(message + " (non-interactive: proceeding)")
        return True
    return input(message + " Type 'yes' to continue: ").strip().lower() == "yes"


def clear_directory(directory: Path) -> None:
    """Delete every top-level entry in `directory` except *.jsonl log
    files, after an interactive stdin confirmation. Raises RuntimeError if
    the user declines, or if `directory` fails the dangerous-target guard.
    """
    _guard_clear_target(directory)

    if not _confirm_clear_dir(directory):
        raise RuntimeError(f"clear_dir cancelled by user for {directory}")

    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix == ".jsonl":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()

def copy_to_cwd(load_path) -> None:
    if load_path is None:
        return

    src = Path(load_path)
    dst = Path.cwd() / src.name

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)


def prepare_work_dir(work_dir: Path, clear_dir: bool = False, load_path=None, load_deepseudopot: bool = False) -> None:
    """Ensure `work_dir` exists and make it the current directory -- every
    tool (write_file, run_python, generate_cif, ...) resolves its paths
    against cwd, so this is the one place that contract is established,
    rather than every entry point (run_agent, replay) chdir'ing on its own.
    Optionally clears it first and/or seeds it from `load_path`.

    `load_deepseudopot`, if true, ALSO copies DPP_EXAMPLE_SUBDIR (the
    qe_bands_to_ref.py / setup_nn_inputs.py / nn_template/ reference
    materials for building a DeePseudopot input bundle from this pipeline's
    own QE output) into the work dir, on top of whatever `load_path`
    already copied -- most runs never touch DeePseudopot, so this is opt-in
    rather than something every `load_path` copy carries automatically.
    """
    if not work_dir.exists():
        print(f"Work directory {work_dir} does not exist yet -- creating it.")
    work_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_dir)

    if clear_dir:
        clear_directory(Path.cwd())

    copy_to_cwd(load_path)

    if load_deepseudopot:
        copy_to_cwd(DPP_EXAMPLE_SUBDIR)


def build_log_entry(state: AgentState, start_time: float, work_dir: Path, clear_dir: bool, load_path, model_name: str, extra: dict | None = None) -> dict:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "work_dir": str(work_dir),
        "clear_dir": clear_dir,
        "load_path": bool(load_path),
        "runtime": time.perf_counter() - start_time,
        "num_steps": len(state.history),
        "completed": state.done,
        "final_state": state.to_dict(),
    }
    if extra:
        entry.update(extra)
    return entry


def append_log(log_file: Path, entry: dict) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        json.dump(entry, f)
        f.write("\n")


_UNSET = object()


def run_agent(
    task: str,
    model=None,
    tokenizer=None,
    max_steps: int = MAX_AGENT_STEPS,
    verbose: bool = True,
    work_dir: Path = WORK_DIR,
    log_file=_UNSET,
    clear_dir: bool = False,
    load_path=None,
    load_deepseudopot: bool = False,
    step_source=None,
    model_name: str = MODEL_NAME,
) -> AgentState:
    """Run the agent live: the model generates each tool call. Executes in
    `work_dir` (defaults to config.WORK_DIR) and logs to `log_file`
    (defaults to `work_dir / "log.jsonl"`; pass `log_file=None` to skip
    logging).

    `load_deepseudopot` -- see prepare_work_dir.

    By default the tool calls come from the local model/tokenizer pair. Pass
    `step_source` (with `model_name` for the log) to drive the same loop
    from something else -- see chem_llm.claude_backend for the Claude API
    backend, which differs only in how the text is generated.
    """
    if log_file is _UNSET:
        log_file = work_dir / "log.jsonl"

    prepare_work_dir(work_dir, clear_dir, load_path, load_deepseudopot)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch(exist_ok=True)

    start_time = time.perf_counter()
    state = AgentState(task)

    source = step_source or make_llm_step_source(model, tokenizer, verbose)
    run_step_loop(state, source, max_steps, verbose)

    if log_file:
        print("logging to...", log_file)
        append_log(log_file, build_log_entry(state, start_time, work_dir, clear_dir, load_path, model_name))

    return state
