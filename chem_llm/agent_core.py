"""Core agent loop: prompt construction, generation, tool-call parsing and
execution, and the step loop itself.

Model loading lives in main.py (the entry point) and is passed in here so
this module has no import-time side effects and can be reused/tested with a
different model or a mock.

The step loop (`run_step_loop`, a LangGraph graph -- see chem_llm.graph) is
decoupled from *where* each tool call comes from: `run_agent` below drives
it with an LLM-backed source, while `chem_llm.replay` drives the exact same loop with a source that reads
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
    ".py file you wrote, and confirmed it exited with returncode 0 (stderr "
    "may still hold harmless warnings; a traceback means it failed). If a script "
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
    "3a. WHEN SOMETHING FAILS -- a run_python with a nonzero returncode, or any tool "
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
    "5. Do not call 'note' more than once in a row (the harness refuses a "
    "second consecutive note). If you already have a plan, act on it instead "
    "of restating it.\n"
    "6. Do not repeat your previous 'search_docs' query (the harness refuses an "
    "identical search; results never change). If you already have the information, "
    "use it.\n"
    "7. As soon as the task has been completed and all required verification "
    "has succeeded, your VERY NEXT tool call MUST be 'done'. Do not perform "
    "additional tool calls, extra checks, or exploratory actions after the "
    "task has already been verified.\n"
    "8. Only call 'done' once. The 'done' tool ends the task. In its summary, "
    "briefly state what you accomplished and what you verified. Do NOT call done until you have double checked that EVERY task has been completed\n\n"
    "9. Write a note when it carries information: a scientific decision and "
    "its reasoning (relativity, cutoffs, k-mesh, occupations, temperature, "
    "displacements), a diagnosis after a failure, or a checkpoint at the end of "
    "a workflow step. Do not write a note before routine actions (reading a "
    "file, running a script you just wrote, validating an input) -- just do "
    "them.\n"
    "10. Keep what you pull into context small. Read large numeric outputs "
    "(bands.dat, .gnu files, a full pw.out) by line range with read_file's "
    "start_line/num_lines, never whole: read a few lines to learn the format, "
    "and parse the numbers in a script. Notes are for "
    "conclusions and decisions, not for transcribing data -- any value a script "
    "needs must be parsed by that script from the file (INVARIANT 2).\n"
    "11. Quantum ESPRESSO runs ONLY through the run_espresso_workflow tool, "
    "which already knows where pw.x / bands.x are and how to launch them on "
    "this machine. Never search the filesystem for QE binaries, load modules, "
    "or call srun / mpirun / pw.x from a script (a `find /global` for pw.x "
    "runs for many minutes on this filesystem). If run_espresso_workflow "
    "fails, the cause is in its result -- read stderr, output_tail and the "
    ".out file it names, and fix your input files.\n\n"

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
    "'nbnd' in a bands-calculation input file. Before an SCF run exists, "
    "estimate the electron count in code by summing z_valence over every "
    "atom -- the value get_pseudopotential reports as file.z_valence, which "
    "a script reads from the same file with `from chem_llm.tools.upf import "
    "read_upf_header; read_upf_header(path)['z_valence']` (do not write your "
    "own UPF parser). A pseudopotential's valence often includes semicore "
    "states, so it is usually MORE than the textbook valence. Pad nbnd above "
    "half of it; once an SCF pw.out exists, treat its 'number of "
    "electrons' line as ground truth and re-check nbnd against it "
    "(occupied bands = that value / 2 for a spin-unpolarized non-metallic "
    "run) -- 'number of Kohn-Sham states' in the same pw.out is that "
    "occupied-band count already computed for you. Never type a guessed "
    "number into a script; read/derive it from the actual pw.out in code. "
    "In a script, read pw.out ONLY through chem_llm.qe_output -- do not "
    "write your own regex for it (hand-written ones have repeatedly missed "
    "QE's actual line formats):\n"
    "    from chem_llm.qe_output import read_pw_out, band_edges\n"
    "    pw = read_pw_out('calculations/000/pw.out')\n"
    "    # pw['n_electrons'], pw['n_occupied'] (already 1 electron/band with spin-orbit),\n"
    "    # pw['fermi_energy_eV'] (smearing runs), pw['spin_orbit'], pw['total_magnetization']\n"
    "    edges = band_edges(E, pw['n_occupied'])   # -> edges['vbm'], edges['cbm'], edges['gap'] in eV\n\n"
    "INVARIANT 3: the k-path of a bands calculation, the cell it runs on, "
    "and the band plot's x-axis all come from ONE k-path, produced by "
    "chem_llm.kpath (the same code as the get_kpath tool) for THIS structure "
    "-- never typed from memory, a textbook table, or another structure. In "
    "setup_jobs.py, for each structure:\n"
    "    from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save\n"
    "    kp = standard_kpath(atoms)          # seekpath's standard path, evenly spaced\n"
    "    prim = kp_atoms(kp)                 # THE cell for pw.in AND bands.in\n"
    "    # bands.in: write_espresso_in(..., prim, ..., kpts=kp_kpts(kp))\n"
    "    kp_save(kp, f'calculations/{nnn}/kpath.json')\n"
    "Write prim -- not the CIF's cell -- into both inputs: the path's "
    "fractional k-points refer to prim's reciprocal basis, and a differently "
    "oriented cell silently puts them on the wrong k-points. Do not "
    "re-project, re-interpolate, re-label or re-order the k-points, and do "
    "not write a separate labels file: kpath.json holds the labels, tick "
    "positions (path discontinuities already joined as 'A|L') and the line "
    "segments. Recompute it per structure -- perturbed structures have lower "
    "symmetry and their own path. Call get_kpath on structure_000 first to "
    "see the path, and validate_qe_input on bands.in afterwards (it checks "
    "the k-points against kpath.json).\n\n"
    "INVARIANT 4: the band plot's x-axis, ticks and line segments come from "
    "chem_llm.kpath.plot_axis(kpath.json, NNN.bands.dat), which also refuses "
    "to run if bands.x computed a different k-point list than kpath.json "
    "(so a plot can never silently mismatch the calculation). Use its x (path "
    "length) for the band curves, draw each of its segments as a separate "
    "line -- segments are slice objects: `for s in ax['segments']: "
    "plt.plot(x[s], E[s, band])`, never x[start:end+1] -- (no line across a "
    "discontinuity), put a vertical line and a tick "
    "label at every tick -- ax['tick_x'] and ax['tick_labels'] are plain "
    "lists for exactly that (ax['ticks'] holds the same as dicts with keys "
    "x/label/indices; do not unpack them as pairs) -- and never build ticks "
    "from k-point indices or typed labels. Choose the y-range deliberately: semicore "
    "bands sit tens of eV below the valence manifold, so after aligning to "
    "the VBM (or the Fermi energy for a metal) restrict the y-axis to about "
    "+/-10 eV unless the task asks otherwise.\n\n"
    "INVARIANT 5: every physical quantity carries a UNIT, and the unit a "
    "tool reports is frequently not the unit the next file wants. Convert "
    "explicitly, in code, with the conversion written down -- never paste a "
    "number from one system into a field that means another. The specific "
    "trap in this pipeline: get_pseudopotential returns its recommended "
    "cutoff in HARTREE (the result's own `recommended_ecutwfc.unit` says so), while "
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
    "here for identical physics). The cell chem_llm.kpath.kp_atoms returns "
    "IS this primitive cell (seekpath's), and it is the one the k-path refers "
    "to -- use it.\n"
    "  CHECK IN CODE: assert the space group is unchanged by the reduction, "
    "and that the reduced atom count divides the original evenly. The space "
    "group of a cell is standard_kpath(atoms)['spacegroup'], e.g. 'P6_3mc "
    "(#186)' -- compare its number with the one generate_cif reported; do "
    "not call spglib or ase.spacegroup for this yourself. Then "
    "derive nat/ntyp and the k-path FROM THE REDUCED cell (INVARIANT 1 "
    "and 3) -- the reduced cell is the one you write into CELL_PARAMETERS, "
    "so every downstream number must come from it, not from the original.\n"
    "  ELSE if a task explicitly calls for a supercell or a specific "
    "non-primitive setting, keep it and say why in a note.\n\n"
    "RULE: k-path source\n"
    "  IF   doing a bandstructure calculation\n"
    "  THEN use chem_llm.kpath / get_kpath per INVARIANT 3 -- never a "
    "hand-built path\n"
    "  ELSE (SCF/relax/NSCF-only) no explicit k-path needed -- use a "
    "Monkhorst-Pack mesh consistent with the cell (state the density and why)\n"
    "  The SCF mesh (pw.in) is NEVER typed by hand: compute it in setup_jobs.py "
    "for the cell the SCF runs on, `kpts = scf_kmesh(prim, metal=...)` from "
    "chem_llm.kpath (even spacing along every reciprocal vector; metal=True "
    "for a metal or semimetal, which needs about twice the density), and pass "
    "it as write_espresso_in(..., kpts=kpts). A hand-picked mesh has silently "
    "under-sampled the short axis (rutile 4x4x2 instead of 6x6x9) and shifted "
    "the conduction bands by 0.1 eV with nothing failing.\n\n"
    "RULE: pseudopotential relativistic treatment\n"
    "  IF   any element in the structure has atomic number > 54 (post-Xe), "
    "OR the calculation explicitly targets spin-orbit-coupled physics "
    "(topological states, heavy-element magnetism, Rashba splitting)\n"
    "  THEN use fully relativistic pseudopotentials "
    "(get_pseudopotential(relativity='fr')) and run with spin-orbit: set "
    "noncolin=.true. and lspinorb=.true. in &SYSTEM (without them pw.x does not "
    "include spin-orbit), and remember each band then holds ONE electron, so "
    "the occupied-band count is the number of electrons, not half of it "
    "(INVARIANT 2)\n"
    "  ELSE use scalar relativistic pseudopotentials "
    "(get_pseudopotential(relativity='sr'))\n"
    "  There is no default: decide from this rule, write the decision and the "
    "atomic numbers that drove it in a note before fetching, and check the "
    "returned file.relativistic matches what you asked for.\n\n"
    "RULE: occupations\n"
    "  IF   the material may be a metal or semimetal (generate_cif's "
    "band_gap_eV near 0, or e.g. graphite, a metal, a doped system)\n"
    "  THEN use smearing in pw.in (occupations='smearing', "
    "smearing='mv' or 'gaussian', degauss about 0.01 Ry) and align the "
    "band plot to the Fermi energy from pw.out instead of a VBM\n"
    "  ELSE occupations='fixed' is fine for a clear insulator or "
    "semiconductor\n\n"
    "RULE: magnetism\n"
    "  IF   generate_cif's magnetism reports ordering 'FM' or 'FiM' (or "
    "is_magnetic true with a nonzero moment) -- e.g. bcc Fe, Co, Ni\n"
    "  THEN run spin-polarized: nspin=2 and starting_magnetization(i)=0.5 for "
    "each magnetic species (the transition-metal / rare-earth ones) in BOTH "
    "pw.in and bands.in, with smearing (a ferromagnet is a metal); in "
    "bands_post.in set spin_component=1 (majority spin) and say in the plot "
    "title that it shows the majority spin. After the SCF, read "
    "total_magnetization from pw.out with read_pw_out and compare it per atom "
    "with generate_cif's moment_per_formula_unit_muB (per formula unit). "
    "Align to the Fermi energy.\n"
    "  IF   ordering is 'AFM': antiferromagnetic order needs a magnetic "
    "supercell -- say so in a note and in the done summary, and run "
    "non-spin-polarized unless the task asks for the supercell.\n"
    "  ELSE (ordering 'NM' / 'Unknown' with ~0 moment) run nspin=1.\n\n"
    "RULE: bands plot construction\n"
    "  Use chem_llm.kpath.plot_axis per INVARIANT 4 and chem_llm.qe_output per "
    "INVARIANT 2 with plain matplotlib:\n"
    "    ax = plot_axis(kpath_json, bands_dat); x, E = ax['x'], ax['energies']\n"
    "    pw = read_pw_out(pw_out)\n"
    "    zero = pw['fermi_energy_eV'] if metal else band_edges(E, pw['n_occupied'])['vbm']\n"
    "    for s in ax['segments']:\n"
    "        for b in range(E.shape[1]): plt.plot(x[s], E[s, b] - zero, 'k-', lw=1)\n"
    "    for t in ax['tick_x']: plt.axvline(t, color='gray', lw=0.5)\n"
    "    plt.xticks(ax['tick_x'], ax['tick_labels']); plt.xlim(x[0], x[-1]); plt.ylim(-10, 10)\n\n"
    f"Available tools:\n{json.dumps(TOOLS, indent=2)}"
)


def build_messages(state: AgentState, max_history: int = MAX_HISTORY, trim_history: bool = False) -> list:
    """The two-message prompt every step sends: the fixed system prompt, and
    the agent's working memory (AgentState.context_summary) as the user turn."""
    from langchain_core.messages import HumanMessage, SystemMessage

    return [
        SystemMessage(SYSTEM_PROMPT_TEMPLATE),
        HumanMessage(state.context_summary(max_history=max_history, trim=trim_history)),
    ]


def build_prompt(state: AgentState, tokenizer, max_history: int = MAX_HISTORY, trim_history: bool = False):
    """build_messages rendered through a tokenizer's chat template (what the
    local transformers backend generates from)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE},
        {"role": "user", "content": state.context_summary(max_history=max_history, trim=trim_history)},
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


# Entries that are not actions the model took, and so do not break a run of
# notes or separate two searches.
_NOT_ACTIONS = ("refused", "parse_error", "api_error")


def _refusal(tool: str, args: dict, state: AgentState) -> str | None:
    """Harness-level guards against loops the prompt rules alone did not
    stop: back-to-back notes (a CsPbBr3 run spent 45 of 80 steps on notes,
    13 in a row), a repeated doc search, and rewriting a file with identical
    content (a ZnO run wrote the same setup_jobs.py 5 times, ~25 steps, while
    its notes described a fix it never applied). Returns the reason a call
    is refused, or None to let it run."""
    actions = [e for e in state.history if e["tool"] not in _NOT_ACTIONS]
    if tool == "note" and actions and actions[-1]["tool"] == "note":
        return (
            f"REFUSED: your previous action (step {actions[-1]['step']}) was already a note, and notes "
            "may not be written twice in a row. Act on it now -- write_file, run_python, read_file, "
            "run_espresso_workflow, search_docs with a new query, or done."
        )
    if tool == "write_file" and args.get("path"):
        try:
            current = Path(args["path"]).read_text()
        except (OSError, UnicodeDecodeError):
            current = None
        if current is not None and current == args.get("content"):
            return (
                f"REFUSED: {args['path']} already has exactly this content, so writing it changes nothing. "
                "If you have not run it since writing it, run it now: only an actual error shows there is a bug. "
                "If you diagnosed a real bug, the fix is not in this version yet -- change the code to apply it."
            )
    if tool == "search_docs":
        last = next((e for e in reversed(actions) if e["tool"] == "search_docs"), None)
        key = lambda a: (str(a.get("query", "")).strip().lower(), tuple(sorted(a.get("sources") or ())))
        if last is not None and key(last["args"]) == key(args):
            return (
                f"REFUSED: identical to your search at step {last['step']}, and search results do not "
                "change -- use those results, or search for something different."
            )
    return None


def execute_tool(call: dict, state: AgentState):
    tool = call["tool"]
    args = call.get("args", {})

    reason = _refusal(tool, args, state)
    if reason is not None:
        # Logged under its own name (not as a note), so it stays visible in a
        # trimmed history and the model sees why nothing happened.
        state.log("refused", {"tool": tool, **args}, reason)
        return reason

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


def make_chat_step_source(
    llm,
    verbose: bool = True,
    *,
    max_history: int = MAX_HISTORY,
    trim_history: bool = False,
    on_error=None,
):
    """Build a `next_tool_call(state, step)` source for `run_step_loop` from
    any LangChain chat model (see chem_llm.llm).

    Each step sends build_messages(state) and parses one tool call out of
    the reply. A reply that is not a tool call is logged as a 'parse_error'
    step (fed back as a note so the model can self-correct) and reported to
    the loop as a no-op step. Token counts from the reply's usage_metadata
    are left in state.llm_usage for the step's timing record.

    on_error(state, step, exc): what to do when the model call raises --
    return None to skip the step, or raise StopIteration to end the run.
    Without it the exception propagates.
    """
    def next_tool_call(state: AgentState, step: int):
        messages = build_messages(state, max_history=max_history, trim_history=trim_history)
        try:
            reply = llm.invoke(messages, config={"run_name": f"step {step}", "metadata": {"step": step}})
        except Exception as e:
            if on_error is None:
                raise
            return on_error(state, step, e)
        usage = getattr(reply, "usage_metadata", None) or {}
        cached = (usage.get("input_token_details") or {}).get("cache_read")
        state.llm_usage = {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "cached_tokens": cached,
            "prompt_chars": sum(len(m.content) for m in messages),
        }
        output = reply.content if isinstance(reply.content, str) else str(reply.content)
        output = output.strip()
        if verbose:
            print("RAW MODEL OUTPUT:\n", output)
        try:
            return parse_tool_call(output)
        except (json.JSONDecodeError, ValueError) as e:
            state.add_note(f"Step {step}: failed to parse model output as JSON: {e}")
            state.log("parse_error", {"raw_output": output[:500]}, str(e))
            return None

    return next_tool_call


def retry_on_error(max_consecutive: int = 3):
    """An on_error for make_chat_step_source: log the failure as an
    'api_error' step and carry on, but end the run after `max_consecutive`
    failures in a row (a dead server fails every step the same way)."""
    def on_error(state: AgentState, step: int, exc: Exception):
        state.log("api_error", {"error_type": type(exc).__name__}, str(exc)[:2000])
        recent = state.history[-max_consecutive:]
        if len(recent) == max_consecutive and all(e["tool"] == "api_error" for e in recent):
            state.add_note(f"Step {step}: the model failed {max_consecutive} times in a row; stopping.")
            raise StopIteration
        return None

    return on_error


def make_llm_step_source(
    model,
    tokenizer,
    verbose: bool = True,
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    do_sample: bool = DO_SAMPLE,
    max_history: int = MAX_HISTORY,
    trim_history: bool = False,
):
    """make_chat_step_source for a transformers model/tokenizer pair loaded
    in this process. The keyword defaults are the legacy chem_llm.config
    values; DFTAgent passes its own settings."""
    from .llm import TransformersChatModel

    llm = TransformersChatModel(hf_model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens,
                                temperature=temperature, do_sample=do_sample,
                                model_name=getattr(getattr(model, "config", None), "_name_or_path", "local"))
    return make_chat_step_source(llm, verbose, max_history=max_history, trim_history=trim_history)


def run_step_loop(state: AgentState, next_tool_call, max_steps: int, verbose: bool = True, observer=None,
                  recorder=None) -> str | None:
    """Drive `state` through up to `max_steps` tool calls, one per step,
    obtained from `next_tool_call(state, step)`. Mutates `state` in place
    and returns why the loop stopped ("done", "max_steps",
    "source_exhausted"). The loop is a LangGraph graph: chem_llm.graph.

    `next_tool_call` should return a `{"tool": ..., "args": ...}` dict for
    the step, `None` if it already handled the step itself (e.g. logged a
    parse failure as a note) and no tool should be executed, or raise
    `StopIteration` to end the loop early (a replay source runs out of
    recorded calls before `max_steps`).

    `observer`, if given, is told about each step as it happens: its
    `start(step, tool_call)` before a tool runs, and its
    `finish(step, entry)` for every history entry the step logged (tool
    results, refusals, parse or API errors). See chem_llm.progress.

    `recorder`, a telemetry.StepRecorder, times each step (one is made if
    not given, so state.steps is always filled).
    """
    from .graph import run_graph
    from .telemetry import StepRecorder

    stop = run_graph(state, next_tool_call, max_steps, verbose=verbose, observer=observer,
                     recorder=recorder or StepRecorder(state))
    if stop == "max_steps" and not state.done:
        state.add_note("Max steps reached without explicit 'done' call.")
    return stop


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
    """Delete everything in `directory` except *.jsonl log files -- at any
    depth, so the log of a run kept in a subdirectory survives too -- after
    an interactive stdin confirmation. Directories left empty are removed;
    a directory that still holds a log keeps just that log. Raises
    RuntimeError if the user declines, or if `directory` fails the
    dangerous-target guard.
    """
    _guard_clear_target(directory)

    if not _confirm_clear_dir(directory):
        raise RuntimeError(f"clear_dir cancelled by user for {directory}")

    _clear_keeping_logs(directory)


def _clear_keeping_logs(directory: Path) -> bool:
    """Remove everything under `directory` except *.jsonl files. Returns
    whether anything was kept (so the caller knows if it may rmdir it)."""
    kept = False
    for entry in directory.iterdir():
        if entry.is_symlink() or entry.is_file():
            if entry.suffix == ".jsonl" and not entry.is_symlink():
                kept = True
            else:
                entry.unlink()
        elif entry.is_dir():
            if _clear_keeping_logs(entry):
                kept = True
            else:
                entry.rmdir()
    return kept

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
    from .rundiff import prompt_snapshot
    from .telemetry import run_metrics

    runtime = time.perf_counter() - start_time
    entry = {
        "run_id": state.run_id,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "work_dir": str(work_dir),
        "clear_dir": clear_dir,
        "load_path": bool(load_path),
        "runtime": runtime,
        "num_steps": len(state.history),
        "completed": state.done,
        "prompt": prompt_snapshot(SYSTEM_PROMPT_TEMPLATE, state.task),
        "metrics": run_metrics(state, runtime, model_load_s=(extra or {}).get("model_load_s")),
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
    log_extra: dict | None = None,
    observer=None,
    trace: bool = True,
    outcome=None,
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

    trace: append each step to trace.jsonl (next to the log) as it finishes,
    so a run killed mid-way still leaves its trajectory and timings.
    outcome: optional callable(work_dir) -> dict, recorded as the log's
    "outcome" (e.g. the band gap read back from disk).
    """
    if log_file is _UNSET:
        log_file = work_dir / "log.jsonl"

    prepare_work_dir(work_dir, clear_dir, load_path, load_deepseudopot)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.touch(exist_ok=True)

    from .telemetry import StepRecorder

    start_time = time.perf_counter()
    state = AgentState(task)
    trace_file = None
    if log_file and trace:
        trace_file = log_file.with_name("trace.jsonl") if log_file.name == "log.jsonl" else log_file.with_suffix(".trace.jsonl")
    recorder = StepRecorder(state, trace_file=trace_file, run_id=state.run_id)

    source = step_source or make_llm_step_source(model, tokenizer, verbose)
    run_step_loop(state, source, max_steps, verbose, observer=observer, recorder=recorder)

    if log_file:
        print("logging to...", log_file)
        entry = build_log_entry(state, start_time, work_dir, clear_dir, load_path, model_name, extra=log_extra)
        if outcome is not None:
            try:
                entry["outcome"] = outcome(work_dir)
            except Exception as e:  # never lose the run record over this
                entry["outcome"] = {"error": f"{type(e).__name__}: {e}"}
        append_log(log_file, entry)

    return state
