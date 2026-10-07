import os
from pathlib import Path
import re
import subprocess

import numpy as np
import sys
from ase.io import write as ase_write
from mp_api.client import MPRester
from pseudohub import get_pseudo, get_hints
# The Materials Project client (mp_api) returns structures as pymatgen
# Structure objects -- that dependency is unavoidable here since it's baked
# into mp_api's own return type, not something this codebase chooses. We
# convert to an ASE Atoms object immediately (via AseAtomsAdaptor) so every
# other tool/script downstream of generate_cif works with ASE, not pymatgen.
from pymatgen.io.ase import AseAtomsAdaptor

# Everything machine-specific comes from the running agent's Settings, read
# at call time -- see chem_llm.settings.active_settings.
from ..settings import active_settings
from .upf import read_upf_header

TOOLS: list[dict] = []
TOOL_DISPATCH: dict[str, callable] = {}

# tool name -> arg names that define a call's identity. Populated by
# register_tool(..., remember_on=...). AgentState uses this to remember
# successful calls for the whole task (surviving history truncation) so the
# agent recognizes it already e.g. fetched a given composition/element (and
# where the result was saved), regardless of how many steps have passed
# since.
REMEMBER_FIELDS: dict[str, tuple[str, ...]] = {}


def register_tool(name: str, description: str, parameters: dict, remember_on: tuple[str, ...] | None = None):
    """Register a tool.

    remember_on: optional tuple of arg names whose values define this call's
    identity (e.g. ("composition", "spacegroup_symbol")). If given,
    AgentState will remember successful calls with matching args -- including
    where their output was written -- for the rest of the task, and warn the
    model not to repeat them.
    """
    def decorator(func):
        TOOLS.append({"name": name, "description": description, "parameters": parameters})
        TOOL_DISPATCH[name] = func
        if remember_on is not None:
            REMEMBER_FIELDS[name] = remember_on
        return func
    return decorator


@register_tool(
    "write_file",
    "Write a file to disk",
    {"path": "string", "content": "string"},
)
def write_file(path: str, content: str):
    Path(path).write_text(content)
    return f"Wrote {path}"


@register_tool(
    "make_dir",
    "Create a directory (and any missing parent directories) on disk",
    {"path": "string"},
    remember_on=("path",),
)
def make_dir(path: str):
    dir_path = Path(path).resolve()
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        return {"success": True, "path": str(dir_path)}
    except OSError as e:
        return {"success": False, "stderr": str(e)}


# Files Quantum ESPRESSO writes; reading one that predates the inputs next to
# it means reading the result of an earlier run.
_QE_OUTPUT = re.compile(r"(\.out|^CRASH|\.bands\.dat.*|\.xml)$")


def _stale_against_inputs(path: Path) -> list[str]:
    """Input files (*.in) in the same directory modified after `path`, if
    `path` is a QE output."""
    if not _QE_OUTPUT.search(path.name):
        return []
    try:
        mtime = path.stat().st_mtime
        return sorted(f.name for f in path.parent.glob("*.in") if f.stat().st_mtime > mtime + 1)
    except OSError:
        return []


@register_tool(
    "read_file",
    (
        "Read a file: the whole file by default (up to a character cap), or num_lines "
        "lines starting at start_line (1-based). The result reports total_lines and, "
        "when anything was left out, which lines were shown. Do not read large numeric "
        "outputs (bands.dat, .gnu files, a full pw.out) whole: read a few lines to learn "
        "the format, then parse the numbers in a script."
    ),
    {
        "path": "string",
        "start_line": "int (optional). 1-based line to start at. Default 1.",
        "num_lines": "int (optional). How many lines to return. Default: to the end of the file.",
    },
)
def read_file(path: str, start_line: int = 1, num_lines: int | None = None, max_chars: int | None = None):
    max_chars = max_chars or active_settings().agent.read_max_chars
    file_path = Path(path)
    if not file_path.exists():
        return f"ERROR: {path} does not exist"
    lines = file_path.read_text(errors="replace").splitlines(keepends=True)
    total = len(lines)
    start = max(int(start_line or 1), 1) - 1
    stop = total if num_lines is None else start + max(int(num_lines), 0)
    chunk = "".join(lines[start:stop])
    char_capped = len(chunk) > max_chars
    content = chunk[:max_chars]
    end = start + content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    result = {
        "path": path,
        "content": content,
        "start_line": start + 1,
        "end_line": end,
        "total_lines": total,
        "truncated": start > 0 or end < total or char_capped,
    }
    stale = _stale_against_inputs(file_path)
    if stale:
        result["stale_warning"] = (
            f"{file_path.name} is OLDER than {', '.join(stale)} in the same directory: it was written by an earlier "
            "run, before those inputs were last changed, so it does not reflect them. Re-run "
            "run_espresso_workflow to get output for the current inputs."
        )
    if start >= total and total:
        result["hint"] = f"start_line {start + 1} is past the end of the file ({total} lines)."
    elif result["truncated"]:
        result["hint"] = f"Showing lines {start + 1}-{end} of {total}. Use start_line/num_lines to read other parts."
    return result


@register_tool(
    "run_python",
    "Execute a python file, returns stdout/stderr and the exit code (0 = success)",
    {"path": "string"},
)
def run_python(path: str):
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        # The agent's own credentials/HF cache, not just whatever this
        # process happened to inherit.
        env=active_settings().subprocess_env(),
    )
    return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}

def _release_cached_gpu_memory() -> None:
    """Hand GPU memory PyTorch is caching (for the loaded model) back to the
    driver, so a GPU build of pw.x on the same GPUs can use it. The model's
    weights stay resident; only the cache from earlier generation is freed."""
    import sys

    torch = sys.modules.get("torch")  # only if the model already imported it
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


@register_tool(
    "get_kpath",
    (
        "Get the standard band-structure k-path for a structure (seekpath, "
        "Hinuma et al. convention -- the same paths the Materials Project "
        "offers) and save it as JSON. Returns the path (e.g. "
        "'GAMMA-X-M-GAMMA-Z-R-A-Z | X-R | M-A'), the space group, the primitive "
        "cell the path refers to, the number of k-points and the plot ticks. "
        "Use it to inspect structure_000; inside setup_jobs.py call the same "
        "code per structure (chem_llm.kpath.standard_kpath) as INVARIANT 3 shows."
    ),
    {
        "structure_path": "string. Structure file ASE can read (e.g. structures/structure_000.cif).",
        "output_path": "string. Where to write the k-path JSON (e.g. kpath_000.json).",
    },
)
def get_kpath(structure_path: str, output_path: str):
    from ase.io import read

    from ..kpath import kp_atoms, kp_save, standard_kpath

    try:
        atoms = read(structure_path)
        kp = standard_kpath(atoms)
    except Exception as e:
        return {"success": False, "stderr": f"{type(e).__name__}: {e}"}
    kp_save(kp, output_path)
    prim = kp_atoms(kp)
    a, b, c, al, be, ga = prim.cell.cellpar()
    same = len(prim) == len(atoms) and np.allclose(prim.cell[:], atoms.cell[:], atol=1e-4)
    return {
        "success": True,
        "output_path": str(Path(output_path).resolve()),
        "spacegroup": kp["spacegroup"],
        "path": kp["path"],
        "n_kpoints": len(kp["kpoints"]),
        "ticks": [t["label"] for t in kp["ticks"]],
        "primitive_cell": {"formula": prim.get_chemical_formula(), "n_atoms": len(prim),
                           "abc": [round(v, 4) for v in (a, b, c)], "angles": [round(v, 2) for v in (al, be, ga)]},
        "input_cell_is_this_cell": bool(same),
        "note": ("Run pw.x and bands.x on kp_atoms(kp), not on the input file's cell"
                 + ("" if same else " (they differ here)") + "; write bands.in with kpts=kp_kpts(kp); "
                 "plot with chem_llm.kpath.plot_axis."),
    }


@register_tool(
    "run_espresso_workflow",
    (
        "Run the Quantum ESPRESSO SCF + bands workflow for one calculation "
        "directory on the current node: pw.x for the SCF step, pw.x for the bands "
        "step, and bands.x for the bands post-processing step -- three commands, in "
        "that order. How QE is launched (module, MPI/GPU layout, parallel flags) is "
        "configured by the harness, not by you; it runs inside the current "
        "allocation, never through the batch queue. Stops and reports "
        "failure at the first step that errors, without running later steps. "
        "The SCF, bands, and bands-post-processing input files must already "
        "exist in `directory` (as written by setup_jobs.py).\n"
        "You MUST check the returned `success` boolean before treating this "
        "workflow as having done anything -- do not infer success from the "
        "presence of an `outputs` entry, from stdout being non-empty, or from "
        "an output file existing on disk: this tool always attempts to write "
        "the step's output file (via shell redirection) even when that step's "
        "command fails, so a failed run still leaves a real .out file behind. "
        "`completed_steps` lists ONLY the steps (by name: scf, bands_scf, "
        "bands_post) that actually finished with exit code 0, in order -- "
        "when `success` is false, the failed step itself is NOT in that list "
        "even if its `outputs` entry is present, and none of the steps after "
        "it ran at all. When `success` is false, `stderr` (and, for a "
        "module-load/shell-level failure, `shell_stderr`) names exactly which "
        "step failed and why; treat that like a run_python error under rule 3 "
        "of the system prompt -- do not write a note claiming the workflow "
        "(or any of its steps) succeeded, and do not proceed to later steps "
        "or plotting/analysis of that directory's outputs until it is fixed "
        "and re-run with success: true."
    ),
    {
        "directory": "string. Directory containing the QE input files, e.g. a calculations/NNN folder.",
        "pw_input": "string (optional, default 'pw.in'). SCF pw.x input filename, relative to directory.",
        "bands_input": "string (optional, default 'bands.in'). Bands pw.x input filename, relative to directory.",
        "bands_post_input": (
            "string (optional, default 'bands_post.in'). bands.x post-processing "
            "input filename, relative to directory."
        ),
    },
    remember_on=("directory",),
)
def run_espresso_workflow(
    directory: str,
    pw_input: str = "pw.in",
    bands_input: str = "bands.in",
    bands_post_input: str = "bands_post.in",
):
    work_dir = Path(directory).resolve()
    if not work_dir.is_dir():
        return {"success": False, "stderr": f"{work_dir} is not a directory"}

    # A CRASH file from an earlier failed run would otherwise survive a
    # successful rerun and read as if this run had crashed.
    (work_dir / "CRASH").unlink(missing_ok=True)

    # How to invoke QE on this machine comes from the agent's settings, not
    # from a hardcoded module name -- see chem_llm.settings.QESettings.
    settings = active_settings()
    qe = settings.qe

    steps = [
        ("scf", qe.pw_x, qe.pw_args, pw_input, "pw.out"),
        ("bands_scf", qe.pw_x, qe.pw_args, bands_input, "bands_pw.out"),
        ("bands_post", qe.bands_x, "", bands_post_input, "bands_post.out"),
    ]
    _release_cached_gpu_memory()

    completed = []
    outputs = {}
    for name, exe, args, in_file, out_file in steps:
        if not (work_dir / in_file).exists():
            return {
                "success": False,
                "stderr": f"{name} failed: input file {in_file} not found in {work_dir}",
                "completed_steps": completed,
            }

        command = qe.shell_command(exe, in_file, out_file, args)
        # Without OMP_NUM_THREADS pinned, pw.x/bands.x default to one OpenMP
        # thread per core on the node for every k-point's serial
        # diagonalization -- for a small unit cell that's massive
        # oversubscription that made even a single SCF iteration take many
        # minutes. Serial runs (no mpi_prefix) have no k-point parallelism
        # to hand those cores to, so the configured default is 1 thread.
        env = {**settings.subprocess_env(), "OMP_NUM_THREADS": str(qe.omp_num_threads)}
        result = subprocess.run(
            ["bash", "-c", command], cwd=work_dir, capture_output=True, text=True,
            env=env, timeout=qe.timeout_seconds,
        )
        outputs[name] = str(work_dir / out_file)

        if result.returncode != 0:
            out_path = work_dir / out_file
            tail = out_path.read_text(errors="replace")[-2000:] if out_path.exists() else ""
            return {
                "success": False,
                "stderr": f"{name} ({exe} -in {in_file}) exited with code {result.returncode}",
                "completed_steps": completed,
                "outputs": outputs,
                "output_tail": tail,
                # module-load / shell-level errors (e.g. Lmod failures) never
                # reach out_file since they occur before the redirection's
                # right-hand side runs -- without these, a shell-level
                # failure is invisible to the agent.
                "shell_stdout": result.stdout[-2000:],
                "shell_stderr": result.stderr[-2000:],
            }

        completed.append(name)

    return {
        "success": True,
        "directory": str(work_dir),
        "completed_steps": completed,
        "outputs": outputs,
    }

@register_tool(
    "generate_cif",
    (
        "Download a crystal structure CIF from the Materials Project. "
        "If multiple structures match the space group, the most stable one is "
        "selected (lowest energy above the convex hull); the result lists the "
        "other candidates. "
        "Space group must be specified using spacegroup_symbol and/or "
        "spacegroup_number. If both are provided, they must refer to the same "
        "space group. Space group symbols must use Materials Project Hermann-Mauguin "
        "format with underscores for subscripts and dashes for inversion bars (e.g. P4_2/mnm, Fd-3m). "
        "Do NOT use tool without specifying space group with either number or symbol"
        "Note: the CIF file metadata may sometimes report an incorrect or lower "
        "symmetry space group (for example P1 or P4) even when the atomic structure "
        "itself corresponds to the requested space group. Do not reject or modify "
        "the CIF solely because the space group label inside the file differs from "
        "the requested space group. Use the Materials Project symmetry information "
        "returned by this tool as the authoritative source."
    ),
    {
        "composition": "string",
        "output_path": "string",
        "spacegroup_symbol": (
            "string (optional). Hermann-Mauguin space group symbol in Materials "
            "Project format, using underscores for subscripts and dashes for inversion bars"
            "(e.g. P4_2/mnm, Fd-3m)."
        ),
        "spacegroup_number": (
            "int (optional). International Tables space group number (1-230)."
        ),
    },
    remember_on=("composition", "spacegroup_symbol", "spacegroup_number"),
)
def generate_cif(
    composition: str,
    output_path: str,
    spacegroup_symbol: str | None = None,
    spacegroup_number: int | None = None,
):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with MPRester(active_settings().mp_api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=composition,
                fields=["material_id", "structure", "symmetry", "energy_above_hull", "band_gap"],
            )

        if not docs:
            return {
                "success": False,
                "stderr": f"No Materials Project entries found for {composition}.",
            }

        # Filter candidates by supplied space group information
        filtered_docs = []

        for doc in docs:
            if doc.symmetry is None:
                continue

            matches = True

            if spacegroup_number is not None:
                matches &= doc.symmetry.number == spacegroup_number

            if spacegroup_symbol is not None:
                matches &= (
                    doc.symmetry.symbol.lower()
                    == spacegroup_symbol.lower()
                )

            if matches:
                filtered_docs.append(doc)

        docs = filtered_docs

        if not docs:
            return {
                "success": False,
                "stderr": (
                    f"No Materials Project entries found for {composition} "
                    f"matching space group "
                    f"{spacegroup_symbol if spacegroup_symbol else ''} "
                    f"{spacegroup_number if spacegroup_number else ''}."
                ),
            }

        # Several polymorphs can share a space group (graphite: mp-48 plus a
        # 12-atom and a 4-atom variant); take the most stable, not whichever
        # the API happened to return first.
        docs = sorted(docs, key=lambda d: (d.energy_above_hull if d.energy_above_hull is not None else float("inf"),
                                          len(d.structure)))
        # Verify symbol and number consistency if both were supplied
        selected_doc = docs[0]

        if (
            spacegroup_symbol is not None
            and spacegroup_number is not None
        ):
            if (
                selected_doc.symmetry.symbol != spacegroup_symbol
                or selected_doc.symmetry.number != spacegroup_number
            ):
                return {
                    "success": False,
                    "stderr": (
                        "Space group mismatch: "
                        f"provided ({spacegroup_symbol}, {spacegroup_number}), "
                        f"but Materials Project returned "
                        f"({selected_doc.symmetry.symbol}, "
                        f"{selected_doc.symmetry.number})."
                    ),
                }

        candidate_material_ids = [
            str(d.material_id) for d in docs
        ]

        atoms = AseAtomsAdaptor.get_atoms(selected_doc.structure)
        ase_write(str(output_path), atoms, format="cif")

        return {
            "success": True,
            "selected_material_id": str(selected_doc.material_id),
            "energy_above_hull_eV_per_atom": selected_doc.energy_above_hull,
            # Materials Project's PBE gap: a hint for occupations (near 0 ->
            # metal or semimetal -> smearing), not a target to reproduce
            "band_gap_eV": selected_doc.band_gap,
            "candidate_material_ids": candidate_material_ids,
            "space_group": {
                "symbol": selected_doc.symmetry.symbol,
                "number": selected_doc.symmetry.number,
            },
            "output_path": str(output_path),
        }

    except Exception as e:
        return {
            "success": False,
            "stderr": str(e),
        }


@register_tool(
    "get_pseudopotential",
    (
        "Fetch a pseudopotential for one element from Pseudo-Dojo (via pseudohub) "
        "and save it to disk. Every parameter is required -- there are no defaults; "
        "decide each one and say why in a note first. Returns (1) facts read from "
        "the saved file itself: z_valence (the electrons this pseudopotential treats "
        "as valence -- sum z_valence over the atoms for the system's electron "
        "count), its pseudo_type, whether it is scalar- or fully-relativistic, and "
        "its functional; and (2) Pseudo-Dojo's recommended wavefunction cutoff for "
        "this same table, IN HARTREE (QE's ecutwfc is in Rydberg: 1 Ha = 2 Ry). "
        "Not every (kind, relativity, generator, accuracy) combination exists -- "
        "if the call fails, read the error's suggestions and retry."
    ),
    {
        "element": "string or int. Element symbol (e.g. 'Si') or atomic number (e.g. 14).",
        "output_path": "string. File or directory to save to; a file name's extension must match format.",
        "kind": "string. 'nc' (norm-conserving) or 'paw'.",
        "relativity": "string. 'sr' (scalar-relativistic) or 'fr' (fully-relativistic, includes spin-orbit).",
        "generator": "string. Functional the pseudopotential was generated with, e.g. 'pbe', 'pbesol', 'pw'.",
        "accuracy": "string. 'standard' or 'stringent'.",
        "format": "string. File format, e.g. 'upf' (what Quantum ESPRESSO reads) or 'psp8'.",
        "hint_level": "string. Which recommended cutoff to report as `value`: 'low', 'normal' or 'high'.",
    },
    remember_on=("element", "kind", "relativity", "generator", "accuracy", "format"),
)
def get_pseudopotential(
    element: str | int,
    output_path: str,
    kind: str,
    relativity: str,
    generator: str,
    accuracy: str,
    format: str,
    hint_level: str,
):
    spec = dict(kind=kind, relativity=relativity, generator=generator, accuracy=accuracy)
    output_path = Path(output_path).resolve()
    target_dir = output_path if output_path.is_dir() else output_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        saved_path = get_pseudo(element, format=format, output=str(output_path), **spec)
    except Exception as e:  # InvalidParameterError carries the valid alternatives
        return {"success": False, "stderr": str(e)}

    result = {"success": True, "element": element, "output_path": str(saved_path), "requested": {**spec, "format": format}}

    # Facts from the file itself -- the authority on what QE will actually use.
    # Decided by the requested format, not the file name: agents pick names
    # like "Ti.pbesol_upf" with no .upf extension.
    if str(format).lower() == "upf" or str(saved_path).lower().endswith(".upf"):
        try:
            result["file"] = read_upf_header(saved_path)
        except ValueError as e:
            result["file_warning"] = str(e)
        else:
            wanted = {"sr": "scalar", "fr": "full"}.get(relativity)
            if wanted and result["file"]["relativistic"] not in (None, wanted):
                result["file_warning"] = (
                    f"requested relativity={relativity!r} but the file says relativistic="
                    f"{result['file']['relativistic']!r}"
                )

    # Recommended cutoffs from the hints table of the SAME kind/relativity/
    # generator/accuracy as the file (pseudohub otherwise falls back to the
    # PBE/standard table, whose cutoffs belong to a different pseudopotential).
    # pseudohub's "n_valence" is the table's `nv`, a count of valence SHELLS,
    # not electrons, so it is not passed on; use file.z_valence.
    try:
        hints = get_hints(element, level=hint_level, **spec)
        raw = hints.get("raw", {})
        result["recommended_ecutwfc"] = {
            "value": hints.get("ecut"),
            "level": hint_level,
            "low": raw.get("hl"),
            "normal": raw.get("hn"),
            "high": raw.get("hh"),
            "unit": "Ha",
        }
    except Exception as e:
        result["recommended_ecutwfc"] = None
        result["hints_warning"] = f"no Pseudo-Dojo cutoff hints for this table: {e}"
    return result


# --- State-mutating tools (schemas only; behavior lives in agent_core) ---
TOOLS.append({
    "name": "note",
    "description": "Record an observation/plan in your scratchpad without taking an action",
    "parameters": {"text": "string"},
})
TOOLS.append({
    "name": "done",
    "description": "Call this when the task is fully complete. Provide a summary.",
    "parameters": {"summary": "string"},
})

# tools/docs.py, tools/qe_validate.py, and tools/deepseudopot.py register
# themselves against TOOLS/TOOL_DISPATCH above via `from tools import
# register_tool`; import them for that side effect.
from . import docs  # noqa: E402,F401
from . import qe_validate  # noqa: E402,F401
from . import deepseudopot  # noqa: E402,F401