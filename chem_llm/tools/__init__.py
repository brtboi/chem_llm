import os
from pathlib import Path
import subprocess
import sys
from mp_api.client import MPRester
from pseudohub import get_pseudo, get_hints
from pseudohub.exceptions import InvalidParameterError

from ..config import READ_MAX_CHARS, MP_API_KEY

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


@register_tool(
    "read_file",
    "Read a file from disk (truncated if very large)",
    {"path": "string"},
)
def read_file(path: str, max_chars: int = READ_MAX_CHARS):
    file_path = Path(path)
    if not file_path.exists():
        return f"ERROR: {path} does not exist"
    content = file_path.read_text(errors="replace")
    truncated = len(content) > max_chars
    return {
        "path": path,
        "content": content[:max_chars],
        "truncated": truncated,
        "total_chars": len(content),
    }


@register_tool(
    "run_python",
    "Execute a python file, returns stdout/stderr",
    {"path": "string"},
)
def run_python(path: str):
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
    )
    return {"stdout": result.stdout, "stderr": result.stderr}

@register_tool(
    "run_espresso_workflow",
    (
        "Run the Quantum ESPRESSO SCF + bands workflow for one calculation "
        "directory directly on the current node (module-loads 'espresso', then "
        "runs pw.x for the SCF step, pw.x for the bands step, and bands.x for "
        "the bands post-processing step, in that order -- three commands total, "
        "matching the pw.x/pw.x/bands.x sequence in submit.sh). Does NOT use "
        "srun or sbatch: this assumes the calling process is already running on "
        "an allocated compute node (e.g. inside the pipeline's own GPU job), so "
        "the three commands run in-process one after another. Stops and reports "
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

    steps = [
        ("scf", "pw.x", pw_input, "pw.out"),
        ("bands_scf", "pw.x", bands_input, "bands_pw.out"),
        ("bands_post", "bands.x", bands_post_input, "bands_post.out"),
    ]

    completed = []
    outputs = {}
    for name, exe, in_file, out_file in steps:
        if not (work_dir / in_file).exists():
            return {
                "success": False,
                "stderr": f"{name} failed: input file {in_file} not found in {work_dir}",
                "completed_steps": completed,
            }

        command = f"module load espresso/7.5-libxc-7.0.0-cpu && {exe} -in {in_file} > {out_file} 2>&1"
        # Without OMP_NUM_THREADS pinned, pw.x/bands.x default to one OpenMP
        # thread per core on the node (128 here) for every k-point's serial
        # diagonalization -- for a small unit cell that's massive
        # oversubscription (~118 threads thrashing on a tiny matrix) that
        # made even a single SCF iteration take many minutes. Since this
        # tool deliberately runs without srun/MPI (see docstring), there is
        # no k-point-level parallelism to hand those cores to, so pin to 1
        # thread and let each k-point's diagonalization run efficiently.
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        result = subprocess.run(["bash", "-c", command], cwd=work_dir, capture_output=True, text=True, env=env)
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
        "If multiple structures match, the first matching result is selected. "
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
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                formula=composition,
                fields=["material_id", "structure", "symmetry"],
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

        selected_doc.structure.to(
            filename=str(output_path),
            fmt="cif",
        )

        return {
            "success": True,
            "selected_material_id": str(selected_doc.material_id),
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
        "Fetch a pseudopotential file for an element from Pseudo-Dojo (via the "
        "pseudohub package) and save it to disk. Also returns the recommended "
        "plane-wave energy cutoff (ecut, in Ha) for the requested accuracy level, "
        "which should be used when building the Quantum ESPRESSO input file. "
        "Not every (kind, relativity, generator, accuracy) combination has a "
        "corresponding Pseudo-Dojo table -- if the call fails, check the error "
        "message's suggestions and retry with a valid combination."
    ),
    {
        "element": (
            "string or int. Element symbol (e.g. 'Si') or atomic number (e.g. 14)."
        ),
        "output_path": "string. Path (file or directory) to save the pseudopotential to. File type must match specified file format.",
        "kind": "string (optional, default 'nc'). 'nc' or 'paw'.",
        "relativity": (
            "string (optional, default 'sr'). 'sr' (scalar-relativistic) or "
            "'fr' (fully-relativistic; required for spin-orbit coupling)."
        ),
        "generator": (
            "string (optional, default 'pbe'). DFT functional used to generate "
            "the pseudopotential, e.g. 'pbe', 'pbesol', 'pw'."
        ),
        "accuracy": (
            "string (optional, default 'standard'). 'standard' or 'stringent'."
        ),
        "format": (
            "string (optional, default 'upf'). Pseudopotential file format, "
            "e.g. 'upf', 'psp8'. Use 'upf' for Quantum ESPRESSO."
        ),
        "hint_level": (
            "string (optional, default 'normal'). Accuracy level used to look up "
            "the recommended ecut: 'low', 'normal', or 'high'."
        ),
    },
    remember_on=("element", "kind", "relativity", "generator", "accuracy", "format"),
)
def get_pseudopotential(
    element: str | int,
    output_path: str,
    kind: str = "nc",
    relativity: str = "sr",
    generator: str = "pbe",
    accuracy: str = "standard",
    format: str = "upf",
    hint_level: str = "normal",
):
    output_path = Path(output_path).resolve()
    target_dir = output_path if output_path.is_dir() else output_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        saved_path = get_pseudo(
            element,
            kind=kind,
            relativity=relativity,
            generator=generator,
            accuracy=accuracy,
            format=format,
            output=str(output_path),
        )

        try:
            hints = get_hints(element, level=hint_level)
        except Exception:
            hints = None

        return {
            "success": True,
            "element": element,
            "kind": kind,
            "relativity": relativity,
            "generator": generator,
            "accuracy": accuracy,
            "format": format,
            "output_path": str(saved_path),
            "hints": hints,
        }

    except InvalidParameterError as e:
        return {
            "success": False,
            "stderr": str(e),
        }
    except Exception as e:
        return {
            "success": False,
            "stderr": str(e),
        }

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