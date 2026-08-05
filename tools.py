import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
from mp_api.client import MPRester
from pseudohub import get_pseudo, get_hints
from pseudohub.exceptions import InvalidParameterError
from tool_modules.pymatgen_docs import scan_pymatgen_docs

from config import READ_MAX_CHARS, MP_API_KEY

TOOLS: list[dict] = []
TOOL_DISPATCH: dict[str, callable] = {}


def register_tool(name: str, description: str, parameters: dict):
    def decorator(func):
        TOOLS.append({"name": name, "description": description, "parameters": parameters})
        TOOL_DISPATCH[name] = func
        return func
    return decorator


@register_tool(
    "write_file",
    "Write a file to disk",
    {"path": "string", "content": "string"},
)
def write_file(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)
    return f"Wrote {path}"


@register_tool(
    "read_file",
    "Read a file from disk (truncated if very large)",
    {"path": "string"},
)
def read_file(path: str, max_chars: int = READ_MAX_CHARS):
    if not os.path.exists(path):
        return f"ERROR: {path} does not exist"
    with open(path, "r", errors="replace") as f:
        content = f.read()
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
    "scan_pymatgen_docs",
    (
        "Look up official pymatgen documentation for a function, class, method, "
        "or property by name, scraped live from https://pymatgen.org. Accepts a "
        "bare name ('Structure', 'get_space_group_info'), a 'Class.method' pair "
        "('Structure.get_space_group_info'), or a property name -- properties "
        "(e.g. getters exposed via @property, like Composition.reduced_formula) "
        "are included, unlike a plain source-code introspection would give. "
        "Returns the exact signature, full docstring, and a doc URL for each "
        "match. If no exact match is found, returns nearest-name suggestions -- "
        "retry with one of those. Use this before calling an unfamiliar pymatgen "
        "member to confirm its signature and behavior rather than guessing."
    ),
    {
        "query": (
            "string. Name to look up: a bare identifier ('Structure', "
            "'get_space_group_info'), or 'Class.method' / 'Class.property' "
            "(e.g. 'Composition.reduced_formula')."
        ),
        "class_hint": (
            "string (optional). When query is a bare method/property name that "
            "exists on multiple classes (e.g. 'get_nn_info' appears on several "
            "neighbor-finder classes), narrow to matches whose containing class "
            "name matches this hint."
        ),
        "max_results": (
            "int (optional, default 5). Maximum number of matches to return "
            "details for."
        ),
    },
)
def get_pymatgen_docs(
    query: str,
    class_hint: str | None = None,
    max_results: int = 5,
):
    return scan_pymatgen_docs(query, class_hint=class_hint, max_results=max_results)

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
)
def generate_cif(
    composition: str,
    output_path: str,
    spacegroup_symbol: str | None = None,
    spacegroup_number: int | None = None,
):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

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
            filename=output_path,
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
            "output_path": output_path,
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
        "kind must be 'nc' (norm-conserving) or 'paw'. relativity must be 'sr' "
        "(scalar-relativistic) or 'fr' (fully-relativistic, required for spin-orbit "
        "coupling calculations). generator is the DFT functional the pseudopotential "
        "was generated with (e.g. 'pbe', 'pbesol', 'pw'). accuracy must be "
        "'standard' or 'stringent'. format must be a valid pseudopotential file "
        "format (e.g. 'upf', 'psp8'); Quantum ESPRESSO requires 'upf'. "
        "Not every (kind, relativity, generator, accuracy) combination has a "
        "corresponding Pseudo-Dojo table -- if the call fails, check the error "
        "message's suggestions and retry with a valid combination."
    ),
    {
        "element": (
            "string or int. Element symbol (e.g. 'Si') or atomic number (e.g. 14)."
        ),
        "output_path": "string. Path (file or directory) to save the pseudopotential to.",
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
    output_path = os.path.abspath(output_path)
    os.makedirs(
        output_path if os.path.isdir(output_path) or output_path.endswith(os.sep)
        else os.path.dirname(output_path),
        exist_ok=True,
    )

    try:
        saved_path = get_pseudo(
            element,
            kind=kind,
            relativity=relativity,
            generator=generator,
            accuracy=accuracy,
            format=format,
            output=output_path,
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