import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
from mp_api.client import MPRester

from config import READ_MAX_CHARS, PSEUDOS_DIR, MP_API_KEY

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
    "generate_cif",
    (
        "Download a crystal structure CIF from the Materials Project. "
        "If multiple structures match, the first matching result is selected. "
        "Space group can be specified using spacegroup_symbol and/or "
        "spacegroup_number. If both are provided, they must refer to the same "
        "space group. Space group symbols must use Materials Project Hermann-Mauguin "
        "format with underscores for subscripts (e.g. P4_2/mnm). "
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
            "Project format, using underscores for subscripts "
            "(e.g. P4_2/mnm)."
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
    "fetch_pseudopotential",
    (
        "Fetch a pseudopotential (.upf) file for a given element "
        "and copy it to output_path."
    ),
    {
        "element": "string (e.g. 'Br', 'Cs', 'Pb')",
        "output_path": "string",
    },
)
def fetch_pseudopotential(element: str, output_path: str):
    element = element.strip()
    src_path = Path(PSEUDOS_DIR) / f"{element}.upf"

    if not src_path.exists():
        candidates = [
            p for p in Path(PSEUDOS_DIR).glob("*.upf")
            if p.stem.lower() == element.lower()
        ]
        if not candidates:
            return {
                "success": False,
                "stderr": f"No pseudopotential found for element '{element}' in {PSEUDOS_DIR}",
            }
        src_path = candidates[0]

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(src_path, output_path)
    except Exception as e:
        return {"success": False, "stderr": str(e)}

    return {
        "success": True,
        "element": element,
        "source_path": str(src_path),
        "output_path": str(output_path),
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