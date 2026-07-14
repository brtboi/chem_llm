import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
from mp_api.client import MPRester

from config import READ_MAX_CHARS, CRYSTALLM_DIR, CRYSTALLM_PYTHON, CRYSTALLM_MODEL_DIR, MP_API_KEY

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
    "Download a crystal structure CIF from the Materials Project. If multiple structures exist, the first matching result is used.",
    {
        "composition": "string",
        "output_path": "string",
        "space_group": "string (optional)",
    },
)
def generate_cif(
    composition: str,
    output_path: str,
    space_group: str | None = None,
):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with MPRester(MP_API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                formula=composition,
                fields=["material_id", "structure", "symmetry"],
            )

        if space_group is not None:
            docs = [
                d
                for d in docs
                if d.symmetry is not None
                and (
                    str(d.symmetry.number) == str(space_group)
                    or d.symmetry.symbol.lower() == str(space_group).lower()
                )
            ]

        if not docs:
            return {
                "success": False,
                "stderr": f"No Materials Project entries found for {composition}"
                + (
                    f" with space group {space_group}"
                    if space_group is not None
                    else ""
                ),
            }

        doc = docs[0]

        doc.structure.to(
            filename=output_path,
            fmt="cif",
        )

        return {
            "success": True,
            "material_id": str(doc.material_id),
            "output_path": output_path,
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