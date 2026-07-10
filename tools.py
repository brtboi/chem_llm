import os
from pathlib import Path
import subprocess
import sys

from config import READ_MAX_CHARS, CRYSTALLM_DIR, CRYSTALLM_PYTHON, CRYSTALLM_MODEL_DIR

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
    "Generate a crystal structure CIF file for a given composition (and optional space group) using CrystaLLM.",
    {"composition": "string", "space_group": "string (optional)", "out_dir": "string"},
)
def generate_cif(composition: str, out_dir: str, space_group: str = None):
    os.makedirs(out_dir, exist_ok=True)
    prompt_path = os.path.join(out_dir, "prompt.txt")

    # 1. Build the prompt
    make_prompt_cmd = [CRYSTALLM_PYTHON, f"{CRYSTALLM_DIR}/bin/make_prompt_file.py", composition, prompt_path]
    if space_group:
        make_prompt_cmd += ["--spacegroup", space_group]
    r1 = subprocess.run(make_prompt_cmd, capture_output=True, text=True)
    if r1.returncode != 0:
        return {"stage": "make_prompt", "stdout": r1.stdout, "stderr": r1.stderr}

    # 2. Sample a raw CIF from the model
    r2 = subprocess.run(
        [CRYSTALLM_PYTHON, f"{CRYSTALLM_DIR}/bin/sample.py",
         f"out_dir={CRYSTALLM_MODEL_DIR}",
         f"start=FILE:{prompt_path}",
         "num_samples=1",
         "target=file"],  # sample.py writes sample_1.cif etc. to cwd/out_dir; adjust per actual flag
        capture_output=True, text=True, cwd=out_dir,
    )
    if r2.returncode != 0:
        return {"stage": "sample", "stdout": r2.stdout, "stderr": r2.stderr}

    # 3. Post-process into a valid CIF
    processed_dir = os.path.join(out_dir, "processed")
    r3 = subprocess.run(
        [CRYSTALLM_PYTHON, f"{CRYSTALLM_DIR}/bin/postprocess.py", out_dir, processed_dir],
        capture_output=True, text=True,
    )
    if r3.returncode != 0:
        return {"stage": "postprocess", "stdout": r3.stdout, "stderr": r3.stderr}

    cif_files = [f for f in os.listdir(processed_dir) if f.endswith(".cif")]
    return {"processed_dir": processed_dir, "cif_files": cif_files}

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