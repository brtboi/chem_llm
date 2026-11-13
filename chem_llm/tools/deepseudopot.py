"""Agent tool exposing the DeePseudopot package (chem_llm/DeePseudopot/) as a
training stage: Monte Carlo over the Zunger-form parameters (main_mc_Zunger.py),
or its main.py end-to-end (Zunger-form pre-fit of
the neural-network pseudopotential, then gradient-descent training against
the reference band structure(s) in `inputs_folder`, then convergence and
writing of the final real/reciprocal-space pseudopotentials) and report
whether it actually finished.

DeePseudopot's main.py is a script, not an importable library entry point
(it does `from utils.read import ...` etc., relative to its own directory),
so this shells out to it with cwd=DeePseudopot's own directory -- exactly
how the package's own README says to invoke it (`python main.py
/path/to/inputs/ /path/to/results/`), just from a subprocess instead of a
shell.
"""
import os
import subprocess
import sys
from pathlib import Path

from . import register_tool

# A git checkout of github.com/TommyLinkl/DeePseudopot. It is vendored in the
# source repository only -- not shipped in the wheel -- so an installed
# package needs $DEEPSEUDOPOT_DIR pointing at a checkout to use this tool.
DEEPSEUDOPOT_DIR = Path(os.environ.get("DEEPSEUDOPOT_DIR") or Path(__file__).resolve().parents[1] / "DeePseudopot")


@register_tool(
    "train_deepseudopot",
    (
        "Run one DeePseudopot training stage on an input bundle written by "
        "chem_llm.dpp.io.write_bundle. CPU only; minutes to an hour.\n"
        "mode='zunger_mc' (the first stage): Monte Carlo over the Zunger-form "
        "parameters in init_<element>Params.par (main_mc_Zunger.py), with the "
        "options in the bundle's mcOpts1.par (write_bundle(..., mc_opts=...)). "
        "Writes best_<element>Params.dat (the lowest-cost parameters, the next "
        "stage's init_params via chem_llm.dpp.io.best_zunger_params), "
        "bestBandStruct_<N>_0.dat and iterations.dat. Success = every element's "
        "best_<element>Params.dat was written.\n"
        "mode='gradient': main.py. Pre-fits the neural network to the Zunger "
        "form in init_<element>Params.par for init_Zunger_num_epochs, then "
        "trains it by gradient descent for max_num_epochs (NN_config.par), and "
        "writes final_pot_<element>.dat and epoch_<N>_PPmodel.pth (the trained "
        "model). Success = main.py exited 0 and wrote final_pot_* for every "
        "element.\n"
        "Check `success` before going on; when false, read stderr_tail, fix "
        "the bundle and call again. Score every stage with "
        "chem_llm.dpp.io.read_fit(results_folder, inputs_folder)."
    ),
    {
        "inputs_folder": "string. The input bundle directory.",
        "results_folder": "string. Where the stage writes its output; created if missing. Use a new one per stage.",
        "mode": "string (optional, default 'gradient'). 'zunger_mc' or 'gradient'.",
        "timeout_seconds": "int (optional, default 5400). Stop the stage and report failure after this long.",
    },
    remember_on=("inputs_folder", "results_folder"),
)
def train_deepseudopot(inputs_folder: str, results_folder: str, mode: str = "gradient", timeout_seconds: int = 5400):
    if not (DEEPSEUDOPOT_DIR / "main.py").exists():
        return {
            "success": False,
            "stderr_tail": (
                f"DeePseudopot is not available: no main.py in {DEEPSEUDOPOT_DIR}. It is not part of "
                "the chem-llm package -- clone https://github.com/TommyLinkl/DeePseudopot and set "
                "DEEPSEUDOPOT_DIR to that checkout."
            ),
        }
    inputs_path = Path(inputs_folder).resolve()
    if not inputs_path.is_dir():
        return {"success": False, "stderr_tail": f"{inputs_path} is not a directory"}

    if mode not in ("gradient", "zunger_mc"):
        return {"success": False, "stderr_tail": f"mode must be 'gradient' or 'zunger_mc', got {mode!r}"}
    required = ["NN_config.par", "system_0.par", "kpoints_0.par", "expBandStruct_0.par", "bandWeights_0.par", "input_0.par"]
    if mode == "zunger_mc":
        required.append("mcOpts1.par")
    missing = [f for f in required if not (inputs_path / f).exists()]
    if missing:
        return {
            "success": False,
            "stderr_tail": f"inputs_folder is missing required file(s): {missing}",
        }

    results_path = Path(results_folder).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    # main.py expects trailing-slash-style folder args (it does string
    # concatenation like inputsFolder + 'NN_config.par'), so pass str(path) + "/".
    inputs_arg = str(inputs_path) + "/"
    results_arg = str(results_path) + "/"

    try:
        result = subprocess.run(
            [sys.executable, "main_mc_Zunger.py" if mode == "zunger_mc" else "main.py", inputs_arg, results_arg],
            cwd=DEEPSEUDOPOT_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "stderr_tail": f"train_deepseudopot timed out after {timeout_seconds}s",
            "stdout_tail": (e.stdout or "")[-3000:] if e.stdout else "",
            "results_folder": str(results_path),
        }

    elements = _elements(inputs_path / "system_0.par")
    # which inputs each results folder came from, for read_fit and the agent's own bookkeeping
    (results_path / "inputs_folder.txt").write_text(str(inputs_path) + "\n")
    if mode == "zunger_mc":
        best = {el: str(results_path / f"best_{el}Params.dat") for el in elements}
        success = all(Path(f).exists() for f in best.values())
        out = {"success": success, "mode": mode, "returncode": result.returncode,
               "inputs_folder": str(inputs_path), "results_folder": str(results_path),
               "best_params_files": best if success else {}}
        if success and result.returncode != 0 and "realSpacePot" in result.stderr:
            # main_mc_Zunger.py writes best_*Params.dat, then crashes in its last step
            # (a tensor-shape bug in realSpacePot writing final_pot_*); the fit is complete.
            out["note"] = ("the Monte Carlo fit finished and wrote best_*Params.dat; DeePseudopot's optional "
                           "real-space write-out after it crashed (a known bug), which does not affect the result")
        out["stdout_tail"] = result.stdout[-3000:]
        out["stderr_tail"] = "" if out.get("note") else result.stderr[-3000:]
        return out

    final_pot_files = sorted(str(p) for p in results_path.glob("final_pot_*.dat"))

    # NNConfig.max_num_epochs isn't known to this tool without re-parsing
    # NN_config.par, so just report the highest-numbered epoch checkpoint
    # actually written, rather than assuming which epoch was "last".
    epoch_checkpoints = sorted(
        results_path.glob("epoch_*_PPmodel.pth"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    trained_model_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    success = result.returncode == 0 and all(
        any(Path(f).name == f"final_pot_{el}.dat" for f in final_pot_files) for el in elements)

    return {
        "success": success,
        "mode": mode,
        "returncode": result.returncode,
        "inputs_folder": str(inputs_path),
        "results_folder": str(results_path),
        "final_pot_files": final_pot_files,
        "trained_model_path": trained_model_path,
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


def _elements(system_par: Path) -> list[str]:
    """The element symbols of system_0.par's atoms block, in order of first appearance."""
    lines = system_par.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().lower() == "atoms") + 1
    return list(dict.fromkeys(line.split()[0] for line in lines[start:] if line.split()))
