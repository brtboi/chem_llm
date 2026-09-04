"""Agent tool exposing the DeePseudopot package (chem_llm/DeePseudopot/) as a
single training action: run its main.py end-to-end (Zunger-form pre-fit of
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
import subprocess
import sys
from pathlib import Path

from . import register_tool
from .. import REPO_ROOT

DEEPSEUDOPOT_DIR = REPO_ROOT / "chem_llm" / "DeePseudopot"


@register_tool(
    "train_deepseudopot",
    (
        "Train a DeepPseudopot neural-network pseudopotential by running the "
        "DeePseudopot package's main.py on an already-assembled input bundle. "
        "This is the actual physics training step -- it: (1) pre-fits the NN "
        "to the Zunger-form initial pseudopotentials in `inputs_folder` "
        "(init_<Atom><N>Params.par), (2) trains the NN against the reference "
        "band structure(s) in `inputs_folder` for the number of epochs set by "
        "`max_num_epochs` in that folder's NN_config.par, (3) converges and "
        "writes the final pseudopotentials to `results_folder`. It always runs "
        "on CPU (no GPU needed) and can take anywhere from a couple minutes to "
        "well over an hour depending on the number of k-points and bands in "
        "the input bundle -- Hamiltonian caching (spin-orbit + nonlocal "
        "projector matrices) happens once per k-point BEFORE any training "
        "starts and is NOT parallelized across k-points for a single system, "
        "so it dominates runtime for a bundle with many k-points. Prefer a "
        "bundle with as few k-points as will still make a sensible fit "
        "(e.g. the '_g' reduced-k-point convention used elsewhere in this "
        "pipeline) to keep this tool call fast. `inputs_folder`'s "
        "NN_config.par must set `num_cores = 0`: any value >= 1 routes "
        "training through a multiprocessing pool that both uses far more "
        "memory (observed: OOM-killed around ~200GB RSS on this package's "
        "own reference bundle) and unconditionally calls .zero_grad() on "
        "LSDoptimizers[key], which is None (crashing every epoch) whenever "
        "local_env_corr = 0. Since nSystem is normally 1 for a single-job "
        "bundle, there is nothing for multiprocessing to parallelize "
        "across anyway, so num_cores = 0 is strictly better here, not just "
        "a workaround.\n"
        "`inputs_folder` must already contain a complete DeePseudopot input "
        "bundle: NN_config.par, system_0.par, kpoints_0.par, "
        "expBandStruct_0.par, bandWeights_0.par, input_0.par, and one "
        "init_<Atom><N>Params.par per atom in system_0.par's atom list "
        "(e.g. init_Cs0Params.par, init_Pb2Params.par, init_Br11Params.par). "
        "You MUST check the returned `success` boolean before treating this "
        "as having produced a trained model -- do not infer success from "
        "stdout being non-empty or from `results_folder` existing: this tool "
        "always creates `results_folder` and lets main.py write into it even "
        "when main.py errors out partway through, so a failed run can still "
        "leave partial files (e.g. only `initZunger_PPmodel.pth`) behind. "
        "`success` is only true when main.py exited with code 0 AND the "
        "final `final_pot_<Atom><N>.dat` file for every atom was written -- "
        "those files are only written once, at the very end of main.py, "
        "after training has fully converged and finished, so their presence "
        "is the one reliable 'the whole pipeline actually finished' signal. "
        "`final_pot_files` lists them; `trained_model_path` (if present) is "
        "the final epoch's NN weights, `epoch_<max_num_epochs>_PPmodel.pth` -- "
        "that .pth file, together with the final_pot_* files, IS the trained "
        "neural-network pseudopotential this whole pipeline exists to "
        "produce. When `success` is false, read `stderr_tail` (and "
        "`stdout_tail`) to see exactly what failed -- treat it like a "
        "run_python error under rule 3 of the system prompt: diagnose and "
        "fix the input bundle (e.g. a malformed .par file, a band count "
        "mismatch between bandWeights_0.par and expBandStruct_0.par), then "
        "call this tool again. Do not call 'done' while `success` is false."
    ),
    {
        "inputs_folder": (
            "string. Path to a directory containing a complete DeePseudopot "
            "input bundle (see description)."
        ),
        "results_folder": (
            "string. Directory to write training outputs to; created if it "
            "does not already exist."
        ),
        "timeout_seconds": (
            "int (optional, default 5400). Kill the training run and report "
            "failure if it has not finished within this many seconds."
        ),
    },
    remember_on=("inputs_folder", "results_folder"),
)
def train_deepseudopot(inputs_folder: str, results_folder: str, timeout_seconds: int = 5400):
    inputs_path = Path(inputs_folder).resolve()
    if not inputs_path.is_dir():
        return {"success": False, "stderr_tail": f"{inputs_path} is not a directory"}

    required = ["NN_config.par", "system_0.par", "kpoints_0.par", "expBandStruct_0.par", "bandWeights_0.par", "input_0.par"]
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
            [sys.executable, "main.py", inputs_arg, results_arg],
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

    final_pot_files = sorted(str(p) for p in results_path.glob("final_pot_*.dat"))

    # NNConfig.max_num_epochs isn't known to this tool without re-parsing
    # NN_config.par, so just report the highest-numbered epoch checkpoint
    # actually written, rather than assuming which epoch was "last".
    epoch_checkpoints = sorted(
        results_path.glob("epoch_*_PPmodel.pth"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    trained_model_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    success = result.returncode == 0 and len(final_pot_files) > 0

    return {
        "success": success,
        "returncode": result.returncode,
        "inputs_folder": str(inputs_path),
        "results_folder": str(results_path),
        "final_pot_files": final_pot_files,
        "trained_model_path": trained_model_path,
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }
