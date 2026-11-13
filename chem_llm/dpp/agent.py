"""Second agent: a finished DFT run -> a trained DeePseudopot neural-network pseudopotential.

A fresh agent with its own system prompt, task and tools, rather than the
DFT agent continuing for another 50 steps: the DFT run's context (MP
lookups, pseudopotential choice, k-path, QE debugging) is irrelevant here,
and starting clean keeps the prompt small and the rules specific.

    from chem_llm import DFTAgent
    from chem_llm.dpp.agent import DeePseudopotAgent

    dft = DFTAgent(backend="vllm")
    result = dft.run("TiO2", "P4_2/mnm")                       # -> work_dir/calculations/000
    dpp = DeePseudopotAgent(dft)                                # same model, already loaded
    fit = dpp.run(result.work_dir / "calculations/000", work_dir="runs/tio2_dpp")
    print(fit)                                                  # RMS error, gap, trained model

    chem-llm-dpp deepseudopot_example/tio2_rutile/qe_run --work-dir runs/tio2_dpp --backend vllm

Input: one chem_llm QE run directory (pw.in, pw.out, kpath.json, the bands.x
filband, the UPF files). It is copied into the work dir as qe_run/.
The deterministic parts (reading QE output, writing and validating the
bundle, scoring the fit) are chem_llm.dpp.io; the agent decides the band
window, k-points, weights, basis and training settings, and checks the fit.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from .. import REPO_ROOT

TOOL_NAMES = ("read_file", "write_file", "run_python", "make_dir", "train_deepseudopot", "note", "done")
DOCS_DIR = REPO_ROOT / "chem_llm" / "DeePseudopot" / "docs"
EXAMPLE_DIR = REPO_ROOT / "deepseudopot_example" / "tio2_rutile"


def _tools():
    from ..tools import REGISTRY

    missing = [n for n in TOOL_NAMES if n not in REGISTRY]
    if missing:
        raise RuntimeError(f"tools not registered: {missing}")
    schemas = [REGISTRY[n][0] for n in TOOL_NAMES]
    dispatch = {n: REGISTRY[n][1] for n in TOOL_NAMES if REGISTRY[n][1] is not None}
    return schemas, dispatch


SYSTEM_PROMPT_BASE = (
    "You are a tool-using coding agent that solves tasks step by step.\n"
    "Your objective is to complete the task using the fewest necessary tool calls. Once the task is complete "
    "and verified, immediately call the 'done' tool.\n"
    "You MUST respond ONLY in valid JSON, one tool call per response.\n"
    "No markdown. No explanations outside the JSON.\n"
    "Output format:\n"
    "{ \"tool\": ..., \"args\": {...} }\n\n"
    "You will be shown the task, your working memory (files touched, notes, recent tool results), and you "
    "must decide the single next tool call.\n\n"
    "The task: turn a finished Quantum ESPRESSO band-structure calculation into a trained DeePseudopot "
    "neural-network empirical pseudopotential. AGENT PROTOCOL is how you use tools. INVARIANTS are facts "
    "each enforced with an explicit assert in the script itself. DECISION RULES are condition tables for "
    "the choices this task requires.\n\n"
    "=== AGENT PROTOCOL ===\n"
    "1. Do not repeat an action listed under 'Actions already completed' unless you are fixing a bug.\n"
    "2. Run (run_python) every .py file you write and confirm returncode 0; a traceback means it failed. "
    "Read back (read_file) any file a later step depends on.\n"
    "3. When a tool result has a `success` field, it alone says whether the tool worked. If it is false, "
    "read stderr / stderr_tail, fix the cause, and retry; never proceed as if it had worked.\n"
    "4. Before writing code that depends on a DeePseudopot file format or NN_config.par key you are not "
    "certain of, read the documentation file the task names for it (read_file, a line range if long). "
    "Do not guess formats.\n"
    "5. Write a note before each consequential decision (the band window, k-points, basis, training "
    "settings), with the numbers and the file they came from. Never write two notes in a row.\n"
    "6. On an error, change the code to fix its cause and rerun; rewriting a script unchanged does nothing.\n\n"
    "=== INVARIANTS ===\n"
    "INVARIANT 1: every number about the DFT run -- cell, atoms, electron count, occupied bands, band "
    "energies, VBM/CBM, k-points, spin-orbit -- is read from the run's files in code with "
    "chem_llm.dpp.io.read_qe_run, never typed in from memory or from the task text.\n\n"
    "INVARIANT 2 (band alignment): DeePseudopot fits its LOWEST nBands eigenvalues, in order, to the "
    "columns of expBandStruct_0.par. An empirical pseudopotential in a small plane-wave basis describes "
    "only the valence shell and the lowest conduction bands -- not the semicore states (filled shells "
    "below the outermost valence shell, e.g. Ti 3s/3p, Ga 3d, Pb 5d, Cs 5s/5p) that the DFT "
    "pseudopotentials include. So first_band (the first DFT band fitted) = the number of DFT bands "
    "holding semicore electrons: from chem_llm.dpp.io.pseudo_orbitals(upf) of each element, decide which "
    "orbitals are semicore, sum their occupations over EVERY atom of the cell, and divide by 2 (by 1 with "
    "spin-orbit, where each band holds one electron). In the script, print and assert that the dropped "
    "and the kept bands are separated by a clear gap: run['energies'][:, first_band - 1].max() is several "
    "eV below run['energies'][:, first_band].min().\n\n"
    "INVARIANT 3: band indices in input_0.par are relative to first_band: idxVB = run['n_occupied'] - "
    "first_band - 1 and idxCB = idxVB + 1. Assert 0 <= idxVB < idxCB < n_bands.\n\n"
    "INVARIANT 4: NN_config.par has SObool = 1 exactly when run['spin_orbit'] is true, and num_cores = 0 "
    "(>= 1 crashes every epoch and runs out of memory). write_bundle rejects both mistakes.\n\n"
    "INVARIANT 5 (units): maxKE is DeePseudopot's own plane-wave cutoff in Hartree -- not QE's ecutwfc. "
    "Energies are eV. write_bundle converts the cell from Angstrom to Bohr itself.\n\n"
    "INVARIANT 6: the fit is judged from DeePseudopot's output files with chem_llm.dpp.io.read_fit "
    "(RMS error, gap of the fit vs the DFT gap), never from training stdout or from a plot alone, and "
    "reported as it is, including when it is poor.\n\n"
    "=== DECISION RULES ===\n"
    "RULE: conduction bands\n"
    "  Fit the lowest conduction manifold: the conduction bands whose minimum lies within about 4 eV of "
    "the CBM. Count them in code from run['energies']; n_bands = (n_occupied - first_band) + that count.\n\n"
    "RULE: k-points\n"
    "  Start with the unique high-symmetry points: the first occurrence of each label in "
    "run['high_symmetry']. Each k-point adds cost to every epoch; add interior path points only if the "
    "high-symmetry fit is already good and time remains.\n\n"
    "RULE: band weights\n"
    "  Weight the bands at the gap (the top 2-4 valence and bottom 2-4 conduction bands) 3-5x the others "
    "(weight 1).\n\n"
    "RULE: basis (maxKE)\n"
    "  Plane waves ~ V * (2 * maxKE)**1.5 / (6 * pi**2), V the cell volume in Bohr**3. Choose maxKE so that "
    "is at least 4x n_bands, compute and print it; cost grows steeply with maxKE, so do not go far beyond.\n\n"
    "RULE: initial pseudopotential\n"
    "  init_params = {el: d['params'] for el, d in initial_zunger_params(run, semicore).items()}, with "
    "semicore the orbitals INVARIANT 2 dropped, per element (e.g. {'Ti': ['3S', '3P']}; elements with "
    "none may be left out). It fits the Zunger form to each element's local potential from the run's own "
    "UPF file, with the semicore shells counted as core and screened by the remaining valence electrons. "
    "Print each element's fit_rms, v0 and z_eff. Never type Zunger parameters in from memory, and never "
    "start from all zeros: a zero potential is a free-electron gas, far from any solid.\n\n"
    "RULE: training -- Monte Carlo first, then gradient descent\n"
    "  Stage 1, Monte Carlo over the Zunger parameters: a bundle written with mc_opts (betas [20, 2], "
    "stepsAtTemp [100, 30], tempStepSizeMod [1.0, 4.0]: the hot phase takes 4x larger steps to escape "
    "local minima; totalIter from the task), trained with train_deepseudopot(..., mode='zunger_mc'). "
    "Read the fit with read_fit: rms_error_eV, gap_fit_eV and mc_best_cost_by_quarter.\n"
    "  IF   rms_error_eV > 2.0 AND the best cost fell by more than 10% over the last quarter of the run\n"
    "  THEN another Monte Carlo round, starting from init_params = best_zunger_params(<that results "
    "folder>), into a new inputs and results folder\n"
    "  ELSE switch to Stage 2 (Monte Carlo has either fit well enough or stopped improving).\n"
    "  Stage 2, gradient descent on the neural network: a bundle with init_params = "
    "best_zunger_params(<the best Monte Carlo results folder>), init_Zunger_num_epochs = 2000 (the "
    "network is first fitted to that Zunger form), mc_bool = 0, optimizer_lr = 0.01, max_num_epochs from "
    "the task and plotEvery dividing it (read_fit needs an epoch_<N>_BS_sys0.dat), trained with "
    "mode='gradient'. Report the stage whose read_fit is best.\n\n"
    "=== HELPERS (chem_llm.dpp.io; use them, do not re-parse QE files by hand) ===\n"
    "    from chem_llm.dpp.io import (read_qe_run, pseudo_orbitals, initial_zunger_params, write_bundle,\n"
    "                                 best_zunger_params, read_fit)\n"
    "    run = read_qe_run('qe_run')     # dict: symbols, cell_angstrom, energies (nk, nbnd) eV, n_occupied,\n"
    "                                    # spin_orbit, kpoints, high_symmetry [(idx, label)], vbm, cbm, gap,\n"
    "                                    # pseudopotentials {element: upf path}\n"
    "    pseudo_orbitals(run['pseudopotentials']['Ti'])  # [{'label': '3S', 'l': 0, 'occupation': 2.0}, ...]\n"
    "    initial_zunger_params(run, {'Ti': ['3S', '3P']})  # {el: {'params': [9], 'fit_rms', 'v0', 'z_eff'}}\n"
    "    write_bundle('mc_inputs', run, first_band=..., n_bands=..., kpoint_indices=[...],\n"
    "                 band_weights=[...], input_params={'maxKE': ..., 'idxVB': ..., 'idxCB': ...,\n"
    "                 'systemName': ..., 'BS_plot_center': 0.0, 'BS_plot_CBVB_range': 8.0,\n"
    "                 'BS_plot_CBVB_range_zoom': 4.0}, nn_config={...}, init_params={el: [9 floats]},\n"
    "                 energy_shift=-run['vbm'],    # validates, writes every bundle file, returns a summary\n"
    "                 mc_opts={...})               # Stage 1 only: writes mcOpts1.par\n"
    "    best_zunger_params('mc_results')         # {el: [9 floats]}: the best a Monte Carlo stage found\n"
    "    read_fit('mc_results', 'mc_inputs')      # stage, rms_error_eV, gap_reference_eV, gap_fit_eV, per-band\n"
    "                                             # errors; for Monte Carlo also mc_best_cost_by_quarter\n\n"
)


def system_prompt() -> str:
    schemas, _ = _tools()
    return SYSTEM_PROMPT_BASE + f"Available tools:\n{json.dumps(schemas, indent=2)}"


TASK = """
Train a DeePseudopot neural-network empirical pseudopotential for {name} that reproduces the DFT band structure in qe_run/ (a finished Quantum ESPRESSO SCF + bands + bands.x run: pw.in, pw.out, kpath.json, the *.bands.dat filband, and the UPF pseudopotentials). Do not run any DFT.

Apply the INVARIANTS and DECISION RULES of the system prompt throughout.

Reference material (read what you need, a line range at a time):
- {docs}/system-data.md: every bundle file's format (system_0.par, input_0.par, kpoints_0.par, bandWeights_0.par, expBandStruct_0.par, init_<atom>Params.par).
- {docs}/configuration.md: every NN_config.par key.
- {docs}/outputs.md: what training writes.
- {template}: a working NN_config.par for this kind of single-system, no-spin-orbit fit (num_cores = 0). Start from it and change only what a rule or your analysis requires, saying why.

WORKFLOW (checkpoint after each step: state in 1-2 sentences what you verified):

STEP 1: Write and run inspect_run.py: read_qe_run('qe_run'); print the atoms, the electron count and occupied bands, spin-orbit, the number of DFT bands and k-points, the high-symmetry points, VBM/CBM/gap, and for every element its pseudo_orbitals. Print each DFT band's minimum and maximum energy so the semicore, valence and conduction groups are visible.

STEP 2: Write a note deciding first_band and the semicore orbitals (INVARIANT 2), n_bands (RULE: conduction bands), idxVB/idxCB (INVARIANT 3), the k-points (RULE: k-points), the band weights and maxKE (RULE: basis), each with the numbers from STEP 1 behind it.

STEP 3: Write and run build_mc_inputs.py: compute those choices in code from read_qe_run's result, with the INVARIANT asserts; compute init_params (RULE: initial pseudopotential); and call write_bundle('mc_inputs', ..., mc_opts={{'totalIter': {mc_iter}}}) with NN_config.par read from the template. Read back mc_inputs/input_0.par, mc_inputs/mcOpts1.par and mc_inputs/bundle_summary.json.

STEP 4: train_deepseudopot(inputs_folder='mc_inputs', results_folder='mc_results', mode='zunger_mc'). Check `success`. Write and run check_fit.py printing read_fit for it. Apply RULE: training -- another Monte Carlo round (mc_inputs_2 / mc_results_2, ...) or Stage 2.

STEP 5: Write and run build_gd_inputs.py: the same bundle choices, init_params = best_zunger_params(<best Monte Carlo results>), max_num_epochs = {gd_epochs}, written to gd_inputs (no mc_opts). train_deepseudopot(inputs_folder='gd_inputs', results_folder='gd_results', mode='gradient'), check `success`, and print read_fit for it.

DONE CRITERIA:
- inspect_run.py, build_mc_inputs.py, build_gd_inputs.py and check_fit.py ran with returncode 0, with the INVARIANT asserts in the build scripts.
- train_deepseudopot returned success: true for the Monte Carlo stage and for the gradient stage.
- The done summary gives: first_band and why (which orbitals are semicore, electron count), n_bands, idxVB/idxCB, the k-points, maxKE and the plane-wave count, the initial Zunger parameters' fit_rms per element, the Monte Carlo rounds and iterations and the gradient epochs, and for every stage read_fit's RMS error, max error, and DFT vs fitted gap; the paths of the trained model (the final epoch's .pth and final_pot_*.dat) and the band-structure plot; and an honest statement of fit quality and what would improve it.
"""


@dataclass
class FitResult:
    """What a DeePseudopot agent run produced, read back from disk."""

    work_dir: Path
    completed: bool
    trained: bool
    fit: dict | None
    results_dir: Path | None
    num_steps: int
    runtime_seconds: float
    summary: str | None = None
    state: object = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.completed and self.trained and self.fit is not None

    def __str__(self) -> str:
        if not self.fit:
            return (f"DeePseudopot: {'OK' if self.ok else 'INCOMPLETE'} | no fit read | {self.num_steps} steps | "
                    f"{self.runtime_seconds / 60:.1f} min")
        f = self.fit
        gap = (f" | gap {f['gap_fit_eV']:.2f} eV fit vs {f['gap_reference_eV']:.2f} DFT"
               if "gap_fit_eV" in f else "")
        return (f"DeePseudopot: {'OK' if self.ok else 'INCOMPLETE'} | RMS {f['rms_error_eV']:.2f} eV{gap} | "
                f"{self.num_steps} steps | {self.runtime_seconds / 60:.1f} min | {self.results_dir}")


def _latest_fit(work_dir: Path) -> tuple[Path | None, dict | None]:
    """The best-scoring stage the run trained (read_fit), if any: every
    results folder train_deepseudopot wrote (it records its inputs folder in
    inputs_folder.txt)."""
    from .io import read_fit

    best = (None, None)
    for marker in sorted(work_dir.glob("*/inputs_folder.txt")):
        res = marker.parent
        try:
            fit = read_fit(res, Path(marker.read_text().strip()))
        except Exception:
            continue
        if best[1] is None or fit["rms_error_eV"] < best[1]["rms_error_eV"]:
            best = (res, fit)
    return best


class DeePseudopotAgent:
    """Runs the DeePseudopot task with the model of a DFTAgent (loaded once,
    shared with DFT runs), or of a new DFTAgent built from `**kwargs`."""

    def __init__(self, agent=None, **kwargs):
        from ..agent import DFTAgent

        self.agent = agent if agent is not None else DFTAgent(**kwargs)

    @property
    def settings(self):
        return self.agent.settings

    def run(self, qe_run: str | Path, work_dir: str | Path | None = None, *, name: str | None = None,
            max_steps: int = 40, mc_iter: int = 3000, gd_epochs: int = 600, clear_dir: bool = True,
            template: str | Path | None = None, verbose: bool = True, on_step=None) -> FitResult:
        from ..core.agent_core import run_agent
        from .io import read_qe_run
        from ..settings import activated

        qe_run = Path(qe_run).expanduser().resolve()
        run = read_qe_run(qe_run)  # fail before loading anything if the input is unusable
        symbols = run["symbols"]
        name = name or "".join(f"{el}{symbols.count(el) if symbols.count(el) > 1 else ''}"
                               for el in dict.fromkeys(symbols))
        work_dir = Path(work_dir or self.settings.work_root / f"{name}_dpp").expanduser().resolve()
        if clear_dir and work_dir.exists():
            for p in work_dir.iterdir():
                if p.name.endswith(".jsonl"):
                    continue
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        work_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(qe_run, work_dir / "qe_run", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("tmp", "*.save", "*.wfc*"))
        template = Path(template or EXAMPLE_DIR / "NN_config_template.par").resolve()

        load_s = self.agent.load()
        task = TASK.format(name=name, docs=DOCS_DIR, template=template, mc_iter=mc_iter, gd_epochs=gd_epochs)
        prompt = system_prompt()
        _, dispatch = _tools()
        loaded = self.agent._loaded
        started = time.perf_counter()
        previous_cwd = Path.cwd()
        try:
            with activated(self.settings):
                state = run_agent(
                    task,
                    step_source=loaded.step_source_factory(self.settings, verbose, system_prompt=prompt),
                    system_prompt=prompt, dispatch=dispatch, model_name=loaded.name, work_dir=work_dir,
                    max_steps=max_steps, verbose=verbose, observer=on_step, clear_dir=False,
                    log_extra={"compound": name, "workflow": "deepseudopot", "qe_run": str(qe_run),
                               "config": self.agent.run_config(task),
                               **({"model_load_s": round(load_s, 1)} if load_s else {})},
                    outcome=lambda d: {"fit": _latest_fit(Path(d))[1]},
                )
        finally:
            import os

            os.chdir(previous_cwd)
        results_dir, fit = _latest_fit(work_dir)
        trained = any(e["tool"] == "train_deepseudopot" and isinstance(e.get("result"), dict)
                      and e["result"].get("success") for e in state.history)
        return FitResult(work_dir=work_dir, completed=state.done, trained=trained, fit=fit,
                         results_dir=results_dir, num_steps=len(state.history),
                         runtime_seconds=time.perf_counter() - started, summary=state.final_result, state=state)


def main(argv=None) -> int:
    """chem-llm-dpp QE_RUN [--work-dir D] [--backend vllm] ..."""
    import argparse

    from ..cli.progress import ProgressPrinter
    from ..settings import Settings

    p = argparse.ArgumentParser(prog="chem-llm-dpp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("qe_run", help="A finished chem_llm QE run dir (e.g. <work_dir>/calculations/000).")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--name", default=None, help="System name for the prompt and plots (default: from the cell).")
    p.add_argument("--config", default=None)
    p.add_argument("--backend", default=None, choices=["huggingface", "vllm", "claude", "qwen"])
    p.add_argument("--model", default=None)
    p.add_argument("--vllm-url", default=None)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--mc-iter", type=int, default=3000, help="Monte Carlo steps per round (stage 1)")
    p.add_argument("--gd-epochs", type=int, default=600, help="gradient-descent epochs (stage 2)")
    p.add_argument("--pretty-progress", action="store_true")
    args = p.parse_args(argv)
    settings = Settings.from_yaml(args.config) if args.config else Settings.discover()
    agent = DeePseudopotAgent(settings=settings, backend=args.backend, model=args.model, vllm_url=args.vllm_url)
    result = agent.run(args.qe_run, args.work_dir, name=args.name, max_steps=args.max_steps,
                       mc_iter=args.mc_iter, gd_epochs=args.gd_epochs,
                       verbose=not args.pretty_progress,
                       on_step=ProgressPrinter() if args.pretty_progress else None)
    print(result)
    if result.summary:
        print(result.summary)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
