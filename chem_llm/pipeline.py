"""One-call entry point: compound + space group -> finished DFT run.

    from chem_llm import configure, run_dft_agent

    configure("config.yaml")
    result = run_dft_agent("TiO2", "P4_2/mnm", spacegroup_name="rutile")
    print(result.completed, result.band_gap, result.work_dir)

Everything between those two lines -- picking the backend, hiding tools the
task does not need, building the prompt, creating the run directory,
driving the agent loop, and reading the outcome back off disk -- is handled
here. The pieces are still importable on their own (chem_llm.tasks,
chem_llm.agent_core, chem_llm.claude_backend) when you want to assemble a
different workflow.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .settings import Settings, get_settings
from .tasks import build_task


@dataclass
class RunResult:
    """What a run produced, read back from disk rather than from the
    agent's own claims about itself."""

    compound: str
    spacegroup: str
    work_dir: Path
    completed: bool               # agent called done
    num_steps: int
    runtime_seconds: float
    model: str
    summary: str | None = None    # the agent's done summary
    scf_converged: bool = False
    dft_finished: bool = False    # all three QE stages reported JOB DONE
    n_atoms: int | None = None
    n_electrons: float | None = None
    band_gap: float | None = None  # eV, from the bands output
    plot: Path | None = None
    state: Any = None              # the full AgentState

    @property
    def ok(self) -> bool:
        """A run is only good if the physics landed, not merely if the
        agent decided it was finished."""
        return self.completed and self.dft_finished and self.band_gap is not None

    def __str__(self) -> str:
        gap = f"{self.band_gap:.3f} eV" if self.band_gap is not None else "no gap"
        return (
            f"{self.compound} ({self.spacegroup}): {'OK' if self.ok else 'INCOMPLETE'} | "
            f"{gap} | {self.num_steps} steps | {self.runtime_seconds / 60:.1f} min | {self.model}"
        )


# ----------------------------------------------------------------------
def _hide_tools(names) -> None:
    """Drop tools from the registry before agent_core snapshots it into the
    system prompt (which happens once, at import time)."""
    from . import tools as tools_module

    names = set(names or ())
    if not names:
        return
    tools_module.TOOLS[:] = [t for t in tools_module.TOOLS if t["name"] not in names]
    for name in names:
        tools_module.TOOL_DISPATCH.pop(name, None)


def _build_step_source(settings: Settings, verbose: bool):
    """Return (step_source, model_name) for the configured backend."""
    backend = settings.model.backend.lower()

    if backend == "claude":
        settings.require("anthropic_api_key")
        from .claude_backend import ClaudeModel, make_claude_step_source

        claude = ClaudeModel(
            model=settings.model.claude_model,
            max_tokens=settings.model.claude_max_tokens,
            effort=settings.model.claude_effort,
            api_key=settings.anthropic_api_key,
        )
        return make_claude_step_source(claude, verbose=verbose), claude.model

    if backend == "qwen":
        settings.require("hf_token")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .agent_core import make_llm_step_source

        name = settings.model.qwen_model
        tokenizer = AutoTokenizer.from_pretrained(name, token=settings.hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            name, device_map="auto", dtype=torch.bfloat16, token=settings.hf_token
        )
        return make_llm_step_source(model, tokenizer, verbose), name

    raise ValueError(f"Unknown model backend {backend!r}; expected 'claude' or 'qwen'")


# ----------------------------------------------------------------------
_FLOAT = r"[-+]?\d+\.?\d*(?:[eEdD][-+]?\d+)?"


def _inspect_outputs(work_dir: Path, job: str = "000") -> dict:
    """Read the real QE/plot artefacts a run should have left behind."""
    calc = work_dir / "calculations" / job
    found: dict = {"scf_converged": False, "dft_finished": False}
    if not calc.is_dir():
        return found

    pw_out = calc / "pw.out"
    if pw_out.exists():
        text = pw_out.read_text(errors="replace")
        found["scf_converged"] = "convergence has been achieved" in text
        for key, pattern in (
            ("n_atoms", r"number of atoms/cell\s*=\s*(\d+)"),
            ("n_electrons", rf"number of electrons\s*=\s*({_FLOAT})"),
            ("n_bands", r"number of Kohn-Sham states\s*=\s*(\d+)"),
        ):
            match = re.search(pattern, text)
            if match:
                found[key] = float(match.group(1)) if "." in match.group(1) else int(match.group(1))

        # Occupied-band count must come from the electron count, NOT from
        # "number of Kohn-Sham states": those coincide only when nbnd
        # defaults to nelec/2 (fixed occupations). Under smearing QE pads
        # nbnd above that, and using the padded value picks two bands deep
        # inside the conduction manifold -- which yields a nonsense
        # (often negative) gap. Spin-orbit/noncollinear runs hold one
        # electron per band rather than two.
        electrons = found.get("n_electrons")
        if electrons:
            noncolin = bool(re.search(r"(?i)non-?colin|noncollinear|spin-orbit", text))
            found["n_occupied"] = int(round(electrons if noncolin else electrons / 2))

    done = sum(
        (calc / name).exists() and "JOB DONE" in (calc / name).read_text(errors="replace")
        for name in ("pw.out", "bands_pw.out", "bands_post.out")
    )
    found["dft_finished"] = done == 3

    plots = sorted(calc.glob("*.pdf")) + sorted(calc.glob("*.png"))
    found["plot"] = plots[0] if plots else None

    gap = _band_gap(calc, job, found.get("n_occupied"))
    if gap is not None:
        found["band_gap"] = gap
    return found


def _band_gap(calc: Path, job: str, n_occupied: int | None) -> float | None:
    """Gap from the bands output, computed here rather than trusted from
    the agent. Handles both shapes bands.x can leave: the gnuplot-format
    .gnu file (one blank-line-separated block per band) and the primary
    filband file (&plot header, then k-point/energy blocks)."""
    if not n_occupied:
        return None
    try:
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover
        return None

    gnu = calc / f"{job}.bands.dat.gnu"
    if gnu.exists() and gnu.stat().st_size > 0:
        blocks, current = [], []
        for line in gnu.read_text(errors="replace").splitlines():
            if not line.strip():
                if current:
                    blocks.append(current)
                    current = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                current.append(float(parts[1]))
        if current:
            blocks.append(current)
        if len(blocks) > n_occupied:
            energies = np.array([b for b in blocks if len(b) == len(blocks[0])]).T
            return _gap_from(energies, n_occupied)

    dat = calc / f"{job}.bands.dat"
    if dat.exists() and dat.stat().st_size > 0:
        text = dat.read_text(errors="replace")
        header = re.search(r"nbnd=\s*(\d+),\s*nks=\s*(\d+)", text)
        if header and "/" in text:
            nbnd, nks = int(header.group(1)), int(header.group(2))
            values = [float(x) for x in text.split("/", 1)[1].split()]
            stride = 3 + nbnd
            if len(values) >= nks * stride and nbnd > n_occupied:
                energies = np.array(
                    [values[i * stride + 3 : (i + 1) * stride] for i in range(nks)]
                )
                return _gap_from(energies, n_occupied)
    return None


def _gap_from(energies, n_occupied: int) -> float | None:
    if energies.ndim != 2 or energies.shape[1] <= n_occupied:
        return None
    vbm = energies[:, n_occupied - 1].max()
    cbm = energies[:, n_occupied].min()
    return float(cbm - vbm)


# ----------------------------------------------------------------------
def run_dft_agent(
    compound: str,
    spacegroup: str,
    *,
    spacegroup_number: int | None = None,
    spacegroup_name: str = "",
    run_name: str | None = None,
    settings: Settings | None = None,
    max_steps: int | None = None,
    n_structures: int = 51,
    relativistic: str = "scalar",
    functional: str = "pbesol",
    task: str | None = None,
    hide_tools=("train_deepseudopot",),
    verbose: bool = True,
    clear_dir: bool = True,
) -> RunResult:
    """Run the full agentic DFT band-structure workflow for one compound.

    Only `compound` and `spacegroup` are required. The run is written to
    settings.work_root / run_name (default: the compound name), and the
    returned RunResult reports what actually landed on disk.
    """
    settings = settings or get_settings()
    settings.apply_env()

    _hide_tools(hide_tools)

    # Imported after _hide_tools: agent_core serialises the tool list into
    # the system prompt at import time.
    from .agent_core import run_agent

    work_dir = (settings.work_root / (run_name or compound)).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    prompt = task or build_task(
        compound=compound,
        spacegroup=spacegroup,
        spacegroup_number=spacegroup_number,
        spacegroup_name=spacegroup_name,
        n_structures=n_structures,
        relativistic=relativistic,
        functional=functional,
    )

    step_source, model_name = _build_step_source(settings, verbose)

    # run_agent chdir's into the work dir (every tool resolves paths against
    # cwd). Restore it afterwards so calling this twice from a notebook does
    # not nest run dirs, and so the caller's relative paths still mean what
    # they did before the call.
    previous_cwd = Path.cwd()
    try:
        state = run_agent(
            prompt,
            step_source=step_source,
            model_name=model_name,
            work_dir=work_dir,
            max_steps=max_steps or settings.agent.max_steps,
            verbose=verbose,
            clear_dir=clear_dir,
            load_path=None,
        )
    finally:
        os.chdir(previous_cwd)

    found = _inspect_outputs(work_dir)
    runtime = 0.0
    log = work_dir / "log.jsonl"
    if log.exists():
        import json

        try:
            runtime = json.loads(log.read_text().strip().splitlines()[-1]).get("runtime", 0.0)
        except (ValueError, IndexError):
            pass

    return RunResult(
        compound=compound,
        spacegroup=spacegroup,
        work_dir=work_dir,
        completed=state.done,
        num_steps=len(state.history),
        runtime_seconds=runtime,
        model=model_name,
        summary=state.final_result,
        scf_converged=found.get("scf_converged", False),
        dft_finished=found.get("dft_finished", False),
        n_atoms=found.get("n_atoms"),
        n_electrons=found.get("n_electrons"),
        band_gap=found.get("band_gap"),
        plot=found.get("plot"),
        state=state,
    )
