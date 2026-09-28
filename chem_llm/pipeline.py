"""Run bookkeeping shared by DFTAgent, plus a one-shot convenience wrapper.

    from chem_llm import run_dft_agent

    result = run_dft_agent("TiO2", "P4_2/mnm", spacegroup_name="rutile")
    print(result.completed, result.band_gap, result.work_dir)

RunResult and the output inspection (_inspect_outputs, _band_gap) read what
a run actually left on disk rather than trusting the agent's own summary.
For more than one run, use chem_llm.DFTAgent directly so the model is
loaded once.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import Settings
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
    """One-shot convenience wrapper: build a DFTAgent from `settings`
    (default: DFTAgent.default_settings()), run one compound, throw the agent away.

    Fine for a single run. For several, build the agent yourself and call
    `agent.run()` per compound -- that loads the model once instead of once
    per compound, which for a 27B local model is minutes apiece:

        agent = DFTAgent(settings)
        for compound, spacegroup in jobs:
            agent.run(compound, spacegroup)
    """
    from .agent import DFTAgent

    agent = DFTAgent(settings)
    return agent.run(
        compound,
        spacegroup,
        spacegroup_number=spacegroup_number,
        spacegroup_name=spacegroup_name,
        work_dir=agent.settings.work_root / (run_name or compound),
        clear_dir=clear_dir,
        max_steps=max_steps,
        n_structures=n_structures,
        relativistic=relativistic,
        functional=functional,
        task=task,
        hide_tools=hide_tools,
        verbose=verbose,
    )
