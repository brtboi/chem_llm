"""Facts from a pw.x output file, for the scripts the agent writes.

Agent scripts kept failing on hand-written regexes over pw.out (the Fermi
energy line, 'number of electrons', spin-orbit band counting). Same idea as
chem_llm.kpath: one tested reader instead of a new parser per run.

    from chem_llm.qe_output import read_pw_out, band_edges
    pw = read_pw_out("calculations/000/pw.out")
    pw["n_electrons"], pw["n_occupied"], pw["fermi_energy_eV"]
    edges = band_edges(E, pw["n_occupied"])     # E: (n_kpoints, n_bands) in eV
    edges["vbm"], edges["cbm"], edges["gap"]
"""
from __future__ import annotations

import re
from pathlib import Path

_F = r"[-+]?\d+\.?\d*(?:[eEdD][-+]?\d+)?"


def _last(pattern: str, text: str, cast=float):
    found = re.findall(pattern, text)
    return cast(found[-1]) if found else None


def read_pw_out(path: str | Path) -> dict:
    """Everything a band-structure script needs from pw.out. Energies in eV.

    n_electrons        'number of electrons' (the pseudopotentials' valence)
    n_occupied         occupied bands: n_electrons / 2, or n_electrons with
                       spin-orbit / noncollinear (one electron per band)
    noncollinear, spin_orbit, spin_polarized (nspin = 2)
    n_bands            'number of Kohn-Sham states'
    fermi_energy_eV    smearing runs (None for fixed occupations)
    highest_occupied_eV, lowest_unoccupied_eV   fixed-occupation runs
    total_magnetization  Bohr magnetons per cell (spin-polarized runs)
    converged, job_done, n_atoms, alat_bohr
    """
    text = Path(path).read_text(errors="replace")
    # pw.out says "Non magnetic calculation with spin-orbit" for an SOC run
    # (never "noncollinear"), and spin-orbit in QE is always noncollinear.
    soc = "with spin-orbit" in text.lower()
    noncolin = soc or bool(re.search(r"(?i)non-?colin|noncollinear", text))
    lsda = bool(re.search(r"(?i)lsda|spin-polarized|nspin\s*=\s*2", text))
    n_el = _last(rf"number of electrons\s*=\s*({_F})", text)
    out = {
        "n_electrons": n_el,
        "n_occupied": None if n_el is None else int(round(n_el if noncolin else n_el / 2)),
        "noncollinear": noncolin,
        "spin_orbit": soc,
        "spin_polarized": lsda and not noncolin,
        "n_bands": _last(r"number of Kohn-Sham states\s*=\s*(\d+)", text, int),
        "n_atoms": _last(r"number of atoms/cell\s*=\s*(\d+)", text, int),
        "alat_bohr": _last(rf"lattice parameter \(alat\)\s*=\s*({_F})", text),
        "fermi_energy_eV": _last(rf"the Fermi energy is\s+({_F})\s*ev", text),
        "highest_occupied_eV": None,
        "lowest_unoccupied_eV": None,
        "total_magnetization": _last(rf"total magnetization\s*=\s*({_F})", text),
        "converged": "convergence has been achieved" in text,
        "job_done": "JOB DONE" in text,
    }
    both = re.findall(rf"highest occupied, lowest unoccupied level \(ev\):\s*({_F})\s+({_F})", text)
    if both:
        out["highest_occupied_eV"], out["lowest_unoccupied_eV"] = map(float, both[-1])
    else:
        out["highest_occupied_eV"] = _last(rf"highest occupied level \(ev\):\s*({_F})", text)
    return out


def band_edges(energies, n_occupied: int) -> dict:
    """VBM, CBM and gap (eV) from band energies of shape (n_kpoints, n_bands),
    e.g. plot_axis(...)['energies'], given the occupied-band count."""
    import numpy as np

    E = np.asarray(energies, dtype=float)
    if E.ndim != 2 or not 0 < n_occupied < E.shape[1]:
        raise ValueError(f"need 0 < n_occupied < n_bands; got n_occupied={n_occupied}, energies shape {E.shape}")
    vbm, cbm = float(E[:, n_occupied - 1].max()), float(E[:, n_occupied].min())
    return {"vbm": vbm, "cbm": cbm, "gap": cbm - vbm,
            "vbm_k": int(E[:, n_occupied - 1].argmax()), "cbm_k": int(E[:, n_occupied].argmin())}
