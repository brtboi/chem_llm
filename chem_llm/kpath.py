"""Standard band-structure k-paths, and one shared definition for QE and the plot.

The agent used to build k-paths by hand from seekpath's named points, and
most of its band-structure failures were bookkeeping: a segment missing its
start point after a discontinuity, labels missing or on the wrong point, plot
ticks that did not match the k-points QE computed, and many steps spent
fixing them. Here the whole path comes from one call, and both ends use it:

    from chem_llm.kpath import standard_kpath, plot_axis

    # setup_jobs.py -- per structure
    kp = standard_kpath(atoms)            # dict, also saved with kp_save()
    prim = kp_atoms(kp)                   # THE cell to write into pw.in and bands.in
    write_espresso_in(f, prim, input_data={"calculation": "bands", ...},
                      kpts=kp_kpts(kp))   # -> an explicit K_POINTS crystal card
    kp_save(kp, "calculations/000/kpath.json")

    # plot_bands.py
    ax = plot_axis("calculations/000/kpath.json", "calculations/000/000.bands.dat")
    x, E = ax["x"], ax["energies"]
    for s in ax["segments"]:              # slice objects: use them as x[s], E[s, band]
        plt.plot(x[s], E[s, band])
    plt.xticks(ax["tick_x"], ax["tick_labels"])

The path is seekpath's (Hinuma et al. convention), with points spaced evenly
in reciprocal space, so the x-axis is the true path length and breaks add no
distance. plot_axis refuses to run if the bands output was computed on a
different k-point list than the one in kpath.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["standard_kpath", "kp_atoms", "kp_kpts", "kp_save", "load_kpath", "plot_axis"]

DEFAULT_DENSITY = 0.025  # 1/A between consecutive k-points (incl. the 2*pi)


def _ticks(labels: list[str], x: list[float]) -> list[dict]:
    """One tick per distinct x; labels meeting at a discontinuity are joined
    as 'A|L' (both sit at the same x because a jump adds no distance)."""
    ticks: list[dict] = []
    for i, name in enumerate(labels):
        if not name:
            continue
        if ticks and abs(ticks[-1]["x"] - x[i]) < 1e-9:
            if name not in ticks[-1]["label"].split("|"):
                ticks[-1]["label"] += "|" + name
                ticks[-1]["indices"].append(i)
        else:
            ticks.append({"x": float(x[i]), "label": name, "indices": [i]})
    return ticks


def standard_kpath(atoms, density: float = DEFAULT_DENSITY, symprec: float = 0.01) -> dict:
    """The standard high-symmetry path for `atoms` (an ase.Atoms).

    Returns a JSON-serialisable dict:
      cell, symbols, scaled_positions  the primitive cell the path refers to --
                                       run pw.x AND bands.x on this cell
      kpoints       (n, 3) fractional coordinates of that cell's reciprocal basis
      x             cumulative path length (1/A); no distance across a break
      labels        per-point label ('' for interior points)
      ticks         [{"x", "label", "indices"}], breaks joined as "A|L"
      segments      [(start, stop)] index ranges to draw as continuous lines
      path          readable form, e.g. "GAMMA-M-K-GAMMA-A-L-H-A | L-M | H-K"
      spacegroup    international symbol and number
    """
    import seekpath

    import warnings

    cell = (np.asarray(atoms.cell[:]), atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    with warnings.catch_warnings():
        # low-symmetry (perturbed) cells trigger seekpath's EdgeCaseWarning
        # about near-90-degree angles; the path is still valid
        warnings.simplefilter("ignore")
        r = seekpath.get_explicit_k_path(cell, symprec=symprec, reference_distance=density)

    from ase.data import chemical_symbols

    labels = list(r["explicit_kpoints_labels"])
    x = [float(v) for v in r["explicit_kpoints_linearcoord"]]
    segments = [(int(a), int(b)) for a, b in r["explicit_segments"]]
    # merge segments that join up (shared endpoint) into continuous runs
    runs: list[list[int]] = []
    for a, b in segments:
        if runs and runs[-1][1] - 1 == a:
            runs[-1][1] = b
        else:
            runs.append([a, b])
    path = " | ".join(
        "-".join([labels[a]] + [labels[i] for i in range(a + 1, b) if labels[i]]) for a, b in runs
    )
    return {
        "cell": np.asarray(r["primitive_lattice"]).tolist(),
        "symbols": [chemical_symbols[z] for z in r["primitive_types"]],
        "scaled_positions": np.asarray(r["primitive_positions"]).tolist(),
        "kpoints": np.asarray(r["explicit_kpoints_rel"]).tolist(),
        "x": x,
        "labels": labels,
        "ticks": _ticks(labels, x),
        "segments": [tuple(s) for s in runs],
        "path": path,
        "spacegroup": f"{r['spacegroup_international']} (#{r['spacegroup_number']})",
        "density_per_angstrom": density,
    }


def kp_atoms(kp: dict):
    """The primitive cell of a k-path dict, as ase.Atoms (periodic)."""
    from ase import Atoms

    return Atoms(kp["symbols"], cell=kp["cell"], scaled_positions=kp["scaled_positions"], pbc=True)


def kp_kpts(kp: dict) -> np.ndarray:
    """The path's k-points as the (n, 4) array ase.io.espresso.write_espresso_in
    takes for an explicit K_POINTS crystal card (weights 1; a bands run
    ignores them)."""
    k = np.asarray(kp["kpoints"], dtype=float)
    return np.hstack([k, np.ones((len(k), 1))])


def kp_save(kp: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kp, indent=1))
    return path


def load_kpath(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _read_filband(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import re

    text = path.read_text()
    m = re.search(r"nbnd=\s*(\d+),\s*nks=\s*(\d+)", text)
    if not m:
        raise ValueError(f"{path}: not a bands.x filband file (no '&plot nbnd=..., nks=... /' header)")
    nbnd, nks = map(int, m.groups())
    vals = np.array(text.split("/", 1)[1].split(), dtype=float)
    if vals.size != nks * (3 + nbnd):
        raise ValueError(f"{path}: expected {nks * (3 + nbnd)} numbers, found {vals.size}")
    block = vals.reshape(nks, 3 + nbnd)
    return block[:, :3], block[:, 3:]          # k cartesian (2pi/alat), energies (eV)


def plot_axis(kpath_file: str | Path, bands_file: str | Path) -> dict:
    """Everything a band plot needs, guaranteed consistent with the QE run.

    bands_file is bands.x's primary filband output (e.g. 000.bands.dat).
    Returns {"x", "energies" (nk, nbnd, eV), "ticks", "tick_x", "tick_labels",
    "segments", "path"}. ticks are dicts ({"x", "label", "indices"}); tick_x and
    tick_labels are the same as two plain lists, ready for
    plt.xticks(ax["tick_x"], ax["tick_labels"]).
    segments are Python slice objects -- one continuous line each, used as
    x[s] and energies[s, band]. (Plain (start, stop) pairs were misread as
    inclusive by four of five agent scripts, which then drew a connector
    across each discontinuity.)
    Raises ValueError if the bands were not computed on kpath_file's k-points
    (different count, or different k-point lengths) -- i.e. if bands.in was not
    written from the same k-path.
    """
    kp = load_kpath(kpath_file)
    k_qe, energies = _read_filband(Path(bands_file))
    if len(k_qe) != len(kp["kpoints"]):
        raise ValueError(f"{bands_file} has {len(k_qe)} k-points but {kpath_file} has {len(kp['kpoints'])}: "
                         "bands.in was not written from this k-path")
    # rotation-invariant comparison: |k| of every point. QE writes cartesian
    # k in units of 2pi/alat, alat = |a1| of the cell it ran on.
    cell = np.asarray(kp["cell"])
    recip = 2 * np.pi * np.linalg.inv(cell).T
    k_ref = np.linalg.norm(np.asarray(kp["kpoints"]) @ recip, axis=1)
    k_run = np.linalg.norm(k_qe, axis=1) * 2 * np.pi / np.linalg.norm(cell[0])
    worst = float(np.max(np.abs(k_ref - k_run)))
    if worst > 1e-3:
        raise ValueError(f"the k-points in {bands_file} differ from {kpath_file} (max |dk| {worst:.4f} 1/A): "
                         "bands.in was not written from this k-path, or pw.x ran on a different cell")
    return {"x": np.asarray(kp["x"]), "energies": energies, "ticks": kp["ticks"],
            "tick_x": [t["x"] for t in kp["ticks"]], "tick_labels": [t["label"] for t in kp["ticks"]],
            "segments": [slice(a, b) for a, b in kp["segments"]], "path": kp["path"]}
