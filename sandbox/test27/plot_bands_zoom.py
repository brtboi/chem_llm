#!/usr/bin/env python
"""Band-structure plot for calculations/000 zoomed to the gap region.

Separate from the agent-written plot_bands.py (left untouched); this one
narrows the y-range to +/-10 eV so the O 2p valence manifold and the Ti 3d
conduction bands are readable instead of being compressed by the Ti 3s/3p
and O 2s semicore states at -57/-34/-17 eV.

Also handles the two discontinuities in rutile's seekpath path
(... A->Z | X->R | M->A): bands.x zeroes the distance increment across a
jump, so a break shows up as a repeated x value. Curves are split there and
the tick is labelled with both sides (Z|X, R|M).
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seekpath
from ase.io import read

CALC_DIR = "calculations/000"
GNU = f"{CALC_DIR}/000.bands.dat.gnu"
LABELS = f"{CALC_DIR}/band_labels.dat"
PW_OUT = f"{CALC_DIR}/pw.out"
STRUCTURE = "structures/structure_000.cif"
OUT_PDF = f"{CALC_DIR}/bands_000_zoom.pdf"

YMIN, YMAX = -10.0, 10.0

_GREEK = {"GAMMA": r"$\Gamma$", "SIGMA": r"$\Sigma$", "DELTA": r"$\Delta$"}


def latex(label):
    return "|".join(_GREEK.get(p, p) for p in label.split("|"))


def read_gnu(path):
    """bands.x .gnu output is gnuplot block format: one blank-line-separated
    block per band, each block (k_distance, energy) per k-point."""
    blocks, current = [], []
    for line in open(path):
        if not line.strip():
            if current:
                blocks.append(np.array(current))
                current = []
            continue
        k, e = line.split()
        current.append([float(k), float(e)])
    if current:
        blocks.append(np.array(current))
    k_col = blocks[0][:, 0]
    energies = np.column_stack([b[:, 1] for b in blocks])
    return k_col, energies


def occupied_bands(pw_out):
    """Occupied-band count from the actual SCF output, never from a
    per-element valence guess."""
    text = open(pw_out).read()
    n_ks = re.search(r"number of Kohn-Sham states\s*=\s*(\d+)", text)
    n_el = re.search(r"number of electrons\s*=\s*([\d.]+)", text)
    if n_ks is None:
        raise ValueError(f"no 'number of Kohn-Sham states' in {pw_out}")
    n_occ = int(n_ks.group(1))
    if n_el:
        assert abs(n_occ - float(n_el.group(1)) / 2) < 0.5, "n_ks != n_electrons/2"
    return n_occ


def ticks(k_col, labels_file, structure_file):
    """band_labels.dat records each segment's END label only, so a path break
    (previous segment ends at Z, next starts at X) is invisible there.
    Recompute the same seekpath path to recover both sides of each break."""
    recorded = [(ln.split()[0], int(ln.split()[1])) for ln in open(labels_file) if ln.strip()]
    atoms = read(structure_file)
    cell = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    path = seekpath.get_path(cell, symprec=0.01)["path"]
    assert len(recorded) == len(path) + 1, "band_labels.dat does not match the seekpath path"

    positions = [k_col[recorded[0][1]]]
    names = [path[0][0]]
    for i in range(1, len(path)):
        positions.append(k_col[recorded[i][1]])
        prev_end, start = path[i - 1][1], path[i][0]
        names.append(start if prev_end == start else f"{prev_end}|{start}")
    positions.append(k_col[recorded[-1][1]])
    names.append(path[-1][1])
    return positions, names


def main():
    k_col, energies = read_gnu(GNU)
    n_occ = occupied_bands(PW_OUT)

    vbm = energies[:, n_occ - 1].max()
    cbm = energies[:, n_occ].min()
    print(f"VBM = {vbm:.4f} eV   CBM = {cbm:.4f} eV   gap = {cbm - vbm:.4f} eV")

    e = energies - vbm
    tick_x, tick_names = ticks(k_col, LABELS, STRUCTURE)

    # Ticks and curves must share x units: both come from the .gnu k column.
    assert np.isclose(tick_x[-1], k_col.max()), "tick/band x-axis unit mismatch"

    # A repeated x value marks a path discontinuity; split so no connector
    # line is drawn between unrelated k-points.
    breaks = [i for i in range(1, len(k_col)) if np.isclose(k_col[i], k_col[i - 1])]

    fig, ax = plt.subplots(figsize=(8, 5))
    for band in range(e.shape[1]):
        for piece_k, piece_e in zip(np.split(k_col, breaks), np.split(e[:, band], breaks)):
            ax.plot(piece_k, piece_e, color="tab:blue", lw=0.8)

    for x in tick_x:
        ax.axvline(x, color="0.6", lw=0.6)
    ax.axhline(0.0, color="black", lw=0.8)

    ax.set_xticks(tick_x)
    ax.set_xticklabels([latex(n) for n in tick_names])
    ax.set_xlim(k_col.min(), k_col.max())
    ax.set_ylim(YMIN, YMAX)
    ax.set_ylabel("Energy (eV)")
    ax.set_title(f"Rutile TiO$_2$ (calculations/000) — gap {cbm - vbm:.2f} eV")
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print("Wrote", OUT_PDF)


if __name__ == "__main__":
    main()
