#!/usr/bin/env python3
"""Band structure of cubic CsPbBr3 (PBEsol + spin-orbit) from calculations/000.

Hand-written replacement for the agent's plot_bands.py (kept as
plot_bands_agent.py), which read the k-point lines of 000.bands.dat as path
distances. In that file each k-point block starts with the k-point's
CARTESIAN coordinates (kx ky kz, 2pi/alat); the distance along the path is
computed here from consecutive k-points.

- Occupied bands come from pw.out: with spin-orbit (noncollinear) every
  band holds one electron, so n_occ = number of electrons.
- High-symmetry ticks are found from the k-point coordinates (segment
  endpoints, named by their simple-cubic coordinates), not from
  band_labels.dat.
- No line is drawn across the path discontinuity, and that tick is
  labelled with both sides ("X|R").
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CALC = Path(__file__).resolve().parent / "calculations" / "000"
EWINDOW = (-6.0, 6.0)  # eV around the VBM


def read_filband(path):
    text = path.read_text()
    nbnd, nks = map(int, re.search(r"nbnd=\s*(\d+),\s*nks=\s*(\d+)", text).groups())
    vals = np.array(text.split("/", 1)[1].split(), dtype=float)
    stride = 3 + nbnd
    assert vals.size == nks * stride, f"expected {nks * stride} numbers, found {vals.size}"
    block = vals.reshape(nks, stride)
    return block[:, :3], block[:, 3:]          # k (cartesian, 2pi/alat), E (eV)


def occupied_bands(pw_out):
    text = pw_out.read_text(errors="replace")
    electrons = float(re.search(r"number of electrons\s*=\s*([\d.]+)", text).group(1))
    soc = bool(re.search(r"spin-orbit|noncollinear", text, re.I))
    n = electrons if soc else electrons / 2
    assert abs(n - round(n)) < 1e-6, f"non-integer occupied-band count {n}"
    return int(round(n)), electrons, soc


def path_axis(k):
    """Cumulative distance; a jump much larger than the typical step is a
    discontinuity and adds no distance."""
    steps = np.linalg.norm(np.diff(k, axis=0), axis=1)
    typical = np.median(steps[steps > 1e-9])
    breaks = np.where(steps > 3 * typical)[0] + 1      # index of the first point after a jump
    steps[breaks - 1] = 0.0
    x = np.concatenate([[0.0], np.cumsum(steps)])
    assert np.all(np.diff(x) >= -1e-12), "x-axis must never decrease"
    return x, list(breaks)


def name_point(kv):
    """Simple-cubic high-symmetry point from its coordinates (2pi/a)."""
    a = sorted(abs(c) % 1.0 for c in kv)
    a = [min(c, 1 - c) for c in a]
    half = sum(abs(c - 0.5) < 1e-6 for c in a)
    zero = sum(c < 1e-6 for c in a)
    if zero == 3:
        return "$\\Gamma$"
    if half == 1 and zero == 2:
        return "X"
    if half == 2 and zero == 1:
        return "M"
    if half == 3:
        return "R"
    return None


def ticks(k, x, breaks):
    """(position, label) for every named point the path passes through."""
    marks = {}
    for i, kv in enumerate(k):
        name = name_point(kv)
        if name is None:
            continue
        marks.setdefault(round(x[i], 9), [])
        if name not in marks[round(x[i], 9)]:
            # the point before a break comes first in the "A|B" label
            marks[round(x[i], 9)].append(name)
    return sorted((pos, "|".join(names)) for pos, names in marks.items())


def main():
    k, E = read_filband(CALC / "000.bands.dat")
    n_occ, electrons, soc = occupied_bands(CALC / "pw.out")
    assert E.shape[1] > n_occ, f"need more than {n_occ} bands, have {E.shape[1]}"

    vbm_i = int(E[:, n_occ - 1].argmax())
    cbm_i = int(E[:, n_occ].argmin())
    vbm, cbm = E[vbm_i, n_occ - 1], E[cbm_i, n_occ]
    gap = cbm - vbm
    x, breaks = path_axis(k)
    tk = ticks(k, x, breaks)

    print(f"{electrons:g} electrons, spin-orbit {'on' if soc else 'off'} -> {n_occ} occupied bands of {E.shape[1]}")
    print(f"VBM {vbm:.4f} eV at k={k[vbm_i]} ({name_point(k[vbm_i])}), CBM {cbm:.4f} eV at k={k[cbm_i]} ({name_point(k[cbm_i])})")
    print(f"band gap {gap:.4f} eV ({'direct' if vbm_i == cbm_i else 'indirect'})")
    print("ticks:", [(round(p, 4), l) for p, l in tk])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    segs = np.split(np.arange(len(x)), breaks)
    Er = E - vbm
    for b in range(E.shape[1]):
        color = "tab:blue" if b < n_occ else "tab:red"
        for s in segs:
            ax.plot(x[s], Er[s, b], color=color, lw=1.0)
    for pos, _ in tk:
        ax.axvline(pos, color="0.6", lw=0.6)
    ax.axhline(0.0, color="k", lw=0.6, ls="--")
    ax.set_xticks([p for p, _ in tk])
    ax.set_xticklabels([l for _, l in tk])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*EWINDOW)
    ax.set_ylabel("Energy relative to VBM (eV)")
    ax.set_title(f"Cubic CsPbBr$_3$, PBEsol + SOC: gap {gap:.2f} eV "
                 f"({'direct' if vbm_i == cbm_i else 'indirect'} at {name_point(k[vbm_i])})")
    fig.tight_layout()
    out = CALC / "bands_000.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    print("saved", out)


if __name__ == "__main__":
    sys.exit(main())
