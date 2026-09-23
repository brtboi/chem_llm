"""Step 6 of tio2_prompt.md: VBM-aligned bands plot for calculations/000.

N_OCC is read from the actual SCF pw.out's "number of electrons" line
(never assumed from a memorized per-element valence-electron count -- the
Ti/O pseudopotentials used here are semicore: z_valence=12 for Ti,
z_valence=6 for O, giving 48 electrons / 24 occupied bands for this 6-atom
cell, not the naive 4+4*6=... guess a textbook valence count would give).

The .gnu file's x-column is bands.x's own cumulative k-path distance (in
2*pi/alat units), NOT a plain point index -- tick positions for the
high-symmetry labels are read off that same x-column at the exact k-point
indices where each label sits (from the K_POINTS crystal_b block used to
run bands.in: 7 segments of 20 points each, Gamma-X-M-Gamma-Z-R-A-Z, plus
the final endpoint), rather than assumed to be evenly spaced.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CALC_DIR = "calculations/000"

# --- N_OCC from the actual pw.out ---
pw_out = open(f"{CALC_DIR}/pw.out").read()
m = re.search(r"number of electrons\s*=\s*([\d.]+)", pw_out)
n_electrons = float(m.group(1))
assert n_electrons == int(n_electrons) and int(n_electrons) % 2 == 0
N_OCC = int(n_electrons) // 2
print(f"N_OCC (from pw.out: {n_electrons} electrons) = {N_OCC}")

# --- parse .gnu (columns: k-distance, energy; blocks separated by blank lines, one block per band) ---
gnu_path = f"{CALC_DIR}/000.bands.dat.gnu"
blocks, current = [], []
for line in open(gnu_path):
    if line.strip() == "":
        if current:
            blocks.append(current)
            current = []
    else:
        current.append([float(x) for x in line.split()])
if current:
    blocks.append(current)

bands = []
kdist = None
for block in blocks:
    rows = np.array(block)
    kdist = rows[:, 0]
    bands.append(rows[:, 1])
bands = np.array(bands)  # shape (nbnd, nk)
nbnd, nk = bands.shape
print(f"Parsed {nbnd} bands x {nk} k-points from {gnu_path}")

VBM = bands[:N_OCC].max()
CBM = bands[N_OCC:].min()
print(f"VBM = {VBM:.4f} eV, CBM = {CBM:.4f} eV, gap = {CBM - VBM:.4f} eV (raw, before shift)")

bands_shifted = bands - VBM

# --- high-symmetry tick positions, read from bands.in's own K_POINTS crystal_b block ---
bands_in = open(f"{CALC_DIR}/bands.in").read()
kpts_block = re.search(r"K_POINTS crystal_b\s*\n\s*(\d+)\s*\n((?:.*\n?)+)", bands_in)
n_labels = int(kpts_block.group(1))
kpt_lines = kpts_block.group(2).strip().splitlines()[:n_labels]
divisions = [int(line.split()[3]) for line in kpt_lines]

label_names = [r"$\Gamma$", "X", "M", r"$\Gamma$", "Z", "R", "A", "Z"]
assert len(label_names) == n_labels, (len(label_names), n_labels)

tick_indices = [0]
for d in divisions[:-1]:
    tick_indices.append(tick_indices[-1] + d)
assert tick_indices[-1] == nk - 1, (tick_indices, nk)
tick_positions = kdist[tick_indices]

fig, ax = plt.subplots(figsize=(7, 6))
for b in range(nbnd):
    ax.plot(kdist, bands_shifted[b], color="tab:blue" if b < N_OCC else "tab:red", lw=1.0)

ax.axhline(0.0, color="k", ls="--", lw=0.7)
for x in tick_positions[1:-1]:
    ax.axvline(x, color="gray", lw=0.5)

ax.set_xticks(tick_positions)
ax.set_xticklabels(label_names)
ax.set_xlim(kdist[0], kdist[-1])
ax.set_ylim(-10, 10)
ax.set_ylabel("Energy (eV, VBM = 0)")
ax.set_title(f"Rutile TiO$_2$ (mp-2657) — representative structure_000\nDFT gap = {CBM - VBM:.2f} eV (PBEsol, occ. bands blue, unocc. red)")
fig.tight_layout()
fig.savefig(f"{CALC_DIR}/bands_000.pdf")
fig.savefig(f"{CALC_DIR}/bands_000.png", dpi=150)
print(f"Wrote {CALC_DIR}/bands_000.pdf and .png")
