import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CALC = 'calculations/000'
GNU = f'{CALC}/000.bands.dat.gnu'
LABELS = f'{CALC}/band_labels.dat'
PWOUT = f'{CALC}/pw.out'

# --- occupied bands from pw.out (INVARIANT 2): occupied = number of electrons / 2
# for a spin-unpolarized non-metallic run. Do NOT hardcode from a per-element guess.
nelec = None
with open(PWOUT) as f:
    for line in f:
        m = re.search(r'number of electrons\s*=\s*([0-9.]+)', line)
        if m:
            nelec = float(m.group(1))
            break
assert nelec is not None, 'could not read number of electrons from pw.out'
n_occ = int(round(nelec / 2))   # 24 for TiO2 primitive with semicore Ti (Zval=12)
print(f'number of electrons = {nelec}, occupied bands = {n_occ}')

# --- parse .gnu: blank-line separated blocks, each = one band.
# col0 = cumulative k-path distance (2*pi/alat units), col1 = energy (eV).
blocks = []
cur = []
with open(GNU) as f:
    for line in f:
        s = line.strip()
        if not s:
            if cur:
                blocks.append(cur)
                cur = []
            continue
        parts = s.split()
        cur.append((float(parts[0]), float(parts[1])))
    if cur:
        blocks.append(cur)

nbands = len(blocks)
nkpts = len(blocks[0])
for b in blocks:
    assert len(b) == nkpts, 'inconsistent nkpts across bands'

# k distance column (same for every band) and energies (nbands, nkpts)
k_col = np.array([p[0] for p in blocks[0]])
ener = np.array([[p[1] for p in b] for b in blocks])  # (nbands, nkpts)
print(f'nbands={nbands}, nkpts={nkpts}')

# --- VBM/CBM alignment using occupied-band count
# 0-based: highest occupied band = index n_occ-1, lowest unoccupied = n_occ
vbm = ener[n_occ - 1, :].max()
cbm = ener[n_occ, :].min()
gap = cbm - vbm
print(f'VBM = {vbm:.4f} eV, CBM = {cbm:.4f} eV, band gap = {gap:.4f} eV')

# --- high-symmetry ticks: band_labels.dat gives POINT INDICES; the .gnu x-axis is
# cumulative DISTANCE (INVARIANT 4). Map each index -> its distance k_col[index].
label_entries = []  # (index, label)
with open(LABELS) as f:
    for line in f:
        s = line.split()
        if len(s) >= 2:
            label_entries.append((int(s[0]), s[1]))

# Merge discontinuities: consecutive indices at the same distance -> 'A|B' tick.
tick_pos = []
tick_lab = []
i = 0
while i < len(label_entries):
    idx, lab = label_entries[i]
    if (i + 1 < len(label_entries)
            and label_entries[i + 1][0] == idx + 1
            and abs(k_col[label_entries[i + 1][0]] - k_col[idx]) < 1e-9):
        # discontinuity (jump): distance repeats -> combine both labels
        lab2 = label_entries[i + 1][1]
        tick_pos.append(k_col[idx])
        tick_lab.append(f'{lab}|{lab2}')
        i += 2
    else:
        tick_pos.append(k_col[idx])
        tick_lab.append(lab)
        i += 1

GREEK = {'GAMMA': r'$\Gamma$'}
tick_lab = [GREEK.get(l, l) for l in tick_lab]

# INVARIANT 4: tick positions and band data must span the same numeric range.
assert abs(max(tick_pos) - k_col.max()) < 0.05 * k_col.max(), \
    'tick positions and band k-column are in different unit systems'

# --- plot, aligning energies to VBM (=0)
fig, ax = plt.subplots(figsize=(6, 5))
# Split each band at discontinuities: bands.x marks a jump by repeating the
# k-distance value (zero increment) -> break the line there so nothing is drawn
# across the jump.
jump_idx = [j for j in range(1, nkpts) if abs(k_col[j] - k_col[j - 1]) < 1e-12]
seg_bounds = [0] + jump_idx + [nkpts]
for bidx in range(nbands):
    y = ener[bidx] - vbm
    for a, b in zip(seg_bounds[:-1], seg_bounds[1:]):
        ax.plot(k_col[a:b], y[a:b], color='b', lw=0.7)

ax.axhline(0.0, color='k', ls=':', lw=0.6)
for xp in tick_pos:
    ax.axvline(xp, color='gray', ls='-', lw=0.4)
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_lab)
ax.set_xlim(k_col.min(), k_col.max())
ax.set_ylim(-10, 10)  # +/-10 eV around VBM
ax.set_ylabel('E - E_VBM (eV)')
ax.set_title(f'Rutile TiO2 (struct 000)  gap = {gap:.3f} eV')
fig.tight_layout()
fig.savefig(f'{CALC}/bands_000.pdf')
print(f'saved {CALC}/bands_000.pdf')
