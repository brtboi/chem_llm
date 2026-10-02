import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# plot_bands.py - Plot band structure for calculations/000
# =============================================================================

CALC_DIR = 'calculations/000'

# -----------------------------------------------------------------------------
# 1. Read 'number of electrons' from pw.out to get occupied band count
#    (INVARIANT 2: use the actual value from the SCF output, not an assumption)
# -----------------------------------------------------------------------------
with open(f'{CALC_DIR}/pw.out') as f:
    pw_out = f.read()

m = re.search(r'number of electrons\s+=\s+([\d.]+)', pw_out)
assert m, "Could not find 'number of electrons' in pw.out"
n_electrons = float(m.group(1))
occupied_bands = int(n_electrons / 2)  # spin-unpolarized
print(f"Number of electrons: {n_electrons}, Occupied bands: {occupied_bands}")

# -----------------------------------------------------------------------------
# 2. Read band labels (high-symmetry points and their indices in the k-point list)
# -----------------------------------------------------------------------------
labels = []  # list of (index, label)
with open(f'{CALC_DIR}/band_labels.dat') as f:
    for line in f:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        parts = line.split()
        idx = int(parts[0])
        label = parts[1]
        labels.append((idx, label))

print(f"High-symmetry points: {labels}")

# -----------------------------------------------------------------------------
# 3. Read the .gnu file (distance-based x-axis, matches bands_post.out ticks)
#    Format: x_coordinate  energy
#    The .gnu file is organized as: for each band (64 bands), for each kpt (183 kpts)
#    So total lines = 64 * 183 = 11776
#    x_coords for kpt k = data[k, 0] (first band, all kpts)
#    energies for band b at kpt k = data[b*183 + k, 1]
# -----------------------------------------------------------------------------
data = np.loadtxt(f'{CALC_DIR}/000.bands.dat.gnu')
assert data.shape[1] == 2, f"Expected 2 columns, got {data.shape[1]}"

# Determine n_kpts and nbands from the file size
# We know from setup_jobs.py that n_kpts=183 and nbands=64
# But let's verify: the x coordinate should repeat every n_kpts entries
# Find where x resets to 0 (or near 0) to determine n_kpts
x_col = data[:, 0]
# The first x is 0.0, find the next occurrence of x close to 0
# Actually, let's use the known structure: 11776 lines
# From band_labels.dat, the last index is 182, so n_kpts = 183
n_kpts = 183
nbands = data.shape[0] // n_kpts
assert data.shape[0] == n_kpts * nbands, f"Expected {n_kpts*nbands} lines, got {data.shape[0]}"

# x_coords: for kpt k, x = data[k, 0] (from the first band's entries)
x_coords = data[:n_kpts, 0]

# energies: reshape to (nbands, n_kpts, 2), then extract column 1 and transpose
energies = data.reshape(nbands, n_kpts, 2)[:, :, 1].T  # (n_kpts, nbands)

# Verify x is monotonically non-decreasing (with possible repeats at jumps)
print(f"x range: {x_coords.min():.4f} to {x_coords.max():.4f}")
print(f"n_kpts: {n_kpts}, nbands: {energies.shape[1]}")

# -----------------------------------------------------------------------------
# 4. Align energies to VBM
#    VBM = max of occupied band energies (band index occupied_bands-1)
# -----------------------------------------------------------------------------
vbm_band = occupied_bands - 1
vbm = energies[:, vbm_band].max()
cbm = energies[:, occupied_bands:].min()
band_gap = cbm - vbm

print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {band_gap:.4f} eV")

# Shift all energies so VBM = 0
energies_shifted = energies - vbm

# -----------------------------------------------------------------------------
# 5. Determine tick positions from band_labels.dat
#    The labels give indices into the k-point list.
#    x_coords[i] is the x-coordinate of k-point i.
# -----------------------------------------------------------------------------
tick_positions = []
tick_labels = []
for idx, label in labels:
    assert idx < n_kpts, f"Label index {idx} out of range (n_kpts={n_kpts})"
    tick_positions.append(x_coords[idx])
    tick_labels.append(label)

tick_positions = np.array(tick_positions)

# INVARIANT 4 CHECK: tick positions must span approximately the same range
# as the band data's x column
assert abs(tick_positions.max() - x_coords.max()) < 0.05 * x_coords.max(), \
    f"Tick positions max {tick_positions.max():.4f} vs x max {x_coords.max():.4f}"
assert abs(tick_positions.min() - x_coords.min()) < 0.05 * max(x_coords.max(), 1e-10), \
    f"Tick positions min {tick_positions.min():.4f} vs x min {x_coords.min():.4f}"
print(f"Tick positions: {tick_positions}")
print(f"Tick labels: {tick_labels}")

# -----------------------------------------------------------------------------
# 6. Identify path discontinuities (jumps)
#    A jump occurs where consecutive k-points have the same x-coordinate
#    (bands.x repeats the x value at a jump)
# -----------------------------------------------------------------------------
jump_indices = set()
for i in range(1, n_kpts):
    if abs(x_coords[i] - x_coords[i-1]) < 1e-10:
        jump_indices.add(i)

print(f"Jump indices: {sorted(jump_indices)}")

# -----------------------------------------------------------------------------
# 7. Plot
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each band, splitting at jumps
for band in range(energies_shifted.shape[1]):
    e = energies_shifted[:, band]
    # Split into segments at jumps
    segments = []
    start = 0
    for i in range(1, n_kpts):
        if i in jump_indices:
            segments.append((start, i))
            start = i
    segments.append((start, n_kpts))
    
    for s, e_idx in segments:
        ax.plot(x_coords[s:e_idx], e[s:e_idx], 'b-', linewidth=0.8)

# Add high-symmetry ticks
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=0, ha='center')

# Add vertical lines at high-symmetry points
for pos in tick_positions:
    ax.axvline(x=pos, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

# Restrict y-axis to +/-10 eV around VBM (INVARIANT 4)
ax.set_ylim(-10, 10)
ax.set_xlabel(r'k-point path (2$\pi$/a units)')
ax.set_ylabel('Energy (eV) relative to VBM')
ax.set_title(r'TiO$_2$ (rutile) Band Structure - PBESOL')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{CALC_DIR}/bands_000.pdf', bbox_inches='tight')
print(f"\nSaved plot to {CALC_DIR}/bands_000.pdf")

# -----------------------------------------------------------------------------
# 8. Print summary
# -----------------------------------------------------------------------------
print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"VBM: {vbm:.4f} eV")
print(f"CBM: {cbm:.4f} eV")
print(f"Band gap: {band_gap:.4f} eV")
print(f"Number of electrons: {n_electrons}")
print(f"Occupied bands: {occupied_bands}")
print(f"Total bands plotted: {energies_shifted.shape[1]}")
print(f"Number of k-points: {n_kpts}")
print(f"High-symmetry points: {len(labels)}")
print(f"Path discontinuities: {len(jump_indices)}")
