import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# plot_bands.py
# Reads the bands.x output (000.bands.dat) and band_labels.dat for calculations/000,
# aligns energies to the VBM, and plots the band structure.
# =============================================================================

# --- Read the number of electrons from pw.out to determine occupied bands ---
# INVARIANT 2: occupied bands = number of electrons / 2 for spin-unpolarized run
with open('calculations/000/pw.out') as f:
    pw_out = f.read()

m = re.search(r'number of electrons\s*=\s*([\d.]+)', pw_out)
assert m, "Could not find 'number of electrons' in pw.out"
n_electrons = float(m.group(1))
n_occ = int(n_electrons / 2)
print(f"Number of electrons: {n_electrons}, occupied bands: {n_occ}")

# --- Parse 000.bands.dat ---
# Format: header '&plot nbnd=48, nks=120 /', then nks blocks.
# Each block: 1 line with 3 cartesian k-coords (2*pi/alat), then lines with energies (eV).
# Total lines per block: 1 (k-coords) + ceil(nbnd/10) energy lines.
with open('calculations/000/000.bands.dat') as f:
    lines = f.readlines()

# Parse header
header = lines[0].strip()
m = re.search(r'nbnd=\s*(\d+),\s*nks=\s*(\d+)', header)
assert m, f"Could not parse header: {header}"
nbnd = int(m.group(1))
nks = int(m.group(2))
print(f"Parsed header: nbnd={nbnd}, nks={nks}")

# Parse k-coordinates and energies
k_coords = np.zeros((nks, 3))  # cartesian, 2*pi/alat
energies = np.zeros((nks, nbnd))  # eV

line_idx = 1
for ik in range(nks):
    # Read k-coordinates line
    k_line = lines[line_idx].strip()
    k_coords[ik] = np.array([float(x) for x in k_line.split()])
    line_idx += 1
    
    # Read energy lines until we have nbnd values
    e_vals = []
    while len(e_vals) < nbnd:
        e_line = lines[line_idx].strip()
        e_vals.extend([float(x) for x in e_line.split()])
        line_idx += 1
    energies[ik] = e_vals[:nbnd]

assert line_idx <= len(lines), f"Read past end of file: line_idx={line_idx}, total={len(lines)}"
print(f"Parsed {nks} k-points, {nbnd} bands each")

# --- Read band_labels.dat ---
# Format: "LABEL INDEX" per line
labels = []
with open('calculations/000/band_labels.dat') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            labels.append((parts[0], int(parts[1])))
print(f"Labels: {labels}")

# --- Compute cumulative path distance ---
# The k-coordinates are in cartesian (2*pi/alat). Compute Euclidean distance
# between consecutive k-points. At a discontinuity (path jump), add 0 distance.
# Detect discontinuities: the distance between consecutive k-points at a jump
# is much larger than the typical step size within a segment.

dists = np.zeros(nks - 1)
for i in range(nks - 1):
    dists[i] = np.linalg.norm(k_coords[i+1] - k_coords[i])

# Typical step size: median of all distances
median_step = np.median(dists)
print(f"Median step size: {median_step:.6f} (2*pi/alat)")

# Discontinuity: distance > 3x median step
# (within a segment, all steps are equal; a jump between different high-symmetry
#  points is typically much larger)
discont_mask = dists > 3.0 * median_step
discont_indices = np.where(discont_mask)[0]  # indices i where dists[i] is a jump
print(f"Discontinuities at k-point indices: {discont_indices.tolist()}")

# Build cumulative x-axis: add 0 distance at discontinuities
x = np.zeros(nks)
for i in range(nks - 1):
    if discont_mask[i]:
        x[i+1] = x[i]  # no distance added across the jump
    else:
        x[i+1] = x[i] + dists[i]

# INVARIANT 4 check: x values never decrease
assert np.all(np.diff(x) >= -1e-10), "x values decrease somewhere!"
print(f"x range: {x[0]:.4f} to {x[-1]:.4f} (2*pi/alat)")

# --- Align energies to VBM ---
# VBM = max of the highest occupied band (index n_occ-1, 0-based) over all k-points
vbm_band = n_occ - 1
vbm = np.max(energies[:, vbm_band])
print(f"VBM (raw): {vbm:.4f} eV (band index {vbm_band})")

# Subtract VBM from all energies
energies_rel = energies - vbm

# --- Find CBM and gap ---
# CBM = min of all conduction bands (indices n_occ to nbnd-1) over all k-points
cbm = np.min(energies_rel[:, n_occ:])
gap = cbm  # since VBM is at 0 after alignment
print(f"CBM (relative to VBM): {cbm:.4f} eV")
print(f"Band gap: {gap:.4f} eV")

# --- Build tick positions and labels ---
# Convert label indices to x positions (cumulative distance)
tick_positions = []
tick_labels = []

# Identify which label indices are at discontinuities
# A discontinuity at index i means k[i] and k[i+1] are not connected.
# The label at index i (end of previous segment) and index i+1 (start of next segment)
# should be combined as "LABEL1|LABEL2".

# Build a dict: index -> label
label_at_index = {}
for label, idx in labels:
    label_at_index[idx] = label

# Process labels in order
for i, (label, idx) in enumerate(labels):
    x_pos = x[idx]
    
    # Check if this label is at a discontinuity
    # A discontinuity at index idx means the jump is between k[idx] and k[idx+1]
    # So the label at idx is the END of the previous segment.
    # The next label (at idx+1 or later) is the START of the next segment.
    
    # Check if idx is a discontinuity point (end of a segment before a jump)
    if idx in discont_indices:
        # This label is at the end of a segment before a jump.
        # The next label should be at the start of the next segment.
        # Find the next label
        if i + 1 < len(labels):
            next_label, next_idx = labels[i+1]
            # Combine: "LABEL|NEXT_LABEL"
            tick_labels.append(f"{label}|{next_label}")
            tick_positions.append(x_pos)
            # Skip the next label (already combined)
            # We'll handle this by marking it as consumed
            # Actually, let's just add both and let the plot handle it
            # For now, add the combined label and skip the next one
            # We need to be careful not to double-add
            # Let's use a different approach: build the list first, then merge
            pass
        else:
            tick_labels.append(label)
            tick_positions.append(x_pos)
    else:
        tick_labels.append(label)
        tick_positions.append(x_pos)

# The above approach is messy. Let's redo it properly.
# Build a list of (x_pos, label, is_discont_end, is_discont_start)
# Then merge adjacent labels at discontinuities.

tick_data = []
for label, idx in labels:
    x_pos = x[idx]
    is_discont_end = idx in discont_indices  # this label is at the end of a segment before a jump
    # is_discont_start: this label is at the start of a segment after a jump
    # A jump at index i means k[i] -> k[i+1] is a jump.
    # The label at i+1 (if any) is the start of the next segment.
    is_discont_start = (idx - 1) in discont_indices  # the previous index is a jump
    tick_data.append((x_pos, label, is_discont_end, is_discont_start))

# Now merge: if a label is is_discont_end and the next label is is_discont_start,
# combine them as "LABEL1|LABEL2" at the same x position.
merged_ticks = []
i = 0
while i < len(tick_data):
    x_pos, label, is_end, is_start = tick_data[i]
    if is_end and i + 1 < len(tick_data):
        next_x, next_label, next_is_end, next_is_start = tick_data[i+1]
        if next_is_start and abs(x_pos - next_x) < 1e-6:
            # Combine
            merged_ticks.append((x_pos, f"{label}|{next_label}"))
            i += 2
            continue
    merged_ticks.append((x_pos, label))
    i += 1

tick_positions = [t[0] for t in merged_ticks]
tick_labels = [t[1] for t in merged_ticks]
print(f"Tick positions: {tick_positions}")
print(f"Tick labels: {tick_labels}")

# INVARIANT 4 check: tick positions span the same range as the band data x column
assert abs(max(tick_positions) - x[-1]) < 0.05 * x[-1], \
    f"Tick positions ({max(tick_positions)}) don't span the same range as x data ({x[-1]})"
assert abs(min(tick_positions) - x[0]) < 0.05 * max(x[-1], 1e-10), \
    f"Tick positions start ({min(tick_positions)}) doesn't match x data start ({x[0]})"

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each band, breaking the line at discontinuities
# For each band, plot segments between discontinuities
for ib in range(nbnd):
    e = energies_rel[:, ib]
    # Find segments: split at discontinuities
    # A discontinuity at index i means no line between k[i] and k[i+1]
    # So segments are: [0..i], [i+1..j], [j+1..], etc.
    
    # Build segment boundaries
    seg_starts = [0]
    for i in discont_indices:
        seg_starts.append(i + 1)
    seg_ends = list(discont_indices) + [nks - 1]
    
    for s, e_idx in zip(seg_starts, seg_ends):
        ax.plot(x[s:e_idx+1], e[s:e_idx+1], 'b-', linewidth=0.5)

# Add tick marks
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

# Restrict y-axis to +/-10 eV around VBM
ax.set_ylim(-10, 10)
ax.set_xlabel('k-point (2$\\pi$/a)')
ax.set_ylabel('Energy (eV)')
ax.set_title('GaAs band structure (PBEsol, 300 K)')
ax.axhline(0, color='k', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.savefig('calculations/000/bands_000.pdf', bbox_inches='tight')
print("Saved plot to calculations/000/bands_000.pdf")

# --- Print summary ---
print(f"\n=== Summary ===")
print(f"VBM: {vbm:.4f} eV (raw)")
print(f"CBM: {vbm + cbm:.4f} eV (raw)")
print(f"Band gap: {gap:.4f} eV")
print(f"Number of k-points: {nks}")
print(f"Number of bands: {nbnd}")
print(f"Discontinuities: {discont_indices.tolist()}")
