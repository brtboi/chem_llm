import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# plot_bands.py
# Plots the band structure for calculations/000 from the bands.x output file
# (000.bands.dat), aligned to the VBM, with high-symmetry tick marks.
# =============================================================================

CALC_DIR = 'calculations/000'
BANDS_FILE = f'{CALC_DIR}/000.bands.dat'
LABELS_FILE = f'{CALC_DIR}/band_labels.dat'
PW_OUT = f'{CALC_DIR}/pw.out'
OUTPUT_PDF = f'{CALC_DIR}/bands_000.pdf'

# --- Read occupied band count from pw.out (INVARIANT 2) ---
with open(PW_OUT) as f:
    pw_content = f.read()

m = re.search(r'number of Kohn-Sham states\s*=\s*(\d+)', pw_content)
assert m, "Could not find 'number of Kohn-Sham states' in pw.out"
n_occ = int(m.group(1))
print(f"Occupied bands (from pw.out): {n_occ}")

m_e = re.search(r'number of electrons\s*=\s*([\d.]+)', pw_content)
assert m_e, "Could not find 'number of electrons' in pw.out"
n_elec = float(m_e.group(1))
print(f"Number of electrons: {n_elec}")
assert abs(n_elec - 2 * n_occ) < 1e-6, \
    f"Electron count {n_elec} != 2 * n_occ {2*n_occ}"

# --- Parse the bands.x output file (000.bands.dat) ---
with open(BANDS_FILE) as f:
    lines = f.readlines()

header = lines[0].strip()
m = re.search(r'nbnd=\s*(\d+),\s*nks=\s*(\d+)', header)
assert m, f"Could not parse header: {header}"
nbnd = int(m.group(1))
nks = int(m.group(2))
print(f"Header: nbnd={nbnd}, nks={nks}")

kpts_cart = []
energies = []

i = 1
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    parts = line.split()
    if len(parts) == 3:
        try:
            kx, ky, kz = float(parts[0]), float(parts[1]), float(parts[2])
            kpts_cart.append((kx, ky, kz))
            e_vals = []
            i += 1
            while len(e_vals) < nbnd and i < len(lines):
                e_line = lines[i].strip()
                if e_line:
                    e_vals.extend([float(x) for x in e_line.split()])
                i += 1
            assert len(e_vals) == nbnd, f"Expected {nbnd} energies, got {len(e_vals)}"
            energies.append(np.array(e_vals))
        except ValueError:
            i += 1
    else:
        i += 1

assert len(kpts_cart) == nks, f"Expected {nks} k-points, got {len(kpts_cart)}"
assert len(energies) == nks, f"Expected {nks} energy blocks, got {len(energies)}"
print(f"Parsed {nks} k-points, {nbnd} bands each")

# --- Read band labels ---
labels = []
with open(LABELS_FILE) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            labels.append((parts[0], int(parts[1])))

print(f"Band labels: {labels}")

# --- Identify segments from the labels ---
# The labels list contains (label, index) pairs. Each segment in the k-path
# contributes: start_label at start_idx, end_label at end_idx.
# Consecutive segments share a boundary: segment i's end_idx == segment i+1's start_idx
# for continuous paths, or they differ for discontinuous paths.
# 
# From setup_jobs.py, the path is built as:
#   for each segment (label_from, label_to):
#     append start point (label_from) at current index
#     append PTS_PER_SEG interior points
#     append end point (label_to) at current index
#
# So the labels list has 2 entries per segment: (label_from, start_idx), (label_to, end_idx)
# The segments are: [(start_idx, end_idx, label_from, label_to), ...]

# Group labels into pairs (each segment contributes exactly 2 labels)
assert len(labels) % 2 == 0, f"Expected even number of labels, got {len(labels)}"
segments = []
for j in range(0, len(labels), 2):
    label_from, idx_from = labels[j]
    label_to, idx_to = labels[j+1]
    segments.append((idx_from, idx_to, label_from, label_to))

print(f"\nSegments (start_idx, end_idx, label_from, label_to):")
for seg in segments:
    print(f"  {seg}")

# --- Compute cumulative path distance ---
# cum_dist[i] = cum_dist[i-1] + dists[i-1] for continuous steps
# cum_dist[i] = cum_dist[i-1] for discontinuities (no distance added across jump)
# This makes cum_dist monotonically non-decreasing and spanning the full path.

kpts_arr = np.array(kpts_cart)
dists = np.linalg.norm(np.diff(kpts_arr, axis=0), axis=1)

# Identify discontinuity indices: where a segment ends and the next begins
# A discontinuity is at index i if i is the start of a segment AND i > 0
# AND the distance from kpts[i-1] to kpts[i] is large (a jump)
# More robustly: use the segment boundaries directly.
# Segment j ends at segments[j][1], segment j+1 starts at segments[j+1][0].
# If segments[j][1] != segments[j+1][0] - 1, there's a gap (discontinuity).
# Actually, in our construction, each segment's start point is explicitly added,
# so segment j+1's start_idx = segment j's end_idx + 1 always.
# The discontinuity is detected by the large distance between kpts[end_j] and kpts[start_{j+1}].

cum_dist = np.zeros(nks)
for i in range(1, nks):
    cum_dist[i] = cum_dist[i-1] + dists[i-1]

# Now, for discontinuities, we need to NOT add the jump distance.
# Identify which consecutive k-point pairs are across a discontinuity.
# A pair (i-1, i) is across a discontinuity if i is the start of a segment
# and the distance is much larger than the typical step.

# Typical step size: median of all distances
median_step = np.median(dists)
print(f"Median step size: {median_step:.6f}")

# For each segment boundary, check if it's a discontinuity
# Segment j ends at idx = segments[j][1], segment j+1 starts at idx = segments[j+1][0]
# The distance between these two points is dists[segments[j][1]] (0-indexed: dists[i] = dist between kpts[i] and kpts[i+1])
# Wait: dists[i] = |kpts[i+1] - kpts[i]|, so the distance between kpts[segments[j][1]] and kpts[segments[j+1][0]]
# is dists[segments[j][1]] if segments[j+1][0] == segments[j][1] + 1.

# Let's just use the threshold approach but verify against segment boundaries.
threshold = 3.0 * median_step
is_discontinuity = dists > threshold
print(f"Discontinuities at indices (i means between kpts[i] and kpts[i+1]): {np.where(is_discontinuity)[0]}")

# Now recompute cum_dist, not adding distance across discontinuities
cum_dist = np.zeros(nks)
for i in range(1, nks):
    if is_discontinuity[i-1]:
        cum_dist[i] = cum_dist[i-1]  # no distance added across jump
    else:
        cum_dist[i] = cum_dist[i-1] + dists[i-1]

# Assert cum_dist is monotonically non-decreasing
for i in range(1, nks):
    assert cum_dist[i] >= cum_dist[i-1] - 1e-10, \
        f"Cumulative distance decreased at index {i}"

print(f"Cumulative distance range: {cum_dist.min():.4f} to {cum_dist.max():.4f}")

# --- Align energies to VBM ---
all_energies = np.array(energies)  # shape (nks, nbnd)

vbm = np.max(all_energies[:, :n_occ])
cbm = np.min(all_energies[:, n_occ:])
band_gap = cbm - vbm

print(f"\nVBM = {vbm:.4f} eV")
print(f"CBM = {cbm:.4f} eV")
print(f"Band gap = {band_gap:.4f} eV")

all_energies_shifted = all_energies - vbm

# --- Determine tick positions and labels ---
# For each segment, the start and end are high-symmetry points.
# Tick position = cum_dist at that index.
# For discontinuities, label with both sides (e.g., 'A|L').

tick_positions = []
tick_labels = []

for seg_idx, (start, end, label_from, label_to) in enumerate(segments):
    # Start tick
    if seg_idx == 0:
        tick_positions.append(cum_dist[start])
        tick_labels.append(label_from)
    else:
        # Check if this is a discontinuity
        # The previous segment ended at segments[seg_idx-1][1]
        prev_end = segments[seg_idx-1][1]
        prev_label_to = segments[seg_idx-1][3]
        # If the distance between prev_end and start is a discontinuity
        if is_discontinuity[prev_end]:
            combined_label = f"{prev_label_to}|{label_from}"
        else:
            # Continuous: the end of prev segment and start of this segment are the same point
            # Use the label from the current segment's start
            combined_label = label_from
        tick_positions.append(cum_dist[start])
        tick_labels.append(combined_label)
    
    # End tick (only for the last segment, to avoid duplication with next segment's start)
    if seg_idx == len(segments) - 1:
        tick_positions.append(cum_dist[end])
        tick_labels.append(label_to)

# Remove duplicate tick positions (within tolerance)
unique_ticks = []
unique_labels = []
for pos, lab in zip(tick_positions, tick_labels):
    if not unique_ticks or abs(pos - unique_ticks[-1]) > 1e-6:
        unique_ticks.append(pos)
        unique_labels.append(lab)

tick_positions = unique_ticks
tick_labels = unique_labels

print(f"\nTick positions: {tick_positions}")
print(f"Tick labels: {tick_labels}")

# INVARIANT 4: assert tick positions span the same range as the band data
assert abs(max(tick_positions) - cum_dist.max()) < 0.05 * cum_dist.max(), \
    f"Tick positions max {max(tick_positions)} != cum_dist max {cum_dist.max()}"
assert abs(min(tick_positions) - cum_dist.min()) < 0.05 * max(cum_dist.max(), 1e-10), \
    f"Tick positions min {min(tick_positions)} != cum_dist min {cum_dist.min()}"

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each segment separately (no line across discontinuities)
for start, end, label_from, label_to in segments:
    x_seg = cum_dist[start:end+1]
    for band in range(nbnd):
        y_seg = all_energies_shifted[start:end+1, band]
        ax.plot(x_seg, y_seg, 'b-', linewidth=0.8)

# Set x-axis ticks
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)

# Set y-axis limits: +/-10 eV around VBM (which is now at 0)
ax.set_ylim(-10, 10)

# Labels
ax.set_xlabel('k-point path (2$\\pi$/a$_{lat}$)')
ax.set_ylabel('Energy (eV) relative to VBM')
ax.set_title('Graphite C - Band Structure (PBEsol)')
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_PDF, dpi=150)
print(f"\nSaved plot to {OUTPUT_PDF}")

print(f"\nSummary:")
print(f"  VBM = {vbm:.4f} eV")
print(f"  CBM = {cbm:.4f} eV")
print(f"  Band gap = {band_gap:.4f} eV")
