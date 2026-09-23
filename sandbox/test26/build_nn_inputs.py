#!/usr/bin/env python3
"""
build_nn_inputs.py -- Assemble a complete DeePseudopot input bundle for ONE
completed QE DFT job (calculations/000) into nn_inputs_g/.

This script ports the logic from:
  - deepseudopot/qe_bands_to_ref.py  (parse_band_file, generate_qe_kpath,
    compute_kpath_distances, write_kpoints_file, write_band_file)
  - deepseudopot/setup_nn_inputs.py  (parse_pw_input, write_system_par)
  - deepseudopot/nn_template/shift.py  (VBM alignment + band-gap opening)
  - deepseudopot/nn_template/select_rows.py  (keep rows 11 and 21 only)

All operations are done in-process with explicit paths -- no subprocess calls,
no cwd= tricks, no reliance on hardcoded output filenames.
"""

import shutil
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

CALC_DIR = Path("calculations/000")
TEMPLATE_DIR = Path("deepseudopot/nn_template")
NN_DIR = Path("nn_inputs_g")

ANG_TO_BOHR = 1.889726125

# Work function (eV) to which the VBM is aligned, and target band gap (eV).
# These are the same values used by deepseudopot/nn_template/shift.py.
WORK_FUNC_EV = -5.25
REF_GAP_EV = 1.7

# Number of bands to drop from the low end (core-like / deep valence bands).
# CsPbBr3 under noncolin/lspinorb has 176 electrons -> VBM = band 176.
# After dropping 72 bands, VBM lands at column 104 (0-indexed) in the
# 128-band array. This split is correct for ANY CsPbBr3 job built from
# example/'s pipeline (perturbed or not) -- do not re-derive.
N_DROP = 72

# 1-indexed rows to keep for the fast '_g' reduced-k-point bundle.
# Row 11 = first point of the Gamma->X segment (X direction),
# Row 21 = first point of the X->M segment (M direction).
# Keeping only these two k-points makes Hamiltonian caching (per k-point,
# NOT parallelized within one system) the single biggest lever on
# train_deepseudopot's runtime.
SELECT_ROWS = (11, 21)


# ============================================================
# PORTED FROM deepseudopot/qe_bands_to_ref.py
# ============================================================

def generate_qe_kpath():
    """
    Generate k-points in fractional (crystal_b) coordinates following the
    same interpolation Quantum ESPRESSO uses for a K_POINTS crystal_b block.

    Hardcoded to the exact 41-point R -> Gamma -> X -> M -> Gamma path that
    setup_jobs.py's bands.in writes (four 10-point segments + closing point).
    This path was NOT changed in STEP A2, so it applies unchanged here.
    """
    k_segments = [
        ([0.5, 0.5, 0.5], 10),  # R
        ([0.0, 0.0, 0.0], 10),  # Gamma
        ([0.5, 0.0, 0.0], 10),  # X
        ([0.5, 0.5, 0.0], 10),  # M
        ([0.0, 0.0, 0.0], 1),   # Gamma (closing)
    ]

    kpoints = []
    for i in range(len(k_segments) - 1):
        start, npts = k_segments[i]
        end, _ = k_segments[i + 1]
        start = np.array(start)
        end = np.array(end)
        # QE includes both endpoints for each segment
        for t in range(npts):
            frac = t / npts if npts > 1 else 0.0
            k = (1 - frac) * start + frac * end
            kpoints.append(k)
    kpoints.append(k_segments[-1][0])
    return np.array(kpoints)


def parse_band_file(filename):
    """
    Parse a QE bands.x-produced .dat file of the form:
      kx ky kz
      e1 e2 e3 ...
    Returns (kpoints, bands) as numpy arrays.
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Remove header lines (&plot ... /)
    data_lines = [
        ln.strip() for ln in lines
        if not ln.strip().startswith("&")
        and not ln.strip().startswith("/")
        and ln.strip()
    ]

    kpoints = []
    bands = []
    i = 0
    while i < len(data_lines):
        parts = data_lines[i].split()
        if len(parts) == 3:  # k-point line
            kpt = list(map(float, parts))
            kpoints.append(kpt)
            i += 1

            # Collect band energies until next k-point or EOF
            energy_vals = []
            while i < len(data_lines):
                next_parts = data_lines[i].split()
                if len(next_parts) == 3:
                    break
                energy_vals.extend(map(float, next_parts))
                i += 1
            bands.append(energy_vals)
        else:
            i += 1

    return np.array(kpoints), np.array(bands)


def compute_kpath_distances(kpoints):
    """Compute cumulative distance along the k-path (fractional coords)."""
    diffs = np.diff(kpoints, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return distances


def write_kpoints_file(filename, kpoints):
    """Write fractional k-points to kpoints_0.par."""
    with open(filename, "w") as f:
        for k in kpoints:
            f.write(f"{k[0]:.6f} {k[1]:.6f} {k[2]:.6f} 1.0\n")


def write_band_file(filename, distances, bands):
    """Write expBandStruct_0.par file."""
    with open(filename, "w") as f:
        for d, bvals in zip(distances, bands):
            f.write(f"{d:.6f} " + " ".join(f"{x:.6f}" for x in bvals) + "\n")


# ============================================================
# PORTED FROM deepseudopot/setup_nn_inputs.py
# ============================================================

def parse_pw_input(filename):
    """
    Parse a pw.in file's CELL_PARAMETERS (angstrom) and ATOMIC_POSITIONS
    crystal blocks into (scale_in_bohr, normalized_cell, atoms).

    scale_in_bohr: the first component of the first lattice vector, converted
      from angstrom to bohr. This is the 'scale' value DeePseudopot expects.
    normalized_cell: each lattice vector divided by the scale (dimensionless).
    atoms: list of (label, [x, y, z]) where label is e.g. 'Cs0', 'Pb1', etc.
    """
    with open(filename) as f:
        lines = f.readlines()

    # --------------------------------------------------------
    # lattice
    # --------------------------------------------------------
    cell_idx = None
    for i, line in enumerate(lines):
        if "CELL_PARAMETERS" in line:
            cell_idx = i
            break

    if cell_idx is None:
        raise RuntimeError(f"No CELL_PARAMETERS in {filename}")

    lattice = []
    for i in range(3):
        vals = list(map(float, lines[cell_idx + 1 + i].split()))
        lattice.append(vals)

    # --------------------------------------------------------
    # scale + normalized cell
    # --------------------------------------------------------
    a_vec = lattice[0]
    scale_angstrom = a_vec[0]
    scale_bohr = scale_angstrom * ANG_TO_BOHR

    normalized = []
    for vec in lattice:
        normalized.append([x / scale_angstrom for x in vec])

    # --------------------------------------------------------
    # atomic positions
    # --------------------------------------------------------
    atom_idx = None
    for i, line in enumerate(lines):
        if "ATOMIC_POSITIONS crystal" in line:
            atom_idx = i
            break

    if atom_idx is None:
        raise RuntimeError(f"No ATOMIC_POSITIONS crystal in {filename}")

    atoms = []
    counts = {}
    for line in lines[atom_idx + 1:]:
        stripped = line.strip()
        if stripped == "":
            break
        vals = stripped.split()
        species = vals[0]
        coords = vals[1:4]
        if species not in counts:
            counts[species] = 0
        label = f"{species}{counts[species]}"
        counts[species] += 1
        atoms.append((label, coords))

    return scale_bohr, normalized, atoms


def write_system_par(outfile, scale, cell, atoms):
    """
    Write DeePseudopot's system_0.par format.

    scale: lattice scale in bohr (first component of first lattice vector).
    cell: normalized (dimensionless) lattice vectors.
    atoms: list of (label, [x, y, z]) with fractional coordinates.
    """
    with open(outfile, "w") as f:
        f.write(f"scale = {scale:.12f}\n\n")
        f.write("cell\n")
        for vec in cell:
            f.write(
                f"{vec[0]:.12f}\t"
                f"{vec[1]:.12f}\t"
                f"{vec[2]:.12f}\n"
            )
        f.write("\n")
        f.write("atoms\n")
        for label, coords in atoms:
            f.write(
                f"{label:<4} "
                f"{coords[0]} "
                f"{coords[1]} "
                f"{coords[2]}\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # Step (a): Copy deepseudopot/nn_template/ to nn_inputs_g/
    # --------------------------------------------------------
    if NN_DIR.exists():
        print(f"Overwriting {NN_DIR}")
        shutil.rmtree(NN_DIR)

    print(f"Copying {TEMPLATE_DIR} -> {NN_DIR}")
    shutil.copytree(TEMPLATE_DIR, NN_DIR)

    # --------------------------------------------------------
    # Step (b): Parse calculations/000/pw.in and write system_0.par
    # --------------------------------------------------------
    pw_in = CALC_DIR / "pw.in"
    print(f"Parsing {pw_in}")
    scale_bohr, normalized_cell, atoms = parse_pw_input(pw_in)

    # INVARIANT 1: verify atom count matches the structure
    assert len(atoms) == 20, f"Expected 20 atoms, got {len(atoms)}"
    species_counts = {}
    for label, _ in atoms:
        # Extract species (strip trailing digits)
        sp = "".join(c for c in label if c.isalpha())
        species_counts[sp] = species_counts.get(sp, 0) + 1
    assert species_counts.get("Cs", 0) == 4, f"Expected 4 Cs, got {species_counts.get('Cs', 0)}"
    assert species_counts.get("Pb", 0) == 4, f"Expected 4 Pb, got {species_counts.get('Pb', 0)}"
    assert species_counts.get("Br", 0) == 12, f"Expected 12 Br, got {species_counts.get('Br', 0)}"

    system_par_path = NN_DIR / "system_0.par"
    write_system_par(system_par_path, scale_bohr, normalized_cell, atoms)
    print(f"Wrote {system_par_path} ({len(atoms)} atoms)")

    # --------------------------------------------------------
    # Step (c): Parse 000.bands.dat, build k-path, write
    #           kpoints_0.par and expBandStruct_0.par
    # --------------------------------------------------------
    bands_dat = CALC_DIR / "000.bands.dat"
    print(f"Parsing {bands_dat}")
    k_cart, bands_full = parse_band_file(bands_dat)

    # Verify we got the expected number of k-points and bands
    assert len(k_cart) == 41, f"Expected 41 k-points, got {len(k_cart)}"
    assert bands_full.shape[1] == 200, f"Expected 200 bands, got {bands_full.shape[1]}"

    # Drop the first 72 bands (core-like / deep valence) to keep the 128
    # bands DeePseudopot fits to. This split comes from CsPbBr3's electron
    # count (176 electrons -> VBM = band 176) and is correct for THIS job
    # unchanged because the composition and pseudopotentials are identical
    # to the validated recipe.
    bands = bands_full[:, N_DROP:]
    assert bands.shape[1] == 128, f"Expected 128 bands after drop, got {bands.shape[1]}"

    # Build the k-path in fractional (crystal_b) coordinates.
    # generate_qe_kpath() is hardcoded to the exact 41-point path used in
    # bands.in -- it applies unchanged here.
    k_frac = generate_qe_kpath()
    assert len(k_frac) == 41, f"Expected 41 k-points in kpath, got {len(k_frac)}"

    # Compute cumulative distances along the k-path (x-axis for DeePseudopot)
    distances = compute_kpath_distances(k_frac)

    # Write kpoints_0.par and expBandStruct_0.par
    kpoints_path = NN_DIR / "kpoints_0.par"
    bandstruct_path = NN_DIR / "expBandStruct_0.par"
    write_kpoints_file(kpoints_path, k_frac)
    write_band_file(bandstruct_path, distances, bands)
    print(f"Wrote {kpoints_path} and {bandstruct_path} (41 rows x 128 bands)")

    # --------------------------------------------------------
    # Step (d): Apply the SAME valence-band-maximum shift that shift.py
    #           performs (work function -5.25 eV, target gap 1.7 eV).
    #
    # shift.py's arithmetic:
    #   vbmax = bs[:, 104].max()   # VBM in the 128-band array
    #   cbmin = bs[:, 105].min()   # CBM in the 128-band array
    #   shift = vbmax - work_func  # shift to align VBM to work function
    #   shift_for_bandgap = ref_gap - (cbmin - vbmax)  # open gap to 1.7 eV
    #   bs[:, 1:] -= shift         # shift all bands
    #   bs[:, 105:] += shift_for_bandgap  # open the gap
    #
    # This MUST run on the full 41-row band structure (searching VBM/CBM
    # over the whole k-path, not a subset) -- BEFORE step (e).
    # --------------------------------------------------------
    print("Applying VBM shift and band-gap opening...")
    bs = np.loadtxt(bandstruct_path)
    assert bs.shape == (41, 129), f"Expected shape (41, 129), got {bs.shape}"

    vbmax = bs[:, 104].max()
    cbmin = bs[:, 105].min()
    shift = vbmax - WORK_FUNC_EV
    band_gap = cbmin - vbmax
    shift_for_bandgap = REF_GAP_EV - band_gap

    print(f"  VBmax = {vbmax:.6f} eV, CBmin = {cbmin:.6f} eV")
    print(f"  Band gap = {band_gap:.6f} eV, shift to open = {shift_for_bandgap:.6f} eV")
    print(f"  VBM shift = {shift:.6f} eV")

    # Subtract shift from every band column (1 onward), keep column 0 (k-distance)
    bs[:, 1:] = bs[:, 1:] - shift
    # Add shift_for_bandgap to columns 105 onward (conduction bands)
    bs[:, 105:] = bs[:, 105:] + shift_for_bandgap

    # Overwrite expBandStruct_0.par with the shifted result
    np.savetxt(bandstruct_path, bs, fmt="%.6f")
    print(f"  Overwrote {bandstruct_path} with shifted bands")

    # --------------------------------------------------------
    # Step (e): Apply the SAME row-selection that select_rows.py performs:
    #           keep only rows 11 and 21 (1-indexed) of BOTH kpoints_0.par
    #           and expBandStruct_0.par, overwriting each file with just
    #           those two rows. This is what makes the bundle fast to train
    #           (Hamiltonian caching is per k-point and NOT parallelized
    #           within one system, so fewer k-points is the single biggest
    #           lever on train_deepseudopot's runtime).
    # --------------------------------------------------------
    print(f"Selecting rows {SELECT_ROWS} (1-indexed)...")

    # kpoints_0.par: keep rows 11 and 21
    with open(kpoints_path) as f:
        kp_lines = f.readlines()
    assert len(kp_lines) == 41, f"Expected 41 rows in kpoints_0.par, got {len(kp_lines)}"
    kp_selected = [kp_lines[i - 1] for i in SELECT_ROWS]
    with open(kpoints_path, "w") as f:
        f.writelines(kp_selected)
    print(f"  kpoints_0.par: kept rows {SELECT_ROWS} -> {len(kp_selected)} rows")

    # expBandStruct_0.par: keep rows 11 and 21
    with open(bandstruct_path) as f:
        bs_lines = f.readlines()
    assert len(bs_lines) == 41, f"Expected 41 rows in expBandStruct_0.par, got {len(bs_lines)}"
    bs_selected = [bs_lines[i - 1] for i in SELECT_ROWS]
    with open(bandstruct_path, "w") as f:
        f.writelines(bs_selected)
    print(f"  expBandStruct_0.par: kept rows {SELECT_ROWS} -> {len(bs_selected)} rows")

    print(f"\nDone. Bundle assembled at {NN_DIR}/")


if __name__ == "__main__":
    main()
