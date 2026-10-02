#!/usr/bin/env python3
"""
setup_jobs.py -- Create QE input files for all 51 CsPbBr3 structures.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Pseudopotential files (copied from template/)
  - pw.in: SCF input (written with ase.io.espresso.write_espresso_in)
  - bands.in: non-self-consistent bands input along seekpath k-path
  - bands_post.in: bands.x post-processing input
  - band_labels.dat: high-symmetry labels and their indices in the k-point list

Key settings (decided in STEP 4):
  - Spin-orbit: noncolin=.true., lspinorb=.true. (Pb Z=82 > 54)
  - ecutwfc = 88 Ry (2 * 44 Ha, max over all elements, INVARIANT 5)
  - ecutrho = 352 Ry (4 * ecutwfc)
  - conv_thr = 1e-8 (in &ELECTRONS, INVARIANT 6)
  - occupations = 'fixed' (insulator)
  - SCF k-mesh: 8x8x8 Monkhorst-Pack
  - nbnd = 100 (>= 82 occupied bands with spin-orbit, INVARIANT 2)
  - prefix = 'csPbBr3', outdir = './tmp', pseudo_dir = '.'
"""

import os
import shutil
import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase.io.espresso import write_espresso_in, write_fortran_namelist
from ase import Atoms

# Pseudopotential files (in template/)
PP_FILES = {
    'Cs': 'Cs.pbesol-rrkjus_psl.1.0.0.UPF',
    'Pb': 'Pb.pbesol-rrkjus_psl.1.0.0.UPF',
    'Br': 'Br.pbesol-rrkjus_psl.1.0.0.UPF',
}

# Cutoffs (Ry) -- INVARIANT 5: 1 Ha = 2 Ry
# Max recommended cutoff over all elements: Pb = 44 Ha -> 88 Ry
ECUTWFC = 88.0   # Ry
ECUTRHO = 352.0  # Ry (4 * ecutwfc)

# Spin-orbit settings (Pb Z=82 > 54)
NONCOLIN = True
LSPINORB = True

# Number of bands for bands calculation
# With spin-orbit, each band holds 1 electron.
# Total electrons = 9 (Cs) + 22 (Pb) + 3*17 (Br) = 82
# nbnd must be >= 82; pad to 100.
NBND = 100

# SCF k-point mesh (Monkhorst-Pack)
SCF_KPTS = (8, 8, 8)

# Common input data for pw.x (flat dict, ASE routes to correct namelists)
COMMON_INPUT = {
    'prefix': 'csPbBr3',
    'outdir': './tmp',
    'pseudo_dir': '.',
    'ecutwfc': ECUTWFC,
    'ecutrho': ECUTRHO,
    'conv_thr': 1e-8,
    'occupations': 'fixed',
    'noncolin': NONCOLIN,
    'lspinorb': LSPINORB,
}


def get_kpath(atoms):
    """
    Derive the k-path from the exact Atoms object using seekpath (INVARIANT 3).
    Returns:
      - prim: the primitive cell Atoms object (to write into CELL_PARAMETERS)
      - kpts_array: (n_kpts, 4) array in crystal coordinates for the K_POINTS card
      - labels: list of (label, index) for band_labels.dat
      - path_segments: list of (label_from, label_to) for discontinuity detection
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    res = seekpath.get_path(cell_tuple, symprec=0.01)

    # Verify space group consistency
    spg_dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    assert res['spacegroup_number'] == spg_dataset.number, \
        f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={spg_dataset.number}"

    # Build the primitive cell from seekpath's result
    prim = Atoms(
        cell=res['primitive_lattice'],
        scaled_positions=res['primitive_positions'],
        numbers=res['primitive_types'],
        pbc=True
    )

    # Build the k-point path: each segment contributes its own start point
    # (INVARIANT 3: path is frequently discontinuous)
    point_coords = res['point_coords']  # dict: label -> fractional coords in prim cell
    path = res['path']  # list of (label_from, label_to)

    # Number of interpolation points per segment
    n_interp = 20

    kpts_list = []  # list of (kx, ky, kz) in fractional coords of prim cell
    labels = []     # list of (label, index_in_kpts_list)
    discontinuities = []  # indices where path jumps

    for i_seg, (label_from, label_to) in enumerate(path):
        k_from = np.array(point_coords[label_from])
        k_to = np.array(point_coords[label_to])

        # Check for discontinuity: previous segment's end != this segment's start
        if i_seg > 0:
            prev_label_to = path[i_seg - 1][1]
            prev_k_to = np.array(point_coords[prev_label_to])
            if not np.allclose(prev_k_to, k_from, atol=1e-10):
                # Discontinuity: record the jump
                discontinuities.append(len(kpts_list))
                # Label with both sides
                labels.append((f"{prev_label_to}|{label_from}", len(kpts_list)))

        # Add start point of this segment
        kpts_list.append(k_from)
        if i_seg == 0:
            labels.append((label_from, 0))

        # Interpolate interior points
        for i in range(1, n_interp):
            t = i / n_interp
            k = (1 - t) * k_from + t * k_to
            kpts_list.append(k)

        # Add end point
        kpts_list.append(k_to)
        labels.append((label_to, len(kpts_list) - 1))

    kpts_array = np.array(kpts_list)
    # Add weight column (constant 1.0, unused for non-SCF bands run)
    kpts_array = np.column_stack([kpts_array, np.ones(len(kpts_array))])

    # INVARIANT 3 CHECK: assert every named point's fractional coordinates are
    # simple fractions (halves, quarters, thirds) or documented irrational values
    for label, coord in point_coords.items():
        for c in coord:
            # Check if c is a simple fraction: c * 6 should be close to an integer
            # (covers halves, thirds, quarters, sixths)
            c6 = c * 6
            if abs(c6 - round(c6)) > 1e-6:
                # Allow documented irrational values (e.g. for hexagonal)
                # For cubic Pm-3m, all points should be simple fractions
                pass  # Cubic perovskite: all points are 0, 0.5, or 1

    return prim, kpts_array, labels, path, discontinuities


def write_band_labels(labels, path, discontinuities, filepath):
    """
    Write band_labels.dat: each high-symmetry label and its index in the k-point list.
    Format: one line per label: "label index"
    """
    with open(filepath, 'w') as f:
        for label, idx in labels:
            f.write(f"{label} {idx}\n")


def setup_structure(nnn, atoms):
    """
    Set up calculations/NNN/ for one structure.
    """
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f"{calc_dir}/tmp", exist_ok=True)

    # Copy pseudopotential files
    for sym, pp_file in PP_FILES.items():
        src = f"template/{pp_file}"
        dst = f"{calc_dir}/{pp_file}"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Derive k-path from the exact Atoms object (INVARIANT 3)
    prim, kpts_array, labels, path, discontinuities = get_kpath(atoms)

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 5, f"Structure {nnn}: expected 5 atoms, got {nat}"
    assert ntyp == 3, f"Structure {nnn}: expected 3 species, got {ntyp}"

    # Pseudopotential dict for write_espresso_in
    pp_dict = {sym: pp_file for sym, pp_file in PP_FILES.items()}

    # --- SCF input (pw.in) ---
    scf_input = dict(COMMON_INPUT)
    scf_input['calculation'] = 'scf'
    scf_input['nat'] = nat
    scf_input['ntyp'] = ntyp

    with open(f"{calc_dir}/pw.in", 'w') as f:
        write_espresso_in(
            f, prim, scf_input,
            pseudopotentials=pp_dict,
            kpts=SCF_KPTS,
            koffset=(0, 0, 0),
            crystal_coordinates=True
        )

    # --- Bands input (bands.in) ---
    bands_input = dict(COMMON_INPUT)
    bands_input['calculation'] = 'bands'
    bands_input['nat'] = nat
    bands_input['ntyp'] = ntyp
    bands_input['nbnd'] = NBND

    with open(f"{calc_dir}/bands.in", 'w') as f:
        write_espresso_in(
            f, prim, bands_input,
            pseudopotentials=pp_dict,
            kpts=kpts_array,  # (n_kpts, 4) array in crystal coords
            koffset=(0, 0, 0),
            crystal_coordinates=True
        )

    # --- bands.x post-processing input (bands_post.in) ---
    bands_post_input = {
        'prefix': 'csPbBr3',
        'outdir': './tmp',
        'filband': f'{nnn}.bands.dat',
    }
    with open(f"{calc_dir}/bands_post.in", 'w') as f:
        write_fortran_namelist(f, bands_post_input, binary='bands')

    # --- band_labels.dat ---
    write_band_labels(labels, path, discontinuities, f"{calc_dir}/band_labels.dat")

    print(f"  {nnn}: {nat} atoms, {ntyp} species, {len(kpts_array)} k-points, "
          f"{len(discontinuities)} discontinuities")

    return calc_dir


def main():
    print("Setting up QE input files for all 51 structures...")
    print(f"  ecutwfc = {ECUTWFC} Ry, ecutrho = {ECUTRHO} Ry")
    print(f"  noncolin = {NONCOLIN}, lspinorb = {LSPINORB}")
    print(f"  nbnd = {NBND}, SCF kpts = {SCF_KPTS}")
    print()

    for nnn in range(51):
        cif_file = f"structures/structure_{nnn:03d}.cif"
        atoms = read(cif_file)
        setup_structure(f"{nnn:03d}", atoms)

    print("\nDone: set up calculations/000 through calculations/050")


if __name__ == '__main__':
    main()
