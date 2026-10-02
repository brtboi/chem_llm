import os
import shutil
import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase.io.espresso import write_espresso_in, write_fortran_namelist
from ase import Atoms

# =============================================================================
# setup_jobs.py -- Generate QE input files for all 51 TiO2 structures
# =============================================================================
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - Pseudopotential files (copied from template/)
#   - pw.in       (SCF calculation)
#   - bands.in    (non-self-consistent bands along k-path)
#   - bands_post.in (bands.x post-processing input)
#   - band_labels.dat (high-symmetry labels and their indices for plotting)
# =============================================================================

# Pseudopotential files (from template/)
# Ti: z_valence=12 (includes semicore 3s2 3p6), recommended ecutwfc=42 Ha
# O:  z_valence=6,  recommended ecutwfc=46 Ha
# INVARIANT 5: 1 Ha = 2 Ry, so ecutwfc = max(42,46)*2 = 92 Ry
PSEUDO_FILES = {
    'Ti': 'Ti.pbesol_s-rrkjus_psl.1.0.0.UPF',
    'O':  'O.pbesol_s-rrkjus_psl.1.0.0.UPF',
}

# Cutoffs in Rydberg (converted from Ha per INVARIANT 5)
ECUTWFC_RY = 92.0   # 46 Ha * 2
ECUTRHO_RY = 368.0  # 4 * ecutwfc (standard ratio for charge density)

# SCF k-point mesh: 4x4x4 Monkhorst-Pack grid
# Reasonable density for a 6-atom primitive cell of rutile TiO2
SCF_KPTS = (4, 4, 4)

# Bands calculation: nbnd must exceed occupied bands
# Total electrons = 2*12 + 4*6 = 48; occupied bands = 48/2 = 24
# nbnd=40 gives 16 conduction bands above VBM
NBND = 40

# Number of k-points per segment in the bands path
PTS_PER_SEGMENT = 20

# Common input parameters (flat dict -- ASE routes to correct namelists, INVARIANT 6)
COMMON_INPUT = {
    'prefix': 'tio2',
    'outdir': './tmp',
    'pseudo_dir': './',
    'ecutwfc': ECUTWFC_RY,
    'ecutrho': ECUTRHO_RY,
}


def build_kpath(atoms):
    """
    Build the k-path for the bands calculation using seekpath.
    
    INVARIANT 3: The k-path must be derived from the EXACT same Atoms object
    written into CELL_PARAMETERS. seekpath's point_coords are fractional
    coordinates of ITS OWN primitive cell, so we use that cell as the one
    written into CELL_PARAMETERS.
    
    Returns:
        prim: Atoms object (seekpath's primitive cell) to use in CELL_PARAMETERS
        kpoints: (n_kpts, 4) numpy array of k-points in crystal coordinates
        labels: list of (index, label) for high-symmetry points
        path_segments: list of (label_from, label_to) for discontinuity detection
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    
    # seekpath with symprec=0.01 (matches pymatgen default; spglib's 1e-5 default
    # is too tight for CIF-round-tripped coordinates)
    res = seekpath.get_path(cell_tuple, symprec=0.01)
    
    # INVARIANT 3 check: seekpath's space group must agree with spglib
    sg_spglib = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01).number
    assert res['spacegroup_number'] == sg_spglib, (
        f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={sg_spglib}")
    
    # Build the primitive cell from seekpath's own reduced cell
    prim = Atoms(
        cell=res['primitive_lattice'],
        scaled_positions=res['primitive_positions'],
        numbers=res['primitive_types'],
        pbc=True,
    )
    
    # Build the k-point list by interpolating each segment independently.
    # INVARIANT 3: EVERY segment must contribute its own start point because
    # res['path'] is frequently DISCONTINUOUS (e.g. Z->X, R->A jumps).
    path = res['path']  # list of (label_from, label_to)
    point_coords = res['point_coords']  # dict of label -> [x, y, z] fractional coords
    
    kpoints_list = []
    labels = []  # (index, label) for high-symmetry points
    
    for i_seg, (label_from, label_to) in enumerate(path):
        k_from = np.array(point_coords[label_from])
        k_to = np.array(point_coords[label_to])
        
        # Interpolate PTS_PER_SEGMENT points from k_from to k_to (inclusive)
        # np.linspace includes both endpoints
        seg_kpts = np.linspace(k_from, k_to, PTS_PER_SEGMENT)
        
        # For the first segment, include all points
        # For subsequent segments, skip the first point IF it's the same as the
        # last point of the previous segment (continuous path), otherwise include it
        if i_seg == 0:
            new_kpts = seg_kpts
        else:
            # Check if this segment's start is the same as the previous segment's end
            prev_k_to = np.array(point_coords[path[i_seg - 1][1]])
            if np.allclose(k_from, prev_k_to, atol=1e-10):
                # Continuous: skip the duplicate start point
                new_kpts = seg_kpts[1:]
            else:
                # Discontinuous: include the start point (it's a new high-symmetry point)
                new_kpts = seg_kpts
        
        # Record the label at the start of this segment (if not already recorded)
        if i_seg == 0:
            labels.append((0, label_from))
        else:
            # The start point of this segment is at index len(kpoints_list)
            # (before we append new_kpts)
            start_idx = len(kpoints_list)
            # Check if this label was already recorded (e.g. GAMMA appears multiple times)
            existing_labels = [l for _, l in labels]
            if label_from not in existing_labels:
                labels.append((start_idx, label_from))
            # For discontinuous jumps, we need to record BOTH labels at the jump point
            # This is handled by checking if the previous segment's end differs
            prev_label = path[i_seg - 1][1]
            if not np.allclose(k_from, prev_k_to, atol=1e-10):
                # Discontinuous jump: record the previous label at the same index
                if prev_label not in existing_labels:
                    labels.append((start_idx, prev_label))
        
        kpoints_list.extend(new_kpts)
        
        # Record the label at the end of this segment
        end_idx = len(kpoints_list) - 1
        if label_to not in [l for _, l in labels]:
            labels.append((end_idx, label_to))
    
    # Convert to (n_kpts, 4) array with weight=1.0
    kpoints = np.zeros((len(kpoints_list), 4))
    for i, kp in enumerate(kpoints_list):
        kpoints[i, :3] = kp
        kpoints[i, 3] = 1.0  # weight (unused for non-self-consistent bands run)
    
    # INVARIANT 3 CHECK: assert every named point's fractional coordinates are
    # simple fractions (halves, quarters, thirds) and NOT arbitrary decimals
    for label, coords in point_coords.items():
        for c in coords:
            # Check if c is a simple fraction: c * 6 should be close to an integer
            # (covers halves, thirds, sixths, quarters via 12, etc.)
            for denom in [1, 2, 3, 4, 6, 12]:
                if abs(c * denom - round(c * denom)) < 1e-10:
                    break
            else:
                raise AssertionError(
                    f"Point {label} has non-simple-fraction coordinate {c}")
    
    return prim, kpoints, labels, path


def setup_structure(nnn, atoms):
    """Set up the calculation directory for structure NNN."""
    calc_dir = f'calculations/{nnn}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/tmp', exist_ok=True)
    
    # Copy pseudopotential files into the calculation directory
    for element, fname in PSEUDO_FILES.items():
        src = f'template/{fname}'
        dst = f'{calc_dir}/{fname}'
        shutil.copy2(src, dst)
    
    # Build the k-path from THIS structure (INVARIANT 3)
    prim, kpoints, labels, path = build_kpath(atoms)
    
    # INVARIANT 1: nat and ntyp computed from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 6, f"Structure {nnn}: expected nat=6, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"
    
    # Pseudopotential mapping for write_espresso_in
    pseudos = {element: fname for element, fname in PSEUDO_FILES.items()}
    
    # --- SCF input (pw.in) ---
    scf_input = dict(COMMON_INPUT)
    scf_input.update({
        'calculation': 'scf',
        'nat': nat,
        'ntyp': ntyp,
        'occupations': 'fixed',  # insulator, no smearing needed
        'conv_thr': 1.0e-8,      # in &ELECTRONS (ASE routes correctly)
    })
    
    write_espresso_in(
        f'{calc_dir}/pw.in',
        prim,
        input_data=scf_input,
        pseudopotentials=pseudos,
        kpts=SCF_KPTS,  # (4,4,4) Monkhorst-Pack grid
    )
    
    # --- Bands input (bands.in) ---
    bands_input = dict(COMMON_INPUT)
    bands_input.update({
        'calculation': 'bands',
        'nat': nat,
        'ntyp': ntyp,
        'nbnd': NBND,
    })
    
    write_espresso_in(
        f'{calc_dir}/bands.in',
        prim,
        input_data=bands_input,
        pseudopotentials=pseudos,
        kpts=kpoints,  # (n_kpts, 4) numpy array = raw crystal-coordinate k-points
    )
    
    # --- Bands post-processing input (bands_post.in) ---
    # bands.x input: &BANDS namelist with prefix, outdir, filband
    bands_post_input = {
        'prefix': 'tio2',
        'outdir': './tmp',
        'filband': f'{nnn}.bands.dat',
    }
    write_fortran_namelist(
        f'{calc_dir}/bands_post.in',
        input_data=bands_post_input,
        binary='bands',
    )
    
    # --- Band labels file (band_labels.dat) ---
    # Each line: index label
    # For discontinuous jumps, both labels are recorded at the same index
    with open(f'{calc_dir}/band_labels.dat', 'w') as f:
        for idx, label in sorted(labels, key=lambda x: x[0]):
            f.write(f"{idx} {label}\n")
    
    # Also write the path segments for discontinuity detection in plotting
    with open(f'{calc_dir}/path_segments.dat', 'w') as f:
        for label_from, label_to in path:
            f.write(f"{label_from} {label_to}\n")
    
    print(f"  {calc_dir}: nat={nat}, ntyp={ntyp}, n_kpts={len(kpoints)}, "
          f"labels={len(labels)}")


def main():
    print("Setting up calculation directories for all 51 structures...")
    
    for i in range(51):
        nnn = f"{i:03d}"
        cif_file = f'structures/structure_{nnn}.cif'
        atoms = read(cif_file)
        setup_structure(nnn, atoms)
    
    print("\nDone. All calculation directories created.")
    print("Next: run validate_qe_input on calculations/000/pw.in and bands.in")


if __name__ == '__main__':
    main()
