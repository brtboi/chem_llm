#!/usr/bin/env python
"""Set up Quantum ESPRESSO calculation directories for wurtzite ZnO band-structure calculations.

For each structure NNN (000-050), creates calculations/NNN/ containing:
  - Pseudopotential files (Zn and O)
  - pw.in (SCF input)
  - bands.in (non-self-consistent bands input)
  - bands_post.in (bands.x post-processing input)
  - band_labels.dat (high-symmetry labels and their indices in the k-point list)

Key settings (see STEP 4 note):
  - ecutwfc = 92.0 Ry (from O's recommended 46 Ha * 2 Ha/Ry; Zn is 42 Ha = 84 Ry)
  - ecutrho = 368.0 Ry (4 * ecutwfc)
  - conv_thr = 1.0e-8 Ry
  - occupations = 'fixed' (insulator)
  - SCF k-mesh: 4x4x2 Monkhorst-Pack
  - nbnd = 40 (occupied bands = 52/2 = 26, so 40 gives 14 unoccupied bands)
  - K-path: from seekpath on the exact Atoms object written to CELL_PARAMETERS
"""
import numpy as np
import os
import shutil
import seekpath
import spglib
from ase.io import read, write
from ase.io.espresso import write_espresso_in, write_fortran_namelist
from ase import Atoms

# Pseudopotential files
PP_ZN = 'template/Zn.pbesol-srn-rrkjus_psl.1.0.0.UPF'
PP_O = 'template/O.pbesol-srn-rrkjus_psl.1.0.0.UPF'

# Pseudopotential filenames (basenames, since pseudo_dir='.' and files are copied into calculations/NNN/)
PP_ZN_BASE = os.path.basename(PP_ZN)
PP_O_BASE = os.path.basename(PP_O)

# Dict mapping species to pseudopotential filename for write_espresso_in
PSEUDOPOTENTIALS = {
    'Zn': PP_ZN_BASE,
    'O': PP_O_BASE,
}

# QE settings
ECUTWFC = 92.0  # Ry (from O's 46 Ha * 2)
ECUTRHO = 368.0  # Ry (4 * ecutwfc)
CONV_THR = 1.0e-8  # Ry
PREFIX = 'zno'
OUTDIR = './tmp'
NBDN = 40  # bands

# K-path settings
NPTS_PER_SEGMENT = 40  # number of k-points per segment (including endpoints)


def get_kpath(atoms):
    """Get the k-path from seekpath for the given Atoms object.
    
    Per INVARIANT 3: use seekpath with symprec=0.01, and use the primitive cell
    that seekpath returns as the one to write into CELL_PARAMETERS.
    
    Returns:
        prim: Atoms object for the primitive cell
        kpoints: list of (kx, ky, kz, weight) tuples
        labels: list of (label, index) for ALL segment endpoints (18 for wurtzite)
        discontinuities: list of indices where path is discontinuous
        res: seekpath result dict
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    res = seekpath.get_path(cell_tuple, symprec=0.01)
    
    # Verify space group consistency
    spglib_spg = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01).number
    seekpath_spg = res['spacegroup_number']
    assert spglib_spg == seekpath_spg, f"Space group mismatch: seekpath={seekpath_spg}, spglib={spglib_spg}"
    
    # Build the primitive cell Atoms object (this is what we write into CELL_PARAMETERS)
    prim = Atoms(
        cell=res['primitive_lattice'],
        scaled_positions=res['primitive_positions'],
        numbers=res['primitive_types'],
        pbc=True
    )
    
    # Build the k-point list by interpolating along each segment
    path = res['path']  # list of (label_from, label_to) tuples
    point_coords = res['point_coords']  # dict of label -> fractional coords
    
    # Build the full k-point list and labels
    kpoints = []  # list of (kx, ky, kz, weight)
    labels = []  # list of (label, index) for ALL segment endpoints
    discontinuities = []  # list of indices where path is discontinuous
    
    for i, (label_from, label_to) in enumerate(path):
        start = np.array(point_coords[label_from])
        end = np.array(point_coords[label_to])
        
        # Check for discontinuity: if this is not the first segment, check if the
        # previous segment's end differs from this segment's start
        if i > 0:
            prev_end = np.array(point_coords[path[i-1][1]])
            if not np.allclose(prev_end, start, atol=1e-10):
                discontinuities.append(len(kpoints))  # index where the jump occurs
        
        # Record the START label of this segment at the current index
        # (for the first segment, this is index 0; for subsequent segments,
        #  this is the index where the new segment begins, which may be a
        #  discontinuity point)
        labels.append((label_from, len(kpoints)))
        
        # Interpolate k-points along this segment
        for j in range(NPTS_PER_SEGMENT):
            t = j / (NPTS_PER_SEGMENT - 1)
            k = start + t * (end - start)
            kpoints.append((k[0], k[1], k[2], 1.0))
        
        # Record the ENDPOINT label of this segment
        labels.append((label_to, len(kpoints) - 1))
    
    return prim, kpoints, labels, discontinuities, res


def write_pw_in(atoms, nnn):
    """Write the SCF input file."""
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    
    # INVARIANT 1: nat and ntyp computed from the actual Atoms object
    assert nat == 4, f"Expected 4 atoms, got {nat}"
    assert ntyp == 2, f"Expected 2 species, got {ntyp}"
    
    input_data = {
        'CONTROL': {
            'calculation': 'scf',
            'prefix': PREFIX,
            'outdir': OUTDIR,
            'pseudo_dir': '.',
        },
        'SYSTEM': {
            'ecutwfc': ECUTWFC,
            'ecutrho': ECUTRHO,
            'occupations': 'fixed',
        },
        'ELECTRONS': {
            'conv_thr': CONV_THR,
        },
    }
    
    # K-point mesh: 4x4x2 Monkhorst-Pack
    kpts = (4, 4, 2)
    
    with open(f'calculations/{nnn}/pw.in', 'w') as f:
        write_espresso_in(f, atoms, input_data=input_data, kpts=kpts,
                          pseudopotentials=PSEUDOPOTENTIALS)


def write_bands_in(atoms, kpoints, nnn):
    """Write the bands input file."""
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    
    # INVARIANT 1: nat and ntyp computed from the actual Atoms object
    assert nat == 4, f"Expected 4 atoms, got {nat}"
    assert ntyp == 2, f"Expected 2 species, got {ntyp}"
    
    input_data = {
        'CONTROL': {
            'calculation': 'bands',
            'prefix': PREFIX,
            'outdir': OUTDIR,
            'pseudo_dir': '.',
        },
        'SYSTEM': {
            'ecutwfc': ECUTWFC,
            'ecutrho': ECUTRHO,
            'nbnd': NBDN,
        },
    }
    
    # K-points: explicit crystal_b path
    kpts_array = np.array(kpoints)
    
    with open(f'calculations/{nnn}/bands.in', 'w') as f:
        write_espresso_in(f, atoms, input_data=input_data, kpts=kpts_array,
                          pseudopotentials=PSEUDOPOTENTIALS)


def write_bands_post_in(nnn):
    """Write the bands.x post-processing input file using write_fortran_namelist."""
    input_data = {
        'BANDS': {
            'prefix': PREFIX,
            'outdir': OUTDIR,
            'filband': f'{nnn}.bands.dat',
        },
    }
    
    with open(f'calculations/{nnn}/bands_post.in', 'w') as f:
        write_fortran_namelist(f, input_data=input_data, binary='bands')


def write_band_labels(labels, discontinuities, nnn):
    """Write the band labels file for plotting.
    
    Writes label-index pairs for ALL segment endpoints (18 for wurtzite ZnO).
    Discontinuity info is stored in a separate file (discontinuities.dat)
    so that band_labels.dat contains only valid label-index pairs.
    """
    with open(f'calculations/{nnn}/band_labels.dat', 'w') as f:
        for label, index in labels:
            f.write(f"{label} {index}\n")
    
    # Write discontinuity info to a separate file
    with open(f'calculations/{nnn}/discontinuities.dat', 'w') as f:
        for idx in discontinuities:
            f.write(f"{idx}\n")


def main():
    # Create calculations directory
    os.makedirs('calculations', exist_ok=True)
    
    # Process each structure
    for nnn in range(51):
        nnn_str = f'{nnn:03d}'
        calc_dir = f'calculations/{nnn_str}'
        os.makedirs(calc_dir, exist_ok=True)
        
        # Copy pseudopotentials
        shutil.copy(PP_ZN, calc_dir)
        shutil.copy(PP_O, calc_dir)
        
        # Read the structure
        atoms = read(f'structures/structure_{nnn_str}.cif')
        
        # Get the k-path and primitive cell
        prim, kpoints, labels, discontinuities, res = get_kpath(atoms)
        
        # Write the input files using the primitive cell
        write_pw_in(prim, nnn_str)
        write_bands_in(prim, kpoints, nnn_str)
        write_bands_post_in(nnn_str)
        write_band_labels(labels, discontinuities, nnn_str)
        
        if nnn <= 2 or nnn == 50:
            print(f"Set up calculations/{nnn_str}/")
    
    print("\nDone. Set up all 51 calculation directories.")


if __name__ == '__main__':
    main()
