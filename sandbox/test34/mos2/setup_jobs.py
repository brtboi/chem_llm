import os
import shutil
import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase.io.espresso import write_espresso_in
from ase import Atoms

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - pseudopotential files (Mo.pbesol_upf, S.pbesol_upf)
#   - pw.in (SCF input)
#   - bands.in (non-self-consistent bands input with explicit k-path)
#   - bands_post.in (bands.x post-processing input)
#   - band_labels.dat (high-symmetry labels and their indices in the k-point list)
# =============================================================================

# --- Pseudopotential files ---
# Mo: z_valence=14, recommended ecutwfc=40.0 Ha = 80.0 Ry (INVARIANT 5: 1 Ha = 2 Ry)
# S:  z_valence=6,  recommended ecutwfc=26.0 Ha = 52.0 Ry
# Use the maximum: ecutwfc = 80.0 Ry (Mo dominates)
# ecutrho = 4 * ecutwfc = 320.0 Ry (standard 4x ratio for charge density)

ECUTWFC_RY = 80.0   # Ry
ECUTRHO_RY = 320.0  # Ry

# --- SCF settings ---
# calculation = 'scf'
# prefix = 'mos2' (consistent across SCF and bands)
# outdir = './tmp' (relative to calculations/NNN, where pw.x runs)
# pseudo_dir = './' (pseudopotential files are in calculations/NNN/)
# occupations = 'fixed' (insulator, 2H MoS2 is a semiconductor)
# conv_thr = 1.0e-8 Ry (tight convergence for band structure)
# kpts = (4, 4, 1) Monkhorst-Pack grid (hexagonal: 4x4 in-plane, 1 along c)
# nspin = 1 (non-magnetic, spin-unpolarized)

# --- Bands settings ---
# calculation = 'bands'
# nbnd = 40 (occupied bands = 28 from 56 electrons / 2; pad to 40 for conduction bands)
# kpts = explicit k-path array (n_kpts, 4) from seekpath, crystal coordinates

NBND = 40

# --- K-path interpolation ---
# 20 points per segment for smooth band structure
POINTS_PER_SEGMENT = 20

# --- Pseudopotential filenames ---
PP_MO = 'Mo.pbesol_upf'
PP_S = 'S.pbesol_upf'

# --- Template directory with pseudopotentials ---
TEMPLATE_DIR = 'template'

# --- Structures directory ---
STRUCTURES_DIR = 'structures'

# --- Calculations directory ---
CALC_DIR = 'calculations'


def get_kpath(atoms):
    """
    Derive the k-path from the exact ASE Atoms object using seekpath.
    Returns:
      - prim: the primitive cell Atoms object (to use in CELL_PARAMETERS)
      - kpts_array: (n_kpts, 4) array of k-points in crystal coordinates
      - label_indices: list of (label, index_in_kpts_list) pairs
      - discontinuities: list of indices where the path jumps
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    
    # Use symprec=0.01 (matching pymatgen's default) to avoid under-detecting symmetry
    res = seekpath.get_path(cell_tuple, symprec=0.01)
    
    # Verify space group consistency between seekpath and spglib
    spg_dataset = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    assert res['spacegroup_number'] == spg_dataset.number, \
        f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={spg_dataset.number}"
    
    # Build the primitive cell from seekpath's own reduced cell
    prim = Atoms(
        cell=res['primitive_lattice'],
        scaled_positions=res['primitive_positions'],
        numbers=res['primitive_types'],
        pbc=True
    )
    
    # res['point_coords'] is a DICT mapping label -> [x, y, z] fractional coords
    # (confirmed via diagnose_seekpath.py: type is dict, keys are label strings)
    point_coords = res['point_coords']
    
    # Build the explicit k-point list by interpolating each segment
    # res['path'] is a list of (label_from, label_to) tuples
    kpts_list = []
    label_indices = []  # (label, index_in_kpts_list)
    discontinuities = []  # indices where the path jumps
    
    for i_seg, (label_from, label_to) in enumerate(res['path']):
        coord_from = np.array(point_coords[label_from])
        coord_to = np.array(point_coords[label_to])
        
        # Check for discontinuity: if this is not the first segment and the
        # previous segment's end differs from this segment's start
        if i_seg > 0:
            prev_label_to = res['path'][i_seg - 1][1]
            prev_coord_to = np.array(point_coords[prev_label_to])
            if not np.allclose(coord_from, prev_coord_to, atol=1e-6):
                # Discontinuity: record the index where the jump occurs
                discontinuities.append(len(kpts_list))
        
        # Record the start label for EVERY segment (INVARIANT 3: every segment
        # must contribute its own start point, and the label must be recorded)
        label_indices.append((label_from, len(kpts_list)))
        
        # Interpolate from coord_from to coord_to
        # Include the start point for every segment (INVARIANT 3)
        for i in range(POINTS_PER_SEGMENT + 1):
            t = i / POINTS_PER_SEGMENT
            kpt = coord_from + t * (coord_to - coord_from)
            kpts_list.append(kpt)
        
        # Record the label at the end of this segment
        label_indices.append((label_to, len(kpts_list) - 1))
    
    # Convert to array: (n_kpts, 4) with weight column = 1.0
    kpts_array = np.zeros((len(kpts_list), 4))
    for i, kpt in enumerate(kpts_list):
        kpts_array[i, :3] = kpt
        kpts_array[i, 3] = 1.0  # weight (unused for non-self-consistent bands)
    
    return prim, kpts_array, label_indices, discontinuities


def write_pw_in(atoms, calc_dir, calculation, kpts, nbnd=None):
    """
    Write a pw.x input file using write_espresso_in.
    """
    # Build the input_data dict (flat, let ASE route to correct namelists)
    input_data = {
        'calculation': calculation,
        'prefix': 'mos2',
        'outdir': './tmp',
        'pseudo_dir': './',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': 1.0e-8,
        'nspin': 1,
    }
    
    if calculation == 'bands':
        input_data['nbnd'] = nbnd
    
    # Pseudopotential filenames
    pseudos = {'Mo': PP_MO, 'S': PP_S}
    
    # Write the input file
    in_file = os.path.join(calc_dir, 'pw.in' if calculation == 'scf' else 'bands.in')
    write_espresso_in(in_file, atoms, input_data, pseudos, kpts=kpts)


def write_bands_post_in(calc_dir, nnn):
    """
    Write the bands.x post-processing input file.
    """
    # bands.x input format:
    # &BANDS
    #   prefix = 'mos2'
    #   outdir = './tmp'
    #   filband = 'NNN.bands.dat'
    # /
    
    content = f"""&BANDS
  prefix = 'mos2'
  outdir = './tmp'
  filband = '{nnn}.bands.dat'
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(content)


def write_band_labels(calc_dir, label_indices, discontinuities):
    """
    Write band_labels.dat with high-symmetry labels and their indices.
    Format: one line per label: "LABEL INDEX"
    For discontinuities, label with both sides: "LABEL1|LABEL2 INDEX"
    """
    # Create a dict of index -> label(s)
    index_to_labels = {}
    for label, idx in label_indices:
        if idx in index_to_labels:
            # Multiple labels at the same index (discontinuity)
            index_to_labels[idx] += f"|{label}"
        else:
            index_to_labels[idx] = label
    
    with open(os.path.join(calc_dir, 'band_labels.dat'), 'w') as f:
        for idx in sorted(index_to_labels.keys()):
            f.write(f"{index_to_labels[idx]} {idx}\n")


def main():
    # Create the calculations directory
    os.makedirs(CALC_DIR, exist_ok=True)
    
    # Process each structure
    for nnn in range(51):
        nnn_str = f"{nnn:03d}"
        struct_file = os.path.join(STRUCTURES_DIR, f"structure_{nnn_str}.cif")
        calc_dir = os.path.join(CALC_DIR, nnn_str)
        
        # Create the calculation directory
        os.makedirs(calc_dir, exist_ok=True)
        
        # Copy pseudopotential files
        for pp_file in [PP_MO, PP_S]:
            src = os.path.join(TEMPLATE_DIR, pp_file)
            dst = os.path.join(calc_dir, pp_file)
            shutil.copy2(src, dst)
        
        # Read the structure
        atoms = read(struct_file)
        
        # Derive the k-path from this exact structure (INVARIANT 3)
        prim, kpts_array, label_indices, discontinuities = get_kpath(atoms)
        
        # INVARIANT 1: assert nat/ntyp from the actual Atoms object
        nat = len(prim)
        ntyp = len(set(prim.get_chemical_symbols()))
        assert nat == len(atoms), f"Structure {nnn_str}: nat mismatch after reduction"
        assert ntyp == len(set(atoms.get_chemical_symbols())), \
            f"Structure {nnn_str}: ntyp mismatch after reduction"
        
        # Write SCF input (pw.in)
        # kpts = (4, 4, 1) Monkhorst-Pack grid
        write_pw_in(prim, calc_dir, 'scf', kpts=(4, 4, 1))
        
        # Write bands input (bands.in)
        # kpts = explicit k-path array
        write_pw_in(prim, calc_dir, 'bands', kpts=kpts_array, nbnd=NBND)
        
        # Write bands.x post-processing input
        write_bands_post_in(calc_dir, nnn_str)
        
        # Write band labels
        write_band_labels(calc_dir, label_indices, discontinuities)
        
        print(f"Setup complete for structure {nnn_str}: "
              f"nat={nat}, ntyp={ntyp}, "
              f"kpts={kpts_array.shape[0]}, "
              f"discontinuities={len(discontinuities)}")
    
    print(f"\nDone. Set up 51 calculation directories in {CALC_DIR}/")


if __name__ == '__main__':
    main()
