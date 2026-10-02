import os
import shutil
import numpy as np
import seekpath
import spglib
from ase.io import read, write
from ase.io.espresso import write_espresso_in, write_fortran_namelist
from ase import Atoms

# =============================================================================
# setup_jobs.py - Generate QE input files for all 51 TiO2 structures
# =============================================================================

# Pseudopotential files (fetched in STEP 3)
# Recommended cutoffs: Ti=42.0 Ha, O=46.0 Ha
# Per INVARIANT 5: convert Ha to Ry (1 Ha = 2 Ry)
# ecutwfc = 2 * max(42.0, 46.0) = 92.0 Ry
ECUTWFC_RY = 92.0
ECUTRHO_RY = 4 * ECUTWFC_RY  # 368.0 Ry

# Read z_valence from pseudopotential files to compute electron count
# This is required by INVARIANT 2 - never assume valence electron counts
# UPF files are XML; z_valence is an attribute of the PP_HEADER element
import xml.etree.ElementTree as ET

def get_z_valence_from_upf(filepath):
    """Parse a UPF pseudopotential file and return z_valence.
    
    In UPF v2.0.1 format, z_valence is an attribute of the <PP_HEADER> element.
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    for elem in root.iter():
        if elem.tag == 'PP_HEADER':
            if 'z_valence' in elem.attrib:
                return float(elem.attrib['z_valence'])
    raise ValueError(f"Could not find z_valence in {filepath}")

# Get z_valence for Ti and O
zval_ti = get_z_valence_from_upf('template/Ti.pbesol_upf')
zval_o = get_z_valence_from_upf('template/O.pbesol_upf')
print(f"z_valence: Ti={zval_ti}, O={zval_o}")

# For the primitive cell: 2 Ti + 4 O
total_electrons = 2 * zval_ti + 4 * zval_o
occupied_bands = int(total_electrons / 2)  # spin-unpolarized
print(f"Total electrons: {total_electrons}, Occupied bands: {occupied_bands}")

# nbnd should be at least 2x occupied bands for a good band structure
NBND = max(2 * occupied_bands, 64)
print(f"nbnd = {NBND}")

# =============================================================================
# K-path generation using seekpath (INVARIANT 3)
# =============================================================================

def generate_kpath(atoms, n_points_per_segment=20):
    """
    Generate a k-path for band structure calculation using seekpath.
    
    Per INVARIANT 3:
    - Use seekpath with symprec=0.01 (not the default 1e-5)
    - Use the primitive cell that seekpath returns
    - The k-point coordinates are in fractional coordinates of seekpath's
      primitive cell, so we must use that same cell in CELL_PARAMETERS
    - Every segment must contribute its own start point
    - Record both labels at jumps
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    
    # seekpath with symprec=0.01
    res = seekpath.get_path(cell_tuple, symprec=0.01)
    
    # Cross-check with spglib
    ds = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    assert res['spacegroup_number'] == ds.number, (
        f"Space group mismatch: seekpath={res['spacegroup_number']}, spglib={ds.number}")
    
    # The primitive cell that seekpath's k-path refers to
    prim = Atoms(
        cell=res['primitive_lattice'],
        scaled_positions=res['primitive_positions'],
        numbers=res['primitive_types'],
        pbc=True,
    )
    
    # Build the k-point list from the path
    # res['path'] is a list of (label_from, label_to) segments
    # res['point_coords'] gives fractional coordinates of named points
    path = res['path']
    point_coords = res['point_coords']
    
    # Build a dictionary of label -> coordinates
    # point_coords is a list of (label, coords) or similar
    if isinstance(point_coords, dict):
        label_to_coords = point_coords
    else:
        # It might be a list of tuples
        label_to_coords = {}
        for item in point_coords:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                label_to_coords[item[0]] = item[1]
            elif isinstance(item, dict):
                for k, v in item.items():
                    label_to_coords[k] = v
    
    # Build the full k-point list
    kpoints = []
    labels = []  # (index, label) for each high-symmetry point
    
    for i, (label_from, label_to) in enumerate(path):
        coords_from = np.array(label_to_coords[label_from], dtype=float)
        coords_to = np.array(label_to_coords[label_to], dtype=float)
        
        # Add the start point of this segment
        if i == 0 or not np.allclose(kpoints[-1], coords_from):
            # This is either the first segment or a jump
            kpoints.append(coords_from)
            if i == 0:
                labels.append((len(kpoints) - 1, label_from))
            else:
                # Jump: record both labels
                labels.append((len(kpoints) - 1, f"{label_from}|{label_to}"))
        
        # Interpolate interior points (excluding the start, including the end)
        for j in range(1, n_points_per_segment + 1):
            t = j / n_points_per_segment
            k = (1 - t) * coords_from + t * coords_to
            kpoints.append(k)
        
        # Record the end point label (if it's not a jump to the next segment)
        if i < len(path) - 1:
            next_label_from = path[i + 1][0]
            if not np.allclose(coords_to, np.array(label_to_coords[next_label_from], dtype=float)):
                # Jump at the end of this segment
                labels.append((len(kpoints) - 1, f"{label_to}|{next_label_from}"))
            else:
                # Continuous: just record the label
                labels.append((len(kpoints) - 1, label_to))
        else:
            # Last segment
            labels.append((len(kpoints) - 1, label_to))
    
    kpoints = np.array(kpoints)
    
    # Add weights column (all 1.0, unused for bands calculation)
    kpoints_with_weights = np.column_stack([kpoints, np.ones(len(kpoints))])
    
    return prim, kpoints_with_weights, labels, res


# =============================================================================
# Main: process each structure
# =============================================================================

for nnn in range(51):
    struct_id = f"{nnn:03d}"
    calc_dir = f"calculations/{struct_id}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into the calculation directory
    # (pseudo_dir='.' means pw.x looks in the current directory)
    shutil.copy('template/Ti.pbesol_upf', os.path.join(calc_dir, 'Ti.pbesol_upf'))
    shutil.copy('template/O.pbesol_upf', os.path.join(calc_dir, 'O.pbesol_upf'))
    
    # Read the structure
    atoms = read(f'structures/structure_{struct_id}.cif')
    
    # INVARIANT 1: nat and ntyp must be computed from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f"Structure {struct_id}: expected nat=6, got {nat}"
    assert ntyp == 2, f"Structure {struct_id}: expected ntyp=2, got {ntyp}"
    
    # Generate k-path using seekpath on THIS structure
    # For structure 000 (unperturbed), this gives the full P4_2/mnm path
    # For perturbed structures, symmetry may be lower, and seekpath will
    # detect the actual symmetry of the perturbed structure
    prim, kpoints, labels, seekpath_res = generate_kpath(atoms, n_points_per_segment=20)
    
    # The primitive cell from seekpath is what we write into CELL_PARAMETERS
    # This ensures the k-point coordinates are consistent with the cell
    # (INVARIANT 3: use seekpath's primitive cell, not the original)
    
    # Pseudopotential mapping
    pseudopotentials = {
        'Ti': 'Ti.pbesol_upf',
        'O': 'O.pbesol_upf',
    }
    
    # =================================================================
    # SCF input (pw.in)
    # =================================================================
    scf_input_data = {
        'calculation': 'scf',
        'prefix': 'tio2',
        'outdir': './tmp',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': 1.0e-8,
    }
    
    # K-points for SCF: Monkhorst-Pack mesh
    # For a 6-atom primitive cell with ~4.6 A lattice constants,
    # a 4x4x4 mesh is reasonable
    scf_kpts = (4, 4, 4)
    
    with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
        write_espresso_in(
            f, prim, scf_input_data,
            pseudopotentials=pseudopotentials,
            kpts=scf_kpts,
            crystal_coordinates=True,
        )
    
    # =================================================================
    # Bands input (bands.in)
    # =================================================================
    bands_input_data = {
        'calculation': 'bands',
        'prefix': 'tio2',
        'outdir': './tmp',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'nbnd': NBND,
    }
    
    with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
        write_espresso_in(
            f, prim, bands_input_data,
            pseudopotentials=pseudopotentials,
            kpts=kpoints,  # (n_kpts, 4) array in crystal coordinates
            crystal_coordinates=True,
        )
    
    # =================================================================
    # Bands post-processing input (bands_post.in)
    # =================================================================
    bands_post_input_data = {
        'prefix': 'tio2',
        'outdir': './tmp',
        'filband': f'{struct_id}.bands.dat',
    }
    
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        write_fortran_namelist(f, bands_post_input_data, binary='bands')
    
    # =================================================================
    # Band labels file (for plotting)
    # =================================================================
    with open(os.path.join(calc_dir, 'band_labels.dat'), 'w') as f:
        f.write("# index label\n")
        for idx, label in labels:
            f.write(f"{idx} {label}\n")
    
    print(f"Structure {struct_id}: nat={nat}, ntyp={ntyp}, "
          f"space_group={seekpath_res['spacegroup_number']}, "
          f"n_kpts={len(kpoints)}, nbnd={NBND}")

print("\nAll calculation directories created.")
