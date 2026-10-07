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
#   - pseudopotential files (copied from template/)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (bands input, written with write_espresso_in)
#   - bands_post.in (bands.x input)
#   - band_labels.dat (high-symmetry labels and their indices)
# =============================================================================

# --- Pseudopotential files ---
# Full paths for copying; basenames for the ATOMIC_SPECIES card (pseudo_dir='./')
pp_files_full = {
    'Ga': 'template/Ga.pbesol_s-rrkjus_psl.1.0.0.UPF',
    'As': 'template/As.pbesol_s-rrkjus_psl.1.0.0.UPF',
}
pp_files = {elem: os.path.basename(path) for elem, path in pp_files_full.items()}

# --- QE settings (from STEP 4 note) ---
# INVARIANT 5: ecutwfc in Ry (converted from Ha: 1 Ha = 2 Ry)
# Ga recommended: 57 Ha -> 114 Ry; As recommended: 52 Ha -> 104 Ry
# Use max: ecutwfc = 114 Ry
ecutwfc = 114.0  # Ry
ecutrho = 4.0 * ecutwfc  # 456 Ry (standard 4x ratio for charge density)

# Total electrons = 21 (Ga) + 23 (As) = 44
# Occupied bands = 44/2 = 22 (spin-unpolarized, insulator)
# nbnd for bands: pad above 22, use 48 (enough for ~10 eV above VBM)
nbnd = 48

# SCF k-mesh: 8x8x8 Monkhorst-Pack (zincblende, 2 atoms, good density)
scf_kpts = (8, 8, 8)

# Bands k-path: from seekpath on the SAME Atoms object written into CELL_PARAMETERS
# Path: GAMMA-X-U-K-GAMMA-L-W-X (6 segments)
# 20 points per segment including both endpoints
npts_per_segment = 20

# --- Common input_data for pw.x ---
common_input = {
    'prefix': 'gaas',
    'outdir': './tmp',
    'pseudo_dir': './',
    'ecutwfc': ecutwfc,
    'ecutrho': ecutrho,
    'conv_thr': 1e-8,
}

# --- SCF input_data ---
scf_input = dict(common_input)
scf_input.update({
    'calculation': 'scf',
    'occupations': 'fixed',  # insulator, 22 occupied bands, no smearing
})

# --- Bands input_data ---
bands_input = dict(common_input)
bands_input.update({
    'calculation': 'bands',
    'nbnd': nbnd,
})


def build_kpath(atoms):
    """Build the k-point list and labels from seekpath for the given Atoms object.
    
    INVARIANT 3: the k-path must be generated from the EXACT same Atoms object
    that is written into CELL_PARAMETERS. We use seekpath with symprec=0.01.
    
    Returns:
        kpoints: list of (kx, ky, kz) in fractional coordinates of the primitive cell
        labels: list of (label, index) for high-symmetry points
        discontinuities: list of indices where the path jumps (no line should be drawn)
    """
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    res = seekpath.get_path(cell_tuple, symprec=0.01)
    
    # INVARIANT 3 check: seekpath space group must match spglib's own analysis
    spg = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    assert res['spacegroup_number'] == spg.number, (res['spacegroup_number'], spg.number)
    
    path = res['path']  # list of (label_from, label_to) segments
    point_coords = res['point_coords']  # dict of label -> [kx, ky, kz] fractional
    
    # Build the k-point list by interpolating along each segment
    kpoints = []
    labels = []  # (label, index) for high-symmetry points
    discontinuities = []  # indices where the path jumps
    
    for i, (from_label, to_label) in enumerate(path):
        from_coord = np.array(point_coords[from_label])
        to_coord = np.array(point_coords[to_label])
        
        # Check for discontinuity: if this is not the first segment and the
        # start of this segment differs from the end of the previous segment
        if i > 0:
            prev_from, prev_to = path[i-1]
            prev_end = np.array(point_coords[prev_to])
            if np.linalg.norm(from_coord - prev_end) > 1e-6:
                # Discontinuity: record the index where the jump occurs
                discontinuities.append(len(kpoints))
        
        # Interpolate npts_per_segment points including both endpoints
        for j in range(npts_per_segment):
            t = j / (npts_per_segment - 1)
            k = from_coord + t * (to_coord - from_coord)
            kpoints.append(tuple(k))
        
        # Record the high-symmetry point at the start of this segment
        if i == 0:
            labels.append((from_label, 0))
        else:
            prev_from, prev_to = path[i-1]
            prev_end = np.array(point_coords[prev_to])
            if np.linalg.norm(from_coord - prev_end) < 1e-6:
                # Same point, already recorded as the end of the previous segment
                pass
            else:
                # Discontinuity: record this start point
                labels.append((from_label, len(kpoints) - npts_per_segment))
        
        # Record the high-symmetry point at the end of this segment
        if i < len(path) - 1:
            next_from, next_to = path[i+1]
            next_start = np.array(point_coords[next_from])
            if np.linalg.norm(to_coord - next_start) > 1e-6:
                # This end point is NOT the start of the next segment (discontinuity)
                labels.append((to_label, len(kpoints) - 1))
        else:
            # Last segment: record the end point
            labels.append((to_label, len(kpoints) - 1))
    
    return kpoints, labels, discontinuities


def write_bands_post_in(path, prefix, outdir, filband):
    """Write a bands.x input file."""
    with open(path, 'w') as f:
        f.write("&BANDS\n")
        f.write(f"  prefix = '{prefix}'\n")
        f.write(f"  outdir = '{outdir}'\n")
        f.write(f"  filband = '{filband}'\n")
        f.write("/\n")
        f.write("EOF\n")


def write_band_labels(path, labels):
    """Write band_labels.dat with high-symmetry labels and their indices."""
    with open(path, 'w') as f:
        for label, index in labels:
            f.write(f"{label} {index}\n")


# --- Main loop over structures ---
for nnn in range(51):
    struct_id = f"{nnn:03d}"
    calc_dir = f"calculations/{struct_id}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files (using full paths for the source)
    for elem, pp_path in pp_files_full.items():
        shutil.copy2(pp_path, calc_dir)
    
    # Read the structure
    atoms = read(f"structures/structure_{struct_id}.cif")
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    
    # Build the k-path from THIS structure (INVARIANT 3)
    kpoints, labels, discontinuities = build_kpath(atoms)
    
    # Write pw.in (SCF)
    scf_input_data = dict(scf_input)
    scf_input_data['nat'] = nat
    scf_input_data['ntyp'] = ntyp
    
    write_espresso_in(
        f"{calc_dir}/pw.in",
        atoms,
        input_data=scf_input_data,
        pseudopotentials=pp_files,  # basenames only, pseudo_dir='./'
        kpts=scf_kpts,
        crystal_coordinates=True,
    )
    
    # Write bands.in (bands)
    bands_input_data = dict(bands_input)
    bands_input_data['nat'] = nat
    bands_input_data['ntyp'] = ntyp
    
    # kpoints as a NumPy array of shape (n_kpts, 4) for explicit k-points
    # weight is unused for a non-self-consistent bands run, set to 1.0
    kpts_array = np.array([(kx, ky, kz, 1.0) for (kx, ky, kz) in kpoints], dtype=float)
    
    write_espresso_in(
        f"{calc_dir}/bands.in",
        atoms,
        input_data=bands_input_data,
        pseudopotentials=pp_files,  # basenames only, pseudo_dir='./'
        kpts=kpts_array,
        crystal_coordinates=True,
    )
    
    # Write bands_post.in (bands.x)
    write_bands_post_in(
        f"{calc_dir}/bands_post.in",
        prefix='gaas',
        outdir='./tmp',
        filband=f"{struct_id}.bands.dat",
    )
    
    # Write band_labels.dat
    write_band_labels(f"{calc_dir}/band_labels.dat", labels)
    
    print(f"Wrote {calc_dir}/ (nat={nat}, ntyp={ntyp}, {len(kpoints)} k-points, "
          f"{len(discontinuities)} discontinuities)")

print("Done: all calculation directories created.")
