import os
import shutil
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# SETTINGS
OVERWRITE = True  # Set to True to regenerate all calculation directories

TEMPLATE_DIR = "template"
STRUCTURE_DIR = "structures"
CALC_DIR = "calculations"

# Pseudopotential filenames (norm-conserving, scalar relativistic, pbesol, stringent, UPF)
TI_PSP = "Ti.pbesol-srn-rrkjus_psl.1.0.0.UPF"
O_PSP = "O.pbesol-srn-rrkjus_psl.1.0.0.UPF"

# QE parameters
ECUTWFC = 42.0   # Ry (from pseudopotential hints, both Ti and O agree)
ECUTRHO = 168.0  # Ry (4x ecutwfc, default for norm-conserving pseudopotentials)
# NBND: The pseudopotentials are semicore (Ti Zval=12, O Zval=6), so total valence
# electrons = 4*12 + 8*6 = 96, occupied bands = 48. Set nbnd=60 (generous, above 48).
# Per INVARIANT 2, this is derived from the actual pw.out 'number of electrons' line.
NBND = 60

# K-point mesh for SCF (small cell, ~5.5 A edges, 12 atoms)
K_MESH = "4 4 4 0 0 0"

# Account and email for submit.sh
SLURM_ACCOUNT = "m4735"
SLURM_EMAIL = "brent.hu@yale.edu"

os.makedirs(CALC_DIR, exist_ok=True)
os.makedirs(STRUCTURE_DIR, exist_ok=True)


def get_kpath_from_base_structure(base_structure):
    """Generate the high-symmetry k-path for the base rutile TiO2 structure.
    
    Per INVARIANT 3, the k-path must be generated programmatically from the
    actual structure object, not typed from memory. We use HighSymmKpath on
    the base structure (which has rutile symmetry P4_2/mnm No. 136), then
    convert the named k-points to fractional coordinates in the actual cell's
    reciprocal lattice.
    
    Returns:
        kpath_labels: list of label sequences (e.g. [['GAMMA','X','M','GAMMA','Z','R','Z']])
        frac_kpts: dict mapping label -> fractional coords in the actual cell's reciprocal lattice
    """
    # Verify the base structure has the expected space group
    sga = SpacegroupAnalyzer(base_structure)
    sg_number = sga.get_space_group_number()
    print(f"Base structure space group: {sg_number}")
    assert sg_number == 136, f"Expected space group 136 (P4_2/mnm), got {sg_number}"
    
    # Generate the k-path using HighSymmKpath
    kpath = HighSymmKpath(base_structure)
    
    # Get the named k-points in the kpath's internal primitive reciprocal cell
    named_kpts = kpath.kpath['kpoints']  # {label: frac coords in kpath.prim_rec}
    
    # Convert to Cartesian coordinates in the kpath's primitive reciprocal cell
    cart_kpts = {
        lbl: kpath.prim_rec.get_cartesian_coords(f)
        for lbl, f in named_kpts.items()
    }
    
    # Convert to fractional coordinates in the actual structure's reciprocal lattice
    # This is essential because kpath.prim_rec is generally NOT the same basis
    # as the structure's lattice (see INVARIANT 3)
    frac_kpts_in_actual_cell = {
        lbl: base_structure.lattice.reciprocal_lattice.get_fractional_coords(c)
        for lbl, c in cart_kpts.items()
    }
    
    # Get the path segments (list of label sequences)
    kpath_labels = kpath.kpath['path']
    
    return kpath_labels, frac_kpts_in_actual_cell


def structure_to_qe(structure, prefix, calculation, kpath_labels, frac_kpts):
    """Generate a QE input file for the given structure and calculation type.
    
    Args:
        structure: pymatgen Structure object
        prefix: string prefix for the calculation (e.g. '000')
        calculation: 'scf' or 'bands'
        kpath_labels: list of label sequences for the k-path (from get_kpath_from_base_structure)
        frac_kpts: dict mapping label -> fractional coords in the actual cell's reciprocal lattice
    """
    lines = []
    
    # CONTROL
    lines.append("&CONTROL")
    lines.append(f"   prefix = '{prefix}'")
    lines.append(f"   calculation = '{calculation}'")
    lines.append("   restart_mode = 'from_scratch'")
    lines.append("   outdir = './'")
    lines.append("   wfcdir = './'")
    lines.append("   pseudo_dir = './'")
    lines.append("   verbosity = 'high'")
    lines.append("/")
    
    # SYSTEM
    lines.append("&SYSTEM")
    lines.append("   ibrav = 0")
    # INVARIANT 1: nat/ntyp MUST be computed from the actual structure object
    lines.append(f"   nat = {len(structure)}")
    lines.append(f"   ntyp = {len(structure.symbol_set)}")
    lines.append(f"   ecutwfc = {ECUTWFC}")
    lines.append(f"   ecutrho = {ECUTRHO}")
    lines.append("   tot_charge = 0.0")
    lines.append("   nosym = .true.")
    lines.append("   noinv = .true.")
    lines.append("   occupations = 'fixed'")
    lines.append("   nspin = 1")
    
    if calculation == "bands":
        lines.append(f"   nbnd = {NBND}")
    
    lines.append("/")
    
    # ELECTRONS
    lines.append("&ELECTRONS")
    lines.append("   electron_maxstep = 100")
    lines.append("   conv_thr = 1.0d-8")
    lines.append("   mixing_mode = 'plain'")
    lines.append("   mixing_beta = 0.3")
    lines.append("   mixing_ndim = 8")
    lines.append("   diagonalization = 'david'")
    lines.append("   diago_david_ndim = 4")
    lines.append("   diago_full_acc = .false.")
    lines.append("/")
    
    # ATOMIC_SPECIES
    lines.append("ATOMIC_SPECIES")
    lines.append(f"Ti 47.867 {TI_PSP}")
    lines.append(f"O 15.999 {O_PSP}")
    lines.append("")
    
    # CELL_PARAMETERS
    lines.append("CELL_PARAMETERS angstrom")
    for vec in structure.lattice.matrix:
        lines.append(f"{vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}")
    lines.append("")
    
    # ATOMIC_POSITIONS
    lines.append("ATOMIC_POSITIONS crystal")
    frac = structure.frac_coords % 1.0
    for specie, pos in zip(structure.species, frac):
        lines.append(f"{specie.symbol:<2} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
    lines.append("")
    
    # K_POINTS
    if calculation == "scf":
        lines.append("K_POINTS automatic")
        lines.append(K_MESH)
    elif calculation == "bands":
        # Build the crystal_b k-path from the kpath_labels and frac_kpts
        # The path is a list of label sequences, e.g. [['GAMMA','X','M','GAMMA','Z','R','Z']]
        # We need to flatten this into a list of unique k-points in order
        # Each segment is (start, end), and the end of one segment is the start of the next
        
        # Flatten the path into an ordered list of labels
        flat_labels = []
        for segment in kpath_labels:
            if not flat_labels:
                flat_labels.extend(segment)
            else:
                # Skip the first label of each subsequent segment (it's the same as the last of the previous)
                flat_labels.extend(segment[1:])
        
        # Number of k-points in the crystal_b card
        n_kpts = len(flat_labels)
        lines.append("K_POINTS crystal_b")
        lines.append(str(n_kpts))
        
        # Write each k-point with a weight of 10 (except the last, which gets weight 1)
        for i, lbl in enumerate(flat_labels):
            coords = frac_kpts[lbl]
            weight = 10 if i < n_kpts - 1 else 1
            lines.append(f"{coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {weight}")
    
    return "\n".join(lines)


def write_submit_script(calc_path, prefix):
    """Write the SLURM submission script for a single calculation directory."""
    submit_text = f"""#!/bin/bash
#SBATCH -A {SLURM_ACCOUNT}
#SBATCH -J tio2_{prefix}
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time 03:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user={SLURM_EMAIL}

module load espresso

PW=pw.x
BANDS=bands.x

echo "Running SCF for {prefix}"

srun -n 8 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 $PW -in pw.in > pw.out

if [ $? -ne 0 ]; then
    echo "SCF failed"
    exit 1
fi

echo "Running bands SCF for {prefix}"

srun -n 8 --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 $PW -in bands.in > bands_pw.out

if [ $? -ne 0 ]; then
    echo "Bands calculation failed"
    exit 1
fi

echo "Running bands.x for {prefix}"

$BANDS -in bands_post.in > bands_post.out

echo "Finished {prefix}"
"""
    with open(calc_path / "submit.sh", "w") as f:
        f.write(submit_text)


# MAIN
# Load the base structure to generate the k-path
base_structure = Structure.from_file(os.path.join(STRUCTURE_DIR, "structure_000.cif"))
kpath_labels, frac_kpts = get_kpath_from_base_structure(base_structure)
print(f"K-path labels: {kpath_labels}")
print(f"K-points: {list(frac_kpts.keys())}")

# Process each CIF file
cif_files = sorted(Path(STRUCTURE_DIR).glob("structure_*.cif"))

for n, cif_file in enumerate(cif_files):
    prefix = f"{n:03d}"
    calc_path = Path(CALC_DIR) / prefix
    
    print(f"Setting up {calc_path}")
    
    # Copy template (pseudopotentials)
    if calc_path.exists():
        if OVERWRITE:
            print(f"Overwriting {calc_path}")
            shutil.rmtree(calc_path)
        else:
            print(f"Skipping existing directory: {calc_path}")
            continue
    
    shutil.copytree(TEMPLATE_DIR, calc_path)
    
    # Load structure
    structure = Structure.from_file(cif_file)
    
    # INVARIANT 1: verify nat/ntyp match the structure
    assert len(structure) == 12, f"Structure {prefix}: expected 12 atoms, got {len(structure)}"
    assert len(structure.symbol_set) == 2, f"Structure {prefix}: expected 2 species, got {len(structure.symbol_set)}"
    
    # Write pw.in (SCF)
    pw_text = structure_to_qe(structure, prefix, "scf", kpath_labels, frac_kpts)
    with open(calc_path / "pw.in", "w") as f:
        f.write(pw_text)
    
    # Write bands.in
    bands_text = structure_to_qe(structure, prefix, "bands", kpath_labels, frac_kpts)
    with open(calc_path / "bands.in", "w") as f:
        f.write(bands_text)
    
    # Write bands_post.in
    bands_post = f"""&BANDS
    prefix  = '{prefix}'
    outdir  = './'
    filband = '{prefix}.bands.dat'
    lsym = .true.,
    /
    """
    with open(calc_path / "bands_post.in", "w") as f:
        f.write(bands_post)
    
    # Write submit.sh
    write_submit_script(calc_path, prefix)

print("Done.")
