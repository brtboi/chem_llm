import os
import shutil
import ase.io
from ase.io.espresso import write_espresso_in, write_fortran_namelist
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (NSCF bands input, written with write_espresso_in)
#   - bands_post.in (bands.x input, written with write_fortran_namelist)
# =============================================================================

# Pseudopotential files (from template/)
# Ti: z_valence=12 (semicore 3s2 3p6 included), O: z_valence=6
# Cutoffs: Ti 42 Ha = 84 Ry, O 46 Ha = 92 Ry
# Use min = 84 Ry for ecutwfc (INVARIANT 5: 1 Ha = 2 Ry)
TI_PSP = 'template/Ti.pbesol_s-rrkjus_psl.1.0.0.UPF'
O_PSP = 'template/O.pbesol_s-rrkjus_psl.1.0.0.UPF'

# Read z_valence from the actual UPF files (INVARIANT 2)
ti_zv = read_upf_header(TI_PSP)['z_valence']
o_zv = read_upf_header(O_PSP)['z_valence']
print(f'Ti z_valence = {ti_zv}, O z_valence = {o_zv}')

# Total electrons for the primitive cell (2 Ti + 4 O)
n_electrons = 2 * ti_zv + 4 * o_zv
n_occupied = int(n_electrons / 2)  # spin-unpolarized
print(f'Total electrons = {n_electrons}, occupied bands = {n_occupied}')

# nbnd: pad above occupied bands. Use 2x occupied to be safe.
nbnd = 2 * n_occupied
print(f'nbnd = {nbnd}')

# Cutoffs (INVARIANT 5: convert from Ha to Ry)
# Ti recommended: 42 Ha = 84 Ry
# O recommended: 46 Ha = 92 Ry
# Use the minimum (84 Ry) for ecutwfc
ecutwfc_ry = 2.0 * 42.0  # 84 Ry (from Ti, the lower of the two)
ecutrho_ry = 4.0 * ecutwfc_ry  # 336 Ry (standard 4x ratio)
print(f'ecutwfc = {ecutwfc_ry} Ry, ecutrho = {ecutrho_ry} Ry')

# Pseudopotential filenames (relative to calculations/NNN/)
pseudopotentials = {
    'Ti': 'Ti.pbesol_s-rrkjus_psl.1.0.0.UPF',
    'O': 'O.pbesol_s-rrkjus_psl.1.0.0.UPF',
}

# SCF input data (flat dict -- ASE routes each key to the correct namelist, INVARIANT 6)
scf_input_data = {
    'calculation': 'scf',
    'prefix': 'scf',
    'outdir': 'tmp_scf',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',  # TiO2 is a semiconductor (band_gap_eV=1.77)
    'conv_thr': 1e-8,
}

# Bands input data (flat dict)
bands_input_data = {
    'calculation': 'bands',
    'prefix': 'scf',  # same as SCF so bands.x can read the SCF wavefunctions
    'outdir': 'tmp_scf',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
    'nbnd': nbnd,
}

# bands.x input data
bands_post_input_data = {
    'prefix': 'scf',
    'outdir': 'tmp_scf',
    'filband': 'NNN.bands.dat',  # will be replaced per structure
    'lsym': False,  # skip band-representation file (not needed, saves time)
}

# Process each structure
for nnn in range(51):
    nnn_str = f'{nnn:03d}'
    calc_dir = f'calculations/{nnn_str}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into the calculation directory
    for psp in [TI_PSP, O_PSP]:
        shutil.copy2(psp, calc_dir)
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{nnn_str}.cif')
    
    # INVARIANT 1: compute nat/ntyp from the actual ASE Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6, f'Structure {nnn_str}: expected 6 atoms, got {nat}'
    assert ntyp == 2, f'Structure {nnn_str}: expected 2 species, got {ntyp}'
    
    # Get the k-path for THIS structure (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # INVARIANT 1: nat/ntyp from the primitive cell
    nat_prim = len(prim)
    ntyp_prim = len(set(prim.get_chemical_symbols()))
    assert nat_prim == 6, f'Structure {nnn_str}: primitive cell expected 6 atoms, got {nat_prim}'
    assert ntyp_prim == 2, f'Structure {nnn_str}: primitive cell expected 2 species, got {ntyp_prim}'
    
    # Save kpath.json
    kp_save(kp, f'{calc_dir}/kpath.json')
    
    # SCF k-mesh (computed, never typed by hand)
    kpts_scf = scf_kmesh(prim, metal=False)
    print(f'Structure {nnn_str}: SCF k-mesh = {kpts_scf}')
    
    # Write pw.in (SCF)
    write_espresso_in(
        f'{calc_dir}/pw.in',
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_scf,
    )
    
    # Write bands.in (NSCF bands)
    # kpts = kp_kpts(kp) from chem_llm.kpath (INVARIANT 3)
    kpts_bands = kp_kpts(kp)
    write_espresso_in(
        f'{calc_dir}/bands.in',
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=kpts_bands,
    )
    
    # Write bands_post.in (bands.x input)
    # Replace NNN in filband with the actual structure number
    bands_post_data = dict(bands_post_input_data)
    bands_post_data['filband'] = f'{nnn_str}.bands.dat'
    write_fortran_namelist(
        f'{calc_dir}/bands_post.in',
        input_data=bands_post_data,
        binary='bands',
    )
    
    print(f'Wrote {calc_dir}/: pw.in, bands.in, bands_post.in, kpath.json')

print('\nDone. Set up 51 calculation directories.')
