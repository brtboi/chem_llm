import os
import shutil
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# ============================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ with:
#   - pseudopotential files (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input)
#   - bands.in (NSCF bands input)
#   - bands_post.in (bands.x post-processing input)
# ============================================================

# --- Pseudopotential setup ---
# Read z_valence from the actual UPF files (INVARIANT 2)
B_pp = read_upf_header('template/B.upf')
N_pp = read_upf_header('template/N.upf')
z_val_B = B_pp['z_valence']  # 3.0
z_val_N = N_pp['z_valence']  # 5.0

# Recommended cutoffs in Hartree (from get_pseudopotential results)
# INVARIANT 5: 1 Ha = 2 Ry, so convert to Rydberg for QE
ecutwfc_B_ha = 38.0  # Ha, from Pseudo-Dojo recommended cutoff for B
ecutwfc_N_ha = 42.0  # Ha, from Pseudo-Dojo recommended cutoff for N

# Use the maximum cutoff (in Ry) to ensure both pseudopotentials are well-converged
ecutwfc_ry = 2.0 * max(ecutwfc_B_ha, ecutwfc_N_ha)  # 84 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 336 Ry (standard 4x ratio)

# INVARIANT 5: assert the conversion is correct
assert ecutwfc_ry == 84.0, f'Expected ecutwfc=84 Ry, got {ecutwfc_ry}'
assert ecutrho_ry == 336.0, f'Expected ecutrho=336 Ry, got {ecutrho_ry}'

# --- Electron count and nbnd ---
# Total valence electrons = 2*B + 2*N = 2*3 + 2*5 = 16
# Occupied bands (spin-unpolarized) = 16/2 = 8
# nbnd must be >= occupied bands; use 16 for safety (INVARIANT 2)
n_electrons = 2 * int(z_val_B) + 2 * int(z_val_N)  # 16
n_occupied = n_electrons // 2  # 8
nbnd = 16  # padded above n_occupied
assert nbnd >= n_occupied, f'nbnd={nbnd} must be >= n_occupied={n_occupied}'

# --- Common QE settings ---
prefix = 'bn'
outdir = './tmp'
pseudo_dir = '../../template'  # relative to calculations/NNN
conv_thr = 1e-8  # in &ELECTRONS (INVARIANT 6)

# --- Pseudopotential filenames ---
pseudopotentials = {
    'B': 'B.upf',
    'N': 'N.upf'
}

# --- Process each structure ---
for nnn in range(51):
    struct_id = f'{nnn:03d}'
    calc_dir = f'calculations/{struct_id}'
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into the calculation directory
    # (pw.x runs inside calculations/NNN, so pseudo_dir must resolve from there)
    for elem, pp_file in pseudopotentials.items():
        src = f'template/{pp_file}'
        dst = f'{calc_dir}/{pp_file}'
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure
    atoms = ase.io.read(f'structures/structure_{struct_id}.cif')
    
    # INVARIANT 1: nat/ntyp from actual ASE object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 4, f'Structure {struct_id}: expected 4 atoms, got {nat}'
    assert ntyp == 2, f'Structure {struct_id}: expected 2 species, got {ntyp}'
    
    # Get the standard k-path and primitive cell (INVARIANT 3)
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in
    
    # Save kpath.json (holds labels, ticks, segments)
    kp_save(kp, f'{calc_dir}/kpath.json')
    
    # --- SCF k-point mesh ---
    # h-BN is an insulator (band_gap_eV=3.97), so metal=False
    scf_kpts = scf_kmesh(prim, metal=False)
    
    # --- SCF input (pw.in) ---
    scf_input_data = {
        'calculation': 'scf',
        'prefix': prefix,
        'outdir': outdir,
        'pseudo_dir': pseudo_dir,
        'ecutwfc': ecutwfc_ry,  # Ry
        'ecutrho': ecutrho_ry,  # Ry
        'occupations': 'fixed',  # insulator
        'conv_thr': conv_thr,  # in &ELECTRONS
        'nat': nat,
        'ntyp': ntyp,
    }
    
    write_espresso_in(
        f'{calc_dir}/pw.in',
        prim,
        input_data=scf_input_data,
        pseudopotentials=pseudopotentials,
        kpts=scf_kpts
    )
    
    # --- Bands input (bands.in) ---
    bands_input_data = {
        'calculation': 'bands',
        'prefix': prefix,
        'outdir': outdir,
        'pseudo_dir': pseudo_dir,
        'ecutwfc': ecutwfc_ry,  # Ry
        'ecutrho': ecutrho_ry,  # Ry
        'occupations': 'fixed',
        'conv_thr': conv_thr,
        'nat': nat,
        'ntyp': ntyp,
        'nbnd': nbnd,
    }
    
    # kpts from the k-path (INVARIANT 3)
    bands_kpts = kp_kpts(kp)
    
    write_espresso_in(
        f'{calc_dir}/bands.in',
        prim,
        input_data=bands_input_data,
        pseudopotentials=pseudopotentials,
        kpts=bands_kpts
    )
    
    # --- Bands post-processing input (bands_post.in) ---
    # bands.x input with &BANDS namelist
    bands_post_content = f"""&BANDS
    prefix = '{prefix}'
    outdir = '{outdir}'
    filband = '{struct_id}.bands.dat'
    lsym = .false.
/
"""
    with open(f'{calc_dir}/bands_post.in', 'w') as f:
        f.write(bands_post_content)
    
    print(f'Set up calculations/{struct_id}/ (nat={nat}, ntyp={ntyp}, kpts={scf_kpts})')

print('Done: all 51 calculation directories set up.')
