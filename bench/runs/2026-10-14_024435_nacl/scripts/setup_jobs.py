import os
import shutil
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# =============================================================================
# setup_jobs.py
# For each structure NNN, create calculations/NNN/ with:
#   - pseudopotential files (copied from template/)
#   - kpath.json
#   - pw.in (SCF)
#   - bands.in (NSCF bands)
#   - bands_post.in (bands.x post-processing)
# =============================================================================

# Pseudopotential files (from template/)
# Na: z_valence=9, Cl: z_valence=7 -> total electrons = 16, occupied bands = 8
# ecutwfc: max(44 Ha Na, 29 Ha Cl) = 44 Ha = 88 Ry (INVARIANT 5: 1 Ha = 2 Ry)
# ecutrho = 4 * 88 = 352 Ry
# nbnd = 20 (well above 8 occupied bands, captures conduction bands)

PSEUDO_FILES = {
    'Na': 'Na.pbesol_s-rrkjus_psl.1.0.0.UPF',
    'Cl': 'Cl.pbesol_s-rrkjus_psl.1.0.0.UPF',
}

ECUTWFC_RY = 88.0   # 44 Ha * 2
ECUTRHO_RY = 352.0  # 4 * 88
NBND = 20           # 8 occupied bands, padded to 20
CONV_THR = 1e-8     # Ry

# Read base structure to get the primitive cell
base = ase.io.read('structures/structure_000.cif')

# Compute k-path for the base structure (unperturbed, Fm-3m #225)
kp = standard_kpath(base)
prim = kp_atoms(kp)

# SCF k-point mesh: Monkhorst-Pack for insulator (metal=False)
kpts_scf = scf_kmesh(prim, metal=False)
print(f"SCF k-mesh: {kpts_scf}")

# Bands k-points from the k-path
kpts_bands = kp_kpts(kp)
print(f"Bands k-points: {len(kpts_bands)} points")

# Pseudopotential filenames for write_espresso_in
pseudos = {sym: fname for sym, fname in PSEUDO_FILES.items()}

for n in range(51):
    nnn = f"{n:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)
    
    # Copy pseudopotential files into the calculation directory
    # (pseudo_dir will be '.' so pw.x finds them in the current directory)
    for sym, fname in PSEUDO_FILES.items():
        src = os.path.join('template', fname)
        dst = os.path.join(calc_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    
    # Read the structure for this NNN
    atoms = ase.io.read(f'structures/structure_{nnn}.cif')
    
    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"Structure {nnn}: expected nat=2, got {nat}"
    assert ntyp == 2, f"Structure {nnn}: expected ntyp=2, got {ntyp}"
    
    # For perturbed structures, recompute the k-path (lower symmetry)
    kp_n = standard_kpath(atoms)
    prim_n = kp_atoms(kp_n)
    kpts_bands_n = kp_kpts(kp_n)
    
    # Save kpath.json
    kp_save(kp_n, os.path.join(calc_dir, 'kpath.json'))
    
    # SCF input (pw.in)
    # Use the primitive cell from the k-path (INVARIANT 3)
    input_data_scf = {
        'calculation': 'scf',
        'prefix': 'nacl',
        'outdir': './tmp',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'nspin': 1,
    }
    
    with open(os.path.join(calc_dir, 'pw.in'), 'w') as f:
        write_espresso_in(f, prim_n, input_data=input_data_scf,
                          pseudopotentials=pseudos, kpts=kpts_scf)
    
    # Bands input (bands.in) - NSCF calculation
    input_data_bands = {
        'calculation': 'bands',
        'prefix': 'nacl',
        'outdir': './tmp',
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'nbnd': NBND,
        'nspin': 1,
    }
    
    with open(os.path.join(calc_dir, 'bands.in'), 'w') as f:
        write_espresso_in(f, prim_n, input_data=input_data_bands,
                          pseudopotentials=pseudos, kpts=kpts_bands_n)
    
    # Bands post-processing input (bands_post.in) for bands.x
    # &BANDS namelist: prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
    prefix = 'nacl'
    outdir = './tmp'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)
    
    print(f"Wrote {calc_dir}/ (nat={nat}, ntyp={ntyp}, kpath={kp_n['spacegroup']})")

print("\nDone: 51 calculation directories created.")
