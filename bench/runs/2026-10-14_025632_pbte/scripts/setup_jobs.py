import os
import shutil
import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh

# =============================================================================
# setup_jobs.py
# For each structure NNN (000-050), create calculations/NNN/ containing:
#   - Pb.upf, Te.upf (copied from template/)
#   - kpath.json (from chem_llm.kpath)
#   - pw.in (SCF input, written with write_espresso_in)
#   - bands.in (NSCF bands input, written with write_espresso_in)
#   - bands_post.in (bands.x input, written manually)
#
# Key decisions (see notes):
#   - ecutwfc = 102 Ry (51 Ha * 2, from Te pseudopotential, INVARIANT 5)
#   - ecutrho = 408 Ry (4 * ecutwfc)
#   - noncolin=.true., lspinorb=.true. (spin-orbit, Pb Z=82 > 54)
#   - occupations='fixed' (PbTe is a semiconductor, gap ~0.8 eV)
#   - conv_thr = 1e-8
#   - nbnd = 60 (n_occupied = 46 with spin-orbit, each band holds 1 electron)
#   - SCF k-mesh from scf_kmesh(prim, metal=False)
#   - bands k-points from kp_kpts(kp)
#   - bands_post.in: lsym=.false. (skip band-representation file, saves 20 min)
# =============================================================================

from ase.io.espresso import write_espresso_in

# Pseudopotential files
PSEUDO_DIR = 'template'

# QE settings
PREFIX = 'pbte'
OUTDIR = './tmp'
ECUTWFC_RY = 102.0  # 51 Ha * 2 Ry/Ha (Te pseudopotential, INVARIANT 5)
ECUTRHO_RY = 408.0  # 4 * ecutwfc
CONV_THR = 1e-8
NBND = 60           # n_occupied = 46 (spin-orbit: 1 electron/band), headroom to 60

# Spin-orbit settings (Pb Z=82 > 54, fully relativistic pseudopotentials)
NONCOLIN = True
LSPINORB = True

# Pseudopotential filenames (relative to calculations/NNN/)
PSEUDOS = {
    'Pb': 'Pb.upf',
    'Te': 'Te.upf',
}

# Number of structures
N_STRUCTURES = 51

for nnn in range(N_STRUCTURES):
    nnn_str = f'{nnn:03d}'
    calc_dir = f'calculations/{nnn_str}'
    os.makedirs(calc_dir, exist_ok=True)
    os.makedirs(f'{calc_dir}/tmp', exist_ok=True)

    # Copy pseudopotentials into calc_dir (pseudo_dir='.' resolves from here)
    for elem, fname in PSEUDOS.items():
        src = os.path.join(PSEUDO_DIR, fname)
        dst = os.path.join(calc_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    cif_path = f'structures/structure_{nnn_str}.cif'
    atoms = ase.io.read(cif_path)

    # Get the standard k-path and the primitive cell it refers to
    # INVARIANT 3: the cell for pw.in AND bands.in is kp_atoms(kp), not the CIF cell
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(prim)
    ntyp = len(set(prim.get_chemical_symbols()))
    assert nat == 2, f'structure_{nnn_str}: expected nat=2, got {nat}'
    assert ntyp == 2, f'structure_{nnn_str}: expected ntyp=2, got {ntyp}'

    # Save kpath.json (holds labels, tick positions, line segments)
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # SCF k-point mesh (Monkhorst-Pack, consistent with the primitive cell)
    # metal=False: PbTe is a semiconductor (gap ~0.8 eV)
    kpts_scf = scf_kmesh(prim, metal=False)

    # Bands k-points from the k-path
    kpts_bands = kp_kpts(kp)

    # Build the flat input_data dict for write_espresso_in
    # INVARIANT 6: pass a FLAT dict; ASE routes each key to the correct namelist
    input_data_scf = {
        'calculation': 'scf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'noncolin': NONCOLIN,
        'lspinorb': LSPINORB,
        'nspin': 1,
    }

    input_data_bands = {
        'calculation': 'nscf',
        'prefix': PREFIX,
        'outdir': OUTDIR,
        'pseudo_dir': '.',
        'ecutwfc': ECUTWFC_RY,
        'ecutrho': ECUTRHO_RY,
        'occupations': 'fixed',
        'conv_thr': CONV_THR,
        'noncolin': NONCOLIN,
        'lspinorb': LSPINORB,
        'nspin': 1,
        'nbnd': NBND,
    }

    # Write pw.in (SCF)
    pw_in_path = os.path.join(calc_dir, 'pw.in')
    with open(pw_in_path, 'w') as f:
        write_espresso_in(
            f, prim, input_data=input_data_scf,
            pseudopotentials=PSEUDOS, kpts=kpts_scf
        )

    # Write bands.in (NSCF)
    bands_in_path = os.path.join(calc_dir, 'bands.in')
    with open(bands_in_path, 'w') as f:
        write_espresso_in(
            f, prim, input_data=input_data_bands,
            pseudopotentials=PSEUDOS, kpts=kpts_bands
        )

    # Write bands_post.in (bands.x input)
    # &BANDS namelist: prefix, outdir, filband, lsym=.false.
    # lsym=.false. skips the band-representation file (saves ~20 min with spin-orbit)
    bands_post_in_path = os.path.join(calc_dir, 'bands_post.in')
    with open(bands_post_in_path, 'w') as f:
        f.write('&BANDS\n')
        f.write(f"  prefix = '{PREFIX}'\n")
        f.write(f"  outdir = '{OUTDIR}'\n")
        f.write(f"  filband = '{nnn_str}.bands.dat'\n")
        f.write('  lsym = .false.\n')
        f.write('/\n')
        f.write('EOF\n')

    if nnn <= 2 or nnn == N_STRUCTURES - 1:
        print(f'structure_{nnn_str}: nat={nat}, ntyp={ntyp}, '
              f'kpts_scf={kpts_scf}, n_kpts_bands={len(kpts_bands)}')

print(f'\nDone: set up {N_STRUCTURES} calculation directories (calculations/000 through calculations/050)')
