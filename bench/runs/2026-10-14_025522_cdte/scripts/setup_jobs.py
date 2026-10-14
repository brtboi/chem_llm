import os
import shutil
import numpy as np
import ase.io
from ase.io.espresso import write_espresso_in
from chem_llm.kpath import standard_kpath, kp_atoms, kp_kpts, kp_save, scf_kmesh
from chem_llm.tools.upf import read_upf_header

# ---------------------------------------------------------------------------
# Pseudopotential info (from get_pseudopotential results)
# Cd: z_valence=20.0, recommended_ecutwfc=60.0 Ha
# Te: z_valence=24.0, recommended_ecutwfc=51.0 Ha
# Total electrons = 20 + 24 = 44, occupied bands = 22 (spin-unpolarized)
# ecutwfc: max(60, 51) = 60 Ha = 120 Ry (INVARIANT 5: 1 Ha = 2 Ry)
# ecutrho = 4 * ecutwfc = 480 Ry
# ---------------------------------------------------------------------------

# Read pseudopotential headers to get z_valence (INVARIANT 2)
cd_pp = read_upf_header('template/Cd.pbesol_s-rrkjus_psl.1.0.0.UPF')
te_pp = read_upf_header('template/Te.pbesol_s-rrkjus_psl.1.0.0.UPF')

z_valence_Cd = cd_pp['z_valence']
z_valence_Te = te_pp['z_valence']

# Total electrons per formula unit (2 atoms: 1 Cd + 1 Te)
total_electrons = z_valence_Cd + z_valence_Te
occupied_bands = int(total_electrons / 2)  # spin-unpolarized

# nbnd: pad well above occupied bands to include conduction bands
nbnd = 48
assert nbnd > occupied_bands, f"nbnd={nbnd} must be > occupied_bands={occupied_bands}"

# Cutoffs (INVARIANT 5: convert Ha -> Ry)
ecutwfc_ha = 60.0  # max of Cd (60) and Te (51) recommended cutoffs in Ha
ecutwfc_ry = 2.0 * ecutwfc_ha  # 120 Ry
ecutrho_ry = 4.0 * ecutwfc_ry  # 480 Ry

assert ecutwfc_ry == 120.0, f"Expected ecutwfc=120 Ry, got {ecutwfc_ry}"
assert ecutrho_ry == 480.0, f"Expected ecutrho=480 Ry, got {ecutrho_ry}"

# Pseudopotential filenames (relative to calculations/NNN/)
pseudo_files = {
    'Cd': 'Cd.pbesol_s-rrkjus_psl.1.0.0.UPF',
    'Te': 'Te.pbesol_s-rrkjus_psl.1.0.0.UPF',
}

# Common input_data for SCF and bands (flat dict, ASE routes to correct namelists)
common_input = {
    'prefix': 'cdte',
    'outdir': 'tmp_000',
    'pseudo_dir': '.',
    'ecutwfc': ecutwfc_ry,
    'ecutrho': ecutrho_ry,
    'occupations': 'fixed',
    'conv_thr': 1e-8,
    'nspin': 1,
}

for i in range(51):
    nnn = f"{i:03d}"
    calc_dir = f"calculations/{nnn}"
    os.makedirs(calc_dir, exist_ok=True)

    # Copy pseudopotentials into calculations/NNN/
    for elem, fname in pseudo_files.items():
        src = os.path.join('template', fname)
        dst = os.path.join(calc_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Read the structure
    atoms = ase.io.read(f"structures/structure_{nnn}.cif")

    # INVARIANT 1: nat and ntyp from the actual Atoms object
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 2, f"{nnn}: expected nat=2, got {nat}"
    assert ntyp == 2, f"{nnn}: expected ntyp=2, got {ntyp}"

    # INVARIANT 3: k-path from chem_llm.kpath for THIS structure
    kp = standard_kpath(atoms)
    prim = kp_atoms(kp)  # THE cell for pw.in AND bands.in

    # Save kpath.json
    kp_save(kp, os.path.join(calc_dir, 'kpath.json'))

    # SCF k-mesh (RULE: never typed by hand)
    kpts_scf = scf_kmesh(prim, metal=False)

    # --- Write pw.in (SCF) ---
    scf_input = dict(common_input)
    scf_input['calculation'] = 'scf'
    write_espresso_in(
        os.path.join(calc_dir, 'pw.in'),
        prim,
        input_data=scf_input,
        pseudopotentials=pseudo_files,
        kpts=kpts_scf,
    )

    # --- Write bands.in (non-self-consistent bands) ---
    bands_input = dict(common_input)
    bands_input['calculation'] = 'bands'
    bands_input['nbnd'] = nbnd
    write_espresso_in(
        os.path.join(calc_dir, 'bands.in'),
        prim,
        input_data=bands_input,
        pseudopotentials=pseudo_files,
        kpts=kp_kpts(kp),
    )

    # --- Write bands_post.in (bands.x input) ---
    # &BANDS namelist: prefix, outdir, filband, lsym
    bands_post_content = f"""&BANDS
    prefix = 'cdte'
    outdir = 'tmp_000'
    filband = '{nnn}.bands.dat'
    lsym = .false.
/
"""
    with open(os.path.join(calc_dir, 'bands_post.in'), 'w') as f:
        f.write(bands_post_content)

    print(f"{nnn}: nat={nat}, ntyp={ntyp}, kpts_scf={kpts_scf}, kpath={kp['path']}")

print("\nDone. All 51 calculation directories set up.")
print(f"ecutwfc={ecutwfc_ry} Ry, ecutrho={ecutrho_ry} Ry, nbnd={nbnd}, occupied_bands={occupied_bands}")
