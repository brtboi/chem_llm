import os, shutil
import numpy as np
import spglib, seekpath
from ase.io import read
from ase.io.espresso import write_espresso_in, write_fortran_namelist

# ---- Global QE settings (INVARIANT 5: cutoff hint 42 Ha -> Ry) ----
ECUT_HA = 42.0                 # get_pseudopotential hint, HARTREE
ECUTWFC = 2.0 * ECUT_HA        # 84 Ry (1 Ha = 2 Ry)
assert ECUTWFC == 84.0
ECUTRHO = 4.0 * ECUTWFC        # 336 Ry, standard NC ratio
PREFIX = 'tio2'
SCF_KPTS = (6, 6, 8)           # rutile c<a: denser along c*; MP grid for primitive cell
NBND = 48                      # padded generously; re-checked vs pw.out electrons
NPTS_SEG = 20                  # interpolation points per k-path segment

PSEUDOS = {'Ti': 'Ti.upf', 'O': 'O.upf'}
EXPECTED_SG = 136              # rutile P4_2/mnm; authoritative for structure_000


def build_kpath(atoms, expect_sg=None):
    """Return (kpts_array (n,4), labels list of (index,label)) via seekpath.
    INVARIANT 3: path derived from THIS atoms object, reprojected onto
    atoms.cell.reciprocal() with the 2*pi convention."""
    cell_tuple = (atoms.cell[:], atoms.get_scaled_positions(),
                  atoms.get_atomic_numbers())
    res = seekpath.get_path(cell_tuple, symprec=0.01)
    sg_seek = res['spacegroup_number']
    ds = spglib.get_symmetry_dataset(cell_tuple, symprec=0.01)
    sg_spg = ds.number if hasattr(ds, 'number') else ds['number']
    assert sg_seek == sg_spg, f'seekpath {sg_seek} != spglib {sg_spg}'
    if expect_sg is not None:
        assert sg_seek == expect_sg, f'sg {sg_seek} != expected {expect_sg}'

    recip_prim = np.array(res['reciprocal_primitive_lattice'])   # includes 2*pi
    recip_actual = 2 * np.pi * np.array(atoms.cell.reciprocal()[:])  # ASE: add 2*pi

    def to_actual_frac(frac_prim):
        cart = np.array(frac_prim) @ recip_prim
        return cart @ np.linalg.inv(recip_actual)

    pcoords = {lbl: to_actual_frac(c) for lbl, c in res['point_coords'].items()}

    kpts = []
    labels = []   # (index_in_kpts, label)
    for seg in res['path']:
        a, b = seg
        ka = pcoords[a]
        kb = pcoords[b]
        # discontinuity handling: every segment contributes its own start point.
        # If start coincides with previous end (continuous), merge the label.
        start_idx = len(kpts)
        if labels and labels[-1][0] == start_idx - 1:
            # previous end is the point just before; check continuity
            prev_lbl = labels[-1][1]
            prev_coord = kpts[start_idx - 1][:3]
            if np.allclose(prev_coord, ka, atol=1e-6):
                # continuous: relabel previous end to prev|a is not needed if same
                if prev_lbl != a:
                    labels[-1] = (start_idx - 1, f'{prev_lbl}|{a}')
                # do not re-add start point
                for t in np.linspace(0, 1, NPTS_SEG + 1)[1:]:
                    kpts.append(list(ka + t * (kb - ka)) + [1.0])
                labels.append((len(kpts) - 1, b))
                continue
        # discontinuous (or first segment): add own start point
        kpts.append(list(ka) + [1.0])
        labels.append((len(kpts) - 1, a))
        for t in np.linspace(0, 1, NPTS_SEG + 1)[1:]:
            kpts.append(list(ka + t * (kb - ka)) + [1.0])
        labels.append((len(kpts) - 1, b))

    return np.array(kpts), labels


def scf_input(atoms):
    nat = len(atoms)
    ntyp = len(set(atoms.get_chemical_symbols()))
    assert nat == 6 and ntyp == 2
    return {
        'control': {
            'calculation': 'scf', 'prefix': PREFIX,
            'outdir': './out', 'pseudo_dir': './',
            'verbosity': 'high', 'tprnfor': False, 'tstress': False,
        },
        'system': {
            'ecutwfc': ECUTWFC, 'ecutrho': ECUTRHO,
            'occupations': 'fixed',  # TiO2 is an insulator
        },
        'electrons': {'conv_thr': 1.0e-8, 'mixing_beta': 0.4},
    }


def bands_input(atoms):
    d = scf_input(atoms)
    d['control']['calculation'] = 'bands'
    d['system']['nbnd'] = NBND
    return d


SLURM = """#!/bin/bash
#SBATCH -A m4735
#SBATCH -J tio2_{nnn}
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brent.hu@yale.edu

module load espresso
srun pw.x -in pw.in > pw.out
srun pw.x -in bands.in > bands.out
srun bands.x -in bands_post.in > bands_post.out
"""


def main():
    for i in range(51):
        nnn = f'{i:03d}'
        d = os.path.join('calculations', nnn)
        os.makedirs(d, exist_ok=True)
        atoms = read(f'structures/structure_{nnn}.cif')
        # copy pseudos (pw.x runs inside this dir, pseudo_dir='./')
        for f in PSEUDOS.values():
            shutil.copy(os.path.join('template', f), os.path.join(d, f))

        # SCF input
        with open(os.path.join(d, 'pw.in'), 'w') as fd:
            write_espresso_in(fd, atoms, input_data=scf_input(atoms),
                              pseudopotentials=PSEUDOS, kpts=SCF_KPTS)

        # Bands: explicit k-path (only assert exact SG on unperturbed 000)
        expect = EXPECTED_SG if i == 0 else None
        kpts, labels = build_kpath(atoms, expect_sg=expect)
        with open(os.path.join(d, 'bands.in'), 'w') as fd:
            write_espresso_in(fd, atoms, input_data=bands_input(atoms),
                              pseudopotentials=PSEUDOS, kpts=kpts)

        # bands.x post-processing
        bands_post = {'BANDS': {'prefix': PREFIX, 'outdir': './out',
                                'filband': f'{nnn}.bands.dat'}}
        with open(os.path.join(d, 'bands_post.in'), 'w') as fd:
            write_fortran_namelist(fd, input_data=bands_post)

        # band_labels.dat: label and its index in the bands k-point list
        with open(os.path.join(d, 'band_labels.dat'), 'w') as fd:
            for idx, lbl in labels:
                fd.write(f'{idx} {lbl}\n')

        with open(os.path.join(d, 'submit.sh'), 'w') as fd:
            fd.write(SLURM.format(nnn=nnn))

    print('setup_jobs done for 51 dirs')


if __name__ == '__main__':
    main()
