"""Step 7 of tio2_prompt.md: build a DeePseudopot input bundle from the
Step 5 structure_000 QE SCF+bands output.

Everything band-count/VBM/CBM-related is derived from THIS run's own
pw.out/bands.dat.gnu (never a memorized/reused constant from another
compound's bundle -- e.g. the CsPbBr3 example bundle's 176/128/72 column
slicing literals do not apply here and are not reused).

Bundle format follows chem_llm/DeePseudopot/utils/read.py (ground truth
parser) and chem_llm/DeePseudopot/docs/*.md, cross-checked against the
validated CsPbBr3 example bundle at example/deepseudopot/nn_template/.

Design choices specific to this TiO2 bundle (see final report for full
justification):
  - system_0.par uses `scale = 1.0` with the full lattice already
    converted to Bohr, rather than the CsPbBr3 example's "normalize by
    a_vec[0]" trick -- that normalization is only valid when the first
    lattice vector lies along x, which needn't hold in general (and this
    project's own history includes a bug from assuming a specific cell
    shape without checking it). scale=1.0 works for any cell orientation.
  - Only 6 of the 141 bands-path k-points are kept (the unique
    high-symmetry points Gamma, X, M, Z, R, A -- dropping the second,
    duplicate Gamma and Z), to keep DeePseudopot's per-k-point Hamiltonian
    caching (not parallelized across k-points) fast for this cold-start
    bundle, per train_deepseudopot's own tool docstring recommendation.
    Their eigenvalues are exact rows already computed by the real QE bands
    run (same k-points, to machine precision) -- no new DFT is run.
  - VBM/CBM (and the VBM=0 shift applied to expBandStruct_0.par) are
    computed over the FULL 141-k-point bands run, not just the 6 kept
    rows, so a true extremum lying at a k-point we then drop is not missed.
  - Initial Zunger-form pseudopotentials (init_TiParams.par/init_OParams.par)
    are cold-started as all-zero 9-parameter vectors (the "near-zero smooth
    guess" option tio2_prompt.md explicitly allows) since no fitted Ti/O
    parameters exist anywhere in this repo (only Cs/Pb/Br do, for a
    different compound) -- NN_config.par's init_Zunger_num_epochs is set to
    0 to skip Zunger pretraining entirely and rely on the NN's own
    He/Xavier initialization, since an all-zero target curve would make
    that pretraining stage a no-op anyway.
"""
import re
from pathlib import Path

import numpy as np

CALC_DIR = Path("calculations/000")
OUT_DIR = Path("nn_inputs")
OUT_DIR.mkdir(exist_ok=True)

ANG_TO_BOHR = 1.889726125
MAXKE_HARTREE = 2.0  # DeePseudopot's own internal basis cutoff (Ha), not QE's ecutwfc; kept small for a fast cold-start fit

# --- parse pw.in: cell + fractional atomic positions ---
pw_in = (CALC_DIR / "pw.in").read_text()

cell_block = re.search(r"CELL_PARAMETERS\s+angstrom\s*\n((?:.*\n){3})", pw_in)
cell_ang = np.array([[float(x) for x in line.split()] for line in cell_block.group(1).strip().splitlines()])
cell_bohr = cell_ang * ANG_TO_BOHR

pos_block = re.search(r"ATOMIC_POSITIONS\s+crystal\s*\n((?:.*\n)*?)\n", pw_in)
atom_lines = [l.split() for l in pos_block.group(1).strip().splitlines() if l.strip()]
species = [l[0] for l in atom_lines]
frac_coords = np.array([[float(x) for x in l[1:4]] for l in atom_lines])
print(f"system_0: {len(species)} atoms ({species}), cell (Bohr) diag ~ {np.diag(cell_bohr)}")

# --- N_OCC from the actual pw.out ---
pw_out = (CALC_DIR / "pw.out").read_text()
m = re.search(r"number of electrons\s*=\s*([\d.]+)", pw_out)
n_electrons = float(m.group(1))
assert n_electrons == int(n_electrons) and int(n_electrons) % 2 == 0
N_OCC = int(n_electrons) // 2
print(f"N_OCC (from pw.out: {n_electrons} electrons) = {N_OCC}")

# --- parse bands.dat.gnu: nbnd blocks of (kdist, energy), one block per band ---
blocks, current = [], []
for line in open(CALC_DIR / "000.bands.dat.gnu"):
    if line.strip() == "":
        if current:
            blocks.append(current)
            current = []
    else:
        current.append([float(x) for x in line.split()])
if current:
    blocks.append(current)
bands = np.array([[row[1] for row in block] for block in blocks])  # (nbnd, nk)
nbnd, nk = bands.shape
print(f"Parsed {nbnd} bands x {nk} k-points from bands.dat.gnu")

VBM = bands[:N_OCC].max()
CBM = bands[N_OCC:].min()
print(f"VBM = {VBM:.4f} eV, CBM = {CBM:.4f} eV, gap = {CBM - VBM:.4f} eV (computed over all {nk} k-points)")

# --- parse bands.in's own K_POINTS crystal_b block for fractional k-coordinates ---
bands_in = (CALC_DIR / "bands.in").read_text()
kblock = re.search(r"K_POINTS crystal_b\s*\n\s*(\d+)\s*\n((?:.*\n?)+)", bands_in)
n_labels = int(kblock.group(1))
kpt_lines = kblock.group(2).strip().splitlines()[:n_labels]
label_fracs = [tuple(float(x) for x in line.split()[:3]) for line in kpt_lines]
divisions = [int(line.split()[3]) for line in kpt_lines]
label_names = [r"Gamma", "X", "M", r"Gamma", "Z", "R", "A", "Z"]

label_indices = [0]
for d in divisions[:-1]:
    label_indices.append(label_indices[-1] + d)
assert label_indices[-1] == nk - 1

# keep only unique high-symmetry points (drop repeated Gamma/Z)
seen = set()
keep = []
for name, idx, frac in zip(label_names, label_indices, label_fracs):
    if name in seen:
        continue
    seen.add(name)
    keep.append((name, idx, frac))
print("Kept high-symmetry k-points:", [k[0] for k in keep])

# --- kpoints_0.par ---
kpts_frac = np.array([k[2] for k in keep])
with open(OUT_DIR / "kpoints_0.par", "w") as f:
    for kx, ky, kz in kpts_frac:
        f.write(f"{kx:.10f} {ky:.10f} {kz:.10f} 1.0\n")

# --- expBandStruct_0.par (cosmetic k-distance column + VBM-shifted energies, all nbnd bands kept) ---
kdist = [0.0]
for i in range(1, len(kpts_frac)):
    kdist.append(kdist[-1] + np.linalg.norm(kpts_frac[i] - kpts_frac[i - 1]))

nBands = nbnd
with open(OUT_DIR / "expBandStruct_0.par", "w") as f:
    for row_i, (name, idx, frac) in enumerate(keep):
        energies = bands[:, idx] - VBM
        f.write(f"{kdist[row_i]:.6f} " + " ".join(f"{e:.6f}" for e in energies) + "\n")

# --- bandWeights_0.par: uniform weights ---
with open(OUT_DIR / "bandWeights_0.par", "w") as f:
    for _ in range(nBands):
        f.write("1.0\n")

# --- system_0.par ---
with open(OUT_DIR / "system_0.par", "w") as f:
    f.write("scale = 1.0\n\n")
    f.write("cell\n")
    for row in cell_bohr:
        f.write(f"{row[0]:.10f} {row[1]:.10f} {row[2]:.10f}\n")
    f.write("\natoms\n")
    for sp, fc in zip(species, frac_coords):
        f.write(f"{sp} {fc[0]:.10f} {fc[1]:.10f} {fc[2]:.10f}\n")

# --- input_0.par ---
idxVB = N_OCC - 1
idxCB = N_OCC
with open(OUT_DIR / "input_0.par", "w") as f:
    f.write(f"nBands = {nBands}\n")
    f.write(f"maxKE = {MAXKE_HARTREE}\n")
    f.write("systemName = TiO2\n")
    f.write(f"idxVB = {idxVB}\n")
    f.write(f"idxCB = {idxCB}\n")
    f.write("BS_plot_center = 0.0\n")
    f.write("BS_plot_CBVB_range = 8.0\n")
    f.write("BS_plot_CBVB_range_zoom = 4.0\n")

# --- init_<Atom>Params.par: cold-start, all-zero 9-parameter Zunger form ---
for atom in sorted(set(species)):
    with open(OUT_DIR / f"init_{atom}Params.par", "w") as f:
        for _ in range(9):
            f.write("0.0\n")

# --- NN_config.par ---
nn_config = """SHOWPLOTS = 0
nSystem = 1
memory_flag = 0
runtime_flag = 0
num_cores = 0
num_threads = 1
cacheSO = 0

local_env_corr = 0
LSDmodel = Net_celu_HeInit_decayGaussian_LSD
LSDmodel_decay_rate = 1.5
LSDmodel_decay_center = 0.0
LSDmodel_gaussian_std = 2.0
LSD_hiddenLayers = 20
init_LSD_num_epochs = 0
init_LSD_optimizer_lr = 0.15
init_LSD_scheduler_gamma = 0.9
init_LSD_plot_every = 100
init_LSD_scheduler_step = 500
init_LSD_force_retrain = 1

PPmodel = Net_celu_HeInit_decayGaussian
PPmodel_decay_rate = 1.5
PPmodel_decay_center = 5.0
PPmodel_gaussian_std = 2.0
hiddenLayers = 20
separateKptGrad = 1
checkpoint = 0
SObool = 0

init_Zunger_optimizer_lr = 0.12
init_Zunger_scheduler_gamma = 0.9
init_Zunger_num_epochs = 0
init_Zunger_plotEvery = 100
init_Zunger_scheduler_step = 1000
force_retrain = 1

pre_adjust_moves = 0
pre_adjust_stepSize = 0.00
pre_adjust_LSD_step_size = 0.0005

mc_bool = 0
mc_beta = 100
mc_percentage = 0.0001
mc_iter = 350

optimizer_lr = 0.001
LSD_optimizer_lr = 0.0005
scheduler_gamma = 0.70
LSD_scheduler_gamma = 0.80
max_num_epochs = 10
plotEvery = 5
schedulerStep = 500
patience = 100
"""
(OUT_DIR / "NN_config.par").write_text(nn_config)

print(f"\nWrote DeePseudopot input bundle to {OUT_DIR}/")
print("Files:", sorted(p.name for p in OUT_DIR.iterdir()))
