# Rutile TiO2 → DeePseudopot pipeline — final report

## Materials Project ID
**mp-2657** (rutile TiO2, space group P4_2/mnm, #136), fetched via `generate_cif`.
Materials Project's own symmetry analysis confirms P4_2/mnm; the raw CIF export is a
12-atom, oblique-looking cell (a=b=c=5.4695 A, angles ~107/107/114.5 deg) rather than the
conventional 6-atom tetragonal setting — a known Materials-Project-CIF artifact, not an
error (see Step 1 below).

## Pipeline outline (end to end)

1. **Base structure** (`step1_2_build_ensemble.py`): fetched mp-2657 via `generate_cif`,
   then standardized it with `pymatgen.symmetry.analyzer.SpacegroupAnalyzer(raw,
   symprec=0.1).get_primitive_standard_structure()`, giving the true 6-atom tetragonal
   cell (a=b=4.5998 A, c=2.9592 A, 90/90/90 — matches experimental rutile lattice
   parameters closely). `symprec=0.1` was needed because the fetched CIF's coordinates
   have small numerical deviations from ideal Wyckoff positions that break exact-symmetry
   detection at the default `symprec=1e-3`.
2. **Perturbed ensemble** (same script): 50 additional structures with per-atom Gaussian
   Cartesian displacements (Ti: sigma=0.05 A, O: sigma=0.08 A) plus an independent +/-1%
   random diagonal lattice strain. 51 structures total (`structures/structure_000..050.cif`).
3. **Pseudopotentials** (`step3_fetch_pseudos.py`): Pseudo-Dojo norm-conserving,
   scalar-relativistic, PBEsol, stringent UPF pseudopotentials for Ti and O
   (`template/Ti.upf`, `template/O.upf`).
4. **QE inputs** (`setup_jobs.py`): wrote `pw.in`/`bands.in`/`bands_post.in`/`submit.sh`
   for all 51 structures under `calculations/NNN/`. The bands k-path is derived
   per-structure via `pymatgen.symmetry.bandstructure.HighSymmKpath(structure)` on the
   already-standardized cell, converted to QE `K_POINTS crystal_b` fractional
   coordinates in that same structure's own reciprocal lattice.
5. **DFT run** (`calculations/000/`): ran SCF -> bands -> bands.x post-processing for the
   representative (unperturbed) structure directly on the allocated node
   (`module load espresso/7.5-libxc-7.0.0-gpu-cu13`, no srun/mpirun,
   `OMP_NUM_THREADS=1`). All three steps completed with exit code 0; SCF converged in 31
   iterations.
6. **Bands plot** (`plot_bands.py` -> `calculations/000/bands_000.pdf`/`.png`): VBM-aligned,
   x-axis ticks placed at the exact cumulative k-path distances (from the `.gnu` file's own
   x-column) corresponding to each high-symmetry label, not assumed evenly spaced.
7. **DeePseudopot bundle** (`build_nn_inputs.py` -> `nn_inputs/`): built from this run's
   real `pw.in`/`pw.out`/`bands.in`/`000.bands.dat.gnu`.
8. **Training** (`step8_train.py`): `train_deepseudopot(nn_inputs/, dpp_results/)` ->
   `success: true`, `final_pot_Ti.dat`/`final_pot_O.dat`/`final_pot_q_Ti.dat`/
   `final_pot_q_O.dat` all written with real numeric data, `trained_model_path =
   dpp_results/epoch_10_PPmodel.pth`.

## Every scientific/engineering assumption and its justification

- **Cell standardization instead of using MP's raw CIF cell as-is.** The raw fetched cell
  is a valid but non-conventional 12-atom representation of the same crystal. Rather than
  deriving a k-path for that arbitrary oblique cell (which would require correctly
  tracking a possible rotation between the CIF's coordinate frame and whatever frame
  pymatgen's internal symmetry routines use — a real, easy-to-get-subtly-wrong step,
  confirmed by testing: `HighSymmKpath` raises a "does not match expected standard
  primitive" warning on the raw cell, and naively reprojecting Cartesian k-points back onto
  the raw cell's reciprocal lattice through that mismatched frame gives non-rational,
  physically meaningless fractional coordinates), the structure was standardized ONCE with
  pymatgen's own symmetry analysis and used consistently for every downstream step
  (perturbation, QE cell, k-path). This is still "derived from the actual fetched
  structure" (pymatgen computes it from mp-2657's real atomic positions), it just also
  fixes the cell shape to the one its own symmetry says it should be — verified
  self-consistent: `HighSymmKpath(structure_000)` raises no warning and its internal
  reciprocal lattice is numerically identical to `structure_000.lattice.reciprocal_lattice`.
  One consequence worth flagging: this makes the working cell 6 atoms/48 electrons/24
  occupied bands, not the 12 atoms/96 electrons/48 occupied bands a prior session's run
  (using the raw, unstandardized cell) reported — both are valid representations of the
  same crystal, just at different cell sizes.
- **Displacement magnitudes** (Ti: 0.05 A, O: 0.08 A, Gaussian, per Cartesian component):
  chosen to match real room-temperature atomic displacement parameters reported for rutile
  TiO2 from diffraction refinements (U_iso ~ 0.005-0.008 A^2, i.e. RMS ~ 0.07-0.09 A per
  axis) and consistent with a Debye-model estimate using TiO2's Debye temperature
  (~660-760 K) at 300 K. O gets a larger sigma than Ti, consistent with its lower mass and
  reported ADPs.
- **Lattice strain** (+/-1% random diagonal, independent per axis): deliberately somewhat
  larger than TiO2's real room-temperature-scale thermal expansion (linear alpha ~
  7-9e-6/K), to give the ensemble some elastic/compositional strain diversity while
  staying well inside the small-strain/harmonic regime.
- **Pseudopotentials are semicore**: Ti's UPF has `z_valence=12` (not the naive
  textbook valence of 4), O's has `z_valence=6`. Read directly from the UPF file headers,
  not assumed. This is why the occupied-band count (24 for the 6-atom cell) does not match
  a naive per-element valence-electron guess, and it was cross-checked against the
  actual SCF `pw.out`'s "number of electrons = 48.00" line (not just the pre-computed
  padding value) everywhere N_OCC is used (plotting and the DeePseudopot bundle).
- **QE settings**: `ecutwfc = 84 Ry` / `ecutrho = 336 Ry`, i.e. `2x`/`8x` the Pseudo-Dojo
  "stringent" hint of 42 Ha for both elements (Pseudo-Dojo hints are in Hartree; QE wants
  Rydberg, hence the factor of 2; `ecutrho = 4*ecutwfc` is the standard dual for
  norm-conserving pseudopotentials). `occupations='fixed'` (TiO2 is a nonmagnetic
  insulator); no spin polarization or SOC (light elements, standard NC pseudopotentials).
  8x8x10 automatic SCF k-mesh (denser along a/b than the shorter c axis, roughly
  proportional to the real-space lattice parameter ratio) — conservative for such a small
  cell, not converged for production use (see below). `nbnd = 32` for the bands run (24
  occupied + 8 empty bands) — enough to see some conduction-band dispersion, not converged
  for accurate CBM/effective-mass work.
- **Resulting DFT gap (1.85 eV, PBEsol)** is smaller than the experimental rutile gap
  (~3.0 eV), as expected for a GGA functional — a known, well-documented DFT band-gap
  underestimation, not a bug.
- **DeePseudopot bundle, k-point selection**: only the 6 unique high-symmetry points
  (Gamma, X, M, Z, R, A) from the 141-point bands path are kept (dropping the second,
  repeated Gamma and Z), to keep training's per-k-point Hamiltonian caching fast, per
  `train_deepseudopot`'s own tool docstring recommendation. Their reference energies are
  exact rows from the real QE bands calculation (same k-points, to machine precision) — no
  additional DFT was run for this step. VBM/CBM (used for the VBM=0 eV shift applied to
  `expBandStruct_0.par`) were computed over the full 141-k-point run, not just these 6
  rows, so a true extremum elsewhere on the path is not missed.
- **`system_0.par`'s cell uses `scale = 1.0`** with the full lattice matrix already
  converted to Bohr, rather than the CsPbBr3 reference pipeline's "normalize by the first
  lattice vector's x-component" trick — that normalization is only valid when the first
  lattice vector lies along x and is not guaranteed in general (this project's own history
  includes exactly this kind of "assumed a specific cell shape without checking it" bug).
  `scale=1.0` is correct for any cell orientation.
- **DeePseudopot cold start (Step 7's explicit "no validated initial-pseudopotential
  parameters exist ... cold-start them yourself" instruction)**: no fitted Zunger-form
  parameters for Ti or O exist anywhere in this repo (only Cs/Pb/Br, for an unrelated
  compound). I used the simplest of the two options the task explicitly allows — an
  **all-zero 9-parameter Zunger form** for both `init_TiParams.par` and `init_OParams.par`
  ("near-zero smooth guess") — rather than `generate_random_qSpace_pot.py`'s random smooth
  curve, since the all-zero choice has no extra file-format/column-order assumptions to get
  wrong under a cold start and is explicitly endorsed by the task prompt. Correspondingly,
  `NN_config.par`'s `init_Zunger_num_epochs = 0` (Zunger pretraining is skipped — with an
  all-zero target curve that stage would just be a no-op — and the NN's own
  He/Xavier-style initialization is used directly). `bandWeights_0.par` uses uniform 1.0
  weights (the simplest valid default, per DeePseudopot's own docs) rather than the
  CsPbBr3 template's graded near-gap weighting scheme.
- **`max_num_epochs = 10`**: enough to demonstrate the training loop runs to completion
  and writes real final pseudopotentials, not remotely enough for a converged fit (see
  below — training cost barely moved: ~160170 -> ~159328, and the optimizer explicitly
  reports "Are the gradients well-conditioned? False" every epoch).

## What would need expert review before trusting the resulting EDPP

- **Training is nowhere near converged.** 10 epochs on an all-zero cold start with only 6
  k-points is enough to prove the pipeline is wired correctly end-to-end (real DFT in,
  real trained-pseudopotential files out) — it is not a physically meaningful empirical
  pseudopotential. A real fit needs far more epochs, almost certainly more k-points (6
  high-symmetry points alone under-constrain a full band structure fit), and the optimizer
  itself is flagging poorly-conditioned gradients throughout.
- **DFT is a single unconverged reference point.** One k-mesh, one plane-wave cutoff, one
  (unrelaxed, as-fetched-then-standardized) representative structure, only 8 empty
  conduction bands. A production reference band structure would need k-point/cutoff
  convergence testing and probably a structural relaxation.
- **The all-zero Zunger cold start is a deliberately uninformative starting point** — a
  domain expert may want to instead seed the local pseudopotential with either a smooth
  random curve (`generate_random_qSpace_pot.py`) or, better, back an initial guess out of
  a bulk pseudopotential-plane-wave calculation for Ti/O, to give the optimizer more
  physically reasonable gradients to descend from.
- **`SObool=0`, `local_env_corr=0`, uniform band weights** are all simplifications
  appropriate for a first cold-start demonstration; a domain expert doing a real
  Ti/O empirical pseudopotential fit would likely want spin-orbit coupling considered (Ti
  3p semicore states can have non-negligible SOC splitting) and a graded band-weighting
  scheme that emphasizes the gap region, as the CsPbBr3 reference pipeline does.
- **The 51-structure perturbed ensemble was generated but never used** beyond providing
  QE input files (per the task's Done criteria, only structure_000 was run through DFT and
  fed into DeePseudopot) — an expert wanting a genuinely EDPP-quality (environment-aware)
  fit would want to actually run DFT on some/all of the other 50 perturbed structures and
  fold them into the training bundle as additional reference systems.
