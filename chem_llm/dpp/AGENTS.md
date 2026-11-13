# DeePseudopot in chem_llm: a guide for coding agents

What DeePseudopot is, how chem_llm drives it, what its options really do, and what we measured.
Every claim here was checked against the code of the vendored checkout or a run; where the
upstream docs disagree with the code, the code wins and the difference is listed.

## 1. What it is and where it lives

- DeePseudopot (github.com/TommyLinkl/DeePseudopot; Lin, Coley-O'Rourke & Rabani, npj Comput
  Mater 11, 381 (2025)) fits a semi-empirical pseudopotential: a small plane-wave Hamiltonian whose
  local potential per element is a neural network v(q) (or the 4-parameter Zunger form), trained
  so its eigenvalues match reference (DFT) bands.
- Vendored at `chem_llm/DeePseudopot/`: a git checkout of upstream `main` (138aa96). Not packaged;
  an installed chem_llm needs `$DEEPSEUDOPOT_DIR`. Do not edit it: it is someone else's repo.
- User's guide: https://tommylinkl.github.io/DeePseudopot/, built from the checkout's `docs/` (also
  readable locally). The repo has no GitHub wiki.
- Scripts are run as `python main.py <inputs>/ <results>/` from the checkout's own directory (they
  import `utils.*` relative to it). Trailing slashes matter: paths are joined by string concatenation.

## 2. How chem_llm uses it

```
finished QE run (pw.in, pw.out, kpath.json, *.bands.dat, UPFs)
  -> dpp.io.read_qe_run                      everything about the DFT run, read from its files
  -> dpp.io.initial_zunger_params            physical starting potential, from the UPF files
  -> dpp.io.write_bundle(..., mc_opts=...)   input bundle + mcOpts1.par
  -> train_deepseudopot(mode='zunger_mc')    stage 1: Monte Carlo over the Zunger parameters
  -> dpp.io.best_zunger_params               carry the best parameters forward
  -> dpp.io.write_bundle(...)                bundle for stage 2
  -> train_deepseudopot(mode='gradient')     stage 2: gradient descent on the neural network
  -> dpp.io.read_fit                         score any stage against the DFT bands
```

- `chem_llm/dpp/io.py`: the deterministic bookkeeping above. `chem_llm/tools/deepseudopot.py`: the
  tool (runs the script, checks the outputs, writes `inputs_folder.txt` into each results folder).
  `chem_llm/dpp/agent.py`: the DeePseudopot agent's prompt (INVARIANTS, RULES) and task (`chem-llm-dpp`).
- Example inputs: `deepseudopot_example/tio2_rutile/` (a QE run + NN_config template + measured
  numbers), `deepseudopot_example/graphene_mc_gradient/` (a collaborator's MC and gradient configs).
- Tests: `tests/test_dpp.py` (no DeePseudopot run; training is faked with the real output layout).

## 3. The input bundle

One system (`_0`) per bundle; `write_bundle` writes and cross-checks all of these.

| file | content | units |
|---|---|---|
| `system_0.par` | `scale`, `cell` (3 rows), `atoms` (symbol + fractional coords) | Bohr |
| `kpoints_0.par` | `kx ky kz weight` per line | fractional (reciprocal basis) |
| `expBandStruct_0.par` | path coordinate, then one column per fitted band | eV |
| `bandWeights_0.par` | one loss weight per fitted band | - |
| `input_0.par` | `nBands`, `maxKE`, `idxVB`, `idxCB`, plot ranges, `systemName` | maxKE in **Hartree** |
| `init_<El>Params.par` | 9 numbers per element (below) | |
| `NN_config.par` | global settings (section 6) | |
| `mcOpts1.par` | Zunger Monte Carlo options (section 5) | |
| `init_qSpace_pot.par` | optional table `q v_El1 v_El2 ...`, header names the order | q in 1/Bohr, v in Ha*Bohr^3 |
| `init_PPmodel.pth` | optional network weights to start from | |

**The 9 parameters per element** (README order, checked in `utils/ham.py`):
`[0:4]` Zunger local form `a0 (q^2 - a1) / (a2 exp(a3 q^2) - 1)` (a2 > 1, or it diverges);
`[4]` long-range Coulomb tail (polar materials; constrained so the tails sum to 0);
`[5]` spin-orbit strength; `[6:8]` non-local strengths; `[8]` strain/deformation.

**Hamiltonian** (Hartree atomic units, `HBAR = MASS = 1`, eigenvalues x 27.2114 to eV):
`H = |k+G|^2/2 + V_loc + V_NL + V_SO`, with `V_loc(G,G') = sum_atoms e^{i(G-G').tau} v(|G-G'|) / cell volume`.
`maxKE` is the plane-wave cutoff in Hartree: about `V (2 maxKE)^1.5 / (6 pi^2)` plane waves.

**Non-local and spin-orbit terms exist only in the spinor Hamiltonian** (`SObool = 1`). The
non-local projectors are l = 1 only. A fit without spin-orbit is a purely local potential.

## 4. Physics rules that are not optional

1. **Band alignment.** DeePseudopot fits its lowest `nBands` eigenvalues, in order, to the columns
   of `expBandStruct_0.par`. Semicore DFT bands (filled shells below the valence shell: Ti 3s/3p,
   Ga 3d, Pb 5d, Cs 5s/5p) must be dropped (`first_band`), and the dropped/kept bands separated by
   a clear gap. Feeding all DFT bands misaligns every band (the September TiO2 attempt).
2. **No exact zeros in the reference.** The Zunger Monte Carlo cost (`fit_mc.calcIndivMSE`) skips
   entries with `abs(E) <= 1e-15` as missing. With the VBM shifted to 0 that is the VBM.
   `write_bundle` writes such entries as 1e-6 eV (since c28fe02; earlier bundles have the bug).
3. **A physical start.** All-zero Zunger parameters are a free-electron gas; a fresh network is
   random. Start from `initial_zunger_params(run, semicore)`: each element's UPF local potential,
   semicore shells counted as core (their UPF pseudo-orbital charge added to the ion; Ti becomes
   +4), Lindhard-screened by the electrons of the fitted bands, Zunger form fitted over q <= 8.
   Never type Zunger parameters in from memory.
4. **Network shape must match a loaded `.pth`** (`PPmodel`, `hiddenLayers`), or the load fails.

## 5. The three ways to optimize

| | Monte Carlo, Zunger form | Monte Carlo, network | Gradient descent, network |
|---|---|---|---|
| script | `main_mc_Zunger.py` | `main.py`, `mc_bool = 1` | `main.py`, `mc_bool = 0`, `max_num_epochs > 0` |
| varies | 9 params per element | every network weight (+ SO/NL) | network weights |
| TiO2, 6 k-pts, ~100 PW | 0.07 s/step | 1.5 s/step | ~1 s/epoch |
| options | `mcOpts1.par`, `<El>ParamSteps.par` | `mc_iter`, `mc_percentage`, `mc_beta`, `mc_perturb_mode` | `optimizer`, `optimizer_lr`, `scheduler_gamma`, `schedulerStep`, `perturbEvery` |
| best output | `best_<El>Params.dat`, `bestBandStruct_<N>_0.dat` | `best_PPmodel.pth`, `best_pot_*` | `epoch_<N>_PPmodel.pth`, `final_pot_*` |
| cost log | `iterations.dat` | `final_mc_cost.dat` | `final_training_cost.dat` |

**Zunger Monte Carlo** (`utils/fit_mc.py::MonteCarloFit`), `mcOpts1.par` keys (format
`key = a b`, spaces, no commas; unknown keys raise):
- `betas = cold hot`: inverse temperatures. A worse step is accepted with probability
  `exp(-beta (sqrt(new) - sqrt(current)))`, i.e. beta is per eV of RMS-like error. A better-than-best
  step is always taken.
- `stepsAtTemp = cold hot`: steps before switching temperature (quasi parallel tempering).
- `tempStepSizeMod = cold hot`: multiplies every step at that temperature. **This is the "learning
  rate" of Monte Carlo.**
- `totalIter`; `optGaps` (fit band differences instead of energies); `fitDefPot`, `fitCoupling`,
  `fitEffMass`, `defPotWeight`, `writePerIter`.
- Step sizes: `<El>ParamSteps.par` (9 values) if given for every element; otherwise 1% of each
  parameter's current value (the exponent `a3`: `log(a3 + 1)/100`). **Parameters that are 0 never
  move**, so non-local/SO terms starting at 0 are never explored.
- `enforceParamConstraints`: long-range tails sum to 0; spin-orbit strength kept positive.
- `iterations.dat`: header `newMSE curMSE bstMSE %Accpt <params>`; the first row lacks the leading
  iteration number the others have (best cost in column 3 there, column 4 after). `read_fit`
  handles it (`mc_best_cost_by_quarter`).
- Known bug: after the fit, `main_mc_Zunger.py` crashes in `realSpacePot` (tensor shapes) while
  writing `final_pot_*`. Exit code 1, but `best_*Params.dat` are complete; the tool reports success
  with a note.

**Network Monte Carlo** (`utils/NN_train.py::runMC_NN`, `perturb_model`): `mc_percentage` = epsilon;
`mc_perturb_mode` 1 (default): each weight x U[1-eps, 1+eps], SO/NL x (1 +- eps/100); 2:
magnitude-aware; 3: normalized Gaussian shifts; 4: +-eps on ~60% of weights; 5: +-eps/10 on
weights, +-eps on SO/NL; 6: SO only; 7: NL only. Mode 1 is multiplicative, so a weight never
changes sign.

**Gradient descent**: `optimizer` adam (default), sgd, asgd, lbfgs, adadelta, adagrad, adamw,
sparseadam, adamax, nadam, radam, rmsprop (`init_optimizer`); `optimizer_lr`;
`scheduler_gamma` every `schedulerStep` epochs; `plotEvery` sets how often `epoch_<N>_*` (and the
`BS_sys0.dat` read_fit needs) are written; `separateKptGrad = 1` / `checkpoint = 1` save memory.

## 6. Initialization (`utils/init_NN_train.py::init_ZungerPP`, in this order)

1. `init_qSpace_pot.par` present: it replaces the Zunger curves as the pre-fit target.
2. `init_PPmodel.pth` present: weights loaded; with `force_retrain = 0` that is the start (no pre-fit).
   `init_AdamState.pth` also restores optimizer momentum.
3. `init_Zunger_num_epochs > 0`: the network is trained to reproduce the Zunger form of
   `init_<El>Params.par` (or the table) first (`init_Zunger_optimizer`, `init_Zunger_optimizer_lr`).
   chem_llm's gradient stage uses 2000 epochs at lr 0.12.
4. Otherwise the torch initializer baked into `PPmodel`: `*HeInit*` (Kaiming), `*xavier*`,
   `*RandInit*` (custom normal). `*_decayGaussian` multiplies the output by exp(-q^2/(2 sigma^2))
   (`PPmodel_gaussian_std`), so v(q) vanishes at large q. `ZeroFunction` outputs 0 with exactly 2
   columns (hard-coded), so it only fits two-element systems.

## 7. Settings that bite

- `num_cores = 0`. On this checkout, `>= 1` with `local_env_corr = 0` calls `.zero_grad()` on a
  `None` optimizer every epoch and ran out of memory (~200 GB) on CsPbBr3. `write_bundle` refuses it.
- `SObool` must match the DFT run's spin-orbit; `write_bundle` checks.
- `SHOWPLOTS = 0` on a cluster. Plots are still written to files.
- `maxKE`: cost grows ~maxKE^4.5 (matrix size^3); 6.0 ran out of memory with spin-orbit on CsPbBr3.

## 8. Upstream docs vs this checkout

- `mc_beta_schedule` (annealing) and `mc_checkpoint.pth`: documented, not in the code.
- `partial_eig`, `NLbool`, `nonlocal_grad`, `nonlocal_grad_lr` (gradient training of non-local
  prefactors): only on upstream branches `irrep-block` and `spin-polarized`, not `main`. The
  collaborator's gradient configs use them.
- `mc_percentage`: `workflows.md` calls it the fraction of parameters perturbed; the code multiplies
  every weight by U[1-eps, 1+eps] (`configuration.md` has it right).

## 9. What we measured on rutile TiO2 (2026-11-12)

Bundle: DFT bands 8-31 (Ti 3s/3p dropped), 6 high-symmetry points, maxKE 2.9 Ha (~100 PW), gap
bands weighted 4x. DFT gap 1.85 eV. RMS over all fitted bands.

| | RMS | gap |
|---|---|---|
| all-zero start, 600 gradient epochs | 1.59 eV | -2.64 eV |
| UPF start, no training | 16.7 eV | - |
| + Zunger MC, 3000 steps | 1.8-1.9 eV | about -1.8 eV |
| + gradient descent, 600 epochs (agent, 21 steps, 19 min) | 1.07 eV | -1.20 eV |
| best of 6 Monte Carlo runs (x25 steps then x1; gap -0.41 eV) + gradient 400 epochs | 1.12 eV | **-0.16 eV** |

- **Monte Carlo step size** (`tempStepSizeMod`, x1 to x25 of the 1% default, same start): big steps
  get down faster early (cost < 10 at step 426 at x25 vs 1073 at x1) but acceptance falls from 32%
  to 1% and all ended at cost 6.7-7.1. **Runs are noisy**: identical settings ended between 5.5 and
  15. Several short big-step runs in parallel, keeping the best, then small steps, beats one run.
  Which Monte Carlo minimum gradient descent starts from matters more than step size or epochs:
  from the one run whose gap was -0.41 eV, gradient descent reached -0.16 eV; from a typical
  one (about -1.8 eV), -1.2 eV. Pick the run with the best gap among those with similar RMS.
- **The gap stays inverted, and it is not the basis or the training**: the Monte Carlo potential at
  maxKE 2.9 / 4.5 / 6 Ha gives -1.84 / -1.67 / -1.65 eV; gradient descent is flat after ~400 epochs;
  counting the VBM (rule 4.2) did not change it. A local potential does not place the Ti 3d
  conduction bands, and this checkout's non-local terms (l = 1, spin-orbit path only) cannot either.

## 10. Next things to try

- A branch with gradient-trained non-local terms (`irrep-block` / `spin-polarized`), or `SObool = 1`
  with spin-orbit strength 0 to switch the non-local projectors on, with nonzero starting values so
  Monte Carlo moves them.
- Heavier weights on the band-edge bands, or `optGaps = True` in Monte Carlo.
- A compound without d conduction bands (Si, GaAs, CdSe-type) to check the pipeline end to end where
  a local empirical pseudopotential is known to work.
