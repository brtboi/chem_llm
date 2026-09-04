# DeePseudopot single-job example

This directory is real, validated data for the "DFT output -> DeePseudopot
neural-network pseudopotential" leg of the pipeline, prepared the same way
`example_quantum_espresso_workflow/` prepares its 50-structure CsPbBr3
ensemble -- just for a single, deliberately UNperturbed structure instead of
an ensemble, so the tool-use exercise is about adapting the reference
scripts to a single job, not about the DFT itself.

## What's here

- `build_structure.py` -- builds `structure_cubic_CsPbBr3.cif`: the par=0.0
  (pure cubic) endpoint of `generate_structures.py`'s cubic<->orthorhombic
  interpolation, with `randomize_structure()` (the thermal-displacement
  perturbation) deliberately skipped. One clean, unperturbed CsPbBr3
  structure -- 4 Cs + 4 Pb + 12 Br, 20 atoms.
- `template/` -- the same three real (USPP, fully-relativistic) Quantum
  ESPRESSO pseudopotentials `example_quantum_espresso_workflow` uses:
  `Cs.rel-pbe-spn-rrkjus_psl.1.0.0.UPF`, `Pb.rel-pbe-dn-rrkjus_psl.1.0.0.UPF`,
  `Br.USPP.FR.PBE.3.4.UPF`.
- `calculations/000/` -- one completed SCF + bands + bands.x DFT run on that
  structure (`pw.in`/`bands.in`/`bands_post.in` written the same way
  `setup_jobs.py` writes them: ibrav=0, ecutwfc=50/ecutrho=250, noncolin
  spin-orbit coupling on, K_POINTS crystal_b band path
  R->Gamma->X->M->Gamma. The ONE intentional change from the validated
  recipe: the SCF k-grid is a single Gamma point (`1 1 1`) instead of
  `8 8 8`, so this single demo DFT run finishes in a practical amount of
  time -- everything else
  (pseudopotentials, cutoffs, SOC, the bands k-path) matches exactly, so
  the electron count and band indexing the rest of the pipeline depends on
  are unchanged from the validated multi-structure ensemble). Real pw.out /
  bands_pw.out / bands_post.out / `000.bands.dat` are here after the run.
- `qe_bands_to_ref.py`, `setup_nn_inputs.py` -- unmodified copies of the
  reference scripts from `example_quantum_espresso_workflow/`. Read them to
  understand the QE-output -> DeePseudopot-input conversion; you will need
  to ADAPT their logic (not just run them as-is) for a single job living at
  a fixed path rather than a `calculations/NNN` loop -- see the task
  instructions for exactly what that means.
- `nn_template/` -- copy of `example_quantum_espresso_workflow/`'s
  DeePseudopot input template for this exact system (real Zunger-form
  initial pseudopotential guesses `init_<Atom><N>Params.par` for every atom,
  `NN_config.par`, `input_0.par`, `bandWeights_0.par`, plus the `shift.py`
  and `select_rows.py` post-processing scripts) -- everything a
  DeePseudopot input bundle needs EXCEPT the three structure/DFT-derived
  files (`system_0.par`, `kpoints_0.par`, `expBandStruct_0.par`), which is
  exactly what `qe_bands_to_ref.py` + `setup_nn_inputs.py` produce.
  Three settings were changed from the original template (verified against
  a real end-to-end `train_deepseudopot` run of this exact bundle) -- do
  not revert them:
  - `NN_config.par`'s epoch counts (`init_Zunger_num_epochs`,
    `max_num_epochs`) are turned down from production values for demo
    speed.
  - `NN_config.par`'s `num_cores` is `0`, not `2`. `num_cores >= 1` routes
    training through a multiprocessing pool that (a) unconditionally calls
    `.zero_grad()` on `LSDoptimizers[key]`, which is `None` when
    `local_env_corr = 0` as it is here, crashing every epoch, and (b) uses
    far more memory (observed: killed by the OOM killer around ~200GB RSS)
    than the equivalent sequential path taken when `num_cores == 0`. Since
    `nSystem = 1` there is nothing for multiprocessing to parallelize
    across anyway.
  - `input_0.par`'s `maxKE` is `2.0`, not `6.0`. `maxKE` sets the
    plane-wave basis cutoff for the pseudopotential representation;
    dense per-k-point spin-orbit/nonlocal matrices scale roughly as
    `maxKE**3` in memory, and `6.0` alone (independent of the num_cores
    issue above) was enough to run this process out of memory during
    Hamiltonian caching, before training even starts. `2.0` caches in
    ~10 seconds instead of crashing.
  - `init_PPmodel.pth`, `init_qSpace_pot.par`, and `old_qSpace_pot.dat`
    (present in the original template) were removed: they're a pretrained
    checkpoint from a differently-configured run, and DeePseudopot's
    `init_ZungerPP` unconditionally tries to `load_state_dict` them when
    present, which fails on an architecture mismatch. Without them,
    training correctly cold-starts from the Zunger-form
    `init_<Atom><N>Params.par` files instead, which is the standard path
    documented in the package's own README.

## Why the physics recipe was kept identical

CsPbBr3's electron count and band indexing (176 electrons, VBM = band 176,
`nBands = 128` keeping bands 73-200, `bandWeights_0.par` etc.) come from the
composition and pseudopotentials, not from the structure being cubic vs.
distorted -- so `nn_template/` and `qe_bands_to_ref.py`'s `bands[:, 72:]`
band slice are valid unchanged for this single cubic job.
