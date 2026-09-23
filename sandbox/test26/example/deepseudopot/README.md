# DeePseudopot reference materials (optional example load)

Only copied into a run's working directory when `run_agent(..., load_deepseudopot=True)`
is set (see `chem_llm/config.py`'s `DPP_EXAMPLE_SUBDIR` and
`chem_llm/agent_core.py`'s `prepare_work_dir`) -- most runs never touch
DeePseudopot, so this is opt-in on top of the normal `example/` load.

These files let a run's own QE DFT output (produced by adapting
`generate_structures.py` + `setup_jobs.py`, exactly like the rest of
`example/`) be carried one leg further: DFT band structure -> a
DeePseudopot input bundle -> a trained neural-network pseudopotential (via
the `train_deepseudopot` tool, `chem_llm/tools/deepseudopot.py`).

## What's here

- `qe_bands_to_ref.py` -- converts a QE `bands.x`-produced `NNN.bands.dat`
  file into DeePseudopot's `kpoints_0.par` + `expBandStruct_0.par` format.
  `generate_qe_kpath()` is hardcoded to the exact 41-point
  R -> Gamma -> X -> M -> Gamma path `setup_jobs.py`'s `bands.in` writes
  (`K_POINTS crystal_b`, four 10-point segments + the closing point) -- it
  applies unchanged as long as that k-path isn't altered. `main()` drops
  the first 72 bands (`bands[:, 72:]`) to keep the 128 bands DeePseudopot
  fits to (see `nn_template/input_0.par`'s `nBands = 128`); this split
  comes from CsPbBr3's electron count under `noncolin`/`lspinorb` (176
  electrons -> VBM = band 176, kept-band column 104 after the drop), not
  from any particular structure, so it's correct for ANY CsPbBr3 job built
  from `example/`'s pipeline (perturbed or not).
- `setup_nn_inputs.py` -- reference driver that, for a `calculations/NNN`
  loop, copies `nn_template/` -> a new `nn_inputs_NNN_<suffix>` dir, writes
  `system_0.par` from that job's `pw.in`, runs `qe_bands_to_ref.py` to
  produce `kpoints_0.par`/`expBandStruct_0.par`, then `nn_template/shift.py`
  (VBM alignment) and, for the fast `_g` suffix, `nn_template/select_rows.py`
  (keeps only k-path rows 11 and 21). Written for a directory LOOP over many
  jobs via subprocess calls with `cwd=` tricks -- for a single job, adapt
  (don't blindly subprocess-call) this logic into one script that writes
  directly to explicit paths. See `deepseudopot_agent.ipynb`'s TASK cell for
  a fully worked-out adaptation recipe (constants, shift math, row
  selection) if you need the exact reference.
- `nn_template/` -- a complete DeePseudopot input bundle template for
  CsPbBr3 MINUS the three structure/DFT-derived files (`system_0.par`,
  `kpoints_0.par`, `expBandStruct_0.par`) that `qe_bands_to_ref.py` +
  `setup_nn_inputs.py`'s logic produce. Contains real Zunger-form initial
  pseudopotential parameters (`init_<Atom><N>Params.par`) for Cs/Pb/Br --
  DeePseudopot's NN training cold-starts from these, and no equivalent
  parameterization exists for other elements, so this bundle only works for
  CsPbBr3 systems. Settings already tuned (verified against a real
  end-to-end `train_deepseudopot` run) for a fast single-job demo rather
  than production accuracy -- do not "fix" these back to more typical
  values:
  - `NN_config.par`: `num_cores = 0` (>=1 crashes/OOMs -- see
    `chem_llm/tools/deepseudopot.py`'s tool description), `max_num_epochs = 5`,
    `plotEvery = 5`.
  - `input_0.par`: `maxKE = 2.0` (higher OOMs during Hamiltonian caching).
  - No `init_PPmodel.pth` / `init_qSpace_pot.par` / `old_qSpace_pot.dat` --
    forces a cold start from the Zunger-form params instead of trying to
    load an incompatible checkpoint.

## What `example/`'s own pipeline already provides

`example/setup_jobs.py` writes exactly the QE input `qe_bands_to_ref.py`
expects: `noncolin = .true.`, `lspinorb = .true.`, `nbnd = 200` for the
bands calculation, and the same R -> Gamma -> X -> M -> Gamma
`K_POINTS crystal_b` path. The SCF `K_POINTS automatic 8 8 8 0 0 0` mesh is
the one thing worth overriding for a single demo job -- a real DFT run
against that mesh with SOC and no symmetry (`nosym = .true.`) is too slow
for a practical agent session; a single Gamma point (`1 1 1 0 0 0`) was the
validated substitution used to build this template (see
`deepseudopot_example/README.md` for the full precedent), and doesn't
change nat/ntyp/electron count/pseudopotentials/k-path, so `qe_bands_to_ref.py`
and this template still apply unchanged.
