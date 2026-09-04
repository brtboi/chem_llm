# DeePseudopot tool-use task — handoff

## Original ask
Expose `chem_llm/DeePseudopot` (NN pseudopotential fitting) as a tool to the
Qwen agent, and build a notebook testing *only* the last leg of the pipeline:
DFT output -> DeePseudopot input -> trained NN. Data must be real (from
actually running QE), from a single unperturbed cubic CsPbBr3 structure (not
`generate_structures.py`'s random ensemble). The agent should adapt
`qe_bands_to_ref.py` + `setup_nn_inputs.py` (not just call them), heavily
guided by the prompt. I (Claude) do DFT myself; the LLM agent does
DFT-output -> nn-input -> trained model.

## What's done and persisted on disk (verified real, not fabricated)

- **`deepseudopot_example/`** — the complete load_path for the notebook:
  - `build_structure.py` + `structure_cubic_CsPbBr3.cif`: pure cubic (par=0.0)
    CsPbBr3, 20 atoms, thermal perturbation deliberately skipped.
  - `calculations/000/`: one real completed QE run on that structure —
    SCF (single Gamma k-point, converged 11 iter) + bands (41-pt crystal_b
    path R->Gamma->X->M->Gamma, noncolin SOC). `000.bands.dat` present and
    verified byte-size-matching a production reference run.
  - `qe_bands_to_ref.py`, `setup_nn_inputs.py`: unmodified copies of the
    reference scripts, for the agent to read and adapt.
  - `nn_template/`: DeePseudopot input template for this system, with **three
    real template bugs found and fixed** (see below) — do not revert.
  - `README.md`: full documentation of every deviation from the production
    recipe and why it's still valid physics.
- **`chem_llm/tools/deepseudopot.py`**: registers `train_deepseudopot(inputs_folder, results_folder, timeout_seconds=5400)`, shells out to `DeePseudopot/main.py`, checks for real `final_pot_*.dat` output before reporting success. Wired into `chem_llm/tools/__init__.py`. Confirmed present in `TOOL_DISPATCH`.
- **`chem_llm/config.py`**: added `DPP_EXAMPLE_DIR` pointing at `deepseudopot_example/`.
- **`deepseudopot_agent.ipynb`**: new notebook (12 cells, validated nbformat), heavily-guided task prompt telling the LLM to port (not subprocess-call) the two reference scripts' logic into a new `build_nn_inputs.py`, producing a 2-k-point (`_g` convention) input bundle, then call `train_deepseudopot`, then verify real output. **This notebook has not actually been run yet** — running the Qwen agent loop end-to-end was out of scope for me to execute (too slow/expensive in-session).

### Three real bugs found in DeePseudopot's template/package (fixed in `nn_template/`, documented in `deepseudopot_example/README.md`)
1. `NN_config.par`: `num_cores` must be `0`. Any value >=1 routes through a multiprocessing path that (a) crashes every epoch calling `.zero_grad()` on a `None` optimizer when `local_env_corr=0`, and (b) OOM'd around ~200GB RSS.
2. `input_0.par`: `maxKE` must be `2.0`, not the production `6.0`. Dense per-k-point SO/NL matrices scale ~`maxKE**3`; `6.0` alone OOM'd during Hamiltonian caching.
3. Removed `init_PPmodel.pth` / `init_qSpace_pot.par` / `old_qSpace_pot.dat` from `nn_template/` — they're a checkpoint for a different NN architecture; `init_ZungerPP` unconditionally tries to load them and crashes on architecture mismatch. Without them it correctly cold-starts from the Zunger-form params (the package's own documented standard path).

## Validation status

Earlier in this session (before a context compaction) I built the exact
`nn_inputs_g` bundle from this real cubic-structure DFT data by hand,
following the same recipe the notebook asks the LLM to reimplement, and ran
it through `train_deepseudopot` end-to-end successfully: Hamiltonian caching
in ~6s, Zunger-form pre-fit (500 epochs, ~40s), VBM/CBM landed exactly on the
intended -5.25 eV / 1.7 eV gap targets, 5 training epochs with monotonically
decreasing loss, and real `final_pot_*.dat` + `epoch_5_PPmodel.pth` output.

**However: that run's output was written to an ephemeral session-scratch
directory that no longer exists** (session/process restarted since). So
right now, on disk, **there is no trained model and no plot** — only the
validated recipe and fixed template described above.

## What DeePseudopot writes automatically (this IS the visualization)

No separate plotting step is needed. `DeePseudopot/main.py` calls
`evalBS_noGrad(...)` every `plotEvery` epochs, which writes into the results
folder:
- `epoch_<N>_plotBS.pdf` / `.png` — reference bands vs NN-predicted bands, overlaid (this is "band gap vs trained network").
- `epoch_<N>_BS_sys0.dat` — the raw numbers.
- `final_pot_<Atom><N>.dat` (all atoms) + `epoch_<max_num_epochs>_PPmodel.pth` — the actual trained model, written only once training fully converges/finishes.

With `nn_template/`'s current settings (`max_num_epochs=5`, `plotEvery=5`),
a run against `deepseudopot_example/nn_inputs_000_g/` lands the final plot at
`results_000_g/epoch_5_plotBS.png` and the model at
`results_000_g/epoch_5_PPmodel.pth` + `results_000_g/final_pot_*.dat`.

## Currently blocked on

This node's `/pscratch` had a severe I/O stall (processes stuck in kernel
`cl_sync_io_wait` for 30+ min, `kill -9` couldn't even touch them) that
turned out to be caused by **the Claude Code process itself restarting**
mid-session — all background tasks got orphaned/marked "stopped" with no
completion record. `deepseudopot_example/run_pipeline_000_g.sh` (already
written, `chmod +x` already set) never got past step 1 (`qe_bands_to_ref.py`)
before the restart. `deepseudopot_example/pipeline_000_g.log` will show
where it last got to.

## Exact next steps for a fresh session

1. Sanity check the environment is responsive: `timeout 15
   /pscratch/sd/b/brenthu/chem_llm/.venv/bin/python -c "import numpy"` should
   return near-instantly. If it hangs, something is still wrong with the
   node/venv — don't stack more commands, just wait and retry.
2. Check for and kill any leftover stuck processes from prior attempts
   (`ps aux | grep qe_bands_to_ref` / `select_rows` / `shift.py` / `main.py`).
3. Run `deepseudopot_example/run_pipeline_000_g.sh` (rebuilds the
   `nn_inputs_000_g` bundle via `qe_bands_to_ref.py` -> `shift.py -5.25` ->
   `select_rows.py`, then trains via `DeePseudopot/main.py` into
   `deepseudopot_example/results_000_g/`). Use `nohup ... &` + `disown` and
   poll/monitor the log rather than a blocking foreground call — the full
   run (caching + Zunger pre-fit + training) took under 2 minutes when it
   worked earlier this session, so if it's taking much longer than that
   something is wrong.
4. Once done, report `results_000_g/epoch_5_plotBS.png` (visualization) and
   `results_000_g/epoch_5_PPmodel.pth` + `results_000_g/final_pot_*.dat`
   (trained model) to the user as the deliverable.
5. Separately (not yet attempted): actually run `deepseudopot_agent.ipynb`
   to see whether the Qwen agent can reproduce this bundle-building +
   training itself, using the tool and heavily-guided prompt already in
   place. This is the actual original ask and hasn't been executed yet.

## Note

`CLAUDE.md` and `sandbox/run_test22_pipeline.py` appeared in `git status` as
untracked files I did not create — there's a separate, concurrent Claude
session working in this same repo on this shared node. Don't assume
ownership of those files.
