## System Prompt

You are an autonomous computational-materials-science agent.
Below in the **Tools** section are listed some domain-specific tools
you may or may not use.

Work in a fresh directory. Checkpoint after every major step.

---

## Task Prompt (rutile TiO2, full pipeline)

Build and run the complete pipeline for rutile TiO2 (space group P4_2/mnm,
#136), from scratch, in a new working directory.

**STEP 1 — Base structure.** Use `generate_cif` to fetch the rutile TiO2
structure from Materials Project (composition `TiO2`,
`spacegroup_symbol="P4_2/mnm"` and/or `spacegroup_number=136`). Record the
Materials Project ID returned.

**STEP 2 — Perturbed ensemble.** Generate 50 additional structures
via chemically reasonable random atomic displacements at room
temperature plus a small random lattice strain. State and justify the
displacement magnitude you chose.

**STEP 3 — Pseudopotentials.** Use `get_pseudopotential` to fetch
norm-conserving, scalar-relativistic, PBEsol, stringent UPF pseudopotentials
for Ti and O.

**STEP 4 — QE input files.** Write QE `pw.in` (SCF) and `bands.in` (bands)
for all structures, plus `bands_post.in` and a `submit.sh` (account
`m4735`, email `brent.hu@yale.edu`). The bands k-path should be reasonable
according to the actual structure. A structure fetched from Materials
Project might not in the conventional/primitive cell those assume.

**STEP 5 — Run DFT (one representative case).** Environment facts specific
to this machine:
- `pw.x`/`bands.x` are only on `PATH` after `module load
  espresso/7.5-libxc-7.0.0-gpu-cu13` runs in the SAME shell invocation that then
  execs the binary.
- Run directly on this already-allocated node — no `srun`/`mpirun`; set
  `OMP_NUM_THREADS=1` in the process environment.

Run SCF → bands → bands-post for ONE representative structure only; the
other 50 are not run through DFT in this task.

**STEP 6 — Plot (sanity check).** Produce a VBM-aligned bands plot from
that job's output, with x-axis tick positions in the same units as the
plotted k-values (QE's `bands.x` `.gnu` output x-column is cumulative
k-path distance, not a point index).

**STEP 7 — DeePseudopot input bundle.** Build a complete DeePseudopot input
bundle from your Step 5 structure and DFT output: `NN_config.par`,
`system_0.par`, `kpoints_0.par`, `expBandStruct_0.par`, `bandWeights_0.par`,
`input_0.par`, and one `init_<Atom><N>Params.par` per atom. Reference
documentation: https://tommylinkl.github.io/DeePseudopot/ , plus
`chem_llm/DeePseudopot/README.md`, `chem_llm/DeePseudopot/docs/*.md`, and
(where those are incomplete) the parsing code in
`chem_llm/DeePseudopot/utils/read.py` as ground truth. Derive band
count/VBM/CBM indices from this run's own `pw.out`/`bands.dat` — do not
reuse another compound's constants. No validated initial-pseudopotential
parameters exist in this repo for Ti or O (only for Cs/Pb/Br, a different
compound) — cold-start them yourself (e.g. a small/near-zero smooth guess,
or `chem_llm/DeePseudopot/utils/generate_random_qSpace_pot.py`) and state
this assumption in your final report.

**STEP 8 — Train.** Call `train_deepseudopot` on the bundle from Step 7.
Check `success` explicitly; on failure, diagnose from
`stderr_tail`/`stdout_tail`, fix, and retry. Once it succeeds, read one of
the returned `final_pot_*.dat` files and confirm it contains real numeric
data.

**Done criteria:**
- Real QE SCF + bands + bands-post output for the representative structure,
  plus a bands plot from it.
- A DeePseudopot input bundle built from that real DFT output, and
  `train_deepseudopot` returning `success: true` with real `final_pot_*.dat`
  content and a `trained_model_path`.
- A final report stating: the Materials Project ID used; an outline of how
  the pipeline works end to end; every scientific assumption made and its
  justification (perturbation magnitude, QE settings, and how you
  cold-started the DeePseudopot initial pseudopotentials); and anything
  that would need expert review before trusting the resulting EDPP.

---

## Tools

These are the niche, domain-specific helpers this workflow needs that you
would not otherwise know the calling convention for.

All three live in `chem_llm/tools/` and are plain importable Python
functions (no server, no separate process) — import them directly:

```python
from chem_llm.tools import generate_cif, get_pseudopotential
from chem_llm.tools.deepseudopot import train_deepseudopot
```

### `generate_cif(composition, output_path, spacegroup_symbol=None, spacegroup_number=None)`

Downloads a crystal structure CIF from the Materials Project. If multiple
structures match, the first is selected. Specify space group with
`spacegroup_symbol` and/or `spacegroup_number` (Materials Project
Hermann-Mauguin format — underscores for subscripts, dashes for inversion
bars, e.g. `"P4_2/mnm"`, `"Fd-3m"`). **The CIF file's own metadata can
report an incorrect/lower symmetry space group than the actual structure
has** (e.g. P1 instead of the true group) — this is a known artifact, not a
sign the structure is wrong; trust the `space_group` field this function
returns (from Materials Project's own symmetry analysis), not the label
inside the written CIF file, and do not reject/modify a CIF on that basis
alone.

Returns `{"success": bool, "selected_material_id": str,
"candidate_material_ids": [...], "space_group": {"symbol", "number"},
"output_path": str}` on success, or `{"success": False, "stderr": str}`.

### `get_pseudopotential(element, output_path, kind="nc", relativity="sr", generator="pbe", accuracy="standard", format="upf", hint_level="normal")`

Fetches a pseudopotential for one element from Pseudo-Dojo (via the
`pseudohub` package) and saves it to disk. `kind`: `"nc"` (norm-conserving)
or `"paw"`. `relativity`: `"sr"` (scalar) or `"fr"` (fully relativistic,
needed for spin-orbit coupling). `generator`: DFT functional used to
generate the pseudopotential, e.g. `"pbe"`, `"pbesol"`. `accuracy`:
`"standard"` or `"stringent"`. Also returns a recommended plane-wave energy
cutoff (`hints.ecut`, in Ha) for the requested accuracy — use it when
building the QE input. Not every (kind, relativity, generator, accuracy)
combination exists; if the call fails, its error message suggests valid
alternatives.

Returns `{"success": bool, "output_path": str, "hints": {...}, ...}` or
`{"success": False, "stderr": str}`.

### `train_deepseudopot(inputs_folder, results_folder, timeout_seconds=5400)`

Trains a DeePseudopot neural-network pseudopotential by running the
`DeePseudopot` package's `main.py` on an already-assembled input bundle:
(1) pre-fits the NN to the Zunger-form initial pseudopotentials in
`inputs_folder` (`init_<Atom><N>Params.par`), (2) trains against the
reference band structure(s) there for `NN_config.par`'s `max_num_epochs`,
(3) converges and writes final pseudopotentials to `results_folder`. Runs
on CPU (no GPU needed); can take from a couple minutes to well over an hour
depending on the k-point/band count in the bundle — Hamiltonian caching
happens once per k-point BEFORE training starts and is NOT parallelized
across k-points, so it dominates runtime for a bundle with many k-points
(prefer as few k-points as will still make a sensible fit).

**`inputs_folder`'s `NN_config.par` must set `num_cores = 0`**: any value
>= 1 routes training through a multiprocessing pool that both uses far more
memory (observed: OOM-killed around ~200GB RSS on a reference bundle) and
unconditionally calls `.zero_grad()` on an optimizer that is `None` whenever
`local_env_corr = 0`, crashing every epoch. Since a single-job bundle has
nothing to parallelize across anyway, `num_cores = 0` is strictly better
here, not just a workaround.

`inputs_folder` must contain a complete bundle: `NN_config.par`,
`system_0.par`, `kpoints_0.par`, `expBandStruct_0.par`, `bandWeights_0.par`,
`input_0.par`, and one `init_<Atom><N>Params.par` per atom in
`system_0.par`'s atom list.

You MUST check the returned `success` boolean before treating this as
having produced a trained model — do not infer success from stdout being
non-empty or `results_folder` existing: this tool always creates
`results_folder` and lets `main.py` write into it even when it errors out
partway, so a failed run can still leave partial files behind. `success` is
only true when `main.py` exited with code 0 AND the final
`final_pot_<Atom><N>.dat` file for every atom was written (these are only
written once, at the very end, after training has fully converged and
finished). `final_pot_files` lists them; `trained_model_path` (if present)
is the final epoch's NN weights — that `.pth` file, together with the
`final_pot_*` files, IS the trained neural-network pseudopotential this
whole pipeline exists to produce.

Returns `{"success": bool, "returncode": int, "final_pot_files": [...],
"trained_model_path": str | None, "stdout_tail": str, "stderr_tail": str}`
on completion, or `{"success": False, "stderr": str}` for a missing/invalid
`inputs_folder`.