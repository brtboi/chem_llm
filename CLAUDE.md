# Handoff: TiO2 band-structure pipeline fix (in progress)

## Context
`main.ipynb` + `/chem_llm` is an LLM agent pipeline (Qwen/Qwen3.8-27B, `chem_llm/agent_core.py`)
that autonomously writes Python scripts to generate perturbed crystal structures, build QE
(Quantum ESPRESSO) input files, run DFT, and plot band structures. It's driven by a system
prompt with numbered rules in `SYSTEM_PROMPT_TEMPLATE` (`chem_llm/agent_core.py`) that mostly
say "never hardcode a value from memory/textbook assumption — derive it from the actual
structure/output in code."

## Two bugs found and fixed this session (both same root cause pattern)

1. **N_OCC / band-count hardcoding** (`sandbox/test21`, original report): the agent's
   `plot_bands.py` hardcoded `N_OCC=32` from a naive per-element valence-electron guess
   (Ti=4, O=6). The actual pseudopotentials used are semicore, giving 96 total electrons /
   48 occupied bands, not 64/32. This misaligned the VBM=0 shift, producing a
   spuriously-metallic-looking band plot (bands crossing zero everywhere).
   **Fix**: added rule 11 to `SYSTEM_PROMPT_TEMPLATE` (`chem_llm/agent_core.py`) instructing
   the agent to always read electron/band counts from the actual `pw.out`, never assume from
   memorized element valence, and to treat `get_pseudopotential`'s `n_valence` hint as
   untrustworthy. Also bumped `MAX_AGENT_STEPS` 36 → 44 in `chem_llm/config.py` (the debug
   loop needed more room). **Verified fixed** — reran in `sandbox/test22`, confirmed
   `bands_000.pdf` now VBM-aligns correctly with no spurious zero-crossings.

2. **K-path / cell-mismatch hardcoding** (found while comparing test21 vs test22, NOT yet
   verified fixed): the base rutile TiO2 structure fetched by `generate_cif` from Materials
   Project (mp-2657) comes back in a non-conventional 12-atom, oblique/rhombohedral-looking
   cell (a=b=c=5.4695 Å, angles ~107°/107°/114.5°, P1) instead of the standard 6-atom
   tetragonal cell (a=b≠c, 90°/90°/90°). The agent's `setup_jobs.py` hardcoded the textbook
   tetragonal Γ/X/M/Z/R fractional k-point coordinates from memory, which do NOT correspond
   to real physical high-symmetry points of this differently-shaped cell's actual reciprocal
   lattice — confirmed numerically (see conversation history). This produced nonphysical band
   dispersion (bands rocketing to ±tens of eV away from Γ).
   **Fix applied**: added rule 12 to `SYSTEM_PROMPT_TEMPLATE` (`chem_llm/agent_core.py`)
   instructing the agent to derive the k-path programmatically via
   `pymatgen.symmetry.bandstructure.HighSymmKpath(structure)`, going through Cartesian
   coordinates (`kpath.prim_rec.get_cartesian_coords(...)`) and re-projecting onto the actual
   cell's reciprocal lattice (`structure.lattice.reciprocal_lattice.get_fractional_coords(...)`)
   rather than typing fractional coordinates from a textbook convention.
   **NOT yet verified** — this is the pending task (see below).

## Pending: rerun sandbox/test22 to verify the rule-12 fix

Run from repo root (`/pscratch/sd/b/brenthu/chem_llm`):

```
yes | ./.venv/bin/python -u sandbox/run_test22_pipeline.py
```

This script replicates `main.ipynb`'s cells (load model, build TASK prompt for rutile TiO2,
`run_agent(..., clear_dir=True, load_path=config.EXAMPLE_DIR)` into `config.WORK_DIR` =
`sandbox/test22`). It pipes `yes` to stdin because `clear_dir=True` prompts for an interactive
confirmation (`chem_llm/agent_core.py::_confirm_clear_dir`). Expect ~30-70 min for a full run.

**After it completes**: read `sandbox/test22/calculations/000/bands_000.pdf` and confirm the
band structure shows sensible dispersion at ALL high-symmetry points along the path (not just
near Γ, and not blowing up to tens of eV) — that's the signal the k-path fix worked. Compare
against `sandbox/test21/tio2_pipeline/calculations/000/bands_000.pdf` (the original broken
one) and the previous `sandbox/test22` run's `bands_000.pdf` if still around, to show
before/after.

## Node/infra note (why this stalled repeatedly this session)

Earlier reruns in this session got stuck for 30+ min at trivial import steps
(`import numpy`) with `/proc/<pid>/wchan` = `cl_sync_io_wait`, while node load average
spiked to 60-160 — this is shared-node filesystem (Lustre) contention on NERSC Perlmutter,
not a bug in the code. If a rerun stalls the same way (check `cat /proc/<pid>/wchan`,
`uptime`), it's worth confirming the current session is actually on an exclusive/dedicated
node before assuming the fix is broken — a Claude Code terminal doesn't follow you when you
switch nodes elsewhere; it stays on whatever node it was started on.

## Other files touched this session
- `chem_llm/agent_core.py` — rules 11 and 12 added to `SYSTEM_PROMPT_TEMPLATE`.
- `chem_llm/config.py` — `MAX_AGENT_STEPS` 36 → 44.
- `sandbox/run_test22_pipeline.py` — persistent copy of the headless run script (mirrors
  `main.ipynb` cells 0,1,2,3,4,6,7).
