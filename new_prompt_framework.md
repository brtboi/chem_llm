# QE Materials Workflow Prompt Template

This template separates three kinds of instructions that were previously mixed
together in one long paragraph:

- **INVARIANTS** — things that must never be false, enforced with runtime
  assertions in the generated code (not just prompt text).
- **DECISION RULES** — branch points, written as condition tables.
- **WORKFLOW** — the sequential steps, now short because the branching logic
  lives in the tables above it.

Reuse the INVARIANTS and DECISION RULES sections across projects with minimal
edits; rewrite the WORKFLOW section per-task.

---

## 1. INVARIANTS (must hold, checked in code, not just "kept in mind")

For each invariant, the agent must add an explicit `assert` (or equivalent
check-and-raise) in the script itself, immediately after the value is produced
— not as a prompt reminder, not as a manual review step at the end.

```
INVARIANT: structure.num_sites == config["n_atoms"]
  CHECK IN CODE: after building/loading each Structure object, before writing
  any QE input file, assert the atom count matches the config entry for that
  structure. On mismatch: raise, do not write the file, log the structure id.

INVARIANT: k-path used in bands calculation was generated FROM this specific
  structure's symmetry, not copied from a template or from the endpoint
  structure.
  CHECK IN CODE: regenerate the k-path with pymatgen's
  HighSymmKpath(structure) for the *actual* structure being run, and assert
  the resulting space group number matches SpacegroupAnalyzer(structure)
  .get_space_group_number() for that same structure. Never reuse a k-path
  object across structures with different symmetry.

INVARIANT: every element present in the structure has a pseudopotential
  assigned, and every assigned pseudopotential's relativistic treatment
  matches the DECISION RULE below.
  CHECK IN CODE: build a dict {element: pseudo_filename}, assert set(dict.keys())
  == set(unique elements in structure), and assert each filename's
  relativistic tag matches what the rule requires for that element.

INVARIANT: number of atoms in the QE `&SYSTEM` nat field == len(structure.sites)
  == number of ATOMIC_POSITIONS lines written.
  CHECK IN CODE: assert all three are equal before the file is closed.
```

Add new invariants here as you discover new failure modes — this section is
meant to grow over time as a running list of "bugs we've seen before."

---

## 2. DECISION RULES (branch points, as condition tables)

```
RULE: pseudopotential relativistic treatment
  IF   any element in the structure has atomic number > 54 (post-Xe),
       OR the calculation explicitly targets spin-orbit-coupled physics
       (topological states, heavy-element magnetism, Rashba splitting)
  THEN use fully relativistic pseudopotentials
  ELSE use scalar relativistic pseudopotentials
  ONLY IF unsure whether a given system needs SOC, use search_docs / ask
       before defaulting to scalar relativistic.

RULE: k-path source
  IF   doing a bandstructure calculation
  THEN generate the k-path from THIS structure via
       pymatgen.symmetry.bandstructure.HighSymmKpath(structure)
  ELSE (SCF/relax/NSCF-only) no explicit k-path needed, use a k-point mesh
       consistent with the structure's symmetry via
       pymatgen's automatic k-point generator or a fixed Monkhorst-Pack grid
       density (specify density here, e.g. 0.03 1/Å³-1 spacing).
  NEVER reuse a k-path generated for one structure on a different structure,
       even if they are "close" perturbations of each other — symmetry can
       break under perturbation.

RULE: pseudopotential functional / type / accuracy
  IF   [fill in: e.g. structure is an oxide needing +U] THEN [...]
  ELSE [default: PBEsol, norm-conserving, stringent, UPF — state explicitly]

RULE: perturbation magnitude for generated structures
  IF   generating room-temperature thermal displacements
  THEN sample displacements from the Boltzmann/harmonic estimate consistent
       with the material's known Debye temperature or a stated fixed
       displacement magnitude (state the number and its justification
       explicitly in the script's docstring, not just in the prompt).
  ELSE if a different physical scenario is intended, state it here.
```

---

## 3. WORKFLOW (short, because branching lives above)

```
STEP 0: Read example/generate_structures.py, example/setup_jobs.py, and
        example/plot_bands.py in full before writing anything. Note per file:
        - what each constant/function does
        - what is CsPbBr3-specific and must be removed/replaced for this system

STEP 1: Obtain base structure via generate_cif. Record the Materials Project
        ID used.

STEP 2: Generate N perturbed structures with pymatgen.
        - Apply INVARIANT checks (atom count) immediately per structure.
        - Apply DECISION RULE (perturbation magnitude) explicitly, with the
          chosen magnitude and justification written into the script.

STEP 3: Get pseudopotentials via get_pseudopotential.
        - Apply DECISION RULE (relativistic treatment, functional/type).
        - Apply INVARIANT check (element coverage + relativistic match).

STEP 4: Before writing setup_jobs code, write a note walking through every
        QE input option and its intended value for this workflow. Use
        search_docs for anything uncertain.

STEP 5: Generate input files + submit.sh (account, email as specified).
        - Apply INVARIANT checks (nat/site count consistency) per file.

STEP 6: Test ONE representative input file with run_espresso_workflow
        (e.g. calculations/000). Fix all errors before scaling to the rest.
        Only after a clean single-case run, generate/submit the remaining
        cases.

STEP 7: Read example/plot_bands.py, note changes needed
        (DECISION RULE: k-path source), write and run the new version.

CHECKPOINT after each step: state in 1-2 sentences what was verified before
proceeding to the next step. Do not batch verification to the end.
```

---

## 4. DONE CRITERIA (unchanged from your original, still worth keeping explicit)

- Execute all three scripts end-to-end without errors.
- Read at least one intermediate CIF and one QE input file; confirm each
  INVARIANT holds for that specific case (not just "looks reasonable").
- Summary must include: MP IDs used, workflow outline, assumptions +
  justification, anything needing expert review.

---

### Why this structure (short version, for your own reference)

- **Invariants → code assertions**, not prompt reminders. This is the fix for
  the "atom count mismatch" and "wrong k-path" class of bug — those are
  consistency failures, not knowledge gaps, and prose reminders don't reliably
  prevent them across a long multi-file task. A runtime assert does.
- **Decision rules → condition tables**, not embedded in workflow prose. This
  is the fix for "used scalar relativistic when it needed full relativistic" —
  that error is best avoided by isolating the rule from the surrounding
  narrative so it can't get lost in a long paragraph.
- **Checkpoints between steps**, not one big verification at the end. Errors
  compound across a pipeline; catching them at step 2 is much cheaper than
  catching them at step 7.