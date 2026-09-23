import os

# Must be set before chem_llm.config is imported (it reads this env var at
# import time to compute WORK_DIR). This supersedes test26's earlier
# ASE-plot-fix content (that ASE work has been rolled back to the
# ase-migration branch, see git log) -- this run is master's own CsPbBr3
# pipeline (example/, pymatgen-based) taken one leg further: run real QE
# DFT for one structure, then adapt it into a DeePseudopot input bundle and
# actually train a neural-network pseudopotential (chem_llm/tools/
# deepseudopot.py's train_deepseudopot tool). Fresh sandbox/test26 dir
# (clear_dir=True below).
os.environ["MATAGENT_WORK_DIR"] = "sandbox/test26"

import torch
import json
from chem_llm import config
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_HOME"] = config.HF_HOME
from transformers import AutoModelForCausalLM, AutoTokenizer

from chem_llm.agent_core import run_agent

if not config.HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set.")

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, token=config.HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    config.MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16,
    token=config.HF_TOKEN,
)

COMPOUND = "CsPbBr3"
JOB = "000"

TASK = f"""
This task has two legs. Leg A (QE DFT) is the same kind of workflow you've done before with example/'s pre-existing generate_structures.py, setup_jobs.py, and plot_bands.py. Leg B (DeePseudopot) is new: carry ONE completed DFT job's band structure through to a trained neural-network pseudopotential.

Unlike prior tasks, the compound here is NOT changing -- example/'s existing scripts already build {COMPOUND} (a cubic<->orthorhombic interpolation). You do not need generate_cif or get_pseudopotential: example/template/ already contains the exact three pseudopotential files example/setup_jobs.py's ATOMIC_SPECIES lines reference by name (Cs.rel-pbe-spn-rrkjus_psl.1.0.0.UPF, Pb.rel-pbe-dn-rrkjus_psl.1.0.0.UPF, Br.USPP.FR.PBE.3.4.UPF -- production PSlibrary ultrasoft pseudopotentials, not something get_pseudopotential's Pseudo-Dojo-backed tool can reproduce by name, so do not call get_pseudopotential for this task -- it would fetch differently-named files that don't match the hardcoded ATOMIC_SPECIES lines). example/'s structure-building logic already matches this compound exactly. Leg B's reference materials (deepseudopot/) are ALSO already specific to this exact {COMPOUND} system (real Zunger-form initial pseudopotential parameters exist only for Cs/Pb/Br) -- this is why the compound must stay {COMPOUND} for this task.

=== LEG A: QE DFT (produce one real, completed DFT job) ===

STEP A0: Read example/generate_structures.py, example/setup_jobs.py, and example/plot_bands.py in full before writing anything. Note what each does.

STEP A1: Write and run a new generate_structures.py adapted from the example (51 structures total, same cubic<->orthorhombic {COMPOUND} interpolation + thermal perturbation the example already implements -- no changes to the physics needed). Apply the nat/ntyp INVARIANT check per structure.

STEP A2: Write and run a new setup_jobs.py adapted from the example, with exactly ONE required change: the SCF calculation's K_POINTS must be a single Gamma point (`K_POINTS automatic` / `1 1 1 0 0 0`) instead of the example's `8 8 8 0 0 0` -- an 8x8x8 mesh with noncolin spin-orbit coupling and nosym=.true. is too slow to converge in a practical session; a single Gamma point is a validated substitution that changes nothing else (nat/ntyp/electron count/pseudopotentials/bands k-path are unchanged), documented in deepseudopot/README.md's last section if you want the justification. Do not change anything else about the QE input (the bands calculation's k-path and nbnd=200 must stay exactly as the example has them -- Leg B depends on them). Apply the nat/ntyp and k-path INVARIANT checks, and call validate_qe_input on calculations/{JOB}'s input files.

STEP A3: Run QE DFT for calculations/{JOB} ONLY via run_espresso_workflow (SCF + bands + bands post-processing). Fix all errors before moving on. Do NOT generate or run any other calculations/NNN directory -- Leg B only uses this one job; the other 50 structures exist only because generate_structures.py always makes an ensemble, and running DFT on them is out of scope for this task.

STEP A4: Read example/plot_bands.py, adapt it for calculations/{JOB}, run it, and confirm bands_{JOB}.pdf is produced. (This step is a normal sanity check of Leg A, not required by Leg B, but do not skip verifying Leg A's DFT output is real and sane before moving to Leg B.)

=== LEG B: DeePseudopot (DFT output -> trained NN pseudopotential) ===

Everything you need for this leg is in deepseudopot/ (already in your working directory). Read deepseudopot/README.md first.

STEP B1: Read deepseudopot/qe_bands_to_ref.py in full. Note in particular:
   - generate_qe_kpath() returns the 41 fractional (crystal_b) k-points along the band path used for calculations/{JOB}/bands.in -- it is hardcoded to that exact path, so it applies UNCHANGED here (you did not change the bands k-path in STEP A2).
   - parse_band_file(filename) reads a QE bands.x-produced .dat file (repeating blocks of "kx ky kz" then that k-point's band energies) into (kpoints, bands) arrays.
   - compute_kpath_distances(kpoints) turns k-points into a cumulative distance along the path -- the x-axis DeePseudopot's expBandStruct_0.par expects in its first column.
   - main() ties these together for ONE job: parses calculations/{JOB}/{JOB}.bands.dat, drops the first 72 bands (`bands[:, 72:]`) to keep the 128 bands DeePseudopot fits to, and writes kpoints_0.par + expBandStruct_0.par -- but it does so by writing hardcoded filenames INTO THE CURRENT WORKING DIRECTORY. You cannot call this script as-is with run_python: run_python always runs with YOUR working directory as cwd, and you need the output written into a new nn_inputs_g/ directory, not wherever main() happens to run.
   - The 72-dropped / 128-kept band split, and shift.py's hardcoded k-path column indices 104 (VBM) / 105 (CBM) in expBandStruct_0.par, come from {COMPOUND}'s electron count (176 electrons -> VBM = band 176) and are correct for THIS job unchanged, because the composition and pseudopotentials are identical to the validated recipe -- only the structure itself differs (a perturbed ensemble member instead of the pristine cubic endpoint). Do not re-derive or change these numbers.

STEP B2: Read deepseudopot/setup_nn_inputs.py in full. Note in particular:
   - parse_pw_input(filename) parses a pw.in file's CELL_PARAMETERS and ATOMIC_POSITIONS crystal blocks into (scale_in_bohr, normalized_cell, atoms).
   - write_system_par(outfile, scale, cell, atoms) writes DeePseudopot's system_0.par format from that.
   - The MAIN LOOP (bottom of the file) does, for each completed calculations/NNN job: copy nn_template/ -> nn_inputs/inputs_NNN_<suffix>/, write system_0.par from that job's pw.in, run qe_bands_to_ref.py as a subprocess with cwd=calculations/NNN to produce kpoints_0.par / expBandStruct_0.par THERE, copy those two files into the new nn_inputs dir, then run nn_template/shift.py (work-function argument -5.25) inside the new nn_inputs dir to align the valence-band maximum, and -- for suffix=="g" only -- run nn_template/select_rows.py to keep only k-path rows 11 and 21 (1-indexed) of kpoints_0.par and expBandStruct_0.par, a fast reduced-k-point bundle.
   - You have exactly ONE completed job (calculations/{JOB}), not a loop over many -- adapt this logic into a single new script, you do not need (and should not try) to reuse the NNN-loop as-is.

STEP B3: Write ONE new Python script, build_nn_inputs.py, that assembles a complete DeePseudopot input bundle for this one job at nn_inputs_g/ by PORTING (not blindly subprocess-calling) the logic from STEP B1-B2, using explicit paths throughout -- no cwd= subprocess tricks, no relying on a script's own hardcoded output filenames:
   a. Copy deepseudopot/nn_template/ to nn_inputs_g/ (shutil.copytree).
   b. Parse calculations/{JOB}/pw.in with an adapted parse_pw_input, and write nn_inputs_g/system_0.par with write_system_par.
   c. Parse calculations/{JOB}/{JOB}.bands.dat with an adapted parse_band_file, build the k-path with generate_qe_kpath and compute_kpath_distances, and write nn_inputs_g/kpoints_0.par and nn_inputs_g/expBandStruct_0.par (dropping the first 72 bands as in STEP B1).
   d. Apply the SAME valence-band-maximum shift shift.py performs (work function -5.25 eV, target gap 1.7 eV -- shift.py's ref_gap): load nn_inputs_g/expBandStruct_0.par as a numpy array `bs` (column 0 is k-distance, so band columns start at 1), compute `vbmax = bs[:, 104].max()`, `cbmin = bs[:, 105].min()`, `shift = vbmax - (-5.25)`, `shift_for_bandgap = 1.7 - (cbmin - vbmax)`; subtract `shift` from every band column (1 onward), then add `shift_for_bandgap` to columns 105 onward -- exactly shift.py's arithmetic, applied in-process instead of via subprocess. This MUST run on the full 41-row band structure (searching VBM/CBM over the whole k-path, not a subset) -- do this step BEFORE step (e). Overwrite expBandStruct_0.par with the result.
   e. Apply the SAME row-selection select_rows.py performs: keep only rows 11 and 21 (1-indexed) of BOTH kpoints_0.par and expBandStruct_0.par, overwriting each file with just those two rows -- this is what makes the bundle fast to train (Hamiltonian caching is per k-point and NOT parallelized within one system, so fewer k-points is the single biggest lever on train_deepseudopot's runtime).

   Run this script with run_python. If it errors, fix it and rerun -- do not proceed with a broken or partial nn_inputs_g/.

STEP B4: Sanity-check nn_inputs_g/ before training: read_file nn_inputs_g/kpoints_0.par and nn_inputs_g/expBandStruct_0.par and confirm each has exactly 2 rows, and that expBandStruct_0.par's values look like real energies in eV (not NaN, not all zero, not wildly outside a plausible few-eV range for a semiconductor band structure). Also confirm nn_inputs_g/system_0.par's atoms block lists 20 atoms matching calculations/{JOB}/pw.in's ATOMIC_POSITIONS (4 Cs + 4 Pb + 12 Br).

STEP B5: Call train_deepseudopot with inputs_folder="nn_inputs_g" and results_folder="dpp_results". Check `success` explicitly. If it is false, read stderr_tail/stdout_tail, diagnose (a malformed file from STEP B3 -- e.g. a column-index or shape mismatch -- is the most likely cause), fix build_nn_inputs.py, rerun it, and call train_deepseudopot again. Do not call done while success is false.

STEP B6: Once train_deepseudopot succeeds, read_file one of the returned final_pot_files entries and confirm it contains real numeric pseudopotential data (not empty, not NaN).

Requirements:
- Organize output files in a clear directory structure.
- Comment the code where nontrivial crystallographic or DeePseudopot-format operations are performed.
- Do NOT modify anything under deepseudopot/ (treat it as read-only reference input); write all new files (build_nn_inputs.py, nn_inputs_g/, dpp_results/) into your own working directory.

DONE CRITERIA:
- Leg A: calculations/{JOB} has real, completed SCF + bands + bands-post output, and bands_{JOB}.pdf was produced.
- Leg B: a working nn_inputs_g/ bundle you verified in STEP B4, a train_deepseudopot call that returned success: true, and a final_pot_*.dat file from dpp_results/ you read and confirmed has real content.
- The done summary must include: which structure (composition, interpolation parameter `par`) was actually run through DFT, trained_model_path and the final_pot_files list from the successful train_deepseudopot call, a one-paragraph outline of how build_nn_inputs.py adapts qe_bands_to_ref.py + setup_nn_inputs.py's logic for a single job, and any assumptions made and why (e.g. why the Gamma-point SCF mesh, and the -5.25 eV work function / k-path rows 11/21, were used unchanged rather than re-derived). No Materials Project lookup is involved in this task -- do not mention an MP ID.
"""

os.makedirs(config.WORK_DIR, exist_ok=True)
os.chdir(config.WORK_DIR)
print(f"Working directory: {os.getcwd()}")

final_state = run_agent(
    TASK, model, tokenizer, verbose=True,
    clear_dir=True, load_path=config.EXAMPLE_DIR, load_deepseudopot=True,
    max_steps=100,  # both legs combined -- prior QE-only runs alone used 14-60 steps
)

print("\nFINAL STATE:\n", json.dumps(final_state.to_dict(), indent=2))
