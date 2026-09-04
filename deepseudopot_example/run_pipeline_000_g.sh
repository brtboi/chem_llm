#!/bin/bash
set -e
PY=/pscratch/sd/b/brenthu/chem_llm/.venv/bin/python
ROOT=/pscratch/sd/b/brenthu/chem_llm/deepseudopot_example
DPP=/pscratch/sd/b/brenthu/chem_llm/chem_llm/DeePseudopot

cd "$ROOT/calculations/000"
echo "STEP 1: qe_bands_to_ref.py"
$PY ../../qe_bands_to_ref.py 000.bands.dat
cp kpoints_0.par expBandStruct_0.par "$ROOT/nn_inputs_000_g/"

cd "$ROOT/nn_inputs_000_g"
echo "STEP 2: shift.py"
$PY shift.py -5.25
mv expBandStruct_0_shift.par expBandStruct_0.par
echo "STEP 3: select_rows.py"
$PY select_rows.py
echo "BUNDLE READY"
cat expBandStruct_0.par

echo "STEP 4: train_deepseudopot"
mkdir -p "$ROOT/results_000_g"
cd "$DPP"
$PY -u main.py "$ROOT/nn_inputs_000_g/" "$ROOT/results_000_g/"
echo "TRAINING DONE rc=$?"
