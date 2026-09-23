import numpy as np
import sys

work_func = float(sys.argv[1])
ref_gap = 1.7

bs = np.loadtxt("expBandStruct_0.par")

vb = bs[:, 104]
cb = bs[:, 105]
vbmax = vb.max()
cbmin = cb.min()
shift = vbmax - work_func
band_gap = cbmin - vbmax
shift_for_bandgap = ref_gap - band_gap
print(f"VBmax = {vbmax} CBmin = {cbmin}")
print(f"Bandgap = {band_gap} shift to open bands = {shift_for_bandgap}")

# Shift the valence band maximum to the inputted work function
bs_shift = np.zeros_like(bs)
bs_shift[:, 1:] = bs[:, 1:] - shift
bs_shift[:, 0] = bs[:, 0]

# Open the bands up to have a band gap of 1.7 eV
bs_shift[:, 105:] = bs_shift[:, 105:] + shift_for_bandgap
#bs_shift[:, 1:20] = 0.0
k_points = bs_shift[:, 0]
#bs_shift = bs_shift[:, 19:]

#output = np.concatenate(bs_shift, axis=1 )
np.savetxt("expBandStruct_0_shift.par", bs_shift, fmt="%.6f")
