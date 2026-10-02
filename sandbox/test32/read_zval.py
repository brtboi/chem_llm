#!/usr/bin/env python3
"""Read z_valence from the Ti and O pseudopotential files."""
from chem_llm.tools.upf import read_upf_header

for elem, path in [('Ti', 'template/Ti.pbesol_upf'), ('O', 'template/O.pbesol_upf')]:
    header = read_upf_header(path)
    z_val = header['z_valence']
    print(f"{elem}: z_valence = {z_val}")

# Compute total electrons for the 6-atom primitive cell (2 Ti + 4 O)
ti_z = read_upf_header('template/Ti.pbesol_upf')['z_valence']
o_z = read_upf_header('template/O.pbesol_upf')['z_valence']
total_electrons = 2 * ti_z + 4 * o_z
occupied_bands = total_electrons // 2  # spin-unpolarized, non-metallic
print(f"\nTotal electrons (2 Ti + 4 O): {total_electrons}")
print(f"Occupied bands (spin-unpolarized): {occupied_bands}")
print(f"Recommended nbnd (2x occupied, rounded up): {2 * occupied_bands}")
