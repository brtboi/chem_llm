from chem_llm.qe_output import read_pw_out

pw = read_pw_out('calculations/000/pw.out')
print(f"number of electrons: {pw['n_electrons']}")
print(f"n_occupied: {pw['n_occupied']}")
print(f"fermi_energy_eV: {pw['fermi_energy_eV']}")
print(f"spin_orbit: {pw['spin_orbit']}")

# nbnd in bands.in is 40
# occupied bands = n_electrons / 2 for spin-unpolarized
n_occ = pw['n_electrons'] / 2
print(f"occupied bands (n_electrons/2): {n_occ}")
assert 40 >= n_occ, f"nbnd=40 is too small: need >= {n_occ}"
print("OK: nbnd=40 >= occupied bands")
