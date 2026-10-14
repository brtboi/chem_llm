#!/usr/bin/env python3
"""Inspect what kp_atoms returns to understand the type issue."""
import ase.io
from chem_llm.kpath import standard_kpath, kp_atoms

atoms = ase.io.read('structures/structure_000.cif')
kp = standard_kpath(atoms)
prim = kp_atoms(kp)

print(f"Type of kp_atoms result: {type(prim)}")
print(f"Is it a dict? {isinstance(prim, dict)}")
if isinstance(prim, dict):
    print(f"Keys: {list(prim.keys())}")
    for k, v in prim.items():
        print(f"  {k}: {type(v)} = {v if not hasattr(v, 'shape') else v.shape}")
else:
    print(f"Is it Atoms? {isinstance(prim, ase.Atoms)}")
    print(f"len(prim) = {len(prim)}")
    print(f"symbols = {prim.get_chemical_symbols()}")
