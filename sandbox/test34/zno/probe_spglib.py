import spglib
import numpy as np
import ase.io

# Quick probe: what attributes does SpglibDataset actually have?
base = ase.io.read('structures/zno_base.cif')
lattice = np.array(base.cell.array)
positions = np.array(base.get_scaled_positions())
numbers = np.array(base.get_atomic_numbers())
cell = (lattice, positions, numbers)
dataset = spglib.get_symmetry_dataset(cell, symprec=1e-5)
print('type:', type(dataset))
print('dir:', [a for a in dir(dataset) if not a.startswith('_')])
print('number:', getattr(dataset, 'number', 'MISSING'))
print('international_number:', getattr(dataset, 'international_number', 'MISSING'))
print('hall_symbol:', getattr(dataset, 'hall_symbol', 'MISSING'))
print('pointgroup:', getattr(dataset, 'pointgroup', 'MISSING'))
print('crystal_system:', getattr(dataset, 'crystal_system', 'MISSING'))
