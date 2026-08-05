from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Load the Materials Project CIF
structure = Structure.from_file('csPbBr3_materials_project.cif')

# Analyze the space group
sg_analyzer = SpacegroupAnalyzer(structure)

# Get the correct space group
correct_space_group = sg_analyzer.get_space_group_symbol()
correct_number = sg_analyzer.get_space_group_number()

print(f'Correct space group: {correct_space_group} (Number: {correct_number})')

# Get the primitive structure to verify Z
primitive_structure = sg_analyzer.get_primitive_standard_structure()
print(f'Primitive structure formula: {primitive_structure.composition.formula}')
print(f'Number of formula units in primitive cell (Z): {primitive_structure.num_sites / len(primitive_structure.composition)}')
