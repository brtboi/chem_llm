from pymatgen.core import Structure

# Load the CIF file
structure = Structure.from_file('CsPbBr3.cif')

# Get the space group information
symbol, number = structure.get_space_group_info(symprec=0.01, angle_tolerance=5.0)

print(f"Space group symbol: {symbol}")
print(f"Space group number: {number}")
