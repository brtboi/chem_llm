import numpy as np
import seekpath
from ase.io import read

# Read the base structure
base = read('structures/MoS2_primitive.cif')
cell_tuple = (base.cell[:], base.get_scaled_positions(), base.get_atomic_numbers())
res = seekpath.get_path(cell_tuple, symprec=0.01)

# Print all keys and their types/shapes
print("Keys in seekpath.get_path return dict:")
for key in sorted(res.keys()):
    val = res[key]
    if isinstance(val, np.ndarray):
        print(f"  {key}: ndarray shape={val.shape} dtype={val.dtype}")
    elif isinstance(val, list):
        print(f"  {key}: list len={len(val)}, first element type={type(val[0]).__name__ if val else 'empty'}")
        if val and isinstance(val[0], (list, np.ndarray, tuple)):
            print(f"    first element: {val[0]}")
    else:
        print(f"  {key}: {type(val).__name__} = {val}")

# Specifically inspect point_coords
print("\npoint_coords details:")
pc = res['point_coords']
print(f"  type: {type(pc)}")
print(f"  len: {len(pc)}")
for i, item in enumerate(pc):
    print(f"  [{i}]: type={type(item).__name__}, value={item}")

# Also check if there's a 'point_labels' or similar key
print("\nAll keys:", sorted(res.keys()))
