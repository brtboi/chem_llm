import numpy as np

old_file = np.loadtxt("old_qSpace_pot.dat", skiprows=1)

n_cs = 4
n_pb = 4
n_br = 12

q = old_file[:, 0].reshape(-1,1)
cs_pot = old_file[:, 2].reshape(-1,1).repeat(n_cs, axis=1)
br_pot = old_file[:, 1].reshape(-1,1).repeat(n_br, axis=1)
pb_pot = old_file[:, 3].reshape(-1,1).repeat(n_pb, axis=1)

outfile = np.concatenate((q, br_pot, cs_pot, pb_pot), axis=1)

np.savetxt("init_qSpace_pot.par", outfile, fmt="%.8f", header="q          " + n_br * " v(q)_Br       " + n_cs * "v(q)_Cs     " + n_pb * "v(q)_Pb      ")

print(f"Done with merge_qSpace.py")