import MDAnalysis as mda
import glob
import tqdm
import numpy as np

u = mda.Universe("../cp2k_input/conf.gro")
u.atoms.dimensions = None

maxH = 0
maxO = 0
distO = []
distH = []
for traj in tqdm.tqdm(glob.glob("../load/*/accepted/*.xyz")[:50], smoothing = 0):
    utmp = mda.Universe(traj)
    for frame in utmp.trajectory:
        u.atoms.positions = utmp.atoms.positions
        sol = u.select_atoms("around 10 resname UNK")
        H = sol.select_atoms("name H*")
        O = sol.select_atoms("name O*")
        distH.append(H.atoms.n_atoms)
        distO.append(O.atoms.n_atoms)
        if H.atoms.n_atoms > maxH:
            maxH = H.atoms.n_atoms
        if O.atoms.n_atoms > maxO:
            maxO = O.atoms.n_atoms

print("H:", maxH, "O:",maxO)
#np.save("distH", distH)
#np.save("distO", distO)
