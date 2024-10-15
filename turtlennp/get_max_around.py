import glob

import MDAnalysis as mda
import tqdm

u = mda.Universe("../cp2k_input/conf.gro")
u.atoms.dimensions = None

maxH = 0
maxO = 0
distO = []
distH = []
for traj in tqdm.tqdm(glob.glob("../load/*/accepted/*.xyz")[:50], smoothing=0):
    utmp = mda.Universe(traj)
    for frame in utmp.trajectory:
        u.atoms.positions = utmp.atoms.positions
        sol = u.select_atoms("around 10 resname UNK")
        selH = sol.select_atoms("name H*")
        selO = sol.select_atoms("name O*")
        distH.append(selH.atoms.n_atoms)
        distO.append(selO.atoms.n_atoms)
        if selH.atoms.n_atoms > maxH:
            maxH = selH.atoms.n_atoms
        if selO.atoms.n_atoms > maxO:
            maxO = selO.atoms.n_atoms

print("H:", maxH, "O:", maxO)
# np.save("distH", distH)
# np.save("distO", distO)
