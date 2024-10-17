import pathlib
from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch


class Dataset(ABC):

    @abstractmethod
    def get_sample(self, idx):
        """Return sample from the dataset.

        Args:
            idx: the sample index to get

        Returns:
            xyz: atomic coordinates
            frc: the forces
            ene: the energy
            at: atomic numbers
            box: box vectors
            subsel: atomic subselection
        """

    @abstractmethod
    def get_random_sample(self):
        """Return a random sample from the dataset.

        Returns:
            xyz: atomic coordinates
            frc: the forces
            ene: the energy
            at: atomic numbers
            box: box vectors
            subsel: atomic subselection
        """


class NumpyDataset(Dataset):
    def __init__(self, fdir, device):
        self.device = device
        self.xyz = torch.tensor(np.load(f"{fdir}/xyz.npy"), device=device)
        self.frc = torch.tensor(np.load(f"{fdir}/frc.npy"), device=device)
        self.at = torch.tensor(np.load(f"{fdir}/at.npy"), device=device)

        # load box data if exits, else 'inf' box
        fbox = pathlib.Path(f"{fdir}/box.npy")
        if fbox.exists():
            self.box = torch.tensor(np.load(f"{fdir}/box.npy"), device=device)
        else:
            self.box = torch.tensor([1e16, 1e16, 1e16])

        # load subsel if exists, else []
        fsubsel = pathlib.Path(fdir) / "subsel.npy"
        if fsubsel.exists():
            self.subsel = list(np.load(f"{fdir}/subsel.npy"))
        else:
            self.subsel = []
        print("Subsel", self.subsel)

        self.Nsamples = self.xyz.shape[0]

    def get_sample(self, idx):
        xyzi = self.xyz[idx].requires_grad_(True)
        frci = self.frc[idx]
        return xyzi, frci, 0.0, self.at, self.box, self.subsel

    def get_random_sample(self):
        idx = np.random.randint(self.Nsamples)
        return self.get_sample(idx)


class h5pyDataset:
    def __init__(self, fpath, device):
        self.device = device
        self.h5f = h5py.File(fpath, "r")
        self.data = self.h5f["data"]
        self.subsel = []
        self.Nsamples = 1e5
        self.mean_e = -7653.337455462715
        self.std_e = 1712.1638370170933


    def get_sample(self, idx):
        print("[INFO] get_sample(idx) not implemented. Giving you a random sample.")
        return self.get_random_sample()

    def get_random_sample(self):
        sys = self.data[np.random.choice(list(self.data.keys()))]
        rxn = sys[np.random.choice(list(sys.keys()))]
        #rxn = rxns[np.random.choice(list(rxns.keys()))]
        state = np.random.choice(["product", "reactant", "transition_state"])
        conf = rxn[state]
        idx = np.random.choice(conf["positions"].shape[0])
        xyz = torch.tensor(conf["positions"][idx], requires_grad=True)
        fk = [i for i in rxn.keys() if "forces" in i][0]
        ek = [i for i in rxn.keys() if ".energy" in i][0]
        frc = torch.tensor(conf[fk][idx])
        ene = torch.tensor(conf[ek][idx])
        at = torch.tensor(conf["atomic_numbers"])
        box = torch.tensor([1e9, 1e9, 1e9])
        return xyz, frc, ene - self.mean_e, at, box, self.subsel
