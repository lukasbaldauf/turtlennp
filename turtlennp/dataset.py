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
        fsubsel = pathlib.Path(f"{dir}/subsel.npy")
        if fsubsel.exists():
            self.subsel = list(np.load(f"{fdir}/subsel.npy"))
        else:
            self.subsel = []

        self.Nsamples = self.xyz.shape[0]

    def get_sample(self, idx):
        xyzi = self.xyz[idx].requires_grad_(True)
        frci = self.frc[idx]
        return xyzi, frci, self.at, self.box, self.subsel

    def get_random_sample(self):
        idx = np.random.randint(self.Nsamples)
        return self.get_sample(idx)


class h5pyDataset:
    def __init__(self, fdir, device):
        self.device = device
        self.h5f = h5py.File(f"{fdir}/transition1x.h5", "r")
        self.data = self.h5f["data"]
        self.subsel = []

    def get_sample(self, idx):
        raise NotImplementedError

    def get_random_sample(self):
        sys = self.data[np.random.choice(list(self.data.keys()))]
        rxn = sys[np.random.choice(list(sys.keys()))]
        state = np.random.choice(["product", "reactnt", "transition_state"])
        conf = rxn[state]
        idx = np.random.choice(conf["positions"].shape[0])
        xyz = torch.tensor(conf["positions"][idx], requires_grad=True)
        fk = [i for i in rxn.keys() if "forces" in i][0]
        frc = torch.tensor(conf[fk][idx])
        at = torch.tensor(conf["atomic_numbers"])
        box = torch.tensor([1e16, 1e16, 1e16])
        return xyz, frc, box, at, self.subsel
