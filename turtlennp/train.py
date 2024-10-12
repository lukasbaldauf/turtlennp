import torch
import ase

import numpy as np
from ase.io import read
from torch import nn, optim

from turtlennp.model import Model
from turtlennp.normalize import normalize

device = torch.device("cuda")
logfile = "out.log"

atoms = read("diels.xyz")
subsel = torch.arange(22, device = device)
typemap = {'H':0,'C':1,'O':2}
at = torch.tensor([typemap[i] for i in atoms.get_chemical_symbols()], device = device)

xyz_arr = torch.tensor(np.load("data_diels/nowrap_diels_coord.npy"), requires_grad = True, device = device)
frc = torch.tensor(np.load("data_diels/nowrap_diels_frc.npy"), device = device)*51.422086190832 # hartree/bohr to eV/Å
ene = torch.tensor(np.load("data_diels/dies_ene.npy"), device = device)*27.2107 # hartree to eV

opts = {"comment":"diels_ene_small",
        "architecture": [40, 30, 20, 10,1],
        "lr": 1e-3,
        "subsel":[i for i in range(22)],
        "cutoff":4.0, # NOTE if this is changed recalc the sel
        "gamma": 0.9,
        "every": 40000,
        "batch_size": 1,
        "a_sel":[48,sum(at==1),24],
        "sel":[sum(at==0), sum(at==1), sum(at==2)],
        "box":[25.81, 25.59, 25.16],
        }

with open(logfile, "a") as wfile:
    print(opts, file = wfile)

m = model(typemap, opts["sel"], opts["architecture"], a_sel = opts["a_sel"], device = device, cutoff = opts["cutoff"])
m.subsel = subsel

box_size = torch.tensor(box, device = device)

res = normalize(m, xyz_arr, at, subsel, box_size)
np.save(f"res_{opts['comment']}", res)
m.norm = torch.tensor(np.load(f"res_{opts['comment']}.npy")[:,:,:2], dtype = torch.float32, device = device)
m.norm[:,:,1] = torch.clip(m.norm[:,:,1], 0.01)
prms = list(m.networks[0].parameters()) + list(m.networks[1].parameters()) + list(m.networks[2].parameters()) # NBNB nr. of prms

optimizer = optim.Adam(prms, lr = opts["lr"])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = opts["gamma"])

idxt = [123]
xyzt = xyz_arr[idxt].clone().detach().requires_grad_(True)
print("epoch losst loss mean(abs(pred_force)), mean(abs(force))")
for epoch in range(2000000):
    optimizer.zero_grad()
    idx = np.random.choice(len(xyz_arr), opts["batch_size"], replace = False)
    idx = torch.tensor(idx)
    xyzi = xyz_arr[idx].clone().detach().requires_grad_(True)
    energy, frci = m.calculate_ef(xyzi, at.expand(opts["batch_size"], at.shape[0]), subsel, box_size, force_gradients = True)
    loss = torch.sqrt(torch.mean((frci[0,m.subsel] - frc[idx,m.subsel])**2))
    loss.backward()

    if epoch%100 == 0:
        et,frct = m.calculate_ef(xyzt, at.expand(len(idxt), at.shape[0]), subsel, box_size)
        losst = torch.sqrt(torch.mean((frct[0,m.subsel] - frc[idxt,m.subsel])**2))
        print(epoch, losst.item(), loss.item(), torch.mean(torch.abs(frci[0,m.subsel])).item(), torch.mean(torch.abs(frc[idx,m.subsel])), scheduler.get_last_lr()[0], flush = True)
        with open(logfile, "a") as wfile:
                print(epoch, losst.item(), loss.item(), torch.mean(torch.abs(frci[0,m.subsel])).item(), torch.mean(torch.abs(frc[idx,m.subsel])), scheduler.get_last_lr()[0], file = wfile)
    if epoch%10000 == 0:
        torch.save(m, f"model_{opts['comment'].pt")
    if epoch%opts["every"]==0:
        if epoch !=0:
            scheduler.step()
    optimizer.step()

torch.save(m, f"model_{opts['comment'].pt")
