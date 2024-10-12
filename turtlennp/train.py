import numpy as np
import torch
from torch import optim

from turtlennp.model import Model
from turtlennp.normalize import normalize


def load_data(at_map, device):
    """Loads the dataset including coordinates, forces, and energies."""
    xyz = torch.tensor(np.load("xyz.npy"), requires_grad=True, device=device)
    frc = torch.tensor(np.load("frc.npy"), device=device)
    # ene = torch.tensor(np.load("ene.npy"), device=device) * 27.210
    ene = None
    box = torch.tensor(np.load("box.npy"), device=device)
    at = torch.tensor([at_map[i] for i in np.load("at.npy")], device=device)

    return xyz, frc, ene, box, at


def normalize_data(model, xyz, box, at, opts):
    """Initializes the model and sets up parameters."""
    res = normalize(model, xyz, at, opts["subsel"], box, N=100)
    np.save(f"res_{opts['comment']}", res)
    model.norm = torch.tensor(
        np.load(f"res_{opts['comment']}.npy")[:, :, :2],
        dtype=xyz.dtype,
        device=model.device,
    )
    model.norm[:, :, 1] = torch.clip(model.norm[:, :, 1], 0.01)


def train_model(m, xyz, frc, at, box, opts, device, logfile="out.log"):
    """Trains the model."""
    params_to_train = []
    for atomic_network in m.networks:
        params_to_train += list(atomic_network.parameters())
    optimizer = optim.Adam(params_to_train, lr=opts["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=opts["gamma"]
    )

    idxt = opts["test_frames"]  # Validation example index
    xyzt = xyz[idxt].clone().detach().requires_grad_(True)

    with open(logfile, "a") as wfile:
        wfile.write(str(opts) + "\n")

    msg = "Epoch  losst  loss  mean(abs(pred_force))  mean(abs(force))  lr"
    print(msg)
    with open(logfile, "a") as wfile:
        wfile.write(msg + "\n")

    for epoch in range(opts["epochs"]):
        optimizer.zero_grad()

        # Random batch selection
        idx = torch.tensor(
            np.random.choice(len(xyz), opts["batch_size"], replace=False)
        )
        xyzi = xyz[idx].clone().detach().requires_grad_(True)

        # Model prediction
        energy, frci = m.calculate_ef(
            xyzi,
            at.expand(opts["batch_size"], at.shape[0]),
            box,
            subsel=opts["subsel"],
            attach_grad=True,
        )
        loss = torch.sqrt(
            torch.mean(
                (frci[:, opts["subsel"], :] - frc[idx][:, opts["subsel"], :])
                ** 2
            )
        )
        loss.backward()

        # Periodic logging and saving
        if epoch % opts["log_every"] == 0:
            et, frct = m.calculate_ef(
                xyzt,
                at.expand(len(idxt), at.shape[0]),
                box,
                subsel=opts["subsel"],
            )
            losst = torch.sqrt(
                torch.mean(
                    (frct[:, opts["subsel"]] - frc[idxt][:, opts["subsel"], :])
                    ** 2
                )
            )
            msg = (
                f"{epoch} {losst.item()} {loss.item()}"
                f"{torch.mean(torch.abs(frci[:, opts['subsel']])).item()}"
                f"{torch.mean(torch.abs(frc[idx][:, opts['subsel']]))}"
                f"{scheduler.get_last_lr()[0]}"
            )
            print(msg)
            with open(logfile, "a") as wfile:
                wfile.write(msg + "\n")

        if epoch % opts["save_every"] == 0:
            torch.save(m, f"model_{opts['comment']}.pt")

        if epoch % opts["lr_every"] == 0 and epoch != 0:
            scheduler.step()

        optimizer.step()

    torch.save(m, f"model_{opts['comment']}.pt")


def main():
    device = torch.device("cpu")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Configuration options
    opts = {
        "comment": "diels_very_small",  # all files are saved with this prefix
        "architecture": [40, 30, 20, 10, 1],  # network architecture
        "lr": 1e-3,  # initial learning rate
        "subsel": torch.tensor([i for i in range(6)]),  # subselection
        "cutoff": 4.0,  # ignore atoms beyond this cutoff
        "gamma": 0.9,  # scale lr by this factor every 'every' steps
        "lr_every": 1000,  # update lr every this many steps
        "batch_size": 1,  # nr. of frames to pass treat each iteration
        "a_sel": [4, 4, 4],  # angular info of the N nearest atoms of type i
        "sel": [4, 4, 4],  # radial info of N nearest atoms of type i
        "at_map": {"H": 0, "C": 1, "O": 2},  # just to save it as a model prop
        "epochs": int(2e6),  # number of training loops
        # if given, we continue training with this model
        "restart_model": "model_diels_very_small.pt",
        "save_every": 1000,  # save model every this many steps
        "log_every": 100,  # log output every this many steps
        "test_frames": [10],  # test set used to calculate errors
    }

    # allowing different number of atoms per frame would also work
    xyz, frc, ene, box, at = load_data(opts["at_map"], device)

    if opts["restart_model"]:
        model = torch.load(opts["restart_model"])

    else:
        model = Model(
            opts["at_map"],
            opts["sel"],
            opts["a_sel"],
            cutoff=opts["cutoff"],
            layer_sizes=opts["architecture"],
            device=device,
        )
        normalize_data(model, xyz, box, at, opts)

    train_model(model, xyz, frc, at, box, opts, device)


if __name__ == "__main__":
    main()
