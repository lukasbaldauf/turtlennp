import numpy as np
import torch
from torch import optim

from turtlennp.dataset import *
from turtlennp.model import Model
from turtlennp.normalize import normalize


def loss_subsel(f, f0, subsel):
    if subsel:
        loss = torch.sqrt(torch.mean((f[subsel, :] - f0[subsel, :]) ** 2))
    else:
        loss = torch.sqrt(torch.mean((f - f0) ** 2))
    return loss


def normalize_features(model, datasets, opts):
    """Standardize the dataset features."""
    if opts["norm_load"]:
        model.norm = torch.tensor(
            np.load(opts["norm_load"])[:, :, :2],
            device=model.device,
            dtype=opts["dtype"],
        )
        model.norm[:, :, 1] = torch.clip(model.norm[:, :, 1], 0.01)
    else:
        for i, dataset in enumerate(datasets):
            nsamples = dataset.Nsamples
            if opts["norm_N"][i] > nsamples:
                opts["norm_N"][i] = nsamples
                print(f"[INFO]  opts['norm_N'][i] > {nsamples} in {dataset}")
        res = normalize(
            model, datasets, N=opts["norm_N"], mode=opts["norm_mode"]
        )
        np.save(f"res_{opts['comment']}", res)
        model.norm = torch.tensor(
            res[:, :, :2], device=model.device, dtype=opts["dtype"]
        )
        model.norm[:, :, 1] = torch.clip(model.norm[:, :, 1], 0.01)


def train_model(m, datasets, opts, device, logfile="out.log"):
    """Trains the model."""
    params_to_train = []
    for atomic_network in m.networks:
        params_to_train += list(atomic_network.parameters())
    optimizer = optim.Adam(params_to_train, lr=opts["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=opts["gamma"]
    )

    t_idx = opts["test_frame"]  # Validation example index
    t_set = opts["test_dataset"]
    xyzt, frct, att, boxt, subselt = datasets[t_set].get_sample(t_idx)

    with open(logfile, "a") as wfile:
        wfile.write(str(opts) + "\n")

    msg = "Epoch  losst  loss  mean(abs(pred_force))  mean(abs(force))  lr"
    print(msg)
    with open(logfile, "a") as wfile:
        wfile.write(msg + "\n")

    for epoch in range(opts["epochs"]):
        optimizer.zero_grad()

        # Random batch selection
        dataset = datasets[np.random.randint(len(datasets))]
        xyzi, frci, at, box, subsel = dataset.get_random_sample()

        # Model prediction
        epred, fpred = m.calculate_ef(
            xyzi,
            at,
            box,
            subsel=subsel,
            attach_grad=True,
        )

        loss = loss_subsel(fpred, frci, subsel)
        loss.backward()

        # Periodic logging and saving
        if epoch % opts["log_every"] == 0:
            etest, ftest = m.calculate_ef(
                xyzt,
                att,
                boxt,
                subsel=subselt,
            )
            losst = loss_subsel(ftest, frct, subsel)
            if subsel:
                mae_fpred = torch.mean(torch.abs(fpred[subsel])).item()
                mae_f = torch.mean(torch.abs(frci[subsel]))
            else:
                mae_fpred = torch.mean(torch.abs(fpred)).item()
                mae_f = torch.mean(torch.abs(frci))
            msg = (
                f"{epoch} {losst.item()} {loss.item()} "
                f"{mae_fpred} "
                f"{mae_f} "
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
    dtype = torch.float32

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
        "a_sel": [4, 4, 4, 4],  # angular info of the N nearest atoms of type i
        "sel": [4, 4, 4, 4],  # radial info of N nearest atoms of type i
        "at_map": {1: 0, 6: 1, 7: 2, 8: 3},  # just to save it as a model prop
        "epochs": int(50),  # number of training loops
        # if given, we continue training with this model
        "restart_model": False,
        "device": "cpu",
        "num_threads": 1,
        "dtype": dtype,
        "save_every": 1000,  # save model every this many steps
        "log_every": 1,  # log output every this many steps
        "test_frame": 3,  # [4,2],
        "test_dataset": 0,
        "norm_load": False,  # "res_diels_very_small.npy", #False,
        "norm_N": [10],
        "sel_prob": [1.0],
        "norm_mode": ["random"],
    }

    device = torch.device(opts["device"])
    torch.set_num_threads(opts["num_threads"])
    torch.set_num_interop_threads(opts["num_threads"])

    datasets = [h5pyDataset("./transition1x.h5", device)]
   # datasets = [DatasetDiels(".", device)]

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
        normalize_features(model, datasets, opts)

    # load norm if we have precomputed it
    # TODO: assert shape
    train_model(model, datasets, opts, device)


if __name__ == "__main__":
    main()
