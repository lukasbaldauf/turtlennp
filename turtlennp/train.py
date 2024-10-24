import pathlib
import numpy as np
import torch
from torch import optim
import tomli

from turtlennp.dataset import *
from turtlennp.model import Model
from turtlennp.normalize import normalize


def loss_subsel(f, f0, e, e0, subsel):
    if subsel:
        lossf = torch.sqrt(torch.mean((f[subsel, :] - f0[subsel, :]) ** 2))
        losse = torch.tensor(0.0)
    else:
        lossf = torch.sqrt(torch.mean((f - f0) ** 2))
        losse = torch.sqrt((e - e0)**2)
    return losse, lossf

def get_escale_fscale(opts, i):
    n = opts["epochs"]

    es, fs = opts["loss_ene_frc_start"]
    ee, fe = opts["loss_ene_frc_end"]

    escale = ee*(i/n) + es*(1 - i/n)
    fscale = fe*(i/n) + fs*(1 - i/n)
    tot = escale + fscale
    escale = escale / tot
    fscale = fscale / tot
    return escale, fscale

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


def train_model(m, datasets, opts, device, logfile="out1.log"):
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
    xyzt, frct, enet, att, boxt, subselt = datasets[t_set].get_sample(t_idx)

    with open(logfile, "a") as wfile:
        wfile.write(str(opts) + "\n")

    msg = "Epoch  rmse_tf rmse_te rmse_e  rmse_f  mean(abs(pred_force))  mean(abs(force))  lr es fs"
    print(msg)
    with open(logfile, "a") as wfile:
        wfile.write(msg + "\n")

    for epoch in range(opts["epochs"]):
        optimizer.zero_grad()

        # Random batch selection
        dataset = np.random.choice(datasets, p = opts["sel_prob"])
        xyzi, frci, enei, at, box, subsel = dataset.get_random_sample()

        # Model prediction
        epred, fpred = m.calculate_ef(
            xyzi,
            at,
            box,
            subsel=subsel,
            attach_grad=True,
        )

        escale, fscale = get_escale_fscale(opts, epoch)
        rmse_e, rmse_f = loss_subsel(fpred, frci, epred, enei, subsel)
        rmse_e = rmse_e/frci.shape[0]
        loss = escale*rmse_e + fscale*rmse_f
        loss.backward()

        # Periodic logging and saving
        if epoch % opts["log_every"] == 0:
            etest, ftest = m.calculate_ef(
                xyzt,
                att,
                boxt,
                subsel=subselt,
            )
            rmse_te, rmse_tf = loss_subsel(ftest, frct, etest, enet, subselt)
            rmse_te = rmse_te/frct.shape[0]
            losst = escale * rmse_te + fscale * rmse_tf

            if subsel:
                mae_fpred = torch.mean(torch.abs(fpred[subsel])).item()
                mae_f = torch.mean(torch.abs(frci[subsel]))
            else:
                mae_fpred = torch.mean(torch.abs(fpred)).item()
                mae_f = torch.mean(torch.abs(frci))
            msg = (
                f"{epoch} {losst.item():.5f} {loss.item():.5f} "
                f"{rmse_te.item():.5f} {rmse_tf.item():.5f} "
                f"{rmse_e.item():.5f} {rmse_f.item():.5f} "
                f"{mae_fpred:.5f} "
                f"{mae_f:.5f} "
                f"{scheduler.get_last_lr()[0]:.3e} "
                f"{escale:.3f} {fscale:.3f}"
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
    with open("turtlennp.toml", "rb") as readfile:
        opts = tomli.load(readfile)
    print(opts)

    opts["dtype"] = dtype
    opts["at_map"] = {int(k):v for k,v in opts["at_map"].items()}

    device = torch.device(opts["device"])
    torch.set_num_threads(opts["num_threads"])
    torch.set_num_interop_threads(opts["num_threads"])

    datasets = []
    for fpath, dformat in zip(opts["datasets"], opts["dataset_formats"]):
        if dformat == "np":
            datasets.append(NumpyDataset(fpath, device))
        elif dformat == "h5":
            datasets.append(H5pyDataset(fpath, device))

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
