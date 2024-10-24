"""Welford's algorithm for calculating mean and stddev."""

import numpy as np
import tqdm


def update(existing_aggregate, new_value):
    """Update an aggregate with a new value."""
    count = existing_aggregate[:, 0]
    mean = existing_aggregate[:, 1]
    M2 = existing_aggregate[:, 2]
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return np.c_[count, mean, M2]


def finalize(existing_aggregate):
    """Retrieve the mean, variance and sample variance from an aggregate."""
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sample_variance) = (
            mean,
            M2 / count,
            M2 / (count - 1),
        )
        # use variance and not sample_variance
        return mean, variance, sample_variance


def normalize(m, datasets, N=[100], mode=["linear"]):
    """Perform the normalization.

    We use one aggregate[i,j] for each atom type i and feature j
    """
    aggs = np.array(
        [
            np.array(
                [(0, 0.0, 0.0) for i in range(sum(m.sel) + 3 * sum(m.a_sel))]
            )
            for j in range(len(m.sel))
        ]
    )
    for i, dataset in enumerate(datasets):
        for j in tqdm.tqdm(range(N[i])):
            if mode[i] == "linear":
                xyz, _, _, at, box, ss = dataset.get_sample(j)
            elif mode[i] == "random":
                xyz, _, _, at, box, ss = dataset.get_random_sample()
            else:
                raise ValueError(f"No such normalizeation mode: '{mode[i]}'")
            dv = m.calculate_ef(xyz, at, box, subsel=ss, descriptors_only=True)
            dv = dv.cpu().detach().numpy()
            if ss:
                atnp = at[ss]
            else:
                atnp = at
            for k, atk in enumerate(atnp):
                atk = m.typemap[atk.item()]
                aggs[atk] = update(aggs[atk], dv[k])

    # mean, variance, sample_variance
    res = np.zeros((len(m.sel), aggs.shape[1], 3))
    for ati in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[ati, j] = finalize(aggs[ati, j])
            res[ati, j, 1:] = np.sqrt(res[ati, j, 1:])
    return res
