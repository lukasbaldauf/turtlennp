import tqdm
import torch
import numpy as np
#An example Python implementation for Welford's algorithm is given below
# For a new value new_value, compute the new count, new mean, the new M2.
# mean accumulates the mean of the entire dataset
# M2 aggregates the squared distance from the mean
# count aggregates the number of samples seen so far

def update(existing_aggregate, new_value):
	count = existing_aggregate[:,0]
	mean = existing_aggregate[:,1]
	M2 = existing_aggregate[:,2]
	count += 1
	delta = new_value - mean
	mean += delta / count
	delta2 = new_value - mean
	M2 += delta * delta2
	return np.c_[count, mean, M2]

# Retrieve the mean, variance and sample variance from an aggregate
def finalize(existing_aggregate):
    (count, mean, M2) = existing_aggregate
    if count < 2:
        return float("nan")
    else:
        (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
        # use variance and not sample_variance
        return mean, variance, sample_variance


# one aggregate[i,j] for type i and feature j
def normalize(m, xyz_arr, at, subsel, box_size, N = None):
    N = xyz_arr.shape[0] if N == None else N
    aggs = np.array([np.array([(0,0.0,0.0) for i in range(sum(m.sel) + 3*sum(m.a_sel))]) for j in range(3)])
    atnp = at[subsel]
    for i in tqdm.tqdm(range(N)):
        dv = m.calculate_ef(xyz_arr[i:i+1], at.unsqueeze(0), subsel, box_size, descriptors_only = True)[0].cpu().detach().numpy()
        for j,atj in enumerate(atnp):
            aggs[atj] = update(aggs[atj], dv[j])

    # mean, variance, sample_variance
    res = np.zeros((len(m.sel),aggs.shape[1],3))
    for ati in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[ati,j] = finalize(aggs[ati,j])
            res[ati,j, 1:] = np.sqrt(res[ati,j, 1:])
    return res
