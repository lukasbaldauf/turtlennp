import pytest
import torch
import numpy as np

from turtlennp.model import (
Model,
get_dm_dist_b,
get_dm_dist_b_subsel,
get_local_frame_vectors,
)

def get_model0():
    """A toy system for tests with 1 linear layer + bias."""
    xyz = torch.tensor([[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        ]])

    typemap = {'Ar':0}
    at = torch.zeros((1,xyz.shape[1]))
    cutoff = 10.0
    box_size = 100.0
    sel = [len(at[0])-1]
    a_sel = [len(at[0])-1]
    model = Model(typemap, sel, a_sel = a_sel, cutoff = cutoff, layer_sizes = [1])
    return model, xyz, cutoff, box_size, at

def test_descriptors_model0():
    """Test that we get the expected descriptors from a toy system."""
    model, xyz, cutoff, box_size, at = get_model0()
    dm, dist = get_dm_dist_b(xyz, box_size)
    dv = get_local_frame_vectors(xyz, dm, dist, model.cutoff, model.sel, model.a_sel, at)[0]
    dv_model0 = np.array([1.0, 1.0, 1/2**(1/2), # distances to neighbors
                          1.0, 0.0, 0.0, # nearest neighbor x', y', z' coords
                          0.0, 1.0, 0.0, # second neighbor coords
                          0.5, 0.5, 0.0, # third neighbor, 1/2**(1/2) * 1/2**(1/2)
                          ])
    # model0 .xyz is symmetric so all atoms should have same local environment
    for i in range(dv.shape[0]):
        assert np.allclose(dv[i], dv_model0)

def test_calculate_ef_model0():
    """Test that the energy increases 1/distance."""
    model, xyz, cutoff, box_size, at = get_model0()
    for i,p in enumerate(model.networks[0].parameters()):
        p.data = p.data*0.0
        if i == 0:
            p.data[0,2] = 1.0
    N = 20
    dists =  torch.linspace(2.0, 5.0, N)
    xyzi = xyz.expand(N, -1, -1).detach().clone()
    xyzi[:, 3, :2] = dists.unsqueeze(1).expand(N, 2)
    xyzi.requires_grad = True
    e,f = model.calculate_ef(xyzi, at.expand(N, -1), box_size, subsel = [0, 3])
    assert np.allclose(e.detach().numpy(),2/(2**(1/2)*dists))
