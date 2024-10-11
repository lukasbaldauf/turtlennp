import torch
from torch import nn


def get_dm_dist_b(xyz, box_size):
    """
    Calculate the distance matrix and the distance tensor for batched atom
    coordinates.
    Args:
        xyz: Tensor of shape (B, N, 3) containing the coordinates of the atoms.
        box_size: Float or Tensor containing the length of the box.
    Returns:
        dm: Tensor of shape (B, N, N, 3) containing the distance vectors.
        dist: Tensor of shape (B, N, N) containing the distance magnitudes.
    """
    B, N, D = xyz.size()
    x = xyz.unsqueeze(2).expand(B, N, N, D)
    y = xyz.unsqueeze(1).expand(B, N, N, D)
    dm = x - y + 1e-16  # avoid NaN derivatives
    dm = dm - box_size * torch.round(dm / box_size)
    dist = torch.sqrt(torch.pow(dm, 2).sum(-1))
    return dm, dist


def get_dm_dist_b_subsel(xyz, box_size, subsel):
    """
    Same as get_dm_dist_b but with subsel.
    """
    B, N, D = xyz.size()
    N_ = len(subsel)
    x = xyz[:, subsel, :].unsqueeze(2).expand(B, N_, N, D)
    y = xyz.unsqueeze(1).expand(B, N_, N, D)
    dm = x - y + 1e-16  # avoid NaN derivatives
    dm = dm - box_size * torch.round(dm / box_size)
    dist = torch.sqrt(torch.pow(dm, 2).sum(-1))
    return dm, dist


def get_local_frame_vectors(xyz, dm, dist, cutoff, sel, a_sel, atomtypes):
    """Get the local coordinate descriptor defined by the DeePMD model."""
    epsilon = 1e-16  # why not 1e-16?
    B, N_, N = dist.size()
    descriptor_dists = torch.zeros(
        (B, N_, sum(sel) + 3 * sum(a_sel)), device=xyz.device
    )
    a_xyz = torch.zeros(
        (B, N_, sum(a_sel), 3), device=xyz.device, dtype=xyz.dtype
    )
    a_dist = torch.zeros(B, N_, sum(a_sel), device=xyz.device, dtype=xyz.dtype)
    # Construct descriptor vectors in a vectorized manner
    beg = 0
    for j, selj in enumerate(sel):
        # Get masks for atoms of type j
        mask = (atomtypes == j).unsqueeze(1).expand(B, N_, N)
        # have to clone for some reason when using fill_diagonal_
        # probably mask is a view, thats why
        mask = mask.clone()
        # Exclude self-interaction
        for b in range(B):
            mask[b].fill_diagonal_(False)
        dists = torch.where(mask, dist, torch.full_like(dist, float("inf")))
        if cutoff not in [None, torch.inf]:
            dists[dists > cutoff] = float("inf")
        sorted_dists, sorted_idx = torch.sort(dists, dim=2)
        # get the nearest neighbor indices. The two nearest are
        # used to construct rotation matrix
        if a_sel and j == 0:
            local_frame_dist = sorted_dists[:, :, :2].clone()
            local_frame_idx = sorted_idx[:, :, :2].clone()
        if a_sel and j != 0:
            mask_a = local_frame_dist > sorted_dists[:, :, :2]
            if torch.any(mask_a):
                local_frame_dist[mask_a] = sorted_dists[:, :, :2][mask_a]
                local_frame_idx[mask_a] = sorted_idx[:, :, :2][mask_a]
        nearest_dists = sorted_dists[:, :, :selj]
        distance_block = 1 / (nearest_dists + epsilon)
        bega = sum(a_sel[:j])
        descriptor_dists[:, :, beg : beg + distance_block.shape[2]] = (
            distance_block
        )
        beg += sel[j]  # distance_block.shape[2]
        if a_sel:
            a_idx = (
                sorted_idx[:, :, : a_sel[j]]
                .unsqueeze(-1)
                .expand(-1, -1, -1, 3)
            )
            a_xyz[:, :, bega : bega + a_sel[j]] = torch.gather(dm, 2, a_idx)
            a_dist[:, :, bega : bega + a_sel[j]] = sorted_dists[
                :, :, : a_sel[j]
            ]
    # since we sort the distances according to closest atoms, we also
    # have to sort the s(x),s(y),s(z) according to closest atoms
    # does the order of scaling matter here?
    # We ave to sort according to atomic indices
    # construct angular descriptors
    if a_sel:
        l_idx = local_frame_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        l_dist = local_frame_dist.unsqueeze(-1)
        inf_vals = torch.where(l_dist == torch.inf)
        if len(inf_vals[1]) > 0:
            raise NotImplementedError(
                f"Atoms {inf_vals[1]} has less than 2 neighbors!",
                "Can't construct local frame, but this might not be",
                "neccessary if this atom is not part of the subselection.",
                "Increase cutoff or implement it so that it works.",
            )
        l_xyz = torch.gather(dm, 2, l_idx)
        l_xyz = l_xyz / (l_dist + epsilon)
        a_dist = a_dist.unsqueeze(-1)
        a_xyz = a_xyz / (a_dist + epsilon)
        # find rotation matrix
        r0 = l_xyz[:, :, 0, :]  # Shape: (Nf, Na, 3)
        r1 = l_xyz[:, :, 1, :]  # Shape: (Nf, Na, 3)
        dot_product = torch.sum(r0 * r1, dim=-1, keepdim=True)
        v2 = r1 - dot_product * r0
        v2 = v2 / v2.norm(dim=-1, keepdim=True)
        v3 = torch.cross(r0, r1, dim=-1)
        v3 = v3 / v3.norm(dim=-1, keepdim=True)
        A = torch.stack([r0, v2, v3], dim=-2)
        # rotate the coordinates
        a_xyz = torch.einsum("...ij,...kj->...ki", A, a_xyz)
        # Then scale by 1/rij
        a_xyz = a_xyz / (a_dist + epsilon)
        descriptor_dists[:, :, -sum(a_sel) * 3 :] = a_xyz.reshape(
            *descriptor_dists.shape[:2], sum(a_sel) * 3
        )
    return descriptor_dists


class Model(nn.Module):
    def __init__(
        self,
        typemap: dict[str, int],
        sel: list[int],
        a_sel: list = [],
        cutoff: torch.float = torch.inf,
        layer_sizes: list[int] = [10, 1],
        device=torch.device("cpu"),
        dynamic_subsel: bool = False,
    ):
        """
        TODO: how to treat transfereability? That is, can subsel,
            dynamic subsel change after training? Usually it wouldn't right?

        TODO: interpretation of atomic energies?

        TODO: where to select distance_function? It depends on
        dynamic_subsel and subsel.
            * if dynamic_subsel use dm_dist_dynamic_subsel, shape (1,N_,N)
              We could also use batches by using full distance_matrix
            * if subsel and not dynamic_subsel, shape (B,N_,N)
            * if not subsel, shape (B,N,N)

        TODO: sel, if its greater than the number of neighbors, we allways
            get one value that is zero? Also, a_sel, second and third value
            are always zero. Reduce the number of these zero_values, as they
            are redundant.

        subsel can be a dynamic region by using the dynamic_subsel keyword.

        For example, for a water-solute qm/mm system where the solute
        is treated with QM and water with MM, the water molecules close
        to the QM solute experience a strong QM force. So we may select
        as subsel all atoms that are within e.g. 4 Å of the solute.

        If dynamic_subsel == True, we calculate the whole distance matrix
        of size (N,N), where N is the number of atoms.
        Else, we a_sel closest atoms to include their 3*a_sel normalized
        local atomic coordscalculate the subsel matrix of shape
        (len(subsel), N).

        If subsel and cutoff, we could created a modified xyz array such that
        coordinates of atoms beyond the cutoff are removed.
        TODO: is this neccessary?

        a_sel selects the closest a_sel[i] atoms to include their 3*a_sel
        normalized local atomic coords

        Arguments
            typemap: map atom type from elemt to integer
            sel: number radial neighbors to construct for each atom type
            a_sel: number of angular neighbors to construct for atom type
            cutoff: atoms beyond cutoff don't see each other
            layer_zies: neural network architecture
            device: 'cuda' or 'cpu'
            dynamic_subsel: is subsel is a dynamic region or not

        """
        super().__init__()
        self.sel = sel
        self.a_sel = a_sel
        self.typemap = typemap
        self.cutoff = cutoff
        self.dynamic_subsel = dynamic_subsel
        self.norm = torch.zeros(
            (len(self.sel), sum(self.sel) + 3 * sum(self.a_sel), 2),
            device=device,
        )
        self.norm[:, :, 1] = 1.0

        assert len(sel) == len(
            typemap
        ), "Length of sel and typemap should be equal"

        # each atom type gets it's own subnet
        self.networks = []

        for i in range(len(sel)):
            # Build the network based on the layer_sizes list
            layers = []
            input_size = sum(self.sel) + 3 * sum(self.a_sel)

            for output_size in layer_sizes[:-1]:
                layers.append(nn.Linear(input_size, output_size))
                layers.append(nn.Tanh())
                input_size = output_size

            # Add the last linear layer without ReLU
            layers.append(nn.Linear(input_size, layer_sizes[-1]))

            network = nn.Sequential(*layers)
            network = network.to(device)
            self.networks.append(network)

    def calculate_ef(
        self,
        xyz: torch.tensor,
        atomtypes: torch.tensor,
        box_size: torch.tensor,
        subsel: list = [],
        attach_grad: bool = False,
        descriptors_only: bool = False,
    ):
        """Calculate the energy and forces.

        Arguments:
            xyz: coordinate array of shape (B, N, 3)
            atomtypes: atomtypes of shape (B, N)
            box_size: box size of shape (B,3)
            subsel: the subselection, useful for training on QM/MM data
            attach_grad: If we attach gradients to forces, used when training

        """
        # calculate the distances and distance_matrices
        if self.dynamic_subsel or len(subsel) == 0:
            dm, dist = get_dm_dist_b(xyz, box_size)

        else:
            dm, dist = get_dm_dist_b_subsel(xyz, box_size, subsel)

        # calculate descriptors
        descriptors = get_local_frame_vectors(
            xyz, dm, dist, self.cutoff, self.sel, self.a_sel, atomtypes
        )
        if descriptors_only:
            return descriptors

        # standardize the inputs
        for j in range(len(self.sel)):
            descriptors[atomtypes[:, subsel] == j] = (
                descriptors[atomtypes[:, subsel] == j] - self.norm[j, :, 0]
            ) / self.norm[j, :, 1]

        # Compute energy in a vectorized manner
        energies = torch.zeros(dist.shape[:2], device=xyz.device)
        for t, network in enumerate(self.networks):
            mask = atomtypes[:, subsel] == t
            if len(mask) > 0:
                energies[mask] = network(descriptors[mask]).squeeze()
        energy = torch.sum(energies, axis=1)
        # Compute forces
        if attach_grad:
            frci = -torch.autograd.grad(
                energy,
                xyz,
                create_graph=True,
                retain_graph=True,
                grad_outputs=torch.ones_like(energy),
            )[0]
        else:
            frci = -torch.autograd.grad(
                energy, xyz, grad_outputs=torch.ones_like(energy)
            )[0]

        return energy, frci
