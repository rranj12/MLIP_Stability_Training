"""Bond list and non-bonded distance floors, computed from training frames.

A pair (i, j) is bonded if its mean training distance is below
`scale` x (sum of covalent radii); b_ij is that mean distance (explainer Sec. 5.2).
"""

import numpy as np

# Cordero et al. (2008) covalent radii in Å, the values ASE uses.
COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66}


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """(n_frames, N, 3) -> (n_frames, N, N) distances."""
    diff = coords[:, :, None, :] - coords[:, None, :, :]
    return np.linalg.norm(diff, axis=-1)


def frame_chunks(coords, budget=2 ** 24):
    """Slices of frames holding at most ~`budget` atom pairs each, so pairwise_distances
    needs the same memory for a 370-atom molecule (122 frames per slice) as for aspirin."""
    step = max(1, budget // coords.shape[1] ** 2)
    return (coords[s:s + step] for s in range(0, len(coords), step))


def bond_list(z, coords, scale=1.2):
    """Return bonds (P, 2) int array and equilibrium lengths b0 (P,) in Å."""
    z, coords = np.asarray(z), np.asarray(coords)
    mean_d = sum(pairwise_distances(c).sum(axis=0) for c in frame_chunks(coords)) / len(coords)
    radii = np.array([COVALENT_RADII[int(zi)] for zi in z])
    cutoff = scale * (radii[:, None] + radii[None, :])
    i, j = np.triu_indices(len(z), k=1)
    mask = mean_d[i, j] < cutoff[i, j]
    bonds = np.stack([i[mask], j[mask]], axis=1)
    return bonds, mean_d[bonds[:, 0], bonds[:, 1]]


def nonbonded_floor(coords, bonds, tol=0.5, min_floor=0.5):
    """Non-bonded pairs and a collision floor per pair.

    The floor is the smallest distance seen for the pair in training minus
    `tol`, clipped at `min_floor`. The bond-deviation criterion only watches
    bonded pairs, so this catches non-bonded collisions it would miss.
    """
    coords = np.asarray(coords)
    d_min = np.minimum.reduce([pairwise_distances(c).min(axis=0) for c in frame_chunks(coords)])
    n = coords.shape[1]
    i, j = np.triu_indices(n, k=1)
    bonded = {(int(a), int(b)) for a, b in bonds}
    keep = np.array([(a, b) not in bonded for a, b in zip(i, j)])
    pairs = np.stack([i[keep], j[keep]], axis=1)
    floor = np.maximum(d_min[pairs[:, 0], pairs[:, 1]] - tol, min_floor)
    return pairs, floor
