"""Structural and thermodynamic observables, computed only on stable frames."""

import numpy as np

from mlipstab.data.bonds import pairwise_distances


def distance_histogram(coords, bins):
    """h(r): normalized histogram of all interatomic distances. coords (n, N, 3)."""
    d = pairwise_distances(np.asarray(coords))
    i, j = np.triu_indices(d.shape[1], k=1)
    hist, _ = np.histogram(d[:, i, j].ravel(), bins=bins, density=True)
    return hist


def dihedral(coords, i, j, k, l):
    """Dihedral angle i-j-k-l in degrees, in (-180, 180]. coords (n, N, 3)."""
    c = np.asarray(coords)
    b0 = c[:, i] - c[:, j]
    b1 = c[:, k] - c[:, j]
    b2 = c[:, l] - c[:, k]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    x = (v * w).sum(-1)
    y = (np.cross(b1, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))


def free_energy_profile(angles, kT, bins=72):
    """F(phi) = -kT ln p(phi), shifted so min F = 0. Empty bins -> nan."""
    p, edges = np.histogram(angles, bins=bins, range=(-180, 180), density=True)
    with np.errstate(divide="ignore"):
        f = -kT * np.log(p)
    f[~np.isfinite(f)] = np.nan
    return 0.5 * (edges[1:] + edges[:-1]), f - np.nanmin(f)
