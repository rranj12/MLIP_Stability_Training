"""Choice of anchor x0 for each failure x_fail.

    rollout         the same replica a few hundred steps before the failure (default).
                    Differs from x_fail in many unrelated motions, so the true energy
                    ordering is uncertain by ~ thermal fluctuations (~5 kcal/mol for aspirin).
    counterfactual  x_fail with only the most-deviated bond moved back to b0. The pair
                    differs in one coordinate whose energy ordering is known.
    labeled         the training frame closest to the rollout anchor. Its energy and
                    forces are pinned by L_EF, so the margin cannot be met by lowering it.
"""

import torch

from mlipstab.detect import worst_bond


def counterfactual_anchor(x_fail, bonds, b0):
    """Restore each configuration's most-deviated bond to its equilibrium length.

    Both atoms move symmetrically along the bond. x_fail (B, N, 3) -> (B, N, 3).
    """
    k = worst_bond(x_fail, bonds, b0)
    rows = torch.arange(x_fail.shape[0], device=x_fail.device)
    i, j = bonds[k, 0], bonds[k, 1]
    rij = x_fail[rows, j] - x_fail[rows, i]
    r = rij.norm(dim=-1, keepdim=True)
    shift = 0.5 * (r - b0[k][:, None]) * rij / r
    x = x_fail.clone()
    x[rows, i] += shift
    x[rows, j] -= shift
    return x


def distance_matrix(x):
    return (x[..., :, None, :] - x[..., None, :, :]).norm(dim=-1)


def nearest_frame(x, frames):
    """Nearest frame by distance-matrix L2 (rotation/translation invariant).

    x (B, N, 3), frames (M, N, 3) -> (B, N, 3) rows of `frames`.
    """
    dx = distance_matrix(x).flatten(1)
    df = distance_matrix(frames).flatten(1)
    return frames[torch.cdist(dx, df).argmin(dim=1)]


def make_anchor_fn(mode, bonds=None, b0=None, train_frames=None):
    """Returns anchor_fn(x_fail, x_rollout_anchor) -> x_anchor."""
    if mode == "rollout":
        return lambda x_fail, x_anchor: x_anchor
    if mode == "counterfactual":
        return lambda x_fail, x_anchor: counterfactual_anchor(x_fail, bonds, b0)
    if mode == "labeled":
        return lambda x_fail, x_anchor: nearest_frame(x_anchor, train_frames)
    raise ValueError(f"unknown anchor mode {mode!r}")
