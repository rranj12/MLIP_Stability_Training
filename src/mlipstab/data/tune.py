"""`tune` split: labeled frames outside train/val/test, used only to select hyperparameters.

Selection simulations start from these frames and measure accuracy on them, so the test
frames are read once, for final numbers. Written by the prep scripts; see
scripts/sweep_lambda.py.
"""

from pathlib import Path

import numpy as np


def tune_pool(n_total, used_idx, n_frames, seed=0):
    """Sorted indices of up to n_frames frames in [0, n_total) not in used_idx."""
    candidates = np.setdiff1d(np.arange(n_total), np.asarray(used_idx))
    n = min(n_frames, len(candidates))
    return np.sort(np.random.default_rng(seed + 1).choice(candidates, size=n, replace=False))


def used_by_positions(raw_R, split_Rs, tol=1e-4):
    """Indices into raw_R (T, N, 3) of every frame in split_Rs, matched on coordinates within tol Å.
    For datasets whose prepared splits do not record raw-file indices (Fu et al.'s archive)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(raw_R[:, 0])  # candidates by the first atom, then confirm on all atoms
    used = []
    for r in np.concatenate(split_Rs):
        match = [i for i in tree.query_ball_point(r[0], tol) if np.abs(raw_R[i] - r).max() < tol]
        if len(match) != 1:
            raise ValueError(f"split frame matched {len(match)} raw frames")
        used.append(match[0])
    return np.unique(used)


def write_tune(base, idx, z, R, E, F):
    """Write <base>/tune/nequip_npz.npz in the other splits' format (no LMDB: never trained on).
    Written to a temp file and renamed, so concurrent readers never see a partial file."""
    d = Path(base) / "tune"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "nequip_npz.tmp.npz"
    np.savez(tmp, z=z, E=E[idx], F=F[idx], R=R[idx])
    tmp.rename(d / "nequip_npz.npz")
