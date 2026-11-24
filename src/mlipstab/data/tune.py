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


def write_tune(base, idx, z, R, E, F):
    """Write <base>/tune/nequip_npz.npz in the other splits' format (no LMDB: never trained on).
    Written to a temp file and renamed, so concurrent readers never see a partial file."""
    d = Path(base) / "tune"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "nequip_npz.tmp.npz"
    np.savez(tmp, z=z, E=E[idx], F=F[idx], R=R[idx])
    tmp.rename(d / "nequip_npz.npz")
