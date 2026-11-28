"""h(r) error of finished Fu et al. MD runs (scripts/fane_md.py), as the authors' observable.ipynb
(MDsim 61ca2cf, MD17 section): h(r) is the density histogram of all interatomic distances,
500 bins on (0, 10] Å; the reference uses every frame of the original MD17 trajectory, a run only
its frames up to and including the first frame whose bond deviation exceeds 0.5 Å; error is
mean |h - h_ref| x 10. Bonds come from the run's first frame, as in the notebook.

    python fane_hr.py <molecule> <runs_root>    # runs_root/<variant>_seed<s>/rep*/atoms.traj

Writes <runs_root>/hr.json (per-run errors) and hr_curves.npz (reference and per-variant mean h(r)).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase.io import Trajectory
from ase.neighborlist import NeighborList, natural_cutoffs

from fane_md_smoke import distance_pbc

DATA = Path('/vol/data/forces_not_enough/md17')
XLIM, N_BINS, THRESHOLD = 10, 500, 0.5
WINDOWS_PS = ((0, 10), (10, 50), (50, 150), (150, 300))  # beyond Fu et al.: does h(r) drift with time?
FRAMES_PER_PS = 20  # fane_md.py saves every 100 steps of 0.5 fs
PAIR_MARGINS = (0.3, 0.5, 1.0)  # Å; strict check (ours, not Fu et al.): any atom pair outside its reference range
BINS = np.linspace(1e-6, XLIM, N_BINS + 1)
CELL = torch.FloatTensor([30., 30., 30.])


def get_hr(traj, chunk=20000):
    """observable.ipynb get_hr: traj (T, N, 3) -> density histogram of all pairwise distances
    (np.histogram density=True, accumulated over chunks of frames to bound memory)."""
    counts = sum(np.histogram(torch.cdist(c, c).flatten().numpy(), BINS)[0] for c in traj.split(chunk))
    return counts / (counts.sum() * np.diff(BINS))


def pair_range(gt_traj, chunk=20000):
    """Per-pair min and max distance over the reference trajectory, (N, N) each."""
    d = [torch.cdist(c, c) for c in gt_traj.split(chunk)]
    return torch.stack([x.min(0).values for x in d]).min(0).values, torch.stack([x.max(0).values for x in d]).max(0).values


def strict_failure_ps(traj, lo, hi, margin):
    """First time any pair distance leaves [ref min - margin, ref max + margin]; None if never.
    Catches distorted-but-bonded states (e.g. H folded onto a non-bonded C) the 0.5 Å bond rule misses."""
    out = [(((d < lo - margin) | (d > hi + margin)).flatten(1).any(1)) for d in
           (torch.cdist(c, c) for c in traj.split(2000))]
    bad = torch.cat(out).nonzero()
    return float(bad[0]) / FRAMES_PER_PS if len(bad) else None


def load_run(run_dir, gt_traj):
    """observable.ipynb load_run: h(r) over frames [0, collapse_pt)."""
    frames = Trajectory(str(run_dir / 'atoms.traj'))
    neighbors = NeighborList(natural_cutoffs(frames[0]), self_interaction=False)
    neighbors.update(frames[0])
    bonds = torch.tensor(np.asarray(neighbors.get_connectivity_matrix().todense().nonzero()))
    traj = torch.from_numpy(np.stack([a.positions for a in frames])).float()
    # Runs retried before fane_md.py cleared stale files have earlier attempts appended in front:
    # keep the final attempt, whose length result.json records.
    n = round(json.loads((run_dir / 'result.json').read_text())['simulated_ps'] * FRAMES_PER_PS) + 1
    assert len(traj) >= n, (run_dir, len(traj), n)
    trimmed, traj = len(traj) - n, traj[-n:]
    means = distance_pbc(gt_traj[:, bonds[0]], gt_traj[:, bonds[1]], CELL).mean(dim=0)
    dev = (distance_pbc(traj[:, bonds[0]], traj[:, bonds[1]], CELL) - means).abs().max(dim=1).values
    over = (dev > THRESHOLD).nonzero()
    collapse_pt = int(over[0]) + 1 if len(over) else len(traj)
    return traj, collapse_pt, trimmed


def main(molecule, root):
    gt_traj = torch.FloatTensor(np.load(DATA / f'md17_{molecule}.npz')['R'])
    hist_gt = get_hr(gt_traj)
    lo, hi = pair_range(gt_traj)
    runs, curves = {}, {}
    for traj in sorted(root.glob('*_seed*/rep*/atoms.traj')):
        full, collapse_pt, trimmed = load_run(traj.parent, gt_traj)
        stable = full[:collapse_pt]
        hist = get_hr(stable)
        name = f'{traj.parent.parent.name}/{traj.parent.name}'
        runs[name] = {'trimmed_frames': trimmed, 'hr_mae': float(np.abs(hist - hist_gt).mean() * XLIM), 'frames': collapse_pt,
                      'window_hr_mae': {f'{a}-{b}ps': float(np.abs(get_hr(stable[a * FRAMES_PER_PS:b * FRAMES_PER_PS]) - hist_gt).mean() * XLIM)
                                        for a, b in WINDOWS_PS if collapse_pt >= b * FRAMES_PER_PS},
                      'bond_failure_ps': (collapse_pt - 1) / FRAMES_PER_PS if collapse_pt < len(full) else None,
                      'strict_failure_ps': {str(m): strict_failure_ps(full, lo, hi, m) for m in PAIR_MARGINS}}
        curves.setdefault(traj.parent.parent.name.rsplit('_seed', 1)[0], []).append(hist)
        print(name, runs[name], flush=True)
    (root / 'hr.json').write_text(json.dumps(runs, indent=1) + '\n')
    np.savez(root / 'hr_curves.npz', r=BINS[1:], reference=hist_gt,
             **{k: np.mean(v, axis=0) for k, v in curves.items()})


if __name__ == '__main__':
    main(sys.argv[1], Path(sys.argv[2]))
