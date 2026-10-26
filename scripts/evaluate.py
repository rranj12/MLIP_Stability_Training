"""Evaluate any checkpoint (baseline, StABlE, CRMSE, ours) under one protocol.

    python scripts/evaluate.py --config configs/eval/rmd17_500K.yml \
        --molecule aspirin --model schnet --checkpoint <path> --tag margin_seed0

Outputs under results/eval/<tag>/:
    stability.npz   per-replica first-failure time + censoring flag
    traj.npy        positions every save_every_fs, stable frames only
    metrics.json    median tau, fraction stable at 100/300 ps, E/F MAE,
                    h(r) error, dihedral F(phi) error, NVE drift
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/StABlE-Training"))

from mlipstab.config import load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--molecule", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--temperature", type=float, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)

    # TODO 1: accuracy -> energy/force MAE on the 1,000 test frames.
    # TODO 2: batched Langevin MD from n_replicas test frames (reuse the simulate glue
    #         from scripts/train_stability.py with gradients off, no early stop).
    # TODO 3: per saved frame, mlipstab.detect.unstable(...) with bond_threshold and the
    #         non-bonded floors from mlipstab.data.bonds.nonbonded_floor(train frames).
    # TODO 4: mlipstab.eval.stability.first_failure_time -> stability.npz.
    # TODO 5: observables on stable frames only: eval.observables.distance_histogram,
    #         dihedral + free_energy_profile, compared against held-out DFT frames.
    raise NotImplementedError


if __name__ == "__main__":
    main()
