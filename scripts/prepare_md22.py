"""Download an MD22 molecule and write it in the layout prepare_rmd17.py uses.

    data/processed/md22/<mol>/<size>/{train,val,test}/{data.lmdb,nequip_npz.npz}
    data/processed/md22/<mol>/<size>/metadata.npy

MD22 has no official splits. Frames are drawn at random (seeded): n_train for training
(default 950, as in our rMD17 "1k" split, so molecules of different size are compared at
the same training-set size), 50 for validation and 1,000 for testing. Labels are converted
to eV and eV/Å.

The dummy cell is 100 Å, not the 30 Å StABlE writes. SchNet and GemNet-T build their graph
with MDsim's periodic neighbor search, so no periodic image may ever come within the
cutoff; 100 Å rules that out for every MD22 molecule. Set data.cell to match in the configs.

StABlE's released Ac-Ala3-NHMe checkpoints were trained on its own split. To use them,
prepare that split with its script instead (run from the repository root):
    python third_party/StABlE-Training/data/md22.py --molecule ac_Ala3_NHMe \
        --size 25percent --data_path data/raw/md22 --db_path data/processed/md22

Usage (needs the full conda env, i.e. MDsim from the StABlE submodule):
    python scripts/prepare_md22.py --molecule AT_AT_CG_CG
"""

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare_rmd17 import write_split  # noqa: E402
from mlipstab.units import EV_TO_KCAL_MOL  # noqa: E402

URL = "https://www.quantum-machine.org/gdml/data/npz/"

# StABlE's molecule names (its NequIP configs are named after them) -> file on the server.
# StABlE's md22.py asks for md22_double-walled_nanotube.npz, which the server does not have.
FILES = {
    "ac_Ala3_NHMe": "md22_Ac-Ala3-NHMe.npz",            # 42 atoms
    "DHA": "md22_DHA.npz",                              # 56
    "AT_AT": "md22_AT-AT.npz",                          # 60
    "stachyose": "md22_stachyose.npz",                  # 87
    "AT_AT_CG_CG": "md22_AT-AT-CG-CG.npz",              # 118
    "buckyball_catcher": "md22_buckyball-catcher.npz",  # 148
    "double_walled_nanotube": "md22_dw_nanotube.npz",   # 370
}
CELL = 100.0  # Å
N_VAL, N_TEST = 50, 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", required=True, choices=sorted(FILES))
    p.add_argument("--n_train", type=int, default=950)
    p.add_argument("--size", default="1k", help="name of the output directory (data.size in the configs)")
    p.add_argument("--raw_dir", default=str(ROOT / "data/raw/md22"))
    p.add_argument("--out_dir", default=str(ROOT / "data/processed"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    npz = Path(args.raw_dir) / FILES[args.molecule]
    if not npz.exists():
        npz.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL + npz.name, npz)
    raw = np.load(npz)
    z = raw["z"].astype(np.int64)
    R = raw["R"]
    E = raw["E"].reshape(-1) / EV_TO_KCAL_MOL
    F = raw["F"] / EV_TO_KCAL_MOL

    counts = (args.n_train, N_VAL, N_TEST)
    if sum(counts) > len(R):
        raise ValueError(f"{args.molecule} has {len(R)} frames, fewer than the {sum(counts)} requested")
    perm = np.random.default_rng(args.seed).permutation(len(R))[:sum(counts)]
    train, val, test = np.split(perm, np.cumsum(counts)[:2])
    splits = {"train": train, "val": val, "test": test}

    base = Path(args.out_dir) / "md22" / args.molecule / args.size
    for name, idx in splits.items():
        write_split(base / name, idx, z, R, E, F, cell=CELL)
    np.save(base / "metadata", {
        "e_mean": E[train].mean(), "e_std": E[train].std(),
        "f_mean": F[train].mean(), "f_std": F[train].std(),
    })
    np.save(base / "splits", splits)
    print(f"wrote {base}")


if __name__ == "__main__":
    main()
