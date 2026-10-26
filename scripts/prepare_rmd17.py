"""Download rMD17 and write it in the directory layout StABlE/MDsim expect.

    data/processed/md17/<mol>/1k/{train,val,test}/{data.lmdb,nequip_npz.npz}
    data/processed/md17/<mol>/1k/metadata.npy
    data/processed/md17/<mol>/10k -> reference split used by StABlE's h(r) target
    data/processed/contiguous-md17/<mol>/10k/val/nequip_npz.npz   (see note)

The dataset is still called "md17" on disk because StABlE's code branches on
config["name"] == "md17"; the labels are rMD17. Energies/forces are converted
to eV and eV/Å exactly as StABlE's data/md17.py does.

--reference same       10k/ is a symlink to 1k/  -> StABlE's h(r) uses our 950 frames (main baseline)
--reference published  10k/ holds 9,500 extra non-test frames, like StABlE's original setup

Note: StABlE also reads contiguous-md17/<mol>/10k/val for a VACF that it only
logs. rMD17 has no contiguous frames, so we write frames sorted by their
original MD17 index. Treat StABlE's logged VACF loss as meaningless on rMD17.

Usage (needs the full conda env, i.e. MDsim from the StABlE submodule):
    python scripts/prepare_rmd17.py --molecule aspirin --split 1 --reference same
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/StABlE-Training/data"))  # arrays_to_graphs

from mlipstab.data.rmd17 import download, make_splits, reference_pool  # noqa: E402
from mlipstab.units import EV_TO_KCAL_MOL  # noqa: E402


def write_split(path, idx, z, R, E, F):
    import lmdb
    from arrays_to_graphs import AtomsToGraphs

    a2g = AtomsToGraphs(max_neigh=1000, radius=6, r_energy=True, r_forces=True,
                        r_distances=False, r_edges=False, device="cpu")
    path.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(str(path / "data.lmdb"), map_size=1099511627776 * 2,
                   subdir=False, meminit=False, map_async=True)
    lengths, angles = np.ones(3)[None] * 30.0, np.ones(3)[None] * 90.0
    natoms = np.array([R.shape[1]], dtype=np.int64)
    for i, k in enumerate(idx):
        data = a2g.convert(natoms, R[k], z, lengths, angles, E[k], F[k])
        data.sid, data.fid = 0, int(k)
        with db.begin(write=True) as txn:
            txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
    with db.begin(write=True) as txn:
        # StABlE/MDsim store the last index under "length" (their convention, kept as-is)
        txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
    db.sync()
    db.close()
    np.savez(path / "nequip_npz", z=z, E=E[idx], F=F[idx], R=R[idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--molecule", required=True)
    p.add_argument("--split", type=int, default=1, help="official rMD17 split 1-5")
    p.add_argument("--reference", choices=["same", "published"], default="same")
    p.add_argument("--raw_dir", default=str(ROOT / "data/raw/rmd17"))
    p.add_argument("--out_dir", default=str(ROOT / "data/processed"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    npz, train_idx, test_idx = download(args.molecule, args.raw_dir, args.split)
    raw = np.load(npz)
    z = raw["nuclear_charges"].astype(np.int64)
    R = raw["coords"]
    E = raw["energies"] / EV_TO_KCAL_MOL
    F = raw["forces"] / EV_TO_KCAL_MOL
    splits = make_splits(train_idx, test_idx, seed=args.seed)

    base = Path(args.out_dir) / "md17" / args.molecule
    for split in ("train", "val", "test"):
        write_split(base / "1k" / split, splits[split], z, R, E, F)
    # Nested low-data subsets: size dirs "n100", "n250" share 1k's val/test.
    for name in (k for k in splits if k.startswith("train_")):
        size_dir = base / name.replace("train_", "n")
        write_split(size_dir / "train", splits[name], z, R, E, F)
        for split in ("val", "test"):
            if not (size_dir / split).exists():
                os.symlink(Path("..") / "1k" / split, size_dir / split)
    np.save(base / "1k" / "metadata", {
        "e_mean": E[splits["train"]].mean(), "e_std": E[splits["train"]].std(),
        "f_mean": F[splits["train"]].mean(), "f_std": F[splits["train"]].std(),
    })
    np.save(base / "1k" / "splits", splits)

    ref = base / "10k"
    if args.reference == "same":
        if not ref.exists():
            os.symlink("1k", ref)
    else:
        pool = reference_pool(len(R), test_idx, 9500, seed=args.seed)
        write_split(ref / "train", pool, z, R, E, F)

    order = np.argsort(raw["old_indices"][splits["val"]])
    contig = Path(args.out_dir) / "contiguous-md17" / args.molecule / "10k" / "val"
    contig.mkdir(parents=True, exist_ok=True)
    v = splits["val"][order]
    np.savez(contig / "nequip_npz", z=z, E=E[v], F=F[v], R=R[v])
    print(f"wrote {base}")


if __name__ == "__main__":
    main()
