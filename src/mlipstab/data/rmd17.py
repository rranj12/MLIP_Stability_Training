"""rMD17 download and splits (Christensen & von Lilienfeld, figshare 12672038).

Files: rmd17_<mol>.npz with keys nuclear_charges, coords (Å), energies
(kcal/mol), forces (kcal/mol/Å), old_indices (index in original MD17), ...
Five official 1,000-frame train/test index files; the authors say to train on
at most 1,000 frames because frames come from one correlated MD trajectory.
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

FIGSHARE = "https://ndownloader.figshare.com/files/"

# molecule -> (figshare file id, md5), from api.figshare.com/v2/articles/12672038
MOLECULES = {
    "aspirin": ("62265757", "17fd6fb69066888613f7e16b358a7553"),
    "azobenzene": ("62265754", "be79df918468eb3579aa73a0becf7390"),
    "benzene": ("62265739", "18c9242bc90fbf28215f6dd81e650f16"),
    "ethanol": ("62265733", "eb837fb8deb27d4e0d52f71a03aff776"),
    "malonaldehyde": ("62265736", "cdc1d70c0c34062ddde6e5071eb6fe21"),
    "naphthalene": ("62265751", "0efba19c9907e3852318b1e6008b3b9e"),
    "paracetamol": ("62265760", "ba10784f7b67635427085f6d7ec2dd97"),
    "salicylic": ("62265748", "900fedf242da438400fa4293348d7dd1"),
    "toluene": ("62265742", "0f2913d51f8149c90ab28d697a076f64"),
    "uracil": ("62265745", "992a4479c28a07e0cce6da964805be31"),
}

SPLITS = {
    "index_train_01.csv": ("62265793", "22264dcf2e42d6133362411abbbe4ce8"),
    "index_train_02.csv": ("62265772", "28dbc9a3d5e7c47e653b1cc01809df35"),
    "index_train_03.csv": ("62265775", "8de68b1b288c790e633c9d05ff5f04dd"),
    "index_train_04.csv": ("62265766", "3764257abf8121a0905f364468ba83da"),
    "index_train_05.csv": ("62265778", "4670362bd554c9c22581c936432236b3"),
    "index_test_01.csv": ("62265781", "124c8a0849e84e244d6ce8424e1c05cf"),
    "index_test_02.csv": ("62265769", "9dc098724c1680daca038b785ee4f602"),
    "index_test_03.csv": ("62265787", "fd96f4623291dc23dd80b24dd634ca0b"),
    "index_test_04.csv": ("62265790", "f0acce759fa4fe46b3a0b2a450e7e260"),
    "index_test_05.csv": ("62265784", "4892da12e88a3c30c9a15f1f95e6eaac"),
}


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, file_id, md5, raw_dir):
    """Download one figshare file into raw_dir (skipped if present) and verify its md5."""
    path = Path(raw_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(FIGSHARE + file_id, path)
    if _md5(path) != md5:
        raise RuntimeError(f"md5 mismatch for {path}; delete it and re-download")
    return path


def download(molecule, raw_dir, split_id=1):
    npz = fetch(f"rmd17_{molecule}.npz", *MOLECULES[molecule], raw_dir)
    train_csv = fetch(f"index_train_0{split_id}.csv", *SPLITS[f"index_train_0{split_id}.csv"], raw_dir)
    test_csv = fetch(f"index_test_0{split_id}.csv", *SPLITS[f"index_test_0{split_id}.csv"], raw_dir)
    return npz, np.loadtxt(train_csv, dtype=int), np.loadtxt(test_csv, dtype=int)


def make_splits(train_idx, test_idx, n_train=950, n_val=50, nested=(100, 250), seed=0):
    """Official split -> {train, val, test, train_100, train_250, ...}.

    val is held out of the official 1,000 training frames; nested low-data
    subsets are prefixes of one permutation so train_100 ⊂ train_250 ⊂ train.
    """
    assert n_train + n_val <= len(train_idx)
    perm = np.random.default_rng(seed).permutation(train_idx)
    splits = {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": np.asarray(test_idx)}
    for n in nested:
        splits[f"train_{n}"] = perm[:n]
    return splits


def reference_pool(n_total, test_idx, n_frames, seed=0):
    """Frames for StABlE's *published-style* reference h(r): n_frames non-test frames.

    Only used for the secondary StABlE baseline; the main baseline computes its
    reference from the same 950 training frames as every other method.
    """
    candidates = np.setdiff1d(np.arange(n_total), test_idx)
    return np.random.default_rng(seed).choice(candidates, size=n_frames, replace=False)
