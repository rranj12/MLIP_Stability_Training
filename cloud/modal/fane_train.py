"""Full Fu et al. Stage A training on Modal, in the pinned environment of fane_pilot.py.

    modal run --detach cloud/modal/fane_train.py --model schnet --molecule naphthalene --seeds 0,1,2
    MLIP_GPU=A100 modal run --detach cloud/modal/fane_train.py ...   # default A10G

All (molecule, seed) runs train in parallel. Missing MD17 molecules are first extracted from the
authors' archive (aspirin, ethanol, naphthalene, salicylic_acid). Each run writes
/vol/results/reproduce/fane_<model>_<molecule>/full_seed<seed>/ and, on success, is linked as
results/checkpoints/fane_seed<seed>/<model>/md17-<molecule>_10k_<model>, the Stage A layout that
scripts/sweep_lambda.py loads (configs/sweep/naphthalene.yml). Rerunning resumes from the last
checkpoint.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

import modal

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).parent))
from fane_pilot import LOCAL, PYTHON, image, volume  # noqa: E402

ARCHIVE = "https://zenodo.org/records/7196767/files/mdsim_data.tar.gz?download=1"  # as scripts/fane_pilot.py
ARCHIVE_MD5 = "2f895e74a32793e37da43b961ce8edb2"

MODELS = ("schnet",)  # configs/reproduce/fane_<model>_naphthalene/train.yml
DATA = Path("/vol/data/forces_not_enough/md17")
SIZES = {"train": 9500, "val": 500, "test": 10000}
image = image.add_local_python_source("fane_pilot").add_local_file(
    LOCAL / "scripts/fane_train.py", "/opt/fane_train.py")
for m in MODELS:
    image = image.add_local_file(LOCAL / f"configs/reproduce/fane_{m}_naphthalene/train.yml",
                                 f"/opt/train_{m}.yml")
app = modal.App("fane-md17-train", image=image)


@app.function(volumes={"/vol": volume}, timeout=3 * 3600, cpu=4, memory=8192, retries=0)
def prepare(molecules: list):
    """Stream the authors' archive once and extract <molecule>/10k for every missing molecule."""
    import hashlib
    import shutil
    import tarfile
    import urllib.request

    todo = [m for m in molecules if not (DATA / m / "10k/test/nequip_npz.npz").exists()]
    if not todo:
        return
    digest = hashlib.md5()

    class Hashed:  # md5 of the bytes as they stream through tarfile
        def __init__(self, raw):
            self.raw = raw

        def read(self, n=-1):
            b = self.raw.read(n)
            digest.update(b)
            return b

    with urllib.request.urlopen(ARCHIVE, timeout=300) as response:
        stream = Hashed(response)
        with tarfile.open(fileobj=stream, mode="r|gz") as tar:
            for member in tar:
                parts = Path(member.name).parts
                if not member.isfile() or len(parts) < 5 or parts[1] != "md17" or parts[2] not in todo \
                        or parts[3] != "10k" or ".." in parts:
                    continue
                dest = DATA.joinpath(*parts[2:])
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as inp, dest.open("wb") as out:
                    shutil.copyfileobj(inp, out)
        while stream.read(1 << 24):  # hash the rest of the archive too
            pass
    if digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError("authors' archive checksum mismatch")
    check = "import numpy as np, sys; assert len(np.load(sys.argv[1])['R']) == int(sys.argv[2]), sys.argv[1]"
    for m in todo:
        for split, n in SIZES.items():
            subprocess.run([PYTHON, "-c", check, str(DATA / m / f"10k/{split}/nequip_npz.npz"), str(n)], check=True)
    volume.commit()
    print(f"extracted {todo}")


@app.function(gpu=os.environ.get("MLIP_GPU", "A10G"), volumes={"/vol": volume},
              timeout=24 * 3600, cpu=8, memory=32768, retries=0)
def train(model: str, molecule: str, seed: int):
    if not (DATA / molecule / "10k/train/nequip_npz.npz").exists():
        raise FileNotFoundError(f"no data for {molecule}; run prepare first")
    subprocess.run(["nvidia-smi"], check=True)
    out = Path(f"/vol/results/reproduce/fane_{model}_{molecule}/full_seed{seed}")
    cmd = [PYTHON, "/opt/fane_train.py", f"/opt/train_{model}.yml", molecule, str(out), str(seed)]
    with subprocess.Popen(cmd, start_new_session=True) as process:
        try:
            while process.poll() is None:  # commit checkpoints every minute
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    volume.commit()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, cmd)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            volume.commit()
    link.local(model, molecule, seed)


@app.function(volumes={"/vol": volume}, timeout=300)
def link(model: str, molecule: str, seed: int):
    """Expose a finished run where the Stage B backend looks for Stage A checkpoints."""
    run = Path(f"/vol/results/reproduce/fane_{model}_{molecule}/full_seed{seed}/md17-{molecule}_10k_{model}_seed{seed}")
    if not (run / "checkpoints/best_checkpoint.pt").exists():
        raise FileNotFoundError(run)
    dest = Path(f"/vol/results/checkpoints/fane_seed{seed}/{model}/md17-{molecule}_10k_{model}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        dest.unlink()
    dest.symlink_to(run)
    volume.commit()
    print(f"{dest} -> {run}")


@app.function(volumes={"/vol": volume}, timeout=24 * 3600, retries=0)
def launch(model: str, molecules: list, seeds: list):
    """Runs on Modal, so --detach keeps everything going without the laptop."""
    prepare.remote(molecules)
    jobs = [(model, m, s) for m in molecules for s in seeds]
    for job, res in zip(jobs, train.starmap(jobs, return_exceptions=True)):
        print("FAILED" if isinstance(res, Exception) else "done", job, repr(res) if res else "")


@app.local_entrypoint()
def main(model: str = "schnet", molecule: str = "naphthalene", seeds: str = "0"):
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    launch.remote(model, molecule.split(","), [int(s) for s in seeds.split(",")])
