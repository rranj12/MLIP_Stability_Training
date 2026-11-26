"""Run only in the isolated historical MDsim environment via cloud/modal/fane_pilot.py."""

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

COMMIT = "61ca2cfba373fd374343592556789fa30fbe3b56"
ARCHIVE = "https://zenodo.org/records/7196767/files/mdsim_data.tar.gz?download=1"
ARCHIVE_MD5 = "2f895e74a32793e37da43b961ce8edb2"
ROOT = Path("/opt/MDsim")
DATA = Path("/vol/data/forces_not_enough/md17/naphthalene/10k")
RESULTS = Path("/vol/results/reproduce/fane_gemnet_t_naphthalene/pilot_seed0_retry1")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    from concurrent.futures import ThreadPoolExecutor, wait
    import pickle
    import tarfile
    import urllib.request

    import lmdb
    import numpy as np
    import torch
    import yaml
    from mdsim.datasets import data_list_collater
    from mdsim.datasets.lmdb_dataset import LmdbDataset
    from mdsim.models.gemnet.gemnet import GemNetT

    provenance_path = DATA / "provenance.json"
    if not provenance_path.exists():
        archive = Path("/tmp/mdsim_data.tar.gz")
        size, workers = 7279215356, 8
        progress = [0] * workers
        with archive.open("wb") as output:
            output.truncate(size)

        def download_part(index):
            start, end = size * index // workers, size * (index + 1) // workers - 1
            request = urllib.request.Request(ARCHIVE, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("r+b") as output:
                assert response.status == 206
                assert response.headers["Content-Range"] == f"bytes {start}-{end}/{size}"
                output.seek(start)
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    progress[index] += len(chunk)
                assert progress[index] == end - start + 1

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(download_part, index) for index in range(workers)}
            while pending:
                done, pending = wait(pending, timeout=30)
                for future in done:
                    future.result()
                print(f"Archive download: {sum(progress) / 1e9:.2f} / 7.28 GB", flush=True)
        received = sum(progress)
        digest = hashlib.md5()
        with archive.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != ARCHIVE_MD5:
            raise ValueError("Authors' archive checksum mismatch")
        members = []
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                parts = Path(member.name).parts
                if "naphthalene" not in parts or not member.isfile():
                    continue
                tail = parts[parts.index("naphthalene") + 1:]
                if not tail or tail[0] != "10k" or ".." in tail:
                    continue
                destination = DATA.joinpath(*tail[1:])
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as inp, destination.open("wb") as out:
                    shutil.copyfileobj(inp, out)
                members.append(member.name)
                print(f"Extracted {member.name}", flush=True)
        if not members:
            raise ValueError("No naphthalene/10k data found in authors' archive")
        write_json(provenance_path, {"url": ARCHIVE, "md5": digest.hexdigest(),
                                    "bytes": received, "members": members, "source_commit": COMMIT})

    report = {"provenance": json.loads(provenance_path.read_text()), "splits": {}}
    for split, expected in (("train", 9500), ("val", 500), ("test", 10000)):
        path = DATA / split
        arrays = np.load(path / "nequip_npz.npz")
        assert arrays["R"].shape == (expected, 18, 3)
        for field in ("R", "E", "F"):
            assert np.isfinite(arrays[field]).all(), (split, field)
        env = lmdb.open(str(path / "data.lmdb"), subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            count = env.stat()["entries"] - 1
            header = pickle.loads(txn.get(b"length"))
            assert count == expected, (split, count)
            for index in (0, expected // 2, expected - 1):
                sample = pickle.loads(txn.get(str(index).encode()))
                np.testing.assert_allclose(sample.pos.numpy(), arrays["R"][index], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(sample.force.numpy(), arrays["F"][index], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(sample.y.numpy().ravel(), arrays["E"][index].ravel(), rtol=1e-5)
        env.close()
        dataset = LmdbDataset({"src": str(path)})
        assert len(dataset) == header
        assert len(dataset[0].pos) == 18
        dataset.close_db()
        report["splits"][split] = {"stored": count, "upstream_loader": header}
    assert np.isfinite(list(np.load(DATA / "metadata.npy", allow_pickle=True).item().values())).all()
    report["torch"] = torch.__version__
    report["python"] = sys.version
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    assert revision == COMMIT
    report["source_commit"] = revision
    model_config = yaml.safe_load(Path("/opt/train.yml").read_text())["model"]
    model_config.pop("name")
    model_config["scale_file"] = str(ROOT / model_config["scale_file"])
    torch.manual_seed(0)
    model = GemNetT(None, 50, 1, **model_config)
    dataset = LmdbDataset({"src": str(DATA / "train")})
    batch = data_list_collater([dataset[0]], otf_graph=True)
    energy, forces = model(batch)
    (energy.square().mean() + forces.square().mean()).backward()
    assert torch.isfinite(energy).all() and torch.isfinite(forces).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    report["cpu_forward_force_backward"] = "passed"
    report["model_parameters"] = model.num_params
    assert model.num_params == 1891025
    dataset.close_db()
    write_json(DATA / "validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def pilot():
    import torch
    import yaml

    started = time.monotonic()
    validation = json.loads((DATA / "validation.json").read_text())
    # Never silently overwrite/restart this budgeted experiment.
    if RESULTS.exists():
        raise FileExistsError(f"Pilot already attempted: {RESULTS}")
    RESULTS.mkdir(parents=True)
    cfg = yaml.safe_load(Path("/opt/train.yml").read_text())
    cfg["optim"]["max_epochs"] = 1
    cfg["optim"]["early_stopping_time"] = 3000
    cfg["identifier"] = "pilot_seed0"
    config = RESULTS / "pilot.yml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    (RESULTS / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    (RESULTS / "nvidia_smi.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
    write_json(RESULTS / "started.json", {"source_commit": COMMIT, "gpu_cap_s": 3300,
                                        "prior_attempt_reserved_s": 300,
                                        "max_epochs": 1, "seed": 0, "data": validation})
    command = [sys.executable, "-u", "main.py", "--mode", "train", "--config-yml", str(config),
               "--run-dir", str(RESULTS), "--identifier", "pilot_seed0", "--seed", "0",
               "--max_epochs", "1", "--print-every", "200"]
    status = {"status": "started", "command": command}
    try:
        train_started = time.monotonic()
        with (RESULTS / "train.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    remaining = 3150 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("Pilot training stopped before the 3300s GPU hard cap")
                    try:
                        process.wait(timeout=min(30, remaining))
                    except subprocess.TimeoutExpired:
                        print("\n".join((RESULTS / "train.log").read_text().splitlines()[-2:]), flush=True)
                if process.returncode:
                    raise RuntimeError(f"Upstream training exited {process.returncode}; see train.log")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        status["training_process_s"] = time.monotonic() - train_started
        checkpoints = list(RESULTS.glob("**/checkpoints/checkpoint.pt"))
        assert len(checkpoints) == 1, checkpoints
        checkpoint = torch.load(checkpoints[0], map_location="cpu")
        best = torch.load(checkpoints[0].with_name("best_checkpoint.pt"), map_location="cpu")
        steps = validation["splits"]["train"]["upstream_loader"]
        assert checkpoint["step"] == steps, checkpoint["step"]
        assert checkpoint["optimizer"]["state"] and checkpoint["ema"]
        assert all(torch.isfinite(value).all() for value in best["state_dict"].values())
        assert all(torch.isfinite(torch.tensor(value["metric"])) for value in best["val_metrics"].values())
        status.update(status="completed", steps=steps, epoch=float(checkpoint["epoch"]),
                      val_metrics=best["val_metrics"], checkpoint=str(checkpoints[0]),
                      gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        status["gpu_function_elapsed_s"] = time.monotonic() - started
        status["gpu_compute_estimate_usd"] = status["gpu_function_elapsed_s"] * 0.000694
        write_json(RESULTS / "result.json", status)
        print(json.dumps(status, indent=2), flush=True)
    return status

def verify_checkpoint():
    """Exercise the authors' ASE calculator on CPU, without an MD rollout."""
    import os

    import numpy as np
    import torch
    import yaml
    from mdsim.md.ase_utils import OCPCalculator, data_to_atoms

    os.chdir(ROOT)
    checkpoints = list(RESULTS.glob("**/checkpoints/checkpoint.pt"))
    assert len(checkpoints) == 1
    raw = torch.load(checkpoints[0], map_location="cpu")
    for state in raw["optimizer"]["state"].values():
        assert all(torch.isfinite(value).all() for value in state.values() if torch.is_tensor(value))
    steps = sorted({int(state["step"]) for state in raw["optimizer"]["state"].values()})
    assert steps == [9499], steps
    cfg = yaml.safe_load(checkpoints[0].with_name("config.yml").read_text())
    cfg["cpu"] = True
    calculator = OCPCalculator(config_yml=cfg,
                               checkpoint=str(checkpoints[0].with_name("best_checkpoint.pt")),
                               test_data_src=str(DATA / "test"))
    atoms = data_to_atoms(calculator.trainer.test_dataset[0])
    atoms.calc = calculator
    energy, forces = atoms.get_potential_energy(), atoms.get_forces()
    assert np.isfinite(energy) and np.isfinite(forces).all() and forces.shape == (18, 3)
    report = {"status": "passed", "device": str(calculator.trainer.device),
              "optimizer_steps": steps, "energy_eV": float(energy),
              "force_shape": list(forces.shape), "md_steps": 0}
    write_json(RESULTS / "cpu_checkpoint_verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    {"prepare": prepare, "pilot": pilot, "verify": verify_checkpoint}[sys.argv[1]]()
