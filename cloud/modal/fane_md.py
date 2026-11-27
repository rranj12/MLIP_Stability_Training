"""Fu et al. stability evaluation (scripts/fane_md.py) of Stage A and Stage B checkpoints, every
(variant, model seed, replica) in parallel.

    modal run --detach cloud/modal/fane_md.py --molecule naphthalene --seeds 0,1,2 --lams 0,0.003
    MLIP_GPU=A10G modal run --detach cloud/modal/fane_md.py ...   # default T4

Stage A: results/checkpoints/fane_seed<seed>/<model>/ (cloud/modal/fane_train.py). Stage B (--lams):
the sweep's ckpt.pt under results/sweeps/<sweep>/. Missing checkpoints are skipped, and finished
replicas (result.json) are not rerun, so rerunning fills in the rest. Results:
results/fu_md/<molecule>_<model>/<variant>_seed<seed>/rep<replica seed>_init<idx>/result.json.
"""

import os
import subprocess
import sys
from pathlib import Path

import modal

if modal.is_local():
    sys.path.insert(0, str(Path(__file__).parent))
from fane_pilot import LOCAL, PYTHON, image, volume  # noqa: E402

PROTOCOL = "configs/reproduce/fane_gemnet_t_naphthalene/protocol.yml"  # the five replicas
image = (image.add_local_python_source("fane_pilot")
         .add_local_file(LOCAL / "scripts/fane_md.py", "/opt/fane_md.py")
         .add_local_file(LOCAL / PROTOCOL, "/opt/protocol.yml"))
app = modal.App("fane-md17-md", image=image)
VOL = Path("/vol/results")


@app.function(gpu=os.environ.get("MLIP_GPU", "T4"), volumes={"/vol": volume},
              timeout=24 * 3600, cpu=2, memory=8192, retries=1)
def md(molecule: str, model_dir: str, out: str, seed: int, init_idx: int, weights: str = "", max_ps: float = 300.):
    cmd = [PYTHON, "/opt/fane_md.py", molecule, model_dir, out, str(seed), str(init_idx), str(max_ps)]
    process = subprocess.Popen(cmd + ([weights] if weights else []))
    while process.poll() is None:  # commit progress every 5 minutes
        try:
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            volume.commit()
    volume.commit()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, cmd)


@app.function(volumes={"/vol": volume}, timeout=24 * 3600)
def launch(molecule: str, model: str, seeds: list, lams: list, sweep: str):
    import json

    read = "import json, yaml; print(json.dumps(yaml.safe_load(open('/opt/protocol.yml'))['evaluation']['replicas']))"
    replicas = json.loads(subprocess.run([PYTHON, "-c", read], capture_output=True, text=True, check=True).stdout)
    jobs = []
    for seed in seeds:
        model_dir = VOL / f"checkpoints/fane_seed{seed}/{model}/md17-{molecule}_10k_{model}"
        variants = [("stageA", "")] + [
            (f"lam{lam:g}", str(VOL / f"sweeps/{sweep}/{molecule}_{model}/train/lam{lam:g}/{molecule}_{model}_seed{seed}/ckpt.pt"))
            for lam in lams]
        for tag, weights in variants:
            if not (model_dir / "checkpoints/best_checkpoint.pt").exists() or (weights and not Path(weights).exists()):
                print("skip, no checkpoint:", tag, seed)
                continue
            for r in replicas:
                out = VOL / f"fu_md/{molecule}_{model}/{tag}_seed{seed}/rep{r['seed']}_init{r['init_idx']}"
                if not (out / "result.json").exists():
                    jobs.append((molecule, str(model_dir), str(out), r["seed"], r["init_idx"], weights))
    print(f"{len(jobs)} replicas")
    for job, res in zip(jobs, md.starmap(jobs, return_exceptions=True)):
        if isinstance(res, Exception):
            print("FAILED", job[2], repr(res))


@app.local_entrypoint()
def main(molecule: str = "naphthalene", model: str = "schnet", seeds: str = "0,1,2",
         lams: str = "", sweep: str = "naphthalene_fu"):
    launch.remote(molecule, model, [int(s) for s in seeds.split(",")],
                  [float(x) for x in lams.split(",") if x], sweep)
