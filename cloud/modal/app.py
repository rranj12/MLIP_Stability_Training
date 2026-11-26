"""Run the pipeline on Modal. Data and results live on the `mlipstab` Volume.

    modal run --detach cloud/modal/app.py --molecule ethanol --model schnet
    modal run --detach cloud/modal/app.py --molecule aspirin --model gemnet_t
    modal run --detach cloud/modal/app.py --dataset md22 --molecule AT_AT_CG_CG --model schnet

Steps (comma-separated, default prepare,pretrain,eval):
    prepare   rMD17 or MD22 download + StABlE layout (skipped if already on the Volume)
    pretrain  Stage A (scripts/pretrain.sh), then link the run where StABlE expects it
    eval      scripts/evaluate.py on the Stage A model, or on --checkpoint
    margin    Stage B with our loss (scripts/train_stability.py on configs/margin/base.yml or md22.yml)

Large molecules: --pretrain-args="--optim.batch_size=16" is passed to scripts/pretrain.sh, and
--eval-overrides="{simulate: {n_replicas: 16}}" is merged over the evaluation config.

Lambda sweep (scripts/sweep_lambda.py), one GPU container per (lambda, seed):
    modal run --detach cloud/modal/app.py::sweep --config configs/sweep/lambda.yml --phase tune
    modal run --detach cloud/modal/app.py::sweep --config ... --phase final   # once, after reading selection.md

GPU: MLIP_GPU=A100 modal run ...   (default A10G)
Bounded runs: MLIP_RUN_TIMEOUT_S=600 caps one run invocation at 10 minutes (no retries).
Workspace: modal profile activate <workspace>, or MODAL_PROFILE=<workspace> modal run ...
Fetch results: modal volume get mlipstab results/eval/<tag> .
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

import modal

ROOT = "/root/repo"
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path(ROOT)  # in the container this file is /root/app.py
STABLE = f"{ROOT}/third_party/StABlE-Training"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch==2.3.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("torch-scatter", "torch-sparse", "torch-cluster",
                 find_links="https://data.pyg.org/whl/torch-2.3.0+cu121.html")
    .pip_install(
        "numpy>=1.25,<2", "nequip==0.6.1",  # nequip >= 0.7 breaks StABlE's loader
        "torch_geometric==2.5.3",  # newer releases need pyg-lib for radius_graph instead of torch-cluster
        "ase", "pymatgen>=2023.10.3", "numba", "orjson", "lmdb", "pyyaml", "submitit",
        "tensorboard", "tqdm", "wandb", "scikit-learn", "gsd", "lifelines",
        "matplotlib", "pandas", "scipy",
    )
    .env({"PYTHONPATH": f"{ROOT}/src:{STABLE}", "WANDB_MODE": "disabled", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(REPO, ROOT, copy=True,
                   ignore=[".git", "data", "results", "**/__pycache__", "**/paper_stable_training_results"])
    # the scripts read and write <repo>/data and <repo>/results; point both at the Volume
    .run_commands(f"ln -s /vol/data {ROOT}/data && ln -s /vol/results {ROOT}/results")
)

app = modal.App("mlipstab", image=image)
volume = modal.Volume.from_name("mlipstab", create_if_missing=True)


def sh(*cmd, commit_every_s=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    cmd = [str(c) for c in cmd]
    if commit_every_s is None:
        subprocess.run(cmd, check=True, cwd=ROOT)
        return
    # Commit partial snapshots while the training subprocess is still running.
    with subprocess.Popen(cmd, cwd=ROOT) as process:
        try:
            while True:
                try:
                    returncode = process.wait(timeout=commit_every_s)
                    break
                except subprocess.TimeoutExpired:
                    volume.commit()
            if returncode:
                raise subprocess.CalledProcessError(returncode, cmd)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            volume.commit()


@app.function(gpu=os.environ.get("MLIP_GPU", "A10G"), volumes={"/vol": volume},
              timeout=int(os.environ.get("MLIP_RUN_TIMEOUT_S", 24 * 3600)), retries=0)
def run(molecule: str, model: str, steps: str, eval_ps: float, checkpoint: str, tag: str,
        margin_overrides: str = "", stage_a: str = "", dataset: str = "md17",
        pretrain_args: str = "", eval_overrides: str = "", seed: Optional[int] = None):
    import yaml
    if seed is not None and stage_a:
        raise ValueError("use --seed for independent Stage A runs or --stage-a for a legacy run, not both")
    os.environ["DATASET"] = dataset  # read by scripts/pretrain.sh
    if seed is not None:
        os.environ["TRAIN_SEED"] = str(seed)
    else:
        os.environ.pop("TRAIN_SEED", None)
    name = f"{dataset}-{molecule}_1k_{model}"  # StABlE's name for a Stage A run
    margin_cfg, eval_cfg = {"md17": ("base.yml", "rmd17_500K.yml"), "md22": ("md22.yml", "md22_500K.yml")}[dataset]
    os.makedirs("/vol/data", exist_ok=True)
    os.makedirs("/vol/results", exist_ok=True)
    sh("nvidia-smi")
    steps = steps.split(",")

    if "prepare" in steps and not os.path.exists(f"/vol/data/processed/{dataset}/{molecule}/1k/metadata.npy"):
        if dataset == "md22":
            sh("python", "scripts/prepare_md22.py", "--molecule", molecule)
        else:
            sh("python", "scripts/prepare_rmd17.py", "--molecule", molecule, "--reference", "same")
        volume.commit()

    if "pretrain" in steps:
        sh("bash", "scripts/pretrain.sh", model, molecule, "1k", *pretrain_args.split())
        volume.commit()

    # stage_a: name of a run under results/pretrain/<model>_<molecule>_1k/ to start from instead
    # of the latest one (e.g. the patience-5 run after a patience-50 run replaced the default link)
    pre = {}
    if seed is not None:
        pre = {"seed": seed, "pretrained": {"dir": f"results/checkpoints/qm_pretrained/seed{seed}",
                                           "seed": seed}}
    dump = lambda d: yaml.safe_dump(d) if d else ""  # safe_dump({}) is "{}", not valid after other keys
    if stage_a:
        dest = Path(f"/vol/results/checkpoints/qm_{stage_a}/{model}/{name}")
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / "checkpoints").is_symlink():
            (dest / "checkpoints").symlink_to(f"/vol/results/pretrain/{model}_{molecule}_1k/{stage_a}/checkpoints")
        pre = {"pretrained": {"dir": f"results/checkpoints/qm_{stage_a}"}}

    if "margin" in steps:
        cfg = "/tmp/margin.yml"  # margin_overrides: YAML merged over base.yml, e.g. "{optim: {lr: 1.0e-5}}"
        Path(cfg).write_text(f"includes: [{ROOT}/configs/margin/{margin_cfg}]\n"
                             + dump({**(yaml.safe_load(margin_overrides or "{}") or {}), **pre}))
        sh("python", "scripts/train_stability.py", "--config", cfg, "--molecule", molecule, "--model", model,
           commit_every_s=30)
        volume.commit()

    if "eval" in steps:
        cfg = "/tmp/eval.yml"
        over = yaml.safe_load(eval_overrides or "{}") or {}
        over.setdefault("simulate", {})["length_ps"] = eval_ps
        Path(cfg).write_text(f"includes: [{ROOT}/configs/eval/{eval_cfg}]\n" + dump({**over, **pre}))
        cmd = ["python", "scripts/evaluate.py", "--config", cfg, "--molecule", molecule, "--model", model,
               "--tag", tag or (f"pretrained_{model}_{molecule}_{eval_ps:g}ps"
                                + (f"_seed{seed}" if seed is not None else "")),
               "--seed", str(seed if seed is not None else 0)]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        sh(*cmd)
        volume.commit()


@app.local_entrypoint()
def main(molecule: str = "ethanol", model: str = "schnet", steps: str = "prepare,pretrain,eval",
         eval_ps: float = 100.0, checkpoint: str = "", tag: str = "", margin_overrides: str = "", stage_a: str = "",
         dataset: str = "md17", pretrain_args: str = "", eval_overrides: str = "", seed: Optional[int] = None):
    run.remote(molecule, model, steps, eval_ps, checkpoint, tag, margin_overrides, stage_a,
               dataset, pretrain_args, eval_overrides, seed)


# ---------------------------------------------------------------- bounded diagnostic
@app.function(gpu="A100", volumes={"/vol": volume}, timeout=300, retries=0)
def bond_distortion_check():
    """Synthetic-pair check only: eight total updates, no MD or sweep."""
    try:
        sh("nvidia-smi")
        sh("python", "scripts/check_bond_distortion.py", "--out",
           "results/smoke/gemnet_t_naphthalene_distortion_20261125", commit_every_s=30)
    finally:
        volume.commit()


# ---------------------------------------------------------------- lambda sweep
@app.function(gpu=os.environ.get("MLIP_GPU", "A10G"), volumes={"/vol": volume}, timeout=24 * 3600)
def sweep_job(args: list):
    """One `scripts/sweep_lambda.py run-one ...` job: Stage B for one (lambda, seed), then its evaluation."""
    sh("python", "scripts/sweep_lambda.py", *args)
    volume.commit()


def _sweep_cli(*args):
    out = subprocess.run(["python", "scripts/sweep_lambda.py", *args], cwd=ROOT, check=True,
                         capture_output=True, text=True)
    print(out.stderr, end="")
    return out.stdout


@app.function(volumes={"/vol": volume}, timeout=24 * 3600)
def sweep_driver(config: str, phase: str, force: bool):
    """Runs remotely so `--detach` keeps the whole sweep going after the laptop disconnects."""
    import json
    if phase == "tune":
        print(_sweep_cli("prepare-tune", "--config", config))
        volume.commit()
    jobs = json.loads(_sweep_cli("plan", "--config", config, "--stage", phase, *(["--force"] if force else [])))
    volume.commit()  # the final lock is written by `plan`
    print(f"{len(jobs)} jobs")
    for job, res in zip(jobs, sweep_job.map(jobs, return_exceptions=True)):
        if isinstance(res, Exception):
            print("FAILED:", " ".join(job), "->", repr(res))
    volume.reload()
    step = ["select", "--config", config, "--allow-missing"] if phase == "tune" else ["summarize", "--config", config]
    print(_sweep_cli(*step))
    volume.commit()


@app.local_entrypoint()
def sweep(config: str = "configs/sweep/lambda.yml", phase: str = "tune", force: bool = False):
    sweep_driver.remote(config, phase, force)
