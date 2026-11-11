"""Run the pipeline on Modal. Data and results live on the `mlipstab` Volume.

    modal run --detach cloud/modal/app.py --molecule ethanol --model schnet

Steps (comma-separated, default prepare,pretrain,eval):
    prepare   rMD17 download + StABlE layout (skipped if already on the Volume)
    pretrain  Stage A (scripts/pretrain.sh), then link the run where StABlE expects it
    eval      scripts/evaluate.py on the Stage A model, or on --checkpoint
    margin    Stage B with our loss (scripts/train_stability.py on configs/margin/base.yml)

GPU: MLIP_GPU=A100 modal run ...   (default A10G)
Workspace: modal profile activate <workspace>, or MODAL_PROFILE=<workspace> modal run ...
Fetch results: modal volume get mlipstab results/eval/<tag> .
"""

import os
import subprocess
from pathlib import Path

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
    .env({"PYTHONPATH": f"{ROOT}/src:{STABLE}", "WANDB_MODE": "disabled"})
    .add_local_dir(REPO, ROOT, copy=True,
                   ignore=[".git", "data", "results", "**/__pycache__", "**/paper_stable_training_results"])
    # the scripts read and write <repo>/data and <repo>/results; point both at the Volume
    .run_commands(f"ln -s /vol/data {ROOT}/data && ln -s /vol/results {ROOT}/results")
)

app = modal.App("mlipstab", image=image)
volume = modal.Volume.from_name("mlipstab", create_if_missing=True)


def sh(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


@app.function(gpu=os.environ.get("MLIP_GPU", "A10G"), volumes={"/vol": volume}, timeout=12 * 3600)
def run(molecule: str, model: str, steps: str, eval_ps: float, checkpoint: str, tag: str):
    os.makedirs("/vol/data", exist_ok=True)
    os.makedirs("/vol/results", exist_ok=True)
    sh("nvidia-smi")
    steps = steps.split(",")

    if "prepare" in steps and not os.path.exists(f"/vol/data/processed/md17/{molecule}/1k/metadata.npy"):
        sh("python", "scripts/prepare_rmd17.py", "--molecule", molecule, "--reference", "same")
        volume.commit()

    if "pretrain" in steps:
        sh("bash", "scripts/pretrain.sh", model, molecule)
        if model != "nequip":  # pretrain.sh leaves this link to the caller
            run_dir = Path(f"/vol/results/pretrain/{model}_{molecule}_1k")
            ckpt = max(run_dir.glob("*/checkpoints/best_checkpoint.pt"), key=os.path.getmtime)
            dest = Path(f"/vol/results/checkpoints/qm_pretrained/{model}/md17-{molecule}_1k_{model}")
            dest.mkdir(parents=True, exist_ok=True)
            link = dest / "checkpoints"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(ckpt.parent)
        volume.commit()

    if "margin" in steps:
        sh("python", "scripts/train_stability.py", "--config", "configs/margin/base.yml",
           "--molecule", molecule, "--model", model)
        volume.commit()

    if "eval" in steps:
        cfg = "/tmp/eval.yml"
        Path(cfg).write_text(
            f"includes: [{ROOT}/configs/eval/rmd17_500K.yml]\nsimulate: {{length_ps: {eval_ps}}}\n")
        cmd = ["python", "scripts/evaluate.py", "--config", cfg, "--molecule", molecule, "--model", model,
               "--tag", tag or f"pretrained_{model}_{molecule}_{eval_ps:g}ps"]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        sh(*cmd)
        volume.commit()


@app.local_entrypoint()
def main(molecule: str = "ethanol", model: str = "schnet", steps: str = "prepare,pretrain,eval",
         eval_ps: float = 100.0, checkpoint: str = "", tag: str = ""):
    run.remote(molecule, model, steps, eval_ps, checkpoint, tag)
