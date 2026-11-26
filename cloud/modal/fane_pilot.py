"""CPU preparation, then one upstream epoch; 3300s cap after a failed import attempt.

    modal run --detach cloud/modal/fane_pilot.py

The supported Modal host runs an isolated Python 3.9 training environment.
"""

import os
import signal
import subprocess
from pathlib import Path

import modal

COMMIT = "61ca2cfba373fd374343592556789fa30fbe3b56"
LOCAL = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt")
PYTHON = "/opt/mdsim-env/bin/python"
PIP = f"{PYTHON} -m pip install"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("uv==0.9.11")
    .run_commands(
        "uv python install 3.9.23",
        "uv venv --python 3.9.23 --seed /opt/mdsim-env",
        f"{PIP} pip==24.0 setuptools==61.3.1 wheel",
        f"{PIP} torch==1.11.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113",
    )
    .run_commands(
        f"{PIP} numpy==1.21.2 scipy==1.7.3",
        f"{PIP} torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 "
        "torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html",
    )
    .run_commands(
        f"{PIP} pyg-nightly==2.0.5.dev20220418 lmdb==1.1.1 numba==0.55.1 "
        "PyYAML==6.0 scikit-learn==1.1.2 submitit==1.4.5 sympy==1.7.1 "
        "tqdm==4.63.1 wandb==0.12.17 ase==3.22.1 matplotlib==3.5.2 "
        "tensorboard==2.9.1 protobuf==3.19.6 orjson",
    )
    # Torch's wheel does not bundle the cuSPARSE runtime required by torch-sparse.
    # Use NVIDIA's CUDA 11.3 packages, rather than mixing in a newer CUDA release.
    .run_commands(
        "python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/"
        "libcusparse-11-3_11.6.0.109-1_amd64.deb', '/tmp/cusparse.deb')\"",
        "dpkg-deb -x /tmp/cusparse.deb /",
        "python -c \"import urllib.request; urllib.request.urlretrieve("
        "'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/"
        "cuda-cudart-11-3_11.3.109-1_amd64.deb', '/tmp/cudart.deb')\"",
        "dpkg-deb -x /tmp/cudart.deb /",
        "ldconfig /usr/local/cuda-11.3/targets/x86_64-linux/lib",
        "rm /tmp/cusparse.deb /tmp/cudart.deb",
    )
    .run_commands(
        "git clone https://github.com/kyonofx/MDsim.git /opt/MDsim",
        f"git -C /opt/MDsim checkout {COMMIT}",
        f"{PIP} --no-deps -e /opt/MDsim",
        f"{PYTHON} -c 'from mdsim.common.utils import setup_imports; setup_imports()'",
    )
    .env({"PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled", "OMP_NUM_THREADS": "4",
          "LD_LIBRARY_PATH": "/usr/local/cuda-11.3/targets/x86_64-linux/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"})
    .add_local_file(LOCAL / "configs/reproduce/fane_gemnet_t_naphthalene/train.yml", "/opt/train.yml")
    .add_local_file(LOCAL / "scripts/fane_pilot.py", "/opt/fane_pilot.py")
)
app = modal.App("fane-gemnet-naphthalene-pilot", image=image)
volume = modal.Volume.from_name("mlipstab")


def run_script(phase):
    with subprocess.Popen([PYTHON, "/opt/fane_pilot.py", phase], start_new_session=True) as process:
        try:
            while True:
                try:
                    returncode = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    volume.commit()
            if returncode:
                raise subprocess.CalledProcessError(returncode, process.args)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            volume.commit()


@app.function(volumes={"/vol": volume}, timeout=10800, cpu=4, memory=16384, retries=0)
def prepare():
    run_script("prepare")


@app.function(gpu="A100-80GB", volumes={"/vol": volume}, timeout=3300,
              cpu=4, memory=16384, retries=0)
def pilot():
    run_script("pilot")


@app.function(volumes={"/vol": volume}, timeout=300, cpu=4, memory=16384, retries=0)
def verify_checkpoint():
    run_script("verify")


@app.local_entrypoint()
def main():
    prepare.remote()
    pilot.remote()
    verify_checkpoint.remote()
