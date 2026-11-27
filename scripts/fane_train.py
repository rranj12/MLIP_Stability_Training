"""Full Stage A training with pinned upstream MDsim (Fu et al. protocol). Runs in the historical
environment of cloud/modal/fane_pilot.py; launched by cloud/modal/fane_train.py.

    python fane_train.py <config.yml> <molecule> <run_dir> <seed>

The config's dataset.molecule is replaced by <molecule>; everything else is the config as is.
Resumes from the newest <run_dir>/**/checkpoints/checkpoint.pt (weights, optimizer, scheduler,
step), so a run cut off by the container timeout continues where it stopped. Stopping rules
are the config's: max_epochs, early_stopping_lr, early_stopping_time.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

MDSIM = Path("/opt/MDsim")


def main(config, molecule, run_dir, seed):
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(config.read_text())
    cfg["dataset"]["molecule"] = molecule
    run_cfg = run_dir / "train.yml"
    run_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    resume = sorted(run_dir.glob("**/checkpoints/checkpoint.pt"), key=lambda p: p.stat().st_mtime)
    command = [sys.executable, "-u", "main.py", "--mode", "train", "--config-yml", str(run_cfg),
               "--run-dir", str(run_dir), "--identifier", f"seed{seed}", "--seed", str(seed),
               "--print-every", "100"]
    if resume:
        command += ["--checkpoint", str(resume[-1])]
    (run_dir / f"launch_{int(time.time())}.json").write_text(json.dumps({"command": command}, indent=1))
    with (run_dir / "train.log").open("a") as log:
        subprocess.run(command, cwd=MDSIM, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]))
