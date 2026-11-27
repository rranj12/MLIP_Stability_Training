"""Exercise seed forwarding and checkpoint links without loading or training a model."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_pretrain_seeds_have_separate_runs_and_checkpoint_roots(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(root / "scripts/pretrain.sh", scripts / "pretrain.sh")
    (tmp_path / "third_party/StABlE-Training").mkdir(parents=True)
    fake = tmp_path / "fake_pretraining.py"
    fake.write_text('''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--run-dir')
p.add_argument('--seed', type=int, default=0)
a, _ = p.parse_known_args()
d = Path(a.run_dir) / 'mock-run' / 'checkpoints'
d.mkdir(parents=True)
(d / 'best_checkpoint.pt').touch()
(d / 'config.yml').write_text(json.dumps({'seed': a.seed}))
''')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(f'''#!/bin/sh
if [ "$(basename "$1")" = pretrain_mdsim.py ]; then
    exec "{sys.executable}" "{fake}" "$@"
fi
exec "{sys.executable}" "$@"
''')
    python.chmod(0o755)
    paths = []
    for seed in (None, 0, 1):
        env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "DATASET": "md17"}
        env.pop("TRAIN_SEED", None)
        if seed is not None:
            env["TRAIN_SEED"] = str(seed)
        result = subprocess.run(["bash", str(scripts / "pretrain.sh"), "gemnet_t", "aspirin"],
                                env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        base = tmp_path / "results/checkpoints/qm_pretrained"
        run = tmp_path / "results/pretrain/gemnet_t_aspirin_1k"
        if seed is not None:
            base, run = base / f"seed{seed}", run / f"seed{seed}"
        link = base / "gemnet_t/md17-aspirin_1k_gemnet_t/checkpoints"
        assert link.resolve() == run / "mock-run/checkpoints"
        assert json.loads((link / "config.yml").read_text())["seed"] == (seed or 0)
        paths.append(link.resolve())
    assert len(set(paths)) == 3
