"""Stage B for our method (and CRMSE): load a Stage A checkpoint, run the loop.

    python scripts/train_stability.py --config configs/margin/aspirin_schnet.yml
    python scripts/train_stability.py --config configs/crmse/base.yml --method crmse \
        --molecule aspirin --model schnet

Writes <results_dir>/<molecule>_<model>_seed<seed>/{ckpt.pt, history.json}.
The loops live in mlipstab.train; the StABlE/MDsim glue is mlipstab.mdsim_backend.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/StABlE-Training"))

from mlipstab.config import load_config  # noqa: E402
from mlipstab.detect import bond_deviation  # noqa: E402
from mlipstab.losses.anchors import make_anchor_fn  # noqa: E402
from mlipstab.mdsim_backend import MDSimBackend  # noqa: E402
from mlipstab.train.crmse_loop import CRMSEConfig, run_crmse  # noqa: E402
from mlipstab.train.stage_b import StageBConfig, run_stage_b  # noqa: E402


def seed_crmse_buffer(backend, cfg, size):
    """Chains start from a rollout of the pretrained model that is allowed to fail:
    frames up to (and including) each replica's first failure, sampled uniformly."""
    sim, det = cfg["simulate"], cfg["detect"]
    every = sim["check_every_steps"]
    dev = lambda x: bond_deviation(x, backend.bonds, backend.b0) > det["bond_threshold"]
    traj, _ = backend.langevin(backend.sample_train(sim["n_replicas"]),
                               int(round(sim["max_ps"] * 1000 / sim["timestep_fs"])), every, dev, sim)
    bad = dev(traj)  # (T, R)
    first = torch.where(bad.any(0), bad.float().argmax(0), torch.tensor(len(traj) - 1, device=bad.device))
    keep = torch.arange(len(traj), device=bad.device)[:, None] <= first[None]
    pool = traj[keep]
    return pool[torch.randint(len(pool), (size,), device=pool.device)].clone()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", choices=["margin", "crmse"], default="margin")
    p.add_argument("--molecule", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    for key in ("molecule", "model", "seed"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if cfg.get("localize", {}).get("enabled"):
        raise NotImplementedError("localized energies need per-atom outputs, which StABlE's "
                                  "calculator only provides for GemNet-T")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    backend = MDSimBackend(cfg, cfg["molecule"], cfg["model"], root=ROOT)
    optimizer = backend.make_optimizer(cfg)
    ef_loss_fn = backend.make_ef_loss_fn(cfg)
    out = ROOT / cfg["results_dir"] / f"{cfg['molecule']}_{cfg['model']}_seed{cfg['seed']}"

    if args.method == "crmse":
        ccfg = CRMSEConfig(**cfg["crmse"], temperature=cfg["simulate"]["temperature"])
        buffer = seed_crmse_buffer(backend, cfg, ccfg.buffer_size)
        history = run_crmse(backend.model, optimizer, ccfg, buffer, backend.energy_fn, backend.forces,
                            ef_loss_fn, backend.sample_train, backend.val_force_mae_fn)
        used = dataclasses.asdict(ccfg)
    else:
        scfg = StageBConfig(**{**cfg["stage_b"], "device": str(backend.device)})
        history = run_stage_b(
            backend.model, optimizer, scfg,
            simulate_fn=backend.make_simulate_fn(cfg), energy_fn=backend.energy_fn,
            ef_loss_fn=ef_loss_fn, val_force_mae_fn=backend.val_force_mae_fn,
            anchor_fn=make_anchor_fn(scfg.anchor, backend.bonds, backend.b0, backend.frames("train")),
            bonds=backend.bonds, b0=backend.b0)
        used = dataclasses.asdict(scfg)

    backend.save(out / "ckpt.pt")
    (out / "history.json").write_text(json.dumps({"config": cfg, "resolved": used, "history": history}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
