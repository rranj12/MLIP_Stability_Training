"""Stage B for our method (and CRMSE): load a Stage A checkpoint, run the loop.

    python scripts/train_stability.py --config configs/margin/aspirin_schnet.yml
    python scripts/train_stability.py --config configs/crmse/base.yml --method crmse

The loop itself (mlipstab.train.stage_b) is implemented and unit-tested. What is
left is the MDsim glue below: four functions that wrap StABlE's model loading,
batched simulator and energy/force loss. Each TODO names the StABlE code to reuse.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/StABlE-Training"))

from mlipstab.config import load_config  # noqa: E402
from mlipstab.train.stage_b import StageBConfig, run_stage_b  # noqa: E402


def load_model_and_optimizer(cfg):
    """TODO: load the Stage A checkpoint the same way stable_training.py does
    (Trainer.load_model_from_training_session / load_pretrained_model), and
    restore Stage A's Adam state when cfg['optim']['reuse_stage_a_state'].
    Returns (model, optimizer, calculator)."""
    raise NotImplementedError


def make_simulate_fn(cfg, calculator):
    """TODO: wrap StABlE's simulator.Simulator (batched replicas, Langevin integrator
    from mdsim.md.integrator). Per rollout, under torch.no_grad():
      1. start R replicas from random training frames,
      2. integrate up to simulate.max_ps, saving positions every check_every_steps,
      3. stop early once end_phase_fail_frac of replicas have failed,
      4. call mlipstab.detect.extract_failure_pairs(traj, bonds, b0, threshold,
         lag_frames=anchor_lag_steps // check_every_steps).
    Returns simulate_fn(model) -> (x_fail, x_anchor, fail_frac)."""
    raise NotImplementedError


def make_energy_fn(calculator):
    """TODO: energy_fn(model, x) -> (B,) eV, via
    calculator.calculate_energy_force(x, retain_grad=True)[0]. For the localized
    variant pass output_individual_energies=True and use losses.margin.local_energy."""
    raise NotImplementedError


def make_ef_loss_fn(cfg, calculator):
    """TODO: energy/force MSE on a labeled batch, matching Stage A's loss weights
    (see simulator_utils.energy_force_gradient for how StABlE batches the train set).
    Also return val_force_mae_fn for the guardrail."""
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--method", choices=["margin", "crmse"], default="margin")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed

    model, optimizer, calculator = load_model_and_optimizer(cfg)
    ef_loss_fn, val_force_mae_fn = make_ef_loss_fn(cfg, calculator)
    if args.method == "crmse":
        # TODO: CRMSE loop (Algorithm 1): persistent buffer seeded from a failed rollout,
        # K ULA steps per epoch (mlipstab.losses.crmse.ula_step), loss = E/F + lam * contrastive_term.
        raise NotImplementedError
    history = run_stage_b(
        model, optimizer, StageBConfig(**cfg["stage_b"]),
        simulate_fn=make_simulate_fn(cfg, calculator),
        energy_fn=make_energy_fn(calculator),
        ef_loss_fn=ef_loss_fn, val_force_mae_fn=val_force_mae_fn,
    )
    # TODO: save model checkpoint + history under cfg["results_dir"].


if __name__ == "__main__":
    main()
