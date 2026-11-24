"""Evaluate any checkpoint (baseline, StABlE, CRMSE, ours) under one protocol.

    python scripts/evaluate.py --config configs/eval/rmd17_500K.yml \
        --molecule aspirin --model schnet --checkpoint <path> --tag margin_seed0

--checkpoint is a weights file (ours: ckpt.pt; StABlE: its ckpt.pt / .pth); omit it to
evaluate the Stage A model. Outputs under results/eval/<tag>/:
    stability.npz   per-replica first-failure time + censoring flag (bond-only and combined)
    traj.npy        positions every save_every_fs; frames from a replica's bond failure on are NaN
    metrics.json    median tau, fraction stable at 100/300 ps, E/F MAE,
                    h(r) error, dihedral F(phi) error, NVE drift
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/StABlE-Training"))

from mlipstab.config import load_config  # noqa: E402
from mlipstab.data.bonds import frame_chunks, pairwise_distances  # noqa: E402
from mlipstab.detect import bond_deviation, nonbonded_collision  # noqa: E402
from mlipstab.eval.observables import dihedral, free_energy_profile  # noqa: E402
from mlipstab.eval.stability import first_failure_time, kaplan_meier, median_survival  # noqa: E402
from mlipstab.mdsim_backend import MDSimBackend  # noqa: E402
from mlipstab.units import EV_TO_KCAL_MOL, kT  # noqa: E402


def hist_density(coords, bins):
    """h(r) over many frames without holding every distance in memory."""
    counts = np.zeros(len(bins) - 1)
    i, j = np.triu_indices(coords.shape[1], k=1)
    for c in frame_chunks(coords):
        counts += np.histogram(pairwise_distances(c)[:, i, j], bins=bins)[0]
    return counts / (counts.sum() * np.diff(bins))


def survival(times, observed, length_ps):
    median = median_survival(*kaplan_meier(times, observed))
    out = {"median_tau_ps": median if np.isfinite(median) else None,  # None: fewer than half failed
           "mean_tau_ps": float(times.mean())}  # censored at the run length, as StABlE reports it
    for t in (100, 300):
        if t <= length_ps:
            out[f"frac_stable_{t}ps"] = float((times >= t).mean())
    return out


def nve_drift(backend, x0, sim, nve):
    """Mean |slope| of total energy per atom (meV/atom/ps) over short NVE runs."""
    from ase import units
    from mdsim.common.utils import initialize_velocities
    from mdsim.md.integrator import Langevin

    n, every = x0.shape[0], 100
    integ = Langevin(backend, backend.masses, n, backend.n_atoms, {
        "timestep": sim["timestep_fs"], "temperature": sim["temperature"],
        "integrator_config": {"gamma": 0.0}}, backend.device)  # gamma = 0 -> no friction, no noise
    x = x0.clone()
    v = initialize_velocities(backend.n_atoms, backend.masses, sim["temperature"] * units.kB, n).to(backend.device)
    f, etot = backend.forces(x), []
    with torch.no_grad():
        for step in range(int(nve["length_ps"] * 1000 / sim["timestep_fs"])):
            x, v, e, f, _ = integ.step(x, v, f)
            if step % every == 0:
                ke = 0.5 * (backend.masses * v ** 2).sum(dim=(1, 2))
                etot.append((e.reshape(-1) + ke).cpu().numpy())
    etot = np.array(etot)  # (T, n)
    t_ps = np.arange(len(etot)) * every * sim["timestep_fs"] / 1000
    ok = np.isfinite(etot).all(axis=0)
    slopes = np.polyfit(t_ps, etot[:, ok], 1)[0]
    return float(np.abs(slopes).mean() / backend.n_atoms * 1000) if ok.any() else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--molecule", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    cfg = load_config(args.config)
    sim, crit = cfg["simulate"], cfg["criteria"]
    if args.temperature is not None:
        sim["temperature"] = args.temperature
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg["seed"] = args.seed
    backend = MDSimBackend(cfg, args.molecule, args.model, root=ROOT, checkpoint=args.checkpoint)
    out = ROOT / "results" / "eval" / args.tag
    out.mkdir(parents=True, exist_ok=True)

    # 1. accuracy on the labeled split
    e_mae, f_mae = backend.errors(cfg["accuracy"]["split"])
    metrics = {"energy_mae_kcal_mol": e_mae * EV_TO_KCAL_MOL, "force_mae_kcal_mol_A": f_mae * EV_TO_KCAL_MOL,
               "temperature": sim["temperature"], "checkpoint": args.checkpoint}

    # 2. Langevin MD from the first n_replicas test frames (same starts for every model)
    # A replica is restarted only when it fails the bond criterion of Fu et al. (or goes
    # non-finite), so the bond-only failure time is not cut short by a non-bonded collision.
    failed_fn = lambda x: bond_deviation(x, backend.bonds, backend.b0) > crit["bond_threshold"]
    every = int(round(sim["save_every_fs"] / sim["timestep_fs"]))
    n_steps = int(round(sim["length_ps"] * 1000 / sim["timestep_fs"]))
    x0 = backend.frames(cfg["data"]["start_split"])[:sim["n_replicas"]]
    traj, _ = backend.langevin(x0, n_steps, every, failed_fn, sim, stop_frac=1.0, store_device="cpu")

    # 3-4. first-failure times. The run stops early once every replica has failed, so
    # survivors are censored at length_ps only if the run went the full length.
    dt_ps = sim["save_every_fs"] / 1000
    bonds, b0 = backend.bonds.cpu(), backend.b0.cpu()
    nb_pairs, nb_floor = (t.cpu() for t in backend.nonbonded(crit["nonbonded_tol"]))
    # frames per chunk: non-bonded pairs grow with atoms squared (190 for aspirin, 67,710 at 370 atoms)
    chunks = traj.split(max(1, 2 ** 25 // (traj.shape[1] * len(nb_pairs))))
    bad_bond = torch.cat([(bond_deviation(c, bonds, b0) > crit["bond_threshold"])
                          | ~torch.isfinite(c).all(dim=(2, 3)) for c in chunks]).numpy()
    collided = torch.cat([nonbonded_collision(c, nb_pairs, nb_floor) for c in chunks]).numpy()
    t_bond, obs_bond = first_failure_time(bad_bond, dt_ps)  # Fu et al. criterion
    # combined: the first bond failure or non-bonded collision (never later than the bond-only time)
    times, observed = first_failure_time(bad_bond | collided, dt_ps)
    np.savez(out / "stability.npz", times_ps=times, observed=observed,
             times_bond_only_ps=t_bond, observed_bond_only=obs_bond)
    metrics["combined"] = survival(times, observed, sim["length_ps"])
    metrics["bond_only"] = survival(t_bond, obs_bond, sim["length_ps"])

    # 5. observables on stable frames only (bond criterion, as in Fu et al.), against held-out DFT frames
    frames = traj.numpy()
    first = np.round(t_bond / dt_ps).astype(int)
    frames[np.arange(len(frames))[:, None] >= first[None]] = np.nan
    np.save(out / "traj.npy", frames.astype(np.float32))
    stable = frames[np.isfinite(frames).all(axis=(2, 3))]  # (n_stable, N, 3)
    ref = backend.data[cfg["observables"].get("reference_split", "test")]["R"]
    r = cfg["observables"]["rdf_bins"]
    bins = np.arange(r["start"], r["stop"] + r["step"] / 2, r["step"])
    metrics["n_stable_frames"] = int(len(stable))
    if len(stable):
        metrics["hr_mae"] = float(np.abs(hist_density(stable, bins) - hist_density(ref, bins)).sum() * r["step"])
        idx = cfg["observables"]["dihedrals"].get(args.molecule)
        if idx:
            kt = kT(sim["temperature"]) * EV_TO_KCAL_MOL
            _, f_md = free_energy_profile(dihedral(stable, *idx), kt)
            _, f_ref = free_energy_profile(dihedral(ref, *idx), kt)
            both = np.isfinite(f_md) & np.isfinite(f_ref)
            metrics["dihedral_F_mae_kcal_mol"] = float(np.abs(f_md - f_ref)[both].mean()) if both.any() else None
            np.savez(out / "dihedral.npz", f_md=f_md, f_ref=f_ref)

    if cfg.get("nve"):
        metrics["nve_drift_meV_atom_ps"] = nve_drift(backend, x0[:cfg["nve"]["n_replicas"]], sim, cfg["nve"])

    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
