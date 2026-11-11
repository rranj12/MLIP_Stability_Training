"""Glue between this package and StABlE/MDsim (third_party/StABlE-Training).

One `MDSimBackend` per (molecule, model) wraps a Stage A checkpoint in StABlE's
batched ForceCalculator and exposes what Stage B, CRMSE and evaluation need:
energies/forces in eV, the labeled energy/force loss, and Langevin rollouts.

NequIP note (checked against nequip 0.6.1): its outputs are in physical units only
in eval() mode, and its forces carry a parameter graph only in train() mode.
StABlE never switches modes, so this class does it explicitly: eval() for MD, and
train() plus a manual rescale wherever a loss needs parameter gradients.
`_check_nequip_modes` verifies at start-up that the two paths agree.
"""

import warnings
from pathlib import Path

import numpy as np
import torch

from mlipstab.data.bonds import bond_list, nonbonded_floor
from mlipstab.detect import extract_failure_pairs, unstable
from mlipstab.md import rollout

CELL = 30.0  # Å, the dummy cubic cell prepare_rmd17.py writes


def _load_state(path, device):
    """State dict from an MDsim/OCP checkpoint, a StABlE checkpoint, or a raw NequIP .pth."""
    sd = torch.load(path, map_location=device)
    for key in ("state_dict", "model_state"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
    if all(k.startswith("module.") for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    return sd


class MDSimBackend:
    def __init__(self, cfg, molecule, model_type, root=".", checkpoint=None, device=None):
        import ase
        from nequip.data import AtomicData
        from nequip.utils.torch_geometric import Batch
        from mdsim.md.calculator import ForceCalculator

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_type, self.is_nequip = model_type, model_type == "nequip"
        root, d = Path(root), cfg["data"]
        base = root / d["src"] / d["name"] / molecule / d["size"]
        self.data = {s: dict(np.load(base / s / "nequip_npz.npz")) for s in ("train", "val", "test")}
        z = self.data["train"]["z"]
        self.n_atoms = len(z)

        # ---- model: architecture + Stage A weights, optionally overwritten by `checkpoint`
        pre = root / cfg["pretrained"]["dir"] / model_type
        matches = sorted(pre.glob(f"{d['name']}-{molecule}_{d['size']}_*{model_type}"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one Stage A run matching {pre}/{d['name']}-{molecule}_{d['size']}_*{model_type}, found {matches}")
        self.stage_a_dir = matches[0]
        if self.is_nequip:
            from nequip.train.trainer import Trainer
            self.model, nequip_cfg = Trainer.load_model_from_training_session(
                str(self.stage_a_dir), model_name="best_model.pth", device=self.device)
            self.model_config, r_max_key = {"model": nequip_cfg}, "r_max"
            self.rescale_layers, layer = [], self.model  # outermost first, as in simulator.py
            while hasattr(layer, "unscale"):
                self.rescale_layers.append(layer)
                layer = getattr(layer, "model", None)
        else:
            from mdsim.models.load_models import load_pretrained_model
            self.model, _, self.model_config = load_pretrained_model(
                model_type, path=str(self.stage_a_dir), device=self.device)
            r_max_key = "cutoff"
        if checkpoint is not None:
            self.model.load_state_dict(_load_state(checkpoint, self.device))
        for p in self.model.parameters():
            p.requires_grad = True

        # ---- StABlE's batched calculator (mirrors Simulator.__init__)
        atoms = ase.Atoms(numbers=z, positions=self.data["train"]["R"][0], cell=np.eye(3) * CELL, pbc=True)
        self.masses = torch.tensor(atoms.get_masses(), dtype=torch.float32, device=self.device).view(1, -1, 1)
        batch = Batch.from_data_list(
            [AtomicData.from_ase(atoms=atoms, r_max=self.model_config["model"][r_max_key])])
        batch["natoms"] = torch.tensor([self.n_atoms], device=self.device)
        batch["cell"] = batch["cell"].to(self.device)
        batch["atomic_numbers"] = batch["atomic_numbers"].squeeze().long().to(self.device)
        if self.is_nequip:
            type_names = nequip_cfg["type_names"]
            type_ids = [type_names.index(s) for s in atoms.get_chemical_symbols()]
            batch = {k: v.to(self.device) for k, v in AtomicData.to_AtomicDataDict(batch).items()}
            batch["atom_types"] = torch.tensor(type_ids, device=self.device).unsqueeze(-1)
            del batch["ptr"], batch["atomic_numbers"]
        self.calc = ForceCalculator(
            self.model, model_type, self.model_config, r_max_key, self.n_atoms,
            torch.tensor(z, dtype=torch.long, device=self.device), batch, False,
            torch.eye(3, device=self.device) * CELL, self.device)

        # ---- topology from the training frames
        bonds, b0 = bond_list(z, self.data["train"]["R"], scale=cfg.get("detect", {}).get("bond_scale", 1.2))
        self.bonds = torch.tensor(bonds, dtype=torch.long, device=self.device)
        self.b0 = torch.tensor(b0, dtype=torch.float32, device=self.device)
        self._bonds_np = bonds
        self.rng = np.random.default_rng(cfg.get("seed", 0))
        if self.is_nequip:
            self._check_nequip_modes()

    # ------------------------------------------------------------------ energies and forces
    def frames(self, split, key="R"):
        return torch.tensor(self.data[split][key], dtype=torch.float32, device=self.device)

    def energy_force(self, x, param_grad=False):
        """x (B, N, 3) Å -> (energy (B,) eV, forces (B, N, 3) eV/Å).

        param_grad=True keeps the graph to the model parameters (and to x)."""
        if self.is_nequip:
            self.model.train(param_grad)
            self.calc.atoms_batch["natoms"] = torch.full((x.shape[0],), self.n_atoms, device=self.device)
        e, f = self.calc.calculate_energy_force(x, retain_grad=param_grad)
        e = e.reshape(-1)
        if self.is_nequip and param_grad:  # train mode skips RescaleOutput; apply it by hand
            for layer in reversed(self.rescale_layers):
                if getattr(layer, "scale_by", None) is not None:
                    if "total_energy" in layer.scale_keys:
                        e = e * layer.scale_by
                    if "forces" in layer.scale_keys:
                        f = f * layer.scale_by
                if getattr(layer, "shift_by", None) is not None:
                    if list(layer.shift_keys) != ["total_energy"]:
                        raise NotImplementedError(f"unsupported NequIP shift keys {layer.shift_keys}")
                    e = e + layer.shift_by
        return e, f

    def calculate_energy_force(self, radii, retain_grad=False):
        """Calculator interface for StABlE's integrators (MD: physical units, no graph)."""
        return self.energy_force(radii, param_grad=retain_grad)

    def forces(self, x):
        return self.energy_force(x)[1]

    def _check_nequip_modes(self):
        x = self.frames("train")[:4]
        e0, f0 = self.energy_force(x.clone())
        e1, f1 = self.energy_force(x.clone(), param_grad=True)
        if not (torch.allclose(e0, e1.detach(), rtol=1e-4, atol=1e-4)
                and torch.allclose(f0, f1.detach(), rtol=1e-3, atol=1e-4)):
            raise RuntimeError("NequIP train-mode outputs + manual rescale do not match eval-mode "
                               "outputs; check the installed nequip version against mdsim_backend.py")

    # ------------------------------------------------------------------ callables for the loops
    def energy_fn(self, model, x):
        return self.energy_force(x, param_grad=True)[0]

    def make_ef_loss_fn(self, cfg):
        """Labeled energy/force loss with Stage A's weights (override with cfg['ef_loss']).

        MDsim models: w_E * loss(E) + w_F * loss(F), loss type and weights from the Stage A config.
        NequIP: w_E * per-atom energy MSE + w_F * force MSE, computed in eV rather than
        NequIP's normalized units (same minimizer; keeps lambda comparable across models)."""
        R, E, F = (self.frames("train", k) for k in ("R", "E", "F"))
        o = cfg.get("ef_loss", {})
        if self.is_nequip:
            lc = self.model_config["model"].get("loss_coeffs", {})
            te = lc.get("total_energy", 1.0)
            w_e = o.get("energy_weight", float(te[0] if isinstance(te, (list, tuple)) else te))
            w_f = o.get("force_weight", float(lc.get("forces", 1000.0)))
            e_loss = lambda e, ref: (((e - ref) / self.n_atoms) ** 2).mean()
            f_loss = torch.nn.functional.mse_loss
        else:
            optim = self.model_config.get("optim", {})
            w_e = o.get("energy_weight", optim.get("energy_coefficient", 1.0))
            w_f = o.get("force_weight", optim.get("force_coefficient", 30.0))
            pick = lambda name: torch.nn.functional.l1_loss if name in ("l1", "mae") else torch.nn.functional.mse_loss
            e_loss, f_loss = pick(optim.get("loss_energy", "mse")), pick(optim.get("loss_force", "mse"))

        def ef_loss_fn(model, batch_size):
            idx = torch.as_tensor(self.rng.choice(len(R), batch_size, replace=False), device=self.device)
            e, f = self.energy_force(R[idx], param_grad=True)
            return w_e * e_loss(e, E[idx].reshape(-1)) + w_f * f_loss(f, F[idx])

        return ef_loss_fn

    def errors(self, split, batch_size=50):
        """(energy MAE eV, force MAE eV/Å) on a labeled split."""
        R, E, F = (self.frames(split, k) for k in ("R", "E", "F"))
        e_err = f_err = 0.0
        for s in range(0, len(R), batch_size):
            e, f = self.energy_force(R[s:s + batch_size].clone())
            e_err += (e - E[s:s + batch_size].reshape(-1)).abs().sum().item()
            f_err += (f - F[s:s + batch_size]).abs().sum().item()
        return e_err / len(R), f_err / F.numel()

    def val_force_mae_fn(self, model):
        return self.errors("val")[1]

    def sample_train(self, batch_size):
        idx = self.rng.choice(len(self.data["train"]["R"]), batch_size, replace=False)
        return self.frames("train")[torch.as_tensor(idx, device=self.device)]

    def make_optimizer(self, cfg):
        o = cfg["optim"]
        opt = getattr(torch.optim, o["name"])(self.model.parameters(), lr=o["lr"])
        if o.get("reuse_stage_a_state"):
            try:
                if self.is_nequip:
                    state = torch.load(self.stage_a_dir / "trainer.pth", map_location=self.device)["state_dict"]["optim"]
                else:
                    state = torch.load(self.stage_a_dir / "checkpoints" / "best_checkpoint.pt",
                                       map_location=self.device)["optimizer"]
                opt.load_state_dict(state)
            except Exception as err:  # different optimizer class, missing key, ...
                warnings.warn(f"could not reuse Stage A optimizer state ({err!r}); starting fresh")
                opt = getattr(torch.optim, o["name"])(self.model.parameters(), lr=o["lr"])
            for g in opt.param_groups:
                g["lr"] = o["lr"]
        return opt

    # ------------------------------------------------------------------ MD
    def langevin(self, x0, n_steps, save_every, failed_fn, sim, stop_frac=None,
                 store_device=None, temperature=None, friction=None):
        """Langevin rollout with StABlE's integrator. Returns (traj (T, R, N, 3), failed (R,))."""
        from ase import units
        from mdsim.common.utils import initialize_velocities
        from mdsim.md.integrator import Langevin

        temperature = sim["temperature"] if temperature is None else temperature
        friction = sim["friction_per_ps"] if friction is None else friction
        n = x0.shape[0]
        integrator = Langevin(self, self.masses, n, self.n_atoms, {
            "timestep": sim["timestep_fs"], "temperature": temperature,
            "integrator_config": {"gamma": friction}}, self.device)
        v0 = initialize_velocities(self.n_atoms, self.masses, temperature * units.kB, n).to(self.device)

        def step_fn(x, v, f):
            x, v, _, f, _ = integrator.step(x, v, f)
            return x, v, f

        return rollout(self.forces, step_fn, x0, v0, n_steps, save_every, failed_fn,
                       stop_frac=stop_frac, store_device=store_device)

    def make_simulate_fn(self, cfg):
        """simulate_fn(model) -> (x_fail, x_anchor, fail_frac) for mlipstab.train.stage_b."""
        sim, det = cfg["simulate"], cfg["detect"]
        every = sim["check_every_steps"]
        n_steps = int(round(sim["max_ps"] * 1000 / sim["timestep_fs"]))
        failed_fn = lambda x: unstable(x, self.bonds, self.b0, det["bond_threshold"])

        def simulate_fn(model):
            traj, failed = self.langevin(self.sample_train(sim["n_replicas"]), n_steps, every,
                                         failed_fn, sim, stop_frac=sim["end_phase_fail_frac"])
            _, _, x_fail, x_anchor = extract_failure_pairs(
                traj, self.bonds, self.b0, det["bond_threshold"], det["anchor_lag_steps"] // every)
            return x_fail, x_anchor, failed.float().mean().item()

        return simulate_fn

    def nonbonded(self, tol):
        pairs, floor = nonbonded_floor(self.data["train"]["R"], self._bonds_np, tol=tol)
        return (torch.tensor(pairs, dtype=torch.long, device=self.device),
                torch.tensor(floor, dtype=torch.float32, device=self.device))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict()}, path)
