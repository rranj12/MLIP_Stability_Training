"""Stage B: alternate simulation (find failures) and learning (margin loss).

The loop is backend-agnostic: everything that touches MDsim / the model zoo is
injected as a callable, so the control flow is unit-testable on CPU and the
same loop drives SchNet, NequIP or MACE. See scripts/train_stability.py for
the MDsim glue.

    simulate_fn(model)            -> (x_fail (B,N,3), x_anchor (B,N,3), fail_frac float)
                                     run under torch.no_grad(); B may be 0
    energy_fn(model, x)           -> (B,) energies in eV for positions x (B,N,3)
    ef_loss_fn(model, batch_size) -> scalar energy/force loss on a labeled batch
    val_force_mae_fn(model)       -> float, validation force MAE (guardrail)
"""

import copy
from dataclasses import dataclass

import torch

from mlipstab.buffer import FailureBuffer
from mlipstab.losses.margin import margin_loss
from mlipstab.units import kT


@dataclass
class StageBConfig:
    n_cycles: int = 6
    steps_per_cycle: int = 500
    batch_labeled: int = 8
    batch_pairs: int = 16
    lam: float = 0.1
    margin_kT: float = 10.0
    temperature: float = 500.0
    stop_grad_anchor: bool = True
    buffer_capacity: int = 2000
    stab_loss_tol: float = 1e-3  # end the learning phase early once failures are suppressed
    guardrail_rel: float = 0.10  # stop if val force MAE rises > 10% over Stage A
    stop_fail_frac: float = 0.05  # stop once < 5% of replicas fail in a rollout
    grad_clip: float = 1.0
    device: str = "cpu"


def run_stage_b(model, optimizer, cfg, simulate_fn, energy_fn, ef_loss_fn,
                val_force_mae_fn, log=print):
    buffer = FailureBuffer(cfg.buffer_capacity)
    margin = cfg.margin_kT * kT(cfg.temperature)
    base_mae = val_force_mae_fn(model)
    history = []

    for cycle in range(cfg.n_cycles):
        x_fail, x_anchor, fail_frac = simulate_fn(model)
        if x_fail.shape[0] > 0:
            buffer.push(x_fail, x_anchor, cycle=cycle)
        record = {"cycle": cycle, "fail_frac": fail_frac, "buffer": len(buffer)}
        history.append(record)
        log(record)
        if fail_frac < cfg.stop_fail_frac:
            log("failure fraction below target; stopping")
            break
        if len(buffer) == 0:
            continue

        last_good = copy.deepcopy(model.state_dict())
        for step in range(cfg.steps_per_cycle):
            xf, xa, _ = buffer.sample(cfg.batch_pairs, device=cfg.device)
            e_fail = energy_fn(model, xf)
            if cfg.stop_grad_anchor:
                with torch.no_grad():
                    e_anchor = energy_fn(model, xa)
            else:
                e_anchor = energy_fn(model, xa)
            l_stab = margin_loss(e_fail, e_anchor, margin, cfg.stop_grad_anchor)
            loss = ef_loss_fn(model, cfg.batch_labeled) + cfg.lam * l_stab
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if l_stab.item() < cfg.stab_loss_tol:
                break
        record.update(steps=step + 1, l_stab=l_stab.item())

        mae = val_force_mae_fn(model)
        record["val_force_mae"] = mae
        if mae > (1 + cfg.guardrail_rel) * base_mae:
            log(f"guardrail: val force MAE {mae:.4f} > {1 + cfg.guardrail_rel:.2f} x "
                f"{base_mae:.4f}; restoring the previous cycle's weights and stopping")
            model.load_state_dict(last_good)
            record["guardrail_tripped"] = True
            break

    return history
