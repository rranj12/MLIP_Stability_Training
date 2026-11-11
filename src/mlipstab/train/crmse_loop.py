"""CRMSE training loop (Tzivrailis et al., arXiv:2606.31660, Algorithm 1), backend-agnostic.

    energy_fn(model, x)           -> (B,) energies in eV, differentiable w.r.t. parameters
    force_fn(x)                   -> (B, N, 3) forces, no parameter graph (used by ULA)
    ef_loss_fn(model, batch_size) -> scalar energy/force loss on a labeled batch
    sample_data(batch_size)       -> (B, N, 3) training positions (the positive phase)
    val_force_mae_fn(model)       -> float, logged only
"""

from dataclasses import dataclass

import torch

from mlipstab.losses.crmse import contrastive_term, ula_step
from mlipstab.units import kT


@dataclass
class CRMSEConfig:
    lam: float = 0.015
    epochs: int = 200
    batch_size: int = 16
    buffer_size: int = 1000  # informational; the buffer is whatever `buffer` holds
    ula_steps_per_epoch: int = 200
    step_size: float = None  # must be tuned for eV/Å
    seed_from: str = "failed_rollout"
    temperature: float = 500.0
    steps_per_epoch: int = 59  # 950 // 16
    ula_chunk: int = 250  # chains advanced per force call
    grad_clip: float = None
    log_every: int = 10


def run_crmse(model, optimizer, cfg, buffer, energy_fn, force_fn, ef_loss_fn,
              sample_data, val_force_mae_fn, log=print):
    """buffer: (M, N, 3) persistent chains, updated in place. Returns the history."""
    if cfg.step_size is None:
        raise ValueError("crmse.step_size is not set; tune it for eV/Å (see configs/crmse/base.yml)")
    beta = 1.0 / kT(cfg.temperature)
    history = []
    for epoch in range(cfg.epochs):
        for _ in range(cfg.ula_steps_per_epoch):
            for s in range(0, buffer.shape[0], cfg.ula_chunk):
                buffer[s:s + cfg.ula_chunk] = ula_step(
                    buffer[s:s + cfg.ula_chunk], None, cfg.step_size, beta, force_fn=force_fn)
        for _ in range(cfg.steps_per_epoch):
            x_neg = buffer[torch.randint(buffer.shape[0], (cfg.batch_size,), device=buffer.device)]
            l_cr = contrastive_term(energy_fn(model, sample_data(cfg.batch_size)),
                                    energy_fn(model, x_neg), beta)
            loss = ef_loss_fn(model, cfg.batch_size) + cfg.lam * l_cr
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        record = {"epoch": epoch, "l_cr": l_cr.item(), "loss": loss.item()}
        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            record["val_force_mae"] = val_force_mae_fn(model)
        history.append(record)
        log(record)
    return history
