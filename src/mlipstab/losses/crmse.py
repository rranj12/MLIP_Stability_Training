"""CRMSE baseline (Tzivrailis et al., arXiv:2606.31660). Code is not released;
this is a reimplementation from the paper (Eq. 8-10, Algorithm 1, App. D-E).

    L_CR = beta * mean U(x_data) - beta * mean U(x_neg)

Negatives are persistent ULA chains in a 1,000-configuration replay buffer,
seeded from a failed simulation of the pretrained model; each epoch every chain
takes K = 200 ULA steps. Paper settings: lambda = 0.015, 200 epochs, batch 16,
Adam lr 1e-4 continuing the pretraining optimizer state.
"""

import torch


def contrastive_term(e_pos, e_neg, beta):
    """e_pos: energies of training frames; e_neg: energies of chain samples (eV)."""
    return beta * e_pos.mean() - beta * e_neg.mean()


def ula_step(x, energy_fn, step_size, beta, precond=None, generator=None):
    """One unadjusted Langevin step: x - eta * H grad U + sqrt(2 eta H / beta) * xi.

    x: (B, N, 3). precond: optional (N,) per-atom weights H (the paper uses
    mass-based weights, App. B tables II-III). Returns a detached tensor.
    """
    x = x.detach().requires_grad_(True)
    grad = torch.autograd.grad(energy_fn(x).sum(), x)[0]
    h = 1.0 if precond is None else precond.view(1, -1, 1)
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return (x - step_size * h * grad + (2 * step_size * h / beta) ** 0.5 * noise).detach()
