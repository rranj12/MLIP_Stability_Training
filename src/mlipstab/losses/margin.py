"""Targeted, bounded contrastive (margin) stability loss.

    L_stab = mean softplus( m - [U(x_fail) - U(x_anchor)] )

Gradients flow through BOTH energies by default, so the penalty gradient is
sigma(m - gap) * [dU(x_fail)/dtheta - dU(x_anchor)/dtheta] and any shift common
to all energies (e.g. a readout bias) cancels. With a stop-gradient on the anchor
the gradient is dU(x_fail)/dtheta alone: it keeps pushing a global energy offset
up, which changes no force but fights the energy term of L_EF and biases energies.
`stop_grad_anchor=True` is kept only as an ablation.
"""

import torch
import torch.nn.functional as F


def margin_loss(e_fail, e_anchor, margin, stop_grad_anchor=False):
    """e_fail, e_anchor: (B,) energies in eV. margin: eV (e.g. 10 * kT(500))."""
    if stop_grad_anchor:
        e_anchor = e_anchor.detach()
    return F.softplus(margin - (e_fail - e_anchor)).mean()


def bond_restoring_loss(x, energy_fn, bond_pairs, b0, scale=1.0):
    """Hinge requiring a restoring force along each flagged bond.

    The margin loss only fixes energy values at two points; MD responds to forces.
    For a stretched bond (r > b0) we want dU/dr > 0 (atoms pulled back together);
    for a compressed bond, dU/dr < 0. dU/dr is the derivative along a symmetric
    stretch of the pair. Uses double backprop (create_graph=True).

    x: (B, N, 3); bond_pairs: (B, 2) long; b0: (B,) Å; scale: eV/Å.
    """
    x = x.detach().requires_grad_(True)
    grad = torch.autograd.grad(energy_fn(x).sum(), x, create_graph=True)[0]
    rows = torch.arange(x.shape[0], device=x.device)
    i, j = bond_pairs[:, 0], bond_pairs[:, 1]
    rij = x[rows, j] - x[rows, i]
    r = rij.norm(dim=-1)
    rhat = rij / r[:, None]
    dU_dr = 0.5 * ((grad[rows, j] - grad[rows, i]) * rhat).sum(-1)
    sign = torch.sign(r - b0).detach()
    return F.softplus(-sign * dU_dr / scale).mean()


def local_energy(atom_energies, mask):
    """Localized variant: sum per-atom energies eps_i over the masked atoms.

    atom_energies: (B, N); mask: (B, N) bool (broken bond's atoms + neighbors).
    """
    return (atom_energies * mask).sum(dim=-1)


def local_mask(pos, bond, cutoff):
    """Atoms within `cutoff` Å of either atom of the broken bond. pos (N, 3), bond (i, j)."""
    i, j = bond
    d = torch.minimum((pos - pos[i]).norm(dim=-1), (pos - pos[j]).norm(dim=-1))
    return d < cutoff
