"""Targeted, bounded contrastive (margin) stability loss.

    L_stab = mean softplus( m - [U(x_fail) - sg(U(x_anchor))] )

The stop-gradient sg() on the anchor is the default: without it the model can
satisfy the margin by lowering U(x_anchor), an unlabeled state, and dig a new
fake valley next to the physical basin. `stop_grad_anchor=False` is the ablation.
"""

import torch
import torch.nn.functional as F


def margin_loss(e_fail, e_anchor, margin, stop_grad_anchor=True):
    """e_fail, e_anchor: (B,) energies in eV. margin: eV (e.g. 10 * kT(500))."""
    if stop_grad_anchor:
        e_anchor = e_anchor.detach()
    return F.softplus(margin - (e_fail - e_anchor)).mean()


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
