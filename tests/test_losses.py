import math

import torch

from mlipstab.losses.crmse import contrastive_term, ula_step
from mlipstab.losses.margin import local_energy, local_mask, margin_loss


def test_margin_loss_value_at_margin_is_log2():
    e_fail = torch.tensor([1.0])
    e_anchor = torch.tensor([0.5])
    assert math.isclose(margin_loss(e_fail, e_anchor, margin=0.5).item(), math.log(2), rel_tol=1e-6)


def test_margin_loss_saturates_above_margin():
    loss = margin_loss(torch.tensor([10.0]), torch.tensor([0.0]), margin=0.4)
    assert loss.item() < 1e-4


def test_stop_grad_anchor_blocks_anchor_gradient():
    e_fail = torch.tensor([0.0], requires_grad=True)
    e_anchor = torch.tensor([0.0], requires_grad=True)
    margin_loss(e_fail, e_anchor, margin=0.43).backward()
    assert e_fail.grad.item() < 0  # descent raises the failure energy
    assert e_anchor.grad is None


def test_free_anchor_gets_gradient_in_ablation():
    e_fail = torch.tensor([0.0], requires_grad=True)
    e_anchor = torch.tensor([0.0], requires_grad=True)
    margin_loss(e_fail, e_anchor, margin=0.43, stop_grad_anchor=False).backward()
    assert e_anchor.grad.item() > 0  # descent would lower the anchor: the failure mode sg() prevents


def test_local_energy_and_mask():
    pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [5.0, 0, 0], [10.0, 0, 0]])
    mask = local_mask(pos, (0, 1), cutoff=4.5)
    assert mask.tolist() == [True, True, True, False]
    eps = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert local_energy(eps, mask[None]).item() == 6.0


def test_contrastive_term_sign():
    # Lower data energies / higher sample energies -> lower loss.
    assert contrastive_term(torch.tensor([0.0]), torch.tensor([1.0]), beta=1.0) < 0


def test_ula_moves_downhill_without_noise_scale():
    # Harmonic well U = 0.5 * |x|^2; with huge beta the noise vanishes.
    x = torch.full((1, 2, 3), 1.0)
    x_new = ula_step(x, lambda y: 0.5 * (y ** 2).sum(dim=(1, 2)), step_size=0.1, beta=1e12)
    assert torch.allclose(x_new, torch.full((1, 2, 3), 0.9), atol=1e-4)
