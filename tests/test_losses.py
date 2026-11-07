import math

import torch

from mlipstab.losses.crmse import contrastive_term, ula_step
from mlipstab.losses.margin import bond_restoring_loss, local_energy, local_mask, margin_loss


def test_margin_loss_value_at_margin_is_log2():
    e_fail = torch.tensor([1.0])
    e_anchor = torch.tensor([0.5])
    assert math.isclose(margin_loss(e_fail, e_anchor, margin=0.5).item(), math.log(2), rel_tol=1e-6)


def test_margin_loss_saturates_above_margin():
    loss = margin_loss(torch.tensor([10.0]), torch.tensor([0.0]), margin=0.4)
    assert loss.item() < 1e-4


def _offset_grad(stop_grad_anchor):
    # U(x) = a * |x|^2 + c: c shifts every energy and changes no force.
    a = torch.tensor(0.0, requires_grad=True)
    c = torch.tensor(0.0, requires_grad=True)
    U = lambda x: a * (x ** 2).sum(dim=(1, 2)) + c
    x_fail, x_anchor = torch.ones(2, 3, 3), torch.zeros(2, 3, 3)
    margin_loss(U(x_fail), U(x_anchor), 0.43, stop_grad_anchor).backward()
    return a.grad.item(), c.grad.item()


def test_default_free_anchor_gives_offset_zero_gradient():
    a_grad, c_grad = _offset_grad(stop_grad_anchor=False)
    assert a_grad < 0  # descent raises the failure relative to the anchor
    assert c_grad == 0.0  # common offset cancels


def test_stop_grad_ablation_leaks_into_offset():
    _, c_grad = _offset_grad(stop_grad_anchor=True)
    assert c_grad < 0  # descent would raise every energy: the bias the free anchor avoids


def _bond_energy(slope):
    # U = slope * r_01: slope > 0 restores a stretched bond, slope < 0 pulls it apart.
    return lambda x: slope * (x[:, 1] - x[:, 0]).norm(dim=-1)


def test_bond_restoring_loss_penalizes_wrong_sign_force():
    x = torch.tensor([[[0.0, 0, 0], [1.5, 0, 0]]])  # stretched: r = 1.5 > b0 = 1.0
    pair, b0 = torch.tensor([[0, 1]]), torch.tensor([1.0])
    good = bond_restoring_loss(x, _bond_energy(5.0), pair, b0)
    bad = bond_restoring_loss(x, _bond_energy(-5.0), pair, b0)
    assert good.item() < 0.01 < 4.9 < bad.item()


def test_bond_restoring_loss_is_differentiable_in_params():
    slope = torch.tensor(-1.0, requires_grad=True)
    x = torch.tensor([[[0.0, 0, 0], [1.5, 0, 0]]])
    bond_restoring_loss(x, lambda y: slope * (y[:, 1] - y[:, 0]).norm(dim=-1),
                        torch.tensor([[0, 1]]), torch.tensor([1.0])).backward()
    assert slope.grad.item() < 0  # descent increases the slope toward a restoring force


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
