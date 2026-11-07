import pytest
import torch

from mlipstab.losses.anchors import counterfactual_anchor, make_anchor_fn, nearest_frame


def test_counterfactual_restores_only_the_worst_bond():
    bonds = torch.tensor([[0, 1], [1, 2]])
    b0 = torch.tensor([1.0, 1.0])
    x = torch.tensor([[[0.0, 0, 0], [1.6, 0, 0], [1.6, 1.05, 0]]])  # bond 0 stretched by 0.6
    y = counterfactual_anchor(x, bonds, b0)
    assert torch.isclose((y[0, 1] - y[0, 0]).norm(), torch.tensor(1.0))
    assert torch.allclose(y[0, 0] + y[0, 1], x[0, 0] + x[0, 1])  # symmetric move
    assert torch.equal(y[0, 2], x[0, 2])
    assert torch.equal(x[0, 1], torch.tensor([1.6, 0, 0]))  # input untouched


def test_counterfactual_also_fixes_compressed_bonds():
    bonds, b0 = torch.tensor([[0, 1]]), torch.tensor([1.0])
    y = counterfactual_anchor(torch.tensor([[[0.0, 0, 0], [0.5, 0, 0]]]), bonds, b0)
    assert torch.isclose((y[0, 1] - y[0, 0]).norm(), torch.tensor(1.0))


def test_nearest_frame_is_rotation_invariant():
    frames = torch.tensor([[[0.0, 0, 0], [1.0, 0, 0]], [[0.0, 0, 0], [2.0, 0, 0]]])
    rotated = torch.tensor([[[5.0, 5, 5], [5.0, 6.9, 5]]])  # length 1.9, rotated and shifted
    assert torch.equal(nearest_frame(rotated, frames)[0], frames[1])


def test_make_anchor_fn_modes():
    x_fail, x_anchor = torch.ones(1, 2, 3), torch.zeros(1, 2, 3)
    assert torch.equal(make_anchor_fn("rollout")(x_fail, x_anchor), x_anchor)
    with pytest.raises(ValueError):
        make_anchor_fn("nope")
