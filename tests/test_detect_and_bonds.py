import numpy as np
import torch

from mlipstab.data.bonds import bond_list, nonbonded_floor
from mlipstab.detect import bond_deviation, extract_failure_pairs, nonbonded_collision


def water(n=5, oh=0.96):
    # O at origin, two H at ~104.5 degrees, small jitter across frames.
    rng = np.random.default_rng(0)
    ang = np.radians(104.5)
    base = np.array([[0, 0, 0], [oh, 0, 0], [oh * np.cos(ang), oh * np.sin(ang), 0]])
    return np.array([8, 1, 1]), base + 0.01 * rng.standard_normal((n, 3, 3))


def test_bond_list_finds_oh_bonds_only():
    z, coords = water()
    bonds, b0 = bond_list(z, coords)
    assert sorted(map(tuple, bonds.tolist())) == [(0, 1), (0, 2)]
    assert np.allclose(b0, 0.96, atol=0.02)


def test_nonbonded_floor_is_hh_pair():
    z, coords = water()
    bonds, _ = bond_list(z, coords)
    pairs, floor = nonbonded_floor(coords, bonds, tol=0.5)
    assert pairs.tolist() == [[1, 2]]
    assert 0.9 < floor[0] < 1.1  # H-H ~1.52 Å minus 0.5


def test_bond_deviation_and_collision():
    bonds = torch.tensor([[0, 1]])
    pos = torch.tensor([[[0.0, 0, 0], [1.5, 0, 0], [0.2, 0, 0]]])
    assert torch.isclose(bond_deviation(pos, bonds, torch.tensor([1.0])), torch.tensor([0.5])).all()
    assert nonbonded_collision(pos, torch.tensor([[0, 2]]), torch.tensor([0.5])).item()


def test_extract_failure_pairs_takes_first_failure_and_lagged_anchor():
    T, R = 6, 3
    bonds, b0 = torch.tensor([[0, 1]]), torch.tensor([1.0])
    traj = torch.zeros(T, R, 2, 3)
    traj[..., 1, 0] = 1.0  # all bonds at equilibrium
    traj[4:, 1, 1, 0] = 1.4  # replica 1 fails from frame 4 on
    traj[5, 2, 1, 0] = 1.3  # replica 2 fails at frame 5
    reps, t_fail, x_fail, x_anchor = extract_failure_pairs(traj, bonds, b0, 0.25, lag_frames=3)
    assert reps.tolist() == [1, 2]
    assert t_fail.tolist() == [4, 5]
    assert torch.isclose(x_fail[0, 1, 0], torch.tensor(1.4)) and x_anchor[0, 1, 0].item() == 1.0
