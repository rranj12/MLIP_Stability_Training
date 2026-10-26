import numpy as np
import torch

from mlipstab.buffer import FailureBuffer
from mlipstab.eval.observables import dihedral, free_energy_profile
from mlipstab.eval.stability import first_failure_time, kaplan_meier, median_survival
from mlipstab.train.stage_b import StageBConfig, run_stage_b


def test_first_failure_time_censors_survivors():
    unstable = np.array([[0, 0], [1, 0], [1, 0]], dtype=bool)
    t, obs = first_failure_time(unstable, dt_ps=10.0)
    assert t.tolist() == [10.0, 30.0] and obs.tolist() == [True, False]


def test_kaplan_meier_hand_example():
    # 4 replicas: failures at 1 and 2, two censored at 3 -> S = 3/4 then 3/4 * 2/3 = 1/2
    t, s = kaplan_meier([1, 2, 3, 3], [True, True, False, False])
    assert t.tolist() == [1, 2] and np.allclose(s, [0.75, 0.5])
    assert median_survival(t, s) == 2


def test_dihedral_known_angles():
    trans = np.array([[[1.0, 1, 0], [0, 0, 0], [0, 0, 1], [-1, -1, 1]]])
    cis = np.array([[[1.0, 0, 0], [0, 0, 0], [0, 0, 1], [1, 0, 1]]])
    assert np.isclose(abs(dihedral(trans, 0, 1, 2, 3)[0]), 180)
    assert np.isclose(dihedral(cis, 0, 1, 2, 3)[0], 0)


def test_free_energy_profile_min_is_zero():
    angles = np.random.default_rng(0).normal(0, 30, 10000)
    _, f = free_energy_profile(angles, kT=1.0, bins=36)
    assert np.isclose(np.nanmin(f), 0)


def test_buffer_is_fifo():
    buf = FailureBuffer(capacity=3)
    x = torch.arange(5.0).view(5, 1, 1).expand(5, 2, 3)
    buf.push(x, x)
    assert len(buf) == 3 and buf.items[0][0][0, 0].item() == 2.0


class Toy(torch.nn.Module):
    """U(x) = a * |x|^2 summed per configuration."""

    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.0))


def test_stage_b_raises_failure_energy_and_stops():
    model = Toy()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    energy = lambda m, x: m.a * (x ** 2).sum(dim=(1, 2))
    fail = torch.ones(4, 2, 3)  # far from origin
    anchor = torch.zeros(4, 2, 3)
    calls = {"n": 0}

    def simulate(m):
        calls["n"] += 1
        frac = 0.5 if calls["n"] == 1 else 0.0
        return fail, anchor, frac

    cfg = StageBConfig(n_cycles=3, steps_per_cycle=300, margin_kT=10.0, lam=1.0)
    hist = run_stage_b(model, opt, cfg, simulate, energy,
                       ef_loss_fn=lambda m, b: torch.zeros(()),
                       val_force_mae_fn=lambda m: 1.0, log=lambda *_: None)
    gap = energy(model, fail[:1]) - energy(model, anchor[:1])
    assert gap.item() > 10.0 * 8.617e-5 * 500  # failure now sits above the margin
    assert len(hist) == 2  # second rollout has no failures -> stop


def test_stage_b_guardrail_restores_weights():
    model = Toy()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    maes = iter([1.0, 2.0])  # baseline, then a 100% jump
    cfg = StageBConfig(n_cycles=3, steps_per_cycle=5)
    hist = run_stage_b(model, opt, cfg,
                       simulate_fn=lambda m: (torch.ones(2, 2, 3), torch.zeros(2, 2, 3), 0.5),
                       energy_fn=lambda m, x: m.a * (x ** 2).sum(dim=(1, 2)),
                       ef_loss_fn=lambda m, b: torch.zeros(()),
                       val_force_mae_fn=lambda m: next(maes), log=lambda *_: None)
    assert hist[-1]["guardrail_tripped"] and model.a.item() == 0.0
