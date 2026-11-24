import torch

from mlipstab.detect import bond_deviation, extract_failure_pairs
from mlipstab.losses.crmse import ula_step
from mlipstab.md import rollout
from mlipstab.train.crmse_loop import CRMSEConfig, run_crmse

BONDS, B0 = torch.tensor([[0, 1]]), torch.tensor([1.0])


def _dimer(n):
    x = torch.zeros(n, 2, 3)
    x[:, 1, 0] = 1.0
    return x


def test_rollout_flags_failures_resets_them_and_keeps_frame_zero():
    # replica 0 feels no force and is pulled apart by its velocity; replica 1 is at rest
    x0, v0 = _dimer(2), torch.zeros(2, 2, 3)
    v0[0, 1, 0] = 0.1
    step = lambda x, v, f: (x + v, v, f)
    failed_fn = lambda x: bond_deviation(x, BONDS, B0) > 0.25
    traj, failed = rollout(lambda x: torch.zeros_like(x), step, x0, v0, n_steps=20, save_every=2,
                           failed_fn=failed_fn)
    assert failed.tolist() == [True, False] and traj.shape == (11, 2, 2, 3)
    assert torch.equal(traj[0], x0)
    dev = bond_deviation(traj, BONDS, B0)[:, 0]
    assert dev[2] > 0.25 and dev[3] < 0.25  # failing frame is saved, then the replica restarts
    _, fail_t, x_fail, x_anchor = extract_failure_pairs(traj, BONDS, B0, 0.25, lag_frames=1)
    assert fail_t.tolist() == [2] and torch.isclose(x_fail[0, 1, 0], torch.tensor(1.4))
    assert torch.isclose(x_anchor[0, 1, 0], torch.tensor(1.2))


def test_rollout_stops_early_at_stop_frac():
    x0, v0 = _dimer(4), torch.zeros(4, 2, 3)
    v0[:, 1, 0] = 0.1
    traj, failed = rollout(lambda x: torch.zeros_like(x), lambda x, v, f: (x + v, v, f), x0, v0,
                           n_steps=100, save_every=1,
                           failed_fn=lambda x: bond_deviation(x, BONDS, B0) > 0.25, stop_frac=0.5)
    assert failed.all() and len(traj) == 4


def test_ula_force_fn_matches_autograd_path():
    energy = lambda x: (x ** 2).sum(dim=(1, 2))
    x = torch.randn(3, 2, 3)
    a = ula_step(x, energy, 0.01, 10.0, generator=torch.Generator().manual_seed(0))
    b = ula_step(x, None, 0.01, 10.0, generator=torch.Generator().manual_seed(0), force_fn=lambda x: -2 * x)
    assert torch.allclose(a, b)


def test_crmse_raises_negative_energy_relative_to_data():
    class Toy(torch.nn.Module):  # U(x) = a * (r - 1)^2 on the dimer bond
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(0.1))

    model = Toy()
    r = lambda x: (x[:, 1] - x[:, 0]).norm(dim=-1)
    energy_fn = lambda m, x: m.a * (r(x) - 1.0) ** 2

    def force_fn(x):
        x = x.detach().requires_grad_(True)
        return -torch.autograd.grad(energy_fn(model, x).sum(), x)[0]

    buffer = _dimer(8)
    buffer[:, 1, 0] = 1.5  # negatives sit at a stretched bond
    cfg = CRMSEConfig(lam=1.0, epochs=3, batch_size=4, ula_steps_per_epoch=2, step_size=1e-5,
                      steps_per_epoch=5, log_every=1)
    hist = run_crmse(model, torch.optim.SGD(model.parameters(), lr=1e-3), cfg, buffer, energy_fn,
                     force_fn, ef_loss_fn=lambda m, b: torch.zeros(()), sample_data=lambda b: _dimer(b),
                     val_force_mae_fn=lambda m: 0.0, log=lambda *_: None)
    assert model.a.item() > 0.1 and len(hist) == 3 and "val_force_mae" in hist[-1]
