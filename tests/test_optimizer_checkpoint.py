"""A resumed update must equal an uninterrupted update, including parameter groups."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mlipstab.mdsim_backend import MDSimBackend, _load_state


def update(model, optimizer):
    optimizer.zero_grad()
    model(torch.tensor([[1., 2.]])).square().sum().backward()
    optimizer.step()


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "pretrain_mdsim", Path(__file__).resolve().parents[1] / "scripts/pretrain_mdsim.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    monkeypatch.setitem(sys.modules, "mdsim.common",
                        SimpleNamespace(distutils=SimpleNamespace(is_master=lambda: True)))

    class Trainer:
        def save(self, metrics, checkpoint_file, training_state):
            # Model the upstream EMA export: different weights, raw weights restored on return.
            torch.save({"state_dict": {n: p + 10 for n, p in self.model.state_dict().items()}},
                       tmp_path / checkpoint_file)

    trainer = Trainer()
    trainer.model = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.AdamW([
        {"params": [trainer.model.bias], "weight_decay": 0.},
        {"params": [trainer.model.weight], "weight_decay": 0.02},
    ], lr=0.01, eps=1e-7, amsgrad=True)
    trainer.step, trainer.is_debug = 3, False
    trainer.config = {"cmd": {"checkpoint_dir": str(tmp_path)}}
    for _ in range(trainer.step):
        update(trainer.model, trainer.optimizer)
    wrapper.preserve_best_training_state(Trainer)
    trainer.save(checkpoint_file="best_checkpoint.pt", training_state=False)

    backend = MDSimBackend.__new__(MDSimBackend)
    backend.model = torch.nn.Linear(2, 1)
    backend.device, backend.is_nequip = "cpu", False
    backend.stage_a_checkpoint = tmp_path / "best_training_checkpoint.pt"
    backend.model.load_state_dict(_load_state(backend.stage_a_checkpoint, "cpu"))
    return trainer, backend


def test_matched_optimizer_next_step_equals_uninterrupted(saved_run):
    trainer, backend = saved_run
    ema = _load_state(backend.stage_a_checkpoint.with_name("best_checkpoint.pt"), "cpu")
    assert not torch.equal(ema["weight"], backend.model.weight)
    optimizer = backend.make_optimizer({"optim": {"name": "AdamW", "lr": 1e-5,
                                                 "reuse_stage_a_state": True}})
    assert len(optimizer.state) == 2
    assert all(s["step"] == 3 for s in optimizer.state.values())
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0., 0.02]
    for group in trainer.optimizer.param_groups:
        group["lr"] = 1e-5
    update(trainer.model, trainer.optimizer)
    update(backend.model, optimizer)
    for expected, actual in zip(trainer.model.parameters(), backend.model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(s["step"] == 4 for s in optimizer.state.values())


@pytest.mark.parametrize("problem, error, message", [
    ("missing", FileNotFoundError, "Missing matched"),
    ("class", ValueError, "does not match"),
    ("weights", ValueError, "weights do not match"),
    ("nequip", ValueError, "not paired"),
])
def test_unsafe_reuse_fails_instead_of_resetting(saved_run, problem, error, message):
    _, backend = saved_run
    name = "AdamW"
    if problem == "missing":
        backend.stage_a_checkpoint.unlink()
    elif problem == "class":
        name = "Adam"
    elif problem == "weights":
        with torch.no_grad():
            backend.model.weight.add_(1)
    else:
        backend.is_nequip = True
    with pytest.raises(error, match=message):
        backend.make_optimizer({"optim": {"name": name, "lr": 1e-5, "reuse_stage_a_state": True}})
    fresh = backend.make_optimizer({"optim": {"name": name, "lr": 1e-5, "reuse_stage_a_state": False}})
    assert not fresh.state
