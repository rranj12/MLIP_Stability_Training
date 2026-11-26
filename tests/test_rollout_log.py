import json

import pytest
import torch

from mlipstab.md import rollout
from mlipstab.rollout_log import RolloutRecorder


def test_snapshot_preserves_failure_frame_and_matches_unobserved_rollout(tmp_path):
    x0 = torch.zeros(2, 1, 3)
    velocity = torch.zeros_like(x0)
    velocity[0, 0, 0] = 1
    args = dict(force_fn=torch.zeros_like, step_fn=lambda x, v, f: (x + v, v, f),
                x0=x0, v0=velocity, n_steps=20, save_every=2,
                failed_fn=lambda x: x[:, 0, 0] > 3, stop_frac=0.5)
    expected, failed = rollout(**args)
    recorder = RolloutRecorder(tmp_path, timestep_fs=0.5, n_steps=20, save_every=2)
    actual, actual_failed = rollout(**args, progress_fn=recorder)
    saved = torch.load(tmp_path / "trajectory.pt")
    assert torch.equal(expected, actual) and torch.equal(failed, actual_failed)
    assert torch.equal(saved["positions"], expected)
    assert saved["positions"][-1, 0, 0, 0] == 4  # failing frame, before reset
    assert saved["failed"].tolist() == [True, False]
    assert saved["frame_steps"].tolist() == [0, 2, 4]
    assert saved["progress"]["step"] == 4
    assert saved["progress"]["simulated_ps"] == 0.002
    assert saved["progress"]["finished"] is True


def test_partial_snapshot_survives_interrupted_rollout(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr("mlipstab.rollout_log.time.monotonic", lambda: clock[0])
    recorder = RolloutRecorder(tmp_path, timestep_fs=0.5, n_steps=20, save_every=2, interval_s=30)

    def step(x, v, f):
        clock[0] += 10
        if clock[0] == 50:
            raise RuntimeError("interrupted")
        return x + 1, v, f

    with pytest.raises(RuntimeError, match="interrupted"):
        rollout(torch.zeros_like, step, torch.zeros(1, 1, 3), torch.zeros(1, 1, 3),
                n_steps=20, save_every=2, failed_fn=lambda x: torch.tensor([False]),
                progress_fn=recorder)
    saved = torch.load(tmp_path / "trajectory.pt")
    assert saved["positions"][:, 0, 0, 0].tolist() == [0, 2, 4]
    assert saved["progress"]["step"] == 4
    assert saved["progress"]["elapsed_s"] == 40
    assert saved["progress"]["finished"] is False
    assert saved["positions"].device.type == "cpu"
    events = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert [e["step"] for e in events] == [0, 4]  # interval skips the step-2 write


def test_interrupted_snapshot_write_keeps_previous_snapshot(tmp_path, monkeypatch):
    recorder = RolloutRecorder(tmp_path, timestep_fs=0.5, n_steps=20, save_every=2, interval_s=0)
    frames, failed = [torch.zeros(1, 1, 3)], torch.tensor([False])
    recorder(0, frames, failed)
    previous = (tmp_path / "trajectory.pt").read_bytes()

    def broken_save(payload, path):
        path.write_bytes(b"incomplete")
        raise OSError("interrupted write")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="interrupted write"):
        recorder(2, frames + [torch.ones(1, 1, 3)], failed)
    assert (tmp_path / "trajectory.pt").read_bytes() == previous
    assert len((tmp_path / "progress.jsonl").read_text().splitlines()) == 1
