"""Periodic, atomic position snapshots for diagnosis, not resumable MD checkpoints."""

import json
import time
from pathlib import Path

import torch


class RolloutRecorder:
    def __init__(self, directory, timestep_fs, n_steps, save_every, interval_s=30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.timestep_fs, self.n_steps, self.save_every = timestep_fs, n_steps, save_every
        self.interval_s = interval_s
        self.started = time.monotonic()
        self.last_saved = None

    def __call__(self, step, frames, failed, finished=False):
        now = time.monotonic()
        if not finished and self.last_saved is not None and now - self.last_saved < self.interval_s:
            return
        elapsed = now - self.started
        progress = {
            "step": step, "requested_steps": self.n_steps,
            "simulated_ps": step * self.timestep_fs / 1000,
            "elapsed_s": elapsed, "steps_per_second": step / elapsed if elapsed else 0.,
            "failed_replicas": int(failed.sum()), "n_replicas": len(failed),
            "saved_frames": len(frames), "finished": finished,
            "timestep_fs": self.timestep_fs, "save_every_steps": self.save_every,
        }
        # CPU copies make the snapshot portable. Replace only after a complete write:
        # interruption during the next save leaves the previous snapshot readable.
        temporary = self.directory / "trajectory.tmp"
        torch.save({"progress": progress, "positions": torch.stack(frames).cpu(),
                    "frame_steps": torch.arange(len(frames)) * self.save_every,
                    "failed": failed.cpu()}, temporary)
        temporary.replace(self.directory / "trajectory.pt")
        with (self.directory / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(progress) + "\n")
        print(f"rollout {self.directory.name}: {json.dumps(progress)}", flush=True)
        self.last_saved = time.monotonic()
