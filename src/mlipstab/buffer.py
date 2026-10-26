"""FIFO replay buffer of (failure, anchor) configuration pairs."""

from collections import deque

import torch


class FailureBuffer:
    def __init__(self, capacity=2000):
        self.items = deque(maxlen=capacity)  # oldest pairs are evicted first

    def __len__(self):
        return len(self.items)

    def push(self, x_fail, x_anchor, local_atoms=None, cycle=0):
        """Add a batch of pairs. x_*: (B, N, 3). local_atoms: optional (B, N) bool mask."""
        for b in range(x_fail.shape[0]):
            mask = None if local_atoms is None else local_atoms[b].detach().cpu()
            self.items.append(
                (x_fail[b].detach().cpu(), x_anchor[b].detach().cpu(), mask, cycle)
            )

    def sample(self, batch_size, device="cpu", generator=None):
        """Uniform sample with replacement -> (x_fail, x_anchor, mask or None)."""
        idx = torch.randint(len(self.items), (batch_size,), generator=generator)
        picked = [self.items[i] for i in idx.tolist()]
        x_fail = torch.stack([p[0] for p in picked]).to(device)
        x_anchor = torch.stack([p[1] for p in picked]).to(device)
        masks = [p[2] for p in picked]
        mask = None if masks[0] is None else torch.stack(masks).to(device)
        return x_fail, x_anchor, mask
