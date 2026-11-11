"""Batched MD rollout with online failure checks (backend-agnostic).

    force_fn(x)        -> forces (R, N, 3)
    step_fn(x, v, f)   -> (x, v, f) after one integrator step
    failed_fn(x)       -> (R,) bool, evaluated on every saved frame

Replicas that fail a check are put back on their starting state so that one
collapsed geometry cannot produce NaN forces for the whole batch (StABlE's
calculator asserts on NaN). Their failing frame is saved before the reset, so
`first failure` is still read off the returned trajectory.
"""

import torch


@torch.no_grad()
def rollout(force_fn, step_fn, x0, v0, n_steps, save_every, failed_fn,
            stop_frac=None, store_device=None):
    """Returns (traj (T, R, N, 3) with frame 0 = x0, failed_ever (R,) bool)."""
    store_device = store_device or x0.device
    x, v = x0.clone(), v0.clone()
    f = force_fn(x)
    failed = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    frames = [x0.detach().to(store_device)]
    for step in range(1, n_steps + 1):
        x, v, f = step_fn(x, v, f)
        if step % save_every:
            continue
        frames.append(x.detach().to(store_device))
        bad = failed_fn(x) | ~torch.isfinite(x).all(dim=(1, 2))
        failed |= bad
        if bad.any():
            sel = bad[:, None, None]
            x, v = torch.where(sel, x0, x), torch.where(sel, v0, v)
            f = force_fn(x)
        if stop_frac is not None and failed.float().mean().item() >= stop_frac:
            break
    return torch.stack(frames), failed
