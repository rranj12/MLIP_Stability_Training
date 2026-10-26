"""Failure detection on batched replica positions (torch, any device)."""

import torch


def _pair_dist(pos, pairs):
    # pos: (..., N, 3); pairs: (P, 2) long -> (..., P)
    return (pos[..., pairs[:, 0], :] - pos[..., pairs[:, 1], :]).norm(dim=-1)


def bond_deviation(pos, bonds, b0):
    """Max |r_ij - b_ij| over bonds (Fu et al. criterion). pos (..., N, 3) -> (...)."""
    return (_pair_dist(pos, bonds) - b0).abs().amax(dim=-1)


def nonbonded_collision(pos, pairs, floor):
    """True where any non-bonded pair is closer than its floor. pos (..., N, 3) -> (...)."""
    return (_pair_dist(pos, pairs) < floor).any(dim=-1)


def unstable(pos, bonds, b0, threshold, nb_pairs=None, nb_floor=None):
    """Combined criterion: bond deviation > threshold, or (optionally) a non-bonded collision."""
    bad = bond_deviation(pos, bonds, b0) > threshold
    if nb_pairs is not None:
        bad = bad | nonbonded_collision(pos, nb_pairs, nb_floor)
    return bad


def extract_failure_pairs(traj, bonds, b0, threshold, lag_frames):
    """Find (x_fail, x_anchor) pairs in a saved rollout.

    traj: (T, R, N, 3) positions saved every few MD steps.
    For each replica that ever trips the criterion, x_fail is the first failing
    frame and x_anchor is the frame `lag_frames` earlier (clamped to frame 0),
    which should still be valid. Returns (replica_idx, fail_t, x_fail, x_anchor).
    """
    bad = bond_deviation(traj, bonds, b0) > threshold  # (T, R)
    failed = bad.any(dim=0)
    replicas = failed.nonzero(as_tuple=True)[0]
    fail_t = bad[:, replicas].float().argmax(dim=0)  # first True per replica
    anchor_t = (fail_t - lag_frames).clamp(min=0)
    x_fail = traj[fail_t, replicas]
    x_anchor = traj[anchor_t, replicas]
    return replicas, fail_t, x_fail, x_anchor
