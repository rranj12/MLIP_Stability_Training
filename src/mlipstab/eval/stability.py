"""Stability time and survival analysis (stability is right-censored data)."""

import numpy as np


def first_failure_time(unstable, dt_ps):
    """unstable: (T, R) bool per saved frame. Returns (times_ps (R,), observed (R,) bool).

    Replicas that never fail are censored at the end of the run (observed=False).
    """
    unstable = np.asarray(unstable, dtype=bool)
    n_frames = unstable.shape[0]
    observed = unstable.any(axis=0)
    first = np.where(observed, unstable.argmax(axis=0), n_frames)
    return first * dt_ps, observed


def kaplan_meier(times, observed):
    """Kaplan-Meier survival curve. Returns (event_times, S) as step-function knots."""
    times = np.asarray(times, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    event_times = np.unique(times[observed])
    surv, s = [], 1.0
    for t in event_times:
        at_risk = (times >= t).sum()
        deaths = ((times == t) & observed).sum()
        s *= 1.0 - deaths / at_risk
        surv.append(s)
    return event_times, np.array(surv)


def median_survival(event_times, surv):
    """First time S(t) <= 0.5; inf if the curve never gets there (median censored)."""
    below = np.nonzero(surv <= 0.5)[0]
    return float(event_times[below[0]]) if len(below) else float("inf")


# Log-rank tests between methods: use lifelines.statistics.logrank_test in
# scripts/analysis/survival.py rather than reimplementing it here.
