"""Kaplan-Meier survival curves per method + log-rank tests (explainer Sec. 9.3-9.4).

    python scripts/analysis/survival.py results/eval/*aspirin_schnet*

TODO: load each stability.npz, plot mlipstab.eval.stability.kaplan_meier per method,
and compare methods pairwise with lifelines.statistics.logrank_test.
Other planned figures (one script each): Pareto (stability vs force MAE over lambda, m),
energy scan along a stretched C-H bond, buffer-energy histogram, h(r) and dihedral
overlays, failing-replica fraction per Stage B cycle, per-failure gradient signal.
"""
