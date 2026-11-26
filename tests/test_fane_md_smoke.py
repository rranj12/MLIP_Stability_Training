"""Boundary checks for the benchmark bond metric and terminal-frame handling."""

import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "fane_md_smoke", Path(__file__).resolve().parents[1] / "scripts/fane_md_smoke.py"
)
md = importlib.util.module_from_spec(spec)
spec.loader.exec_module(md)


def test_pbc_matches_notebook_and_counts_terminal_failure():
    bonds = np.array([[0], [1]])
    # Across the 30 A box: lengths 1.0, 1.5, 1.6 A.
    positions = np.array([[[0., 0., 0.], [29., 0., 0.]],
                          [[0., 0., 0.], [28.5, 0., 0.]],
                          [[0., 0., 0.], [28.4, 0., 0.]]])
    values = md.deviations(positions, bonds, np.array([1.]))
    np.testing.assert_allclose(values, [0., 0.5, 0.6], atol=1e-12)
    assert md.first_failure(values) == 2
    assert md.first_failure(values[:-1]) is None  # Exactly 0.5 is not a failure.


def test_deviation_checks_all_bonds_against_reference_means():
    positions = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]],
                              [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]])
    bonds = np.array([[0, 0], [1, 2]])
    means = md.distance_pbc(positions[:, bonds[0]], positions[:, bonds[1]],
                            torch.FloatTensor([30., 30., 30.])).mean(0).numpy()
    np.testing.assert_array_equal(means, [1.5, 2.])
    frame = np.array([[[0., 0., 0.], [1.5, 0., 0.], [0., 2.6, 0.]]])
    assert md.first_failure(md.deviations(frame, bonds, means)) == 0
