import importlib.util
from pathlib import Path

import torch

from mlipstab.detect import bond_deviation


def test_controlled_ch_distortion_preserves_anchors_and_other_atoms():
    spec = importlib.util.spec_from_file_location(
        'check_bond_distortion', Path(__file__).resolve().parents[1] / 'scripts/check_bond_distortion.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames = torch.tensor([[[0., 0, 0], [1.12, 0, 0], [-1.4, 0, 0]],
                           [[0., 0, 0], [0, 1.06, 0], [-1.4, 0, 0]]])
    original = frames.clone()
    bonds, b0 = torch.tensor([[0, 2], [0, 1]]), torch.tensor([1.4, 1.1])
    xf, xa, detail = module.distort_ch(frames, bonds, b0, [6, 1, 6])
    assert detail['bond_index'] == 1 and detail['hydrogen'] == 1
    torch.testing.assert_close(frames, original)
    torch.testing.assert_close(xa, original.repeat_interleave(2, dim=0))
    torch.testing.assert_close(xf[:, [0, 2]], xa[:, [0, 2]])
    torch.testing.assert_close((xf[:, 1] - xf[:, 0]).norm(dim=-1), torch.tensor([1.4, .8, 1.4, .8]))
    assert (bond_deviation(xf, bonds, b0) > .216182).all()
    assert (bond_deviation(xa, bonds, b0) < .216182).all()
