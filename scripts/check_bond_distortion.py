"""Small synthetic-pair diagnostic, not an MD stability benchmark or campaign checkpoint."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mlipstab.config import load_config, _merge
from mlipstab.detect import bond_deviation
from mlipstab.losses.margin import margin_loss
from mlipstab.mdsim_backend import MDSimBackend
from mlipstab.units import kT


def distort_ch(frames, bonds, b0, atomic_numbers, displacement=0.30):
    """First C-H bond, first two frames, both signs; move only H to b0 +/- delta."""
    for index, (i, j) in enumerate(bonds.tolist()):
        if {int(atomic_numbers[i]), int(atomic_numbers[j])} == {1, 6}:
            carbon, hydrogen = (i, j) if int(atomic_numbers[i]) == 6 else (j, i)
            break
    else:
        raise ValueError('diagnostic needs a C-H bond')
    anchors = frames[:2].detach().repeat_interleave(2, dim=0).clone()
    distorted = anchors.clone()
    direction = anchors[:, hydrogen] - anchors[:, carbon]
    direction = direction / direction.norm(dim=-1, keepdim=True)
    target = b0[index] + displacement * frames.new_tensor([1., -1., 1., -1.])
    distorted[:, hydrogen] = anchors[:, carbon] + target[:, None] * direction
    return distorted, anchors, {'bond_index': index, 'carbon': carbon, 'hydrogen': hydrogen,
                                'target_lengths_A': target.tolist()}


def optimizer_summary(opt):
    steps = [float(s['step']) for s in opt.state.values() if 'step' in s]
    return {'state_count': len(opt.state), 'step_min': min(steps), 'step_max': max(steps)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    seed = 20261126
    cfg = _merge(load_config(ROOT / 'configs/margin/base.yml'), {
        'seed': seed, 'pretrained': {'dir': f'results/checkpoints/qm_pretrained/seed{seed}', 'seed': seed},
        'optim': {'name': 'AdamW', 'lr': 1e-5}})
    torch.manual_seed(seed)
    np.random.seed(seed)
    b = MDSimBackend(cfg, 'naphthalene', 'gemnet_t', root=ROOT)
    threshold = b.train_threshold(cfg['detect'])
    pairs, metadata = {}, {}
    for split in ('train', 'val'):
        xf, xa, detail = distort_ch(b.frames(split), b.bonds, b.b0, b.data['train']['z'])
        assert (bond_deviation(xa, b.bonds, b.b0) < threshold).all()
        assert (bond_deviation(xf, b.bonds, b.b0) > threshold).all()
        pairs[split], metadata[split] = (xf, xa), detail
    torch.save({split: {'distorted': xf.cpu(), 'anchor': xa.cpu()} for split, (xf, xa) in pairs.items()}, out / 'pairs.pt')
    baseline = copy.deepcopy(b.model.state_dict())
    params = [p for p in b.model.parameters() if p.requires_grad]
    scale, default_margin = kT(500.), 10 * kT(500.)

    def energies(split, grad=False):
        xf, xa = pairs[split]
        return b.energy_force(xf.clone(), param_grad=grad)[0], b.energy_force(xa.clone(), param_grad=grad)[0]

    ef, ea = energies('train')
    # Deliberately exercise an unsatisfied margin if these synthetic pairs already
    # clear the production margin. Freeze this diagnostic margin before any update.
    active_margin = max(default_margin, (ef - ea).max().item() + 2 * scale)
    report = {'kind': 'synthetic_bond_distortion_only', 'seed': seed, 'config': cfg,
              'threshold_A': threshold, 'pairs': metadata, 'default_margin_eV': default_margin,
              'diagnostic_margin_eV': active_margin, 'updates_per_arm': 4, 'lambda': 1.,
              'probe_parameter_step_norm': 1e-3, 'arms': {}}

    def save():
        report['elapsed_s'] = time.monotonic() - started
        temporary = out / 'report.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
        temporary.replace(out / 'report.json')
        print(json.dumps(report, allow_nan=False), flush=True)

    def measure():
        result = {}
        for split in pairs:
            ef, ea = energies(split)
            result[split] = {'gaps_eV': (ef - ea).tolist(),
                             'default_loss': margin_loss(ef, ea, default_margin, scale).item(),
                             'diagnostic_loss': margin_loss(ef, ea, active_margin, scale).item()}
        result['val_force_mae_eV_A'] = b.errors('val', batch_size=2)[1]
        return result

    report['before'] = measure()
    ef, ea = energies('train', grad=True)
    loss = margin_loss(ef, ea, active_margin, scale)
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    norm = sum(g.double().square().sum() for g in grads if g is not None).sqrt().item()
    assert np.isfinite(norm) and norm > 0
    report['margin_gradient_norm'] = norm
    report['direction_probe'] = {}
    for sign, name in ((-1, 'descent'), (1, 'ascent')):
        b.model.load_state_dict(baseline)
        with torch.no_grad():
            for param, grad in zip(params, grads):
                if grad is not None:
                    param.add_(grad, alpha=sign * 1e-3 / norm)
        ef, ea = energies('train')
        report['direction_probe'][name] = {'loss': margin_loss(ef, ea, active_margin, scale).item(),
                                           'mean_gap_eV': (ef - ea).mean().item()}
    del grads, loss, ef, ea
    b.model.load_state_dict(baseline)
    save()
    for name, lam in (('control', 0.), ('margin', 1.)):
        b.model.load_state_dict(baseline)
        opt = b.make_optimizer(cfg)
        b.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        ef_loss_fn = b.make_ef_loss_fn(cfg)
        arm = {'initial_optimizer': optimizer_summary(opt), 'updates': []}
        report['arms'][name] = arm
        for step in range(4):
            labeled = ef_loss_fn(b.model, 2)
            ef, ea = energies('train', grad=True)
            stab = margin_loss(ef, ea, active_margin, scale)
            opt.zero_grad()
            total = labeled + lam * stab
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step()
            arm['updates'].append({'step': step + 1, 'ef_loss': labeled.item(),
                                   'margin_loss': stab.item(), 'gradient_norm': float(grad_norm)})
            save()
        arm['after'] = measure()
        arm['final_optimizer'] = optimizer_summary(opt)
        arm['frozen_parameters_unchanged'] = all(torch.equal(param, baseline[n])
            for n, param in b.model.named_parameters() if not param.requires_grad)
        assert arm['initial_optimizer']['step_min'] == arm['initial_optimizer']['step_max'] == 119
        assert arm['final_optimizer']['step_min'] == arm['final_optimizer']['step_max'] == 123
        assert arm['frozen_parameters_unchanged']
        save()
    probe = report['direction_probe']
    report['gradient_direction_pass'] = (probe['descent']['loss'] < report['before']['train']['diagnostic_loss'] < probe['ascent']['loss']
                                         and probe['descent']['mean_gap_eV'] > probe['ascent']['mean_gap_eV'])
    report['margin_vs_control_gap_gain_eV'] = {
        split: float(np.mean(report['arms']['margin']['after'][split]['gaps_eV'])
                     - np.mean(report['arms']['control']['after'][split]['gaps_eV'])) for split in pairs}
    report['completed'] = True
    save()
    # The diagnostic never replaces a production checkpoint.
    b.model.load_state_dict(baseline)


if __name__ == '__main__':
    main()
