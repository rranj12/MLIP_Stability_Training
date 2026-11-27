"""Fu et al. stability evaluation for any molecule and checkpoint: fane_md_smoke.py generalized.

    python fane_md.py <molecule> <model_dir> <out_dir> <seed> <init_idx> [max_ps] [weights.pt]

One replica: Nosé–Hoover 500 K, 0.5 fs, from test frame <init_idx>, until a bond deviates more than
0.5 Å from its mean over the full original MD17 trajectory, or max_ps (default 300). Same physics,
seeding order, topology and criterion as fane_md_smoke.py; replicas come from
configs/reproduce/fane_gemnet_t_naphthalene/protocol.yml. <model_dir> is an upstream MDsim run
(checkpoints/{config.yml,best_checkpoint.pt}); [weights.pt] replaces its weights (Stage B ckpt.pt).
Runs in the pinned environment of cloud/modal/fane_pilot.py; launched by cloud/modal/fane_md.py.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from fane_md_smoke import THRESHOLD, DT_PS, SAVE_EVERY, distance_pbc, deviations, write_json

DATA = Path('/vol/data/forces_not_enough/md17')
RAW_URL = 'https://www.quantum-machine.org/gdml/data/npz/md17_{molecule}.npz'
KCAL_TO_EV = 1 / 23.06052


def reference(molecule, atoms):
    """Bonds of the starting frame and their mean lengths over the full original MD17 trajectory,
    after checking the authors' LMDB splits against the raw file (as fane_md_smoke.prepare_reference)."""
    import urllib.request
    import numpy as np
    import torch
    from ase.neighborlist import NeighborList, natural_cutoffs
    from mdsim.datasets.lmdb_dataset import LmdbDataset

    raw_path = DATA / f'md17_{molecule}.npz'
    if not raw_path.exists():
        tmp = raw_path.with_name(f'{raw_path.name}.{os.getpid()}.part')
        urllib.request.urlretrieve(RAW_URL.format(molecule=molecule), tmp)
        tmp.replace(raw_path)
    raw = np.load(raw_path)
    for split in ('train', 'val', 'test'):
        dataset = LmdbDataset({'src': str(DATA / molecule / '10k' / split)})
        for i in (0, len(dataset) // 2, len(dataset) - 1):
            sample = dataset[i]
            fid = int(sample.fid)
            np.testing.assert_allclose(sample.pos.numpy(), raw['R'][fid], atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(sample.force.numpy(), raw['F'][fid] * KCAL_TO_EV, atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(sample.atomic_numbers.numpy(), raw['z'])
        dataset.close_db()
    neighbors = NeighborList(natural_cutoffs(atoms), self_interaction=False)
    neighbors.update(atoms)
    bonds = np.asarray(neighbors.get_connectivity_matrix().todense().nonzero())
    gt = torch.FloatTensor(raw['R'])
    means = distance_pbc(gt[:, bonds[0]], gt[:, bonds[1]], torch.FloatTensor([30., 30., 30.])).mean(dim=0).numpy()
    sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    return bonds, means, {'raw_sha256': sha, 'reference_frames': len(raw['R'])}


def run(molecule, model_dir, out, seed, init_idx, max_ps=300., weights=None):
    import numpy as np
    import torch
    from ase import units
    from mdsim.datasets.lmdb_dataset import LmdbDataset
    from mdsim.md.ase_utils import OCPCalculator, Simulator, data_to_atoms
    from mdsim.md.integrator import NoseHoover
    sys.path.insert(0, '/opt/MDsim')
    from simulate import seed_everywhere

    if (out / 'result.json').exists():
        print(f'done already: {out}')
        return
    out.mkdir(parents=True, exist_ok=True)
    for stale in ('bond_progress.jsonl', 'progress.json'):  # an interrupted attempt restarts from zero
        (out / stale).unlink(missing_ok=True)
    started = time.monotonic()
    os.chdir('/opt/MDsim')
    test = DATA / molecule / '10k' / 'test'
    dataset = LmdbDataset({'src': str(test)})
    sample = dataset[init_idx]
    atoms = data_to_atoms(sample)
    dataset.close_db()
    bonds, means, provenance = reference(molecule, atoms)
    # Preserve upstream order: seed_everywhere -> calculator/model seed -> velocity initialization.
    seed_everywhere(seed)
    checkpoint = model_dir / 'checkpoints/best_checkpoint.pt'
    calculator = OCPCalculator(config_yml=str(model_dir / 'checkpoints/config.yml'),
                               checkpoint=str(checkpoint), test_data_src=str(test))
    if not calculator.config['model'].get('use_pbc', True):
        # MDsim 7678721 (2024) wraps positions into the 30 Å cell before graph construction; a
        # non-periodic model (SchNet, use_pbc: false) then sees a molecule torn across the cell edge
        # (force MAE ~1.7 eV/Å on test frames). Use the pre-2024 conversion, under which Fu et al.'s
        # SchNet numbers were produced.
        import mdsim.md.ase_utils as ase_utils
        from torch_geometric.data import Data

        def atoms_to_batch(a):
            return Data(cell=torch.Tensor(a.get_cell()).view(1, 3, 3), pos=torch.Tensor(a.get_positions()),
                        atomic_numbers=torch.Tensor(a.get_atomic_numbers()), natoms=len(a))
        ase_utils.atoms_to_batch = atoms_to_batch
    if weights is not None:  # Stage B: same architecture and normalizers, new weights
        model = calculator.trainer.model
        state = torch.load(weights, map_location='cpu')['state_dict']
        prefix = 'module.' if next(iter(model.state_dict())).startswith('module.') else ''
        missing, unexpected = model.load_state_dict(
            {prefix + k.removeprefix('module.'): v for k, v in state.items()}, strict=False)
        # Our Stage B SchNet omits upstream's constant atomic_mass buffer; anything else is an error.
        assert not unexpected and {k.removeprefix('module.') for k in missing} <= {'atomic_mass'}, (missing, unexpected)
        calculator.trainer.ema = None  # predict() would otherwise swap the EMA weights back in
    atoms.calc = calculator
    start_force_mae = float(np.abs(atoms.get_forces() - sample.force.numpy()).mean())
    assert start_force_mae < 0.25, f'force MAE {start_force_mae:.3f} eV/A on the starting frame: model not loaded correctly'
    integrator = NoseHoover(atoms, timestep=0.5 * units.fs, temperature=500. * units.kB, ttime=20.)
    simulator = Simulator(atoms, integrator, 500., save_dir=out, save_frequency=SAVE_EVERY)
    evaluated = weights or checkpoint
    state = {'status': 'running', 'molecule': molecule, 'model_dir': str(model_dir),
             'weights': str(evaluated), 'weights_sha256': hashlib.sha256(Path(evaluated).read_bytes()).hexdigest(),
             'seed': seed, 'init_idx': init_idx, 'timestep_fs': 0.5, 'temperature_K': 500., 'ttime': 20.,
             'max_ps': max_ps, 'save_every_steps': SAVE_EVERY, 'threshold_A': THRESHOLD,
             'bonds': bonds.tolist(), 'means_A': means.tolist(), **provenance,
             'start_force_mae_eV_A': start_force_mae,
             'initial_max_deviation_A': float(deviations(atoms.positions[None], bonds, means)[0]),
             'first_failure_ps': None, 'max_observed_deviation_A': 0.}
    assert state['initial_max_deviation_A'] < THRESHOLD

    def observe():
        step = integrator.nsteps
        deviation = float(deviations(atoms.positions[None], bonds, means)[0])
        record = {'step': step, 'ps': step * DT_PS, 'max_deviation_A': deviation,
                  'temperature_K': float(atoms.get_temperature())}
        with (out / 'bond_progress.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        state['max_observed_deviation_A'] = max(state['max_observed_deviation_A'], deviation)
        if deviation > THRESHOLD and state['first_failure_ps'] is None:
            state['first_failure_ps'] = record['ps']
        if step % 20000 == 0:
            print(f"{record['ps']:.0f}/{max_ps:.0f} ps; deviation {deviation:.3f} A; "
                  f"T {record['temperature_K']:.0f} K; {time.monotonic() - started:.0f} s", flush=True)
            write_json(out / 'progress.json', state)

    integrator.attach(observe, interval=SAVE_EVERY)
    try:
        for _ in range(round(max_ps / DT_PS)):
            integrator.run(1)  # same stepping as upstream Simulator.run
            if state['first_failure_ps'] is not None:
                state['status'] = 'bond_failure'
                break
            temperature = atoms.get_temperature()
            if not np.isfinite(atoms.positions).all() or not np.isfinite(temperature):
                state['status'] = 'numerical_failure'
                break
            if temperature < simulator.min_temp or temperature > simulator.max_temp:
                state['status'] = 'temperature_guard'
                break
        else:
            state['status'] = 'horizon_reached'
    finally:
        simulator.traj.close()
        state.update(md_steps=integrator.nsteps, simulated_ps=integrator.nsteps * DT_PS,
                     elapsed_s=time.monotonic() - started)
        write_json(out / ('result.json' if state['status'] != 'running' else 'progress.json'), state)
        print(json.dumps({k: v for k, v in state.items() if k not in ('bonds', 'means_A')}, indent=1), flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    run(a[0], Path(a[1]), Path(a[2]), int(a[3]), int(a[4]),
        float(a[5]) if len(a) > 5 else 300., Path(a[6]) if len(a) > 6 else None)
