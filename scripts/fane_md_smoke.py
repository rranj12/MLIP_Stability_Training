"""Bounded MD diagnostic using pinned MDsim physics and the notebook bond criterion."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

DATA = Path('/vol/data/forces_not_enough/md17/naphthalene/10k')
BASE = Path('/vol/results/reproduce/fane_gemnet_t_naphthalene')
MODEL = BASE / 'pilot_seed0_retry1/md17-naphthalene_10k_gemnet_t_pilot_seed0'
OUT = BASE / 'md_smoke_seed123_init857_20ps'
RAW = DATA.parent / 'md17_naphthalene.npz'
REFERENCE = DATA.parent / 'md_smoke_reference.npz'
RAW_URL = 'https://www.quantum-machine.org/gdml/data/npz/md17_naphthalene.npz'
SEED, INIT_IDX, MAX_STEPS, SAVE_EVERY = 123, 857, 40000, 100
DT_PS, THRESHOLD, WALL_STOP_S = 0.0005, 0.5, 1650


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def distance_pbc(x0, x1, lattices):
    # Exact helper from the pinned authors' observable.ipynb.
    import torch
    delta = torch.abs(x0 - x1)
    lattices = lattices.view(-1, 1, 3)
    delta = torch.where(delta > 0.5 * lattices, delta - lattices, delta)
    return torch.sqrt((delta ** 2).sum(dim=-1))


def deviations(positions, bonds, means):
    import torch
    xyz = torch.as_tensor(positions)
    lens = distance_pbc(xyz[:, bonds[0]], xyz[:, bonds[1]], torch.FloatTensor([30., 30., 30.]))
    return (lens - torch.as_tensor(means)).abs().max(dim=1).values.numpy()


def first_failure(values, threshold=THRESHOLD):
    return next((i for i, value in enumerate(values) if value > threshold), None)


def prepare_reference():
    import urllib.request
    import numpy as np
    import torch
    from ase.neighborlist import NeighborList, natural_cutoffs
    from mdsim.datasets.lmdb_dataset import LmdbDataset
    from mdsim.md.ase_utils import data_to_atoms

    if not RAW.exists():
        temporary = RAW.with_suffix('.part')
        count, previous = 0, time.monotonic()
        with urllib.request.urlopen(RAW_URL, timeout=120) as response, temporary.open('wb') as out:
            while chunk := response.read(4 * 1024 * 1024):
                out.write(chunk)
                count += len(chunk)
                if time.monotonic() - previous >= 30:
                    print(f'Original MD17 download: {count / 1e6:.1f} MB', flush=True)
                    previous = time.monotonic()
        temporary.replace(RAW)
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    raw = np.load(RAW)
    positions, forces, energies = raw['R'], raw['F'], raw['E']
    assert positions.shape == (326250, 18, 3)
    assert np.isfinite(positions).all()
    for split in ('train', 'val', 'test'):
        dataset = LmdbDataset({'src': str(DATA / split)})
        for i in (0, len(dataset) // 2, len(dataset) - 1):
            sample = dataset[i]
            fid = int(sample.fid)
            np.testing.assert_allclose(sample.pos.numpy(), positions[fid], atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(sample.force.numpy(), forces[fid] / 23.06052, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(sample.y.numpy().ravel(), energies[fid].ravel() / 23.06052, rtol=1e-5)
            np.testing.assert_array_equal(sample.atomic_numbers.numpy(), raw['z'])
        if split == 'test':
            atoms = data_to_atoms(dataset[INIT_IDX])
        dataset.close_db()
    neighbors = NeighborList(natural_cutoffs(atoms), self_interaction=False)
    neighbors.update(atoms)
    bonds = np.asarray(neighbors.get_connectivity_matrix().todense().nonzero())
    gt = torch.FloatTensor(positions)
    means = distance_pbc(gt[:, bonds[0]], gt[:, bonds[1]], torch.FloatTensor([30., 30., 30.])).mean(dim=0)
    np.savez(REFERENCE, bonds=bonds, means=means.numpy(), initial_positions=atoms.positions, z=raw['z'])
    report = {'raw_url': RAW_URL, 'raw_sha256': digest, 'reference_frames': len(positions),
              'released_split_crosscheck': 'passed', 'init_idx': INIT_IDX,
              'bonds': bonds.tolist(), 'means_A': means.tolist(),
              'initial_max_deviation_A': float(deviations(atoms.positions[None], bonds, means.numpy())[0])}
    assert report['initial_max_deviation_A'] < THRESHOLD
    write_json(REFERENCE.with_suffix('.json'), report)
    print(json.dumps(report, indent=2), flush=True)


def run_md():
    import numpy as np
    from ase import units
    from mdsim.datasets.lmdb_dataset import LmdbDataset
    from mdsim.md.ase_utils import OCPCalculator, Simulator, data_to_atoms
    from mdsim.md.integrator import NoseHoover
    sys.path.insert(0, '/opt/MDsim')
    from simulate import seed_everywhere

    started = time.monotonic()
    os.chdir('/opt/MDsim')
    if OUT.exists():
        raise FileExistsError(f'MD diagnostic already attempted: {OUT}')
    OUT.mkdir(parents=True)
    reference = np.load(REFERENCE)
    bonds, means = reference['bonds'], reference['means']
    seed_everywhere(SEED)
    checkpoint = MODEL / 'checkpoints/best_checkpoint.pt'
    calculator = OCPCalculator(config_yml=str(MODEL / 'checkpoints/config.yml'),
                               checkpoint=str(checkpoint), test_data_src=str(DATA / 'test'))
    dataset = LmdbDataset({'src': str(DATA / 'test')})
    atoms = data_to_atoms(dataset[INIT_IDX])
    dataset.close_db()
    np.testing.assert_array_equal(atoms.positions, reference['initial_positions'])
    atoms.calc = calculator
    integrator = NoseHoover(atoms, timestep=0.5 * units.fs, temperature=500. * units.kB, ttime=20.)
    simulator = Simulator(atoms, integrator, 500., save_dir=OUT, save_frequency=SAVE_EVERY)
    # Preserve upstream order: seed_everywhere -> calculator/model seed -> velocity initialization.
    np.savez(OUT / 'initial_state.npz', positions=atoms.positions, momenta=atoms.get_momenta(), z=atoms.numbers)
    state = {'status': 'running', 'checkpoint': str(checkpoint),
             'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
             'requested_seed': SEED, 'calculator_seed': calculator.trainer.config['cmd']['seed'],
             'init_idx': INIT_IDX, 'timestep_fs': 0.5, 'temperature_K': 500., 'ttime': 20.,
             'max_ps': 20., 'save_every_steps': SAVE_EVERY, 'threshold_A': THRESHOLD,
             'gpu_hard_cap_s': 1800, 'wall_stop_s': WALL_STOP_S, 'training_updates': 0,
             'reference': json.loads(REFERENCE.with_suffix('.json').read_text()),
             'first_failure_ps': None, 'sampled_frames': 0, 'max_observed_deviation_A': 0.}
    previous = [atoms.positions.copy(), 0]

    def observe():
        step = integrator.nsteps
        deviation = float(deviations(atoms.positions[None], bonds, means)[0])
        record = {'step': step, 'ps': step * DT_PS, 'max_deviation_A': deviation,
                  'temperature_K': float(atoms.get_temperature())}
        with (OUT / 'bond_progress.jsonl').open('a') as out:
            out.write(json.dumps(record) + '\n')
        state.update(last_saved_ps=record['ps'], sampled_frames=state['sampled_frames'] + 1,
                     max_observed_deviation_A=max(state['max_observed_deviation_A'], deviation))
        if deviation > THRESHOLD and state['first_failure_ps'] is None:
            state['first_failure_ps'] = record['ps']
            np.savez(OUT / 'failure_pair.npz', anchor=previous[0], failure=atoms.positions,
                     anchor_step=previous[1], failure_step=step, z=atoms.numbers, bonds=bonds, means_A=means)
        previous[:] = [atoms.positions.copy(), step]
        if step % 500 == 0 or state['first_failure_ps'] is not None:
            print(f"MD {record['ps']:.3f}/20 ps; deviation {deviation:.4f}/0.5 A; "
                  f"T {record['temperature_K']:.1f} K; elapsed {time.monotonic()-started:.0f}s", flush=True)
        write_json(OUT / 'progress.json', state)

    integrator.attach(observe, interval=SAVE_EVERY)
    write_json(OUT / 'progress.json', state)
    try:
        for _ in range(MAX_STEPS):
            if time.monotonic() - started >= WALL_STOP_S:
                state['status'] = 'runtime_cap'
                break
            integrator.run(1)  # Same stepping as upstream Simulator.run.
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
    except Exception as exc:
        state.update(status='error', error=repr(exc))
        raise
    finally:
        simulator.traj.close()
        state.update(md_steps=integrator.nsteps, simulated_ps=integrator.nsteps * DT_PS,
                     elapsed_s=time.monotonic() - started, thermostat_zeta=float(integrator.zeta))
        state['gpu_compute_estimate_usd'] = state['elapsed_s'] * 0.000694
        np.savez(OUT / 'final_state.npz', positions=atoms.positions, momenta=atoms.get_momenta(), z=atoms.numbers)
        write_json(OUT / 'result.json', state)
        print(json.dumps({k:v for k,v in state.items() if k != 'reference'}, indent=2), flush=True)


def audit():
    import numpy as np
    from ase.io import Trajectory

    result = json.loads((OUT / 'result.json').read_text())
    ref = np.load(REFERENCE)
    with Trajectory(OUT / 'atoms.traj') as traj:
        positions = np.stack([atoms.positions for atoms in traj])
    records = [json.loads(line) for line in (OUT / 'bond_progress.jsonl').read_text().splitlines()]
    values = deviations(positions, ref['bonds'], ref['means'])
    assert len(values) == len(records)
    np.testing.assert_allclose(values, [row['max_deviation_A'] for row in records], atol=1e-12)
    assert [row['step'] for row in records] == list(range(0, len(records) * SAVE_EVERY, SAVE_EVERY))
    thermo = np.loadtxt(OUT / 'thermo.log', skiprows=1, ndmin=2)
    np.testing.assert_allclose(thermo[:, 0], [row['ps'] for row in records], atol=5e-5)
    failure = first_failure(values)
    failure_ps = None if failure is None else records[failure]['ps']
    assert failure_ps == result['first_failure_ps']
    if failure is not None:
        pair = np.load(OUT / 'failure_pair.npz')
        np.testing.assert_array_equal(pair['failure'], positions[failure])
        assert first_failure(values[:failure]) is None
    # The literal notebook loop omits the terminal frame; record its result separately.
    literal_failure = first_failure(values[:-1])
    literal_collapse_ps = (len(records)-1 if literal_failure is None else literal_failure) * 0.05
    report = {'status': 'passed', 'frames': len(values), 'first_bond_failure_ps': failure_ps,
              'last_audited_ps': records[-1]['ps'], 'max_deviation_A': float(values.max()),
              'literal_notebook_collapse_ps': literal_collapse_ps,
              'literal_notebook_detected_failure': literal_failure is not None,
              'terminal_frame_included_in_our_detection': True,
              'md_steps': result['md_steps'], 'stop_reason': result['status']}
    write_json(OUT / 'audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    {'prepare': prepare_reference, 'run': run_md, 'audit': audit}[sys.argv[1]]()
