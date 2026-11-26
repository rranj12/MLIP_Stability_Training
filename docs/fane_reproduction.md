# GemNet-T / naphthalene protocol reproduction — prepared, not launched

Target: the GemNet-T naphthalene entry in *Forces are not Enough*, Table 3 of the
user's downloaded / arXiv v2 paper: 1.5 meV/Å force MAE and 8 ± 2 ps stability.
These are reference observations, not guaranteed outcomes or training targets to
select runs against. No GPU job, dataset download, environment installation or
training was performed for this setup.

## Configs and execution stack

- `configs/reproduce/fane_gemnet_t_naphthalene/train.yml`: complete upstream MDsim
  training config. Original MD17 10k split (9,500 train / 500 validation), full
  GemNet-T, batch 1, AdamW LR .001, force/energy weights 1000/1, EMA .999,
  ReduceLROnPlateau factor .8 and patience 5. Stop at 2,000 epochs, seven days of
  recorded training time, or LR floor 1e-6, whichever comes first.
- `simulate.yml`: upstream MDsim simulation config: 500 K, NoseHoover, 0.5 fs,
  `ttime: 20` (20 timesteps = 10 fs in this implementation), 600,000 steps / 300 ps,
  save every 100 steps / 0.05 ps, original MD17 test starts.
- `protocol.yml`: source revision and hashes, dataset sizes, checkpoint semantics,
  five predeclared seeds/starts, metric, pricing scenarios and unresolved items.
  This is an experiment manifest, not an executable trainer config.

Use the **authors' MDsim entrypoints**, pinned to commit
`61ca2cfba373fd374343592556789fa30fbe3b56`, from https://github.com/kyonofx/MDsim.
The configs are intentionally separate from our rMD17 sweep. Do not use
`scripts/pretrain.sh`, `scripts/evaluate.py`, or `cloud/modal/app.py::main` for this
reproduction: those use our StABlE/data defaults and Langevin evaluation. Merely
setting `integrator: NoseHoover` in our current evaluation YAML would not implement
that thermostat.

The upstream checkout/environment is not installed by this setup. Follow its
installation instructions in a separate environment; its reference versions are
Python 3.9.11, Torch 1.11.0+cu113 and CUDA 11.3. ASE and PyG were originally nightly
versions, so the README alone does not specify an exactly reproducible environment.
Our current Torch 2.3/StABlE Modal image is not that historical stack.

## Data, paths and weights

Keep original MD17 isolated at `/vol/data/forces_not_enough/md17/naphthalene/10k/`.
Never point this at our existing rMD17 `data/processed/md17/...` directory or its
`10k` reference symlink. Prefer the authors' [preprocessed release](https://zenodo.org/records/7196767),
whose `mdsim_data.tar.gz` is ~7.28 GB; verify its published checksum and archive
layout before extracting. The [source preprocessing](https://github.com/kyonofx/MDsim/blob/61ca2cfba373fd374343592556789fa30fbe3b56/preprocessing/md17.py)
uses original MD17 and split random state 123; its old download URL has reported
availability problems. Validate 9,500 / 500 / 10,000 examples after preparation.
Raw original MD17 coordinates are also needed for the reference bond-length metric.

`normalize_labels: false` is intentional: upstream still subtracts training mean
energy but uses unit scale for energy and force. Do not disable mean subtraction.

Training output root: `/vol/results/reproduce/fane_gemnet_t_naphthalene`.
With identifier `seed0`, the expected model directory is
`md17-naphthalene_10k_gemnet_t_seed0/`. Evaluate upstream **EMA**
`checkpoints/best_checkpoint.pt`; do not substitute our matched raw
`best_training_checkpoint.pt`. Model seed 0 is the upstream CLI default, not a
verified seed for the published table checkpoint. We did not find that exact
checkpoint in the authors' public repository listing.

## Future commands — do not run until separately authorized

These commands assume a prepared upstream environment/data and that this repository
is mounted at `/root/repo`. Run from the pinned upstream MDsim checkout so the
GemNet scale-file path resolves. They are not wired into an automatic launcher.

```bash
TASK_REPO=/root/repo
python main.py --mode train \
  --config-yml "$TASK_REPO/configs/reproduce/fane_gemnet_t_naphthalene/train.yml" \
  --seed 0 --identifier seed0 \
  --run-dir /vol/results/reproduce/fane_gemnet_t_naphthalene
```

After training and force-MAE validation, one of the five planned trajectories is:

```bash
python simulate.py \
  --config_yml "$TASK_REPO/configs/reproduce/fane_gemnet_t_naphthalene/simulate.yml" \
  --seed 123 --init_idx 857
```

Use pairs `(seed, init_idx)` = `(123,857), (124,4385), (125,1428), (126,6672),
`(127,4367)` from the manifest. Starts are a fixed sample without replacement from
10,000 test entries using Python Random(123), declared before observing any result.
The paper does not publish these exact five starts/seeds. There is no requirement
that every trajectory fail at 8 ps or that five fresh runs reproduce the same mean.

## Stability metric and known discrepancies

The authors' [observable notebook](https://github.com/kyonofx/MDsim/blob/61ca2cfba373fd374343592556789fa30fbe3b56/observable.ipynb)
constructs connectivity from the first simulated frame with ASE
`NeighborList(natural_cutoffs(atoms), self_interaction=False)`. It computes mean
reference lengths over the **full original MD17 reference trajectory**, using
30 Å periodic distances. It scans saved frames for a deviation >0.5 Å. This differs
from our training-only topology/mean and 0.216 Å collection threshold.
The upstream MD runner's early-stop flag instead detects temperatures outside
0.1–100,000 K. Do not report that flag as the 0.5 Å bond failure time. The notebook
metric needs a validated analysis step, including frame/log alignment and censoring,
before this is a complete executable reproduction pipeline. No metric implementation
or extra simulation was added in this configuration-only task.

Other differences are recorded rather than silently chosen:

- **2,000 versus 10,000 epochs:** paper Table 7 says 2,000; the upstream GemNet YAML
  says 10,000. Prepared config follows the paper's 2,000. Do not assume either limit
  identifies the exact training length of the published checkpoint.
- **2024 correction:** pinned upstream includes the periodic-coordinate wrapping fix.
  The downloaded paper is arXiv v2; the corrected OpenReview manuscript was inaccessible
  in prior checks. Verify its current table before claiming strict replication.
- **Boundary conditions:** paper describes MD17 as nonperiodic, while implementation
  uses 30 Å boxes with PBC flags. Preserve/audit the actual upstream behavior.
- **Hardware:** proposed A100 differs from the paper's V100 PCIe. A seven-day stop on
  A100 is not an identical compute budget. It is also an epoch-boundary stopping rule,
  not a cloud hard runtime or spending cap. A bounded, resumable execution plan must
  be selected before any paid launch.

## Time and cost: scenarios, not a measured convergence forecast

[Modal pricing](https://modal.com/pricing), checked November 25, 2026:
A100 80 GB $0.000694/s = **$2.4984/hour**; 40 GB $0.000583/s = **$2.0988/hour**.
The current generic `A100` request can select either; choose explicitly for the
future reproduction. Prices below are GPU only, before CPU, RAM, storage, startup/
shutdown overhead or credits. Evaluation is additional.

| Training duration | A100 80 GB | A100 40 GB |
|---|---:|---:|
| 12 hours | $29.98 | $25.19 |
| 24 hours | $59.96 | $50.37 |
| 72 hours | $179.88 | $151.11 |
| 168 hours (seven days) | $419.73 | $352.60 |

We have **not measured batch-size-1 reproduction throughput or convergence**.
Our one-epoch smoke took 12.15 s for 119 updates at batch 8 on 950 structures;
that cannot establish runtime for the paper setup. At batch 1, an epoch is roughly
9,500 updates; 2,000 epochs would be about 19 million updates. Illustrative step
costs of 0.02–0.10 s imply 3.2–15.8 minutes per epoch, excluding validation/I/O;
these step costs are assumptions, not measurements. For example, 100 epochs would
then be 5.3–26.4 hours, whereas 500 epochs would be 26.4–131.9 hours. No basis yet
exists for predicting how many epochs achieve the paper's accuracy.

Budget for potentially days of training; do not promise a fixed completion cost.
The first paid step should be a separately capped timing/convergence pilot at the
matching batch size, after data/environment verification. No pilot is authorized
or launched by these files. A smaller training budget can be a useful pilot but
must not be presented as a completed reproduction if its accuracy has not converged.

## Meaning of the previous gradient check

It tested our existing smooth, temperature-scaled margin loss:

`L_margin = kT * mean(softplus((m - [U(distorted) - U(anchor)]) / kT))`.

Descent on that term should increase the energy gap when it is below the requested
margin. The real GemNet gradient check observed exactly that. The diagnostic
changed **m** from 0.430867 to 1.611350 eV because the chosen distorted structures
already exceeded the normal margin; it did not introduce a new formula or change
production defaults. The matched four-update mixed-loss comparison at lambda 1
showed negligible difference from its control. That is a functional verification,
not demonstrated improvement in MD stability. The paper-baseline training config
above contains only the standard energy/force objective; it has no margin loss.
