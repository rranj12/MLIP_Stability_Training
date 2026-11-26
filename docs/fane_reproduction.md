# GemNet-T / naphthalene protocol reproduction

Target: the GemNet-T naphthalene entry in *Forces are not Enough*, Table 3 of the
user's downloaded / arXiv v2 paper: 1.5 meV/Å force MAE and 8 ± 2 ps stability.
These are reference observations, not guaranteed outcomes or training targets to
select runs against. Original MD17 data and the isolated environment are prepared;
a one-epoch timing pilot **completed** within a cumulative one-GPU-hour cap.
The subsequent single-trajectory MD diagnostic completed **20 ps without a detected
bond failure**. This has **not reproduced the older paper's reported ~8 ps failure**.
Full training and sweeps remain unauthorized.

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

The isolated pilot environment is defined in `cloud/modal/fane_pilot.py`; its
training/data entrypoint is `scripts/fane_pilot.py`. The historical reference is
Python 3.9.11, Torch 1.11.0+cu113 and CUDA 11.3. Correction to the initial setup
notes: the README **does** pin PyG to `pyg-nightly==2.0.5.dev20220418`; ASE master
was unpinned. Our current Torch 2.3/StABlE Modal image is not that historical stack.
Modern Modal rejected a Python 3.9 host before allocating any GPU, so the pilot
uses an isolated Python 3.9.23 environment inside a supported Python 3.11 host.
Python patch version, Linux base, ASE 3.22.1 and A100 hardware are disclosed
differences from the historical environment; exact installed packages are saved.
The CUDA 11.3 cuSPARSE/cudart runtime packages come from
[NVIDIA's archive](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/).
They are explicitly included in the library search path because the GPU container
did not retain the lookup behavior of the image-build loader cache.

## Data, paths and weights

Keep original MD17 isolated at `/vol/data/forces_not_enough/md17/naphthalene/10k/`.
Never point this at our existing rMD17 `data/processed/md17/...` directory or its
`10k` reference symlink. Prefer the authors' [preprocessed release](https://zenodo.org/records/7196767),
whose `mdsim_data.tar.gz` is ~7.28 GB; verify its published checksum and archive
layout before extracting. The [source preprocessing](https://github.com/kyonofx/MDsim/blob/61ca2cfba373fd374343592556789fa30fbe3b56/preprocessing/md17.py)
uses original MD17 and split random state 123; its old download URL has reported
availability problems. Validate 9,500 / 500 / 10,000 examples after preparation.
Raw original MD17 coordinates are also needed for the reference bond-length metric.

**Pilot preparation verified:** the full 7,279,215,356-byte archive matches published
MD5 `2f895e74a32793e37da43b961ce8edb2`. The extracted NPZ/LMDB files store
9,500/500/10,000 examples. The release writes the last index into LMDB `length`,
and the unmodified upstream loader uses that as a count: it exposes **9,499/499/9,999**.
The pilot preserves this behavior, so one epoch means **9,499 updates** at batch 1.
Finite-array, sample-label/position agreement, metadata and full-model CPU
energy/force/backward checks passed. Provenance and checks live alongside the
data in `provenance.json` and `validation.json` on volume `mlipstab`.
The full original reference trajectory is now prepared: 326,250 frames from the
[official original-MD17 file](https://www.quantum-machine.org/gdml/data/npz/md17_naphthalene.npz).
Its SHA256 is `8939b910d1e91870db7515baa9d49d23169fdf8108687c4c0659afe38a6f053d`.
Sample original frame IDs, positions, atom identities, energies and forces agree
with all three released splits. The capped diagnostic uses full-trajectory bond means.

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
0.1–100,000 K. Do not report that flag as the 0.5 Å bond failure time.
`scripts/fane_md_smoke.py` now implements the notebook geometry criterion and
independently audits trajectory/log alignment and censoring. The diagnostic checks
the terminal frame too, which the literal notebook loop omits; both results are
reported separately. The full five-trajectory benchmark has not been run.

## Capped MD diagnostic on the one-epoch checkpoint

Requested seed 123, predeclared test index 857; 500 K Nosé–Hoover, 0.5 fs,
`ttime=20`, 100-step saving, one trajectory up to 20 ps or first bond failure.
GPU hard timeout 1,800 s, internal wall stop 1,650 s; no training or automatic retries.
The authors' initialization order is preserved: calculator construction reseeds
from its training config (seed 0), then Maxwell–Boltzmann momenta are drawn.
Actual initial momenta and checkpoint identity are saved.

**Outcome:** 40,000 steps / **20.000 ps**, 401 saved frames, **no detected bond
failure**. Maximum deviation **0.216945365 Å at 18.7 ps**, below the **0.5 Å**
criterion. No temperature guard/numerical failure; **zero failure pairs**.
Runtime **1,106.792 s / 18.45 min**, approximately **$0.7681 GPU compute**, before
CPU/RAM, startup/shutdown overhead, storage or credits. Independent CPU audit passed.
This is a censored observation at the horizon, not a failure time of 20 ps.

Artifacts: volume `mlipstab` and local ignored results folder
`results/reproduce/fane_gemnet_t_naphthalene/md_smoke_seed123_init857_20ps/`.
Entry point: `modal run --detach cloud/modal/fane_pilot.py::md_test`; it refuses to
overwrite this completed diagnostic. See the ongoing log for the app and full audit.

This matches the documented **MD recipe**, with explicitly noted implementation
details, on a **one-epoch** checkpoint. It is not a replication of the paper's
converged checkpoint, five 300-ps trajectories, final test accuracy, or its older
8 ± 2 ps result. More training is not guaranteed to create failures. Reconcile the
2024 correction before funding a longer training run on that assumption.

## Remaining replication differences

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

## Measured one-epoch pilot and cost

[Modal pricing](https://modal.com/pricing), checked November 25, 2026:
A100 80 GB $0.000694/s = **$2.4984/hour**; 40 GB $0.000583/s = **$2.0988/hour**.
The pilot explicitly requested A100 80 GB. Prices below are GPU only, before CPU,
RAM, storage, startup/
shutdown overhead or credits. Evaluation is additional.

| Training duration | A100 80 GB | A100 40 GB |
|---|---:|---:|
| 12 hours | $29.98 | $25.19 |
| 24 hours | $59.96 | $50.37 |
| 72 hours | $179.88 | $151.11 |
| 168 hours (seven days) | $419.73 | $352.60 |

The November 25 pilot measured the actual upstream batch-size-1 pipeline:

| Measurement | Result |
|---|---:|
| Epochs / optimizer updates | 1 / 9,499 |
| Upstream training + validation loop | 1,362.635 s / 22.71 min |
| Entire training subprocess | 1,380.070 s / 23.00 min |
| Successful GPU function | 1,381.670 s / 23.03 min |
| Both GPU attempts' recorded function work | 1,384.149 s / 23.07 min |
| Rate-based GPU compute estimate, both attempts | $0.9606 |
| EMA validation force MAE (499 examples) | 17.642864 meV/Å |
| EMA validation energy MAE | 0.048213615 eV |

The loop averages about **6.97 updates/s**, including validation. Rough extrapolation:
**100 epochs ≈38 hours / $95 GPU-only**. This replaces the earlier unmeasured
0.02–0.10 s/update scenario; our old 119-update rMD17 smoke is not representative.
**Convergence is still unmeasured**: one epoch cannot predict the number needed
to reach the paper's final test accuracy. Costs are estimates, not an invoice.

Both raw training/optimizer and EMA checkpoints were saved. The EMA checkpoint
reloaded through the authors' ASE `OCPCalculator` on CPU and produced finite energy
and 18×3 forces for one test structure; raw optimizer states were finite and their
step counters were all 9,499. **No MD steps or margin updates were performed.**
The training and verification apps are stopped. Artifacts live on volume `mlipstab`
at `results/reproduce/fane_gemnet_t_naphthalene/pilot_seed0_retry1/` and in the same
relative local results directory. See [the ongoing log](experiment_log.md) for
app IDs, setup fixes and complete provenance.

Budget for potentially days of training; do not promise a fixed completion cost.
The authorized first step is a separately capped **one-epoch timing pilot** at the
matching batch size, after CPU data/environment verification. `cloud/modal/fane_pilot.py`
has no automatic retries and refuses to overwrite an existing pilot. The first
GPU invocation failed on an import before any update (2.479 s of recorded function
work). Its output is retained in `pilot_seed0/`. The corrected invocation uses
`pilot_seed0_retry1/`, a **3,300-second hard timeout** and a 3,150-second subprocess
stop. Reserving 300 seconds for the failed invocation/startup keeps total GPU use
below one hour. CPU preparation and image builds precede GPU use.
A smaller training budget can be a useful pilot but
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

### Numerical evidence from the saved diagnostic

For one pair, writing `gap = U(distorted) - U(anchor)` and `s = kT`,
`dL/dgap = -sigmoid((m-gap)/s)` (with an additional `1/N` for a batch mean).
The sign encourages a larger gap. Shared model parameters and the supervised
objective mean this does not guarantee that every pair's gap grows during training.

The saved real-GemNet report is
`results/smoke/gemnet_t_naphthalene_distortion_20261125/report.json`.
The diagnostic restored the same weights before perturbing them in each direction,
using `v = gradient / ||gradient||` and parameter step norm `h = 0.001`:

| Measurement | Value |
| --- | ---: |
| Original margin loss | 0.3425830305 eV |
| Loss at `theta - h*v` | 0.3385159969 eV |
| Loss at `theta + h*v` | 0.3466821313 eV |
| Autograd directional derivative `gradient dot v` | 4.0834300429 |
| Central finite difference `(L(theta+h*v)-L(theta-h*v))/(2*h)` | 4.0830671787 |
| Relative discrepancy | 0.0088863% |

The finite-difference comparison was calculated from existing saved values; no
model execution or new experiment was needed. Descent also increased the mean
distorted-minus-anchor energy gap from 1.271212 to 1.275649 eV; ascent reduced it
to 1.266790 eV. This is evidence for a correct local parameter-gradient direction
and magnitude at one point, one direction and one step size. A stronger future
numerical audit would test several step sizes and independent directions. It is
not a proof of all gradient components, restoring forces, or longer MD survival.
This check used the earlier rMD17 checkpoint and deliberately active diagnostic
margin, not the original-MD17 pilot checkpoint or naturally collected failures.

### 2024 correction audit (2026-11-25)

The [official README](https://github.com/kyonofx/MDsim) announces an October 29,
2024 code and TMLR paper update. The linked
[FAIR-Chem PR 783](https://github.com/facebookresearch/fairchem/pull/783) explains
that periodic graph construction could miss neighbors when unwrapped coordinates
became too far apart for its finite set of periodic image cells. The correction
wraps coordinates before graph construction. MDsim commit
[`7678721`](https://github.com/kyonofx/MDsim/commit/7678721f22b2be44aeaa65016c4d0f3053afeec0)
adds this operation in `atoms_to_batch`; our pinned code includes it.

**The effect on the published GemNet-T/naphthalene number remains unverified.**
The current OpenReview forum/PDF, public APIs and a search-indexed TMLR PDF asset
were blocked by browser verification/HTTP 403. The downloaded arXiv v2 is from
2023, so its 8 +/- 2 ps result cannot establish what the 2024 revision reports.
The bug description alone does not show that it caused that particular result.
Do not attribute our 20 ps survival to the fix without further evidence, and do
not treat 8 +/- 2 ps as a verified target for the corrected implementation.
Next required evidence is the corrected manuscript's MD17 table and correction
statement from [the TMLR record](https://openreview.net/forum?id=A8pqQipwkt).
No new training, MD or sweep was launched for this audit.
