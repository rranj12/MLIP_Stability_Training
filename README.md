# Stability-Aware MLIP Training

CS230 project. Neural-network interatomic potentials can match DFT on test data
and still blow up in molecular dynamics. We find the failures a model produces in
its own simulations and raise their energy with a **margin loss** applied only to
physically flagged states, then compare head-to-head with the two closest published
fixes, **StABlE** (observable matching) and **CRMSE** (contrastive maximum likelihood),
under one MD protocol on rMD17 (and MD22).

```
L = L_EF + λ · mean softplus( m − [U(x_fail) − sg(U(x_anchor))] )
```

`x_fail` fails a bond-deviation check during a Langevin rollout, `x_anchor` is the
same replica a few hundred steps earlier, `m` is a margin in units of k_BT, and
`sg` is a stop-gradient. See [docs/design_notes.md](docs/design_notes.md).

## Status

| Piece | State |
|---|---|
| Margin loss, localized variant, CRMSE term + ULA step | implemented, unit-tested |
| Failure detection (bonds, non-bonded collisions), pair extraction, replay buffer | implemented, unit-tested |
| Stage B loop (simulate → detect → buffer → learn → guardrail) | implemented, unit-tested with toy callables |
| Stability time, Kaplan–Meier, h(r), dihedral free energy | implemented, unit-tested |
| rMD17 download + splits | implemented, checked on real ethanol data |
| rMD17 → StABlE LMDB layout (`scripts/prepare_rmd17.py`) | implemented, needs a run in the full env |
| Pretraining / StABlE baseline wrappers | thin wrappers around the submodule |
| MDsim glue for our Stage B, CRMSE loop, evaluation harness | `TODO` stubs with pointers to the StABlE code to reuse |
| NequIP rMD17 pretraining config, MACE | to add in the pilot |

## Layout

```
configs/
  margin/        our method (base.yml + per molecule/model)
  crmse/         CRMSE baseline (paper settings)
  stable/        StABlE baseline: includes StABlE's own configs, overrides paths only
  eval/          one evaluation protocol for every method
src/mlipstab/
  losses/        margin.py, crmse.py
  train/         stage_b.py (backend-agnostic loop)
  data/          rmd17.py (download, splits), bonds.py (bond list, collision floors)
  eval/          stability.py (tau, Kaplan-Meier), observables.py (h(r), dihedrals)
  detect.py  buffer.py  units.py  config.py
scripts/
  prepare_rmd17.py      data -> StABlE layout
  pretrain.sh           Stage A via StABlE's pretraining.py
  run_stable.sh         StABlE baseline, unmodified
  train_stability.py    Stage B for ours / CRMSE
  evaluate.py           shared evaluation harness
  analysis/             survival curves, Pareto, energy scans, ...
third_party/StABlE-Training   pinned git submodule (MDsim models, integrators, StABlE)
cloud/aws/      instance setup and bootstrap
docs/           design notes, compute plan
data/ results/  local only (gitignored)
tests/          CPU unit tests
```

## Quickstart

```bash
git clone --recursive https://github.com/rranj12/MLIP_Stability_Training.git
cd MLIP_Stability_Training
mamba env create -f environment.yml && conda activate mlipstab
pytest                                                         # CPU unit tests

python scripts/prepare_rmd17.py --molecule aspirin --reference same
scripts/pretrain.sh schnet aspirin                             # Stage A
scripts/run_stable.sh configs/stable/aspirin_schnet_train.yml  # StABlE baseline
python scripts/train_stability.py --config configs/margin/aspirin_schnet.yml
python scripts/evaluate.py --config configs/eval/rmd17_500K.yml \
    --molecule aspirin --model schnet --checkpoint <ckpt> --tag margin_seed0
```

## Experiment plan

Core: {ethanol, aspirin} × {SchNet, NequIP} × {baseline, StABlE, CRMSE, ours} × 3 seeds,
all from the same Stage A checkpoint, evaluated with 256 Langevin replicas at 500 K for
300 ps (0.5 Å bond criterion + non-bonded collisions). Then: MD22 Ac-Ala3-NHMe, MACE,
temperature transfer, xTB margins. Estimates and budget: [docs/compute.md](docs/compute.md).

Fairness rules: every baseline runs with its authors' optimizer and settings; StABlE's
main baseline computes its reference h(r) from the same 950 training frames as everything
else (`--reference same`), with the published extra-data setting reported separately.

## Attribution

Built on [StABlE-Training](https://github.com/ASK-Berkeley/StABlE-Training)
(Raja et al., TMLR 2025), which builds on MDsim (Fu et al., MIT). StABlE has no license
file, so it is included as an unmodified, pinned submodule rather than copied.
CRMSE (Tzivrailis et al., arXiv:2606.31660) is reimplemented from the paper. Data: rMD17
(Christensen & von Lilienfeld, 2020), MD22 (Chmiela et al., 2023).
