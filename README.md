# Stability-Aware MLIP Training

Neural-network interatomic potentials can match DFT on test data
and still blow up in molecular dynamics. We find the failures a model produces in
its own simulations and raise their energy with a **margin loss** applied only to
physically flagged states, then compare head-to-head with the two closest published
fixes, **StABlE** (observable matching) and **CRMSE** (contrastive maximum likelihood),
under one MD protocol on rMD17 (and MD22).

```
L = L_EF + λ · k_BT · mean softplus( (m − [U(x_fail) − U(x_anchor)]) / k_BT )
```

`x_fail` fails a bond-deviation check during a Langevin rollout, `x_anchor` is by
default the same replica a few hundred steps earlier, and `m` is a margin in units of
k_BT. Gradients flow through both energies, so a global energy offset cancels. Both
snapshots are detached from the rollout. Optional: a counterfactual anchor (the broken
bond reset to its equilibrium length), a labeled anchor (nearest training frame), and a
bond-restoring force hinge. See [docs/design_notes.md](docs/design_notes.md).

## Status

Current experiments, fixes, run IDs, and next steps are maintained in
[the ongoing experiment log](docs/experiment_log.md).

| Piece | State |
|---|---|
| Margin loss, localized variant, CRMSE term + ULA step | implemented, unit-tested |
| Anchor options (rollout / counterfactual / labeled), bond-restoring force hinge | implemented, unit-tested |
| Failure detection (bonds, non-bonded collisions), pair extraction, replay buffer | implemented, unit-tested |
| Stage B loop (simulate → detect → buffer → learn → guardrail) | implemented, unit-tested with toy callables |
| Stability time, Kaplan–Meier, h(r), dihedral free energy | implemented, unit-tested |
| rMD17 download + splits | implemented, checked on real ethanol data |
| rMD17 → StABlE LMDB layout (`scripts/prepare_rmd17.py`) | verified on real naphthalene in the GPU smoke pipeline |
| Pretraining / StABlE baseline wrappers | thin wrappers around the submodule |
| MDsim glue (`mlipstab.mdsim_backend`), Stage B and CRMSE scripts, evaluation harness | GemNet-T/naphthalene training, optimizer continuation and evaluation GPU smoke verified; see experiment log for active-margin verification; CRMSE and other models are not validated by this smoke |
| NequIP rMD17 pretraining (`scripts/pretrain.sh nequip <mol>`) | implemented from StABlE's MD17 NequIP configs, not yet run |
| MD22 molecules and GemNet-T (`scripts/prepare_md22.py`, `DATASET=md22 scripts/pretrain.sh`, model `gemnet_t`) | GemNet-T GPU smoke run on real rMD17 naphthalene; real MD22 GPU validation remains outstanding |
| MACE, analysis scripts | to add |

## Layout

```
configs/
  margin/        our method (base.yml + per molecule/model; md22.yml for MD22 molecules)
  crmse/         CRMSE baseline (paper settings)
  stable/        StABlE baseline: includes StABlE's own configs, overrides paths only
  eval/          one evaluation protocol for every method (rmd17_500K.yml, md22_500K.yml)
src/mlipstab/
  losses/        margin.py, anchors.py, crmse.py
  train/         stage_b.py (backend-agnostic loop)
  data/          rmd17.py (download, splits), bonds.py (bond list, collision floors)
  eval/          stability.py (tau, Kaplan-Meier), observables.py (h(r), dihedrals)
  detect.py  buffer.py  units.py  config.py
scripts/
  prepare_rmd17.py      data -> StABlE layout
  prepare_md22.py       same layout for the seven MD22 molecules (42 to 370 atoms)
  pretrain.sh           Stage A via StABlE's pretraining.py (schnet, gemnet_t) or nequip-train
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

GemNet-T uses the same commands with the model name `gemnet_t`
(`scripts/pretrain.sh gemnet_t aspirin`, `configs/margin/aspirin_gemnet_t.yml`).
An MD22 molecule takes its own data script and configs:

```bash
python scripts/prepare_md22.py --molecule AT_AT_CG_CG          # 950 / 50 / 1,000 random frames
DATASET=md22 scripts/pretrain.sh schnet AT_AT_CG_CG
python scripts/train_stability.py --config configs/margin/md22.yml \
    --molecule AT_AT_CG_CG --model schnet
python scripts/evaluate.py --config configs/eval/md22_500K.yml \
    --molecule AT_AT_CG_CG --model schnet --checkpoint <ckpt> --tag margin_seed0
```

SchNet and GemNet-T find neighbors by comparing every atom pair in 27 periodic images, so
memory grows with replicas (or batch size) times atoms squared: about 0.05 GB per replica
at 118 atoms and 0.5 GB at 370, measured on CPU. At 370 atoms (`double_walled_nanotube`)
lower `simulate.n_replicas` to about 16 and pass `--optim.batch_size=16` to
`scripts/pretrain.sh`.

## Experiment plan

GemNet-T on naphthalene and aspirin has a per-system lambda sweep in
`configs/sweep/gemnet_t.yml`, with fixed Stage B update budgets and independent
Stage A checkpoints per seed. Setup, protocol and launch commands:
[lambda sweeps](docs/lambda_sweeps.md).

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
