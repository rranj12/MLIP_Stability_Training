# GemNet-T lambda sweeps

`configs/sweep/gemnet_t.yml` configures naphthalene and aspirin on rMD17, with a
separate selected lambda for each molecule. It inherits the data paths and held-out
tune/test protocol from `configs/sweep/lambda.yml`. Configuration alone launches no jobs.

## Controlled comparison

- Stage B uses a fixed 3 × 500 = 1,500 optimizer updates for every lambda, including
  zero. Low failure fractions and satisfied margins do not stop these runs. If no
  failure pairs exist, supervised energy/force updates still run.
- The accuracy guardrail remains active. It restores the previous cycle's weights
  and aborts on a >15% validation-force-MAE increase over Stage A. Such candidates
  are excluded from selection; if the zero-lambda control aborts, selection stops
  with an error. `history.json` records planned, attempted and retained updates.
- Each seed loads its own independently pretrained Stage A model. Within that seed,
  all lambdas and the Stage A evaluation use the same checkpoint. The train/val/tune/test
  split stays fixed across model seeds. The backend checks the recorded Stage A seed.
- MDsim pretraining preserves the upstream EMA export (`best_checkpoint.pt`) and
  saves `best_training_checkpoint.pt`: raw model weights and optimizer state from
  the same selected training step. Our backend uses this matched raw checkpoint
  for both Stage A evaluation and all Stage B arms whenever it is present. Selection
  of the step still uses upstream EMA validation; these are not EMA weights.
- Optimizer reuse restores parameter groups, moments, step counters and group
  settings, then applies the configured Stage B learning rate. GemNet's fixed
  scaling factors remain frozen during Stage B. A missing matched
  checkpoint, wrong optimizer class or mismatched weights raises an error. Older
  runs need new pretraining, or explicit `reuse_stage_a_state: false`; there is no
  silent fresh-optimizer fallback. NequIP reuse is rejected because its latest
  `trainer.pth` is not guaranteed to match `best_model.pth`. `history.json` records
  the optimizer source and initial/final step ranges for auditing.
- Tuning uses seeds 0 and 1; final evaluation uses 0, 1 and 2. The final evaluation
  reuses the trained candidates for seeds 0 and 1 on separate test frames.

The provisional grid is `0, 0.001, 0.01, 0.1, 1, 10, 100`. It brackets a much wider
range than the old SchNet sweep because GemNet-T uses a different supervised loss
scale. These are not validated weights; inspect pilot behavior before committing to
the full sweep. Selection minimizes bond-only failure fraction within a 5% tune
force-MAE budget relative to the control, choosing the smallest lambda within one
estimated standard error of the best. Combined failures remain in evaluation outputs.

Training uses 32 rollout replicas and batches of 8 labeled structures and 8 failure
pairs. Both tuning (100 ps) and final evaluation (300 ps) use 64 replicas per seed
to limit GemNet-T memory use. If more precision is needed, increase the evaluation
replica count before running the campaign. The full tuning grid contains 28 jobs.

## Independent Stage A checkpoints

The sweep resolves `pretrained_dir: results/checkpoints/qm_pretrained/seed{seed}`.
For example, seed 1 for aspirin must be at:

```
results/checkpoints/qm_pretrained/seed1/gemnet_t/md17-aspirin_1k_gemnet_t/checkpoints/
```

`TRAIN_SEED` supplies the model seed and isolates the pretraining output directory.
The pretraining wrapper links the completed checkpoint into this layout automatically.
Omitting `TRAIN_SEED` retains the legacy unseeded directory layout, which this sweep
does not use. Existing unseeded checkpoints are not silently substituted.

Commands below are for a future launch, after reviewing the configuration. Prepare
each molecule once before starting independent pretraining jobs:

```bash
python scripts/prepare_rmd17.py --molecule naphthalene --reference same
python scripts/prepare_rmd17.py --molecule aspirin --reference same

# Repeat for each molecule and seeds 0, 1, 2; seed 2 is needed for final evaluation.
TRAIN_SEED=0 scripts/pretrain.sh gemnet_t naphthalene
TRAIN_SEED=0 scripts/pretrain.sh gemnet_t aspirin
```

On Modal, use the new `--seed` option, which forwards the same model seed and selects
the corresponding checkpoint root for subsequent margin/evaluation steps:

```bash
modal run --detach cloud/modal/app.py::main --molecule naphthalene --model gemnet_t --seed 0 --steps prepare,pretrain
modal run --detach cloud/modal/app.py::main --molecule aspirin --model gemnet_t --seed 0 --steps prepare,pretrain
```

Repeat with seeds 1 and 2, using `--steps pretrain` after data preparation is complete.
Do not run two pretraining jobs for the same molecule/model/seed concurrently.

## Plan, tune and evaluate

This command only prints the job plan; it does not train or evaluate:

```bash
python scripts/sweep_lambda.py plan --config configs/sweep/gemnet_t.yml
```

After Stage A checkpoints are ready and the pilot is satisfactory, these commands
launch the tuning and final jobs respectively:

```bash
modal run --detach cloud/modal/app.py::sweep --config configs/sweep/gemnet_t.yml --phase tune
modal run --detach cloud/modal/app.py::sweep --config configs/sweep/gemnet_t.yml --phase final
```

Review `results/sweeps/gemnet_t_fixed_seeded/selection.md` before final evaluation.
The new campaign name keeps older variable-budget, shared-checkpoint results from
being reused. Changing an already-run experiment requires a fresh campaign name;
the sweep skips output files that already exist.
