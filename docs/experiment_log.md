# Ongoing experiment log

Last updated: 2026-11-25. Append dated entries as experiments finish; keep this
status summary current. This is the teammate handoff record. Detailed binaries,
logs and metrics live in ignored `results/` locally and on Modal volume `mlipstab`;
the essential decisions and results are recorded here so they can travel with Git.

## Current status

- Full sweeps have **not** been launched. Only bounded GemNet-T/naphthalene smoke
  tests are authorized so far. Aspirin and independent campaign seeds 0–2 remain unrun.
- GemNet-T uses the full 1,891,025-parameter architecture, not a reduced toy model.
- Seed-isolated pretraining, fixed update budgets, matching optimizer restoration,
  fixed GemNet scaling factors, checkpoint reload and short evaluation have passed.
- Progress logging and partial-trajectory persistence are **verified on GPU**, including
  post-timeout recovery. The latest **20 ps / 30-minute-capped** diagnostic preserved
  **19.155 ps per replica**, two replicas, **zero failures**. Timeout occurred before
  training or evaluation; **0/4 updates** were reached. App `ap-SSODGspmlyG82BVDbHHj6o`
  is confirmed stopped. Active margin on naturally collected failures remains
  **unverified**. A subsequent synthetic bond-distortion diagnostic verified the
  gradient direction and matched optimizer updates; its mixed-loss benefit at
  lambda 1 was negligible. Details and limitations are recorded below.
- Latest local suite: **60 tests passed**, including progress/partial-snapshot regressions.
  The subsequent distortion/loss/optimizer targeted check passed **17 tests**.
- Fixes, configs, tests and this handoff were committed and pushed as `6ffeab0` on
  `fix/margin-scale-and-bond-only-eval`. The final 20 ps diagnostic outcome below
  is included in a follow-up documentation commit. No further GPU work is running.

## 2026-11-25 — Repository and experiment direction

1. Pulled/merged main into `fix/margin-scale-and-bond-only-eval`; local merge commit
   `e961c22`. Preserved work on the current branch.
2. Reviewed the codebase around the user's research question: force accuracy versus
   MD stability, with “Forces Are Not Enough” as a motivating paper supplied locally.
   A smoke test is an execution check, not evidence of a paper-level improvement.
3. Selected GemNet-T on rMD17 naphthalene and aspirin for the next configured studies.
   Lambda selection is per molecule; the current code/config does not establish a
   transferable optimum across molecules.

## 2026-11-25 — Control budget, seeds and sweep preparation

- Stage B gained `fixed_budget`: lambda zero and nonzero arms receive the same
  planned number of updates, including supervised updates when the failure buffer
  is empty. Low failure fraction or a satisfied margin no longer ends a fixed-budget run.
- The validation-force-MAE guardrail can still stop a run and restore the previous
  cycle's weights. Histories distinguish planned, attempted and retained updates.
  Guardrail-aborted candidates are excluded; an aborted zero-lambda control blocks selection.
- `TRAIN_SEED` and Modal `--seed` isolate Stage A outputs. Sweep paths resolve
  `seed{seed}`, and the backend checks the recorded seed rather than silently sharing
  one model across seeds. Dataset splits stay fixed across model seeds.
- `configs/sweep/gemnet_t.yml`: naphthalene and aspirin, lambdas
  `0, .001, .01, .1, 1, 10, 100`; tuning seeds 0/1, final seeds 0/1/2;
  3 × 500 Stage B updates per candidate; per-system, bond-only selection.
  This is a provisional grid, not a calibrated recommendation.
- Detailed protocol and future commands: [lambda_sweeps.md](lambda_sweeps.md).
  No sweep was started.

## 2026-11-25 — First real GPU smoke

- Seed `20261125`, A100, real rMD17 naphthalene; one Stage A epoch, 950 train frames,
  batch size 8; lambda 0 and 1, each 2 cycles × 2 updates; 2-replica short evaluation.
- Both arms retained four updates. The failure buffer grew from 2 to 4 pairs and
  finite margin losses were recorded, but optimizer continuation was **not** valid:
  upstream `best_checkpoint.pt` lacked optimizer state, so the backend reset AdamW.
- The first A10G/L4 requests stayed queued and were canceled; completed jobs used A100.
- Local detailed report: `results/smoke/gemnet_t_naphthalene_20261125/README.md`.
- Completed Modal apps: Stage A `ap-T5PG2GLqRkE1fzzUPM3Axm`, margin
  `ap-PdK37QxMjtw1M7Co2lgYiu`, control `ap-uHl8pTYhFAQNZbjU4juadd`.

## 2026-11-25 — Optimizer and frozen-parameter fixes

- Upstream best weights are EMA weights; a latest/raw optimizer is not an exact
  continuation of those weights. `scripts/pretrain_mdsim.py` wraps upstream saving
  without editing the submodule: preserve the EMA export, additionally save
  `best_training_checkpoint.pt` with raw weights and optimizer at the same selected step.
- Our backend uses this raw checkpoint consistently for baseline and Stage B when
  present. Step selection still uses upstream EMA validation. It reconstructs exact
  optimizer parameter groups/order and settings, then applies the Stage B LR.
- Missing/mismatched optimizer checkpoints now raise rather than silently reset.
  Legacy runs need new pretraining or explicit `reuse_stage_a_state: false`.
  NequIP reuse is rejected because its latest trainer state is not guaranteed to
  match the best weights; explicit fresh state is still possible.
- A rerun exposed unconditional unfreezing of GemNet's 17 fixed scaling factors.
  MDsim constructor freeze flags are now preserved.
- New Stage A seed `20261126`: one epoch / 119 updates, 121 initialized AdamW states
  with finite nonzero moments. Final lambda-0 and lambda-1 tests restored all 121
  states at step 119 and finished at step 123, without creating scaling-factor states.
  Downloaded checkpoints confirmed all 17 scales unchanged and model weights updated.
- Both final tests retained all four updates; checkpoint reload and evaluation passed.
  All 57 CPU tests passed, including exact resumed-versus-uninterrupted update equality.
- **Limitation:** no failures occurred in these 0.1 ps rollouts. The lambda-1 arm
  therefore used only supervised loss; active margin training with the repaired
  optimizer remained unverified.
- Local report/evidence: `results/smoke/gemnet_t_naphthalene_optimizer_20261125/`.
  Only `*_verified` arms include the freeze fix; plain control/margin files are diagnostic.
- Completed apps: [Stage A](https://modal.com/apps/rishabh12ranjan/main/ap-OaxmuY9yjg884eZYg1tomm),
  [final control](https://modal.com/apps/rishabh12ranjan/main/ap-MZENCjIlqt6NHBhsf9PBjF),
  [final margin](https://modal.com/apps/rishabh12ranjan/main/ap-Fr3IoUTz6RwB8KRrTx4yzG).

## 2026-11-25 — Longer active-margin smoke (timed out; incomplete)

Purpose: collect a genuine model-generated failure and exercise the real margin
loss with the corrected optimizer, with a tiny update budget and bounded runtime.

- Reuse Stage A seed `20261126`; no new pretraining. One A100 GPU job, lambda 1 only.
- One cycle × four updates, labeled/pair batch sizes 2, AdamW LR `1e-5`, fixed budget.
- Two replicas at 500 K, 0.5 fs timestep, up to **10 ps** collection. Check bonds every
  2 integration steps and stop when at least one of the two replicas fails.
- Preserve the automatically calibrated bond threshold (~0.21618 Å), physical
  temperature, and default 500-step anchor lag. Do not manufacture failures by
  changing thresholds or temperature.
- Short post-training evaluation: two tune replicas, 0.1 ps, save every 5 fs,
  tune accuracy/reference, no NVE. This is not the production stability evaluation.
- `MLIP_RUN_TIMEOUT_S=600` sets the Modal function execution timeout; retries are
  disabled. The cap applies to training plus evaluation, excluding scheduling/build
  time, with possible shutdown grace. Do not extend it automatically if no failure is found.
- Estimate: 5–15 minutes wall time; approximately $0.20–$0.50 compute, with minor
  startup/storage overhead. [Modal pricing](https://modal.com/pricing), checked
  2026-11-25: A100 40 GB $0.000583/s, 80 GB $0.000694/s. GPU execution at the
  600-second limit is $0.35–$0.42 before CPU/RAM/startup/credits; this is not an invoice.
- Verification: nonempty failure buffer, finite positive `l_stab`, four retained
  updates, 121 optimizer states advanced from 119 to 123, unchanged 17 scaling
  factors, finite checkpoint and evaluation. Stop and report any unmet condition.
- Artifact root: `results/smoke/gemnet_t_naphthalene_active_margin_20261125/`.
  [Modal run ap-Pivm4HfQOHZ5Jdn10M3IdB](https://modal.com/apps/rishabh12ranjan/main/ap-Pivm4HfQOHZ5Jdn10M3IdB).
  Submitted approximately 17:42 UTC. Local capture: `/tmp/mlip_active_margin_smoke.log`.

Exact submitted options (local `python` was `/opt/anaconda3/envs/modal-env/bin/python`):

```bash
MLIP_GPU=A100 MLIP_RUN_TIMEOUT_S=600 python -m modal run --detach cloud/modal/app.py::main \
  --molecule naphthalene --model gemnet_t --seed 20261126 --steps margin,eval \
  --margin-overrides='{simulate: {n_replicas: 2, max_ps: 10.0, check_every_steps: 2, end_phase_fail_frac: 0.5}, stage_b: {n_cycles: 1, steps_per_cycle: 4, fixed_budget: true, batch_labeled: 2, batch_pairs: 2, lam: 1.0}, optim: {name: AdamW, lr: 1.0e-5}, results_dir: results/smoke/gemnet_t_naphthalene_active_margin_20261125}' \
  --checkpoint results/smoke/gemnet_t_naphthalene_active_margin_20261125/naphthalene_gemnet_t_seed20261126/ckpt.pt \
  --eval-ps 0.1 --tag smoke_gemnet_t_naphthalene_active_margin_20261125 \
  --eval-overrides='{data: {start_split: tune}, accuracy: {split: tune}, observables: {reference_split: tune}, simulate: {n_replicas: 2, save_every_fs: 5}, nve: null}'
```

Do not rerun into the same output paths; use a fresh tag/root to retain evidence.

### Observed result

- A100-SXM4-40GB execution started at **17:42:29 UTC**; Modal canceled the input at
  **17:52:29 UTC** on the configured **600-second timeout**. App shutdown was confirmed
  at **17:52:31 UTC**. A subsequent explicit stop request returned “App is already stopped.”
- Submission-to-stop was approximately **10 minutes 20 seconds**, excluding local setup.
- The matched checkpoint and all **121 AdamW states at step 119** loaded successfully.
- The job remained inside its first failure-collection rollout. It returned no
  failure-pair batch, reached **zero of the four planned optimizer updates**, and
  did not run evaluation. This is **not an active-margin pass**.
- Read-only volume checks found no training `history.json` or evaluation `metrics.json`.
  The loop currently emits no intra-rollout step progress and saves no partial
  trajectory, so the simulated duration reached is unknown. Do **not** interpret this
  as demonstrated 10 ps survival or as a numerical failure; the only observed error
  was the configured timeout.
- GPU execution cost estimate: **$0.3498** (600 × $0.000583). Allow approximately
  **$0.40–$0.50 total** for CPU/RAM/startup/termination overhead, before credits.
  These are estimates from published prices, not an account billing receipt.
- Local durable evidence: `results/smoke/gemnet_t_naphthalene_active_margin_20261125/`
  contains `run.log`, `summary.json`, `artifact_status.json` and `README.md`.
- No retry, new pretraining, lambda-zero run or full sweep was launched.
- This step's code change was limited to the single-run Modal timeout environment
  setting and disabled retries; the training/model code was unchanged. The capped
  Modal app imported successfully locally, and `git diff --check` passed.
- Next bounded diagnostic should first add MD-step progress and partial trajectory
  saving. It must explicitly distinguish an active-margin plumbing check from a
  natural-failure stability benchmark. No additional GPU attempt is authorized by
  this log alone.

## 2026-11-25 — Progress and partial-trajectory rerun (completed: persistence verified; collection timed out)

New user authorization: add rollout progress and partial saving, then rerun one
capped smoke test. The four-update budget and no-full-sweep constraint remain.

Changes:

- `simulate.progress_every_s: 30` enables Stage B diagnostics. Logging reports MD
  step, requested steps, simulated ps, elapsed seconds, average steps/second,
  failed replicas, frame count, and whether collection finished.
- A cycle's `rollouts/cycle000/trajectory.pt` is atomically replaced at each report.
  It contains CPU positions (Å), explicit frame-step indices, cumulative failed
  flags and progress metadata. Frame zero and pre-reset failure frames are retained.
  `progress.jsonl` retains the event history. These are diagnostic snapshots,
  **not restart checkpoints**: velocities/RNG/integrator state are not included.
- `run_started.json` records resolved config, initial optimizer state summary and
  bond/atomic topology before collection, even if the training phase is never reached.
- The single-run Modal subprocess runner commits the volume every 30 seconds during Stage B,
  and commits again on exit. Thus evidence survives a timeout; the last committed
  snapshot can lag the last completed MD step by roughly one save-plus-commit interval
  (about 60 seconds), or longer if a single MD step/commit stalls.
- A partial write leaves the previous complete `trajectory.pt` intact. Logging is
  opt-in and uses no random sampling, so it does not alter the trajectory/seed.
- Tests cover trajectory equivalence and failure-frame retention, a simulated
  interruption with periodic saving, and interruption during an atomic write.

Rerun protocol: same full GemNet-T / rMD17 naphthalene, same Stage A seed 20261126,
same 500 K, calibrated bond threshold, two replicas, up to 10 ps, stop at first
failed replica. One cycle × four lambda-1 updates; AdamW LR 1e-5; 0.1 ps tune
evaluation if training finishes. One A100 invocation, timeout 600 seconds, no
retries or automatic budget extension. Only the logging/persistence changes differ.

Fresh artifact root: `results/smoke/gemnet_t_naphthalene_progress_20261125/`.
Local stdout capture: `/tmp/mlip_progress_smoke.log`.
[Modal run ap-7xjXd3YijCm5Yb125BZqr3](https://modal.com/apps/rishabh12ranjan/main/ap-7xjXd3YijCm5Yb125BZqr3).
Submitted November 25 at approximately **7:42:50 p.m. EST**. Assigned A100 80 GB.
Local suite: **60 passed in 27.66 seconds**; capped Modal app import succeeded.

Exact command (the local interpreter was `/opt/anaconda3/envs/modal-env/bin/python`):

```bash
MLIP_GPU=A100 MLIP_RUN_TIMEOUT_S=600 python -m modal run --detach cloud/modal/app.py::main \
  --molecule naphthalene --model gemnet_t --seed 20261126 --steps margin,eval \
  --margin-overrides='{simulate: {n_replicas: 2, max_ps: 10.0, check_every_steps: 2, end_phase_fail_frac: 0.5, progress_every_s: 30}, stage_b: {n_cycles: 1, steps_per_cycle: 4, fixed_budget: true, batch_labeled: 2, batch_pairs: 2, lam: 1.0}, optim: {name: AdamW, lr: 1.0e-5}, results_dir: results/smoke/gemnet_t_naphthalene_progress_20261125}' \
  --checkpoint results/smoke/gemnet_t_naphthalene_progress_20261125/naphthalene_gemnet_t_seed20261126/ckpt.pt \
  --eval-ps 0.1 --tag smoke_gemnet_t_naphthalene_progress_20261125 \
  --eval-overrides='{data: {start_split: tune}, accuracy: {split: tune}, observables: {reference_split: tune}, simulate: {n_replicas: 2, save_every_fs: 5}, nve: null}'
```

Use fresh output names for future attempts.

Live verification (while the GPU job was still running): downloaded and loaded the
committed snapshot at **step 1,610 / 0.805 ps / 60.10 seconds**. It contained 806
frames × 2 replicas × 18 atoms × 3 coordinates, all finite, with correct frame-step
indices. Maximum bond deviations were **0.11477 Å** and **0.14180 Å**, below the
**0.21618 Å** threshold; re-extraction found no failure pairs. This verifies periodic
volume persistence before process completion, not merely local file creation.
Measured throughput was ~26.8 steps/s; a full 20,000-step / 10 ps collection would
take ~12.4 minutes at that rate, exceeding the existing 10-minute execution cap.

### Final result

- **Logging and partial persistence verified. Active margin still unverified.**
- The 600-second timeout canceled collection at **7:53:08 p.m. EST** on November 25,
  following a **7:43:08 p.m. EST** start. App shutdown was confirmed at **7:53:09 p.m.**;
  an explicit stop request returned “App is already stopped.” No retry was started.
- Downloaded the final committed snapshot **after shutdown**: step **15,142**, saved
  duration **7.571 ps**, **7,572 frames × 2 replicas × 18 atoms × 3 coordinates**.
  It matches the last of **20 progress reports** and contains only finite coordinates.
  Its first 806 frames exactly match the earlier live download.
- Independent CPU bond-deviation calculation and failure-pair extraction found
  **zero failures**. Largest deviations were **0.161408 Å** and **0.165382 Å**, below
  **0.216182 Å**. The requested 10 ps was not completed; only the saved interval is established.
- Initial optimizer restoration remained correct: **121 states at step 119**.
  Collection never returned; **zero of four planned updates** were reached, and
  evaluation did not start. No final history or evaluation metrics were present.
- Measured throughput at the last snapshot: **26.518 steps/second**. A full 10 ps /
  20,000-step collection at that rate requires approximately **12.57 minutes**,
  plus setup/training/evaluation. The observed issue was an insufficient execution
  budget for that horizon, not an observed stall. Longer collection would not
  guarantee a failure or validate a stability improvement.
- The last snapshot was saved at about 571 seconds into collection. The 600-second
  function budget also includes setup/validation; the exact unsaved final step is
  unknown. Steps after the last snapshot are not claimed or reconstructed.
- Estimated GPU execution cost: **$0.4164** on A100 80 GB, plus CPU/RAM/startup overhead,
  before credits. This is an estimate from the pricing reference above, not an invoice.
- Local evidence: `results/smoke/gemnet_t_naphthalene_progress_20261125/` contains
  final and live snapshots, metadata, progress JSONL, verification JSON, logs and
  `bond_deviation.png`. A detailed local `README.md` explains interpretation and paths.
- Remote volume: `mlipstab`, under the artifact root above plus
  `naphthalene_gemnet_t_seed20261126/`; snapshots are in `rollouts/cycle000/`.
- **60 tests passed**. The capped Modal app import and `git diff --check` passed.
  No full sweep, additional pretraining, new lambda-zero job, or budget extension ran.

## 2026-11-25 — Literature check: GemNet-T / naphthalene

Checked the user's `~/Downloads/forces_are_not_enough.pdf` against the
[arXiv v2 paper](https://arxiv.org/pdf/2210.07237v2). Table 3 (printed page 8)
reports naphthalene GemNet-T force MAE **1.5 meV/Å** and stability **8 ± 2 ps**
(mean ± standard deviation over five simulations). GemNet-dT is a separate
variant, with **25 ± 10 ps** stability and **1.9 meV/Å** force MAE.

Protocol in Section 4 / Table 1 and the failure definition in Section 5:

- Original MD17; 9,500 training, 500 validation, 10,000 force-test configurations.
- Five randomly sampled test starting configurations; 500 K, Nosé–Hoover,
  0.5 fs timestep, no periodic boundaries for MD17.
- Planned horizon **300 ps / 600,000 steps per trajectory**, with stability
  measured until failure or the horizon. Eight ps is observed time to failure,
  not the planned simulation length or GPU wall-clock runtime.
- Failure: any bond deviates by more than **0.5 Å** from its mean equilibrium length.
- Appendix B / Table 7: GemNet-T/dT MD17 batch size 1, initial learning rate 0.001.
  Training stops at the first of 2,000 epochs, seven days on V100, or learning rate
  below 1e-6; these are limits, not an exact naphthalene checkpoint training duration.

Our smoke is not a replication: rMD17 with 950/50 train/validation structures,
one Stage A epoch, two training-set starts, Langevin dynamics, and an automatic
training-collection threshold of 0.216182 Å. Temperature and timestep match.
The 7.571 ps retained interval without failures cannot establish improved stability
or contradict the paper. No failure pairs also means active-margin training remains
unverified. The separate evaluation threshold is 0.5 Å; that evaluation did not run.

Version caveat: the [authors' repository](https://github.com/kyonofx/MDsim)
announces an October 29, 2024 code/paper correction linked to
[periodic-coordinate wrapping](https://github.com/facebookresearch/fairchem/pull/783).
The current [TMLR PDF](https://openreview.net/pdf?id=A8pqQipwkt) was inaccessible
to this check (browser verification / public API HTTP 403). Consequently the
numbers above are explicitly verified for the downloaded / arXiv v2 version;
we have not confirmed whether the updated table differs. Do not silently treat
them as verified against the corrected manuscript. MD17 is nonperiodic, but that
alone is not proof that its reported results were unchanged.

Documentation only: no new GPU run or full sweep launched during this check.

## Before a full campaign

- Logging and persistence are now verified. Choose a separately bounded failure-collection
  diagnostic before claiming active-margin verification; account for the measured
  throughput and the lack of failures in these two 500 K trajectories.
- Smoke results cannot select lambda or establish a force/stability improvement.
- Independent adequate Stage A training for the campaign seeds, aspirin validation,
  and production-sized rollout/batch/resource checks remain outstanding.
- Obtain explicit authorization before starting the full sweep.

## 2026-11-25 — Authorized 20 ps failure-collection smoke

User requested GemNet-T / naphthalene for up to 20 ps or the first failure pairs,
then tiny margin updates, a runtime cap, and ongoing monitoring; no full sweep.
Plan: reuse matched Stage A model/AdamW checkpoint, seed 20261126; two replicas,
500 K Langevin, 0.5 fs timestep, existing automatic threshold. Start fresh because
saved position snapshots lack velocity/RNG state. Stop at the first failed replica
(0.5 of two), or 40,000 MD steps / 20 ps. One cycle, at most four updates with
lambda 1, LR 1e-5, labeled/pair batches 2. `fixed_budget: false` skips updates if
no failure pairs; `stab_loss_tol: 0` permits all four updates when pairs exist,
subject to the existing accuracy guardrail. Follow with 0.1 ps tune evaluation.
Hard function timeout: 1,800 seconds, A100, retries zero. Previous throughput
suggests ~25 minutes for collection; runtime includes loading and subsequent work.
Exact invocation: `results/smoke/gemnet_t_naphthalene_20ps_20261125/launch.sh`.
Progress every 30 seconds; partial trajectories committed periodically. Verify
failure pairs independently from saved coordinates, nonzero finite margin loss
if present, and restored optimizer step progression. No failure is guaranteed:
the paper's 8 ± 2 ps is an average under a different training/simulation protocol.

Launch confirmed: https://modal.com/apps/rishabh12ranjan/main/ap-SSODGspmlyG82BVDbHHj6o
Actual allocation is **A100-SXM4-40GB** (the generic `A100` request does not fix
memory size), unlike the preceding 80 GB allocation. Initial optimizer restoration:
121 states, all at step 119, LR 1e-5. First minute: 0.671 ps, zero failures,
22.31 steps/s. At this initial rate the 20 ps horizon alone takes ~29.9 minutes;
the 30-minute function cap may stop it earlier once setup is included. The cap
will not be extended automatically.

Research interpretation discussed with user while the run executes: SchNet is a
potential cheaper pipeline diagnostic, but is not an automatic remedy for a lack
of failures. The downloaded paper reports SchNet/naphthalene 18 ± 2 ps versus
GemNet-T/naphthalene 8 ± 2 ps. Current Stage A is only one epoch / 119 updates,
so scientific model selection should follow a properly trained baseline, force-MAE
verification and repeated rollouts under a consistent evaluation protocol.
No model switch or additional job was authorized or launched in this discussion.

### Reproducible command for the active 20 ps smoke

The local `results/` directory is ignored by Git; the exact command is included
here for the teammate handoff. It launches one capped diagnostic, not a sweep.

```bash
MLIP_GPU=A100 MLIP_RUN_TIMEOUT_S=1800 /opt/anaconda3/envs/modal-env/bin/python -m modal run --detach cloud/modal/app.py::main \
  --molecule naphthalene --model gemnet_t --seed 20261126 --steps margin,eval \
  --margin-overrides='{simulate: {n_replicas: 2, max_ps: 20.0, check_every_steps: 2, end_phase_fail_frac: 0.5, progress_every_s: 30}, stage_b: {n_cycles: 1, steps_per_cycle: 4, fixed_budget: false, stab_loss_tol: 0.0, batch_labeled: 2, batch_pairs: 2, lam: 1.0}, optim: {name: AdamW, lr: 1.0e-5}, results_dir: results/smoke/gemnet_t_naphthalene_20ps_20261125}' \
  --checkpoint results/smoke/gemnet_t_naphthalene_20ps_20261125/naphthalene_gemnet_t_seed20261126/ckpt.pt \
  --eval-ps 0.1 --tag smoke_gemnet_t_naphthalene_20ps_20261125 \
  --eval-overrides='{data: {start_split: tune}, accuracy: {split: tune}, observables: {reference_split: tune}, simulate: {n_replicas: 2, save_every_fs: 5}, nve: null}'
```

### Final 20 ps diagnostic result — verified after shutdown

- **No failure pairs; active margin remains unverified.** The 1,800-second function
  timeout canceled collection at **20:55:49 EST**. The final committed snapshot is
  step **38,310 / 19.155 ps per replica**, not the full requested 20 ps. It contains
  **19,156 frames × 2 replicas × 18 atoms × 3 coordinates**. Unsaved steps are unknown.
- Independent CPU checks found all coordinates finite and **zero failure pairs**.
  Maximum bond deviations: **0.162823 Å** and **0.165451 Å**, below the existing
  **0.216182 Å** collection threshold; no events at the paper's 0.5 Å threshold either.
- **60 progress records**; the 5.573 ps live snapshot exactly matches the corresponding
  prefix of the final trajectory. Progress was steady at ~21.594 steps/s; this was
  not a stalled run. Twenty ps alone would take ~30.87 minutes at that rate, before
  setup or training. The slower 40 GB allocation explains why the hard cap won.
- **0/4 optimizer updates** were reached. No final `history.json` or `ckpt.pt` was
  present under the run directory; evaluation never started. Initial optimizer
  restoration was correct: 121 AdamW states at step 119, LR 1e-5.
- The function timeout left an app container briefly present. Explicitly stopped
  only this app and verified **state stopped, zero tasks, 20:57:03 EST**. The execution
  cap does not remove container shutdown overhead. No retry, new job, model switch,
  additional pretraining or full sweep was launched.
- GPU execution estimate: **$1.0494** for 1,800 seconds at the previously checked
  A100 40 GB rate of $0.000583/s. Excludes CPU/RAM, startup/shutdown overhead and
  credits; not an invoice.
- Local evidence: `results/smoke/gemnet_t_naphthalene_20ps_20261125/` contains
  final/live trajectories, metadata, progress, verification JSON, remote-file listing,
  logs, exact launch command, read-only CPU verification script, README and bond plot.
  Binaries stay outside Git; essential findings and the command are recorded here.
- Re-ran the local suite before sharing the code: **60 tests passed in 26.93 s**.
  No application code was changed during this diagnostic.

### Research direction discussed after the negative result

The user needs a class-project demonstration and asked about deliberately eliciting
failures. A transparent, controlled stress test is a possible next experiment;
no such test has been launched or claimed successful.

1. Controlled bond distortions can check that the margin loss raises distorted
   structures' energies relative to valid anchors. This is a mechanism/gradient
   check, not evidence of better MD trajectories.
2. A higher-temperature or other predeclared challenge can test actual stability
   improvement. Pilot the challenge using the baseline, freeze the settings before
   comparing methods, and evaluate on separate trajectories/starts/seeds. Compare
   with lambda zero under equal update budgets; report force MAE and failures,
   including ordinary-condition evaluation. Keep the criterion identical across arms.
3. A paper reproduction requires a comparably trained baseline and matching data
   and simulation protocol. The present one-epoch rMD17/Langevin smoke is not that.

A lack of baseline failures limits measurable improvement over the chosen horizon.
It neither validates nor disproves the proposed loss. Do not infer improved accuracy
or dynamics merely from intact bonds, or count synthetic distortion checks as natural
rollout failures. Establish baseline suitability before another expensive failure
search; the existing sweep configs remain provisional and unlaunched.

## 2026-11-25 — Controlled bond-distortion diagnostic (authorized)

User requested a cheap, capped mechanism check before any stress rollout or sweep.
New `scripts/check_bond_distortion.py` uses the existing matched GemNet-T/naphthalene
Stage A checkpoint (seed 20261126). First C-H bond; first two training and first two
validation frames; hydrogen moved along the bond to equilibrium length ±0.30 Å.
Four fixed synthetic pairs per split; validate that anchors are below the existing
0.216182 Å threshold and distorted structures above it. No MD failure is claimed.

Predeclared checks:

- Record default 10 kT loss and energy gaps first.
- Diagnostic margin is max(10 kT, largest initial training-pair gap + 2 kT), frozen
  before updates. This deliberately tests the active-loss regime if the production
  margin is already satisfied; it is not a selected campaign hyperparameter.
- Compute parameter gradients and compare ±1e-3 normalized-gradient perturbations.
  Descent should lower the loss and raise the distorted-versus-anchor energy gap.
  Restore the exact initial weights after these temporary probes.
- Two arms from identical weights and matched AdamW state: lambda 0 and lambda 1,
  four updates each, LR 1e-5, same labeled batches, gradient clipping 1.0. Record
  full loss, margin loss, gradients, energy gaps and validation force MAE. Separate
  validation distortions are diagnostic holdout structures, not a final test set.
- Verify optimizer steps 119→123 in both arms and unchanged frozen parameters.
  Report any absence of gap improvement instead of tuning the test after seeing it.
- Never overwrite production checkpoints. Eight optimizer updates total, no MD,
  no new pretraining, no lambda sweep. CPU tests: 17 passed (2.80 seconds).
- Modal `bond_distortion_check` is a single A100 function with **300-second timeout,
  zero retries** and periodic volume commits. Generic A100 allocation may be 40/80GB.
  Execution-cost ceiling at previously checked rates: ~$0.175–$0.208 GPU only,
  excluding startup/shutdown, CPU/RAM and credits; this is not an invoice.

Exact launch (local interpreter may differ):

```bash
python -m modal run --timestamps cloud/modal/app.py::bond_distortion_check
```

Artifacts: `results/smoke/gemnet_t_naphthalene_distortion_20261125/`.

### Distortion diagnostic result — mechanism passed, efficacy not established

App: https://modal.com/apps/rishabh12ranjan/main/ap-53tDEev7KkuA0roqwmfChZ
A100-SXM4-80GB; app created **21:02:46 EST**, script completed in **9.20 seconds**,
app stopped **21:03:14 EST**, zero remaining tasks. Five-minute cap was not approached.
Rough GPU component is on the order of one to two cents at the previously checked
rate, excluding CPU/RAM and overhead; no billing receipt was consulted.

- Fixed C0–H10 target lengths: **1.400652 Å / 0.800652 Å** (equilibrium ±0.30 Å),
  four train pairs and four validation pairs. Only H moved; anchors and other atoms
  were preserved. These are synthetic violations of our collection threshold,
  not naturally occurring failures or the paper's 0.5 Å instability events.
- Baseline mean training-pair energy gap: **1.271212 eV**. Normal 10 kT margin:
  **0.430867 eV**. Normal margin loss: **2.34e-8 eV**, already essentially saturated.
  A geometric threshold violation does not necessarily have too little energy
  relative to its anchor; this loss penalizes the latter.
- Explicit diagnostic margin: **1.611350 eV**, determined from initial training
  gaps by the predeclared rule. This is a mechanism-test setting, not a production
  recommendation or result from a lambda sweep.
- Finite nonzero parameter-gradient norm: **4.08343**. A temporary normalized
  descent perturbation reduced diagnostic loss **0.342583 → 0.338516 eV** and
  increased mean gap **1.271212 → 1.275649 eV**. Opposite-direction perturbation
  increased loss to **0.346682 eV**. Weights were reset before both training arms.
- Lambda-zero and lambda-one arms each restored **121 AdamW states at step 119**,
  performed four updates and ended at **123**. Frozen parameters stayed unchanged.
  All losses/gradient norms were finite; no campaign checkpoint was overwritten.
- Combined training did **not** raise the mean gap above baseline: control final
  **1.267482 eV**, margin final **1.267484 eV**. Margin-minus-control difference:
  **2.03e-6 eV** (train) and **8.34e-7 eV** (validation). These tiny differences are
  near numerical variability and are not meaningful improvement evidence.
- Validation force MAE: baseline **0.370468 eV/Å**, control **0.376813203 eV/Å**,
  margin **0.376813230 eV/Å**. The two arms are effectively identical at this scale;
  both worsened slightly after these few supervised updates. Labeled losses were
  ~714–853 while the weighted diagnostic margin term was ~0.34, consistent with
  a weak margin contribution at lambda 1. Loss values alone are not gradient ratios.
- Independently downloaded report and pairs; CPU recomputation verified geometry,
  softplus loss values, update counts and reported optimizer continuation.
- Corrected the derivative sign in the margin-loss **docstring**: gradient is
  sigmoid × (dU_anchor − dU_distorted). The implemented loss was already correct.
- No MD, new pretraining, production checkpoint replacement, or sweep ran.

Interpretation: the margin gradient has the intended direction on real GemNet-T
parameters and the restored optimizer executes with active synthetic pairs. This
is a functional diagnostic, not proof of improved dynamics or adequate weighting
against the supervised objective. Natural-failure Stage B still needs validation.

### Concrete paper-reproduction route and unresolved details

Primary sources checked on November 25:

- [Authors' repository](https://github.com/kyonofx/MDsim), including recursive file
  inventory and issue listings. The distributed example checkpoint is SchNet/water;
  no GemNet-T/naphthalene checkpoint was found in those published listings.
- [Zenodo dataset](https://zenodo.org/records/7196767) describes preprocessed data;
  its three listed files are `mdsim_data.tar.gz` (7.28 GB), `alanine_dipeptide.npy`,
  and `water.npy`. We inspected metadata only, not the large archive's contents.
- [Original preprocessing](https://github.com/kyonofx/MDsim/blob/main/preprocessing/md17.py):
  MD17, 9,500 training / 500 validation / 10,000 test; split random state 123.
- [Naphthalene simulation config](https://github.com/kyonofx/MDsim/blob/main/configs/simulate/naphthalene.yml):
  test starts, default seed 123, 500 K, NoseHoover, timestep 0.5 fs, `ttime: 20`,
  600,000 steps, save every 100 steps. A five-trajectory benchmark needs separate
  starts/seeds; the config's single default seed is not the full five-run specification.
- [GemNet-T config](https://github.com/kyonofx/MDsim/blob/main/configs/md17/gemnet-T.yml):
  batch 1, initial LR .001, AdamW, force coefficient 1000, energy coefficient 1,
  EMA .999. Source currently says `max_epochs: 10000`; the downloaded paper's
  Table 7 says 2,000 for MD17. Resolve/version this discrepancy before claiming
  exact replication. Current backend prefers matched raw weights; paper-style
  evaluation may instead use the upstream EMA export. Preserve the distinction.

A sensible reproduction uses the authors' released data/split, model and training
protocol (or their exact checkpoint if obtained), verifies force MAE, then runs
five independent test trajectories with the published thermostat and failure
criterion. Aim to reproduce a distribution of outcomes, not guarantee failure at
exactly 8 ps. The paper's 2024 correction remains an additional version-check item
as documented above. No reproduction training or large download was started.

We cannot yet identify which checkpoint/data/integrator difference caused the
absence of failures in our smoke. Architecture alone does not determine the
learned energy surface or failure time. The smoke checkpoint's validation force
MAE is ~370 meV/Å; the downloaded paper reports 1.5 meV/Å for its GemNet-T naphthalene
model on a different MD17 test set. These are not matched evaluations, but they
underscore that we have not recreated the paper's trained baseline.
