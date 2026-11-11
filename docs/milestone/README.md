# Milestone: first runs (SchNet, ethanol)

Numbers and figures: [results.md](results.md). Regenerate with

```bash
python scripts/analysis/milestone.py fetch   # needs the modal CLI and access to the mlipstab Volume
python scripts/analysis/milestone.py plot    # numpy + matplotlib only
```

Everything here is one molecule, one architecture, one seed, 100 ps. Treat it as preliminary.

## Setup

- **Data:** rMD17 ethanol, official split 1: 950 train / 50 validation / 1,000 test frames.
- **Model:** SchNet with StABlE's architecture and optimizer settings (64 channels, 6 interactions,
  cutoff 5 Å, Adam 1e-3, batch 100). 117,953 parameters.
- **Stage A (standard loss):** energy + force MSE. Two learning-rate schedules, see below.
- **Stage B (margin loss):** Stage A loss + λ · softplus(m − [U(x_fail) − U(x_anchor)]), m = 10 k_BT.
  Failures come from 128-replica, 20 ps Langevin rollouts at 500 K (bond deviation > 0.25 Å);
  up to 3 cycles of 500 steps, Adam 1e-5, 100 labeled frames per step.
- **Evaluation:** 256 Langevin replicas from test frames, 500 K, 0.5 fs, 100 ps. A replica fails when a
  bond deviates > 0.5 Å from its mean training length, or two non-bonded atoms collide. Every failure
  so far was a bond failure, so the combined and bond-only (Fu et al.) criteria give identical numbers.
- **Compute:** Modal, one A10G per job. Stage A: 6 min (short schedule), 48 min (long). Evaluation: 30 min.

## Headline result

On the short-schedule model (the one with real instability), Stage B with the margin loss cut failures
within 100 ps from 66/256 to 41/256 (λ = 0.0005) and to 6/256 (λ = 0.005), with test force MAE unchanged
(53.3 → 52.0 and 53.0 meV/Å). Two-proportion z-test against the standard-loss model: p = 0.007 and
p < 1e-10 (against the control below: p = 0.11 and p < 1e-10). On the long-schedule model, which fails 8/256 to begin with, the margin loss changed nothing
(8 and 9 failures) and λ = 0.005 cost 11% in force MAE.

**Control.** Stage B also adds 1,500 ordinary gradient steps to an underfit model, so we ran the same
Stage B with the margin term off (λ = 0). It leaves 55/256 failures: not distinguishable from the
standard-loss model's 66 (p = 0.25), and far above the 6 with λ = 0.005 (p < 1e-10). In its training
rollouts the failing fraction did not fall (5.5%, 10.2%, 11.7% per cycle; with λ = 0.005: 8.6%, 2.3%,
2.3%). So the improvement at λ = 0.005 comes from the margin term, not from the extra training. The
smaller weight is weaker evidence: 41 failures against the control's 55 is p = 0.11.

Single seed, and all runs share the same 256 starting frames.

## Why the converged model shows no difference (low signal, not a null result)

On the long-schedule model the standard and margin-loss runs are indistinguishable (8, 8 and 9 failures
out of 256). We read this as the test being too easy to separate them, not as evidence about the method:

- **100 ps instead of the planned 300 ps.** Evaluations were shortened for time. With ~3% of replicas
  failing by 100 ps, a method can change the count by only a few replicas, which is inside the noise of
  256 replicas. Failures accumulate with time, so 300 ps (or 1 ns) leaves more room for a difference.
- **Ethanol is an easy molecule.** It has 9 atoms, and in Fu et al. (2022) SchNet stays stable on ethanol
  for 247 ps on average at 500 K, against 26 ps on aspirin and 18 ps on naphthalene. A converged SchNet
  on ethanol has little instability left to remove.
- **Few failures to train on.** Stage B found 0 to 3 failing replicas per 20 ps rollout on this model, so
  the margin term saw only a handful of examples.

Next: aspirin and naphthalene with the long Stage A schedule, evaluated for 300 ps, where a converged
SchNet should still fail often enough to compare methods.

## What did not go as assumed

1. **StABlE's Stage A schedule does not converge on the 950-frame split.** ReduceLROnPlateau with
   patience 5 epochs is 50 gradient steps here (10 steps per epoch, 50 noisy validation frames). The LR
   reached its 1e-6 stopping value after 422 epochs at 54 meV/Å. With patience 50 (the same number of
   steps per patience window as on the 9,500-frame split) training ran 3,502 epochs to 19 meV/Å.
   StABlE's released SchNet checkpoint was trained with patience 5 on its 1k split.
2. **Most of ethanol's instability was undertraining.** Failures within 100 ps: 25.8% (short schedule)
   vs 3.1% (long schedule). A converged SchNet leaves little to fix on ethanol at 100 ps, so the next
   molecules should be ones where SchNet is unstable (aspirin, naphthalene in Fu et al.).
3. **The planned margin weight λ = 0.1 destroys force accuracy.** On a trained model the standard loss
   is ~0.005 and the margin term starts near 0.9, so at λ = 0.1 it is ~20× the standard loss. Validation
   force MAE after one cycle, short-schedule model (baseline 53.5 meV/Å; the run stops and restores
   the previous weights above +10%):

   | Change from the planned settings (λ = 0.1, lr 1e-4, 8 labeled frames, 10 ps rollouts) | Force MAE change |
   |---|---|
   | none | +52% |
   | lr 1e-5 | +37% |
   | lr 1e-5, stop the cycle once the margin is met | +34% |
   | lr 1e-6, 100 labeled frames, stop once the margin is met | +17% to +31% |
   | failure threshold 0.5 Å instead of 0.25 Å | +21% |
   | **λ = 0 (control: same fine-tuning, no margin term)** | **+2%** |

   On the long-schedule model (baseline 18.7 meV/Å): λ = 0.005 gave +9% to +11%, λ = 0.0005 gave 0%.
4. **Bugs found only on the first real runs** (all fixed on this branch): rMD17 energies have shape (N,),
   MD17's (N, 1); StABlE's `--molecule` flag does not reach its dataset config; MDsim models predict
   E − mean(E_train), which the labeled loss and the reported energy MAE did not account for.

## Caveats

- Failure counts in the 20 ps Stage B rollouts are noisy: the same unmodified model gave 2% to 10%
  across repeated 128-replica rollouts.
- No StABlE or CRMSE comparison yet, and no seeds.
- Stage B started from a fresh Adam state; the Stage A checkpoint does not store the optimizer.
