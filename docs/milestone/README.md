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
