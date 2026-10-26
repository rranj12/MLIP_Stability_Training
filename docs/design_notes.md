# Design notes

Facts checked against the StABlE code (pinned at `54d6808`) and the CRMSE paper
(arXiv:2606.31660), plus decisions they drive in this repo.

## Losses side by side

All three methods update `θ ← θ + η Σᵢ wᵢ ∂U(xᵢ)/∂θ` (wᵢ > 0 raises a state's energy):

| Method | States | Weight wᵢ |
|---|---|---|
| StABlE | all snapshots of fresh 1 ps rollouts | ∝ (⟨g⟩ − g_ref)·(gᵢ − ⟨g⟩), from the Boltzmann covariance estimator |
| CRMSE | model samples (+) and training frames (−) | +β/M, −β/N (maximum likelihood) |
| Ours | flagged failures only | +σ(m − Δᵢ)/\|B\|, switching off once the margin is met |

Ours is a margin loss in the sense of LeCun et al. (2006); CRMSE is the negative
log-likelihood loss. Expected trade-off: similar stability, smaller accuracy cost,
no improvement in in-basin thermodynamics (dihedral free energies).

## StABlE facts that affect the comparison

- **Failures are a trigger, not training data.** Learning starts once ≥ 75% of
  replicas are unstable (`max_frac_unstable_threshold`); then every replica is reset
  to a DFT frame and the gradient uses fresh 1 ps rollouts (`set_starting_states`).
- **Reference h(r) uses 10× more frames.** `MAX_SIZES = {"md17": "10k"}`: g_ref is
  computed from 9,500 frames while the model trains on 950.
  `prepare_rmd17.py --reference same` gives StABlE only our 950 frames (main baseline);
  `--reference published` reproduces the extra-data setting (secondary baseline).
- **Original MD17, not rMD17.** StABlE downloads `md17_aspirin.npz` from
  quantum-machine.org (noisier labels). Its released MD17 checkpoints are therefore
  *not* usable as our rMD17 baseline; we retrain on rMD17. MD22 is the same dataset in
  both, so StABlE's released Ac-Ala3-NHMe checkpoints are directly reusable.
- **Units are eV.** `data/md17.py` divides kcal/mol labels by 23.06. Margins are
  configured in k_BT and converted with `mlipstab.units.kT` (0.0431 eV at 500 K).
- **Thermostats.** Training uses Nosé–Hoover; evaluation (`simulate.yml`) uses
  Langevin, 500 K, 0.5 Å threshold, 256 replicas. Our eval config matches it.
- **Optimizer.** SGD lr 1e-3, grad clip 1.0, ReduceLROnPlateau ×0.2 keyed to the
  unstable fraction. SGD fits its hand-built gradient written into `param.grad`.
- **License.** StABlE has no LICENSE file, so this repo uses it as a pinned
  submodule rather than copying its code. MDsim (which it builds on) is MIT.

## Detection threshold finding (run on real rMD17 ethanol, official split 1)

The largest bond deviation among the 1,000 **DFT test frames** is **0.226 Å**.
So the 0.25 Å training threshold sits just above normal (hot) DFT vibrations, and a
0.15 Å threshold would flag real DFT geometries as failures. (MD17's effective
temperature is ~650 K for ethanol, so 500 K MD vibrates less, but the margin is thin.)

Implications:
- Drop 0.15 Å from the sweep or calibrate per molecule: e.g. threshold =
  max DFT deviation + 0.05 Å, computed on training frames.
- The margin and threshold are coupled: a 0.25 Å C–H stretch costs ≈ 22 kcal/mol
  in DFT (k ≈ 700 kcal/mol/Å²), so m = 10 k_BT (≈ 10 kcal/mol) is below the true cost.
  At smaller thresholds a 10–20 k_BT margin would exceed it and bias the surface.
- The non-bonded collision floors had 0 false positives on the same test frames.

## Anchor stop-gradient

`margin_loss(..., stop_grad_anchor=True)` is the default. Without it the model can
meet the margin by lowering U(x₀), an unlabeled state, which digs a new spurious
minimum next to the basin. `stop_grad_anchor: false` is an ablation.

## CRMSE reimplementation notes

Paper settings: λ = 0.015, 200 epochs, batch 16, buffer 1,000 seeded from a failed
MALA chain of the pretrained model, K = 200 ULA steps/epoch, Adam lr 1e-4 continuing
the pretraining optimizer state. Its step sizes are in reduced units with mass-based
per-atom preconditioning (App. B); re-tune for eV/Å. Reported cost: aspirin energy
MAE 0.06 → 0.33 kcal/mol (forces unchanged); ethanol 0.01 → 0.03 / 0.07 → 0.13.
