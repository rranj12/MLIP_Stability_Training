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

## Anchor: no stop-gradient (changed from the first draft)

The first draft detached U(x₀). That makes the penalty gradient σ(m − Δ)·∂U(x_fail)/∂θ
alone, which includes every direction that raises *all* energies. Every architecture
we use has one: the readout bias / per-element energy shift (SchNet atomref, NequIP
per-species shift, MACE E0s). The forward gap Δ is unaffected by that offset, so the
push never satisfies the loss and never stops. Only the weak energy term of L_EF
(λ_E:λ_F ≈ 1:100–1:1000) resists it, so expect a systematic upward energy bias,
the same symptom CRMSE reports on aspirin.

Default now: gradients flow through both energies (`stop_grad_anchor: false`), so
the gradient is σ(m − Δ)·[∂U(x_fail)/∂θ − ∂U(x₀)/∂θ] and common offsets cancel.
Both snapshots are still detached from the rollout. The worry that motivated the
stop-gradient, the model lowering U(x₀), is small: x₀ sits near the training data,
where L_EF pins forces with 3N labels per frame. `tests/test_losses.py` checks that
an offset parameter gets exactly zero gradient by default and a nonzero one under
the ablation. Keep `stop_grad_anchor: true` as an ablation; it should show the bias.

## Anchor choice and forces (`stage_b.anchor`, `stage_b.restore_weight`)

- **Energy ordering of rollout anchors is uncertain.** x₀ is 250 fs earlier and also
  differs in torsions, angles and other bonds. Aspirin's thermal energy fluctuation at
  500 K is ≈ √(N_df/2)·k_BT ≈ 5 kcal/mol, about half the 10 k_BT margin.
  - `anchor: counterfactual` uses x_fail with only its most-deviated bond moved back to
    b0, so the pair differs in one coordinate whose ordering is known.
  - `anchor: labeled` uses the nearest training frame (distance-matrix L2), whose
    energy and forces L_EF holds in place, so the margin can't be met by lowering it.
  - Localized energies (`localize.enabled`) also remove most unrelated thermal noise.
- **Energy values at two points aren't a restoring force.** MD follows −∇U, and a
  narrow spike at x_fail would satisfy the margin without redirecting trajectories.
  `restore_weight > 0` adds a hinge requiring ∂U/∂r_ij > 0 along a stretched flagged
  bond (< 0 if compressed). It needs double backprop on buffer frames only.
  `extract_failure_pairs` already takes the *first* threshold crossing rather than the
  collapsed end state. The energy scan along a stretched C–H bond (planned figure)
  is the diagnostic for wall vs. spike.

## CRMSE reimplementation notes

Paper settings: λ = 0.015, 200 epochs, batch 16, buffer 1,000 seeded from a failed
MALA chain of the pretrained model, K = 200 ULA steps/epoch, Adam lr 1e-4 continuing
the pretraining optimizer state. Its step sizes are in reduced units with mass-based
per-atom preconditioning (App. B); re-tune for eV/Å. Reported cost: aspirin energy
MAE 0.06 → 0.33 kcal/mol (forces unchanged); ethanol 0.01 → 0.03 / 0.07 → 0.13.
