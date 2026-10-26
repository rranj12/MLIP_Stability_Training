# Compute plan

## What is measured vs. estimated

Measured by StABlE on one A100 (paper, Supplementary Table 3):

| System | Model | QM pretraining | StABlE training |
|---|---|---|---|
| Aspirin (MD17) | SchNet | 2 h | 4 h |
| Ac-Ala3-NHMe (MD22) | NequIP | 16 h | 4.7 h |
| Water | GemNet-T | 31 h | 3.5 h |

**Not reported anywhere:** the cost of evaluation (256 replicas × 300 ps – 1 ns of
MD). That is likely the largest line item, so the first pilot task is to measure
MD throughput (steps/s for 256 batched replicas) for SchNet and NequIP on the GPUs
we actually have, then update the estimates below.

## Estimates (A100-hours; assume ~2× longer on an A10G/L4)

Core matrix: {ethanol, aspirin} × {SchNet, NequIP} × {baseline, StABlE, CRMSE, ours}
× 3 seeds. One Stage A checkpoint per (molecule, model), shared by all methods;
seeds vary Stage B and evaluation.

| Item | Runs | Per run | Total |
|---|---|---|---|
| Stage A pretraining | 4 | SchNet ~2 h, NequIP ~6–12 h | ~15–30 h |
| Stage B (3 methods) | 36 | SchNet ~3–4 h, NequIP ~5 h | ~120–160 h |
| Evaluation, 300 ps | 48 | SchNet ~1–3 h, NequIP ~3–8 h | ~100–260 h |
| Sweeps + ablations (SchNet aspirin, 1 seed, 100 ps eval) | ~15 | ~3 h | ~45 h |
| **Core total** | | | **~280–500 A100-h** |

Cheap savings: stop an evaluation once every replica has failed (baselines fail in
tens of ps); pack 2–3 SchNet jobs per GPU (they use little memory); run sweeps at
100 ps and only headline configurations at 1 ns.

## Budget

- AWS on-demand (us-east-1, approximate; check the console): g5.xlarge (A10G 24 GB)
  ≈ $1.01/h, g6.xlarge (L4 24 GB) ≈ $0.80/h, g6e.xlarge (L40S 48 GB) ≈ $1.86/h.
  Spot is usually far cheaper but can be interrupted, so checkpoint often.
- A few hundred dollars ≈ 200–300 on-demand A10G hours, or more on spot. That covers
  roughly the SchNet half of the core matrix. Put the long NequIP / MACE runs on the
  rented GPU and the many short SchNet jobs on AWS spot.
- **Request AWS GPU quota now.** New accounts often have 0 vCPUs for G instances.
  Request "Running On-Demand G and VT instances" and "All G and VT Spot Instance
  Requests" (≥ 8 vCPUs). Approval can take days.

## If there is room to be more ambitious (in order of value per GPU-hour)

1. **Ac-Ala3-NHMe (MD22) with NequIP.** StABlE's own system; MD22 data is identical,
   so its released pretrained and StABlE-trained checkpoints can be reused directly
   (no 16 h pretraining). Gives a head-to-head against published numbers. ~100–150 h.
2. **MACE as a third architecture** on ethanol + aspirin: the current state of the
   art, which the explainer already lists. ~80–150 h.
3. **Temperature transfer** (350 K, 700 K) and **1 ns** evaluations for headline
   configurations: eval-only. ~50–100 h.
4. **GFN2-xTB per-pair margins**: CPU only (a 2,000-pair buffer is minutes of xTB),
   so nearly free on GPU budget.
5. More rMD17 molecules with SchNet (naphthalene, paracetamol): cheap, more breadth.

Skip water/GemNet-T (31 h pretraining alone) unless everything above is done.
