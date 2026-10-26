# Running on AWS

1. **Quota first** (can take days): Service Quotas → EC2 → "Running On-Demand G and
   VT instances" and "All G and VT Spot Instance Requests", ask for ≥ 8 vCPUs.
2. **Instance**: g5.xlarge (A10G 24 GB) or g6.xlarge (L4 24 GB) with an AWS Deep
   Learning Base AMI (Ubuntu, CUDA 12.x drivers). Spot for SchNet sweeps and evals;
   on-demand for long runs you can't afford to lose.
3. **Storage**: keep `data/` and `results/` in an S3 bucket; instances are disposable.
4. **Bootstrap**: `bash cloud/aws/bootstrap.sh` on the instance.
5. **Spot interruptions**: run inside `tmux`, checkpoint every cycle/epoch, and sync
   `results/` to S3 periodically (`aws s3 sync results s3://<bucket>/results`).
6. **Stop instances when idle.** A forgotten g5.xlarge costs ≈ $24/day.
