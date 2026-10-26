#!/usr/bin/env bash
# Set up the environment on a fresh GPU instance (Deep Learning Base AMI, Ubuntu).
#   bash cloud/aws/bootstrap.sh [s3://bucket/prefix]
set -euo pipefail
BUCKET=${1:-}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
git submodule update --init --recursive

if ! command -v mamba >/dev/null; then
  curl -L -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
fi
mamba env create -f environment.yml || mamba env update -f environment.yml
echo "Activate with: conda activate mlipstab"

if [ -n "$BUCKET" ]; then
  aws s3 sync "$BUCKET/data/processed" data/processed
  aws s3 sync "$BUCKET/results/checkpoints" results/checkpoints
fi
nvidia-smi
