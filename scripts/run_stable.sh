#!/usr/bin/env bash
# StABlE baseline, run unmodified from the pinned submodule.
#   scripts/run_stable.sh configs/stable/aspirin_schnet_train.yml [--seed 1]
# Extra arguments are passed through as StABlE overrides (key=value) or flags.
set -euo pipefail
CFG=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT/third_party/StABlE-Training"
python stable_training.py --config-yml "$CFG" "$@"
