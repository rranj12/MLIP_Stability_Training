#!/usr/bin/env bash
# Stage A: standard energy/force pretraining with StABlE's own pretraining.py.
#   scripts/pretrain.sh schnet aspirin [size=1k]
#   scripts/pretrain.sh nequip aspirin [size=1k] [l_max]   (uses nequip-train)
# Every method's Stage B starts from the checkpoint this produces.
set -euo pipefail
MODEL=$1; MOL=$2; SIZE=${3:-1k}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR="$ROOT/results/pretrain/${MODEL}_${MOL}_${SIZE}"
mkdir -p "$RUN_DIR"
cd "$ROOT/third_party/StABlE-Training"

if [ "$MODEL" = "nequip" ]; then
  # StABlE ships NequIP MD17 configs written for its 10k split. Rewrite the paths and sizes
  # for our 950/50 rMD17 split and train with nequip-train directly, into the directory
  # StABlE's stable_training.py and our backend load from. Optional 4th argument: l_max
  # (default: the config's 3; StABlE's per-sample gradient memory grows with model size).
  LMAX=${4:-}
  CKPT_ROOT="$ROOT/results/checkpoints/qm_pretrained/nequip"
  python - "$MOL" "$SIZE" "$ROOT" "$CKPT_ROOT" "$RUN_DIR/nequip.yml" "$LMAX" <<'PY'
import sys, yaml
mol, size, root, ckpt_root, out, lmax = sys.argv[1:7]
cfg = yaml.safe_load(open(f"configs/pretraining/md17/nequip/{mol}.yml"))
if lmax:
    cfg["l_max"] = int(lmax)
data = f"{root}/data/processed/md17/{mol}/{size}"
import numpy as np
cfg.update(
    root=ckpt_root, run_name=f"md17-{mol}_{size}_lmax={cfg['l_max']}_nequip",
    dataset_file_name=f"{data}/train/nequip_npz.npz",
    validation_dataset_file_name=f"{data}/val/nequip_npz.npz",
    n_train=len(np.load(f"{data}/train/nequip_npz.npz")["E"]),
    n_val=len(np.load(f"{data}/val/nequip_npz.npz")["E"]), wandb=False)
yaml.safe_dump(cfg, open(out, "w"))
PY
  nequip-train "$RUN_DIR/nequip.yml"
  echo "Done: $CKPT_ROOT (StABlE's layout, nothing to link)."
  exit 0
else
  python pretraining.py --mode train \
    --config-yml "configs/pretraining/md17/${MODEL}.yml" \
    --molecule "$MOL" --size "$SIZE" \
    --run-dir "$RUN_DIR" --identifier "${MODEL}_${MOL}_${SIZE}" \
    --dataset.src="$ROOT/data/processed/md17" \
    --logger.name=tensorboard
fi

# StABlE's stable_training.py looks for <model_dir>/<model>/md17-<mol>_<size>_<model>/checkpoints/.
DEST="$ROOT/results/checkpoints/qm_pretrained/$MODEL/md17-${MOL}_${SIZE}_${MODEL}"
echo "Done. Link the trained run into StABlE's expected location, e.g.:"
echo "  mkdir -p $(dirname "$DEST") && ln -s <run_dir>/checkpoints/<timestamp-id> $DEST/checkpoints"
