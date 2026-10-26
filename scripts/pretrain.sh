#!/usr/bin/env bash
# Stage A: standard energy/force pretraining with StABlE's own pretraining.py.
#   scripts/pretrain.sh schnet aspirin [size=1k]
#   scripts/pretrain.sh nequip aspirin            (uses nequip-train)
# Every method's Stage B starts from the checkpoint this produces.
set -euo pipefail
MODEL=$1; MOL=$2; SIZE=${3:-1k}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR="$ROOT/results/pretrain/${MODEL}_${MOL}_${SIZE}"
mkdir -p "$RUN_DIR"
cd "$ROOT/third_party/StABlE-Training"

if [ "$MODEL" = "nequip" ]; then
  # TODO(pilot): StABlE's NequIP configs are written for MD22; add an rMD17 config
  # under configs/pretrain/ (NequIP's published rMD17 settings) and point --config-yml at it.
  python pretraining.py --nequip --config-yml "$ROOT/configs/pretrain/nequip_rmd17.yml" \
    --molecule "$MOL" --size "$SIZE"
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
