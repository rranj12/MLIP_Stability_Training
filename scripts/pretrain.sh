#!/usr/bin/env bash
# Stage A: standard energy/force pretraining with StABlE's own pretraining.py.
#   scripts/pretrain.sh schnet aspirin [size=1k] [overrides, e.g. --optim.batch_size=20]
#   scripts/pretrain.sh gemnet_t aspirin
#   scripts/pretrain.sh nequip aspirin [size=1k] [l_max]   (uses nequip-train)
#   DATASET=md22 scripts/pretrain.sh schnet AT_AT_CG_CG     (after scripts/prepare_md22.py)
# Every method's Stage B starts from the checkpoint this produces.
# TRAIN_SEED=0 writes an independent run and checkpoint root under seed0; omit for legacy paths.
set -euo pipefail
MODEL=$1; MOL=$2; SIZE=${3:-1k}
DATASET=${DATASET:-md17}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_DIR="$ROOT/results/pretrain/${MODEL}_${MOL}_${SIZE}"
CKPT_BASE="$ROOT/results/checkpoints/qm_pretrained"
if [ -n "${TRAIN_SEED:-}" ]; then
  RUN_DIR="$RUN_DIR/seed${TRAIN_SEED}"
  CKPT_BASE="$CKPT_BASE/seed${TRAIN_SEED}"
fi
mkdir -p "$RUN_DIR"
cd "$ROOT/third_party/StABlE-Training"

if [ "$MODEL" = "nequip" ]; then
  # StABlE ships NequIP configs written for its own splits. Rewrite the paths and sizes
  # for ours and train with nequip-train directly, into the directory
  # StABlE's stable_training.py and our backend load from. Optional 4th argument: l_max
  # (default: the config's value; StABlE's per-sample gradient memory grows with model size).
  LMAX=${4:-}
  CKPT_ROOT="$CKPT_BASE/nequip"
  python - "$MOL" "$SIZE" "$ROOT" "$CKPT_ROOT" "$RUN_DIR/nequip.yml" "$DATASET" "$LMAX" "${TRAIN_SEED:-}" <<'PY'
import sys, yaml
mol, size, root, ckpt_root, out, dataset, lmax, seed = sys.argv[1:9]
cfg = yaml.safe_load(open(f"configs/pretraining/{dataset}/nequip/{mol}.yml"))
if seed:
    cfg["seed"] = int(seed)
if lmax:
    cfg["l_max"] = int(lmax)
cfg.pop("wandb_project", None)  # nequip-train rejects it as an unused key once wandb is off
data = f"{root}/data/processed/{dataset}/{mol}/{size}"
import numpy as np
cfg.update(
    root=ckpt_root, run_name=f"{dataset}-{mol}_{size}_lmax={cfg['l_max']}_nequip",
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
  # StABlE's patience of 5 epochs is 50 gradient steps on the 950-frame split (10 steps per
  # epoch): the LR hits its 1e-6 floor after ~4k steps, still underfit. 50 epochs gives the
  # same steps per patience window as on the 9,500-frame split. PATIENCE=5 reproduces StABlE.
  PATIENCE=${PATIENCE:-50}
  # Every dataset uses StABlE's md17 model configs: its md22 ones include files that do not
  # exist, and the two sets differ only in the dataset block, which is overridden here.
  ARGS=(--molecule "$MOL" --dataset.src="$ROOT/data/processed/md17")
  if [ "$DATASET" != md17 ]; then  # MDsim appends the molecule to dataset.src for md17 only
    ARGS=(--dataset.name="$DATASET" --dataset.src="$ROOT/data/processed/$DATASET/$MOL")
  fi
  if [ "$MODEL" = gemnet_t ]; then  # the config names a scale file at a path that does not exist
    ARGS+=(--model.scale_file=configs/pretraining/md17/gemnet-T-scale.json)
  fi
  if [ -n "${TRAIN_SEED:-}" ]; then
    ARGS+=(--seed "$TRAIN_SEED")
  fi
  python "$ROOT/scripts/pretrain_mdsim.py" --mode train \
    --config-yml "configs/pretraining/md17/${MODEL/gemnet_t/gemnet-T}.yml" \
    --size "$SIZE" --patience "$PATIENCE" \
    --run-dir "$RUN_DIR" --identifier "${MODEL}_${MOL}_${SIZE}_p${PATIENCE}" \
    --dataset.molecule="$MOL" \
    --logger.name=tensorboard "${ARGS[@]}" "${@:4}"
fi

# StABlE's stable_training.py looks for <model_dir>/<model>/<dataset>-<mol>_<size>_<model>/checkpoints/.
DEST="$CKPT_BASE/$MODEL/${DATASET}-${MOL}_${SIZE}_${MODEL}"
python - "$RUN_DIR" "$DEST" <<'PY'
import sys
from pathlib import Path
run_dir, dest = map(Path, sys.argv[1:])
ckpt = max(run_dir.glob("*/checkpoints/best_checkpoint.pt"), key=lambda p: p.stat().st_mtime)
dest.mkdir(parents=True, exist_ok=True)
link = dest / "checkpoints"
if link.is_symlink():
    link.unlink()
link.symlink_to(ckpt.parent)
print(f"Done: {link} -> {ckpt.parent}")
PY
