#!/usr/bin/env bash
set -euo pipefail

# arg check
[[ $# -ge 1 ]] || { echo "usage: $0 /path/to/BASE_DIR (e.g., .../040625-LundoBin)"; exit 1; }

BASE="${1%/}"
SCENE="$BASE/$(basename "$BASE")-All"
LOG="$SCENE/pose_log.txt"
#rm -rf "$SCENE"
#mkdir -p "$SCENE"

# show on screen AND save to log
exec > >(tee -a "$LOG") 2>&1
trap 'echo "[ERROR] failed at line $LINENO (exit $?)"' ERR

echo "==> $(date -Is) Start | BASE=$BASE | SCENE=$SCENE"

# 0) collect all images
#python3 make_all.py "$BASE"

# 1) Optional pre-processing
bash local_colmap_and_resize.sh "$SCENE"

# 2) .bin -> .txt
colmap model_converter \
  --input_path "$SCENE/undistortion_sparse/0" \
  --output_path "$SCENE/undistortion_sparse/0" \
  --output_type TXT

# 3) .txt -> .json
python3 colmap2nerf.py \
  --aabb_scale 128 \
  --images "$SCENE/undistortion_images" \
  --out "$SCENE/transforms.json" \
  --text "$SCENE/undistortion_sparse/0"

python3 make_split.py "$SCENE"


echo "==> $(date -Is) Done. Log saved at: $LOG"

