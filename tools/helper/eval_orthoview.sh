#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash ./tools/helper/eval_orthoview.sh [RESULTS_ROOT] [DATASET] [RUN_NAME] [PARAM_STAGE] [SCENE...]
#
# Examples:
#   bash ./tools/helper/eval_orthoview.sh
#   RES=0.05 SAVE_VIEWS=0 bash ./tools/helper/eval_orthoview.sh results Replica run_0_keyframe final room2

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

RESULTS_ROOT="${1:-$REPO_ROOT/results}"
DATASET="${2:-Replica}"
RUN_NAME="${3:-run_0_keyframe}"
PARAM_STAGE="${4:-exploration_stage_1}"

EVAL_SCRIPT="$REPO_ROOT/tools/helper/eval_orthoview_miou.py"
RENDER_MODE="${RENDER_MODE:-gaussian}"
RES="${RES:-0.01}"
REMOVE_FRONT_PERCENT="${REMOVE_FRONT_PERCENT:-20.0}"
SAVE_VIEWS="${SAVE_VIEWS:-1}"
OUT_SUBDIR="${OUT_SUBDIR:-visualization/eval_orthoview_${PARAM_STAGE}}"

REPLICA_SCENES=(
    office0
    office1
    office2
    office3
    office4
    room0
    room1
    room2
)

MP3D_SCENES=(
    GdvgFV5R1Z5
    HxpKQynjfin
    gZ6f7yhEvPG
    pLe4wQe7qrG
)

if (( $# > 4 )); then
  SCENES=("${@:5}")
elif [[ "$DATASET" == "Replica" ]]; then
  SCENES=("${REPLICA_SCENES[@]}")
elif [[ "$DATASET" == "MP3D" ]]; then
  SCENES=("${MP3D_SCENES[@]}")
else
  echo "[ERROR] Unknown dataset '$DATASET'. Pass SCENE names after PARAM_STAGE." >&2
  exit 1
fi

HEIGHT_ARGS=()
if [[ -n "${MIN_HEIGHT:-}" ]]; then
  HEIGHT_ARGS+=(--min-height "$MIN_HEIGHT")
fi
if [[ -n "${MAX_HEIGHT:-}" ]]; then
  HEIGHT_ARGS+=(--max-height "$MAX_HEIGHT")
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "[ERROR] Missing evaluator: $EVAL_SCRIPT" >&2
  exit 1
fi

cd "$REPO_ROOT"

echo "[INFO] RESULTS_ROOT=$RESULTS_ROOT"
echo "[INFO] DATASET=$DATASET RUN_NAME=$RUN_NAME PARAM_STAGE=$PARAM_STAGE"
echo "[INFO] RENDER_MODE=$RENDER_MODE RES=$RES REMOVE_FRONT_PERCENT=$REMOVE_FRONT_PERCENT"

for scene in "${SCENES[@]}"; do
  # npz="$RESULTS_ROOT/$DATASET/$scene/ActiveSem/$RUN_NAME/splatam/$PARAM_STAGE/params.npz"

  npz="$RESULTS_ROOT/$DATASET/$scene/SemanticHeat/$RUN_NAME/splatam/$PARAM_STAGE/params.npz"
  if [[ ! -f "$npz" ]]; then
    echo "[WARN] Missing npz: $npz"
    continue
  fi

  # out_dir="$RESULTS_ROOT/$DATASET/$scene/ActiveSem/$RUN_NAME/$OUT_SUBDIR"
  out_dir="$RESULTS_ROOT/$DATASET/$scene/SemanticHeat/$RUN_NAME/$OUT_SUBDIR"
  mkdir -p "$out_dir"

  echo "[INFO] Evaluating $scene -> $out_dir"
  python "$EVAL_SCRIPT" \
    --npz "$npz" \
    --dataset "$DATASET" \
    --scene "$scene" \
    --out-dir "$out_dir" \
    --render-mode "$RENDER_MODE" \
    --res "$RES" \
    --remove-front-percent "$REMOVE_FRONT_PERCENT" \
    "${HEIGHT_ARGS[@]}" \
    --mp3d-palette eval\
    --save-views
done
