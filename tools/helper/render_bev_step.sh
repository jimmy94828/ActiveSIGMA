#!/usr/bin/env bash
# bash ./tools/helper/render_bev.sh --version keyframe
set -euo pipefail

SCENES=(
  office0
  office1
  office2
  office3
  office4
  room0
  room1
  room2
)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJ=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
RENDERER="$PROJ/src/visualization/render_rgb_sem_entro_bev.py"

VERSION=""
DATASET="Replica"
METHOD="SemanticHeat"
NPZ_REL="splatam/final/params.npz"
OUT_DIR="visualization"
OUT_PREFIX="bev"
COORD_SYSTEM="sim"
RENDER_MODE="gaussian"
REMOVE_FRONT_PERCENT="25"
RENDER_GT_SEMANTIC=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash tools/helper/render_bev.sh --version <version> [options]
  bash tools/helper/render_bev.sh <version> [options]

Examples:
  bash tools/helper/render_bev.sh keyframe
  bash tools/helper/render_bev.sh poster --npz-rel splatam/exploration_stage_1/params.npz
  bash tools/helper/render_bev.sh run_0_v6 --no-gt-semantic --dry-run

Version mapping:
  keyframe      -> run_0_keyframe
  poster        -> run_0_poster
  v6 / _v6      -> run_0_v6
  run_0_v6      -> run_0_v6
  run_0         -> run_0

Options:
  --version <value>              Run version or suffix.
  --dataset <value>              Dataset name. Default: Replica
  --method <value>               Method directory. Default: SemanticHeat
  --npz-rel <path>               Relative npz path under run dir.
                                 Default: splatam/final/params.npz
  --out-dir <path>               Output dir under run dir. Default: visualization
  --out-prefix <name>            Output filename prefix. Default: bev
  --coord-system <sim|slam>      Default: sim
  --render-mode <gaussian|nearest>
                                 Default: gaussian
  --remove-front-percent <num>   Default: 25
  --no-gt-semantic               Do not render GT semantic.
  --dry-run                      Print commands only.
  -h, --help                     Show this help.
EOF
}

normalize_run_name() {
  local raw="$1"
  if [[ -z "$raw" ]]; then
    echo ""
    return
  fi

  if [[ "$raw" == run_* ]]; then
    echo "$raw"
    return
  fi

  if [[ "$raw" == _* ]]; then
    echo "run_0$raw"
    return
  fi

  if [[ "$raw" =~ ^v[0-9A-Za-z._-]+$ ]]; then
    echo "run_0_$raw"
    return
  fi

  if [[ "$raw" == "0" || "$raw" == "base" ]]; then
    echo "run_0"
    return
  fi

  echo "run_0_$raw"
}

derive_npz_tag() {
  local npz_rel="$1"
  local npz_name parent_dir stem

  npz_name=$(basename "$npz_rel")
  parent_dir=$(basename "$(dirname "$npz_rel")")
  stem="${npz_name%.npz}"

  if [[ "$npz_name" == "params.npz" && "$parent_dir" != "." && "$parent_dir" != "splatam" ]]; then
    echo "$parent_dir"
    return
  fi

  echo "$stem"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="${2:-}"
      shift 2
      ;;
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    --method)
      METHOD="${2:-}"
      shift 2
      ;;
    --npz-rel)
      NPZ_REL="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --out-prefix)
      OUT_PREFIX="${2:-}"
      shift 2
      ;;
    --coord-system)
      COORD_SYSTEM="${2:-}"
      shift 2
      ;;
    --render-mode)
      RENDER_MODE="${2:-}"
      shift 2
      ;;
    --remove-front-percent)
      REMOVE_FRONT_PERCENT="${2:-}"
      shift 2
      ;;
    --no-gt-semantic)
      RENDER_GT_SEMANTIC=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "$VERSION" ]]; then
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 1
      fi
      VERSION="$1"
      shift
      ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  echo "Missing version." >&2
  usage >&2
  exit 1
fi

RUN_NAME=$(normalize_run_name "$VERSION")
NPZ_TAG=$(derive_npz_tag "$NPZ_REL")

echo "Project root         : $PROJ"
echo "Dataset / Method     : $DATASET / $METHOD"
echo "Run name             : $RUN_NAME"
echo "NPZ relative path    : $NPZ_REL"
echo "Output prefix        : $OUT_PREFIX"
echo "Scenes               : ${SCENES[*]}"
echo

for scene in "${SCENES[@]}"; do
  run_dir="$PROJ/results/$DATASET/$scene/$METHOD/$RUN_NAME"
  npz_path="$run_dir/$NPZ_REL"
  out_path="$run_dir/$OUT_DIR/${OUT_PREFIX}_${NPZ_TAG}.png"

  if [[ ! -f "$npz_path" ]]; then
    echo "[skip] $scene"
    echo "  missing npz: $npz_path"
    echo
    continue
  fi

  cmd=(
    "$PYTHON_BIN"
    "$RENDERER"
    --npz "$npz_path"
    --dataset "$DATASET"
    --scene "$scene"
    --out "$out_path"
    --coord-system "$COORD_SYSTEM"
    --render-mode "$RENDER_MODE"
    --remove-front-percent "$REMOVE_FRONT_PERCENT"
  )

  if [[ "$RENDER_GT_SEMANTIC" -eq 1 ]]; then
    cmd+=(--render-gt-semantic)
  fi

  echo "[run] $scene"
  echo "  npz: $npz_path"
  echo "  out: $out_path"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  cmd:'
    printf ' %q' "${cmd[@]}"
    printf '\n\n'
    continue
  fi

  "${cmd[@]}"
  echo
done
