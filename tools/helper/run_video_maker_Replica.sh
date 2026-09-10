#!/usr/bin/env bash
# Convert Replica render_all_info image folders into videos.
#
# Examples:
#   bash tools/helper/run_video_maker_Replica.sh keyframe --scene office1
#   bash tools/helper/run_video_maker_Replica.sh --version run_0_keyframe
#   bash tools/helper/run_video_maker_Replica.sh v6 --fps 10 --dry-run
#   bash tools/helper/run_video_maker_Replica.sh keyframe --scene office1 --python-bin /home/phudh/.conda/envs/visual/bin/python
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
PHOTO2VIDEO="$SCRIPT_DIR/photos2video.py"

VERSION=""
DATASET="Replica"
METHOD="SemanticHeat"
INPUT_REL="visualization/render_all_info"
OUTPUT_NAME="render_all_info.mp4"
FPS=3
EXTS="png"
RECURSIVE=0
RESIZE=0
DRY_RUN=0
SELECTED_SCENES=()

usage() {
  cat <<'EOF'
Usage:
  bash tools/helper/run_video_maker_Replica.sh --version <version> [options]
  bash tools/helper/run_video_maker_Replica.sh <version> [options]

Examples:
  bash tools/helper/run_video_maker_Replica.sh keyframe --scene office1
  bash tools/helper/run_video_maker_Replica.sh run_0_keyframe
  bash tools/helper/run_video_maker_Replica.sh v6 --fps 10 --dry-run
  bash tools/helper/run_video_maker_Replica.sh keyframe --scene office1 --python-bin /home/phudh/.conda/envs/visual/bin/python

Version mapping:
  keyframe      -> run_0_keyframe
  poster        -> run_0_poster
  v6 / _v6      -> run_0_v6
  run_0_v6      -> run_0_v6
  run_0         -> run_0

Options:
  --version <value>       Run version or suffix.
  --scene <name>          Scene to process. Can be repeated. Default: all Replica scenes.
  --dataset <value>       Dataset name. Default: Replica
  --method <value>        Method directory. Default: SemanticHeat
  --input-rel <path>      Image folder relative to the run dir.
                          Default: visualization/render_all_info
  --output-name <name>    Output mp4 filename under the input folder's parent.
                          Default: render_all_info.mp4
  --fps <num>             Frames per second. Default: 3
  --exts <list>           Comma-separated image extensions. Default: png
  --python-bin <path>     Python executable with cv2 installed.
                          Default: PYTHON_BIN env var, or python
  --recursive             Recursively search the input folder.
  --resize                Resize images to match the first frame.
  --dry-run               Print commands only.
  -h, --help              Show this help.
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version)
      VERSION="${2:-}"
      shift 2
      ;;
    --scene)
      SELECTED_SCENES+=("${2:-}")
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
    --input-rel)
      INPUT_REL="${2:-}"
      shift 2
      ;;
    --output-name)
      OUTPUT_NAME="${2:-}"
      shift 2
      ;;
    --fps)
      FPS="${2:-}"
      shift 2
      ;;
    --exts)
      EXTS="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --recursive)
      RECURSIVE=1
      shift
      ;;
    --resize)
      RESIZE=1
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

if [[ ! -f "$PHOTO2VIDEO" ]]; then
  echo "photos2video.py not found: $PHOTO2VIDEO" >&2
  exit 1
fi

if [[ "${#SELECTED_SCENES[@]}" -eq 0 ]]; then
  SELECTED_SCENES=("${SCENES[@]}")
fi

RUN_NAME=$(normalize_run_name "$VERSION")

echo "Project root      : $PROJ"
echo "Dataset / Method  : $DATASET / $METHOD"
echo "Run name          : $RUN_NAME"
echo "Input relative dir: $INPUT_REL"
echo "Output filename   : $OUTPUT_NAME"
echo "FPS / extensions  : $FPS / $EXTS"
echo "Python            : $PYTHON_BIN"
echo "Scenes            : ${SELECTED_SCENES[*]}"
echo

for scene in "${SELECTED_SCENES[@]}"; do
  run_dir="$PROJ/results/$DATASET/$scene/$METHOD/$RUN_NAME"
  input_dir="$run_dir/$INPUT_REL"
  output_dir=$(dirname "$input_dir")
  output_path="$output_dir/$OUTPUT_NAME"

  if [[ ! -d "$input_dir" ]]; then
    echo "[skip] $scene"
    echo "  missing input dir: $input_dir"
    echo
    continue
  fi

  cmd=(
    "$PYTHON_BIN"
    "$PHOTO2VIDEO"
    --input-dir "$input_dir"
    --output "$output_path"
    --fps "$FPS"
    --exts "$EXTS"
  )

  if [[ "$RECURSIVE" -eq 1 ]]; then
    cmd+=(--recursive)
  fi

  if [[ "$RESIZE" -eq 1 ]]; then
    cmd+=(--resize)
  fi

  echo "[run] $scene"
  echo "  in : $input_dir"
  echo "  out: $output_path"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '  cmd:'
    printf ' %q' "${cmd[@]}"
    printf '\n\n'
    continue
  fi

  "${cmd[@]}"
  echo
done
