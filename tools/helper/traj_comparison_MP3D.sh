#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash tools/helper/traj_comparison_MP3D.sh [SCENE|all] [NUM_RUN] [EXP] [ENABLE_VIS] [GPU]

Arguments:
  SCENE       MP3D scene id, or "all". Default: GdvgFV5R1Z5
  NUM_RUN     Number of run_i result folders to evaluate from run_0. Default: 1
  EXP         Experiment/result folder name. Default: SemanticHeat
  ENABLE_VIS  Forwarded to evaluators. Default: 0
  GPU         CUDA_VISIBLE_DEVICES value. Default: 0

Common environment variables:
  CODE_ROOT=/media/phudh/X9_Pro/undergraduate/ActiveMapping
  RESULT_ROOT=<HDD result root>
    Default is /media/phudh/HDD/Active_mapping_results_MP3D for SemanticHeat,
    and /media/phudh/HDD/ActiveSGM_result/results for ActiveSem.
  TRAJ_ROOT=/media/phudh/HDD/trajectory_generate/keyboard_trajectories/MP3D
  STAGE=exploration_stage_1
  METRICS=both
  EVAL_SUFFIX=new_traj
  SKIP_EXISTING=0
  PLOT_EVAL_DATA=1
  PLOT_STRIDE=1
  SCENES="GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG"
  SUMMARY_DIR=results

New trajectory data:
  By default, each scene uses:
    $TRAJ_ROOT/<scene>/test
  The folder must contain traj.txt and results_habitat/.

  To evaluate one explicit prepared folder:
    EVAL_DATA_DIR=/path/to/folder bash tools/helper/traj_comparison_MP3D.sh GdvgFV5R1Z5

  To pair a new traj.txt with an existing results_habitat/:
    NEW_TRAJ_FILE=/path/to/traj.txt bash tools/helper/traj_comparison_MP3D.sh GdvgFV5R1Z5

  Optional knobs for NEW_TRAJ_FILE:
    SOURCE_EVAL_DATA_DIR=<folder containing results_habitat/>
    SOURCE_RESULTS_HABITAT=<direct results_habitat path>
    PREPARED_EVAL_DATA_DIR=<output folder>
    PREPARED_EVAL_ROOT=/tmp/activemapping_eval_trajectory_data
    COPY_RESULTS_HABITAT=0
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCENE_ARG=${1:-GdvgFV5R1Z5}
NUM_RUN=${2:-1}
EXP=${3:-SemanticHeat}
ENABLE_VIS=${4:-1}
GPU=${5:-0,1}

if ! [[ "$NUM_RUN" =~ ^[0-9]+$ ]] || [[ "$NUM_RUN" -lt 1 ]]; then
  echo "[error] NUM_RUN must be a positive integer, got: $NUM_RUN" >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CODE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET=MP3D
CODE_ROOT=${CODE_ROOT:-$DEFAULT_CODE_ROOT}
if [[ -z "${RESULT_ROOT+x}" ]]; then
  if [[ "$EXP" == "ActiveSem" ]]; then
    RESULT_ROOT=/media/phudh/HDD/ActiveSGM_result/MP3D_52
  else
    RESULT_ROOT=/media/phudh/HDD/results_MP3D_reduce_candidates
  fi
fi
TRAJ_ROOT=${TRAJ_ROOT:-/media/phudh/HDD/trajectory_generate/keyboard_trajectories/MP3D}
# TRAJ_ROOT=${TRAJ_ROOT:-/media/phudh/X9_Pro/undergraduate/ActiveMapping/data/mp3d_sim_nvs_v2}
STAGE=${STAGE:-exploration_stage_1}
METRICS=${METRICS:-both}
EVAL_SUFFIX=${EVAL_SUFFIX:-new_traj}
SKIP_EXISTING=${SKIP_EXISTING:-0}
PLOT_EVAL_DATA=${PLOT_EVAL_DATA:-1}
PLOT_STRIDE=${PLOT_STRIDE:-1}
SUMMARY_DIR=${SUMMARY_DIR:-results}
PREPARED_EVAL_ROOT=${PREPARED_EVAL_ROOT:-/tmp/activemapping_eval_trajectory_data}
COPY_RESULTS_HABITAT=${COPY_RESULTS_HABITAT:-0}

export CUDA_VISIBLE_DEVICES="$GPU"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/home/phudh/.Xauthority}"
export PYTHONNOUSERSITE=1
export PYTHONPATH=
if [[ -f /lib/x86_64-linux-gnu/libGLdispatch.so.0 ]]; then
  export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
fi

read -r -a SCENE_LIST <<< "${SCENES:-GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG}"

if [[ "$SCENE_ARG" == "all" ]]; then
  SELECTED_SCENES=("${SCENE_LIST[@]}")
else
  SELECTED_SCENES=("$SCENE_ARG")
fi

cd "$CODE_ROOT"

prepare_eval_data_dir() {
  local scene=$1
  local default_eval_data_dir="$TRAJ_ROOT/$scene/test"
  # local default_eval_data_dir="$TRAJ_ROOT/$scene"
  local source_eval_data_dir="${SOURCE_EVAL_DATA_DIR:-$default_eval_data_dir}"
  local eval_data_dir="${EVAL_DATA_DIR:-$source_eval_data_dir}"

  if [[ -n "${NEW_TRAJ_FILE:-}" && -z "${EVAL_DATA_DIR:-}" ]]; then
    eval_data_dir="${PREPARED_EVAL_DATA_DIR:-$PREPARED_EVAL_ROOT/$DATASET/$scene/$EVAL_SUFFIX}"
    local prep_cmd=(
      python src/evaluation/prepare_eval_trajectory_data.py
      --traj-file "$NEW_TRAJ_FILE"
      --out-dir "$eval_data_dir"
      --overwrite-traj
    )

    if [[ -n "${SOURCE_RESULTS_HABITAT:-}" ]]; then
      prep_cmd+=(--source-results-habitat "$SOURCE_RESULTS_HABITAT")
    else
      prep_cmd+=(--source-eval-data-dir "$source_eval_data_dir")
    fi
    if [[ "$COPY_RESULTS_HABITAT" == "1" ]]; then
      prep_cmd+=(--copy-results-habitat)
    fi

    "${prep_cmd[@]}" >&2
  fi

  printf '%s\n' "$eval_data_dir"
}

run_one() {
  local scene=$1
  local run_idx=$2
  local run_name="run_${run_idx}"
  local cfg="configs/$DATASET/$scene/$EXP.py"
  local result_dir="$RESULT_ROOT/$DATASET/$scene/$EXP/$run_name"
  local eval_data_dir
  local summary_csv

  if ! eval_data_dir=$(prepare_eval_data_dir "$scene"); then
    return 1
  fi
  summary_csv="$SUMMARY_DIR/eval_compare_${DATASET}_${scene}_${EXP}_${run_name}_${EVAL_SUFFIX}.csv"

  if [[ ! -f "$cfg" ]]; then
    echo "[error] Missing config: $CODE_ROOT/$cfg" >&2
    return 1
  fi
  if [[ ! -d "$result_dir" ]]; then
    echo "[error] Missing result dir: $result_dir" >&2
    return 1
  fi
  if [[ ! -f "$result_dir/splatam/$STAGE/params.npz" ]]; then
    echo "[error] Missing checkpoint: $result_dir/splatam/$STAGE/params.npz" >&2
    return 1
  fi
  if [[ ! -f "$eval_data_dir/traj.txt" ]]; then
    echo "[error] Missing trajectory file: $eval_data_dir/traj.txt" >&2
    return 1
  fi
  if [[ ! -d "$eval_data_dir/results_habitat" ]]; then
    echo "[error] Missing eval RGB-D folder: $eval_data_dir/results_habitat" >&2
    return 1
  fi

  mkdir -p "$SUMMARY_DIR"

  echo
  echo "### MP3D trajectory rendering eval"
  echo "Scene      : $scene"
  echo "Run        : $run_name"
  echo "EXP        : $EXP"
  echo "Code root  : $CODE_ROOT"
  echo "Result dir : $result_dir"
  echo "Eval data  : $eval_data_dir"
  echo "Checkpoint : splatam/$STAGE/params.npz"
  echo "Output dir : $result_dir/splatam/eval_$EVAL_SUFFIX"
  echo "Summary CSV: $summary_csv"

  local cmd=(
    python src/evaluation/eval_trajectory_compare.py
    --eval_data_dir "$eval_data_dir"
    --dataset "$DATASET"
    --run "$EXP=$cfg=$result_dir"
    --stage "$STAGE"
    --metrics "$METRICS"
    --eval_suffix "$EVAL_SUFFIX"
    --gpu "$GPU"
    --enable_vis "$ENABLE_VIS"
    --summary_csv "$summary_csv"
    --plot_stride "$PLOT_STRIDE"
  )

  if [[ "$SKIP_EXISTING" == "1" ]]; then
    cmd+=(--skip_existing)
  fi
  if [[ "$PLOT_EVAL_DATA" != "1" ]]; then
    cmd+=(--no_plot_eval_data)
  fi
  if [[ -n "${PLOT_DIR:-}" ]]; then
    cmd+=(--plot_dir "$PLOT_DIR")
  fi

  "${cmd[@]}"
}

failed=0
for scene in "${SELECTED_SCENES[@]}"; do
  for ((run_idx = 0; run_idx < NUM_RUN; run_idx++)); do
    if ! run_one "$scene" "$run_idx"; then
      if [[ "$SCENE_ARG" == "all" ]]; then
        echo "[warn] Skip $DATASET/$scene/run_$run_idx because required inputs are missing." >&2
        failed=1
        continue
      fi
      exit 1
    fi
  done
done

if [[ "$SCENE_ARG" == "all" && "$failed" -eq 1 ]]; then
  echo "[warn] Finished with skipped scenes/runs because some inputs were missing." >&2
fi
