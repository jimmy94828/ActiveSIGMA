#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash ./tools/helper/traj_comparison.sh [DATASET] [SCENE] [VERSION]

Arguments:
  DATASET   MP3D or Replica
  SCENE     scene name, or "all"
  VERSION   result version, e.g. run_0, run_0_keyframe

Optional environment variables:
  GPU=0,1
  STAGE=final
  METRICS=both
  EVAL_SUFFIX=final_traj
  SKIP_EXISTING=0
  PLOT_EVAL_DATA=1
  PLOT_STRIDE=1
  PLOT_DIR=<each run splatam/eval_${EVAL_SUFFIX}/trajectory_rgbd_semantic>
  EVAL_DATA_DIR=<existing folder with traj.txt and results_habitat/>
  NEW_TRAJ_FILE=<new traj.txt to pair with an existing results_habitat/>
  SOURCE_EVAL_DATA_DIR=<source folder for results_habitat; defaults to dataset trajectory root>
  SOURCE_RESULTS_HABITAT=<source results_habitat path; overrides SOURCE_EVAL_DATA_DIR>
  PREPARED_EVAL_DATA_DIR=<output folder created when NEW_TRAJ_FILE is set>
  PREPARED_EVAL_ROOT=/tmp/activemapping_eval_trajectory_data
  COPY_RESULTS_HABITAT=0
  COMPARE_ACTIVE_SGM=1
  ACTIVE_SGM_ROOT=/media/phudh/HDD/ActiveSGM_result/results
  ACTIVE_MAPPING_ROOT=/media/phudh/HDD/ActiveMapping_result_keyframe
  ACTIVE_SGM_VERSION=<VERSION>
  ACTIVE_MAPPING_VERSION=<VERSION>
  SUMMARY_DIR=results
USAGE
}

if [[ $# -ne 3 ]]; then
  usage
  exit 1
fi

DATASET=$1
SCENE_ARG=$2
VERSION=$3

PROJ_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$PROJ_DIR"

GPU=${GPU:-0,1}
STAGE=${STAGE:-exploration_stage_1}
METRICS=${METRICS:-both}
EVAL_SUFFIX=${EVAL_SUFFIX:-new_traj}
SKIP_EXISTING=${SKIP_EXISTING:-0}
PLOT_EVAL_DATA=${PLOT_EVAL_DATA:-1}
PLOT_STRIDE=${PLOT_STRIDE:-1}
SUMMARY_DIR=${SUMMARY_DIR:-results}
PREPARED_EVAL_ROOT=${PREPARED_EVAL_ROOT:-/tmp/activemapping_eval_trajectory_data}
COPY_RESULTS_HABITAT=${COPY_RESULTS_HABITAT:-0}
COMPARE_ACTIVE_SGM=${COMPARE_ACTIVE_SGM:-1}

ACTIVE_SGM_ROOT=${ACTIVE_SGM_ROOT:-/media/phudh/HDD/ActiveSGM_result/newresults_0609}
ACTIVE_MAPPING_ROOT=${ACTIVE_MAPPING_ROOT:-/media/phudh/HDD/ActiveMapping_result_local_global}
ACTIVE_SGM_VERSION=${ACTIVE_SGM_VERSION:-run_1_origin}
# ACTIVE_SGM_VERSION=${ACTIVE_SGM_VERSION:-$VERSION}
ACTIVE_MAPPING_VERSION=${ACTIVE_MAPPING_VERSION:-$VERSION}

case "$DATASET" in
  MP3D)
    DATA_DIR_ROOT="/media/phudh/HDD/trajectory_generate/keyboard_trajectories/MP3D"
    SCENES=(GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG YmJkqBEsHnH)
    ;;
  Replica)
    DATA_DIR_ROOT="/media/phudh/HDD/trajectory_generate/keyboard_trajectories/Replica"
    SCENES=(room0 room1 room2 office0 office1 office2 office3 office4)
    ;;
  *)
    echo "[error] DATASET must be MP3D or Replica, got: $DATASET" >&2
    usage
    exit 1
    ;;
esac

if [[ "$SCENE_ARG" == "all" ]]; then
  SELECTED_SCENES=("${SCENES[@]}")
else
  SELECTED_SCENES=("$SCENE_ARG")
fi

join_by() {
  local delimiter=$1
  shift
  local first=1
  local item
  for item in "$@"; do
    if [[ $first -eq 1 ]]; then
      printf '%s' "$item"
      first=0
    else
      printf '%s%s' "$delimiter" "$item"
    fi
  done
}


run_scene() {
  local scene=$1
  local default_eval_data_dir="$DATA_DIR_ROOT/$scene/test"
  local source_eval_data_dir="${SOURCE_EVAL_DATA_DIR:-$default_eval_data_dir}"
  local eval_data_dir="${EVAL_DATA_DIR:-$source_eval_data_dir}"
  local active_sgm_cfg="configs/$DATASET/$scene/ActiveSem.py"
  local active_mapping_cfg="configs/$DATASET/$scene/SemanticHeat.py"
  local active_sgm_result="$ACTIVE_SGM_ROOT/$DATASET/$scene/ActiveSem/$ACTIVE_SGM_VERSION"
  local active_mapping_result="$ACTIVE_MAPPING_ROOT/$DATASET/$scene/SemanticHeat/$ACTIVE_MAPPING_VERSION"
  local summary_csv

  if [[ -n "${NEW_TRAJ_FILE:-}" && -z "${EVAL_DATA_DIR:-}" ]]; then
    eval_data_dir="${PREPARED_EVAL_DATA_DIR:-$PREPARED_EVAL_ROOT/$DATASET/$scene/$EVAL_SUFFIX}"
    prep_cmd=(
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
    if ! "${prep_cmd[@]}"; then
      echo "[error] Failed to prepare eval data dir: $eval_data_dir" >&2
      return 1
    fi
  fi

  if [[ ! -d "$eval_data_dir" ]]; then
    echo "[error] Missing eval data dir: $eval_data_dir" >&2
    return 1
  fi
  if [[ "$COMPARE_ACTIVE_SGM" == "1" && ! -f "$active_sgm_cfg" ]]; then
    echo "[error] Missing config: $active_sgm_cfg" >&2
    return 1
  fi
  if [[ ! -f "$active_mapping_cfg" ]]; then
    echo "[error] Missing config: $active_mapping_cfg" >&2
    return 1
  fi
  if [[ "$COMPARE_ACTIVE_SGM" == "1" && ! -d "$active_sgm_result" ]]; then
    echo "[error] Missing ActiveSGM result dir: $active_sgm_result" >&2
    return 1
  fi
  if [[ ! -d "$active_mapping_result" ]]; then
    echo "[error] Missing ActiveMapping result dir: $active_mapping_result" >&2
    return 1
  fi

  mkdir -p "$SUMMARY_DIR"
  summary_csv="$SUMMARY_DIR/eval_compare_${DATASET}_${scene}_${ACTIVE_MAPPING_VERSION}.csv"

  echo
  echo "### Trajectory comparison: $DATASET/$scene"
  if [[ "$COMPARE_ACTIVE_SGM" == "1" ]]; then
    echo "ActiveSGM     : $active_sgm_result"
  fi
  echo "ActiveMapping : $active_mapping_result"
  echo "Summary CSV   : $summary_csv"

  cmd=(
    python src/evaluation/eval_trajectory_compare.py
    --eval_data_dir "$eval_data_dir"
    --dataset "$DATASET"
  )
  if [[ "$COMPARE_ACTIVE_SGM" == "1" ]]; then
    cmd+=(--run "ActiveSGM=$active_sgm_cfg=$active_sgm_result")
  fi
  cmd+=(
    --run "ActiveMapping=$active_mapping_cfg=$active_mapping_result"
    --stage "$STAGE"
    --metrics "$METRICS"
    --eval_suffix "$EVAL_SUFFIX"
    --gpu "$GPU"
    --summary_csv "$summary_csv"
    --plot_stride "$PLOT_STRIDE"
  )

  if [[ -n "${PLOT_DIR:-}" ]]; then
    cmd+=(--plot_dir "$PLOT_DIR")
  fi
  if [[ "$SKIP_EXISTING" == "1" ]]; then
    cmd+=(--skip_existing)
  fi
  if [[ "$PLOT_EVAL_DATA" != "1" ]]; then
    cmd+=(--no_plot_eval_data)
  fi

  "${cmd[@]}"
}

failed=0
for scene in "${SELECTED_SCENES[@]}"; do
  if ! run_scene "$scene"; then
    if [[ "$SCENE_ARG" == "all" ]]; then
      echo "[warn] Skip $DATASET/$scene/$VERSION" >&2
      failed=1
      continue
    fi
    exit 1
  fi
done

if [[ "$SCENE_ARG" == "all" && $failed -eq 1 ]]; then
  echo "[warn] Finished with skipped scenes because some inputs were missing." >&2
fi
