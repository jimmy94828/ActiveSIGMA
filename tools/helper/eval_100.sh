#!/usr/bin/env bash
set -euo pipefail

PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RESULTS_ROOT=${1:-/media/phudh/HDD/ActiveMapping_result_local_global}
DATASET=${2:-Replica}
SEED=0

shopt -s nullglob

for scene_dir in "$RESULTS_ROOT"/$DATASET/room*; do
  [[ -d "$scene_dir" ]] || continue
  scene=$(basename "$scene_dir")
  run_root="$scene_dir/SemanticHeat"
  [[ -d "$run_root" ]] || continue

  cfg="$PROJ/configs/$DATASET/$scene/SemanticHeat.py"
  if [[ ! -f "$cfg" ]]; then
    echo "[warn] Missing config: $cfg"
    continue
  fi

  mesh_scene=${scene/office/office_}
  mesh_scene=${mesh_scene/room/room_}
  gt_mesh="$PROJ/data/replica_v1/${mesh_scene}/mesh.ply"
  traj="$PROJ/data/Replica/$scene/traj.txt"

  if [[ ! -f "$gt_mesh" ]]; then
    echo "[warn] Missing GT mesh: $gt_mesh"
    continue
  fi
  if [[ ! -f "$traj" ]]; then
    echo "[warn] Missing traj: $traj"
    continue
  fi

  for run_dir in "$run_root"/*; do
    [[ -d "$run_dir" ]] || continue

    npz_files=("$run_dir"/splatam/params*.npz)
    if [[ ${#npz_files[@]} -eq 0 ]]; then
      echo "[warn] No checkpoints: $run_dir/splatam/params*.npz"
      continue
    fi

    for npz in "${npz_files[@]}"; do
      base=$(basename "$npz")
      step=${base#params}
      step=${step%.npz}
      if [[ -z "$step" ]]; then
        step=0
      fi

      if [[ ! "$step" =~ ^[0-9]+$ ]]; then
        echo "[warn] Skip non-numeric step: $npz"
        continue
      fi

      if (( step % 100 != 0 )); then
        continue
      fi

      # Semantic evaluation (skip step 0 because eval_semantic uses stage/params.npz when step==0)
      if [[ "$step" != "0" ]]; then
        python "$PROJ/src/evaluation/eval_semantic.py" \
          --cfg "$cfg" \
          --seed "$SEED" \
          --result_dir "$run_dir" \
          --step "$step" \
          --stage "step_${step}"

        # RGB evaluation (render_result.txt, psnr/ssim/lpips)
        python "$PROJ/src/evaluation/eval_rgb_by_step.py" \
          --cfg "$cfg" \
          --seed "$SEED" \
          --result_dir "$run_dir" \
          --step "$step" \
          --stage "step_${step}"
      fi

      # 3D reconstruction evaluation
      python "$PROJ/src/evaluation/eval_splatam_recon_v2.py" \
        --ckpt "$npz" \
        --gt_mesh "$gt_mesh" \
        --transform_traj "$traj" \
        --result_dir "$run_dir/eval_3d/step_${step}"
    done
  done
done