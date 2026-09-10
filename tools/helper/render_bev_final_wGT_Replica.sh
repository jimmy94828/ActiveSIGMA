#!/usr/bin/env bash
# bash ./tools/helper/render_bev_explorationstage1_wGT.sh
set -euo pipefail
for item in \
  "room2://media/phudh/X9_Pro/undergraduate/ActiveMapping/results_all/Replica/room2/SemanticHeat/run_0"
  # "office0:/media/phudh/X9_Pro/undergraduate/ActiveMapping/results_all/Replica/office0/SemanticHeat/run_0" \
  # "office1:/media/phudh/X9_Pro/undergraduate/ActiveMapping/results_all/Replica/office1/SemanticHeat/run_0" \
  # "office2:/media/phudh/X9_Pro/undergraduate/ActiveMapping/results_all/Replica/office2/SemanticHeat/run_0" \
  # "office3:/media/phudh/X9_Pro/undergraduate/ActiveMapping/results_all/Replica/office3/SemanticHeat/run_0" \
  # "office4://media/phudh/X9_Pro/undergraduate/ActiveMapping/results_all/Replica/office4/SemanticHeat/run_0" \
  # "room0:/media/phudh/X9_Pro/undergraduate/ActiveMapping/results_all/Replica/room0/SemanticHeat/run_0" \
  # "room1:/media/phudh/X9_Pro/undergraduate/ActiveMapping/results_all/Replica/room1/SemanticHeat/run_0" \

do
  scene="${item%%:*}"
  run_root="${item#*:}"
  npz="${run_root}/splatam/final/params.npz"

  if [[ ! -f "$npz" ]]; then
    echo "[WARN] Missing npz: $npz"
    continue
  fi

  out_dir="${run_root}/visualization/bev_rgb_sem_entropy_steps/final"
  mkdir -p "$out_dir"

  python src/visualization/render_rgb_sem_entro_bev.py \
    --npz "$npz" \
    --dataset Replica \
    --scene "$scene" \
    --traj "data/replica_sim_nvs/${scene}/traj.txt" \
    --out "${out_dir}/bev.png" \
    --coord-system sim \
    --render-mode gaussian \
    --res 0.015 \
    --remove-front-percent 40 \
    --max-height 0.4 \
    --dpi 400 \
    --render-gt-semantic \
    --mp3d-palette eval
done