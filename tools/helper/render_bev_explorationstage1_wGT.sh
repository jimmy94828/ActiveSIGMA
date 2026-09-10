#!/usr/bin/env bash
# bash ./tools/helper/render_bev_explorationstage1_wGT.sh
set -euo pipefail
for item in \
  "GdvgFV5R1Z5:/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/GdvgFV5R1Z5/SemanticHeat/run_0" \
  "HxpKQynjfin:/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/HxpKQynjfin/SemanticHeat/run_0" \
  "gZ6f7yhEvPG:/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/gZ6f7yhEvPG/SemanticHeat/run_0" \
  "pLe4wQe7qrG:/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/pLe4wQe7qrG/SemanticHeat/run_0"
  # "GdvgFV5R1Z5:/media/phudh/HDD/ActiveSGM_result/MP3D_52/MP3D/GdvgFV5R1Z5/ActiveSem/run_0" \
  # "HxpKQynjfin:/media/phudh/HDD/ActiveSGM_result/MP3D_52/MP3D/HxpKQynjfin/ActiveSem/run_0" \
  # "gZ6f7yhEvPG:/media/phudh/HDD/ActiveSGM_result/MP3D_52/MP3D/gZ6f7yhEvPG/ActiveSem/run_0" \
  # "pLe4wQe7qrG:/media/phudh/HDD/ActiveSGM_result/MP3D_52/MP3D/pLe4wQe7qrG/ActiveSem/run_0" \
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
    --dataset MP3D \
    --scene "$scene" \
    --traj "data/mp3d_sim_nvs_v2/${scene}/traj.txt" \
    --out "${out_dir}/bev.png" \
    --coord-system sim \
    --render-mode gaussian \
    --res 0.015 \
    --remove-front-percent 0 \
    --max-height 1.0 \
    --dpi 400 \
    --render-gt-semantic \
    --mp3d-palette eval
done