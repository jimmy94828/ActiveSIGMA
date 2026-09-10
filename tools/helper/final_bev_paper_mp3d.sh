# MP3D
cd /media/phudh/X9_Pro/undergraduate/ActiveMapping

#GdvgFV5R1Z5 gZ6f7yhEvPG HxpKQynjfin pLe4wQe7qrG
conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/GdvgFV5R1Z5/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "/media/phudh/X9_Pro/undergraduate/ActiveMapping/data/mp3d_sim_nvs_v2/GdvgFV5R1Z5/traj.txt" \
  --scene "GdvgFV5R1Z5" \
  --out-dir "/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/GdvgFV5R1Z5/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
  --dataset MP3D \
  --res 0.01 \
  --ceiling-clearance 0.0 \
  --elevation 75 \
  --azimuths 0 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 


conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/gZ6f7yhEvPG/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "/media/phudh/X9_Pro/undergraduate/ActiveMapping/data/mp3d_sim_nvs_v2/gZ6f7yhEvPG/traj.txt" \
  --scene "gZ6f7yhEvPG" \
  --out-dir "/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/gZ6f7yhEvPG/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
  --dataset MP3D \
  --res 0.01 \
  --ceiling-clearance 3.0 \
  --elevation 75 \
  --azimuths 0 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 

conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/HxpKQynjfin/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "/media/phudh/X9_Pro/undergraduate/ActiveMapping/data/mp3d_sim_nvs_v2/HxpKQynjfin/traj.txt" \
  --scene "HxpKQynjfin" \
  --out-dir "/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/HxpKQynjfin/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
  --dataset MP3D \
  --res 0.01 \
  --ceiling-clearance -1.6 \
  --elevation 65 \
  --azimuths 180\
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 


conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/pLe4wQe7qrG/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "/media/phudh/X9_Pro/undergraduate/ActiveMapping/data/mp3d_sim_nvs_v2/pLe4wQe7qrG/traj.txt" \
  --scene "pLe4wQe7qrG" \
  --out-dir "/media/phudh/HDD/results_MP3D_reduce_candidates/MP3D/pLe4wQe7qrG/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
  --dataset MP3D \
  --res 0.01 \
  --ceiling-clearance 1.2 \
  --elevation 75 \
  --azimuths 90 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 1 \
  --canvas-size 3200 \
  --frustum-depth 0.10

# --elevation: 旋轉角度