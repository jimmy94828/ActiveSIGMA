# Replica
cd /media/phudh/X9_Pro/undergraduate/ActiveMapping

# for scene in office0 office1 office2 office3 office4 room0 room1 room2; do
#   conda run -n base python src/visualization/render_canonical_oblique_bev.py \
#     --npz "results_all/Replica/${scene}/SemanticHeat/run_0/splatam/final/params.npz" \
#     --traj "data/Replica/${scene}/traj.txt" \
#     --scene "${scene}" \
#     --out-dir "results_all/Replica/${scene}/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
#     --dataset Replica \
#     --res 0.01 \
#     --elevation 75 \
#     --azimuths 0 90 180 270 \
#     --pose-stride 10 \
#     --trajectory-dash-length 4 \
#     --trajectory-gap-length 3 \
#     --canvas-size 3200
# done

conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "results_all/Replica/office0/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "data/Replica/office0/traj.txt" \
  --scene "office0" \
  --out-dir "results_all/Replica/office0/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
  --dataset Replica \
  --res 0.01 \
  --elevation 75 \
  --azimuths 0 90 180 270 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 

conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "results_all/Replica/office1/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "data/Replica/office1/traj.txt" \
  --scene "office1" \
  --out-dir "results_all/Replica/office1/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
  --dataset Replica \
  --res 0.01 \
  --elevation 75 \
  --azimuths 0 90 180 270 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 

conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "results_all/Replica/office2/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "data/Replica/office2/traj.txt" \
  --scene "office2" \
  --out-dir "results_all/Replica/office2/SemanticHeat/run_0/visualization/bev_final_corrected_v3" \
  --dataset Replica \
  --res 0.01 \
  --ceiling-clearance 0.0 \
  --elevation 80 \
  --azimuths 180 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 

conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "results_all/Replica/office3/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "data/Replica/office3/traj.txt" \
  --scene "office3" \
  --out-dir "results_all/Replica/office3/SemanticHeat/run_0/visualization/bev_final_corrected_v3" \
  --dataset Replica \
  --res 0.01 \
  --elevation 75 \
  --azimuths 180 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 

conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "results_all/Replica/office4/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "data/Replica/office4/traj.txt" \
  --scene "office4" \
  --out-dir "results_all/Replica/office4/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
  --dataset Replica \
  --res 0.01 \
  --elevation 75 \
  --azimuths 0 90 180 270 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 

conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "results_all/Replica/room0/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "data/Replica/room0/traj.txt" \
  --scene "room0" \
  --out-dir "results_all/Replica/room0/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
  --dataset Replica \
  --res 0.01 \
  --elevation 75 \
  --azimuths 0 90 180 270 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 

conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "results_all/Replica/room1/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "data/Replica/room1/traj.txt" \
  --scene "room1" \
  --out-dir "results_all/Replica/room1/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
  --dataset Replica \
  --res 0.01 \
  --elevation 75 \
  --azimuths 0 90 180 270 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 

conda run -n base python src/visualization/render_canonical_oblique_bev.py \
  --npz "results_all/Replica/room2/SemanticHeat/run_0/splatam/final/params.npz" \
  --traj "data/Replica/room2/traj.txt" \
  --scene "room2" \
  --out-dir "results_all/Replica/room2/SemanticHeat/run_0/visualization/bev_final_corrected_v2" \
  --dataset Replica \
  --res 0.01 \
  --elevation 75 \
  --azimuths 0 90 180 270 \
  --pose-stride 10 \
  --trajectory-dash-length 4 \
  --trajectory-gap-length 3 \
  --canvas-size 3200 \
  --frustum-depth 0.10 


# --elevation: 旋轉角度