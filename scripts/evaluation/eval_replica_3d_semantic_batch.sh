#!/bin/bash
set -euo pipefail

##################################################
### Batch 3D semantic evaluation for Replica.
###
### Usage:
###   bash scripts/evaluation/eval_replica_3d_semantic_batch.sh [scene|all] [result_root] [final|steps|all] [num_samples] [distance_threshold] [opacity_threshold]
###
### Example:
###   conda run -n activemapping bash scripts/evaluation/eval_replica_3d_semantic_batch.sh all /media/phudh/HDD/ActiveMapping_result_local_global/Replica final
##################################################

scene_arg=${1:-all}
result_root=${2:-/media/phudh/HDD/ActiveMapping_result_local_global/Replica}
mode=${3:-final}
num_samples=${4:-200000}
distance_threshold=${5:-0.05}
opacity_threshold=${6:-0.05}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
proj_dir="$(cd "${script_dir}/../.." && pwd)"

gt_root="${proj_dir}/data/replica_v1"
traj_root="${proj_dir}/data/Replica"
eval_script="${proj_dir}/src/evaluation/eval_3d_semantic.py"

scenes=(office0 office1 office2 office3 office4 room0 room1 room2)

if [[ "${scene_arg}" == "all" ]]; then
    selected_scenes=("${scenes[@]}")
else
    selected_scenes=("${scene_arg}")
fi

if [[ "${mode}" != "final" && "${mode}" != "steps" && "${mode}" != "all" ]]; then
    echo "mode must be one of: final, steps, all" >&2
    exit 1
fi

replica_scene_name() {
    local scene=$1
    if [[ "${scene}" == office* ]]; then
        echo "office_${scene#office}"
    elif [[ "${scene}" == room* ]]; then
        echo "room_${scene#room}"
    else
        echo "${scene}"
    fi
}

eval_one_npz() {
    local scene=$1
    local params_path=$2
    local output_tag=$3

    local replica_scene
    replica_scene=$(replica_scene_name "${scene}")

    # local run_dir="${result_root}/${scene}/SemanticHeat/run_0_local_global"
    local run_dir="${result_root}/${scene}/sgsslam/run_0"
    local gt_mesh="${gt_root}/${replica_scene}/habitat/mesh_semantic.ply"
    local semantic_info="${gt_root}/${replica_scene}/habitat/info_semantic.json"
    local transform_traj="${traj_root}/${scene}/traj.txt"
    local output_dir="${run_dir}/eval_3d_semantic/${output_tag}"

    if [[ ! -f "${params_path}" ]]; then
        echo "[skip] ${scene}: missing params ${params_path}"
        return
    fi
    if [[ ! -f "${gt_mesh}" || ! -f "${semantic_info}" || ! -f "${transform_traj}" ]]; then
        echo "[skip] ${scene}: missing GT or trajectory files"
        return
    fi

    echo "[eval] scene=${scene} params=${params_path} output=${output_dir}"
    python "${eval_script}" \
        --params "${params_path}" \
        --gt_mesh "${gt_mesh}" \
        --semantic_info "${semantic_info}" \
        --transform_traj "${transform_traj}" \
        --output_dir "${output_dir}" \
        --num_samples "${num_samples}" \
        --distance_threshold "${distance_threshold}" \
        --opacity_threshold "${opacity_threshold}"
}

for scene in "${selected_scenes[@]}"; do
    # run_dir="${result_root}/${scene}/SemanticHeat/run_0_local_global"
    run_dir="${result_root}/${scene}/sgsslam/run_0"
    splatam_dir="${run_dir}/splatam"

    if [[ ! -d "${splatam_dir}" ]]; then
        echo "[skip] ${scene}: missing ${splatam_dir}"
        continue
    fi

    if [[ "${mode}" == "final" || "${mode}" == "all" ]]; then
        eval_one_npz "${scene}" "${splatam_dir}/final/params.npz" "final"
    fi

    if [[ "${mode}" == "steps" || "${mode}" == "all" ]]; then
        while IFS= read -r params_path; do
            base_name=$(basename "${params_path}" .npz)
            step_id=${base_name#params}
            eval_one_npz "${scene}" "${params_path}" "step_${step_id}"
        done < <(find "${splatam_dir}" -maxdepth 1 -type f -name 'params*.npz' | sort -V)
    fi
done

summary_csv="${result_root}/eval_3d_semantic_summary_${mode}.csv"
python - "${result_root}" "${summary_csv}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
rows = []

# for metrics_path in sorted(root.glob("*/SemanticHeat/run_0_local_global/eval_3d_semantic/*/metrics.json")):
for metrics_path in sorted(root.glob("*/sgsslam/run_0/eval_3d_semantic/*/metrics.json")):
    scene = metrics_path.relative_to(root).parts[0]
    tag = metrics_path.parent.name
    with metrics_path.open("r") as f:
        metrics = json.load(f)
    rows.append({
        "scene": scene,
        "tag": tag,
        "miou": metrics.get("gt_domain_miou"),
        "macc": metrics.get("gt_domain_macc"),
        "overall_acc": metrics.get("gt_domain_overall_acc"),
        "semantic_fscore": metrics.get("semantic_fscore_at_threshold"),
        "semantic_precision": metrics.get("pred_semantic_precision_at_threshold"),
        "semantic_recall": metrics.get("gt_semantic_recall_at_threshold"),
        "geometry_precision": metrics.get("pred_geometry_precision_at_threshold"),
        "geometry_recall": metrics.get("gt_geometry_recall_at_threshold"),
        "num_pred_points": metrics.get("num_pred_points_after_filter"),
    })

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "scene",
        "tag",
        "miou",
        "macc",
        "overall_acc",
        "semantic_fscore",
        "semantic_precision",
        "semantic_recall",
        "geometry_precision",
        "geometry_recall",
        "num_pred_points",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote summary CSV: {summary_csv}")
PY
