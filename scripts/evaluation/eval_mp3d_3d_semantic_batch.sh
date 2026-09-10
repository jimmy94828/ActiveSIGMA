#!/bin/bash
set -euo pipefail

##################################################
### Batch 3D semantic evaluation for MP3D.
###
### Usage:
###   bash scripts/evaluation/eval_mp3d_3d_semantic_batch.sh [scene|all] [result_root] [run_name] [final|steps|all] [num_samples] [distance_threshold] [opacity_threshold]
###
### Example:
###   conda run -n activemapping bash scripts/evaluation/eval_mp3d_3d_semantic_batch.sh all /media/phudh/HDD/ActiveMapping_result_local_global/MP3D run_0_random_global final
##################################################

scene_arg=${1:-all}
result_root=${2:-/media/phudh/HDD/ActiveMapping_result_local_global/MP3D}
run_name=${3:-run_0_random_global}
mode=${4:-final}
num_samples=${5:-200000}
distance_threshold=${6:-0.05}
opacity_threshold=${7:-0.05}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
proj_dir="$(cd "${script_dir}/../.." && pwd)"

task_root="${proj_dir}/data/MP3D/v1/tasks/mp3d"
scan_root="${proj_dir}/data/MP3D/v1/scans"
traj_root="${proj_dir}/data/mp3d_sim_nvs_v2"
class_info="${proj_dir}/configs/MP3D/class_info_file.json"
category_mapping="${proj_dir}/configs/MP3D/category_mapping.tsv"
eval_script="${proj_dir}/src/evaluation/eval_3d_semantic.py"
plot_script="${proj_dir}/src/evaluation/plot_3d_semantic_summary.py"

if [[ "${mode}" != "final" && "${mode}" != "steps" && "${mode}" != "all" ]]; then
    echo "mode must be one of: final, steps, all" >&2
    exit 1
fi

if [[ "${scene_arg}" == "all" ]]; then
    selected_scenes=()
    for scene_dir in "${result_root}"/*; do
        [[ -d "${scene_dir}" ]] && selected_scenes+=("$(basename "${scene_dir}")")
    done
else
    selected_scenes=("${scene_arg}")
fi

resolve_mesh() {
    local scene=$1
    local clean_mesh="${task_root}/${scene}/semantic_clean.ply"
    local raw_mesh="${task_root}/${scene}/${scene}_semantic.ply"
    if [[ -f "${clean_mesh}" ]]; then
        echo "${clean_mesh}"
    elif [[ -f "${raw_mesh}" ]]; then
        echo "${raw_mesh}"
    else
        echo ""
    fi
}

eval_one_npz() {
    local scene=$1
    local params_path=$2
    local output_tag=$3

    local run_dir="${result_root}/${scene}/SemanticHeat/${run_name}"
    # local run_dir="${result_root}/${scene}/ActiveSem/${run_name}"
    local gt_mesh
    gt_mesh=$(resolve_mesh "${scene}")
    local semseg_json="${scan_root}/${scene}/${scene}/house_segmentations/${scene}.semseg.json"
    local instance_map="${proj_dir}/configs/MP3D/${scene}/instance_to_mpcat40.json"
    local transform_traj="${traj_root}/${scene}/traj.txt"
    local output_dir="${run_dir}/eval_3d_semantic/${output_tag}"

    if [[ ! -f "${params_path}" ]]; then
        echo "[skip] ${scene}: missing params ${params_path}"
        return
    fi
    if [[ -z "${gt_mesh}" || ! -f "${semseg_json}" || ! -f "${class_info}" || ! -f "${category_mapping}" || ! -f "${transform_traj}" ]]; then
        echo "[skip] ${scene}: missing MP3D GT, mapping, or trajectory files"
        return
    fi

    echo "[eval] scene=${scene} run=${run_name} params=${params_path} output=${output_dir}"
    python "${eval_script}" \
        --dataset MP3D \
        --scene "${scene}" \
        --params "${params_path}" \
        --gt_mesh "${gt_mesh}" \
        --class_info "${class_info}" \
        --semseg_json "${semseg_json}" \
        --category_mapping "${category_mapping}" \
        --instance_map "${instance_map}" \
        --transform_traj "${transform_traj}" \
        --output_dir "${output_dir}" \
        --num_samples "${num_samples}" \
        --distance_threshold "${distance_threshold}" \
        --opacity_threshold "${opacity_threshold}"
}

for scene in "${selected_scenes[@]}"; do
    run_dir="${result_root}/${scene}/SemanticHeat/${run_name}"
    # run_dir="${result_root}/${scene}/ActiveSem/${run_name}"
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

summary_csv="${result_root}/eval_3d_semantic_summary_${run_name}_${mode}.csv"
python - "${result_root}" "${run_name}" "${summary_csv}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_name = sys.argv[2]
summary_csv = Path(sys.argv[3])
rows = []

for metrics_path in sorted(root.glob(f"*/SemanticHeat/{run_name}/eval_3d_semantic/*/metrics.json")):
# for metrics_path in sorted(root.glob(f"*/ActiveSem/{run_name}/eval_3d_semantic/*/metrics.json")):
    scene = metrics_path.relative_to(root).parts[0]
    tag = metrics_path.parent.name
    with metrics_path.open("r") as f:
        metrics = json.load(f)
    rows.append({
        "scene": scene,
        "run_name": run_name,
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
        "run_name",
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

plot_dir="${result_root}/eval_3d_semantic_plots_${run_name}_${mode}"
python "${plot_script}" \
    --summary_csv "${summary_csv}" \
    --output_dir "${plot_dir}" \
    --title "MP3D 3D Semantic Evaluation (${run_name}, ${mode})"
